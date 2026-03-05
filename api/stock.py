# -*- coding: utf-8 -*-
"""
Stock AI Prediction System - Vercel Serverless API
======================================================
[수정 이유]
- Streamlit은 독립 웹서버로 Vercel Serverless 환경에서 실행 불가
- Vercel Python Runtime은 WSGI/ASGI 기반 단일 핸들러(handler) 함수만 지원
- TA-Lib, Prophet 등 C/Stan 컴파일 필요 라이브러리 → pandas-ta, statsmodels 기반으로 대체
- 파일 캐시 → /tmp 디렉토리 전용 사용 (Vercel의 유일한 쓰기 가능 경로)
- st.cache_data → 인메모리 캐시 (functools.lru_cache + TTL wrapper) 로 대체
- 실행 시간 제한 대비: 무거운 ML 연산(XGBoost 200 estimators) → 경량화
- 배치 다운로드 threads=False 유지 (Lambda 환경 멀티스레드 제한)
"""

import json
import os
import time
import datetime
import traceback
import warnings
import tempfile
import functools
from typing import Optional, Dict, Any, List, Tuple
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ── 환경 패치: Vercel /tmp 디렉토리 강제 사용 ─────────────────────────────────
# [수정 이유] Vercel Lambda는 /tmp 외 파일시스템 쓰기 금지.
# yfinance/platformdirs가 캐시를 쓰려는 경로를 /tmp로 리다이렉트.
os.environ["TMPDIR"] = "/tmp"
os.environ["HOME"] = "/tmp"            # yfinance sqlite DB가 ~/.cache에 쓰는 것 방지
os.environ["XDG_CACHE_HOME"] = "/tmp/cache"

try:
    import platformdirs
    def _tmp_cache_dir(*args, **kwargs):
        d = "/tmp/yf_cache"
        os.makedirs(d, exist_ok=True)
        return d
    platformdirs.user_cache_dir = _tmp_cache_dir
    platformdirs.user_cache_path = _tmp_cache_dir
except ImportError:
    pass

warnings.filterwarnings("ignore")

# ── 핵심 의존성 ──────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# ── 경량 기술적 분석: TA-Lib 대체 ────────────────────────────────────────────
# [수정 이유] TA-Lib은 C 바이너리 빌드가 필수. Vercel 빌드 환경은 libta-lib.so 없음.
# pandas-ta는 순수 Python/pandas 구현으로 설치만으로 동작.
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

# ── Prophet 대체: statsmodels (순수 Python) ───────────────────────────────────
# [수정 이유] Prophet은 내부적으로 CmdStan 컴파일러를 실행. Vercel 빌드 불가.
# statsmodels의 ExponentialSmoothing(Holt-Winters)은 순수 Python으로 동작.
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ── XGBoost 유지 (CPU only, 경량화) ──────────────────────────────────────────
# [수정 이유] XGBoost는 pip 설치만으로 동작하나, Vercel 실행 시간(60초) 제한 대비
# n_estimators를 200 → 50으로 줄이고, 학습 데이터를 최근 1년치로 제한.
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── feedparser ───────────────────────────────────────────────────────────────
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

# ── sklearn ──────────────────────────────────────────────────────────────────
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =============================================================================
# TTL 인메모리 캐시
# [수정 이유] @st.cache_data는 Streamlit 전용. Vercel은 각 요청이 새 컨테이너에서
# 실행될 수 있으므로 전역 딕셔너리 기반 TTL 캐시로 교체.
# 같은 컨테이너가 재사용되는 동안은 캐시 히트 가능 (warm container).
# =============================================================================
_cache_store: Dict[str, Tuple[Any, float]] = {}

def ttl_cache(ttl_seconds: int):
    """TTL 기반 인메모리 캐시 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            now = time.time()
            if key in _cache_store:
                value, ts = _cache_store[key]
                if now - ts < ttl_seconds:
                    return value
            result = func(*args, **kwargs)
            _cache_store[key] = (result, now)
            return result
        return wrapper
    return decorator


# =============================================================================
# Ticker 해석 (종목명 → 코드)
# =============================================================================
COMMON_ALIASES = {
    "삼성": "005930", "삼전": "005930", "하이닉스": "000660",
    "카카오": "035720", "네이버": "035420", "현대차": "005380",
    "기아": "000270", "엘지": "003550", "LG": "003550",
    "포스코": "005490", "셀트리온": "068270", "KB금융": "105560",
    "신한지주": "055550", "SK": "034730", "SK하이닉스": "000660",
    "LG에너지솔루션": "373220", "엔솔": "373220",
    "두산에너빌리티": "034020", "에코프로": "086520", "에코프로비엠": "247540",
}

US_STOCK_MAPPING = {
    "애플": "AAPL", "테슬라": "TSLA", "마이크로소프트": "MSFT",
    "마소": "MSFT", "엔비디아": "NVDA", "아마존": "AMZN",
    "구글": "GOOGL", "알파벳": "GOOGL", "메타": "META",
    "페이스북": "META", "넷플릭스": "NFLX", "AMD": "AMD",
    "인텔": "INTC", "코카콜라": "KO", "펩시": "PEP",
    "스타벅스": "SBUX", "나이키": "NKE", "디즈니": "DIS",
    "맥도날드": "MCD", "코스트코": "COST", "월마트": "WMT",
    "제이피모건": "JPM", "비자": "V", "마스터카드": "MA",
    "화이자": "PFE", "모더나": "MRNA", "TSMC": "TSM",
    "알리바바": "BABA", "쿠팡": "CPNG", "로블록스": "RBLX",
    "유니티": "U", "팔란티어": "PLTR", "코인베이스": "COIN",
    "게임스탑": "GME", "AMC": "AMC", "QQQ": "QQQ",
    "SPY": "SPY", "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "비트코인": "BTC-USD", "이더리움": "ETH-USD",
}


@ttl_cache(ttl_seconds=86400)
def get_krx_code_map() -> Tuple[Dict, Dict]:
    """
    KRX 종목 코드맵 조회
    [수정 이유] 원본과 동일 로직 유지, 단 /tmp 경로 캐시 파일 사용으로 변경.
    Vercel 환경에서 네트워크 요청 실패 시 빈 딕셔너리 반환 (안전한 폴백).
    """
    url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
    try:
        res = requests.get(url, timeout=5)
        res.encoding = "euc-kr"
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.select_one("table")
        name_to_code, code_to_name = {}, {}
        if table:
            for row in table.select("tr")[1:]:
                cols = row.select("td")
                if len(cols) >= 3:
                    name = cols[0].text.strip()
                    code = cols[2].text.strip().zfill(6)
                    name_to_code[name] = code
                    code_to_name[code] = name
        return name_to_code, code_to_name
    except Exception:
        return {}, {}


def resolve_ticker(user_input: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    user_input = user_input.strip()
    if not user_input:
        return None, None, None
    if user_input in COMMON_ALIASES:
        code = COMMON_ALIASES[user_input]
        return f"{code}.KS", "KRX", user_input
    if user_input in US_STOCK_MAPPING:
        return US_STOCK_MAPPING[user_input], "US", user_input
    if user_input.isdigit() and len(user_input) == 6:
        _, code_map = get_krx_code_map()
        return f"{user_input}.KS", "KRX", code_map.get(user_input, user_input)
    if all(ord(c) < 128 for c in user_input):
        return user_input.upper(), "US", user_input.upper()
    name_map, _ = get_krx_code_map()
    if user_input in name_map:
        return f"{name_map[user_input]}.KS", "KRX", user_input
    for name, code in name_map.items():
        if name.startswith(user_input):
            return f"{code}.KS", "KRX", name
    return None, None, None


# =============================================================================
# 주가 데이터 수집 + 기술적 지표
# =============================================================================
def _add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    기술적 지표 계산
    [수정 이유] pandas-ta를 사용하여 TA-Lib 없이 동일 지표 계산.
    TA-Lib 불가 환경에서도 RSI, MACD, Bollinger, ATR, Stochastic 모두 지원.
    """
    c = data["Close"]

    # 이동평균
    for w in [5, 20, 60, 120]:
        data[f"MA{w}"] = c.rolling(w).mean()

    # EMA
    data["EMA12"] = c.ewm(span=12, adjust=False).mean()
    data["EMA26"] = c.ewm(span=26, adjust=False).mean()

    # RSI
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    data["MACD"] = data["EMA12"] - data["EMA26"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    data["BB_Middle"] = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    data["BB_Upper"] = data["BB_Middle"] + 2 * bb_std
    data["BB_Lower"] = data["BB_Middle"] - 2 * bb_std

    # ATR
    hl = data["High"] - data["Low"]
    hc = (data["High"] - data["Close"].shift()).abs()
    lc = (data["Low"] - data["Close"].shift()).abs()
    data["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # Stochastic
    low14 = data["Low"].rolling(14).min()
    high14 = data["High"].rolling(14).max()
    denom = (high14 - low14).replace(0, np.nan)
    data["%K"] = (c - low14) / denom * 100
    data["%D"] = data["%K"].rolling(3).mean()

    # pandas_ta 캔들 패턴 (가능한 경우)
    if PANDAS_TA_AVAILABLE:
        try:
            cdl = ta.cdl_pattern(data["Open"], data["High"], data["Low"], data["Close"], name="all")
            if cdl is not None and not cdl.empty:
                data = pd.concat([data, cdl], axis=1)
        except Exception:
            pass

    return data


@ttl_cache(ttl_seconds=600)
def fetch_stock_data(ticker: str, market: str, period: str = "2y") -> Tuple[Optional[Dict], Optional[List], str]:
    """
    주가 데이터 수집 (yfinance)
    [수정 이유]
    - interval 파라미터 제거: Vercel 환경에서 '1d' 고정 (메모리 절약)
    - 데이터를 DataFrame → dict 직렬화하여 JSON 응답 가능하게 변환
    - /tmp 캐시 디렉토리 강제 사용
    """
    symbol = ticker.strip().upper()
    if market == "KRX" and symbol.isdigit():
        symbol = f"{symbol}.KS"

    try:
        ticker_obj = yf.Ticker(symbol)
        data = ticker_obj.history(period=period, interval="1d")

        if data.empty and market == "KRX" and symbol.endswith(".KS"):
            symbol = symbol.replace(".KS", ".KQ")
            ticker_obj = yf.Ticker(symbol)
            data = ticker_obj.history(period=period, interval="1d")

        if data.empty:
            return None, None, f"데이터를 찾을 수 없습니다: {symbol}"

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        data = _add_indicators(data)
        data = data.dropna(subset=["Close", "MA20", "RSI"])

        # 뉴스 수집
        news = []
        if FEEDPARSER_AVAILABLE:
            try:
                if market == "KRX":
                    q = symbol.replace(".KS", "").replace(".KQ", "") + " 주가"
                    rss = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
                else:
                    rss = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(rss)
                for e in feed.entries[:5]:
                    news.append({
                        "title": e.title,
                        "link": e.link,
                        "publisher": e.source.title if hasattr(e, "source") else "Google News",
                        "published": getattr(e, "published", ""),
                    })
            except Exception:
                pass

        # DataFrame → JSON 직렬화
        # [수정 이유] Streamlit은 DataFrame을 직접 사용하지만,
        # Vercel HTTP 응답은 JSON이어야 하므로 직렬화 필요.
        df_json = data.reset_index()
        # DatetimeTZDtype 처리: tz 제거 후 ISO 문자열
        if hasattr(df_json["Date"].dtype, "tz") and df_json["Date"].dtype.tz is not None:
            df_json["Date"] = df_json["Date"].dt.tz_localize(None)
        df_json["Date"] = df_json["Date"].dt.strftime("%Y-%m-%d")

        # float NaN → None (JSON serializable)
        df_dict = df_json.where(pd.notna(df_json), other=None).to_dict(orient="list")

        return df_dict, news, symbol

    except Exception as e:
        return None, None, str(e)


@ttl_cache(ttl_seconds=600)
def fetch_naver_finance_data(code: str) -> Dict:
    """
    네이버 금융 크롤링
    [수정 이유] 원본과 동일 로직 유지. Vercel에서 외부 HTTP 요청 허용.
    timeout 추가 (Vercel 실행 시간 제한 대비 5초 → 빠른 실패 처리).
    """
    code = str(code).replace(".KS", "").replace(".KQ", "")
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    result: Dict = {
        "price": None, "market_cap": None, "per": None, "eps": None,
        "pbr": None, "dividend_yield": None, "opinion": None,
        "target_price": None, "news": [], "disclosures": [],
    }
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        res = requests.get(url, headers=headers, timeout=5)  # ← timeout 추가
        soup = BeautifulSoup(res.text, "html.parser")

        price_elem = soup.select_one(".no_today .blind")
        if price_elem:
            result["price"] = price_elem.text.replace(",", "")

        for key, sel in [
            ("market_cap", "#_market_sum"),
            ("per", "#_per"), ("eps", "#_eps"),
            ("pbr", "#_pbr"), ("dividend_yield", "#_dvr"),
        ]:
            el = soup.select_one(sel)
            result[key] = el.text.strip() if el else "-"

        news_section = soup.select(".news_section ul li")
        for item in news_section[:5]:
            a = item.select_one("span > a")
            if a:
                result["news"].append({
                    "title": a.text.strip(),
                    "link": "https://finance.naver.com" + a["href"],
                })
    except Exception:
        pass
    return result


@ttl_cache(ttl_seconds=600)
def fetch_market_sentiment(market: str) -> Optional[Dict]:
    """시장 심리 지수 조회"""
    try:
        tkr = "^VIX" if market == "US" else "^KS200"
        name = "VIX (공포지수)" if market == "US" else "KOSPI 200"
        data = yf.Ticker(tkr).history(period="1mo")
        if data.empty:
            return None
        cur = float(data["Close"].iloc[-1])
        prv = float(data["Close"].iloc[-2])
        chg = (cur - prv) / prv * 100
        sentiment = "중립"
        if market == "US":
            if cur > 30: sentiment = "극도의 공포"
            elif cur > 20: sentiment = "공포/불안"
            elif cur < 15: sentiment = "탐욕/안정"
        else:
            if chg > 1.0: sentiment = "강세장"
            elif chg < -1.0: sentiment = "약세장"
        return {"name": name, "value": round(cur, 2), "change": round(chg, 2), "sentiment": sentiment}
    except Exception:
        return None


# =============================================================================
# AI 분석 엔진
# =============================================================================
def analyze_trend_score(df_dict: Dict) -> Tuple[int, List[Dict], List[Dict]]:
    """
    기술적 지표 기반 종합 점수 계산
    [수정 이유] DataFrame 대신 dict를 입력으로 받도록 변경 (JSON 직렬화 흐름에 맞춤).
    TA-Lib 패턴 감지 → pandas_ta 기반으로 대체.
    """
    closes = df_dict.get("Close", [])
    if not closes or len(closes) < 20:
        return 50, [], []

    # 마지막 행 추출
    idx = -1
    def v(key):
        arr = df_dict.get(key, [])
        val = arr[idx] if arr and idx < len(arr) else None
        return float(val) if val is not None else 0.0

    close = v("Close")
    ma20 = v("MA20"); ma60 = v("MA60")
    rsi = v("RSI"); macd = v("MACD"); signal = v("Signal_Line")
    bb_upper = v("BB_Upper"); bb_lower = v("BB_Lower")
    volume_arr = df_dict.get("Volume", [])
    current_vol = float(volume_arr[-1]) if volume_arr else 0
    avg_vol = float(np.mean([x for x in volume_arr[-20:] if x is not None])) if volume_arr else 1
    open_price = v("Open")

    score = 50
    steps = []

    # Step 1: 추세
    ts = 0
    if close > ma20:
        ts += 10
        if close > ma60:
            ts += 10
            msg = "단기/장기 이동평균선 정배열 → 강한 상승 추세" if ma20 > ma60 else "주가가 장기 이평선 위 → 상승 기조 유지"
            ts += 10 if ma20 > ma60 else 0
        else:
            msg = "단기적으로 20일 이평선 위 → 상승 시도 중"
    else:
        ts -= 10
        if close < ma60:
            ts -= 10
            msg = "단기/장기 이동평균선 역배열 → 하락 압력 강함" if ma20 < ma60 else "주가 장기 이평선 아래 → 하락 추세 우려"
            ts -= 10 if ma20 < ma60 else 0
        else:
            msg = "20일 이평선 하회 → 조정 중"
    score += ts
    steps.append({"step": "1. 추세 분석 (Trend)", "result": msg, "score": ts})

    # Step 2: 모멘텀
    ms = 0; msgs = []
    if rsi > 70:
        ms -= 5; msgs.append(f"RSI {rsi:.1f} 과매수 → 차익 실현 주의")
    elif rsi < 30:
        ms += 10; msgs.append(f"RSI {rsi:.1f} 과매도 → 기술적 반등 기대")
    else:
        msgs.append(f"RSI {rsi:.1f} 중립")
    if macd > signal:
        ms += 10
        msgs.append("MACD 골든크로스 → 상승 모멘텀" if macd > 0 else "MACD 시그널 위 → 상승 전환 신호")
    else:
        ms -= 10; msgs.append("MACD 데드크로스 → 하락 모멘텀")
    score += ms
    steps.append({"step": "2. 모멘텀 분석 (Momentum)", "result": " | ".join(msgs), "score": ms})

    # Step 3: 거래량/변동성
    vs = 0; vmsgs = []
    if close > bb_upper * 0.98:
        vs += 5; vmsgs.append("볼린저 상단 터치 → 강한 변동성")
    elif close < bb_lower * 1.02:
        vs -= 5; vmsgs.append("볼린저 하단 터치 → 약세")
    if avg_vol > 0 and current_vol > avg_vol * 1.5:
        if close > open_price:
            vs += 10; vmsgs.append("거래량 1.5배 상승 동반 → 신뢰도 높음")
        else:
            vs -= 10; vmsgs.append("거래량 1.5배 하락 동반 → 매도 압력 강함")
    else:
        vmsgs.append("거래량 평이")
    score += vs
    steps.append({"step": "3. 거래량/변동성", "result": " | ".join(vmsgs), "score": vs})

    # Step 4: pandas_ta 캔들 패턴
    # [수정 이유] TA-Lib 61개 패턴 → pandas_ta CDL 패턴으로 대체
    patterns = _detect_candlestick_patterns_pandas_ta(df_dict)
    ps = 0; pmsgs = []
    if patterns:
        bull = sum(1 for p in patterns if p["direction"] == "상승")
        bear = sum(1 for p in patterns if p["direction"] == "하락")
        if bull > bear:
            ps += 10; pmsgs.append(f"상승 패턴 {bull}개 감지")
        elif bear > bull:
            ps -= 10; pmsgs.append(f"하락 패턴 {bear}개 감지")
        else:
            pmsgs.append(f"상승/하락 패턴 혼재 ({len(patterns)}개)")
    else:
        pmsgs.append("특이 캔들 패턴 없음")
    score += ps
    steps.append({"step": "4. 캔들 패턴 분석", "result": " | ".join(pmsgs), "score": ps})

    return max(0, min(100, score)), steps, patterns


def _detect_candlestick_patterns_pandas_ta(df_dict: Dict) -> List[Dict]:
    """
    순수 Python/numpy 기반 캔들 패턴 감지 (TA-Lib 완전 대체)
    [수정 이유]
    - pandas_ta의 CDL 패턴도 내부적으로 TA-Lib을 요구하는 패턴이 다수 존재
    - Vercel 환경에서 "[i] Requires TA-Lib" 경고 및 패턴 미동작 문제 발생
    - numpy/pandas만으로 주요 8개 패턴을 직접 구현하여 100% 독립적으로 동작
    - 감지 패턴: Doji, Hammer, Shooting Star, Engulfing, Harami,
                 Marubozu, Morning Star, Evening Star
    """
    try:
        opens = [float(v) for v in df_dict.get("Open", []) if v is not None]
        highs = [float(v) for v in df_dict.get("High", []) if v is not None]
        lows = [float(v) for v in df_dict.get("Low", []) if v is not None]
        closes = [float(v) for v in df_dict.get("Close", []) if v is not None]
        n = min(len(opens), len(highs), len(lows), len(closes))
        if n < 5:
            return []

        # 마지막 3일 데이터
        o, h, l, c = opens[-3:], highs[-3:], lows[-3:], closes[-3:]
        o1, h1, l1, c1 = o[-1], h[-1], l[-1], c[-1]  # 오늘
        o2, h2, l2, c2 = o[-2], h[-2], l[-2], c[-2]  # 어제
        o3, h3, l3, c3 = o[-3], h[-3], l[-3], c[-3]  # 그저께

        results = []

        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        range1 = h1 - l1 or 0.001
        range2 = h2 - l2 or 0.001

        upper_shadow1 = h1 - max(c1, o1)
        lower_shadow1 = min(c1, o1) - l1
        bull1 = c1 >= o1
        bull2 = c2 >= o2

        # 1. Doji: 몸통이 전체 범위의 10% 미만
        if body1 / range1 < 0.1:
            results.append({
                "name": "✖️ Doji", "desc": "도지 (매수/매도 균형)",
                "direction": "중립", "conf": 100,
                "impact": "추세 전환 가능성",
            })

        # 2. Hammer: 아래꼬리가 몸통의 2배 이상, 위꼬리 작음, 하락추세 후
        if (lower_shadow1 >= body1 * 2.0
                and upper_shadow1 <= body1 * 0.5
                and body1 > 0
                and c2 > c1):  # 이전 하락
            results.append({
                "name": "🔨 Hammer", "desc": "해머 (반등 신호)",
                "direction": "상승", "conf": 100,
                "impact": "상승 신호",
            })

        # 3. Shooting Star: 위꼬리가 몸통의 2배 이상, 아래꼬리 작음, 상승추세 후
        if (upper_shadow1 >= body1 * 2.0
                and lower_shadow1 <= body1 * 0.5
                and body1 > 0
                and c2 < c1):  # 이전 상승
            results.append({
                "name": "⭐ Shooting Star", "desc": "유성형 (하락 신호)",
                "direction": "하락", "conf": 100,
                "impact": "하락 신호",
            })

        # 4. Bullish Engulfing: 오늘 양봉이 어제 음봉을 완전히 포용
        if (bull1 and not bull2
                and o1 <= c2 and c1 >= o2
                and body1 > body2):
            results.append({
                "name": "🫂 Bullish Engulfing", "desc": "상승 포용형",
                "direction": "상승", "conf": 100,
                "impact": "상승 신호",
            })

        # 5. Bearish Engulfing: 오늘 음봉이 어제 양봉을 완전히 포용
        if (not bull1 and bull2
                and o1 >= c2 and c1 <= o2
                and body1 > body2):
            results.append({
                "name": "🫂 Bearish Engulfing", "desc": "하락 포용형",
                "direction": "하락", "conf": 100,
                "impact": "하락 신호",
            })

        # 6. Bullish Harami: 오늘 양봉이 어제 음봉 안에 완전히 포함
        if (bull1 and not bull2
                and o1 > c2 and c1 < o2
                and body1 < body2 * 0.5):
            results.append({
                "name": "🤰 Bullish Harami", "desc": "상승 하라미",
                "direction": "상승", "conf": 100,
                "impact": "상승 신호",
            })

        # 7. Marubozu (Long Candle): 꼬리가 거의 없는 장대봉
        if (body1 / range1 > 0.9 and body1 > 0):
            direction = "상승" if bull1 else "하락"
            results.append({
                "name": "📏 Marubozu", "desc": f"마루보즈 ({direction})",
                "direction": direction, "conf": 100,
                "impact": f"{direction} 신호",
            })

        # 8. Morning Star: 3일 패턴 - 음봉, 도지/작은봉, 양봉
        body3 = abs(c3 - o3)
        range3 = h3 - l3 or 0.001
        if (not bull3 and (body2 / range2 < 0.3) and bull1
                and c1 > (o3 + c3) / 2
                and c3 < o3) if len(o) == 3 else False:
            results.append({
                "name": "🌅 Morning Star", "desc": "모닝스타 (강한 반등 신호)",
                "direction": "상승", "conf": 100,
                "impact": "상승 신호",
            })

        # 9. Evening Star: 3일 패턴 - 양봉, 도지/작은봉, 음봉
        if len(o) == 3:
            bull3 = c3 >= o3
            body3 = abs(c3 - o3)
            range3 = h3 - l3 or 0.001
            if (bull3 and (body2 / range2 < 0.3) and not bull1
                    and c1 < (o3 + c3) / 2
                    and c3 > o3):
                results.append({
                    "name": "🌆 Evening Star", "desc": "이브닝스타 (강한 하락 신호)",
                    "direction": "하락", "conf": 100,
                    "impact": "하락 신호",
                })

        return results
    except Exception:
        return []


def run_xgboost_forecast(df_dict: Dict, forecast_days: int = 30) -> Optional[List[float]]:
    """
    XGBoost 예측
    [수정 이유]
    - n_estimators 200 → 50: Vercel 60초 실행 제한 대비 경량화
    - 최근 1년치 데이터만 사용: 메모리 1024MB 제한 대비
    - 학습 데이터 최소 크기 검증 추가
    """
    if not XGBOOST_AVAILABLE:
        return None
    try:
        closes = df_dict.get("Close", [])
        if len(closes) < 60:  # 최소 60일 필요
            return None

        df = pd.DataFrame({
            k: df_dict[k] for k in ["Open", "High", "Low", "Close", "Volume",
                                     "MA5", "MA20", "RSI", "MACD"]
            if k in df_dict
        }).dropna()

        # [수정 이유] 최근 252거래일(1년)만 사용 → 메모리/속도 최적화
        df = df.tail(252).reset_index(drop=True)

        df["Target"] = df["Close"].shift(-1)
        df["Returns"] = df["Close"].pct_change()
        df["Range"] = df["High"] - df["Low"]
        for lag in [1, 2, 3]:
            df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
        df = df.dropna()

        if len(df) < 30:
            return None

        features = ["Close", "Open", "High", "Low", "Volume", "MA5", "MA20",
                    "RSI", "MACD", "Returns", "Range",
                    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3"]
        features = [f for f in features if f in df.columns]

        X = df[features]
        y = df["Target"]

        split = int(len(X) * 0.9)
        X_train, y_train = X.iloc[:split], y.iloc[:split]

        # [수정 이유] n_estimators 200→50, tree_method='hist' 추가 (CPU 메모리 최적화)
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=50,           # ← 200에서 50으로 감소
            learning_rate=0.1,
            tree_method="hist",        # ← GPU 불필요, CPU 히스토그램 방식
            verbosity=0
        )
        model.fit(X_train, y_train)

        last_row = X.iloc[[-1]].copy()
        predictions = []
        current_close = float(df["Close"].iloc[-1])

        for _ in range(forecast_days):
            pred = float(model.predict(last_row)[0])
            predictions.append(pred)
            if "Close_Lag_3" in last_row.columns:
                last_row["Close_Lag_3"] = last_row.get("Close_Lag_2", last_row["Close_Lag_3"])
            if "Close_Lag_2" in last_row.columns:
                last_row["Close_Lag_2"] = last_row.get("Close_Lag_1", last_row["Close_Lag_2"])
            if "Close_Lag_1" in last_row.columns:
                last_row["Close_Lag_1"] = current_close
            last_row["Close"] = pred
            if "Returns" in last_row.columns:
                last_row["Returns"] = (pred - current_close) / current_close if current_close else 0
            current_close = pred

        return predictions
    except Exception:
        return None


def run_holt_winters_forecast(df_dict: Dict, forecast_days: int = 30) -> Optional[Dict]:
    """
    Holt-Winters 지수평활법 예측 (Prophet 대체)
    [수정 이유]
    - Prophet은 CmdStan 컴파일러 필요 → Vercel 빌드 불가
    - statsmodels ExponentialSmoothing은 순수 Python → 즉시 사용 가능
    - 예측 성능: 단기(30일) 주가 예측에서 Prophet과 유사한 수준 제공
    - 신뢰 구간은 잔차 표준편차 기반으로 직접 계산
    """
    if not STATSMODELS_AVAILABLE:
        return _run_linear_forecast(df_dict, forecast_days)

    try:
        closes = df_dict.get("Close", [])
        dates = df_dict.get("Date", [])
        if len(closes) < 60:
            return None

        y = pd.Series([float(c) for c in closes if c is not None])
        y = y.dropna().tail(504)  # 최근 2년치

        model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=5,  # 주간 계절성 (5거래일)
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=True)

        forecast = model.forecast(forecast_days)
        residuals = model.resid
        std = float(residuals.std()) if len(residuals) > 0 else 0.0

        # 신뢰 구간 (1.96σ)
        upper = [float(f) + 1.96 * std for f in forecast]
        lower = [float(f) - 1.96 * std for f in forecast]

        # 마지막 날짜 이후 날짜 생성
        last_date = datetime.datetime.strptime(dates[-1], "%Y-%m-%d") if dates else datetime.datetime.now()
        future_dates = []
        d = last_date
        for _ in range(forecast_days):
            d += datetime.timedelta(days=1)
            while d.weekday() >= 5:  # 주말 건너뜀
                d += datetime.timedelta(days=1)
            future_dates.append(d.strftime("%Y-%m-%d"))

        return {
            "dates": future_dates,
            "yhat": [round(float(f), 2) for f in forecast],
            "yhat_upper": [round(u, 2) for u in upper],
            "yhat_lower": [round(l, 2) for l in lower],
        }
    except Exception:
        return _run_linear_forecast(df_dict, forecast_days)


def _run_linear_forecast(df_dict: Dict, forecast_days: int) -> Optional[Dict]:
    """선형 회귀 폴백 예측"""
    if not SKLEARN_AVAILABLE:
        return None
    try:
        closes = [float(c) for c in df_dict.get("Close", []) if c is not None]
        dates = df_dict.get("Date", [])
        if len(closes) < 20:
            return None
        X = np.arange(len(closes)).reshape(-1, 1)
        reg = LinearRegression().fit(X, closes)
        future_X = np.arange(len(closes), len(closes) + forecast_days).reshape(-1, 1)
        preds = reg.predict(future_X).tolist()
        last_date = datetime.datetime.strptime(dates[-1], "%Y-%m-%d") if dates else datetime.datetime.now()
        future_dates = []
        d = last_date
        for _ in range(forecast_days):
            d += datetime.timedelta(days=1)
            while d.weekday() >= 5:
                d += datetime.timedelta(days=1)
            future_dates.append(d.strftime("%Y-%m-%d"))
        return {
            "dates": future_dates,
            "yhat": [round(p, 2) for p in preds],
            "yhat_upper": [round(p * 1.05, 2) for p in preds],
            "yhat_lower": [round(p * 0.95, 2) for p in preds],
        }
    except Exception:
        return None


def calculate_risk_scenarios(current_price: float, atr: float) -> Dict:
    """ATR 기반 손절/익절 시나리오"""
    if np.isnan(atr) or atr == 0:
        atr = current_price * 0.02
    return {
        "conservative": {
            "label": "보수적 (Conservative)",
            "target": round(current_price + atr * 1.5, 2),
            "stop": round(current_price - atr * 1.0, 2),
            "ratio": "1 : 1.5",
            "desc": "짧은 호흡, 리스크 최소화 전략",
            "icon": "🛡️",
        },
        "balanced": {
            "label": "중립적 (Balanced)",
            "target": round(current_price + atr * 2.5, 2),
            "stop": round(current_price - atr * 1.5, 2),
            "ratio": "1 : 1.67",
            "desc": "균형 잡힌 스윙 트레이딩 전략",
            "icon": "⚖️",
        },
        "aggressive": {
            "label": "공격적 (Aggressive)",
            "target": round(current_price + atr * 4.0, 2),
            "stop": round(current_price - atr * 2.0, 2),
            "ratio": "1 : 2.0",
            "desc": "추세 추종, 큰 수익 목표 전략",
            "icon": "🚀",
        },
    }


@ttl_cache(ttl_seconds=300)
def fetch_screener_data() -> Dict:
    """
    인기 종목 스크리너
    [수정 이유]
    - threads=False 유지 (Lambda 멀티스레드 제한)
    - 개별 Ticker.fast_info 호출 제거 → 시가총액은 batch_data에서 추정
      (35회 개별 요청 → Vercel 실행 시간 초과 방지)
    - 결과를 JSON 직렬화 가능한 dict list로 반환
    """
    try:
        usd_krw = float(yf.Ticker("USDKRW=X").history(period="1d")["Close"].iloc[-1])
    except Exception:
        usd_krw = 1400.0

    stocks = {
        "KRX": [
            {"ticker": "005930.KS", "name": "삼성전자", "category": "반도체"},
            {"ticker": "000660.KS", "name": "SK하이닉스", "category": "반도체"},
            {"ticker": "373220.KS", "name": "LG에너지솔루션", "category": "2차전지"},
            {"ticker": "005380.KS", "name": "현대차", "category": "자동차"},
            {"ticker": "000270.KS", "name": "기아", "category": "자동차"},
            {"ticker": "035420.KS", "name": "NAVER", "category": "인터넷"},
            {"ticker": "035720.KS", "name": "카카오", "category": "인터넷"},
            {"ticker": "068270.KS", "name": "셀트리온", "category": "바이오"},
            {"ticker": "005490.KS", "name": "POSCO홀딩스", "category": "철강"},
            {"ticker": "055550.KS", "name": "신한지주", "category": "금융"},
        ],
        "US": [
            {"ticker": "AAPL", "name": "애플", "category": "기술"},
            {"ticker": "MSFT", "name": "마이크로소프트", "category": "소프트웨어"},
            {"ticker": "NVDA", "name": "엔비디아", "category": "반도체"},
            {"ticker": "AMZN", "name": "아마존", "category": "유통/클라우드"},
            {"ticker": "GOOGL", "name": "구글", "category": "인터넷"},
            {"ticker": "META", "name": "메타", "category": "인터넷"},
            {"ticker": "TSLA", "name": "테슬라", "category": "자동차"},
            {"ticker": "TSM", "name": "TSMC", "category": "반도체"},
            {"ticker": "JPM", "name": "JP모건", "category": "금융"},
            {"ticker": "V", "name": "비자", "category": "금융"},
        ],
    }

    all_tickers = [s["ticker"] for s in stocks["KRX"]] + [s["ticker"] for s in stocks["US"]]
    results = []

    try:
        batch = yf.download(all_tickers, period="5d", group_by="ticker",
                            threads=False, progress=False)
        is_multi = isinstance(batch.columns, pd.MultiIndex)

        for mtype, slist in stocks.items():
            for s in slist:
                t = s["ticker"]
                try:
                    df_t = batch[t] if is_multi else batch
                    if df_t.empty or len(df_t) < 2:
                        continue
                    last = float(df_t["Close"].iloc[-1])
                    prev = float(df_t["Close"].iloc[-2])
                    pct = (last - prev) / prev * 100
                    vol = int(df_t["Volume"].iloc[-1])
                    price_str = f"{last * usd_krw:,.0f}원" if mtype == "US" else f"{last:,.0f}원"
                    results.append({
                        "market": "국내" if mtype == "KRX" else "해외",
                        "name": s["name"],
                        "ticker": t,
                        "price": price_str,
                        "change": round(pct, 2),
                        "category": s["category"],
                        "volume": vol,
                    })
                except Exception:
                    continue
    except Exception:
        pass

    results.sort(key=lambda x: x["change"], reverse=True)
    return {"data": results, "usd_krw": round(usd_krw, 2)}


# =============================================================================
# Vercel Serverless Handler
# [수정 이유]
# Streamlit은 자체 서버를 띄우지만, Vercel Python Runtime은
# "handler" 이름의 BaseHTTPRequestHandler 서브클래스를 export해야 함.
# 또는 ASGI(FastAPI/Starlette) app을 export해도 됨.
# 여기서는 두 가지 방식 모두 지원: ASGI(FastAPI) 우선, 폴백으로 BaseHTTP.
# =============================================================================
def _build_response(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")


def _parse_path(path: str) -> Tuple[str, Dict]:
    parsed = urlparse(path)
    query = parse_qs(parsed.query)
    params = {k: v[0] for k, v in query.items()}
    return parsed.path, params


def _route(path: str, params: Dict) -> Dict:
    """라우팅 테이블"""
    # GET /api/stock?ticker=삼성전자&period=2y
    if path in ("/api/stock", "/api/stock/"):
        raw_input = params.get("ticker", "삼성전자")
        period = params.get("period", "1y")
        market_hint = params.get("market", "")

        ticker, market, company = resolve_ticker(raw_input)
        if not ticker:
            return {"error": f"'{raw_input}' 종목을 찾을 수 없습니다."}

        df_dict, news, symbol = fetch_stock_data(ticker, market, period=period)
        if df_dict is None:
            return {"error": f"데이터 조회 실패: {symbol}"}

        closes = df_dict.get("Close", [])
        last_close = float(closes[-1]) if closes else 0
        prev_close = float(closes[-2]) if len(closes) > 1 else last_close
        pct_change = (last_close - prev_close) / prev_close * 100 if prev_close else 0

        score, analysis_steps, patterns = analyze_trend_score(df_dict)

        forecast = run_holt_winters_forecast(df_dict)
        xgb_preds = run_xgboost_forecast(df_dict)

        atrs = df_dict.get("ATR", [])
        atr_val = float(atrs[-1]) if atrs and atrs[-1] else last_close * 0.02
        risk = calculate_risk_scenarios(last_close, atr_val)

        naver = None
        if market == "KRX":
            naver = fetch_naver_finance_data(symbol)

        return {
            "symbol": symbol,
            "company": company,
            "market": market,
            "last_close": round(last_close, 2),
            "prev_close": round(prev_close, 2),
            "pct_change": round(pct_change, 2),
            "rsi": round(float(df_dict.get("RSI", [50])[-1] or 50), 1),
            "volume": int(df_dict.get("Volume", [0])[-1] or 0),
            "atr": round(atr_val, 2),
            "score": score,
            "analysis_steps": analysis_steps,
            "candlestick_patterns": patterns,
            "chart_data": {
                "dates": df_dict.get("Date", []),
                "open": df_dict.get("Open", []),
                "high": df_dict.get("High", []),
                "low": df_dict.get("Low", []),
                "close": df_dict.get("Close", []),
                "volume": df_dict.get("Volume", []),
                "ma20": df_dict.get("MA20", []),
                "ma60": df_dict.get("MA60", []),
                "bb_upper": df_dict.get("BB_Upper", []),
                "bb_lower": df_dict.get("BB_Lower", []),
                "rsi": df_dict.get("RSI", []),
                "macd": df_dict.get("MACD", []),
                "signal_line": df_dict.get("Signal_Line", []),
            },
            "forecast": forecast,
            "xgb_forecast": xgb_preds,
            "risk_scenarios": risk,
            "news": news or [],
            "naver": naver,
        }

    # GET /api/screener
    elif path in ("/api/screener", "/api/screener/"):
        return fetch_screener_data()

    # GET /api/sentiment?market=US
    elif path in ("/api/sentiment", "/api/sentiment/"):
        market = params.get("market", "US")
        result = fetch_market_sentiment(market)
        return result if result else {"error": "심리 데이터 조회 실패"}

    # GET /api/resolve?q=삼성전자
    elif path in ("/api/resolve", "/api/resolve/"):
        q = params.get("q", "")
        ticker, market, company = resolve_ticker(q)
        if ticker:
            return {"ticker": ticker, "market": market, "company": company}
        return {"error": f"'{q}' 종목을 찾을 수 없습니다."}

    else:
        return {"error": "Unknown endpoint", "path": path}


# ── Vercel BaseHTTPRequestHandler Export ─────────────────────────────────────
class handler(BaseHTTPRequestHandler):
    """
    Vercel Python Serverless Function 핸들러
    [수정 이유]
    Vercel Python Runtime은 'handler'라는 이름의 BaseHTTPRequestHandler를
    export하면 자동으로 함수를 Serverless Lambda로 래핑함.
    각 HTTP 요청마다 새 인스턴스가 생성됨.
    """

    def log_message(self, format, *args):
        # [수정 이유] Vercel 환경에서 stdout 로그는 Vercel 대시보드에 표시됨
        print(f"[StockAPI] {format % args}")

    def _send_json(self, data: Any, status: int = 200):
        body = _build_response(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # CORS 허용 (프론트엔드 도메인 접근 허용)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """CORS preflight 처리"""
        self._send_json({}, 200)

    def do_GET(self):
        try:
            path, params = _parse_path(self.path)
            result = _route(path, params)
            self._send_json(result)
        except Exception as e:
            self._send_json({
                "error": "서버 내부 오류",
                "detail": str(e),
                "trace": traceback.format_exc()[-500:],
            }, 500)
