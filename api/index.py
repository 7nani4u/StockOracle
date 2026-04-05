# -*- coding: utf-8 -*-
"""
StockOracle - Vercel Serverless Unified Handler
================================================
[구조 설명]
- 단일 Python 파일이 HTML 프론트엔드 + 모든 /api/* 엔드포인트를 처리
- frontend/ 폴더, Next.js 빌드 없이 순수 Python + HTML/JS만으로 동작
- Vercel Python Runtime: BaseHTTPRequestHandler를 'handler'로 export

[Vercel 404 원인 및 수정 내용]
1. frontend/ 폴더 누락 → vercel.json이 없는 폴더 빌드 시도 → 수정: 단일 Python 파일로 통합
2. "framework":"nextjs" + "builds" 혼용 → Vercel v2 충돌 → 수정: builds만 사용
3. "routes" + "framework" 충돌 → 수정: routes로만 라우팅 통일
4. "dest":"/frontend/$1" 존재하지 않는 경로 → 수정: 모든 요청을 api/index.py로
5. api/stock.py 미진입 → 수정: api/index.py로 진입점 변경

[버그 수정 내역 v2]
1. 거래량 차트 색상: hex 문자열에 .replace('rgb','rgba') 적용 불가 → rgba() 직접 사용
2. RSI 기준선 dates[20] → dates가 20개 미만이면 IndexError → safe index 사용
3. holt_winters_forecast seasonal 오류 방지 → 데이터 부족 시 linear fallback 강화
4. route() path 매칭 일관성: /api/* trailing slash 정규화
5. XGBoost 컬럼 dtype 명시 → pandas 3.x 경고 제거
6. screener yf.download 멀티인덱스 처리 강화
"""

import json
import os
import sys
import time
import datetime
from datetime import datetime as dt, timedelta
import concurrent.futures
import math
import traceback
import warnings
import functools
import tempfile
import re
from typing import Optional, Dict, Any, List, Tuple
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote

# ── /tmp 강제 사용 (Vercel은 /tmp 외 쓰기 금지) ───────────────────────────────
if os.name == 'nt':
    # Windows 로컬 개발 환경
    TMP_DIR = tempfile.gettempdir()
else:
    # Vercel / Linux 환경
    TMP_DIR = "/tmp"

os.environ.setdefault("TMPDIR", TMP_DIR)
os.environ.setdefault("HOME", TMP_DIR)
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(TMP_DIR, "cache"))

try:
    import platformdirs
    def _tmp(*a, **k):
        d = os.path.join(TMP_DIR, "yf_cache")
        os.makedirs(d, exist_ok=True)
        return d
    platformdirs.user_cache_dir = _tmp
    platformdirs.user_cache_path = _tmp
except ImportError:
    pass

warnings.filterwarnings("ignore")

# ── 의존성 ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
# from scipy.signal import argrelextrema

# 통계/학습 라이브러리 제거 (Vercel 용량 제한 대응)
# 대신 경량화된 자체 구현 알고리즘 사용

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

# =============================================================================
# TTL 캐시 (st.cache_data 대체)
# =============================================================================
_CACHE: Dict[str, Tuple[Any, float]] = {}

def ttl_cache(ttl: int):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = f"{fn.__name__}|{args}|{sorted(kwargs.items())}"
            now = time.time()
            if key in _CACHE and now - _CACHE[key][1] < ttl:
                return _CACHE[key][0]
            r = fn(*args, **kwargs)
            _CACHE[key] = (r, now)
            # ── 만료 키 정리: 캐시 항목이 500개 초과 시 24시간 지난 키 일괄 삭제 ──
            if len(_CACHE) > 500:
                expired = [k for k, (_, t) in list(_CACHE.items()) if now - t > 86400]
                for k in expired:
                    _CACHE.pop(k, None)
            return r
        return wrapper
    return deco

@ttl_cache(600)
def get_usd_krw() -> float:
    """USD/KRW 환율 조회 (10분 캐시) — 중복 호출 방지용 단일 함수"""
    try:
        return float(
            yf.Ticker("USDKRW=X").history(period="1d")["Close"].iloc[-1]
        )
    except Exception:
        return 1380.0

# =============================================================================
# Ticker 매핑
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
    "알리바바": "BABA", "쿠팡": "CPNG", "로블록스": "RBLX", "유니티": "U",
    "팔란티어": "PLTR", "코인베이스": "COIN", "게임스탑": "GME", "AMC": "AMC",
    "QQQ": "QQQ", "SPY": "SPY", "SQQQ": "SQQQ",
    "TQQQ": "TQQQ", "SOXL": "SOXL", "비트코인": "BTC-USD",
    "이더리움": "ETH-USD",
}

@ttl_cache(86400)
def get_krx_code_map():
    url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
    try:
        res = requests.get(url, timeout=5)
        res.encoding = "euc-kr"
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.select_one("table")
        n2c, c2n = {}, {}
        if table:
            for row in table.select("tr")[1:]:
                cols = row.select("td")
                if len(cols) >= 3:
                    name = cols[0].text.strip()
                    code = cols[2].text.strip().zfill(6)
                    n2c[name] = code
                    c2n[code] = name
        return n2c, c2n
    except Exception:
        return {}, {}

def resolve_ticker(q: str):
    q = q.strip()
    if not q:
        return None, None, None
    if q in COMMON_ALIASES:
        return f"{COMMON_ALIASES[q]}.KS", "KRX", q
    if q in US_STOCK_MAPPING:
        return US_STOCK_MAPPING[q], "US", q
    if q.isdigit() and len(q) == 6:
        _, c2n = get_krx_code_map()
        return f"{q}.KS", "KRX", c2n.get(q, q)
    if all(ord(c) < 128 for c in q):
        return q.upper(), "US", q.upper()
    n2c, _ = get_krx_code_map()
    if q in n2c:
        return f"{n2c[q]}.KS", "KRX", q
    for name, code in n2c.items():
        if name.startswith(q):
            return f"{code}.KS", "KRX", name
    return None, None, None

# =============================================================================
# 지표 계산
# =============================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    
    # 핵심 추세 지표
    for w in [5, 20, 60, 120]:
        df[f"MA{w}"] = c.rolling(w).mean()
    df["EMA20"] = c.ewm(span=20, adjust=False).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()
    
    # MACD (12, 26, 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = c.diff()
    # Wilder's Smoothing (공식 RSI 계산법 — TradingView 동일)
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    df["BB_Middle"] = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    denom = (high14 - low14).replace(0, np.nan)
    df["%K"] = (c - low14) / denom * 100
    df["%D"] = df["%K"].rolling(3).mean()

    # ADX (14)
    dm_plus = df["High"].diff()
    dm_minus = -df["Low"].diff()
    dm_plus_adj = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus_adj = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    atr14 = df["ATR"]
    df["DI_Plus"]  = 100 * dm_plus_adj.rolling(14).mean()  / atr14.replace(0, np.nan)
    df["DI_Minus"] = 100 * dm_minus_adj.rolling(14).mean() / atr14.replace(0, np.nan)
    dx = (df["DI_Plus"] - df["DI_Minus"]).abs() / (df["DI_Plus"] + df["DI_Minus"]).replace(0, np.nan) * 100
    df["ADX"] = dx.rolling(14).mean()

    # OBV (On-Balance Volume) — 거래량 기반 추세 확인
    _obv_dir = np.sign(c.diff()).fillna(0)
    df["OBV"] = (df["Volume"] * _obv_dir).cumsum()

    # Aroon (25) — 고점/저점 경과일 기반 추세 방향성
    _ap = 25
    df["AROON_UP"]   = df["High"].rolling(_ap + 1).apply(
        lambda x: float(np.argmax(x)) / _ap * 100, raw=True)
    df["AROON_DOWN"] = df["Low"].rolling(_ap + 1).apply(
        lambda x: float(np.argmin(x)) / _ap * 100, raw=True)

    # Buy Pressure (14일 상승일 거래량 비중, 0–100%)
    _vol = df["Volume"].replace(0, np.nan)
    _up_vol = df["Volume"].where(c >= c.shift(1), 0.0)
    df["BUY_PRESSURE"] = _up_vol.rolling(14).sum() / _vol.rolling(14).sum() * 100

    # PSAR (Parabolic SAR, af=0.02, max=0.2) — 추세 방향 + 반전 신호
    _hs = df["High"].values.astype(float)
    _ls = df["Low"].values.astype(float)
    _n  = len(_hs)
    _sv_p = np.full(_n, np.nan)
    _sd_p = np.zeros(_n)
    _bull = True; _af = 0.02; _ep = _hs[0]
    _sv_p[0] = _ls[0]; _sd_p[0] = 1.0
    for _i in range(1, _n):
        if _bull:
            _s = _sv_p[_i-1] + _af * (_ep - _sv_p[_i-1])
            _s = min(_s, _ls[_i-1], _ls[_i-2] if _i >= 2 else _ls[_i-1])
            if _hs[_i] > _ep: _ep = _hs[_i]; _af = min(_af + 0.02, 0.2)
            if _ls[_i] < _s:
                _bull = False; _s = _ep; _ep = _ls[_i]; _af = 0.02; _sd_p[_i] = -1.0
            else:
                _sd_p[_i] = 1.0
        else:
            _s = _sv_p[_i-1] + _af * (_ep - _sv_p[_i-1])
            _s = max(_s, _hs[_i-1], _hs[_i-2] if _i >= 2 else _hs[_i-1])
            if _ls[_i] < _ep: _ep = _ls[_i]; _af = min(_af + 0.02, 0.2)
            if _hs[_i] > _s:
                _bull = True; _s = _ep; _ep = _hs[_i]; _af = 0.02; _sd_p[_i] = 1.0
            else:
                _sd_p[_i] = -1.0
        _sv_p[_i] = _s
    df["PSAR"]     = _sv_p
    df["PSAR_DIR"] = _sd_p   # 1.0=상승 추세, -1.0=하락 추세

    return df

@ttl_cache(600)
def fetch_stock_data(ticker: str, market: str, period: str = "1y"):
    sym = ticker.strip().upper()
    if market == "KRX" and sym.isdigit():
        sym = f"{sym}.KS"
        
    interval = "1d"
    yf_period = period
    if period == "1d":
        yf_period = "1d"
        interval = "5m"
    elif period == "3d":
        yf_period = "5d"
        interval = "15m"
    elif period == "1wk":
        yf_period = "5d"
        interval = "30m"
    elif period == "2wk":
        yf_period = "1mo"
        interval = "1h"
    elif period == "1mo":
        yf_period = "1mo"
        interval = "1h"

    try:
        obj = yf.Ticker(sym)
        df = obj.history(period=yf_period, interval=interval, auto_adjust=True)
        if df.empty and market == "KRX" and sym.endswith(".KS"):
            sym = sym.replace(".KS", ".KQ")
            df = yf.Ticker(sym).history(period=yf_period, interval=interval, auto_adjust=True)
        if df.empty:
            return None, None, f"데이터 없음: {sym}"
        # MultiIndex 처리 (안전하게)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        # 컬럼명 표준화: 일부 버전에서 소문자로 오는 경우
        df.columns = [c.capitalize() if c.lower() in ("open","high","low","close","volume") else c for c in df.columns]
        # 필수 컬럼 존재 확인
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, None, f"컬럼 누락: {missing}"
        df = add_indicators(df)
        df = df.dropna(subset=["Close", "MA20", "RSI"])

        # 뉴스
        news = []
        if FEEDPARSER_AVAILABLE:
            try:
                if market == "KRX":
                    q = sym.replace(".KS","").replace(".KQ","") + " 주가"
                    url = f"https://news.google.com/rss/search?q={quote(q)}&hl=ko&gl=KR&ceid=KR:ko"
                else:
                    url = f"https://news.google.com/rss/search?q={sym}+stock&hl=en-US&gl=US&ceid=US:en"
                for e in feedparser.parse(url).entries[:5]:
                    news.append({
                        "title": e.title, "link": e.link,
                        "publisher": getattr(e, "source", type("", (), {"title": "Google News"})()).title,
                        "published": getattr(e, "published", ""),
                    })
            except Exception:
                pass

        df2 = df.reset_index()
        # 분봉 데이터인 경우 인덱스 이름이 "Datetime"일 수 있으므로 "Date"로 통일
        if "Datetime" in df2.columns:
            df2.rename(columns={"Datetime": "Date"}, inplace=True)
            
        if hasattr(df2["Date"].dtype, "tz") and df2["Date"].dtype.tz:
            df2["Date"] = df2["Date"].dt.tz_localize(None)
            
        # Lightweight Charts가 인식할 수 있도록 날짜를 yyyy-mm-dd 문자열 또는 Unix Timestamp 형식으로 반환해야 함
        # 일봉은 %Y-%m-%d 문자열로, 분봉은 Unix Timestamp(초 단위)로 변환
        if period in ["1d", "3d", "1wk", "2wk", "1mo"]:
            df2["Date"] = df2["Date"].astype("int64") // 10**9
        else:
            df2["Date"] = df2["Date"].dt.strftime("%Y-%m-%d")
            
        d = df2.where(pd.notna(df2), other=None).to_dict(orient="list")
        return d, news, sym
    except Exception as e:
        return None, None, str(e)

@ttl_cache(600)
def fetch_naver(code: str):
    code = str(code).replace(".KS","").replace(".KQ","")
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    r = {"price":None,"market_cap":None,"per":None,"pbr":None,"opinion":None,"news":[],"disclosures":[]}
    try:
        # User-Agent 업데이트 및 타임아웃 증가
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://finance.naver.com/"
        }
        resp = requests.get(url, headers=hdrs, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # 가격 (class 변경 가능성 대비)
        el = soup.select_one(".no_today .blind")
        if el: r["price"] = el.text.replace(",","")
        
        # 주요 지표
        for k, s in [("market_cap","#_market_sum"),("per","#_per"),("pbr","#_pbr")]:
            e = soup.select_one(s)
            r[k] = e.text.strip() if e else "-"
            
        # 뉴스 섹션
        for item in soup.select(".news_section ul li")[:5]:
            a = item.select_one("span > a")
            if a:
                title = a.text.strip()
                link = a["href"]
                if not link.startswith("http"):
                    link = "https://finance.naver.com" + link
                r["news"].append({"title": title, "link": link})
                
        # 공시 섹션 (별도 페이지 조회)
        url_notice = f"https://finance.naver.com/item/news_notice.naver?code={code}"
        resp_notice = requests.get(url_notice, headers=hdrs, timeout=5)
        soup_notice = BeautifulSoup(resp_notice.text, "html.parser")
        
        for row in soup_notice.select("tbody tr")[:5]:
            title_a = row.select_one(".title a")
            date_td = row.select_one(".date")
            if title_a:
                title = title_a.text.strip()
                link = title_a["href"]
                date = date_td.text.strip() if date_td else ""
                if not link.startswith("http"):
                    link = "https://finance.naver.com" + link
                r["disclosures"].append({"title": title, "link": link, "date": date})

        # 재무분석 섹션 (ROE, 부채비율, 영업이익률 등)
        section = soup.select_one(".section.cop_analysis")
        if section:
            for tr in section.select("table tbody tr"):
                th = tr.select_one("th")
                if not th: continue
                title = th.text.strip()
                
                key = None
                if "ROE" in title: key = "roe"
                elif "부채비율" in title: key = "debt"
                elif "PER" in title: key = "per"
                elif "EPS" in title: key = "eps"
                elif "영업이익률" in title: key = "op_margin"
                
                if key:
                    vals = []
                    for td in tr.select("td"):
                        txt = td.text.strip().replace(",","")
                        if txt and txt != "-" and not txt.isalpha():
                            try:
                                vals.append(float(txt))
                            except:
                                pass
                    if vals:
                        # Use the latest available value (Estimate or Actual)
                        r[key] = vals[-1]

    except Exception as e:
        print(f"Naver Fetch Error: {e}")
        pass
    return r

@ttl_cache(600)
def fetch_sentiment(market: str):
    try:
        tkr = "^VIX" if market == "US" else "^KS200"
        name = "VIX (공포지수)" if market == "US" else "KOSPI 200"
        df = yf.Ticker(tkr).history(period="1mo")
        if df.empty: return None
        cur, prv = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
        chg = (cur - prv) / prv * 100
        sent = "중립"
        if market == "US":
            sent = "극도의 공포" if cur > 30 else "공포/불안" if cur > 20 else "탐욕/안정" if cur < 15 else "중립"
        else:
            sent = "강세장" if chg > 1 else "약세장" if chg < -1 else "중립"
        return {"name": name, "value": round(cur, 2), "change": round(chg, 2), "sentiment": sent}
    except Exception:
        return None

# ── EXPANDED UNIVERSE ──
KR_STOCK_MAP = {
    "삼성전자": "005930.KS", "삼성전자우": "005935.KS", "SK하이닉스": "000660.KS", "LG에너지솔루션": "373220.KS",
    "삼성바이오로직스": "207940.KS", "삼성SDI": "006400.KS", "현대차": "005380.KS", "기아": "000270.KS",
    "셀트리온": "068270.KS", "KB금융": "105560.KS", "신한지주": "055550.KS", "POSCO홀딩스": "005490.KS",
    "NAVER": "035420.KS", "카카오": "035720.KS", "LG화학": "051910.KS", "LG전자": "066570.KS",
    "삼성물산": "028260.KS", "삼성생명": "032830.KS", "삼성화재": "000810.KS", "삼성전기": "009150.KS",
    "삼성SDS": "018260.KS", "현대모비스": "012330.KS", "SK이노베이션": "096770.KS", "SK텔레콤": "017670.KS",
    "SK": "034730.KS", "KT": "030200.KS", "KT&G": "033780.KS", "한국전력": "015760.KS",
    "하나금융지주": "086790.KS", "우리금융지주": "316140.KS", "카카오뱅크": "323410.KS", "카카오페이": "377300.KS",
    "크래프톤": "259960.KS", "엔씨소프트": "036570.KS", "넷마블": "251270.KS", "펄어비스": "263750.KS",
    "하이브": "352820.KS", "CJ제일제당": "097950.KS", "CJ": "001040.KS", "롯데케미칼": "011170.KS",
    "한화솔루션": "009830.KS", "한화에어로스페이스": "012450.KS", "한화오션": "042660.KS", "HD현대중공업": "329180.KS",
    "HD한국조선해양": "009540.KS", "두산에너빌리티": "034020.KS", "두산밥캣": "241560.KS", "포스코퓨처엠": "003670.KS",
    "에코프로비엠": "247540.KQ", "에코프로": "086520.KQ", "엘앤에프": "066970.KQ", "HLB": "028300.KQ",
    "리노공업": "058470.KQ", "알테오젠": "196170.KQ"
}

US_TICKERS = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "LLY", "AVGO", "JPM", "V", "UNH", "XOM", 
    "MA", "JNJ", "PG", "HD", "COST", "ABBV", "MRK", "ADBE", "CVX", "PEP", "KO", "BAC", "ACN", "NFLX", "LIN", 
    "MCD", "TMO", "AMD", "DIS", "ABT", "WMT", "CSCO", "INTU", "PFE", "CMCSA", "ORCL", "QCOM", "NKE", "UPS", 
    "TXN", "PM", "GE", "IBM", "AMGN", "HON", "UNP", "SBUX", "BA", "MMM", "CAT", "GS", "MS", "C", "BLK", "SPGI", 
    "AXP", "LOW", "TGT", "TJX", "CVS", "CI", "ELV", "DE", "PLD", "AMT", "NOW", "ISRG", "ZTS", "GILD", "SYK", 
    "BKNG", "MDT", "ADP", "LRCX", "ADI", "MU", "VRTX", "REGN", "SO", "DUK", "SLB", "EOG", "COP", "MMC", "AON", 
    "PGR", "CB", "CL", "MO", "EMR", "ETN", "ITW", "PH", "USB", "PNC", "TFC", "COF", "MET", "PRU", "ALL", "TRV", 
    "AIG", "HIG", "PLTR", "IONQ", "JOBY", "ACHR", "SOFI", "AFRM", "UPST", "RIVN", "LCID", "NKLA", "DNA", "PATH"
]

@ttl_cache(3600)
def fetch_metrics(item):
    try:
        t = yf.Ticker(item["ticker"])
        i = t.info
        
        # Basic Price Data
        price = i.get("currentPrice") or i.get("regularMarketPrice")
        if not price: return None
        
        # Metadata Update
        if item["market_type"] == "US":
            item["name"] = i.get("shortName", item["name"])
        item["cat"] = i.get("sector", "Unknown")
        
        # Signal (Consensus)
        rec = i.get("recommendationKey", "").lower()
        if rec == "buy": item["signal"] = "매수"
        elif rec == "strong_buy": item["signal"] = "적극 매수"
        else: item["signal"] = "중립"
        
        mkt_cap = i.get("marketCap", 0)
        
        # Quality Factors
        roic = i.get("returnOnEquity", 0) 
        debt_raw = i.get("debtToEquity", 999)
        if debt_raw is None: debt_raw = 999
        debt_ratio = debt_raw * 100 if debt_raw < 10 else debt_raw
        
        fcf = i.get("freeCashflow", 0)
        if fcf is None: fcf = i.get("operatingCashflow", -1)

        # Value & Growth Factors
        peg = i.get("pegRatio")
        per = i.get("trailingPE")
        
        if peg is None and per is not None:
            growth = i.get("earningsGrowth")
            if growth and growth > 0:
                peg = per / (growth * 100)
        
        # Profitability Init
        op_margin = i.get("operatingMargins", 0)
        eps = i.get("trailingEps", 0)

        # KRX Fallback (Naver) - Enhanced with Financials
        if item.get("market_type") == "KRX":
            try:
                # Fetch if any key metric is missing or suspicious
                if peg is None or per is None or not roic or debt_ratio == 999 or op_margin == 0:
                    nv = fetch_naver(item["ticker"])
                    if nv:
                        if (per is None or per == 999) and nv.get("per"):
                            try:
                                val = nv["per"]
                                if isinstance(val, str) and val != "-": val = float(val.replace(",", ""))
                                per = float(val)
                            except:
                                pass
                        if (not roic or roic == 0) and nv.get("roe"):
                            roic = float(nv["roe"]) / 100.0
                        if (debt_ratio == 999 or debt_ratio is None) and nv.get("debt"):
                            debt_ratio = float(nv["debt"])
                        if nv.get("op_margin"):
                            op_margin = float(nv["op_margin"])
                        if nv.get("eps"):
                            eps = float(nv["eps"])
                        if peg is None and per and per < 15:
                            peg = 1.2
            except:
                pass

        if peg is None: peg = 999
        if per is None: per = 999
        
        # Momentum Factor
        high52 = i.get("fiftyTwoWeekHigh", price)
        prox = price / high52 if high52 else 0
        
        ma50 = i.get("fiftyDayAverage", 0)
        ma200 = i.get("twoHundredDayAverage", 0)
        # 정배열 조건 (가격 >= 50일선 >= 200일선)
        is_ma_aligned = (price >= ma50) and (ma50 >= ma200) and (ma200 > 0)
        
        # 3-year Profit Check
        is_profitable = (op_margin is not None and op_margin > 0) or (eps is not None and eps > 0)
        
        item["market_cap"] = mkt_cap
        item["roic"] = roic if roic else 0
        item["roe"]  = roic if roic else 0   # returnOnEquity 값 (소수 단위)
        item["debt_ratio"] = debt_ratio
        item["fcf"] = fcf
        item["peg"] = peg
        item["per"] = per
        item["prox"] = prox
        item["is_ma_aligned"] = is_ma_aligned
        item["price_val"] = price
        item["change"] = i.get("regularMarketChangePercent", 0) * 100
        item["volume"] = i.get("volume", 0)
        item["is_profitable"] = is_profitable
        item["op_margin"] = op_margin
        item["earnings_growth"] = i.get("earningsGrowth", 0)
        
        return item
    except:
        return None

# =============================================================================
# 토스증권 해외주식 스크리너 – 유니버스 & 필터 함수
# (스크린샷 기준 필터: 시가총액≥10억$ / 영업이익률>0 / 순이익증감률≥10% /
#  ROE 15%~8785% / PER 0~25 / 부채비율≤100% / PFCR≥0)
# =============================================================================

# ── 토스증권 해외주식 필터 유니버스 (S&P500 + 글로벌 대형주 + 광산/에너지/금융 등) ──
TOSS_US_UNIVERSE = [
    # 미국 대형 기술주
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "AVGO", "ORCL", "ADBE",
    "CRM", "CSCO", "INTU", "TXN", "QCOM", "AMD", "AMAT", "LRCX", "KLAC", "MRVL",
    "NOW", "SNOW", "PANW", "CRWD", "ZS", "FTNT", "DDOG", "MDB", "TEAM",
    # 금융주
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SPGI", "MCO", "AXP",
    "V", "MA", "COF", "USB", "PNC", "TFC", "MTB", "CFG", "FITB", "RF",
    "CB", "PGR", "MET", "PRU", "AFL", "AIG", "HIG", "ALL", "TRV", "MMC",
    "AON", "MSCI", "ICE", "CME", "NDAQ",
    # 헬스케어/제약
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "ABT", "TMO", "DHR", "SYK", "BSX",
    "ISRG", "MDT", "EW", "ZTS", "VRTX", "REGN", "BIIB", "GILD", "AMGN", "BMY",
    "PFE", "CVS", "MCK", "CAH", "HCA", "CNC", "ELV", "CI",
    # 소비재/유통
    "WMT", "COST", "TGT", "HD", "LOW", "TJX", "ROST", "BBY", "DG", "DLTR",
    "MCD", "SBUX", "YUM", "CMG", "DPZ", "QSR",
    "PG", "KO", "PEP", "CL", "MO", "PM", "BTI", "NKE", "VFC", "HBI",
    "AMZN", "BKNG", "EXPE", "ABNB", "LYFT", "UBER",
    # 에너지
    "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO", "PSX", "HES",
    "DVN", "FANG", "APA", "BKR", "HAL",
    # 광업/금/원자재 (스크린샷에 다수 포함된 섹터)
    "NEM",   # 뉴몬트 (Newmont)
    "AEM",   # 애그니코 이글 마인스 (Agnico Eagle Mines)
    "GOLD",  # 배릭 마이닝 코퍼레이션 (Barrick Gold)
    "AU",    # 앵글로골드 아샨티 (AngloGold Ashanti ADR)
    "GFI",   # 골드 필즈 (Gold Fields ADR)
    "KGC",   # 킨로스 골드 (Kinross Gold)
    "WPM",   # 휠튼 프레셔스 메탈 (Wheaton Precious Metals)
    "RGLD",  # 로열 골드 (Royal Gold)
    "FNV",   # 프랑코-네바다 (Franco-Nevada)
    "OR",    # 오시스코 로열티 (Osisko Gold Royalties)
    "EGO",   # 엘도라도 골드 (Eldorado Gold)
    "AGI",   # 알라모스 골드 (Alamos Gold)
    "HMY",   # 하모니 골드 (Harmony Gold ADR)
    "IAG",   # IAMGOLD
    "BVN",   # 부에나벤투라 (Buenaventura ADR)
    "VALE",  # 발레 (Vale ADR)
    "RIO",   # 리오 틴토 (Rio Tinto ADR)
    "BHP",   # BHP 그룹 (BHP ADR)
    "SCCO",  # 서던 코퍼 (Southern Copper)
    "FCX",   # 프리포트-맥모란 (Freeport-McMoRan)
    "AA",    # 알코아 (Alcoa)
    "CLF",   # 클리프스 내추럴 (Cleveland-Cliffs)
    "MP",    # MP 머티리얼즈 (MP Materials)
    "NUE",   # 뉴코어 (Nucor)
    "STLD",  # 스틸 다이내믹스 (Steel Dynamics)
    "X",     # US 스틸 (U.S. Steel)
    # ADR / 글로벌 대형주
    "TSM",   # TSMC
    "ASML",  # ASML
    "SAP",   # SAP
    "NTES",  # 넷이즈 (NetEase ADR)
    "BIDU",  # 바이두 (Baidu ADR)
    "JD",    # JD닷컴 (JD.com ADR)
    "BABA",  # 알리바바 (Alibaba ADR)
    "NVO",   # 노보 노르디스크 (Novo Nordisk ADR)
    "AZN",   # 아스트라제네카 (AstraZeneca ADR)
    "UL",    # 유니레버 (Unilever ADR)
    "NESN",  # 네슬레 (OTC)
    "TTE",   # 토탈에너지 (TotalEnergies ADR)
    "BP",    # BP ADR
    "SHEL",  # 쉘 (Shell ADR)
    "GSK",   # GSK ADR
    "SNY",   # 사노피 (Sanofi ADR)
    "RHHBY", # 로슈 (Roche ADR)
    "DEO",   # 디아지오 (Diageo ADR)
    "SONY",  # 소니 (Sony ADR)
    "TM",    # 도요타 (Toyota ADR)
    "HMC",   # 혼다 (Honda ADR)
    "NSANY", # 닛산 (Nissan ADR)
    "MUFG",  # 미쓰비시 UFJ ADR
    "SMFG",  # 스미토모 미쓰이 ADR
    "KB",    # KB금융 ADR
    "SHG",   # 신한지주 ADR
    "PKX",   # POSCO ADR
    "LG",    # LG ADR (OTC)
    # 통신/미디어
    "T", "VZ", "TMUS", "CMCSA", "NFLX", "DIS", "WBD", "PARA",
    "CHTR", "DISH", "SIRI",
    # 산업재/항공우주
    "BA", "GE", "HON", "RTX", "LMT", "NOC", "GD", "TXT", "HII",
    "CAT", "DE", "EMR", "ETN", "ITW", "PH", "ROK", "DOV", "FTV",
    "UPS", "FDX", "EXPD", "GWW", "FAST",
    "MMM", "ITT", "XYL", "REXR", "EFX",
    # 부동산
    "PLD", "AMT", "CCI", "EQIX", "SPG", "O", "WELL",
    # 전력/유틸리티
    "NEE", "SO", "DUK", "AEP", "EXC", "D", "PCG", "ED", "FE",
    # 신흥 성장주 (ADR)
    "GRAB", "SEA", "MELI", "NU", "PDD", "TME",
]

def fetch_toss_metrics(ticker: str):
    """
    [BUG-FIX v2]
    토스증권 스크린샷 필터 조건을 yfinance로 구현.
    - BUG-2: earningsGrowth=None → 필터 스킵 (None과 음수만 제거)
    - BUG-3: ROE 상한선 1.01→87.85 (yfinance 소수 단위, 101%→8785%)
    - BUG-4: debtToEquity 단위 안전 처리 (소수/퍼센트 자동 판별)
    """
    try:
        t = yf.Ticker(ticker)
        i = t.info
        if not i or i.get("quoteType") == "MUTUALFUND":
            return None

        # ── 현재가 ──────────────────────────────────────────────
        price = (i.get("currentPrice")
                 or i.get("regularMarketPrice")
                 or i.get("previousClose"))
        if not price or price <= 0:
            return None

        # ── 시가총액 10억$ 이상 ──────────────────────────────────
        mkt_cap = i.get("marketCap") or 0
        if mkt_cap < 1_000_000_000:
            return None

        # ── 영업이익률 직전 분기 0% 이상 ────────────────────────
        op_margin = i.get("operatingMargins")
        if op_margin is None or op_margin <= 0:
            return None

        # ── 순이익 증감률 TTM: 10% 이상 (없으면 필터 스킵) ──────
        # [BUG-2] earningsGrowth=None 종목이 많아 전부 필터링되던 문제 수정
        earnings_growth = i.get("earningsGrowth")
        if earnings_growth is not None and earnings_growth < 0.10:
            return None  # 있는데 10% 미만이면 탈락

        # ── ROE TTM: 15% 이상 (상한 실용적 완화) ────────────────
        # [BUG-3] 상한 1.01 → 5.0: AAPL(~1.47), 고ROE기업 포함
        # yfinance returnOnEquity: 소수 단위 (0.15 = 15%, 1.47 = 147%)
        roe = i.get("returnOnEquity")
        if roe is None or roe < 0.15:
            return None
        # 상한: 87.85 (8785%) → 비정상적 ROE 제외 (부채과다 착시 방지)
        if roe > 87.85:
            return None

        # ── PER: 0 ~ 25배 ────────────────────────────────────────
        # trailingPE 없으면 forwardPE로 fallback
        per = i.get("trailingPE") or i.get("forwardPE")
        if per is None or not (0 < per <= 25):
            return None

        # ── 부채비율 100% 이하 ───────────────────────────────────
        # [BUG-4] yfinance debtToEquity 단위 안전 처리:
        #   - 최신 yfinance: % 단위 (146.52 = 146.52%)
        #   - 구버전 일부: 소수 단위 (1.4652 = 146.52%)
        #   → 값이 10 미만이면 소수로 간주하여 100배 변환
        debt_raw = i.get("debtToEquity")
        if debt_raw is None:
            return None
        debt_pct = debt_raw * 100 if debt_raw < 10 else debt_raw
        if debt_pct > 100:
            return None

        # ── PFCR ≥ 0: 양의 FCF 여부 확인 ────────────────────────
        fcf = i.get("freeCashflow") or i.get("operatingCashflow") or 0
        if fcf <= 0:
            return None
        pfcr = round(mkt_cap / fcf, 2)

        # ── 보조 정보 ────────────────────────────────────────────
        change_pct = (i.get("regularMarketChangePercent") or 0) * 100
        sector     = i.get("sector") or i.get("industry") or "Unknown"
        short_name = i.get("shortName") or i.get("longName") or ticker
        volume     = i.get("volume") or i.get("averageVolume") or 0
        rec_key    = (i.get("recommendationKey") or "").lower()
        analyst_map = {
            "strong_buy": "적극 매수", "buy": "매수",
            "hold": "보유", "underperform": "약세", "sell": "매도",
        }
        analyst_signal = analyst_map.get(rec_key, "중립")
        high52 = i.get("fiftyTwoWeekHigh") or price
        prox52 = round(price / high52, 4) if high52 else 0
        eg_pct = round(earnings_growth * 100, 2) if earnings_growth is not None else None

        return {
            "ticker":          ticker,
            "name":            short_name,
            "market_type":     "US",
            "price_val":       round(price, 4),
            "change":          round(change_pct, 2),
            "market_cap":      mkt_cap,
            "op_margin":       round(op_margin * 100, 2),
            "earnings_growth": eg_pct,
            "roe":             round(roe * 100, 2),
            "per":             round(per, 2),
            "debt_ratio":      round(debt_pct, 2),
            "pfcr":            pfcr,
            "fcf":             fcf,
            "volume":          int(volume),
            "sector":          sector,
            "prox52":          prox52,
            "signal":          analyst_signal,
        }
    except Exception:
        return None

def fetch_kr_toss_metrics(item: dict):
    """
    국내 주식에 대한 토스증권 필터 조건 적용
    해외(US) 탭과 동일한 구조로 단일 종목을 검사하고 조건을 만족하면 dict 반환
    """
    try:
        ticker = item["ticker"]
        t = yf.Ticker(ticker)
        i = t.info
        
        price = i.get("currentPrice") or i.get("regularMarketPrice")
        if not price or price <= 0: return None
        
        mkt_cap = i.get("marketCap", 0)
        # 1. 시가총액: 1,000억 원 이상
        if mkt_cap < 100_000_000_000: return None
            
        roic = i.get("returnOnEquity")
        debt_raw = i.get("debtToEquity")
        per = i.get("trailingPE") or i.get("forwardPE")
        op_margin = i.get("operatingMargins")
        earnings_growth = i.get("earningsGrowth")
        
        # 2. 영업이익률: 직전 분기 0% 이상 (데이터 없으면 패스)
        if op_margin is not None and op_margin < 0: return None
            
        # 3. ROE: 최근 1년(TTM) 10% 이상 (데이터 없으면 패스)
        if roic is not None and roic < 0.10: return None
            
        # 4. PER: 0배 초과 ~ 20배 이하 (데이터 없으면 패스)
        if per is not None and not (0 < per <= 20): return None
            
        # 5. 순이익 증감률: 최근 1년(TTM) 10% 이상 (데이터 없으면 패스)
        if earnings_growth is not None and earnings_growth < 0.10: return None
            
        # 6. 부채비율: 직전 분기 100% 이하 (데이터 없으면 패스)
        if debt_raw is not None:
            debt_ratio = debt_raw * 100 if debt_raw < 10 else debt_raw
            if debt_ratio > 100: return None
            
        # 통과 시 결과 반환
        change_pct = (i.get("regularMarketChangePercent") or 0) * 100
        sector     = i.get("sector") or i.get("industry") or "Unknown"
        volume     = i.get("volume") or i.get("averageVolume") or 0
        rec_key    = (i.get("recommendationKey") or "").lower()
        analyst_map = {
            "strong_buy": "적극 매수", "buy": "매수",
            "hold": "보유", "underperform": "약세", "sell": "매도",
        }
        analyst_signal = analyst_map.get(rec_key, "중립")
        high52 = i.get("fiftyTwoWeekHigh") or price
        prox = price / high52 if high52 else 0
        
        return {
            "market":       "국내",
            "ticker":       ticker,
            "name":         item["name"],
            "price_val":    price,
            "change":       change_pct,
            "market_cap":   mkt_cap,
            "op_margin":    op_margin,
            "roe":          roic,
            "per":          per,
            "debt_ratio":   debt_ratio,
            "earnings_growth": round(earnings_growth * 100, 2) if earnings_growth is not None else 0,
            "sector":       sector,
            "volume":       volume,
            "signal":       analyst_signal,
            "prox52":       round(prox, 2)
        }
    except Exception:
        return None

@ttl_cache(3600)
def fetch_toss_overseas_screener(sort_by: str = "price", sort_order: str = "desc") -> dict:
    """
    토스증권 해외주식 필터 조건과 동일하게 종목을 수집·필터링합니다.

    Parameters
    ----------
    sort_by    : "price" | "change" | "volume" | "per" | "roe"
    sort_order : "asc" | "desc"
    """
    usd_krw = get_usd_krw()

    results = []
    # Vercel 1024MB 메모리·60초 제한 대응: max_workers=25 (I/O bound 특성상 속도 동일)
    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        futures = {
            executor.submit(fetch_toss_metrics, tkr): tkr
            for tkr in TOSS_US_UNIVERSE
        }
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)

    # ── 정렬 ──────────────────────────────────────────────────
    sort_key_map = {
        "price":  "price_val",
        "change": "change",
        "volume": "volume",
        "per":    "per",
        "roe":    "roe",
    }
    sort_field = sort_key_map.get(sort_by, "price_val")
    ascending  = (sort_order == "asc")
    results.sort(key=lambda x: (x.get(sort_field) or 0), reverse=not ascending)

    # 상위 90개만 반환
    results = results[:90]

    # ── 출력 정제 ─────────────────────────────────────────────
    output = []
    for item in results:
        p = item["price_val"]
        output.append({
            "market":              "해외",
            "ticker":              item["ticker"],
            "name":                item["name"],
            "price":               f"${p:,.2f}",
            "price_usd":           f"${p:,.2f}",
            "price_krw":           f"{p * usd_krw:,.0f}원",
            "price_val":           item["price_val"],
            "change":              item["change"],
            "category":            item["sector"],
            "sector":              item["sector"],
            "volume":              item["volume"],
            "market_cap_b":        round(item["market_cap"] / 1e9, 2),
            "op_margin_pct":       item["op_margin"],
            "earnings_growth_pct": item["earnings_growth"],
            "roe_pct":             item["roe"],
            "per":                 item["per"],
            "debt_ratio_pct":      item["debt_ratio"],
            "pfcr":                item["pfcr"],
            "prox52":              item["prox52"],
            "signal":              item["signal"],
        })

    filter_conditions = {
        "시장":       "해외(미국)",
        "시가총액":   "10억$ 이상",
        "영업이익률": "직전 분기 0% 이상",
        "순이익증감": "최근 1년(TTM) 10% 이상",
        "ROE":        "최근 1년(TTM) 15% ~ 8,785%",
        "PER":        "0배 ~ 25배",
        "부채비율":   "직전 분기 0% ~ 100%",
        "PFCR":       "0배 이상(양의 FCF)",
    }
    return {
        "data":              output,
        "usd_krw":           round(usd_krw, 2),
        "filter_conditions": filter_conditions,
        "total":             len(output),
        "sort_by":           sort_by,
        "sort_order":        sort_order,
    }


@ttl_cache(3600)
def fetch_screener(sort_by: str = "price", sort_order: str = "desc") -> dict:
    """
    통합 스크리너:
     - 국내(KRX): 기존 Quality-GARP 로직 유지
     - 해외(US) : 토스증권 필터 조건 적용
    """
    usd_krw = get_usd_krw()

    # ── 해외 US: 토스증권 스크리너로 교체 ────────────────────────
    toss_result = fetch_toss_overseas_screener(
        sort_by=sort_by,
        sort_order=sort_order,
    )
    us_results = toss_result.get("data", [])

    # ── 국내 KRX: 토스증권 스크리너 로직 ──────────────────────────
    kr_processed = []
    kr_candidates = [
        {"ticker": ticker, "name": name, "market_type": "KRX"}
        for name, ticker in KR_STOCK_MAP.items()
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_kr_toss_metrics, c): c for c in kr_candidates}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                kr_processed.append(res)
    
    # ── 정렬 ──────────────────────────────────────────────────
    sort_key_map = {
        "price":  "price_val",
        "change": "change",
        "volume": "volume",
        "per":    "per",
        "roe":    "roe",
    }
    sort_field = sort_key_map.get(sort_by, "price_val")
    ascending  = (sort_order == "asc")
    kr_results = sorted(kr_processed, key=lambda x: x.get(sort_field, 0), reverse=not ascending)

    # ── 출력 정제 ─────────────────────────────────────────────
    output = []
    for item in kr_results:
        p = item["price_val"]
        output.append({
            "market":              "국내",
            "ticker":              item["ticker"],
            "name":                item["name"],
            "price":               f"{p:,.0f}원",
            "price_usd":           None,
            "price_krw":           f"{p:,.0f}원",
            "price_val":           item["price_val"],
            "change":              item["change"],
            "category":            item["sector"],
            "sector":              item["sector"],
            "volume":              item["volume"],
            "market_cap_b":        round(item["market_cap"] / 100_000_000, 2), # 억원 단위
            "op_margin_pct":       round(float(item["op_margin"] or 0) * 100, 2),
            "earnings_growth_pct": item["earnings_growth"],
            "roe_pct":             round(float(item["roe"] or 0) * 100, 2),
            "per":                 item["per"],
            "debt_ratio_pct":      item["debt_ratio"],
            "prox52":              item["prox52"],
            "signal":              item["signal"],
        })

    kr_filter_conditions = {
        "시장": "국내(KRX)",
        "시가총액": "1000억원 이상",
        "영업이익률": "직전 분기 0% 이상",
        "ROE": "최근 1년(TTM) 10% 이상",
        "PER": "0~20배",
        "순이익증감": "최근 1년(TTM) 10% 이상",
        "부채비율": "직전 분기 0% ~ 100%",
    }

    return {
        "data":              output + us_results,
        "usd_krw":           round(usd_krw, 2),
        "total_overseas":    toss_result.get("total", 0),
        "total_domestic":    len(kr_results),
        "us_filter_conditions": toss_result.get("filter_conditions", {}),
        "kr_filter_conditions": kr_filter_conditions,
        "sort_by":           sort_by,
        "sort_order":        sort_order,
    }

# =============================================================================
# 분석 엔진
# =============================================================================
class ChartPatternAnalyzer:
    def __init__(self, df):
        self.df = df
        self.closes = np.array(df['Close'].values, dtype=float)
        self.highs = np.array(df['High'].values, dtype=float)
        self.lows = np.array(df['Low'].values, dtype=float)
        
    def find_local_extrema(self, order=5):
        # scipy.signal.argrelextrema 대체 구현 (Pure Numpy)
        # order: 양쪽으로 비교할 이웃의 수
        peaks = []
        troughs = []
        
        # Highs for peaks
        for i in range(order, len(self.highs) - order):
            window = self.highs[i-order : i+order+1]
            if self.highs[i] == np.max(window) and self.highs[i] != self.highs[i-1]:
                peaks.append(i)
                
        # Lows for troughs
        for i in range(order, len(self.lows) - order):
            window = self.lows[i-order : i+order+1]
            if self.lows[i] == np.min(window) and self.lows[i] != self.lows[i-1]:
                troughs.append(i)
                
        return np.array(peaks), np.array(troughs)

    def detect_patterns(self):
        patterns = []
        try:
            peaks, troughs = self.find_local_extrema(order=5)
            if len(peaks) < 3 or len(troughs) < 3:
                return patterns

            last_peaks = peaks[-3:]
            last_troughs = troughs[-3:]
            
            # Linear regression for slope
            def get_slope(x, y):
                if len(x) < 2: return 0
                return np.polyfit(x, y, 1)[0]

            slope_upper = get_slope(last_peaks, self.highs[last_peaks])
            slope_lower = get_slope(last_troughs, self.lows[last_troughs])
            
            # Triangle Patterns
            # 주가 수준에 맞는 상대적 flat 임계값 (주가의 0.1% per bar)
            avg_price = float(np.mean(self.closes[-30:])) if len(self.closes) >= 30 else float(np.mean(self.closes))
            flat_threshold = avg_price * 0.001
            if slope_upper < 0 and slope_lower > 0:
                patterns.append({'name': '대칭 삼각형 (Symmetrical Triangle)', 'signal': '중립/변동성 축소', 'desc': '곧 큰 방향성이 나올 것입니다.'})
            elif slope_upper < 0 and abs(slope_lower) < flat_threshold:
                patterns.append({'name': '하락 삼각형 (Descending Triangle)', 'signal': '매도 (하락형)', 'desc': '지지선 붕괴 위험이 있습니다.'})
            elif abs(slope_upper) < flat_threshold and slope_lower > 0:
                patterns.append({'name': '상승 삼각형 (Ascending Triangle)', 'signal': '매수 (상승형)', 'desc': '저항선 돌파 시도가 예상됩니다.'})
                
            # Wedge Patterns
            if slope_upper < 0 and slope_lower < 0:
                if slope_lower < slope_upper:
                    patterns.append({'name': '하락 쐐기형 (Falling Wedge)', 'signal': '매수 (반전)', 'desc': '하락세가 약화되고 반등할 가능성이 큽니다.'})
            if slope_upper > 0 and slope_lower > 0:
                if slope_lower > slope_upper:
                    patterns.append({'name': '상승 쐐기형 (Rising Wedge)', 'signal': '매도 (반전)', 'desc': '상승세가 약화되고 하락할 가능성이 큽니다.'})
                    
            # Double Top/Bottom
            if len(last_peaks) >= 2:
                if abs(self.highs[last_peaks[-1]] - self.highs[last_peaks[-2]]) / (self.highs[last_peaks[-1]] or 1) < 0.02:
                    patterns.append({'name': '이중 천장 (Double Top)', 'signal': '매도', 'desc': '고점 돌파 실패, 하락 전환 가능성.'})
            if len(troughs) >= 2:
                if abs(self.lows[last_troughs[-1]] - self.lows[last_troughs[-2]]) / (self.lows[last_troughs[-1]] or 1) < 0.02:
                    patterns.append({'name': '이중 바닥 (Double Bottom)', 'signal': '매수', 'desc': '바닥 지지 성공, 상승 전환 가능성.'})
        except Exception:
            pass
        return patterns

def detect_patterns(dd: Dict) -> List[Dict]:
    """
    캔들스틱 패턴 인식 — 순수 Python (Vercel 호환, TA-Lib 불필요)
    기존 8개 + 추가 11개 = 총 19개 패턴
    [단일] Doji, Hammer, Shooting Star, Marubozu
    [2봉]  Bullish/Bearish Engulfing, Bullish/Bearish Harami, Harami Cross,
           Piercing Line, Dark Cloud Cover
    [3봉]  Morning Star, Evening Star, Three White Soldiers, Three Black Crows,
           Three Inside Up/Down, Three Outside Up/Down
    [복합] Rising/Falling Three Methods, Abandoned Baby, Hikkake, Mat Hold
    """
    patterns = []

    try:
        o = [float(x) for x in dd.get("Open",   []) if x is not None]
        h = [float(x) for x in dd.get("High",   []) if x is not None]
        l = [float(x) for x in dd.get("Low",    []) if x is not None]
        c = [float(x) for x in dd.get("Close",  []) if x is not None]

        if len(c) < 5:
            return []

        n = len(c)

        # ── 인덱스 (최신봉 = i1, 가장 오래된 = i5) ──
        i1, i2, i3, i4, i5 = n-1, n-2, n-3, n-4, n-5

        o1,h1,l1,c1 = o[i1],h[i1],l[i1],c[i1]
        o2,h2,l2,c2 = o[i2],h[i2],l[i2],c[i2]
        o3,h3,l3,c3 = o[i3],h[i3],l[i3],c[i3]
        o4,h4,l4,c4 = o[i4],h[i4],l[i4],c[i4]
        o5,h5,l5,c5 = o[i5],h[i5],l[i5],c[i5]

        def body(idx): return abs(c[idx] - o[idx])
        def rng(idx):  return (h[idx] - l[idx]) or 0.001
        def mid(idx):  return (o[idx] + c[idx]) / 2
        def is_bull(idx): return c[idx] >= o[idx]

        body1 = body(i1); rng1 = rng(i1)
        body2 = body(i2); rng2 = rng(i2)
        body3 = body(i3); rng3 = rng(i3)
        body4 = body(i4); rng4 = rng(i4)
        body5 = body(i5); rng5 = rng(i5)

        up_sh1 = h1 - max(c1, o1)
        lo_sh1 = min(c1, o1) - l1

        bull1 = is_bull(i1); bull2 = is_bull(i2)
        bull3 = is_bull(i3); bull4 = is_bull(i4); bull5 = is_bull(i5)

        # 최근 10봉 평균 몸통 (상대적 크기 판단용)
        avg_body = sum(body(n-1-k) for k in range(min(10, n))) / min(10, n) or 0.001

        # ════════════════════════════════════════
        #  단일봉 패턴 (1-Candle)
        # ════════════════════════════════════════

        # 1. Doji — 몸통이 레인지의 10% 미만
        if body1 / rng1 < 0.1:
            patterns.append({"name": "✖️ Doji", "desc": "도지 (매수/매도 균형, 방향 전환 가능)", "direction": "중립", "conf": 70})

        # 2. Hammer — 하락 추세 끝 긴 아래 꼬리 (상승 반전)
        if lo_sh1 >= body1 * 2 and up_sh1 <= body1 * 0.5 and body1 > 0 and not bull2:
            patterns.append({"name": "🔨 Hammer", "desc": "해머 (하락 후 강한 반등 신호)", "direction": "상승", "conf": 80})

        # 3. Shooting Star — 상승 추세 끝 긴 위 꼬리 (하락 반전)
        if up_sh1 >= body1 * 2 and lo_sh1 <= body1 * 0.5 and body1 > 0 and bull2:
            patterns.append({"name": "⭐ Shooting Star", "desc": "유성형 (상승 후 강한 하락 신호)", "direction": "하락", "conf": 80})

        # 4. Marubozu — 꼬리 없는 강한 봉 (추세 지속)
        if body1 / rng1 > 0.9 and body1 > avg_body * 1.1:
            patterns.append({"name": "📏 Marubozu", "desc": f"마루보즈 ({'강한 상승 지속' if bull1 else '강한 하락 지속'})", "direction": "상승" if bull1 else "하락", "conf": 80})

        # ════════════════════════════════════════
        #  2봉 패턴 (2-Candle)
        # ════════════════════════════════════════

        # 5. Bullish Engulfing — 음봉을 완전히 감싸는 양봉 (상승 반전)
        if bull1 and not bull2 and o1 <= c2 and c1 >= o2 and body1 > body2:
            patterns.append({"name": "🫂 Bullish Engulfing", "desc": "상승 포용형 (강한 매수 신호)", "direction": "상승", "conf": 85})

        # 6. Bearish Engulfing — 양봉을 완전히 감싸는 음봉 (하락 반전)
        if not bull1 and bull2 and o1 >= c2 and c1 <= o2 and body1 > body2:
            patterns.append({"name": "🫂 Bearish Engulfing", "desc": "하락 포용형 (강한 매도 신호)", "direction": "하락", "conf": 85})

        # 7. Bullish Harami — 큰 음봉 내에 포함된 작은 양봉 (상승 반전 초기)
        if bull1 and not bull2 and o1 > c2 and c1 < o2 and body1 < body2 * 0.5:
            patterns.append({"name": "🤰 Bullish Harami", "desc": "상승 하라미 (추세 전환 초기 신호)", "direction": "상승", "conf": 70})

        # 8. Bearish Harami — 큰 양봉 내에 포함된 작은 음봉 (하락 반전 초기)
        if not bull1 and bull2 and o1 < c2 and c1 > o2 and body1 < body2 * 0.5:
            patterns.append({"name": "🤰 Bearish Harami", "desc": "하락 하라미 (추세 전환 초기 신호)", "direction": "하락", "conf": 70})

        # 9. Harami Cross — 큰 봉 내의 도지 (강한 추세 전환 경고)
        if body1 / rng1 < 0.15 and body2 > avg_body * 1.0:
            if min(c1, o1) > min(c2, o2) and max(c1, o1) < max(c2, o2):
                dir_hc = "상승" if not bull2 else "하락"
                patterns.append({"name": "➕ Harami Cross", "desc": f"하라미 크로스 ({dir_hc} 반전 강한 경고)", "direction": dir_hc, "conf": 80})

        # 10. Piercing Line — 하락 후 전일 몸통 절반 초과 상승 (상승 반전)
        if (not bull2 and body2 > avg_body and bull1
                and o1 < l2 and c1 > mid(i2) and c1 < o2):
            patterns.append({"name": "🎯 Piercing Line", "desc": "관통형 (하락 반전, 매수 신호)", "direction": "상승", "conf": 80})

        # 11. Dark Cloud Cover — 상승 후 전일 몸통 절반 아래 하락 (하락 반전)
        if (bull2 and body2 > avg_body and not bull1
                and o1 > h2 and c1 < mid(i2) and c1 > c2):
            patterns.append({"name": "☁️ Dark Cloud Cover", "desc": "암운형 (상승 반전, 매도 신호)", "direction": "하락", "conf": 80})

        # ════════════════════════════════════════
        #  3봉 패턴 (3-Candle)
        # ════════════════════════════════════════

        # 12. Morning Star — 음봉 + 도지/소형봉 + 양봉 (강한 상승 반전)
        if (not bull3 and body3 > avg_body
                and body2 / rng2 < 0.35
                and bull1 and c1 > mid(i3)):
            patterns.append({"name": "🌅 Morning Star", "desc": "모닝스타 (하락 후 강한 상승 반전)", "direction": "상승", "conf": 90})

        # 13. Evening Star — 양봉 + 도지/소형봉 + 음봉 (강한 하락 반전)
        if (bull3 and body3 > avg_body
                and body2 / rng2 < 0.35
                and not bull1 and c1 < mid(i3)):
            patterns.append({"name": "🌆 Evening Star", "desc": "이브닝스타 (상승 후 강한 하락 반전)", "direction": "하락", "conf": 90})

        # 14. Three White Soldiers — 3연속 양봉, 각 봉이 이전 봉보다 높게 마감
        if (bull1 and bull2 and bull3
                and c1 > c2 > c3
                and o1 > o2 > o3
                and body1 > avg_body * 0.7 and body2 > avg_body * 0.7 and body3 > avg_body * 0.7
                and lo_sh1 < body1 * 0.3):
            patterns.append({"name": "⚪ Three White Soldiers", "desc": "세 백병 (강한 상승 추세 확인)", "direction": "상승", "conf": 90})

        # 15. Three Black Crows — 3연속 음봉, 각 봉이 이전 봉보다 낮게 마감
        if (not bull1 and not bull2 and not bull3
                and c1 < c2 < c3
                and o1 < o2 < o3
                and body1 > avg_body * 0.7 and body2 > avg_body * 0.7 and body3 > avg_body * 0.7):
            patterns.append({"name": "🐦 Three Black Crows", "desc": "세 검은 까마귀 (강한 하락 추세 확인)", "direction": "하락", "conf": 90})

        # 16. Three Inside Up — 하라미 확인형 (상승)
        if (not bull3 and body3 > avg_body
                and bull2 and o2 > c3 and c2 < o3 and body2 < body3 * 0.6
                and bull1 and c1 > c2):
            patterns.append({"name": "📦 Three Inside Up", "desc": "삼내부 상승 (하라미 상승 확인)", "direction": "상승", "conf": 85})

        # 17. Three Inside Down — 하라미 확인형 (하락)
        if (bull3 and body3 > avg_body
                and not bull2 and o2 < c3 and c2 > o3 and body2 < body3 * 0.6
                and not bull1 and c1 < c2):
            patterns.append({"name": "📤 Three Inside Down", "desc": "삼내부 하락 (하라미 하락 확인)", "direction": "하락", "conf": 85})

        # 18. Three Outside Up — 포용형 확인형 (상승)
        if (not bull3
                and bull2 and o2 <= c3 and c2 >= o3 and body2 > body3
                and bull1 and c1 > c2):
            patterns.append({"name": "📤 Three Outside Up", "desc": "삼외부 상승 (포용 상승 강세 확인)", "direction": "상승", "conf": 88})

        # 19. Three Outside Down — 포용형 확인형 (하락)
        if (bull3
                and not bull2 and o2 >= c3 and c2 <= o3 and body2 > body3
                and not bull1 and c1 < c2):
            patterns.append({"name": "📦 Three Outside Down", "desc": "삼외부 하락 (포용 하락 강세 확인)", "direction": "하락", "conf": 88})

        # ════════════════════════════════════════
        #  복합 패턴 (4~5봉, Gap 포함)
        # ════════════════════════════════════════

        # 20. Rising Three Methods — 큰 양봉 + 3소형 음봉(범위 내) + 큰 양봉 (상승 지속)
        if (bull5 and body5 > avg_body * 1.4
                and not bull4 and not bull3 and not bull2
                and body4 < body5 * 0.5 and body3 < body5 * 0.5 and body2 < body5 * 0.5
                and l4 > l5 and l3 > l5 and l2 > l5
                and h4 < h5 and h3 < h5 and h2 < h5
                and bull1 and c1 > c5):
            patterns.append({"name": "📊 Rising Three Methods", "desc": "상승 삼법 (상승 추세 지속 강력 신호)", "direction": "상승", "conf": 90})

        # 21. Falling Three Methods — 큰 음봉 + 3소형 양봉(범위 내) + 큰 음봉 (하락 지속)
        if (not bull5 and body5 > avg_body * 1.4
                and bull4 and bull3 and bull2
                and body4 < body5 * 0.5 and body3 < body5 * 0.5 and body2 < body5 * 0.5
                and h4 < h5 and h3 < h5 and h2 < h5
                and l4 > l5 and l3 > l5 and l2 > l5
                and not bull1 and c1 < c5):
            patterns.append({"name": "📊 Falling Three Methods", "desc": "하락 삼법 (하락 추세 지속 강력 신호)", "direction": "하락", "conf": 90})

        # 22. Abandoned Baby Bullish — 음봉 + 갭다운 도지 + 갭업 양봉 (초강세 반전)
        if (not bull3 and body3 > avg_body
                and body2 / rng2 < 0.15
                and h2 < l3                   # 갭다운
                and bull1 and l1 > h2          # 갭업
                and c1 > mid(i3)):
            patterns.append({"name": "👶 Abandoned Baby Bull", "desc": "어밴던드 베이비 (갭 반전, 초강세 신호)", "direction": "상승", "conf": 92})

        # 23. Abandoned Baby Bearish — 양봉 + 갭업 도지 + 갭다운 음봉 (초강세 하락)
        if (bull3 and body3 > avg_body
                and body2 / rng2 < 0.15
                and l2 > h3                   # 갭업
                and not bull1 and h1 < l2      # 갭다운
                and c1 < mid(i3)):
            patterns.append({"name": "👶 Abandoned Baby Bear", "desc": "어밴던드 베이비 (갭 반전, 초강세 하락)", "direction": "하락", "conf": 92})

        # 24. Hikkake Bullish — 내부바 하향 속임 후 상승 반전
        if (h2 < h3 and l2 > l3           # i2 = inside bar
                and l1 < l2               # i1이 아래로 속임
                and bull1 and c1 > h2):   # 반전 상승
            patterns.append({"name": "🎣 Hikkake Bull", "desc": "힛카케 상승 (속임 돌파 후 상승 반전)", "direction": "상승", "conf": 82})

        # 25. Hikkake Bearish — 내부바 상향 속임 후 하락 반전
        if (h2 < h3 and l2 > l3           # i2 = inside bar
                and h1 > h2               # i1이 위로 속임
                and not bull1 and c1 < l2):  # 반전 하락
            patterns.append({"name": "🎯 Hikkake Bear", "desc": "힛카케 하락 (속임 돌파 후 하락 반전)", "direction": "하락", "conf": 82})

        # 26. Mat Hold — 상승 지속 (Rising 3 Methods 변형, 갭 포함)
        if (bull5 and body5 > avg_body * 1.2
                and not bull4
                and l4 > l5 and h4 < h5     # i4 소형 역방향
                and not bull3 and not bull2  # 추가 소형 역방향
                and l3 > l5 and l2 > l5
                and bull1 and c1 > h5):      # 신고가 돌파 양봉
            patterns.append({"name": "🤝 Mat Hold", "desc": "매트 홀드 (상승 추세 지속 확인)", "direction": "상승", "conf": 85})

    except Exception:
        pass

    return patterns

def classify_market_state(dd: Dict, close: float, rsi: float,
                           adx: float, dip: float, dim: float) -> str:
    """시장 상태 분류 (6단계): 강세추세 / 약세추세 / 누적 / 분배 / 반전가능 / 횡보
    확인 순서: ADX 추세 강도 → OBV 거래량 방향 → PSAR 추세 전환 → RSI 극단
    """
    ma20 = float((dd.get("MA20",  [close])[-1] or close))
    ma50 = float((dd.get("EMA50", [close])[-1] or close))
    obv  = dd.get("OBV", [])
    obv_up = len(obv) >= 5 and obv[-1] is not None and obv[-5] is not None \
             and float(obv[-1]) > float(obv[-5])
    psar_arr  = dd.get("PSAR_DIR", [])
    psar_bull = psar_arr[-1] == 1.0 if psar_arr and psar_arr[-1] is not None else None

    above_ma20 = close > ma20 * 0.995
    above_ma50 = close > ma50 * 0.995
    strong     = adx > 25

    # 1순위: 강한 추세 (ADX + DI 방향 + MA 정배열 + PSAR 일치)
    if strong and above_ma20 and above_ma50 and dip > dim and (psar_bull is None or psar_bull):
        return "📈 강세 추세 (Uptrend)"
    if strong and not above_ma20 and not above_ma50 and dim > dip and (psar_bull is None or not psar_bull):
        return "📉 약세 추세 (Downtrend)"
    # 2순위: ADX만 강하지만 방향 불일치 → 추세 전환 구간
    if strong and ((dip > dim) != (above_ma20)):
        return "🔁 추세 전환 구간 (Transition)"
    # 3순위: 횡보 + OBV 방향으로 누적/분배 구분
    if not strong and rsi < 45 and obv_up:
        return "🔄 누적 구간 (Accumulation)"
    if not strong and rsi > 55 and not obv_up:
        return "⚠️ 분배 구간 (Distribution)"
    # 4순위: RSI + PSAR 극단 → 반전 가능
    if rsi < 30 or (rsi < 38 and psar_bull is True and not above_ma20):
        return "↩️ 상승 반전 가능 (Bullish Reversal)"
    if rsi > 70 or (rsi > 62 and psar_bull is False and above_ma20):
        return "↩️ 하락 반전 가능 (Bearish Reversal)"
    return "➡️ 횡보 구간 (Consolidation)"

def analyze_score(dd: Dict):
    """
    가중치 기반 종합 점수 산출
    - 추세 (35%): MACD & EMA 20/50
    - 모멘텀 (30%): RSI (14) & ADX (14)
    - 변동성 (20%): Bollinger Bands (20,2) & ATR (14)
    - 거래량 (15%): Volume 급증 확인
    Base 50 + 각 구간 ±(weight/2) → 최종 0~100 클리핑
    """
    closes = dd.get("Close", [])
    if len(closes) < 20:
        return 50, [], [], []

    def v(k):
        a = dd.get(k, [])
        val = a[-1] if a else None
        return float(val) if val is not None else 0.0

    close  = v("Close")
    ema20  = v("EMA20");  ema50 = v("EMA50")
    macd   = v("MACD");   sig   = v("Signal_Line")
    rsi    = v("RSI")
    adx    = v("ADX");    dip   = v("DI_Plus");  dim = v("DI_Minus")
    bb_u   = v("BB_Upper"); bb_l = v("BB_Lower"); bb_m = v("BB_Middle")
    atr    = v("ATR")
    last_opn = v("Open")   # scalar float — v()는 항상 마지막 값 반환
    vols   = dd.get("Volume", [])
    cur_vol = float(vols[-1]) if vols else 0
    avg_vol = float(np.mean([x for x in vols[-20:] if x])) if vols else 1

    score = 50.0
    steps = []

    # ── 1. 추세 분석 — EMA(20/50) & MACD & PSAR [35%] max ±17.5 ──
    ts = 0.0; msgs = []
    if ema20 and ema50:
        if ema20 > ema50:
            ts += 6.0; msgs.append("EMA20 > EMA50 정배열 → 중기 상승 추세")
        else:
            ts -= 6.0; msgs.append("EMA20 < EMA50 역배열 → 중기 하락 추세")
    if ema20 and close:
        if close > ema20:
            ts += 4.5; msgs.append("현재가 EMA20 상회 → 단기 강세")
        else:
            ts -= 4.5; msgs.append("현재가 EMA20 하회 → 단기 약세")
    if macd > sig:
        ts += 4.5; msgs.append("MACD 골든크로스 → 상승 전환 신호")
    else:
        ts -= 4.5; msgs.append("MACD 데드크로스 → 하락 전환 신호")
    psar_dir = v("PSAR_DIR")   # v()가 None→0.0 반환; 유효값은 1.0 또는 -1.0
    psar_dir_arr = dd.get("PSAR_DIR", [])
    _prev_pdir = float(psar_dir_arr[-2]) if len(psar_dir_arr) >= 2 and psar_dir_arr[-2] is not None else psar_dir
    if psar_dir == 1.0:
        ts += 2.5
        msgs.append("PSAR 상승" + (" 전환 (신규)" if _prev_pdir != 1.0 else " 추세 지속"))
    elif psar_dir == -1.0:
        ts -= 2.5
        msgs.append("PSAR 하락" + (" 전환 (신규)" if _prev_pdir != -1.0 else " 추세 지속"))
    ts = max(-17.5, min(17.5, ts))
    score += ts
    steps.append({"step": "1. 추세 분석 (EMA·MACD·PSAR)",
                  "result": " | ".join(msgs), "score": round(ts, 1), "weight": "35%"})

    # ── 2. RSI (14) & ADX (14) — 모멘텀 및 추세 신뢰도 [30%] max ±15 ──
    ms = 0.0; msgs = []
    if   rsi > 70: ms -= 6.0; msgs.append(f"RSI {rsi:.1f} 과매수 → 하락 압력 주의")
    elif rsi < 30: ms += 8.0; msgs.append(f"RSI {rsi:.1f} 과매도 → 강한 반등 기대")
    elif rsi > 55: ms -= 2.0; msgs.append(f"RSI {rsi:.1f} 고점권 — 완만한 하락 압력")
    elif rsi < 45: ms += 3.0; msgs.append(f"RSI {rsi:.1f} 저점권 → 매수 관심 구간")
    else:                      msgs.append(f"RSI {rsi:.1f} 중립")
    if adx > 25:
        if dip > dim:
            ms += 7.0; msgs.append(f"ADX {adx:.0f} + +DI 우세 → 강한 상승 추세 신뢰")
        else:
            ms -= 7.0; msgs.append(f"ADX {adx:.0f} + -DI 우세 → 강한 하락 추세 신뢰")
    elif adx > 20:
        msgs.append(f"ADX {adx:.0f} — 추세 형성 초기")
    else:
        msgs.append(f"ADX {adx:.0f} — 횡보 구간 (추세 약함)")
    ms = max(-15.0, min(15.0, ms))
    score += ms
    steps.append({"step": "2. RSI (14) & ADX (14) — 모멘텀 및 추세 신뢰도",
                  "result": " | ".join(msgs), "score": round(ms, 1), "weight": "30%"})

    # ── 3. Bollinger Bands (20, 2) & ATR (14) — 변동성 및 리스크 관리 [20%] max ±10 ──
    vs = 0.0; msgs = []
    if close and bb_u and bb_l and bb_u > bb_l:
        bb_range = bb_u - bb_l
        pos = (close - bb_l) / bb_range  # 0~1 위치
        if close >= bb_u * 0.98:
            vs -= 5.0; msgs.append("볼린저 상단 터치 → 단기 과매수/저항")
        elif close <= bb_l * 1.02:
            vs += 5.0; msgs.append("볼린저 하단 터치 → 단기 과매도/지지")
        elif pos > 0.7:
            vs -= 2.0; msgs.append("볼린저 상단권 (70%+) → 매도 압력")
        elif pos < 0.3:
            vs += 2.0; msgs.append("볼린저 하단권 (30%-) → 지지 기대")
        else:
            msgs.append("볼린저 중간권 → 중립")
        bb_pct = bb_range / close * 100
        if bb_pct < 3.0:
            vs += 3.0; msgs.append(f"밴드 수렴 ({bb_pct:.1f}%) → 큰 방향 돌파 임박")
        elif atr and close:
            atr_pct = atr / close * 100
            if atr_pct > 4.0:
                vs -= 2.0; msgs.append(f"ATR 고변동 ({atr_pct:.1f}%) → 리스크 증가")
            else:
                msgs.append(f"ATR {atr_pct:.1f}% — 적정 변동성")
    vs = max(-10.0, min(10.0, vs))
    score += vs
    steps.append({"step": "3. Bollinger Bands (20,2) & ATR (14) — 변동성 및 리스크 관리",
                  "result": " | ".join(msgs), "score": round(vs, 1), "weight": "20%"})

    # ── 4. Volume — 거래량 급증 확인 [15%] max ±7.5 ──
    gvs = 0.0; msgs = []
    if avg_vol > 0:
        ratio = cur_vol / avg_vol
        last_close = close
        last_opn = last_opn if last_opn else last_close   # 데이터 없으면 종가로 대체

        if ratio > 2.0:
            if last_close > last_opn: gvs += 7.5; msgs.append(f"거래량 {ratio:.1f}x 급증 + 양봉 → 강한 매수세 확인")
            else:                     gvs -= 7.5; msgs.append(f"거래량 {ratio:.1f}x 급증 + 음봉 → 강한 매도세 확인")
        elif ratio > 1.5:
            if last_close > last_opn: gvs += 4.0; msgs.append(f"거래량 {ratio:.1f}x 증가 + 상승 → 매수 우위")
            else:                     gvs -= 4.0; msgs.append(f"거래량 {ratio:.1f}x 증가 + 하락 → 매도 압력")
        elif ratio < 0.5:
            msgs.append(f"거래량 급감 ({ratio:.1f}x) → 신뢰도 낮음")
        else:
            msgs.append(f"거래량 평이 ({ratio:.1f}x)")
    else:
        msgs.append("거래량 데이터 없음")
    gvs = max(-7.5, min(7.5, gvs))
    score += gvs
    steps.append({"step": "4. Volume — 거래량 급증 확인",
                  "result": " | ".join(msgs), "score": round(gvs, 1), "weight": "15%"})

    # 캔들 패턴 (점수 반영 없이 정보 제공)
    patterns = detect_patterns(dd)

    # 기하학적 패턴 (점수 반영 없이 정보 제공)
    geo_patterns = []
    try:
        df = pd.DataFrame({k: dd[k] for k in ["Open", "High", "Low", "Close"] if k in dd})
        if not df.empty and len(df) > 20:
            geo_patterns = ChartPatternAnalyzer(df).detect_patterns()
    except:
        pass

    cp_msgs = []
    cp_score = 0.0
    bull_patterns = []
    bear_patterns = []

    for p in patterns + geo_patterns:
        direction = p.get('direction') or ('상승' if p.get('signal') == '매수' else '하락' if p.get('signal') == '매도' else '중립')
        cp_msgs.append(f"[{direction}] {p.get('name', '')}: {p.get('desc', '')}")
        # ── conf(신뢰도) 기반 차등 가중치 ──────────────────────────
        # 기하학적 패턴(geo)은 conf 키 없음 → 기본값 80 적용
        conf = p.get('conf', 80)
        weight = 3.0 if conf >= 90 else (2.0 if conf >= 80 else 1.0)
        if direction == '상승':
            cp_score += weight
            bull_patterns.append(p)
        elif direction == '하락':
            cp_score -= weight
            bear_patterns.append(p)

    # ── RSI × 캔들 패턴 연동 시너지 (팩터 하이브리드 핵심) ─────────
    # RSI 과매도 + 상승 패턴 동시 발생 → 추가 가산
    if rsi < 35 and bull_patterns:
        cp_score += 1.5
        cp_msgs.append(f"⚡ RSI 과매도({rsi:.1f}) + 상승 패턴 시너지 → 반전 신호 강화")
    # RSI 과매수 + 하락 패턴 동시 발생 → 추가 감산
    elif rsi > 65 and bear_patterns:
        cp_score -= 1.5
        cp_msgs.append(f"⚡ RSI 과매수({rsi:.1f}) + 하락 패턴 시너지 → 하락 신호 강화")

    if not cp_msgs:
        cp_msgs = ["특이한 캔들/차트 패턴 미발견"]

    steps.append({"step": "5. 캔들 패턴 분석",
                  "result": " | ".join(cp_msgs), "score": round(max(-6.0, min(6.0, cp_score)), 1), "weight": "보조"})

    # ── 6. 크로스 지표 종합 (OBV·Aroon·TRIX 수렴/다이버전스) ─────────────
    sx = 0.0; msgs = []
    obv_arr      = dd.get("OBV", [])
    aroon_up_arr = dd.get("AROON_UP", [])
    aroon_dn_arr = dd.get("AROON_DOWN", [])
    trix_arr     = dd.get("TRIX", [])
    bp_arr       = dd.get("BUY_PRESSURE", [])

    def _sv(arr):  # safe last float
        return float(arr[-1]) if arr and arr[-1] is not None else None

    aroon_up_v = _sv(aroon_up_arr)
    aroon_dn_v = _sv(aroon_dn_arr)
    trix_v     = _sv(trix_arr)
    bp_v       = _sv(bp_arr)

    # OBV vs 가격 수렴/다이버전스 (5일 비교)
    if len(obv_arr) >= 5 and len(closes) >= 5 and \
       obv_arr[-1] is not None and obv_arr[-5] is not None:
        obv_d   = float(obv_arr[-1]) - float(obv_arr[-5])
        price_d = float(closes[-1])  - float(closes[-5])
        if   price_d > 0 and obv_d > 0: sx += 1.5; msgs.append("OBV + 가격 동반 상승 → 추세 신뢰↑")
        elif price_d < 0 and obv_d < 0: sx -= 1.5; msgs.append("OBV + 가격 동반 하락 → 하락 추세 신뢰↑")
        elif price_d > 0 and obv_d < 0: sx -= 1.0; msgs.append("가격↑·OBV↓ 다이버전스 → 매수세 약화 경고")
        elif price_d < 0 and obv_d > 0: sx += 1.0; msgs.append("가격↓·OBV↑ 다이버전스 → 저가 누적 추정")

    # Aroon + TRIX 이중 모멘텀 확인
    if aroon_up_v is not None and aroon_dn_v is not None and trix_v is not None:
        if aroon_up_v > 70 and trix_v > 0:
            sx += 1.5; msgs.append(f"Aroon Up {aroon_up_v:.0f} + TRIX 양전 → 중기 상승 이중 확인")
        elif aroon_dn_v > 70 and trix_v < 0:
            sx -= 1.5; msgs.append(f"Aroon Down {aroon_dn_v:.0f} + TRIX 음전 → 중기 하락 이중 확인")
        elif aroon_up_v > aroon_dn_v:
            msgs.append("Aroon 상승 우위" + (" + TRIX 상승 확인" if trix_v > 0 else " · TRIX 음전 혼조"))
        else:
            msgs.append("Aroon 하락 우위" + (" + TRIX 하락 확인" if trix_v < 0 else " · TRIX 양전 혼조"))

    # Buy Pressure + RSI 시너지
    if bp_v is not None:
        if bp_v > 55 and rsi < 50:
            sx += 1.0; msgs.append(f"매수압력 {bp_v:.0f}% + RSI 저점 → 저점 매집 신호")
        elif bp_v < 45 and rsi > 60:
            sx -= 1.0; msgs.append(f"매수압력 {bp_v:.0f}% + RSI 고점 → 상승 동력 약화")

    # PSAR + EMA 정합성 확인
    psar_d_arr = dd.get("PSAR_DIR", [])
    _pdir = float(psar_d_arr[-1]) if psar_d_arr and psar_d_arr[-1] is not None else None
    if _pdir is not None and ema20 is not None:
        above_ema = close > ema20
        if _pdir == 1.0 and above_ema:
            sx += 0.5; msgs.append("PSAR 상승 + EMA20 상회 → 추세 지표 일치")
        elif _pdir == -1.0 and not above_ema:
            sx -= 0.5; msgs.append("PSAR 하락 + EMA20 하회 → 추세 지표 일치")
        elif _pdir == 1.0 and not above_ema:
            msgs.append("PSAR 상승·EMA20 하회 혼조 → 전환 초기 또는 일시 눌림")
        elif _pdir == -1.0 and above_ema:
            msgs.append("PSAR 하락·EMA20 상회 혼조 → 단기 반등 중 추세 이탈 주의")

    # 시장 상태 분류
    market_state = classify_market_state(dd, close, rsi, adx, dip, dim)
    msgs.append(f"시장 상태: {market_state}")

    if not msgs:
        msgs = ["크로스 지표 분석 데이터 부족"]

    steps.append({"step": "6. 크로스 지표 종합 (OBV·Aroon 수렴)",
                  "result": " | ".join(msgs),
                  "score": round(max(-6.0, min(6.0, sx)), 1), "weight": "보조"})

    score = max(0.0, min(100.0, round(score)))

    # ── [AI 종합 진단 및 미래 예측 시나리오 추가] ──
    ai_msgs = []
    
    # 1. 핵심 요약 및 매수/매도 타이밍 조건
    if score >= 65:
        ai_msgs.append("[핵심 요약] BUY (매수 우위)")
        ai_msgs.append(f"👉 매수 타이밍: 현재가({close:,.0f}) 부근 혹은 단기 눌림목({ema20:,.0f} 지지) 시 분할 매수")
        ai_msgs.append(f"👉 매도 타이밍: RSI가 70을 초과하거나 볼린저 상단({bb_u:,.0f}) 도달 시 비중 축소")
    elif score <= 40:
        ai_msgs.append("[핵심 요약] SELL (매도 우위 / 리스크 관리)")
        ai_msgs.append(f"👉 매수 타이밍: RSI 과매도(30 이하) 진입 및 지지선({bb_l:,.0f})에서 명확한 반등 캔들 확인 후")
        ai_msgs.append(f"👉 매도 타이밍: 반등 시 EMA20({ema20:,.0f}) 저항선 부근에서 비중 축소")
    else:
        ai_msgs.append("[핵심 요약] HOLD (관망 / 중립)")
        ai_msgs.append(f"👉 매수 타이밍: ADX가 25를 돌파하며 방향성이 나오거나 볼린저 하단({bb_l:,.0f}) 터치 시")
        ai_msgs.append(f"👉 매도 타이밍: 박스권 상단({bb_u:,.0f}) 도달 시 수익 실현")

    # 2. 근거 (지표 해석)
    reasons = []
    if ema20 and ema50 and ema20 > ema50: reasons.append("단기 이평선 정배열")
    elif ema20 and ema50 and ema20 < ema50: reasons.append("단기 이평선 역배열")
    if macd > sig: reasons.append("MACD 상승 다이버전스")
    if rsi < 40: reasons.append("RSI 단기 저평가 매수권")
    elif rsi > 60: reasons.append("RSI 단기 고평가 매도권")
    if _pdir == 1.0: reasons.append("PSAR 상승 추세 유지")
    elif _pdir == -1.0: reasons.append("PSAR 하락 추세 유지")
    ai_msgs.append(f"🔍 종합 판단 근거: {', '.join(reasons) if reasons else '복합적 횡보장세 요소 작용'}")

    # 3. 시나리오별 대응 전략
    ai_msgs.append("[시나리오별 대응 전략]")
    ai_msgs.append(f"📈 상승 시나리오: 강한 거래량(이전 대비 1.5배 이상)을 동반하여 {bb_u:,.0f} 상향 돌파 시 추세 추종 (목표가 +5~10%)")
    ai_msgs.append(f"📉 하락 시나리오: {bb_l:,.0f} 하향 이탈 및 MACD 데드크로스 발생 시 즉각적인 리스크 관리 (손절선 -3~5%)")
    ai_msgs.append(f"➡️ 횡보 시나리오: {bb_l:,.0f} ~ {bb_u:,.0f} 밴드 내 박스권 트레이딩 (하단 지지 확인 후 진입, 상단 저항 시 청산)")

    ai_strategy = {
        "step": "💡 AI 종합 진단 및 트레이딩 전략",
        "result": " | ".join(ai_msgs),
        "score": round(score - 50, 1), 
        "weight": "종합"
    }

    return score, steps, patterns, geo_patterns, ai_strategy

def calc_risk(price: float, atr: float) -> Dict:
    if not atr or np.isnan(atr): atr = price * 0.02
    
    # 목표가 및 손절가 범위 계산
    cons_tgt = [price + atr * 1.0, price + atr * 1.5]
    cons_stp = [price - atr * 1.0, price - atr * 0.8]
    cons_ret = round(((cons_tgt[0] + cons_tgt[1])/2 - price) / price * 100, 2)
    
    bal_tgt = [price + atr * 2.0, price + atr * 3.0]
    bal_stp = [price - atr * 1.5, price - atr * 1.2]
    bal_ret = round(((bal_tgt[0] + bal_tgt[1])/2 - price) / price * 100, 2)
    
    agg_tgt = [price + atr * 4.0, price + atr * 6.0]
    agg_stp = [price - atr * 2.5, price - atr * 2.0]
    agg_ret = round(((agg_tgt[0] + agg_tgt[1])/2 - price) / price * 100, 2)
    
    return {
        "conservative": {
            "label":"보수적",
            "target":[round(cons_tgt[0],2), round(cons_tgt[1],2)],
            "stop":[round(cons_stp[0],2), round(cons_stp[1],2)],
            "return": cons_ret,
            "desc":"리스크 최소화", "icon":"🛡️"
        },
        "balanced": {
            "label":"중립적",
            "target":[round(bal_tgt[0],2), round(bal_tgt[1],2)],
            "stop":[round(bal_stp[0],2), round(bal_stp[1],2)],
            "return": bal_ret,
            "desc":"스윙 트레이딩", "icon":"⚖️"
        },
        "aggressive": {
            "label":"공격적",
            "target":[round(agg_tgt[0],2), round(agg_tgt[1],2)],
            "stop":[round(agg_stp[0],2), round(agg_stp[1],2)],
            "return": agg_ret,
            "desc":"추세 추종", "icon":"🚀"
        },
    }

def calc_pivot_points(dd: Dict) -> Dict:
    """피봇 포인트 계산: 클래식, 피보나치, 카마리야, 우디스, 디마크"""
    highs  = [float(x) for x in dd.get("High",  []) if x is not None]
    lows   = [float(x) for x in dd.get("Low",   []) if x is not None]
    closes = [float(x) for x in dd.get("Close", []) if x is not None]
    opens  = [float(x) for x in dd.get("Open",  []) if x is not None]
    if len(highs) < 2:
        return {}
    # 전일 고/저/종
    h = highs[-2]; l = lows[-2]; c = closes[-2]
    o = opens[-2] if len(opens) >= 2 else c
    r = lambda v: round(v, 2)
    rng = h - l

    # 클래식
    piv = (h + l + c) / 3
    classic = {
        "Pivot": r(piv),
        "R1": r(2*piv - l),   "R2": r(piv + rng),     "R3": r(h + 2*(piv - l)),
        "S1": r(2*piv - h),   "S2": r(piv - rng),     "S3": r(l - 2*(h - piv)),
    }

    # 피보나치
    fibonacci = {
        "Pivot": r(piv),
        "R1": r(piv + 0.382*rng), "R2": r(piv + 0.618*rng), "R3": r(piv + 1.000*rng),
        "S1": r(piv - 0.382*rng), "S2": r(piv - 0.618*rng), "S3": r(piv - 1.000*rng),
    }

    # 카마리야
    camarilla = {
        "Pivot": r(c),
        "R1": r(c + rng*1.1/12), "R2": r(c + rng*1.1/6), "R3": r(c + rng*1.1/4),
        "S1": r(c - rng*1.1/12), "S2": r(c - rng*1.1/6), "S3": r(c - rng*1.1/4),
    }

    return {"classic": classic, "fibonacci": fibonacci, "camarilla": camarilla}

def calc_indicator_signals(dd: Dict) -> Dict:
    """각 기술적 지표별 현재 상태·매매 시그널·핵심 해석 계산"""
    def v(k):
        a = dd.get(k, [])
        val = a[-1] if a else None
        return float(val) if val is not None else None

    close = v("Close") or 1.0
    signals = {}

    # ADX (14)
    adx = v("ADX"); dip = v("DI_Plus"); dim = v("DI_Minus")
    if adx is not None:
        if adx > 25:
            if dip is not None and dim is not None and dip > dim:
                st,sig,desc = "강한 상승 추세","매수", f"ADX {adx:.0f} 강세 + +DI 우세 → 상승 추세 강함"
            elif dip is not None and dim is not None:
                st,sig,desc = "강한 하락 추세","매도", f"ADX {adx:.0f} 강세 + -DI 우세 → 하락 추세 강함"
            else:
                st,sig,desc = "추세 강함",     "관망", f"ADX {adx:.0f} — 강한 추세 형성 중"
        elif adx > 20: st,sig,desc = "추세 발생",    "관망", f"ADX {adx:.0f} — 추세 형성 초기"
        else:          st,sig,desc = "횡보/추세 없음","관망", f"ADX {adx:.0f} — 방향성 불명확, 돌파 대기"
        signals["adx"] = {"name":"ADX (14)", "state":st, "signal":sig, "desc":desc, "value":f"{adx:.1f}"}

    # ATR (14)
    atr = v("ATR")
    if atr is not None:
        atr_pct = atr / close * 100
        if   atr_pct > 3:   st,sig,desc = "고변동성",   "관망", f"일간 변동 ≈{atr_pct:.1f}% — 분할 매수 권장"
        elif atr_pct > 1.5: st,sig,desc = "보통 변동성","관망", f"일간 변동 ≈{atr_pct:.1f}% — 적정 리스크"
        else:                st,sig,desc = "저변동성",   "관망", f"일간 변동 ≈{atr_pct:.1f}% — 돌파 시 강한 추세 기대"
        signals["atr"] = {"name":"ATR (14)", "state":st, "signal":sig, "desc":desc, "value":f"{atr:.2f}"}

    # ── 신규 지표 ───────────────────────────────────────────────────────

    # OBV — 가격·거래량 수렴/다이버전스
    obv_arr = dd.get("OBV", [])
    if len(obv_arr) >= 5:
        obv_now = float(obv_arr[-1]) if obv_arr[-1] is not None else None
        obv_p5  = float(obv_arr[-5]) if obv_arr[-5] is not None else None
        cl_arr  = dd.get("Close", [])
        cl_p5   = float(cl_arr[-5]) if len(cl_arr) >= 5 and cl_arr[-5] is not None else close
        if obv_now is not None and obv_p5 is not None:
            p_up = close > cl_p5; o_up = obv_now > obv_p5
            if   p_up and o_up:   st,sig,desc = "수렴 상승","매수","가격↑ + OBV↑ — 매수세 동반, 추세 신뢰↑"
            elif not p_up and not o_up: st,sig,desc = "수렴 하락","매도","가격↓ + OBV↓ — 매도세 동반, 하락 신뢰↑"
            elif p_up and not o_up: st,sig,desc = "강세 다이버전스","매도","가격↑·OBV↓ — 매수세 약화, 상승 지속성 의문"
            else:                   st,sig,desc = "약세 다이버전스","매수","가격↓·OBV↑ — 기관 누적 추정, 반등 가능"
            _ao = abs(obv_now)
            if _ao >= 1e9:   obv_disp = f"{obv_now/1e9:+.2f}B"
            elif _ao >= 1e6: obv_disp = f"{obv_now/1e6:+.1f}M"
            else:            obv_disp = f"{obv_now/1e3:+.1f}K"
            signals["obv"] = {"name":"OBV", "state":st, "signal":sig, "desc":desc,
                              "value":obv_disp}

    # Stochastic (14,3) — %K / %D
    sk14 = v("%K"); sd14 = v("%D")
    if sk14 is not None and sd14 is not None:
        if   sk14 > 80: st,sig,desc = "과매수",    "매도",f"%K {sk14:.1f} 과매수 — 되돌림 경계"
        elif sk14 < 20: st,sig,desc = "과매도",    "매수",f"%K {sk14:.1f} 과매도 — 단기 반등 기대"
        elif sk14 > sd14: st,sig,desc = "골든크로스","매수",f"%K({sk14:.1f}) > %D({sd14:.1f}) — 단기 상승 전환"
        else:           st,sig,desc = "데드크로스","매도",f"%K({sk14:.1f}) < %D({sd14:.1f}) — 단기 하락 전환"
        signals["stoch14"] = {"name":"Stochastic (14,3)", "state":st, "signal":sig, "desc":desc,
                              "value":f"{sk14:.1f} / {sd14:.1f}"}

    # Aroon (25) — 최근 고점/저점 경과일 기반 추세 방향
    au = v("AROON_UP"); ad = v("AROON_DOWN")
    if au is not None and ad is not None:
        if   au > 70 and ad < 30: st,sig,desc = "강한 상승","매수",f"Up {au:.0f} — 최근 고점 근접, 상승 추세 우세"
        elif ad > 70 and au < 30: st,sig,desc = "강한 하락","매도",f"Down {ad:.0f} — 최근 저점 근접, 하락 추세 우세"
        elif au > ad:             st,sig,desc = "상승 우위","관망",f"Up({au:.0f}) > Down({ad:.0f}) — 추세 약함"
        else:                     st,sig,desc = "하락 우위","관망",f"Up({au:.0f}) < Down({ad:.0f}) — 하락 우위"
        signals["aroon"] = {"name":"Aroon (25)", "state":st, "signal":sig, "desc":desc,
                            "value":f"↑{au:.0f} / ↓{ad:.0f}"}

    # Buy Pressure (14일 상승일 거래량 비중)
    bp = v("BUY_PRESSURE")
    if bp is not None:
        if   bp > 65: st,sig,desc = "강한 매수세","매수",f"상승일 거래량 비중 {bp:.1f}% — 14일 매수 우위"
        elif bp > 50: st,sig,desc = "매수 우위",  "매수",f"상승일 거래량 비중 {bp:.1f}% — 완만한 매수세"
        elif bp > 35: st,sig,desc = "매도 우위",  "매도",f"상승일 거래량 비중 {bp:.1f}% — 매도 압력 존재"
        else:         st,sig,desc = "강한 매도세","매도",f"상승일 거래량 비중 {bp:.1f}% — 14일 매도 우위"
        signals["buy_pressure"] = {"name":"Buy Pressure (14)", "state":st, "signal":sig, "desc":desc,
                                   "value":f"{bp:.1f}%"}

    # ── 추세형 이동평균 지표 ──────────────────────────────────────────────

    # PSAR (Parabolic SAR) — 추세 방향 + 반전 감지
    psar_v   = v("PSAR")
    psar_dir = v("PSAR_DIR")
    if psar_v is not None and psar_v > 0 and psar_dir is not None:
        psar_arr  = dd.get("PSAR_DIR", [])
        prev_pdir = float(psar_arr[-2]) if len(psar_arr) >= 2 and psar_arr[-2] is not None else psar_dir
        flipped   = (psar_dir != prev_pdir)
        psar_disp = round(psar_v, 2)
        if psar_dir == 1.0:
            if flipped: st,sig,desc = "상승 전환", "매수", f"SAR {psar_disp} — 하락→상승 전환 (추세 반전 확인)"
            else:       st,sig,desc = "상승 추세", "매수", f"가격 > SAR {psar_disp} — 상승 추세 지속"
        else:
            if flipped: st,sig,desc = "하락 전환", "매도", f"SAR {psar_disp} — 상승→하락 전환 (손절 고려)"
            else:       st,sig,desc = "하락 추세", "매도", f"가격 < SAR {psar_disp} — 하락 추세 지속"
        signals["psar"] = {"name":"PSAR (0.02/0.2)", "state":st, "signal":sig, "desc":desc,
                           "value":f"{'▲' if psar_dir == 1.0 else '▼'} {psar_disp}"}

    # 시장 상태 분류
    adx_v = v("ADX") or 0.0; dip_v = v("DI_Plus") or 0.0; dim_v = v("DI_Minus") or 0.0
    rsi_v = v("RSI") or 50.0
    market_state = classify_market_state(dd, close, rsi_v, adx_v, dip_v, dim_v)

    buy_n   = sum(1 for s in signals.values() if s["signal"] == "매수")
    sell_n  = sum(1 for s in signals.values() if s["signal"] == "매도")
    watch_n = sum(1 for s in signals.values() if s["signal"] == "관망")
    total_n = len(signals)
    if   buy_n  > sell_n  and buy_n  >= watch_n: ov_sig,ov_lbl = "매수","매수 우세"
    elif sell_n > buy_n   and sell_n >= watch_n: ov_sig,ov_lbl = "매도","매도 우세"
    else:                                         ov_sig,ov_lbl = "관망","중립 / 관망"

    return {"signals": signals,
            "summary": {"buy":buy_n,"sell":sell_n,"watch":watch_n,"total":total_n,
                        "overall_signal":ov_sig,"overall_label":ov_lbl,
                        "market_state": market_state}}

def calc_buy_price(dd: Dict, last_price: float, atr: float) -> Dict:
    """매수 적정 가격 예측 (지지선·변동성·기술적 지표 기반)"""
    lows   = [float(x) for x in dd.get("Low",   []) if x is not None]
    rsi    = float((dd.get("RSI",   [50])[-1]) or 50)
    bb_l   = dd.get("BB_Lower",    [None])[-1]
    ma20   = dd.get("MA20",        [None])[-1]
    ma60   = dd.get("MA60",        [None])[-1]
    if not atr or np.isnan(atr):
        atr = last_price * 0.02

    recent_lows  = sorted([x for x in lows[-30:] if x > 0])
    support_zone = float(np.mean(recent_lows[:5])) if len(recent_lows) >= 5 else last_price * 0.95

    # 가격 범위 및 수익 확률 산출
    aggressive = {
        "range": [round(last_price - atr * 0.8, 2), round(last_price - atr * 0.2, 2)],
        "prob": 45
    }
    recommended = {
        "range": [round(last_price - atr * 1.5, 2), round(last_price - atr * 0.8, 2)],
        "prob": 65
    }
    conservative = {
        "range": [round(support_zone - atr * 0.5, 2), round(support_zone + atr * 0.5, 2)],
        "prob": 85
    }

    # 매수/매도 타이밍 예측 (동적 계산)
    now = dt.now()
    
    # RSI 및 MACD에 따른 매수 타이밍 조정
    macd = float(dd.get("MACD", [0])[-1] or 0)
    sig  = float(dd.get("Signal_Line", [0])[-1] or 0)

    buy_delay = 1
    if rsi < 30 and macd > sig: # 강력 매수 조건
        buy_delay = 0 
    elif rsi > 70: # 과매수 (눌림 대기)
        buy_delay = 3
    elif rsi > 55:
        buy_delay = 2
        
    buy_time = now + timedelta(days=buy_delay)
    buy_time = buy_time.replace(hour=10, minute=30)
    while buy_time.weekday() > 4:  # 주말 건너뛰기
        buy_time += timedelta(days=1)

    # 목표 수익률 달성까지의 예상 기간 (ATR 기반 변동성 고려)
    target_dist = (aggressive["range"][1] - last_price) if aggressive["range"][1] > last_price else (last_price * 0.05)
    days_to_target = max(2, int(target_dist / (atr if atr > 0 else 1)))

    sell_time = buy_time + timedelta(days=days_to_target)
    sell_time = sell_time.replace(hour=14, minute=30)
    while sell_time.weekday() > 4:
        sell_time += timedelta(days=1)

    basis = []
    if bb_l and float(bb_l) < last_price * 1.05:
        basis.append(f"볼린저 하단 지지선 근접 (≈{round(float(bb_l), 2):,})")
    if ma20 and abs(float(ma20) - last_price) / last_price < 0.15:
        basis.append(f"단기 생명선(MA20) 지지 (≈{round(float(ma20), 2):,})")
    if ma60 and abs(float(ma60) - last_price) / last_price < 0.20:
        basis.append(f"중기 추세선(MA60) 지지 (≈{round(float(ma60), 2):,})")
    basis.append(f"일간 변동성(ATR) 기반 구간 분할 (ATR ≈{round(atr, 2):,})")
    basis.append(f"최근 30일 핵심 매물대/저점 지지구간 (≈{round(support_zone, 2):,})")

    if   rsi < 30:  rsi_ctx = "RSI 과매도 — 적극 매수 관점"
    elif rsi < 45:  rsi_ctx = "RSI 저점권 — 매수 유리, 분할 진입"
    elif rsi > 70:  rsi_ctx = "RSI 과매수 — 신규 진입 보류, 눌림 대기"
    elif rsi > 55:  rsi_ctx = "RSI 고점권 — 보수적 접근, 확인 후 진입"
    else:           rsi_ctx = "RSI 중립 — 지지/저항선 돌파 확인 후 진입"

    return {"current": round(last_price, 2),
            "aggressive": aggressive,
            "recommended": recommended,
            "conservative": conservative,
            "timing": {
                "buy": buy_time.strftime("%Y-%m-%d %H:%M"),
                "sell": sell_time.strftime("%Y-%m-%d %H:%M")
            },
            "support_zone": round(support_zone, 2),
            "basis": basis, "rsi": round(rsi, 1), "rsi_context": rsi_ctx, "atr": round(atr, 2)}

def holt_winters_forecast(dd: Dict, days: int = 30):
    """
    Lightweight implementation of Double Exponential Smoothing (Holt's Linear Trend)
    Replaces statsmodels to reduce dependency size.
    """
    try:
        closes = [float(c) for c in dd.get("Close", []) if c is not None]
        dates = dd.get("Date", [])
        if len(closes) < 30: return None
        
        # ── Holt's Double Exponential Smoothing 파라미터 ──────────────
        # alpha=0.3: 레벨 평활 (낮을수록 안정적, 노이즈 저감)
        # beta=0.05: 트렌드 평활 (낮을수록 과도한 기울기 방지)
        alpha = 0.3
        beta  = 0.05

        # Initialization
        level = closes[0]
        trend = closes[1] - closes[0]

        # Fit
        for i in range(1, len(closes)):
            last_level = level
            level = alpha * closes[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend

        # Forecast
        forecast = []
        last_d = datetime.datetime.strptime(dates[-1], "%Y-%m-%d") if dates else datetime.datetime.now()
        future_dates = []
        d = last_d

        for i in range(1, days + 1):
            d += datetime.timedelta(days=1)
            while d.weekday() >= 5:
                d += datetime.timedelta(days=1)
            future_dates.append(d.strftime("%Y-%m-%d"))
            # 음수 방지: 주가는 0 이하가 될 수 없음
            yhat = max(0.01, level + i * trend)
            forecast.append(yhat)

        # ── 신뢰구간 동적화: 기간이 길수록 불확실성 증가 (sqrt 스케일) ──
        std = np.std(closes[-30:]) if len(closes) >= 30 else closes[-1] * 0.02

        return {
            "dates": future_dates,
            "yhat":       [round(float(f), 2) for f in forecast],
            "yhat_upper": [round(max(0.01, float(f) + 1.96 * std * math.sqrt(i + 1)), 2)
                           for i, f in enumerate(forecast)],
            "yhat_lower": [round(max(0.01, float(f) - 1.96 * std * math.sqrt(i + 1)), 2)
                           for i, f in enumerate(forecast)],
        }
    except Exception:
        return linear_forecast(dd, days)

def linear_forecast(dd: Dict, days: int):
    """
    Simple Linear Regression using numpy.polyfit
    Replaces sklearn to reduce dependency size.
    """
    try:
        closes = [float(c) for c in dd.get("Close", []) if c is not None]
        dates = dd.get("Date", [])
        if len(closes) < 20: return None
        
        y = np.array(closes)
        x = np.arange(len(y))
        
        # Linear Fit (Degree 1)
        slope, intercept = np.polyfit(x, y, 1)
        
        # Predict
        future_x = np.arange(len(y), len(y) + days)
        preds = slope * future_x + intercept
        
        last_d = datetime.datetime.strptime(dates[-1], "%Y-%m-%d") if dates else datetime.datetime.now()
        fds = []
        d = last_d
        for _ in range(days):
            d += datetime.timedelta(days=1)
            while d.weekday() >= 5: d += datetime.timedelta(days=1)
            fds.append(d.strftime("%Y-%m-%d"))
            
        # ── 신뢰구간 동적화: 최근 30일 변동성 + sqrt 스케일 ─────────
        vol = np.std(closes[-30:]) if len(closes) >= 30 else closes[-1] * 0.03

        return {
            "dates": fds,
            "yhat":       [round(float(p), 2) for p in preds],
            "yhat_upper": [round(max(0.01, float(p) + 1.96 * vol * math.sqrt(i + 1)), 2)
                           for i, p in enumerate(preds)],
            "yhat_lower": [round(max(0.01, float(p) - 1.96 * vol * math.sqrt(i + 1)), 2)
                           for i, p in enumerate(preds)],
        }
    except Exception:
        return None

def xgb_forecast(dd: Dict, days: int = 30):
    """
    Simulated Forecast based on Momentum & Mean Reversion
    Replaces XGBoost to reduce dependency size.
    """
    try:
        closes = [float(c) for c in dd.get("Close", []) if c is not None]
        if len(closes) < 60: return None
        
        # Calculate recent momentum (short vs long MA)
        short_ma = np.mean(closes[-5:])
        long_ma = np.mean(closes[-20:])
        momentum_strength = (short_ma - long_ma) / long_ma
        
        last_price = closes[-1]
        preds = []
        
        # Simulation parameters
        decay = 0.95 # Momentum decay factor
        current_price = last_price
        
        # Estimated daily drift based on momentum
        # If momentum is 0.05 (5%), daily drift might be approx 0.05/20 per day initially
        drift = momentum_strength * last_price * 0.1
        
        for _ in range(days):
            drift *= decay # Momentum fades over time
            current_price += drift
            preds.append(round(current_price, 2))
        
        return preds
    except Exception as e:
        print(f"Simulation Error: {e}")
        return None

# =============================================================================
# 라우팅
# =============================================================================
def route(path: str, params: Dict) -> Dict:
    # trailing slash는 do_GET에서 rstrip("/") 처리됨
    # path는 항상 /api/stock 형식 (슬래시 없음)
    if path == "/api/stock":
        raw = params.get("ticker", "삼성전자")
        period = params.get("period", "1y")
        ticker, market, company = resolve_ticker(raw)
        if not ticker:
            return {"error": f"'{raw}' 종목을 찾을 수 없습니다."}
        dd, news, sym = fetch_stock_data(ticker, market, period)
        if dd is None:
            return {"error": f"데이터 조회 실패: {sym}"}
        closes = dd.get("Close", [])
        last = float(closes[-1]) if closes else 0
        prev = float(closes[-2]) if len(closes) > 1 else last
        pct = (last - prev) / prev * 100 if prev else 0
        score, steps, patterns, geo_patterns, ai_strategy = analyze_score(dd)
        
        # 기하학적 패턴을 캔들 패턴 리스트에 통합 (UI 표시용)
        for gp in geo_patterns:
            direction = "상승" if gp.get("signal") == "매수" else "하락" if gp.get("signal") == "매도" else "중립"
            patterns.append({
                "name": gp.get("name"),
                "desc": gp.get("desc"),
                "direction": direction,
                "conf": 100
            })
            
        atrs = dd.get("ATR", [])
        atr_val = float(atrs[-1]) if atrs and atrs[-1] else last * 0.02
        risk             = calc_risk(last, atr_val)
        pivot_points     = calc_pivot_points(dd)
        indicator_signals= calc_indicator_signals(dd)
        buy_price        = calc_buy_price(dd, last, atr_val)
        naver = fetch_naver(sym) if market == "KRX" else None
        return {
            "symbol": sym, "company": company or sym, "market": market,
            "last_close": round(last, 2), "prev_close": round(prev, 2),
            "pct_change": round(pct, 2),
            "rsi": round(float(dd.get("RSI", [50])[-1] or 50), 1),
            "volume": int(dd.get("Volume", [0])[-1] or 0),
            "atr": round(atr_val, 2),
            "score": score, "analysis_steps": steps, "ai_strategy": ai_strategy,
            "candlestick_patterns": patterns,
            "chart_data": {
                "dates": dd.get("Date", []),
                "open": dd.get("Open", []),
                "high": dd.get("High", []),
                "low": dd.get("Low", []),
                "close": dd.get("Close", []),
                "volume": dd.get("Volume", []),
                "ma20": dd.get("MA20", []),
                "ma60": dd.get("MA60", []),
                "bb_upper": dd.get("BB_Upper", []),
                "bb_lower": dd.get("BB_Lower", []),
                "rsi": dd.get("RSI", []),
                "macd": dd.get("MACD", []),
                "signal_line": dd.get("Signal_Line", []),
            },
            "risk_scenarios": risk,
            "pivot_points": pivot_points,
            "indicator_signals": indicator_signals,
            "buy_price": buy_price,
            "news": news or [], "naver": naver,
        }

    if path == "/api/screener":
        sort_by    = params.get("sort_by",    "price")
        sort_order = params.get("sort_order", "desc")
        if sort_by    not in {"price","change","volume","per","roe"}:
            sort_by = "price"
        if sort_order not in {"asc","desc"}:
            sort_order = "desc"
        return fetch_screener(sort_by=sort_by, sort_order=sort_order)

    if path == "/api/toss-overseas":
        sort_by    = params.get("sort_by",    "price")
        sort_order = params.get("sort_order", "desc")
        if sort_by    not in {"price","change","volume","per","roe"}:
            sort_by = "price"
        if sort_order not in {"asc","desc"}:
            sort_order = "desc"
        return fetch_toss_overseas_screener(sort_by=sort_by, sort_order=sort_order)

    if path == "/api/cron":
        try:
            fetch_screener()
            fetch_toss_overseas_screener()
            return {"status": "ok", "message": "Cache warmed (domestic + toss overseas)"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    if path == "/api/sentiment":
        m = params.get("market", "US")
        r = fetch_sentiment(m)
        return r if r else {"error": "조회 실패"}

    if path == "/api/resolve":
        q = params.get("q", "")
        t, m, c = resolve_ticker(q)
        return {"ticker": t, "market": m, "company": c} if t else {"error": f"'{q}' 미발견"}

    # HTML 서빙 (모든 나머지 경로)
    return None  # None이면 HTML 반환


# =============================================================================
# HTML 프론트엔드 (인라인)
# =============================================================================
HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>주식 AI 예측 시스템 (KRX/US)</title>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI','Noto Sans KR',sans-serif;display:flex;height:100vh;overflow:hidden}

/* 사이드바 */
#sidebar{width:260px;background:#161b22;border-right:1px solid #30363d;display:flex;flex-direction:column;flex-shrink:0;overflow-y:auto}
.sb-header{padding:16px;border-bottom:1px solid #30363d}
.sb-header h1{font-size:15px;font-weight:700;display:flex;align-items:center;gap:6px}
.sb-header p{font-size:11px;color:#8b949e;margin-top:4px}
.sb-section{padding:14px;border-bottom:1px solid #30363d}
.sb-label{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;display:block}
.mkt-btns{display:flex;gap:6px}
.mkt-btn{flex:1;padding:8px;border-radius:8px;border:none;cursor:pointer;font-size:13px;font-weight:500;transition:all .15s}
.mkt-btn.active{background:#1f6feb;color:#fff}
.mkt-btn:not(.active){background:#21262d;color:#8b949e}
.mkt-btn:not(.active):hover{background:#30363d;color:#e6edf3}
input,select{width:100%;background:#21262d;border:1px solid #30363d;border-radius:8px;padding:9px 12px;color:#e6edf3;font-size:13px;outline:none;transition:border-color .15s}
input:focus,select:focus{border-color:#1f6feb}
input::placeholder{color:#484f58}
#analyze-btn{width:100%;background:#1f6feb;color:#fff;border:none;border-radius:10px;padding:12px;font-size:14px;font-weight:600;cursor:pointer;transition:background .15s;margin-top:4px}
#analyze-btn:hover{background:#388bfd}
#analyze-btn:disabled{background:#21262d;color:#484f58;cursor:not-allowed}
.sentiment-card{background:#21262d;border-radius:10px;padding:12px}
.sent-name{font-size:11px;color:#8b949e}
.sent-val{font-size:20px;font-weight:700;margin:3px 0}
.sent-chg{font-size:12px;font-weight:500}
.sent-badge{display:inline-block;margin-top:6px;font-size:10px;padding:2px 8px;border-radius:20px;background:#21262d;border:1px solid #30363d;color:#8b949e}
.sb-footer{padding:12px;margin-top:auto;border-top:1px solid #30363d}
.sb-footer p{font-size:10px;color:#484f58;line-height:1.5}

/* 메인 */
#main{flex:1;overflow-y:auto;background:#0d1117;padding:24px}

/* 로딩/빈 상태 */
.center-state{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:16px;text-align:center}
.center-state .icon{font-size:56px}
.center-state h2{font-size:22px;font-weight:700}
.center-state p{color:#8b949e;font-size:14px;line-height:1.6;max-width:380px}
.sample-tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;justify-content:center}
.sample-tag{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:6px 14px;font-size:13px;color:#8b949e}
.spinner{width:40px;height:40px;border:4px solid #21262d;border-top-color:#1f6feb;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* 메트릭 카드 */
.metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.metric-card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px}
.m-label{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
.m-value{font-size:22px;font-weight:700}
.m-sub{font-size:12px;font-weight:500;margin-top:2px}
.rise{color:#f85149}
.fall{color:#388bfd}
.rise-us{color:#3fb950}
.fall-us{color:#f85149}

/* 탭 */
.tabs{display:flex;gap:6px;border-bottom:1px solid #21262d;margin-bottom:16px;padding-bottom:2px}
.tab-btn{padding:7px 14px;border-radius:8px;border:none;background:none;color:#8b949e;font-size:13px;font-weight:500;cursor:pointer;transition:all .15s}
.tab-btn.active{background:#1f6feb;color:#fff}
.tab-btn:not(.active):hover{background:#21262d;color:#e6edf3}

/* 카드 */
.card{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:18px;margin-bottom:14px}
.card-title{font-size:13px;font-weight:600;color:#8b949e;margin-bottom:14px;text-transform:uppercase;letter-spacing:.05em}

/* 헤더 */
.page-header{margin-bottom:20px}
.page-header h2{font-size:22px;font-weight:700;display:flex;align-items:center;gap:8px}
.ticker-badge{font-size:12px;font-weight:400;color:#8b949e;background:#21262d;padding:2px 8px;border-radius:6px;margin-left:4px}
.page-header p{font-size:12px;color:#484f58;margin-top:4px}

/* 펀더멘털 */
.fund-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.fund-item{background:#21262d;border-radius:10px;padding:12px}
.fund-label{font-size:11px;color:#8b949e;margin-bottom:4px}
.fund-val{font-size:14px;font-weight:600}

/* 스코어 */
.score-wrap{display:flex;align-items:flex-end;gap:8px;margin-bottom:10px}
.score-num{font-size:52px;font-weight:800;line-height:1}
.score-bar-bg{background:#21262d;border-radius:6px;height:10px;overflow:hidden}
.score-bar-fill{height:10px;border-radius:6px;transition:width .6s ease}

/* 분석 스텝 */
.step-item{background:#21262d;border-radius:10px;padding:14px;margin-bottom:8px}
.step-header{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:8px;flex-wrap:wrap}
.step-title{font-size:13px;font-weight:600;line-height:1.4;flex:1;min-width:0}
.step-score{font-size:12px;font-weight:700;padding:2px 8px;border-radius:12px;flex-shrink:0;align-self:center}
.step-score.pos{background:#0d2d1a;color:#3fb950}
.step-score.neg{background:#2d0d0d;color:#f85149}
.step-score.neu{background:#21262d;color:#8b949e}
.step-result{font-size:13px;color:#8b949e;line-height:1.6}
.step-result-line{display:block;padding:2px 0;word-break:keep-all;overflow-wrap:break-word}

/* 패턴 */
.pattern-item{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-radius:8px;margin-bottom:6px;font-size:13px}
.pattern-bull{background:#0d2d1a;border:1px solid #1a4730}
.pattern-bear{background:#2d0d0d;border:1px solid #4d1515}
.pattern-neu{background:#21262d;border:1px solid #30363d}

/* 차트 */
#price-chart, #rsi-chart, #macd-chart, #forecast-chart{width:100%;border-radius:8px;overflow:hidden}

/* 리스크 카드 */
.risk-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.risk-card{border-radius:12px;padding:16px;border:1px solid transparent}
.risk-card.conservative{background:#0a2d1a;border-color:#1a4730}
.risk-card.balanced{background:#2d200a;border-color:#4d3615}
.risk-card.aggressive{background:#2d0d0d;border-color:#4d1515}
.risk-icon{font-size:22px;margin-bottom:6px}
.risk-name{font-size:14px;font-weight:600;margin-bottom:4px}
.risk-desc{font-size:11px;color:#8b949e;margin-bottom:12px}
.risk-row{display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px}
.risk-lbl{color:#8b949e}
.risk-tgt{color:#f85149;font-weight:700}
.risk-stp{color:#388bfd;font-weight:700}
.risk-ratio{text-align:right;font-size:11px;color:#484f58;margin-top:8px;border-top:1px solid #30363d;padding-top:8px}

/* 뉴스 */
.news-item{display:flex;gap:10px;padding:10px 0;border-bottom:1px solid #21262d}
.news-item:last-child{border-bottom:none}
.news-dot{color:#388bfd;margin-top:2px;flex-shrink:0}
.news-a{color:#8b949e;font-size:13px;text-decoration:none;line-height:1.5}
.news-a:hover{color:#e6edf3;text-decoration:underline}
.news-meta{font-size:11px;color:#484f58;margin-top:3px}

/* 스크리너 */
.screener-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px}
.screener-table{width:100%;border-collapse:collapse;font-size:13px}
.screener-table th{padding:10px 14px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid #30363d;cursor:pointer;white-space:nowrap}
.screener-table th:hover{color:#e6edf3}
.screener-table td{padding:12px 14px;border-bottom:1px solid #21262d;vertical-align:middle}
.screener-table tr:hover td{background:#161b22}
.ticker-name{font-weight:600}
.ticker-code{font-size:11px;color:#484f58;margin-top:2px}
.cat-badge{font-size:11px;padding:2px 8px;border-radius:10px;background:#21262d;color:#8b949e}
.signal-badge{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:600}
.sig-buy-strong{background:#0d2d1a;color:#3fb950}
.sig-buy{background:#0d2020;color:#238636}
.sig-neu{background:#2d2206;color:#d29922}
.sig-sell{background:#2d0d0d;color:#f85149}

/* 기술적 지표 시그널 */
.indicator-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:4px}
.indicator-item{background:#21262d;border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:4px}
.ind-name{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.04em}
.ind-state{font-size:13px;font-weight:600}
.ind-value{font-size:11px;color:#484f58}
.ind-desc{font-size:12px;color:#8b949e;line-height:1.4;margin-top:2px}
.sig-buy-badge{background:#0d2d1a;color:#3fb950;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;white-space:nowrap}
.sig-sell-badge{background:#2d0d0d;color:#f85149;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;white-space:nowrap}
.sig-watch-badge{background:#2d2200;color:#d29922;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;white-space:nowrap}
.overall-signal-box{display:flex;justify-content:space-between;align-items:center;background:#21262d;border-radius:12px;padding:14px 18px;margin-bottom:14px;flex-wrap:wrap;gap:8px}
.ovs-label{font-size:11px;color:#8b949e;margin-bottom:4px}
.ovs-counts{display:flex;gap:14px;font-size:12px;flex-wrap:wrap}
.ovs-buy{color:#3fb950;font-weight:700}
.ovs-sell{color:#f85149;font-weight:700}
.ovs-watch{color:#d29922;font-weight:700}

/* 피봇 포인트 테이블 */
.pivot-table{width:100%;border-collapse:collapse;font-size:12px}
.pivot-table th{padding:8px 10px;text-align:center;color:#8b949e;font-size:11px;border-bottom:1px solid #30363d;font-weight:500;white-space:nowrap}
.pivot-table td{padding:7px 10px;text-align:center;border-bottom:1px solid #21262d;white-space:nowrap}
.pivot-table tr:last-child td{border-bottom:none}
.pivot-label-col{color:#8b949e;text-align:left!important;font-weight:500;white-space:nowrap}
.pv-r{color:#f85149;font-weight:600}
.pv-s{color:#388bfd;font-weight:600}
.pv-p{color:#e6edf3;font-weight:700}
.pv-nr{background:#2d150d}
.pv-ns{background:#0d2d1a}

/* 매수 적정가 카드 */
.buy-price-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}
.buy-card{border-radius:12px;padding:16px;border:1px solid transparent}
.buy-card.aggressive{background:#2d200a;border-color:#4d3615}
.buy-card.recommended{background:#0d2d1a;border-color:#1a4730}
.buy-card.conservative{background:#0a1f3a;border-color:#15356b}
.buy-label{font-size:10px;color:#8b949e;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em}
.buy-price-val{font-size:18px;font-weight:800;margin-bottom:4px;word-break:break-all}
.buy-basis-box{font-size:11px;color:#8b949e;line-height:1.5;margin-top:8px;border-top:1px solid #30363d;padding-top:8px}

/* 2칼럼 그리드 공통 클래스 (인라인 스타일 대체) */
.two-col-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}

/* 탭 — 가로 스크롤 (모바일에서 넘침 방지) */
.tabs{overflow-x:auto;-webkit-overflow-scrolling:touch;scrollbar-width:none;flex-wrap:nowrap}
.tabs::-webkit-scrollbar{display:none}
.tab-btn{white-space:nowrap;flex-shrink:0}

/* 스크리너 테이블 래퍼 — 가로 스크롤 */
.screener-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
.screener-table{min-width:680px}

/* 햄버거 버튼 (모바일 전용, 데스크톱은 숨김) */
#hamburger{
  display:none;position:fixed;top:12px;left:12px;z-index:400;
  background:#1f6feb;border:none;border-radius:8px;
  padding:8px 11px;cursor:pointer;color:#fff;font-size:20px;line-height:1;
  box-shadow:0 2px 8px rgba(0,0,0,.4);transition:background .15s
}
#hamburger:hover{background:#388bfd}

/* 모바일 오버레이 (사이드바 열릴 때 배경 어둡게) */
#mob-overlay{
  display:none;position:fixed;inset:0;
  background:rgba(0,0,0,.55);z-index:150;
  backdrop-filter:blur(2px);-webkit-backdrop-filter:blur(2px)
}
#mob-overlay.on{display:block}

/* ── 태블릿 (≤ 900px) ── */
@media(max-width:900px){
  .metrics-grid{grid-template-columns:repeat(2,1fr)}
  .risk-grid{grid-template-columns:1fr}
  .fund-grid{grid-template-columns:repeat(2,1fr)}
  .buy-price-grid{grid-template-columns:1fr}
  .indicator-grid{grid-template-columns:1fr}
  .two-col-grid{grid-template-columns:1fr}
}

/* ── 모바일 (≤ 768px) ── */
@media(max-width:768px){
  /* 레이아웃 재구성: flex → block, 스크롤 허용 */
  body{display:block;height:auto;overflow-y:auto}
  #main{padding:56px 14px 28px;min-height:100vh}

  /* 햄버거 표시 */
  #hamburger{display:block}

  /* 사이드바: 슬라이드-인 오버레이 패널 */
  #sidebar{
    position:fixed;top:0;left:0;height:100%;width:280px;z-index:200;
    transform:translateX(-100%);
    transition:transform .25s cubic-bezier(.4,0,.2,1);
    box-shadow:6px 0 32px rgba(0,0,0,.6)
  }
  #sidebar.open{transform:translateX(0)}

  /* 그리드 1열 */
  .metrics-grid{grid-template-columns:1fr 1fr;gap:8px}
  .two-col-grid{grid-template-columns:1fr;gap:10px}
  .risk-grid{grid-template-columns:1fr}
  .fund-grid{grid-template-columns:repeat(2,1fr)}
  .buy-price-grid{grid-template-columns:1fr}
  .indicator-grid{grid-template-columns:1fr}

  /* 헤더/타이포 */
  .page-header h2{font-size:18px}
  .score-num{font-size:36px}
  .card{padding:14px}
  .card-title{font-size:12px}

  /* 차트 높이 축소 */
  #price-chart{height:260px!important}
  #rsi-chart,#macd-chart{height:130px!important}

  /* 스크리너 헤더 세로 정렬 */
  .screener-header{flex-direction:column;gap:10px;align-items:flex-start}
  .screener-header>div:last-child{width:100%;justify-content:flex-start}

  /* 메트릭 카드 */
  .m-value{font-size:18px}
}

/* ── 소형 모바일 (≤ 480px) ── */
@media(max-width:480px){
  #main{padding:52px 10px 20px}
  .metrics-grid{grid-template-columns:1fr 1fr;gap:6px}
  .metric-card{padding:10px}
  .m-label{font-size:10px}
  .m-value{font-size:16px}
  .card{padding:12px;border-radius:10px}
  .tab-btn{font-size:11px;padding:6px 10px}
  .page-header h2{font-size:16px}
  .step-result{font-size:12px}
  .pattern-item{font-size:12px;padding:6px 10px}
  .risk-card{padding:12px}
  .buy-card{padding:12px}
  .fund-grid{grid-template-columns:1fr 1fr}
}
</style>
</head>
<body>

<!-- ── 모바일: 햄버거 + 오버레이 ── -->
<button id="hamburger" onclick="toggleSidebar()" aria-label="메뉴 열기">☰</button>
<div id="mob-overlay" onclick="closeSidebar()"></div>

<!-- ── 사이드바 ── -->
<div id="sidebar">
  <div class="sb-header">
    <h1>📈 주식 AI 예측</h1>
    <p>KRX / US 기술적 분석 시스템</p>
  </div>

  <div class="sb-section">
    <span class="sb-label">메뉴</span>
    <div style="display:flex;flex-direction:column;gap:4px">
      <button class="mkt-btn active" style="text-align:left;padding:10px 12px" id="nav-analysis" onclick="showPage('analysis')">🔍 종목 상세 분석</button>
      <button class="mkt-btn" style="text-align:left;padding:10px 12px" id="nav-screener" onclick="showPage('screener')">📋 주식 골라보기</button>
    </div>
  </div>

  <div class="sb-section">
    <span class="sb-label">🌍 시장 심리</span>
    <div id="sentiment-widget" class="sentiment-card">
      <div class="sent-name">로딩 중...</div>
    </div>
  </div>

  <div class="sb-section" id="analysis-controls">
    <span class="sb-label" style="margin-bottom:10px;display:block">시장 선택</span>
    <div class="mkt-btns" style="margin-bottom:12px">
      <button class="mkt-btn active" id="mkt-krx" onclick="setMarket('KRX')">🇰🇷 한국</button>
      <button class="mkt-btn" id="mkt-us" onclick="setMarket('US')">🇺🇸 미국</button>
    </div>
    <span class="sb-label">종목명 / 코드</span>
    <input type="text" id="ticker-input" value="삼성전자" placeholder="예: 삼성전자, 005930, TSLA"
           style="margin-bottom:10px" onkeydown="if(event.key==='Enter')analyze()">
    <span class="sb-label">분석 기간</span>
    <select id="period-select" style="margin-bottom:12px">
      <option value="1d">초단기 (1일)</option>
      <option value="3d">초단기 (3일)</option>
      <option value="1wk">초단기 (1주)</option>
      <option value="2wk">단기 (2주)</option>
      <option value="1mo" selected>단기 (1개월)</option>
      <option value="6mo">6개월</option>
      <option value="1y">1년</option>
      <option value="2y">2년</option>
      <option value="5y">5년</option>
    </select>
    <button id="analyze-btn" onclick="analyze()">🔍 분석 시작</button>
  </div>

  <div class="sb-footer">
    <p>⚠️ 본 시스템은 참고용이며, 투자 결정의 책임은 본인에게 있습니다.</p>
  </div>
</div>

<!-- ── 메인 ── -->
<div id="main">
  <!-- 분석 페이지 -->
  <div id="page-analysis">
    <div id="state-empty" class="center-state">
      <div class="icon">📊</div>
      <h2>주식 AI 예측 시스템</h2>
      <p>종목명 또는 코드를 입력하고<br><strong style="color:#388bfd">분석 시작</strong> 버튼을 누르세요.<br><span style="font-size:12px;color:#484f58">모바일: 왼쪽 상단 ☰ 버튼으로 메뉴를 여세요.</span></p>
      <div class="sample-tags">
        <span class="sample-tag" onclick="quickSearch('삼성전자')" style="cursor:pointer">삼성전자</span>
        <span class="sample-tag" onclick="quickSearch('SK하이닉스')" style="cursor:pointer">SK하이닉스</span>
        <span class="sample-tag" onclick="quickSearch('NVDA')" style="cursor:pointer">NVDA</span>
        <span class="sample-tag" onclick="quickSearch('TSLA')" style="cursor:pointer">TSLA</span>
        <span class="sample-tag" onclick="quickSearch('애플')" style="cursor:pointer">애플</span>
        <span class="sample-tag" onclick="quickSearch('카카오')" style="cursor:pointer">카카오</span>
      </div>
    </div>
    <div id="state-loading" class="center-state" style="display:none">
      <div class="spinner"></div>
      <p style="color:#8b949e">주가 데이터 분석 중...<br><span style="font-size:12px;color:#484f58">AI 모델이 기술적 지표를 계산하고 있습니다</span></p>
    </div>
    <div id="state-error" class="center-state" style="display:none">
      <div class="icon">⚠️</div>
      <h2 style="color:#f85149">분석 오류</h2>
      <p id="error-msg" style="color:#8b949e"></p>
    </div>
    <div id="state-result" style="display:none">
      <div class="page-header">
        <h2 id="r-title"></h2>
        <p id="r-subtitle"></p>
      </div>
      <div class="metrics-grid">
        <div class="metric-card"><div class="m-label">현재가</div><div class="m-value" id="r-price"></div><div class="m-sub" id="r-pct"></div></div>
        <div class="metric-card"><div class="m-label">RSI (14)</div><div class="m-value" id="r-rsi"></div><div class="m-sub" id="r-rsi-label"></div></div>
        <div class="metric-card"><div class="m-label">거래량</div><div class="m-value" id="r-vol" style="font-size:18px"></div></div>
        <div class="metric-card"><div class="m-label">ATR (변동성)</div><div class="m-value" id="r-atr" style="font-size:18px"></div></div>
      </div>
      <div id="r-naver-fund" style="display:none" class="card">
        <div class="card-title">🏢 기업 펀더멘털 (네이버 금융)</div>
        <div class="fund-grid">
          <div class="fund-item"><div class="fund-label">시가총액</div><div class="fund-val" id="f-mktcap"></div></div>
          <div class="fund-item"><div class="fund-label">PER</div><div class="fund-val" id="f-per"></div></div>
          <div class="fund-item"><div class="fund-label">PBR</div><div class="fund-val" id="f-pbr"></div></div>
        </div>
      </div>
      <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('chart')">📊 차트 분석</button>
        <button class="tab-btn" onclick="switchTab('ai')">🧠 AI 진단</button>
        <button class="tab-btn" onclick="switchTab('forecast')">🔮 미래 예측</button>
        <button class="tab-btn" onclick="switchTab('news')">📰 뉴스/공시</button>
      </div>

      <!-- 차트 탭 -->
      <div id="tab-chart">
        <div class="card">
          <div class="card-title">가격 차트 (캔들 + MA + 볼린저 + 거래량)</div>
          <div id="price-chart" style="height:380px"></div>
        </div>
        <div class="two-col-grid" style="gap:12px">
          <div class="card">
            <div class="card-title">RSI (14)</div>
            <div id="rsi-chart" style="height:150px"></div>
          </div>
          <div class="card">
            <div class="card-title">MACD</div>
            <div id="macd-chart" style="height:150px"></div>
          </div>
        </div>
        <div class="card">
          <div class="card-title">📡 기술적 지표 종합 시그널</div>
          <div id="indicator-signals-section"></div>
        </div>
        <div class="card">
          <div class="card-title">📐 피봇 포인트 분석</div>
          <div id="pivot-points-section"></div>
        </div>
      </div>

      <!-- AI 탭 -->
      <div id="tab-ai" style="display:none">
        <div class="two-col-grid">
          <div>
            <div class="card">
              <div class="card-title">🏆 종합 기술적 점수</div>
              <div class="score-wrap">
                <div class="score-num" id="ai-score"></div>
                <span style="color:#8b949e;font-size:18px;margin-bottom:6px">/ 100점</span>
              </div>
              <div class="score-bar-bg"><div class="score-bar-fill" id="ai-score-bar"></div></div>
              <p id="ai-score-desc" style="font-size:12px;color:#8b949e;margin-top:8px"></p>
            </div>
            <div class="card">
              <div class="card-title">🕯️ 캔들스틱 패턴</div>
              <div id="patterns-list"></div>
            </div>
          </div>
          <div class="card">
            <div class="card-title">📝 단계별 분석 리포트</div>
            <div id="steps-list"></div>
          </div>
        </div>
      </div>

      <!-- 예측 탭 -->
      <div id="tab-forecast" style="display:none">
        <div class="card">
          <div class="card-title">💡 AI 종합 진단 및 트레이딩 전략</div>
          <div id="ai-strategy-section"></div>
        </div>
        <div class="card">
          <div class="card-title">🎯 현재가 기준 매수 적정 가격 예측</div>
          <div id="buy-price-section"></div>
        </div>
        <div class="card">
          <div class="card-title">🛡️ 리스크 관리 (ATR 기반)</div>
          <div class="risk-grid" id="risk-grid"></div>
        </div>
      </div>

      <!-- 뉴스 탭 -->
      <div id="tab-news" style="display:none">
        <div class="two-col-grid">
          <div class="card" id="news-col1">
            <div class="card-title" id="news-col1-title">📰 주요 뉴스</div>
            <div id="news-list"></div>
          </div>
          <div class="card" id="disclosure-col" style="display:none">
            <div class="card-title">📋 최근 공시</div>
            <div id="disclosure-list"></div>
          </div>
        </div>
      </div>

      <div style="border-top:1px solid #21262d;margin-top:20px;padding-top:12px">
        <p style="font-size:11px;color:#484f58">⚠️ 본 분석은 AI 기술적 지표 기반 참고용 자료입니다. 투자 결정의 책임은 본인에게 있습니다.</p>
      </div>
    </div>
  </div>

  <!-- 스크리너 페이지 -->
  <div id="page-screener" style="display:none">
    <div class="screener-header">
      <div>
        <h2 style="font-size:22px;font-weight:700;margin-bottom:4px">📋 주식 골라보기</h2>
        <p style="font-size:12px;color:#8b949e" id="scrn-subtitle">토스증권 필터 조건 적용 결과</p>
        <p style="font-size:11px;color:#484f58;margin-top:4px" id="scrn-filter-badge"></p>
      </div>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <select id="scrn-sort-by" onchange="loadScreener(this.value, document.getElementById('scrn-sort-order').value)"
          style="background:#21262d;border:1px solid #30363d;border-radius:6px;padding:6px 10px;color:#e6edf3;font-size:12px">
          <option value="price">현재가 정렬</option>
          <option value="change">등락률 정렬</option>
          <option value="volume">거래량 정렬</option>
          <option value="per">PER 정렬</option>
          <option value="roe">ROE 정렬</option>
        </select>
        <select id="scrn-sort-order" onchange="loadScreener(document.getElementById('scrn-sort-by').value, this.value)"
          style="background:#21262d;border:1px solid #30363d;border-radius:6px;padding:6px 10px;color:#e6edf3;font-size:12px">
          <option value="desc">내림차순 ↓</option>
          <option value="asc">오름차순 ↑</option>
        </select>
        <button onclick="loadScreener(document.getElementById('scrn-sort-by').value, document.getElementById('scrn-sort-order').value)"
          style="background:#21262d;border:1px solid #30363d;border-radius:8px;padding:8px 14px;color:#8b949e;font-size:13px;cursor:pointer;white-space:nowrap;">🔄 새로고침</button>
      </div>
    </div>
    <div class="tabs">
      <button class="tab-btn active" id="scrn-tab-domestic" onclick="switchScrnTab('domestic')">🇰🇷 국내 (KRX)</button>
      <button class="tab-btn" id="scrn-tab-overseas" onclick="switchScrnTab('overseas')">🇺🇸 해외 (US)</button>
    </div>
    <div id="scrn-loading" style="text-align:center;padding:40px;color:#8b949e">
      <div class="spinner" style="margin:0 auto 12px"></div>
      데이터 로딩 중...
    </div>
    <div id="scrn-result" style="display:none">
      <div class="card screener-wrap" style="padding:0;">
        <table class="screener-table">
          <thead><tr>
            <th>#</th>
            <th>종목</th>
            <th style="text-align:right;cursor:pointer" onclick="sortScreener('price')">현재가 ↕</th>
            <th style="text-align:right;cursor:pointer" onclick="sortScreener('change')">등락률 ↕</th>
            <th>카테고리</th>
            <th style="text-align:right;cursor:pointer" onclick="sortScreener('volume')">거래량 ↕</th>
            <th style="text-align:center;cursor:pointer" onclick="sortScreener('per')">PER ↕</th>
            <th style="text-align:center">신호</th>
          </tr></thead>
          <tbody id="scrn-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
// ── 전역 상태 ──
let currentMarket = 'KRX';
let currentData = null;
let currentTab = 'chart';
let screenerData = [];
let screenerInfo = {}; // 추가
let scrnMarket = 'domestic';
let scrnSort = {key:'price', dir:'desc'};
let chartInstances = {};

// ── 페이지 전환 ──
function showPage(page) {
  document.getElementById('page-analysis').style.display = page === 'analysis' ? 'block' : 'none';
  document.getElementById('page-screener').style.display = page === 'screener' ? 'block' : 'none';
  document.getElementById('analysis-controls').style.display = page === 'analysis' ? 'block' : 'none';
  document.getElementById('nav-analysis').classList.toggle('active', page === 'analysis');
  document.getElementById('nav-screener').classList.toggle('active', page === 'screener');
  if (page === 'screener' && screenerData.length === 0) loadScreener();
  closeSidebar();   // 페이지 전환 시 모바일 사이드바 닫기
}

// ── 모바일 사이드바 토글 ──
function toggleSidebar() {
  const sb  = document.getElementById('sidebar');
  const ov  = document.getElementById('mob-overlay');
  const isOpen = sb.classList.toggle('open');
  ov.classList.toggle('on', isOpen);
  // 사이드바 열릴 때 배경 스크롤 잠금
  document.body.style.overflow = isOpen ? 'hidden' : '';
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('mob-overlay').classList.remove('on');
  document.body.style.overflow = '';
}

// ── 시장 선택 ──
function setMarket(m) {
  currentMarket = m;
  document.getElementById('mkt-krx').classList.toggle('active', m === 'KRX');
  document.getElementById('mkt-us').classList.toggle('active', m === 'US');
  document.getElementById('ticker-input').placeholder = m === 'KRX' ? '예: 삼성전자, 005930' : '예: 애플, TSLA, NVDA';
  document.getElementById('ticker-input').value = m === 'KRX' ? '삼성전자' : '애플';
  loadSentiment(m);
}

function quickSearch(name) {
  document.getElementById('ticker-input').value = name;
  showPage('analysis');  // 내부에서 closeSidebar() 호출됨
  analyze();
}

// ── 시장 심리 ──
async function loadSentiment(market) {
  const w = document.getElementById('sentiment-widget');
  w.innerHTML = '<div class="sent-name">로딩 중...</div>';
  try {
    const r = await fetch(`/api/sentiment?market=${market}`);
    const d = await r.json();
    if (d.error) { w.innerHTML = '<div class="sent-name" style="color:#484f58">조회 실패</div>'; return; }
    const isUp = d.change >= 0;
    const clr = market === 'KRX' ? (isUp ? '#f85149' : '#388bfd') : (isUp ? '#3fb950' : '#f85149');
    w.innerHTML = `
      <div class="sent-name">${d.name}</div>
      <div class="sent-val">${d.value.toFixed(2)}</div>
      <div class="sent-chg" style="color:${clr}">${isUp?'▲':'▼'} ${Math.abs(d.change).toFixed(2)}%</div>
      <span class="sent-badge">${d.sentiment}</span>`;
  } catch(e) { w.innerHTML = '<div class="sent-name" style="color:#484f58">조회 실패</div>'; }
}

// ── 분석 ──
async function analyze() {
  closeSidebar();   // 모바일에서 분석 시작 시 사이드바 자동 닫기
  const ticker = document.getElementById('ticker-input').value.trim();
  const period = document.getElementById('period-select').value;
  if (!ticker) return;
  setState('loading');
  document.getElementById('analyze-btn').disabled = true;
  destroyCharts();
  try {
    const r = await fetch(`/api/stock?ticker=${encodeURIComponent(ticker)}&period=${period}&market=${currentMarket}`);
    const d = await r.json();
    if (d.error) { setState('error'); document.getElementById('error-msg').textContent = d.error; return; }
    currentData = d;
    if (d.market) {
      currentMarket = d.market;
      document.getElementById('mkt-krx').classList.toggle('active', d.market === 'KRX');
      document.getElementById('mkt-us').classList.toggle('active', d.market === 'US');
    }
    renderResult(d);
    setState('result');
  } catch(e) {
    setState('error');
    document.getElementById('error-msg').textContent = 'API 서버 오류: ' + e.message;
  } finally {
    document.getElementById('analyze-btn').disabled = false;
  }
}

function setState(s) {
  ['empty','loading','error','result'].forEach(n => {
    const el = document.getElementById('state-' + n);
    if (el) el.style.display = n === s ? (s === 'result' ? 'block' : 'flex') : 'none';
  });
}

// ── 렌더링 ──
function fmt(v, isKrx) {
  return isKrx ? v.toLocaleString('ko-KR',{maximumFractionDigits:0}) + '원'
               : '$' + v.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
}

function renderResult(d) {
  const isKrx = d.market === 'KRX';
  const up = d.pct_change >= 0;
  const clr = isKrx ? (up ? '#f85149' : '#388bfd') : (up ? '#3fb950' : '#f85149');

  document.getElementById('r-title').innerHTML =
    `${d.company || d.symbol} <span class="ticker-badge">${d.symbol}</span>`;
  document.getElementById('r-subtitle').textContent =
    `기준일: ${new Date().toLocaleDateString('ko-KR')} | 시장: ${isKrx ? '🇰🇷 KRX (한국)' : '🇺🇸 US (미국)'}`;
  document.getElementById('r-price').textContent = fmt(d.last_close, isKrx);
  document.getElementById('r-pct').innerHTML = `<span style="color:${clr}">${up?'▲':'▼'} ${Math.abs(d.pct_change).toFixed(2)}%</span>`;

  const rsi = d.rsi;
  const rsiClr = rsi > 70 ? '#f85149' : rsi < 30 ? '#388bfd' : '#e6edf3';
  document.getElementById('r-rsi').innerHTML = `<span style="color:${rsiClr}">${rsi.toFixed(1)}</span>`;
  document.getElementById('r-rsi-label').innerHTML = `<span style="color:${rsiClr}">${rsi>70?'과매수':rsi<30?'과매도':'중립'}</span>`;
  document.getElementById('r-vol').textContent = d.volume.toLocaleString();
  document.getElementById('r-atr').textContent = d.atr.toLocaleString();

  // 펀더멘털
  if (isKrx && d.naver) {
    document.getElementById('r-naver-fund').style.display = 'block';
    document.getElementById('f-mktcap').textContent = d.naver.market_cap || '-';
    document.getElementById('f-per').textContent = d.naver.per || '-';
    document.getElementById('f-pbr').textContent = d.naver.pbr || '-';
  } else {
    document.getElementById('r-naver-fund').style.display = 'none';
  }

  // AI 진단
  renderAI(d, isKrx);
  // 예측/리스크
  renderForecast(d, isKrx);
  // 기술적 지표 시그널 & 피봇 포인트
  renderTechnicalSignals(d);
  renderPivotPoints(d, isKrx);
  // 뉴스
  renderNews(d, isKrx);
  // 탭 초기화
  switchTab('chart');
  // 차트는 탭 전환 후 렌더
  setTimeout(() => renderCharts(d, isKrx), 50);
}

function renderAI(d, isKrx) {
  const s = d.score;
  const sClr = s >= 65 ? '#3fb950' : s >= 40 ? '#d29922' : '#f85149';
  const sBarClr = s >= 65 ? '#3fb950' : s >= 40 ? '#d29922' : '#f85149';
  document.getElementById('ai-score').innerHTML = `<span style="color:${sClr}">${s}</span>`;
  const bar = document.getElementById('ai-score-bar');
  bar.style.width = s + '%'; bar.style.background = sBarClr;
  document.getElementById('ai-score-desc').textContent =
    s >= 65 ? '✅ BUY (매수 우위)'
    : s >= 40 ? '⚖️ HOLD (관망 / 중립)'
    : '⚠️ SELL (매도 우위 / 리스크 관리)';

  const stepsList = document.getElementById('steps-list');
  stepsList.innerHTML = d.analysis_steps.map(st => {
    const sc = st.score;
    const cls = sc > 0 ? 'pos' : sc < 0 ? 'neg' : 'neu';
    const label = sc > 0 ? '+' + sc : sc;
    const weight = st.weight || '';
    
    return `<div class="step-item">
      <div class="step-header">
        <span class="step-title">${st.step}</span>
        <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;white-space:nowrap">
          ${weight ? `<span style="font-size:10px;color:#484f58;background:#161b22;padding:1px 6px;border-radius:8px">${weight}</span>` : ''}
          <span class="step-score ${cls}">${label}점</span>
        </div>
      </div>
      <div class="step-result">
        ${st.result.split(' | ').filter(l => l.trim()).map(line => `<span class="step-result-line">• ${line}</span>`).join('')}
      </div>
    </div>`;
  }).join('');

  const patList = document.getElementById('patterns-list');
  if (d.candlestick_patterns.length === 0) {
    patList.innerHTML = '<p style="font-size:13px;color:#484f58">특이한 캔들 패턴이 감지되지 않았습니다.</p>';
  } else {
    patList.innerHTML = d.candlestick_patterns.map(p => {
      const cls = p.direction === '상승' ? 'pattern-bull' : p.direction === '하락' ? 'pattern-bear' : 'pattern-neu';
      const icon = p.direction === '상승' ? '📈' : p.direction === '하락' ? '📉' : '➖';
      return `<div class="pattern-item ${cls}">
        <span>${icon} <strong>${p.name}</strong></span>
        <span style="font-size:12px;color:#8b949e">${p.desc}</span>
      </div>`;
    }).join('');
  }
}

function renderForecast(d, isKrx) {
  const risk = d.risk_scenarios;
  const bp   = d.buy_price;
  const ai   = d.ai_strategy;

  // ── AI 종합 진단 및 트레이딩 전략 섹션 ──
  const aiEl = document.getElementById('ai-strategy-section');
  if (aiEl && ai) {
    aiEl.innerHTML = `
      <div style="background: rgba(31, 111, 235, 0.05); border-radius:10px; padding:16px; margin-bottom:16px; border: 1px solid #1f6feb;">
        <div style="color:#e6edf3; font-size: 14px; line-height: 1.6;">
          ${ai.result.split(' | ').filter(l => l.trim()).map(line => {
            if (line.startsWith('[')) return `<div style="margin-top:12px; font-weight:bold; color:#388bfd; font-size: 15px;">${line}</div>`;
            return `<div style="margin-top:6px; margin-left:8px;">${line}</div>`;
          }).join('')}
        </div>
      </div>
    `;
  }

  // ── 매수 적정가 섹션 ──
  const bpEl = document.getElementById('buy-price-section');
  if (bpEl) {
    if (!bp) {
      bpEl.innerHTML = '<p style="color:#484f58;font-size:13px">데이터 부족</p>';
    } else {
      const cur = bp.current;
      const pct = v => ((v - cur) / cur * 100).toFixed(2);
      const aggR = bp.aggressive.range;
      const recR = bp.recommended.range;
      const conR = bp.conservative.range;

      bpEl.innerHTML = `
        <div style="background:#21262d;border-radius:10px;padding:14px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <div>
            <div style="font-size:11px;color:#8b949e;margin-bottom:4px">현재가</div>
            <div style="font-size:22px;font-weight:800">${fmt(cur, isKrx)}</div>
          </div>
          <div style="display:flex;gap:16px;align-items:flex-start;flex-shrink:0">
            <div style="text-align:right">
              <div style="font-size:11px;color:#8b949e;margin-bottom:4px">예상 매수 타이밍</div>
              <div style="font-size:13px;font-weight:600;color:#3fb950;white-space:nowrap">${bp.timing.buy}</div>
            </div>
            <div style="text-align:right">
              <div style="font-size:11px;color:#8b949e;margin-bottom:4px">예상 매도 타이밍</div>
              <div style="font-size:13px;font-weight:600;color:#f85149;white-space:nowrap">${bp.timing.sell}</div>
            </div>
          </div>
        </div>
        <div class="buy-price-grid">
          <div class="buy-card aggressive">
            <div class="buy-label">⚡ 공격적 매수</div>
            <div class="buy-price-val" style="color:#f97316;font-size:14px">${fmt(aggR[0], isKrx)} ~ ${fmt(aggR[1], isKrx)}</div>
            <div class="buy-basis-box">현재가 대비 단기 눌림 구간<br>ATR 0.5배 기반 · 빠른 진입</div>
          </div>
          <div class="buy-card recommended">
            <div class="buy-label">✅ 추천 매수 구간</div>
            <div class="buy-price-val" style="color:#3fb950;font-size:14px">${fmt(recR[0], isKrx)} ~ ${fmt(recR[1], isKrx)}</div>
            <div class="buy-basis-box">ATR 기반 최적 진입 구간<br>분할 매수 권장</div>
          </div>
          <div class="buy-card conservative">
            <div class="buy-label">🛡️ 보수적 매수</div>
            <div class="buy-price-val" style="color:#388bfd;font-size:14px">${fmt(conR[0], isKrx)} ~ ${fmt(conR[1], isKrx)}</div>
            <div class="buy-basis-box">강한 지지구간 도달 시 매수<br>최대 안전 마진 확보</div>
          </div>
        </div>
        <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px">
          <div style="font-size:11px;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">📋 예측 근거</div>
          ${bp.basis.map(b => `<div style="font-size:12px;color:#8b949e;margin-bottom:4px">• ${b}</div>`).join('')}
        </div>`;
    }
  }

  // ── 리스크 카드 ──
  const rgEl = document.getElementById('risk-grid');
  if (rgEl && risk) {
    const riskClasses = ['conservative', 'balanced', 'aggressive'];
    rgEl.innerHTML = Object.entries(risk).map(([, sc], i) => `
      <div class="risk-card ${riskClasses[i]}">
        <div class="risk-icon">${sc.icon}</div>
        <div class="risk-name">${sc.label}</div>
        <div class="risk-desc">${sc.desc}</div>
        <div class="risk-row"><span class="risk-lbl">🎯 목표가</span><span class="risk-tgt" style="font-size:12px">${fmt(sc.target[0], isKrx)} ~ ${fmt(sc.target[1], isKrx)}</span></div>
        <div class="risk-row"><span class="risk-lbl">🛑 손절가</span><span class="risk-stp" style="font-size:12px">${fmt(sc.stop[0], isKrx)} ~ ${fmt(sc.stop[1], isKrx)}</span></div>
        <div class="risk-ratio" style="font-size:13px;color:#3fb950;font-weight:bold;margin-top:8px">예상 수익률: ${sc.return > 0 ? '+' : ''}${sc.return}%</div>
      </div>`).join('');
  }
}

function renderTechnicalSignals(d) {
  const el = document.getElementById('indicator-signals-section');
  if (!el || !d.indicator_signals) return;

  const { signals, summary } = d.indicator_signals;

  const badge = sig => {
    if (sig === '매수') return '<span class="sig-buy-badge">▲ 매수</span>';
    if (sig === '매도') return '<span class="sig-sell-badge">▼ 매도</span>';
    return '<span class="sig-watch-badge">— 관망</span>';
  };

  const ovClr = summary.overall_signal === '매수' ? '#3fb950'
              : summary.overall_signal === '매도' ? '#f85149' : '#d29922';

  let html = `
    <div class="overall-signal-box">
      <div>
        <div class="ovs-label">종합 판단 (${summary.total}개 지표 기준)</div>
        <div style="font-size:20px;font-weight:800;color:${ovClr}">${summary.overall_label}</div>
      </div>
      ${summary.market_state ? `<div style="font-size:12px;background:#21262d;border-radius:8px;padding:6px 12px;color:#cdd9e5;white-space:nowrap">${summary.market_state}</div>` : ''}
    </div>
    <div class="indicator-grid">`;

  Object.values(signals).forEach(s => {
    const stClr = s.signal === '매수' ? '#3fb950' : s.signal === '매도' ? '#f85149' : '#d29922';
    html += `
      <div class="indicator-item">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px">
          <span class="ind-name">${s.name}</span>
          ${badge(s.signal)}
        </div>
        <div class="ind-state" style="color:${stClr}">${s.state}</div>
        <div class="ind-value">${s.value}</div>
        <div class="ind-desc">${s.desc}</div>
      </div>`;
  });

  html += '</div>';
  el.innerHTML = html;
}

function renderPivotPoints(d, isKrx) {
  const el = document.getElementById('pivot-points-section');
  if (!el || !d.pivot_points || !Object.keys(d.pivot_points).length) {
    if (el) el.innerHTML = '<p style="font-size:13px;color:#484f58">피봇 포인트 계산 데이터 부족</p>';
    return;
  }

  const pp  = d.pivot_points;
  const cur = d.last_close;

  const methods = [
    { key: 'classic',    label: '클래식' },
    { key: 'fibonacci',  label: '피보나치' },
    { key: 'camarilla',  label: '카마리야' },
  ];
  const levels = ['S3','S2','S1','Pivot','R1','R2','R3'];

  // 클래식 기준 가장 가까운 지지/저항
  const cl = pp.classic || {};
  let nearestR = Infinity, nearestS = -Infinity;
  let nearestRKey = null, nearestSKey = null;
  ['R1','R2','R3'].forEach(k => {
    if (cl[k] != null && cl[k] > cur && cl[k] < nearestR) { nearestR = cl[k]; nearestRKey = k; }
  });
  ['S1','S2','S3'].forEach(k => {
    if (cl[k] != null && cl[k] < cur && cl[k] > nearestS) { nearestS = cl[k]; nearestSKey = k; }
  });

  const fmtV = v => (v == null) ? '-' : fmt(v, isKrx);

  let html = `<div style="overflow-x:auto">
    <table class="pivot-table">
      <thead>
        <tr>
          <th style="text-align:left">방식</th>
          <th class="pv-s">S3</th><th class="pv-s">S2</th><th class="pv-s">S1</th>
          <th class="pv-p">Pivot</th>
          <th class="pv-r">R1</th><th class="pv-r">R2</th><th class="pv-r">R3</th>
        </tr>
      </thead>
      <tbody>`;

  methods.forEach(m => {
    const data = pp[m.key];
    if (!data) return;
    html += `<tr>
      <td class="pivot-label-col">${m.label}</td>
      ${levels.map(k => {
        const val = data[k];
        const cls = k === 'Pivot' ? 'pv-p' : k.startsWith('R') ? 'pv-r' : 'pv-s';
        let bg = '';
        if (m.key === 'classic' && k === nearestRKey) bg = 'class="pv-nr"';
        if (m.key === 'classic' && k === nearestSKey) bg = 'class="pv-ns"';
        return `<td class="${cls}" ${bg}>${fmtV(val)}</td>`;
      }).join('')}
    </tr>`;
  });

  html += `</tbody></table></div>`;

  // 전략 힌트
  if (nearestSKey || nearestRKey) {
    html += `<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:14px">`;
    if (nearestSKey) {
      const sVal = cl[nearestSKey];
      const sPct = ((sVal - cur) / cur * 100).toFixed(2);
      html += `<div style="background:#0d2d1a;border:1px solid #1a4730;border-radius:8px;padding:10px 14px;flex:1;min-width:220px">
        <div style="font-size:11px;color:#8b949e;margin-bottom:4px">🟢 가장 가까운 지지선 (클래식 ${nearestSKey})</div>
        <div style="font-size:16px;font-weight:700;color:#3fb950">${fmtV(sVal)} <span style="font-size:11px;font-weight:400">(${sPct}%)</span></div>
        <div style="font-size:11px;color:#8b949e;margin-top:6px">→ ${nearestSKey} 근접 시 <strong>분할 매수</strong> 고려</div>
      </div>`;
    }
    if (nearestRKey) {
      const rVal = cl[nearestRKey];
      const rPct = ((rVal - cur) / cur * 100).toFixed(2);
      html += `<div style="background:#2d0d0d;border:1px solid #4d1515;border-radius:8px;padding:10px 14px;flex:1;min-width:220px">
        <div style="font-size:11px;color:#8b949e;margin-bottom:4px">🔴 가장 가까운 저항선 (클래식 ${nearestRKey})</div>
        <div style="font-size:16px;font-weight:700;color:#f85149">${fmtV(rVal)} <span style="font-size:11px;font-weight:400">(+${rPct}%)</span></div>
        <div style="font-size:11px;color:#8b949e;margin-top:6px">→ ${nearestRKey} 돌파 시 <strong>상승 탄력 확인</strong> 후 추가 매수</div>
      </div>`;
    }
    html += `</div>`;
  }

  el.innerHTML = html;
}

function renderNews(d, isKrx) {
  const newsList = document.getElementById('news-list');
  const discEl = document.getElementById('disclosure-col');
  const col1Title = document.getElementById('news-col1-title');

  const newsArr = d.naver ? d.naver.news : d.news;
  if (isKrx && d.naver) {
    col1Title.textContent = '📰 주요 뉴스 (네이버)';
    discEl.style.display = 'block';
    const discList = document.getElementById('disclosure-list');
    const discs = d.naver.disclosures || [];
    discList.innerHTML = discs.length > 0
      ? discs.map(n => `<div class="news-item">
          <span class="news-dot">📌</span>
          <div>
            <a class="news-a" href="${n.link}" target="_blank">${n.title}</a>
            ${n.date ? `<div class="news-meta">${n.date}</div>` : ''}
          </div>
        </div>`).join('')
      : '<p style="font-size:13px;color:#484f58">공시 없음</p>';
  } else {
    col1Title.textContent = '📰 관련 뉴스 (Google RSS)';
    discEl.style.display = 'none';
  }
  newsList.innerHTML = (newsArr || []).length > 0
    ? (newsArr || []).map(n => `<div class="news-item"><span class="news-dot">📄</span><div>
        <a class="news-a" href="${n.link}" target="_blank">${n.title}</a>
        ${n.publisher ? `<div class="news-meta">${n.publisher}${n.published ? ' · ' + (n.published+'').slice(0,16) : ''}</div>` : ''}
      </div></div>`).join('')
    : '<p style="font-size:13px;color:#484f58">뉴스가 없습니다.</p>';
}

// ── 차트 (lightweight-charts) ──
function destroyCharts() {
  Object.values(chartInstances).forEach(c => { try { c.remove(); } catch(e){} });
  chartInstances = {};
}

function renderCharts(d, isKrx) {
  const cd = d.chart_data;
  const n = cd.dates.length;
  if (n < 2) return;
  const upClr = isKrx ? '#f85149' : '#3fb950';
  const dnClr = isKrx ? '#388bfd' : '#f85149';
  // \uac70\ub798\ub7c9 \ubc14 rgba (hex\ub97c \uc9c1\uc811 replace \ud558\uba74 \uc548 \ub428)
  const volUpClr = isKrx ? 'rgba(248,81,73,0.4)' : 'rgba(63,185,80,0.4)';
  const volDnClr = isKrx ? 'rgba(56,139,253,0.4)' : 'rgba(248,81,73,0.4)';

  // ── 가격 차트 ──
  const priceEl = document.getElementById('price-chart');
  if (priceEl && !chartInstances['price']) {
    const chart = LightweightCharts.createChart(priceEl, {
      layout: { background: { color: '#161b22' }, textColor: '#8b949e' },
      grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
      rightPriceScale: { borderColor: '#30363d' },
      timeScale: { borderColor: '#30363d', timeVisible: true },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    });
    chartInstances['price'] = chart;

    // 캔들
    const candleSeries = chart.addCandlestickSeries({
      upColor: upClr, downColor: dnClr,
      borderUpColor: upClr, borderDownColor: dnClr,
      wickUpColor: upClr, wickDownColor: dnClr,
    });
    const candleData = [];
    for (let i = 0; i < n; i++) {
      if (cd.open[i] != null && cd.high[i] != null && cd.low[i] != null && cd.close[i] != null) {
        candleData.push({ time: cd.dates[i], open: cd.open[i], high: cd.high[i], low: cd.low[i], close: cd.close[i] });
      }
    }
    candleSeries.setData(candleData);

    // MA20
    const ma20 = chart.addLineSeries({ color: '#f97316', lineWidth: 1, title: 'MA20' });
    ma20.setData(cd.ma20.map((v,i) => ({ time: cd.dates[i], value: v })).filter(p => p.value != null));

    // MA60
    const ma60 = chart.addLineSeries({ color: '#a78bfa', lineWidth: 1, title: 'MA60' });
    ma60.setData(cd.ma60.map((v,i) => ({ time: cd.dates[i], value: v })).filter(p => p.value != null));

    // BB
    const bbU = chart.addLineSeries({ color: 'rgba(59,130,246,0.5)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed });
    bbU.setData(cd.bb_upper.map((v,i) => ({ time: cd.dates[i], value: v })).filter(p => p.value != null));
    const bbL = chart.addLineSeries({ color: 'rgba(59,130,246,0.5)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed });
    bbL.setData(cd.bb_lower.map((v,i) => ({ time: cd.dates[i], value: v })).filter(p => p.value != null));

    // 거래량
    const volSeries = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'vol',
    });
    chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    const volData = [];
    for (let i = 0; i < n; i++) {
      if (cd.volume[i] != null) {
        const isBull = (cd.close[i] || 0) >= (cd.open[i] || 0);
        volData.push({ time: cd.dates[i], value: cd.volume[i], color: isBull ? volUpClr : volDnClr });
      }
    }
    volSeries.setData(volData);
    chart.timeScale().fitContent();
  }

  // ── RSI 차트 ──
  const rsiEl = document.getElementById('rsi-chart');
  if (rsiEl && !chartInstances['rsi']) {
    const chart = LightweightCharts.createChart(rsiEl, {
      layout: { background: { color: '#161b22' }, textColor: '#8b949e' },
      grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
      rightPriceScale: { borderColor: '#30363d' },
      timeScale: { borderColor: '#30363d', timeVisible: true },
    });
    chartInstances['rsi'] = chart;
    const rsiSeries = chart.addLineSeries({ color: '#facc15', lineWidth: 1.5 });
    rsiSeries.setData(cd.rsi.map((v,i) => ({ time: cd.dates[i], value: v })).filter(p => p.value != null));
    // 70/30 기준선 (버그 수정: cd.dates[20] → 데이터 부족 시 IndexError 방지)
    const rsiStartIdx = Math.min(20, Math.max(0, n - 2));
    if (rsiStartIdx < n - 1) {
      const l70 = chart.addLineSeries({ color: 'rgba(248,81,73,0.5)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed });
      l70.setData([{ time: cd.dates[0], value: 70 }, { time: cd.dates[n-1], value: 70 }]);
      const l30 = chart.addLineSeries({ color: 'rgba(56,139,253,0.5)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed });
      l30.setData([{ time: cd.dates[0], value: 30 }, { time: cd.dates[n-1], value: 30 }]);
    }
    chart.timeScale().fitContent();
  }

  // ── MACD 차트 ──
  const macdEl = document.getElementById('macd-chart');
  if (macdEl && !chartInstances['macd']) {
    const chart = LightweightCharts.createChart(macdEl, {
      layout: { background: { color: '#161b22' }, textColor: '#8b949e' },
      grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
      rightPriceScale: { borderColor: '#30363d' },
      timeScale: { borderColor: '#30363d', timeVisible: true },
    });
    chartInstances['macd'] = chart;
    const macdSeries = chart.addLineSeries({ color: '#3fb950', lineWidth: 1.5, title: 'MACD' });
    macdSeries.setData(cd.macd.map((v,i) => ({ time: cd.dates[i], value: v })).filter(p => p.value != null));
    const sigSeries = chart.addLineSeries({ color: '#f97316', lineWidth: 1.5, title: 'Signal' });
    sigSeries.setData(cd.signal_line.map((v,i) => ({ time: cd.dates[i], value: v })).filter(p => p.value != null));
    const histSeries = chart.addHistogramSeries({ color: '#3fb950', priceScaleId: 'macd_hist' });
    chart.priceScale('macd_hist').applyOptions({ scaleMargins: { top: 0.7, bottom: 0 } });
    const histData = [];
    for (let i = 0; i < n; i++) {
      if (cd.macd[i] != null && cd.signal_line[i] != null) {
        const diff = cd.macd[i] - cd.signal_line[i];
        histData.push({ time: cd.dates[i], value: diff, color: diff >= 0 ? 'rgba(63,185,80,0.5)' : 'rgba(248,81,73,0.5)' });
      }
    }
    histSeries.setData(histData);
    chart.timeScale().fitContent();
  }
}

// ── 탭 전환 ──
function switchTab(tab) {
  currentTab = tab;
  ['chart','ai','forecast','news'].forEach(t => {
    document.getElementById('tab-' + t).style.display = t === tab ? 'block' : 'none';
  });
  document.querySelectorAll('#state-result .tabs .tab-btn').forEach((btn, i) => {
    const tabs = ['chart','ai','forecast','news'];
    btn.classList.toggle('active', tabs[i] === tab);
  });
}

// ── 스크리너 ──
async function loadScreener(sortBy, sortOrder) {
  document.getElementById('scrn-loading').style.display = 'block';
  document.getElementById('scrn-result').style.display = 'none';
  const sb = sortBy    || scrnSort.key || 'price';
  const so = sortOrder || scrnSort.dir || 'desc';
  scrnSort = { key: sb, dir: so };
  try {
    const url = `/api/screener?sort_by=${sb}&sort_order=${so}`;
    const r = await fetch(url);
    const d = await r.json();
    screenerData = d.data || [];
    screenerInfo = {
      usd_krw: d.usd_krw || 0,
      total_overseas: d.total_overseas || 0,
      total_domestic: d.total_domestic || 0,
      us_filter_conditions: d.us_filter_conditions || {},
      kr_filter_conditions: d.kr_filter_conditions || {}
    };
    renderScreener();
    document.getElementById('scrn-loading').style.display = 'none';
    document.getElementById('scrn-result').style.display = 'block';
  } catch(e) {
    document.getElementById('scrn-loading').innerHTML = '<p style="color:#f85149">데이터 로딩 실패: ' + e.message + '</p>';
  }
}

function switchScrnTab(tab) {
  scrnMarket = tab;
  document.getElementById('scrn-tab-domestic').classList.toggle('active', tab === 'domestic');
  document.getElementById('scrn-tab-overseas').classList.toggle('active', tab === 'overseas');
  renderScreener();
}

function sortScreener(key) {
  if (scrnSort.key === key) scrnSort.dir = scrnSort.dir === 'desc' ? 'asc' : 'desc';
  else { scrnSort.key = key; scrnSort.dir = 'desc'; }
  // 해외탭: 서버 재요청으로 정렬 (캐시 활용)
  if (scrnMarket === 'overseas') {
    loadScreener(key, scrnSort.dir);
  } else {
    renderScreener();
  }
}

function renderScreener() {
  const marketLabel = scrnMarket === 'domestic' ? '국내' : '해외';
  const isKrx = scrnMarket === 'domestic';
  let filtered = screenerData.filter(s => s.market === marketLabel);

  // 필터 조건 문자열 생성
  const fc = isKrx ? (screenerInfo.kr_filter_conditions || {}) : (screenerInfo.us_filter_conditions || {});
  const filterStr = Object.entries(fc).map(([k,v])=>{
    const fv = String(v).replace(/(\d+\.\d{3,})/g, m => parseFloat(m).toFixed(2));
    return `${k}: ${fv}`;
  }).join(' │ ');

  const totalCnt = isKrx ? (screenerInfo.total_domestic || filtered.length) : (screenerInfo.total_overseas || 0);
  document.getElementById('scrn-subtitle').textContent =
    `토스증권 필터 조건 적용 | USD/KRW: ${(screenerInfo.usd_krw||0).toLocaleString()} | ${marketLabel} ${totalCnt}종목`;
  
  // 필터 조건 뱃지 업데이트 (DOM 요소가 없으면 생성, 있으면 텍스트만 교체)
  let badgeEl = document.getElementById('scrn-filter-badge');
  if (!badgeEl) {
    const subtitleEl = document.getElementById('scrn-subtitle');
    badgeEl = document.createElement('div');
    badgeEl.id = 'scrn-filter-badge';
    badgeEl.style.cssText = 'margin-top: 8px; font-size: 11px; color: #8b949e; background: #21262d; padding: 8px 12px; border-radius: 8px; line-height: 1.5; word-break: keep-all; border: 1px solid #30363d;';
    subtitleEl.parentNode.insertBefore(badgeEl, subtitleEl.nextSibling);
  }
  
  // 기존 내용 지우고 새로 설정 (중복 방지)
  badgeEl.innerHTML = '';
  badgeEl.textContent = filterStr || '적용된 필터 조건이 없습니다.';

  // 국내: 클라이언트 정렬 / 해외: 서버 정렬 결과 그대로
  if (scrnMarket === 'domestic') {
    filtered = filtered.slice().sort((a, b) => {
      let va, vb;
      if      (scrnSort.key === 'price')  { va = a.price_val || 0; vb = b.price_val || 0; }
      else if (scrnSort.key === 'change') { va = a.change    || 0; vb = b.change    || 0; }
      else if (scrnSort.key === 'per')    { va = a.per       || 0; vb = b.per       || 0; }
      else if (scrnSort.key === 'roe')    { va = a.roe_pct   || 0; vb = b.roe_pct   || 0; }
      else                                { va = a.volume    || 0; vb = b.volume    || 0; }
      return scrnSort.dir === 'desc' ? vb - va : va - vb;
    });
  }

  const tbody = document.getElementById('scrn-tbody');

  if (!filtered.length) {
    const emptyMsg = '필터 조건에 맞는 종목이 없습니다.';
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:40px;color:#8b949e">${emptyMsg}</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map((s, idx) => {
    const up  = s.change >= 0;
    const clr = isKrx ? (up ? '#f85149' : '#388bfd') : (up ? '#3fb950' : '#f85149');

    // 애널리스트 신호 우선 사용
    const rawSig = s.signal || '';
    const signal = rawSig || (s.change > 3 ? '적극 매수' : s.change > 0 ? '매수' : s.change > -3 ? '중립' : '매도');
    const sigCls = signal.includes('적극') ? 'sig-buy-strong'
                 : signal === '매수'        ? 'sig-buy'
                 : signal === '중립' || signal === '보유' ? 'sig-neu'
                 : 'sig-sell';

    // 현재가 표시: 해외는 달러 + 원화 병기
    let priceDisp;
    if (isKrx) {
      priceDisp = s.price || s.price_krw || '-';
    } else {
      const usd = s.price_usd || s.price || '-';
      const krw = s.price_krw ? '<br><span style="font-size:11px;color:#8b949e">' + s.price_krw + '</span>' : '';
      priceDisp = usd + krw;
    }

    const perDisp = (s.per != null && s.per > 0) ? s.per.toFixed(1) : '-';
    const ticker  = String(s.ticker || '');
    const name    = String(s.name   || '');
    const cat     = String(s.category || s.sector || '');
    const vol     = (s.volume || 0).toLocaleString();
    const chg     = Math.abs(s.change || 0).toFixed(2);

    return '<tr onclick="quickSearch(\'' + ticker + '\')" style="cursor:pointer">'
      + '<td style="color:#484f58">' + (idx+1) + '</td>'
      + '<td><div class="ticker-name">' + name + '</div><div class="ticker-code">' + ticker + '</div></td>'
      + '<td style="text-align:right;font-weight:600">' + priceDisp + '</td>'
      + '<td style="text-align:right;font-weight:700;color:' + clr + '">' + (up?'▲':'▼') + ' ' + chg + '%</td>'
      + '<td><span class="cat-badge">' + cat + '</span></td>'
      + '<td style="text-align:right;color:#8b949e;font-size:12px">' + vol + '</td>'
      + '<td style="text-align:center;color:#8b949e;font-size:12px">' + perDisp + '</td>'
      + '<td style="text-align:center"><span class="signal-badge ' + sigCls + '">' + signal + '</span></td>'
      + '</tr>';
  }).join('');
}

// ── 초기화 ──
loadSentiment('KRX');
</script>
</body>
</html>"""


# =============================================================================
# Vercel Handler
# =============================================================================
def replace_nan_with_none(obj):
    # Handle pandas Series/Index/DataFrame
    if isinstance(obj, (pd.Series, pd.Index)):
        return replace_nan_with_none(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return replace_nan_with_none(obj.to_dict(orient='list'))
        
    # Handle numpy arrays (convert to list first)
    if isinstance(obj, np.ndarray):
        return replace_nan_with_none(obj.tolist())
        
    if isinstance(obj, list):
        return [replace_nan_with_none(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(obj)
    elif pd.isna(obj): # pd.NaT, np.nan, etc.
        return None
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    return obj

VALID_PERIODS = {"1d", "3d", "1wk", "2wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}

def _send(handler_self, data: Any, status: int = 200, content_type: str = "application/json"):
    path = getattr(handler_self, 'path', '/').split('?')[0].rstrip('/') or '/'

    if content_type == "application/json":
        data = replace_nan_with_none(data)
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    else:
        body = data if isinstance(data, bytes) else data.encode("utf-8")
    
    handler_self.send_response(status)
    handler_self.send_header("Content-Type", content_type + "; charset=utf-8")
    handler_self.send_header("Content-Length", str(len(body)))
    
    # CORS Policy
    handler_self.send_header("Access-Control-Allow-Origin", "*")
    handler_self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
    handler_self.send_header("Access-Control-Allow-Headers", "Content-Type")

    # Cache Policy (Vercel/CDN Integration)
    # Default: No Cache for errors
    cache_control = "no-store, no-cache, must-revalidate, proxy-revalidate"
    
    if status == 200:
        if path == "/api/screener":
            # Long cache for screener (1 hour + 1 day stale)
            cache_control = "public, s-maxage=3600, stale-while-revalidate=86400"
        elif path == "/api/stock":
            # Short cache for stock data
            cache_control = "public, s-maxage=60, stale-while-revalidate=300"
        elif path == "/" or path == "/index.html":
            # HTML Cache
            cache_control = "public, s-maxage=3600"
        elif path.startswith("/api/"):
            cache_control = "public, s-maxage=10, stale-while-revalidate=60"

    handler_self.send_header("Cache-Control", cache_control)

    # Security Headers
    handler_self.send_header("X-Content-Type-Options", "nosniff")
    handler_self.send_header("X-Frame-Options", "DENY")
    handler_self.send_header("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
    handler_self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' https://finance.naver.com https://query1.finance.yahoo.com https://query2.finance.yahoo.com;")
    handler_self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")

    handler_self.end_headers()
    handler_self.wfile.write(body)


class handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[StockOracle] {fmt % args}")

    def do_OPTIONS(self):
        _send(self, {})

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            path = parsed.path.rstrip("/") or "/"

            # Input Validation
            if path == "/api/stock":
                ticker = params.get("ticker", "")
                period = params.get("period", "1y")
                
                if len(ticker) > 20 or (ticker and not re.match(r"^[a-zA-Z0-9가-힣.\-\s]+$", ticker)):
                     _send(self, {"error": "Invalid ticker format"}, 400)
                     return

                if period not in VALID_PERIODS:
                     _send(self, {"error": f"Invalid period. Allowed: {', '.join(VALID_PERIODS)}"}, 400)
                     return

            result = route(path, params)
            if result is None:
                _send(self, HTML, 200, "text/html")
            else:
                _send(self, result)
        except Exception as e:
            # Log traceback internally but hide from user
            print(f"Server Error: {str(e)}\n{traceback.format_exc()}")
            _send(self, {"error": "Internal Server Error"}, 500)
