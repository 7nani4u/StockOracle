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
import traceback
import warnings
import functools
from typing import Optional, Dict, Any, List, Tuple
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote

# ── /tmp 강제 사용 (Vercel은 /tmp 외 쓰기 금지) ───────────────────────────────
os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache")

try:
    import platformdirs
    def _tmp(*a, **k):
        d = "/tmp/yf_cache"
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

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
            return r
        return wrapper
    return deco

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
    "알리바바": "BABA", "쿠팡": "CPNG", "팔란티어": "PLTR",
    "코인베이스": "COIN", "QQQ": "QQQ", "SPY": "SPY",
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
    for w in [5, 20, 60, 120]:
        df[f"MA{w}"] = c.rolling(w).mean()
    df["EMA12"] = c.ewm(span=12, adjust=False).mean()
    df["EMA26"] = c.ewm(span=26, adjust=False).mean()
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
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
    return df

@ttl_cache(600)
def fetch_stock_data(ticker: str, market: str, period: str = "1y"):
    sym = ticker.strip().upper()
    if market == "KRX" and sym.isdigit():
        sym = f"{sym}.KS"
    try:
        obj = yf.Ticker(sym)
        df = obj.history(period=period, interval="1d", auto_adjust=True)
        if df.empty and market == "KRX" and sym.endswith(".KS"):
            sym = sym.replace(".KS", ".KQ")
            df = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
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
        if hasattr(df2["Date"].dtype, "tz") and df2["Date"].dtype.tz:
            df2["Date"] = df2["Date"].dt.tz_localize(None)
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
        hdrs = {"User-Agent": "Mozilla/5.0"}
        soup = BeautifulSoup(requests.get(url, headers=hdrs, timeout=5).text, "html.parser")
        el = soup.select_one(".no_today .blind")
        if el: r["price"] = el.text.replace(",","")
        for k, s in [("market_cap","#_market_sum"),("per","#_per"),("pbr","#_pbr")]:
            e = soup.select_one(s)
            r[k] = e.text.strip() if e else "-"
        for item in soup.select(".news_section ul li")[:5]:
            a = item.select_one("span > a")
            if a:
                r["news"].append({"title": a.text.strip(),
                                   "link": "https://finance.naver.com" + a["href"]})
    except Exception:
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

@ttl_cache(300)
def fetch_screener():
    try:
        usd_krw = float(yf.Ticker("USDKRW=X").history(period="1d")["Close"].iloc[-1])
    except Exception:
        usd_krw = 1400.0
    stocks = {
        "KRX": [
            {"ticker":"005930.KS","name":"삼성전자","cat":"반도체"},
            {"ticker":"000660.KS","name":"SK하이닉스","cat":"반도체"},
            {"ticker":"373220.KS","name":"LG에너지솔루션","cat":"2차전지"},
            {"ticker":"005380.KS","name":"현대차","cat":"자동차"},
            {"ticker":"000270.KS","name":"기아","cat":"자동차"},
            {"ticker":"035420.KS","name":"NAVER","cat":"인터넷"},
            {"ticker":"035720.KS","name":"카카오","cat":"인터넷"},
            {"ticker":"068270.KS","name":"셀트리온","cat":"바이오"},
            {"ticker":"005490.KS","name":"POSCO홀딩스","cat":"철강"},
            {"ticker":"055550.KS","name":"신한지주","cat":"금융"},
        ],
        "US": [
            {"ticker":"AAPL","name":"애플","cat":"기술"},
            {"ticker":"MSFT","name":"마이크로소프트","cat":"소프트웨어"},
            {"ticker":"NVDA","name":"엔비디아","cat":"반도체"},
            {"ticker":"AMZN","name":"아마존","cat":"유통/클라우드"},
            {"ticker":"GOOGL","name":"구글","cat":"인터넷"},
            {"ticker":"META","name":"메타","cat":"인터넷"},
            {"ticker":"TSLA","name":"테슬라","cat":"자동차"},
            {"ticker":"TSM","name":"TSMC","cat":"반도체"},
            {"ticker":"JPM","name":"JP모건","cat":"금융"},
            {"ticker":"V","name":"비자","cat":"금융"},
        ],
    }
    all_tickers = [s["ticker"] for sl in stocks.values() for s in sl]
    results = []
    try:
        batch = yf.download(
            all_tickers, period="5d",
            group_by="ticker", threads=False, progress=False, auto_adjust=True
        )
        is_multi = isinstance(batch.columns, pd.MultiIndex)
        for mtype, slist in stocks.items():
            for s in slist:
                t = s["ticker"]
                try:
                    if is_multi:
                        # yfinance v0.2.x: (field, ticker) 또는 (ticker, field) 둘 다 처리
                        if t in batch.columns.get_level_values(0):
                            df_t = batch[t]
                        elif t in batch.columns.get_level_values(1):
                            df_t = batch.xs(t, axis=1, level=1)
                        else:
                            continue
                    else:
                        df_t = batch
                    # MultiIndex 평탄화
                    if isinstance(df_t.columns, pd.MultiIndex):
                        df_t.columns = df_t.columns.droplevel(0)
                    df_t = df_t.dropna(subset=["Close"])
                    if len(df_t) < 2: continue
                    last = float(df_t["Close"].iloc[-1])
                    prev = float(df_t["Close"].iloc[-2])
                    pct = (last - prev) / prev * 100 if prev != 0 else 0.0
                    price_str = f"{last * usd_krw:,.0f}원" if mtype == "US" else f"{last:,.0f}원"
                    vol = df_t["Volume"].iloc[-1]
                    results.append({
                        "market": "국내" if mtype == "KRX" else "해외",
                        "name": s["name"], "ticker": t,
                        "price": price_str, "change": round(pct, 2),
                        "category": s["cat"],
                        "volume": int(vol) if pd.notna(vol) else 0,
                    })
                except Exception:
                    continue
    except Exception:
        pass
    results.sort(key=lambda x: x["change"], reverse=True)
    return {"data": results, "usd_krw": round(usd_krw, 2)}

# =============================================================================
# 분석 엔진
# =============================================================================
def analyze_score(dd: Dict):
    closes = dd.get("Close", [])
    if len(closes) < 20:
        return 50, [], []
    def v(k):
        a = dd.get(k, [])
        val = a[-1] if a else None
        return float(val) if val is not None else 0.0
    close, ma20, ma60 = v("Close"), v("MA20"), v("MA60")
    rsi, macd, sig = v("RSI"), v("MACD"), v("Signal_Line")
    bb_u, bb_l = v("BB_Upper"), v("BB_Lower")
    vols = dd.get("Volume", [])
    cur_vol = float(vols[-1]) if vols else 0
    avg_vol = float(np.mean([x for x in vols[-20:] if x])) if vols else 1
    opn = v("Open")
    score, steps = 50, []

    # 추세
    ts, msg = 0, ""
    if close > ma20:
        ts += 10
        if close > ma60:
            ts += 10
            if ma20 > ma60: ts += 10; msg = "단기/장기 이동평균 정배열 → 강한 상승 추세"
            else: msg = "장기 이평선 위 → 상승 기조"
        else: msg = "20일 이평선 위 → 단기 상승 시도"
    else:
        ts -= 10
        if close < ma60:
            ts -= 10
            if ma20 < ma60: ts -= 10; msg = "역배열 → 하락 압력 강함"
            else: msg = "장기 이평선 아래 → 하락 추세 우려"
        else: msg = "20일 이평선 하회 → 조정 중"
    score += ts
    steps.append({"step": "1. 추세 분석 (MA)", "result": msg, "score": ts})

    # 모멘텀
    ms, msgs = 0, []
    if rsi > 70: ms -= 5; msgs.append(f"RSI {rsi:.1f} 과매수")
    elif rsi < 30: ms += 10; msgs.append(f"RSI {rsi:.1f} 과매도 → 반등 기대")
    else: msgs.append(f"RSI {rsi:.1f} 중립")
    if macd > sig: ms += 10; msgs.append("MACD 골든크로스 → 상승 신호")
    else: ms -= 10; msgs.append("MACD 데드크로스 → 하락 신호")
    score += ms
    steps.append({"step": "2. 모멘텀 (RSI/MACD)", "result": " | ".join(msgs), "score": ms})

    # 거래량/BB
    vs, vmsgs = 0, []
    if close > bb_u * 0.98: vs += 5; vmsgs.append("볼린저 상단 터치")
    elif close < bb_l * 1.02: vs -= 5; vmsgs.append("볼린저 하단 터치")
    if avg_vol > 0 and cur_vol > avg_vol * 1.5:
        if close > opn: vs += 10; vmsgs.append("거래량 급증 + 상승 → 신뢰도 높음")
        else: vs -= 10; vmsgs.append("거래량 급증 + 하락 → 매도 압력")
    else: vmsgs.append("거래량 평이")
    score += vs
    steps.append({"step": "3. 거래량/변동성", "result": " | ".join(vmsgs), "score": vs})

    # 캔들 패턴
    patterns = detect_patterns(dd)
    ps = 0; pmsgs = []
    if patterns:
        bull = sum(1 for p in patterns if p["direction"] == "상승")
        bear = sum(1 for p in patterns if p["direction"] == "하락")
        if bull > bear: ps += 10; pmsgs.append(f"상승 패턴 {bull}개")
        elif bear > bull: ps -= 10; pmsgs.append(f"하락 패턴 {bear}개")
        else: pmsgs.append(f"패턴 혼재 {len(patterns)}개")
    else: pmsgs.append("특이 패턴 없음")
    score += ps
    steps.append({"step": "4. 캔들 패턴", "result": " | ".join(pmsgs), "score": ps})
    return max(0, min(100, score)), steps, patterns

def detect_patterns(dd: Dict) -> List[Dict]:
    try:
        o = [float(x) for x in dd.get("Open", []) if x is not None]
        h = [float(x) for x in dd.get("High", []) if x is not None]
        l = [float(x) for x in dd.get("Low", []) if x is not None]
        c = [float(x) for x in dd.get("Close", []) if x is not None]
        if len(c) < 3: return []
        o1,h1,l1,c1 = o[-1],h[-1],l[-1],c[-1]
        o2,h2,l2,c2 = o[-2],h[-2],l[-2],c[-2]
        o3,h3,l3,c3 = o[-3],h[-3],l[-3],c[-3]
        body1 = abs(c1-o1); rng1 = h1-l1 or 0.001
        body2 = abs(c2-o2); rng2 = h2-l2 or 0.001
        body3 = abs(c3-o3); rng3 = h3-l3 or 0.001
        up_sh1 = h1-max(c1,o1); lo_sh1 = min(c1,o1)-l1
        bull1 = c1>=o1; bull2 = c2>=o2; bull3 = c3>=o3
        res = []
        if body1/rng1 < 0.1:
            res.append({"name":"✖️ Doji","desc":"도지","direction":"중립","conf":100})
        if lo_sh1 >= body1*2 and up_sh1 <= body1*0.5 and body1>0 and c2>c1:
            res.append({"name":"🔨 Hammer","desc":"해머 (반등 신호)","direction":"상승","conf":100})
        if up_sh1 >= body1*2 and lo_sh1 <= body1*0.5 and body1>0 and c2<c1:
            res.append({"name":"⭐ Shooting Star","desc":"유성형 (하락 신호)","direction":"하락","conf":100})
        if bull1 and not bull2 and o1<=c2 and c1>=o2 and body1>body2:
            res.append({"name":"🫂 Bullish Engulfing","desc":"상승 포용형","direction":"상승","conf":100})
        if not bull1 and bull2 and o1>=c2 and c1<=o2 and body1>body2:
            res.append({"name":"🫂 Bearish Engulfing","desc":"하락 포용형","direction":"하락","conf":100})
        if bull1 and not bull2 and o1>c2 and c1<o2 and body1<body2*0.5:
            res.append({"name":"🤰 Bullish Harami","desc":"상승 하라미","direction":"상승","conf":100})
        if body1/rng1 > 0.9 and body1>0:
            res.append({"name":"📏 Marubozu","desc":f"마루보즈({'상승' if bull1 else '하락'})","direction":"상승" if bull1 else "하락","conf":100})
        if bull3 and body2/rng2 < 0.3 and bull1 and c1 > (o3+c3)/2 and c3 > o3:
            res.append({"name":"🌆 Evening Star","desc":"이브닝스타 (하락 반전)","direction":"하락","conf":100})
        if not bull3 and body2/rng2 < 0.3 and bull1 and c1 < (o3+c3)/2 and c3 < o3:
            res.append({"name":"🌅 Morning Star","desc":"모닝스타 (상승 반전)","direction":"상승","conf":100})
        return res
    except Exception:
        return []

def calc_risk(price: float, atr: float) -> Dict:
    if not atr or np.isnan(atr): atr = price * 0.02
    return {
        "conservative": {"label":"보수적","target":round(price+atr*1.5,2),"stop":round(price-atr,2),"ratio":"1:1.5","desc":"리스크 최소화","icon":"🛡️"},
        "balanced":      {"label":"중립적","target":round(price+atr*2.5,2),"stop":round(price-atr*1.5,2),"ratio":"1:1.67","desc":"스윙 트레이딩","icon":"⚖️"},
        "aggressive":    {"label":"공격적","target":round(price+atr*4,2),"stop":round(price-atr*2,2),"ratio":"1:2.0","desc":"추세 추종","icon":"🚀"},
    }

def holt_winters_forecast(dd: Dict, days: int = 30):
    if not STATSMODELS_AVAILABLE:
        return linear_forecast(dd, days)
    try:
        closes = [float(c) for c in dd.get("Close", []) if c is not None]
        dates = dd.get("Date", [])
        if len(closes) < 60: return None
        y = pd.Series(closes).tail(504)
        m = ExponentialSmoothing(y, trend="add", seasonal="add",
                                  seasonal_periods=5, initialization_method="estimated").fit(optimized=True)
        fc = m.forecast(days)
        std = float(m.resid.std()) if len(m.resid) > 0 else 0
        last_d = datetime.datetime.strptime(dates[-1], "%Y-%m-%d") if dates else datetime.datetime.now()
        future_dates = []
        d = last_d
        for _ in range(days):
            d += datetime.timedelta(days=1)
            while d.weekday() >= 5: d += datetime.timedelta(days=1)
            future_dates.append(d.strftime("%Y-%m-%d"))
        return {
            "dates": future_dates,
            "yhat": [round(float(f), 2) for f in fc],
            "yhat_upper": [round(float(f)+1.96*std, 2) for f in fc],
            "yhat_lower": [round(float(f)-1.96*std, 2) for f in fc],
        }
    except Exception:
        return linear_forecast(dd, days)

def linear_forecast(dd: Dict, days: int):
    if not SKLEARN_AVAILABLE: return None
    try:
        closes = [float(c) for c in dd.get("Close", []) if c is not None]
        dates = dd.get("Date", [])
        if len(closes) < 20: return None
        X = np.arange(len(closes)).reshape(-1,1)
        reg = LinearRegression().fit(X, closes)
        preds = reg.predict(np.arange(len(closes), len(closes)+days).reshape(-1,1)).tolist()
        last_d = datetime.datetime.strptime(dates[-1], "%Y-%m-%d") if dates else datetime.datetime.now()
        fds = []
        d = last_d
        for _ in range(days):
            d += datetime.timedelta(days=1)
            while d.weekday() >= 5: d += datetime.timedelta(days=1)
            fds.append(d.strftime("%Y-%m-%d"))
        return {"dates": fds, "yhat": [round(p,2) for p in preds],
                "yhat_upper": [round(p*1.05,2) for p in preds],
                "yhat_lower": [round(p*0.95,2) for p in preds]}
    except Exception:
        return None

def xgb_forecast(dd: Dict, days: int = 30):
    if not XGBOOST_AVAILABLE: return None
    try:
        needed = ["Open","High","Low","Close","Volume","MA5","MA20","RSI","MACD"]
        df = pd.DataFrame({k: dd[k] for k in needed if k in dd}).dropna().tail(252).reset_index(drop=True)
        if len(df) < 30: return None
        df["Target"] = df["Close"].shift(-1)
        df["Returns"] = df["Close"].pct_change()
        df["Range"] = df["High"] - df["Low"]
        for lag in [1,2,3]:
            df[f"Lag{lag}"] = df["Close"].shift(lag)
        df = df.dropna()
        feats = [c for c in ["Close","Open","High","Low","Volume","MA5","MA20","RSI","MACD","Returns","Range","Lag1","Lag2","Lag3"] if c in df.columns]
        X, y = df[feats], df["Target"]
        split = int(len(X)*0.9)
        model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, tree_method="hist", verbosity=0)
        model.fit(X.iloc[:split], y.iloc[:split])
        last = X.iloc[[-1]].copy()
        preds = []
        cur = float(df["Close"].iloc[-1])
        for _ in range(days):
            p = float(model.predict(last)[0])
            preds.append(round(p, 2))
            if "Lag3" in last: last["Lag3"] = last.get("Lag2", last["Lag3"])
            if "Lag2" in last: last["Lag2"] = last.get("Lag1", last["Lag2"])
            if "Lag1" in last: last["Lag1"] = cur
            last["Close"] = p
            if "Returns" in last: last["Returns"] = (p - cur) / cur if cur else 0
            cur = p
        return preds
    except Exception:
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
        score, steps, patterns = analyze_score(dd)
        forecast = holt_winters_forecast(dd)
        xgbp = xgb_forecast(dd)
        atrs = dd.get("ATR", [])
        atr_val = float(atrs[-1]) if atrs and atrs[-1] else last * 0.02
        risk = calc_risk(last, atr_val)
        naver = fetch_naver(sym) if market == "KRX" else None
        return {
            "symbol": sym, "company": company or sym, "market": market,
            "last_close": round(last, 2), "prev_close": round(prev, 2),
            "pct_change": round(pct, 2),
            "rsi": round(float(dd.get("RSI", [50])[-1] or 50), 1),
            "volume": int(dd.get("Volume", [0])[-1] or 0),
            "atr": round(atr_val, 2),
            "score": score, "analysis_steps": steps,
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
            "forecast": forecast, "xgb_forecast": xgbp,
            "risk_scenarios": risk, "news": news or [], "naver": naver,
        }

    if path == "/api/screener":
        return fetch_screener()

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
.fund-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
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
.step-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.step-title{font-size:13px;font-weight:600}
.step-score{font-size:12px;font-weight:700;padding:2px 8px;border-radius:12px}
.step-score.pos{background:#0d2d1a;color:#3fb950}
.step-score.neg{background:#2d0d0d;color:#f85149}
.step-score.neu{background:#21262d;color:#8b949e}
.step-result{font-size:13px;color:#8b949e;line-height:1.5}

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

/* 반응형 */
@media(max-width:900px){
  .metrics-grid{grid-template-columns:repeat(2,1fr)}
  .risk-grid{grid-template-columns:1fr}
  .fund-grid{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:640px){
  #sidebar{width:220px}
  .metrics-grid{grid-template-columns:1fr 1fr}
}
</style>
</head>
<body>

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
      <option value="6mo">6개월</option>
      <option value="1y" selected>1년</option>
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
      <p>왼쪽 패널에서 종목명 또는 코드를 입력하고<br><strong style="color:#388bfd">분석 시작</strong> 버튼을 누르세요.</p>
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
          <div class="fund-item"><div class="fund-label">투자의견</div><div class="fund-val" id="f-opinion"></div></div>
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
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
          <div class="card">
            <div class="card-title">RSI (14)</div>
            <div id="rsi-chart" style="height:150px"></div>
          </div>
          <div class="card">
            <div class="card-title">MACD</div>
            <div id="macd-chart" style="height:150px"></div>
          </div>
        </div>
      </div>

      <!-- AI 탭 -->
      <div id="tab-ai" style="display:none">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
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
          <div class="card-title">🔮 가격 예측 (30일 / Holt-Winters + XGBoost)</div>
          <div id="forecast-chart" style="height:300px"></div>
          <div id="forecast-summary" style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px"></div>
        </div>
        <div>
          <div style="font-size:13px;font-weight:600;color:#8b949e;margin-bottom:12px;text-transform:uppercase;letter-spacing:.05em">🛡️ 리스크 관리 (ATR 기반)</div>
          <div class="risk-grid" id="risk-grid"></div>
        </div>
      </div>

      <!-- 뉴스 탭 -->
      <div id="tab-news" style="display:none">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
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
        <p style="font-size:12px;color:#8b949e" id="scrn-subtitle">국내/해외 주요 종목 실시간 시세</p>
      </div>
      <button onclick="loadScreener()" style="background:#21262d;border:1px solid #30363d;border-radius:8px;padding:8px 14px;color:#8b949e;font-size:13px;cursor:pointer">🔄 새로고침</button>
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
      <div class="card" style="padding:0;overflow:hidden">
        <table class="screener-table">
          <thead><tr>
            <th>#</th>
            <th>종목</th>
            <th style="text-align:right">현재가</th>
            <th style="text-align:right" onclick="sortScreener('change')">등락률 ↕</th>
            <th>카테고리</th>
            <th style="text-align:right" onclick="sortScreener('volume')">거래량 ↕</th>
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
let scrnMarket = 'domestic';
let scrnSort = {key:'change', dir:'desc'};
let chartInstances = {};

// ── 페이지 전환 ──
function showPage(page) {
  document.getElementById('page-analysis').style.display = page === 'analysis' ? 'block' : 'none';
  document.getElementById('page-screener').style.display = page === 'screener' ? 'block' : 'none';
  document.getElementById('analysis-controls').style.display = page === 'analysis' ? 'block' : 'none';
  document.getElementById('nav-analysis').classList.toggle('active', page === 'analysis');
  document.getElementById('nav-screener').classList.toggle('active', page === 'screener');
  if (page === 'screener' && screenerData.length === 0) loadScreener();
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
    document.getElementById('f-opinion').textContent = d.naver.opinion || '-';
  } else {
    document.getElementById('r-naver-fund').style.display = 'none';
  }

  // AI 진단
  renderAI(d, isKrx);
  // 예측/리스크
  renderForecast(d, isKrx);
  // 뉴스
  renderNews(d, isKrx);
  // 탭 초기화
  switchTab('chart');
  // 차트는 탭 전환 후 렌더
  setTimeout(() => renderCharts(d, isKrx), 50);
}

function renderAI(d, isKrx) {
  const s = d.score;
  const sClr = s >= 70 ? '#3fb950' : s >= 40 ? '#d29922' : '#f85149';
  const sBarClr = s >= 70 ? '#3fb950' : s >= 40 ? '#d29922' : '#f85149';
  document.getElementById('ai-score').innerHTML = `<span style="color:${sClr}">${s}</span>`;
  const bar = document.getElementById('ai-score-bar');
  bar.style.width = s + '%'; bar.style.background = sBarClr;
  document.getElementById('ai-score-desc').textContent =
    s >= 70 ? '✅ 상승 우위 — 매수 신호 강함'
    : s >= 40 ? '⚖️ 중립 — 추세 확인 필요'
    : '⚠️ 하락 우위 — 리스크 주의';

  const stepsList = document.getElementById('steps-list');
  stepsList.innerHTML = d.analysis_steps.map(st => {
    const sc = st.score;
    const cls = sc > 0 ? 'pos' : sc < 0 ? 'neg' : 'neu';
    const label = sc > 0 ? '+' + sc : sc;
    return `<div class="step-item">
      <div class="step-header">
        <span class="step-title">${st.step}</span>
        <span class="step-score ${cls}">${label}점</span>
      </div>
      <div class="step-result">${st.result}</div>
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
  const fc = d.forecast;
  const xgb = d.xgb_forecast;
  const risk = d.risk_scenarios;
  const last = d.last_close;

  // 예측 요약
  const sumEl = document.getElementById('forecast-summary');
  if (!fc) {
    sumEl.innerHTML = '<p style="color:#484f58;font-size:13px">예측 데이터 부족</p>';
  } else {
    const hwF = fc.yhat[fc.yhat.length - 1];
    const xgbF = xgb ? xgb[xgb.length - 1] : null;
    const ens = xgbF != null ? hwF * 0.6 + xgbF * 0.4 : hwF;
    const ensChg = (ens - last) / last * 100;
    const ensUp = ensChg >= 0;
    const clr = isKrx ? (ensUp ? '#f85149' : '#388bfd') : (ensUp ? '#3fb950' : '#f85149');
    sumEl.innerHTML = `
      <div style="background:#21262d;border-radius:10px;padding:12px;text-align:center">
        <div style="font-size:11px;color:#8b949e;margin-bottom:4px">HW 예측 (30일)</div>
        <div style="font-size:15px;font-weight:700;color:#388bfd">${fmt(hwF, isKrx)}</div>
      </div>
      ${xgbF != null ? `<div style="background:#21262d;border-radius:10px;padding:12px;text-align:center">
        <div style="font-size:11px;color:#8b949e;margin-bottom:4px">XGBoost (30일)</div>
        <div style="font-size:15px;font-weight:700;color:#d29922">${fmt(xgbF, isKrx)}</div>
      </div>` : ''}
      <div style="background:#0d1f4f;border:1px solid #1f4f8e;border-radius:10px;padding:12px;text-align:center">
        <div style="font-size:11px;color:#8b949e;margin-bottom:4px">🤖 AI 앙상블</div>
        <div style="font-size:17px;font-weight:800;color:#fff">${fmt(ens, isKrx)}</div>
        <div style="font-size:12px;color:${clr}">${ensUp?'▲':'▼'} ${Math.abs(ensChg).toFixed(2)}%</div>
      </div>`;
  }

  // 리스크 카드
  const rgEl = document.getElementById('risk-grid');
  const riskList = Object.entries(risk);
  const riskClasses = ['conservative', 'balanced', 'aggressive'];
  rgEl.innerHTML = riskList.map(([key, sc], i) => `
    <div class="risk-card ${riskClasses[i]}">
      <div class="risk-icon">${sc.icon}</div>
      <div class="risk-name">${sc.label}</div>
      <div class="risk-desc">${sc.desc}</div>
      <div class="risk-row"><span class="risk-lbl">🎯 목표가</span><span class="risk-tgt">${fmt(sc.target, isKrx)}</span></div>
      <div class="risk-row"><span class="risk-lbl">🛑 손절가</span><span class="risk-stp">${fmt(sc.stop, isKrx)}</span></div>
      <div class="risk-ratio">손익비 ${sc.ratio}</div>
    </div>`).join('');
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
      ? discs.map(n => `<div class="news-item"><span class="news-dot">📌</span><div><a class="news-a" href="${n.link}" target="_blank">${n.title}</a></div></div>`).join('')
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
        // 버그 수정: hex 코드에 .replace()로 rgba 변환 안 됨 → volUpClr/volDnClr 변수 사용
        volData.push({ time: cd.dates[i], value: cd.volume[i],
          color: isBull ? volUpClr : volDnClr });
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
      l70.setData([{ time: cd.dates[rsiStartIdx], value: 70 }, { time: cd.dates[n-1], value: 70 }]);
      const l30 = chart.addLineSeries({ color: 'rgba(56,139,253,0.5)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed });
      l30.setData([{ time: cd.dates[rsiStartIdx], value: 30 }, { time: cd.dates[n-1], value: 30 }]);
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

  // ── 예측 차트 ──
  renderForecastChart(d, isKrx);
}

function renderForecastChart(d, isKrx) {
  const fc = d.forecast;
  const xgb = d.xgb_forecast;
  const cd = d.chart_data;
  const n = cd.dates.length;
  const fcEl = document.getElementById('forecast-chart');
  if (!fcEl || !fc) return;

  const chart = LightweightCharts.createChart(fcEl, {
    layout: { background: { color: '#161b22' }, textColor: '#8b949e' },
    grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
    rightPriceScale: { borderColor: '#30363d' },
    timeScale: { borderColor: '#30363d', timeVisible: true },
  });
  chartInstances['forecast'] = chart;

  // 실제 가격
  const actualSeries = chart.addLineSeries({ color: '#e6edf3', lineWidth: 1.5, title: '실제' });
  const histSlice = Math.min(n, 60);
  actualSeries.setData(
    cd.close.slice(-histSlice).map((v,i) => ({ time: cd.dates[n-histSlice+i], value: v }))
    .filter(p => p.value != null)
  );

  // HW 예측
  const hwSeries = chart.addLineSeries({
    color: '#388bfd', lineWidth: 1.5,
    lineStyle: LightweightCharts.LineStyle.Dashed, title: 'HW'
  });
  hwSeries.setData(fc.yhat.map((v,i) => ({ time: fc.dates[i], value: v })));

  // 신뢰구간
  const upperSeries = chart.addLineSeries({ color: 'rgba(59,130,246,0.2)', lineWidth: 1 });
  upperSeries.setData(fc.yhat_upper.map((v,i) => ({ time: fc.dates[i], value: v })));
  const lowerSeries = chart.addLineSeries({ color: 'rgba(59,130,246,0.2)', lineWidth: 1 });
  lowerSeries.setData(fc.yhat_lower.map((v,i) => ({ time: fc.dates[i], value: v })));

  // XGBoost
  if (xgb && fc.dates.length === xgb.length) {
    const xgbSeries = chart.addLineSeries({
      color: '#d29922', lineWidth: 1.5,
      lineStyle: LightweightCharts.LineStyle.Dashed, title: 'XGB'
    });
    xgbSeries.setData(xgb.map((v,i) => ({ time: fc.dates[i], value: v })));
  }
  chart.timeScale().fitContent();
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
  if (tab === 'forecast' && currentData && !chartInstances['forecast']) {
    setTimeout(() => renderForecastChart(currentData, currentData.market === 'KRX'), 50);
  }
}

// ── 스크리너 ──
async function loadScreener() {
  document.getElementById('scrn-loading').style.display = 'block';
  document.getElementById('scrn-result').style.display = 'none';
  try {
    const r = await fetch('/api/screener');
    const d = await r.json();
    screenerData = d.data || [];
    document.getElementById('scrn-subtitle').textContent =
      `국내/해외 주요 종목 실시간 시세 | USD/KRW: ${(d.usd_krw||0).toLocaleString()}`;
    renderScreener();
    document.getElementById('scrn-loading').style.display = 'none';
    document.getElementById('scrn-result').style.display = 'block';
  } catch(e) {
    document.getElementById('scrn-loading').innerHTML = '<p style="color:#f85149">데이터 로딩 실패</p>';
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
  renderScreener();
}

function renderScreener() {
  const marketLabel = scrnMarket === 'domestic' ? '국내' : '해외';
  const filtered = screenerData
    .filter(s => s.market === marketLabel)
    .sort((a, b) => {
      const va = scrnSort.key === 'change' ? a.change : a.volume;
      const vb = scrnSort.key === 'change' ? b.change : b.volume;
      return scrnSort.dir === 'desc' ? vb - va : va - vb;
    });

  const isKrx = scrnMarket === 'domestic';
  const tbody = document.getElementById('scrn-tbody');
  tbody.innerHTML = filtered.map((s, i) => {
    const up = s.change >= 0;
    const clr = isKrx ? (up ? '#f85149' : '#388bfd') : (up ? '#3fb950' : '#f85149');
    const signal = s.change > 3 ? '강력 매수' : s.change > 0 ? '매수' : s.change > -3 ? '중립' : '매도';
    const sigCls = signal === '강력 매수' ? 'sig-buy-strong' : signal === '매수' ? 'sig-buy' : signal === '중립' ? 'sig-neu' : 'sig-sell';
    return `<tr>
      <td style="color:#484f58">${i+1}</td>
      <td><div class="ticker-name">${s.name}</div><div class="ticker-code">${s.ticker}</div></td>
      <td style="text-align:right;font-weight:600">${s.price}</td>
      <td style="text-align:right;font-weight:700;color:${clr}">${up?'▲':'▼'} ${Math.abs(s.change).toFixed(2)}%</td>
      <td><span class="cat-badge">${s.category}</span></td>
      <td style="text-align:right;color:#8b949e;font-size:12px">${s.volume.toLocaleString()}</td>
      <td style="text-align:center"><span class="signal-badge ${sigCls}">${signal}</span></td>
    </tr>`;
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
def _send(handler_self, data: Any, status: int = 200, content_type: str = "application/json"):
    if content_type == "application/json":
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    else:
        body = data if isinstance(data, bytes) else data.encode("utf-8")
    handler_self.send_response(status)
    handler_self.send_header("Content-Type", content_type + "; charset=utf-8")
    handler_self.send_header("Content-Length", str(len(body)))
    handler_self.send_header("Access-Control-Allow-Origin", "*")
    handler_self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
    handler_self.send_header("Access-Control-Allow-Headers", "Content-Type")
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

            result = route(path, params)
            if result is None:
                # HTML 서빙
                _send(self, HTML, 200, "text/html")
            else:
                _send(self, result)
        except Exception as e:
            _send(self, {"error": str(e), "trace": traceback.format_exc()[-400:]}, 500)
