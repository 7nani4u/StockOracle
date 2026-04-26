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
import shutil
import re
import certifi
import ssl
from typing import Optional, Dict, Any, List, Tuple
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote

# 환경에 따른 SSL 문제 해결용 (curl_cffi 관련 오류 회피)
os.environ["YFINANCE_DISABLE_HTTP2"] = "1"

# Windows 환경에서 한글 경로(예: C:\Users\박성곤\...)가 포함될 경우 curl_cffi가 CA 인증서를 못 찾는 오류 방지
cert_path = certifi.where()
if os.name == 'nt' and not cert_path.isascii():
    safe_cert_path = "C:\\Users\\Public\\cacert.pem"
    try:
        if not os.path.exists(safe_cert_path):
            shutil.copy2(cert_path, safe_cert_path)
        cert_path = safe_cert_path
        
        # curl_cffi 내부에서 certifi.where()를 호출할 때 우회된 경로를 반환하도록 몽키패치
        certifi.where = lambda: safe_cert_path
    except Exception:
        pass
os.environ["CURL_CA_BUNDLE"] = cert_path
os.environ["SSL_CERT_FILE"] = cert_path

# ── /tmp 강제 사용 (Vercel은 /tmp 외 쓰기 금지) ───────────────────────────────
if os.name == 'nt':
    # Windows 로컬 개발 환경
    TMP_DIR = tempfile.gettempdir()
else:
    # Vercel / Linux 환경
    TMP_DIR = "/tmp"

os.environ["TMPDIR"] = TMP_DIR
os.environ["HOME"] = TMP_DIR
os.environ["XDG_CACHE_HOME"] = os.path.join(TMP_DIR, "cache")
os.environ["YF_CACHE_DIR"] = os.path.join(TMP_DIR, "yf_cache")

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

import yfinance as yf

# yfinance 내부에서 curl_cffi.curl이 이미 임포트되었을 수 있으므로 명시적으로 덮어쓰기
try:
    import curl_cffi.curl
    if cert_path and cert_path != curl_cffi.curl.DEFAULT_CACERT:
        curl_cffi.curl.DEFAULT_CACERT = cert_path
except Exception:
    pass

# yfinance 내부 curl_cffi 사용 강제 비활성화 (타임아웃, curl: 28 오류 방지)
import yfinance.utils
if hasattr(yfinance.utils, '_HAS_CURL_CFFI'):
    yfinance.utils._HAS_CURL_CFFI = False

# yfinance SQLite 캐시 에러 방지 (DB 파일 생성/접근 완전 차단)
try:
    yf.cache.get_tz_cache().dummy = True
    yf.cache.get_cookie_cache().dummy = True
except:
    pass

# yfinance SQLite 캐시 에러 방지를 위해 메모리 DB 사용 (peewee SqliteDatabase)
try:
    from peewee import SqliteDatabase
    class SafeMemDB(SqliteDatabase):
        def connect(self, reuse_if_open=False):
            try:
                super().connect(reuse_if_open=True)
            except Exception:
                pass
    db = SafeMemDB(':memory:')
    yf.cache.get_tz_cache().db = db
    yf.cache.get_cookie_cache().db = db
except:
    pass

warnings.filterwarnings("ignore")

# ── 의존성 ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# yfinance 타임아웃 및 차단 방지를 위한 전역 설정 (session 래핑 제거)
import yfinance.utils
import yfinance.data
import curl_cffi.curl

# IPv6 네트워크 지연으로 인한 타임아웃(curl: 28) 오류를 방지하기 위해 
# curl_cffi의 setopt 메서드를 몽키패치하여 IPv4를 강제로 사용하도록 설정합니다.
_original_setopt = curl_cffi.curl.Curl.setopt

def _patched_setopt(self, option, value):
    # CurlOpt.URL = 10002
    if option == curl_cffi.curl.CurlOpt.URL:
        # IPv4 강제 (CurlOpt.IPRESOLVE = 113, CURL_IPRESOLVE_V4 = 1)
        _original_setopt(self, curl_cffi.curl.CurlOpt.IPRESOLVE, 1)
        # 타임아웃 기본값 연장 (CurlOpt.TIMEOUT_MS = 115)
        _original_setopt(self, curl_cffi.curl.CurlOpt.TIMEOUT_MS, 30000)
    
    # yfinance가 설정하는 timeout 값이 30초보다 작으면 30초로 덮어쓰기
    if option == curl_cffi.curl.CurlOpt.TIMEOUT_MS and value < 30000:
        value = 30000
        
    return _original_setopt(self, option, value)

curl_cffi.curl.Curl.setopt = _patched_setopt

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

@ttl_cache(60)  # 1분 캐시
def get_usd_krw() -> float:
    """USD/KRW 환율 조회 (1분 캐시) — 중복 호출 방지용 단일 함수"""
    try:
        # 1분봉으로 당일 최신 환율 조회
        df = yf.Ticker("USDKRW=X").history(period="1d", interval="1m")
        if not df.empty:
            return float(df["Close"].iloc[-1])
        # Fallback: 5일 일봉
        df = yf.Ticker("USDKRW=X").history(period="5d")
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
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

@ttl_cache(120)   # 2분
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

    # 지표 계산(MA60, MA120 등)을 위해 항상 요청 기간보다 충분히 긴 과거 데이터를 가져오도록 매핑
    fetch_period = yf_period
    if interval == "1d":
        if period in ["1d", "3d", "1wk", "2wk", "1mo", "3mo", "6mo"]:
            fetch_period = "1y"
        elif period == "1y":
            fetch_period = "2y"
        elif period == "2y":
            fetch_period = "5y"
        elif period == "5y":
            fetch_period = "10y"
    else:
        # 분봉 데이터일 때도 과거 지표 계산을 위해 넉넉하게 가져옴
        if interval == "5m":
            fetch_period = "5d"  # 1일 요청 시 5일치 가져와서 자름
        elif interval == "15m":
            fetch_period = "1mo" # 3일 요청 시 1달치
        elif interval == "30m":
            fetch_period = "1mo" # 1주 요청 시 1달치
        elif interval == "1h":
            fetch_period = "3mo" # 2주, 1달 요청 시 3달치

    try:
        obj = yf.Ticker(sym)
        
        # 1. 1차 시도: 요청받은 분봉 단위로 조회
        try:
            df = obj.history(period=fetch_period, interval=interval, auto_adjust=True)
        except TypeError as e:
            # yfinance 내부에서 분봉 데이터가 없을 때 발생하는 TypeError 방어
            if "NoneType" in str(e):
                df = pd.DataFrame()
            else:
                raise e
        
        # 코스닥 종목(.KS -> .KQ) 교체 후 재시도 로직
        if (df is None or df.empty) and market == "KRX" and sym.endswith(".KS"):
            sym = sym.replace(".KS", ".KQ")
            try:
                df = yf.Ticker(sym).history(period=fetch_period, interval=interval, auto_adjust=True)
            except TypeError as e:
                if "NoneType" in str(e):
                    df = pd.DataFrame()
                else:
                    raise e
            
        # 2. 2차 시도: 분봉 조회가 실패하거나 데이터가 부족한 경우 일봉으로 Fallback
        # TypeError: 'NoneType' object is not subscriptable 오류가 발생한 경우 df가 빈 DataFrame일 수 있음
        if (df is None or df.empty or len(df) < 20) and interval != "1d":
            interval = "1d"
            fetch_period = "1y"  # 일봉 Fallback 시 지표 계산을 위해 1y 사용
            try:
                df = yf.Ticker(sym).history(period=fetch_period, interval=interval, auto_adjust=True)
            except TypeError as e:
                if "NoneType" in str(e):
                    df = pd.DataFrame()
                else:
                    raise e
            
        if df is None or df.empty:
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
        df = df.dropna(subset=["Close"])
        
        # 원래 요청한 기간(period)에 맞게 데이터 자르기 (지표 계산 후)
        # 일봉/분봉 조회일 경우 지표 계산용 과거 데이터를 잘라내고 원래 원했던 기간만큼만 필터링
        if interval == "1d":
            if period == "1d": df = df.tail(5)
            elif period == "3d" or period == "1wk": df = df.tail(10)
            elif period == "2wk" or period == "1mo": df = df.tail(25)
            elif period == "3mo": df = df.tail(65)
            elif period == "6mo": df = df.tail(130)
            elif period == "1y": df = df.tail(252)
            elif period == "2y": df = df.tail(504)
            elif period == "5y": df = df.tail(1260)
        else:
            # 분봉일 때 원래 요청 기간에 맞춰 필터링 (영업일 기준)
            unique_dates = pd.Series(df.index.date).unique()
            if period == "1d":
                target_dates = unique_dates[-1:]
            elif period == "3d":
                target_dates = unique_dates[-3:]
            elif period == "1wk":
                target_dates = unique_dates[-5:] # 1주는 약 5영업일
            elif period == "2wk":
                target_dates = unique_dates[-10:] # 2주는 약 10영업일
            elif period == "1mo":
                target_dates = unique_dates[-21:] # 1달은 약 21영업일
            else:
                target_dates = unique_dates
            
            df = df[np.isin(df.index.date, target_dates)]

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
        if interval == "1d":
            df2["Date"] = df2["Date"].dt.strftime("%Y-%m-%d")
        else:
            df2["Date"] = df2["Date"].astype("int64") // 10**9
            
        d = df2.where(pd.notna(df2), other=None).to_dict(orient="list")
        return d, news, sym
    except Exception as e:
        print(f"[StockOracle Error] fetch_stock_data 예외 발생: {e}")
        return None, None, f"조회 중 예외 발생: {str(e)}"

@ttl_cache(60)  # 1분 캐시
def fetch_naver(code: str):
    code = str(code).replace(".KS","").replace(".KQ","")
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    r = {"price":None,"prev_close":None,"market_cap":None,"per":None,"pbr":None,"opinion":None,"news":[],"disclosures":[]}
    try:
        # JSON API로 현재가 및 전일종가 조회 (1분 캐시)
        # 1차 시도: Mobile JSON API
        try:
            m_url = f"https://m.stock.naver.com/api/stock/{code}/basic"
            m_resp = requests.get(m_url, timeout=5)
            if m_resp.status_code == 200:
                m_data = m_resp.json()
                cur_val = m_data.get("closePrice")
                if cur_val:
                    r["price"] = cur_val.replace(",", "")
                    # compareToPreviousClosePrice 값 추출 로직
                    diff_val = m_data.get("compareToPreviousClosePrice", "0").replace(",", "")
                    # compareToPreviousPrice.code가 5(하락)이면 빼주고, 그 외는 더함
                    code_type = m_data.get("compareToPreviousPrice", {}).get("code", "3")
                    if code_type == "5":
                        r["prev_close"] = str(float(r["price"]) + float(diff_val))
                    else:
                        r["prev_close"] = str(float(r["price"]) - float(diff_val))
        except Exception as e:
            pass

        # 2차 시도: 기존 polling API (Mobile API 실패 시)
        if not r["price"]:
            json_url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}"
            try:
                j_resp = requests.get(json_url, timeout=5)
                j_data = j_resp.json()
                if j_data and "datas" in j_data and len(j_data["datas"]) > 0:
                    item = j_data["datas"][0]
                    cur_val = item.get("closePriceRaw")
                    diff_val = item.get("compareToPreviousClosePriceRaw")
                    if cur_val:
                        r["price"] = cur_val
                        if diff_val:
                            r["prev_close"] = str(float(cur_val) - float(diff_val))
            except Exception as e:
                print(f"Naver JSON API Error: {e}")

        # User-Agent 업데이트 및 타임아웃 증가
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://finance.naver.com/"
        }
        resp = requests.get(url, headers=hdrs, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # 가격 (JSON 실패시 Fallback)
        if not r["price"]:
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

def get_us_realtime_price(ticker_obj) -> Tuple[float, str, float]:
    """
    현재 시간(US Eastern Time 기준)과 서머타임(DST) 적용 여부를 고려하여
    카카오페이증권 장 구분(데이, 프리, 정규, 애프터)에 맞는 최적의 현재가와 세션명, 전일종가를 반환.
    """
    try:
        from zoneinfo import ZoneInfo
        us_tz = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        us_tz = pytz.timezone("America/New_York")
        
    now_et = dt.now(us_tz)
    time_float = now_et.hour + now_et.minute / 60.0
    
    info = ticker_obj.info
    
    # yfinance 가격 데이터 수집
    regular_price = info.get("regularMarketPrice")
    current_price = info.get("currentPrice")
    pre_price = info.get("preMarketPrice")
    post_price = info.get("postMarketPrice")
    prev_close = info.get("previousClose")
    
    # fast_info를 통한 최신 체결가 (가장 빠름)
    fast_last_price = None
    try:
        fi = ticker_obj.fast_info
        if hasattr(fi, 'last_price'):
            fast_last_price = fi.last_price
        if not prev_close and hasattr(fi, 'previous_close'):
            prev_close = fi.previous_close
    except:
        pass

    # 시장 세션 판별 및 최우선 가격 추출 (카카오페이증권 기준)
    if 4.0 <= time_float < 9.5:
        session_name = "프리마켓"
        price = pre_price or fast_last_price or current_price or regular_price
    elif 9.5 <= time_float < 16.0:
        session_name = "정규장"
        price = fast_last_price or current_price or regular_price
    elif 16.0 <= time_float < 20.0:
        session_name = "애프터마켓"
        price = post_price or fast_last_price or current_price or regular_price
    else:
        session_name = "데이마켓"
        # yfinance는 24시간 대체거래소(Blue Ocean 등) 시세를 공식 지원하지 않을 수 있음.
        # 따라서 데이마켓 중에는 가장 최근에 갱신된 애프터마켓 종가나 정규장 종가를 참조.
        price = post_price or regular_price or prev_close

    if price is None:
        price = prev_close or 0.0
        
    return float(price), session_name, float(prev_close) if prev_close else 0.0


@ttl_cache(120)  # 2분 캐시 — 장중 수급 변화 대응
def fetch_investor_flow(ticker: str) -> dict:
    """토스증권 공개 API에서 투자자별 순매수 + 외국인 보유율 조회 (KRX 전용).

    Returns:
        성공: {"ok": True, "date": ..., "개인": ..., ...}
        실패: {"ok": False, "reason": "<오류 원인>"}
    """
    code = str(ticker).replace(".KS", "").replace(".KQ", "").strip()
    if not code.isdigit() or len(code) != 6:
        return {"ok": False, "reason": "KRX 6자리 코드 아님"}

    url = (
        "https://wts-info-api.tossinvest.com/api/v1/stock-infos/trade/trend/trading-trend"
        f"?productCode=A{code}&size=60"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Origin":  "https://tossinvest.com",
        "Referer": "https://tossinvest.com/",
        "Accept":  "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        resp.raise_for_status()
        body = resp.json().get("body", [])
        if not body:
            return {"ok": False, "reason": "수급 데이터 없음 (거래 없는 종목)"}
        row = body[0]
    except requests.exceptions.Timeout:
        return {"ok": False, "reason": "API 타임아웃 (6초 초과)"}
    except requests.exceptions.HTTPError as e:
        return {"ok": False, "reason": f"API HTTP {e.response.status_code}"}
    except Exception as e:
        return {"ok": False, "reason": f"조회 실패: {type(e).__name__}"}

    def _i(v):
        try: return int(v or 0)
        except (TypeError, ValueError): return 0

    def _f(v):
        try: return round(float(v or 0), 2)
        except (TypeError, ValueError): return 0.0

    return {
        "ok":         True,
        "date":       str(row.get("baseDate", "")),
        "개인":       _i(row.get("netIndividualsBuyVolume")),
        "외국인":     _i(row.get("netForeignerBuyVolume")),
        "기관":       _i(row.get("netInstitutionBuyVolume")),
        "연기금":     _i(row.get("netPensionFundBuyVolume")),
        "금융투자":   _i(row.get("netFinancialInvestmentBuyVolume")),
        "투신":       _i(row.get("netTrustBuyVolume")),
        "사모":       _i(row.get("netPrivateEquityFundBuyVolume")),
        "보험":       _i(row.get("netInsuranceBuyVolume")),
        "은행":       _i(row.get("netBankBuyVolume")),
        "기타금융":   _i(row.get("netOtherFinancialInstitutionsBuyVolume")),
        "기타법인":   _i(row.get("netOtherCorporationBuyVolume")),
        "외국인비율": _f(row.get("foreignerRatio")),
    }


@ttl_cache(60)
def fetch_metrics(item):
    try:
        t = yf.Ticker(item["ticker"])
        i = t.info
        
        # Basic Price Data
        if item["market_type"] == "US":
            price, _, _ = get_us_realtime_price(t)
        else:
            fi = t.fast_info
            price = fi.last_price if hasattr(fi, 'last_price') else (i.get("currentPrice") or i.get("regularMarketPrice"))
            
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
        fi = t.fast_info
        if not i or i.get("quoteType") == "MUTUALFUND":
            return None

        # ── 현재가 ──────────────────────────────────────────────
        price = (fi.last_price if hasattr(fi, 'last_price') else None) or (i.get("currentPrice")
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
        fi = t.fast_info
        
        price = (fi.last_price if hasattr(fi, 'last_price') else None) or i.get("currentPrice") or i.get("regularMarketPrice")
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

def get_market_weights(market: str) -> dict:
    """시장별 최적화된 팩터 가중치 반환"""
    if market == "US":
        return {"trend": 40.0, "momentum": 25.0, "volatility": 15.0, "volume": 10.0, "quality": 10.0}
    else:
        return {"trend": 20.0, "momentum": 35.0, "volatility": 25.0, "volume": 20.0, "quality": 0.0}

def check_market_regime(market: str) -> str:
    """시장 전체의 추세를 판단하여 투자 비중 조절 신호 발생"""
    index_ticker = "^KS11" if market == "KRX" else "^GSPC"
    try:
        df = yf.Ticker(index_ticker).history(period="6mo")
        if df.empty or len(df) < 120:
            return "NEUTRAL"
        current_close = df['Close'].iloc[-1]
        ma60 = df['Close'].rolling(60).mean().iloc[-1]
        ma120 = df['Close'].rolling(120).mean().iloc[-1]
        
        if current_close < ma120 and ma60 < ma120:
            return "BEAR"
        elif current_close > ma60 > ma120:
            return "BULL"
        else:
            return "NEUTRAL"
    except:
        return "NEUTRAL"

def validate_financial_health(ticker_info: dict) -> bool:
    """투자 전략 수립 시 부채 비율(레버리지) 반드시 확인"""
    debt_to_equity = ticker_info.get("debtToEquity")
    if debt_to_equity is None:
        return False
    debt_pct = debt_to_equity * 100 if debt_to_equity < 10 else debt_to_equity
    return debt_pct <= 150.0

def analyze_score(dd: Dict, market: str = "KRX"):
    """
    가중치 기반 종합 점수 산출 (시장별 동적 가중치 적용)
    """
    weights = get_market_weights(market)
    w_trend = weights["trend"]
    w_mom = weights["momentum"]
    w_vol = weights["volatility"]
    w_volm = weights["volume"]
    w_qual = weights["quality"]

    closes = dd.get("Close", [])
    if len(closes) < 20:
        fallback_strategy = {
            "step": "💡 AI 종합 진단 및 트레이딩 전략",
            "result": "데이터 부족으로 분석 불가",
            "score": 0,
            "weight": "종합"
        }
        return 50, [], [], [], fallback_strategy

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

    # ── 1. 추세 분석 — EMA(20/50) & MACD & PSAR ──
    ts = 0.0; msgs = []
    max_ts = w_trend / 2.0
    if ema20 and ema50:
        if ema20 > ema50:
            ts += max_ts * 0.35; msgs.append("EMA20 > EMA50 정배열 → 중기 상승 추세")
        else:
            ts -= max_ts * 0.35; msgs.append("EMA20 < EMA50 역배열 → 중기 하락 추세")
    if ema20 and close:
        if close > ema20:
            ts += max_ts * 0.25; msgs.append("현재가 EMA20 상회 → 단기 강세")
        else:
            ts -= max_ts * 0.25; msgs.append("현재가 EMA20 하회 → 단기 약세")
    if macd > sig:
        ts += max_ts * 0.25; msgs.append("MACD 골든크로스 → 상승 전환 신호")
    else:
        ts -= max_ts * 0.25; msgs.append("MACD 데드크로스 → 하락 전환 신호")
    psar_dir = v("PSAR_DIR")
    psar_dir_arr = dd.get("PSAR_DIR", [])
    _prev_pdir = float(psar_dir_arr[-2]) if len(psar_dir_arr) >= 2 and psar_dir_arr[-2] is not None else psar_dir
    if psar_dir == 1.0:
        ts += max_ts * 0.15
        msgs.append("PSAR 상승" + (" 전환 (신규)" if _prev_pdir != 1.0 else " 추세 지속"))
    elif psar_dir == -1.0:
        ts -= max_ts * 0.15
        msgs.append("PSAR 하락" + (" 전환 (신규)" if _prev_pdir != -1.0 else " 추세 지속"))
    ts = max(-max_ts, min(max_ts, ts))
    score += ts
    steps.append({"step": "1. 추세 분석 (EMA·MACD·PSAR)",
                  "result": " | ".join(msgs), "score": round(ts, 1), "weight": f"{w_trend}%"})

    # ── 2. RSI (14) & ADX (14) — 모멘텀 및 추세 신뢰도 ──
    ms = 0.0; msgs = []
    max_ms = w_mom / 2.0
    if   rsi > 70: ms -= max_ms * 0.4; msgs.append(f"RSI {rsi:.1f} 과매수 → 하락 압력 주의")
    elif rsi < 30: ms += max_ms * 0.5; msgs.append(f"RSI {rsi:.1f} 과매도 → 강한 반등 기대")
    elif rsi > 55: ms -= max_ms * 0.1; msgs.append(f"RSI {rsi:.1f} 고점권 — 완만한 하락 압력")
    elif rsi < 45: ms += max_ms * 0.2; msgs.append(f"RSI {rsi:.1f} 저점권 → 매수 관심 구간")
    else:                      msgs.append(f"RSI {rsi:.1f} 중립")
    if adx > 25:
        if dip > dim:
            ms += max_ms * 0.5; msgs.append(f"ADX {adx:.0f} + +DI 우세 → 강한 상승 추세 신뢰")
        else:
            ms -= max_ms * 0.5; msgs.append(f"ADX {adx:.0f} + -DI 우세 → 강한 하락 추세 신뢰")
    elif adx > 20:
        msgs.append(f"ADX {adx:.0f} — 추세 형성 초기")
    else:
        msgs.append(f"ADX {adx:.0f} — 횡보 구간 (추세 약함)")
    ms = max(-max_ms, min(max_ms, ms))
    score += ms
    steps.append({"step": "2. RSI (14) & ADX (14) — 모멘텀 및 추세 신뢰도",
                  "result": " | ".join(msgs), "score": round(ms, 1), "weight": f"{w_mom}%"})

    # ── 3. Bollinger Bands (20, 2) & ATR (14) — 변동성 및 리스크 관리 ──
    vs = 0.0; msgs = []
    max_vs = w_vol / 2.0
    if close and bb_u and bb_l and bb_u > bb_l:
        bb_range = bb_u - bb_l
        pos = (close - bb_l) / bb_range  # 0~1 위치
        if close >= bb_u * 0.98:
            vs -= max_vs * 0.5; msgs.append("볼린저 상단 터치 → 단기 과매수/저항")
        elif close <= bb_l * 1.02:
            vs += max_vs * 0.5; msgs.append("볼린저 하단 터치 → 단기 과매도/지지")
        elif pos > 0.7:
            vs -= max_vs * 0.2; msgs.append("볼린저 상단권 (70%+) → 매도 압력")
        elif pos < 0.3:
            vs += max_vs * 0.2; msgs.append("볼린저 하단권 (30%-) → 지지 기대")
        else:
            msgs.append("볼린저 중간권 → 중립")
        bb_pct = bb_range / close * 100
        if bb_pct < 3.0:
            vs += max_vs * 0.3; msgs.append(f"밴드 수렴 ({bb_pct:.1f}%) → 큰 방향 돌파 임박")
        elif atr and close:
            atr_pct = atr / close * 100
            if atr_pct > 4.0:
                vs -= max_vs * 0.2; msgs.append(f"ATR 고변동 ({atr_pct:.1f}%) → 리스크 증가")
            else:
                msgs.append(f"ATR {atr_pct:.1f}% — 적정 변동성")
    vs = max(-max_vs, min(max_vs, vs))
    score += vs
    steps.append({"step": "3. Bollinger Bands (20,2) & ATR (14) — 변동성 및 리스크 관리",
                  "result": " | ".join(msgs), "score": round(vs, 1), "weight": f"{w_vol}%"})

    # ── 4. Volume — 거래량 급증 확인 ──
    gvs = 0.0; msgs = []
    max_gvs = w_volm / 2.0
    if avg_vol > 0:
        ratio = cur_vol / avg_vol
        last_close = close
        last_opn = last_opn if last_opn else last_close   # 데이터 없으면 종가로 대체

        if ratio > 2.0:
            if last_close > last_opn: gvs += max_gvs; msgs.append(f"거래량 {ratio:.1f}x 급증 + 양봉 → 강한 매수세 확인")
            else:                     gvs -= max_gvs; msgs.append(f"거래량 {ratio:.1f}x 급증 + 음봉 → 강한 매도세 확인")
        elif ratio > 1.5:
            if last_close > last_opn: gvs += max_gvs * 0.5; msgs.append(f"거래량 {ratio:.1f}x 증가 + 상승 → 매수 우위")
            else:                     gvs -= max_gvs * 0.5; msgs.append(f"거래량 {ratio:.1f}x 증가 + 하락 → 매도 압력")
        elif ratio < 0.5:
            msgs.append(f"거래량 급감 ({ratio:.1f}x) → 신뢰도 낮음")
        else:
            msgs.append(f"거래량 평이 ({ratio:.1f}x)")
    else:
        msgs.append("거래량 데이터 없음")
    gvs = max(-max_gvs, min(max_gvs, gvs))
    score += gvs
    steps.append({"step": "4. Volume — 거래량 급증 확인",
                  "result": " | ".join(msgs), "score": round(gvs, 1), "weight": f"{w_volm}%"})

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

def calc_risk(price: float, atr: float, market: str = "KRX", dd: Dict = None) -> Dict:
    if not atr or np.isnan(atr): atr = price * 0.02
    rnd = 4 if market == "US" else 2

    # ── 변동성 동적 계수 산출 ──────────────────────────────────────────
    atr_pct = atr / price * 100  # ATR의 현재가 대비 비율(%)

    # ATR 추세: 최근 5일 ATR vs 이전 20일 ATR 비교 → 변동성 확대/수축 판단
    vol_trend = "normal"
    if dd is not None:
        atrs = [float(x) for x in dd.get("ATR", []) if x is not None]
        if len(atrs) >= 20:
            recent_atr_avg = float(np.mean(atrs[-5:]))
            prev_atr_avg   = float(np.mean(atrs[-20:-5]))
            if prev_atr_avg > 0:
                vol_ratio = recent_atr_avg / prev_atr_avg
                if   vol_ratio > 1.3: vol_trend = "expanding"   # 변동성 확대 → 배수 축소
                elif vol_ratio < 0.7: vol_trend = "contracting" # 변동성 수축 → 배수 확대

    # 변동성 수준별 ATR 배수 조정 테이블
    # atr_pct > 4%: 고변동성 → 보수/중립 배수 줄이고 공격 배수도 제한
    # atr_pct < 1%: 저변동성 → 배수 확대
    if   atr_pct > 4.0: vmul = 0.75
    elif atr_pct > 2.5: vmul = 0.90
    elif atr_pct < 1.0: vmul = 1.25
    elif atr_pct < 1.5: vmul = 1.10
    else:               vmul = 1.00

    if   vol_trend == "expanding":   vmul *= 0.85
    elif vol_trend == "contracting": vmul *= 1.10

    # ── 기술적 지표 참조 (dd 있을 때) ────────────────────────────────
    bb_u  = None; bb_l = None; ma20 = None; ma60 = None
    rsi   = 50.0;  macd = 0.0; sig_line = 0.0
    if dd is not None:
        def _last(k): a = dd.get(k, []); return float(a[-1]) if a and a[-1] is not None else None
        bb_u     = _last("BB_Upper")
        bb_l     = _last("BB_Lower")
        ma20     = _last("MA20")
        ma60     = _last("MA60")
        rsi      = float(dd.get("RSI",          [50])[-1] or 50)
        macd     = float(dd.get("MACD",         [0])[-1]  or 0)
        sig_line = float(dd.get("Signal_Line",  [0])[-1]  or 0)

    # ── 추세 강도 점수 (0~4) ──────────────────────────────────────────
    trend = 0
    if ma20  and price > ma20:  trend += 1
    if ma60  and price > ma60:  trend += 1
    if macd  > sig_line:        trend += 1
    if rsi   > 50:              trend += 1

    # ── 각 리스크 성향별 ATR 배수 (변동성 조정 포함) ─────────────────
    # 보수적: 짧은 손절 · 작은 목표
    cons_stp_mul  = round(0.80 * vmul, 2)
    cons_tgt_mul  = round(1.20 * vmul + trend * 0.10, 2)
    # 중립적: 스윙 트레이딩 배율
    bal_stp_mul   = round(1.30 * vmul, 2)
    bal_tgt_mul   = round(2.50 * vmul + trend * 0.15, 2)
    # 공격적: 추세 추종 배율 (트렌드 강할수록 목표 확대)
    agg_stp_mul   = round(1.80 * vmul, 2)
    agg_tgt_mul   = round(4.50 * vmul + trend * 0.30, 2)

    def _rng(base_mul, width_pct=0.3):
        mid = price + atr * base_mul
        delta = price * width_pct / 100
        return [mid - delta, mid + delta]

    def _stp_rng(base_mul, width_pct=0.3):
        mid = price - atr * base_mul
        delta = price * width_pct / 100
        return [mid - delta, mid + delta]

    # 보수적
    cons_tgt_range = _rng(cons_tgt_mul, 0.25)
    cons_stp_range = _stp_rng(cons_stp_mul, 0.20)
    # BB 하단을 손절 참조로 활용
    if bb_l and float(bb_l) < cons_stp_range[1]:
        cons_stp_range[0] = min(cons_stp_range[0], float(bb_l) - atr * 0.1)
        cons_stp_range[1] = min(cons_stp_range[1], float(bb_l))
    cons_risk   = price - (cons_stp_range[0] + cons_stp_range[1]) / 2
    cons_reward = (cons_tgt_range[0] + cons_tgt_range[1]) / 2 - price
    cons_rr     = round(cons_reward / cons_risk, 2) if cons_risk > 0 else 0
    cons_ret    = round(cons_reward / price * 100, 2)
    cons_stp_pct= round(-cons_risk / price * 100, 2)

    # 중립적
    bal_tgt_range = _rng(bal_tgt_mul, 0.35)
    bal_stp_range = _stp_rng(bal_stp_mul, 0.25)
    if ma20 and float(ma20) < price:
        bal_stp_range[0] = min(bal_stp_range[0], float(ma20) - atr * 0.2)
        bal_stp_range[1] = min(bal_stp_range[1], float(ma20))
    bal_risk   = price - (bal_stp_range[0] + bal_stp_range[1]) / 2
    bal_reward = (bal_tgt_range[0] + bal_tgt_range[1]) / 2 - price
    bal_rr     = round(bal_reward / bal_risk, 2) if bal_risk > 0 else 0
    bal_ret    = round(bal_reward / price * 100, 2)
    bal_stp_pct= round(-bal_risk / price * 100, 2)

    # 공격적: BB 상단 또는 MA 기반 목표가 참조
    agg_tgt_range = _rng(agg_tgt_mul, 0.50)
    if bb_u and float(bb_u) > agg_tgt_range[0]:
        agg_tgt_range[1] = max(agg_tgt_range[1], float(bb_u) + atr * 0.5)
    agg_stp_range = _stp_rng(agg_stp_mul, 0.30)
    agg_risk   = price - (agg_stp_range[0] + agg_stp_range[1]) / 2
    agg_reward = (agg_tgt_range[0] + agg_tgt_range[1]) / 2 - price
    agg_rr     = round(agg_reward / agg_risk, 2) if agg_risk > 0 else 0
    agg_ret    = round(agg_reward / price * 100, 2)
    agg_stp_pct= round(-agg_risk / price * 100, 2)

    # ── 변동성 상태 텍스트 ────────────────────────────────────────────
    vol_state_txt = (
        f"고변동성 구간 (ATR {atr_pct:.1f}%) — 배수 축소 적용"   if atr_pct > 4.0 else
        f"중고변동성 (ATR {atr_pct:.1f}%) — 표준 배수 소폭 축소" if atr_pct > 2.5 else
        f"저변동성 구간 (ATR {atr_pct:.1f}%) — 배수 확대 적용"   if atr_pct < 1.0 else
        f"적정 변동성 (ATR {atr_pct:.1f}%)"
    )
    vol_trend_txt = {
        "expanding":   "⚠️ 최근 변동성 확대 중 → 손절선 여유 권장",
        "contracting": "✅ 변동성 수축 중 → 돌파 시 빠른 대응 유리",
        "normal":      "→ 변동성 안정 구간"
    }[vol_trend]

    r = lambda v: round(v, rnd)
    return {
        "conservative": {
            "label": "보수적",
            "icon": "🛡️",
            "desc": f"리스크 최소화 · 손절 {abs(cons_stp_pct):.2f}%",
            "target": [r(cons_tgt_range[0]), r(cons_tgt_range[1])],
            "stop":   [r(cons_stp_range[0]), r(cons_stp_range[1])],
            "return": cons_ret,
            "rr_ratio": cons_rr,
            "stop_pct": cons_stp_pct,
            "atr_mul_tgt": cons_tgt_mul,
            "atr_mul_stp": cons_stp_mul,
            "interpretation": f"BB 하단 참조 손절 · 단기 반등 목표 (R/R {cons_rr:.1f}:1)"
        },
        "balanced": {
            "label": "중립적",
            "icon": "⚖️",
            "desc": f"스윙 트레이딩 · 손절 {abs(bal_stp_pct):.2f}%",
            "target": [r(bal_tgt_range[0]), r(bal_tgt_range[1])],
            "stop":   [r(bal_stp_range[0]), r(bal_stp_range[1])],
            "return": bal_ret,
            "rr_ratio": bal_rr,
            "stop_pct": bal_stp_pct,
            "atr_mul_tgt": bal_tgt_mul,
            "atr_mul_stp": bal_stp_mul,
            "interpretation": f"MA20 지지 손절 · 중기 추세 목표 (R/R {bal_rr:.1f}:1)"
        },
        "aggressive": {
            "label": "공격적",
            "icon": "🚀",
            "desc": f"추세 추종 · 손절 {abs(agg_stp_pct):.2f}%",
            "target": [r(agg_tgt_range[0]), r(agg_tgt_range[1])],
            "stop":   [r(agg_stp_range[0]), r(agg_stp_range[1])],
            "return": agg_ret,
            "rr_ratio": agg_rr,
            "stop_pct": agg_stp_pct,
            "atr_mul_tgt": agg_tgt_mul,
            "atr_mul_stp": agg_stp_mul,
            "interpretation": f"BB 상단 참조 목표 · 추세 지속 시 최대 수익 (R/R {agg_rr:.1f}:1)"
        },
        "vol_state": vol_state_txt,
        "vol_trend": vol_trend_txt,
        "atr_pct": round(atr_pct, 2),
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

def calc_buy_price(dd: Dict, last_price: float, atr: float, score: float, indicator_signals: Dict, market: str = "KRX") -> Dict:
    """매수 적정 가격 예측 — 다중 지표 기반 정밀 구간 산출"""
    lows     = [float(x) for x in dd.get("Low",   []) if x is not None]
    highs    = [float(x) for x in dd.get("High",  []) if x is not None]
    closes   = [float(x) for x in dd.get("Close", []) if x is not None]
    volumes  = [float(x) for x in dd.get("Volume",[]) if x is not None]

    def _last(k):
        a = dd.get(k, [])
        return float(a[-1]) if a and a[-1] is not None else None

    rsi      = float((dd.get("RSI",          [50])[-1]) or 50)
    bb_l_raw = _last("BB_Lower")
    bb_u_raw = _last("BB_Upper")
    bb_m_raw = _last("BB_Middle")
    ma20_raw = _last("MA20")
    ma60_raw = _last("MA60")
    ema20    = _last("EMA20") or _last("MA20")
    macd     = float(dd.get("MACD",          [0])[-1] or 0)
    sig_line = float(dd.get("Signal_Line",   [0])[-1] or 0)
    adx      = float(dd.get("ADX",           [20])[-1] or 20)
    bp       = _last("BUY_PRESSURE") or 50.0

    if not atr or np.isnan(atr):
        atr = last_price * 0.02

    rnd = 4 if market == "US" else 2

    # ── 변동성 동적 계수 ──────────────────────────────────────────────
    atr_pct = atr / last_price * 100
    atrs = [float(x) for x in dd.get("ATR", []) if x is not None]
    vol_trend = "normal"
    if len(atrs) >= 20:
        recent_vol = float(np.mean(atrs[-5:]))
        prev_vol   = float(np.mean(atrs[-20:-5]))
        if prev_vol > 0:
            vr = recent_vol / prev_vol
            if   vr > 1.3: vol_trend = "expanding"
            elif vr < 0.7: vol_trend = "contracting"

    # ── 지지/저항 분석 ────────────────────────────────────────────────
    recent_lows  = sorted([x for x in lows[-30:] if x > 0])
    support_zone = float(np.mean(recent_lows[:5])) if len(recent_lows) >= 5 else last_price * 0.95

    # 최근 20일 저점 클러스터링 → 핵심 지지대 산출
    lows20 = sorted([x for x in lows[-20:] if x > 0])
    strong_support = float(np.mean(lows20[:3])) if len(lows20) >= 3 else support_zone

    # 피보나치 되돌림 (최근 60일 고점/저점 기준)
    h60 = max(highs[-60:]) if len(highs) >= 60 else max(highs) if highs else last_price * 1.1
    l60 = min(lows[-60:])  if len(lows)  >= 60 else min(lows)  if lows  else last_price * 0.9
    fib_range = h60 - l60
    fib_382 = h60 - fib_range * 0.382
    fib_500 = h60 - fib_range * 0.500
    fib_618 = h60 - fib_range * 0.618

    # ── 거래량 가중 평균가 (최근 20일 VWAP 근사) ──────────────────────
    vwap_approx = None
    if closes and volumes and len(closes) >= 20 and len(volumes) >= 20:
        c20 = np.array(closes[-20:])
        v20 = np.array(volumes[-20:])
        if v20.sum() > 0:
            vwap_approx = float(np.average(c20, weights=v20))

    # ── 구간별 핵심 앵커 가격 선택 ────────────────────────────────────
    bb_l  = float(bb_l_raw) if bb_l_raw else last_price * 0.97
    bb_m  = float(bb_m_raw) if bb_m_raw else last_price
    ma20  = float(ma20_raw) if ma20_raw else last_price * 0.98
    ma60  = float(ma60_raw) if ma60_raw else last_price * 0.94

    # ── RSI 조정 계수 ─────────────────────────────────────────────────
    if   rsi < 30:  rsi_adj = -0.15  # 과매도 → 구간 더 좁히기(더 적극)
    elif rsi < 40:  rsi_adj = -0.08
    elif rsi > 70:  rsi_adj =  0.15  # 과매수 → 구간 낮추기(더 보수)
    elif rsi > 60:  rsi_adj =  0.07
    else:           rsi_adj =  0.00

    # ── ⚡ 공격적 매수 구간 ────────────────────────────────────────────
    # 근거: ATR 단기 눌림목 + MA20 이탈 전 + MACD 시그널
    # RSI, 거래량 매수압력 조정
    agg_center = last_price - atr * (0.40 + rsi_adj)
    agg_half   = atr * 0.25
    # MA20 가격이 공격 구간 상단보다 낮으면 MA20을 상단 앵커로 활용
    if ma20 < last_price and ma20 > agg_center:
        agg_high = min(ma20, last_price - atr * 0.15)
        agg_low  = agg_high - atr * 0.50
    else:
        agg_low  = agg_center - agg_half
        agg_high = agg_center + agg_half
    agg_pct_l = round((agg_low  - last_price) / last_price * 100, 2)
    agg_pct_h = round((agg_high - last_price) / last_price * 100, 2)

    agg_basis = []
    agg_basis.append(f"단기 ATR 눌림 구간 (ATR×{0.40+rsi_adj:.2f} ≈ {atr*(0.40+rsi_adj):,.2f})")
    if ma20 and abs(ma20 - last_price) / last_price < 0.10:
        agg_basis.append(f"MA20 지지 근접 ({ma20:,.{rnd}f})")
    if macd > sig_line:
        agg_basis.append("MACD 매수 우위 → 단기 상승 동력 확인")
    else:
        agg_basis.append("MACD 매도 우위 → 빠른 반등 기대 시 공격 진입")
    if bp and bp > 55:
        agg_basis.append(f"매수압력 {bp:.1f}% — 상승일 거래량 우세")
    if vol_trend == "contracting":
        agg_basis.append("변동성 수축 중 → 돌파 임박 시 빠른 진입 유리")

    agg_interp = (
        "MA20 위 단기 눌림목. RSI가 중립~저점권이면 빠른 반등 기대 가능."
        if rsi < 55 else
        "RSI 고점권으로 추가 하락 여지 존재. 확인 캔들 후 진입 권장."
    )

    # ── ✅ 추천 매수 구간 ──────────────────────────────────────────────
    # 근거: 볼린저 중간~하단 + MA20 지지 + 피보나치 38.2% 되돌림
    anchors = [bb_l, bb_m * 0.995, ma20 * 0.995]
    if vwap_approx:
        anchors.append(vwap_approx * 0.995)
    if fib_382 > last_price * 0.85 and fib_382 < last_price:
        anchors.append(fib_382)
    rec_low_anchor  = float(np.mean([a for a in anchors if a < last_price])) if any(a < last_price for a in anchors) else last_price - atr * 1.2
    rec_half = atr * (0.35 + abs(rsi_adj) * 0.5)
    rec_low  = rec_low_anchor - rec_half * 0.5
    rec_high = rec_low_anchor + rec_half * 0.5
    rec_pct_l = round((rec_low  - last_price) / last_price * 100, 2)
    rec_pct_h = round((rec_high - last_price) / last_price * 100, 2)

    rec_basis = []
    rec_basis.append(f"볼린저 하단~중간 지지 구간 (BB_L {bb_l:,.{rnd}f} ~ BB_M {bb_m:,.{rnd}f})")
    rec_basis.append(f"MA20 단기 생명선 지지 ({ma20:,.{rnd}f})")
    if vwap_approx:
        rec_basis.append(f"20일 거래량 가중 평균(VWAP≈) {vwap_approx:,.{rnd}f} — 기관 매집 참조")
    if fib_382 > last_price * 0.85 and fib_382 < last_price:
        rec_basis.append(f"피보나치 38.2% 되돌림 ({fib_382:,.{rnd}f}) — 유효 지지")
    rec_basis.append(f"RSI {rsi:.1f} — {('저점권 분할 매수 유리' if rsi < 45 else '중립권 지지 확인 후 진입' if rsi < 60 else '고점권 눌림 대기 필요')}")

    rec_interp = (
        f"볼린저 밴드 중간선과 MA20이 모이는 {rec_low_anchor:,.{rnd}f} 부근이 핵심 지지대. 분할 매수 권장."
    )

    # ── 🛡️ 보수적 매수 구간 ───────────────────────────────────────────
    # 근거: MA60 + 최근 30일 저점군 + 피보나치 50~61.8% 되돌림
    fib_anchor = (fib_500 + fib_618) / 2
    con_anchors = [strong_support, ma60]
    if last_price * 0.75 < fib_anchor < last_price:
        con_anchors.append(fib_anchor)
    con_center = float(np.mean(con_anchors))
    con_half   = atr * (0.60 + abs(rsi_adj) * 0.3)
    con_low    = con_center - con_half * 0.5
    con_high   = con_center + con_half * 0.5
    con_pct_l  = round((con_low  - last_price) / last_price * 100, 2)
    con_pct_h  = round((con_high - last_price) / last_price * 100, 2)

    con_basis = []
    con_basis.append(f"MA60 중기 추세선 지지 ({ma60:,.{rnd}f})")
    con_basis.append(f"최근 30일 핵심 저점군 ({strong_support:,.{rnd}f} ~ {support_zone:,.{rnd}f})")
    if last_price * 0.75 < fib_anchor < last_price:
        con_basis.append(f"피보나치 50~61.8% 되돌림 구간 ({fib_500:,.{rnd}f} ~ {fib_618:,.{rnd}f})")
    if vol_trend == "expanding":
        con_basis.append("변동성 확대 중 → 충분한 하락 소화 후 지지 확인 진입")
    con_basis.append(f"ATR 기반 변동 폭 완충 (ATR {atr:,.{rnd}f} / {atr_pct:.1f}%)")

    con_interp = (
        f"MA60({ma60:,.{rnd}f})과 중기 지지구간이 겹치는 안전지대. "
        "하락 추세 지속 시에도 반등 확률이 높은 구간."
    )

    # ── 타이밍 산출 ──────────────────────────────────────────────────
    now = dt.now()
    signals_dict = indicator_signals.get("signals", {})
    buy_count  = sum(1 for s in signals_dict.values() if s.get("signal") in ["매수", "적극 매수"])
    total_count= len(signals_dict)
    buy_ratio  = buy_count / total_count if total_count > 0 else 0

    buy_delay = 1
    if score >= 70 and buy_ratio >= 0.6:
        buy_delay = 0
    elif rsi < 30 and macd > sig_line:
        buy_delay = 0
    elif score < 40 or rsi > 70:
        buy_delay = 3
    elif score < 50 or rsi > 60:
        buy_delay = 2

    buy_time = now + timedelta(days=buy_delay)
    buy_time = buy_time.replace(hour=10, minute=30)
    while buy_time.weekday() > 4:
        buy_time += timedelta(days=1)

    bb_u_f = float(bb_u_raw) if bb_u_raw else None
    target_dist = (bb_u_f - last_price) if bb_u_f and bb_u_f > last_price else last_price * 0.05
    base_days = target_dist / (atr if atr > 0 else 1)
    if score >= 70:
        days_to_target = max(1, int(base_days * 0.7))
    elif score <= 40:
        days_to_target = max(3, int(base_days * 1.5))
    else:
        days_to_target = max(2, int(base_days))

    sell_time = buy_time + timedelta(days=days_to_target)
    sell_time = sell_time.replace(hour=14, minute=30)
    while sell_time.weekday() > 4:
        sell_time += timedelta(days=1)

    if buy_delay == 0:
        buy_timing_str = f"즉각 진입 가능 (현재 장세 유리, 점수: {round(score)}점)"
    elif buy_delay == 1:
        buy_timing_str = f"단기 눌림목 대기 후 진입 ({buy_time.strftime('%m/%d')} 경)"
    else:
        buy_timing_str = f"관망 후 지지선 확인 진입 ({buy_time.strftime('%m/%d')} 이후)"
    sell_timing_str = f"단기 저항/목표가 도달 예상 ({sell_time.strftime('%m/%d')} 경)"

    if   rsi < 30:  rsi_ctx = "RSI 과매도 — 적극 매수 관점"
    elif rsi < 45:  rsi_ctx = "RSI 저점권 — 매수 유리, 분할 진입"
    elif rsi > 70:  rsi_ctx = "RSI 과매수 — 신규 진입 보류, 눌림 대기"
    elif rsi > 55:  rsi_ctx = "RSI 고점권 — 보수적 접근, 확인 후 진입"
    else:           rsi_ctx = "RSI 중립 — 지지/저항선 돌파 확인 후 진입"

    r = lambda v: round(v, rnd)
    return {
        "current": r(last_price),
        "aggressive": {
            "range":  [r(agg_low), r(agg_high)],
            "pct":    [agg_pct_l, agg_pct_h],
            "basis":  agg_basis,
            "interpretation": agg_interp,
        },
        "recommended": {
            "range":  [r(rec_low), r(rec_high)],
            "pct":    [rec_pct_l, rec_pct_h],
            "basis":  rec_basis,
            "interpretation": rec_interp,
        },
        "conservative": {
            "range":  [r(con_low), r(con_high)],
            "pct":    [con_pct_l, con_pct_h],
            "basis":  con_basis,
            "interpretation": con_interp,
        },
        "timing": {
            "buy":  buy_timing_str,
            "sell": sell_timing_str,
        },
        "support_zone": r(support_zone),
        "fib": {
            "h60": r(h60), "l60": r(l60),
            "f382": r(fib_382), "f500": r(fib_500), "f618": r(fib_618),
        },
        "rsi": round(rsi, 1),
        "rsi_context": rsi_ctx,
        "atr": r(atr),
        "atr_pct": round(atr_pct, 2),
        "vol_trend": vol_trend,
    }

def calc_target_price(dd: Dict, last_price: float, atr: float, period: str, market: str = "KRX") -> Dict:
    """향후 주가 상승 가능 범위(목표가) 예측"""
    ma20 = float(dd.get("MA20", [0])[-1] or 0)
    ma60 = float(dd.get("MA60", [0])[-1] or 0)
    rsi = float(dd.get("RSI", [50])[-1] or 50)
    macd = float(dd.get("MACD", [0])[-1] or 0)
    sig = float(dd.get("Signal_Line", [0])[-1] or 0)
    bb_u = float(dd.get("BB_Upper", [0])[-1] or 0)
    
    if not atr or np.isnan(atr):
        atr = last_price * 0.02
        
    rnd = 4 if market == "US" else 2

    # 추세 강도 분석 (-2 ~ 2)
    trend_strength = 0
    if ma20 and last_price > ma20: trend_strength += 1
    if ma60 and last_price > ma60: trend_strength += 1
    if macd > sig: trend_strength += 1
    if rsi > 50: trend_strength += 1
    if rsi > 70: trend_strength -= 1

    # 사용자 선택 period 기반 목표 기간 및 기본 타겟 계산
    if period in ["1d", "3d", "1wk"]:
        pred_period = "단기 (1주 ~ 2주)"
        base_target = last_price + (atr * 2)
        if bb_u and bb_u > last_price:
            base_target = max(base_target, bb_u)
    elif period in ["2wk", "1mo", "3mo"]:
        pred_period = "중기 (1개월 ~ 3개월)"
        base_target = last_price + (atr * 4)
    else:
        pred_period = "장기 (6개월 이상)"
        base_target = last_price + (atr * 8)
        
    # 추세 강도에 따른 보정 및 근거 작성
    if trend_strength >= 3:
        min_target = base_target
        max_target = base_target + (atr * 2)
        reason = "강한 상승 추세(이동평균선 정배열 및 MACD 매수 우위)가 지속되고 있어 추가 상승 여력이 높습니다."
    elif trend_strength >= 1:
        min_target = base_target - (atr * 1)
        max_target = base_target + (atr * 1)
        reason = "완만한 상승 추세 또는 박스권 상향 돌파 시도 중입니다. 단기 저항선 돌파 여부가 중요합니다."
    else:
        min_target = last_price + (atr * 0.5)
        max_target = last_price + (atr * 1.5)
        reason = "현재 하락 추세 또는 조정 구간입니다. 기술적 반등 시 일차적인 저항선을 목표로 보수적인 접근이 필요합니다."

    # 수익률 계산
    min_return = (min_target - last_price) / last_price * 100
    max_return = (max_target - last_price) / last_price * 100

    return {
        "min_price": round(min_target, rnd),
        "max_price": round(max_target, rnd),
        "min_return": round(min_return, 1),
        "max_return": round(max_return, 1),
        "period": pred_period,
        "reason": reason,
        "trend_strength": trend_strength
    }

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
# 섹터 흐름 기본 종목 목록 (메인 페이지 /api/market/sector-summary 용)
# =============================================================================
_SECTOR_DEFAULT_STOCKS: list = [
    # ── 1행 (7섹터) ──────────────────────────────────────────────────
    # 🔧 반도체
    {"code": "005930", "name": "삼성전자",        "market": "KOSPI", "sector": "반도체"},
    {"code": "000660", "name": "SK하이닉스",      "market": "KOSPI", "sector": "반도체"},
    # ⚡ 전력기기
    {"code": "010120", "name": "LS일렉트릭",      "market": "KOSPI", "sector": "전력기기"},
    {"code": "267260", "name": "HD현대일렉트릭",  "market": "KOSPI", "sector": "전력기기"},
    # 🚢 조선
    {"code": "009540", "name": "HD한국조선해양",  "market": "KOSPI", "sector": "조선"},
    {"code": "010140", "name": "삼성중공업",      "market": "KOSPI", "sector": "조선"},
    # 🛡️ 방산
    {"code": "012450", "name": "한화에어로스페이스", "market": "KOSPI", "sector": "방산"},
    {"code": "079550", "name": "LIG넥스원",       "market": "KOSPI", "sector": "방산"},
    # ⛽ 정유
    {"code": "096770", "name": "SK이노베이션",    "market": "KOSPI", "sector": "정유"},
    {"code": "010950", "name": "S-Oil",           "market": "KOSPI", "sector": "정유"},
    # 🚗 자동차
    {"code": "005380", "name": "현대차",          "market": "KOSPI", "sector": "자동차"},
    {"code": "000270", "name": "기아",            "market": "KOSPI", "sector": "자동차"},
    # 🔋 배터리
    {"code": "373220", "name": "LG에너지솔루션",  "market": "KOSPI", "sector": "배터리"},
    {"code": "006400", "name": "삼성SDI",         "market": "KOSPI", "sector": "배터리"},
    # ── 2행 (7섹터) ──────────────────────────────────────────────────
    # 🧬 바이오
    {"code": "068270", "name": "셀트리온",        "market": "KOSPI", "sector": "바이오"},
    {"code": "207940", "name": "삼성바이오로직스", "market": "KOSPI", "sector": "바이오"},
    # 🏦 금융
    {"code": "105560", "name": "KB금융",          "market": "KOSPI", "sector": "금융"},
    {"code": "055550", "name": "신한지주",        "market": "KOSPI", "sector": "금융"},
    # 🌐 인터넷
    {"code": "035420", "name": "NAVER",           "market": "KOSPI", "sector": "인터넷"},
    {"code": "035720", "name": "카카오",          "market": "KOSPI", "sector": "인터넷"},
    # 📡 통신
    {"code": "017670", "name": "SK텔레콤",        "market": "KOSPI", "sector": "통신"},
    {"code": "030200", "name": "KT",              "market": "KOSPI", "sector": "통신"},
    # 🧪 화학
    {"code": "051910", "name": "LG화학",          "market": "KOSPI", "sector": "화학"},
    {"code": "011170", "name": "롯데케미칼",      "market": "KOSPI", "sector": "화학"},
    # 🏭 철강
    {"code": "005490", "name": "POSCO홀딩스",     "market": "KOSPI", "sector": "철강"},
    {"code": "004020", "name": "현대제철",        "market": "KOSPI", "sector": "철강"},
    # 💡 에너지
    {"code": "015760", "name": "한국전력",        "market": "KOSPI", "sector": "에너지"},
    {"code": "036460", "name": "한국가스공사",    "market": "KOSPI", "sector": "에너지"},
]

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
        dd, news, err_or_sym = fetch_stock_data(ticker, market, period)
        if dd is None:
            return {"error": f"데이터 조회 실패: {err_or_sym}"}
        sym = err_or_sym
        closes = dd.get("Close", [])
        last = float(closes[-1]) if closes else 0
        prev = float(closes[-2]) if len(closes) > 1 else last
        pct = (last - prev) / prev * 100 if prev else 0
        score, steps, patterns, geo_patterns, ai_strategy = analyze_score(dd, market)
        
        # Market Regime 필터 적용
        regime = check_market_regime(market)
        if regime == "BEAR":
            if isinstance(ai_strategy, dict):
                ai_strategy["result"] += " | [시장 상태] 시장 전체 하락장(BEAR) 진입: 신규 매수 금지 및 현금 비중 확대 권장"
            else:
                ai_strategy = {"step": "💡 AI 종합 진단", "result": "시장 전체 하락장(BEAR) 진입: 신규 매수 금지 및 현금 비중 확대 권장"}
            score = min(score, 40) # 하락장에서는 점수 강제 하향
        elif regime == "BULL":
            if isinstance(ai_strategy, dict):
                ai_strategy["result"] += " | [시장 상태] 시장 전체 상승장(BULL) 진행 중: 적극 매수 유리"
            else:
                ai_strategy = {"step": "💡 AI 종합 진단", "result": "시장 전체 상승장(BULL) 진행 중: 적극 매수 유리"}
            
        # 부채비율 검증 로직 적용
        try:
            info = yf.Ticker(sym).info
            if not validate_financial_health(info):
                if isinstance(ai_strategy, dict):
                    ai_strategy["result"] += " | ⚠️ [경고] 부채비율 150% 초과 또는 재무 데이터 누락으로 투자 위험 높음"
                else:
                    ai_strategy = {"step": "💡 AI 종합 진단", "result": "⚠️ [경고] 부채비율 150% 초과 또는 재무 데이터 누락으로 투자 위험 높음"}
                score = min(score, 45)
        except:
            pass
        
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
        risk             = calc_risk(last, atr_val, market, dd)
        pivot_points     = calc_pivot_points(dd)
        indicator_signals= calc_indicator_signals(dd)
        buy_price        = calc_buy_price(dd, last, atr_val, score, indicator_signals, market)
        target_price     = calc_target_price(dd, last, atr_val, period, market)
        naver = fetch_naver(sym) if market == "KRX" else None

        # ── 투자자 수급 (KRX 전용, 토스증권 API) ───────────────────────────
        investor_flow = {"ok": False, "reason": "KRX 종목 아님"}
        if market == "KRX":
            investor_flow = fetch_investor_flow(sym)
            if investor_flow.get("ok"):
                foreign  = investor_flow.get("외국인", 0)
                inst     = investor_flow.get("기관", 0)
                pension  = investor_flow.get("연기금", 0)
                # 방향성 신호 → 점수 소폭 보정 (상한 ±5)
                adj = 0
                if foreign > 0: adj += 2
                elif foreign < 0: adj -= 2
                if inst > 0: adj += 2
                elif inst < 0: adj -= 2
                if pension > 0: adj += 1   # 연기금은 신뢰도 높음
                elif pension < 0: adj -= 1
                score = max(0, min(100, score + max(-5, min(5, adj))))
                # AI 전략 텍스트에 수급 메모 추가
                flow_notes = []
                if foreign != 0: flow_notes.append(f"외국인 {foreign:+,}주")
                if inst    != 0: flow_notes.append(f"기관 {inst:+,}주")
                if pension != 0: flow_notes.append(f"연기금 {pension:+,}주")
                if flow_notes and isinstance(ai_strategy, dict):
                    ai_strategy["result"] += " | [투자자 수급] " + " / ".join(flow_notes)

        # US 주식 보강 데이터 (Finnhub / Alpha Vantage / Tiingo)
        us_enriched = None
        if market == "US":
            try:
                import sys as _sys, os as _os
                _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
                from market_briefing.us_enricher import fetch_us_enriched
                us_enriched = fetch_us_enriched(sym)
            except Exception:
                pass

        # 현재가 보정: 한국 시장인 경우 네이버 금융의 최신 현재가를 최우선으로 사용
        # 미국 시장은 yfinance의 fast_info 객체를 활용해 실시간 데이터 보정
        if market == "KRX" and naver and naver.get("price"):
            try:
                real_price = float(naver["price"])
                # 네이버 현재가와 yfinance 종가의 차이가 30% 이내일 때만 보정 (액면분할 등 비정상적 차이 방지)
                if last > 0 and abs(real_price - last) / last < 0.3:
                    last = real_price
                    # 전일종가를 네이버에서 가져온 prev_close로 교체
                    nv_prev = naver.get("prev_close")
                    if nv_prev:
                        try:
                            prev = float(nv_prev)  # 실제 전일종가 사용
                        except Exception:
                            pass
                    if prev > 0:
                        pct = (last - prev) / prev * 100
            except:
                pass
        elif market == "US":
            try:
                fast_info = yf.Ticker(sym).fast_info
                # hasattr를 통해 안전하게 last_price 조회
                if hasattr(fast_info, 'last_price'):
                    real_price = fast_info.last_price
                    if real_price and real_price > 0:
                        if last > 0 and abs(real_price - last) / last < 0.3:
                            last = float(real_price)
                            # 이전 종가도 보정
                            if hasattr(fast_info, 'previous_close'):
                                real_prev = fast_info.previous_close
                                if real_prev and real_prev > 0:
                                    prev = float(real_prev)
                            if prev > 0:
                                pct = (last - prev) / prev * 100
                else:
                    # fallback: info 객체 사용
                    info = yf.Ticker(sym).info
                    real_price = info.get("currentPrice") or info.get("regularMarketPrice")
                    if real_price and real_price > 0:
                        if last > 0 and abs(real_price - last) / last < 0.3:
                            last = float(real_price)
                            real_prev = info.get("previousClose")
                            if real_prev and real_prev > 0:
                                prev = float(real_prev)
                            if prev > 0:
                                pct = (last - prev) / prev * 100
            except:
                pass
                
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
            "target_price": target_price,
            "news": news or [], "naver": naver, "us_enriched": us_enriched,
            "investor_flow": investor_flow,
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

    # ── market_briefing 통합 엔드포인트 ─────────────────────────────────────
    # k-ant-daily 로직을 이식한 3가지 분석 모듈
    # /api/market/summary  → ⭐ 오늘의 핵심 (거시 지표 + 뉴스 무드)
    # /api/market/sectors  → 🏭 섹터 흐름 (종목 리스트 기반 섹터 집계)
    # /api/market/stocks   → 📈 종목별 분석 (3-신호 매트릭스 + 검증)

    if path == "/api/market/summary":
        try:
            import sys as _sys
            import os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from market_briefing.data_fetcher import fetch_macro_context
            from market_briefing.core_summary import build_core_summary
            ctx = fetch_macro_context()
            return build_core_summary(ctx)
        except Exception as e:
            return {"error": f"거시 요약 조회 실패: {e}"}

    if path == "/api/market/sectors":
        # ?codes=005930,000660,...  &markets=KOSPI,KOSPI,...
        # &proxies=^SOX:005930,NVDA:000660,...  (선택)
        try:
            import sys as _sys
            import os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from market_briefing.data_fetcher import fetch_stock_list_snapshot
            from market_briefing.sector_flow import build_sector_flow
            import yaml as _yaml

            codes_raw   = params.get("codes", "")
            markets_raw = params.get("markets", "")
            sectors_raw = params.get("sectors", "")
            codes   = [c.strip() for c in codes_raw.split(",") if c.strip()]
            markets_list = [m.strip() for m in markets_raw.split(",") if m.strip()]
            sectors_list = [s.strip() for s in sectors_raw.split(",") if s.strip()]
            if not codes:
                return {"error": "codes 파라미터 필요 (예: ?codes=005930,000660)"}
            stock_cfgs = []
            for i, code in enumerate(codes):
                stock_cfgs.append({
                    "code":   code,
                    "market": markets_list[i] if i < len(markets_list) else "KOSPI",
                    "sector": sectors_list[i] if i < len(sectors_list) else None,
                })
            snapshots = fetch_stock_list_snapshot(stock_cfgs)
            return build_sector_flow(snapshots)
        except Exception as e:
            return {"error": f"섹터 흐름 조회 실패: {e}"}

    if path == "/api/market/sector-summary":
        # 코드 파라미터 없이 기본 대표 종목으로 섹터 흐름 반환
        # 메인 페이지 자동 로드용 — 캐시 s-maxage=600 (10분)
        # quote만 병렬 수집 (history/news 생략) → 대폭 빠름
        try:
            import sys as _sys
            import os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from market_briefing.data_fetcher import fetch_stock_list_quote_only
            from market_briefing.sector_flow import build_sector_flow
            snapshots = fetch_stock_list_quote_only(_SECTOR_DEFAULT_STOCKS)
            return build_sector_flow(snapshots)
        except Exception as e:
            return {"error": f"섹터 흐름 조회 실패: {e}"}

    if path == "/api/market/stocks":
        # ?codes=005930,000660,...  필수
        # &markets=KOSPI,KOSPI,...  선택 (순서 대응)
        # &evening=1               선택 (저녁 검증 모드: 당일 종가로 적중 판정)
        try:
            import sys as _sys
            import os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from market_briefing.data_fetcher import fetch_stock_list_snapshot
            from market_briefing.stock_analyzer import build_stock_report

            codes_raw   = params.get("codes", "")
            markets_raw = params.get("markets", "")
            evening_mode = params.get("evening", "0") == "1"
            codes        = [c.strip() for c in codes_raw.split(",") if c.strip()]
            markets_list = [m.strip() for m in markets_raw.split(",") if m.strip()]
            if not codes:
                return {"error": "codes 파라미터 필요 (예: ?codes=005930,000660)"}
            stock_cfgs = [
                {
                    "code":   code,
                    "market": markets_list[i] if i < len(markets_list) else "KOSPI",
                }
                for i, code in enumerate(codes)
            ]
            snapshots = fetch_stock_list_snapshot(stock_cfgs)
            # 저녁 모드: 스냅샷 quote 를 evening_quotes 로 사용 (당일 종가 반영)
            evening_quotes = None
            if evening_mode:
                evening_quotes = {s["code"]: s.get("quote", {}) for s in snapshots}
            return build_stock_report(snapshots, evening_quotes=evening_quotes)
        except Exception as e:
            return {"error": f"종목별 분석 조회 실패: {e}"}

    if path == "/api/alert/quote":
        # 알림 모니터용 — 등록된 KRX 종목들의 현재가 + 등락률 일괄 조회
        # ?codes=005930,035720,...  (최대 20종목, 6자리 숫자 코드만)
        codes_raw = params.get("codes", "")
        codes = [c.strip() for c in codes_raw.split(",")
                 if c.strip().isdigit() and len(c.strip()) == 6]
        if not codes:
            return {"quotes": {}}
        quotes = {}
        for code in codes[:20]:
            data = fetch_naver(code)
            if data and data.get("price"):
                try:
                    price = float(data["price"])
                    prev  = float(data["prev_close"]) if data.get("prev_close") else price
                    chg   = round((price - prev) / prev * 100, 2) if prev else 0.0
                    quotes[code] = {"price": price, "change_pct": chg}
                except (ValueError, TypeError):
                    pass
        return {"quotes": quotes}

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
/* 메인 */
#main{flex:1;overflow-y:auto;background:#0d1117;padding:24px}

/* 로딩/빈 상태 */
.center-state{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:16px;text-align:center}
.center-state .icon{font-size:56px}
.center-state h2{font-size:22px;font-weight:700}
.center-state p{color:#8b949e;font-size:14px;line-height:1.6;max-width:380px}
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
.tab-btn{padding:7px 14px;border-radius:8px;border:none;background:none;color:#8b949e;font-size:13px;font-weight:500;cursor:pointer;transition:all .15s;position:relative}
.tab-btn.active{background:#1f6feb;color:#fff}
.tab-btn:not(.active):hover{background:#21262d;color:#e6edf3}
/* 탭 배지 (수급 데이터 있을 때 AI 탭에 표시) */
.tab-badge{position:absolute;top:3px;right:3px;width:7px;height:7px;border-radius:50%;background:#3fb950;display:none}
.tab-badge.visible{display:block}
/* Skeleton 로딩 */
.skel{background:linear-gradient(90deg,#21262d 25%,#2d333b 50%,#21262d 75%);background-size:200% 100%;animation:skel-shine 1.4s infinite}
@keyframes skel-shine{0%{background-position:200% 0}100%{background-position:-200% 0}}

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
.score-wrap{display:flex;align-items:flex-end;gap:8px;margin-bottom:6px}
.score-num{font-size:42px;font-weight:800;line-height:1}
.score-bar-bg{background:#21262d;border-radius:6px;height:8px;overflow:hidden}
.score-bar-fill{height:8px;border-radius:6px;transition:width .6s ease}

/* AI 진단 레이아웃 */
.ai-diagnosis-layout{display:flex;flex-direction:column;gap:14px;align-items:stretch}
.ai-top-grid{display:grid;grid-template-columns:minmax(240px,320px) minmax(0,1fr);gap:14px;align-items:stretch}
.ai-bottom-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;align-items:stretch}
.ai-score-card,.ai-patterns-card,.ai-report-card,.ai-flow-card{margin-bottom:0;height:100%}
.ai-flow-card{display:flex;flex-direction:column;gap:8px}
.ai-score-card .score-bar-bg{max-width:300px}
.ai-score-card #ai-score-desc{line-height:1.5}
#steps-list,#flow-sector-content{display:flex;flex-direction:column;gap:10px}
.flow-rationale-text{font-size:12px;color:#8b949e;text-align:center;line-height:1.5;word-break:keep-all;overflow-wrap:anywhere}
.empty-note{font-size:13px;color:#484f58;line-height:1.6;text-align:left}

/* 분석 스텝 */
.step-item{background:#21262d;border-radius:14px;padding:18px;margin-bottom:0;border:1px solid #30363d;display:flex;flex-direction:column;gap:12px}
.step-header{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap}
.step-title{font-size:13px;font-weight:600;line-height:1.6;flex:1;min-width:0;word-break:keep-all;overflow-wrap:anywhere}
.step-meta{display:flex;align-items:center;gap:6px;flex-shrink:0;white-space:nowrap}
.step-weight{font-size:10px;color:#8b949e;background:#161b22;padding:3px 8px;border-radius:999px;border:1px solid #30363d}
.step-score{font-size:12px;font-weight:700;padding:3px 10px;border-radius:999px;flex-shrink:0;align-self:center}
.step-score.pos{background:#0d2d1a;color:#3fb950}
.step-score.neg{background:#2d0d0d;color:#f85149}
.step-score.neu{background:#21262d;color:#8b949e}
.step-result{font-size:13px;color:#8b949e;line-height:1.8;display:flex;flex-direction:column;gap:8px;padding-top:12px;border-top:1px solid #30363d;text-align:left}
.step-result-line{position:relative;display:block;padding-left:14px;word-break:keep-all;overflow-wrap:anywhere}
.step-result-line::before{content:'•';position:absolute;left:0;top:0;color:#388bfd}

/* 패턴 */
.pattern-item{display:flex;flex-direction:column;justify-content:flex-start;align-items:flex-start;gap:8px;padding:14px 16px;border-radius:12px;margin-bottom:0;font-size:13px;width:100%;text-align:left}
.pattern-head{display:flex;align-items:flex-start;gap:8px;font-weight:700;line-height:1.5;word-break:keep-all;overflow-wrap:anywhere}
.pattern-icon{flex-shrink:0}
.pattern-desc{font-size:12px;color:#8b949e;line-height:1.7;word-break:keep-all;overflow-wrap:anywhere}
.pattern-bull{background:#0d2d1a;border:1px solid #1a4730}
.pattern-bear{background:#2d0d0d;border:1px solid #4d1515}
.pattern-neu{background:#21262d;border:1px solid #30363d}

/* 흐름 분석 보조 UI */
.flow-subtext{font-size:11px;color:#484f58;margin-top:4px;line-height:1.6;word-break:keep-all;overflow-wrap:anywhere}
.flow-detail-grid{display:grid;grid-template-columns:minmax(0,1fr) repeat(2,minmax(120px,180px));gap:12px;align-items:stretch}
.flow-detail-main,.flow-stat-card,.flow-chip{background:#21262d;border:1px solid #30363d;border-radius:12px}
.flow-detail-main{padding:14px 16px;display:flex;flex-direction:column;gap:8px}
.flow-detail-label,.flow-stat-label{font-size:11px;color:#8b949e}
.flow-range-meta{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font-size:11px;color:#484f58}
.flow-range-meta strong{color:#e6edf3}
.flow-stat-card{padding:14px 12px;text-align:center;display:flex;flex-direction:column;justify-content:center;gap:6px}
.flow-stat-value{font-size:16px;font-weight:700;line-height:1.4;word-break:keep-all;overflow-wrap:anywhere}
.flow-chip-row{display:flex;flex-wrap:wrap;gap:10px}
.flow-chip{padding:10px 14px;font-size:13px;line-height:1.6;word-break:keep-all;overflow-wrap:anywhere}
.flow-chip strong{color:#e6edf3}
.flow-helper-text{font-size:12px;color:#484f58;line-height:1.7;word-break:keep-all;overflow-wrap:anywhere}

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

@media(max-width:1100px){
  .ai-top-grid{grid-template-columns:1fr}
  .ai-bottom-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
  /* 1100px 이하: 7열 → 4열 */
  .sector-cards{grid-template-columns:repeat(4,minmax(0,1fr))}
}

/* ── 태블릿 (≤ 900px) ── */
@media(max-width:900px){
  .metrics-grid{grid-template-columns:repeat(2,1fr)}
  .risk-grid{grid-template-columns:1fr}
  .fund-grid{grid-template-columns:repeat(2,1fr)}
  .buy-price-grid{grid-template-columns:1fr}
  .indicator-grid{grid-template-columns:1fr}
  .ai-top-grid,.ai-bottom-grid{grid-template-columns:1fr}
  .flow-detail-grid{grid-template-columns:1fr}
  .two-col-grid{grid-template-columns:1fr}
  /* 900px 이하: 3열 */
  .sector-cards{grid-template-columns:repeat(3,minmax(0,1fr))}
}

/* ── 모바일 (≤ 768px) ── */
@media(max-width:768px){
  /* 레이아웃 재구성: flex → block, 스크롤 허용 */
  /* overscroll-behavior:none → 브라우저 네이티브 PTR(전체 새로고침) 차단 */
  body{display:block;height:auto;overflow-y:auto;overscroll-behavior-y:none}
  html{overscroll-behavior-y:none}
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
  .ai-top-grid,.ai-bottom-grid{grid-template-columns:1fr}
  .flow-detail-grid{grid-template-columns:1fr}

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
  /* 768px 이하: 2열 */
  .sector-cards{grid-template-columns:repeat(2,minmax(0,1fr))}
}

/* ── 소형 모바일 (≤ 480px) ── */
@media(max-width:480px){
  #main{padding:52px 10px 20px}
  .metrics-grid{grid-template-columns:1fr 1fr;gap:6px}
  /* 480px 이하: 2열 */
  .sector-cards{grid-template-columns:repeat(2,minmax(0,1fr))}
  .metric-card{padding:10px}
  .m-label{font-size:10px}
  .m-value{font-size:16px}
  .card{padding:12px;border-radius:10px}
  .tab-btn{font-size:11px;padding:6px 10px}
  .page-header h2{font-size:16px}
  .step-result{font-size:12px}
  .pattern-item{font-size:12px;padding:12px}
  .flow-chip{width:100%}
  .flow-range-meta{flex-direction:column;align-items:flex-start}
  .risk-card{padding:12px}
  .buy-card{padding:12px}
  .fund-grid{grid-template-columns:1fr 1fr}
}

/* ── Pull-to-Refresh 인디케이터 ── */
#ptr-indicator{
  position:fixed;top:0;left:0;right:0;z-index:600;
  display:flex;align-items:center;justify-content:center;
  height:0;overflow:hidden;
  background:#1f6feb;color:#fff;
  font-size:13px;font-weight:600;
  transition:height .2s ease,opacity .2s ease;
  will-change:height
}
.ptr-spinner{
  width:16px;height:16px;
  border:2px solid rgba(255,255,255,.35);
  border-top-color:#fff;
  border-radius:50%;
  animation:ptr-spin .7s linear infinite;
  margin-right:8px;flex-shrink:0
}
@keyframes ptr-spin{to{transform:rotate(360deg)}}

/* ── 홈 섹션 공통 헤더 ── */
.home-section{margin-bottom:20px}
.home-section-header{
  display:flex;justify-content:space-between;align-items:center;
  margin-bottom:10px;flex-wrap:wrap;gap:8px
}
.home-section-title{
  font-size:14px;font-weight:700;color:#e6edf3;
  display:flex;align-items:center;gap:8px
}
.home-section-refresh{
  background:none;border:1px solid #30363d;border-radius:6px;
  padding:3px 8px;color:#8b949e;font-size:11px;cursor:pointer;
  transition:border-color .15s,color .15s
}
.home-section-refresh:hover{border-color:#388bfd;color:#388bfd}

/* ── 📊 시장 현황 ── */
#market-core{margin-bottom:20px}
.core-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:8px}
.core-header-left{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.core-title{font-size:14px;font-weight:700;color:#e6edf3}
.mood-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:16px;font-size:11px;font-weight:600}
.mood-positive{background:#0d2d1a;color:#3fb950;border:1px solid #1a4730}
.mood-neutral{background:#21262d;color:#8b949e;border:1px solid #30363d}
.mood-negative{background:#2d0d0d;color:#f85149;border:1px solid #4d1515}
.core-indices{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:0}
.core-index-card{
  background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px 12px;
  transition:border-color .15s
}
.core-index-card:hover{border-color:#388bfd}
.ci-name{font-size:11px;color:#8b949e;margin-bottom:3px;font-weight:500}
.ci-val{font-size:17px;font-weight:700}
.ci-chg{font-size:12px;font-weight:600;margin-top:2px}

/* ── 📰 주요 뉴스 ── */
#market-news{margin-bottom:20px}
.core-news-item{
  background:#161b22;border:1px solid #30363d;border-radius:8px;
  padding:9px 12px;display:flex;gap:10px;align-items:flex-start;margin-bottom:5px;
  transition:border-color .12s
}
.core-news-item:hover{border-color:#30363d80}
.core-news-item:last-child{margin-bottom:0}
.cn-impact{font-size:10px;font-weight:700;padding:2px 6px;border-radius:8px;white-space:nowrap;flex-shrink:0;margin-top:2px}
.cn-positive{background:#0d2d1a;color:#3fb950}
.cn-negative{background:#2d0d0d;color:#f85149}
.cn-neutral{background:#21262d;color:#8b949e}
.cn-title{font-size:13px;color:#cdd9e5;text-decoration:none;line-height:1.45}
.cn-title:hover{color:#388bfd}
.cn-meta{font-size:10px;color:#484f58;margin-top:2px}
.core-loading{text-align:center;padding:24px;color:#8b949e;font-size:13px}
.news-loading{text-align:center;padding:16px;color:#484f58;font-size:12px}

/* ── 🌊 흐름 분석 탭 ── */
.signal-matrix{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:8px}
.sig-cell{background:#21262d;border-radius:10px;padding:8px 10px;text-align:center}
.sig-cell-label{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
.sig-cell-val{font-size:13px;font-weight:700}
.sig-up{color:#f85149}.sig-down{color:#388bfd}.sig-neutral{color:#8b949e}
.rec-badge-lg{display:inline-block;padding:5px 16px;border-radius:20px;font-size:13px;font-weight:700;margin-bottom:4px}
.rec-strong-buy{background:#0d2d1a;color:#3fb950;border:1px solid #1a4730}
.rec-buy{background:#0d2020;color:#238636;border:1px solid #155724}
.rec-hold{background:#2d2200;color:#d29922;border:1px solid #4a3800}
.rec-sell{background:#2d1515;color:#f85149;border:1px solid #4d1515}
.rec-strong-sell{background:#2d0d0d;color:#f85149;border:2px solid #f85149}
.flow-pos-bar-bg{background:#21262d;border-radius:6px;height:8px;overflow:hidden;margin-top:6px}
.flow-pos-bar-fill{height:8px;border-radius:6px;background:#1f6feb;transition:width .6s ease}

/* ── 🏭 섹터 흐름 (메인 페이지) ── */
#sector-flow{margin-top:16px}
.sector-flow-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.sector-flow-title{font-size:14px;font-weight:700;color:#e6edf3}

/* 7열 고정 그리드 — 14종목 → 7×2 레이아웃 */
.sector-cards{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px}

/* 카드 기본 스타일 */
.sector-card{
  background:#161b22;border:1px solid #30363d;border-radius:12px;padding:12px;
  cursor:pointer;user-select:none;
  transition:border-color .15s,background .15s
}
.sector-card:hover{border-color:#388bfd;background:#1a2233}
.sector-card.expanded{border-color:#388bfd;background:#161e2e}

/* 카드 내부 요소 */
.sector-card-head{display:flex;align-items:center;gap:6px;margin-bottom:6px}
.sector-card-emoji{font-size:16px;flex-shrink:0}
.sector-card-name{font-size:12px;font-weight:600;color:#e6edf3;line-height:1.3}
.sector-card-cnt{font-size:10px;color:#484f58;margin-top:1px;transition:color .15s}
.sector-card:hover .sector-card-cnt,.sector-card.expanded .sector-card-cnt{color:#388bfd}
.sector-card-pct{font-size:13px;font-weight:700;margin-bottom:5px}
.sector-card-mood{font-size:11px;padding:2px 7px;border-radius:8px;display:inline-block;font-weight:500}
.sector-mood-pos{background:#0d2d1a;color:#3fb950}
.sector-mood-neg{background:#2d0d0d;color:#f85149}
.sector-mood-neu{background:#21262d;color:#8b949e}

/* 토글 종목 리스트 — 기본 숨김, expanded 시 노출 */
.sector-stock-list{
  display:none;flex-wrap:wrap;gap:4px;
  margin-top:8px;padding-top:8px;
  border-top:1px solid #21262d
}
.sector-card.expanded .sector-stock-list{display:flex}
.sector-stock-tag{
  font-size:10px;padding:2px 7px;border-radius:6px;
  background:#21262d;color:#8b949e;
  cursor:pointer;transition:background .12s,color .12s
}
.sector-stock-tag:hover{background:#1f6feb;color:#fff}

/* ── 단계별 리포트 내 캔들 패턴 카드 ── */
.step-patterns{display:flex;flex-direction:column;gap:6px;margin-top:10px;padding-top:10px;border-top:1px solid #21262d}

/* ── 투자자 수급 카드 ── */
.investor-main-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px}
.investor-main-item{
  background:#161b22;border:1px solid #30363d;border-radius:10px;
  padding:8px 10px;text-align:center
}
.investor-sub-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
.investor-sub-item{
  background:#0d1117;border:1px solid #21262d;border-radius:8px;
  padding:6px 4px;text-align:center
}
.investor-label{font-size:10px;color:#8b949e;margin-bottom:3px}
.investor-val{font-size:13px;font-weight:700}
.investor-val-lg{font-size:16px;font-weight:800}
.investor-pos{color:#3fb950}
.investor-neg{color:#f85149}
.investor-neu{color:#8b949e}
.investor-ratio{font-size:11px;color:#8b949e;margin-top:3px}
@media(max-width:900px){
  .investor-main-grid{grid-template-columns:repeat(3,1fr)}
  .investor-sub-grid{grid-template-columns:repeat(4,1fr)}
}
@media(max-width:600px){
  .investor-main-grid{grid-template-columns:repeat(3,1fr)}
  .investor-sub-grid{grid-template-columns:repeat(2,1fr)}
}

/* ── 🌙 저녁 검증 모드 ── */
.ev-result-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px}
.ev-card{border-radius:12px;padding:16px;border:1px solid transparent}
.ev-hit{background:#0d2d1a;border-color:#1a4730}
.ev-partial{background:#2d2200;border-color:#4a3800}
.ev-miss{background:#2d0d0d;border-color:#4d1515}
.ev-neutral{background:#21262d;border-color:#30363d}
.ev-label{font-size:11px;color:#8b949e;margin-bottom:6px}
.ev-val{font-size:20px;font-weight:700}
.ev-note{font-size:12px;color:#8b949e;margin-top:4px;line-height:1.4}
.ev-outcome-badge{font-size:12px;font-weight:600;padding:4px 12px;border-radius:12px}
.ev-hit-badge{background:#0d2d1a;color:#3fb950}
.ev-partial-badge{background:#2d2200;color:#d29922}
.ev-miss-badge{background:#2d0d0d;color:#f85149}
.ev-na-badge{background:#21262d;color:#8b949e}
.ev-sig-row{display:flex;justify-content:space-between;align-items:center;padding:10px 12px;background:#21262d;border-radius:8px;margin-bottom:6px}
.ev-sig-row:last-child{margin-bottom:0}

@media(max-width:900px){
  .core-indices{grid-template-columns:repeat(3,1fr)}
  .signal-matrix{grid-template-columns:1fr}
  .ev-result-grid{grid-template-columns:1fr}
}
@media(max-width:600px){
  .core-indices{grid-template-columns:repeat(3,1fr)}
  .home-section-title{font-size:13px}
}
@media(max-width:400px){.core-indices{grid-template-columns:repeat(2,1fr)}}

/* ── 🔔 알림 시스템 ── */
.alert-bell-btn{display:flex;align-items:center;gap:6px;width:100%;text-align:left;padding:9px 12px;background:#21262d;border:1px solid #30363d;border-radius:8px;color:#8b949e;font-size:13px;cursor:pointer;transition:all .15s;position:relative;margin-top:6px}
.alert-bell-btn:hover{border-color:#388bfd;color:#e6edf3}
.alert-bell-count{position:absolute;top:-5px;right:6px;min-width:16px;height:16px;border-radius:8px;background:#f85149;color:#fff;font-size:10px;font-weight:700;display:none;align-items:center;justify-content:center;padding:0 4px;line-height:1}
.alert-bell-count.visible{display:flex}
/* 모달 */
.alert-modal-overlay{position:fixed;inset:0;z-index:50;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.65);padding:16px;opacity:0;pointer-events:none;transition:opacity .2s}
.alert-modal-overlay.open{opacity:1;pointer-events:auto}
.alert-modal{width:100%;max-width:360px;background:#161b22;border:1px solid #30363d;border-radius:16px;padding:20px;display:flex;flex-direction:column;gap:14px}
.alert-modal-header{display:flex;justify-content:space-between;align-items:flex-start}
.alert-modal-title{font-size:15px;font-weight:700}
.alert-modal-close{background:none;border:none;color:#8b949e;font-size:18px;cursor:pointer;padding:2px 6px;border-radius:6px;line-height:1}
.alert-modal-close:hover{background:#21262d;color:#e6edf3}
.alert-field-label{font-size:11px;color:#8b949e;font-weight:600;margin-bottom:6px;display:block}
.alert-preview-dir{font-size:11px;font-weight:600}
.alert-preview-up{color:#3fb950}
.alert-preview-dn{color:#f85149}
.alert-price-input{width:100%;background:#21262d;border:1px solid #30363d;border-radius:8px;padding:8px 12px;color:#e6edf3;font-size:13px;outline:none;transition:border-color .15s}
.alert-price-input:focus{border-color:#8b949e}
.alert-toggle-row{display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none;list-style:none}
.alert-toggle-row input[type=checkbox]{width:16px;height:16px;cursor:pointer;accent-color:#3fb950;flex-shrink:0}
.alert-pct-row{display:flex;align-items:center;gap:8px;margin-left:24px;margin-top:6px}
.alert-pct-input{width:70px;background:#21262d;border:1px solid #30363d;border-radius:8px;padding:6px 10px;color:#e6edf3;font-size:13px;outline:none}
.alert-pct-input:focus{border-color:#8b949e}
.alert-btn-row{display:flex;gap:8px;padding-top:2px}
.alert-btn-save{flex:1;background:#e6edf3;color:#0d1117;border:none;border-radius:8px;padding:9px;font-size:13px;font-weight:700;cursor:pointer;transition:opacity .15s}
.alert-btn-save:hover{opacity:.85}
.alert-btn-del{padding:8px 14px;border-radius:8px;border:1px solid #f85149;color:#f85149;background:none;font-size:13px;cursor:pointer;transition:background .15s}
.alert-btn-del:hover{background:rgba(248,81,73,.1)}
.alert-btn-cancel{padding:8px 14px;border-radius:8px;border:1px solid #30363d;color:#8b949e;background:none;font-size:13px;cursor:pointer;transition:background .15s}
.alert-btn-cancel:hover{background:#21262d}
/* 결과 페이지 알림 버튼 */
.alert-result-btn{display:none;align-items:center;gap:5px;padding:5px 11px;background:#21262d;border:1px solid #30363d;border-radius:8px;color:#8b949e;font-size:12px;cursor:pointer;transition:all .15s;white-space:nowrap}
.alert-result-btn:hover{border-color:#388bfd;color:#e6edf3}
.alert-result-btn.has-alert{border-color:#3fb950;color:#3fb950}
/* 알림 시트 */
.alert-sheet-backdrop{position:fixed;inset:0;z-index:40;background:rgba(0,0,0,.6);opacity:0;pointer-events:none;transition:opacity .25s}
.alert-sheet-backdrop.open{opacity:1;pointer-events:auto}
.alert-sheet{position:fixed;bottom:0;left:50%;z-index:45;width:100%;max-width:880px;max-height:44vh;display:flex;flex-direction:column;background:#161b22;border:1px solid #30363d;border-bottom:none;border-radius:16px 16px 0 0;box-shadow:0 -8px 32px rgba(0,0,0,.5);transform:translateX(-50%) translateY(100%);transition:transform .3s cubic-bezier(.4,0,.2,1)}
.alert-sheet.open{transform:translateX(-50%) translateY(0)}
.alert-sheet-handle{display:flex;justify-content:center;padding-top:10px;flex-shrink:0}
.alert-sheet-drag{width:40px;height:4px;border-radius:2px;background:#30363d}
.alert-sheet-header{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;border-bottom:1px solid #30363d;flex-shrink:0}
.alert-sheet-tabs{display:flex;gap:4px}
.alert-sheet-tab{padding:5px 12px;border-radius:6px;border:none;font-size:13px;font-weight:500;cursor:pointer;background:none;color:#8b949e;transition:all .15s;display:flex;align-items:center;gap:6px}
.alert-sheet-tab.active{background:#21262d;color:#e6edf3}
.alert-sheet-close{background:none;border:none;color:#8b949e;font-size:12px;cursor:pointer;padding:4px 8px;border-radius:6px}
.alert-sheet-close:hover{background:#21262d;color:#e6edf3}
.alert-sheet-body{flex:1;min-height:0;overflow-y:auto;padding:12px 16px}
.alert-tab-badge{font-size:10px;padding:1px 5px;border-radius:8px;font-weight:700;line-height:1.4}
.alert-fired-badge{background:#f85149;color:#fff}
.alert-config-badge{background:#30363d;color:#8b949e}
.alert-noti-item{display:flex;align-items:center;gap:10px;border-radius:8px;border:1px solid transparent;padding:10px 12px;margin-bottom:6px}
.alert-noti-up{background:rgba(63,185,80,.06);border-color:rgba(63,185,80,.25)}
.alert-noti-dn{background:rgba(248,81,73,.06);border-color:rgba(248,81,73,.25)}
.alert-noti-main{flex:1;min-width:0}
.alert-noti-symbol{font-size:13px;font-weight:700}
.alert-noti-label{font-size:12px;font-weight:600;margin-top:2px}
.alert-noti-up .alert-noti-label{color:#3fb950}
.alert-noti-dn .alert-noti-label{color:#f85149}
.alert-noti-time{font-size:11px;color:#8b949e;text-align:right;line-height:1.6;flex-shrink:0}
.alert-noti-dismiss{background:none;border:none;color:#484f58;font-size:14px;cursor:pointer;padding:2px 4px;border-radius:4px;flex-shrink:0}
.alert-noti-dismiss:hover{color:#e6edf3;background:#21262d}
.alert-cfg-item{display:flex;align-items:center;gap:10px;border-radius:8px;background:#21262d;border:1px solid #30363d;padding:10px 12px;margin-bottom:6px}
.alert-cfg-code{font-size:13px;font-weight:700;width:76px;flex-shrink:0}
.alert-cfg-tags{display:flex;flex-wrap:wrap;gap:5px;flex:1}
.alert-cfg-tag{font-size:11px;padding:2px 8px;border-radius:6px}
.alert-cfg-tag-target{background:#21262d;border:1px solid #30363d;color:#e6edf3}
.alert-cfg-tag-surge{background:rgba(63,185,80,.1);color:#3fb950}
.alert-cfg-tag-plunge{background:rgba(248,81,73,.1);color:#f85149}
.alert-cfg-del{background:none;border:none;color:#484f58;font-size:14px;cursor:pointer;padding:2px 4px;border-radius:4px;flex-shrink:0}
.alert-cfg-del:hover{color:#f85149;background:rgba(248,81,73,.1)}
.alert-sheet-empty{text-align:center;padding:28px 0;color:#484f58;font-size:13px;line-height:1.8}
/* 토스트 */
#alert-toast-container{position:fixed;top:14px;right:14px;z-index:60;display:flex;flex-direction:column;gap:8px;pointer-events:none}
.alert-toast{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:12px 14px;min-width:210px;max-width:280px;box-shadow:0 8px 24px rgba(0,0,0,.4);display:flex;gap:10px;align-items:flex-start;opacity:0;transform:translateX(40px);transition:opacity .3s,transform .3s;pointer-events:auto}
.alert-toast.visible{opacity:1;transform:translateX(0)}
.alert-toast-icon{font-size:18px;flex-shrink:0;line-height:1.3}
.alert-toast-body{flex:1;min-width:0}
.alert-toast-title{font-size:13px;font-weight:700}
.alert-toast-desc{font-size:11px;color:#8b949e;margin-top:2px;line-height:1.4}
.alert-toast-close{background:none;border:none;color:#484f58;font-size:14px;cursor:pointer;padding:0;flex-shrink:0;line-height:1}
.alert-toast-close:hover{color:#e6edf3}
</style>
</head>
<body>

<!-- ── Pull-to-Refresh 인디케이터 ── -->
<div id="ptr-indicator"><span id="ptr-text">↓ 당겨서 새로고침</span></div>

<!-- ── 모바일: 햄버거 + 오버레이 ── -->
<button id="hamburger" onclick="toggleSidebar()" aria-label="메뉴 열기">☰</button>
<div id="mob-overlay" onclick="closeSidebar()"></div>

<!-- ── 사이드바 ── -->
<div id="sidebar">
  <div class="sb-header">
    <h1>📈 StockOracle</h1>
    <p>AI 기반 기술적 분석 · 투자자 수급</p>
  </div>

  <button onclick="setState('empty');closeSidebar()" style="width:100%;text-align:left;padding:9px 12px;margin-bottom:8px;background:#21262d;border:1px solid #30363d;border-radius:8px;color:#8b949e;font-size:13px;cursor:pointer">🏠 메인 홈페이지로 이동</button>

  <div class="sb-section">
    <span class="sb-label">메뉴</span>
    <div style="display:flex;flex-direction:column;gap:4px">
      <button class="mkt-btn active" style="text-align:left;padding:10px 12px" id="nav-analysis" onclick="showPage('analysis')">🔍 종목 상세 분석</button>
      <button class="mkt-btn" style="text-align:left;padding:10px 12px" id="nav-screener" onclick="showPage('screener')">📋 주식 골라보기</button>
    </div>
    <button class="alert-bell-btn" onclick="openAlertsSheet()" aria-label="알림 관리">
      🔔 알림 관리
      <span class="alert-bell-count" id="alert-bell-count"></span>
    </button>
  </div>


  <div class="sb-section" id="analysis-controls">
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

</div>

<!-- ── 메인 ── -->
<div id="main">
  <!-- 분석 페이지 -->
  <div id="page-analysis">
    <div id="state-empty">

      <!-- 1. 📊 시장 현황 (지수 + 시장 무드) -->
      <div id="market-core" class="home-section">
        <div class="core-loading" id="core-loading">
          <div class="spinner" style="margin:0 auto 10px"></div>
          시장 현황 로딩 중...
        </div>
        <div id="core-content" style="display:none">
          <div class="core-header">
            <div class="core-header-left">
              <span class="core-title">📊 시장 현황</span>
              <span id="core-mood-badge" class="mood-badge mood-neutral">—</span>
              <span id="core-vix-badge" style="display:none;font-size:10px;padding:2px 7px;border-radius:10px;background:#2d0d0d;color:#f85149;border:1px solid #4d1515"></span>
            </div>
            <button onclick="loadMarketCore()" class="home-section-refresh" title="새로고침">🔄 새로고침</button>
          </div>
          <div class="core-indices" id="core-indices"></div>
        </div>
        <div id="core-error" style="display:none;text-align:center;padding:20px;color:#484f58;font-size:13px">
          시장 데이터를 불러오지 못했습니다
          <button onclick="loadMarketCore()" class="home-section-refresh" style="margin-left:8px">재시도</button>
        </div>
      </div>

      <!-- 2. 🏭 섹터 흐름 -->
      <div id="sector-flow" class="home-section">
        <div id="sector-flow-loading" style="text-align:center;padding:14px;color:#484f58;font-size:12px">
          <div class="spinner" style="margin:0 auto 8px;width:22px;height:22px;border-width:3px"></div>
          섹터 데이터 로딩 중...
        </div>
        <div id="sector-flow-content" style="display:none">
          <div class="sector-flow-header">
            <span class="sector-flow-title">🏭 섹터 흐름</span>
            <button onclick="loadSectorFlow()" class="home-section-refresh" title="새로고침">🔄 새로고침</button>
          </div>
          <div class="sector-cards" id="sector-cards"></div>
        </div>
        <div id="sector-flow-error" style="display:none;text-align:center;padding:12px;color:#484f58;font-size:12px">
          섹터 데이터를 불러오지 못했습니다
          <button onclick="loadSectorFlow()" class="home-section-refresh" style="margin-left:6px">재시도</button>
        </div>
      </div>

      <!-- 3. 📰 주요 뉴스 (배경·원인) -->
      <div id="market-news" class="home-section" style="display:none">
        <div class="home-section-header">
          <span class="home-section-title">📰 주요 뉴스</span>
          <span style="font-size:11px;color:#484f58">시장에 영향을 주는 오늘의 이슈</span>
        </div>
        <div id="core-news"></div>
      </div>

    </div>
    <div id="state-loading" class="center-state" style="display:none">
      <div class="spinner"></div>
      <p style="color:#8b949e" id="loading-msg">데이터 수집 중...<br><span style="font-size:12px;color:#484f58">가격 · 기술 지표 · 투자자 수급을 분석하고 있습니다</span></p>
    </div>
    <div id="state-error" class="center-state" style="display:none">
      <div class="icon">⚠️</div>
      <h2 style="color:#f85149">분석 오류</h2>
      <p id="error-msg" style="color:#8b949e"></p>
    </div>
    <div id="state-result" style="display:none">
      <div class="page-header">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:4px">
          <h2 id="r-title"></h2>
          <button id="result-alert-btn" class="alert-result-btn" onclick="openAlertModal(_currentAlertSymbol, _currentAlertPrice)">🔔 알림 설정</button>
        </div>
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
      <div id="r-us-fund" style="display:none" class="card">
        <div class="card-title">🏢 기업 펀더멘털 (Alpha Vantage)</div>
        <div class="fund-grid">
          <div class="fund-item"><div class="fund-label">섹터</div><div class="fund-val" id="f-us-sector" style="font-size:12px"></div></div>
          <div class="fund-item"><div class="fund-label">PER</div><div class="fund-val" id="f-us-per"></div></div>
          <div class="fund-item"><div class="fund-label">PBR</div><div class="fund-val" id="f-us-pbr"></div></div>
          <div class="fund-item"><div class="fund-label">EPS</div><div class="fund-val" id="f-us-eps"></div></div>
          <div class="fund-item"><div class="fund-label">베타</div><div class="fund-val" id="f-us-beta"></div></div>
          <div class="fund-item"><div class="fund-label">시총</div><div class="fund-val" id="f-us-mktcap"></div></div>
        </div>
        <div id="f-us-sentiment" style="display:none;margin-top:10px;font-size:12px;color:#8b949e"></div>
      </div>
      <div class="tabs" id="result-tabs">
        <button class="tab-btn active" onclick="switchTab('chart')">📊 차트</button>
        <button class="tab-btn" onclick="switchTab('ai')" id="tab-ai-btn">🧠 AI 진단<span class="tab-badge" id="investor-badge" title="투자자 수급 데이터 있음"></span></button>
        <button class="tab-btn" onclick="switchTab('forecast')">🔮 예측</button>
        <button class="tab-btn" onclick="switchTab('news')">📰 뉴스</button>
        <button class="tab-btn" id="tab-evening-btn" onclick="switchTab('evening')" style="display:none">📋 KRX</button>
      </div>

      <!-- 차트 탭 -->
      <div id="tab-chart">
        <div class="card">
          <div class="card-title">📡 기술적 지표 종합 시그널</div>
          <div id="indicator-signals-section"></div>
        </div>
        <div class="card">
          <div class="card-title">📐 피봇 포인트 분석</div>
          <div id="pivot-points-section"></div>
        </div>
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
      </div>

      <!-- AI 탭 -->
      <div id="tab-ai" style="display:none">
        <div class="ai-diagnosis-layout">
          <!-- 1행: 종합 점수 + 3-신호 매트릭스 -->
          <div class="ai-top-grid">
            <div class="card ai-score-card">
              <div class="card-title" style="margin-bottom:10px">🏆 종합 기술적 점수</div>
              <div class="score-wrap">
                <div class="score-num" id="ai-score"></div>
                <span style="color:#8b949e;font-size:15px;margin-bottom:4px">/ 100점</span>
              </div>
              <div class="score-bar-bg"><div class="score-bar-fill" id="ai-score-bar"></div></div>
              <p id="ai-score-desc" style="font-size:12px;color:#8b949e;margin-top:6px"></p>
            </div>
            <div class="card ai-flow-card">
              <div class="card-title" style="margin-bottom:8px">📡 3-신호 분석 매트릭스</div>
              <div class="signal-matrix" id="flow-matrix"></div>
              <div style="margin:4px 0">
                <span id="flow-rec-badge" class="rec-badge-lg rec-hold">분석 중...</span>
              </div>
              <div class="flow-rationale-text" id="flow-rationale"></div>
            </div>
          </div>
          <!-- 2행: 투자자 수급 (KRX 전용, JS가 표시/숨김 제어) -->
          <div class="card" id="investor-flow-card" style="display:none">
            <div class="card-title" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
              <span>💰 투자자 수급 <span style="font-size:10px;color:#484f58;font-weight:400">· 토스증권 기준</span></span>
              <button id="investor-flow-retry" onclick="retryInvestorFlow()" style="display:none;background:none;border:1px solid #30363d;border-radius:6px;padding:3px 8px;color:#8b949e;font-size:11px;cursor:pointer">🔄 재시도</button>
            </div>
            <!-- 로딩 skeleton -->
            <div id="investor-flow-skeleton" style="display:none">
              <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px">
                <div class="skel" style="height:56px;border-radius:10px"></div>
                <div class="skel" style="height:56px;border-radius:10px"></div>
                <div class="skel" style="height:56px;border-radius:10px"></div>
              </div>
              <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px">
                <div class="skel" style="height:42px;border-radius:8px"></div>
                <div class="skel" style="height:42px;border-radius:8px"></div>
                <div class="skel" style="height:42px;border-radius:8px"></div>
                <div class="skel" style="height:42px;border-radius:8px"></div>
              </div>
            </div>
            <div id="investor-flow-content"></div>
          </div>
          <!-- 3행: 단계별 분석 리포트 (전체 너비) -->
          <div class="card ai-report-card">
            <div class="card-title">📝 단계별 분석 리포트</div>
            <div id="steps-list"></div>
          </div>
          <!-- 4행: 섹터 / 업종 정보 (캔들·52주 섹션 제거 후 단독 카드) -->
          <div class="card ai-flow-card" id="flow-sector-card" style="display:none">
            <div class="card-title">🏭 섹터 / 업종 정보</div>
            <div id="flow-sector-content"></div>
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
          <div class="card-title">📈 향후 주가 상승 가능 범위 (목표가 예측)</div>
          <div id="target-price-section"></div>
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

      <!-- 📋 KRX 전용 탭 -->
      <div id="tab-evening" style="display:none">
        <div id="evening-loading" style="text-align:center;padding:32px;color:#8b949e;display:none">
          <div class="spinner" style="margin:0 auto 10px"></div>
          KRX 검증 데이터 로딩 중...
        </div>
        <div id="evening-content" style="display:none">
          <div class="card">
            <div class="card-title">⚖️ 예측 vs 실제 비교</div>
            <div class="ev-result-grid">
              <div class="ev-card ev-neutral">
                <div class="ev-label">🌅 아침 예측</div>
                <div class="ev-val" id="ev-pred-rec">—</div>
                <div class="ev-note" id="ev-pred-conf"></div>
              </div>
              <div class="ev-card ev-neutral" id="ev-actual-card">
                <div class="ev-label">📊 실제 종가 등락</div>
                <div class="ev-val" id="ev-actual-pct">—</div>
                <div class="ev-note" id="ev-actual-dir"></div>
              </div>
              <div class="ev-card ev-neutral" id="ev-verdict-card">
                <div class="ev-label">🎯 판정</div>
                <div style="margin-bottom:6px"><span id="ev-verdict-badge" class="ev-outcome-badge ev-na-badge">대기</span></div>
                <div class="ev-note" id="ev-verdict-note"></div>
              </div>
            </div>
            <div style="font-size:12px;color:#484f58;border-top:1px solid #21262d;padding-top:10px">
              💡 장 마감(15:30) 이후 실제 종가 데이터로 아침 예측을 자동 검증합니다.
            </div>
          </div>
          <div class="card">
            <div class="card-title">🔍 신호 기여 분석</div>
            <div id="ev-signal-breakdown"></div>
          </div>
        </div>
        <div id="evening-error" style="display:none;text-align:center;padding:24px;color:#484f58;font-size:13px">
          KRX 검증 데이터 조회 실패 — 장 마감(15:30) 이후 사용 가능합니다
        </div>
        <div id="evening-guide" style="text-align:center;padding:32px;color:#8b949e;font-size:13px">
          <div style="font-size:32px;margin-bottom:12px">📋</div>
          <div style="font-weight:600;margin-bottom:8px">KRX 종목 전용 분석</div>
          <div style="line-height:1.6">KRX 종목 검색 시 자동으로 예측 검증 데이터를 불러옵니다.<br>
            <span style="font-size:11px;color:#484f58">장 마감(15:30) 이후 실제 종가 데이터가 반영됩니다</span></div>
        </div>
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
function quickSearch(name) {
  document.getElementById('ticker-input').value = name;
  showPage('analysis');  // 내부에서 closeSidebar() 호출됨
  analyze();
}

// ── 분석 ──
// 로딩 메시지 단계별 표시
const _LOADING_MSGS = [
  ['데이터 수집 중...', '가격 · 기술 지표 · 투자자 수급을 분석하고 있습니다'],
  ['지표 계산 중...', 'EMA · RSI · MACD · 볼린저밴드를 산출하고 있습니다'],
  ['AI 종합 진단 중...', '패턴 탐지 · 점수 산출 · 전략 생성 중입니다'],
  ['거의 완료됩니다...', '결과 화면을 준비하고 있습니다'],
];
let _loadingTimer = null;

function _startLoadingAnimation() {
  let idx = 0;
  const el = document.getElementById('loading-msg');
  if (el) {
    el.innerHTML = `${_LOADING_MSGS[0][0]}<br><span style="font-size:12px;color:#484f58">${_LOADING_MSGS[0][1]}</span>`;
  }
  _loadingTimer = setInterval(() => {
    idx = (idx + 1) % _LOADING_MSGS.length;
    if (el) el.innerHTML = `${_LOADING_MSGS[idx][0]}<br><span style="font-size:12px;color:#484f58">${_LOADING_MSGS[idx][1]}</span>`;
  }, 3000);
}

function _stopLoadingAnimation() {
  if (_loadingTimer) { clearInterval(_loadingTimer); _loadingTimer = null; }
}

async function analyze() {
  closeSidebar();   // 모바일에서 분석 시작 시 사이드바 자동 닫기
  const ticker = document.getElementById('ticker-input').value.trim();
  const period = document.getElementById('period-select').value;
  if (!ticker) return;
  // 배지 초기화
  const badge = document.getElementById('investor-badge');
  if (badge) badge.classList.remove('visible');
  setState('loading');
  _startLoadingAnimation();
  document.getElementById('analyze-btn').disabled = true;
  destroyCharts();
  try {
    const r = await fetch(`/api/stock?ticker=${encodeURIComponent(ticker)}&period=${period}&market=${currentMarket}`);
    let text = await r.text();
    let d;
    try {
      d = JSON.parse(text);
    } catch(e) {
      throw new Error(`API 응답이 올바르지 않습니다. (상태: ${r.status}, 서버 오류나 타임아웃일 수 있습니다.)`);
    }
    if (d.error) { setState('error'); document.getElementById('error-msg').textContent = d.error; return; }
    currentData = d;
    if (d.market) {
      currentMarket = d.market;
      document.getElementById('ticker-input').placeholder = d.market === 'KRX' ? '예: 삼성전자, 005930' : '예: 애플, TSLA, NVDA';
    }
    renderResult(d);
    setState('result');
    // 흐름 분석: AI 탭에 통합, 항상 즉시 렌더
    renderFlowTab(d);
    // KRX 전용 탭: KRX 종목만 표시 + 자동 데이터 로드
    const krxCode = extractKrxCode(d.symbol);
    const eveningTabBtn = document.getElementById('tab-evening-btn');
    if (d.market === 'KRX' && krxCode) {
      eveningTabBtn.style.display = '';
      resetEveningTab();
      loadKrxVerification(krxCode);  // 자동 로드
    } else {
      eveningTabBtn.style.display = 'none';
      resetEveningTab();
    }
  } catch(e) {
    setState('error');
    document.getElementById('error-msg').textContent = 'API 서버 오류: ' + e.message;
  } finally {
    _stopLoadingAnimation();
    document.getElementById('analyze-btn').disabled = false;
  }
}

function setState(s) {
  // 'empty': block (오늘의 핵심 레이아웃) | 'loading','error': flex (중앙 정렬) | 'result': block
  const displayMap = { empty: 'block', loading: 'flex', error: 'flex', result: 'block' };
  ['empty','loading','error','result'].forEach(n => {
    const el = document.getElementById('state-' + n);
    if (el) el.style.display = n === s ? displayMap[n] : 'none';
  });
}

// ── 렌더링 ──
function fmt(v, isKrx) {
  if (v == null || isNaN(v)) return '-';
  return isKrx ? Number(v).toLocaleString('ko-KR',{maximumFractionDigits:0}) + '원'
               : '$' + Number(v).toLocaleString('en-US',{minimumFractionDigits:4,maximumFractionDigits:4});
}

function renderResult(d) {
  const isKrx = d.market === 'KRX';
  const up = d.pct_change >= 0;
  const clr = isKrx ? (up ? '#f85149' : '#388bfd') : (up ? '#3fb950' : '#f85149');

  // 🔔 알림 버튼 연동 (KRX 전용)
  _currentAlertSymbol = isKrx ? d.symbol : null;
  _currentAlertPrice  = isKrx ? d.last_close : null;
  const alertBtn = document.getElementById('result-alert-btn');
  if (alertBtn) {
    alertBtn.style.display = isKrx ? 'flex' : 'none';
    if (isKrx) _alertResultBtnUpdate(d.symbol);
  }

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
    document.getElementById('r-us-fund').style.display = 'none';
  } else if (!isKrx && d.us_enriched && d.us_enriched.overview && d.us_enriched.overview.sector) {
    document.getElementById('r-naver-fund').style.display = 'none';
    document.getElementById('r-us-fund').style.display = 'block';
    const ov = d.us_enriched.overview;
    document.getElementById('f-us-sector').textContent = ov.sector || '-';
    document.getElementById('f-us-per').textContent = ov.per || '-';
    document.getElementById('f-us-pbr').textContent = ov.pbr || '-';
    document.getElementById('f-us-eps').textContent = ov.eps ? '$' + ov.eps : '-';
    document.getElementById('f-us-beta').textContent = ov.beta ? Number(ov.beta).toFixed(2) : '-';
    const mc = ov.market_cap && Number(ov.market_cap) > 0
      ? '$' + (Number(ov.market_cap) / 1e9).toFixed(1) + 'B' : '-';
    document.getElementById('f-us-mktcap').textContent = mc;
    const sentEl = document.getElementById('f-us-sentiment');
    const sent = d.us_enriched.sentiment || {};
    if (sent.bullish_pct != null) {
      const bull = (Number(sent.bullish_pct) * 100).toFixed(0);
      const bear = (Number(sent.bearish_pct || 0) * 100).toFixed(0);
      sentEl.style.display = 'block';
      sentEl.innerHTML = `뉴스 감성: <span style="color:#3fb950">▲ ${bull}%</span> / <span style="color:#f85149">▼ ${bear}%</span><span style="color:#484f58;margin-left:8px">(${sent.article_count||0}건/주)</span>`;
    } else { sentEl.style.display = 'none'; }
  } else {
    document.getElementById('r-naver-fund').style.display = 'none';
    document.getElementById('r-us-fund').style.display = 'none';
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

  // 캔들 패턴 카드 (step-5 인라인 삽입용)
  const patterns = d.candlestick_patterns || [];
  const patternCardsHtml = patterns.length === 0
    ? '<p class="empty-note">특이한 캔들 패턴이 감지되지 않았습니다.</p>'
    : patterns.map(p => {
        const pcls = p.direction === '상승' ? 'pattern-bull' : p.direction === '하락' ? 'pattern-bear' : 'pattern-neu';
        const icon = p.direction === '상승' ? '📈' : p.direction === '하락' ? '📉' : '➖';
        return `<div class="pattern-item ${pcls}">
          <div class="pattern-head"><span class="pattern-icon">${icon}</span><span>${p.name}</span></div>
          <div class="pattern-desc">${p.desc}</div>
        </div>`;
      }).join('');

  const stepsList = document.getElementById('steps-list');
  stepsList.innerHTML = d.analysis_steps.map(st => {
    const sc = st.score;
    const cls = sc > 0 ? 'pos' : sc < 0 ? 'neg' : 'neu';
    const label = sc > 0 ? '+' + sc : sc;
    const weight = st.weight || '';
    // step-5(캔들 패턴 분석)이면 패턴 카드를 본문 아래에 삽입
    const isStep5 = st.step.startsWith('5.');
    const inlinePatterns = isStep5
      ? `<div class="step-patterns">${patternCardsHtml}</div>`
      : '';
    return `<div class="step-item">
      <div class="step-header">
        <span class="step-title">${st.step}</span>
        <div class="step-meta">
          ${weight ? `<span class="step-weight">${weight}</span>` : ''}
          <span class="step-score ${cls}">${label}점</span>
        </div>
      </div>
      ${isStep5 ? '' : `<div class="step-result">
        ${st.result.split(' | ').filter(l => l.trim()).map(line => `<span class="step-result-line">${line}</span>`).join('')}
      </div>`}
      ${inlinePatterns}
    </div>`;
  }).join('');

  renderInvestorFlow(d, isKrx);
}

// ── 투자자 수급 렌더 (KRX 전용) ────────────────────────────────────────────
function renderInvestorFlow(d, isKrx) {
  const card     = document.getElementById('investor-flow-card');
  const badge    = document.getElementById('investor-badge');
  const retryBtn = document.getElementById('investor-flow-retry');
  const skelEl   = document.getElementById('investor-flow-skeleton');
  const contEl   = document.getElementById('investor-flow-content');
  if (!card) return;

  // US 종목 → 카드 완전 숨김
  if (!isKrx) { card.style.display = 'none'; if (badge) badge.classList.remove('visible'); return; }

  // KRX 종목이면 항상 카드 표시
  card.style.display = '';
  if (skelEl) skelEl.style.display = 'none';
  if (retryBtn) retryBtn.style.display = 'none';

  const flow = d.investor_flow;

  // API 실패 상태
  if (!flow || !flow.ok) {
    const reason = (flow && flow.reason) ? flow.reason : '수급 데이터 조회 실패';
    if (contEl) contEl.innerHTML = `
      <div style="text-align:center;padding:20px 0;color:#484f58;font-size:13px">
        <div style="font-size:22px;margin-bottom:8px">📡</div>
        ${reason}
        <div style="font-size:11px;margin-top:6px;color:#30363d">장 개시 전이거나 API 일시 불가 상태입니다</div>
      </div>`;
    if (retryBtn) retryBtn.style.display = '';
    if (badge) badge.classList.remove('visible');
    return;
  }

  // 성공 — AI 탭 배지 활성화
  if (badge) badge.classList.add('visible');

  const fmtV = v => v === 0 ? '—' : (v > 0 ? '+' : '') + v.toLocaleString();
  const cls  = v => v > 0 ? 'investor-pos' : v < 0 ? 'investor-neg' : 'investor-neu';

  const mainItem = (label, val, extra = '') => `
    <div class="investor-main-item">
      <div class="investor-label">${label}</div>
      <div class="investor-val investor-val-lg ${cls(val)}">${fmtV(val)}</div>
      ${extra}
    </div>`;

  const subItem = (label, val) => `
    <div class="investor-sub-item">
      <div class="investor-label">${label}</div>
      <div class="investor-val ${cls(val)}">${fmtV(val)}</div>
    </div>`;

  const ratio = flow['외국인비율'] || 0;
  const ratioHtml = ratio > 0
    ? `<div class="investor-ratio">보유율 ${ratio.toFixed(2)}%</div>` : '';

  const dateStr = (flow.date || '')
    .replace(/(\d{4})(\d{2})(\d{2})/, '$1.$2.$3') || '—';

  if (contEl) contEl.innerHTML = `
    <div style="font-size:11px;color:#484f58;margin-bottom:12px">
      기준일: ${dateStr} &nbsp;·&nbsp; 단위: 주(株) &nbsp;·&nbsp; 순매수(+) / 순매도(−)
    </div>
    <div class="investor-main-grid">
      ${mainItem('외국인', flow['외국인'], ratioHtml)}
      ${mainItem('기관', flow['기관'])}
      ${mainItem('개인', flow['개인'])}
    </div>
    <div style="font-size:11px;color:#8b949e;font-weight:600;margin-bottom:8px">기관 세부 내역</div>
    <div class="investor-sub-grid">
      ${subItem('연기금',   flow['연기금'])}
      ${subItem('금융투자', flow['금융투자'])}
      ${subItem('투신',     flow['투신'])}
      ${subItem('사모',     flow['사모'])}
      ${subItem('보험',     flow['보험'])}
      ${subItem('은행',     flow['은행'])}
      ${subItem('기타금융', flow['기타금융'])}
      ${subItem('기타법인', flow['기타법인'])}
    </div>
  `;
}

// ── 투자자 수급 재시도 ────────────────────────────────────────────────────
async function retryInvestorFlow() {
  if (!currentData) return;
  const isKrx = currentData.market === 'KRX';
  const skelEl   = document.getElementById('investor-flow-skeleton');
  const contEl   = document.getElementById('investor-flow-content');
  const retryBtn = document.getElementById('investor-flow-retry');
  if (skelEl)   { skelEl.style.display = ''; }
  if (contEl)   { contEl.innerHTML = ''; }
  if (retryBtn) { retryBtn.style.display = 'none'; }

  try {
    const ticker = currentData.symbol;
    const r = await fetch(`/api/stock?ticker=${encodeURIComponent(ticker)}&period=1mo`);
    const nd = await r.json();
    if (nd && nd.investor_flow) {
      currentData.investor_flow = nd.investor_flow;
      renderInvestorFlow(currentData, isKrx);
    } else {
      throw new Error('데이터 없음');
    }
  } catch(e) {
    if (contEl) contEl.innerHTML = '<div style="text-align:center;padding:16px;color:#484f58;font-size:12px">재시도 실패 — 잠시 후 다시 시도해주세요</div>';
    if (retryBtn) retryBtn.style.display = '';
  } finally {
    if (skelEl) skelEl.style.display = 'none';
  }
}

function renderForecast(d, isKrx) {
  const risk = d.risk_scenarios;
  const bp   = d.buy_price;
  const tp   = d.target_price;
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

  // ── 목표가 예측 섹션 ──
  const tpEl = document.getElementById('target-price-section');
  if (tpEl) {
    if (!tp) {
      tpEl.innerHTML = '<p style="color:#484f58;font-size:13px">데이터 부족</p>';
    } else {
      const cur = d.last_close;
      tpEl.innerHTML = `
        <div style="background:#21262d;border-radius:10px;padding:16px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
          <div>
            <div style="font-size:12px;color:#8b949e;margin-bottom:6px">예상 목표가 범위 (${tp.period})</div>
            <div style="font-size:24px;font-weight:800;color:#3fb950">${fmt(tp.min_price, isKrx)} ~ ${fmt(tp.max_price, isKrx)}</div>
            <div style="font-size:13px;color:#8b949e;margin-top:4px">현재가 대비 예상 수익률: <span style="color:#3fb950">+${tp.min_return}% ~ +${tp.max_return}%</span></div>
          </div>
          <div style="text-align:right">
            <div style="font-size:12px;color:#8b949e;margin-bottom:4px">추세 강도 분석</div>
            <div style="font-size:16px;font-weight:700;color:${tp.trend_strength >= 2 ? '#3fb950' : tp.trend_strength > 0 ? '#d29922' : '#f85149'}">
              ${tp.trend_strength >= 2 ? '강세 돌파' : tp.trend_strength > 0 ? '완만한 상승' : '조정 / 하락'}
            </div>
          </div>
        </div>
        <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px">
          <div style="font-size:11px;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">📋 예측 근거</div>
          <div style="font-size:13px;color:#e6edf3;line-height:1.5;">${tp.reason}</div>
        </div>
      `;
    }
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

      const fmtPct = p => (p >= 0 ? `<span style="color:#3fb950">+${p}%</span>` : `<span style="color:#f85149">${p}%</span>`);
      const fib = bp.fib || {};

      const buyZone = (zone, color, label, icon) => {
        const z = bp[zone];
        if (!z) return '';
        const basisHtml = (z.basis || []).map(b => `<div style="font-size:11px;color:#8b949e;margin-bottom:3px">• ${b}</div>`).join('');
        return `
          <div class="buy-card ${zone}" style="display:flex;flex-direction:column;gap:6px">
            <div class="buy-label">${icon} ${label}</div>
            <div class="buy-price-val" style="color:${color};font-size:15px;font-weight:800">
              ${fmt(z.range[0], isKrx)} ~ ${fmt(z.range[1], isKrx)}
            </div>
            <div style="font-size:12px;color:#8b949e">
              현재가 대비 ${fmtPct(z.pct[0])} ~ ${fmtPct(z.pct[1])}
            </div>
            <div style="background:#0d1117;border-radius:7px;padding:8px;margin-top:2px">
              ${basisHtml}
            </div>
            <div style="font-size:11px;color:#cdd9e5;line-height:1.5;border-top:1px solid #21262d;padding-top:6px;margin-top:2px">
              💡 ${z.interpretation || ''}
            </div>
          </div>`;
      };

      bpEl.innerHTML = `
        <div style="background:#21262d;border-radius:10px;padding:14px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
          <div style="flex-shrink:0">
            <div style="font-size:11px;color:#8b949e;margin-bottom:4px">현재가</div>
            <div style="font-size:22px;font-weight:800">${fmt(cur, isKrx)}</div>
            <div style="font-size:11px;color:#8b949e;margin-top:4px">ATR ${bp.atr_pct}% · ${
              bp.vol_trend === 'expanding'   ? '<span style="color:#f85149">변동성 확대↑</span>' :
              bp.vol_trend === 'contracting' ? '<span style="color:#3fb950">변동성 수축↓</span>' :
              '<span style="color:#d29922">변동성 안정</span>'
            }</div>
          </div>
          <div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;min-width:0">
            <div style="min-width:0">
              <div style="font-size:11px;color:#8b949e;margin-bottom:4px">예상 매수 타이밍</div>
              <div style="font-size:13px;font-weight:600;color:#3fb950;word-break:keep-all;white-space:normal;line-height:1.4">${bp.timing.buy}</div>
            </div>
            <div style="min-width:0">
              <div style="font-size:11px;color:#8b949e;margin-bottom:4px">예상 매도 타이밍</div>
              <div style="font-size:13px;font-weight:600;color:#f85149;word-break:keep-all;white-space:normal;line-height:1.4">${bp.timing.sell}</div>
            </div>
          </div>
        </div>
        ${fib.f382 ? `
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px;margin-bottom:14px;display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:#8b949e">
          <span>📐 피보나치 기준 (60일)</span>
          <span>▲ 고점 <b style="color:#cdd9e5">${fmt(fib.h60, isKrx)}</b></span>
          <span>▼ 저점 <b style="color:#cdd9e5">${fmt(fib.l60, isKrx)}</b></span>
          <span>38.2% <b style="color:#f97316">${fmt(fib.f382, isKrx)}</b></span>
          <span>50.0% <b style="color:#d29922">${fmt(fib.f500, isKrx)}</b></span>
          <span>61.8% <b style="color:#388bfd">${fmt(fib.f618, isKrx)}</b></span>
        </div>` : ''}
        <div class="buy-price-grid">
          ${buyZone('aggressive',  '#f97316', '공격적 매수', '⚡')}
          ${buyZone('recommended', '#3fb950', '추천 매수 구간', '✅')}
          ${buyZone('conservative','#388bfd', '보수적 매수', '🛡️')}
        </div>`;
    }
  }

  // ── 리스크 카드 ──
  const rgEl = document.getElementById('risk-grid');
  if (rgEl && risk) {
    const riskEntries = ['conservative', 'balanced', 'aggressive'].map(k => risk[k]).filter(Boolean);
    const rrColor = rr => rr >= 2.0 ? '#3fb950' : rr >= 1.5 ? '#d29922' : '#f85149';
    rgEl.innerHTML = `
      ${risk.vol_state ? `<div style="grid-column:1/-1;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px;font-size:12px;color:#8b949e;margin-bottom:2px">
        📊 ${risk.vol_state}<br><span style="font-size:11px">${risk.vol_trend || ''}</span>
      </div>` : ''}
      ${riskEntries.map(sc => `
      <div class="risk-card ${sc.label === '보수적' ? 'conservative' : sc.label === '중립적' ? 'balanced' : 'aggressive'}">
        <div class="risk-icon">${sc.icon}</div>
        <div class="risk-name">${sc.label}</div>
        <div class="risk-desc" style="font-size:11px;color:#8b949e;margin-bottom:8px">${sc.desc}</div>
        <div class="risk-row" style="margin-bottom:6px">
          <span class="risk-lbl">🎯 목표가</span>
          <span class="risk-tgt" style="font-size:12px">${fmt(sc.target[0], isKrx)} ~ ${fmt(sc.target[1], isKrx)}</span>
        </div>
        <div class="risk-row" style="margin-bottom:6px">
          <span class="risk-lbl">🛑 손절가</span>
          <span class="risk-stp" style="font-size:12px">${fmt(sc.stop[0], isKrx)} ~ ${fmt(sc.stop[1], isKrx)}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;padding-top:8px;border-top:1px solid #21262d">
          <div>
            <div style="font-size:10px;color:#8b949e">손절 %</div>
            <div style="font-size:12px;color:#f85149;font-weight:700">${sc.stop_pct}%</div>
          </div>
          <div style="text-align:center">
            <div style="font-size:10px;color:#8b949e">R/R 비율</div>
            <div style="font-size:14px;font-weight:800;color:${rrColor(sc.rr_ratio)}">${sc.rr_ratio}:1</div>
          </div>
          <div style="text-align:right">
            <div style="font-size:10px;color:#8b949e">목표 수익</div>
            <div style="font-size:12px;color:#3fb950;font-weight:700">+${sc.return}%</div>
          </div>
        </div>
        <div style="font-size:10px;color:#8b949e;margin-top:6px;line-height:1.5">💡 ${sc.interpretation || ''}</div>
        <div style="font-size:10px;color:#484f58;margin-top:4px">ATR배수 — 목표:×${sc.atr_mul_tgt} / 손절:×${sc.atr_mul_stp}</div>
      </div>`).join('')}`;
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

  const finnhubNews = !isKrx && d.us_enriched && (d.us_enriched.news || []).length > 0
    ? d.us_enriched.news : null;
  const newsArr = d.naver ? d.naver.news : (finnhubNews || d.news);
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
    col1Title.textContent = finnhubNews ? '📰 관련 뉴스 (Finnhub)' : '📰 관련 뉴스 (Google RSS)';
    discEl.style.display = 'none';
  }
  const renderNewsItem = (n) => {
    const link = n.link || n.url || '#';
    const src  = n.publisher || n.source || '';
    const dt   = n.date || (n.published ? (n.published+'').slice(0,16) : '');
    return `<div class="news-item"><span class="news-dot">📄</span><div>
      <a class="news-a" href="${link}" target="_blank">${n.title||''}</a>
      ${(src||dt) ? `<div class="news-meta">${src}${src&&dt?' · ':''}${dt}</div>` : ''}
    </div></div>`;
  };
  newsList.innerHTML = (newsArr || []).length > 0
    ? (newsArr || []).map(renderNewsItem).join('')
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
const ALL_TABS = ['chart','ai','forecast','news','evening'];
function switchTab(tab) {
  currentTab = tab;
  ALL_TABS.forEach(t => {
    const el = document.getElementById('tab-' + t);
    if (el) el.style.display = t === tab ? 'block' : 'none';
  });
  // 버튼 active 상태 (display:none 버튼도 포함)
  document.querySelectorAll('#result-tabs .tab-btn').forEach(btn => {
    const onclick = btn.getAttribute('onclick') || '';
    const m = onclick.match(/switchTab\('(\w+)'\)/);
    if (m) btn.classList.toggle('active', m[1] === tab);
  });
  // 저녁 검증 탭 첫 진입 시 가이드 표시
  if (tab === 'evening') {
    const content = document.getElementById('evening-content');
    const guide   = document.getElementById('evening-guide');
    const loading = document.getElementById('evening-loading');
    const err     = document.getElementById('evening-error');
    if (content && content.style.display === 'none' &&
        loading && loading.style.display === 'none' &&
        err     && err.style.display === 'none') {
      if (guide) guide.style.display = 'block';
    }
  }
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
    const text = await r.text();
    let d;
    try {
      d = JSON.parse(text);
    } catch(e) {
      throw new Error(`API 응답 오류 (상태: ${r.status})`);
    }
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

// ═══════════════════════════════════════════════════════════════
// 📊 시장 현황 — 거시 시장 요약
// ═══════════════════════════════════════════════════════════════
async function loadMarketCore() {
  document.getElementById('core-loading').style.display = 'block';
  document.getElementById('core-content').style.display = 'none';
  document.getElementById('core-error').style.display = 'none';
  document.getElementById('market-news').style.display = 'none';
  try {
    const r = await fetch('/api/market/summary');
    if (!r.ok) throw new Error('서버 오류 ' + r.status);
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    renderMarketCore(d);
    document.getElementById('core-loading').style.display = 'none';
    document.getElementById('core-content').style.display = 'block';
    document.getElementById('market-news').style.display = '';   // 뉴스 섹션 노출
  } catch(e) {
    document.getElementById('core-loading').style.display = 'none';
    document.getElementById('core-error').style.display = 'block';
    console.warn('[market-core] 로드 실패:', e.message);
  }
}

function renderMarketCore(d) {
  // 시장 무드 배지
  const moodMap = {
    positive: ['mood-positive', '🟢 우호적'],
    neutral:  ['mood-neutral',  '🟡 혼조'],
    negative: ['mood-negative', '🔴 부담적'],
  };
  const [moodCls, moodTxt] = moodMap[d.market_mood] || moodMap.neutral;
  const badge = document.getElementById('core-mood-badge');
  badge.className = 'mood-badge ' + moodCls;
  badge.textContent = moodTxt;

  // VIX 신호
  const vixBadge = document.getElementById('core-vix-badge');
  const vixMap = {
    extreme_fear: '😱 극단 공포 VIX≥30',
    fear:         '😨 공포 VIX≥20',
    complacency:  '😴 과열 VIX≤13',
    normal:       null,
  };
  const vixTxt = vixMap[d.vix_signal];
  if (vixTxt) { vixBadge.textContent = vixTxt; vixBadge.style.display = ''; }
  else        { vixBadge.style.display = 'none'; }

  // 국내 지수 3종 (KOSPI / KOSDAQ / KOSPI200)
  const indices   = d.indices || {};
  const idxOrder  = ['KOSPI', 'KOSDAQ', 'KOSPI200'];
  const idxLabels = { KOSPI: '코스피', KOSDAQ: '코스닥', KOSPI200: 'KOSPI 200' };
  document.getElementById('core-indices').innerHTML = idxOrder.map(k => {
    const idx = indices[k]; if (!idx) return '';
    const up  = idx.direction === 'up';
    const clr = up ? '#f85149' : idx.direction === 'down' ? '#388bfd' : '#8b949e';
    const arrow = up ? '▲' : idx.direction === 'down' ? '▼' : '—';
    return `<div class="core-index-card">
      <div class="ci-name">${idxLabels[k] || k}</div>
      <div class="ci-val">${idx.value || '—'}</div>
      <div class="ci-chg" style="color:${clr}">${arrow} ${idx.change_pct || idx.change_abs || ''}</div>
    </div>`;
  }).join('');

  // 주요 뉴스 (최대 6건) — #market-news 섹션 안의 #core-news에 렌더
  const news   = (d.top_news || []).slice(0, 6);
  const newsEl = document.getElementById('core-news');
  if (!news.length) {
    newsEl.innerHTML = '<p style="font-size:13px;color:#484f58;padding:8px 0">뉴스를 불러올 수 없습니다.</p>';
    return;
  }
  const impCls = { positive: 'cn-positive', negative: 'cn-negative', neutral: 'cn-neutral' };
  const impTxt = { positive: '호재', negative: '악재', neutral: '중립' };
  newsEl.innerHTML = news.map(n => {
    const imp = n.impact || 'neutral';
    return `<div class="core-news-item">
      <span class="cn-impact ${impCls[imp] || 'cn-neutral'}">${impTxt[imp] || '중립'}</span>
      <div style="min-width:0">
        <a class="cn-title" href="${n.link || '#'}" target="_blank" rel="noopener">${n.title || ''}</a>
        <div class="cn-meta">${n.source || ''}${n.date ? ' · ' + n.date : ''}</div>
      </div>
    </div>`;
  }).join('');
}

// ═══════════════════════════════════════════════════════════════
// 🏭 섹터 흐름 — 메인 페이지 자동 로드
// ═══════════════════════════════════════════════════════════════
async function loadSectorFlow() {
  const elLoad = document.getElementById('sector-flow-loading');
  const elCont = document.getElementById('sector-flow-content');
  const elErr  = document.getElementById('sector-flow-error');
  if (!elLoad) return;
  elLoad.style.display = 'block';
  elCont.style.display = 'none';
  elErr.style.display  = 'none';
  try {
    const r = await fetch('/api/market/sector-summary');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    renderSectorFlow(d);
    elLoad.style.display = 'none';
    elCont.style.display = 'block';
  } catch(e) {
    elLoad.style.display = 'none';
    elErr.style.display  = 'block';
    console.warn('[sector-flow] 로드 실패:', e.message);
  }
}

// ── 섹터 카드 단일 렌더 헬퍼 ──────────────────────────────────────────────
function _buildSectorCardHtml(s) {
  const pct    = s.avg_change_pct;
  const pctTxt = pct != null ? (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%' : '—';
  const pctClr = pct == null ? '#8b949e' : pct > 0 ? '#3fb950' : pct < 0 ? '#f85149' : '#8b949e';

  const moodCls = s.mood === 'positive' ? 'sector-mood-pos'
                : s.mood === 'negative' ? 'sector-mood-neg'
                : 'sector-mood-neu';
  const moodTxt = s.mood === 'positive' ? '강세' : s.mood === 'negative' ? '약세' : '혼조';

  // 종목명 태그 — 클릭 시 해당 종목 분석으로 이동
  const names = s.stock_names || [];
  const stockTagsHtml = names.map(n =>
    `<span class="sector-stock-tag"
          onclick="event.stopPropagation();quickSearch('${n.replace(/'/g,"\\'")}')
          " title="${n} 분석">${n}</span>`
  ).join('');

  // 개수 표시 + 펼치기 화살표 (종목 있을 때만)
  const cntHtml = names.length
    ? `<div class="sector-card-cnt">${names.length}종목 ▾</div>`
    : '';

  // 종목 리스트 영역 (기본 숨김 — CSS .expanded 시 display:flex)
  const listHtml = names.length
    ? `<div class="sector-stock-list">${stockTagsHtml}</div>`
    : '';

  return `<div class="sector-card" onclick="toggleSectorCard(this)">
    <div class="sector-card-head">
      <span class="sector-card-emoji">${s.emoji || '🏭'}</span>
      <div>
        <div class="sector-card-name">${s.name}</div>
        ${cntHtml}
      </div>
    </div>
    <div class="sector-card-pct" style="color:${pctClr}">${pctTxt}</div>
    <span class="sector-card-mood ${moodCls}">${moodTxt}</span>
    ${listHtml}
  </div>`;
}

// ── 섹터 흐름 전체 렌더 ────────────────────────────────────────────────────
function renderSectorFlow(d) {
  const cardsEl = document.getElementById('sector-cards');
  if (!cardsEl) return;

  const sectors = d.sectors || [];
  if (!sectors.length) {
    cardsEl.innerHTML = '<p style="color:#484f58;font-size:13px">섹터 데이터 없음</p>';
    return;
  }

  // 강세→혼조→약세 순으로 정렬해 시각적 우선순위 부여
  const moodOrder = {positive: 0, neutral: 1, negative: 2};
  const sorted = [...sectors].sort(
    (a, b) => (moodOrder[a.mood] ?? 1) - (moodOrder[b.mood] ?? 1)
  );

  cardsEl.innerHTML = sorted.map(_buildSectorCardHtml).join('');
}

// ── 섹터 카드 토글 (펼치기 / 접기) ───────────────────────────────────────
function toggleSectorCard(el) {
  const isOpen = el.classList.toggle('expanded');
  // 화살표 방향 갱신 (▾ ↔ ▴)
  const cntEl = el.querySelector('.sector-card-cnt');
  if (cntEl) {
    cntEl.textContent = cntEl.textContent.replace(/[▾▴]/, isOpen ? '▴' : '▾');
  }
}

// ═══════════════════════════════════════════════════════════════
// 🌊 흐름 분석 탭 — /api/stock 데이터로 즉시 렌더 (추가 API 호출 없음)
// ═══════════════════════════════════════════════════════════════
function extractKrxCode(symbol) {
  if (!symbol) return null;
  const m = String(symbol).match(/^(\d{6})\.(KS|KQ)$/);
  return m ? m[1] : null;
}

function renderFlowTab(d) {
  const isKrx = d.market === 'KRX';
  const closes = (d.chart_data || {}).close || [];
  const ma20arr = (d.chart_data || {}).ma20 || [];
  const n = closes.length;

  // ── Signal 1: 뉴스 감성 ──
  let newsSent, newsSentLbl, newsSentClr, posN = 0, negN = 0;
  const finnSent = !isKrx && d.us_enriched && d.us_enriched.sentiment;
  if (finnSent && finnSent.bullish_pct != null) {
    // Finnhub 감성 점수 활용 (US)
    const bull = Number(finnSent.bullish_pct);
    newsSent = bull > 0.55 ? 'positive' : bull < 0.4 ? 'negative' : 'neutral';
    newsSentLbl = {positive:'긍정 (Finnhub)', negative:'부정 (Finnhub)', neutral:'중립 (Finnhub)'}[newsSent];
    posN = Math.round(bull * 100);
    negN = 100 - posN;
  } else {
    // 키워드 기반 (KRX / Finnhub 없을 때)
    const posKw = /상승|급등|강세|반등|신고가|호실적|수주|승인|흑자|수혜|성장|호재/;
    const negKw = /하락|급락|약세|부진|적자|감소|제재|우려|손실|위기|논란|규제/;
    const newsArr = (d.naver ? (d.naver.news || []) : (d.news || []));
    newsArr.forEach(nw => { const t = nw.title||''; if(posKw.test(t))posN++; if(negKw.test(t))negN++; });
    newsSent = posN > negN ? 'positive' : negN > posN ? 'negative' : 'neutral';
    newsSentLbl = {positive:'긍정', negative:'부정', neutral:'중립'}[newsSent];
  }
  newsSentClr = {positive:'sig-up', negative:'sig-down', neutral:'sig-neutral'}[newsSent];

  // ── Signal 2: 추세 방향 (MA20 기울기) ──
  let trendDir = 'neutral', trendLbl = '혼조', trendClr = 'sig-neutral';
  if (ma20arr.length >= 5) {
    const recent = ma20arr.filter(v => v != null).slice(-5);
    if (recent.length >= 2) {
      const slope = (recent[recent.length-1] - recent[0]) / recent[0] * 100;
      if (slope > 0.5) { trendDir='up'; trendLbl='상승 추세'; trendClr='sig-up'; }
      else if (slope < -0.5) { trendDir='down'; trendLbl='하락 추세'; trendClr='sig-down'; }
    }
  }

  // ── Signal 3: RSI 기반 과매수/과매도 위치 ──
  const rsi = d.rsi || 50;
  let posZone = 'neutral', posZoneLbl = '중립 구간', posZoneClr = 'sig-neutral';
  if (rsi > 70) { posZone='high_zone'; posZoneLbl='과매수 (RSI ' + rsi.toFixed(0) + ')'; posZoneClr='sig-down'; }
  else if (rsi < 30) { posZone='low_zone'; posZoneLbl='과매도 (RSI ' + rsi.toFixed(0) + ')'; posZoneClr='sig-up'; }
  else { posZoneLbl = 'RSI ' + rsi.toFixed(0) + ' (중립)'; }

  // ── 3-신호 → 추천 ──
  const score = d.score || 50;
  let rec, recLbl, recCls, conf, rationale;
  const upSig = [newsSent==='positive', trendDir==='up', posZone==='low_zone'].filter(Boolean).length;
  const dnSig = [newsSent==='negative', trendDir==='down', posZone==='high_zone'].filter(Boolean).length;
  if      (upSig === 3)               { rec='strong_buy';  recLbl='강한 상승 기대'; recCls='rec-strong-buy'; conf='높음'; rationale='3개 신호 모두 상승 일치'; }
  else if (upSig === 2 && score >= 50){ rec='buy';          recLbl='상승 기대';      recCls='rec-buy';        conf='중간'; rationale='2개 상승 신호 · 기술적 점수 양호'; }
  else if (dnSig === 3)               { rec='strong_sell'; recLbl='강한 하락 경계'; recCls='rec-strong-sell'; conf='높음'; rationale='3개 신호 모두 하락 일치'; }
  else if (dnSig === 2 && score < 50) { rec='sell';         recLbl='하락 경계';      recCls='rec-sell';       conf='중간'; rationale='2개 하락 신호 · 기술적 점수 미흡'; }
  else                                { rec='hold';         recLbl='관망';           recCls='rec-hold';        conf='낮음'; rationale='신호 혼재 — 추가 확인 필요'; }

  // 기술적 점수 보정
  if (rec==='buy'  && score >= 65) { rec='strong_buy';  recLbl='강한 상승 기대'; recCls='rec-strong-buy'; conf='높음'; rationale += ' · 기술 점수 ' + score + '점'; }
  if (rec==='sell' && score <= 35) { rec='strong_sell'; recLbl='강한 하락 경계'; recCls='rec-strong-sell'; conf='높음'; rationale += ' · 기술 점수 ' + score + '점'; }

  // 신호 매트릭스 렌더
  const newsSub = finnSent && finnSent.bullish_pct != null
    ? `긍정 ${posN}% · 부정 ${negN}%`
    : `호재 ${posN}건 · 악재 ${negN}건`;
  document.getElementById('flow-matrix').innerHTML = `
    <div class="sig-cell">
      <div class="sig-cell-label">📰 뉴스 감성</div>
      <div class="sig-cell-val ${newsSentClr}">${newsSentLbl}</div>
      <div class="flow-subtext">${newsSub}</div>
    </div>
    <div class="sig-cell">
      <div class="sig-cell-label">📈 MA20 추세</div>
      <div class="sig-cell-val ${trendClr}">${trendLbl}</div>
      <div class="flow-subtext">기술점수 ${score}점</div>
    </div>
    <div class="sig-cell">
      <div class="sig-cell-label">🎯 RSI 위치</div>
      <div class="sig-cell-val ${posZoneClr}">${posZoneLbl}</div>
      <div class="flow-subtext">
        ${rsi > 70 ? '과매수 구간' : rsi < 30 ? '과매도 구간' : '중립 구간'}
      </div>
    </div>`;
  document.getElementById('flow-rec-badge').className = 'rec-badge-lg ' + recCls;
  document.getElementById('flow-rec-badge').textContent = recLbl + ' · 신뢰도 ' + conf;
  document.getElementById('flow-rationale').textContent = rationale;

  // 섹터 정보 (스크리너 데이터 활용)
  const sectorCard = document.getElementById('flow-sector-card');
  const sectorContent = document.getElementById('flow-sector-content');
  if (sectorCard && sectorContent && d.naver) {
    const sector = d.naver.sector || '';
    const industry = d.naver.industry || '';
    if (sector || industry) {
      sectorContent.innerHTML = `
        <div class="flow-chip-row">
          ${sector ? `<div class="flow-chip">🏭 섹터 <strong>${sector}</strong></div>` : ''}
          ${industry ? `<div class="flow-chip">🏢 업종 <strong>${industry}</strong></div>` : ''}
        </div>
        <p class="flow-helper-text">💡 동일 섹터 종목 비교는 스크리너(📋)에서 확인하세요.</p>`;
      sectorCard.style.display = 'block';
    } else {
      sectorCard.style.display = 'none';
    }
  } else if (sectorCard) {
    sectorCard.style.display = 'none';
  }
}

// ═══════════════════════════════════════════════════════════════
// 📋 KRX 전용 탭
// ═══════════════════════════════════════════════════════════════
let eveningModeActive = false;

function resetEveningTab() {
  eveningModeActive = false;
  ['evening-loading','evening-content','evening-error'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
  const guide = document.getElementById('evening-guide');
  if (guide) guide.style.display = '';
}

async function loadKrxVerification(krxCode) {
  if (!krxCode) return;
  eveningModeActive = true;
  const guide   = document.getElementById('evening-guide');
  const loading = document.getElementById('evening-loading');
  const content = document.getElementById('evening-content');
  const err     = document.getElementById('evening-error');
  if (guide)   guide.style.display = 'none';
  if (loading) loading.style.display = 'block';
  if (content) content.style.display = 'none';
  if (err)     err.style.display = 'none';
  try {
    const r = await fetch(`/api/market/stocks?codes=${krxCode}&markets=KOSPI&evening=1`);
    if (!r.ok) throw new Error('서버 오류 ' + r.status);
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    renderEveningVerification(d, krxCode);
    if (loading) loading.style.display = 'none';
    if (content) content.style.display = 'block';
  } catch(e) {
    if (loading) loading.style.display = 'none';
    if (err) {
      err.style.display = 'block';
      err.textContent = 'KRX 검증 실패: ' + e.message + ' — 장 마감(15:30) 이후 사용 가능합니다';
    }
    console.warn('[krx-tab] 로드 실패:', e.message);
  }
}

function renderEveningVerification(d, krxCode) {
  const stocks = d.stocks || [];
  const stock = stocks.find(s => s.code === krxCode) || stocks[0];
  if (!stock) {
    document.getElementById('evening-error').style.display = 'block';
    document.getElementById('evening-content').style.display = 'none';
    return;
  }

  // 예측
  const rec = stock.recommendation || 'hold';
  const recLbl = {strong_buy:'강한 상승 기대',buy:'상승 기대',hold:'관망',sell:'하락 경계',strong_sell:'강한 하락 경계'}[rec] || rec;
  const conf = stock.confidence || 'medium';
  const confLbl = {high:'높음',medium:'중간',low:'낮음'}[conf] || conf;
  document.getElementById('ev-pred-rec').textContent = recLbl;
  document.getElementById('ev-pred-conf').textContent = '신뢰도: ' + confLbl + ' · 근거: ' + (stock.rationale || '');

  // 실제 종가
  const quote = stock.quote || {};
  const actualPct = quote.change_pct_num;
  const actualDir = quote.direction || 'flat';
  if (actualPct != null) {
    const up = actualPct > 0;
    const clr = up ? '#f85149' : actualPct < 0 ? '#388bfd' : '#8b949e';
    document.getElementById('ev-actual-pct').innerHTML =
      `<span style="color:${clr}">${up?'▲':actualPct<0?'▼':'—'} ${Math.abs(actualPct).toFixed(2)}%</span>`;
    document.getElementById('ev-actual-dir').textContent = quote.price ? '종가: ' + quote.price : '';
  } else {
    document.getElementById('ev-actual-pct').textContent = '—';
    document.getElementById('ev-actual-dir').textContent = '종가 데이터 없음';
  }

  // 판정 결과
  const result = stock.result || {};
  const outcome = result.outcome || 'n/a';
  const outcomeMap = {
    hit:     ['ev-hit-badge',     '적중 ✅'],
    partial: ['ev-partial-badge', '부분 ⚖️'],
    miss:    ['ev-miss-badge',    '실패 ❌'],
    'n/a':   ['ev-na-badge',      '데이터 없음'],
  };
  const [outCls, outTxt] = outcomeMap[outcome] || outcomeMap['n/a'];
  const verdictBadge = document.getElementById('ev-verdict-badge');
  verdictBadge.className = 'ev-outcome-badge ' + outCls;
  verdictBadge.textContent = outTxt;
  document.getElementById('ev-verdict-note').textContent = result.note || '';

  // 판정 카드 색상
  const verdictCard = document.getElementById('ev-verdict-card');
  const cardCls = {hit:'ev-hit', partial:'ev-partial', miss:'ev-miss', 'n/a':'ev-neutral'};
  verdictCard.className = 'ev-card ' + (cardCls[outcome] || 'ev-neutral');

  // 신호 기여 분석
  const breakdown = document.getElementById('ev-signal-breakdown');
  const rows = [
    ['뉴스 감성', {positive:'긍정',negative:'부정',neutral:'중립'}[stock.news_sentiment||'neutral'] || '—'],
    ['간밤 신호', {up:'상승 강세',down:'하락 약세',neutral:'중립'}[stock.overnight_signal_dir||'neutral'] || '—'],
    ['52주 위치', {high_zone:'고가권 경계',low_zone:'저가권 반등 기대',neutral:'중립 구간'}[stock.price_zone||'neutral'] || '—'],
    ['거래량', stock.volume_spike ? '🚨 급증 (2× 이상)' : '정상'],
    ['기술 점수', currentData ? String(currentData.score) + '점' : '—'],
  ];
  breakdown.innerHTML = rows.map(([label, val]) =>
    `<div class="ev-sig-row"><span style="font-size:12px;color:#8b949e">${label}</span><span style="font-size:13px;font-weight:600">${val}</span></div>`
  ).join('');
}

// ══════════════════════════════════════════════════════
// 🔔 알림 시스템 — localStorage 기반 (stock-dashboard 이식)
// ══════════════════════════════════════════════════════

const AlertsStore = {
  _KA: 'so_alerts',
  _KN: 'so_notifications',
  _KF: 'so_alert_fired',

  // ── 설정 CRUD ──
  getAll() { try { return JSON.parse(localStorage.getItem(this._KA) || '[]'); } catch { return []; } },
  get(symbol) { return this.getAll().find(a => a.symbol === symbol) || null; },
  save(cfg) {
    const all = this.getAll().filter(a => a.symbol !== cfg.symbol);
    all.push(cfg);
    localStorage.setItem(this._KA, JSON.stringify(all));
    _alertBellUpdate();
  },
  remove(symbol) {
    localStorage.setItem(this._KA, JSON.stringify(this.getAll().filter(a => a.symbol !== symbol)));
    this._clearFiredFor(symbol);
    localStorage.setItem(this._KN, JSON.stringify(this.getNotifs().filter(n => n.symbol !== symbol)));
    _alertBellUpdate();
  },

  // ── 알림 이력 ──
  getNotifs() { try { return JSON.parse(localStorage.getItem(this._KN) || '[]'); } catch { return []; } },
  addNotif(n) {
    const notifs = this.getNotifs();
    if (notifs.some(x => x.symbol === n.symbol && x.type === n.type)) return;
    notifs.unshift({ ...n, id: n.symbol + '-' + n.type + '-' + Date.now(), triggeredAt: Date.now() });
    localStorage.setItem(this._KN, JSON.stringify(notifs.slice(0, 50)));
    _alertBellUpdate();
  },
  dismissNotif(id) {
    localStorage.setItem(this._KN, JSON.stringify(this.getNotifs().filter(n => n.id !== id)));
    _alertBellUpdate();
  },
  clearNotifs() { localStorage.setItem(this._KN, '[]'); _alertBellUpdate(); },

  // ── 중복 발화 방지 ──
  getFired() { try { return JSON.parse(localStorage.getItem(this._KF) || '[]'); } catch { return []; } },
  isFired(key) { return this.getFired().includes(key); },
  markFired(key) {
    const f = this.getFired();
    if (!f.includes(key)) { f.push(key); localStorage.setItem(this._KF, JSON.stringify(f.slice(-200))); }
  },
  _clearFiredFor(symbol) {
    localStorage.setItem(this._KF, JSON.stringify(this.getFired().filter(k => !k.startsWith(symbol + '-'))));
  },
};

function _alertBellUpdate() {
  const count = AlertsStore.getNotifs().length;
  const el = document.getElementById('alert-bell-count');
  if (!el) return;
  el.textContent = count > 0 ? count : '';
  el.classList.toggle('visible', count > 0);
}

// ── 모달 ──────────────────────────────────────────────
var _currentAlertSymbol = null;
var _currentAlertPrice  = null;

function openAlertModal(symbol, price) {
  if (!symbol) return;
  _currentAlertSymbol = symbol;
  _currentAlertPrice  = price;
  const ex = AlertsStore.get(symbol);
  document.getElementById('am-symbol').textContent    = symbol + ' 알림 설정';
  document.getElementById('am-cur-price').textContent = price ? '현재가 ' + Math.round(price).toLocaleString() + '원' : '';
  document.getElementById('am-target').value          = ex && ex.targetPrice != null ? Math.round(ex.targetPrice) : '';
  const surgeChk  = document.getElementById('am-surge-chk');
  const plungeChk = document.getElementById('am-plunge-chk');
  surgeChk.checked  = !!(ex && ex.surgeEnabled);
  plungeChk.checked = !!(ex && ex.plungeEnabled);
  document.getElementById('am-surge-pct').value    = ex ? ex.surgeThreshold  : 5;
  document.getElementById('am-plunge-pct').value   = ex ? ex.plungeThreshold : 5;
  document.getElementById('am-surge-row').style.display  = surgeChk.checked  ? '' : 'none';
  document.getElementById('am-plunge-row').style.display = plungeChk.checked ? '' : 'none';
  document.getElementById('am-del-btn').style.display    = ex ? '' : 'none';
  _updateAlertPreview();
  document.getElementById('alert-modal').classList.add('open');
}

function closeAlertModal() {
  document.getElementById('alert-modal').classList.remove('open');
}

function _updateAlertPreview() {
  const v   = parseFloat(document.getElementById('am-target').value);
  const el  = document.getElementById('am-target-preview');
  if (!el) return;
  if (!isNaN(v) && v > 0 && _currentAlertPrice) {
    const dir = v > _currentAlertPrice ? 'above' : 'below';
    el.textContent = dir === 'above' ? '↑ 상향 돌파 시 알림' : '↓ 하향 도달 시 알림';
    el.className   = 'alert-preview-dir ' + (dir === 'above' ? 'alert-preview-up' : 'alert-preview-dn');
    el.style.display = '';
  } else {
    el.style.display = 'none';
  }
}

function saveAlert() {
  if (!_currentAlertSymbol) return;
  const raw = parseFloat(document.getElementById('am-target').value);
  const targetPrice = (!isNaN(raw) && raw > 0) ? raw : null;
  let targetDirection = null;
  if (targetPrice != null && _currentAlertPrice) {
    targetDirection = targetPrice > _currentAlertPrice ? 'above' : 'below';
  }
  AlertsStore.save({
    symbol:           _currentAlertSymbol,
    targetPrice,
    targetDirection,
    surgeEnabled:     document.getElementById('am-surge-chk').checked,
    surgeThreshold:   Math.max(0.1, parseFloat(document.getElementById('am-surge-pct').value)  || 5),
    plungeEnabled:    document.getElementById('am-plunge-chk').checked,
    plungeThreshold:  Math.max(0.1, parseFloat(document.getElementById('am-plunge-pct').value) || 5),
  });
  AlertsStore._clearFiredFor(_currentAlertSymbol);
  closeAlertModal();
  _alertResultBtnUpdate(_currentAlertSymbol);
  if (_alertSheetOpen) renderAlertsSheet();
}

function deleteAlert() {
  if (!_currentAlertSymbol) return;
  AlertsStore.remove(_currentAlertSymbol);
  closeAlertModal();
  _alertResultBtnUpdate(_currentAlertSymbol);
  if (_alertSheetOpen) renderAlertsSheet();
}

function _alertResultBtnUpdate(symbol) {
  const btn = document.getElementById('result-alert-btn');
  if (!btn || btn.style.display === 'none') return;
  const has = !!AlertsStore.get(symbol);
  btn.classList.toggle('has-alert', has);
}

// ── 알림 시트 ─────────────────────────────────────────
var _alertSheetOpen = false;
var _alertSheetTab  = 'triggered';

function openAlertsSheet() {
  _alertSheetOpen = true;
  _alertSheetTab  = 'triggered';
  document.getElementById('sheet-tab-triggered').classList.add('active');
  document.getElementById('sheet-tab-configured').classList.remove('active');
  renderAlertsSheet();
  document.getElementById('alert-sheet').classList.add('open');
  document.getElementById('alert-sheet-backdrop').classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeAlertsSheet() {
  _alertSheetOpen = false;
  document.getElementById('alert-sheet').classList.remove('open');
  document.getElementById('alert-sheet-backdrop').classList.remove('open');
  document.body.style.overflow = '';
}

function switchSheetTab(tab) {
  _alertSheetTab = tab;
  document.getElementById('sheet-tab-triggered').classList.toggle('active', tab === 'triggered');
  document.getElementById('sheet-tab-configured').classList.toggle('active', tab === 'configured');
  _renderSheetBody();
}

function renderAlertsSheet() {
  const notifs  = AlertsStore.getNotifs();
  const configs = AlertsStore.getAll().filter(a => a.targetPrice != null || a.surgeEnabled || a.plungeEnabled);
  const fb = document.getElementById('sheet-fired-badge');
  const cb = document.getElementById('sheet-config-badge');
  if (fb) { fb.textContent = notifs.length;   fb.style.display = notifs.length  > 0 ? '' : 'none'; }
  if (cb) { cb.textContent = configs.length; cb.style.display = configs.length > 0 ? '' : 'none'; }
  _renderSheetBody();
}

function _renderSheetBody() {
  const body = document.getElementById('alert-sheet-body');
  if (!body) return;
  if (_alertSheetTab === 'triggered') {
    const notifs = AlertsStore.getNotifs();
    if (!notifs.length) {
      body.innerHTML = '<div class="alert-sheet-empty">발생한 알림이 없습니다.</div>';
      return;
    }
    body.innerHTML =
      '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">' +
        '<span style="font-size:11px;color:#8b949e">' + notifs.length + '개의 알림</span>' +
        '<button onclick="AlertsStore.clearNotifs();renderAlertsSheet()" style="font-size:11px;color:#8b949e;background:none;border:none;cursor:pointer">모두 지우기</button>' +
      '</div>' +
      notifs.map(function(n) {
        var isUp = n.type === 'surge' || n.type === 'target_above';
        var cls  = isUp ? 'alert-noti-up' : 'alert-noti-dn';
        var t    = _alertFormatTime(n.triggeredAt);
        return '<div class="alert-noti-item ' + cls + '">' +
          '<div class="alert-noti-main">' +
            '<div class="alert-noti-symbol">' + n.symbol + '</div>' +
            '<div class="alert-noti-label">' + _alertNotifLabel(n) + '</div>' +
          '</div>' +
          '<div class="alert-noti-time">' + t.absolute + '<br><span style="color:#484f58">(' + t.relative + ')</span></div>' +
          '<button class="alert-noti-dismiss" onclick="AlertsStore.dismissNotif(\'' + n.id + '\');renderAlertsSheet()">✕</button>' +
        '</div>';
      }).join('');
  } else {
    var configs = AlertsStore.getAll().filter(function(a) {
      return a.targetPrice != null || a.surgeEnabled || a.plungeEnabled;
    });
    if (!configs.length) {
      body.innerHTML = '<div class="alert-sheet-empty">설정된 알림이 없습니다.<br><span style="font-size:11px">종목 분석 후 🔔 버튼으로 추가하세요.</span></div>';
      return;
    }
    body.innerHTML = configs.map(function(a) {
      var tags = [];
      if (a.targetPrice != null) {
        var dir = a.targetDirection === 'above' ? '↑' : a.targetDirection === 'below' ? '↓' : '';
        tags.push('<span class="alert-cfg-tag alert-cfg-tag-target">목표가 ' + Math.round(a.targetPrice).toLocaleString() + '원 ' + dir + '</span>');
      }
      if (a.surgeEnabled)  tags.push('<span class="alert-cfg-tag alert-cfg-tag-surge">급등 +' + a.surgeThreshold + '%</span>');
      if (a.plungeEnabled) tags.push('<span class="alert-cfg-tag alert-cfg-tag-plunge">급락 -' + a.plungeThreshold + '%</span>');
      return '<div class="alert-cfg-item">' +
        '<div class="alert-cfg-code">' + a.symbol + '</div>' +
        '<div class="alert-cfg-tags">' + tags.join('') + '</div>' +
        '<button class="alert-cfg-del" onclick="AlertsStore.remove(\'' + a.symbol + '\');renderAlertsSheet()" title="삭제">✕</button>' +
      '</div>';
    }).join('');
  }
}

function _alertNotifLabel(n) {
  var price = Math.round(n.price).toLocaleString() + '원';
  if (n.type === 'target_above') return '목표가 도달 ↑  ' + price;
  if (n.type === 'target_below') return '목표가 도달 ↓  ' + price;
  if (n.type === 'surge')        return '급등 +' + (n.changePercent || 0).toFixed(2) + '%  (' + price + ')';
  if (n.type === 'plunge')       return '급락 ' + (n.changePercent || 0).toFixed(2) + '%  (' + price + ')';
  return price;
}

function _alertFormatTime(ts) {
  var d    = new Date(ts);
  var mon  = String(d.getMonth() + 1).padStart(2, '0');
  var day  = String(d.getDate()).padStart(2, '0');
  var h    = d.getHours();
  var min  = String(d.getMinutes()).padStart(2, '0');
  var ampm = h < 12 ? '오전' : '오후';
  var h12  = h % 12 === 0 ? 12 : h % 12;
  var diff = Math.floor((Date.now() - ts) / 1000);
  var rel;
  if (diff < 60) rel = '방금';
  else if (diff < 3600) rel = Math.floor(diff / 60) + '분 전';
  else { var rh = Math.floor(diff / 3600); var rm = Math.floor((diff % 3600) / 60); rel = rm > 0 ? rh + '시간 ' + rm + '분 전' : rh + '시간 전'; }
  return { absolute: mon + '월' + day + '일 ' + ampm + ' ' + h12 + '시' + min + '분', relative: rel };
}

// ── 토스트 알림 ──────────────────────────────────────
function showAlertToast(n) {
  var isUp  = n.type === 'surge' || n.type === 'target_above';
  var icon  = isUp ? '📈' : '📉';
  var toast = document.createElement('div');
  toast.className = 'alert-toast';
  toast.innerHTML =
    '<div class="alert-toast-icon">' + icon + '</div>' +
    '<div class="alert-toast-body">' +
      '<div class="alert-toast-title">' + n.symbol + '</div>' +
      '<div class="alert-toast-desc">' + _alertNotifLabel(n) + '</div>' +
    '</div>' +
    '<button class="alert-toast-close" onclick="this.parentElement.remove()">✕</button>';
  document.getElementById('alert-toast-container').appendChild(toast);
  requestAnimationFrame(function() { toast.classList.add('visible'); });
  setTimeout(function() {
    toast.classList.remove('visible');
    setTimeout(function() { toast.remove(); }, 350);
  }, 6000);
}

// ── 알림 모니터 (2분 폴링) ───────────────────────────
var AlertMonitor = {
  _t: null,
  start: function() {
    if (this._t) return;
    // 페이지 로드 직후 서버 요청 경쟁 방지: 첫 체크는 120초 후부터 시작
    this._t = setInterval(function() { AlertMonitor._check(); }, 120000);
  },
  stop: function() { if (this._t) { clearInterval(this._t); this._t = null; } },
  _check: async function() {
    var alerts = AlertsStore.getAll();
    if (!alerts.length) return;
    var codes = alerts.map(function(a) { return a.symbol; }).filter(function(s) { return /^\d{6}$/.test(s); });
    if (!codes.length) return;
    try {
      var r    = await fetch('/api/alert/quote?codes=' + codes.join(','));
      var data = await r.json();
      var qs   = data.quotes || {};
      alerts.forEach(function(alert) {
        var q = qs[alert.symbol];
        if (!q) return;
        var price = q.price, chg = q.change_pct;
        // 목표가
        if (alert.targetPrice != null && alert.targetDirection != null) {
          if (alert.targetDirection === 'above') {
            var k = alert.symbol + '-target_above';
            if (price >= alert.targetPrice && !AlertsStore.isFired(k)) {
              AlertsStore.markFired(k);
              var n = { symbol: alert.symbol, type: 'target_above', price: price, targetPrice: alert.targetPrice };
              AlertsStore.addNotif(n); showAlertToast(n);
            }
          } else {
            var k = alert.symbol + '-target_below';
            if (price <= alert.targetPrice && !AlertsStore.isFired(k)) {
              AlertsStore.markFired(k);
              var n = { symbol: alert.symbol, type: 'target_below', price: price, targetPrice: alert.targetPrice };
              AlertsStore.addNotif(n); showAlertToast(n);
            }
          }
        }
        // 급등
        if (alert.surgeEnabled) {
          var k = alert.symbol + '-surge';
          if (chg >= alert.surgeThreshold && !AlertsStore.isFired(k)) {
            AlertsStore.markFired(k);
            var n = { symbol: alert.symbol, type: 'surge', price: price, changePercent: chg };
            AlertsStore.addNotif(n); showAlertToast(n);
          }
        }
        // 급락
        if (alert.plungeEnabled) {
          var k = alert.symbol + '-plunge';
          if (chg <= -alert.plungeThreshold && !AlertsStore.isFired(k)) {
            AlertsStore.markFired(k);
            var n = { symbol: alert.symbol, type: 'plunge', price: price, changePercent: chg };
            AlertsStore.addNotif(n); showAlertToast(n);
          }
        }
      });
    } catch(e) { /* 네트워크 오류 무시 */ }
  },
};

function initAlerts() {
  _alertBellUpdate();
  AlertMonitor.start();
  document.addEventListener('keydown', function(e) {
    if (e.key !== 'Escape') return;
    if (document.getElementById('alert-modal').classList.contains('open')) closeAlertModal();
    else if (_alertSheetOpen) closeAlertsSheet();
  });
}

// ── 초기화 ──
loadMarketCore();   // ⭐ 페이지 로드 시 오늘의 핵심 자동 로드
loadSectorFlow();   // 🏭 섹터 흐름 자동 로드
initAlerts();       // 🔔 알림 시스템 초기화

// ── Pull-to-Refresh (모바일) ──
(function(){
  var THRESHOLD  = 72;   // 트리거까지 필요한 최소 드래그 거리(px)
  var el         = document.getElementById('ptr-indicator');
  var txt        = document.getElementById('ptr-text');
  if (!el || !txt) return;

  var startY   = 0;
  var pulling  = false;   // touchstart 에서 PTR 후보로 등록됨
  var dragging = false;   // 실제로 아래로 당기고 있는 중
  var busy     = false;   // 새로고침 실행 중 (중복 방지)

  // ── 스크롤 최상단 여부 판별 ──
  // 모바일: body 스크롤 / 데스크탑: #main 스크롤 → 둘 다 확인
  function atTop() {
    var bodyTop = window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0;
    var mainEl  = document.getElementById('main');
    var mainTop = mainEl ? mainEl.scrollTop : 0;
    return bodyTop === 0 && mainTop === 0;
  }

  // ── 현재 활성 페이지 ──
  function getActivePage() {
    var s = document.getElementById('page-screener');
    return (s && s.style.display !== 'none') ? 'screener' : 'analysis';
  }

  // ── 인디케이터 초기화 ──
  function resetIndicator() {
    busy = false;
    el.style.height  = '0';
    el.style.opacity = '0';
    setTimeout(function(){ txt.textContent = '↓ 당겨서 새로고침'; }, 200);
  }

  // ── 새로고침 실행 ──
  function doRefresh() {
    busy = true;
    el.style.height  = '48px';
    el.style.opacity = '1';
    txt.innerHTML    = '<span class="ptr-spinner"></span>새로고침 중...';

    var page     = getActivePage();
    var savedTab = currentTab;   // 현재 탭 저장
    var p;

    if (page === 'screener') {
      p = loadScreener();
    } else {
      var resultEl  = document.getElementById('state-result');
      var hasResult = resultEl && resultEl.style.display !== 'none';
      if (hasResult) {
        // analyze() 완료 후 renderResult() 가 강제로 switchTab('chart') 하므로
        // .then() 에서 저장해 둔 탭으로 즉시 복원
        p = analyze().then(function(){ switchTab(savedTab); });
      }
    }

    if (p && typeof p.finally === 'function') {
      p.finally(resetIndicator);
    } else {
      setTimeout(resetIndicator, 600);
    }
  }

  // ── touchstart ──
  document.addEventListener('touchstart', function(e) {
    if (busy) return;
    if (atTop()) {
      startY   = e.touches[0].clientY;
      pulling  = true;
      dragging = false;
    }
  }, { passive: true });

  // ── touchmove  (passive:false → preventDefault 가능) ──
  // passive:false 는 브라우저 네이티브 PTR(전체 새로고침)을 막는 데 필수
  document.addEventListener('touchmove', function(e) {
    if (!pulling || busy) return;
    var dy = e.touches[0].clientY - startY;

    if (dy > 0 && atTop()) {
      // 아래로 당기는 중 → 네이티브 스크롤·PTR 억제
      e.preventDefault();
      dragging = true;
      el.style.height  = Math.min(dy * 0.45, 52) + 'px';
      el.style.opacity = String(Math.min(dy / THRESHOLD, 1));
      txt.textContent  = dy >= THRESHOLD ? '↑ 놓으면 새로고침' : '↓ 당겨서 새로고침';
    } else if (dy <= 0) {
      // 위로 올리면 PTR 취소
      pulling  = false;
      dragging = false;
      resetIndicator();
    }
  }, { passive: false });

  // ── touchend ──
  document.addEventListener('touchend', function(e) {
    if (!pulling) return;
    pulling = false;

    if (!dragging) return;   // 실제 당김 없이 끝남
    dragging = false;

    var dy = e.changedTouches[0].clientY - startY;
    if (dy >= THRESHOLD) {
      doRefresh();
    } else {
      resetIndicator();
    }
  }, { passive: true });

  // ── touchcancel (전화·알림 등 인터럽트) ──
  document.addEventListener('touchcancel', function() {
    pulling = dragging = false;
    if (!busy) resetIndicator();
  }, { passive: true });
})();
</script>

<!-- ── 🔔 알림 설정 모달 ── -->
<div class="alert-modal-overlay" id="alert-modal" onclick="if(event.target===this)closeAlertModal()">
  <div class="alert-modal">
    <div class="alert-modal-header">
      <div>
        <div class="alert-modal-title" id="am-symbol"></div>
        <div style="font-size:11px;color:#8b949e;margin-top:2px" id="am-cur-price"></div>
      </div>
      <button class="alert-modal-close" onclick="closeAlertModal()">✕</button>
    </div>
    <!-- 목표가 -->
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <label class="alert-field-label" style="margin-bottom:0">목표가 (원)</label>
        <span class="alert-preview-dir" id="am-target-preview" style="display:none"></span>
      </div>
      <input type="number" min="0" id="am-target" class="alert-price-input" placeholder="예: 95000" oninput="_updateAlertPreview()">
    </div>
    <!-- 급등 -->
    <div>
      <label class="alert-toggle-row">
        <input type="checkbox" id="am-surge-chk" onchange="document.getElementById('am-surge-row').style.display=this.checked?'':'none'">
        <span style="font-size:13px;font-weight:600;color:#3fb950">📈 급등 알림</span>
      </label>
      <div class="alert-pct-row" id="am-surge-row" style="display:none">
        <span style="font-size:12px;color:#8b949e">전일대비</span>
        <input type="number" min="0.1" step="0.5" id="am-surge-pct" class="alert-pct-input" value="5">
        <span style="font-size:12px;color:#8b949e">% 이상 상승</span>
      </div>
    </div>
    <!-- 급락 -->
    <div>
      <label class="alert-toggle-row">
        <input type="checkbox" id="am-plunge-chk" onchange="document.getElementById('am-plunge-row').style.display=this.checked?'':'none'">
        <span style="font-size:13px;font-weight:600;color:#f85149">📉 급락 알림</span>
      </label>
      <div class="alert-pct-row" id="am-plunge-row" style="display:none">
        <span style="font-size:12px;color:#8b949e">전일대비</span>
        <input type="number" min="0.1" step="0.5" id="am-plunge-pct" class="alert-pct-input" value="5">
        <span style="font-size:12px;color:#8b949e">% 이상 하락</span>
      </div>
    </div>
    <!-- 버튼 -->
    <div class="alert-btn-row">
      <button class="alert-btn-save" onclick="saveAlert()">저장</button>
      <button class="alert-btn-del" id="am-del-btn" onclick="deleteAlert()" style="display:none">삭제</button>
      <button class="alert-btn-cancel" onclick="closeAlertModal()">취소</button>
    </div>
  </div>
</div>

<!-- ── 🔔 알림 시트 (bottom sheet) ── -->
<div class="alert-sheet-backdrop" id="alert-sheet-backdrop" onclick="closeAlertsSheet()"></div>
<div class="alert-sheet" id="alert-sheet" role="dialog" aria-label="알림">
  <div class="alert-sheet-handle"><div class="alert-sheet-drag"></div></div>
  <div class="alert-sheet-header">
    <div class="alert-sheet-tabs">
      <button class="alert-sheet-tab active" id="sheet-tab-triggered" onclick="switchSheetTab('triggered')">
        발생한 알림
        <span class="alert-tab-badge alert-fired-badge" id="sheet-fired-badge" style="display:none"></span>
      </button>
      <button class="alert-sheet-tab" id="sheet-tab-configured" onclick="switchSheetTab('configured')">
        설정한 알림
        <span class="alert-tab-badge alert-config-badge" id="sheet-config-badge" style="display:none"></span>
      </button>
    </div>
    <button class="alert-sheet-close" onclick="closeAlertsSheet()">닫기</button>
  </div>
  <div class="alert-sheet-body" id="alert-sheet-body"></div>
</div>

<!-- 🔔 토스트 컨테이너 -->
<div id="alert-toast-container"></div>

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
        elif path == "/api/market/sector-summary":
            # 10분 캐시 — 섹터 흐름은 빠르게 바뀌지 않음
            cache_control = "public, s-maxage=600, stale-while-revalidate=3600"
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
