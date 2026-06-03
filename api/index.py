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

# =============================================================================
# 🔬 7단계 스캔 엔진 — 출력/선정 공통 설정 (한국·미국 동일 적용)
# =============================================================================
# Vercel maxDuration=60s / 1024MB 환경 기준.
# 단일 종목 수집 비용 = yf history(1y) + tk.info 1회씩 (info 호출이 가장 느림).
# 12 워커 병렬 수집이라도 수집 대상 수가 곧 타임아웃 위험이므로,
# "수집 상한"과 "출력 개수"를 상수화하고 양 시장에 동일 적용한다.
SCAN_COLLECT_CAP_FULL = 48   # FULL 수집 상한 — 대형+중형+중소형 풀 커버(다양성+15개 확보)
SCAN_COLLECT_CAP_LITE = 24   # CORE_LITE 수집 상한
SCAN_DISPLAY_CAP      = 15   # 최종 출력 종목 수 (양 시장 공통, 목표=정확히 15)

# 병렬 수집 — 워커 수 / 월클럭 예산 (Vercel 60s 내 안전 마진)
SCAN_COLLECT_WORKERS  = 16
SCAN_COLLECT_BUDGET_S = 42.0   # 이 시간 내 도착한 종목만으로 진행 (타임아웃 하드가드)

# 출력 선정 게이트 — 진입 가능권(상태)만 노출 (신호 조건 제거)
SCAN_GOOD_STATES = ("READY", "WATCH", "WAIT_PULLBACK")

# ── 시가총액 티어 임계값 (시장별) — info.marketCap 우선, 없으면 정적 태그 ──
#   KRX: ≥10조=대형 / 1~10조=중형 / <1조=중소형 (KRW)
#   US : ≥$10B=대형 / $2~10B=중형 / <$2B=중소형 (USD)
SCAN_CAP_TIER_THRESHOLDS = {
    "KRX": {"LARGE": 10e12, "MID": 1e12},
    "US":  {"LARGE": 10e9,  "MID": 2e9},
}
SCAN_CAP_TIER_KO = {"LARGE": "대형", "MID": "중형", "SMALL": "중소형"}

# ── 다양성 선정(MMR) — 동일 섹터/시총티어 누적 시 점수 페널티 ──
#   고득점이라도 같은 섹터·티어가 쌓이면 유효점수가 낮아져
#   우수한 타 섹터·중소형주가 자연스럽게 상위로 진입한다. (하드 제한 아님)
SCAN_SECTOR_PENALTY   = 8.0    # 섹터 소프트캡 초과 1종목당 -8점
SCAN_SECTOR_SOFT_CAP  = 2      # 섹터당 2종목까지는 페널티 면제
SCAN_CAP_TIER_PENALTY = 6.0    # 티어 소프트캡 초과 1종목당 -6점
SCAN_CAP_TIER_SOFT    = 2      # 티어당 2종목까지는 페널티 면제 (대형주 독점 억제)

# 상태 → 점수 (0~100): 진입이 임박할수록 높음
SCAN_STATE_SCORE = {
    "READY": 100.0, "WATCH": 70.0, "WAIT_PULLBACK": 60.0,
    "COOLDOWN": 20.0, "EARNINGS_BLOCK": 10.0, "FAR": 0.0,
}

# 퀀트 모멘텀(QMJ 품질 티어) → 점수 (0~100)
SCAN_MOMENTUM_TIER_SCORE = {
    "high": 100.0, "medium": 70.0, "unknown": 50.0, "low": 20.0, "junk": 0.0,
}

# 종합 점수 가중치 — 상태·순복합점수·퀀트모멘텀 (합 = 1.0)
SCAN_W_STATE    = 0.35
SCAN_W_NCS      = 0.40
SCAN_W_MOMENTUM = 0.25


def _scan_num(v, default: float = 0.0) -> float:
    """방어적 숫자 변환 — None/빈값/문자열 등은 default로 (런타임·타입 오류 방지)."""
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def scan_quant_momentum_score(cd: dict) -> float:
    """퀀트 모멘텀 점수 (0~100) — QMJ 품질 티어 + 모멘텀 배수 종합.

    한국·미국 동일 기준. 누락 필드는 'unknown'(50) 기본값으로 방어한다.
    """
    tier    = cd.get("quality_tier") or "unknown"
    tier_sc = SCAN_MOMENTUM_TIER_SCORE.get(tier, 50.0)
    mult    = max(0.0, min(1.0, _scan_num(cd.get("quality_multiplier"), 0.5)))
    return round(tier_sc * 0.7 + mult * 100.0 * 0.3, 2)


def scan_ncs_score(cd: dict) -> float:
    """순복합 점수 (0~100) — adjusted_ncs 우선, 없으면 ncs (0.0도 유효값으로 처리)."""
    adj = cd.get("adjusted_ncs")
    return _scan_num(adj if adj is not None else cd.get("ncs"), 0.0)


def scan_composite_score(cd: dict) -> float:
    """상태·순복합점수·퀀트모멘텀 3축 종합 점수 (0~100) — 양 시장 공통 스코어러.

    세 기준을 가중 평균하여 우수 종목이 상위에 오도록 한다.
    """
    state_sc = SCAN_STATE_SCORE.get(cd.get("status"), 0.0)
    ncs_sc   = scan_ncs_score(cd)
    mom_sc   = scan_quant_momentum_score(cd)
    return round(
        state_sc * SCAN_W_STATE + ncs_sc * SCAN_W_NCS + mom_sc * SCAN_W_MOMENTUM, 2
    )


def scan_cap_tier(market: str, market_cap, static_tier: str = "") -> str:
    """시가총액 → 티어(LARGE/MID/SMALL). info.marketCap 우선, 없으면 정적 태그.

    marketCap이 없거나 0이면 정적 태그(static_tier)로 폴백, 그것도 없으면 MID.
    """
    th = SCAN_CAP_TIER_THRESHOLDS.get("KRX" if market == "KRX" else "US")
    mc = _scan_num(market_cap, 0.0)
    if mc > 0 and th:
        if mc >= th["LARGE"]:
            return "LARGE"
        if mc >= th["MID"]:
            return "MID"
        return "SMALL"
    st = (static_tier or "").upper()
    return st if st in ("LARGE", "MID", "SMALL") else "MID"


def _scan_sector_key(cd: dict) -> str:
    """후보의 섹터 키 (빈 값은 분산 페널티 제외 대상)."""
    return (cd.get("category") or cd.get("sector") or "").strip()


def scan_diversified_fill(
    pool: list, need: int,
    sector_count: Dict[str, int], tier_count: Dict[str, int],
) -> list:
    """pool에서 need개를 MMR 방식으로 분산 선정해 반환.

    매 단계 '유효점수(종합점수 − 섹터페널티 − 티어페널티)'가 가장 높은 후보를
    선택한다. 같은 섹터/티어가 누적될수록 페널티가 커져, 고득점 대형주가
    독점하지 않고 우수한 타 섹터·중소형주가 자연스럽게 진입한다.

    sector_count / tier_count는 호출자 소유 누적 카운터로 in-place 갱신되어,
    여러 품질 등급(tier)을 순차 보강할 때도 분산이 이어진다.
    """
    candidates: list = list(pool)
    picked: list = []
    while candidates and len(picked) < need:
        best, best_eff = None, None
        for c in candidates:
            sec  = _scan_sector_key(c)
            cap  = c.get("cap_tier") or "MID"
            sec_pen = (
                SCAN_SECTOR_PENALTY * max(0, sector_count.get(sec, 0) - SCAN_SECTOR_SOFT_CAP)
                if sec else 0.0
            )
            cap_pen = SCAN_CAP_TIER_PENALTY * max(0, tier_count.get(cap, 0) - SCAN_CAP_TIER_SOFT)
            eff = _scan_num(c.get("composite_score")) - sec_pen - cap_pen
            if best_eff is None or eff > best_eff:
                best_eff, best = eff, c
        picked.append(best)
        candidates.remove(best)
        sec = _scan_sector_key(best)
        cap = best.get("cap_tier") or "MID"
        if sec:
            sector_count[sec] = sector_count.get(sec, 0) + 1
        tier_count[cap] = tier_count.get(cap, 0) + 1
        best["div_effective_score"] = round(best_eff, 2)
    return picked


def scan_diversified_select(cands: list, target: int) -> list:
    """단일 풀 분산 선정 (하위호환 래퍼) — 빈 누적 카운터로 fill 호출."""
    return scan_diversified_fill(list(cands), target, {}, {})


# 스캔 응답 캐시 — warm 인스턴스 재사용 시 yfinance 재호출/재계산 방지
_SCAN_RESULT_CACHE: Dict[str, Tuple[Any, float]] = {}
_SCAN_RESULT_TTL: float = 300.0   # 5분


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

@ttl_cache(3600)   # 1시간 캐시 — 실패 시 24시간 고착 방지 (기존 86400 → 3600)
def get_krx_code_map():
    """KRX 전체 상장 종목 코드↔이름 맵 반환.
    실패 시 빈 dict 반환 — resolve_ticker가 KR_STOCK_MAP 폴백으로 처리함.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}
    urls = [
        # HTTPS 우선, HTTP 폴백
        "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13",
        "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13",
    ]
    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=8, verify=False)
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
                        if name and code:
                            n2c[name] = code
                            c2n[code] = name
            if n2c:          # 파싱 성공 시 반환 (빈 결과면 다음 URL 시도)
                return n2c, c2n
        except Exception:
            continue
    return {}, {}            # 모두 실패 → resolve_ticker가 KR_STOCK_MAP 폴백 사용

def resolve_ticker(q: str):
    """종목명 / 코드 / 별칭 → (yfinance_ticker, market, display_name)

    우선순위 (빠른 오프라인 조회 우선, 외부 API는 최후 수단):
    1. COMMON_ALIASES  — 별칭·줄임말 (KR_STOCK_MAP으로 올바른 시장접미사 검증)
    2. US_STOCK_MAPPING — 한국어 미국 종목명
    3. 6자리 숫자 코드 — KR_STOCK_MAP 역조회 → KRX API 폴백
    4. 전체 ASCII    — US ticker 직접 입력
    5. KR_STOCK_MAP 완전 일치 (오프라인, KQ/KS 올바름)  ← 핵심 추가
    6. KRX API 완전 일치 (전체 상장 종목, 네트워크 필요)
    7. KRX API 전방 일치
    8. KR_STOCK_MAP 전방 일치 (KRX API 실패 시 최종 폴백)  ← 핵심 추가
    """
    q = q.strip()
    if not q:
        return None, None, None

    # ── 1. 별칭 / 줄임말 ─────────────────────────────────────────────
    if q in COMMON_ALIASES:
        code = COMMON_ALIASES[q]
        # KR_STOCK_MAP에서 시장접미사 확인 — KOSDAQ 종목(.KQ) 오분류 방지
        for name, tkr in KR_STOCK_MAP.items():
            if tkr.startswith(code + "."):
                return tkr, "KRX", q
        return f"{code}.KS", "KRX", q   # KR_STOCK_MAP 미적재면 KS 기본값

    # ── 2. 한국어 미국 종목명 ─────────────────────────────────────────
    if q in US_STOCK_MAPPING:
        return US_STOCK_MAPPING[q], "US", q

    # ── 3. 6자리 숫자 코드 ───────────────────────────────────────────
    if q.isdigit() and len(q) == 6:
        # KR_STOCK_MAP 역조회 (오프라인, 빠름)
        for name, tkr in KR_STOCK_MAP.items():
            if tkr.startswith(q + "."):
                return tkr, "KRX", name
        # KRX API 폴백 (네트워크 필요)
        _, c2n = get_krx_code_map()
        return f"{q}.KS", "KRX", c2n.get(q, q)

    # ── 3.5. 한국 시장 접미사(.KS/.KQ) 부착 코드 → KRX 직접 처리 ──────
    #   "041510.KQ" / "000660.KS" 처럼 접미사가 붙은 완전한 KR 티커가
    #   규칙 4(전체 ASCII → US)로 US 종목으로 오분류되는 것을 방지한다.
    #   (🔬 스캔 엔진 결과 클릭 시 전달되는 형식 — 직접 검색과 동일 동작 보장)
    qu = q.upper()
    if qu.endswith((".KS", ".KQ")):
        code = qu[:-3]
        if code.isdigit() and len(code) == 6:
            # 표시용 한글 종목명: KR_STOCK_MAP 역조회 → KRX API 폴백 (없으면 코드)
            for nm, tkr in KR_STOCK_MAP.items():
                if tkr.upper() == qu or tkr.startswith(code + "."):
                    return qu, "KRX", nm
            try:
                _, c2n = get_krx_code_map()
            except Exception:
                c2n = {}
            return qu, "KRX", c2n.get(code, code)

    # ── 4. 전체 ASCII → US ticker 직접 입력 ──────────────────────────
    if all(ord(c) < 128 for c in q):
        return q.upper(), "US", q.upper()

    # ── 5. KR_STOCK_MAP 완전 일치 (오프라인, KQ/KS 정확) ─────────────
    if q in KR_STOCK_MAP:
        return KR_STOCK_MAP[q], "KRX", q

    # ── 6~7. KRX API (전체 상장 종목 조회, 네트워크 필요) ────────────
    n2c, _ = get_krx_code_map()
    if q in n2c:
        return f"{n2c[q]}.KS", "KRX", q
    for name, code in n2c.items():
        if name.startswith(q):
            return f"{code}.KS", "KRX", name

    # ── 8. KR_STOCK_MAP 전방 일치 (KRX API 실패 시 최종 폴백) ────────
    for name, tkr in KR_STOCK_MAP.items():
        if name.startswith(q):
            return tkr, "KRX", name

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
            
        # Lightweight Charts가 인식할 수 있도록 날짜를 yyyy-mm-dd 문자열 또는 Unix Timestamp 형식으로 반환해야 함
        # 일봉은 %Y-%m-%d 문자열로, 분봉은 Unix Timestamp(초 단위)로 변환
        if interval == "1d":
            # tz 정보 제거 후 날짜 문자열 변환
            if hasattr(df2["Date"].dtype, "tz") and df2["Date"].dtype.tz:
                df2["Date"] = df2["Date"].dt.tz_localize(None)
            df2["Date"] = df2["Date"].dt.strftime("%Y-%m-%d")
        else:
            # pandas 버전에 따라 astype("int64")가 나노초(2.x) 또는 초(3.x)를 반환하는
            # 호환성 문제를 방지하기 위해 Timestamp.timestamp()를 사용해 항상 초 단위로 변환
            df2["Date"] = [int(ts.timestamp()) for ts in df2["Date"]]
            
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
    "리노공업": "058470.KQ", "알테오젠": "196170.KQ",
    # 업종별 흐름(_SECTOR_DEFAULT_STOCKS) 클릭 검색 지원 — KRX API 불필요
    "LS일렉트릭": "010120.KS", "HD현대일렉트릭": "267260.KS",
    "삼성중공업": "010140.KS", "LIG넥스원": "079550.KS",
    "S-Oil": "010950.KS", "현대제철": "004020.KS", "한국가스공사": "036460.KS",
    # 자주 검색되는 추가 종목
    "삼성전자우": "005935.KS", "현대자동차": "005380.KS", "카카오뱅크": "323410.KS",
    "카카오페이": "377300.KS", "삼성바이오": "207940.KS", "현대중공업": "329180.KS",
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

# ── 미국 주가 보조 API 키 (yfinance 실패 시 폴백) ────────────────────────────
_TIINGO_KEY = os.getenv("TIINGO_API_KEY",   "12ebd1feef89b6728cc15808864b7402449a5637")
_AV_KEY     = os.getenv("ALPHAVANTAGE_KEY", "E0ODFSRNDU4P9HDU")


def _tiingo_price(ticker: str, session_name: str) -> Optional[Tuple[float, float]]:
    """Tiingo IEX API로 미국 주식 현재가·전일종가 조회.
    - 프리/애프터마켓 → tngoLast (Tiingo 집계가, 시간외 지원)
    - 정규장           → last (IEX 실시간) 우선, 없으면 tngoLast
    반환: (price, prev_close) or None
    """
    try:
        url = f"https://api.tiingo.com/iex/{ticker.upper()}?token={_TIINGO_KEY}"
        r = requests.get(url, timeout=5, headers={"Accept": "application/json"})
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, list):
            data = data[0] if data else {}
        if not data:
            return None
        prev_close = float(data.get("prevClose") or 0)
        if session_name in ("프리마켓", "애프터마켓", "데이마켓"):
            # 시간외 세션: tngoLast(집계) 우선
            price = float(data.get("tngoLast") or data.get("last") or 0)
        else:
            # 정규장: IEX 실시간 last 우선
            price = float(data.get("last") or data.get("tngoLast") or 0)
        if price > 0:
            return price, prev_close
    except Exception:
        pass
    return None


def _av_price(ticker: str) -> Optional[Tuple[float, float]]:
    """AlphaVantage GLOBAL_QUOTE로 미국 주식 현재가·전일종가 조회 (무료 플랜).
    반환: (price, prev_close) or None
    """
    try:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={ticker.upper()}&apikey={_AV_KEY}"
        )
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return None
        q = r.json().get("Global Quote", {})
        price     = float(q.get("05. price")          or 0)
        prev_close = float(q.get("08. previous close") or 0)
        if price > 0:
            return price, prev_close
    except Exception:
        pass
    return None


_us_price_fetcher = None   # 싱글톤 — 최초 호출 시 초기화

def _get_us_price_fetcher():
    """USStockPriceFetcher 싱글톤 반환 (lazy init)."""
    global _us_price_fetcher
    if _us_price_fetcher is None:
        try:
            import sys as _sys, os as _os
            _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from us_price_fetcher import USStockPriceFetcher
            _us_price_fetcher = USStockPriceFetcher()
        except Exception as _e:
            pass  # 로드 실패 시 None 유지 → fallback 경로 사용
    return _us_price_fetcher


# price_type → 한국어 세션 라벨 매핑
_PRICE_TYPE_LABEL = {
    "overnight":      "오버나이트",   # Blue Ocean ATS (8 PM~4 AM ET)
    "pre_market":     "프리마켓",     # 4 AM~9:30 AM ET
    "post_market":    "애프터마켓",   # 4 PM~8 PM ET
    "extended_hours": "시간외",
    "real_time":      "정규장",
    "regular_close":  "정규장",
    "last_close":     "장마감",
}


def get_us_realtime_price(ticker_obj) -> Tuple[float, str, float]:
    """
    미국 주식 현재 세션에 맞는 최신 가격·세션명·전일종가 반환.

    [우선순위]
      1. USStockPriceFetcher  — overnightMarketPrice(Blue Ocean ATS) / preMarketPrice /
                               postMarketPrice / regularMarketPrice 를 세션별 자동 선택
      2. yfinance info        — 폴백 (fetcher 초기화 실패 시)
      3. Tiingo IEX           — 폴백
      4. AlphaVantage         — 폴백

    반환: (price, session_label_ko, prev_close)
      - session_label_ko: "오버나이트" | "프리마켓" | "정규장" | "애프터마켓" | "장마감"
    """
    ticker_str = getattr(ticker_obj, 'ticker', None) or ""

    # ── ① USStockPriceFetcher (overnightMarketPrice 포함 전체 세션 지원) ──────
    if ticker_str:
        fetcher = _get_us_price_fetcher()
        if fetcher is not None:
            try:
                result = fetcher.fetch(ticker_str)
                if result and result.price > 0:
                    session_label = _PRICE_TYPE_LABEL.get(
                        result.price_type,
                        result.session.label_ko(),
                    )
                    return float(result.price), session_label, float(result.prev_close or 0)
            except Exception:
                pass

    # ── ② yfinance 폴백 ──────────────────────────────────────────────────────
    try:
        from zoneinfo import ZoneInfo
        us_tz = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        us_tz = pytz.timezone("America/New_York")

    now_et     = dt.now(us_tz)
    time_float = now_et.hour + now_et.minute / 60.0

    info         = ticker_obj.info
    regular_price = info.get("regularMarketPrice")
    current_price = info.get("currentPrice")
    pre_price     = info.get("preMarketPrice")
    post_price    = info.get("postMarketPrice")
    prev_close    = info.get("previousClose")

    fast_last_price = None
    try:
        fi = ticker_obj.fast_info
        if hasattr(fi, 'last_price'):
            fast_last_price = fi.last_price
        if not prev_close and hasattr(fi, 'previous_close'):
            prev_close = fi.previous_close
    except Exception:
        pass

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
        session_name = "오버나이트"
        price = post_price or regular_price or prev_close

    # ── ③ Tiingo / AlphaVantage 폴백 ─────────────────────────────────────────
    if not price and ticker_str:
        t_res = _tiingo_price(ticker_str, session_name)
        if t_res:
            price, t_prev = t_res
            if not prev_close and t_prev:
                prev_close = t_prev
    if not price and ticker_str:
        av_res = _av_price(ticker_str)
        if av_res:
            price, av_prev = av_res
            if not prev_close and av_prev:
                prev_close = av_prev

    if not price:
        price = prev_close or 0.0

    return float(price), session_name, float(prev_close) if prev_close else 0.0


@ttl_cache(120)  # 2분 캐시 — 장중 수급 변화 대응
def fetch_investor_flow(ticker: str) -> dict:
    """투자자별 순매수 + 외국인 보유율 조회 (KRX 전용).

    1차: Toss 증권 API (유연한 응답 구조 파싱)
    2차: Naver Finance HTML 파싱 (폴백)

    Returns:
        성공: {"ok": True, "date": ..., "개인": ..., ...}
        실패: {"ok": False, "reason": "<상세 원인>"}
    """
    code = str(ticker).replace(".KS", "").replace(".KQ", "").strip()
    if not code.isdigit() or len(code) != 6:
        return {"ok": False, "reason": "KRX 6자리 코드 아님"}

    # ── 공통 파서: 다양한 응답 구조에서 행 리스트 추출 ────────────────────
    def _extract_rows(raw) -> list:
        """body / result / data / tradingTrend 등 다양한 키 자동 탐색"""
        if isinstance(raw, list):
            return raw
        if not isinstance(raw, dict):
            return []
        # 1레벨 탐색
        for k1 in ("body", "result", "data", "list", "items",
                   "content", "tradingTrend", "records", "rows"):
            v = raw.get(k1)
            if isinstance(v, list) and v:
                return v
            if isinstance(v, dict):
                # 2레벨 탐색 ({"body": {"list": [...]}})
                for k2 in ("list", "items", "tradingTrend", "body",
                           "content", "records", "data"):
                    inner = v.get(k2)
                    if isinstance(inner, list) and inner:
                        return inner
        return []

    # ── 공통 파서: 행 dict → 반환 dict 변환 ──────────────────────────────
    def _parse_row(row: dict, source: str = "") -> dict:
        def _i(*keys):
            for k in keys:
                v = row.get(k)
                if v is not None:
                    try: return int(float(str(v).replace(",", "")))
                    except: continue
            return 0
        def _f(*keys):
            for k in keys:
                v = row.get(k)
                if v is not None:
                    try: return round(float(str(v).replace(",", "")), 2)
                    except: continue
            return 0.0
        date_val = (row.get("baseDate") or row.get("date") or
                    row.get("tradeDate") or row.get("stndDt") or
                    row.get("trdDt") or "")
        return {
            "ok":         True,
            "source":     source,
            "date":       str(date_val),
            # 현재 필드명 + 구버전/대체 필드명 순으로 시도
            "개인":       _i("netIndividualsBuyVolume",  "individualNetBuyVolume",  "retlNetBuyTrdvol"),
            "외국인":     _i("netForeignerBuyVolume",    "foreignerNetBuyVolume",   "frgnNetBuyTrdvol"),
            "기관":       _i("netInstitutionBuyVolume",  "institutionNetBuyVolume", "instNetBuyTrdvol"),
            "연기금":     _i("netPensionFundBuyVolume",  "pensionFundNetBuyVolume", "pnsionNetBuyTrdvol"),
            "금융투자":   _i("netFinancialInvestmentBuyVolume", "financialInvestmentNetBuyVolume"),
            "투신":       _i("netTrustBuyVolume",        "trustNetBuyVolume"),
            "사모":       _i("netPrivateEquityFundBuyVolume", "privateEquityNetBuyVolume"),
            "보험":       _i("netInsuranceBuyVolume",    "insuranceNetBuyVolume"),
            "은행":       _i("netBankBuyVolume",         "bankNetBuyVolume"),
            "기타금융":   _i("netOtherFinancialInstitutionsBuyVolume", "otherFinancialNetBuyVolume"),
            "기타법인":   _i("netOtherCorporationBuyVolume", "otherCorporationNetBuyVolume"),
            "외국인비율": _f("foreignerRatio", "foreignOwnershipRatio", "frgnHoldingRate"),
        }

    # ══════════════════════════════════════════════════════════════════════
    # 1차: Toss 증권 API
    # ══════════════════════════════════════════════════════════════════════
    toss_url = (
        "https://wts-info-api.tossinvest.com/api/v1/stock-infos/trade/trend/trading-trend"
        f"?productCode=A{code}&size=60"
    )
    toss_headers = {
        # 최신 Chrome/Windows UA — 구버전 Mac UA 교체 (Toss UA 필터링 대응)
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Origin":          "https://tossinvest.com",
        "Referer":         f"https://tossinvest.com/stocks/A{code}/order",  # 종목별 URL
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
        "Sec-Fetch-Dest":  "empty",
        "Sec-Fetch-Mode":  "cors",
        "Sec-Fetch-Site":  "same-site",
    }
    toss_reason = ""
    try:
        resp = requests.get(toss_url, headers=toss_headers, timeout=8)
        print(f"[수급|Toss] HTTP {resp.status_code} | CT: {resp.headers.get('Content-Type','?')[:40]}")
        resp.raise_for_status()

        raw = resp.json()
        top_keys = list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
        print(f"[수급|Toss] 응답 최상위 키: {top_keys}")

        rows = _extract_rows(raw)
        print(f"[수급|Toss] 파싱된 행 수: {len(rows)}")

        if rows:
            row0 = rows[0]
            print(f"[수급|Toss] 첫 행 키 샘플: {list(row0.keys())[:10] if isinstance(row0, dict) else '?'}")
            result = _parse_row(row0, source="toss")
            print(f"[수급|Toss] 성공 — 개인:{result['개인']:+,} 외국인:{result['외국인']:+,} 기관:{result['기관']:+,}")
            return result

        # 빈 응답: 원인 세분화
        print(f"[수급|Toss] 빈 응답 샘플: {str(raw)[:500]}")
        if isinstance(raw, dict):
            if "body" in raw:
                toss_reason = "장 개시 전 또는 해당일 거래 없음 (body 빈 배열)"
            else:
                toss_reason = f"API 응답 구조 불일치 (최상위 키: {top_keys})"
        else:
            toss_reason = f"API 응답 형식 비정상 ({type(raw).__name__})"

    except requests.exceptions.Timeout:
        toss_reason = "Toss API 타임아웃 (8초 초과)"
    except requests.exceptions.HTTPError as e:
        sc = e.response.status_code if e.response else "?"
        toss_reason = {
            401: "Toss API 인증 필요 (401)",
            403: "Toss API 접근 차단 (403)",
            429: "Toss API 요청 한도 초과 (429)",
        }.get(sc, f"Toss API HTTP {sc}")
    except Exception as e:
        toss_reason = f"Toss 조회 오류: {type(e).__name__}"
        print(f"[수급|Toss] 예외: {type(e).__name__}: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # 2차: Naver Finance HTML 파싱 (폴백)
    # ══════════════════════════════════════════════════════════════════════
    print(f"[수급|Toss] 실패 ({toss_reason}) → Naver 폴백 시도")
    try:
        naver_url = f"https://finance.naver.com/item/investor.naver?code={code}"
        nheaders = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "Referer":        "https://finance.naver.com/",
            "Accept-Language": "ko-KR,ko;q=0.9",
        }
        nresp = requests.get(naver_url, headers=nheaders, timeout=8)
        print(f"[수급|Naver] HTTP {nresp.status_code}")
        nresp.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(nresp.text, "html.parser")

        # 투자자별 순매수 테이블 탐색 (class="type2" 또는 첫 번째 data table)
        table = soup.find("table", class_="type2") or soup.find("table")
        if not table:
            print("[수급|Naver] 테이블 없음")
            return {"ok": False, "reason": toss_reason}

        all_trs = table.find_all("tr")
        print(f"[수급|Naver] 테이블 행 수: {len(all_trs)}")

        def _nv(td_el):
            txt = td_el.get_text(strip=True).replace(",", "").replace("+", "").replace(" ", "")
            try: return int(txt)
            except: return 0

        # 헤더 제외, 데이터 행 탐색 (첫 번째 유효 행 = 최신 거래일)
        for tr in all_trs[2:]:
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            date_txt = tds[0].get_text(strip=True)
            # 날짜 형식 확인 (예: "2024.05.06")
            if not any(c.isdigit() for c in date_txt):
                continue

            개인 = _nv(tds[1])
            외국인 = _nv(tds[2])
            기관   = _nv(tds[3])

            # 유효 데이터 확인 (모두 0이면 스킵)
            if 개인 == 0 and 외국인 == 0 and 기관 == 0:
                continue

            연기금   = _nv(tds[4])  if len(tds) > 4  else 0
            금융투자 = _nv(tds[5])  if len(tds) > 5  else 0
            보험     = _nv(tds[6])  if len(tds) > 6  else 0
            투신     = _nv(tds[7])  if len(tds) > 7  else 0
            사모     = _nv(tds[8])  if len(tds) > 8  else 0
            은행     = _nv(tds[9])  if len(tds) > 9  else 0
            기타금융 = _nv(tds[10]) if len(tds) > 10 else 0
            기타법인 = _nv(tds[11]) if len(tds) > 11 else 0

            # 날짜를 YYYYMMDD 형식으로 변환 (2024.05.06 → 20240506)
            date_clean = date_txt.replace(".", "").replace(" ", "").strip()
            print(f"[수급|Naver] 성공 — {date_txt} | 개인:{개인:+,} 외국인:{외국인:+,} 기관:{기관:+,}")
            return {
                "ok":         True,
                "source":     "naver",
                "date":       date_clean,
                "개인":       개인,
                "외국인":     외국인,
                "기관":       기관,
                "연기금":     연기금,
                "금융투자":   금융투자,
                "투신":       투신,
                "사모":       사모,
                "보험":       보험,
                "은행":       은행,
                "기타금융":   기타금융,
                "기타법인":   기타법인,
                "외국인비율": 0.0,  # Naver investor.naver에는 보유율 미포함
            }

        print("[수급|Naver] 유효 데이터 행 없음 (장 개시 전 또는 휴장일)")
    except ImportError:
        print("[수급|Naver] BeautifulSoup 미설치 — 폴백 불가")
    except Exception as e:
        print(f"[수급|Naver] 예외: {type(e).__name__}: {e}")

    # 두 소스 모두 실패
    return {"ok": False, "reason": toss_reason}


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

def to_toss_product_code(ticker: str, market: str = None) -> str:
    """KRX ticker → 토스증권 productCode 즉시 변환 (국내 전용)."""
    if not ticker:
        return ""
    if ticker.endswith((".KS", ".KQ")):
        return "A" + ticker.split(".")[0]
    return ""


# ── 토스증권 API 공통 요청 헤더 ──────────────────────────────────────────
_TOSS_API_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
    "Origin": "https://www.tossinvest.com",
    "Referer": "https://www.tossinvest.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}
_TOSS_AI_BATCH_URL  = "https://wts-info-api.tossinvest.com/api/v1/dashboard/wts/overview/ai-signals"
_TOSS_AI_DETAIL_URL = "https://wts-info-api.tossinvest.com/api/v1/dashboard/wts/overview/ai-signals/detail"
# MD 분석: 랭킹 API의 productCode가 해외 종목의 가장 신뢰성 높은 식별자
_TOSS_RANKING_URL   = "https://wts-cert-api.tossinvest.com/api/v2/dashboard/wts/overview/ranking"

# ── 토스증권 API 전용 HTTP Session ──────────────────────────────────────────
# TCP/TLS 연결을 재사용(keep-alive)하여 동일 호스트 반복 호출 시
# 핸드셰이크 오버헤드(100~300ms/콜)를 제거한다.
# wts-info-api / wts-cert-api 두 호스트 각 최대 8개 연결 풀 유지.
_toss_session = requests.Session()
_toss_session.headers.update(_TOSS_API_HEADERS)
_toss_http_adapter = requests.adapters.HTTPAdapter(
    pool_connections=4,   # 고유 호스트별 연결 풀 수
    pool_maxsize=8,       # 풀당 최대 연결 수
    max_retries=0,        # 재시도는 전략 레벨에서 직접 제어
)
_toss_session.mount("https://", _toss_http_adapter)

# ── 랭킹 캐시 (빈 결과 캐시 방지용 수동 관리) ───────────────────────────────
# ttl_cache는 실패(빈 dict)도 캐싱하므로 수동 캐시로 교체.
# 성공한 결과만 저장해 API 오류 후 즉시 재시도가 가능하다.
_TOSS_RANKING_CACHE: dict      = {}
_TOSS_RANKING_CACHE_TS: float  = 0.0
_TOSS_RANKING_CACHE_TTL: float = 60.0   # 1분

# ── US productCode 캐시 (성공 결과만 저장; 빈 문자열 캐싱 방지) ─────────────
# ttl_cache는 실패("")도 캐싱해서 1시간 동안 재시도 불가 → 수동 캐시로 교체.
_TOSS_US_PC_CACHE: dict      = {}     # {ticker_upper: productCode}
_TOSS_US_PC_CACHE_TS: dict   = {}     # {ticker_upper: monotonic_timestamp}
_TOSS_US_PC_CACHE_TTL: float = 3600.0

# ── AI 요약 캐시 (supported=True 결과만 저장; supported=False 캐싱 방지) ──────
# supported=False를 캐싱하면 productCode 검색 로직 개선 후에도 5분간 재시도 불가.
_TOSS_AI_CACHE: dict      = {}     # {(ticker_upper, market): result_dict}
_TOSS_AI_CACHE_TS: dict   = {}     # {(ticker_upper, market): monotonic_timestamp}
_TOSS_AI_CACHE_TTL: float = 300.0


def _fetch_toss_us_ranking_productcodes() -> dict:
    """
    Toss 해외 실시간 랭킹 3개를 병렬로 조회해 ticker→productCode 완전 매핑을 반환한다.

    핵심 동작:
    - ETF·레버리지 상품(NVDU, BEX 등): name 필드 == 영문 ticker → 즉시 저장
    - 개별주(캘러보 그로우어스=CVGW, 마이크론=MU 등): name 필드가 한국어 →
      stock-infos/{productCode} API로 역조회하여 symbol(ticker)을 확보
    - 빈 결과(API 전체 실패)는 캐시하지 않아 즉시 재시도 가능

    Toss productCode 형식: US20020322001 / NAS0230913004 / AMX0251113001 ...
    (NAS.CVGW 같은 단순 접두사·티커 조합 형식이 아님에 주의)

    반환값: {ticker_upper: productCode, ...}
    """
    global _TOSS_RANKING_CACHE, _TOSS_RANKING_CACHE_TS

    # ── 캐시 히트 ────────────────────────────────────────────────────────
    now = time.monotonic()
    if _TOSS_RANKING_CACHE and (now - _TOSS_RANKING_CACHE_TS) < _TOSS_RANKING_CACHE_TTL:
        return _TOSS_RANKING_CACHE

    # ── 단일 랭킹 조회 헬퍼 (스레드 내 실행) ─────────────────────────────
    def _fetch_one(ranking_id: str) -> list:
        body = {
            "id": ranking_id,
            "filters": [
                "KRX_MANAGEMENT_STOCK",
                "MARKET_CAP_GREATER_THAN_50M",
                "STOCKS_PRICE_GREATER_THAN_ONE_DOLLAR",
            ],
            "duration": "realtime",
            "tag": "us",
        }
        try:
            resp = _toss_session.post(
                _TOSS_RANKING_URL,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json().get("result", {}).get("products", [])
        except Exception as e:
            print(f"[Toss US 랭킹] {ranking_id} 오류: {e}")
        return []

    # ── 3개 랭킹 병렬 조회 ───────────────────────────────────────────────
    ticker_to_code: dict = {}     # 최종 결과: ticker_upper → productCode
    korean_codes: list  = []      # 한국어명 종목 productCode (역조회 대상)

    ranking_ids = ("biggest_total_amount", "biggest_change_rate", "biggest_volume")
    seen_codes: set = set()        # 중복 productCode 방지
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            for products in ex.map(_fetch_one, ranking_ids, timeout=7):
                for p in products:
                    name = (p.get("name") or "").strip()
                    code = p.get("productCode") or ""
                    if not (name and code) or code in seen_codes:
                        continue
                    seen_codes.add(code)
                    # 영문명 종목(ETF·레버리지): name == ticker → 즉시 저장
                    # 한국어명 종목(개별주): name ≠ ticker → stock-infos 역조회 필요
                    if all(ord(c) < 128 for c in name):
                        ticker_to_code[name.upper()] = code
                    else:
                        korean_codes.append(code)
    except Exception as e:
        print(f"[Toss US 랭킹] 병렬 조회 오류: {e}")

    # ── 한국어명 종목: stock-infos/{productCode} 역조회로 ticker 확보 ─────
    # 예: productCode=US20020322001 → symbol="CVGW" (캘러보 그로우어스)
    #     productCode=US19890516001 → symbol="MU"   (마이크론 테크놀로지)
    # stock-infos/search?query={ticker} 는 항상 null 반환 → 역조회만 가능
    if korean_codes:
        def _resolve_ticker(pc: str) -> tuple:
            try:
                r = _toss_session.get(
                    f"https://wts-info-api.tossinvest.com/api/v1/stock-infos/{pc}",
                    timeout=4,
                )
                if r.status_code == 200:
                    result = r.json().get("result") or {}
                    sym = (result.get("symbol") or "").upper().strip()
                    if sym:
                        return (pc, sym)
            except Exception:
                pass
            return (pc, "")

        try:
            # pool_maxsize=8이므로 실질 동시 연결 수는 8개.
            # 53개 한국어명 종목 → ceil(53/8)=7라운드 × 4s = ~28s → timeout=35s
            _workers = min(8, len(korean_codes))
            with concurrent.futures.ThreadPoolExecutor(max_workers=_workers) as ex:
                _futs = {ex.submit(_resolve_ticker, pc): pc for pc in korean_codes}
                for _fut in concurrent.futures.as_completed(_futs, timeout=35):
                    try:
                        pc, sym = _fut.result()
                        if sym:
                            ticker_to_code[sym] = pc
                    except Exception:
                        pass
        except concurrent.futures.TimeoutError:
            print("[Toss US 랭킹] 한국어명 역조회 timeout (부분 결과 사용)")
        except Exception as e:
            print(f"[Toss US 랭킹] 한국어명 역조회 오류: {e}")

    # 성공한 경우만 캐시 갱신 (실패 시 빈 dict 캐싱 방지)
    if ticker_to_code:
        _TOSS_RANKING_CACHE    = ticker_to_code
        _TOSS_RANKING_CACHE_TS = now
        print(f"[Toss US 랭킹] 캐시 갱신: {len(ticker_to_code)}개")
    else:
        print("[Toss US 랭킹] 전체 실패 — 이전 캐시 유지")

    return ticker_to_code


def _toss_batch_api(product_code: str) -> dict:
    """
    Toss AI signals 배치 API 호출.
    POST /api/v1/dashboard/wts/overview/ai-signals
    {"productCodes": [product_code], "filters": []}
    Session keep-alive 사용, timeout 5s (기존 8s).
    """
    try:
        resp = _toss_session.post(
            _TOSS_AI_BATCH_URL,
            json={"productCodes": [product_code], "filters": []},
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        if resp.status_code != 200:
            return {"ai_summary": "", "product_code": product_code, "supported": True}
        signals = resp.json().get("result", {}).get("signals", [])
        for s in signals:
            if isinstance(s, dict) and s.get("productCode") == product_code:
                txt = s.get("reasoningDescription", "")
                if txt:
                    print(f"[Toss Batch] {product_code} → {txt[:30]}...")
                return {"ai_summary": txt, "product_code": product_code, "supported": True}
        return {"ai_summary": "", "product_code": product_code, "supported": True}
    except Exception as e:
        print(f"[Toss Batch] 실패 ({product_code}): {e}")
        return {"ai_summary": "", "product_code": product_code, "supported": True}


def _toss_detail_api(product_code: str, product_type: str) -> str:
    """
    Toss AI signals 상세 API 호출.
    GET /detail?productCode={}&productType={}
    Session keep-alive 사용, timeout 4s (기존 6s).
    성공 시 요약 텍스트, 실패 시 "".
    """
    try:
        resp = _toss_session.get(
            _TOSS_AI_DETAIL_URL,
            params={"productCode": product_code, "productType": product_type},
            timeout=4,
        )
        if resp.status_code != 200:
            return ""
        desc = (resp.json().get("result") or {}).get("reasoning", {}).get("description", "")
        if desc:
            print(f"[Toss Detail] {product_code}/{product_type} → {desc[:30]}...")
        return desc or ""
    except Exception:
        return ""


def _parse_toss_candidates(data, ticker_upper: str) -> str:
    """
    Toss 검색 API 응답(JSON)에서 ticker에 해당하는 productCode 추출.
    응답 구조가 다양할 수 있어 여러 키를 탐색.
    """
    # 후보 리스트 추출
    candidates: list = []
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        root = data.get("result") or data
        if isinstance(root, list):
            candidates = root
        elif isinstance(root, dict):
            for key in ("stocks", "products", "items", "data", "list", "body",
                        "content", "searchResults", "results"):
                val = root.get(key)
                if isinstance(val, list):
                    candidates = val
                    break

    # ticker 매칭 — Toss API가 사용하는 다양한 필드명 전부 시도
    for item in candidates:
        if not isinstance(item, dict):
            continue
        sym = (
            item.get("stockCode") or item.get("ticker") or item.get("symbol")
            or item.get("symbolCode") or item.get("code")
            or item.get("tickerSymbol") or item.get("shortCode")
            or item.get("stockSymbol") or item.get("nameEn") or ""
        ).upper().strip()
        if sym == ticker_upper:
            code = item.get("productCode") or item.get("id") or ""
            if code and isinstance(code, str):
                return code
    return ""


def _deep_find_product_code(obj, ticker_upper: str) -> str:
    """JSON 객체에서 ticker에 매칭되는 productCode를 재귀 탐색 (웹 스크래핑용)."""
    if isinstance(obj, dict):
        sym = (
            obj.get("stockCode") or obj.get("ticker") or obj.get("symbol")
            or obj.get("symbolCode") or obj.get("code")
            or obj.get("tickerSymbol") or obj.get("shortCode")
            or obj.get("stockSymbol") or ""
        ).upper()
        if sym == ticker_upper:
            code = obj.get("productCode") or obj.get("id") or ""
            if code and isinstance(code, str):
                return code
        for v in obj.values():
            r = _deep_find_product_code(v, ticker_upper)
            if r:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _deep_find_product_code(item, ticker_upper)
            if r:
                return r
    return ""


def _fetch_toss_us_product_code(ticker: str) -> str:
    """
    미국 주식 ticker → 토스증권 productCode 다단계 조회.

    전략 순서:
      0) 랭킹 API 캐시에서 ticker 매칭 (ETF·레버리지 + 한국어명 개별주 모두 커버)
         캐시에는 영문명 종목(name==ticker 직접)과 한국어명 종목(stock-infos 역조회)
         모두 포함됨 → CVGW, MU 같은 한국어명 종목도 여기서 해결됨.
      1) Toss 검색 API GET 10개 병렬 (stock-infos, search, products namespaces)
      2) Toss 검색 API POST 4개 병렬
      3) tossinvest.com 웹 페이지 __NEXT_DATA__ 스크래핑
         (/stock/{ticker} 직접 상품 페이지 포함)

    성공한 결과만 캐시 저장 → 빈 문자열("")은 절대 캐시되지 않음.
    Toss productCode 형식: US20020322001 / NAS0230913004 / AMX0251113001 ...
    (NAS.{TICKER} 형식이 아님에 주의)
    """
    global _TOSS_US_PC_CACHE, _TOSS_US_PC_CACHE_TS
    import re as _re, json as _json
    ticker_upper = ticker.upper().strip()

    # ── 수동 캐시 히트 (성공 결과만 저장됨; 실패는 캐시 안함) ─────────────
    _now = time.monotonic()
    if ticker_upper in _TOSS_US_PC_CACHE and \
            (_now - _TOSS_US_PC_CACHE_TS.get(ticker_upper, 0)) < _TOSS_US_PC_CACHE_TTL:
        return _TOSS_US_PC_CACHE[ticker_upper]

    def _save_and_return(code: str, src: str) -> str:
        """캐시에 저장하고 반환하는 헬퍼."""
        _TOSS_US_PC_CACHE[ticker_upper] = code
        _TOSS_US_PC_CACHE_TS[ticker_upper] = _now
        print(f"[Toss US코드] {ticker} → {code} ({src})")
        return code

    # ── 전략 0: 랭킹 API 캐시에서 name 기준 매칭 ────────────────────────
    # ETF·레버리지 상품(NVDU, BEX, SMCY 등)은 name == ticker라 즉시 매칭됨.
    # 개별 종목(마이크론=MU, CVGW 등)은 name이 한국어라 미매칭 → 전략 0.5~3으로 fallback.
    try:
        ranking_cache = _fetch_toss_us_ranking_productcodes()
        if ticker_upper in ranking_cache:
            return _save_and_return(ranking_cache[ticker_upper], "랭킹 캐시")
    except Exception as _e:
        print(f"[Toss US코드] 랭킹 캐시 조회 오류: {_e}")

    # ── 전략 1: GET 검색 엔드포인트 10개를 병렬 실행 ────────────────────
    # CVGW처럼 한국어명 종목은 랭킹 캐시에서 못 찾고 여기서 검색해야 함.
    # 어떤 엔드포인트가 성공하든 5s 이내에 결과 반환.
    get_attempts = [
        # stock-infos 네임스페이스
        ("https://wts-info-api.tossinvest.com/api/v1/stock-infos/search",
         {"query": ticker_upper, "size": 10}),
        ("https://wts-info-api.tossinvest.com/api/v1/stock-infos/search",
         {"query": ticker_upper, "size": 10, "market": "OVERSEAS"}),
        # search 네임스페이스
        ("https://wts-info-api.tossinvest.com/api/v1/search/stocks",
         {"query": ticker_upper, "size": 10}),
        ("https://wts-info-api.tossinvest.com/api/v1/search",
         {"query": ticker_upper, "size": 10, "market": "OVERSEAS"}),
        ("https://wts-info-api.tossinvest.com/api/v1/search",
         {"query": ticker_upper, "size": 10, "productType": "FOREIGN_STOCKS"}),
        ("https://wts-info-api.tossinvest.com/api/v1/search",
         {"query": ticker_upper, "size": 10}),
        # products 네임스페이스
        ("https://wts-info-api.tossinvest.com/api/v1/products/search",
         {"query": ticker_upper, "size": 10, "market": "OVERSEAS"}),
        ("https://wts-info-api.tossinvest.com/api/v1/products/search",
         {"query": ticker_upper, "size": 10}),
        ("https://wts-info-api.tossinvest.com/api/v2/search",
         {"query": ticker_upper, "size": 10}),
        ("https://wts-info-api.tossinvest.com/api/v1/search/products",
         {"query": ticker_upper, "size": 10}),
    ]

    def _do_get(url_params):
        _url, _params = url_params
        try:
            r = _toss_session.get(_url, params=_params, timeout=5)
            if r.status_code == 200:
                return _parse_toss_candidates(r.json(), ticker_upper)
        except Exception:
            pass
        return ""

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(get_attempts)) as _ex:
            _futs = {_ex.submit(_do_get, item): item for item in get_attempts}
            for _fut in concurrent.futures.as_completed(_futs, timeout=7):
                try:
                    code = _fut.result()
                    if code:
                        return _save_and_return(code, "GET 병렬")
                except Exception:
                    pass
    except concurrent.futures.TimeoutError:
        print(f"[Toss US코드] {ticker}: GET 병렬 timeout")

    # ── 전략 2: POST 검색 엔드포인트 4개를 병렬 실행 ─────────────────────
    post_attempts = [
        ("https://wts-info-api.tossinvest.com/api/v1/search",
         {"query": ticker_upper, "market": "OVERSEAS", "size": 10}),
        ("https://wts-info-api.tossinvest.com/api/v1/search",
         {"query": ticker_upper, "productType": "FOREIGN_STOCKS", "size": 10}),
        ("https://wts-info-api.tossinvest.com/api/v1/search/stocks",
         {"query": ticker_upper, "size": 10}),
        ("https://wts-info-api.tossinvest.com/api/v1/products/search",
         {"query": ticker_upper, "market": "OVERSEAS", "size": 10}),
    ]

    def _do_post(url_body):
        _url, _body = url_body
        try:
            r = _toss_session.post(
                _url, json=_body,
                headers={"Content-Type": "application/json"}, timeout=5,
            )
            if r.status_code == 200:
                return _parse_toss_candidates(r.json(), ticker_upper)
        except Exception:
            pass
        return ""

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(post_attempts)) as _ex:
            _futs = {_ex.submit(_do_post, item): item for item in post_attempts}
            for _fut in concurrent.futures.as_completed(_futs, timeout=7):
                try:
                    code = _fut.result()
                    if code:
                        return _save_and_return(code, "POST 병렬")
                except Exception:
                    pass
    except concurrent.futures.TimeoutError:
        print(f"[Toss US코드] {ticker}: POST 병렬 timeout")

    # ── 전략 3: 웹 페이지 __NEXT_DATA__ 스크래핑 ────────────────────────
    # /stock/{ticker} 직접 상품 페이지를 최우선으로 시도 (가장 정확한 JSON 포함)
    scrape_urls = [
        f"https://www.tossinvest.com/stock/{ticker_upper}",
        f"https://www.tossinvest.com/search?q={ticker_upper}",
        f"https://www.tossinvest.com/search?query={ticker_upper}",
    ]
    for surl in scrape_urls:
        try:
            resp = _toss_session.get(
                surl,
                headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
                timeout=6,
            )
            if resp.status_code == 200:
                m = _re.search(
                    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                    resp.text, _re.DOTALL
                )
                if m:
                    nd = _json.loads(m.group(1))
                    code = _deep_find_product_code(nd, ticker_upper)
                    if code:
                        return _save_and_return(code, "__NEXT_DATA__")
        except Exception as e:
            print(f"[Toss US코드] 스크래핑 {surl} 오류: {e}")

    print(f"[Toss US코드] {ticker}: 전 전략 실패 — productCode 조회 불가 (캐시 안함)")
    return ""


def fetch_toss_ai_summary(ticker: str, market: str) -> dict:
    """
    토스증권 AI 요약 조회 — KRX + US/해외 모두 지원.

    KRX 전략:
      · productCode "A{code}" 즉시 변환 → 배치 API

    US/해외 전략 (성능 최적화 순서):
      0) 랭킹 캐시(60s TTL) name 매칭 → productCode 즉시 확보 후 배치 API
         성공 시 전략 1~3 완전 스킵 (ETF·레버리지·랭킹 상위 종목)
      1) 상세 API(detail)에 ticker + productType 직접 전달 (6가지 병렬)
      2) 배치 API에 ticker 직접 전달
      3) 검색 API(_fetch_toss_us_product_code, 성공만 캐싱)로
         productCode 획득 → 배치 API + 상세 API 재시도

    캐시 정책: supported=True 결과만 300s 캐싱.
               supported=False(미지원·조회실패)는 캐시 안함 → 즉시 재시도 가능.
    실패 시 ai_summary="" 반환 — UI 절대 깨지지 않음.
    """
    _ticker_upper = ticker.upper().strip()
    _cache_key = (_ticker_upper, market)
    _now = time.monotonic()

    # ── 수동 캐시 히트 (supported=True 결과만 저장됨) ─────────────────────
    if _cache_key in _TOSS_AI_CACHE and \
            (_now - _TOSS_AI_CACHE_TS.get(_cache_key, 0)) < _TOSS_AI_CACHE_TTL:
        return _TOSS_AI_CACHE[_cache_key]

    def _cache_ok(result: dict) -> dict:
        """supported=True 결과를 캐시에 저장 후 반환."""
        if result.get("supported", False):
            _TOSS_AI_CACHE[_cache_key] = result
            _TOSS_AI_CACHE_TS[_cache_key] = _now
        return result

    # ── KRX ──────────────────────────────────────────────────────────────
    if market == "KRX":
        code = to_toss_product_code(ticker, market)
        if not code:
            return {"ai_summary": "", "product_code": "", "supported": False}
        return _cache_ok(_toss_batch_api(code))

    # ── US / 해외 ─────────────────────────────────────────────────────────
    ticker_upper = _ticker_upper

    # 전략 0: 랭킹 캐시에서 productCode 즉시 확보 (전략 1~2의 헛된 7회 API 콜 스킵)
    # 랭킹 상위 ETF·레버리지 상품(NVDU, BEX, SMCY 등)은 이 단계에서 완료
    try:
        ranking_cache = _fetch_toss_us_ranking_productcodes()
        if ticker_upper in ranking_cache:
            code = ranking_cache[ticker_upper]
            result = _toss_batch_api(code)
            if result.get("ai_summary"):
                return _cache_ok(result)
            # productCode는 확보했으나 토스가 아직 미분석
            return _cache_ok({"ai_summary": "", "product_code": code, "supported": True})
    except Exception as _e:
        print(f"[Toss AI] 랭킹 캐시 오류: {_e}")

    # 전략 1: ticker 자체를 productCode로 6가지 productType에 병렬 시도
    # 직렬 6×4s=24s → 병렬 ~4s. 일부 productType은 ticker 직접 수락 가능.
    _pt_list = ("STOCKS", "FOREIGN_STOCKS", "US_STOCKS",
                "OVERSEAS_STOCKS", "OVERSEAS", "LISTED_OVERSEAS")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(_pt_list)) as _ex:
            _futs = [_ex.submit(_toss_detail_api, ticker_upper, pt) for pt in _pt_list]
            for _fut in concurrent.futures.as_completed(_futs, timeout=5):
                try:
                    txt = _fut.result()
                    if txt:
                        return _cache_ok({"ai_summary": txt, "product_code": ticker_upper, "supported": True})
                except Exception:
                    pass
    except concurrent.futures.TimeoutError:
        pass

    # 전략 2: 배치 API에 ticker 직접 시도 (1회 HTTP 콜)
    result = _toss_batch_api(ticker_upper)
    if result.get("ai_summary"):
        return _cache_ok(result)

    # 전략 3: 검색 API로 실제 productCode 획득 후 재시도 (성공만 캐싱)
    code = _fetch_toss_us_product_code(ticker_upper)
    if not code:
        # productCode 조회 실패 → 캐시 안함, 다음 요청에서 재시도
        return {"ai_summary": "", "product_code": "", "supported": False}

    # productCode로 배치 API 시도
    result = _toss_batch_api(code)
    if result.get("ai_summary"):
        return _cache_ok(result)

    # productCode로 상세 API도 시도
    for pt in ("STOCKS", "FOREIGN_STOCKS", "US_STOCKS", "OVERSEAS_STOCKS", "OVERSEAS"):
        txt = _toss_detail_api(code, pt)
        if txt:
            return _cache_ok({"ai_summary": txt, "product_code": code, "supported": True})

    # productCode는 얻었지만 AI 요약 없음 (토스 미분석 상태) → 캐시 OK
    return _cache_ok({"ai_summary": "", "product_code": code, "supported": True})


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
# US 추천 시스템 — 장기 투자 & 개장 급등
# =============================================================================

# S&P 100 + 주요 성장주 유니버스 (장기/급등 공용)
_US_RECO_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","LLY","AVGO",
    "JPM","V","UNH","XOM","MA","ORCL","COST","NFLX","JNJ","WMT",
    "PG","BAC","HD","CVX","MRK","ABBV","CRM","KO","AMD","ACN",
    "PEP","TMO","LIN","MCD","CSCO","PM","DIS","TXN","ABT","ISRG",
    "DHR","AMGN","NEE","WFC","RTX","CAT","VZ","CMCSA","INTU","ADBE",
    "IBM","GS","MS","BKNG","HON","SPGI","QCOM","NOW","UNP","ETN",
    "LOW","T","GE","AXP","SYK","BLK","MDT","GILD","ELV","DE",
    "ADI","PLD","CI","MMC","VRTX","CB","SO","COP","SBUX","PANW",
    "MO","APD","EOG","BSX","LRCX","TT","ADP","ITW","ANET","REGN",
    "PGR","KLAC","ZTS","CME","ICE","ECL","HUM","MCO","PSA","NOC",
    "MU","INTC","F","GM","BA","LMT","MRNA","PYPL","CRWD","DDOG",
    "NET","APP","SMCI","AXON","MELI","COIN","SNOW","PLTR","OKTA","SE",
    "GRAB","PDD","BABA","UBER","DASH","ABNB","RBLX","U","ZM","ARM",
]

# 급등 스캐너 전용 유니버스 — 고베타·변동성 중심, 뉴스·실적에 강하게 반응하는 종목
_US_SURGE_UNIVERSE = list(dict.fromkeys([
    # AI/반도체 — 뉴스·실적에 가장 민감
    "NVDA","AMD","SMCI","ARM","MU","QCOM","INTC","MRVL","AMAT","LRCX","KLAC",
    # 빅테크 — 개별 이슈 빈번
    "AAPL","MSFT","META","AMZN","GOOGL","NFLX","TSLA",
    # 클라우드/AI 소프트웨어 — 실적 시즌에 급변
    "CRWD","NET","DDOG","SNOW","PLTR","APP","AXON","PANW","NOW","ZS","OKTA",
    # 핀테크/블록체인 — 변동성 극상
    "COIN","MSTR","HOOD","SOFI","AFRM","UPST","PYPL","SQ",
    # 소비/중형 성장주 — 개별 모멘텀
    "MELI","UBER","DASH","ABNB","SPOT","SNAP","RBLX","SE","PDD","BABA","GRAB",
    "PINS","TWLO","ZM","U",
    # 바이오/제약 — 임상·FDA에 급등
    "MRNA","BNTX","LLY","ABBV","VRTX","REGN","GILD","AMGN","SGEN",
    # 에너지/원자재 — 유가·지정학 민감
    "CVX","XOM","OXY","SLB","FCX","COP",
    # 방산/항공 — 수주·갈등 이슈
    "LMT","RTX","BA","NOC","GD",
    # 중국 ADR — 규제·무역 민감
    "NIO","XPEV","LI","JD","BIDU","PDD",
    # 이머징 테크 — 소형 고변동
    "IONQ","JOBY","RIVN","LCID","ACHR",
    # 전통 대형주 (실적 반응)
    "JPM","GS","MS","BAC","V","MA","UNH","HD","WMT","COST","MCD","KO",
    "JNJ","PG","CAT","GE","IBM","ISRG","HON","DE","TMO","MRK","ABBV",
    "CRM","ADBE","ORCL","ACN","INTU","TXN","ADI","BKNG","SPGI","BLK",
    "UNP","ETN","ADP","CB","NEE","DUK","SO","EOG",
]))

def _us_calc_rsi(close, period=14):
    """RSI 계산"""
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    v = float((100 - (100 / (1 + rs))).iloc[-1])
    return {"v": round(v, 2), "oversold": v < 30, "overbought": v > 70}

def _us_calc_macd(close, fast=12, slow=26, signal=9):
    """MACD 계산"""
    if len(close) < slow + signal:
        return None
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=signal, adjust=False).mean()
    hist = ml - sl
    cross = len(ml) >= 2 and float(ml.iloc[-1]) > float(sl.iloc[-1]) and float(ml.iloc[-2]) <= float(sl.iloc[-2])
    return {"macd": round(float(ml.iloc[-1]), 4), "signal": round(float(sl.iloc[-1]), 4),
            "hist": round(float(hist.iloc[-1]), 4), "cross": cross}

def _us_calc_bollinger(close, period=20):
    """볼린저밴드 계산"""
    if len(close) < period:
        return None
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    z = float(((close - sma) / std).iloc[-1])
    return {"sma": round(float(sma.iloc[-1]), 2),
            "upper": round(float((sma + 2*std).iloc[-1]), 2),
            "lower": round(float((sma - 2*std).iloc[-1]), 2),
            "z": round(z, 2), "near_lower": -1.2 <= z <= -1.0}

def _us_calc_adx(df, period=14):
    """ADX 계산"""
    if len(df) < period * 2 or not all(c in df.columns for c in ["High","Low","Close"]):
        return None
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    pdm = h.diff().where(lambda x: (x > (-l.diff())) & (x > 0), 0.0)
    ndm = (-l.diff()).where(lambda x: (x > h.diff()) & (x > 0), 0.0)
    atr_e = tr.ewm(span=period, adjust=False).mean()
    pdi = 100 * pdm.ewm(span=period, adjust=False).mean() / atr_e
    ndi = 100 * ndm.ewm(span=period, adjust=False).mean() / atr_e
    dx = 100 * abs(pdi - ndi) / (pdi + ndi).replace(0, np.inf)
    adx_v = float(dx.ewm(span=period, adjust=False).mean().iloc[-1])
    p, n = float(pdi.iloc[-1]), float(ndi.iloc[-1])
    strength = "strong" if adx_v >= 40 else ("moderate" if adx_v >= 25 else "weak")
    direction = "bullish" if p > n else ("bearish" if n > p else "neutral")
    return {"adx": round(adx_v, 2), "pdi": round(p, 2), "ndi": round(n, 2),
            "strength": strength, "direction": direction}

def _us_calc_volume(df, period=20):
    """거래량 비율 계산"""
    if "Volume" not in df.columns or len(df) < period:
        return None
    vol = df["Volume"]
    avg = float(vol.rolling(period).mean().iloc[-1])
    cur = float(vol.iloc[-1])
    if avg == 0:
        return None
    ratio = cur / avg
    return {"ratio": round(ratio, 2), "spike": ratio >= 1.5}

def _us_calc_rs(close, spy_close, period=20):
    """SPY/KOSPI 대비 상대강도 (rs5/rs10/rs20/rs60)"""
    n = min(len(close), len(spy_close))
    if n < period:
        return None
    c = close.iloc[-n:]; s = spy_close.iloc[-n:]
    rs20 = float((c.iloc[-1]/c.iloc[-period] - 1)*100 - (s.iloc[-1]/s.iloc[-period] - 1)*100)
    rs5  = float((c.iloc[-1]/c.iloc[-5]  - 1)*100 - (s.iloc[-1]/s.iloc[-5]  - 1)*100) if n >= 5  else 0.0
    rs10 = float((c.iloc[-1]/c.iloc[-10] - 1)*100 - (s.iloc[-1]/s.iloc[-10] - 1)*100) if n >= 10 else 0.0
    rs60 = float((c.iloc[-1]/c.iloc[-60] - 1)*100 - (s.iloc[-1]/s.iloc[-60] - 1)*100) if n >= 60 else None
    return {"rs20": round(rs20, 2), "outperform": sum([rs5 > 0, rs10 > 0, rs20 > 0]) >= 2,
            "rs60": round(rs60, 2) if rs60 is not None else None}

def _us_calc_week52(close, lookback=252):
    """52주 위치 계산"""
    n = min(len(close), lookback)
    if n < 20:
        return None
    cv = close.iloc[-n:]
    hi, lo, cur = float(cv.max()), float(cv.min()), float(close.iloc[-1])
    pos = ((cur - lo) / (hi - lo) * 100) if hi != lo else 50.0
    return {"pos": round(pos, 2), "near_low": pos <= 10.0, "high": round(hi, 2), "low": round(lo, 2)}

def _us_calc_atr(df, period=14):
    """ATR 계산"""
    if len(df) < period or not all(c in df.columns for c in ["High","Low","Close"]):
        return None
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    return round(float(tr.rolling(period).mean().iloc[-1]), 4)

def _us_calc_kalman(close, Q=1e-5, R=1e-2, N=1):
    """칼만 필터 (N-step 예측)"""
    prices = close.values.astype(float)
    if len(prices) < 10:
        return None
    x = np.array([prices[0], 0.0])
    P = np.eye(2)
    F = np.array([[1.0, 1.0],[0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Qm = Q * np.eye(2)
    Rm = np.array([[R]])
    for z in prices:
        xp = F @ x; Pp = F @ P @ F.T + Qm
        y = z - (H @ xp)[0]
        S = float((H @ Pp @ H.T + Rm)[0, 0])
        K = (Pp @ H.T).flatten() / S
        x = xp + K * y
        P = (np.eye(2) - np.outer(K, H)) @ Pp
    xn = x.copy()
    for _ in range(N):
        xn = F @ xn
    return {"filtered": round(float(x[0]), 2), "predicted": round(float(xn[0]), 2),
            "velocity": round(float(x[1]), 4)}

def _us_calc_obv(df):
    """OBV 계산"""
    if "Close" not in df.columns or "Volume" not in df.columns or len(df) < 21:
        return None
    c = df["Close"].values.astype(float)
    v = df["Volume"].values.astype(float)
    obv = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] > c[i-1]: obv[i] = obv[i-1] + v[i]
        elif c[i] < c[i-1]: obv[i] = obv[i-1] - v[i]
        else: obv[i] = obv[i-1]
    cur_obv = obv[-1]
    obv_sma = float(np.mean(obv[-20:]))
    obv5 = obv[-5] if len(obv) >= 5 else obv[0]
    obv_chg = cur_obv - obv5
    price_chg = c[-1] - (c[-5] if len(c) >= 5 else c[0])
    if obv_chg > 0 and cur_obv > obv_sma: trend = "accumulation"
    elif obv_chg < 0 and cur_obv < obv_sma: trend = "distribution"
    else: trend = "neutral"
    return {"trend": trend, "divergence": bool(obv_chg > 0 and price_chg <= 0 and cur_obv > obv_sma),
            "obv": round(cur_obv, 0), "obv_sma": round(obv_sma, 0)}

def _us_calc_stochastic(df, k_period=14, d_period=3):
    """Stochastic Oscillator 계산"""
    if not all(c in df.columns for c in ["High","Low","Close"]) or len(df) < k_period + d_period:
        return None
    h, l, c = df["High"], df["Low"], df["Close"]
    lo_k = l.rolling(k_period).min()
    hi_k = h.rolling(k_period).max()
    k = 100 * (c - lo_k) / (hi_k - lo_k).replace(0, np.inf)
    d = k.rolling(d_period).mean()
    kv, dv = float(k.iloc[-1]), float(d.iloc[-1])
    cross = len(k) >= 2 and kv > dv and float(k.iloc[-2]) <= float(d.iloc[-2]) and kv < 30
    return {"k": round(kv, 2), "d": round(dv, 2),
            "oversold": kv < 20, "overbought": kv > 80, "cross": cross}

def _us_calc_squeeze(df, bb_period=20, bb_mult=2.0, kc_period=20, kc_mult=1.5):
    """TTM Squeeze 계산"""
    if not all(c in df.columns for c in ["High","Low","Close"]) or len(df) < bb_period + 5:
        return None
    h, l, c = df["High"], df["Low"], df["Close"]
    bb_sma = c.rolling(bb_period).mean()
    bb_std = c.rolling(bb_period).std()
    bb_up = bb_sma + bb_mult * bb_std
    bb_lo = bb_sma - bb_mult * bb_std
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    kc_atr = tr.rolling(kc_period).mean()
    kc_mid = c.rolling(kc_period).mean()
    kc_up = kc_mid + kc_mult * kc_atr
    kc_lo = kc_mid - kc_mult * kc_atr
    sq_on = bool(float(bb_lo.iloc[-1]) > float(kc_lo.iloc[-1]) and float(bb_up.iloc[-1]) < float(kc_up.iloc[-1]))
    sq_series = (bb_lo > kc_lo) & (bb_up < kc_up)
    sq_count = 0
    for i in range(len(sq_series)-1, -1, -1):
        if sq_series.iloc[i]: sq_count += 1
        else: break
    don_mid = (h.rolling(bb_period).max() + l.rolling(bb_period).min()) / 2
    mom_series = c - (don_mid + bb_sma) / 2
    mom = float(mom_series.iloc[-1])
    prev_mom = float(mom_series.iloc[-2]) if len(mom_series) >= 2 else 0.0
    direction = "increasing" if mom > prev_mom else ("decreasing" if mom < prev_mom else "neutral")
    return {"on": sq_on, "count": sq_count, "momentum": round(mom, 4), "direction": direction}

def _us_analyze_ticker(df, spy_df=None):
    """단일 종목 전체 기술적 분석"""
    if df is None or df.empty or "Close" not in df.columns:
        return {}
    c = df["Close"]
    cl = float(c.iloc[-1])
    chg = float((c.iloc[-1]/c.iloc[-2] - 1)*100) if len(c) >= 2 else 0.0
    bb = _us_calc_bollinger(c)
    kalman = _us_calc_kalman(c, Q=1e-5, R=1e-2, N=1)
    # 단기 칼만 blended target
    blended = None
    if kalman and bb:
        blended = 0.5 * kalman["predicted"] + 0.5 * bb["sma"]
    atr_v = _us_calc_atr(df)
    result = {
        "close": round(cl, 2), "change_pct": round(chg, 2),
        "rsi": _us_calc_rsi(c),
        "macd": _us_calc_macd(c),
        "bollinger": bb,
        "adx": _us_calc_adx(df),
        "volume": _us_calc_volume(df),
        "week52": _us_calc_week52(c),
        "obv": _us_calc_obv(df),
        "stochastic": _us_calc_stochastic(df),
        "squeeze": _us_calc_squeeze(df),
        "atr": atr_v,
        "kalman_lt": _us_calc_kalman(c, Q=1e-3, R=1e-2, N=40),
    }
    if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns:
        result["rs"] = _us_calc_rs(c, spy_df["Close"])
    else:
        result["rs"] = None
    return result

def _us_longterm_score(a):
    """장기 투자 점수 (하드 필터 실패 시 -1, 통과 시 0~100)"""
    rsi = a.get("rsi"); macd = a.get("macd"); bb = a.get("bollinger")
    vol = a.get("volume"); adx = a.get("adx"); rs = a.get("rs")
    w52 = a.get("week52"); klt = a.get("kalman_lt"); obv = a.get("obv")
    stoch = a.get("stochastic"); sq = a.get("squeeze")
    # 하드 필터
    if not klt or klt["velocity"] <= 0: return -1
    if adx and adx["direction"] == "bearish": return -1
    if rsi and rsi["v"] >= 75: return -1
    if vol and vol["ratio"] < 0.7: return -1
    score = 0
    # RSI (max 8)
    if rsi:
        v = rsi["v"]
        if 45 <= v <= 60: score += 8
        elif 40 <= v < 45 or 60 < v <= 65: score += 5
        elif 35 <= v < 40 or 65 < v <= 70: score += 3
    # MACD (max 12)
    if macd:
        s = 0
        if macd["macd"] > macd["signal"]: s += 7
        if macd["macd"] > 0: s += 5
        score += min(s, 12)
    # 볼린저 (max 8): z 0.3~1.0 이 최적 (SMA 상방, 과열 아님)
    if bb:
        z = bb["z"]
        if 0.3 <= z <= 1.0: score += 8
        elif 0.0 <= z < 0.3: score += 5
        elif 1.0 < z <= 1.5: score += 3
    # 거래량 (max 8)
    if vol:
        r = vol["ratio"]
        if r >= 1.5: score += 8
        elif r >= 1.3: score += 6
        elif r >= 1.0: score += 4
    # ADX (max 15)
    if adx:
        if adx["direction"] == "bullish":
            if adx["adx"] >= 30: score += 15
            elif adx["adx"] >= 25: score += 10
            elif adx["adx"] >= 20: score += 6
        elif adx["direction"] == "neutral" and adx["adx"] >= 25: score += 4
    # 상대강도 (max 12)
    if rs:
        if rs["rs20"] > 5: score += 12
        elif rs["rs20"] > 2: score += 8
        elif rs["rs20"] > 0: score += 5
    # 52주 위치 (max 8): 40-70% 성장 여력
    if w52:
        p = w52["pos"]
        if 40 <= p <= 70: score += 8
        elif 30 <= p < 40 or 70 < p <= 80: score += 5
        elif 20 <= p < 30 or 80 < p <= 90: score += 2
    # 칼만 velocity (max 8)
    if klt:
        cl = a.get("close", 1) or 1
        vp = (klt["velocity"] / cl) * 100
        if vp > 0.5: score += 8
        elif vp > 0.3: score += 6
        elif vp > 0.1: score += 3
    # OBV (max 8)
    if obv:
        if obv["trend"] == "accumulation":
            score += 8 if obv["obv"] > obv["obv_sma"] else 5
        elif obv.get("divergence"): score += 4
    # Stochastic (max 5): 50-70 건강한 모멘텀
    if stoch:
        k = stoch["k"]
        if 50 <= k <= 70: score += 5
        elif 40 <= k < 50 or 70 < k <= 75: score += 3
        elif 30 <= k < 40: score += 2
    # Squeeze (max 8)
    if sq:
        if not sq["on"] and sq["momentum"] > 0 and sq["direction"] == "increasing": score += 8
        elif not sq["on"] and sq["momentum"] > 0: score += 5
        elif sq["on"] and sq["momentum"] > 0 and sq["count"] >= 5: score += 6
        elif sq["on"] and sq["momentum"] > 0: score += 4
    return score

def _us_surge_score(a, pm_change_pct, rvol=0.0):
    """개장 급등 점수 (하드 필터 실패 시 -1, 통과 시 0~100)
    PM모멘텀30 + RVOL20 + 전일거래량10 + Squeeze15 + 상대강도10 + ADX8 + BB돌파5 + Stoch5 = 103
    """
    rsi = a.get("rsi"); adx = a.get("adx"); rs = a.get("rs")
    bb = a.get("bollinger"); vol = a.get("volume"); sq = a.get("squeeze")
    stoch = a.get("stochastic"); w52 = a.get("week52")

    # 하드 필터 — 기준 완화 (명백한 하락/극단 과열만 제거)
    if adx and adx["adx"] > 35 and adx["direction"] == "bearish": return -1
    if rsi and rsi["v"] > 88: return -1
    if pm_change_pct < 1.0: return -1  # 1.5% → 1.0% 완화

    score = 0

    # 1. PM 모멘텀 (max 30) — 핵심 트리거
    if pm_change_pct >= 10.0: score += 30
    elif pm_change_pct >= 7.0:  score += 26
    elif pm_change_pct >= 5.0:  score += 22
    elif pm_change_pct >= 3.0:  score += 17
    elif pm_change_pct >= 2.0:  score += 13
    elif pm_change_pct >= 1.5:  score += 9
    else:                       score += 5   # 1.0~1.5%

    # 2. PM RVOL — 프리마켓 상대거래량 (max 20) — 기관/세력 관심도
    if rvol >= 5.0:   score += 20
    elif rvol >= 3.0: score += 15
    elif rvol >= 2.0: score += 10
    elif rvol >= 1.5: score += 6
    elif rvol >= 1.0: score += 3

    # 3. 전일 일봉 거래량 (max 10) — RVOL 없을 때 보완
    if vol:
        if vol["spike"]: score += 10
        elif vol["ratio"] >= 1.3: score += 7
        elif vol["ratio"] >= 1.0: score += 4

    # 4. Squeeze 해제 (max 15) — 압축 후 돌파 = 급등 핵심 패턴
    if sq:
        if not sq["on"] and sq["momentum"] > 0 and sq["direction"] == "increasing": score += 15
        elif not sq["on"] and sq["momentum"] > 0: score += 10
        elif sq["on"] and sq["momentum"] > 0 and sq["count"] >= 5: score += 8
        elif sq["on"] and sq["momentum"] > 0: score += 5

    # 5. 상대강도 (max 10)
    if rs:
        r60 = rs.get("rs60") or 0; r20 = rs.get("rs20") or 0
        best = max(r60, r20)
        if best > 10: score += 10
        elif best > 5:  score += 7
        elif best > 0:  score += 4

    # 6. ADX (max 8)
    if adx:
        if adx["direction"] == "bullish":
            if adx["adx"] >= 30: score += 8
            elif adx["adx"] >= 25: score += 5
            elif adx["adx"] >= 20: score += 3
        elif adx["strength"] == "weak": score += 2

    # 7. BB 상단 돌파 (max 5)
    pm_p = a.get("pm_price", a.get("close", 0)) or 0
    if bb and pm_p and pm_p > bb["upper"]: score += 5

    # 8. Stochastic (max 5)
    if stoch:
        k = stoch["k"]
        if 30 <= k <= 70: score += 5
        elif k < 30: score += 3

    return score

# =============================================================================
# KR 장기 투자 전용 분석 모듈
# =============================================================================

_KR_THEME_MAP = {
    "005930.KS": "AI/반도체", "000660.KS": "AI/반도체", "009150.KS": "AI/반도체",
    "018260.KS": "AI/반도체",
    "373220.KS": "2차전지", "006400.KS": "2차전지", "247540.KQ": "2차전지",
    "086520.KQ": "2차전지", "066970.KQ": "2차전지", "003670.KS": "2차전지",
    "051910.KS": "2차전지소재",
    "015760.KS": "전력/에너지", "036460.KS": "전력/에너지",
    "010120.KS": "전력/에너지", "267260.KS": "전력/에너지",
    "009830.KS": "신재생에너지",
    "012450.KS": "방산", "079550.KS": "방산",
    "034020.KS": "원전/방산",
    "329180.KS": "조선/방산", "042660.KS": "조선", "009540.KS": "조선", "010140.KS": "조선",
    "068270.KS": "바이오/헬스케어", "207940.KS": "바이오/헬스케어",
    "196170.KQ": "바이오/헬스케어", "028300.KQ": "바이오/헬스케어",
    "058470.KQ": "AI/전자부품",
    "035420.KS": "인터넷/플랫폼", "035720.KS": "인터넷/플랫폼",
    "323410.KS": "핀테크", "377300.KS": "핀테크",
    "259960.KS": "게임", "036570.KS": "게임", "251270.KS": "게임",
    "352820.KS": "엔터", "005380.KS": "자동차/전기차", "000270.KS": "자동차/전기차",
    "012330.KS": "자동차부품",
    "105560.KS": "금융", "055550.KS": "금융", "086790.KS": "금융", "316140.KS": "금융",
    "005490.KS": "소재/철강", "004020.KS": "소재/철강",
}
_KR_THEME_PRIORITY = {
    "AI/반도체", "2차전지", "전력/에너지", "방산", "조선/방산", "조선",
    "원전/방산", "바이오/헬스케어", "2차전지소재", "AI/전자부품",
}

def _kr_calc_ma_align(df):
    """중장기 이동평균선 정배열 분석 (60/120/240일)"""
    if "Close" not in df.columns or len(df) < 60:
        return None
    c = df["Close"]
    cur = float(c.iloc[-1])
    ma60 = float(c.rolling(60).mean().iloc[-1])
    result = {
        "ma60": round(ma60, 0), "ma120": None, "ma240": None,
        "aligned": False, "partial_aligned": False, "trend_reversal": False,
        "above_ma60": cur > ma60, "above_ma120": None, "above_ma240": None,
    }
    if len(df) >= 120:
        ma120 = float(c.rolling(120).mean().iloc[-1])
        result["ma120"] = round(ma120, 0)
        result["above_ma120"] = cur > ma120
        result["partial_aligned"] = bool(cur > ma60 > ma120)
        if len(df) >= 240:
            ma240 = float(c.rolling(240).mean().iloc[-1])
            result["ma240"] = round(ma240, 0)
            result["above_ma240"] = cur > ma240
            result["aligned"] = bool(cur > ma60 > ma120 > ma240)
            result["trend_reversal"] = bool(ma60 > ma120 and ma120 < ma240)
        else:
            result["aligned"] = result["partial_aligned"]
    return result

def _kr_calc_mdd(df, window=252):
    """최대 낙폭(MDD) 계산"""
    if "Close" not in df.columns or len(df) < 20:
        return None
    c = df["Close"].iloc[-min(window, len(df)):]
    rolling_max = c.cummax()
    drawdown = (c - rolling_max) / rolling_max * 100
    mdd = float(drawdown.min())
    cur_dd = float(drawdown.iloc[-1])
    return {"mdd": round(mdd, 1), "cur_drawdown": round(cur_dd, 1)}

def _kr_calc_trend_status(a, ma_align, w52):
    """추세 상태 레이블 판별"""
    adx = a.get("adx"); macd = a.get("macd")
    if ma_align:
        if ma_align.get("aligned"):
            if adx and adx["adx"] >= 25 and adx["direction"] == "bullish":
                return "강한 상승"
            return "상승"
        if ma_align.get("trend_reversal"):
            return "추세 전환"
        if ma_align.get("partial_aligned"):
            return "눌림목" if (adx and adx.get("strength") == "weak") else "중기 상승"
    if w52 and w52["pos"] <= 25:
        return "초기 반등" if (macd and macd["macd"] > macd["signal"]) else "바닥권"
    if adx and adx.get("strength") == "weak":
        return "횡보"
    return "혼조"

def _kr_longterm_score(a, tkr, fundamentals=None):
    """KR 장기 투자 100점 종합 스코어링
    추세(25) + 수급(20) + 밸류(15) + 성장성(15) + 안정성(10) + 저평가/반등(10) + 테마(5)
    """
    rsi = a.get("rsi"); macd = a.get("macd"); bb = a.get("bollinger")
    vol = a.get("volume"); adx = a.get("adx"); rs = a.get("rs")
    w52 = a.get("week52"); klt = a.get("kalman_lt"); obv = a.get("obv")
    sq = a.get("squeeze"); ma_align = a.get("ma_align"); mdd_d = a.get("mdd_data")
    f = fundamentals or {}

    # ── 하드 필터 ──────────────────────────────────────────────────────
    if rsi and rsi["v"] >= 80: return -1, {}
    if vol and vol["ratio"] < 0.5: return -1, {}
    if adx and adx["adx"] >= 30 and adx["direction"] == "bearish": return -1, {}
    if mdd_d and mdd_d["mdd"] < -50:
        if not (w52 and w52["pos"] <= 15 and macd and macd["macd"] > macd["signal"]):
            return -1, {}

    bd = {}

    # 1. 추세 (max 25)
    ts = 0
    if ma_align:
        ts += 10 if ma_align.get("aligned") else (6 if ma_align.get("partial_aligned") else
              (4 if ma_align.get("trend_reversal") else (2 if ma_align.get("above_ma60") else 0)))
    if adx:
        if adx["direction"] == "bullish":
            ts += 8 if adx["adx"] >= 30 else (5 if adx["adx"] >= 25 else 3)
        elif adx["direction"] == "neutral": ts += 1
    if macd:
        if macd["macd"] > macd["signal"] and macd["macd"] > 0: ts += 7
        elif macd["macd"] > macd["signal"]: ts += 4
        elif macd.get("cross"): ts += 5
    bd["trend"] = min(ts, 25)

    # 2. 수급 (max 20)
    ss = 0
    if obv:
        ss += (8 if obv["obv"] > obv["obv_sma"] else 5) if obv["trend"] == "accumulation" else (4 if obv.get("divergence") else 0)
    if vol:
        r = vol["ratio"]
        ss += 7 if r >= 1.5 else (5 if r >= 1.2 else (3 if r >= 1.0 else 0))
    if rs:
        r60 = rs.get("rs60") or 0; r20 = rs.get("rs20") or 0
        ss += 5 if r60 > 5 else (3 if r60 > 0 else (2 if r20 > 0 else 0))
    bd["supply"] = min(ss, 20)

    # 3. 밸류 (max 15)
    vs = 0
    if w52:
        p = w52["pos"]
        vs += 6 if p <= 30 else (4 if p <= 50 else (2 if p <= 70 else 0))
    per = f.get("per")
    if per and per > 0:
        vs += 5 if per < 10 else (4 if per < 15 else (3 if per < 20 else (2 if per < 30 else 0)))
    pbr = f.get("pbr")
    if pbr and pbr > 0:
        vs += 4 if pbr < 1.0 else (3 if pbr < 1.5 else (2 if pbr < 2.0 else (1 if pbr < 3.0 else 0)))
    bd["value"] = min(vs, 15)

    # 4. 성장성 (max 15)
    gs = 0
    eps_growth = f.get("eps_growth")
    if eps_growth is not None:
        gs += 6 if eps_growth > 30 else (4 if eps_growth > 15 else (2 if eps_growth > 0 else 0))
    roe = f.get("roe")
    if roe is not None:
        gs += 5 if roe > 20 else (4 if roe > 15 else (3 if roe > 10 else (1 if roe > 5 else 0)))
    rev_growth = f.get("rev_growth")
    if rev_growth is not None:
        gs += 4 if rev_growth > 20 else (3 if rev_growth > 10 else (1 if rev_growth > 0 else 0))
    bd["growth"] = min(gs, 15)

    # 5. 안정성 (max 10)
    sts = 0
    if rsi:
        v = rsi["v"]
        sts += 4 if 40 <= v <= 60 else (3 if (35 <= v < 40 or 60 < v <= 65) else (1 if (30 <= v < 35 or 65 < v <= 70) else 0))
    if mdd_d:
        sts += 3 if mdd_d["mdd"] > -15 else (2 if mdd_d["mdd"] > -25 else (1 if mdd_d["mdd"] > -35 else 0))
    if bb:
        z = bb["z"]
        sts += 3 if 0.0 <= z <= 1.5 else (2 if -0.5 <= z < 0.0 else 0)
    bd["stability"] = min(sts, 10)

    # 6. 저평가/반등 특수 (max 10)
    us = 0
    if w52:
        p = w52["pos"]
        if p <= 20 and macd and macd["macd"] > macd["signal"]: us += 5
        elif p <= 30 and obv and obv["trend"] == "accumulation": us += 4
        elif p <= 30: us += 2
    if klt and klt["velocity"] > 0:
        cl_v = a.get("close", 1) or 1
        vp = (klt["velocity"] / cl_v) * 100
        us += 3 if vp > 0.3 else (2 if vp > 0.1 else 1)
    if sq and sq["momentum"] > 0:
        us += 2 if not sq["on"] else (1 if sq.get("count", 0) >= 5 else 0)
    bd["undervalue"] = min(us, 10)

    # 7. 테마 (max 5)
    theme = _KR_THEME_MAP.get(tkr, "")
    bd["theme"] = 5 if theme in _KR_THEME_PRIORITY else (2 if theme else 0)

    return sum(bd.values()), bd


@ttl_cache(14400)  # 4시간 캐시
def fetch_kr_longterm_reco():
    """국내(KRX) 장기 투자 추천 Top 10 (기술·수급·펀더멘털 통합 스코어링)"""
    try:
        name_map = {v: k for k, v in KR_STOCK_MAP.items()}
        kr_tickers = list(dict.fromkeys(KR_STOCK_MAP.values()))  # dedup
        all_dl = kr_tickers + ["^KS11"]
        # 2년 데이터 — 240일 MA 계산 필요
        raw = yf.download(all_dl, period="2y", interval="1d",
                          progress=False, auto_adjust=True, threads=True)
        if raw.empty:
            return {"error": "데이터 없음", "items": []}
        kospi_df = None
        if isinstance(raw.columns, pd.MultiIndex):
            if "^KS11" in raw.columns.get_level_values(1):
                kospi_df = raw.xs("^KS11", axis=1, level=1).dropna(how="all")

        # ── 펀더멘털 병렬 수집 ─────────────────────────────────────────
        fundamentals = {}
        def _fetch_fund(tkr):
            try:
                info = yf.Ticker(tkr).info
                per = info.get("trailingPE") or info.get("forwardPE")
                pbr = info.get("priceToBook")
                roe_raw = info.get("returnOnEquity")
                roe = round(float(roe_raw) * 100, 1) if roe_raw else None
                eps_curr = float(info.get("trailingEps") or 0)
                eps_fwd  = float(info.get("forwardEps")  or 0)
                eps_growth = round(((eps_fwd / eps_curr) - 1) * 100, 1) if eps_curr > 0 and eps_fwd else None
                rev_raw = info.get("revenueGrowth")
                rev_growth = round(float(rev_raw) * 100, 1) if rev_raw else None
                return {
                    "per": round(float(per), 1) if per else None,
                    "pbr": round(float(pbr), 2) if pbr else None,
                    "roe": roe, "eps_growth": eps_growth,
                    "rev_growth": rev_growth,
                    "mktcap": info.get("marketCap"),
                }
            except Exception:
                return {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_fetch_fund, tkr): tkr for tkr in kr_tickers}
            for fut in concurrent.futures.as_completed(futs, timeout=20):
                tkr2 = futs[fut]
                try: fundamentals[tkr2] = fut.result()
                except Exception: fundamentals[tkr2] = {}

        # ── 후보 종목 분석 ──────────────────────────────────────────────
        candidates = []
        for tkr in kr_tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if tkr not in raw.columns.get_level_values(1): continue
                    df = raw.xs(tkr, axis=1, level=1).dropna(how="all")
                else:
                    df = raw.copy()
                if len(df) < 60: continue
                a = _us_analyze_ticker(df, kospi_df)
                if not a: continue
                a["ma_align"] = _kr_calc_ma_align(df)
                a["mdd_data"] = _kr_calc_mdd(df)
                score, breakdown = _kr_longterm_score(a, tkr, fundamentals.get(tkr, {}))
                if score < 0: continue
                candidates.append((tkr, score, a, breakdown))
            except Exception:
                continue

        candidates.sort(key=lambda x: x[1], reverse=True)
        results = []
        for tkr, score, a, breakdown in candidates[:10]:
            rsi = a.get("rsi"); macd = a.get("macd"); adx = a.get("adx")
            rs = a.get("rs"); w52 = a.get("week52"); klt = a.get("kalman_lt")
            obv = a.get("obv"); sq = a.get("squeeze")
            ma_align = a.get("ma_align"); mdd_d = a.get("mdd_data")
            f = fundamentals.get(tkr, {})
            cl = a.get("close", 0)
            atr_v = a.get("atr") or 0

            # 목표가 (칼만 예측 + 장기 ATR 혼합)
            if klt and klt["predicted"] > cl:
                target = round(max(klt["predicted"] * 1.05, cl * 1.10), 0)
            elif atr_v > 0:
                target = round(cl + atr_v * 3.0, 0)
            else:
                target = round(cl * 1.15, 0)
            stop = round(cl - atr_v * 2.0, 0) if atr_v > 0 else round(cl * 0.85, 0)
            exp_ret = round(((target - cl) / cl * 100), 1) if cl else 0

            trend_status = _kr_calc_trend_status(a, ma_align, w52)

            # AI 방향성
            if klt and cl:
                vp = (klt["velocity"] / cl * 100)
                ai_dir = "강한 상승" if vp > 0.3 else ("상승" if vp > 0.05 else ("중립" if vp > -0.05 else "하락"))
            else:
                ai_dir = "분석 불가"

            # 핵심 추천 사유
            reasons = []
            if ma_align:
                if ma_align.get("aligned"):
                    reasons.append("60/120/240일 이동평균 완전 정배열 — 강한 장기 상승 구조")
                elif ma_align.get("partial_aligned"):
                    reasons.append("60/120일 이동평균 정배열 — 중장기 상승 추세 유지")
                elif ma_align.get("trend_reversal"):
                    reasons.append("MA60이 MA120 돌파 — 장기 추세 전환 신호")
            if adx and adx["direction"] == "bullish" and adx["adx"] >= 20:
                reasons.append(f"ADX {adx['adx']:.1f} — 상승 추세 강도 확인")
            rs60 = (rs.get("rs60") or 0) if rs else 0
            rs20_v = (rs.get("rs20") or 0) if rs else 0
            if rs and rs60 > 0:
                reasons.append(f"KOSPI 대비 60일 +{rs60:.1f}% 아웃퍼폼 (업종 상대 강도 우위)")
            elif rs and rs20_v > 0:
                reasons.append(f"KOSPI 대비 20일 +{rs20_v:.1f}% 아웃퍼폼")
            if macd and macd["macd"] > macd["signal"]:
                reasons.append("MACD 상승 모멘텀" + (" (골든크로스)" if macd.get("cross") else ""))
            if klt and cl:
                kr_ret = ((klt["predicted"] - cl) / cl) * 100
                reasons.append(f"칼만 40일 예측 {klt['predicted']:,.0f}원 ({'+' if kr_ret >= 0 else ''}{kr_ret:.1f}%)")
            if obv and obv["trend"] == "accumulation":
                reasons.append("OBV 매집 구간 — 기관/외국인 선행 매수 포착")
            if w52 and w52["pos"] <= 30:
                reasons.append(f"52주 저점 대비 {w52['pos']:.0f}% 위치 — 저점 반등 초기 구간")
            if f.get("per") and f["per"] < 15:
                reasons.append(f"PER {f['per']:.1f}배 저평가 — 실적 대비 주가 매력")
            if f.get("roe") and f["roe"] and f["roe"] > 10:
                reasons.append(f"ROE {f['roe']:.1f}% — 우량 자본 수익성")
            theme = _KR_THEME_MAP.get(tkr, "")
            if theme in _KR_THEME_PRIORITY:
                reasons.append(f"시장 주도 테마 [{theme}] — 섹터 상승 모멘텀")
            if sq and not sq["on"] and sq["momentum"] > 0:
                reasons.append("TTM Squeeze 해제 — 압축 후 상승 돌파 진행")

            # 리스크 요소
            risks = []
            if rsi and rsi["v"] > 65:
                risks.append(f"RSI {rsi['v']:.0f} 과열 — 단기 눌림 가능")
            if w52 and w52["pos"] >= 80:
                risks.append("52주 고점 근처 — 차익 실현 압력")
            if mdd_d and mdd_d["mdd"] < -30:
                risks.append(f"최대 낙폭 {mdd_d['mdd']:.0f}% — 고변동성 종목")
            if not (ma_align and (ma_align.get("aligned") or ma_align.get("partial_aligned"))):
                risks.append("이동평균 정배열 미달 — 추세 재확인 필요")
            if f.get("per") and f["per"] > 30:
                risks.append(f"PER {f['per']:.0f}배 — 고평가 가능성")
            if not risks:
                risks.append("특이 리스크 없음 (지속적인 모니터링 권장)")

            # 보유기간
            if trend_status in ("강한 상승", "상승") and score >= 70:
                holding = "6개월~1년"
            elif trend_status in ("중기 상승", "추세 전환", "초기 반등"):
                holding = "3~6개월"
            else:
                holding = "3개월~"

            confidence_label = ("매우 높음" if score >= 80 else
                               ("높음" if score >= 65 else ("보통" if score >= 50 else "주의")))

            results.append({
                "ticker": tkr,
                "name": name_map.get(tkr, tkr),
                "close": cl,
                "change_pct": a.get("change_pct", 0),
                "score": score,
                "score_breakdown": breakdown,
                "confidence": "High" if score >= 65 else "Medium",
                "confidence_label": confidence_label,
                "target_price": target,
                "stop_loss": stop,
                "expected_return": exp_ret,
                "holding_period": holding,
                "trend_status": trend_status,
                "ai_direction": ai_dir,
                "reasons": reasons[:5],
                "risks": risks[:3],
                "theme": theme,
                "currency": "KRW",
                "rsi": rsi["v"] if rsi else None,
                "adx": adx["adx"] if adx else None,
                "rs20": rs20_v,
                "rs60": rs60,
                "kalman_predicted": klt["predicted"] if klt else None,
                "ma_aligned": ma_align.get("aligned") if ma_align else False,
                "mdd": mdd_d["mdd"] if mdd_d else None,
                "per": f.get("per"), "pbr": f.get("pbr"), "roe": f.get("roe"),
            })
        return {"items": results, "ts": int(time.time())}
    except Exception as e:
        return {"error": str(e), "items": []}

@ttl_cache(14400)  # 4시간 캐시
def fetch_us_longterm_reco():
    """미국 장기 투자 추천 Top 10 (기술·수급·펀더멘털 통합 스코어링)"""
    try:
        tickers = [t for t in _US_RECO_UNIVERSE[:100] if t not in {"SPY","QQQ","DIA","IWM"}]
        all_dl = tickers + ["SPY"]
        raw = yf.download(all_dl, period="1y", interval="1d",
                          progress=False, auto_adjust=True, threads=True)
        if raw.empty:
            return {"error": "데이터 없음", "items": []}
        spy_df = None
        if isinstance(raw.columns, pd.MultiIndex):
            if "SPY" in raw.columns.get_level_values(1):
                spy_df = raw.xs("SPY", axis=1, level=1).dropna(how="all")
        candidates = []
        for tkr in tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if tkr not in raw.columns.get_level_values(1): continue
                    df = raw.xs(tkr, axis=1, level=1).dropna(how="all")
                else:
                    df = raw.copy()
                if len(df) < 60: continue
                a = _us_analyze_ticker(df, spy_df)
                if not a: continue
                score = _us_longterm_score(a)
                if score < 0: continue
                candidates.append((tkr, score, a))
            except Exception:
                continue
        candidates.sort(key=lambda x: x[1], reverse=True)
        results = []
        for tkr, score, a in candidates[:10]:
            rsi = a.get("rsi"); macd = a.get("macd"); adx = a.get("adx")
            rs = a.get("rs"); w52 = a.get("week52"); klt = a.get("kalman_lt")
            obv = a.get("obv"); sq = a.get("squeeze")
            cl = a.get("close", 0)
            atr_v = a.get("atr") or 0
            if klt and klt["predicted"] > cl:
                target = round(max(klt["predicted"] * 1.05, cl * 1.08), 2)
            elif atr_v > 0:
                target = round(cl + atr_v * 3.0, 2)
            else:
                target = round(cl * 1.15, 2)
            stop = round(cl - atr_v * 2.0, 2) if atr_v > 0 else round(cl * 0.85, 2)
            exp_ret = round(((target - cl) / cl * 100), 1) if cl else 0
            # 추세 상태
            rs60 = (rs.get("rs60") or 0) if rs else 0
            rs20_v = (rs.get("rs20") or 0) if rs else 0
            if adx and adx["adx"] >= 25 and adx["direction"] == "bullish":
                trend_status = "강한 상승"
            elif adx and adx["direction"] == "bullish":
                trend_status = "상승"
            elif w52 and w52["pos"] <= 25 and macd and macd["macd"] > macd["signal"]:
                trend_status = "초기 반등"
            elif w52 and w52["pos"] <= 25:
                trend_status = "바닥권"
            elif adx and adx.get("strength") == "weak":
                trend_status = "횡보"
            else:
                trend_status = "혼조"
            # AI 방향성
            if klt and cl:
                vp = (klt["velocity"] / cl * 100)
                ai_dir = "강한 상승" if vp > 0.3 else ("상승" if vp > 0.05 else ("중립" if vp > -0.05 else "하락"))
            else:
                ai_dir = "분석 불가"
            # 보유기간
            if trend_status in ("강한 상승", "상승") and score >= 70:
                holding = "6개월~1년"
            elif trend_status in ("초기 반등",):
                holding = "3~6개월"
            else:
                holding = "3개월~"
            # 추천 사유
            reasons = []
            if adx and adx["direction"] == "bullish" and adx["adx"] >= 20:
                reasons.append(f"ADX {adx['adx']:.1f} — 상승 추세 강도 확인")
            if rs and rs60 > 0:
                reasons.append(f"SPY 대비 60일 +{rs60:.1f}% 아웃퍼폼")
            elif rs and rs20_v > 0:
                reasons.append(f"SPY 대비 20일 +{rs20_v:.1f}% 아웃퍼폼")
            if macd and macd["macd"] > macd["signal"]:
                reasons.append("MACD 상승 모멘텀" + (" (골든크로스)" if macd.get("cross") else ""))
            if klt and cl:
                kr_ret = ((klt["predicted"] - cl) / cl) * 100
                reasons.append(f"칼만 40일 예측 ${klt['predicted']:.2f} ({'+' if kr_ret >= 0 else ''}{kr_ret:.1f}%)")
            if obv and obv["trend"] == "accumulation":
                reasons.append("OBV 매집 구간 — 기관/세력 선행 매수")
            if w52 and w52["pos"] <= 30:
                reasons.append(f"52주 저점 대비 {w52['pos']:.0f}% — 저점 반등 초기 구간")
            if sq and not sq["on"] and sq["momentum"] > 0:
                reasons.append("TTM Squeeze 해제 — 압축 후 상승 돌파 진행")
            if rsi:
                reasons.append(f"RSI {rsi['v']:.1f} — 건강한 모멘텀 구간")
            # 리스크
            risks = []
            if rsi and rsi["v"] > 65:
                risks.append(f"RSI {rsi['v']:.0f} 과열 — 단기 눌림 가능")
            if w52 and w52["pos"] >= 80:
                risks.append("52주 고점 근처 — 차익 실현 압력")
            if not risks:
                risks.append("특이 리스크 없음 (지속적인 모니터링 권장)")
            confidence_label = ("매우 높음" if score >= 80 else
                               ("높음" if score >= 65 else ("보통" if score >= 50 else "주의")))
            results.append({
                "ticker": tkr,
                "close": cl,
                "change_pct": a.get("change_pct", 0),
                "score": score,
                "confidence": "High" if score >= 65 else "Medium",
                "confidence_label": confidence_label,
                "target_price": target,
                "stop_loss": stop,
                "expected_return": exp_ret,
                "holding_period": holding,
                "trend_status": trend_status,
                "ai_direction": ai_dir,
                "reasons": reasons[:5],
                "risks": risks[:3],
                "rsi": rsi["v"] if rsi else None,
                "adx": adx["adx"] if adx else None,
                "rs20": rs20_v,
                "rs60": rs60,
                "week52_pos": w52["pos"] if w52 else None,
                "kalman_predicted": klt["predicted"] if klt else None,
            })
        return {"items": results, "ts": int(time.time())}
    except Exception as e:
        return {"error": str(e), "items": []}

@ttl_cache(900)  # 15분 캐시 (프리마켓 빠른 갱신)
def fetch_us_opening_surge():
    """미국 개장 급등 추천 Top 10 (PM RVOL + ATR 스캐닝)"""
    try:
        # ── 세션 감지 ────────────────────────────────────────────────
        try:
            from zoneinfo import ZoneInfo as _ZI
            _et_now = dt.now(_ZI("America/New_York"))
        except Exception:
            _utc_now = dt.utcnow()
            _m = _utc_now.month
            _et_now = _utc_now + timedelta(hours=(-4 if 3 <= _m <= 10 else -5))
        _et_h = _et_now.hour + _et_now.minute / 60.0
        _time_str = _et_now.strftime("%H:%M")
        if 4.0 <= _et_h < 9.5:
            _session = "premarket"
            _session_label = f"프리마켓 (ET {_time_str})"
        elif 9.5 <= _et_h < 16.0:
            _session = "regular"
            _session_label = f"정규장 (ET {_time_str})"
        elif 16.0 <= _et_h < 20.0:
            _session = "afterhours"
            _session_label = f"시간외거래 (ET {_time_str})"
        else:
            _session = "closed"
            _session_label = f"장 외 시간 (ET {_time_str})"

        tickers = _US_SURGE_UNIVERSE  # 대형주+고변동 전용 유니버스

        # ── 기준 종가 + 평균 거래량 조회 ────────────────────────────
        daily = yf.download(tickers, period="5d", interval="1d",
                            progress=False, auto_adjust=True, threads=True)
        if daily.empty:
            return {"error": "데이터 없음", "items": []}

        today_utc = dt.utcnow().date()
        try:
            last_bar_date = pd.Timestamp(daily.index[-1]).date()
        except Exception:
            last_bar_date = today_utc

        # prev_close 인덱스 결정
        # - afterhours: 오늘 정규장이 닫혔으므로 오늘 종가(-1) 기준
        # - premarket/regular: 오늘 봉이 미완성이면 -2(전일 종가) 기준
        if _session == "afterhours":
            _pc_idx = -1
        elif last_bar_date == today_utc and len(daily) >= 2:
            _pc_idx = -2
        else:
            _pc_idx = -1

        avg_daily_vol: Dict[str, float] = {}
        if isinstance(daily.columns, pd.MultiIndex):
            prev_close_s = daily["Close"].iloc[_pc_idx]
            try:
                vdf = daily["Volume"]
                for tkr in tickers:
                    if tkr in vdf.columns:
                        v = vdf[tkr].dropna()
                        avg_daily_vol[tkr] = float(v.mean()) if len(v) > 0 else 0.0
            except Exception:
                pass
        else:
            prev_close_s = pd.Series({tickers[0]: float(daily["Close"].iloc[_pc_idx])})

        # ── 실시간/PM 1분봉 조회 ────────────────────────────────────
        pm_raw = yf.download(tickers, period="1d", interval="1m",
                             prepost=True, progress=False, auto_adjust=True, threads=True)
        if pm_raw.empty:
            _note = f"현재 시세 없음 ({_session_label}) — 프리마켓(04:00~09:30 ET)에 다시 확인"
            return {"items": [], "note": _note, "session": _session, "session_label": _session_label}

        # PM 거래량 합산 (RVOL 계산용)
        pm_vol_map: Dict[str, float] = {}
        try:
            if isinstance(pm_raw.columns, pd.MultiIndex):
                vdf = pm_raw["Volume"]
                for tkr in tickers:
                    if tkr in vdf.columns:
                        pm_vol_map[tkr] = float(vdf[tkr].dropna().sum())
        except Exception:
            pass

        if isinstance(pm_raw.columns, pd.MultiIndex):
            latest_price = pm_raw["Close"].iloc[-1]
        else:
            latest_price = pd.Series({tickers[0]: float(pm_raw["Close"].iloc[-1])})

        # ── PM 변동률 계산 + RVOL 추정 ──────────────────────────────
        _pm_elapsed = max(0.1, min(_et_h - 4.0, 5.5)) if _session == "premarket" else 6.5

        gainers = []
        for tkr in tickers:
            try:
                pc = float(prev_close_s.get(tkr, 0) or 0)
                lp = float(latest_price.get(tkr, 0) or 0)
                if pc <= 0 or lp <= 0: continue
                chg = (lp - pc) / pc * 100
                if chg < 1.0: continue  # 1.5% → 1.0% 완화
                # RVOL = PM 누적 거래량 / (평균 일일거래량 × PM 경과비율)
                pm_vol = pm_vol_map.get(tkr, 0)
                avg_vol = avg_daily_vol.get(tkr, 0)
                if avg_vol > 0 and pm_vol > 0:
                    rvol = round(pm_vol / (avg_vol * _pm_elapsed / 6.5), 2)
                else:
                    rvol = 0.0
                gainers.append((tkr, round(lp, 2), round(chg, 2), rvol))
            except Exception:
                continue

        if not gainers:
            if _session == "closed":
                _note = f"장 외 시간 ({_time_str} ET) — 프리마켓(04:00~09:30 ET)에 다시 확인하세요"
            elif _session == "regular":
                _note = f"정규장 중 ({_time_str} ET) — 전일 대비 1% 이상 상승 종목 없음"
            elif _session == "afterhours":
                _note = f"시간외 ({_time_str} ET) — 전일 종가 대비 1% 이상 상승 종목 없음"
            else:
                _note = f"+1.0% 이상 급등 종목 없음 ({_session_label})"
            return {"items": [], "note": _note, "session": _session, "session_label": _session_label}

        gainers.sort(key=lambda x: x[2], reverse=True)
        gainers = gainers[:60]  # 30 → 60 확대
        gainer_dict = {g[0]: (g[1], g[2], g[3]) for g in gainers}
        gainer_tickers = [g[0] for g in gainers]

        # ── 기술적 분석 (3개월 일봉) ─────────────────────────────────
        hist_raw = yf.download(gainer_tickers, period="3mo", interval="1d",
                               progress=False, auto_adjust=True, threads=True)
        spy_raw = yf.download(["SPY"], period="3mo", interval="1d",
                              progress=False, auto_adjust=True, threads=True)
        spy_df = spy_raw if not spy_raw.empty else None

        candidates = []
        for tkr, (pm_price, pm_chg, rvol) in gainer_dict.items():
            try:
                if isinstance(hist_raw.columns, pd.MultiIndex):
                    if tkr not in hist_raw.columns.get_level_values(1): continue
                    df = hist_raw.xs(tkr, axis=1, level=1).dropna(how="all")
                else:
                    df = hist_raw.copy()
                if len(df) < 10: continue  # 20 → 10 완화
                a = _us_analyze_ticker(df, spy_df)
                a["pm_price"] = pm_price
                s = _us_surge_score(a, pm_chg, rvol)
                if s < 0: continue  # 하드 필터만 (30점 임계값 제거)
                candidates.append((tkr, s, a, pm_price, pm_chg, rvol))
            except Exception:
                continue

        # 점수 부족 시 fallback: 하드필터만 통과한 종목 중 PM 변동률 기준 보완
        if len(candidates) < 5:
            for g in gainers:
                tkr, lp, chg, rvol = g
                if any(c[0] == tkr for c in candidates): continue
                candidates.append((tkr, max(0, int(chg * 3)), {
                    "close": lp, "change_pct": chg, "pm_price": lp,
                    "atr": None, "rsi": None, "adx": None, "rs": None,
                    "volume": None, "squeeze": None, "stochastic": None,
                }, lp, chg, rvol))

        candidates.sort(key=lambda x: x[1], reverse=True)
        results = []
        _surge_label = {"premarket": "프리마켓", "regular": "정규장",
                        "afterhours": "시간외", "closed": "전일비"}.get(_session, "전일비")

        for tkr, score, a, pm_price, pm_chg, rvol in candidates[:10]:
            rsi = a.get("rsi"); adx = a.get("adx"); rs = a.get("rs")
            vol = a.get("volume"); sq = a.get("squeeze"); stoch = a.get("stochastic")
            atr_v = a.get("atr") or 0
            if atr_v > 0:
                target = round(pm_price + atr_v * 1.5, 2)
                stop   = round(pm_price - atr_v * 0.5, 2)
            else:
                est    = pm_price * 0.025
                target = round(pm_price + est, 2)
                stop   = round(pm_price - est * 0.5, 2)
            ret = round(((target - pm_price) / pm_price) * 100, 2) if pm_price > 0 else 0
            rr  = round((target - pm_price) / (pm_price - stop), 2) if pm_price > stop > 0 else 0

            reasons = [f"{_surge_label} +{pm_chg:.2f}% 급등 (전일 종가 대비)"]
            if rvol >= 2.0: reasons.append(f"PM RVOL {rvol:.1f}x — 프리마켓 거래량 폭증 (강한 매수세)")
            elif rvol >= 1.0: reasons.append(f"PM RVOL {rvol:.1f}x — 평균 대비 거래량 증가")
            if sq and not sq["on"] and sq["momentum"] > 0:
                reasons.append("TTM Squeeze 해제 — 압축 후 상승 돌파 진행")
            elif sq and sq["on"] and sq["momentum"] > 0:
                reasons.append(f"Squeeze ON {sq['count']}일 지속 — 돌파 임박")
            rs60 = (rs.get("rs60") or 0) if rs else 0
            rs20_v = (rs.get("rs20") or 0) if rs else 0
            if rs and rs60 > 0: reasons.append(f"SPY 대비 60일 +{rs60:.1f}% 아웃퍼폼")
            elif rs and rs20_v > 0: reasons.append(f"SPY 대비 20일 +{rs20_v:.1f}% 아웃퍼폼")
            if adx and adx["direction"] == "bullish":
                reasons.append(f"ADX {adx['adx']:.1f} — 상승 추세 강도 확인")
            if vol and vol["spike"]: reasons.append(f"전일 거래량 스파이크 ({vol['ratio']:.1f}x)")

            warning = []
            if rsi and rsi["v"] > 70: warning.append(f"RSI {rsi['v']:.1f} 과매수 — 단기 과열")
            if stoch and stoch["overbought"]: warning.append(f"Stochastic 과매수 (%K:{stoch['k']:.1f})")
            if pm_chg < 1.5: warning.append("상승률 1.5% 미만 — 관심 수준, 추가 확인 필요")

            if score >= 50:   confidence_label = "강력 추천"
            elif score >= 35: confidence_label = "추천"
            elif score >= 20: confidence_label = "주목"
            else:             confidence_label = "관심"

            results.append({
                "ticker": tkr,
                "pm_price": pm_price,
                "pm_change_pct": pm_chg,
                "rvol": rvol,
                "close": a.get("close", 0),
                "score": score,
                "confidence_label": confidence_label,
                "target_price": target,
                "stop_loss": stop,
                "target_return": ret,
                "risk_reward": rr,
                "holding_period": "30분~2시간",
                "reasons": reasons[:5],
                "warning": warning,
                "rsi": rsi["v"] if rsi else None,
                "adx": adx["adx"] if adx else None,
                "rs20": rs20_v,
                "rs60": rs60,
            })
        return {
            "items": results, "ts": int(time.time()),
            "session": _session, "session_label": _session_label,
        }
    except Exception as e:
        return {"error": str(e), "items": []}

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

def calc_probability(score: float, dd: Dict) -> tuple:
    """
    상승/하락 가능성 계산 (0–100%)

    계산 방식:
    1. 기본값: score(0~100) 편차의 70%를 확률 편차로 선형 변환
       - score=50 → 50%, score=70 → 64%, score=30 → 36%
    2. RSI 보조 조정: 과매도(<30) +5pp 상승, 과매수(>70) -5pp 하락
    3. 볼린저 밴드 위치: 하단 20% 이내 근접 → +3pp, 상단 80% 초과 → -3pp
    4. 거래량 급증(>20일 평균 1.5배): 현재 추세 방향으로 추가 ±2pp
    5. 최종 클리핑: [15%, 85%] — 극단값 방지
    """
    prob_up = 50.0 + (score - 50) * 0.70  # 점수 편차를 확률로 변환

    def _last(k):
        a = dd.get(k, [])
        return float(a[-1]) if a and a[-1] is not None else None

    # RSI 보조 조정
    rsi = _last("RSI") or 50.0
    if rsi < 30:    prob_up += 5.0   # 과매도 → 반등 기대
    elif rsi > 70:  prob_up -= 5.0   # 과매수 → 하락 압력

    # 볼린저 밴드 내 현재가 위치
    close = _last("Close")
    bb_u  = _last("BB_Upper")
    bb_l  = _last("BB_Lower")
    if close and bb_u and bb_l and bb_u > bb_l:
        pos = (close - bb_l) / (bb_u - bb_l)  # 0=하단, 1=상단
        if pos < 0.2:    prob_up += 3.0   # 하단 근접 → 반등 기대
        elif pos > 0.8:  prob_up -= 3.0   # 상단 근접 → 하락 경계

    # 거래량 급증: 추세 방향 강화
    vols = dd.get("Volume", [])
    if vols:
        cur_vol = float(vols[-1] or 0)
        avg_vol = float(np.mean([x for x in vols[-20:] if x])) if len(vols) >= 2 else cur_vol
        if avg_vol > 0 and cur_vol > avg_vol * 1.5:
            if score > 50:  prob_up += 2.0   # 상승 추세에 거래량 확인
            else:           prob_up -= 2.0   # 하락 추세에 거래량 확인

    prob_up   = max(15.0, min(85.0, prob_up))  # 극단값 클리핑
    prob_down = round(100.0 - prob_up, 1)
    return round(prob_up, 1), prob_down

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

    # ── TP 확률 산출 (백테스트 분포 기반) ────────────────────────────
    # 기준점: KRX Zone C 승률 74.3% − 6pp 시장보정 = 68.3% (0.8 ATR 거리)
    #         US  Zone C 승률 75.2% + 5pp 시장보정 = 80.2% (0.8 ATR 거리)
    # 거리 감쇄: 0.8 ATR 초과분당 12pp 감소 (백테스트 Zone A→C 구간 분포 보간)
    _is_us    = (market == "US")
    _tp1_base = 80.2 if _is_us else 68.3
    _tp_mom   = (trend - 2) * 3.0          # 추세 강도별 ±6pp
    _tp_rsi   = 5.0 if rsi < 40 else (-5.0 if rsi > 70 else 0.0)
    _base_days = 12.3 if _is_us else 13.0  # Zone B 평균 보유일 기준
    _fl_b = -1.22 if _is_us else -1.11     # Zone B 실패 손실
    _fl_a = -1.67 if _is_us else -1.61     # Zone A 실패 손실 (더 깊음)

    def _make_tp_from_tgt(tgt_range):
        """목표가 범위(tgt_range)를 TP1/TP2/TP3로 분할 → 각각 도달 확률·기간 산출.
        TP 가격은 반드시 해당 리스크 프로필의 실제 목표가에서 나와야 한다.
        확률은 현재가 대비 ATR 거리 기반으로 백테스트 분포를 보간하여 계산."""
        lo, hi = tgt_range[0], tgt_range[1]
        tp_prices = [lo, (lo + hi) / 2, hi]   # TP1=하단, TP2=중간, TP3=상단
        result = []
        for tp_price in tp_prices:
            # ATR 단위 거리 (현재가 기준, 목표가가 위에 있어야 양수)
            dist_atr = (tp_price - price) / atr if atr > 0 else 1.0
            # 0.8 ATR 이내: 기준 확률 유지 / 초과분: ATR당 12pp 감소
            decay    = max(0.0, dist_atr - 0.8) * 12.0
            prob     = round(min(93.0, max(5.0, _tp1_base - decay + _tp_mom + _tp_rsi)), 1)
            # 도달 기간: 0.8 ATR = base_days, 이후 거리에 비례하여 증가
            days     = round(_base_days * max(0.5, dist_atr / 0.8), 1)
            ret_pct  = round((tp_price - price) / price * 100, 2) if price > 0 else 0.0
            # 실패 손실: 목표가가 2.5 ATR 이상이면 더 넓은 하락 범위 적용
            fl       = _fl_b if dist_atr < 2.5 else _fl_a
            result.append({
                "price":             round(tp_price, rnd),
                "return_pct":        ret_pct,
                "prob_pct":          prob,
                "avg_days":          days,
                "fail_avg_loss_pct": fl,
            })
        return result

    # 각 리스크 프로필의 실제 목표가 범위에서 TP 레벨 생성
    cons_tp = _make_tp_from_tgt(cons_tgt_range)
    bal_tp  = _make_tp_from_tgt(bal_tgt_range)
    agg_tp  = _make_tp_from_tgt(agg_tgt_range)

    r = lambda v: round(v, rnd)
    # ── 시나리오별 실패 조건 산출 ─────────────────────────────────────
    def _fail_conditions(scenario: str) -> list[str]:
        conds: list[str] = []
        if vol_trend == "expanding":
            conds.append("변동성 확대 중 — 손절 이탈 가능성 증가")
        if rsi > 65:
            conds.append(f"RSI {rsi:.0f} — 과열권, 조정 압력 상존")
        if scenario == "conservative":
            if trend < 2:
                conds.append("추세 약함 — 단기 반등이 제한적일 수 있음")
            conds.append("BB 하단 이탈 시 손절 즉시 집행 필요")
        elif scenario == "balanced":
            if macd <= sig_line:
                conds.append("MACD 하락 전환 — 중기 추세 약화 가능성")
            if ma20 and price < ma20:
                conds.append("MA20 이탈 상태 — 지지 복귀 실패 시 하락 가속")
            conds.append("MA20 재이탈 시 포지션 재검토 권장")
        elif scenario == "aggressive":
            if trend < 3:
                conds.append("추세 강도 부족 — 목표가 도달 난이도 높음")
            conds.append("거래량 감소 시 상승 모멘텀 소멸 위험")
            conds.append("고변동성 구간 — 목표가 이전 손절 이탈 가능")
        if not conds:
            conds.append("현재 기준 주요 실패 요인 없음")
        return conds

    return {
        "conservative": {
            "label": "보수적",
            "icon": "🛡️",
            "desc": "단기 반등 목표 — 추세가 약하거나 변동성이 높을 때 적합",
            "target": [r(cons_tgt_range[0]), r(cons_tgt_range[1])],
            "stop":   [r(cons_stp_range[0]), r(cons_stp_range[1])],
            "return": cons_ret,
            "rr_ratio": cons_rr,
            "stop_pct": cons_stp_pct,
            "atr_mul_tgt": cons_tgt_mul,
            "atr_mul_stp": cons_stp_mul,
            "interpretation": f"BB 하단 참조 손절 · 단기 반등 목표 (R/R {cons_rr:.1f}:1)",
            "tp_levels": cons_tp,
            "failure_conditions": _fail_conditions("conservative"),
        },
        "balanced": {
            "label": "중립적",
            "icon": "⚖️",
            "desc": "스윙 트레이딩 기준 — 추세가 형성되고 있을 때 적합",
            "target": [r(bal_tgt_range[0]), r(bal_tgt_range[1])],
            "stop":   [r(bal_stp_range[0]), r(bal_stp_range[1])],
            "return": bal_ret,
            "rr_ratio": bal_rr,
            "stop_pct": bal_stp_pct,
            "atr_mul_tgt": bal_tgt_mul,
            "atr_mul_stp": bal_stp_mul,
            "interpretation": f"MA20 지지 손절 · 중기 추세 목표 (R/R {bal_rr:.1f}:1)",
            "tp_levels": bal_tp,
            "failure_conditions": _fail_conditions("balanced"),
        },
        "aggressive": {
            "label": "공격적",
            "icon": "🚀",
            "desc": "추세 추종 최대 수익 — 강한 모멘텀·거래량 뒷받침 시만 적합",
            "target": [r(agg_tgt_range[0]), r(agg_tgt_range[1])],
            "stop":   [r(agg_stp_range[0]), r(agg_stp_range[1])],
            "return": agg_ret,
            "rr_ratio": agg_rr,
            "stop_pct": agg_stp_pct,
            "atr_mul_tgt": agg_tgt_mul,
            "atr_mul_stp": agg_stp_mul,
            "interpretation": f"BB 상단 참조 목표 · 추세 지속 시 최대 수익 (R/R {agg_rr:.1f}:1)",
            "tp_levels": agg_tp,
            "failure_conditions": _fail_conditions("aggressive"),
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
    """
    각 기술적 지표별 현재 상태·매매 시그널·핵심 해석 계산.

    ▶ 개선 사항 (v2):
      - RSI / MACD / 이동평균 / 볼린저밴드 추가 (기존 패널에 없던 핵심 지표)
      - ATR: 변동성 맥락 정보만 표시, 방향성 판단(가중치)에서 제외
      - 단순 다수결 → 가중 점수(weighted_score) 기반 5단계 종합 판단
        (강한매수 / 매수우세 / 중립 / 매도우세 / 강한매도)
      - 지표별 가중치 체계:
          RSI·MACD 1.5 → MA 1.2 → ADX·BB·PSAR 1.0 → OBV·Stoch 0.8 → Aroon·BuyPressure 0.7
    """
    def v(k):
        a = dd.get(k, [])
        val = a[-1] if a else None
        return float(val) if val is not None else None

    def v2(k):
        """직전봉(끝에서 두 번째) 값"""
        a = dd.get(k, [])
        val = a[-2] if len(a) >= 2 else None
        return float(val) if val is not None else None

    close = v("Close") or 1.0
    signals: Dict = {}
    # (score: -1.0~+1.0,  weight,  label)
    weighted: list = []

    # ── 1. RSI (14) ──────────────────────────────────────────────────────────
    rsi = v("RSI")
    if rsi is not None:
        if   rsi < 25: st,sig,desc,sc = "극도 과매도","매수",f"RSI {rsi:.1f} — 강한 반등 가능, 역추세 진입 구간",+1.0
        elif rsi < 35: st,sig,desc,sc = "과매도",      "매수",f"RSI {rsi:.1f} — 과매도권 진입, 단기 반등 기대",   +0.7
        elif rsi < 45: st,sig,desc,sc = "약세 권역",   "관망",f"RSI {rsi:.1f} — 하락 기조, 추가 신호 대기",       +0.1
        elif rsi < 55: st,sig,desc,sc = "중립",        "관망",f"RSI {rsi:.1f} — 중립 구간, 방향성 확인 필요",      0.0
        elif rsi < 65: st,sig,desc,sc = "강세 권역",   "관망",f"RSI {rsi:.1f} — 상승 기조 유지 중",              -0.1
        elif rsi < 75: st,sig,desc,sc = "과매수",      "매도",f"RSI {rsi:.1f} — 과매수권 진입, 차익 실현 고려",   -0.7
        else:          st,sig,desc,sc = "극도 과매수", "매도",f"RSI {rsi:.1f} — 강한 과열, 조정 가능성 높음",     -1.0
        signals["rsi"] = {"name":"RSI (14)", "state":st, "signal":sig, "desc":desc, "value":f"{rsi:.1f}"}
        weighted.append((sc, 1.5, "RSI"))

    # ── 2. MACD (12,26,9) ────────────────────────────────────────────────────
    macd_val  = v("MACD"); macd_sig_val = v("Signal_Line")
    macd_hist = (macd_val - macd_sig_val) if (macd_val is not None and macd_sig_val is not None) else None
    prev_macd = v2("MACD"); prev_sig_line = v2("Signal_Line")
    prev_hist = (prev_macd - prev_sig_line) if (prev_macd is not None and prev_sig_line is not None) else None
    if macd_val is not None and macd_sig_val is not None:
        above    = macd_val > macd_sig_val
        hist_exp = (macd_hist is not None and prev_hist is not None and
                    abs(macd_hist) > abs(prev_hist))
        if above and macd_val > 0 and hist_exp:
            st,sig,desc,sc = "강한 상승","매수",f"MACD 골든크로스 + 히스토그램 확대 (양수 구간)",       +1.0
        elif above and macd_val > 0:
            st,sig,desc,sc = "상승 추세","매수",f"MACD > Signal, 양수 구간 — 상승 추세 유효",           +0.7
        elif above and macd_val <= 0:
            st,sig,desc,sc = "반등 시도","관망",f"MACD > Signal이나 음수 구간 — 추세 전환 확인 중",     +0.3
        elif not above and macd_val < 0 and hist_exp:
            st,sig,desc,sc = "강한 하락","매도",f"MACD 데드크로스 + 히스토그램 확대 (음수 구간)",      -1.0
        elif not above and macd_val < 0:
            st,sig,desc,sc = "하락 추세","매도",f"MACD < Signal, 음수 구간 — 하락 추세 유효",          -0.7
        else:
            st,sig,desc,sc = "하락 전환","관망",f"MACD < Signal이나 양수 구간 — 하락 전환 주시",       -0.3
        h_disp = f"H:{macd_hist:+.3f}" if macd_hist is not None else f"{macd_val:.3f}"
        signals["macd"] = {"name":"MACD (12,26,9)", "state":st, "signal":sig, "desc":desc, "value":h_disp}
        weighted.append((sc, 1.5, "MACD"))

    # ── 3. 이동평균 정렬 (MA20 / MA60 / MA120) ───────────────────────────────
    ma20  = v("MA20"); ma60 = v("MA60"); ma120 = v("MA120")
    pma20 = v2("MA20"); pma60 = v2("MA60")
    if ma20 is not None:
        ab20  = close > ma20
        ab60  = (close > ma60)  if ma60  else None
        ab120 = (close > ma120) if ma120 else None
        gc = (pma20 is not None and pma60 is not None and ma60 is not None
              and pma20 <= pma60 and ma20 > ma60)
        dc = (pma20 is not None and pma60 is not None and ma60 is not None
              and pma20 >= pma60 and ma20 < ma60)
        if gc:
            st,sig,desc,sc = "골든크로스","매수",f"MA20({ma20:.2f}) > MA60 크로스 — 강한 상승 신호",+1.0
        elif dc:
            st,sig,desc,sc = "데드크로스","매도",f"MA20({ma20:.2f}) < MA60 크로스 — 강한 하락 신호",-1.0
        elif ab20 and (ab60 is None or ab60) and (ab120 is None or ab120):
            st,sig,desc,sc = "완전 정배열","매수",f"가격 > MA20 > MA60 > MA120 — 강세 정배열",       +0.8
        elif not ab20 and (ab60 is False) and (ab120 is False or ab120 is None):
            st,sig,desc,sc = "완전 역배열","매도",f"가격 < MA20 < MA60 < MA120 — 약세 역배열",      -0.8
        elif ab20:
            st,sig,desc,sc = "단기 상승","관망",f"가격 > MA20({ma20:.2f}), 중장기 MA 혼조",          +0.3
        else:
            st,sig,desc,sc = "단기 하락","관망",f"가격 < MA20({ma20:.2f}), 단기 추세 약화",          -0.3
        ma_val = f"{ma20:.2f}"
        if ma60:  ma_val += f" / {ma60:.2f}"
        if ma120: ma_val += f" / {ma120:.2f}"
        signals["ma"] = {"name":"이동평균 (20/60/120)", "state":st, "signal":sig, "desc":desc, "value":ma_val}
        weighted.append((sc, 1.2, "MA"))

    # ── 4. ADX (14) ──────────────────────────────────────────────────────────
    adx = v("ADX"); dip = v("DI_Plus"); dim = v("DI_Minus")
    if adx is not None:
        if adx > 25:
            if dip is not None and dim is not None and dip > dim:
                st,sig,desc,sc = "강한 상승 추세","매수",f"ADX {adx:.0f} 강세 + +DI 우세 → 상승 추세 강함", +0.8
            elif dip is not None and dim is not None:
                st,sig,desc,sc = "강한 하락 추세","매도",f"ADX {adx:.0f} 강세 + -DI 우세 → 하락 추세 강함",-0.8
            else:
                st,sig,desc,sc = "추세 강함",     "관망",f"ADX {adx:.0f} — 강한 추세 형성 중",              0.0
        elif adx > 20:
            sc_dir = +0.1 if (dip or 0) > (dim or 0) else -0.1
            st,sig,desc,sc = "추세 발생","관망",f"ADX {adx:.0f} — 추세 형성 초기", sc_dir
        else:
            st,sig,desc,sc = "횡보/추세 없음","관망",f"ADX {adx:.0f} — 방향성 불명확, 돌파 대기", 0.0
        signals["adx"] = {"name":"ADX (14)", "state":st, "signal":sig, "desc":desc, "value":f"{adx:.1f}"}
        weighted.append((sc, 1.0, "ADX"))

    # ── 5. 볼린저 밴드 (20,2) ────────────────────────────────────────────────
    bb_up = v("BB_Upper"); bb_lo = v("BB_Lower"); bb_mi = v("BB_Middle")
    if bb_up is not None and bb_lo is not None and bb_up > bb_lo:
        bb_mid   = bb_mi or ((bb_up + bb_lo) / 2)
        bb_width = (bb_up - bb_lo) / bb_mid * 100
        pct_b    = (close - bb_lo) / (bb_up - bb_lo)
        if   pct_b >= 0.95: st,sig,desc,sc = "상단 돌파",  "매도",f"%B {pct_b:.2f} — 밴드 상단 이탈, 과열/저항",-0.7
        elif pct_b >= 0.80: st,sig,desc,sc = "상단 권역",  "관망",f"%B {pct_b:.2f} — 상단 권역, 조정 주의",     -0.2
        elif pct_b <= 0.05: st,sig,desc,sc = "하단 이탈",  "매수",f"%B {pct_b:.2f} — 밴드 하단 이탈, 반등 가능",+0.7
        elif pct_b <= 0.20: st,sig,desc,sc = "하단 권역",  "매수",f"%B {pct_b:.2f} — 하단 권역, 지지선 테스트", +0.3
        elif bb_width < 3.0:st,sig,desc,sc = "밴드 수렴",  "관망",f"밴드폭 {bb_width:.1f}% — 변동성 수축, 돌파 임박",0.1
        else:               st,sig,desc,sc = "밴드 중간",  "관망",f"%B {pct_b:.2f} — 중립 구간",              0.0
        signals["bb"] = {"name":"볼린저 밴드 (20,2)", "state":st, "signal":sig, "desc":desc,
                         "value":f"%B:{pct_b:.2f} | 폭:{bb_width:.1f}%"}
        weighted.append((sc, 1.0, "BB"))

    # ── 6. ATR (14) — 변동성 맥락 정보만 (가중치 점수 제외) ─────────────────
    atr = v("ATR")
    if atr is not None:
        atr_pct = atr / close * 100
        if   atr_pct > 3:   st,sig,desc = "고변동성",   "관망",f"일간 변동 ≈{atr_pct:.1f}% — 분할 매수 권장"
        elif atr_pct > 1.5: st,sig,desc = "보통 변동성","관망",f"일간 변동 ≈{atr_pct:.1f}% — 적정 리스크"
        else:                st,sig,desc = "저변동성",   "관망",f"일간 변동 ≈{atr_pct:.1f}% — 돌파 시 강한 추세 기대"
        signals["atr"] = {"name":"ATR (14)", "state":st, "signal":sig, "desc":desc,
                          "value":f"{atr:.2f} ({atr_pct:.1f}%)", "context_only": True}
        # ※ ATR은 방향성 없음 → weighted 미포함

    # ── 7. PSAR (Parabolic SAR) ──────────────────────────────────────────────
    psar_v   = v("PSAR")
    psar_dir = v("PSAR_DIR")
    if psar_v is not None and psar_v > 0 and psar_dir is not None:
        psar_arr  = dd.get("PSAR_DIR", [])
        prev_pdir = float(psar_arr[-2]) if len(psar_arr) >= 2 and psar_arr[-2] is not None else psar_dir
        flipped   = (psar_dir != prev_pdir)
        psar_disp = round(psar_v, 2)
        if psar_dir == 1.0:
            st,sig,desc,sc = ("상승 전환","매수",f"SAR {psar_disp} — 하락→상승 전환 (추세 반전 확인)",+0.9) if flipped \
                        else ("상승 추세","매수",f"가격 > SAR {psar_disp} — 상승 추세 지속",           +0.5)
        else:
            st,sig,desc,sc = ("하락 전환","매도",f"SAR {psar_disp} — 상승→하락 전환 (손절 고려)",    -0.9) if flipped \
                        else ("하락 추세","매도",f"가격 < SAR {psar_disp} — 하락 추세 지속",          -0.5)
        signals["psar"] = {"name":"PSAR (0.02/0.2)", "state":st, "signal":sig, "desc":desc,
                           "value":f"{'▲' if psar_dir == 1.0 else '▼'} {psar_disp}"}
        weighted.append((sc, 1.0, "PSAR"))

    # ── 8. OBV — 가격·거래량 수렴/다이버전스 ────────────────────────────────
    obv_arr = dd.get("OBV", [])
    if len(obv_arr) >= 10:
        obv_now = float(obv_arr[-1])  if obv_arr[-1]  is not None else None
        obv_p10 = float(obv_arr[-10]) if obv_arr[-10] is not None else None
        cl_arr  = dd.get("Close", [])
        cl_p10  = float(cl_arr[-10]) if len(cl_arr) >= 10 and cl_arr[-10] is not None else close
        if obv_now is not None and obv_p10 is not None:
            p_up = close > cl_p10; o_up = obv_now > obv_p10
            if   p_up and o_up:         st,sig,desc,sc = "수렴 상승",    "매수","가격↑ + OBV↑ — 매수세 동반, 추세 신뢰↑",    +0.6
            elif not p_up and not o_up: st,sig,desc,sc = "수렴 하락",    "매도","가격↓ + OBV↓ — 매도세 동반, 하락 신뢰↑",    -0.6
            elif p_up and not o_up:     st,sig,desc,sc = "강세 다이버전스","매도","가격↑·OBV↓ — 매수세 약화, 상승 지속 의문",-0.4
            else:                       st,sig,desc,sc = "약세 다이버전스","매수","가격↓·OBV↑ — 기관 누적 추정, 반등 가능", +0.4
            _ao = abs(obv_now)
            obv_disp = f"{obv_now/1e9:+.2f}B" if _ao >= 1e9 else \
                       f"{obv_now/1e6:+.1f}M" if _ao >= 1e6 else f"{obv_now/1e3:+.1f}K"
            signals["obv"] = {"name":"OBV (10봉)", "state":st, "signal":sig, "desc":desc, "value":obv_disp}
            weighted.append((sc, 0.8, "OBV"))

    # ── 9. Stochastic (14,3) ─────────────────────────────────────────────────
    sk14 = v("%K"); sd14 = v("%D"); psk14 = v2("%K")
    if sk14 is not None and sd14 is not None:
        if sk14 > 80:
            st,sig,desc,sc = "과매수",    "매도",f"%K {sk14:.1f} 과매수 — 되돌림 경계",                 -0.6
        elif sk14 < 20:
            st,sig,desc,sc = "과매도",    "매수",f"%K {sk14:.1f} 과매도 — 단기 반등 기대",              +0.6
        elif sk14 > sd14:
            arrow = "↑" if (psk14 is not None and sk14 > psk14) else ""
            st,sig,desc,sc = f"골든크로스{arrow}","매수",f"%K({sk14:.1f}) > %D({sd14:.1f}) — 단기 상승 전환",+0.4
        else:
            st,sig,desc,sc = "데드크로스","매도",f"%K({sk14:.1f}) < %D({sd14:.1f}) — 단기 하락 전환",   -0.4
        signals["stoch14"] = {"name":"Stochastic (14,3)", "state":st, "signal":sig, "desc":desc,
                              "value":f"{sk14:.1f} / {sd14:.1f}"}
        weighted.append((sc, 0.8, "Stochastic"))

    # ── 10. Aroon (25) ───────────────────────────────────────────────────────
    au = v("AROON_UP"); ard = v("AROON_DOWN")
    if au is not None and ard is not None:
        if   au > 70 and ard < 30: st,sig,desc,sc = "강한 상승","매수",f"Up {au:.0f} — 최근 고점 근접, 상승 추세 우세", +0.6
        elif ard > 70 and au < 30: st,sig,desc,sc = "강한 하락","매도",f"Down {ard:.0f} — 최근 저점 근접, 하락 추세 우세",-0.6
        elif au > ard:             st,sig,desc,sc = "상승 우위","관망",f"Up({au:.0f}) > Down({ard:.0f}) — 추세 약함",   +0.2
        else:                      st,sig,desc,sc = "하락 우위","관망",f"Up({au:.0f}) < Down({ard:.0f}) — 하락 우위",   -0.2
        signals["aroon"] = {"name":"Aroon (25)", "state":st, "signal":sig, "desc":desc,
                            "value":f"↑{au:.0f} / ↓{ard:.0f}"}
        weighted.append((sc, 0.7, "Aroon"))

    # ── 11. Buy Pressure (14) ────────────────────────────────────────────────
    bp = v("BUY_PRESSURE")
    if bp is not None:
        if   bp > 65: st,sig,desc,sc = "강한 매수세","매수",f"상승일 거래량 비중 {bp:.1f}% — 14일 매수 우위",+0.6
        elif bp > 50: st,sig,desc,sc = "매수 우위",  "매수",f"상승일 거래량 비중 {bp:.1f}% — 완만한 매수세", +0.3
        elif bp > 35: st,sig,desc,sc = "매도 우위",  "매도",f"상승일 거래량 비중 {bp:.1f}% — 매도 압력 존재",-0.3
        else:         st,sig,desc,sc = "강한 매도세","매도",f"상승일 거래량 비중 {bp:.1f}% — 14일 매도 우위",-0.6
        signals["buy_pressure"] = {"name":"Buy Pressure (14)", "state":st, "signal":sig, "desc":desc,
                                   "value":f"{bp:.1f}%"}
        weighted.append((sc, 0.7, "BuyPressure"))

    # ── 종합 판단: 가중 점수 기반 5단계 ─────────────────────────────────────
    adx_v = v("ADX") or 0.0; dip_v = v("DI_Plus") or 0.0; dim_v = v("DI_Minus") or 0.0
    rsi_v = v("RSI") or 50.0
    market_state = classify_market_state(dd, close, rsi_v, adx_v, dip_v, dim_v)

    # 카운트 (ATR "context_only" 제외)
    buy_n   = sum(1 for s in signals.values() if s["signal"] == "매수" and not s.get("context_only"))
    sell_n  = sum(1 for s in signals.values() if s["signal"] == "매도" and not s.get("context_only"))
    watch_n = sum(1 for s in signals.values() if s["signal"] == "관망" and not s.get("context_only"))
    total_n = buy_n + sell_n + watch_n

    # 가중 점수 정규화 (-1.0 ~ +1.0)
    if weighted:
        tw  = sum(w for _, w, _ in weighted)
        raw = sum(sc * w for sc, w, _ in weighted)
        norm = raw / tw if tw > 0 else 0.0
    else:
        norm = 0.0

    # 5단계 종합 판단
    if   norm >= 0.45: ov_sig, ov_lbl = "매수", "강한 매수 🔥"
    elif norm >= 0.15: ov_sig, ov_lbl = "매수", "매수 우세 ▲"
    elif norm >= -0.15:ov_sig, ov_lbl = "관망", "중립 / 관망 —"
    elif norm >= -0.45:ov_sig, ov_lbl = "매도", "매도 우세 ▽"
    else:              ov_sig, ov_lbl = "매도", "강한 매도 ❄️"

    return {
        "signals": signals,
        "summary": {
            "buy": buy_n, "sell": sell_n, "watch": watch_n, "total": total_n,
            "overall_signal": ov_sig,
            "overall_label":  ov_lbl,
            "market_state":   market_state,
            "weighted_score": round(norm * 100, 1),   # -100 ~ +100 (UI 진행 바용)
        },
    }

def calc_buy_price(dd: Dict, last_price: float, atr: float, score: float, indicator_signals: Dict, market: str = "KRX", period: str = "1y") -> Dict:
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

    # 피보나치 되돌림 — 선택한 분석 기간 기준으로 동적 계산
    _fib_bars_map = {
        '1d': 5,  '3d': 10, '1wk': 10, '2wk': 14,
        '1mo': 25, '3mo': 65, '6mo': 130,
        '1y': 252, '2y': 504, '5y': 1260,
    }
    _fib_lbl_map = {
        '1d': '초단기·1일',  '3d': '초단기·3일', '1wk': '초단기·1주', '2wk': '단기·2주',
        '1mo': '단기·1개월', '3mo': '중기·3개월', '6mo': '장기·6개월',
        '1y': '장기·1년',   '2y': '장기·2년',   '5y': '장기·5년',
    }
    _fib_n = min(_fib_bars_map.get(period, len(highs)), len(highs)) if highs else 1
    fib_period_label = _fib_lbl_map.get(period, f'최근 {_fib_n}봉')
    h60 = max(highs[-_fib_n:]) if highs else last_price * 1.1
    l60 = min(lows[-_fib_n:])  if lows  else last_price * 0.9
    fib_range = h60 - l60 if h60 > l60 else last_price * 0.01
    fib_236 = h60 - fib_range * 0.236
    fib_382 = h60 - fib_range * 0.382
    fib_500 = h60 - fib_range * 0.500
    fib_618 = h60 - fib_range * 0.618
    fib_786 = h60 - fib_range * 0.786

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

    # ── 백테스트 기반 가격 밴드 (A/B/C) 산출 ─────────────────────────
    # 근거: tools/simulate_backtest.py (seed=2718, N=252, KRX/US 각 8종목 동일가중)
    # ATR 배수별 20일 TP/SL 시뮬레이션 → 진입 구간별 승률·기대수익·Sharpe 도출
    _BT_Z = {
        "KRX": {
            "A": {"k": 0.45, "k1": 0.25, "k2": 0.65, "win": 49.5, "ret": 1.94, "loss_p": 50.5, "hold": 15.1, "floss": -1.61, "sharpe": 3.36},
            "B": {"k": 0.875,"k1": 0.65, "k2": 1.10, "win": 60.4, "ret": 2.74, "loss_p": 39.6, "hold": 13.0, "floss": -1.11, "sharpe": 5.68},
            "C": {"k": 1.40, "k1": 1.10, "k2": 1.70, "win": 74.3, "ret": 3.72, "loss_p": 25.7, "hold": 9.2,  "floss": -0.79, "sharpe": 11.21},
        },
        "US": {
            "A": {"k": 0.45, "k1": 0.25, "k2": 0.65, "win": 59.4, "ret": 1.49, "loss_p": 40.6, "hold": 14.0, "floss": -1.67, "sharpe": 3.07},
            "B": {"k": 0.875,"k1": 0.65, "k2": 1.10, "win": 66.3, "ret": 2.00, "loss_p": 33.7, "hold": 12.3, "floss": -1.22, "sharpe": 4.86},
            "C": {"k": 1.40, "k1": 1.10, "k2": 1.70, "win": 75.2, "ret": 2.70, "loss_p": 24.8, "hold": 9.0,  "floss": -0.68, "sharpe": 9.18},
        },
    }
    _mkt = market if market in _BT_Z else "KRX"
    _btz = _BT_Z[_mkt]
    # 한국: 수급 급등락 반영 → 밴드 폭 18% 확대 / 미국: 추세 안정 → 8% 축소
    _bw = 1.18 if _mkt == "KRX" else 0.92

    # 추세 강도(0~4) + RSI 기반 확률 보정
    # 백테스트 기저 확률에 모멘텀·기술적 상태를 반영한 조정치 적용
    _trend = sum([
        1 if (ma20_raw and last_price > float(ma20_raw)) else 0,
        1 if (ma60_raw and last_price > float(ma60_raw)) else 0,
        1 if macd > sig_line else 0,
        1 if rsi > 50 else 0,
    ])
    _mom_adj  = (_trend - 2) * 2.5   # 추세 중립(2)에서 ±5pp
    _rsi_adj2 = 5.0 if rsi < 40 else (-5.0 if rsi > 70 else 0.0)

    def _ap(base_prob):
        """백테스트 기저확률 + 모멘텀·RSI 보정 → 최종 확률 (5~95% 클램프)"""
        return round(min(95.0, max(5.0, base_prob + _mom_adj + _rsi_adj2)), 1)

    # ── 공격적 매수 밴드 A/B/C ────────────────────────────────────────
    # 개념: ATR 눌림목 깊이를 3단계로 세분 → 각 단계별 진입 근거·기대수익·손실확률
    _agg_tech = {
        "A": (f"MA20({ma20:,.{rnd}f}) 근접 단기 지지" if abs(ma20 - last_price) / last_price < 0.05 else "단기 변동폭 최소 눌림"),
        "B": (f"BB중간({bb_m:,.{rnd}f}) 지지 수렴" if bb_m_raw else "볼린저 중간선 기준"),
        "C": (f"BB하단({bb_l:,.{rnd}f}) 구조적 지지" if bb_l_raw else "볼린저 하단 기준"),
    }
    aggressive_bands = []
    for _zn in ["A", "B", "C"]:
        _z = _btz[_zn]
        _center = last_price - _z["k"] * atr
        _hw     = (_z["k2"] - _z["k1"]) * atr * _bw * 0.5
        _lo, _hi = _center - _hw, _center + _hw
        _win = _ap(_z["win"]); _los = round(100.0 - _win, 1)
        aggressive_bands.append({
            "band": _zn,
            "range": [round(_lo, rnd), round(_hi, rnd)],
            "pct":   [round((_lo - last_price) / last_price * 100, 2),
                      round((_hi - last_price) / last_price * 100, 2)],
            "atr_basis": f"ATR×{_z['k1']:.2f}~{_z['k2']:.2f} 눌림 ({_mkt} 백테스트 {_z['win']}% 기저승률)",
            "tech_note": _agg_tech[_zn],
            "expected_return_pct": _z["ret"],
            "win_prob_pct":  _win,
            "loss_prob_pct": _los,
            "avg_failed_loss_pct": _z["floss"],
        })

    # ── 추천 매수 밴드 A/B/C ─────────────────────────────────────────
    # 개념: 기술적 지표 앵커(VWAP·BB·MA·Fib) 기반 고확률 진입 구간
    # 각 밴드는 백테스트 Zone B·C 데이터(더 높은 승률 구간)에 매핑
    _vwap = vwap_approx if vwap_approx else last_price - 0.875 * atr

    # 앵커 산출: 각 지표가 현재가보다 아래에 있어야 유효한 지지선
    # ── Band A: VWAP·BB중간 수렴 (현재가보다 낮은 경우만) ──────────────
    _anc_A_raw = (_vwap + bb_m) / 2 if bb_m_raw else _vwap
    _anc_A = min(_anc_A_raw, last_price - atr * 0.20)   # 최소 0.2 ATR 아래 보장

    # ── Band B: BB하단·MA20 수렴 (현재가보다 낮은 경우만) ───────────────
    if bb_l_raw and ma20_raw:
        _anc_B_raw = (bb_l + ma20) / 2
    elif bb_l_raw:
        _anc_B_raw = bb_l
    else:
        _anc_B_raw = last_price - 1.10 * atr
    # Band A보다 최소 0.2 ATR 아래에 위치하도록 보장
    _anc_B = min(_anc_B_raw, _anc_A - atr * 0.20)

    # ── Band C: MA60·Fib 38.2~50% 수렴 ──────────────────────────────
    # fib_382가 현재가보다 낮을 때만(= 유효한 지지선) 사용
    # fib_382 > last_price인 경우(하락 후 피보나치가 현재가 위) → ma60만 사용
    _fib_valid = (fib_382 < last_price) and (fib_382 > last_price * 0.70)
    if ma60_raw and _fib_valid:
        _anc_C_raw = (ma60 + fib_382) / 2
    elif ma60_raw:
        _anc_C_raw = ma60
    else:
        _anc_C_raw = last_price - 1.40 * atr
    # Band B보다 최소 0.2 ATR 아래에 위치하도록 보장
    _anc_C = min(_anc_C_raw, _anc_B - atr * 0.20)

    _rec_anchor = {"A": _anc_A, "B": _anc_B, "C": _anc_C}
    _rec_hw = {"A": atr * 0.25 * _bw, "B": atr * 0.35 * _bw, "C": atr * 0.50 * _bw}
    _rec_btmap = {"A": "B", "B": "C", "C": "C"}  # 기술적 앵커 구간 → 백테스트 Zone 매핑
    _rec_basis = {
        "A": (f"VWAP({_vwap:,.{rnd}f}) + BB중간({bb_m:,.{rnd}f}) 수렴 지지 — 기관 매집 참조" if bb_m_raw
              else f"20일 거래량 가중 평균({_vwap:,.{rnd}f}) 기관 매집 참조"),
        "B": (f"BB하단({bb_l:,.{rnd}f}) + MA20({ma20:,.{rnd}f}) 쌍지지 수렴" if (bb_l_raw and ma20_raw)
              else "볼린저 하단 + 단기 이평 구조적 지지"),
        "C": (f"MA60({ma60:,.{rnd}f}) + Fib 38.2~50%({fib_382:,.{rnd}f}~{fib_500:,.{rnd}f}) 중기 지지" if (ma60_raw and _fib_valid)
              else f"중기 이평({ma60:,.{rnd}f}) 구조적 지지"),
    }
    _rec_hold = {
        "A": f"상승 지속 시 약 {_btz['B']['hold']:.0f}일 내 목표 도달 기대",
        "B": f"분할 매수 구간 · 평균 {_btz['C']['hold']:.0f}일 보유 전략",
        "C": "중기 저점 매집 · 리스크 분산 보유 (ATR 낙폭 소화 후 반등)",
    }
    _price_ceiling = last_price - atr * 0.05   # 모든 밴드 상단은 현재가보다 낮아야 함
    recommended_bands = []
    for _zn in ["A", "B", "C"]:
        _bk  = _rec_btmap[_zn]
        _z   = _btz[_bk]
        _anc = _rec_anchor[_zn]
        _hw  = _rec_hw[_zn]
        _lo  = _anc - _hw
        _hi  = min(_anc + _hw, _price_ceiling)   # 현재가 위로 올라가지 않도록 클램프
        _lo  = min(_lo, _hi - atr * 0.05)        # 하단이 상단보다 낮도록 보장
        _win = _ap(_z["win"])
        recommended_bands.append({
            "band": _zn,
            "range": [round(_lo, rnd), round(_hi, rnd)],
            "pct":   [round((_lo - last_price) / last_price * 100, 2),
                      round((_hi - last_price) / last_price * 100, 2)],
            "basis":         _rec_basis[_zn],
            "hold_note":     _rec_hold[_zn],
            "expected_sharpe": _z["sharpe"],
            "win_prob_pct":  _win,
            "avg_hold_days": _z["hold"],
        })

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

    # ── 종목 상태 기반 동적 전략 판단 ──────────────────────────────────
    _above_ma20 = ma20_raw and last_price > float(ma20_raw)
    _above_ma60 = ma60_raw and last_price > float(ma60_raw)
    _trend_cnt  = sum([bool(_above_ma20), bool(_above_ma60), macd > sig_line, rsi > 50])
    _adx_strong = adx >= 25

    if rsi > 70 and _trend_cnt >= 3:
        _ctx = "overbought"
    elif _trend_cnt >= 3 and rsi <= 70:
        _ctx = "strong_uptrend" if _trend_cnt == 4 and _adx_strong else "uptrend"
    elif rsi < 40 and macd > sig_line:
        _ctx = "recovery"
    elif _trend_cnt <= 1:
        _ctx = "downtrend"
    else:
        _ctx = "sideways"

    _ctx_map = {
        "overbought":     ("⏸️ 관망 권장",            "wait",         None,  [],         max(20, 50 - int((rsi-70)*2))),
        "strong_uptrend": ("⚡ 밴드 A 단기 진입",      "band_a",       "A",   ["A"],      min(88, 62+_trend_cnt*5)),
        "uptrend":        ("✅ 밴드 A~B 분할 매수",    "split_buy",    "B",   ["A","B"],  60+_trend_cnt*5),
        "recovery":       ("✅ 밴드 B~C 저점 분할",    "recovery_buy", "C",   ["B","C"],  58+(5 if macd>sig_line else 0)),
        "downtrend":      ("⚠️ 관망 후 지지선 확인",   "wait_support", "C",   [],         max(15, 40-(_trend_cnt)*8)),
        "sideways":       ("✅ 밴드 B 분할 매수",       "split_buy",    "B",   ["A","B"],  55+(5 if rsi<50 else 0)),
    }
    _action, _akey, _pband, _abands, _conf = _ctx_map.get(_ctx, _ctx_map["sideways"])

    _rationale: list[str] = []
    if _ctx == "overbought":
        _rationale = [f"RSI {rsi:.0f} — 과매수 구간, 단기 하락 위험 증가", "현재가가 단기 고점 부근 — 신규 진입 불리", "기존 보유자는 일부 차익 실현 고려"]
    elif _ctx == "strong_uptrend":
        _rationale = ["이평선 완전 정배열 + ADX 강세 — 추세 강도 높음", "MA20 근접 단기 눌림목 진입 적기", "MACD 매수 우위 유지"]
    elif _ctx == "uptrend":
        _rationale = ["상승 추세 유지 중", f"RSI {rsi:.0f} — 과열 없이 건전한 상승", "분할 매수로 평균 단가 관리 권장"]
    elif _ctx == "recovery":
        _rationale = [f"RSI {rsi:.0f} — 과매도 반등 가능성 높음", "MACD 매수 전환 시그널 포착", "단기 급락 후 반등 구간 — 저가 분할 매수 유리"]
    elif _ctx == "downtrend":
        _rationale = ["하락 추세 진행 중 — 추가 하락 가능", f"RSI {rsi:.0f} — 아직 바닥 반전 신호 없음", "하락 추세 반전 확인(캔들·거래량) 후 진입 권장"]
    else:
        _rationale = ["횡보 구간 — 지지선 근접 시 매수 기회", f"RSI {rsi:.0f} — 중립 구간, 방향성 탐색 중", "밴드 B 진입 후 저항선 도달 시 차익 실현 전략"]

    if vol_trend == "expanding" and _akey not in ("wait", "wait_support"):
        _rationale.append("⚠️ 변동성 확대 중 — 한 번에 몰지 말고 분산 진입 권장")
    if _adx_strong and _ctx not in ("overbought", "downtrend"):
        _rationale.append(f"ADX {adx:.0f} — 추세 강도 양호, 진입 타이밍 유리")

    _chase_pct  = 1.04 if _ctx in ("sideways", "recovery") else 1.03 if _ctx == "uptrend" else 1.015
    _chase_price = round(last_price * _chase_pct, rnd)
    _chase_map  = {
        "overbought":     "현재가 자체가 과열 구간 — 어느 가격에서도 신규 매수 신중",
        "strong_uptrend": f"현재가 대비 +3% 이상({_chase_price:,.{rnd}f}) 급등 시 추격 주의",
        "uptrend":        f"현재가 대비 +3% 이상({_chase_price:,.{rnd}f}) 이미 올랐다면 진입 지양",
        "recovery":       f"반등 초기 — 현재가 대비 +4% 이상({_chase_price:,.{rnd}f}) 급등 시 추격 지양",
        "downtrend":      f"하락 추세 중 반등은 단기에 그칠 수 있음 — 추격 매수 금지",
        "sideways":       f"횡보 중 갑작스러운 3% 이상 급등({_chase_price:,.{rnd}f}) 시 추격 지양",
    }

    strategy_rec = {
        "action":         _action,
        "action_key":     _akey,
        "confidence_pct": int(min(95, max(5, _conf))),
        "priority_band":  _pband,
        "active_bands":   _abands,
        "rationale":      _rationale,
        "chase_zone":     {"price": _chase_price, "reason": _chase_map.get(_ctx, "")},
        "context":        _ctx,
    }

    r = lambda v: round(v, rnd)
    return {
        "current": r(last_price),
        # 기존 단일 구간 (백워드 호환)
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
        # 백테스트 기반 A/B/C 세분화 밴드 (신규)
        "aggressive_bands":  aggressive_bands,
        "recommended_bands": recommended_bands,
        "timing": {
            "buy":  buy_timing_str,
            "sell": sell_timing_str,
        },
        "support_zone": r(support_zone),
        "fib": {
            "h60": r(h60), "l60": r(l60),
            "f236": r(fib_236), "f382": r(fib_382), "f500": r(fib_500),
            "f618": r(fib_618), "f786": r(fib_786),
            "period_label": fib_period_label,
        },
        "rsi": round(rsi, 1),
        "rsi_context": rsi_ctx,
        "atr": r(atr),
        "atr_pct": round(atr_pct, 2),
        "vol_trend": vol_trend,
        "market": _mkt,
        "strategy_rec": strategy_rec,
    }

def calc_pullback_analysis(dd: Dict, last_price: float, atr: float, score: float, market: str = "KRX", target_price_data: Dict = None) -> Dict:
    """
    눌림목 분석 + 실전형 손익비 자리 진단
    ─────────────────────────────────────────────────────────────────────
    이미지 1: 급등 추세 종목의 눌림목 공통 특징 (5단계 흐름 구조)
      - 거래량 감소 / RSI 50~60 재정렬 / 20MA 지지 / OBV 유지 / 급등봉 저가 미이탈
    이미지 2: 실전형 손익비 좋은 자리 (구간(zone) 기반 분할매수)
      - 핵심 일치가격대(지지/저항 전환) / 세력 흔들림 패턴 점검 / 4단계 분할 진입
    """
    def _last(k, default=0.0):
        a = dd.get(k, [])
        v = a[-1] if a else None
        return float(v) if v is not None else default

    def _arr(k):
        return [float(x) for x in dd.get(k, []) if x is not None]

    closes  = _arr("Close");  highs   = _arr("High");   lows    = _arr("Low")
    volumes = _arr("Volume"); opens   = _arr("Open")
    obv_arr = _arr("OBV")
    ema20   = _arr("EMA20");  ema50   = _arr("EMA50")
    ma20    = _arr("MA20");   ma60    = _arr("MA60");    ma120   = _arr("MA120")
    adx_arr = _arr("ADX");    dip_arr = _arr("DI_Plus"); dim_arr = _arr("DI_Minus")
    rsi_arr = _arr("RSI")
    bb_u_arr = _arr("BB_Upper"); bb_l_arr = _arr("BB_Lower"); bb_m_arr = _arr("BB_Middle")
    bb_u    = bb_u_arr[-1] if bb_u_arr else 0.0
    bb_l    = bb_l_arr[-1] if bb_l_arr else 0.0
    bb_m    = bb_m_arr[-1] if bb_m_arr else 0.0
    rsi_val = rsi_arr[-1] if rsi_arr else 50.0
    rnd     = 4 if market == "US" else 0

    if not atr or atr != atr:
        atr = last_price * 0.02

    vol_avg = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else (float(np.mean(volumes)) if volumes else 1.0)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION A: 눌림목 분석 (급등 추세 종목의 눌림목 공통 특징)
    # ═══════════════════════════════════════════════════════════════════

    # ── A1. 급등봉 감지 (최근 60봉 내 거래량 2x 이상 + 양봉 + 3% 이상) ──
    surge_candles = []
    if len(closes) >= 5 and len(volumes) >= 20:
        for i in range(max(-60, -len(closes)), -1):
            v_ratio = volumes[i] / vol_avg if vol_avg > 0 else 1.0
            body_pct = (closes[i] - opens[i]) / opens[i] * 100 if opens[i] > 0 else 0
            if v_ratio >= 2.0 and closes[i] > opens[i] and body_pct >= 3.0:
                surge_candles.append({"idx": i, "close": closes[i], "low": lows[i],
                                      "vol_ratio": round(v_ratio, 1), "body_pct": round(body_pct, 1)})
    last_surge = surge_candles[-1] if surge_candles else None

    # ── A2. 급등봉 저가 미이탈 확인 ──────────────────────────────────
    surge_low_intact = False
    surge_low_price  = None
    surge_low_desc   = "급등봉 없음 (기준 미설정)"
    if last_surge:
        surge_low_price = last_surge["low"]
        if last_price > surge_low_price:
            surge_low_intact = True
            surge_low_desc = f"급등봉 저가 {surge_low_price:,.{rnd}f} 유지 중 (구조 붕괴 아님)"
        else:
            surge_low_desc = f"급등봉 저가 {surge_low_price:,.{rnd}f} 이탈 — 추세 붕괴 위험"

    # ── A3. 거래량 감소 확인 (눌림목 핵심: 매도세 약화) ──────────────
    vol_decreasing = False
    vol_trend_desc = "데이터 부족"
    if len(volumes) >= 10:
        recent_5  = float(np.mean(volumes[-5:]))
        prev_5_10 = float(np.mean(volumes[-10:-5]))
        vol_dec_ratio = recent_5 / prev_5_10 if prev_5_10 > 0 else 1.0
        vol_vs_surge  = recent_5 / vol_avg if vol_avg > 0 else 1.0
        if vol_dec_ratio < 0.8 and vol_vs_surge < 1.0:
            vol_decreasing = True
            vol_trend_desc = f"거래량 {vol_dec_ratio:.1f}x 감소 — 매도세 약화, 눌림목 이상적"
        elif vol_dec_ratio >= 1.2:
            vol_trend_desc = f"거래량 {vol_dec_ratio:.1f}x 증가 — 매도 지속 가능성 (눌림목 불안정)"
        else:
            vol_trend_desc = f"거래량 {vol_dec_ratio:.1f}x 보통"

    # ── A4. RSI 50~60 재정렬 확인 (과열 해소 후 재상승 준비) ──────────
    rsi_pullback_zone = False
    rsi_zone_desc = f"RSI {rsi_val:.1f}"
    if 45 <= rsi_val <= 65:
        rsi_pullback_zone = True
        rsi_zone_desc = f"RSI {rsi_val:.1f} — 50~60 재정렬 구간 (눌림목 이상적, 40 이하 붕괴 아님)"
    elif rsi_val > 65:
        rsi_zone_desc = f"RSI {rsi_val:.1f} — 과열 구간, 눌림목 미완성"
    elif rsi_val < 40:
        rsi_zone_desc = f"RSI {rsi_val:.1f} — 40 이하 붕괴 위험 (눌림목 실패 패턴)"

    # ── A5. 20MA(중기) 지지 확인 ──────────────────────────────────────
    ma20_support = False
    ma20_desc = "MA20 데이터 없음"
    ma20_val = ma20[-1] if ma20 else None
    ma60_val = ma60[-1] if ma60 else None
    if ma20_val:
        dist_pct = (last_price - ma20_val) / ma20_val * 100
        if -3.0 <= dist_pct <= 5.0:
            ma20_support = True
            ma20_desc = f"MA20({ma20_val:,.{rnd}f}) 근접 지지 중 ({dist_pct:+.1f}%)"
        elif dist_pct > 5.0:
            ma20_desc = f"MA20({ma20_val:,.{rnd}f}) 상회 ({dist_pct:+.1f}%) — 아직 눌림목 미진행"
        else:
            ma20_desc = f"MA20({ma20_val:,.{rnd}f}) 하회 ({dist_pct:+.1f}%) — 지지 붕괴 주의"

    # ── A6. OBV 유지/상승 확인 (매집 강도 유지) ──────────────────────
    obv_healthy = False
    obv_desc = "OBV 데이터 없음"
    if len(obv_arr) >= 10:
        obv_trend = obv_arr[-1] - obv_arr[-10]
        obv_5_trend = obv_arr[-1] - obv_arr[-5]
        if obv_trend > 0:
            obv_healthy = True
            obv_desc = "OBV 상승 추세 유지 — 세력 이탈 없음, 매집 강도 유지"
        elif obv_5_trend > 0:
            obv_healthy = True
            obv_desc = "OBV 단기 반등 — 소폭 조정 후 회복 중"
        else:
            obv_desc = "OBV 하락 — 세력 이탈 가능성, 눌림목 실패 패턴 주의"

    # ── A7. 음봉 크기 축소 확인 (에너지 재정렬 진행 중) ──────────────
    bear_shrinking = False
    bear_shrink_desc = "데이터 부족"
    if len(closes) >= 8 and len(opens) >= 8:
        bear_bodies = []
        for i in range(-8, 0):
            if closes[i] < opens[i] and opens[i] > 0:
                bear_bodies.append(abs(closes[i] - opens[i]) / opens[i] * 100)
        if len(bear_bodies) >= 3:
            first_half = float(np.mean(bear_bodies[:len(bear_bodies)//2]))
            second_half = float(np.mean(bear_bodies[len(bear_bodies)//2:]))
            if second_half < first_half * 0.8:
                bear_shrinking = True
                bear_shrink_desc = f"음봉 크기 축소 ({first_half:.1f}% → {second_half:.1f}%) — 매도세 약화"
            else:
                bear_shrink_desc = f"음봉 크기 유지/확대 ({second_half:.1f}%) — 매도 지속"
        else:
            bear_shrink_desc = "음봉 부족 — 횡보 또는 양봉 구간"

    # ── A8. 눌림목 점수 종합 (7개 항목 체크리스트) ───────────────────
    pb_checks = [
        {"item": "거래량 감소",         "pass": vol_decreasing,    "desc": vol_trend_desc,   "weight": 2},
        {"item": "RSI 50~60 재정렬",    "pass": rsi_pullback_zone, "desc": rsi_zone_desc,    "weight": 2},
        {"item": "20MA 지지",           "pass": ma20_support,      "desc": ma20_desc,        "weight": 2},
        {"item": "OBV 유지/상승",       "pass": obv_healthy,       "desc": obv_desc,         "weight": 2},
        {"item": "급등봉 저가 미이탈",  "pass": surge_low_intact,  "desc": surge_low_desc,   "weight": 2},
        {"item": "음봉 크기 축소",      "pass": bear_shrinking,    "desc": bear_shrink_desc, "weight": 1},
    ]
    pb_pass_count    = sum(1 for c in pb_checks if c["pass"])
    pb_score         = sum(c["weight"] for c in pb_checks if c["pass"])
    pb_max_score     = sum(c["weight"] for c in pb_checks)
    pb_score_pct     = round(pb_score / pb_max_score * 100)

    if pb_pass_count >= 5:
        pb_grade = "이상적인 눌림목"; pb_color = "green"
        pb_desc  = "재상승 가능성 높음 — 모든 조건 거의 충족, 분할 진입 검토 가능"
    elif pb_pass_count >= 3:
        pb_grade = "보통 눌림목";    pb_color = "yellow"
        pb_desc  = "일부 조건 미충족 — 추가 확인 후 신중한 진입"
    else:
        pb_grade = "실패 패턴 의심"; pb_color = "red"
        pb_desc  = "추세 붕괴 가능성 — 진입 자제, 손절 기준 재설정"

    # ── A9. 현재 흐름 단계 감지 (1~5단계) ────────────────────────────
    flow_stage = 1; flow_desc = ""
    adx_val = adx_arr[-1] if adx_arr else 20.0
    if ma20_val and ma60_val and len(closes) >= 60:
        is_uptrend = ma20_val > ma60_val and last_price > ma20_val
        # 최근 60봉 내 급등봉 존재 여부
        has_surge  = len(surge_candles) > 0
        vol_surge_now = (volumes[-1] / vol_avg >= 1.5) if vol_avg > 0 and volumes else False

        if not has_surge and not is_uptrend:
            flow_stage = 1; flow_desc = "장기 바닥 매집 — 횡보+거래량 감소+OBV 바닥 전환 확인 필요"
        elif has_surge and not vol_decreasing and vol_surge_now:
            flow_stage = 2; flow_desc = "거래량 폭증 및 돌파 — 이슈/테마 결합 + 이격 확대 단계"
        elif has_surge and not vol_decreasing and rsi_val > 65:
            flow_stage = 3; flow_desc = "1차 급등 진행 중 — RSI 과열(70+), 거래량 급증, 강한 상승"
        elif has_surge and vol_decreasing and rsi_pullback_zone:
            flow_stage = 4; flow_desc = "눌림목 형성 중 — 에너지 재정렬, 주황 점선 구간 진입 적기"
        elif has_surge and vol_surge_now and rsi_val > 60:
            flow_stage = 5; flow_desc = "재상승 및 재돌파 — 거래량 증가+RSI 60이상, 추세 가속"
        else:
            flow_stage = 3; flow_desc = "급등 후 조정 — 단계 판별 추가 확인 필요"

    # ═══════════════════════════════════════════════════════════════════
    # SECTION B: 실전형 손익비 자리 분석 (구간(zone) 기반)
    # ═══════════════════════════════════════════════════════════════════

    # ── B1. 핵심 구간(Zone) 산출 ─────────────────────────────────────
    n60  = min(60,  len(closes)); n120 = min(120, len(closes)); n250 = min(250, len(closes))
    h60  = max(highs[-n60:])   if highs else last_price * 1.1
    l60  = min(lows[-n60:])    if lows  else last_price * 0.9
    h120 = max(highs[-n120:])  if highs else last_price * 1.15
    l120 = min(lows[-n120:])   if lows  else last_price * 0.85

    # 핵심 일치가격대: 최근 20일 종가 25~75 백분위 구간 (지지/저항 전환 밀집대)
    closes_20 = sorted(closes[-20:]) if len(closes) >= 20 else sorted(closes)
    core_zone_low   = float(np.percentile(closes_20, 25)) if closes_20 else last_price * 0.95
    core_zone_high  = float(np.percentile(closes_20, 75)) if closes_20 else last_price * 1.05
    # 핵심 구간이 역전되는 경우 보정
    if core_zone_high <= core_zone_low:
        core_zone_high = last_price * 1.03
        core_zone_low  = last_price * 0.97

    # 상단 저항대: 60일 고점 (현재가보다 높아야 의미 있음, 낮으면 Fib 연장 사용)
    resist_high = h60 if h60 > last_price * 1.01 else last_price * 1.08
    resist_low  = resist_high * 0.97

    # 하단 방어 구간: MA60 또는 60일 저점 중 현재가에 가장 가까운 값 기준
    ma_floor_candidates = [x for x in [
        ma60[-1]  if ma60  and ma60[-1]  < last_price * 0.98 else None,
        ma20[-1]  if ma20  and ma20[-1]  < last_price * 0.95 else None,
        l60 * 1.01,
    ] if x is not None]
    ma_floor = max(ma_floor_candidates) if ma_floor_candidates else last_price * 0.88
    defense_high = ma_floor * 1.02
    defense_low  = ma_floor * 0.98

    # ── B2. 4단계 분할 진입 구간 산출 ────────────────────────────────
    entry_zones = []
    rnd2 = rnd
    # 1차: 탐색 매수 (현재가 기준 상단 저항대 하단)
    e1_high = min(last_price * 1.005, resist_low)
    e1_low  = e1_high - atr * 0.5
    entry_zones.append({
        "stage": "1차 (탐색 매수)",
        "range": [round(e1_low, rnd2), round(e1_high, rnd2)],
        "ratio": "10~20%",
        "desc":  "일치가격대 상단 최초 도달 — 반응 확인 소액 진입",
        "color": "#58a6ff",
    })
    # 2차: 눌림목 매수 (핵심 일치가격대)
    e2_high = round(core_zone_high, rnd2)
    e2_low  = round(core_zone_low,  rnd2)
    entry_zones.append({
        "stage": "2차 (눌림목 매수)",
        "range": [e2_low, e2_high],
        "ratio": "20~40%",
        "desc":  "지지대 재테스트 구간 — 거래량 감소 + 지지 확인",
        "color": "#3fb950",
    })
    # 3차: 재확인 매수 (반등 확인 후)
    e3_high = round(core_zone_high * 1.01, rnd2)
    e3_low  = round(core_zone_low  * 1.00, rnd2)
    entry_zones.append({
        "stage": "3차 (재확인 매수)",
        "range": [e3_low, e3_high],
        "ratio": "20~30%",
        "desc":  "반등 후 지지 유지 확인 — 단기 고점 돌파 시",
        "color": "#d29922",
    })
    # 4차: 돌파 추격 (저항 돌파 확인)
    e4_trigger = round(resist_low * 1.005, rnd2)
    entry_zones.append({
        "stage": "4차 (돌파 추격)",
        "range": [e4_trigger, round(resist_high * 1.01, rnd2)],
        "ratio": "10~20%",
        "desc":  f"저항선({resist_low:,.{rnd}f}) 돌파 시 — 거래량 동반 확인 필수",
        "color": "#f78166",
    })

    # ── B3. ATR 기반 손절/익절 산출 ──────────────────────────────────
    # 손절: ATR×2 또는 구조 붕괴 기준 (60MA 이탈) 중 낮은 값
    atr_sl      = last_price - atr * 2.0
    struct_sl   = defense_low
    sl_price    = round(max(atr_sl, struct_sl), rnd2)   # 더 높은(타이트한) 손절선
    sl_pct      = round((sl_price - last_price) / last_price * 100, 2)

    # 목표가: 앙상블 예측 → 저항대 → 최소 5% 상승 순으로 우선 적용
    forecast_min = None; forecast_max = None; forecast_source = "기술적 저항대 기반"
    if target_price_data and isinstance(target_price_data, dict):
        _fmin = target_price_data.get("min_price")
        _fmax = target_price_data.get("max_price")
        if _fmin and float(_fmin) > last_price:
            forecast_min = float(_fmin); forecast_max = float(_fmax) if _fmax else forecast_min * 1.05
            forecast_source = f"앙상블 예측 ({target_price_data.get('period','—')})"

    if forecast_min:
        target_main  = round(forecast_min, rnd2)
        target_high2 = round(forecast_max, rnd2)
    else:
        target_raw   = (resist_high + resist_low) / 2
        target_main  = round(max(target_raw, last_price * 1.05), rnd2)
        target_high2 = round(max(resist_high * 1.05, last_price * 1.10), rnd2)

    risk_amt     = last_price - sl_price if last_price > sl_price else atr * 2.0
    reward_main  = target_main - last_price
    rr_main      = round(reward_main / risk_amt, 2) if risk_amt > 0 else 0.0

    # 트레일링: 고점 대비 ATR×1.5
    trail_stop = round(last_price - atr * 1.5, rnd2)

    # ── B4. 세력 흔들림 패턴 점검 ────────────────────────────────────
    manipulation_flags = []
    # 패턴 1: 지지선 이탈 척 (장중 이탈 후 종가 회복)
    if len(closes) >= 3 and len(lows) >= 3:
        for i in range(-5, -1):
            if (lows[i] < core_zone_low * 0.99 and
                    closes[i] > core_zone_low * 0.99):
                manipulation_flags.append({
                    "pattern": "지지선 이탈 척 (손절 유도)",
                    "color": "orange",
                    "desc":  "장중 지지선 일시 이탈 후 종가 회복 — 손절 털기 후 재매집 신호",
                    "action": "종가 기준 확인, 다음날 반등 시 매수 검토",
                })
                break
    # 패턴 2: 거래량 급증 + 장대 음봉 후 다음날 양봉 회복
    if len(closes) >= 3 and len(volumes) >= 3:
        for i in range(-5, -1):
            v_ratio = volumes[i] / vol_avg if vol_avg > 0 else 1.0
            body_dn = (opens[i] - closes[i]) / opens[i] * 100 if opens[i] > 0 else 0
            if v_ratio >= 1.5 and body_dn >= 2.0:
                if i + 1 < 0 and closes[i+1] > opens[i+1]:
                    manipulation_flags.append({
                        "pattern": "장중 급락 후 회복 (공포 자극)",
                        "color": "yellow",
                        "desc":  f"거래량 {v_ratio:.1f}x 음봉 후 양봉 회복 — 공포 극대화 후 흡수",
                        "action": "다음날 흐름 확인 후 대응",
                    })
                    break
    # 패턴 3: 볼린저 밴드 수축 (긴 횡보 — 인내심 소멸 유도)
    bb_squeeze_here = False
    if bb_u and bb_l and bb_m and bb_m > 0:
        bb_width = (bb_u - bb_l) / bb_m * 100
        if len(bb_u_arr) >= 20 and len(bb_l_arr) >= 20 and len(bb_m_arr) >= 20:
            widths = [(bb_u_arr[i] - bb_l_arr[i]) / bb_m_arr[i] * 100
                      for i in range(-20, 0) if bb_m_arr[i] > 0]
            if widths:
                avg_w = float(np.mean(widths))
                if bb_width < avg_w * 0.75:
                    bb_squeeze_here = True
                    manipulation_flags.append({
                        "pattern": "긴 횡보 — 인내심 소멸 유도",
                        "color": "blue",
                        "desc":  f"볼린저 밴드 수축 ({bb_width:.1f}%, 평균 {avg_w:.1f}%) — 에너지 응축, 방향 돌파 임박",
                        "action": "상단 돌파 시 추세 동승, 거래량 감소 구간 유지",
                    })

    # ── B5. 구조 붕괴 손절 기준 (4개 조건) ──────────────────────────
    sl_conditions = [
        {"cond": "종가 기준 60MA 이탈",
         "triggered": (ma60_val is not None and last_price < ma60_val),
         "desc": f"현재가({last_price:,.{rnd}f}) vs MA60({ma60_val:,.{rnd}f})" if ma60_val else "MA60 없음"},
        {"cond": "거래량 실린 하락 (평균 150%+)",
         "triggered": (volumes[-1] / vol_avg >= 1.5 if volumes and vol_avg > 0 else False) and
                      (closes[-1] < opens[-1] if closes and opens else False),
         "desc": f"거래량 {volumes[-1]/vol_avg:.1f}x + 음봉" if volumes and vol_avg > 0 else "데이터 없음"},
        {"cond": "이탈 후 회복 실패 (2~3일 내 재하락)",
         "triggered": (len(closes) >= 3 and
                       closes[-3] < core_zone_low and
                       closes[-2] < core_zone_low and
                       closes[-1] < core_zone_low),
         "desc": f"3일 연속 핵심 지지대({core_zone_low:,.{rnd}f}) 하회"},
        {"cond": "60MA 하향 이탈",
         "triggered": (ma60_val is not None and
                       len(ma60) >= 2 and
                       ma60[-1] < ma60[-2] and
                       last_price < ma60[-1]),
         "desc": f"MA60 하향 전환 + 현재가 하회" if ma60_val else "MA60 없음"},
    ]
    sl_triggered = sum(1 for s in sl_conditions if s["triggered"])

    if sl_triggered >= 2:
        breakdown_verdict = "구조 붕괴 — 손절 실행"
        breakdown_color   = "red"
    elif sl_triggered == 1:
        breakdown_verdict = "경계 신호 — 추가 확인 필요"
        breakdown_color   = "orange"
    else:
        breakdown_verdict = "구조 유지 중"
        breakdown_color   = "green"

    # ── B6. 손익비 시나리오 3가지 ────────────────────────────────────
    rr_scenarios = []
    for label, tp, entry_adj in [
        ("시나리오 ① 이상적 흐름",   target_main,   0.0),
        ("시나리오 ② 횡보 후 돌파",  target_main,   atr * 0.3),
        ("시나리오 ③ 흔들림 후 반등", target_main,  atr * 0.6),
    ]:
        entry_p = last_price + entry_adj
        risk_p  = entry_p - sl_price
        reward_p= tp - entry_p
        rr_s    = round(reward_p / risk_p, 2) if risk_p > 0 else 0.0
        rr_scenarios.append({
            "label":       label,
            "entry":       round(entry_p, rnd2),
            "target":      round(tp, rnd2),
            "stop":        sl_price,
            "rr":          rr_s,
            "viable":      rr_s >= 2.0,
        })

    return {
        # ─ 눌림목 분석 ─
        "flow_stage":          flow_stage,
        "flow_desc":           flow_desc,
        "pullback_checks":     pb_checks,
        "pullback_pass_count": pb_pass_count,
        "pullback_score":      pb_score,
        "pullback_score_pct":  pb_score_pct,
        "pullback_grade":      pb_grade,
        "pullback_grade_color":pb_color,
        "pullback_desc":       pb_desc,
        "surge_candles_count": len(surge_candles),
        "last_surge_low":      surge_low_price,
        # ─ 손익비 자리 분석 ─
        "zones": {
            "resistance": {"high": round(resist_high, rnd2), "low": round(resist_low, rnd2)},
            "core":       {"high": round(core_zone_high, rnd2), "low": round(core_zone_low, rnd2)},
            "defense":    {"high": round(defense_high, rnd2),   "low": round(defense_low, rnd2)},
        },
        "entry_zones":     entry_zones,
        "stop_loss":       sl_price,
        "stop_loss_pct":   sl_pct,
        "target_main":     target_main,
        "target_ext":      target_high2,
        "target_source":   forecast_source,
        "trail_stop":      trail_stop,
        "rr_main":         rr_main,
        "rr_scenarios":    rr_scenarios,
        # ─ 세력 흔들림 패턴 점검 ─
        "manipulation_flags":  manipulation_flags,
        "bb_squeeze":          bb_squeeze_here,
        # ─ 구조 붕괴 판단 ─
        "sl_conditions":       sl_conditions,
        "sl_triggered":        sl_triggered,
        "breakdown_verdict":   breakdown_verdict,
        "breakdown_color":     breakdown_color,
        "atr":                 round(atr, rnd2),
        "market":              market,
    }


def calc_target_price(dd: Dict, last_price: float, atr: float, period: str, market: str = "KRX") -> Dict:
    """
    향후 주가 상승 가능 범위(목표가) 예측.

    last_price: 현재가 보정 완료된 실시간 가격 (Pre-Market / Overnight 포함).
    수익률 계산, 목표가 범위 모두 이 가격을 기준값으로 사용합니다.
    """
    if last_price <= 0:
        return {}

    def _last_val(key: str, default: float = 0.0) -> float:
        a = dd.get(key, [])
        v = a[-1] if a else None
        return float(v) if v is not None else default

    ma20 = _last_val("MA20")
    ma60 = _last_val("MA60")
    rsi  = _last_val("RSI", 50.0)
    macd = _last_val("MACD")
    sig  = _last_val("Signal_Line")
    bb_u = _last_val("BB_Upper")

    if not atr or np.isnan(atr):
        atr = last_price * 0.02

    rnd = 4 if market == "US" else 2

    # ── 추세 강도 분석 (0 ~ 4, RSI 과매수 시 -1 보정) ──────────────────
    trend_strength = (
        (1 if ma20 and last_price > ma20 else 0)
        + (1 if ma60 and last_price > ma60 else 0)
        + (1 if macd > sig else 0)
        + (1 if rsi > 50 else 0)
        - (1 if rsi > 70 else 0)   # 과매수 패널티
    )

    # ── period 기반 기본 목표가 산출 ──────────────────────────────────
    if period in ("1d", "3d", "1wk"):
        pred_period = "단기 (1주 ~ 2주)"
        base_target = last_price + atr * 2
        # 볼린저 상단이 유효한 저항선일 경우 반영
        if bb_u > last_price:
            base_target = max(base_target, bb_u)
    elif period in ("2wk", "1mo", "3mo"):
        pred_period = "중기 (1개월 ~ 3개월)"
        base_target = last_price + atr * 4
    else:
        pred_period = "장기 (6개월 이상)"
        base_target = last_price + atr * 8

    # ── 추세 강도에 따른 범위·근거 산출 ──────────────────────────────
    if trend_strength >= 3:
        min_target = base_target
        max_target = base_target + atr * 2
        reason = ("강한 상승 추세(이동평균선 정배열 및 MACD 매수 우위)가 지속되고 있어 "
                  "추가 상승 여력이 높습니다.")
    elif trend_strength >= 1:
        min_target = base_target - atr
        max_target = base_target + atr
        reason = ("완만한 상승 추세 또는 박스권 상향 돌파 시도 중입니다. "
                  "단기 저항선 돌파 여부가 중요합니다.")
    else:
        min_target = last_price + atr * 0.5
        max_target = last_price + atr * 1.5
        reason = ("현재 하락 추세 또는 조정 구간입니다. 기술적 반등 시 "
                  "일차적인 저항선을 목표로 보수적인 접근이 필요합니다.")

    # ── 수익률: 현재가(last_price) 기준 ──────────────────────────────
    min_return = (min_target - last_price) / last_price * 100
    max_return = (max_target - last_price) / last_price * 100

    # ── 도달 확률 산출 ────────────────────────────────────────────────
    # 기준: 추세강도(0~4)를 핵심 드라이버, RSI·ATR·거리를 보정 인자로 활용
    atr_pct = (atr / last_price * 100) if last_price > 0 else 2.0
    dist_pct = min_return  # 최소 목표가까지 상승 필요 %
    _base_prob = 40.0 + trend_strength * 10.0   # 40~80% 기본 범위
    _rsi_adj   =  8.0 if rsi < 40 else (-8.0 if rsi > 70 else 0.0)
    _vol_adj   = -6.0 if atr_pct > 4.0 else (4.0 if atr_pct < 1.5 else 0.0)
    _dist_adj  = max(-20.0, -(dist_pct - 5.0) * 1.5) if dist_pct > 5.0 else 0.0
    reach_probability = round(min(92.0, max(8.0, _base_prob + _rsi_adj + _vol_adj + _dist_adj)), 1)

    # ── 예상 소요 기간 ─────────────────────────────────────────────────
    _period_days = {"1d":5, "3d":7, "1wk":10, "2wk":14, "1mo":30, "3mo":65, "6mo":130, "1y":252}.get(period, 20)
    _speed = max(0.5, min(1.8, 1.0 + (trend_strength - 2) * 0.2 + (0.1 if macd > sig else -0.1)))
    _min_days = max(3, int(_period_days * 0.4 / _speed))
    _max_days = max(_min_days + 3, int(_period_days * 0.9 / _speed))
    expected_trading_days = [_min_days, _max_days]

    # ── 실패 요인 분석 ─────────────────────────────────────────────────
    failure_factors: list[str] = []
    # 거래량
    vols = [float(x) for x in dd.get("Volume", []) if x is not None]
    if len(vols) >= 10:
        _vol_recent = float(np.mean(vols[-5:])) if len(vols) >= 5 else vols[-1]
        _vol_prev   = float(np.mean(vols[-15:-5])) if len(vols) >= 15 else _vol_recent
        if _vol_prev > 0 and _vol_recent / _vol_prev < 0.75:
            failure_factors.append("거래량 감소 — 상승 동력 약화 가능성")
    # 저항선 근접
    if bb_u > last_price and (bb_u - last_price) / last_price < 0.03:
        failure_factors.append(f"볼린저 상단({round(bb_u, rnd):,}) 근접 — 단기 저항 가능")
    if ma60 and last_price > ma60 and (last_price - ma60) / last_price > 0.15:
        failure_factors.append("중기 이평선 대비 큰 폭 상승 — 되돌림 가능성 주의")
    # 과열
    if rsi > 65:
        failure_factors.append(f"RSI {rsi:.0f} — 단기 과열, 조정 시 목표가 지연 가능")
    # 추세 약함
    if trend_strength <= 1:
        failure_factors.append("추세 지표 대부분 하락 우위 — 목표가 도달 난이도 높음")
    # MACD 약세
    if macd < sig:
        failure_factors.append("MACD 하락 전환 — 단기 상승 모멘텀 약화")
    if not failure_factors:
        failure_factors.append("현재 기준 주요 실패 요인 없음 — 추세 유지 확인 권장")

    # ── 리스크 수준 평가 ──────────────────────────────────────────────
    _fail_cnt = len([f for f in failure_factors if "없음" not in f])
    if _fail_cnt == 0 and trend_strength >= 3:
        risk_level = "낮음"
        risk_reason = f"추세 양호, 실패 요인 미검출 ({pred_period})"
    elif _fail_cnt >= 3 or trend_strength <= 0:
        risk_level = "높음"
        risk_reason = f"다수 실패 요인 감지, 진입 신중 필요"
    else:
        risk_level = "중간"
        _risk_parts = []
        if trend_strength >= 2: _risk_parts.append("추세 양호")
        if atr_pct > 2.5: _risk_parts.append("변동성 증가 중")
        if bb_u > last_price and (bb_u - last_price) / last_price < 0.04: _risk_parts.append("저항선 인접")
        risk_reason = " · ".join(_risk_parts) if _risk_parts else "지표 혼조 상태"

    return {
        "min_price":               round(min_target, rnd),
        "max_price":               round(max_target, rnd),
        "min_return":              round(min_return, 1),
        "max_return":              round(max_return, 1),
        "period":                  pred_period,
        "reason":                  reason,
        "trend_strength":          trend_strength,
        "base_price":              round(last_price, rnd),
        "reach_probability":       reach_probability,
        "expected_trading_days":   expected_trading_days,
        "failure_factors":         failure_factors,
        "risk_level":              risk_level,
        "risk_reason":             risk_reason,
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
    except Exception:
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
        prob_up, prob_down = calc_probability(score, dd)  # 상승/하락 가능성 계산

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
        
        # ── 기하학적 패턴 → 캔들 패턴 리스트 통합 (UI 표시용) ───────────────
        for gp in geo_patterns:
            direction = "상승" if gp.get("signal") == "매수" else "하락" if gp.get("signal") == "매도" else "중립"
            patterns.append({"name": gp.get("name"), "desc": gp.get("desc"),
                              "direction": direction, "conf": 100})

        # ── Step 1: 외부 데이터 선제 수집 (현재가 보정에 필요) ───────────────
        # Naver 금융 (KRX 현재가·전일가 보정 소스)
        naver = fetch_naver(sym) if market == "KRX" else None

        # US 보강 데이터 (Finnhub / AV / Tiingo — 재무지표·뉴스 보완)
        us_enriched = None
        if market == "US":
            try:
                import sys as _sys, os as _os
                _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
                from market_briefing.us_enricher import fetch_us_enriched
                us_enriched = fetch_us_enriched(sym)
            except Exception:
                pass

        # ── Step 2: 현재가 보정 ───────────────────────────────────────────────
        # 반드시 calc_buy_price / calc_target_price / calc_risk 호출 전에 완료해야 함.
        # 보정된 last 가 예측·리스크 계산의 기준값(현재가)으로 사용됩니다.
        #
        # KRX: 네이버 금융 실시간 현재가 최우선
        # US : USStockPriceFetcher → Pre-Market / Overnight / After-Hours / 정규장 순 자동 선택
        session_name = "정규장"   # 기본값
        if market == "KRX" and naver and naver.get("price"):
            try:
                real_price = float(naver["price"])
                # 30% 이내 차이일 때만 보정 (액면분할 등 비정상 이격 방지)
                if last > 0 and abs(real_price - last) / last < 0.3:
                    last = real_price
                    nv_prev = naver.get("prev_close")
                    if nv_prev:
                        try:
                            prev = float(nv_prev)
                        except Exception:
                            pass
                    if prev > 0:
                        pct = (last - prev) / prev * 100
            except Exception:
                pass
        elif market == "US":
            # ① USStockPriceFetcher: overnightMarketPrice / preMarketPrice / postMarketPrice
            #
            # ※ 퍼센트 이격 차단(50% 등)은 사용하지 않습니다.
            #   USStockPriceFetcher 는 내부적으로 타임스탬프 신선도·marketState·
            #   Yahoo Finance 크럼 인증을 거쳐 이미 검증된 가격만 반환합니다.
            #   외부에서 추가로 퍼센트 체크를 하면 CUE처럼 뉴스·이벤트로 인해
            #   Pre-Market / Overnight 에서 50% 이상 급등·급락한 정상 종목이
            #   역사적 종가(closes[-1])로 강제 복귀되는 오류가 발생합니다.
            #   유일한 가드: price > 0 (0이나 음수 가격만 거부)
            _fetched = False
            try:
                _fetcher = _get_us_price_fetcher()
                if _fetcher is not None:
                    _res = _fetcher.fetch(sym)
                    if _res and _res.price > 0:
                        last = float(_res.price)
                        if _res.prev_close and float(_res.prev_close) > 0:
                            prev = float(_res.prev_close)
                        elif last > 0:
                            # prev_close 없는 경우: 역사적 종가를 전일종가로 사용
                            prev = prev or last
                        if prev > 0:
                            pct = (last - prev) / prev * 100
                        session_name = _PRICE_TYPE_LABEL.get(
                            _res.price_type,
                            getattr(_res, "session", None) and _res.session.label_ko() or "정규장",
                        )
                        _fetched = True
            except Exception:
                pass
            # ② fallback: yfinance fast_info (fetcher 로드 실패 또는 API 오류 시)
            #   fast_info.last_price 는 extended hours 포함 최신가이므로
            #   퍼센트 체크 없이 그대로 신뢰합니다.
            if not _fetched:
                try:
                    fast_info = yf.Ticker(sym).fast_info
                    if hasattr(fast_info, 'last_price'):
                        real_price = fast_info.last_price
                        if real_price and real_price > 0:
                            last = float(real_price)
                            if hasattr(fast_info, 'previous_close'):
                                real_prev = fast_info.previous_close
                                if real_prev and real_prev > 0:
                                    prev = float(real_prev)
                            if prev > 0:
                                pct = (last - prev) / prev * 100
                except Exception:
                    pass

        # ── Step 3: 투자자 수급 (KRX 전용) — score 보정 포함 ────────────────
        # calc_buy_price 가 score 를 사용하므로 현재가 보정 직후, 예측 계산 전 실행
        investor_flow = {"ok": False, "reason": "KRX 종목 아님"}
        if market == "KRX":
            investor_flow = fetch_investor_flow(sym)
            if investor_flow.get("ok"):
                foreign = investor_flow.get("외국인", 0)
                inst    = investor_flow.get("기관", 0)
                pension = investor_flow.get("연기금", 0)
                adj = 0
                if foreign > 0: adj += 2
                elif foreign < 0: adj -= 2
                if inst > 0: adj += 2
                elif inst < 0: adj -= 2
                if pension > 0: adj += 1
                elif pension < 0: adj -= 1
                score = max(0, min(100, score + max(-5, min(5, adj))))
                flow_notes = []
                if foreign != 0: flow_notes.append(f"외국인 {foreign:+,}주")
                if inst    != 0: flow_notes.append(f"기관 {inst:+,}주")
                if pension != 0: flow_notes.append(f"연기금 {pension:+,}주")
                if flow_notes and isinstance(ai_strategy, dict):
                    ai_strategy["result"] += " | [투자자 수급] " + " / ".join(flow_notes)

        # ── Step 4: 예측·리스크 계산 — 보정된 현재가(last) 기준 ─────────────
        # ATR fallback 도 보정된 last 기준으로 재계산
        atrs = dd.get("ATR", [])
        atr_val          = float(atrs[-1]) if atrs and atrs[-1] else last * 0.02
        pivot_points     = calc_pivot_points(dd)
        indicator_signals= calc_indicator_signals(dd)
        risk             = calc_risk(last, atr_val, market, dd)
        buy_price        = calc_buy_price(dd, last, atr_val, score, indicator_signals, market, period)
        target_price     = calc_target_price(dd, last, atr_val, period, market)
        pullback_analysis = calc_pullback_analysis(dd, last, atr_val, score, market, target_price)

        # ── Step 5: HybridTurtle 복합 점수 (NCS/BQS/FWS) ────────────────────
        hybrid_score = None
        try:
            import sys as _sys, os as _os
            _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from market_briefing.stock_analyzer import enrich_with_hybrid
            _closes  = [float(c) for c in (dd.get("Close") or []) if c is not None]
            _highs   = [float(c) for c in (dd.get("High")  or []) if c is not None]
            _lows    = [float(c) for c in (dd.get("Low")   or []) if c is not None]
            _vols    = [float(c) for c in (dd.get("Volume") or []) if c is not None]
            _opens   = [float(c) for c in (dd.get("Open")  or []) if c is not None]
            if len(_closes) >= 20:
                hybrid_score = enrich_with_hybrid(
                    closes      = _closes,
                    highs       = _highs  if len(_highs) == len(_closes) else None,
                    lows        = _lows   if len(_lows)  == len(_closes) else None,
                    volumes     = _vols   if len(_vols)  == len(_closes) else None,
                    open_prices = _opens  if len(_opens) == len(_closes) else None,
                )
                # NCS 기반 score 보정 (NCS가 높으면 최대 +5, 낮으면 최대 -10)
                if hybrid_score and "ncs" in hybrid_score and "error" not in hybrid_score:
                    ncs_v = float(hybrid_score["ncs"])
                    fws_v = float(hybrid_score.get("fws", 50))
                    if ncs_v >= 70 and fws_v <= 30:
                        score = min(100, score + 5)
                        if isinstance(ai_strategy, dict):
                            ai_strategy["result"] += f" | [NCS {ncs_v:.0f}] 브레이크아웃 품질 우수 — 신뢰도 상향"
                    elif ncs_v < 40 or fws_v > 65:
                        score = max(0, score - 10)
                        if isinstance(ai_strategy, dict):
                            ai_strategy["result"] += f" | [NCS {ncs_v:.0f}] 기술적 취약 — 주의"
                    # 레짐 BEARISH 시 score 추가 하향
                    if hybrid_score.get("regime") == "BEARISH" and score > 40:
                        score = min(score, 40)
                        if isinstance(ai_strategy, dict):
                            ai_strategy["result"] += " | [레짐 BEARISH] 약세장 국면"
        except Exception as _e:
            pass   # hybrid 실패 시 기존 점수 유지

        # ── Step 6: 신호 신뢰도 종합 엔진 (거시·섹터·실적·불일치·뉴스감정·신뢰구간) ──
        #   confidence_engine에서 4개 소스 점수를 받아 단일 confidence + 신뢰구간 산출.
        #   외부 호출(yfinance/HF) 실패 시 부분 결과로 degrade — 전체 분석 비중단(additive).
        signal_confidence = None
        try:
            from market_briefing.confidence_engine import build_signal_confidence
            # 4개 소스 점수 (0~100) 매핑
            _tech_sc = float(score)
            _ai_sc = None
            if hybrid_score and "ncs" in hybrid_score and "error" not in hybrid_score:
                _ai_sc = float(hybrid_score["ncs"])           # NCS → AI 점수 대용
            _mkt_sc = {"BULL": 70.0, "NEUTRAL": 50.0, "BEAR": 30.0}.get(regime, 50.0)
            # 종목 5일 변화율 (섹터 상대 비교용)
            _cl5 = [float(c) for c in (dd.get("Close") or []) if c is not None]
            _pct5 = ((_cl5[-1] - _cl5[-6]) / _cl5[-6] * 100.0) if len(_cl5) >= 6 and _cl5[-6] else None
            # 뉴스 → 감정 분석 입력 (title + published + 출처유형)
            _news_in = [{"title": n.get("title"), "source": n.get("publisher"),
                         "source_type": "google_news",
                         "published": n.get("published")} for n in (news or []) if n.get("title")]
            # KRX: 네이버 종목뉴스 + 공시 + DART(키 설정 시)를 감정 입력에 병합.
            #   출처유형(source_type)을 부여 → confidence_engine이 신뢰도 가중을 적용
            #   (DART/공시 > 네이버 뉴스 > 포털 RSS). 한국어 헤드라인은 KR-FinBERT로 자동 분석.
            if market == "KRX" and naver:
                for _n in (naver.get("news") or []):
                    if _n.get("title"):
                        _news_in.append({"title": _n["title"], "source": "naver",
                                         "source_type": "naver", "published": _n.get("date")})
                for _d in (naver.get("disclosures") or []):
                    if _d.get("title"):
                        _news_in.append({"title": _d["title"], "source": "공시",
                                         "source_type": "disclosure", "published": _d.get("date")})
                try:
                    from market_briefing.data_fetcher import fetch_dart_disclosures
                    _code = sym.replace(".KS", "").replace(".KQ", "")
                    for _d in (fetch_dart_disclosures(_code) or []):
                        if _d.get("title"):
                            _news_in.append({"title": _d["title"], "source": "DART",
                                             "source_type": "dart", "published": _d.get("date")})
                except Exception:
                    pass
            signal_confidence = build_signal_confidence(
                technical_score = _tech_sc,
                ai_score        = _ai_sc,
                market_score    = _mkt_sc,
                symbol          = sym,
                market          = market,
                stock_pct5d     = _pct5,
                news_items      = _news_in or None,
                # 거시/섹터/실적은 미국 종목에서 의미가 크나, KRX도 거시·실적은 적용.
                include_sector  = (market == "US"),   # 섹터 ETF는 US 한정
            )
        except Exception:
            signal_confidence = None   # 엔진 실패 시 기존 응답 유지

        return {
            "symbol": sym, "company": company or sym, "market": market,
            "last_close": round(last, 2), "prev_close": round(prev, 2),
            "pct_change": round(pct, 2),
            "session_name": session_name,
            "rsi": round(float(dd.get("RSI", [50])[-1] or 50), 1),
            "volume": int(dd.get("Volume", [0])[-1] or 0),
            "atr": round(atr_val, 2),
            "score": score, "prob_up": prob_up, "prob_down": prob_down,
            "analysis_steps": steps, "ai_strategy": ai_strategy,
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
            "pullback_analysis": pullback_analysis,
            "news": news or [], "naver": naver, "us_enriched": us_enriched,
            "investor_flow": investor_flow,
            "hybrid_score": hybrid_score,  # HybridTurtle NCS/BQS/FWS
            "signal_confidence": signal_confidence,  # 신뢰도 종합(거시·섹터·실적·불일치·뉴스감정·신뢰구간)
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

    if path == "/api/toss-ai-summary":
        # 토스증권 AI 요약 단일 종목 조회 (KRX + US, TTL 5분)
        t_raw  = params.get("ticker", "").strip()
        m_raw  = params.get("market", "").strip().upper()
        if not t_raw:
            return {"error": "ticker 파라미터 필요", "ai_summary": "", "supported": False}
        return fetch_toss_ai_summary(t_raw, m_raw)

    if path == "/api/toss-debug":
        # 토스증권 productCode 조회 디버그 (US 종목 문제 진단용)
        # 캐시 우회 버전 — 실제 네트워크 응답 상태를 직접 반환
        t_raw = params.get("ticker", "").strip().upper()
        if not t_raw:
            return {"error": "ticker 파라미터 필요"}
        # 랭킹 캐시 현재 상태 확인 (Korean 역조회 완료 여부)
        _ranking_snap = dict(_TOSS_RANKING_CACHE)
        debug_info: dict = {
            "ticker": t_raw,
            "ranking_cache_size": len(_ranking_snap),
            "ticker_in_ranking_cache": t_raw in _ranking_snap,
            "ranking_productCode": _ranking_snap.get(t_raw, "(없음)"),
            "ranking_cache_ts_age_s": round(time.monotonic() - _TOSS_RANKING_CACHE_TS, 1),
            "endpoints": [],
        }
        test_urls = [
            ("GET", "https://wts-info-api.tossinvest.com/api/v1/stock-infos/search",
             {"query": t_raw, "size": 5}),
            ("GET", "https://wts-info-api.tossinvest.com/api/v1/search/stocks",
             {"query": t_raw, "size": 5}),
            ("GET", "https://wts-info-api.tossinvest.com/api/v1/search",
             {"query": t_raw, "size": 5, "market": "OVERSEAS"}),
            ("GET", "https://wts-info-api.tossinvest.com/api/v1/search",
             {"query": t_raw, "size": 5}),
            ("GET", "https://wts-info-api.tossinvest.com/api/v1/products/search",
             {"query": t_raw, "size": 5}),
        ]
        for method, url, p in test_urls:
            entry: dict = {"method": method, "url": url, "params": p}
            try:
                r = requests.get(url, params=p, headers=_TOSS_API_HEADERS, timeout=6)
                entry["status"] = r.status_code
                try:
                    body = r.json()
                    # 응답 상위 구조만 반환 (전체는 너무 큼)
                    if isinstance(body, dict):
                        entry["top_keys"] = list(body.keys())[:10]
                        result_part = body.get("result") or body
                        if isinstance(result_part, dict):
                            entry["result_keys"] = list(result_part.keys())[:10]
                    entry["sample"] = str(body)[:400]
                except Exception:
                    entry["body_raw"] = r.text[:200]
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {e}"
            debug_info["endpoints"].append(entry)
        # 상세 API 직접 시도 결과도 포함
        detail_results = []
        for pt in ("STOCKS", "FOREIGN_STOCKS", "US_STOCKS", "OVERSEAS_STOCKS", "OVERSEAS"):
            txt = _toss_detail_api(t_raw, pt)
            detail_results.append({"productType": pt, "result": txt or "(없음)"})
        debug_info["detail_api_attempts"] = detail_results
        # 배치 API 직접 시도
        batch_result = _toss_batch_api(t_raw)
        debug_info["batch_api_with_ticker"] = batch_result
        return debug_info

    if path == "/api/cron":
        try:
            fetch_screener()
            fetch_toss_overseas_screener()
            return {"status": "ok", "message": "Cache warmed (domestic + toss overseas)"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    if path == "/api/price":
        # 미국 주식 현재가 전용 경량 엔드포인트 — 캐시 없음, 폴링용
        ticker_raw = params.get("ticker", "").strip()
        market_p   = params.get("market", "US").strip().upper()
        if not ticker_raw or market_p != "US":
            return {"error": "ticker required, US market only"}
        ticker_r, _, sym_r = resolve_ticker(ticker_raw)
        if not ticker_r:
            return {"error": f"'{ticker_raw}' not found"}
        fetcher = _get_us_price_fetcher()
        if fetcher is None:
            return {"error": "fetcher unavailable"}
        try:
            res = fetcher.fetch(sym_r)
            if res and res.price > 0:
                p    = float(res.price)
                pc   = float(res.prev_close) if res.prev_close else 0.0
                pct  = (p - pc) / pc * 100 if pc else 0.0
                sn   = _PRICE_TYPE_LABEL.get(res.price_type, "정규장")
                return {"price": round(p, 4), "prev_close": round(pc, 4),
                        "pct_change": round(pct, 4), "session_name": sn}
        except Exception:
            pass
        return {"error": "price unavailable"}

    if path == "/api/investor-flow":
        # 투자자 수급 전용 경량 엔드포인트 — 메인 분석과 분리하여 타임아웃 경합 제거
        ticker_raw = params.get("ticker", "").strip()
        if not ticker_raw:
            return {"ok": False, "reason": "ticker 파라미터 없음"}
        # resolve_ticker 우회: "005930.KS" 같은 ASCII 심볼이 US로 오분류되는 문제 방지
        # (resolve_ticker 규칙 4 — 전체 ASCII → US ticker 직접 처리)
        code = ticker_raw.replace(".KS", "").replace(".KQ", "").strip()
        if code.isdigit() and len(code) == 6:
            # yfinance KRX 형식(005930.KS / 005930.KQ) 또는 6자리 코드 → 직접 전달
            return fetch_investor_flow(ticker_raw)
        # 종목명 입력(예: 삼성전자) → resolve_ticker 경유
        ticker, market, _ = resolve_ticker(ticker_raw)
        if not ticker or market != "KRX":
            return {"ok": False, "reason": "KRX 종목이 아닙니다"}
        return fetch_investor_flow(ticker)

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
            return {"error": f"업종별 흐름 조회 실패: {e}"}

    if path == "/api/market/sector-summary":
        # 코드 파라미터 없이 기본 대표 종목으로 섹터 흐름 반환
        # 메인 페이지 자동 로드용 — 캐시 s-maxage=600 (10분)
        # quote만 병렬 수집 (history/news 생략) → 대폭 빠름
        try:
            import sys as _sys
            import os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from market_briefing.data_fetcher import fetch_stock_list_quote_cached
            from market_briefing.sector_flow import build_sector_flow
            snapshots = fetch_stock_list_quote_cached(_SECTOR_DEFAULT_STOCKS)
            return build_sector_flow(snapshots)
        except Exception as e:
            return {"error": f"업종별 흐름 조회 실패: {e}"}

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

    if path == "/api/kr/longterm":
        return fetch_kr_longterm_reco()

    if path == "/api/us/longterm":
        return fetch_us_longterm_reco()

    if path == "/api/us/opening-surge":
        return fetch_us_opening_surge()

    # ── HybridTurtle 통합 엔드포인트 ─────────────────────────────────────────
    # /api/scan             → 7단계 스캔 엔진 실행
    # /api/dashboard-status → 대시보드 커맨드 센터 상태
    # /api/portfolio        → 포트폴리오 리스크 평가
    # /api/market-immune    → 시장 위험 면역 시스템

    if path == "/api/scan":
        """7단계 스캔 엔진 API.

        파라미터:
          tickers  — 쉼표 구분 종목 코드 (선택, 없으면 기본 유니버스)
          market   — KRX / US (기본 KRX)
          equity   — 포트폴리오 자본 (숫자, 기본 100000000)
          risk_pct — 거래당 리스크 비율 (기본 1.0)
          mode     — FULL / CORE_LITE (기본 FULL)
        """
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from market_briefing.scan_engine import (
                StockUniverse, run_full_scan, build_snapshot_from_ohlcv
            )
            from market_briefing.quality_filter import get_quality_score_from_info
            from market_briefing.hybrid_signals import compute_regime, detect_vol_regime, _calc_atr, _calc_adx, _calc_ma
            from market_briefing.dual_score_v2 import REGIME_BULLISH, REGIME_BEARISH, REGIME_SIDEWAYS, VOL_NORMAL

            market_p = params.get("market", "KRX").upper()
            equity   = float(params.get("equity", 100_000_000))
            risk_pct = float(params.get("risk_pct", 1.0))
            mode_p   = params.get("mode", "FULL").upper()
            if mode_p not in ("FULL", "CORE_LITE"):
                mode_p = "FULL"

            # 종목명 조회 헬퍼 ─────────────────────────────────────────────────
            # KRX 기본 유니버스 한국어 이름 사전 (오프라인 우선 조회)
            _KRX_NAME_MAP = {
                "005930.KS": "삼성전자",        "005930.KQ": "삼성전자",
                "000660.KS": "SK하이닉스",
                "035420.KS": "NAVER",
                "051910.KS": "LG화학",
                "035720.KS": "카카오",
                "207940.KS": "삼성바이오로직스",
                "006400.KS": "삼성SDI",
                "028260.KS": "삼성물산",
                "105560.KS": "KB금융",
                "055550.KS": "신한지주",
                "000270.KS": "기아",           "005380.KS": "현대차",
                "068270.KS": "셀트리온",        "003550.KS": "LG",
                "005490.KS": "POSCO홀딩스",    "034730.KS": "SK",
                "373220.KS": "LG에너지솔루션", "247540.KS": "에코프로비엠",
                "086520.KS": "에코프로",        "323410.KS": "카카오뱅크",
                "352820.KS": "하이브",          "259960.KS": "크래프톤",
                "034020.KS": "두산에너빌리티", "012330.KS": "현대모비스",
                "066570.KS": "LG전자",         "003670.KS": "포스코퓨처엠",
                "028050.KS": "삼성엔지니어링",  "010130.KS": "고려아연",
            }

            def _get_name(tkr: str, info_dict: dict = None) -> str:
                """ticker → 한국어/영어 종목명 조회 (우선순위: 사전 → resolve_ticker → info → ticker)."""
                # 1. 사전 조회 (오프라인, 가장 빠름)
                if tkr in _KRX_NAME_MAP:
                    return _KRX_NAME_MAP[tkr]
                # 2. resolve_ticker() — KR_STOCK_MAP + KRX API 활용
                try:
                    code = tkr.replace(".KS", "").replace(".KQ", "").strip()
                    if code.isdigit() and len(code) == 6:
                        _, _, cname = resolve_ticker(code)
                        if cname and cname != code:
                            return cname
                except Exception:
                    pass
                # 3. yfinance info shortName / longName
                if info_dict:
                    name = info_dict.get("shortName") or info_dict.get("longName")
                    if name:
                        return name
                # 4. 최후 fallback: ticker 코드
                return tkr

            # ── 확장 유니버스 (KOSPI/KOSDAQ · S&P500/중소형) ──
            # (ticker, 종목명, 시총티어) — 티어는 정적 힌트(실측 marketCap이 우선 보정).
            #   대형주뿐 아니라 중형·중소형까지 풀에 포함해야 균형 선정이 가능하다.
            _SCAN_KRX_UNIVERSE = [
                # ── 대형주 (LARGE) ──
                ("005930.KS", "삼성전자", "LARGE"),       ("000660.KS", "SK하이닉스", "LARGE"),
                ("373220.KS", "LG에너지솔루션", "LARGE"), ("207940.KS", "삼성바이오로직스", "LARGE"),
                ("005380.KS", "현대차", "LARGE"),         ("000270.KS", "기아", "LARGE"),
                ("068270.KS", "셀트리온", "LARGE"),       ("105560.KS", "KB금융", "LARGE"),
                ("055550.KS", "신한지주", "LARGE"),       ("005490.KS", "POSCO홀딩스", "LARGE"),
                ("035420.KS", "NAVER", "LARGE"),          ("035720.KS", "카카오", "LARGE"),
                ("051910.KS", "LG화학", "LARGE"),         ("006400.KS", "삼성SDI", "LARGE"),
                ("012330.KS", "현대모비스", "LARGE"),     ("028260.KS", "삼성물산", "LARGE"),
                ("012450.KS", "한화에어로스페이스", "LARGE"), ("034020.KS", "두산에너빌리티", "LARGE"),
                ("247540.KQ", "에코프로비엠", "LARGE"),   ("196170.KQ", "알테오젠", "LARGE"),
                # ── 중형주 (MID) ──
                ("009150.KS", "삼성전기", "MID"),         ("009540.KS", "HD한국조선해양", "MID"),
                ("015760.KS", "한국전력", "MID"),         ("033780.KS", "KT&G", "MID"),
                ("086790.KS", "하나금융지주", "MID"),     ("003670.KS", "포스코퓨처엠", "MID"),
                ("259960.KS", "크래프톤", "MID"),         ("086520.KQ", "에코프로", "MID"),
                ("028300.KQ", "HLB", "MID"),              ("145020.KQ", "휴젤", "MID"),
                ("263750.KQ", "펄어비스", "MID"),         ("293490.KQ", "카카오게임즈", "MID"),
                ("112040.KQ", "위메이드", "MID"),         ("041510.KQ", "에스엠", "MID"),
                ("011200.KS", "HMM", "MID"),              ("010140.KS", "삼성중공업", "MID"),
                # ── 중소형주 (SMALL) ──
                ("058470.KQ", "리노공업", "SMALL"),       ("035900.KQ", "JYP Ent.", "SMALL"),
                ("214150.KQ", "클래시스", "SMALL"),       ("357780.KQ", "솔브레인", "SMALL"),
                ("095340.KQ", "ISC", "SMALL"),            ("140860.KQ", "파크시스템스", "SMALL"),
                ("098460.KQ", "고영", "SMALL"),           ("222800.KQ", "심텍", "SMALL"),
                ("240810.KQ", "원익IPS", "SMALL"),        ("178320.KQ", "서진시스템", "SMALL"),
            ]
            _SCAN_US_UNIVERSE = [
                # ── 대형주 (LARGE) ──
                ("AAPL", "Apple", "LARGE"),       ("NVDA", "NVIDIA", "LARGE"),
                ("MSFT", "Microsoft", "LARGE"),   ("GOOGL", "Alphabet", "LARGE"),
                ("AMZN", "Amazon", "LARGE"),      ("META", "Meta", "LARGE"),
                ("TSLA", "Tesla", "LARGE"),       ("AVGO", "Broadcom", "LARGE"),
                ("LLY", "Eli Lilly", "LARGE"),    ("JPM", "JPMorgan", "LARGE"),
                ("V", "Visa", "LARGE"),           ("UNH", "UnitedHealth", "LARGE"),
                ("XOM", "Exxon Mobil", "LARGE"),  ("MA", "Mastercard", "LARGE"),
                ("JNJ", "Johnson & Johnson", "LARGE"), ("COST", "Costco", "LARGE"),
                ("NFLX", "Netflix", "LARGE"),     ("AMD", "AMD", "LARGE"),
                ("ORCL", "Oracle", "LARGE"),      ("CRM", "Salesforce", "LARGE"),
                # ── 중형주 (MID) ──
                ("ABBV", "AbbVie", "MID"),        ("MRK", "Merck", "MID"),
                ("ADBE", "Adobe", "MID"),         ("CVX", "Chevron", "MID"),
                ("PEP", "PepsiCo", "MID"),        ("KO", "Coca-Cola", "MID"),
                ("BAC", "Bank of America", "MID"),("HD", "Home Depot", "MID"),
                ("PG", "Procter & Gamble", "MID"),("MCD", "McDonald's", "MID"),
                ("RBLX", "Roblox", "MID"),        ("DKNG", "DraftKings", "MID"),
                ("HOOD", "Robinhood", "MID"),     ("NET", "Cloudflare", "MID"),
                ("DDOG", "Datadog", "MID"),       ("ROKU", "Roku", "MID"),
                # ── 중소형주 (SMALL) ──
                ("RIVN", "Rivian", "SMALL"),      ("AFRM", "Affirm", "SMALL"),
                ("U", "Unity", "SMALL"),          ("PINS", "Pinterest", "SMALL"),
                ("IONQ", "IonQ", "SMALL"),        ("RKLB", "Rocket Lab", "SMALL"),
                ("SOFI", "SoFi", "SMALL"),        ("LMND", "Lemonade", "SMALL"),
                ("CHPT", "ChargePoint", "SMALL"), ("FUBO", "fuboTV", "SMALL"),
            ]

            # 종목 목록 파싱 — tickers 파라미터 우선, 없으면 시장별 확장 유니버스
            tickers_raw = params.get("tickers", "")
            name_hint: dict = {}
            cap_hint:  dict = {}   # ticker → 정적 시총티어 힌트
            if tickers_raw:
                raw_list = [t.strip() for t in tickers_raw.split(",") if t.strip()]
            else:
                triples   = _SCAN_KRX_UNIVERSE if market_p == "KRX" else _SCAN_US_UNIVERSE
                name_hint = {t: nm for t, nm, _ in triples}
                cap_hint  = {t: (tier or "MID").upper() for t, _, tier in triples}
                # 티어 라운드로빈 인터리브(LARGE→MID→SMALL→…) — 수집 상한 truncation
                # 시에도 대형/중형/중소형이 고르게 수집되도록 한다.
                _by_tier = {"LARGE": [], "MID": [], "SMALL": []}
                for t, _, tier in triples:
                    _by_tier.setdefault((tier or "MID").upper(), _by_tier["MID"]).append(t)
                from itertools import zip_longest as _zl
                raw_list = [
                    t for grp in _zl(_by_tier["LARGE"], _by_tier["MID"], _by_tier["SMALL"])
                    for t in grp if t
                ]

            # 스캔 모드별 수집 상한 — 한국/미국 동일 적용 (Vercel 60s 타임아웃 방지)
            #   상수: SCAN_COLLECT_CAP_FULL(48) / SCAN_COLLECT_CAP_LITE(24)
            #   인터리브된 raw_list이므로 truncation 후에도 티어가 고르게 남는다.
            _scan_cap = SCAN_COLLECT_CAP_FULL if mode_p == "FULL" else SCAN_COLLECT_CAP_LITE
            raw_list  = raw_list[:_scan_cap]

            # ── 응답 캐시 조회 (warm 인스턴스 재사용) ──────────────────────────
            #   동일 (시장·모드·자본·리스크·종목집합) 요청은 5분간 재사용.
            #   refresh=1 파라미터로 강제 갱신 가능.
            _refresh   = str(params.get("refresh", "")).lower() in ("1", "true", "yes")
            _cache_key = f"scan|{market_p}|{mode_p}|{equity}|{risk_pct}|{','.join(raw_list)}"
            if not _refresh:
                _hit = _SCAN_RESULT_CACHE.get(_cache_key)
                if _hit and (time.time() - _hit[1]) < _SCAN_RESULT_TTL:
                    return _hit[0]

            def _hist_to_lists(hist):
                """yf.Ticker().history() DataFrame → (closes, highs, lows, volumes, opens)."""
                if hist is None or hist.empty:
                    return None
                cl = hist["Close"].dropna().tolist()
                hi = hist["High"].dropna().tolist()
                lo = hist["Low"].dropna().tolist()
                vo = hist["Volume"].dropna().tolist()
                op = hist["Open"].dropna().tolist()
                n  = min(len(cl), len(hi), len(lo), len(vo), len(op))
                if n < 60:
                    return None
                return cl[:n], hi[:n], lo[:n], vo[:n], op[:n]

            # 벤치마크 데이터 (레짐 감지용) — Ticker.history() 사용
            bench_sym = "^KS200" if market_p == "KRX" else "SPY"
            bench_closes = []
            bench_highs  = []
            bench_lows   = []
            try:
                _bh = yf.Ticker(bench_sym).history(period="1y")
                if not _bh.empty:
                    bench_closes = _bh["Close"].dropna().tolist()
                    bench_highs  = _bh["High"].dropna().tolist()
                    bench_lows   = _bh["Low"].dropna().tolist()
            except Exception:
                pass

            # 레짐 감지
            regime     = REGIME_SIDEWAYS
            vol_regime = VOL_NORMAL
            if len(bench_closes) >= 20:
                _bn = min(len(bench_closes), len(bench_highs), len(bench_lows))
                adx_d = _calc_adx(
                    bench_highs[-50:] if _bn >= 50 else bench_highs,
                    bench_lows[-50:]  if _bn >= 50 else bench_lows,
                    bench_closes[-50:] if _bn >= 50 else bench_closes,
                ) if _bn >= 30 else None
                ma200 = _calc_ma(bench_closes, 200) if len(bench_closes) >= 200 else None
                vix_v = None
                try:
                    _vh = yf.Ticker("^VIX").history(period="5d")
                    if not _vh.empty:
                        vix_v = float(_vh["Close"].iloc[-1])
                except Exception:
                    pass
                rd = compute_regime(bench_closes[-1], ma200 or bench_closes[-1], adx_d, vix_v)
                regime = rd["regime"]
                atr_v  = _calc_atr(bench_highs[-20:], bench_lows[-20:], bench_closes[-20:]) if _bn >= 20 else None
                if atr_v and bench_closes[-1] > 0:
                    vol_regime = detect_vol_regime(atr_v / bench_closes[-1] * 100)

            # OHLCV 수집 + 스냅샷/품질 빌드
            universe    = []
            snap_map    = {}
            quality_map = {}
            change_map  = {}   # ticker → 등락률 %
            sector_map  = {}   # ticker → 섹터/카테고리
            signal_map  = {}   # ticker → 애널리스트 신호
            cap_map     = {}   # ticker → 시총티어 (LARGE/MID/SMALL)
            mcap_map    = {}   # ticker → 시가총액 (원/달러)

            _ANALYST_MAP = {
                "strong_buy": "적극 매수", "buy": "매수",
                "hold": "보유", "underperform": "약세", "sell": "매도",
            }

            _ETF_SET = {"QQQ", "SPY", "SOXL", "TQQQ", "SQQQ", "DIA", "IWM"}

            def _scan_collect_one(tkr: str):
                """단일 종목 정밀 수집 — OHLCV·스냅샷·품질(QMJ)·섹터·애널리스트 신호·이름.

                병렬 워커로 호출됨. info 조회 실패 시에도 가격/기술 분석은 유지(graceful degrade).
                """
                try:
                    tk    = yf.Ticker(tkr)
                    ohlcv = _hist_to_lists(tk.history(period="1y"))
                    if ohlcv is None:
                        return None
                    closes, highs, lows, volumes, opens = ohlcv
                    snap = build_snapshot_from_ohlcv(
                        ticker=tkr, closes=closes, highs=highs,
                        lows=lows, volumes=volumes, opens=opens,
                        bench_closes=bench_closes,
                    )
                    # 등락률: 전일 대비 당일 종가 변화율
                    change = round((closes[-1] - closes[-2]) / closes[-2] * 100, 2) \
                             if len(closes) >= 2 and closes[-2] != 0 else 0.0
                    # info: 품질(QMJ)·섹터·애널리스트 신호 (실패 허용)
                    _info, quality = {}, None
                    try:
                        _info   = tk.info or {}
                        quality = get_quality_score_from_info(tkr, _info)
                    except Exception:
                        pass
                    sector  = _info.get("sector") or _info.get("industry") or ""
                    rec_key = (_info.get("recommendationKey") or "").lower()
                    signal  = _ANALYST_MAP.get(rec_key, "중립")
                    name    = name_hint.get(tkr) or _get_name(tkr, _info)
                    sleeve  = "ETF" if tkr in _ETF_SET else "CORE"
                    # 시가총액 → 티어 (실측 marketCap 우선, 없으면 정적 힌트)
                    mcap     = _scan_num(_info.get("marketCap"), 0.0)
                    cap_tier = scan_cap_tier(market_p, mcap, cap_hint.get(tkr, ""))
                    return {
                        "tkr": tkr, "snap": snap, "change": change, "quality": quality,
                        "sector": sector, "signal": signal, "name": name, "sleeve": sleeve,
                        "cap_tier": cap_tier, "market_cap": mcap,
                    }
                except Exception:
                    return None

            # 병렬 수집 — 시간 예산(SCAN_COLLECT_BUDGET_S) 내 도착분만으로 진행.
            #   대형 유니버스(최대 48)라도 느린 종목이 전체를 막지 않도록 하드가드.
            def _ingest(res):
                if not res:
                    return
                t = res["tkr"]
                snap_map[t]   = res["snap"]
                change_map[t] = res["change"]
                if res["quality"] is not None:
                    quality_map[t] = res["quality"]
                sector_map[t] = res["sector"]
                signal_map[t] = res["signal"]
                cap_map[t]    = res.get("cap_tier", "MID")
                mcap_map[t]   = res.get("market_cap", 0.0)
                universe.append(StockUniverse(t, res["name"], res["sleeve"], sector=res["sector"]))

            _workers = min(SCAN_COLLECT_WORKERS, max(1, len(raw_list)))
            _ex = concurrent.futures.ThreadPoolExecutor(max_workers=_workers)
            try:
                _futs = [_ex.submit(_scan_collect_one, t) for t in raw_list]
                try:
                    for _fut in concurrent.futures.as_completed(_futs, timeout=SCAN_COLLECT_BUDGET_S):
                        try:
                            _ingest(_fut.result())
                        except Exception:
                            continue
                except concurrent.futures.TimeoutError:
                    # 예산 초과 — 이미 수집된 종목만으로 진행 (부분 결과 허용)
                    pass
            finally:
                # 미착수 작업 취소, 진행 중 작업은 대기하지 않고 즉시 반환
                try:
                    _ex.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    _ex.shutdown(wait=False)   # py<3.9 호환

            if not snap_map:
                return {"error": "데이터 수집 실패 — 네트워크 연결을 확인하거나 잠시 후 다시 시도하세요"}

            result = run_full_scan(
                universe           = [u for u in universe if u.ticker in snap_map],
                snap_map           = snap_map,
                quality_map        = quality_map,
                regime             = regime,
                vol_regime         = vol_regime,
                portfolio_equity   = equity,
                risk_pct_per_trade = risk_pct,
                scan_mode          = mode_p,
            )

            from dataclasses import asdict
            cands = []
            for c in result.candidates:
                d = asdict(c)
                d["change_pct"]     = change_map.get(c.ticker, 0.0)
                d["category"]       = sector_map.get(c.ticker, "") or c.sector or ""
                d["analyst_signal"] = signal_map.get(c.ticker, "중립")
                d["cap_tier"]       = cap_map.get(c.ticker, "MID")
                d["cap_tier_ko"]    = SCAN_CAP_TIER_KO.get(d["cap_tier"], "중형")
                d["market_cap"]     = mcap_map.get(c.ticker, 0.0)
                cands.append(d)

            # ── 출력 선정: 정확히 15개 보장 + 품질 우선 단계적 보강 (한국/미국 동일) ──
            #   모든 후보에 종합점수 부여 후, '품질 등급(tier)'을 순서대로 채운다.
            #     · 등급1(최우선): 기술필터 통과 + 진입 가능권(READY/WATCH/눌림목)
            #     · 등급2(보강) : 기술필터 통과 + 그 외(주로 FAR — 추세 양호, 진입가만 먼 종목)
            #     · 등급3(최후) : 기술필터 미통과 (종합점수 상위로만 최소 보강)
            #   각 등급 내부는 종합점수 순 + 시총티어·섹터 분산(MMR)으로 선정하며,
            #   15개에 도달할 때까지만 다음 등급으로 내려간다(품질 우선). 실적대기 제외.
            for cd in cands:
                cd["quant_momentum_score"] = scan_quant_momentum_score(cd)
                cd["composite_score"]      = scan_composite_score(cd)

            def _is_tier1(cd: dict) -> bool:
                return bool(cd.get("passes_tech_filters")) and cd.get("status") in SCAN_GOOD_STATES

            def _is_tier2(cd: dict) -> bool:
                return (bool(cd.get("passes_tech_filters"))
                        and not _is_tier1(cd)
                        and cd.get("status") != "EARNINGS_BLOCK")

            def _is_tier3(cd: dict) -> bool:
                return (not bool(cd.get("passes_tech_filters"))
                        and cd.get("status") != "EARNINGS_BLOCK")

            def _by_score(lst):
                return sorted(lst, key=lambda cd: (-cd["composite_score"], -scan_ncs_score(cd)))

            tier1 = _by_score([cd for cd in cands if _is_tier1(cd)])
            tier2 = _by_score([cd for cd in cands if _is_tier2(cd)])
            tier3 = _by_score([cd for cd in cands if _is_tier3(cd)])

            # 등급 순차 보강 — 분산 카운터를 공유해 등급을 넘어 다양성 유지
            _sector_cnt: dict = {}
            _cap_cnt:    dict = {}
            selected: list = []
            _tier_used = {1: 0, 2: 0, 3: 0}
            for _lvl, _tpool in ((1, tier1), (2, tier2), (3, tier3)):
                if len(selected) >= SCAN_DISPLAY_CAP:
                    break
                _need = SCAN_DISPLAY_CAP - len(selected)
                _got  = scan_diversified_fill(_tpool, _need, _sector_cnt, _cap_cnt)
                for _c in _got:
                    _c["selection_tier"] = _lvl
                _tier_used[_lvl] = len(_got)
                selected += _got

            _relaxed = (_tier_used[2] > 0 or _tier_used[3] > 0)

            # 분포 집계 (투명성·UI 표기용)
            _tier_dist = {"LARGE": 0, "MID": 0, "SMALL": 0}
            _sector_dist: dict = {}
            for cd in selected:
                _tier_dist[cd.get("cap_tier", "MID")] = _tier_dist.get(cd.get("cap_tier", "MID"), 0) + 1
                _sec = _scan_sector_key(cd) or "기타"
                _sector_dist[_sec] = _sector_dist.get(_sec, 0) + 1

            _resp = {
                "regime":          result.regime,
                "vol_regime":      result.vol_regime,
                "total_scanned":   result.total_scanned,
                "passed_filters":  result.passed_filters,
                "ready_count":     result.ready_count,
                "watch_count":     result.watch_count,
                "far_count":       result.far_count,
                "good_count":      len(selected),
                "premium_count":   _tier_used[1],          # 진입 가능권(최우선) 충족 수
                "relaxed":         _relaxed,               # 보강 로직 사용 여부
                "tier_used":       _tier_used,             # 등급별 선정 수
                "display_cap":     SCAN_DISPLAY_CAP,
                "tier_distribution":   _tier_dist,
                "sector_distribution": _sector_dist,
                "filter_desc":     "진입 가능권 우선 + 품질 등급 단계 보강(정확히 15개) + 시총·섹터 분산(MMR)",
                "candidates":      selected,
                "generated_at":    result.generated_at,
            }
            # 응답 캐시 저장 (만료 키 정리 포함)
            _now = time.time()
            _SCAN_RESULT_CACHE[_cache_key] = (_resp, _now)
            if len(_SCAN_RESULT_CACHE) > 50:
                for _k in [k for k, (_, t) in list(_SCAN_RESULT_CACHE.items())
                           if _now - t > _SCAN_RESULT_TTL]:
                    _SCAN_RESULT_CACHE.pop(_k, None)
            return _resp
        except Exception as e:
            return {"error": f"스캔 실행 실패: {e}"}

    if path == "/api/dashboard-status":
        """대시보드 커맨드 센터 상태."""
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from market_briefing.dashboard_payload import DashboardPayloadBuilder
            builder = DashboardPayloadBuilder()
            return builder.build()
        except Exception as e:
            return {"error": f"대시보드 상태 조회 실패: {e}",
                    "status": "ERROR", "generated_at": dt.now().isoformat()}

    if path == "/api/market-immune":
        """시장 위험 면역 시스템 — VIX + MA200 이격도 기반.

        파라미터:
          vix       — VIX 현재값 (선택)
          ma200_dev — MA200 이격도 % (선택)
          atr_pct   — 지수 ATR% (선택)
          index     — 지수 심볼 (선택, 기본 SPY)
        """
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from market_briefing.market_immune import MarketImmune

            mi = MarketImmune()

            # 직접 파라미터 입력 우선
            vix_p     = params.get("vix")
            ma200_p   = params.get("ma200_dev")
            atr_pct_p = params.get("atr_pct")

            if vix_p is not None or ma200_p is not None:
                vix_f    = float(vix_p) if vix_p else None
                ma200_f  = float(ma200_p) if ma200_p else None
                atr_f    = float(atr_pct_p) if atr_pct_p else None
                result   = mi.quick_check(vix_f, ma200_f, atr_f)
                return {**result, "mode": "quick_check"}

            # 지수 데이터 자동 수집 — Ticker.history() 사용
            idx_sym = params.get("index", "SPY")
            try:
                _ih   = yf.Ticker(idx_sym).history(period="1y")
                vix_v = None
                try:
                    _vh = yf.Ticker("^VIX").history(period="5d")
                    if not _vh.empty:
                        vix_v = float(_vh["Close"].iloc[-1])
                except Exception:
                    pass

                if _ih.empty:
                    return {"error": f"지수({idx_sym}) 데이터 없음"}

                ir = mi.assess(
                    index_closes = _ih["Close"].dropna().tolist(),
                    index_highs  = _ih["High"].dropna().tolist(),
                    index_lows   = _ih["Low"].dropna().tolist(),
                    vix          = vix_v,
                )
                return ir.to_dict()
            except Exception as e2:
                return {"error": f"지수 데이터 수집 실패: {e2}"}
        except Exception as e:
            return {"error": f"면역 시스템 오류: {e}"}

    if path == "/api/portfolio":
        """포트폴리오 리스크 평가 — 신규 거래 사전 평가.

        파라미터:
          action      — assess (리스크 평가) / status (상태 조회)
          ticker      — 종목 코드
          entry_price — 진입가
          stop_price  — 손절가
          equity      — 자본 (기본 100000000)
          risk_pct    — 거래당 리스크 % (기본 1.0)
        """
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from market_briefing.portfolio_manager import PortfolioManager

            action = params.get("action", "status")
            equity = float(params.get("equity", 100_000_000))
            rp     = float(params.get("risk_pct", 1.0))
            mgr    = PortfolioManager(equity=equity, cash_balance=equity, risk_pct_per_trade=rp)

            if action == "assess":
                tkr    = params.get("ticker", "").strip().upper()
                entry  = params.get("entry_price")
                stop   = params.get("stop_price")
                if not tkr or not entry or not stop:
                    return {"error": "assess 액션은 ticker, entry_price, stop_price 필요"}
                assessment = mgr.assess_new_trade(
                    ticker      = tkr,
                    entry_price = float(entry),
                    stop_price  = float(stop),
                    sector      = params.get("sector", ""),
                    sleeve      = params.get("sleeve", "CORE"),
                )
                return assessment.to_dict()
            else:
                return mgr.get_dashboard_payload()
        except Exception as e:
            return {"error": f"포트폴리오 조회 실패: {e}"}

    if path == "/api/cross-reference":
        """스캔 × 분석 교차 참조 — 최종 통합 점수.

        파라미터:
          ticker      — 종목 코드
          model_score — analyze_score() 결과 (0~100, 선택)
        """
        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from market_briefing.cross_reference import CrossReferenceEngine

            engine = CrossReferenceEngine()
            tkr_raw = params.get("ticker", "").strip()
            if not tkr_raw:
                return {"formula": CrossReferenceEngine.final_score_formula()}

            ticker_r, market_r, _ = resolve_ticker(tkr_raw)
            if not ticker_r:
                return {"error": f"'{tkr_raw}' 종목 미발견"}

            model_score_p = float(params.get("model_score", 50.0))
            result = engine.merge(None, None, model_score_p)
            result.ticker = ticker_r
            return result.to_dict()
        except Exception as e:
            return {"error": f"교차 참조 실패: {e}"}

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
.sb-header-top{display:flex;align-items:center;justify-content:space-between;gap:8px}
.sb-header h1{font-size:15px;font-weight:700;display:flex;align-items:center;gap:6px;flex:1;min-width:0}
.sb-header p{font-size:11px;color:#8b949e;margin-top:4px}
.sb-home-btn{background:none;border:none;cursor:pointer;font-size:18px;line-height:1;color:#8b949e;border-radius:8px;min-width:44px;min-height:44px;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:background .15s,color .15s}
.sb-home-btn:hover{background:#30363d;color:#e6edf3}
.sb-section{padding:14px;border-bottom:1px solid #30363d}
.sb-label{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;display:block}
.mkt-btns{display:flex;gap:6px}
.mkt-btn{flex:1;padding:8px;border-radius:8px;border:none;cursor:pointer;font-size:13px;font-weight:500;transition:all .15s}
.mkt-btn.active{background:#1f6feb;color:#fff}
.mkt-btn:not(.active){background:#21262d;color:#8b949e}
.mkt-btn:not(.active):hover{background:#30363d;color:#e6edf3}
/* ── 개장 급등 추천 서브메뉴 ── */
.nav-reco-parent{display:flex;align-items:center;justify-content:space-between;width:100%}
.nav-reco-arrow{font-size:10px;transition:transform .2s;color:#8b949e;flex-shrink:0}
.nav-reco-arrow.open{transform:rotate(90deg)}
.nav-subbtn{flex:unset;width:100%;text-align:left;padding:7px 10px;font-size:12px;border-radius:6px;border-left:2px solid #30363d}
.nav-subbtn.active{border-left-color:#1f6feb}
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
.metrics-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}
.metric-card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px}
/* 카드 기본 span — 데스크탑 기준 (5열 그리드: 2+1+1+1) */
.metric-price-card{grid-column:span 2}
.metric-volume-card{grid-column:span 1}
.metric-atr-card{grid-column:span 1}
.metric-toss-card{grid-column:span 1}
/* 토스증권 AI 요약 카드 내부 스타일 */
.toss-ai-summary{font-size:13px;font-weight:600;color:#e6edf3;line-height:1.55;margin-top:4px;word-break:keep-all}
.toss-ai-time{font-size:10px;color:#484f58;margin-top:6px}
.toss-ai-spinner{display:inline-block;width:10px;height:10px;border:2px solid #30363d;border-top-color:#1f6feb;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:5px}
.metric-price-row{display:flex;align-items:flex-start;gap:20px;flex-wrap:nowrap}
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
/* ── 스캔 테이블 상태 배지 ── */
.scan-status-ready{background:#0d2d1a;color:#3fb950;border:1px solid #1a4730;border-radius:5px;padding:2px 7px;font-size:11px;font-weight:700;white-space:nowrap}
.scan-status-watch{background:#2d2200;color:#d29922;border:1px solid #5a4500;border-radius:5px;padding:2px 7px;font-size:11px;font-weight:700;white-space:nowrap}
.scan-status-far{background:#21262d;color:#484f58;border:1px solid #30363d;border-radius:5px;padding:2px 7px;font-size:11px;font-weight:700;white-space:nowrap}
.scan-status-block{background:#2d0d0d;color:#f85149;border:1px solid #4d1515;border-radius:5px;padding:2px 7px;font-size:11px;font-weight:700;white-space:nowrap}
.scan-status-pullback{background:#1a1033;color:#bc8cff;border:1px solid #3a2060;border-radius:5px;padding:2px 7px;font-size:11px;font-weight:700;white-space:nowrap}
/* 스캔 요약 카드 */
.scan-sum-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px;text-align:center}
.scan-sum-val{font-size:22px;font-weight:800;margin-bottom:2px}
.scan-sum-label{font-size:10px;color:#484f58;text-transform:uppercase;letter-spacing:.04em}
/* 면역 레벨 색상 */
.immune-clear{background:#0d2d1a!important;border-color:#1a4730!important;color:#3fb950}
.immune-caution{background:#2d2200!important;border-color:#5a4500!important;color:#d29922}
.immune-alert{background:#2d1810!important;border-color:#6b3010!important;color:#f97316}
.immune-immune{background:#2d0d0d!important;border-color:#4d1515!important;color:#f85149}
/* 면역 지표 카드 */
.immune-metric{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;text-align:center}
.immune-metric-val{font-size:20px;font-weight:800;margin-bottom:3px}
.immune-metric-label{font-size:10px;color:#484f58;text-transform:uppercase;letter-spacing:.04em}
/* 위기 유사도 바 */
.crisis-bar{height:6px;background:#21262d;border-radius:3px;overflow:hidden;margin-top:4px}
.crisis-bar-fill{height:100%;border-radius:3px;transition:width .6s ease}

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
.pattern-bull{background:#0d2d1a}
.pattern-bear{background:#2d0d0d}
.pattern-neu{background:#21262d}

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
.buy-price-grid{display:flex;flex-direction:column;gap:16px;margin-bottom:16px}
.buy-card{border-radius:12px;padding:12px 14px;border:1px solid transparent}
.buy-card.aggressive{background:#2d200a;border-color:#4d3615}
.buy-card.recommended{background:#0d2d1a;border-color:#1a4730}
.buy-card.conservative{background:#0a1f3a;border-color:#15356b}
.buy-bands-row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:8px}
@media(max-width:900px){.buy-bands-row{grid-template-columns:repeat(2,1fr)}}
@media(max-width:600px){.buy-bands-row{grid-template-columns:1fr}}
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
  /* 900px: 2열 그리드 — 토스 카드 전체 너비 (텍스트 공간 확보) */
  .metric-toss-card{grid-column:span 2}
  .risk-grid{grid-template-columns:1fr}
  .fund-grid{grid-template-columns:repeat(2,1fr)}
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

  /* 그리드: 4열 유지, span 재배치 */
  .metrics-grid{grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;align-items:start}
  .metric-price-card{grid-column:span 3}
  .metric-volume-card{grid-column:span 1}
  .metric-atr-card{grid-column:span 2}
  .metric-toss-card{grid-column:span 2}
  .metric-price-row{gap:14px}
  .two-col-grid{grid-template-columns:1fr;gap:10px}
  .risk-grid{grid-template-columns:1fr}
  .fund-grid{grid-template-columns:repeat(2,1fr)}
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
  /* 768px 이하: 3열 (세로가 짧아도 카드당 여유 확보) */
  .sector-cards{grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}
}

/* ── 소형 모바일 (≤ 480px) ── */
@media(max-width:480px){
  #main{padding:52px 10px 20px}
  .metrics-grid{grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
  .metric-price-card{grid-column:1/-1}
  .metric-volume-card,.metric-atr-card{grid-column:span 2}
  .metric-toss-card{grid-column:1/-1}
  /* 480px 이하: 2열, 카드 패딩 축소로 내용 확보 */
  .sector-cards{grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}
  .sector-card{padding:9px 8px;border-radius:10px}
  .sector-card-emoji{font-size:14px}
  .sector-card-name{font-size:11px}
  .sector-card-pct{font-size:12px}
  .sector-card-mood{font-size:10px;padding:2px 5px}
  .metric-card{padding:10px}
  .metric-price-row{justify-content:space-between;gap:10px}
  #r-prob{font-size:10px!important;gap:3px!important}
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
  #result-tabs{gap:5px}
  #result-tabs .tab-btn{font-size:11px;padding:7px 9px}
  .overall-signal-box{flex-direction:column;align-items:flex-start;padding:12px;gap:10px}
  .overall-signal-box>div{width:100%;text-align:left!important}
  .ovs-counts{gap:8px;font-size:11px}
  .indicator-item{padding:10px}
  .ind-desc{font-size:11px}
  /* 종목 진단 등급행 모바일: rec-badge를 아래 행으로 */
  .diag-grade-row{flex-wrap:wrap;gap:10px;align-items:flex-start}
  #flow-rec-badge{flex:0 0 100%;text-align:center;display:block;margin:0}
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
.header-alert-btn{position:relative;display:inline-flex;align-items:center;gap:5px;background:none;border:1px solid #30363d;border-radius:6px;padding:3px 10px;color:#8b949e;font-size:11px;cursor:pointer;transition:border-color .15s,color .15s;white-space:nowrap}
.header-alert-btn:hover{border-color:#388bfd;color:#388bfd}
.header-alert-count{position:absolute;top:-6px;right:-6px;min-width:16px;height:16px;border-radius:8px;background:#f85149;color:#fff;font-size:10px;font-weight:700;display:none;align-items:center;justify-content:center;padding:0 3px;line-height:1}
.header-alert-count.visible{display:flex}

/* ── 📊 시장 현황 ── */
#market-core{margin-bottom:20px}
.core-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:8px}
.core-header-left{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.core-title{font-size:14px;font-weight:700;color:#e6edf3}
.mood-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:16px;font-size:11px;font-weight:600}
.mood-positive{background:#0d2d1a;color:#3fb950;border:1px solid #1a4730}
.mood-neutral{background:#21262d;color:#8b949e;border:1px solid #30363d}
.mood-negative{background:#2d0d0d;color:#f85149;border:1px solid #4d1515}
/* 클릭 가능한 시장 상태 배지 (버튼) */
.market-mood-btn{cursor:pointer;font-family:inherit;line-height:1.2;transition:filter .15s,transform .1s,box-shadow .15s}
.market-mood-btn:hover{filter:brightness(1.18);box-shadow:0 0 0 1px currentColor inset}
.market-mood-btn:active{transform:scale(.96)}
.market-mood-btn::after{content:'›';margin-left:4px;font-weight:700;opacity:.6}

/* ── 🛡️ 시장 위험 면역 슬라이드 드로어 ── */
#market-drawer-overlay{
  position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:1200;
  opacity:0;visibility:hidden;transition:opacity .25s ease,visibility .25s ease
}
#market-drawer-overlay.open{opacity:1;visibility:visible}
#market-drawer{
  position:fixed;top:0;right:0;height:100%;width:420px;max-width:92vw;z-index:1201;
  background:#0d1117;border-left:1px solid #30363d;box-shadow:-8px 0 30px rgba(0,0,0,.5);
  transform:translateX(100%);transition:transform .28s cubic-bezier(.4,0,.2,1);
  display:flex;flex-direction:column
}
#market-drawer.open{transform:translateX(0)}
.drawer-header{
  display:flex;align-items:center;justify-content:space-between;gap:10px;
  padding:16px 18px;border-bottom:1px solid #21262d;flex-shrink:0
}
.drawer-title{font-size:16px;font-weight:700;color:#e6edf3;display:flex;align-items:center;gap:8px}
.drawer-sub{font-size:11px;color:#8b949e;margin-top:2px}
.drawer-close{
  background:none;border:1px solid #30363d;border-radius:8px;width:32px;height:32px;
  color:#8b949e;font-size:18px;cursor:pointer;flex-shrink:0;line-height:1;
  transition:border-color .15s,color .15s
}
.drawer-close:hover{border-color:#f85149;color:#f85149}
.drawer-body{padding:16px 18px;overflow-y:auto;flex:1;-webkit-overflow-scrolling:touch}
@media(max-width:520px){
  #market-drawer{width:100%;max-width:100vw}
  #market-drawer.open{transform:translateX(0)}
}
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
.rec-badge-lg{display:inline-flex;align-items:center;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700;white-space:nowrap;line-height:1.2}
.rec-strong-buy{background:#0d2d1a;color:#3fb950;border:1px solid #1a4730}
.rec-buy{background:#0d2020;color:#238636;border:1px solid #155724}
.rec-hold{background:#2d2200;color:#d29922;border:1px solid #4a3800}
.rec-sell{background:#2d1515;color:#f85149;border:1px solid #4d1515}
.rec-strong-sell{background:#2d0d0d;color:#f85149;border:2px solid #f85149}
.flow-pos-bar-bg{background:#21262d;border-radius:6px;height:8px;overflow:hidden;margin-top:6px}
.flow-pos-bar-fill{height:8px;border-radius:6px;background:#1f6feb;transition:width .6s ease}

/* ── 🏭 업종별 흐름 (메인 페이지) ── */
#sector-flow{margin-top:16px}
.sector-flow-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.sector-flow-title{font-size:14px;font-weight:700;color:#e6edf3}

/* 7열 그리드 — 데스크탑 기준; 반응형은 하단 @media 참고 */
.sector-cards{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px}

/* 카드 기본 스타일 */
.sector-card{
  background:#161b22;border:1px solid #30363d;border-radius:12px;padding:12px;
  cursor:pointer;user-select:none;overflow:hidden;min-width:0;
  transition:border-color .15s,background .15s
}
.sector-card:hover{border-color:#388bfd;background:#1a2233}
.sector-card.expanded{border-color:#388bfd;background:#161e2e}

/* 카드 내부 요소 */
.sector-card-head{display:flex;align-items:flex-start;gap:5px;margin-bottom:6px;min-width:0}
.sector-card-emoji{font-size:15px;flex-shrink:0;line-height:1.3}
/* flex 자식이 넘치지 않도록 min-width:0 필수 */
.sector-card-head>div{min-width:0;overflow:hidden}
.sector-card-name{font-size:12px;font-weight:600;color:#e6edf3;line-height:1.3;
  word-break:break-word;overflow-wrap:break-word}
.sector-card-cnt{font-size:10px;color:#484f58;margin-top:1px;transition:color .15s}
.sector-card:hover .sector-card-cnt,.sector-card.expanded .sector-card-cnt{color:#388bfd}
.sector-card-pct{font-size:13px;font-weight:700;margin-bottom:5px}
.sector-card-mood{font-size:11px;padding:2px 6px;border-radius:8px;display:inline-block;font-weight:500;max-width:100%}
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

/* ── 업종별 흐름 반응형 (base repeat(7) 이후에 선언해야 override 적용됨) ── */
@media(max-width:1100px){
  .sector-cards{grid-template-columns:repeat(4,minmax(0,1fr))}
}
@media(max-width:900px){
  .sector-cards{grid-template-columns:repeat(3,minmax(0,1fr))}
}
@media(max-width:768px){
  .sector-cards{grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}
  .sector-card{padding:10px 8px}
  .sector-card-name{font-size:11px}
  .sector-card-pct{font-size:12px}
}
@media(max-width:480px){
  .sector-cards{grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}
  .sector-card{padding:9px 8px;border-radius:10px}
  .sector-card-emoji{font-size:14px}
  .sector-card-name{font-size:11px}
  .sector-card-pct{font-size:12px}
  .sector-card-mood{font-size:10px;padding:2px 5px}
}
@media(max-width:360px){
  .sector-cards{grid-template-columns:repeat(2,minmax(0,1fr));gap:5px}
  .sector-card{padding:8px 6px;border-radius:8px}
  .sector-card-emoji{font-size:13px}
  .sector-card-name{font-size:10px}
  .sector-card-pct{font-size:11px}
  .sector-card-mood{font-size:9px;padding:1px 4px}
}

/* ── 단계별 리포트 내 캔들 패턴 카드 ── */
.step-patterns{display:flex;flex-direction:column;gap:6px;margin-top:10px;padding-top:10px;border-top:1px solid #21262d}

/* ── 종목 진단 (AI진단 탭 신규) ── */
.diag-grade-row{display:flex;align-items:center;gap:16px;padding:14px 16px;background:#0d1117;border-radius:12px;margin-bottom:18px}
.diag-grade-badge{font-size:26px;font-weight:900;width:58px;height:58px;display:flex;align-items:center;justify-content:center;border:3px solid;border-radius:50%;flex-shrink:0;letter-spacing:0;line-height:1;white-space:nowrap}
.grade-hyphen{display:inline-block;font-size:0.78em;transform:translateY(-1px);margin-left:0.5px;font-weight:900}
.diag-grade-info{display:flex;flex-direction:column;gap:3px}
.diag-grade-title{font-size:15px;font-weight:700;word-break:keep-all;overflow-wrap:anywhere}
.diag-grade-sub{font-size:11px;color:#8b949e;word-break:keep-all;overflow-wrap:anywhere}
.diag-dims{display:flex;flex-direction:column;gap:10px}
.diag-dim{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 14px}
.diag-dim-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.diag-dim-label{font-size:13px;color:#e6edf3;font-weight:600}
.diag-dim-score{font-size:12px;font-weight:700}
.diag-bar-bg{height:6px;background:#21262d;border-radius:3px;overflow:hidden;margin-bottom:6px}
.diag-bar-fill{height:100%;border-radius:3px;transition:width .7s cubic-bezier(.4,0,.2,1)}
.diag-dim-desc{font-size:11px;color:#8b949e;line-height:1.5}

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
/* ── US 추천 섹션 ──────────────────────────────────────────────── */
.us-reco-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.us-reco-title{font-size:14px;font-weight:600;color:#e6edf3;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.us-reco-cards{display:flex;flex-direction:column;gap:10px}
.us-reco-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 14px}
.us-reco-card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.us-reco-ticker{font-size:16px;font-weight:700;color:#e6edf3;font-family:monospace;letter-spacing:.5px}
.us-reco-badge{font-size:11px;padding:2px 8px;border-radius:8px;font-weight:600}
.us-reco-badge-high{background:rgba(63,185,80,.15);color:#3fb950;border:1px solid rgba(63,185,80,.3)}
.us-reco-badge-med{background:rgba(210,153,34,.15);color:#d2a522;border:1px solid rgba(210,153,34,.3)}
.us-reco-score{font-size:11px;color:#8b949e}
.us-reco-prices{display:flex;gap:14px;margin-bottom:8px;flex-wrap:wrap}
.us-reco-pi{display:flex;flex-direction:column;gap:2px}
.us-reco-pi-label{font-size:10px;color:#484f58;text-transform:uppercase;letter-spacing:.3px}
.us-reco-pi-val{font-size:14px;font-weight:600;color:#e6edf3}
.us-reco-pi-val.grn{color:#3fb950}
.us-reco-pi-val.red{color:#f85149}
.us-reco-reasons{display:flex;flex-direction:column;gap:3px}
.us-reco-reason{font-size:11px;color:#8b949e;display:flex;align-items:flex-start;gap:5px;line-height:1.4}
.us-reco-reason::before{content:"▸";color:#388bfd;flex-shrink:0;margin-top:1px}
.us-reco-warning{font-size:11px;color:#d2a522;margin-top:5px;padding:5px 8px;background:rgba(210,153,34,.08);border-radius:6px;border-left:2px solid rgba(210,153,34,.4)}
.us-reco-holding{font-size:11px;color:#484f58;margin-top:6px;display:flex;align-items:center;gap:4px}
.us-reco-empty{text-align:center;padding:24px 0;color:#484f58;font-size:12px;line-height:1.8}
.us-surge-pm{font-size:17px;font-weight:700;color:#3fb950}
.us-surge-inds{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap}
.us-surge-ind{font-size:10px;padding:2px 6px;background:#21262d;border:1px solid #30363d;border-radius:6px;color:#8b949e}
/* ── KR 장기 투자 전용 ── */
.kr-lt-card{border-top:2px solid #388bfd}
.kr-lt-status-badge{font-size:11px;padding:2px 8px;border-radius:12px;border:1px solid;font-weight:600;letter-spacing:.02em}
.kr-lt-breakdown{margin:10px 0;display:flex;flex-direction:column;gap:5px}
.kr-lt-bd-row{display:flex;align-items:center;gap:6px;font-size:11px}
.kr-lt-bd-label{color:#8b949e;min-width:46px;text-align:right}
.kr-lt-bd-bar-wrap{flex:1;height:6px;background:#21262d;border-radius:3px;overflow:hidden}
.kr-lt-bd-bar{height:100%;border-radius:3px;transition:width .4s}
.kr-lt-bd-val{color:#8b949e;min-width:36px;text-align:right;font-variant-numeric:tabular-nums}
.kr-lt-risks{display:flex;flex-direction:column;gap:3px;margin-bottom:6px}
.kr-lt-risk{font-size:11px;color:#d2a522;padding:4px 8px;background:rgba(210,162,34,.08);border-radius:6px;border-left:2px solid rgba(210,162,34,.4);line-height:1.4}
.kr-lt-fund{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}
.kr-lt-fund-tag{font-size:10px;padding:2px 7px;background:#21262d;border:1px solid #30363d;border-radius:10px;color:#8b949e;font-variant-numeric:tabular-nums}
.kr-lt-theme-badge{font-size:10px;padding:1px 7px;border-radius:10px;background:rgba(56,139,253,.15);color:#58a6ff;border:1px solid rgba(56,139,253,.3);font-weight:600}
.us-reco-reason::before{display:none}
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
    <div class="sb-header-top">
      <h1>📈 StockOracle</h1>
      <button class="sb-home-btn" onclick="showPage('analysis');setState('empty')" title="홈으로 이동" aria-label="홈으로 이동">🏠</button>
    </div>
    <p>AI 기반 기술적 분석 · 투자자 수급</p>
  </div>

  <div class="sb-section">
    <span class="sb-label">메뉴</span>
    <div style="display:flex;flex-direction:column;gap:4px">
      <button class="mkt-btn active" style="text-align:left;padding:10px 12px" id="nav-analysis" onclick="showPage('analysis')">🔍 종목 상세 분석</button>
      <button class="mkt-btn" style="text-align:left;padding:10px 12px" id="nav-scan" onclick="showPage('scan')">🔬 7단계 스캔 엔진</button>
      <!-- ⚡ 개장 급등 추천 아코디언 -->
      <button class="mkt-btn" style="text-align:left;padding:10px 12px" id="nav-recommendations" onclick="toggleRecoMenu()">
        <span class="nav-reco-parent">
          <span>⚡ 개장 급등 추천</span>
          <span class="nav-reco-arrow" id="nav-reco-arrow">▶</span>
        </span>
      </button>
      <div id="nav-reco-submenu" style="display:none;flex-direction:column;gap:2px;padding-left:10px">
        <button class="mkt-btn nav-subbtn" id="nav-kr-longterm" onclick="showPage('kr-longterm')">🇰🇷 국내 장기 투자 추천</button>
        <button class="mkt-btn nav-subbtn" id="nav-us-longterm" onclick="showPage('us-longterm')">🇺🇸 미국 장기 투자 추천</button>
        <button class="mkt-btn nav-subbtn" id="nav-us-surge"    onclick="showPage('us-surge')">🇺🇸 미국 개장 급등 추천</button>
      </div>
    </div>
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
              <button type="button" id="core-mood-badge-kr" class="mood-badge mood-neutral market-mood-btn" onclick="openMarketDrawer('KR')" aria-label="한국 시장 위험 면역 상세 보기" title="클릭 — 한국 시장 위험 면역 시스템">🇰🇷 한국 —</button>
              <button type="button" id="core-mood-badge-us" class="mood-badge mood-neutral market-mood-btn" onclick="openMarketDrawer('US')" aria-label="미국 시장 위험 면역 상세 보기" title="클릭 — 미국 시장 위험 면역 시스템">🇺🇸 미국 —</button>
              <span id="core-vix-badge" style="display:none;font-size:10px;padding:2px 7px;border-radius:10px;background:#2d0d0d;color:#f85149;border:1px solid #4d1515"></span>
            </div>
            <button class="header-alert-btn" onclick="openAlertsSheet()" aria-label="알림 관리">
              🔔 알림 관리
              <span class="header-alert-count" id="header-alert-count"></span>
            </button>
          </div>
          <div class="core-indices" id="core-indices"></div>
        </div>
        <div id="core-error" style="display:none;text-align:center;padding:20px;color:#484f58;font-size:13px">
          시장 데이터를 불러오지 못했습니다
          <button onclick="loadMarketCore()" class="home-section-refresh" style="margin-left:8px">재시도</button>
        </div>
      </div>

      <!-- 2. 🏭 업종별 흐름 -->
      <div id="sector-flow" class="home-section">
        <div id="sector-flow-loading" style="text-align:center;padding:14px;color:#484f58;font-size:12px">
          <div class="spinner" style="margin:0 auto 8px;width:22px;height:22px;border-width:3px"></div>
          업종별 흐름 로딩 중...
        </div>
        <div id="sector-flow-content" style="display:none">
          <div class="sector-flow-header">
            <span class="sector-flow-title">🏭 업종별 흐름</span>
            <button onclick="loadSectorFlow()" class="home-section-refresh" title="새로고침">🔄 새로고침</button>
          </div>
          <div class="sector-cards" id="sector-cards"></div>
        </div>
        <div id="sector-flow-error" style="display:none;text-align:center;padding:12px;color:#484f58;font-size:12px">
          업종별 흐름 데이터를 불러오지 못했습니다
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
        <div class="metric-card metric-price-card"><div class="m-label">현재가 <span id="r-session-badge" style="display:none;font-size:10px;font-weight:600;padding:1px 6px;border-radius:4px;background:#1f6feb33;color:#58a6ff;margin-left:4px;vertical-align:middle"></span></div><div class="metric-price-row"><div style="display:flex;flex-direction:column;align-items:flex-start;flex-shrink:0"><div class="m-value" id="r-price" style="white-space:nowrap"></div><div class="m-sub" id="r-pct" style="margin-top:0"></div></div><div id="r-prob" style="display:none;flex-direction:column;gap:4px;align-items:flex-start;font-size:11px;font-weight:600;padding-top:4px"></div></div></div>
        <div class="metric-card metric-volume-card"><div class="m-label">거래량</div><div class="m-value" id="r-vol" style="font-size:18px"></div></div>
        <div class="metric-card metric-atr-card"><div class="m-label">ATR (변동성)</div><div class="m-value" id="r-atr" style="font-size:18px"></div><div id="r-atr-pct" style="display:none;font-size:11px;color:#8b949e;margin-top:4px"></div></div>
        <div class="metric-card metric-toss-card" id="r-toss-card">
          <div class="m-label">토스증권 AI 요약</div>
          <div class="toss-ai-summary" id="r-toss-summary" style="color:#484f58;font-size:11px">-</div>
          <div class="toss-ai-time" id="r-toss-time"></div>
        </div>
      </div>
      <div id="signal-confidence-card" style="display:none" class="card"></div>
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
        <button class="tab-btn" onclick="switchTab('report')" style="display:none">📝 단계별 리포트</button>
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
          <div class="card-title">📐 피봇 포인트 (지지·저항 구간)</div>
          <div id="pivot-points-section"></div>
        </div>
      </div>

      <!-- AI 탭 -->
      <div id="tab-ai" style="display:none">
        <div class="ai-diagnosis-layout">
          <!-- 종목 진단 (흐름 단계 → 5차원 진단 → 눌림목 분석 통합 출력) -->
          <div class="card ai-report-card">
            <div class="card-title">🔬 종목 진단</div>
            <div id="ai-diagnosis-chart"></div>
          </div>
          <!-- 섹터 / 업종 정보 -->
          <div class="card ai-flow-card" id="flow-sector-card" style="display:none">
            <div class="card-title">🏭 섹터 / 업종 정보</div>
            <div id="flow-sector-content"></div>
          </div>
        </div>
      </div>

      <!-- 단계별 분석 리포트 탭 -->
      <div id="tab-report" style="display:none">
        <div class="card">
          <div class="card-title">📝 단계별 분석 리포트</div>
          <div id="steps-list" style="display:flex;flex-direction:column;gap:10px"></div>
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
        <!-- 매수 전략 카드: 현재가 분석 → 가격 구간 → 분할 매수 흐름 통합 -->
        <div class="card">
          <div class="card-title">🎯 현재가 기준 매수 전략</div>
          <div id="buy-price-section"></div>
          <div id="pullback-forecast-section"></div>
        </div>
        <!-- 리스크 관리 카드: 시나리오 + ATR 기반 정밀 가격 통합 -->
        <div class="card">
          <div class="card-title">🛡️ 리스크 관리 (ATR 기반)</div>
          <div class="risk-grid" id="risk-grid"></div>
          <div id="pullback-atr-section"></div>
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

  <!-- ── 🇰🇷 국내 장기 투자 추천 페이지 ── -->
  <div id="page-kr-longterm" style="display:none">
    <div class="screener-header" style="margin-bottom:16px">
      <div>
        <h2 style="font-size:20px;font-weight:700;margin-bottom:3px">🇰🇷 국내 장기 투자 추천</h2>
        <p style="font-size:12px;color:#8b949e">기술·수급·펀더멘털 통합 스코어링 Top 10</p>
      </div>
    </div>
    <div class="home-section">
      <div id="kr-lt-loading" style="text-align:center;padding:14px;color:#484f58;font-size:12px">
        <div class="spinner" style="margin:0 auto 8px;width:22px;height:22px;border-width:3px"></div>
        국내 장기 추천 분석 중...
      </div>
      <div id="kr-lt-content" style="display:none">
        <div class="us-reco-header">
          <span class="us-reco-title">🇰🇷 국내 장기 투자 추천 <span style="font-size:11px;color:#484f58;font-weight:400">100점 종합 스코어링 Top 10</span></span>
          <button onclick="loadKrLongterm(true)" class="home-section-refresh" title="새로고침">🔄 새로고침</button>
        </div>
        <div class="us-reco-cards" id="kr-lt-cards"></div>
      </div>
      <div id="kr-lt-error" style="display:none;text-align:center;padding:12px;color:#484f58;font-size:12px">
        데이터를 불러오지 못했습니다
        <button onclick="loadKrLongterm(true)" class="home-section-refresh" style="margin-left:6px">재시도</button>
      </div>
    </div>
  </div>

  <!-- ── 🇺🇸 미국 장기 투자 추천 페이지 ── -->
  <div id="page-us-longterm" style="display:none">
    <div class="screener-header" style="margin-bottom:16px">
      <div>
        <h2 style="font-size:20px;font-weight:700;margin-bottom:3px">🇺🇸 미국 장기 투자 추천</h2>
        <p style="font-size:12px;color:#8b949e">기술·수급·펀더멘털 통합 스코어링 Top 10</p>
      </div>
    </div>
    <div class="home-section">
      <div id="us-lt-loading" style="text-align:center;padding:14px;color:#484f58;font-size:12px">
        <div class="spinner" style="margin:0 auto 8px;width:22px;height:22px;border-width:3px"></div>
        미국 장기 추천 분석 중...
      </div>
      <div id="us-lt-content" style="display:none">
        <div class="us-reco-header">
          <span class="us-reco-title">🇺🇸 미국 장기 투자 추천 <span style="font-size:11px;color:#484f58;font-weight:400">기술·수급 통합 스코어링 Top 10</span></span>
          <button onclick="loadUsLongterm(true)" class="home-section-refresh" title="새로고침">🔄 새로고침</button>
        </div>
        <div class="us-reco-cards" id="us-lt-cards"></div>
      </div>
      <div id="us-lt-error" style="display:none;text-align:center;padding:12px;color:#484f58;font-size:12px">
        데이터를 불러오지 못했습니다
        <button onclick="loadUsLongterm(true)" class="home-section-refresh" style="margin-left:6px">재시도</button>
      </div>
    </div>
  </div>

  <!-- ── 🔬 7단계 스캔 엔진 페이지 ── -->
  <div id="page-scan" style="display:none">
    <div class="screener-header" style="margin-bottom:16px;flex-wrap:wrap;gap:18px">
      <div style="margin-right:auto">
        <h2 style="font-size:20px;font-weight:700;margin-bottom:3px">🔬 7단계 스캔 엔진</h2>
        <p style="font-size:12px;color:#8b949e">BQS·FWS·NCS 복합 점수 기반 후보 종목 발굴</p>
      </div>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:nowrap">
        <select id="scan-market" style="background:#21262d;border:1px solid #30363d;border-radius:6px;padding:6px 10px;color:#e6edf3;font-size:12px;flex:1 1 0;min-width:0">
          <option value="KRX">🇰🇷 KRX (한국)</option>
          <option value="US">🇺🇸 US (미국)</option>
        </select>
        <select id="scan-mode" style="background:#21262d;border:1px solid #30363d;border-radius:6px;padding:6px 10px;color:#e6edf3;font-size:12px;flex:1 1 0;min-width:0">
          <option value="FULL">전체 스캔</option>
          <option value="CORE_LITE">핵심 종목 (빠름)</option>
        </select>
        <button onclick="runScan()" id="scan-run-btn"
          style="background:#1f6feb;border:none;border-radius:8px;padding:8px 16px;color:#fff;font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap;flex-shrink:0">
          🔍 스캔 실행
        </button>
        <button onclick="runScan(true)" style="background:#21262d;border:1px solid #30363d;border-radius:8px;padding:8px 12px;color:#8b949e;font-size:13px;cursor:pointer;white-space:nowrap;flex-shrink:0">🔄</button>
      </div>
    </div>
    <!-- 스캔 전 안내 -->
    <div id="scan-intro" class="home-section" style="text-align:center;padding:32px 20px;color:#484f58">
      <div style="font-size:36px;margin-bottom:12px">🔬</div>
      <div style="font-size:14px;font-weight:600;color:#8b949e;margin-bottom:8px">7단계 스캔 엔진 준비 완료</div>
      <div style="font-size:12px;line-height:1.7">종목 유니버스 → 기술필터 → 상태분류 → BQS 랭킹 → 리스크 게이트 → 추격 방지 → 포지션 사이징</div>
      <div style="font-size:11px;color:#484f58;margin-top:8px">⚠️ 실시간 데이터 수집으로 10~30초 소요될 수 있습니다</div>
    </div>
    <!-- 로딩 -->
    <div id="scan-loading" style="display:none;text-align:center;padding:40px;color:#8b949e">
      <div class="spinner" style="margin:0 auto 12px"></div>
      <div id="scan-loading-msg">종목 데이터 수집 및 스캔 중...</div>
      <div style="font-size:11px;color:#484f58;margin-top:8px">복합 점수 계산 중...</div>
    </div>
    <!-- 오류 -->
    <div id="scan-error" style="display:none;text-align:center;padding:32px;color:#f85149;font-size:13px"></div>
    <!-- 결과 -->
    <div id="scan-result" style="display:none">
      <!-- 요약 카드 -->
      <div id="scan-summary-cards" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:16px"></div>
      <!-- 레짐 배지 -->
      <div id="scan-regime-bar" style="margin-bottom:14px;padding:10px 14px;background:#161b22;border:1px solid #30363d;border-radius:10px;display:flex;gap:16px;flex-wrap:wrap;align-items:center;font-size:12px"></div>
      <!-- 후보 테이블 -->
      <div class="card" style="padding:0;overflow-x:auto">
        <table class="screener-table" id="scan-table">
          <thead><tr>
            <th>#</th>
            <th>종목</th>
            <th style="text-align:center">상태</th>
            <th style="text-align:right">현재가</th>
            <th style="text-align:right">등락률</th>
            <th>카테고리</th>
            <th style="text-align:center">신호</th>
            <th style="text-align:right">진입 트리거</th>
            <th style="text-align:right">손절가</th>
            <th style="text-align:center">브레이크아웃 품질</th>
            <th style="text-align:center">치명적 약점</th>
            <th style="text-align:center">순복합 점수</th>
            <th style="text-align:center">퀀트 모멘텀 점수</th>
          </tr></thead>
          <tbody id="scan-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ── 🛡️ 시장 위험 면역 페이지 ── -->
  <div id="page-immune" style="display:none">
    <div class="screener-header" style="margin-bottom:16px">
      <div>
        <h2 style="font-size:20px;font-weight:700;margin-bottom:3px">🛡️ 시장 위험 면역 시스템</h2>
        <p style="font-size:12px;color:#8b949e">VIX · MA200 이격도 · ATR 기반 현재 시장 위험도 평가</p>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <select id="immune-index" style="background:#21262d;border:1px solid #30363d;border-radius:6px;padding:6px 10px;color:#e6edf3;font-size:12px">
          <option value="SPY">S&amp;P 500 (SPY)</option>
          <option value="^KS200">KOSPI 200</option>
          <option value="QQQ">NASDAQ 100 (QQQ)</option>
        </select>
        <button onclick="loadImmuneFull(true)" style="background:#21262d;border:1px solid #30363d;border-radius:8px;padding:8px 14px;color:#8b949e;font-size:13px;cursor:pointer">🔄 새로고침</button>
      </div>
    </div>
    <div id="immune-loading" style="text-align:center;padding:40px;color:#8b949e">
      <div class="spinner" style="margin:0 auto 12px"></div>
      시장 위험 데이터 수집 중...
    </div>
    <div id="immune-error" style="display:none;text-align:center;padding:32px;color:#f85149;font-size:13px"></div>
    <div id="immune-content" style="display:none">
      <!-- 메인 면역 레벨 카드 -->
      <div id="immune-level-card" style="margin-bottom:16px;border-radius:14px;padding:24px;border:2px solid;text-align:center"></div>
      <!-- 지표 그리드 -->
      <div id="immune-metrics" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px"></div>
      <!-- 경고 목록 -->
      <div id="immune-warnings" style="margin-bottom:16px"></div>
      <!-- Kill 스위치 권고 -->
      <div id="immune-killswitch" style="margin-bottom:16px"></div>
      <!-- 과거 위기 유사도 -->
      <div id="immune-crisis" style="margin-bottom:16px"></div>
    </div>
  </div>

  <!-- ── 🇺🇸 미국 개장 급등 추천 페이지 ── -->
  <div id="page-us-surge" style="display:none">
    <div class="screener-header" style="margin-bottom:16px">
      <div>
        <h2 style="font-size:20px;font-weight:700;margin-bottom:3px">🇺🇸 미국 개장 급등 추천</h2>
        <p style="font-size:12px;color:#8b949e">PM RVOL · ATR 기반 당일 모멘텀 Top 10</p>
      </div>
    </div>
    <div class="home-section">
      <div id="us-surge-loading" style="text-align:center;padding:14px;color:#484f58;font-size:12px">
        <div class="spinner" style="margin:0 auto 8px;width:22px;height:22px;border-width:3px"></div>
        미국 개장 급등 종목 스캔 중...
      </div>
      <div id="us-surge-content" style="display:none">
        <div class="us-reco-header">
          <span class="us-reco-title">🇺🇸 미국 개장 급등 추천
            <span style="font-size:11px;color:#484f58;font-weight:400">PM RVOL·ATR 기반 Top 10</span>
            <span id="us-surge-session-label" style="font-size:10px;color:#8b949e;font-weight:400;margin-left:6px"></span>
          </span>
          <button onclick="loadUsSurge(true)" class="home-section-refresh" title="새로고침">🔄 새로고침</button>
        </div>
        <div class="us-reco-cards" id="us-surge-cards"></div>
      </div>
      <div id="us-surge-error" style="display:none;text-align:center;padding:12px;color:#484f58;font-size:12px">
        데이터를 불러오지 못했습니다
        <button onclick="loadUsSurge(true)" class="home-section-refresh" style="margin-left:6px">재시도</button>
      </div>
    </div>
  </div>
</div>

<!-- ── 🛡️ 시장 위험 면역 슬라이드 드로어 (한국/미국 공용) ── -->
<div id="market-drawer-overlay" onclick="closeMarketDrawer()"></div>
<aside id="market-drawer" role="dialog" aria-modal="true" aria-labelledby="market-drawer-title">
  <div class="drawer-header">
    <div>
      <div class="drawer-title" id="market-drawer-title">🛡️ 시장 위험 면역</div>
      <div class="drawer-sub" id="market-drawer-sub"></div>
    </div>
    <button type="button" class="drawer-close" onclick="closeMarketDrawer()" aria-label="닫기">✕</button>
  </div>
  <div class="drawer-body" id="market-drawer-body">
    <div style="text-align:center;padding:40px;color:#8b949e">
      <div class="spinner" style="margin:0 auto 12px"></div>
      시장 위험 데이터 수집 중...
    </div>
  </div>
</aside>

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
var _recoMenuOpen = false;
var _krLongtermLoaded = false;
var _usLongtermLoaded = false;
var _usSurgeLoaded   = false;

var _recoSubPages = ['kr-longterm', 'us-longterm', 'us-surge'];
var _allPages = ['analysis', 'screener', 'kr-longterm', 'us-longterm', 'us-surge', 'scan', 'immune'];
var _scanLoaded  = false;
var _immuneLoaded = false;

function toggleRecoMenu() {
  _recoMenuOpen = !_recoMenuOpen;
  var submenu = document.getElementById('nav-reco-submenu');
  var arrow   = document.getElementById('nav-reco-arrow');
  submenu.style.display = _recoMenuOpen ? 'flex' : 'none';
  if (arrow) arrow.classList.toggle('open', _recoMenuOpen);
  // 서브메뉴가 열리면 부모 버튼도 active
  document.getElementById('nav-recommendations').classList.toggle('active', _recoMenuOpen);
}

function showPage(page) {
  // 모든 페이지 숨기기
  _allPages.forEach(function(p) {
    var el = document.getElementById('page-' + p);
    if (el) el.style.display = (p === page) ? 'block' : 'none';
  });
  // 분석 컨트롤 패널 (종목 입력 등) analysis 페이지에서만 노출
  document.getElementById('analysis-controls').style.display = page === 'analysis' ? 'block' : 'none';

  // 상단 nav 버튼 active 상태
  document.getElementById('nav-analysis').classList.toggle('active', page === 'analysis');
  document.getElementById('nav-scan').classList.toggle('active', page === 'scan');

  var isRecoPage = _recoSubPages.indexOf(page) !== -1;
  // 서브페이지를 선택하면 아코디언 부모도 active, 서브메뉴 열기
  if (isRecoPage && !_recoMenuOpen) toggleRecoMenu();
  document.getElementById('nav-recommendations').classList.toggle('active', isRecoPage || _recoMenuOpen);
  ['kr-longterm', 'us-longterm', 'us-surge'].forEach(function(p) {
    var btn = document.getElementById('nav-' + p);
    if (btn) btn.classList.toggle('active', p === page);
  });

  // 데이터 최초 로드 (각 서브페이지 첫 방문 시)
  if (page === 'screener'    && screenerData.length === 0) loadScreener();
  if (page === 'kr-longterm' && !_krLongtermLoaded) { _krLongtermLoaded = true; loadKrLongterm(); }
  if (page === 'us-longterm' && !_usLongtermLoaded) { _usLongtermLoaded = true; loadUsLongterm(); }
  if (page === 'us-surge'    && !_usSurgeLoaded)    { _usSurgeLoaded    = true; loadUsSurge(); }

  // HybridTurtle 통합 페이지 첫 방문 시 자동 로드
  if (page === 'scan')   { var scanEl = document.getElementById('nav-scan');   if(scanEl) scanEl.classList.add('active'); loadImmuneBanner(); }
  if (page === 'immune') { var imEl   = document.getElementById('nav-immune'); if(imEl)   imEl.classList.add('active');   loadImmuneFull(); }
  ['nav-scan','nav-immune'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.classList.toggle('active', id === 'nav-' + page);
  });

  closeSidebar();   // 모바일: 페이지 전환 시 사이드바 닫기
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

// ── 공통 상세분석 진입점 ──────────────────────────────────────────────
// 직접 검색(quickSearch)과 🔬 스캔 결과 클릭이 동일한 분석 로직을 타도록 통합.
// market을 전달받아 currentMarket을 동기화하므로, KRX 종목이 US로 처리되는
// 문제를 방지한다. (티커 해석 자체는 백엔드 resolve_ticker가 최종 판정)
function openStockDetail(ticker, market) {
  if (market) currentMarket = market;
  const inp = document.getElementById('ticker-input');
  if (inp) inp.value = ticker;
  showPage('analysis');   // 내부에서 closeSidebar() 호출됨
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

// ── 미국 주식 실시간 가격 폴링 ──────────────────────────────────────────────
let _pricePoller    = null;
let _pollTicker     = null;
// ── 목표가 섹션 동기화용 상태 (Single Source of Truth) ───────────────────────
let _lastTp         = null;   // 마지막 렌더된 target_price 객체 {min_price, max_price, ...}
let _lastIsKrx      = false;  // 마지막 분석 시장 구분

function _stopPricePolling() {
  if (_pricePoller) { clearInterval(_pricePoller); _pricePoller = null; }
}

function _startPricePolling(symbol) {
  _stopPricePolling();
  _pollTicker = symbol;
  _pricePoller = setInterval(async () => {
    if (!_pollTicker) return;
    try {
      const r = await fetch(`/api/price?ticker=${encodeURIComponent(_pollTicker)}&market=US`);
      if (!r.ok) return;
      const d = await r.json();
      if (d.error || !d.price) return;
      _applyPriceUpdate(d);
    } catch(e) {}
  }, 5000);
}

function _applyPriceUpdate(d) {
  const up  = d.pct_change >= 0;
  const clr = up ? '#3fb950' : '#f85149';
  const priceEl  = document.getElementById('r-price');
  const pctEl    = document.getElementById('r-pct');
  const badgeEl  = document.getElementById('r-session-badge');
  if (priceEl) priceEl.textContent = fmtPrice(d.price, false);
  if (pctEl)   pctEl.innerHTML = `<span style="color:${clr}">${up?'▲':'▼'} ${Math.abs(d.pct_change).toFixed(2)}%</span>`;
  if (badgeEl) {
    const hiddenSessions = new Set(['정규장', '장마감', '']);
    const sn = (d.session_name || '').trim();
    if (sn && !hiddenSessions.has(sn)) {
      const sessionColors = {
        '프리마켓':   { bg: '#1f6feb33', fg: '#58a6ff' },
        '오버나이트': { bg: '#6e40c933', fg: '#bc8cff' },
        '애프터마켓': { bg: '#388bfd22', fg: '#79c0ff' },
        '시간외':     { bg: '#30363d',   fg: '#8b949e' },
      };
      const col = sessionColors[sn] || { bg: '#1f6feb33', fg: '#58a6ff' };
      badgeEl.textContent = sn;
      badgeEl.style.background = col.bg;
      badgeEl.style.color = col.fg;
      badgeEl.style.display = 'inline';
    } else {
      badgeEl.style.display = 'none';
    }
  }
  // ── 목표가 예측 섹션 현재가 실시간 동기화 ────────────────────────────────
  _syncTargetPriceSection(d.price);
}

// ── 목표가 예측 섹션 현재가 동기화 (폴링 업데이트 시 호출) ──────────────────
// Single Source of Truth: _lastTp에 저장된 목표가 기준으로 재계산
function _syncTargetPriceSection(newPrice) {
  if (!_lastTp || !newPrice || isNaN(newPrice)) return;
  const tpEl = document.getElementById('target-price-section');
  if (!tpEl) return;

  // 최신 현재가 기준으로 수익률 재계산
  const minReturn = ((_lastTp.min_price - newPrice) / newPrice * 100).toFixed(1);
  const maxReturn = ((_lastTp.max_price - newPrice) / newPrice * 100).toFixed(1);
  const fmtUs = v => fmtPrice(v, false);

  // data 속성으로 타겟 요소만 핀포인트 업데이트 (전체 re-render 없음)
  const curEl    = tpEl.querySelector('[data-tp-cur]');
  const returnEl = tpEl.querySelector('[data-tp-return]');
  if (curEl)    curEl.textContent    = fmtUs(newPrice);
  if (returnEl) returnEl.textContent = `+${minReturn}% ~ +${maxReturn}%`;
}

async function analyze() {
  _stopPricePolling();   // 새 검색 시 이전 폴링 중단
  _lastTp    = null;     // 새 종목 분석 시 동기화 상태 초기화
  _lastIsKrx = false;
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
    // 입력창("종목명/코드")에 해석된 종목명 반영 — 스캔 클릭/코드 검색 시에도
    // 티커(041510.KQ)가 아닌 한글/영문 종목명(에스엠, Apple)이 표시되도록 통일.
    // 조회는 이미 티커로 끝났으므로, 표시값만 종목명으로 교체한다.
    const _tickerInputEl = document.getElementById('ticker-input');
    if (_tickerInputEl && d.company && d.company !== d.symbol) {
      _tickerInputEl.value = d.company;
    }
    renderResult(d);
    renderSignalConfidence(d);
    setState('result');
    // 미국 주식: 5초마다 현재가 자동 갱신
    if (d.market === 'US' && d.symbol) {
      _startPricePolling(d.symbol);
    }
    // KRX 전용: 투자자 수급 자동 비동기 로드
    // 메인 API 응답에서 ok=false인 경우(타임아웃·API 지연 등) 전용 엔드포인트로 자동 재시도
    if (d.market === 'KRX' && (!d.investor_flow || !d.investor_flow.ok)) {
      loadInvestorFlowAsync(d.symbol);
    }
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
// ════════════════════════════════════════════════════════════════════════
// 공통 가격/종목코드 포맷터 (Single Source of Truth)
//   · KRX: 정수(소수점 제거) + '원'
//   · US : '$' + 불필요한 trailing zero 제거 (200.00→200, 200.35→200.35)
//   · 종목코드: KRX는 시장 접미사(.KS/.KQ) 제거, US는 티커 유지
// ════════════════════════════════════════════════════════════════════════
function _fmtKrNum(v) { return Number(v).toLocaleString('ko-KR', {maximumFractionDigits:0}); }
// US: $1 미만(서브달러) 종목은 소수 4자리까지 표시(시장 관행 — 백엔드도 4자리로 산출),
//     $1 이상은 기존대로 2자리. trailing zero는 자동 제거(200.00→200, 0.5→0.5).
function _fmtUsNum(v) {
  const n = Number(v);
  const maxFD = (n !== 0 && Math.abs(n) < 1) ? 4 : 2;
  return n.toLocaleString('en-US', {minimumFractionDigits:0, maximumFractionDigits:maxFD});
}

// 통화기호 포함 가격 (현재가·목표가·매수전략·ATR 리스크 등)
function fmtPrice(v, isKrx) {
  if (v == null || isNaN(v)) return '-';
  return isKrx ? _fmtKrNum(v) + '원' : '$' + _fmtUsNum(v);
}
// 통화기호 없는 숫자 (진입가·손절가·표·배지 등)
function fmtNum(v, isKrx) {
  if (v == null || isNaN(v)) return '-';
  return isKrx ? _fmtKrNum(v) : _fmtUsNum(v);
}
// 종목코드 표시 — KRX: 시장 접미사 제거 / US: 티커 유지
function fmtSymbol(sym, isKrx) {
  if (!sym) return '';
  return isKrx ? String(sym).replace(/\.(KS|KQ|KX)$/i, '') : String(sym);
}
// 하위호환 — 기존 fmt() 호출부는 모두 fmtPrice로 위임
function fmt(v, isKrx) { return fmtPrice(v, isKrx); }

// ── 🧭 신호 신뢰도 종합 카드 (confidence_engine 결과) ──────────────────────
function renderSignalConfidence(d) {
  const el = document.getElementById('signal-confidence-card');
  if (!el) return;
  const sc = d && d.signal_confidence;
  if (!sc || sc.confidence == null) { el.style.display = 'none'; return; }
  el.style.display = '';

  const conf = sc.confidence;
  const ci   = sc.confidence_interval || {};
  const confColor = conf >= 70 ? '#3fb950' : conf >= 55 ? '#58a6ff' : conf >= 45 ? '#d29922' : '#f85149';

  // 거시 체제 배지
  const regimeKoMap = { 'Risk-On':'위험선호', 'Risk-Off':'위험회피', 'Neutral':'중립', 'Transition':'전환' };
  const reg = (sc.macro_regime && sc.macro_regime.regime) || 'Neutral';
  const regColor = reg === 'Risk-On' ? '#3fb950' : reg === 'Risk-Off' ? '#f85149' : reg === 'Transition' ? '#d29922' : '#8b949e';
  const regChip = '<span style="font-size:11px;font-weight:700;color:' + regColor +
    ';border:1px solid ' + regColor + '55;border-radius:4px;padding:2px 8px">거시 ' + (regimeKoMap[reg]||reg) + '</span>';

  // 섹터 상대
  let sectorChip = '';
  const sr = sc.sector_relative;
  if (sr && sr.sector && sr.sector.pct5d != null) {
    const aligned = sr.aligned;
    const sColor = aligned === true ? '#3fb950' : aligned === false ? '#f85149' : '#8b949e';
    sectorChip = '<span style="font-size:11px;color:' + sColor +
      ';border:1px solid ' + sColor + '55;border-radius:4px;padding:2px 8px">섹터 ' +
      (sr.sector.name||sr.sector.etf) + ' ' + (sr.sector.pct5d>=0?'+':'') + sr.sector.pct5d + '%' +
      (aligned === true ? ' ▲정렬' : aligned === false ? ' ⚠불일치' : '') + '</span>';
  }

  // 실적 임박 경고
  let earnChip = '';
  if (sc.earnings_risk && sc.days_to_earnings != null) {
    earnChip = '<span style="font-size:11px;font-weight:700;color:#f97316;border:1px solid #f9731655;border-radius:4px;padding:2px 8px">⚠️ 실적 D-' + sc.days_to_earnings + '</span>';
  }

  // 뉴스 감정 (FinBERT/키워드)
  let sentChip = '';
  let sentInfo = '';
  const se = sc.sentiment;
  if (se && se.sentiment_score != null) {
    const sentColor = se.overall === 'positive' ? '#3fb950' : se.overall === 'negative' ? '#f85149' : '#8b949e';
    const srcLbl = se.sentiment_source === 'finbert' ? 'FinBERT' : se.sentiment_source === 'kr-finbert' ? 'KR-FinBERT' : se.sentiment_source === 'keyword' ? '금융사전' : '없음';
    sentChip = '<span style="font-size:11px;color:' + sentColor +
      ';border:1px solid ' + sentColor + '55;border-radius:4px;padding:2px 8px">뉴스감정 ' +
      se.sentiment_score + ' (' + srcLbl + ')</span>';

    // 📰 뉴스감정 상세 — 출처 구성 · 품질 · 신뢰도 가중 여부 (보강 내용 가시화)
    const _parts = [];
    if (se.source_breakdown) {
      const _lbl = { dart:'DART', disclosure:'공시', naver:'네이버', google_news:'포털뉴스', news:'뉴스', x:'X' };
      const _segs = Object.keys(se.source_breakdown)
        .map(k => (_lbl[k] || k) + ' ' + se.source_breakdown[k]);
      if (_segs.length) _parts.push('출처 ' + _segs.join(' · '));
    }
    if (se.quality != null) _parts.push('품질 ' + se.quality + (se.effective_n != null ? ' (유효표본 ' + se.effective_n + ')' : ''));
    if (se.credibility_weighted) _parts.push('출처 신뢰도 가중 적용');
    if (_parts.length) {
      sentInfo = '<div style="margin-top:8px;font-size:10px;color:#8b949e">📰 뉴스감정 — ' + _parts.join(' · ') + '</div>';
    }
  }

  // 신뢰 구간 막대 (lower ~ upper)
  const lo = ci.lower != null ? ci.lower : conf;
  const hi = ci.upper != null ? ci.upper : conf;
  const spread = ci.spread != null ? ci.spread : 0;
  const spreadWarn = spread >= 20;   // 의미 있는 분산이면 강조

  // 캡/불일치 사유 + 뉴스감정 사유(한글)
  const reasons = [];
  (sc.cap_reasons || []).forEach(r => reasons.push(r));
  if (ci.reason) ci.reason.forEach(r => { if (!reasons.includes(r)) reasons.push(r); });
  if (se && se.reasons) se.reasons.forEach(r => { if (!reasons.includes(r)) reasons.push(r); });
  const reasonHtml = reasons.length
    ? '<div style="margin-top:8px;font-size:11px;color:#8b949e">' +
      reasons.map(r => '<div style="display:flex;gap:5px"><span style="color:#f97316">•</span><span>' + r + '</span></div>').join('') + '</div>'
    : '';

  el.innerHTML =
    '<div class="card-title">🧭 신호 신뢰도 종합' +
      (spreadWarn ? ' <span style="font-size:10px;color:#d29922">· 불확실성 높음(±' + Math.round(spread/2) + ')</span>' : '') +
    '</div>' +
    '<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">' +
      '<div style="flex-shrink:0">' +
        '<div style="font-size:30px;font-weight:800;color:' + confColor + ';line-height:1">' + conf + '<span style="font-size:14px;color:#8b949e">%</span></div>' +
        '<div style="font-size:11px;color:#8b949e;margin-top:2px">신뢰 구간 ' + lo + '~' + hi + ' (폭 ' + spread + ')</div>' +
      '</div>' +
      '<div style="flex:1;min-width:160px">' +
        '<div style="position:relative;height:8px;background:#21262d;border-radius:4px;overflow:hidden">' +
          '<div style="position:absolute;left:' + lo + '%;width:' + Math.max(2,(hi-lo)) + '%;height:100%;background:' + confColor + '55"></div>' +
          '<div style="position:absolute;left:' + Math.min(99,conf) + '%;width:2px;height:100%;background:' + confColor + '"></div>' +
        '</div>' +
        '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:10px">' + regChip + sectorChip + earnChip + sentChip + '</div>' +
      '</div>' +
    '</div>' + sentInfo + reasonHtml;
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
    `${d.company || fmtSymbol(d.symbol, isKrx)} <span class="ticker-badge">${fmtSymbol(d.symbol, isKrx)}</span>`;
  document.getElementById('r-subtitle').textContent =
    `기준일: ${new Date().toLocaleDateString('ko-KR')} | 시장: ${isKrx ? '🇰🇷 KRX (한국)' : '🇺🇸 US (미국)'}`;
  document.getElementById('r-price').textContent = fmt(d.last_close, isKrx);
  document.getElementById('r-pct').innerHTML = `<span style="color:${clr}">${up?'▲':'▼'} ${Math.abs(d.pct_change).toFixed(2)}%</span>`;

  // 상승/하락 가능성 표시
  const probEl = document.getElementById('r-prob');
  if (probEl && d.prob_up != null) {
    probEl.style.display = 'flex';
    probEl.style.flexDirection = 'column';
    probEl.innerHTML =
      `<span style="color:#3fb950;background:#3fb95018;padding:2px 6px;border-radius:3px;white-space:nowrap">▲ 상승 가능성 ${d.prob_up.toFixed(1)}%</span>` +
      `<span style="color:#f85149;background:#f8514918;padding:2px 6px;border-radius:3px;white-space:nowrap">▼ 하락 가능성 ${d.prob_down.toFixed(1)}%</span>`;
  } else if (probEl) {
    probEl.style.display = 'none';
  }

  // 미국 주식 세션 배지: 프리마켓 / 오버나이트 / 애프터마켓 등 표시
  const sessionBadge = document.getElementById('r-session-badge');
  if (sessionBadge) {
    const hiddenSessions = new Set(['정규장', '장마감', '']);
    const sn = (d.session_name || '').trim();
    if (!isKrx && sn && !hiddenSessions.has(sn)) {
      // 세션별 색상 구분
      const sessionColors = {
        '프리마켓':   { bg: '#1f6feb33', fg: '#58a6ff' },
        '오버나이트': { bg: '#6e40c933', fg: '#bc8cff' },
        '애프터마켓': { bg: '#388bfd22', fg: '#79c0ff' },
        '시간외':     { bg: '#30363d',   fg: '#8b949e' },
      };
      const col = sessionColors[sn] || { bg: '#1f6feb33', fg: '#58a6ff' };
      sessionBadge.textContent = sn;
      sessionBadge.style.background = col.bg;
      sessionBadge.style.color = col.fg;
      sessionBadge.style.display = 'inline';
    } else {
      sessionBadge.style.display = 'none';
    }
  }

  document.getElementById('r-vol').textContent = d.volume.toLocaleString();
  document.getElementById('r-atr').textContent = d.atr.toLocaleString();

  // ATR% + 변동성 추세 → ATR 카드 서브텍스트
  const atrPctEl = document.getElementById('r-atr-pct');
  if (atrPctEl && d.buy_price && d.buy_price.atr_pct != null) {
    const vt = d.buy_price.vol_trend;
    const vtHtml = vt === 'expanding'   ? '<span style="color:#f85149">변동성 확대↑</span>' :
                   vt === 'contracting' ? '<span style="color:#3fb950">변동성 수축↓</span>' :
                                         '<span style="color:#d29922">변동성 안정</span>';
    atrPctEl.innerHTML = `${d.buy_price.atr_pct}% · ${vtHtml}`;
    atrPctEl.style.display = 'block';
  } else if (atrPctEl) {
    atrPctEl.style.display = 'none';
  }

  // 📐 피보나치 되돌림 기준 카드 및 "🔗 연계 밴드" 연동 표시는 모두 폐지됨.
  //    (ATR 밴드 가격과 피보나치 레벨 가격이 독립 계산되어 정합이 어려움 — renderForecast 참조)

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
  // 단계별 분석 리포트 (별도 탭)
  renderReport(d);
  // 예측/리스크
  renderForecast(d, isKrx);
  // 예측 탭: 핵심 구간·분할 매수·ATR·손익비
  renderPullbackIntoForecast(d, isKrx);
  // 기술적 지표 시그널 & 피봇 포인트
  renderTechnicalSignals(d);
  renderPivotPoints(d, isKrx);
  // 뉴스
  renderNews(d, isKrx);
  // 토스증권 AI 요약 비동기 로드 (KRX + US 공통, 결과 카드 상단 4번째 메트릭 카드)
  fetchTossAiSummary(d.symbol, d.market);
  // 탭 초기화
  switchTab('chart');
  // 차트는 탭 전환 후 렌더
  setTimeout(() => renderCharts(d, isKrx), 50);
}

function renderAI(d, isKrx) {
  renderDiagnosis(d, isKrx);
  renderInvestorFlow(d, isKrx);
  renderPullbackIntoAI(d, isKrx);
}

// ── 눌림목 분석 공통 초기화 헬퍼 ────────────────────────────────────────────
function _pullbackInit(d, isKrx) {
  const pa = d.pullback_analysis;
  if (!pa) return null;
  const C = { red:'#f85149', orange:'#d29922', yellow:'#e3b341', green:'#3fb950', blue:'#58a6ff', purple:'#bc8cff', gray:'#8b949e' };
  const fmtP = v => fmtPrice(v, isKrx);
  const stageColors = ['','#484f58','#d29922','#f85149','#3fb950','#58a6ff'];
  const stageLabels = ['','바닥 매집','돌파','1차 급등','눌림목','재급등'];
  return { pa, C, fmtP, stageColors, stageLabels };
}

// ── AI 진단 탭: 눌림목 관련 섹션 HTML 생성 (순수 문자열 반환) ──────────────
// display:contents + flex 컨테이너에서 innerHTML 동적 삽입이 비신뢰적이므로
// 순수 HTML 문자열을 반환해 renderDiagnosis 템플릿에 직접 삽입한다.
function _getPullbackDimsHtml(d, isKrx) {
  const init = _pullbackInit(d, isKrx);
  if (!init) return { topHtml: '', bottomHtml: '' };
  const { pa, C, fmtP, stageColors, stageLabels } = init;

  // ── 점수/색상 계산 ────────────────────────────────────────────────
  // ① 흐름 단계: stage 1~5 → 20~100점
  const stageVal = (pa.flow_stage || 0) * 20;
  const stageC   = stageColors[pa.flow_stage] || C.gray;
  const stageLbl = stageVal >= 80 ? '급등/재급등' : stageVal >= 60 ? '양호' : stageVal >= 40 ? '보통' : '초기';

  // ⑧ 눌림목 조건 점검: pullback_score_pct 그대로
  const pbVal = pa.pullback_score_pct || 0;
  const pbC   = pbVal >= 65 ? C.green : pbVal >= 40 ? C.orange : C.red;
  const pbLbl = pbVal >= 65 ? '양호' : pbVal >= 40 ? '보통' : '주의';

  // ⑨ 세력 흔들림 패턴 점검: 패턴 0개=85, 1개=55, 2개+=25
  const mfCount = (pa.manipulation_flags || []).length;
  const mfVal   = mfCount === 0 ? 85 : mfCount === 1 ? 55 : 25;
  const mfC     = mfVal >= 65 ? C.green : mfVal >= 40 ? C.orange : C.red;
  const mfLbl   = mfVal >= 65 ? '패턴 없음' : mfVal >= 40 ? '주의 1건' : '다수 감지';

  // ⑩ 구조 붕괴 · 손절 기준 점검: sl_triggered 0=85, 1=50, 2+=20
  const slVal = pa.sl_triggered === 0 ? 85 : pa.sl_triggered === 1 ? 50 : 20;
  const slC   = slVal >= 65 ? C.green : slVal >= 40 ? C.orange : C.red;
  const slLbl = slVal >= 65 ? '안전' : slVal >= 40 ? '경고' : '위험';

  // ── 아코디언 콘텐츠 빌더 ─────────────────────────────────────────
  // ① 흐름 단계 아코디언
  const flowAccContent = `
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px">
      <div style="background:${stageC};border-radius:8px;padding:6px 14px;font-size:15px;font-weight:700;color:#fff">
        ${stageLabels[pa.flow_stage] || '단계 불명'}
      </div>
      <div style="color:#cdd9e5;font-size:13px;line-height:1.6;flex:1">${pa.flow_desc}</div>
    </div>
    <div style="display:flex;gap:4px">
      ${[1,2,3,4,5].map(i => `<div style="flex:1;height:6px;border-radius:3px;background:${i <= pa.flow_stage ? stageColors[i] : '#21262d'}"></div>`).join('')}
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:4px;font-size:10px;color:#484f58">
      <span>바닥</span><span>돌파</span><span>급등</span><span style="color:${C.green};font-weight:700">눌림목</span><span>재급등</span>
    </div>`;

  // ⑧ 눌림목 조건 점검 아코디언
  const checkRows = (pa.pullback_checks || []).map(c => {
    const ic = c.pass ? '✅' : '❌';
    const tc = c.pass ? C.green : C.red;
    return `<tr>
      <td style="padding:5px 8px;white-space:nowrap">${ic} <span style="color:#cdd9e5;font-size:12px">${c.item}</span></td>
      <td style="padding:5px 8px;color:${tc};font-size:12px;line-height:1.4">${c.desc}</td>
    </tr>`;
  }).join('');
  const checkAccContent = `
    <table style="width:100%;border-collapse:collapse">${checkRows}</table>
    ${pa.last_surge_low ? `<div style="margin-top:8px;padding:6px 10px;background:#21262d;border-radius:6px;font-size:12px;color:#8b949e">급등봉 ${pa.surge_candles_count}개 감지 | 급등봉 저가 기준선: <b style="color:#e6edf3">${fmtP(pa.last_surge_low)}</b></div>` : ''}`;

  // ⑨ 세력 흔들림 패턴 점검 아코디언
  const mfItems = pa.manipulation_flags && pa.manipulation_flags.length > 0
    ? pa.manipulation_flags.map(f => `
      <div style="border-left:3px solid ${C[f.color]||C.orange};padding:8px 10px;background:#21262d;border-radius:0 6px 6px 0">
        <div style="color:${C[f.color]||C.orange};font-weight:700;font-size:12px">${f.pattern}</div>
        <div style="color:#8b949e;font-size:12px;margin-top:3px">${f.desc}</div>
        <div style="color:${C.blue};font-size:12px;margin-top:3px">대응: ${f.action}</div>
      </div>`).join('')
    : `<div style="color:#8b949e;font-size:13px;padding:4px 0">감지된 세력 흔들림 패턴 없음</div>`;
  const mfAccContent = mfItems;

  // ⑩ 구조 붕괴 · 손절 기준 점검 아코디언
  const slRows = (pa.sl_conditions || []).map(s => {
    const ic = s.triggered ? '🔴' : '🟢';
    const tc = s.triggered ? C.red : C.green;
    return `<tr>
      <td style="padding:5px 8px;white-space:nowrap">${ic} <span style="color:#cdd9e5;font-size:12px">${s.cond}</span></td>
      <td style="padding:5px 8px;color:${tc};font-size:12px;line-height:1.4">${s.desc}</td>
    </tr>`;
  }).join('');
  const slAccContent = `
    <table style="width:100%;border-collapse:collapse">${slRows}</table>
    <div style="margin-top:8px;padding:6px 10px;background:#21262d;border-radius:6px;font-size:12px;color:#8b949e">
      손절 원칙: 손실은 짧게 (-5%~-8% 이내) | 구조가 무너지면 미련 없이 정리
    </div>`;

  // ── 눌림목 dim 점수 → 색상·레이블 (5단계, renderDiagnosis의 _dg와 동일 기준) ──
  const _pbDg = v => v >= 75 ? { c:C.green,  lbl:'우수'  }
                   : v >= 55 ? { c:C.blue,   lbl:'양호'  }
                   : v >= 40 ? { c:C.orange, lbl:'보통'  }
                   : v >= 25 ? { c:'#f97316',lbl:'주의'  }
                   :           { c:C.red,    lbl:'위험'  };

  // ── 아코디언 행 빌더 ──────────────────────────────────────────────
  const pbDimBar = (emoji, label, val, desc, aId, content, extraBorder) => {
    const { c: pbC, lbl: pbLbl } = _pbDg(val);
    return `
    <div class="diag-dim diag-dim-clickable" onclick="toggleDimAccordion('${aId}')" ${extraBorder ? `style="border-color:${extraBorder}"` : ''}>
      <div class="diag-dim-head">
        <span class="diag-dim-label">${emoji} ${label} <span id="arrow-${aId}" style="font-size:11px;color:#8b949e;display:inline-block;transition:transform .25s">▼</span></span>
        <span class="diag-dim-score" style="color:${pbC}">${val}점 · ${pbLbl}</span>
      </div>
      <div class="diag-bar-bg"><div class="diag-bar-fill" style="width:${val}%;background:${pbC}"></div></div>
      <div class="diag-dim-desc">${desc}</div>
    </div>
    <div id="${aId}" style="display:none;padding:12px;background:#0d1117;border-radius:10px;border:1px solid #30363d;margin-top:-2px">
      <div style="display:flex;flex-direction:column;gap:8px">${content}</div>
    </div>`;
  };

  // HTML 문자열 반환 (DOM 조작 없음)
  const topHtml =
      pbDimBar('🔄', '현재 흐름 단계', stageVal,
          `${stageLabels[pa.flow_stage] || '단계 불명'} · ${pa.flow_desc ? pa.flow_desc.substring(0,40) + (pa.flow_desc.length > 40 ? '…' : '') : ''}`,
          'pb-flow', flowAccContent);

  const bottomHtml =
      pbDimBar('✅', '눌림목 조건 점검', pbVal,
          `${pa.pullback_grade} · ${pa.pullback_pass_count}/${(pa.pullback_checks||[]).length} 조건 충족 · ${pa.pullback_desc ? pa.pullback_desc.substring(0,35) + (pa.pullback_desc.length > 35 ? '…' : '') : ''}`,
          'pb-check', checkAccContent)
    + pbDimBar('🎭', `세력 흔들림 패턴 점검${pa.bb_squeeze ? ' · 볼린저 수축' : ''}`, mfVal,
          mfCount > 0 ? `${mfCount}개 패턴 감지 — 속임수 가능성 확인 필요` : '세력 흔들림 패턴 미감지 — 구조 유지 중',
          'pb-manip', mfAccContent)
    + pbDimBar('🛑', '구조 붕괴 · 손절 기준 점검', slVal,
          `${pa.breakdown_verdict} · ${pa.sl_triggered}개 조건 충족`,
          'pb-sl', slAccContent);

  return { topHtml, bottomHtml };
}

// 하위 호환 래퍼 (renderAI에서 호출하지만 이제 renderDiagnosis 내부에서 처리됨)
function renderPullbackIntoAI(d, isKrx) {
  // renderDiagnosis가 이미 _getPullbackDimsHtml()로 렌더링했으므로 no-op
}

// ── 예측 탭: ③ 핵심 구간  ④ 분할 매수  ⑤ ATR 리스크  ⑥ 손익비 시나리오 ──────
function renderPullbackIntoForecast(d, isKrx) {
  const el = document.getElementById('pullback-forecast-section');
  if (!el) return;
  // 📍 핵심 가격 구간 · 📊 분할 매수 전략 섹션은 폐지됨.
  //   두 섹션의 전략 설명(가격 숫자 제외)은 "🎯 현재가 기준 매수 전략"의
  //   ⚡ 1차 매수 구간(ATR 기반) / 📍 2차 매수 구간 카드에 통합되었다. (renderForecast 참조)
  el.innerHTML = '';
}

// ── (하위 호환) 눌림목/손익비 분석 렌더 — 더 이상 사용하지 않음 ───────────────
function renderPullbackAnalysis(d, isKrx) {
  const el = document.getElementById('pullback-analysis-section');
  if (!el) return;
  const pa = d.pullback_analysis;
  if (!pa) { el.innerHTML = '<div style="padding:20px;color:#8b949e;text-align:center">분석 데이터 없음</div>'; return; }

  const C = { red:'#f85149', orange:'#d29922', yellow:'#e3b341', green:'#3fb950', blue:'#58a6ff', purple:'#bc8cff', gray:'#8b949e' };
  const fmtP = v => fmtPrice(v, isKrx);
  const fmtN = v => Number(v).toLocaleString('ko-KR');

  // 단계 배지 색상
  const stageColors = ['','#484f58','#d29922','#f85149','#3fb950','#58a6ff'];
  const stageLabels = ['','① 바닥 매집','② 돌파','③ 1차 급등','④ 눌림목','⑤ 재급등'];
  const stageC = stageColors[pa.flow_stage] || C.gray;

  // 눌림목 조건 점검
  const checkRows = (pa.pullback_checks || []).map(c => {
    const ic = c.pass ? '✅' : '❌';
    const tc = c.pass ? C.green : C.red;
    return `<tr>
      <td style="padding:5px 8px;white-space:nowrap">${ic} <span style="color:#cdd9e5">${c.item}</span></td>
      <td style="padding:5px 8px;color:${tc};font-size:12px">${c.desc}</td>
    </tr>`;
  }).join('');

  // 분할 진입 구간
  const entryRows = (pa.entry_zones || []).map(z => `
    <div style="border-left:3px solid ${z.color};padding:8px 10px;margin-bottom:8px;background:#1c2128;border-radius:0 6px 6px 0">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px">
        <span style="color:${z.color};font-weight:700;font-size:13px">${z.stage}</span>
        <span style="color:#e6edf3;font-size:13px;font-weight:600">${fmtP(z.range[0])} ~ ${fmtP(z.range[1])}</span>
        <span style="background:#30363d;border-radius:4px;padding:2px 8px;color:#8b949e;font-size:11px">${z.ratio}</span>
      </div>
      <div style="color:#8b949e;font-size:12px;margin-top:4px">${z.desc}</div>
    </div>`).join('');

  // 손익비 시나리오
  const rrRows = (pa.rr_scenarios || []).map(s => {
    const rrC = s.rr >= 2.3 ? C.green : s.rr >= 2.0 ? C.yellow : C.red;
    return `<div style="background:#1c2128;border-radius:6px;padding:10px;border:1px solid ${s.viable ? C.green : '#30363d'}">
      <div style="color:#cdd9e5;font-size:12px;font-weight:600;margin-bottom:6px">${s.label}</div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;font-size:12px">
        <span>진입 <b style="color:#e6edf3">${fmtP(s.entry)}</b></span>
        <span>목표 <b style="color:${C.green}">${fmtP(s.target)}</b></span>
        <span>손절 <b style="color:${C.red}">${fmtP(s.stop)}</b></span>
        <span style="color:${rrC};font-weight:700">R/R ${s.rr}:1</span>
      </div>
    </div>`;
  }).join('');

  // 구조 붕괴 손절 조건
  const slRows = (pa.sl_conditions || []).map(s => {
    const ic = s.triggered ? '🔴' : '🟢';
    const tc = s.triggered ? C.red : C.green;
    return `<tr>
      <td style="padding:5px 8px;white-space:nowrap">${ic} <span style="color:#cdd9e5;font-size:12px">${s.cond}</span></td>
      <td style="padding:5px 8px;color:${tc};font-size:11px">${s.desc}</td>
    </tr>`;
  }).join('');

  // 세력 흔들림 패턴 점검
  const mfHtml = pa.manipulation_flags && pa.manipulation_flags.length > 0
    ? pa.manipulation_flags.map(f => `
      <div style="border-left:3px solid ${C[f.color]||C.orange};padding:8px 10px;background:#1c2128;border-radius:0 6px 6px 0;margin-bottom:6px">
        <div style="color:${C[f.color]||C.orange};font-weight:700;font-size:12px">${f.pattern}</div>
        <div style="color:#8b949e;font-size:12px;margin-top:3px">${f.desc}</div>
        <div style="color:#58a6ff;font-size:12px;margin-top:3px">대응: ${f.action}</div>
      </div>`).join('')
    : `<div style="color:#8b949e;font-size:13px;padding:8px 0">감지된 세력 흔들림 패턴 없음</div>`;

  const pbGC = C[pa.pullback_grade_color] || C.blue;
  const bvC  = C[pa.breakdown_color]      || C.gray;

  el.innerHTML = `
  <!-- ══ SECTION A: 눌림목 분석 ══ -->

  <!-- A1. 현재 흐름 단계 -->
  <div class="card" style="margin-bottom:12px">
    <div class="card-title">① 현재 흐름 단계 (5단계 구조)</div>
    <div style="display:flex;align-items:center;gap:14px;padding:10px 0;flex-wrap:wrap">
      <div style="background:${stageC};border-radius:8px;padding:8px 18px;font-size:20px;font-weight:700;color:#fff">
        ${stageLabels[pa.flow_stage] || '단계 불명'}
      </div>
      <div style="color:#cdd9e5;font-size:13px;line-height:1.6;flex:1">${pa.flow_desc}</div>
    </div>
    <div style="display:flex;gap:4px;margin-top:8px">
      ${[1,2,3,4,5].map(i => `
        <div style="flex:1;height:6px;border-radius:3px;background:${i <= pa.flow_stage ? stageColors[i] : '#21262d'}"></div>
      `).join('')}
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:4px;font-size:10px;color:#484f58">
      <span>바닥</span><span>돌파</span><span>급등</span><span style="color:${C.green};font-weight:700">눌림목</span><span>재급등</span>
    </div>
  </div>

  <!-- A2. 눌림목 조건 점검 -->
  <div class="card" style="margin-bottom:12px;border:2px solid ${pbGC}">
    <div class="card-title">② 눌림목 조건 점검</div>
    <div style="display:flex;align-items:center;gap:14px;padding:10px 0 8px;flex-wrap:wrap">
      <div style="text-align:center;min-width:80px">
        <div style="font-size:28px;font-weight:700;color:${pbGC}">${pa.pullback_pass_count}/${(pa.pullback_checks||[]).length}</div>
        <div style="font-size:11px;color:#8b949e">조건 충족</div>
      </div>
      <div>
        <div style="color:${pbGC};font-weight:700;font-size:16px">${pa.pullback_grade}</div>
        <div style="color:#cdd9e5;font-size:12px;margin-top:4px;max-width:280px">${pa.pullback_desc}</div>
      </div>
      <div style="margin-left:auto;text-align:right">
        <div style="font-size:22px;font-weight:700;color:${pbGC}">${pa.pullback_score_pct}%</div>
        <div style="font-size:11px;color:#8b949e">품질 점수</div>
      </div>
    </div>
    <table style="width:100%;border-collapse:collapse;margin-top:4px">
      ${checkRows}
    </table>
    ${pa.last_surge_low ? `<div style="margin-top:8px;padding:6px 10px;background:#1c2128;border-radius:6px;font-size:12px;color:#8b949e">급등봉 ${pa.surge_candles_count}개 감지 | 급등봉 저가 기준선: <b style="color:#e6edf3">${fmtP(pa.last_surge_low)}</b></div>` : ''}
  </div>

  <!-- ══ SECTION B: 실전형 손익비 자리 ══ -->

  <!-- B1. 핵심 구간(Zone) -->
  <div class="card" style="margin-bottom:12px">
    <div class="card-title">③ 핵심 가격 구간 (Zone)</div>
    <div style="display:flex;flex-direction:column;gap:8px;padding:8px 0">
      <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#1c2128;border-left:4px solid ${C.red};border-radius:0 6px 6px 0">
        <span style="color:#cdd9e5;font-weight:600">상단 저항대 (목표)</span>
        <span style="color:${C.red};font-weight:700">${fmtP(pa.zones.resistance.low)} ~ ${fmtP(pa.zones.resistance.high)}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#1c2128;border-left:4px solid ${C.green};border-radius:0 6px 6px 0">
        <span style="color:#cdd9e5;font-weight:600">핵심 일치가격대 (지지/저항 전환)</span>
        <span style="color:${C.green};font-weight:700">${fmtP(pa.zones.core.low)} ~ ${fmtP(pa.zones.core.high)}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#1c2128;border-left:4px solid ${C.blue};border-radius:0 6px 6px 0">
        <span style="color:#cdd9e5;font-weight:600">하단 방어 구간 (추세선+MA 밀집)</span>
        <span style="color:${C.blue};font-weight:700">${fmtP(pa.zones.defense.low)} ~ ${fmtP(pa.zones.defense.high)}</span>
      </div>
    </div>
  </div>

  <!-- B2. 분할 진입 전략 -->
  <div class="card" style="margin-bottom:12px">
    <div class="card-title">④ 분할 매수 전략 (한 번에 물량 투입 금지)</div>
    <div style="padding:8px 0">${entryRows}</div>
    <div style="padding:6px 10px;background:#1c2128;border-radius:6px;color:#8b949e;font-size:12px;margin-top:4px">
      원칙: 한 번에 물량 투입 금지 → 분할매수로 평균단가 낮추고 리스크 분산
    </div>
  </div>

  <!-- B3. ATR 기반 손절/익절 -->
  <div class="card" style="margin-bottom:12px">
    <div class="card-title">⑤ ATR 기반 리스크 관리</div>
    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;padding:8px 0">
      <div style="background:#1c2128;border-radius:6px;padding:10px;border:1px solid ${C.red}">
        <div style="font-size:11px;color:#8b949e">손절선 (구조 붕괴 기준)</div>
        <div style="font-size:16px;font-weight:700;color:${C.red}">${fmtP(pa.stop_loss)}</div>
        <div style="font-size:11px;color:${C.orange};margin-top:3px">${pa.stop_loss_pct}% | 손실은 작게 (-5~8% 이내)</div>
      </div>
      <div style="background:#1c2128;border-radius:6px;padding:10px;border:1px solid ${C.green}">
        <div style="font-size:11px;color:#8b949e">목표가 (1차)</div>
        <div style="font-size:16px;font-weight:700;color:${C.green}">${fmtP(pa.target_main)}</div>
        <div style="font-size:11px;color:${pa.target_source && pa.target_source.includes('앙상블') ? C.blue : C.gray};margin-top:3px">
          R/R ${pa.rr_main}:1 · ${pa.target_source || '기술적 분석'}
        </div>
      </div>
      <div style="background:#1c2128;border-radius:6px;padding:10px">
        <div style="font-size:11px;color:#8b949e">2차 목표 (돌파 후)</div>
        <div style="font-size:16px;font-weight:700;color:${C.blue}">${fmtP(pa.target_ext)}</div>
      </div>
      <div style="background:#1c2128;border-radius:6px;padding:10px">
        <div style="font-size:11px;color:#8b949e">트레일링 스탑</div>
        <div style="font-size:16px;font-weight:700;color:${C.purple}">${fmtP(pa.trail_stop)}</div>
        <div style="font-size:11px;color:#8b949e;margin-top:3px">ATR × 1.5</div>
      </div>
    </div>
  </div>

  <!-- B4. 손익비 시나리오 -->
  <div class="card" style="margin-bottom:12px">
    <div class="card-title">⑥ 손익비 시나리오 (3가지)</div>
    <div style="display:flex;flex-direction:column;gap:8px;padding:8px 0">${rrRows}</div>
    <div style="padding:8px 10px;background:#1c2128;border-radius:6px;margin-top:4px;font-size:12px;color:#8b949e">
      손익비는 "얼마를 벌까?"가 아니라 <b style="color:#e6edf3">"얼마를 잃지 않을까?"</b>에서 시작한다.
    </div>
  </div>

  <!-- B5. 세력 흔들림 패턴 점검 -->
  <div class="card" style="margin-bottom:12px">
    <div class="card-title">⑦ 세력 흔들림 패턴 점검${pa.bb_squeeze ? ' — <span style="color:'+C.yellow+'">볼린저 수축 감지</span>' : ''}</div>
    <div style="padding:8px 0">${mfHtml}</div>
  </div>

  <!-- B6. 구조 붕괴 · 손절 기준 점검 -->
  <div class="card" style="border:2px solid ${bvC}">
    <div class="card-title">⑧ 구조 붕괴 · 손절 기준 점검 (${pa.sl_triggered}개 조건 충족)</div>
    <div style="display:inline-flex;align-items:center;gap:10px;padding:10px 0 8px">
      <span style="font-size:20px">${pa.sl_triggered >= 2 ? '🔴' : pa.sl_triggered === 1 ? '🟡' : '🟢'}</span>
      <span style="color:${bvC};font-weight:700;font-size:15px">${pa.breakdown_verdict}</span>
    </div>
    <table style="width:100%;border-collapse:collapse">${slRows}</table>
    <div style="padding:8px 10px;background:#1c2128;border-radius:6px;margin-top:8px;font-size:12px;color:#8b949e">
      손절 원칙: 손실은 짧게 (-5%~-8% 이내) | 구조가 무너지면 미련 없이 정리
    </div>
  </div>
  `;
}

// ── 단계별 분석 리포트 렌더 (tab-report 전용) ────────────────────────────────
function renderReport(d) {
  const stepsList = document.getElementById('steps-list');
  if (!stepsList) return;
  stepsList.innerHTML = `<div style="text-align:center;padding:32px 16px;color:#8b949e;font-size:13px">
    <div style="font-size:28px;margin-bottom:12px">🔬</div>
    단계별 분석 내용이 <strong style="color:#58a6ff">AI 진단</strong> 탭으로 이동되었습니다.<br>
    <span style="font-size:12px;margin-top:8px;display:inline-block">각 지표 항목을 클릭하면 상세 분석을 확인할 수 있습니다.</span>
  </div>`;
}
// ── 수급 흐름 해석 문구 생성 ─────────────────────────────────────────────────
function calcSupplyDesc(flow) {
  if (!flow || !flow.ok) return '수급 데이터 부족 — 관망';

  const fore = flow['외국인'] || 0;
  const inst = flow['기관']   || 0;
  const indi = flow['개인']   || 0;

  const total  = Math.abs(fore) + Math.abs(inst) + Math.abs(indi);
  const thresh = Math.max(total * 0.05, 50000);

  const foreDir = fore >  thresh ? 'buy'  : fore < -thresh ? 'sell' : 'neu';
  const instDir = inst >  thresh ? 'buy'  : inst < -thresh ? 'sell' : 'neu';
  const indiDir = indi >  thresh ? 'buy'  : indi < -thresh ? 'sell' : 'neu';

  const keyInst = ['연기금','금융투자','투신','사모'].map(k => flow[k] || 0);
  const kiBuy   = keyInst.filter(v => v > 0).length;
  const kiSell  = keyInst.filter(v => v < 0).length;
  const instMix = kiBuy >= 1 && kiSell >= 1;

  let head, tail;

  if (foreDir === 'sell' && instDir === 'sell' && indiDir === 'buy') {
    head = '외국인·기관 동반 매도, 개인 매수 방어';  tail = '수급 부담 우세';
  } else if (foreDir === 'buy' && instDir === 'buy' && indiDir === 'sell') {
    head = '외국인·기관 동반 매수, 개인 차익실현';    tail = '매수 우세';
  } else if (foreDir === 'buy' && instDir === 'buy') {
    head = '외국인·기관 동반 매수 유입';              tail = '매수 우세 흐름';
  } else if (foreDir === 'sell' && instDir === 'sell') {
    head = '외국인·기관 동반 이탈';                   tail = '수급 부담';
  } else if (foreDir === 'buy' && instMix) {
    head = '외국인 매수 주도, 기관 내부 혼조';        tail = '수급 중립권 관망';
  } else if (foreDir === 'buy' && instDir === 'neu') {
    head = '외국인 주도 매수세 유입, 기관 관망';      tail = '수급 개선';
  } else if (foreDir === 'buy' && instDir === 'sell') {
    head = '외국인 매수·기관 매도 혼조';              tail = '방향성 제한';
  } else if (instDir === 'buy' && foreDir === 'neu') {
    head = instMix ? '기관 내부 혼조 속 순매수 우세' : '기관 주도 매수세 강화, 외국인 관망';
    tail = '매수 우세';
  } else if (instDir === 'buy' && foreDir === 'sell') {
    head = '기관 매수, 외국인 이탈 혼조';             tail = '방향성 제한';
  } else if (foreDir === 'sell' && instDir === 'neu') {
    head = instMix ? '외국인 이탈, 기관 내부 혼조' : '외국인 이탈, 기관 관망';
    tail = '수급 부담';
  } else if (instDir === 'sell' && foreDir === 'neu') {
    head = instMix ? '기관 내부 혼조 속 순매도' : '기관 약세, 외국인 관망';
    tail = '수급 중립 하단';
  } else if (foreDir === 'sell' && indiDir === 'buy') {
    head = '개인 매수 집중, 외국인 이탈';             tail = '단기 수급 불균형';
  } else if (instDir === 'sell' && indiDir === 'buy') {
    head = '개인 매수 집중, 기관 이탈';               tail = '단기 수급 불균형';
  } else if (foreDir === 'neu' && instDir === 'neu') {
    if      (indiDir === 'buy')  { head = '개인 매수 집중, 기관·외국인 관망'; tail = '수급 중립'; }
    else if (indiDir === 'sell') { head = '개인 차익실현, 전반적 관망';        tail = '중립 관망'; }
    else                         { head = '수급 방향성 미미';                  tail = '중립 관망'; }
  } else {
    head = instMix ? '기관 내부 혼조 속 방향성 불명확' : '수급 방향성 불명확';
    tail = '중립 — 관망';
  }

  return `${head} — ${tail}`;
}

// ── 5-차원 종목 진단 렌더 ────────────────────────────────────────────────────
// ── 복합 기술 신호 점수 섹션 렌더러 (HybridTurtle NCS) ──────────────────────
function renderHybridSection(d) {
  const hs = d.hybrid_score;
  if (!hs || hs.error) return '';
  const isKrx = d.market === 'KRX';   // 시장 구분 — 진입가·손절가 포맷에 사용

  const ncs = hs.ncs ?? 0;
  const bqs = hs.bqs ?? 0;
  const fws = hs.fws ?? 0;
  const action = hs.action || 'CONDITIONAL';
  const regime = hs.regime || 'SIDEWAYS';
  const hurst  = hs.hurst;
  const bis    = hs.bis_score ?? 0;
  const adx    = hs.adx ?? 0;
  const atrPct = hs.atr_percent;
  const volRatio = hs.vol_ratio;
  const antiChase = hs.anti_chase || {};
  const entryTrigger = hs.entry_trigger;
  const stopPrice    = hs.stop_price;

  // 색상 팔레트
  const C = { red:'#f85149', orange:'#f97316', yellow:'#d29922', green:'#3fb950', blue:'#58a6ff', purple:'#bc8cff', gray:'#8b949e', teal:'#39d353' };

  // 종합 점수 색상 및 등급
  const ncsColor = ncs >= 70 ? C.green : ncs >= 50 ? C.blue : ncs >= 35 ? C.yellow : C.red;
  const ncsLabel = ncs >= 70 ? '우수' : ncs >= 50 ? '양호' : ncs >= 35 ? '보통' : '주의';

  // 하락 위험 강도 색상 (높을수록 위험)
  const fwsColor = fws <= 30 ? C.green : fws <= 50 ? C.yellow : fws <= 65 ? C.orange : C.red;
  const fwsLabel = fws <= 30 ? '낮음' : fws <= 50 ? '보통' : fws <= 65 ? '높음' : '매우 높음';

  // 진입 판단 배지
  const actionMap = {
    'AUTO_YES':    { color: C.green,  bg: '#0d2d1a', text: '✅ 진입 가능' },
    'AUTO_NO':     { color: C.red,    bg: '#2d1515', text: '❌ 진입 지양' },
    'CONDITIONAL': { color: C.yellow, bg: '#2d2200', text: '⚠️ 조건 확인 후 진입' },
  };
  const act = actionMap[action] || actionMap['CONDITIONAL'];

  // 시장 국면 배지
  const regimeMap = {
    'BULLISH':  { color: C.green,  bg: '#0d2d1a', icon: '🐂', text: '강세장' },
    'BEARISH':  { color: C.red,    bg: '#2d1515', icon: '🐻', text: '약세장' },
    'SIDEWAYS': { color: C.gray,   bg: '#21262d', icon: '↔️', text: '횡보장' },
  };
  const reg = regimeMap[regime] || regimeMap['SIDEWAYS'];

  // 추세 지속성 (허스트 지수) 레이블
  const hurstLabel = hurst == null ? '—'
    : hurst >= 0.65 ? `${hurst.toFixed(2)} — 추세 지속 가능성 높음`
    : hurst >= 0.5  ? `${hurst.toFixed(2)} — 방향성 불명확`
    : `${hurst.toFixed(2)} — 추세 반전 가능성`;
  const hurstColor = hurst == null ? C.gray : hurst >= 0.6 ? C.green : hurst >= 0.5 ? C.blue : C.orange;

  // 고점 추격 경고
  const chaseText = antiChase.chasing
    ? `<span style="color:${C.red}">⛔ ${antiChase.reason || '현재가가 단기 급등 구간 — 추격 매수 주의'}</span>`
    : `<span style="color:${C.green}">✓ ${antiChase.reason || '현재가가 과도하게 올라있지 않습니다'}</span>`;

  // 권장 진입가 / 손절 기준가
  //   · entryTrigger : 20일 고점 위 적응형 돌파 트리거 (HybridTurtle Module 11b).
  //                    "돌파 매수" 진입 판단 로직과 일치하므로 현재가가 아닌 돌파가를 표시한다.
  //   · stopPrice    : 진입가 − 1.5×ATR. 백엔드 값이 누락/역전(진입가 이상)된 경우
  //                    ATR(절대값) 기준으로 재계산해 "항상 진입가보다 낮은" 손절선을 보장한다.
  const _atrAbs = (typeof hs.atr === 'number' && hs.atr > 0) ? hs.atr : null;
  const _entry  = (typeof entryTrigger === 'number' && entryTrigger > 0) ? entryTrigger : null;
  let   _stop   = (typeof stopPrice === 'number' && stopPrice > 0) ? stopPrice : null;
  if (_entry && _atrAbs && (!_stop || _stop >= _entry)) {
    _stop = _entry - 1.5 * _atrAbs;   // 논리 보정: 손절가는 반드시 진입가 아래
  }
  const _riskPct = (_entry && _stop) ? ((_stop - _entry) / _entry * 100) : null;
  const entryRow = (_entry && _stop && _stop > 0) ? `
    <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
      <div style="flex:1;min-width:120px;background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:8px 10px">
        <div style="font-size:10px;color:#8b949e;margin-bottom:2px">권장 진입가 <span style="font-size:9px">(20일 고점 돌파 시)</span></div>
        <div style="font-size:13px;font-weight:700;color:${C.blue}">${fmtPrice(_entry, isKrx)}</div>
      </div>
      <div style="flex:1;min-width:120px;background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:8px 10px">
        <div style="font-size:10px;color:#8b949e;margin-bottom:2px">손절 기준가 <span style="font-size:9px">(이 가격 이탈 시 손실 제한)</span></div>
        <div style="font-size:13px;font-weight:700;color:${C.red}">${fmtPrice(_stop, isKrx)}${_riskPct != null ? ` <span style="font-size:10px;color:#8b949e;font-weight:400">진입가 대비 ${_riskPct.toFixed(1)}%</span>` : ''}</div>
      </div>
    </div>` : '';

  // 보조 지표 그리드 (초보자용 레이블)
  const detailItems = [
    {
      label: '추세 강도',
      tooltip: '방향성 강도. 25 이상이면 뚜렷한 추세, 미만이면 방향 없이 횡보 중',
      val: adx ? adx.toFixed(1) : '—',
      color: adx >= 25 ? C.green : C.yellow,
    },
    {
      label: '일일 변동폭',
      tooltip: '하루 평균 가격이 얼마나 움직이는지 (%). 낮을수록 안정적',
      val: atrPct ? atrPct.toFixed(2) + '%' : '—',
      color: atrPct && atrPct <= 4 ? C.green : atrPct && atrPct <= 7 ? C.yellow : C.red,
    },
    {
      label: '거래량 배율',
      tooltip: '오늘 거래량이 최근 20일 평균 대비 몇 배인지. 1.2배 이상이면 관심 높음',
      val: volRatio ? volRatio.toFixed(2) + 'x' : '—',
      color: volRatio && volRatio >= 1.2 ? C.green : C.yellow,
    },
    {
      label: '추세 지속성',
      tooltip: '현재 추세가 계속 이어질 가능성. 0.6 이상이면 방향 유지 가능성 높음',
      val: hurstLabel,
      color: hurstColor,
    },
    {
      label: '돌파 신뢰도',
      tooltip: '상승 돌파 신호가 얼마나 믿을 만한지 (0~15점). 10점 이상이면 신뢰도 높음',
      val: bis + '/15점',
      color: bis >= 10 ? C.green : bis >= 5 ? C.yellow : C.gray,
    },
    {
      label: '고점 추격',
      tooltip: '이미 많이 오른 가격에 뒤늦게 사는 위험 여부',
      val: antiChase.chasing ? '⚠️ 주의' : '✓ 안전',
      color: antiChase.chasing ? C.red : C.green,
    },
  ].map(item => `
    <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:8px;text-align:center" title="${item.tooltip}">
      <div style="font-size:10px;color:#8b949e;margin-bottom:3px">${item.label}</div>
      <div style="font-size:12px;font-weight:600;color:${item.color}">${item.val}</div>
    </div>`).join('');

  // 아코디언 세부 내용
  const accordionContent = `
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">
      <div style="flex:1;min-width:80px">
        <div style="font-size:10px;color:#8b949e;margin-bottom:2px">상승 신호 강도</div>
        <div style="font-size:10px;color:#484f58;margin-bottom:4px">상승 돌파 신호가 얼마나 강하고 믿을 만한지 (높을수록 좋음)</div>
        <div style="background:#21262d;border-radius:6px;height:8px;overflow:hidden">
          <div style="height:100%;width:${bqs}%;background:${C.blue};border-radius:6px;transition:width .4s"></div>
        </div>
        <div style="font-size:11px;color:${C.blue};margin-top:2px">${bqs.toFixed(1)}점</div>
      </div>
      <div style="flex:1;min-width:80px">
        <div style="font-size:10px;color:#8b949e;margin-bottom:2px">하락 위험 강도</div>
        <div style="font-size:10px;color:#484f58;margin-bottom:4px">추세가 꺾이거나 하락할 위험 정도 (낮을수록 안전)</div>
        <div style="background:#21262d;border-radius:6px;height:8px;overflow:hidden">
          <div style="height:100%;width:${fws}%;background:${fwsColor};border-radius:6px;transition:width .4s"></div>
        </div>
        <div style="font-size:11px;color:${fwsColor};margin-top:2px">${fws.toFixed(1)}점 · ${fwsLabel}</div>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:10px">
      ${detailItems}
    </div>
    ${entryRow}
    <div style="margin-top:10px;font-size:11px;color:#8b949e">
      고점 추격 경고: ${chaseText}
    </div>
    <div style="margin-top:8px;font-size:10px;color:#484f58;line-height:1.6;border-top:1px solid #21262d;padding-top:8px">
      이 점수는 상승 신호의 강도에서 하락 위험 강도를 빼고 보정한 종합 기술 점수입니다.<br>
      점수가 높을수록 기술적으로 진입에 유리한 조건이며, 단독 판단보다 다른 지표와 함께 참고하세요.
    </div>`;

  return `
    <div class="diag-dim diag-dim-clickable" onclick="toggleDimAccordion('dim-ncs')">
      <div class="diag-dim-head">
        <span class="diag-dim-label">🤖 복합 기술 신호 점수 <span id="arrow-dim-ncs" style="font-size:11px;color:#8b949e;display:inline-block;transition:transform .25s">▼</span></span>
        <span class="diag-dim-score" style="color:${ncsColor}">${ncs.toFixed(0)}점 · ${ncsLabel}</span>
      </div>
      <div class="diag-bar-bg"><div class="diag-bar-fill" style="width:${ncs}%;background:${ncsColor}"></div></div>
      <div class="diag-dim-desc" style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
        <span style="background:${act.bg};color:${act.color};border:1px solid ${act.color};border-radius:5px;padding:1px 7px;font-size:11px;font-weight:600">${act.text}</span>
        <span style="background:${reg.bg};color:${reg.color};border:1px solid ${reg.color};border-radius:5px;padding:1px 7px;font-size:11px">${reg.icon} ${reg.text}</span>
        <span style="color:#8b949e;font-size:11px">추세강도 ${adx ? adx.toFixed(0) : '—'} · 돌파신뢰도 ${bis}/15</span>
      </div>
    </div>
    <div id="dim-ncs" style="display:none;padding:12px;background:#0d1117;border-radius:10px;border:1px solid #30363d;margin-top:-2px">
      ${accordionContent}
    </div>`;
}

function renderDiagnosis(d, isKrx) {
  const diagEl = document.getElementById('ai-diagnosis-chart');
  if (!diagEl) return;

  const score  = d.score   || 50;
  const rsi    = d.rsi     || 50;
  const bp     = d.buy_price;
  const flow   = d.investor_flow;
  const patterns = d.candlestick_patterns || [];

  // ── 눌림목 관련 HTML 미리 생성 (display:contents 우회) ──────────────
  // flex 컨테이너 내 동적 innerHTML 삽입 대신 템플릿 문자열에 직접 주입
  const { topHtml: pbTopHtml, bottomHtml: pbBottomHtml } = _getPullbackDimsHtml(d, isKrx);

  // ── 1. 기술적 추세 (Technical Trend) ────────────────────────────
  const techScore = Math.min(100, Math.max(0, score));

  // ── 2. 모멘텀 (RSI 기반) ─────────────────────────────────────────
  let mScore = 50;
  if      (rsi > 70)              mScore = Math.max(10, 45 - (rsi - 70) * 2);
  else if (rsi > 60)              mScore = 60;
  else if (rsi >= 45 && rsi<=60)  mScore = 75;
  else if (rsi >= 30)             mScore = 52;
  else                            mScore = Math.min(75, 55 + (30 - rsi) * 1.5); // 과매도 = 반등기회
  const momentumScore = Math.min(100, Math.max(0, Math.round(mScore)));

  // ── 3. 변동성 수준 (ATR 기반) ───────────────────────────────────
  let vScore = 60;
  if (bp) {
    const atrPct = parseFloat(bp.atr_pct) || 0;
    if      (bp.vol_trend === 'contracting') vScore = 82;
    else if (bp.vol_trend === 'expanding')   vScore = 28;
    else                                     vScore = 62;
    if      (atrPct > 6) vScore = Math.max(15, vScore - 25);
    else if (atrPct > 3) vScore = Math.max(25, vScore - 12);
    else if (atrPct < 2) vScore = Math.min(92, vScore + 10);
  }
  const volScore = Math.min(100, Math.max(0, Math.round(vScore)));

  // ── 4. 수급 흐름 ────────────────────────────────────────────────
  let sScore = 50;
  if (isKrx && flow && flow.ok) {
    const net = (flow['외국인'] || 0) + (flow['기관'] || 0);
    sScore = 50 + Math.min(35, Math.max(-35, net / 4000));
  } else if (!isKrx && d.us_enriched && d.us_enriched.sentiment) {
    const bull = Number(d.us_enriched.sentiment.bullish_pct || 0.5);
    sScore = Math.round(bull * 100);
  }
  const supplyScore = Math.min(100, Math.max(0, Math.round(sScore)));

  // ── 5. 캔들·차트 패턴 신호 ──────────────────────────────────────
  const bullPat = patterns.filter(p => p.direction === '상승').length;
  const bearPat = patterns.filter(p => p.direction === '하락').length;
  let pScore = 50;
  if (bullPat + bearPat > 0) {
    pScore = Math.round((bullPat / (bullPat + bearPat)) * 80 + 10);
  }
  const patScore = Math.min(100, Math.max(0, pScore));

  // ── 종합 등급 계산 (8단계 세분화 체계) ─────────────────────────────
  // threshold: A(82+) A-(72+) B(62+) B-(52+) C(42+) C-(30+) D(15+) D-(0+)
  const hasSupply = (isKrx && flow && flow.ok) || (!isKrx && d.us_enriched && d.us_enriched && d.us_enriched.sentiment);
  const dims = [techScore, momentumScore, volScore, supplyScore, patScore];
  const dimNameMap = ['기술추세', '모멘텀', '변동성', '수급', '패턴신호'];
  const activeDimsInfo = dims.map((v, i) => ({ v, name: dimNameMap[i], active: i !== 3 || hasSupply }));
  const activeDims = activeDimsInfo.filter(x => x.active);
  const activeItemCount = activeDims.length;
  const avg  = Math.round(activeDims.reduce((s, x) => s + x.v, 0) / activeItemCount);

  // 분산(표준편차) — 항목 간 불균형 감지
  const dimVariance = Math.round(Math.sqrt(activeDims.reduce((s, x) => s + (x.v - avg) ** 2, 0) / activeItemCount));

  const _gm = (() => {
    if      (avg >= 82) return { grade:'A',  color:'#3fb950', text:'최우수',   };
    else if (avg >= 72) return { grade:'A-', color:'#3fb950', text:'우수',     };
    else if (avg >= 62) return { grade:'B',  color:'#58a6ff', text:'양호',     };
    else if (avg >= 52) return { grade:'B-', color:'#d29922', text:'보통 이상', };
    else if (avg >= 42) return { grade:'C',  color:'#d29922', text:'보통',     };
    else if (avg >= 30) return { grade:'C-', color:'#f97316', text:'주의',     };
    else if (avg >= 15) return { grade:'D',  color:'#f85149', text:'위험',     };
    else                return { grade:'D-', color:'#f85149', text:'매우 위험', };
  })();
  const grade      = _gm.grade;
  const gradeColor = _gm.color;
  const gradeText  = _gm.text;

  // 등급 색상 → 어두운 배경 매핑 (rec-badge 배경에 사용)
  const _gradeBgMap = {'#3fb950':'#0d2d1a','#58a6ff':'#0d1b33','#d29922':'#2d2200','#f97316':'#2d1500','#f85149':'#2d1515'};
  const gradeBg = _gradeBgMap[gradeColor] || '#21262d';

  // 하이픈(-) 포함 등급의 baseline 보정용 HTML
  const gradeHtml = grade.includes('-')
    ? `${grade[0]}<span class="grade-hyphen">-</span>`
    : grade;

  // ── 등급별 행동 지침 (8단계) — 배지 좌측 레이블로 사용 ──────────────────
  const _grActionMap = {
    'A' : '적극 매수',
    'A-': '매수 우위',
    'B' : '단계적 매수',
    'B-': '신중 접근',
    'C' : '관망 유지',
    'C-': '보유 재검토',
    'D' : '매도 고려',
    'D-': '즉시 점검',
  };
  // 초기 배지: flow 계산 완료 전 등급 기반 레이블만 표시
  // (flow 함수에서 종합 점수·신뢰도 포함한 최종 텍스트로 덮어씀)
  const badgeInitText = `${_grActionMap[grade] || gradeText} · ${gradeText}`;

  // 동적 설명문 — 강세/약세 항목 이름 명시
  const strongItems = activeDimsInfo.filter(x => x.active && x.v >= 72).map(x => x.name);
  const weakItems   = activeDimsInfo.filter(x => x.active && x.v <  40).map(x => x.name);
  const varNote = '';
  const gradeDesc = (() => {
    if (weakItems.length === 0 && strongItems.length >= 3)
      return `${strongItems.join(' · ')} 등 ${strongItems.length}개 지표 강세 — 추세가 명확합니다.${varNote}`;
    if (weakItems.length === 0 && strongItems.length >= 1)
      return `전 지표가 안정 구간입니다. ${strongItems.join(' · ')} 중심으로 긍정적.${varNote}`;
    if (weakItems.length === 0)
      return `전 지표가 안정 구간에 있습니다. ${activeItemCount}개 항목 모두 경고 없음.${varNote}`;
    if (weakItems.length === 1)
      return `[${weakItems[0]}] 주의 — 나머지 ${activeItemCount - 1}개 지표는 양호합니다.${varNote}`;
    if (weakItems.length === 2)
      return `[${weakItems.join(' · ')}] 두 지표에서 경고 신호가 감지됩니다.${varNote}`;
    return `[${weakItems.join(' · ')}] 등 ${weakItems.length}개 지표에서 경고 신호가 감지됩니다.${varNote}`;
  })();

  // ── 각 차원 설명 텍스트 ────────────────────────────────────────
  const techDesc   = `종합 기술점수 ${score}점 · ${score >= 65 ? '매수 우위' : score >= 40 ? '중립' : '매도 우위'}`;
  const rsiLabel   = rsi > 70 ? `RSI ${rsi.toFixed(0)} 과매수 — 조정 주의`
                   : rsi < 30 ? `RSI ${rsi.toFixed(0)} 과매도 — 반등 기대`
                   :            `RSI ${rsi.toFixed(0)} 안정 구간`;
  const volDesc    = bp ? `ATR ${bp.atr_pct}% · ${
    bp.vol_trend === 'expanding'   ? '변동성 확대 진행중' :
    bp.vol_trend === 'contracting' ? '변동성 수축 (안정화)' : '변동성 안정'}` : '변동성 데이터 없음';
  const supplyDesc = isKrx ? calcSupplyDesc(flow) : '수급 데이터 없음';
  const patDesc    = patterns.length === 0 ? '특이 캔들 패턴 없음'
    : `상승패턴 ${bullPat}개 · 하락패턴 ${bearPat}개 감지`;

  // ── 단계별 분석 스텝 HTML 빌더 ─────────────────────────────────────────────
  const allSteps    = d.analysis_steps || [];
  const patCardHtml = patterns.map(p => {
    const pcls = p.direction === '상승' ? 'pattern-bull' : p.direction === '하락' ? 'pattern-bear' : 'pattern-neu';
    const icon  = p.direction === '상승' ? '📈' : p.direction === '하락' ? '📉' : '➖';
    return `<div class="pattern-item ${pcls}">
      <div class="pattern-head"><span class="pattern-icon">${icon}</span><span>${p.name}</span></div>
      <div class="pattern-desc">${p.desc}</div>
    </div>`;
  }).join('') || '<p class="empty-note">특이한 캔들 패턴이 감지되지 않았습니다.</p>';

  const buildStepHtml = (steps) => {
    if (!steps || !steps.length) return '<p style="font-size:12px;color:#484f58;padding:6px 0">해당 분석 데이터가 없습니다.</p>';
    return steps.map(st => {
      const isS5    = st.step.startsWith('5.');
      const title   = st.step.replace(/^\d+\.\s*/, '');
      return `<div class="step-item">
        <div class="step-header">
          <span class="step-title">${title}</span>
        </div>
        ${isS5
          ? `<div class="step-patterns">${patCardHtml}</div>`
          : `<div class="step-result">${st.result.split(' | ').filter(l => l.trim()).map(line => `<span class="step-result-line">${line}</span>`).join('')}</div>`
        }
      </div>`;
    }).join('');
  };

  // 차원별 단계 그룹핑: 1→기술추세 / 2→모멘텀 / 3→변동성 / 4→거래량(독립) / 5·6→패턴·신호
  const stepTech   = allSteps.filter(st => st.step.startsWith('1.'));
  const stepMom    = allSteps.filter(st => st.step.startsWith('2.'));
  const stepVol    = allSteps.filter(st => st.step.startsWith('3.'));
  const stepVolume = allSteps.filter(st => st.step.startsWith('4.'));
  const stepPat    = allSteps.filter(st => !st.step.match(/^[1234]\./));

  // ── dim 점수 → 색상·레이블 헬퍼 (5단계) ────────────────────────────
  // 75+:우수(초록) / 55+:양호(파랑) / 40+:보통(노랑) / 25+:주의(주황) / 0+:위험(빨강)
  const _dg = v => v >= 75 ? { c:'#3fb950', lbl:'우수'  }
                 : v >= 55 ? { c:'#58a6ff', lbl:'양호'  }
                 : v >= 40 ? { c:'#d29922', lbl:'보통'  }
                 : v >= 25 ? { c:'#f97316', lbl:'주의'  }
                 :           { c:'#f85149', lbl:'위험'  };

  // ── 렌더 헬퍼 ─────────────────────────────────────────────────────
  const dimBar = (emoji, label, val, desc, opts = {}) => {
    const { c, lbl } = _dg(val);
    if (opts.accordionId) {
      const aId = opts.accordionId;
      return `<div class="diag-dim diag-dim-clickable" onclick="toggleDimAccordion('${aId}')">
        <div class="diag-dim-head">
          <span class="diag-dim-label">${emoji} ${label} <span id="arrow-${aId}" style="font-size:11px;color:#8b949e;display:inline-block;transition:transform .25s">▼</span></span>
          <span class="diag-dim-score" style="color:${c}">${val}점 · ${lbl}</span>
        </div>
        <div class="diag-bar-bg"><div class="diag-bar-fill" style="width:${val}%;background:${c}"></div></div>
        <div class="diag-dim-desc">${desc}</div>
      </div>
      <div id="${aId}" style="display:none;padding:12px;background:#0d1117;border-radius:10px;border:1px solid #30363d;margin-top:-2px">
        <div style="display:flex;flex-direction:column;gap:8px">${opts.accordionContent || ''}</div>
      </div>`;
    }
    if (opts.clickable) {
      return `<div class="diag-dim diag-dim-clickable" onclick="toggleInvestorFlowAccordion()">
        <div class="diag-dim-head">
          <span class="diag-dim-label">${emoji} ${label} <span id="investor-flow-arrow" style="font-size:11px;color:#8b949e;display:inline-block;transition:transform .25s">▼</span></span>
          <span class="diag-dim-score" style="color:${c}">${val}점 · ${lbl}</span>
        </div>
        <div class="diag-bar-bg"><div class="diag-bar-fill" style="width:${val}%;background:${c}"></div></div>
        <div class="diag-dim-desc">${desc}</div>
      </div>`;
    }
    return `<div class="diag-dim">
      <div class="diag-dim-head">
        <span class="diag-dim-label">${emoji} ${label}</span>
        <span class="diag-dim-score" style="color:${c}">${val}점 · ${lbl}</span>
      </div>
      <div class="diag-bar-bg"><div class="diag-bar-fill" style="width:${val}%;background:${c}"></div></div>
      <div class="diag-dim-desc">${desc}</div>
    </div>`;
  };

  diagEl.innerHTML = `
    <div class="diag-grade-row">
      <div class="diag-grade-badge" style="border-color:${gradeColor};color:${gradeColor}">${gradeHtml}</div>
      <div class="diag-grade-info" style="flex:1">
        <div class="diag-grade-title" style="color:${gradeColor}">${gradeText} <span style="color:#484f58;font-size:11px;font-weight:400">· ${activeItemCount}항목 평균 ${avg}점</span></div>
        <div class="diag-grade-sub">${gradeDesc}</div>
      </div>
      <span id="flow-rec-badge" class="rec-badge-lg" style="flex-shrink:0;color:${gradeColor};border:1px solid ${gradeColor};background:${gradeBg}" data-grade="${grade}" data-grade-color="${gradeColor}" data-grade-bg="${gradeBg}" data-badge-text="${badgeInitText}">${badgeInitText}</span>
    </div>
    <div id="flow-rationale" style="display:none"></div>
    <div class="diag-dims">
      ${pbTopHtml}
      ${dimBar('📊', '기술적 추세',   techScore,    techDesc,  {accordionId:'dim-tech', accordionContent: buildStepHtml(stepTech)})}
      ${dimBar('⚡', '모멘텀 강도',   momentumScore, rsiLabel, {accordionId:'dim-mom',  accordionContent: buildStepHtml(stepMom)})}
      ${dimBar('🌊', '변동성 수준',   volScore,      volDesc,   {accordionId:'dim-vol',  accordionContent: buildStepHtml(stepVol)})}
      ${isKrx ? dimBar('💰', '수급 흐름', supplyScore, supplyDesc, {clickable: true}) : ''}
      ${isKrx ? `<div id="investor-flow-accordion" style="display:none;padding:12px;background:#0d1117;border-radius:10px;border:1px solid #30363d;margin-top:-2px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <span style="font-size:12px;color:#8b949e;font-weight:600">💰 투자자 수급 <span style="font-size:10px;color:#484f58;font-weight:400">· 토스증권 기준</span></span>
          <button id="investor-flow-retry" onclick="retryInvestorFlow()" style="display:none;background:none;border:1px solid #30363d;border-radius:6px;padding:3px 8px;color:#8b949e;font-size:11px;cursor:pointer">🔄 재시도</button>
        </div>
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
      </div>` : ''}
      ${dimBar('🕯️', '캔들·차트 패턴 신호', patScore, patDesc, {accordionId:'dim-pat', accordionContent: buildStepHtml(stepPat)})}
      ${renderHybridSection(d)}
      ${pbBottomHtml}
    </div>
    <div style="font-size:11px;color:#484f58;margin-top:12px;padding-top:10px;border-top:1px solid #21262d">
      ⚠️ 본 진단은 기술적 지표 기반 참고 자료이며 투자 판단의 단독 근거로 사용하지 마세요. 각 항목을 클릭하면 단계별 상세 분석을 확인할 수 있습니다.
    </div>`;
}

// ── 투자자 수급 렌더 (KRX 전용) ────────────────────────────────────────────
function renderInvestorFlow(d, isKrx) {
  const badge    = document.getElementById('investor-badge');
  const retryBtn = document.getElementById('investor-flow-retry');
  const skelEl   = document.getElementById('investor-flow-skeleton');
  const contEl   = document.getElementById('investor-flow-content');

  // US 종목 → 배지만 제거 (수급 흐름 행 자체가 KRX 전용이므로 아코디언 없음)
  if (!isKrx) { if (badge) badge.classList.remove('visible'); return; }

  const flow = d.investor_flow;

  // ok=false: 비동기 자동 로드 진행 중 — 아코디언 내부에 스켈레톤 표시
  if (!flow || !flow.ok) {
    if (skelEl) skelEl.style.display = '';
    if (contEl) contEl.innerHTML = '';
    if (retryBtn) retryBtn.style.display = 'none';
    if (badge) badge.classList.remove('visible');
    return;
  }

  if (skelEl) skelEl.style.display = 'none';
  if (retryBtn) retryBtn.style.display = 'none';

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

// ── 투자자 수급 비동기 로드 (전용 경량 엔드포인트 사용) ─────────────────────
// 메인 /api/stock 호출과 완전히 분리 — Toss API 타임아웃 경합 문제 근본 해결
async function loadInvestorFlowAsync(symbol) {
  const skelEl   = document.getElementById('investor-flow-skeleton');
  const contEl   = document.getElementById('investor-flow-content');
  const retryBtn = document.getElementById('investor-flow-retry');

  console.log(`[투자자수급] 로드 시작 → ticker: ${symbol}`);

  // 로딩 상태: 스켈레톤 표시
  if (skelEl)   skelEl.style.display = '';
  if (contEl)   contEl.innerHTML = '';
  if (retryBtn) retryBtn.style.display = 'none';

  try {
    const url = `/api/investor-flow?ticker=${encodeURIComponent(symbol)}`;
    console.log(`[투자자수급] API 요청: ${url}`);
    const r  = await fetch(url);

    if (!r.ok) throw new Error(`HTTP ${r.status}`);

    const nd = await r.json();
    console.log(`[투자자수급] 응답 수신:`, nd);

    // 응답 도착 전 다른 종목으로 전환된 경우 무시 (race condition 방지)
    if (!currentData || currentData.symbol !== symbol) {
      console.log(`[투자자수급] 종목 전환 감지 — 렌더 취소 (${symbol} → ${currentData && currentData.symbol})`);
      return;
    }

    if (nd && nd.ok) {
      // 성공 상태
      console.log(`[투자자수급] 성공 — 데이터 렌더링`);
      currentData.investor_flow = nd;
      renderInvestorFlow(currentData, true);
    } else {
      // API 실패(ok:false) — 재시도 버튼 표시
      const reason = (nd && nd.reason) ? nd.reason : '수급 데이터 없음';
      console.warn(`[투자자수급] ok:false — ${reason}`);
      if (contEl) contEl.innerHTML = `
        <div style="text-align:center;padding:20px 0;color:#484f58;font-size:13px">
          <div style="font-size:22px;margin-bottom:8px">📡</div>
          ${reason}
          <div style="font-size:11px;margin-top:6px;color:#30363d">장 개시 전이거나 API 일시 불가 상태입니다</div>
        </div>`;
      if (retryBtn) retryBtn.style.display = '';
    }
  } catch(e) {
    // 네트워크 / 파싱 오류
    console.error(`[투자자수급] 오류:`, e);
    if (!currentData || currentData.symbol !== symbol) return;
    if (contEl) contEl.innerHTML = `
      <div style="text-align:center;padding:20px 0;color:#484f58;font-size:13px">
        <div style="font-size:22px;margin-bottom:8px">📡</div>
        수급 데이터 조회 실패
        <div style="font-size:11px;margin-top:6px;color:#30363d">네트워크 오류 — 잠시 후 재시도해주세요</div>
      </div>`;
    if (retryBtn) retryBtn.style.display = '';
  } finally {
    if (skelEl) skelEl.style.display = 'none';
  }
}

// ── 투자자 수급 재시도 (전용 엔드포인트로 위임) ───────────────────────────
async function retryInvestorFlow() {
  if (!currentData || currentData.market !== 'KRX') return;
  await loadInvestorFlowAsync(currentData.symbol);
}

function toggleInvestorFlowAccordion() {
  const acc   = document.getElementById('investor-flow-accordion');
  const arrow = document.getElementById('investor-flow-arrow');
  if (!acc) return;
  const nowHidden = acc.style.display === 'none';
  acc.style.display = nowHidden ? '' : 'none';
  if (arrow) arrow.style.transform = nowHidden ? 'rotate(180deg)' : 'rotate(0deg)';
}

function toggleDimAccordion(id) {
  const acc   = document.getElementById(id);
  const arrow = document.getElementById('arrow-' + id);
  if (!acc) return;
  const nowHidden = acc.style.display === 'none';
  acc.style.display = nowHidden ? '' : 'none';
  if (arrow) arrow.style.transform = nowHidden ? 'rotate(180deg)' : 'rotate(0deg)';
}

// ── 리스크 관리 카드 하단: 눌림목 기반 ATR 정밀 가격 설정 ────────────────────
// [DEPRECATED] "📌 눌림목 분석 기반 정밀 가격 설정" 섹션 렌더러.
//   정밀 가격(손절선·1·2차 목표·트레일링 스탑)은 renderForecast의 리스크 시나리오
//   카드(보수적/중립적/공격적)에 통합되어 더 이상 호출되지 않는다. (참고용 보존)
function renderPullbackATR(d, isKrx) {
  const el = document.getElementById('pullback-atr-section');
  if (!el) return;
  const pa = d.pullback_analysis;
  if (!pa || !pa.stop_loss) { el.innerHTML = ''; return; }

  const C = { red:'#f85149', orange:'#d29922', green:'#3fb950', blue:'#58a6ff', purple:'#bc8cff', gray:'#8b949e' };
  const fmtP = v => fmtPrice(v, isKrx);

  // R/R 색상
  const rrC = pa.rr_main >= 2.3 ? C.green : pa.rr_main >= 1.5 ? C.orange : C.red;
  // 목표가 출처 강조
  const srcIsEnsemble = pa.target_source && pa.target_source.includes('앙상블');
  const srcC = srcIsEnsemble ? C.blue : C.gray;

  el.innerHTML = `
    <div style="margin-top:18px;border-top:1px solid #21262d;padding-top:16px">
      <div style="font-size:11px;font-weight:700;color:#8b949e;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">
        📌 눌림목 분석 기반 정밀 가격 설정
      </div>
      <div style="font-size:12px;color:#8b949e;line-height:1.6;margin-bottom:12px">
        ATR(14) 변동성과 구조 분석을 결합해 산출한 손절·목표가입니다.
        위 시나리오 카드와 함께 참고하여 실제 매매 기준으로 활용하세요.
      </div>
      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px">
        <div style="background:#0d1117;border-radius:8px;padding:11px 12px;border:1px solid ${C.red}">
          <div style="font-size:10px;color:#8b949e;margin-bottom:5px;text-transform:uppercase;letter-spacing:.05em">손절선</div>
          <div style="font-size:17px;font-weight:800;color:${C.red}">${fmtP(pa.stop_loss)}</div>
          <div style="font-size:11px;color:${C.orange};margin-top:4px">${pa.stop_loss_pct}% 손실 — 이탈 시 미련 없이 정리</div>
        </div>
        <div style="background:#0d1117;border-radius:8px;padding:11px 12px;border:1px solid ${C.green}">
          <div style="font-size:10px;color:#8b949e;margin-bottom:5px;text-transform:uppercase;letter-spacing:.05em">1차 목표가</div>
          <div style="font-size:17px;font-weight:800;color:${C.green}">${fmtP(pa.target_main)}</div>
          <div style="font-size:11px;color:${rrC};margin-top:4px">R/R ${pa.rr_main}:1</div>
          <div style="font-size:10px;color:${srcC};margin-top:2px">${pa.target_source || '기술적 분석'}</div>
        </div>
        <div style="background:#0d1117;border-radius:8px;padding:11px 12px;border:1px solid #30363d">
          <div style="font-size:10px;color:#8b949e;margin-bottom:5px;text-transform:uppercase;letter-spacing:.05em">2차 목표 (돌파 후)</div>
          <div style="font-size:17px;font-weight:800;color:${C.blue}">${fmtP(pa.target_ext)}</div>
          <div style="font-size:11px;color:#8b949e;margin-top:4px">1차 돌파 확인 후 홀딩 기준</div>
        </div>
        <div style="background:#0d1117;border-radius:8px;padding:11px 12px;border:1px solid #30363d">
          <div style="font-size:10px;color:#8b949e;margin-bottom:5px;text-transform:uppercase;letter-spacing:.05em">트레일링 스탑</div>
          <div style="font-size:17px;font-weight:800;color:${C.purple}">${fmtP(pa.trail_stop)}</div>
          <div style="font-size:11px;color:#8b949e;margin-top:4px">ATR × 1.5 — 수익 보전 기준</div>
        </div>
      </div>
    </div>`;
}

function renderForecast(d, isKrx) {
  const risk = d.risk_scenarios;
  const bp   = d.buy_price;
  const tp   = d.target_price;
  const ai   = d.ai_strategy;

  // ── AI 종합 진단 및 트레이딩 전략 섹션 ──
  const aiEl = document.getElementById('ai-strategy-section');
  if (aiEl && ai) {
    const hiddenAiStrategyPatterns = [
      /^\[시장 상태\]/,
      /^⚠️\s*\[경고\]\s*부채비율/,
      /^\[투자자 수급\]/,
    ];
    const visibleAiStrategyLines = (ai.result || '')
      .split(' | ')
      .map(line => line.trim())
      .filter(line => line && !hiddenAiStrategyPatterns.some(pattern => pattern.test(line)));

    aiEl.innerHTML = `
      <div style="background: rgba(31, 111, 235, 0.05); border-radius:10px; padding:16px; margin-bottom:16px; border: 1px solid #1f6feb;">
        <div style="color:#e6edf3; font-size: 14px; line-height: 1.6;">
          ${visibleAiStrategyLines.map(line => {
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
      const sn  = (!isKrx && d.session_name && !['정규장','장마감'].includes(d.session_name))
                  ? ` <span style="font-size:10px;padding:1px 5px;border-radius:3px;background:#6e40c933;color:#bc8cff;margin-left:4px">${d.session_name}</span>`
                  : '';
      const probColor  = (tp.reach_probability || 50) >= 65 ? '#3fb950' : (tp.reach_probability || 50) >= 45 ? '#d29922' : '#f85149';
      const riskColors = { '낮음':'#3fb950', '중간':'#d29922', '높음':'#f85149' };
      const riskC      = riskColors[tp.risk_level] || '#d29922';
      const failHtml   = (tp.failure_factors || [])
        .map(f => `<div style="display:flex;align-items:flex-start;gap:6px;margin-bottom:4px"><span style="color:#f85149;flex-shrink:0">•</span><span>${f}</span></div>`)
        .join('');
      tpEl.innerHTML = `
        <div style="background:#21262d;border-radius:10px;padding:16px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
          <div>
            <div style="font-size:12px;color:#8b949e;margin-bottom:6px">예상 목표가 범위 (${tp.period})</div>
            <div style="font-size:24px;font-weight:800;color:#3fb950">${fmt(tp.min_price, isKrx)} ~ ${fmt(tp.max_price, isKrx)}</div>
            <div style="font-size:13px;color:#8b949e;margin-top:4px">
              현재가${sn} <b data-tp-cur style="color:#e6edf3">${fmt(cur, isKrx)}</b> 기준 예상 수익률:
              <span data-tp-return style="color:#3fb950">+${tp.min_return}% ~ +${tp.max_return}%</span>
            </div>
          </div>
          <div style="display:flex;flex-direction:column;gap:8px;align-items:flex-end">
            <div style="text-align:center;background:#0d1117;border-radius:10px;padding:8px 14px;border:1px solid ${probColor}44">
              <div style="font-size:10px;color:#8b949e;margin-bottom:2px">목표가 도달 확률</div>
              <div style="font-size:20px;font-weight:800;color:${probColor}">${tp.reach_probability || '—'}%</div>
            </div>
            ${tp.expected_trading_days ? `
            <div style="text-align:center;background:#0d1117;border-radius:8px;padding:6px 12px">
              <div style="font-size:10px;color:#8b949e;margin-bottom:1px">예상 소요 기간</div>
              <div style="font-size:13px;font-weight:700;color:#58a6ff">거래일 기준 ${tp.expected_trading_days[0]}~${tp.expected_trading_days[1]}일</div>
            </div>` : ''}
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">
          <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px">
            <div style="font-size:11px;color:#8b949e;margin-bottom:6px">📋 예측 근거</div>
            <div style="font-size:12px;color:#e6edf3;line-height:1.5">${tp.reason}</div>
          </div>
          <div style="background:#161b22;border:1px solid ${riskC}33;border-radius:10px;padding:12px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
              <div style="font-size:11px;color:#8b949e">⚠️ 실패 가능성</div>
              <div style="font-size:11px;font-weight:700;color:${riskC};background:${riskC}22;border-radius:4px;padding:1px 7px">리스크 ${tp.risk_level || '—'}</div>
            </div>
            <div style="font-size:11px;color:#8b949e;margin-bottom:6px">${tp.risk_reason || ''}</div>
            <div style="font-size:11px;color:#cdd9e5;line-height:1.5">${failHtml}</div>
          </div>
        </div>
      `;
      _lastTp    = tp;
      _lastIsKrx = isKrx;
    }
  }

  // ── 매수 전략 섹션 ──
  const bpEl = document.getElementById('buy-price-section');
  if (bpEl) {
    if (!bp) {
      bpEl.innerHTML = '<p style="color:#484f58;font-size:13px">데이터 부족</p>';
    } else {
      const sr = bp.strategy_rec || {};
      const cur = bp.current;
      const fmtPct = p => (p >= 0 ? `<span style="color:#3fb950">+${p}%</span>` : `<span style="color:#f85149">${p}%</span>`);

      // ── 전략 추천 배너 ──
      const ctxColorMap = {
        overbought:     ['#f85149','#2d1515'], strong_uptrend: ['#3fb950','#0d2d1a'],
        uptrend:        ['#3fb950','#0d2d1a'], recovery:       ['#d29922','#2d2200'],
        downtrend:      ['#f85149','#2d1515'], sideways:       ['#58a6ff','#0d1b33'],
      };
      const [bannerC, bannerBg] = ctxColorMap[sr.context] || ['#d29922','#2d2200'];
      const confColor = (sr.confidence_pct || 50) >= 70 ? '#3fb950' : (sr.confidence_pct || 50) >= 50 ? '#d29922' : '#f85149';
      const ratHtml   = (sr.rationale || []).map(r => `<div style="display:flex;align-items:flex-start;gap:6px;margin-bottom:3px"><span style="color:${bannerC};flex-shrink:0">•</span><span>${r}</span></div>`).join('');
      const stratBanner = sr.action ? `
        <div style="background:${bannerBg};border:1px solid ${bannerC}55;border-radius:10px;padding:14px;margin-bottom:14px">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px">
            <div style="font-size:15px;font-weight:800;color:${bannerC}">${sr.action}</div>
            <div style="text-align:right">
              <div style="font-size:10px;color:#8b949e">신뢰도</div>
              <div style="font-size:18px;font-weight:800;color:${confColor}">${sr.confidence_pct}%</div>
            </div>
          </div>
          <div style="font-size:12px;color:#cdd9e5;line-height:1.6">${ratHtml}</div>
          ${sr.chase_zone && sr.chase_zone.reason ? `
          <div style="margin-top:10px;padding:8px 10px;background:#2d0d0d;border-left:3px solid #f85149;border-radius:0 6px 6px 0;font-size:11px;color:#f85149">
            ⛔ 추격 매수 위험: ${sr.chase_zone.reason}
          </div>` : ''}
        </div>` : '';

      // ── 추천 밴드 (active_bands 기준으로 필터링) ──
      const activeBands = sr.active_bands || ['A','B','C'];
      const bandColor   = ['#f97316','#d29922','#3fb950'];

      // ── "🔗 연계 밴드"(피보나치 연동) · 핵심 구간(저항대/방어) · 분할 매수 단계(1~4차)
      //    설명은 모두 매수 구간 카드에서 폐지됨 → 카드에는 밴드 가격대/근거만 표시한다.

      const renderBandCard = (b, i, isRec) => {
        const bc = bandColor[i] || '#58a6ff';
        const isActive = activeBands.includes(b.band);
        const isPriority = b.band === sr.priority_band;
        const dimStyle = isActive ? '' : 'opacity:0.4;';
        const priTag   = isPriority ? `<span style="font-size:9px;background:${bc}33;color:${bc};border:1px solid ${bc};border-radius:3px;padding:1px 5px;margin-left:4px">권장</span>` : '';
        if (isRec) {
          return `<div style="background:#0d1117;border-radius:8px;padding:10px 12px;box-sizing:border-box;${dimStyle}border:1px solid ${isPriority ? bc+'55' : '#21262d'}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
              <span style="font-size:12px;font-weight:700;color:${bc}">밴드 ${b.band}${priTag}</span>
              <span style="font-size:10px;color:#8b949e;background:#161b22;border-radius:4px;padding:2px 6px">${fmtPct(b.pct[0])} ~ ${fmtPct(b.pct[1])}</span>
            </div>
            <div style="font-size:14px;font-weight:800;color:${bc};margin-bottom:5px">${fmt(b.range[0], isKrx)} ~ ${fmt(b.range[1], isKrx)}</div>
            <div style="font-size:10px;color:#8b949e;margin-bottom:2px">• ${b.basis}</div>
            <div style="font-size:10px;color:#3fb950">→ ${b.hold_note}</div>
          </div>`;
        } else {
          return `<div style="background:#0d1117;border-radius:8px;padding:10px 12px;box-sizing:border-box;${dimStyle}border:1px solid ${isPriority ? bc+'55' : '#21262d'}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
              <span style="font-size:12px;font-weight:700;color:${bc}">밴드 ${b.band}${priTag}</span>
              <span style="font-size:10px;color:#8b949e;background:#161b22;border-radius:4px;padding:2px 6px">${fmtPct(b.pct[0])} ~ ${fmtPct(b.pct[1])}</span>
            </div>
            <div style="font-size:14px;font-weight:800;color:${bc};margin-bottom:5px">${fmt(b.range[0], isKrx)} ~ ${fmt(b.range[1], isKrx)}</div>
            <div style="font-size:10px;color:#8b949e">• ${b.atr_basis}</div>
            <div style="font-size:10px;color:#8b949e">• ${b.tech_note}</div>
          </div>`;
        }
      };

      const recBandsHtml = (bp.recommended_bands && bp.recommended_bands.length)
        ? `<div class="buy-card recommended" style="padding:12px 14px">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap;gap:4px">
              <div class="buy-label" style="margin-bottom:0;font-size:13px">📍 2차 매수 구간</div>
              <div style="font-size:10px;color:#484f58">※ 지지선·이평선·VWAP 앵커 기반</div>
            </div>
            <div class="buy-bands-row">${bp.recommended_bands.map((b, i) => renderBandCard(b, i, true)).join('')}</div>
          </div>` : '';

      const aggBandsHtml = (bp.aggressive_bands && bp.aggressive_bands.length)
        ? `<div class="buy-card aggressive" style="padding:12px 14px">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap;gap:4px">
              <div class="buy-label" style="margin-bottom:0;font-size:13px">⚡ 1차 매수 구간 (ATR 기반)</div>
              <div style="font-size:10px;color:#484f58">※ 백테스트(1년·${bp.market||'KRX'}) 기저확률 + 추세·RSI 보정</div>
            </div>
            <div class="buy-bands-row">${bp.aggressive_bands.map((b, i) => renderBandCard(b, i, false)).join('')}</div>
          </div>` : '';

      bpEl.innerHTML = stratBanner + `<div class="buy-price-grid">${aggBandsHtml}${recBandsHtml}</div>`;
    }
  }

  // ── 리스크 카드 ──
  const rgEl = document.getElementById('risk-grid');
  if (rgEl && risk) {
    const riskEntries = ['conservative', 'balanced', 'aggressive'].map(k => risk[k]).filter(Boolean);
    const rrColor = rr => rr >= 2.0 ? '#3fb950' : rr >= 1.5 ? '#d29922' : '#f85149';
    // 📌 눌림목 분석 기반 정밀 가격 — 시나리오 카드에 통합 (별도 섹션 폐지)
    // 고대비 구분선 — 카드 배경(녹/적 틴트·다크·라이트)에 무관하게 항상 보이도록
    //   진한 검정 선 + 안쪽 미세 하이라이트(engraved)로 대비 확보. 4개 영역 공통 사용.
    const DIVIDER = 'border-top:2px solid rgba(0,0,0,0.85);box-shadow:inset 0 2px 0 rgba(255,255,255,0.06)';
    // 📌 공통 손절선 — 카드별 반복 출력 폐지, 3개 시나리오 카드 하단에 1회만 출력.
    //    카드 영역 전체 폭으로 표시 (grid-column:1/-1). 정밀 손절선(pullback) 우선, 없으면 중립 시나리오 stop.
    const _stopPa  = d.pullback_analysis || null;
    const _stopBal = risk.balanced || risk.conservative || riskEntries[0] || null;
    let _stopVal = null, _stopPct = null;
    if (_stopPa && _stopPa.stop_loss != null) {
      _stopVal = _stopPa.stop_loss; _stopPct = _stopPa.stop_loss_pct;
    } else if (_stopBal && _stopBal.stop) {
      _stopVal = _stopBal.stop[0]; _stopPct = _stopBal.stop_pct;
    }
    const commonStopHtml = (_stopVal == null) ? '' : `
      <div style="grid-column:1 / -1;background:#0d1117;border-radius:8px;padding:11px 14px;border:1px solid #f85149">
        <div style="font-size:10px;color:#8b949e;margin-bottom:5px;text-transform:uppercase;letter-spacing:.05em">손절선</div>
        <div style="font-size:17px;font-weight:800;color:#f85149">${fmt(_stopVal, isKrx)}</div>
        <div style="font-size:11px;color:#d29922;margin-top:4px">${_stopPct != null ? _stopPct + '% 손실 — ' : ''}이탈 시 미련 없이 정리</div>
      </div>`;
    rgEl.innerHTML = `
      ${riskEntries.map(sc => {
        const failHtml = (sc.failure_conditions || [])
          .map(f => `<div style="display:flex;align-items:flex-start;gap:5px;margin-bottom:2px"><span style="color:#f97316;flex-shrink:0">•</span><span>${f}</span></div>`)
          .join('');
        // ── 눌림목 정밀 목표가 — 1차(중립적) · 2차(공격적)만 복원, 손절/트레일링은 제외 ──
        const pbHtml = (() => {
          const pa = d.pullback_analysis || null;
          if (!pa) return '';
          const TOL = 0.0025;
          const tpPrices = (sc.tp_levels || []).map(t => t && t.price).filter(v => v != null);
          const tpRel = price => {
            if (price == null || !tpPrices.length) return { type: 'none' };
            for (let i = 0; i < tpPrices.length; i++) {
              if (tpPrices[i] && Math.abs(price - tpPrices[i]) / tpPrices[i] <= TOL) return { type: 'eq', n: i + 1 };
            }
            const lo = Math.min(...tpPrices), hi = Math.max(...tpPrices);
            if (price > lo && price < hi) {
              const s = tpPrices.map((p, i) => ({ p, n: i + 1 })).sort((a, b) => a.p - b.p);
              for (let i = 0; i < s.length - 1; i++) {
                if (price >= s[i].p && price <= s[i + 1].p) return { type: 'between', a: s[i].n, b: s[i + 1].n };
              }
            }
            return { type: 'out' };
          };
          const targetItem = (price, baseLabel, color, sub) => {
            const r = tpRel(price);
            if (r.type === 'eq')      return { label: baseLabel, rel: `TP${r.n} 연계` };
            if (r.type === 'between') return { label: baseLabel, rel: `TP${r.a}~TP${r.b} 구간` };
            return { label: baseLabel, value: price, color, sub };
          };
          let items = [];
          if (sc.label === '중립적' && pa.target_main != null) {
            const sub = [pa.rr_main != null ? `R/R ${pa.rr_main}:1` : '', pa.target_source || ''].filter(Boolean).join(' · ');
            items.push(targetItem(pa.target_main, '1차 정밀 목표가', '#3fb950', sub));
          } else if (sc.label === '공격적' && pa.target_ext != null) {
            items.push(targetItem(pa.target_ext, '2차 목표 (돌파 후)', '#58a6ff', '1차 돌파 확인 후 홀딩 기준'));
          }
          if (!items.length) return '';
          const rows = items.map(it => {
            if (it.rel) {
              return `<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:5px">
                <span style="font-size:11px;color:#cdd9e5">${it.label}</span>
                <span style="font-size:11px;font-weight:700;color:#58a6ff;background:#58a6ff1a;border-radius:4px;padding:1px 7px">${it.rel}</span>
              </div>`;
            }
            return `<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px">
                <span style="font-size:11px;color:#cdd9e5">${it.label}</span>
                <span style="font-size:13px;font-weight:700;color:${it.color}">${fmt(it.value, isKrx)}</span>
              </div>${it.sub ? `<div style="font-size:10px;color:#8b949e;margin-top:1px;margin-bottom:5px">${it.sub}</div>` : '<div style="margin-bottom:4px"></div>'}`;
          }).join('');
          return `<div style="margin-top:8px;padding-top:8px;${DIVIDER}">
            <div style="font-size:10px;color:#8b949e;margin-bottom:5px">📌 눌림목 정밀가</div>
            ${rows}
          </div>`;
        })();
        return `
        <div class="risk-card ${sc.label === '보수적' ? 'conservative' : sc.label === '중립적' ? 'balanced' : 'aggressive'}">
          <div class="risk-icon">${sc.icon}</div>
          <div class="risk-name">${sc.label}</div>
          <div class="risk-desc" style="font-size:11px;color:#8b949e;margin-bottom:8px">${sc.desc}</div>
          <div class="risk-row" style="margin-bottom:6px">
            <span class="risk-lbl">🎯 목표가</span>
            <span class="risk-tgt" style="font-size:12px">${fmt(sc.target[0], isKrx)} ~ ${fmt(sc.target[1], isKrx)}</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;padding-top:8px;${DIVIDER}">
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
          ${pbHtml}
          <div style="font-size:10px;color:#8b949e;margin-top:6px;line-height:1.5">💡 ${sc.interpretation || ''}</div>
          ${sc.tp_levels && sc.tp_levels.length ? `
          <div style="margin-top:8px;padding-top:8px;${DIVIDER}">
            <div style="font-size:10px;color:#8b949e;margin-bottom:5px">📊 목표가 레벨별 도달 확률</div>
            ${sc.tp_levels.map((lv, i) => {
              const tpC = lv.prob_pct >= 65 ? '#3fb950' : lv.prob_pct >= 45 ? '#d29922' : '#f97316';
              return `<div style="display:flex;justify-content:space-between;align-items:center;background:#0d1117;border-radius:5px;padding:4px 8px;margin-bottom:3px">
                <span style="font-size:10px;font-weight:700;color:${tpC}">TP${i+1}</span>
                <span style="font-size:10px;color:#cdd9e5;font-weight:600">${fmt(lv.price, isKrx)}</span>
                <span style="font-size:10px;color:#3fb950">+${lv.return_pct}%</span>
                <span style="font-size:10px;color:${tpC}">도달 ${lv.prob_pct}%</span>
                <span style="font-size:10px;color:#8b949e">~${lv.avg_days}일</span>
              </div>`;
            }).join('')}
          </div>` : ''}
          ${failHtml ? `
          <div style="margin-top:8px;padding-top:8px;${DIVIDER}">
            <div style="font-size:10px;color:#f97316;margin-bottom:4px;font-weight:600">⚠️ 이 시나리오가 실패할 수 있는 조건</div>
            <div style="font-size:11px;color:#8b949e;line-height:1.5">${failHtml}</div>
          </div>` : ''}
        </div>`;
      }).join('')}
      ${commonStopHtml}`;
  }
  // 📌 "눌림목 분석 기반 정밀 가격 설정" 섹션은 위 시나리오 카드(보수적/중립적/공격적)에
  //    정밀 가격으로 통합됨 → 별도 섹션 비표시 (잔존 콘텐츠 방지 위해 컨테이너 클리어).
  const _pbSec = document.getElementById('pullback-atr-section');
  if (_pbSec) _pbSec.innerHTML = '';
  // renderPullbackATR(d, isKrx);  // (통합으로 사용 중단 — 함수는 보존하되 호출 안 함)
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

  // 5단계 색상 매핑
  const ws = summary.weighted_score ?? 0;
  const ovClr = ws >= 45  ? '#3fb950'
              : ws >= 15  ? '#57c55a'
              : ws >= -15 ? '#d29922'
              : ws >= -45 ? '#e87a5a'
              :              '#f85149';

  // 가중 점수 진행 바 (−100 ~ +100 → 0~100% 너비, 중앙 50%)
  const barPct  = Math.round((ws + 100) / 2);   // 0~100
  const barClr  = ws >= 0 ? '#3fb950' : '#f85149';
  // 중앙에서 확장되는 방향 계산
  const barLeft  = ws >= 0 ? 50 : barPct;
  const barWidth = Math.abs(ws) / 2;

  let html = `
    <div class="overall-signal-box">
      <div style="flex:1">
        <div class="ovs-label">종합 판단 (가중 점수 기반 · ${summary.total}개 지표)</div>
        <div style="font-size:20px;font-weight:800;color:${ovClr};margin-bottom:8px">${summary.overall_label}</div>
        <!-- 가중 점수 바 -->
        <div style="position:relative;background:#30363d;border-radius:4px;height:6px;margin-bottom:4px">
          <div style="position:absolute;left:${barLeft}%;width:${barWidth}%;height:100%;background:${barClr};border-radius:4px;transition:width .4s"></div>
          <div style="position:absolute;left:50%;top:-3px;width:2px;height:12px;background:#484f58"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#484f58">
          <span>매도 −100</span>
          <span style="color:${ovClr};font-weight:600">점수: ${ws > 0 ? '+' : ''}${ws}</span>
          <span>+100 매수</span>
        </div>
      </div>
      ${summary.market_state ? `<div style="font-size:12px;background:#21262d;border-radius:8px;padding:6px 12px;color:#cdd9e5;white-space:nowrap;align-self:flex-start">${summary.market_state}</div>` : ''}
    </div>
    <!-- 매수/관망/매도 카운트 바 -->
    <div style="display:flex;gap:4px;margin-bottom:12px;font-size:11px;font-weight:600">
      <div style="flex:${summary.buy||0};background:#1b3a1f;color:#3fb950;text-align:center;padding:3px 0;border-radius:4px 0 0 4px;min-width:0"
           title="매수 ${summary.buy}개">${summary.buy ? '▲ ' + summary.buy : ''}</div>
      <div style="flex:${summary.watch||0};background:#2d2200;color:#d29922;text-align:center;padding:3px 0;min-width:0"
           title="관망 ${summary.watch}개">${summary.watch ? '— ' + summary.watch : ''}</div>
      <div style="flex:${summary.sell||0};background:#3d1a1a;color:#f85149;text-align:center;padding:3px 0;border-radius:0 4px 4px 0;min-width:0"
           title="매도 ${summary.sell}개">${summary.sell ? '▽ ' + summary.sell : ''}</div>
    </div>
    <div class="indicator-grid">`;

  Object.values(signals).forEach(s => {
    const isCtx  = s.context_only;   // ATR 등 변동성 맥락 전용
    const stClr  = isCtx ? '#8b949e'
                 : s.signal === '매수' ? '#3fb950'
                 : s.signal === '매도' ? '#f85149' : '#d29922';
    const ctxTag = isCtx ? '<span style="font-size:9px;background:#21262d;color:#8b949e;padding:1px 4px;border-radius:3px;margin-left:4px">맥락</span>' : '';
    html += `
      <div class="indicator-item">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px">
          <span class="ind-name">${s.name}${ctxTag}</span>
          ${isCtx ? '<span class="sig-watch-badge" style="opacity:.6">— 참고</span>' : badge(s.signal)}
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

  const cl  = d.pivot_points.classic || {};
  const cur = d.last_close;
  const levels = ['S3','S2','S1','Pivot','R1','R2','R3'];

  // 가장 가까운 지지/저항 탐색
  let nearestR = Infinity, nearestS = -Infinity;
  let nearestRKey = null, nearestSKey = null;
  ['R1','R2','R3'].forEach(k => {
    if (cl[k] != null && cl[k] > cur && cl[k] < nearestR) { nearestR = cl[k]; nearestRKey = k; }
  });
  ['S1','S2','S3'].forEach(k => {
    if (cl[k] != null && cl[k] < cur && cl[k] > nearestS) { nearestS = cl[k]; nearestSKey = k; }
  });

  const fmtV = v => (v == null) ? '-' : fmt(v, isKrx);

  // 설명 문구
  let html = `<p style="font-size:12px;color:#8b949e;margin-bottom:12px">전일 고/저/종가를 기준으로 산출한 오늘의 예상 지지·저항 구간입니다. S(지지선)에 근접하면 반등, R(저항선)에 근접하면 차익 실현을 검토하세요.</p>`;

  // 클래식 피봇 테이블 (1가지 방식만 표시)
  html += `<div style="overflow-x:auto">
    <table class="pivot-table">
      <thead>
        <tr>
          <th class="pv-s">S3</th><th class="pv-s">S2</th><th class="pv-s">S1</th>
          <th class="pv-p">Pivot</th>
          <th class="pv-r">R1</th><th class="pv-r">R2</th><th class="pv-r">R3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          ${levels.map(k => {
            const val = cl[k];
            const cls = k === 'Pivot' ? 'pv-p' : k.startsWith('R') ? 'pv-r' : 'pv-s';
            let extra = '';
            if (k === nearestRKey) extra = 'class="pv-nr"';
            if (k === nearestSKey) extra = 'class="pv-ns"';
            return `<td class="${cls}" ${extra}>${fmtV(val)}</td>`;
          }).join('')}
        </tr>
      </tbody>
    </table>
  </div>`;

  // 가장 가까운 지지/저항 요약 카드
  if (nearestSKey || nearestRKey) {
    html += `<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:14px">`;
    if (nearestSKey) {
      const sVal = cl[nearestSKey];
      const sPct = ((sVal - cur) / cur * 100).toFixed(2);
      html += `<div style="background:#0d2d1a;border:1px solid #1a4730;border-radius:8px;padding:10px 14px;flex:1;min-width:220px">
        <div style="font-size:11px;color:#8b949e;margin-bottom:4px">🟢 가장 가까운 지지선 (${nearestSKey})</div>
        <div style="font-size:16px;font-weight:700;color:#3fb950">${fmtV(sVal)} <span style="font-size:11px;font-weight:400">(${sPct}%)</span></div>
        <div style="font-size:11px;color:#8b949e;margin-top:6px">→ 이 구간 근접 시 <strong>분할 매수</strong> 고려</div>
      </div>`;
    }
    if (nearestRKey) {
      const rVal = cl[nearestRKey];
      const rPct = ((rVal - cur) / cur * 100).toFixed(2);
      html += `<div style="background:#2d0d0d;border:1px solid #4d1515;border-radius:8px;padding:10px 14px;flex:1;min-width:220px">
        <div style="font-size:11px;color:#8b949e;margin-bottom:4px">🔴 가장 가까운 저항선 (${nearestRKey})</div>
        <div style="font-size:16px;font-weight:700;color:#f85149">${fmtV(rVal)} <span style="font-size:11px;font-weight:400">(+${rPct}%)</span></div>
        <div style="font-size:11px;color:#8b949e;margin-top:6px">→ 이 구간 돌파 시 <strong>상승 탄력 확인</strong> 후 추가 매수</div>
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
const ALL_TABS = ['chart','ai','report','forecast','news','evening'];
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
  // 시장 무드 배지 — 한국/미국 개별 표시
  const moodMap = {
    positive: ['mood-positive', '우호적'],
    neutral:  ['mood-neutral',  '혼조'],
    negative: ['mood-negative', '부담적'],
  };
  // 한국 배지 (kr_market_mood 우선, 없으면 기존 market_mood 폴백)
  const krMood = d.kr_market_mood || d.market_mood || 'neutral';
  const [krCls, krTxt] = moodMap[krMood] || moodMap.neutral;
  const krBadge = document.getElementById('core-mood-badge-kr');
  if (krBadge) {
    krBadge.className = 'mood-badge market-mood-btn ' + krCls;
    krBadge.textContent = '🇰🇷 한국 ' + krTxt;
  }
  // 미국 배지 (us_market_mood)
  const usMood = d.us_market_mood || 'neutral';
  const [usCls, usTxt] = moodMap[usMood] || moodMap.neutral;
  const usBadge = document.getElementById('core-mood-badge-us');
  if (usBadge) {
    usBadge.className = 'mood-badge market-mood-btn ' + usCls;
    usBadge.textContent = '🇺🇸 미국 ' + usTxt;
  }

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
// 🏭 업종별 흐름 — 메인 페이지 자동 로드
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

// ── 업종별 흐름 전체 렌더 ────────────────────────────────────────────────────
function renderSectorFlow(d) {
  const cardsEl = document.getElementById('sector-cards');
  if (!cardsEl) return;

  const sectors = d.sectors || [];
  if (!sectors.length) {
    cardsEl.innerHTML = '<p style="color:#484f58;font-size:13px">섹터 데이터 없음</p>';
    return;
  }

  // 등락률(%) 내림차순 정렬 — 동률 시 업종명 가나다순
  // (강세/혼조/약세 상태값은 카드 내 표시용이며 정렬 기준으로 사용하지 않음)
  const sorted = [...sectors].sort((a, b) => {
    const pa = (a.avg_change_pct != null) ? a.avg_change_pct : -Infinity;
    const pb = (b.avg_change_pct != null) ? b.avg_change_pct : -Infinity;
    if (pa !== pb) return pb - pa;
    return (a.name || '').localeCompare(b.name || '', 'ko');
  });

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

  // ── 핵심 데이터 추출 ──
  const score = d.score || 50;
  const sigSum = ((d.indicator_signals || {}).summary) || {};
  // weighted_score: -100 ~ +100 (indicator_signals에서 계산된 종합 가중 점수)
  const wscore  = sigSum.weighted_score || 0;
  const buyN    = sigSum.buy   || 0;
  const sellN   = sigSum.sell  || 0;
  const watchN  = sigSum.watch || 0;
  const totalN  = sigSum.total || Math.max(1, buyN + sellN + watchN);

  // ── 보조 신호 (뉴스·추세·RSI) ──
  const upBonus = [newsSent==='positive', trendDir==='up', posZone==='low_zone'].filter(Boolean).length;
  const dnBonus = [newsSent==='negative', trendDir==='down', posZone==='high_zone'].filter(Boolean).length;

  // ── 눌림목 분석 데이터 ──
  const pa = d.pullback_analysis;
  const flowStage  = pa ? (pa.flow_stage  || 0) : 0;
  const pbQuality  = pa ? (pa.pullback_score_pct || 0) : 0;
  const slTriggered= pa ? (pa.sl_triggered || 0) : 0;

  // ── 통합 효과 점수 계산 ──
  // score(0-100) + wscore 보정(-25~+25) + 보조신호 보정(-10~+10)
  const wNorm  = wscore / 4;                    // -25 ~ +25
  const bNorm  = (upBonus - dnBonus) * 4;       // -12 ~ +12
  // 눌림목 단계(4)이면서 체크리스트 품질이 높을 때 최대 +8점 부스트
  const pbBoost = (flowStage === 4 && pbQuality >= 50)
    ? Math.round((pbQuality - 50) / 6.25)       // 0 ~ +8
    : 0;
  // 구조 붕괴 조건이 트리거 됐을 때 페널티
  const slPenalty = slTriggered >= 2 ? -10 : slTriggered === 1 ? -4 : 0;
  const effScore = Math.min(100, Math.max(0, score + wNorm + bNorm + pbBoost + slPenalty));

  // ── 추천 결정 (effScore 1차 기준) ──
  let rec, recLbl, recCls, confLabel, rationale;

  if      (effScore >= 73) { rec='strong_buy';  recLbl='적극 매수';       recCls='rec-strong-buy'; }
  else if (effScore >= 62) { rec='buy';          recLbl='매수 우위';        recCls='rec-buy';        }
  else if (effScore >= 54) { rec='weak_buy';     recLbl='단기 반등 가능';   recCls='rec-buy';        }
  else if (effScore >= 44) { rec='hold';         recLbl='관망 권장';        recCls='rec-hold';       }
  else if (effScore >= 34) { rec='weak_sell';    recLbl='추세 약화';        recCls='rec-sell';       }
  else                     { rec='sell';         recLbl='리스크 높음';      recCls='rec-sell';       }

  // ── 눌림목 특별 케이스 오버라이드 ──
  if (flowStage === 4 && pbQuality >= 65 && score >= 48 && slTriggered === 0) {
    rec='buy'; recLbl='눌림목 진입 기회'; recCls='rec-buy';
  }
  // 구조 붕괴 2개 이상: 최소 추세 약화 이상으로 격하
  if (slTriggered >= 2 && (rec==='strong_buy' || rec==='buy' || rec==='weak_buy')) {
    rec='hold'; recLbl='구조 점검 필요'; recCls='rec-hold';
  }

  // ── 신뢰도: 지표 신호 일관성 + ADX 추세 강도 기반 ──
  // buyN 또는 sellN이 totalN 중 얼마나 지배적인지
  const dominance = totalN > 0 ? Math.max(buyN, sellN) / totalN : 0;
  const absWscore = Math.abs(wscore);
  if      (dominance >= 0.65 || absWscore >= 50) confLabel = '높음';
  else if (dominance >= 0.50 || absWscore >= 30) confLabel = '보통';
  else if (dominance >= 0.38 || absWscore >= 15) confLabel = '중간';
  else                                            confLabel = '낮음';

  // sell 계열이면 신뢰도 대신 리스크 강조
  const confText = (rec==='sell' || rec==='weak_sell')
    ? (confLabel === '높음' ? '리스크 높음' : confLabel === '보통' ? '리스크 보통' : '리스크 주의')
    : ('신뢰도 ' + confLabel);

  // ── 근거 문구 ──
  const reasonParts = [`기술점수 ${score}점`];
  if      (wscore >= 30)  reasonParts.push('지표 매수 우세');
  else if (wscore <= -30) reasonParts.push('지표 매도 우세');
  if (trendDir === 'up')   reasonParts.push('MA20 상승');
  else if (trendDir==='down') reasonParts.push('MA20 하락');
  if (newsSent === 'positive') reasonParts.push('뉴스 긍정');
  else if (newsSent==='negative') reasonParts.push('뉴스 부정');
  if (posZone === 'low_zone')  reasonParts.push('RSI 과매도');
  else if (posZone==='high_zone') reasonParts.push('RSI 과매수 주의');
  if (flowStage === 4 && pbQuality >= 65) reasonParts.push('눌림목 진입 조건 충족');
  if (slTriggered >= 2) reasonParts.push('구조 붕괴 경고 ' + slTriggered + '건');
  rationale = reasonParts.join(' · ');

  // ── 이하 기존 conf 변수 참조 교체 ──
  const conf = confLabel;

  // 신호 매트릭스 렌더
  const newsSub = finnSent && finnSent.bullish_pct != null
    ? `긍정 ${posN}% · 부정 ${negN}%`
    : `호재 ${posN}건 · 악재 ${negN}건`;
  const flowMatrix = document.getElementById('flow-matrix');
  if (flowMatrix) flowMatrix.innerHTML = `
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
  const flowRecBadge = document.getElementById('flow-rec-badge');
  if (flowRecBadge) {
    const _gc  = flowRecBadge.dataset.gradeColor;
    const _gb  = flowRecBadge.dataset.gradeBg;
    const _bt  = flowRecBadge.dataset.badgeText; // 등급 행동 지침 포함 초기 텍스트
    // 등급 행동 지침(좌측) + 종합점수(중간) + 신뢰도(우측)
    // ex) "관망 유지 · 종합 47점 · 신뢰도 보통"
    const _action = _bt ? _bt.split(' · ')[0] : recLbl;
    flowRecBadge.className = 'rec-badge-lg';
    flowRecBadge.textContent = `${_action} · 종합 ${effScore.toFixed(1)}점 · ${confText}`;
    if (_gc) {
      flowRecBadge.style.color       = _gc;
      flowRecBadge.style.borderColor = _gc;
      flowRecBadge.style.background  = _gb || '#21262d';
    } else {
      flowRecBadge.className = 'rec-badge-lg ' + recCls;
    }
  }
  const flowRationale = document.getElementById('flow-rationale');
  if (flowRationale) flowRationale.textContent = rationale;

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
  // 사이드바 뱃지
  const el = document.getElementById('alert-bell-count');
  if (el) { el.textContent = count > 0 ? count : ''; el.classList.toggle('visible', count > 0); }
  // 시장 현황 헤더 뱃지
  const hel = document.getElementById('header-alert-count');
  if (hel) { hel.textContent = count > 0 ? count : ''; hel.classList.toggle('visible', count > 0); }
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

// ── 섹션 간 중복 제거용 전역 Set ──────────────────────────────────
var _usLtTickers = new Set();   // US 장기 추천에 이미 표시된 티커
var _krLtTickers = new Set();   // KR 장기 추천에 이미 표시된 티커

// ── 국내 장기 투자 추천 ───────────────────────────────────────────
async function loadKrLongterm(force) {
  var ldg = document.getElementById('kr-lt-loading');
  var cnt = document.getElementById('kr-lt-content');
  var err = document.getElementById('kr-lt-error');
  if (ldg) { ldg.style.display = 'block'; }
  if (cnt) { cnt.style.display = 'none'; }
  if (err) { err.style.display = 'none'; }
  try {
    var r = await fetch('/api/kr/longterm');
    var d = await r.json();
    if (ldg) ldg.style.display = 'none';
    if (d.error && !(d.items && d.items.length)) {
      if (err) err.style.display = 'block';
      return;
    }
    renderKrLongtermCards(d.items || []);
    if (cnt) cnt.style.display = 'block';
  } catch(e) {
    if (ldg) ldg.style.display = 'none';
    if (err) err.style.display = 'block';
  }
}

function renderKrLongtermCards(items) {
  var el = document.getElementById('kr-lt-cards');
  if (!el) return;
  _krLtTickers = new Set((items || []).map(function(it) { return it.ticker; }));
  if (!items.length) {
    el.innerHTML = '<div class="us-reco-empty">현재 국내 장기 추천 조건에 부합하는 종목이 없습니다.<br>장기 투자 스코어링 조건을 충족하는 종목이 발견되면 표시됩니다.</div>';
    return;
  }

  var trendColors = {
    '강한 상승': '#3fb950', '상승': '#56d364', '중기 상승': '#7ee787',
    '눌림목': '#d29922', '추세 전환': '#58a6ff', '초기 반등': '#79c0ff',
    '바닥권': '#ff7b72', '횡보': '#8b949e', '혼조': '#6e7681'
  };
  var aiDirColors = { '강한 상승': '#3fb950', '상승': '#56d364', '중립': '#8b949e', '하락': '#ff7b72' };

  el.innerHTML = items.map(function(it, idx) {
    var pctCls  = (it.change_pct || 0) >= 0 ? 'grn' : 'red';
    var pctSign = (it.change_pct || 0) >= 0 ? '+' : '';
    var expSign = (it.expected_return || 0) >= 0 ? '+' : '';
    var badgeCls = it.confidence === 'High' ? 'us-reco-badge-high' : 'us-reco-badge-med';
    var trendColor = trendColors[it.trend_status] || '#8b949e';
    var aiColor    = aiDirColors[it.ai_direction]  || '#8b949e';

    // 추천 사유
    var reasons = (it.reasons || []).map(function(r) {
      return '<div class="us-reco-reason">▸ ' + r + '</div>';
    }).join('');

    // 리스크
    var risks = (it.risks || []).map(function(r) {
      return '<div class="kr-lt-risk">⚠ ' + r + '</div>';
    }).join('');

    // 칼만 예측
    var kpredHtml = '';
    if (it.kalman_predicted && it.close) {
      var kret = (((it.kalman_predicted - it.close) / it.close) * 100).toFixed(1);
      kpredHtml = '<div class="us-reco-pi"><div class="us-reco-pi-label">칼만 40일 예측</div>' +
        '<div class="us-reco-pi-val">' + it.kalman_predicted.toLocaleString() + '원' +
        ' <span style="font-size:11px;color:#8b949e">' + (parseFloat(kret) >= 0 ? '+' : '') + kret + '%</span></div></div>';
    }

    // 펀더멘털 배지
    var fundHtml = '';
    var fundItems = [];
    if (it.per  != null) fundItems.push('PER ' + it.per + '배');
    if (it.pbr  != null) fundItems.push('PBR ' + it.pbr + '배');
    if (it.roe  != null) fundItems.push('ROE ' + it.roe + '%');
    if (it.rs60 != null && it.rs60 !== 0) fundItems.push('RS60 ' + (it.rs60 >= 0 ? '+' : '') + it.rs60.toFixed(1) + '%');
    if (it.mdd  != null) fundItems.push('MDD ' + it.mdd + '%');
    if (fundItems.length) {
      fundHtml = '<div class="kr-lt-fund">' + fundItems.map(function(f) {
        return '<span class="kr-lt-fund-tag">' + f + '</span>';
      }).join('') + '</div>';
    }

    // 테마 배지
    var themeHtml = it.theme ? '<span class="kr-lt-theme-badge"># ' + it.theme + '</span>' : '';

    return '<div class="us-reco-card kr-lt-card">' +
      /* ── 헤더 ── */
      '<div class="us-reco-card-header">' +
        '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">' +
          '<span style="font-size:13px;color:#8b949e;font-weight:600">#' + (idx+1) + '</span>' +
          '<span class="us-reco-ticker">' + (it.name || it.ticker) + '</span>' +
          '<span style="font-size:10px;color:#484f58;font-family:monospace">' + it.ticker + '</span>' +
          themeHtml +
        '</div>' +
        '<div style="display:flex;align-items:center;gap:6px">' +
          '<span class="us-reco-score">점수 ' + it.score + '</span>' +
          '<span class="us-reco-badge ' + badgeCls + '">' + (it.confidence_label || (it.confidence === 'High' ? '확신도 높음' : '확신도 보통')) + '</span>' +
        '</div>' +
      '</div>' +
      /* ── 추세 상태 + AI 방향 ── */
      '<div style="display:flex;gap:8px;margin:8px 0 4px;flex-wrap:wrap">' +
        '<span class="kr-lt-status-badge" style="background:' + trendColor + '22;color:' + trendColor + ';border-color:' + trendColor + '55">📊 ' + (it.trend_status || '') + '</span>' +
        '<span class="kr-lt-status-badge" style="background:' + aiColor + '22;color:' + aiColor + ';border-color:' + aiColor + '55">🤖 AI: ' + (it.ai_direction || '') + '</span>' +
        '<span class="kr-lt-status-badge" style="background:#21262d;color:#8b949e;border-color:#30363d">⏱ ' + (it.holding_period || '') + '</span>' +
      '</div>' +
      /* ── 가격 정보 ── */
      '<div class="us-reco-prices">' +
        '<div class="us-reco-pi"><div class="us-reco-pi-label">현재가</div>' +
          '<div class="us-reco-pi-val">' + (it.close || 0).toLocaleString() + '원 ' +
          '<span class="' + pctCls + '" style="font-size:12px">' + pctSign + (it.change_pct || 0).toFixed(2) + '%</span></div></div>' +
        '<div class="us-reco-pi"><div class="us-reco-pi-label">목표가</div>' +
          '<div class="us-reco-pi-val grn">' + (it.target_price || 0).toLocaleString() + '원' +
          ' <span style="font-size:11px;color:#3fb950">' + expSign + (it.expected_return || 0).toFixed(1) + '%</span></div></div>' +
        '<div class="us-reco-pi"><div class="us-reco-pi-label">손절가</div>' +
          '<div class="us-reco-pi-val red">' + (it.stop_loss || 0).toLocaleString() + '원</div></div>' +
        kpredHtml +
      '</div>' +
      /* ── 핵심 추천 사유 ── */
      '<div style="margin:10px 0 4px;font-size:11px;color:#8b949e;font-weight:600;letter-spacing:.04em">핵심 추천 사유</div>' +
      '<div class="us-reco-reasons">' + reasons + '</div>' +
      /* ── 리스크 ── */
      (risks ? '<div style="margin:8px 0 4px;font-size:11px;color:#8b949e;font-weight:600;letter-spacing:.04em">리스크 요소</div>' +
        '<div class="kr-lt-risks">' + risks + '</div>' : '') +
      /* ── 펀더멘털 태그 ── */
      fundHtml +
    '</div>';
  }).join('');
}

// ── US 장기 투자 추천 ─────────────────────────────────────────────
async function loadUsLongterm(force) {
  var ldg = document.getElementById('us-lt-loading');
  var cnt = document.getElementById('us-lt-content');
  var err = document.getElementById('us-lt-error');
  if (ldg) { ldg.style.display = 'block'; }
  if (cnt) { cnt.style.display = 'none'; }
  if (err) { err.style.display = 'none'; }
  try {
    var r = await fetch('/api/us/longterm');
    var d = await r.json();
    if (ldg) ldg.style.display = 'none';
    if (d.error && !(d.items && d.items.length)) {
      if (err) err.style.display = 'block';
      return;
    }
    renderUsLongtermCards(d.items || []);
    if (cnt) cnt.style.display = 'block';
  } catch(e) {
    if (ldg) ldg.style.display = 'none';
    if (err) err.style.display = 'block';
  }
}

function renderUsLongtermCards(items) {
  var el = document.getElementById('us-lt-cards');
  if (!el) return;
  _usLtTickers = new Set((items || []).map(function(it) { return it.ticker; }));
  if (!items.length) {
    el.innerHTML = '<div class="us-reco-empty">현재 장기 추천 조건에 부합하는 종목이 없습니다.<br>기술·수급 통합 조건을 충족하는 종목이 발견되면 표시됩니다.</div>';
    return;
  }

  var trendColors = {
    '강한 상승': '#3fb950', '상승': '#56d364', '중기 상승': '#7ee787',
    '눌림목': '#d29922', '추세 전환': '#58a6ff', '초기 반등': '#79c0ff',
    '바닥권': '#ff7b72', '횡보': '#8b949e', '혼조': '#6e7681'
  };
  var aiDirColors = { '강한 상승': '#3fb950', '상승': '#56d364', '중립': '#8b949e', '하락': '#ff7b72' };

  el.innerHTML = items.map(function(it, idx) {
    var pctCls  = (it.change_pct || 0) >= 0 ? 'grn' : 'red';
    var pctSign = (it.change_pct || 0) >= 0 ? '+' : '';
    var expSign = (it.expected_return || 0) >= 0 ? '+' : '';
    var badgeCls = it.confidence === 'High' ? 'us-reco-badge-high' : 'us-reco-badge-med';
    var trendColor = trendColors[it.trend_status] || '#8b949e';
    var aiColor    = aiDirColors[it.ai_direction]  || '#8b949e';

    // 추천 사유
    var reasons = (it.reasons || []).map(function(r) {
      return '<div class="us-reco-reason">▸ ' + r + '</div>';
    }).join('');

    // 리스크
    var risks = (it.risks || []).map(function(r) {
      return '<div class="kr-lt-risk">⚠ ' + r + '</div>';
    }).join('');

    // 칼만 예측
    var kpredHtml = '';
    if (it.kalman_predicted && it.close) {
      var kret = (((it.kalman_predicted - it.close) / it.close) * 100).toFixed(1);
      kpredHtml = '<div class="us-reco-pi"><div class="us-reco-pi-label">칼만 40일 예측</div>' +
        '<div class="us-reco-pi-val">$' + it.kalman_predicted.toFixed(2) +
        ' <span style="font-size:11px;color:#8b949e">' + (parseFloat(kret) >= 0 ? '+' : '') + kret + '%</span></div></div>';
    }

    // 보조 지표 태그
    var fundItems = [];
    if (it.rsi  != null) fundItems.push('RSI ' + it.rsi.toFixed(1));
    if (it.adx  != null) fundItems.push('ADX ' + it.adx.toFixed(1));
    if (it.rs60 != null && it.rs60 !== 0) fundItems.push('RS60 ' + (it.rs60 >= 0 ? '+' : '') + it.rs60.toFixed(1) + '%');
    else if (it.rs20 != null && it.rs20 !== 0) fundItems.push('RS20 ' + (it.rs20 >= 0 ? '+' : '') + it.rs20.toFixed(1) + '%');
    if (it.week52_pos != null) fundItems.push('52주 ' + it.week52_pos.toFixed(0) + '%');
    var fundHtml = fundItems.length ? '<div class="kr-lt-fund">' + fundItems.map(function(f) {
      return '<span class="kr-lt-fund-tag">' + f + '</span>';
    }).join('') + '</div>' : '';

    return '<div class="us-reco-card kr-lt-card">' +
      /* ── 헤더 ── */
      '<div class="us-reco-card-header">' +
        '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">' +
          '<span style="font-size:13px;color:#8b949e;font-weight:600">#' + (idx+1) + '</span>' +
          '<span class="us-reco-ticker">' + it.ticker + '</span>' +
        '</div>' +
        '<div style="display:flex;align-items:center;gap:6px">' +
          '<span class="us-reco-score">점수 ' + it.score + '</span>' +
          '<span class="us-reco-badge ' + badgeCls + '">' + (it.confidence_label || (it.confidence === 'High' ? '확신도 높음' : '확신도 보통')) + '</span>' +
        '</div>' +
      '</div>' +
      /* ── 추세 상태 + AI 방향 ── */
      '<div style="display:flex;gap:8px;margin:8px 0 4px;flex-wrap:wrap">' +
        '<span class="kr-lt-status-badge" style="background:' + trendColor + '22;color:' + trendColor + ';border-color:' + trendColor + '55">📊 ' + (it.trend_status || '') + '</span>' +
        '<span class="kr-lt-status-badge" style="background:' + aiColor + '22;color:' + aiColor + ';border-color:' + aiColor + '55">🤖 AI: ' + (it.ai_direction || '') + '</span>' +
        '<span class="kr-lt-status-badge" style="background:#21262d;color:#8b949e;border-color:#30363d">⏱ ' + (it.holding_period || '') + '</span>' +
      '</div>' +
      /* ── 가격 정보 ── */
      '<div class="us-reco-prices">' +
        '<div class="us-reco-pi"><div class="us-reco-pi-label">현재가</div>' +
          '<div class="us-reco-pi-val">$' + (it.close || 0).toFixed(2) +
          ' <span class="' + pctCls + '" style="font-size:12px">' + pctSign + (it.change_pct || 0).toFixed(2) + '%</span></div></div>' +
        '<div class="us-reco-pi"><div class="us-reco-pi-label">목표가</div>' +
          '<div class="us-reco-pi-val grn">$' + (it.target_price || 0).toFixed(2) +
          ' <span style="font-size:11px;color:#3fb950">' + expSign + (it.expected_return || 0).toFixed(1) + '%</span></div></div>' +
        '<div class="us-reco-pi"><div class="us-reco-pi-label">손절가</div>' +
          '<div class="us-reco-pi-val red">$' + (it.stop_loss || 0).toFixed(2) + '</div></div>' +
        kpredHtml +
      '</div>' +
      /* ── 핵심 추천 사유 ── */
      '<div style="margin:10px 0 4px;font-size:11px;color:#8b949e;font-weight:600;letter-spacing:.04em">핵심 추천 사유</div>' +
      '<div class="us-reco-reasons">' + reasons + '</div>' +
      /* ── 리스크 ── */
      (risks ? '<div style="margin:8px 0 4px;font-size:11px;color:#8b949e;font-weight:600;letter-spacing:.04em">리스크 요소</div>' +
        '<div class="kr-lt-risks">' + risks + '</div>' : '') +
      /* ── 보조 지표 태그 ── */
      fundHtml +
    '</div>';
  }).join('');
}

// ── US 개장 급등 추천 ─────────────────────────────────────────────
async function loadUsSurge(force) {
  var ldg = document.getElementById('us-surge-loading');
  var cnt = document.getElementById('us-surge-content');
  var err = document.getElementById('us-surge-error');
  if (ldg) { ldg.style.display = 'block'; }
  if (cnt) { cnt.style.display = 'none'; }
  if (err) { err.style.display = 'none'; }
  try {
    var r = await fetch('/api/us/opening-surge');
    var d = await r.json();
    if (ldg) ldg.style.display = 'none';
    if (d.error && !(d.items && d.items.length)) {
      if (err) err.style.display = 'block';
      return;
    }
    renderUsSurgeCards(d.items || [], d.note);
    // 세션 레이블 배지 업데이트 (있는 경우)
    var slEl = document.getElementById('us-surge-session-label');
    if (slEl && d.session_label) slEl.textContent = d.session_label;
    if (cnt) cnt.style.display = 'block';
  } catch(e) {
    if (ldg) ldg.style.display = 'none';
    if (err) err.style.display = 'block';
  }
}

function renderUsSurgeCards(items, note) {
  var el = document.getElementById('us-surge-cards');
  if (!el) return;
  // US 장기 추천에 이미 포함된 티커는 중복 제거
  var filtered = (items || []).filter(function(it) {
    return !_usLtTickers.has(it.ticker);
  });
  if (!filtered.length) {
    var msg = note || '현재 급등 조건에 부합하는 종목이 없습니다.<br>프리마켓 시간(미국 동부 오전 4~9시30분)에 확인해 주세요.';
    el.innerHTML = '<div class="us-reco-empty">' + msg + '</div>';
    return;
  }
  var confColors = {
    '강력 추천': {bg:'rgba(46,160,67,.15)', border:'rgba(46,160,67,.5)', color:'#3fb950'},
    '추천':      {bg:'rgba(56,139,253,.12)', border:'rgba(56,139,253,.4)', color:'#58a6ff'},
    '주목':      {bg:'rgba(187,128,9,.12)', border:'rgba(187,128,9,.4)', color:'#d29922'},
    '관심':      {bg:'rgba(110,118,129,.12)', border:'rgba(110,118,129,.4)', color:'#8b949e'}
  };
  el.innerHTML = filtered.map(function(it, idx) {
    var rank = idx + 1;
    var conf = it.confidence_label || '관심';
    var cc = confColors[conf] || confColors['관심'];
    var confBadge = '<span class="kr-lt-status-badge" style="background:' + cc.bg + ';border-color:' + cc.border + ';color:' + cc.color + '">' + conf + '</span>';

    var pmChg = it.pm_change_pct || 0;
    var pmColor = pmChg >= 0 ? '#3fb950' : '#f85149';
    var pmSign  = pmChg >= 0 ? '+' : '';

    var reasons = (it.reasons || []).map(function(r) {
      return '<div class="us-reco-reason">' + r + '</div>';
    }).join('');
    var warnings = (it.warning || []).map(function(w) {
      return '<div class="kr-lt-risk">⚠ ' + w + '</div>';
    }).join('');

    var inds = [];
    if (it.rsi  != null) inds.push('RSI '  + it.rsi.toFixed(1));
    if (it.adx  != null) inds.push('ADX '  + it.adx.toFixed(1));
    if (it.rvol != null) inds.push('RVOL ' + it.rvol.toFixed(1) + 'x');
    if (it.rs60 != null) inds.push('RS60 ' + (it.rs60 >= 0 ? '+' : '') + it.rs60.toFixed(1) + '%');
    else if (it.rs20 != null) inds.push('RS20 ' + (it.rs20 >= 0 ? '+' : '') + it.rs20.toFixed(1) + '%');

    var indHtml = inds.length
      ? '<div class="kr-lt-fund">' + inds.map(function(i) { return '<span class="kr-lt-fund-tag">' + i + '</span>'; }).join('') + '</div>'
      : '';

    return '<div class="us-reco-card kr-lt-card">' +
      '<div class="us-reco-card-header" style="gap:6px">' +
        '<span style="font-size:13px;font-weight:700;color:#8b949e;min-width:26px">#' + rank + '</span>' +
        '<span class="us-reco-ticker">' + it.ticker + '</span>' +
        confBadge +
        '<span class="us-reco-score" style="margin-left:auto">점수 ' + it.score + '</span>' +
      '</div>' +
      '<div class="us-reco-prices">' +
        '<div class="us-reco-pi"><div class="us-reco-pi-label">PM 가격</div>' +
          '<div class="us-surge-pm">$' + (it.pm_price || 0).toFixed(2) +
          ' <span style="font-size:13px;color:' + pmColor + '">' + pmSign + pmChg.toFixed(2) + '%</span></div></div>' +
        '<div class="us-reco-pi"><div class="us-reco-pi-label">목표가 (ATR)</div>' +
          '<div class="us-reco-pi-val grn">$' + (it.target_price || 0).toFixed(2) +
          ' <span style="font-size:11px">(+' + (it.target_return || 0).toFixed(1) + '%)</span></div></div>' +
        '<div class="us-reco-pi"><div class="us-reco-pi-label">손절가</div>' +
          '<div class="us-reco-pi-val red">$' + (it.stop_loss || 0).toFixed(2) + '</div></div>' +
        '<div class="us-reco-pi"><div class="us-reco-pi-label">R:R</div>' +
          '<div class="us-reco-pi-val">' + (it.risk_reward || 0).toFixed(2) + ':1</div></div>' +
      '</div>' +
      (reasons ? '<div class="us-reco-reasons" style="margin-top:8px"><div style="font-size:11px;color:#8b949e;font-weight:600;margin-bottom:4px">⚡ 급등 신호</div>' + reasons + '</div>' : '') +
      (warnings ? '<div class="kr-lt-risks" style="margin-top:6px">' + warnings + '</div>' : '') +
      indHtml +
      '<div class="us-reco-holding" style="margin-top:8px">⏱ ' + (it.holding_period || '당일 매도 필수') + '</div>' +
    '</div>';
  }).join('');
}

// ── 토스증권 AI 요약 ──────────────────────────────────────────────────────
/**
 * 토스증권 AI 요약 비동기 조회 및 렌더링 (KRX + US 공통).
 *
 * [KRX] productCode = "A" + 종목코드 → 즉시 AI signals 배치 API 호출
 * [US]  전략 순서:
 *   0) 랭킹 API 캐시에서 name 매칭 (ETF·레버리지 상품 즉시 매칭)
 *   1) Toss 검색 GET 엔드포인트 다수 시도
 *   2) Toss 검색 POST 엔드포인트 다수 시도
 *   3) tossinvest.com __NEXT_DATA__ 스크래핑
 *   → productCode 확보 후 배치 API + 상세 API 재시도
 *
 * 모든 상태(로딩·성공·실패·빈값·타임아웃)를 명확히 표시.
 * AbortController로 race condition 방지: 연속 검색 시 이전 요청 자동 취소.
 *
 * @param {string} ticker - yfinance 형식 ticker (e.g. "005930.KS", "AAPL", "NVDU")
 * @param {string} market - "KRX" | "US"
 */

// 진행 중인 fetch를 취소하기 위한 컨트롤러.
// 연속 검색 시 이전 응답이 현재 카드를 덮어쓰는 race condition을 방지한다.
var _tossAiController = null;

function fetchTossAiSummary(ticker, market) {
  var cardEl    = document.getElementById('r-toss-card');
  var summaryEl = document.getElementById('r-toss-summary');
  var timeEl    = document.getElementById('r-toss-time');
  if (!cardEl || !summaryEl) return;

  // ── 이전 진행 중 요청 즉시 취소 (race condition 방지) ────────────────
  if (_tossAiController) {
    _tossAiController.abort();
  }
  _tossAiController = new AbortController();
  var signal     = _tossAiController.signal;
  var isUS       = (market !== 'KRX');
  var timedOut   = false;

  // ── 로딩 상태 ────────────────────────────────────────────────────────
  summaryEl.innerHTML =
    '<span class="toss-ai-spinner"></span>' +
    '<span style="color:#484f58;font-size:11px">' +
      (isUS ? '해외 요약 조회 중...' : '조회 중...') +
    '</span>';
  if (timeEl) timeEl.textContent = '';

  // ── 클라이언트 사이드 타임아웃 ──────────────────────────────────────
  // 백엔드 전략 1~3 직렬 실행 시 최악 수십 초 소요 방지.
  // 랭킹 캐시 히트 시 KRX 수준(~1s)으로 빠르게 완료되므로 실제론 드물게 발동.
  var timeoutMs = isUS ? 22000 : 10000;
  var timeoutId = setTimeout(function() {
    timedOut = true;
    if (_tossAiController) _tossAiController.abort();
  }, timeoutMs);

  // ── API 호출 ─────────────────────────────────────────────────────────
  var url = '/api/toss-ai-summary'
          + '?ticker=' + encodeURIComponent(ticker)
          + '&market=' + encodeURIComponent(market);

  fetch(url, { signal: signal })
    .then(function(r) {
      clearTimeout(timeoutId);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(function(data) {
      var txt = (data && data.ai_summary) || '';

      if (data && data.error) {
        // 백엔드에서 명시적 에러 반환
        summaryEl.style.color      = '#484f58';
        summaryEl.style.fontSize   = '11px';
        summaryEl.style.fontWeight = '';
        summaryEl.textContent = '조회 불가';
        return;
      }

      // supported=false: productCode 조회 자체가 실패
      // US 종목은 토스 랭킹에 없거나 검색 API가 인식 못하는 종목
      if (data && data.supported === false) {
        summaryEl.style.color      = '#484f58';
        summaryEl.style.fontSize   = '11px';
        summaryEl.style.fontWeight = '';
        summaryEl.textContent = isUS ? '토스증권 미지원 종목' : '종목 코드 조회 실패';
        return;
      }

      if (!txt) {
        // productCode 확보됐으나 토스가 아직 미분석
        summaryEl.style.color      = '#484f58';
        summaryEl.style.fontSize   = '11px';
        summaryEl.style.fontWeight = '';
        summaryEl.textContent = '요약 없음';
      } else {
        // ── 성공: textContent로 XSS 방어 ──────────────────────────────
        summaryEl.style.color      = '#e6edf3';
        summaryEl.style.fontSize   = '13px';
        summaryEl.style.fontWeight = '600';
        summaryEl.textContent = txt;
      }

      // 기준 시간 표시
      if (timeEl) {
        var now = new Date();
        var hh  = now.getHours().toString().padStart(2, '0');
        var mm  = now.getMinutes().toString().padStart(2, '0');
        timeEl.textContent = '토스증권 기준 ' + hh + ':' + mm;
      }
    })
    .catch(function(err) {
      clearTimeout(timeoutId);
      // AbortError: 새 검색이 시작되어 이전 요청이 취소된 경우 → 무시
      if (err && err.name === 'AbortError' && !timedOut) return;
      // 타임아웃으로 인한 abort
      summaryEl.style.color      = '#484f58';
      summaryEl.style.fontSize   = '11px';
      summaryEl.style.fontWeight = '';
      summaryEl.textContent = timedOut ? '조회 시간 초과' : '조회 실패';
      if (timeEl) timeEl.textContent = '';
    });
}

// ── 초기화 ──
loadMarketCore();   // ⭐ 페이지 로드 시 오늘의 핵심 자동 로드
loadSectorFlow();   // 🏭 업종별 흐름 자동 로드
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
    var pages = ['screener', 'kr-longterm', 'us-longterm', 'us-surge'];
    for (var i = 0; i < pages.length; i++) {
      var el = document.getElementById('page-' + pages[i]);
      if (el && el.style.display !== 'none') return pages[i];
    }
    return 'analysis';
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
    } else if (page === 'kr-longterm') {
      p = loadKrLongterm(true);
    } else if (page === 'us-longterm') {
      p = loadUsLongterm(true);
    } else if (page === 'us-surge') {
      p = loadUsSurge(true);
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

<script>
// ══════════════════════════════════════════════════════
// 🔬 7단계 스캔 엔진 UI
// ══════════════════════════════════════════════════════

async function runScan(force) {
  var market = document.getElementById('scan-market').value;
  var mode   = document.getElementById('scan-mode').value;
  var introEl  = document.getElementById('scan-intro');
  var loadEl   = document.getElementById('scan-loading');
  var errorEl  = document.getElementById('scan-error');
  var resultEl = document.getElementById('scan-result');
  var runBtn   = document.getElementById('scan-run-btn');

  if (introEl)  introEl.style.display  = 'none';
  if (errorEl)  errorEl.style.display  = 'none';
  if (resultEl) resultEl.style.display = 'none';
  if (loadEl)   loadEl.style.display   = '';
  if (runBtn)   runBtn.disabled = true;

  try {
    var url = '/api/scan?market=' + market + '&mode=' + mode + (force ? '&refresh=1' : '');
    var r   = await fetch(url);
    var d   = await r.json();

    if (d.error) {
      if (errorEl) { errorEl.textContent = '오류: ' + d.error; errorEl.style.display = ''; }
      return;
    }

    _scanLoaded = true;
    renderScanResult(d, market);
    if (resultEl) resultEl.style.display = '';

  } catch(e) {
    if (errorEl) { errorEl.textContent = '네트워크 오류: ' + e.message; errorEl.style.display = ''; }
  } finally {
    if (loadEl) loadEl.style.display = 'none';
    if (runBtn) runBtn.disabled = false;
  }
}

function renderScanResult(d, market) {
  var isKrx = market === 'KRX';
  var fmtP  = function(v) {
    if (v == null) return '—';
    return fmtPrice(v, isKrx);
  };

  // ── 섹터명 한글 변환 ──────────────────────────────────────────────────────
  var _sectorKo = {
    'Technology':              '기술',
    'Healthcare':              '헬스케어',
    'Communication Services':  '통신 서비스',
    'Industrials':             '산업재',
    'Financial Services':      '금융',
    'Consumer Cyclical':       '경기소비재',
    'Consumer Defensive':      '필수소비재',
    'Basic Materials':         '기초소재',
    'Energy':                  '에너지',
    'Real Estate':             '부동산',
    'Utilities':               '유틸리티',
    'Semiconductor':           '반도체',
    'Software':                '소프트웨어',
    'Electronics':             '전자',
    'Biotechnology':           '바이오',
    'Pharmaceuticals':         '제약',
    'Banks':                   '은행',
    'Insurance':               '보험',
    'Chemicals':               '화학',
    'Steel':                   '철강',
    'Automobile':              '자동차',
    'Retail':                  '유통',
    'Media':                   '미디어',
    'Internet':                '인터넷',
  };
  function _toKoSector(s) { return _sectorKo[s] || s || '—'; }

  // ── 레짐 한글 변환 ────────────────────────────────────────────────────────
  var _regimeKo  = { 'BULLISH': '상승장', 'BEARISH': '하락장', 'SIDEWAYS': '횡보장' };
  var _volKo     = { 'HIGH_VOL': '고변동성', 'LOW_VOL': '저변동성', 'NORMAL_VOL': '보통' };
  var _tierKo    = { 'high': '상', 'medium': '중', 'low': '하', 'junk': '불량', 'unknown': '—' };

  // 요약 카드
  var sumEl = document.getElementById('scan-summary-cards');
  if (sumEl) {
    var cards = [
      { val: d.total_scanned,  label: '전체 스캔',  color: '#8b949e' },
      { val: d.passed_filters, label: '필터 통과',  color: '#58a6ff' },
      { val: d.ready_count,    label: '진입 준비',  color: '#3fb950' },
      { val: d.watch_count,    label: '관찰 대기',  color: '#d29922' },
      { val: (d.good_count != null ? d.good_count : (d.candidates||[]).length), label: '선정 종목', color: '#3fb950' },
    ];
    sumEl.innerHTML = cards.map(function(c) {
      return '<div class="scan-sum-card"><div class="scan-sum-val" style="color:' + c.color + '">' + (c.val || 0) + '</div><div class="scan-sum-label">' + c.label + '</div></div>';
    }).join('');
  }

  // 레짐 배지
  var regEl = document.getElementById('scan-regime-bar');
  if (regEl) {
    var regColor = d.regime === 'BULLISH' ? '#3fb950' : d.regime === 'BEARISH' ? '#f85149' : '#8b949e';
    var volColor = d.vol_regime === 'HIGH_VOL' ? '#f97316' : d.vol_regime === 'LOW_VOL' ? '#58a6ff' : '#8b949e';
    var ts = d.generated_at ? new Date(d.generated_at).toLocaleTimeString('ko-KR') : '';
    regEl.innerHTML =
      '<span style="color:#484f58;font-size:11px">시장 국면</span>' +
      '<span style="color:' + regColor + ';font-weight:700">' + (_regimeKo[d.regime] || d.regime || '—') + '</span>' +
      '<span style="color:#484f58;font-size:11px;margin-left:16px">변동성</span>' +
      '<span style="color:' + volColor + ';font-weight:700">' + (_volKo[d.vol_regime] || d.vol_regime || '—') + '</span>' +
      (function() {
        var td = d.tier_distribution; if (!td) return '';
        return '<span style="color:#484f58;font-size:11px;margin-left:16px">시총 분포</span>' +
          '<span style="font-weight:700;color:#8b949e">대 ' + (td.LARGE||0) + '</span>' +
          '<span style="font-weight:700;color:#58a6ff">중 ' + (td.MID||0) + '</span>' +
          '<span style="font-weight:700;color:#3fb950">소 ' + (td.SMALL||0) + '</span>';
      })() +
      (d.premium_count != null
        ? '<span style="color:#484f58;font-size:11px;margin-left:16px">진입 가능권</span>' +
          '<span style="font-weight:700;color:#3fb950">' + d.premium_count + '</span>' +
          (d.relaxed ? '<span style="font-size:10px;color:#d29922;border:1px solid #d2992255;border-radius:3px;padding:0 5px;margin-left:6px">관찰 후보 보강</span>' : '')
        : '') +
      (ts ? '<span style="color:#484f58;font-size:10px;margin-left:auto">생성: ' + ts + '</span>' : '');
  }

  // 후보 테이블
  var tbody = document.getElementById('scan-tbody');
  if (!tbody) return;
  var cands = d.candidates || [];
  if (cands.length === 0) {
    tbody.innerHTML = '<tr><td colspan="13" style="text-align:center;padding:24px;color:#484f58">선정 종목 없음 — 현재 진입 가능권(상태)에 든 종목이 없습니다</td></tr>';
    return;
  }

  var statusBadge = function(s) {
    var map = {
      'READY':         '<span class="scan-status-ready">진입 준비</span>',
      'WATCH':         '<span class="scan-status-watch">관찰 중</span>',
      'WAIT_PULLBACK': '<span class="scan-status-pullback">눌림목</span>',
      'FAR':           '<span class="scan-status-far">원거리</span>',
      'EARNINGS_BLOCK':'<span class="scan-status-block">실적 대기</span>',
      'COOLDOWN':      '<span class="scan-status-block">쿨다운</span>',
    };
    return map[s] || '<span class="scan-status-far">' + s + '</span>';
  };

  var scoreBar = function(v, color) {
    var pct = Math.min(100, Math.max(0, v));
    return '<div style="font-weight:700;color:' + color + ';font-size:13px">' + pct.toFixed(1) + '</div>' +
           '<div style="height:4px;background:#21262d;border-radius:2px;margin-top:3px;overflow:hidden">' +
           '<div style="height:100%;width:' + pct + '%;background:' + color + ';border-radius:2px"></div></div>';
  };

  var rows = cands.slice(0, d.display_cap || 15).map(function(c, i) {
    var ncsColor = c.ncs >= 70 ? '#3fb950' : c.ncs >= 50 ? '#58a6ff' : c.ncs >= 35 ? '#d29922' : '#f85149';
    var fwsColor = c.fws <= 30 ? '#3fb950' : c.fws <= 50 ? '#d29922' : c.fws <= 65 ? '#f97316' : '#f85149';
    var bqsColor = c.bqs >= 65 ? '#3fb950' : c.bqs >= 45 ? '#58a6ff' : '#8b949e';
    var tier     = c.quality_tier || 'unknown';
    var qmjColor = tier === 'high' ? '#3fb950' : tier === 'medium' ? '#58a6ff' : '#484f58';
    var qmjLabel = _tierKo[tier] || '—';
    // 퀀트 모멘텀 점수 (0~100) — 백엔드 제공값 우선, 없으면 티어로 폴백
    var qmScore  = c.quant_momentum_score != null ? Number(c.quant_momentum_score)
                 : (tier === 'high' ? 100 : tier === 'medium' ? 70 : tier === 'low' ? 20 : tier === 'junk' ? 0 : 50);
    var qmColor  = qmScore >= 70 ? '#3fb950' : qmScore >= 50 ? '#58a6ff' : qmScore >= 30 ? '#d29922' : '#f85149';

    var chg    = c.change_pct != null ? c.change_pct : 0;
    var chgUp  = chg >= 0;
    var chgClr = isKrx ? (chgUp ? '#f85149' : '#388bfd') : (chgUp ? '#3fb950' : '#f85149');
    var chgTxt = (chgUp ? '▲' : '▼') + ' ' + Math.abs(chg).toFixed(2) + '%';

    var cat = _toKoSector(c.category || c.sector || '');

    var sig = c.analyst_signal || '중립';
    var sigCls = sig.includes('적극') ? 'sig-buy-strong'
               : sig === '매수'        ? 'sig-buy'
               : sig === '중립' || sig === '보유' ? 'sig-neu'
               : 'sig-sell';

    // 종목명 우선 — 이름 크게, 코드 작게 (KRX는 시장 접미사 제거)
    var displayName = c.name && c.name !== c.ticker ? c.name : fmtSymbol(c.ticker, isKrx);
    var displayCode = c.name && c.name !== c.ticker ? fmtSymbol(c.ticker, isKrx) : '';

    // 시총 티어 배지 (대형/중형/중소형)
    var capTier  = c.cap_tier || 'MID';
    var capKo    = c.cap_tier_ko || ({LARGE:'대형', MID:'중형', SMALL:'중소형'}[capTier] || '중형');
    var capClr   = capTier === 'LARGE' ? '#8b949e' : capTier === 'MID' ? '#58a6ff' : '#3fb950';
    var capBadge = '<span style="font-size:9px;font-weight:700;color:' + capClr +
                   ';border:1px solid ' + capClr + '55;border-radius:3px;padding:0 4px;margin-left:5px">' + capKo + '</span>';

    return '<tr onclick="openStockDetail(\'' + c.ticker + '\', \'' + (isKrx ? 'KRX' : 'US') + '\')" style="cursor:pointer">' +
      '<td style="color:#484f58;font-size:11px">' + (i+1) + '</td>' +
      '<td><div style="font-weight:700;font-size:13px;color:#e6edf3">' + displayName + capBadge + '</div>' +
           (displayCode ? '<div style="font-size:10px;color:#484f58;margin-top:2px">' + displayCode + '</div>' : '') + '</td>' +
      '<td style="text-align:center">' + statusBadge(c.status) + '</td>' +
      '<td style="text-align:right;font-size:13px;font-weight:600">' + fmtP(c.price) + '</td>' +
      '<td style="text-align:right;font-weight:700;color:' + chgClr + '">' + chgTxt + '</td>' +
      '<td><span class="cat-badge">' + cat + '</span></td>' +
      '<td style="text-align:center"><span class="signal-badge ' + sigCls + '">' + sig + '</span></td>' +
      '<td style="text-align:right;font-size:12px;color:#58a6ff">' + fmtP(c.entry_trigger) + '</td>' +
      '<td style="text-align:right;font-size:12px;color:#f85149">' + fmtP(c.stop_price) + '</td>' +
      '<td style="min-width:60px">' + scoreBar(c.bqs, bqsColor) + '</td>' +
      '<td style="min-width:60px">' + scoreBar(c.fws, fwsColor) + '</td>' +
      '<td style="min-width:60px">' + scoreBar(c.ncs, ncsColor) + '</td>' +
      '<td style="min-width:60px">' + scoreBar(qmScore, qmColor) +
           '<div style="font-size:9px;color:' + qmjColor + ';text-align:center;margin-top:2px">품질 ' + qmjLabel + '</div></td>' +
    '</tr>';
  }).join('');

  tbody.innerHTML = rows;
}


// ══════════════════════════════════════════════════════
// 🛡️ 시장 위험 면역 UI
// ══════════════════════════════════════════════════════

async function loadImmuneBanner() {
  // 홈 화면 배너용 빠른 체크 (SPY 기본)
  try {
    var r = await fetch('/api/market-immune?index=SPY');
    var d = await r.json();
    if (d.error) return;
    _renderImmuneBanner(d);
  } catch(e) { /* 배너는 실패해도 무시 */ }
}

function _renderImmuneBanner(d) {
  var bannerEl = document.getElementById('immune-banner');
  var inner    = document.getElementById('immune-banner-inner');
  var icon     = document.getElementById('immune-banner-icon');
  var title    = document.getElementById('immune-banner-title');
  var sub      = document.getElementById('immune-banner-sub');
  if (!bannerEl) return;

  var level = d.immune_level || 'CLEAR';
  var score = d.immune_score || 0;

  var cfg = {
    'CLEAR':   { cls:'immune-clear',   icon:'🟢', t:'시장 정상 (CLEAR)',      s:'신규 진입 허용 · 정상 운영' },
    'CAUTION': { cls:'immune-caution', icon:'🟡', t:'주의 단계 (CAUTION)',    s:'포지션 50% 축소 권고' },
    'ALERT':   { cls:'immune-alert',   icon:'🟠', t:'경보 단계 (ALERT)',      s:'신규 진입 극도 제한 · 수동 확인 필수' },
    'IMMUNE':  { cls:'immune-immune',  icon:'🔴', t:'면역 발동 (IMMUNE)',     s:'신규 매수 전면 금지 · 현금 비중 확대' },
  }[level] || { cls:'immune-clear', icon:'⚪', t:level, s:'' };

  if (icon)  icon.textContent   = cfg.icon;
  if (title) title.textContent  = cfg.t + ' (위험 점수 ' + score + '/100)';
  if (sub)   sub.textContent    = cfg.s;
  if (inner) { inner.className  = ''; inner.classList.add(cfg.cls); }
  if (bannerEl) bannerEl.style.display = '';
}

async function loadImmuneFull(force) {
  var idx    = (document.getElementById('immune-index') || {}).value || 'SPY';
  var loadEl = document.getElementById('immune-loading');
  var errEl  = document.getElementById('immune-error');
  var contEl = document.getElementById('immune-content');

  if (loadEl) loadEl.style.display = '';
  if (errEl)  errEl.style.display  = 'none';
  if (contEl) contEl.style.display = 'none';

  try {
    var r = await fetch('/api/market-immune?index=' + idx);
    var d = await r.json();

    if (d.error) {
      if (errEl) { errEl.textContent = '오류: ' + d.error; errEl.style.display = ''; }
      return;
    }

    _immuneLoaded = true;
    _renderImmuneFull(d);
    if (contEl) contEl.style.display = '';
    // 홈 배너도 갱신
    _renderImmuneBanner(d);

  } catch(e) {
    if (errEl) { errEl.textContent = '네트워크 오류: ' + e.message; errEl.style.display = ''; }
  } finally {
    if (loadEl) loadEl.style.display = 'none';
  }
}

function _renderImmuneFull(d) {
  var C = { green:'#3fb950', yellow:'#d29922', orange:'#f97316', red:'#f85149', blue:'#58a6ff', gray:'#8b949e' };
  var level = d.immune_level || 'CLEAR';
  var score = d.immune_score || 0;

  var levelCfg = {
    'CLEAR':   { cls:'immune-clear',   label:'🟢 정상 (CLEAR)',   desc:'신규 진입 허용 · 모든 전략 가동 가능' },
    'CAUTION': { cls:'immune-caution', label:'🟡 주의 (CAUTION)', desc:'포지션 사이즈 50% 축소 권고' },
    'ALERT':   { cls:'immune-alert',   label:'🟠 경보 (ALERT)',   desc:'신규 진입 극도 제한 · 수동 확인 필수' },
    'IMMUNE':  { cls:'immune-immune',  label:'🔴 면역 발동 (IMMUNE)', desc:'신규 매수 전면 금지 · 현금 비중 확대' },
  }[level] || { cls:'immune-clear', label:level, desc:'' };

  // 메인 카드
  var lvCard = document.getElementById('immune-level-card');
  if (lvCard) {
    lvCard.className = levelCfg.cls;
    lvCard.innerHTML =
      '<div style="font-size:28px;font-weight:900;margin-bottom:6px">' + levelCfg.label + '</div>' +
      '<div style="font-size:36px;font-weight:900;margin:8px 0">' + score + '<span style="font-size:16px;font-weight:400;opacity:.7">/100</span></div>' +
      '<div style="font-size:13px;opacity:.9">' + levelCfg.desc + '</div>' +
      '<div style="height:8px;background:rgba(0,0,0,.3);border-radius:4px;overflow:hidden;margin-top:14px;width:80%;margin-left:auto;margin-right:auto">' +
      '<div style="height:100%;width:' + score + '%;background:currentColor;border-radius:4px;transition:width .7s"></div></div>';
  }

  // 지표 그리드
  var metEl = document.getElementById('immune-metrics');
  if (metEl) {
    var metrics = [
      { label:'VIX', val: d.vix_level != null ? d.vix_level.toFixed(1) : '—',
        color: d.vix_level == null ? C.gray : d.vix_level >= 40 ? C.red : d.vix_level >= 30 ? C.orange : d.vix_level >= 20 ? C.yellow : C.green },
      { label:'MA200 이격도', val: d.ma200_deviation != null ? d.ma200_deviation.toFixed(1) + '%' : '—',
        color: d.ma200_deviation == null ? C.gray : d.ma200_deviation < -20 ? C.red : d.ma200_deviation < -10 ? C.orange : d.ma200_deviation < 0 ? C.yellow : C.green },
      { label:'지수 ATR%', val: d.vol_pct != null ? d.vol_pct.toFixed(2) + '%' : '—',
        color: d.vol_pct == null ? C.gray : d.vol_pct >= 5 ? C.red : d.vol_pct >= 3 ? C.orange : C.green },
      { label:'위험 점수', val: score + '/100', color: score >= 65 ? C.red : score >= 40 ? C.orange : score >= 20 ? C.yellow : C.green },
    ];
    metEl.innerHTML = metrics.map(function(m) {
      return '<div class="immune-metric"><div class="immune-metric-val" style="color:' + m.color + '">' + m.val + '</div><div class="immune-metric-label">' + m.label + '</div></div>';
    }).join('');
  }

  // 경고 목록
  var warnEl = document.getElementById('immune-warnings');
  if (warnEl) {
    var warnings = d.warnings || [];
    if (warnings.length > 0) {
      warnEl.innerHTML = '<div style="font-size:12px;font-weight:600;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">⚠️ 감지된 경고</div>' +
        '<div style="display:flex;flex-direction:column;gap:6px">' +
        warnings.map(function(w) {
          return '<div style="background:#2d1515;border:1px solid #4d1515;border-radius:8px;padding:8px 12px;font-size:12px;color:#f85149">' + w + '</div>';
        }).join('') + '</div>';
    } else {
      warnEl.innerHTML = '<div style="background:#0d2d1a;border:1px solid #1a4730;border-radius:8px;padding:10px 14px;font-size:12px;color:#3fb950">✓ 감지된 이상 신호 없음</div>';
    }
  }

  // Kill 스위치 권고
  var ksEl = document.getElementById('immune-killswitch');
  if (ksEl) {
    var ks = d.kill_switch || {};
    var items = [
      { key:'disable_new_entries',        label:'신규 진입 금지',   val: ks.disable_new_entries },
      { key:'disable_automated_entries',  label:'자동 진입 금지',   val: ks.disable_automated_entries },
      { key:'reduce_position_size',       label:'포지션 축소 권고', val: ks.reduce_position_size },
      { key:'require_manual_confirm',     label:'수동 확인 필수',   val: ks.require_manual_confirm },
    ];
    ksEl.innerHTML = '<div style="font-size:12px;font-weight:600;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">🔐 Kill 스위치 상태</div>' +
      '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px">' +
      items.map(function(item) {
        var on = item.val === true;
        var color = on ? '#f85149' : '#3fb950';
        var bg    = on ? '#2d0d0d' : '#0d2d1a';
        var border= on ? '#4d1515' : '#1a4730';
        return '<div style="background:' + bg + ';border:1px solid ' + border + ';border-radius:8px;padding:8px 12px;display:flex;justify-content:space-between;align-items:center">' +
          '<span style="font-size:12px;color:#cdd9e5">' + item.label + '</span>' +
          '<span style="font-size:12px;font-weight:700;color:' + color + '">' + (on ? '활성' : '비활성') + '</span></div>';
      }).join('') + '</div>';
  }

  // 과거 위기 유사도
  var crisisEl = document.getElementById('immune-crisis');
  if (crisisEl) {
    var matches = d.crisis_matches || [];
    if (matches.length === 0) { crisisEl.innerHTML = ''; return; }
    crisisEl.innerHTML = '<div style="font-size:12px;font-weight:600;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">📜 과거 위기 유사도 (상위 3개)</div>' +
      '<div style="display:flex;flex-direction:column;gap:8px">' +
      matches.slice(0,3).map(function(c) {
        var sim = c.similarity || 0;
        var sevColor = c.severity === 'EXTREME' ? C.red : c.severity === 'SEVERE' ? C.orange : c.severity === 'MODERATE' ? C.yellow : C.green;
        return '<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px">' +
          '<div style="display:flex;justify-content:space-between;margin-bottom:6px">' +
          '<span style="font-size:12px;font-weight:600;color:#cdd9e5">' + c.name + '</span>' +
          '<span style="font-size:11px;font-weight:700;color:' + sevColor + '">' + c.severity + '</span></div>' +
          '<div class="crisis-bar"><div class="crisis-bar-fill" style="width:' + sim + '%;background:' + sevColor + '"></div></div>' +
          '<div style="display:flex;justify-content:space-between;margin-top:4px">' +
          '<span style="font-size:10px;color:#484f58">유사도 ' + sim.toFixed(1) + '%</span>' +
          '<span style="font-size:10px;color:#f85149">최대 낙폭 ' + c.drawdown + '%</span></div></div>';
      }).join('') + '</div>';
  }
}

// ══════════════════════════════════════════════════════
// 🛡️ 시장 위험 면역 슬라이드 드로어 (한국 / 미국 공용 컴포넌트)
// ══════════════════════════════════════════════════════

// 시장별 설정 — 한국은 KOSPI 200(^KS200), 미국은 S&P 500(SPY) 기준 데이터
var MARKET_DRAWER_CFG = {
  KR: { flag:'🇰🇷', name:'한국 시장',  index:'^KS200', indexLabel:'KOSPI 200 기준' },
  US: { flag:'🇺🇸', name:'미국 시장',  index:'SPY',    indexLabel:'S&P 500 기준' },
};

function openMarketDrawer(market) {
  var cfg = MARKET_DRAWER_CFG[market] || MARKET_DRAWER_CFG.KR;
  var overlay = document.getElementById('market-drawer-overlay');
  var drawer  = document.getElementById('market-drawer');
  var titleEl = document.getElementById('market-drawer-title');
  var subEl   = document.getElementById('market-drawer-sub');
  var bodyEl  = document.getElementById('market-drawer-body');
  if (!drawer) return;

  // 상단에 현재 선택된 시장 명확히 표시
  if (titleEl) titleEl.textContent = cfg.flag + ' ' + cfg.name;
  if (subEl)   subEl.textContent   = '🛡️ 시장 위험 면역 시스템 · ' + cfg.indexLabel;

  // 로딩 표시
  if (bodyEl) bodyEl.innerHTML =
    '<div style="text-align:center;padding:40px;color:#8b949e">' +
    '<div class="spinner" style="margin:0 auto 12px"></div>시장 위험 데이터 수집 중...</div>';

  // 패널 열기
  if (overlay) overlay.classList.add('open');
  drawer.classList.add('open');
  document.body.style.overflow = 'hidden';   // 배경 스크롤 잠금

  // 시장별 독립 데이터 fetch (한국/미국 서로 다른 index 파라미터)
  fetch('/api/market-immune?index=' + encodeURIComponent(cfg.index))
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (!bodyEl) return;
      if (d.error) {
        bodyEl.innerHTML = '<div style="text-align:center;padding:32px;color:#f85149;font-size:13px">오류: ' + d.error + '</div>';
        return;
      }
      bodyEl.innerHTML = _buildImmuneHtml(d, cfg);
    })
    .catch(function(e){
      if (bodyEl) bodyEl.innerHTML = '<div style="text-align:center;padding:32px;color:#f85149;font-size:13px">네트워크 오류: ' + e.message + '</div>';
    });
}

function closeMarketDrawer() {
  var overlay = document.getElementById('market-drawer-overlay');
  var drawer  = document.getElementById('market-drawer');
  if (overlay) overlay.classList.remove('open');
  if (drawer)  drawer.classList.remove('open');
  document.body.style.overflow = '';
}

// ESC 키로 드로어 닫기
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    var drawer = document.getElementById('market-drawer');
    if (drawer && drawer.classList.contains('open')) closeMarketDrawer();
  }
});

// 면역 데이터 → HTML 문자열 빌더 (드로어 전용, 면역 페이지 렌더와 동일 구성 재사용)
function _buildImmuneHtml(d, cfg) {
  var C = { green:'#3fb950', yellow:'#d29922', orange:'#f97316', red:'#f85149', gray:'#8b949e' };
  var level = d.immune_level || 'CLEAR';
  var score = d.immune_score || 0;

  var levelCfg = {
    'CLEAR':   { cls:'immune-clear',   label:'🟢 정상 (CLEAR)',       desc:'신규 진입 허용 · 모든 전략 가동 가능' },
    'CAUTION': { cls:'immune-caution', label:'🟡 주의 (CAUTION)',     desc:'포지션 사이즈 50% 축소 권고' },
    'ALERT':   { cls:'immune-alert',   label:'🟠 경보 (ALERT)',       desc:'신규 진입 극도 제한 · 수동 확인 필수' },
    'IMMUNE':  { cls:'immune-immune',  label:'🔴 면역 발동 (IMMUNE)', desc:'신규 매수 전면 금지 · 현금 비중 확대' },
  }[level] || { cls:'immune-clear', label:level, desc:'' };

  var html = '';

  // 선택된 시장 헤더 (드로어 본문 상단에도 명확히 표기)
  if (cfg) {
    html += '<div style="font-size:13px;font-weight:700;color:#e6edf3;margin-bottom:12px">' +
            cfg.flag + ' ' + cfg.name +
            '<span style="font-size:11px;font-weight:400;color:#8b949e;margin-left:6px">(' + cfg.indexLabel + ')</span></div>';
  }

  // 메인 면역 레벨 카드
  html += '<div class="' + levelCfg.cls + '" style="border-radius:14px;padding:20px;border:2px solid;text-align:center;margin-bottom:16px">' +
    '<div style="font-size:24px;font-weight:900;margin-bottom:6px">' + levelCfg.label + '</div>' +
    '<div style="font-size:34px;font-weight:900;margin:8px 0">' + score + '<span style="font-size:15px;font-weight:400;opacity:.7">/100</span></div>' +
    '<div style="font-size:12px;opacity:.9">' + levelCfg.desc + '</div>' +
    '<div style="height:8px;background:rgba(0,0,0,.3);border-radius:4px;overflow:hidden;margin-top:14px;width:80%;margin-left:auto;margin-right:auto">' +
    '<div style="height:100%;width:' + score + '%;background:currentColor;border-radius:4px;transition:width .7s"></div></div></div>';

  // 지표 그리드
  var metrics = [
    { label:'VIX', val: d.vix_level != null ? d.vix_level.toFixed(1) : '—',
      color: d.vix_level == null ? C.gray : d.vix_level >= 40 ? C.red : d.vix_level >= 30 ? C.orange : d.vix_level >= 20 ? C.yellow : C.green },
    { label:'MA200 이격도', val: d.ma200_deviation != null ? d.ma200_deviation.toFixed(1) + '%' : '—',
      color: d.ma200_deviation == null ? C.gray : d.ma200_deviation < -20 ? C.red : d.ma200_deviation < -10 ? C.orange : d.ma200_deviation < 0 ? C.yellow : C.green },
    { label:'지수 ATR%', val: d.vol_pct != null ? d.vol_pct.toFixed(2) + '%' : '—',
      color: d.vol_pct == null ? C.gray : d.vol_pct >= 5 ? C.red : d.vol_pct >= 3 ? C.orange : C.green },
    { label:'위험 점수', val: score + '/100', color: score >= 65 ? C.red : score >= 40 ? C.orange : score >= 20 ? C.yellow : C.green },
  ];
  html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:16px">' +
    metrics.map(function(m) {
      return '<div class="immune-metric"><div class="immune-metric-val" style="color:' + m.color + '">' + m.val + '</div><div class="immune-metric-label">' + m.label + '</div></div>';
    }).join('') + '</div>';

  // 경고 목록
  var warnings = d.warnings || [];
  if (warnings.length > 0) {
    html += '<div style="font-size:12px;font-weight:600;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">⚠️ 감지된 경고</div>' +
      '<div style="display:flex;flex-direction:column;gap:6px;margin-bottom:16px">' +
      warnings.map(function(w) {
        return '<div style="background:#2d1515;border:1px solid #4d1515;border-radius:8px;padding:8px 12px;font-size:12px;color:#f85149">' + w + '</div>';
      }).join('') + '</div>';
  } else {
    html += '<div style="background:#0d2d1a;border:1px solid #1a4730;border-radius:8px;padding:10px 14px;font-size:12px;color:#3fb950;margin-bottom:16px">✓ 감지된 이상 신호 없음</div>';
  }

  // Kill 스위치 권고
  var ks = d.kill_switch || {};
  var ksItems = [
    { label:'신규 진입 금지',   val: ks.disable_new_entries },
    { label:'자동 진입 금지',   val: ks.disable_automated_entries },
    { label:'포지션 축소 권고', val: ks.reduce_position_size },
    { label:'수동 확인 필수',   val: ks.require_manual_confirm },
  ];
  html += '<div style="font-size:12px;font-weight:600;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">🔐 Kill 스위치 상태</div>' +
    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px">' +
    ksItems.map(function(item) {
      var on = item.val === true;
      var color = on ? '#f85149' : '#3fb950';
      var bg    = on ? '#2d0d0d' : '#0d2d1a';
      var border= on ? '#4d1515' : '#1a4730';
      return '<div style="background:' + bg + ';border:1px solid ' + border + ';border-radius:8px;padding:8px 10px;display:flex;justify-content:space-between;align-items:center;gap:6px">' +
        '<span style="font-size:11px;color:#cdd9e5">' + item.label + '</span>' +
        '<span style="font-size:11px;font-weight:700;color:' + color + '">' + (on ? '활성' : '비활성') + '</span></div>';
    }).join('') + '</div>';

  // 과거 위기 유사도 (상위 3개)
  var matches = d.crisis_matches || [];
  if (matches.length > 0) {
    html += '<div style="font-size:12px;font-weight:600;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">📜 과거 위기 유사도 (상위 3개)</div>' +
      '<div style="display:flex;flex-direction:column;gap:8px">' +
      matches.slice(0,3).map(function(c) {
        var sim = c.similarity || 0;
        var sevColor = c.severity === 'EXTREME' ? C.red : c.severity === 'SEVERE' ? C.orange : c.severity === 'MODERATE' ? C.yellow : C.green;
        return '<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px">' +
          '<div style="display:flex;justify-content:space-between;margin-bottom:6px">' +
          '<span style="font-size:12px;font-weight:600;color:#cdd9e5">' + c.name + '</span>' +
          '<span style="font-size:11px;font-weight:700;color:' + sevColor + '">' + c.severity + '</span></div>' +
          '<div class="crisis-bar"><div class="crisis-bar-fill" style="width:' + sim + '%;background:' + sevColor + '"></div></div>' +
          '<div style="display:flex;justify-content:space-between;margin-top:4px">' +
          '<span style="font-size:10px;color:#484f58">유사도 ' + sim.toFixed(1) + '%</span>' +
          '<span style="font-size:10px;color:#f85149">최대 낙폭 ' + c.drawdown + '%</span></div></div>';
      }).join('') + '</div>';
  }

  return html;
}

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
            # 1시간 캐시 + 1일 stale (스크리너는 실시간성 불필요)
            cache_control = "public, s-maxage=3600, stale-while-revalidate=86400"

        elif path == "/api/market/summary":
            # 시장 현황 — 가변 TTL (시장 급변 대응)
            # 장중(KST 09:00~15:30): 2분  → 지수·환율 분 단위 변동 반영
            # 미국 정규장(KST 22:30~05:00): 3분 → 간밤 지표 반영
            # 장 외(야간·주말): 8분         → 데이터 거의 고정, 비용 절감
            # stale-while-revalidate: CDN이 만료 후에도 즉시 반환 + 백그라운드 갱신
            try:
                from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                _KST = _tz(timedelta(hours=9))
                _now = _dt.now(_KST)
                _h   = _now.hour + _now.minute / 60.0
                _wd  = _now.weekday()
                if _wd >= 5:                      # 주말
                    _smx, _swr = 480, 900
                elif 9.0 <= _h < 15.5:            # 국내 정규장
                    _smx, _swr = 120, 300
                elif 22.5 <= _h or _h < 5.0:      # 미국 정규장(EDT)
                    _smx, _swr = 180, 360
                else:                              # 장 외 (장전·장후·심야)
                    _smx, _swr = 480, 900
            except Exception:
                _smx, _swr = 180, 360              # 계산 실패 시 3분 기본값
            cache_control = f"public, s-maxage={_smx}, stale-while-revalidate={_swr}"

        elif path == "/api/market/sector-summary":
            # 섹터 흐름 — 장중 5분, 장 외 15분
            try:
                from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                _KST = _tz(timedelta(hours=9))
                _now = _dt.now(_KST)
                _h   = _now.hour + _now.minute / 60.0
                _wd  = _now.weekday()
                if _wd >= 5 or not (9.0 <= _h < 15.5):
                    _smx, _swr = 900, 1800         # 장 외: 15분
                else:
                    _smx, _swr = 300, 600          # 장중: 5분
            except Exception:
                _smx, _swr = 300, 600
            cache_control = f"public, s-maxage={_smx}, stale-while-revalidate={_swr}"

        elif path == "/api/stock":
            # 종목 분석 — 장중 60초, 장 외 5분
            cache_control = "public, s-maxage=60, stale-while-revalidate=300"

        elif path == "/" or path == "/index.html":
            # HTML — 1시간 캐시 (정적에 가깝고 자주 바뀌지 않음)
            cache_control = "public, s-maxage=3600"

        elif path.startswith("/api/"):
            # 기타 API — 10초 캐시 (기본값 유지)
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


# ── 서버 시작 시 랭킹 캐시 예열 (백그라운드) ────────────────────────────────
# 첫 해외 주식 검색 시 랭킹 API 병렬 호출 대기(~5s)를 없앤다.
# 모듈 로드 직후 백그라운드 스레드로 캐시를 미리 채운다.
def _prewarm_toss_ranking():
    try:
        result = _fetch_toss_us_ranking_productcodes()
        print(f"[Toss Prewarm] 랭킹 캐시 예열 완료: {len(result)}개")
    except Exception as e:
        print(f"[Toss Prewarm] 예열 실패 (무시): {e}")

_prewarm_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="toss_prewarm")
_prewarm_pool.submit(_prewarm_toss_ranking)


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
