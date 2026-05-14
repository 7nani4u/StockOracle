#!/usr/bin/env python3
"""
us_price_fetcher.py  (StockOracle 프로젝트용)
================================================================================
KST 기준 언제 실행해도 항상 가장 최신·정확한 미국 주가를 반환합니다.
"Overnight"(심야 연장 거래) 가격 지원이 추가되었습니다.

[Yahoo Finance 세션 구분 — marketState 기반]
  ┌─────────────┬──────────────┬────────────────────────────────────────────────┐
  │ marketState │ ET 시간대    │ Yahoo Finance 표시                             │
  ├─────────────┼──────────────┼────────────────────────────────────────────────┤
  │ PREPRE      │ 00:00~04:00  │ "Overnight" 레이블 — 심야 Cboe EDGX 연장 거래  │
  │ PRE         │ 04:00~09:30  │ "Before hours" / Pre-market                    │
  │ REGULAR     │ 09:30~16:00  │ 정규장 실시간                                   │
  │ POST        │ 16:00~20:00  │ "After hours" / Post-market                    │
  │ POSTPOST    │ 20:00~00:00  │ "Overnight" 레이블 — 심야 연장 거래             │
  │ CLOSED      │ 주말/공휴일  │ 완전 마감 (거래 없음)                           │
  └─────────────┴──────────────┴────────────────────────────────────────────────┘

[KST 기준 Overnight 발생 구간]
  KST 06:00 ≈ ET 21:00 (EDT) → POSTPOST (Overnight)
  KST 13:00 ≈ ET 04:00 (EDT) → PREPRE 끝 / PRE 시작
  → KST 06:00~22:29 중 약 06:00~13:00 구간이 Yahoo Overnight 표시

[API 우선순위]
  ┌──────────────┬──────────────────────────────────────────────────────────────┐
  │ Session      │ 우선순위                                                     │
  ├──────────────┼──────────────────────────────────────────────────────────────┤
  │ OVERNIGHT    │ YFDirect(postMarket) → yfinance(postMarket) → 당일종가       │
  │ PRE_MARKET   │ yfinance(preMarket)  → Tiingo(tngoLast)     → 전일종가       │
  │ REGULAR      │ Tiingo IEX(last)     → yfinance(current)    → Finnhub → AV  │
  │ AFTER_HOURS  │ yfinance(postMarket) → Tiingo(tngoLast)     → 당일종가       │
  │ CLOSED       │ yfinance(prevClose)  → Finnhub(pc) → Tiingo EOD → AV        │
  └──────────────┴──────────────────────────────────────────────────────────────┘

[시간대]
  KST(UTC+9) ↔ ET(America/New_York, DST 자동)
  EDT(3~11월): 정규장 22:30~05:00 KST
  EST(11~3월): 정규장 23:30~06:00 KST

[API 키]
  Finnhub   : d7lm0o9r01qm7o0cb440d7lm0o9r01qm7o0cb44g
  Tiingo    : 12ebd1feef89b6728cc15808864b7402449a5637
  AV        : E0ODFSRNDU4P9HDU

Dependencies:
  pip install requests yfinance
"""

from __future__ import annotations

import os
import time
import logging
import warnings
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False
    warnings.warn("yfinance 미설치 — pip install yfinance 권장 (Extended Hours 정확도 저하)")

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# API 키 (환경변수 우선, 없으면 하드코딩 기본값 사용)
# ──────────────────────────────────────────────────────────────────────────────
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY",  "d7lm0o9r01qm7o0cb440d7lm0o9r01qm7o0cb44g")
TIINGO_KEY  = os.getenv("TIINGO_API_KEY",   "12ebd1feef89b6728cc15808864b7402449a5637")
AV_KEY      = os.getenv("ALPHAVANTAGE_KEY", "E0ODFSRNDU4P9HDU")

ET_TZ  = ZoneInfo("America/New_York")
KST_TZ = ZoneInfo("Asia/Seoul")

REQUEST_TIMEOUT = 10  # seconds

# ──────────────────────────────────────────────────────────────────────────────
# NYSE 공휴일 / 조기 종료일 (2025–2028)
# ──────────────────────────────────────────────────────────────────────────────
NYSE_HOLIDAYS: frozenset[date] = frozenset({
    # 2025
    date(2025, 1, 1),  date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4),  date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1),  date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3),  date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    # 2027
    date(2027, 1, 1),  date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5),  date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
    # 2028
    date(2028, 1, 17), date(2028, 2, 21), date(2028, 4, 14),
    date(2028, 5, 29), date(2028, 6, 19), date(2028, 7, 4),  date(2028, 9, 4),
    date(2028, 11, 23), date(2028, 12, 25),
})
NYSE_EARLY_CLOSE: frozenset[date] = frozenset({
    date(2025, 11, 28), date(2025, 12, 24),
    date(2026, 11, 27), date(2026, 12, 24),
    date(2027, 11, 26), date(2027, 12, 24),
    date(2028, 11, 24), date(2028, 12, 24),
})

# Yahoo Finance marketState 값 중 "Overnight" 레이블에 해당하는 상태들
# PREPRE: 자정~04:00 ET / POSTPOST: 20:00~자정 ET
_YF_OVERNIGHT_STATES: frozenset[str] = frozenset({"PREPRE", "POSTPOST"})
# Overnight + After-hours 합산 (postMarketPrice가 유효한 모든 extended 상태)
_YF_EXTENDED_STATES:  frozenset[str] = frozenset({"PREPRE", "POSTPOST", "POST", "PRE"})


# ──────────────────────────────────────────────────────────────────────────────
# 세션 Enum + PriceResult 데이터 클래스
# ──────────────────────────────────────────────────────────────────────────────
class MarketSession(Enum):
    PRE_MARKET  = "pre_market"
    REGULAR     = "regular"
    AFTER_HOURS = "after_hours"
    OVERNIGHT   = "overnight"   # ← 신규: 20:00~04:00 ET (Yahoo "Overnight" 레이블)
    CLOSED      = "closed"      # 주말·공휴일 완전 마감

    @property
    def is_extended(self) -> bool:
        """정규장 외 거래 세션 여부."""
        return self in (
            MarketSession.PRE_MARKET,
            MarketSession.AFTER_HOURS,
            MarketSession.OVERNIGHT,
        )

    def label_ko(self) -> str:
        return {
            "pre_market":  "프리마켓",
            "regular":     "정규장",
            "after_hours": "애프터마켓",
            "overnight":   "오버나이트",
            "closed":      "장마감",
        }[self.value]


@dataclass
class PriceResult:
    ticker:      str
    price:       float          # 현재가 (세션에 맞는 최신가)
    prev_close:  float          # 전일 정규장 종가
    change:      float          # 전일 종가 대비 변동액
    change_pct:  float          # 전일 종가 대비 변동률 (%)
    session:     MarketSession
    price_type:  str            # "real_time" | "pre_market" | "post_market" |
                                # "overnight" | "last_close"
    source:      str            # 데이터 출처
    price_time:  Optional[datetime]
    fetch_time:  datetime = field(default_factory=lambda: datetime.now(KST_TZ))
    notes:       str = ""

    def __str__(self) -> str:
        G, R, RST = "\033[32m", "\033[31m", "\033[0m"
        clr  = G if self.change >= 0 else R
        sign = "+" if self.change >= 0 else ""
        arr  = "▲" if self.change >= 0 else "▼"
        t_str = (self.price_time.strftime("%Y-%m-%d %H:%M:%S %Z")
                 if self.price_time else "N/A")
        session_lbl = self.session.label_ko()
        # Overnight 세션은 강조 표시
        if self.session == MarketSession.OVERNIGHT:
            session_lbl = f"🌙 {session_lbl} (Yahoo Overnight)"
        lines = [
            f"┌─ {self.ticker} {'─'*40}",
            f"│  가격     : {clr}${self.price:>10,.4f}  "
            f"{arr} {sign}{self.change:,.4f} ({sign}{self.change_pct:.2f}%){RST}",
            f"│  전일종가 : ${self.prev_close:>10,.4f}",
            f"│  세션     : {session_lbl} ({self.session.value})",
            f"│  가격유형 : {self.price_type}",
            f"│  출처     : {self.source}",
            f"│  가격시각 : {t_str}",
            f"│  조회시각 : {self.fetch_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        ]
        if self.notes:
            lines.append(f"│  비고     : {self.notes}")
        lines.append("└" + "─" * 47)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker":     self.ticker,
            "price":      self.price,
            "prev_close": self.prev_close,
            "change":     self.change,
            "change_pct": self.change_pct,
            "session":    self.session.value,
            "price_type": self.price_type,
            "source":     self.source,
            "price_time": self.price_time.isoformat() if self.price_time else None,
            "fetch_time": self.fetch_time.isoformat(),
            "notes":      self.notes,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 세션 감지 유틸리티
# ──────────────────────────────────────────────────────────────────────────────
def now_et()  -> datetime: return datetime.now(ET_TZ)
def now_kst() -> datetime: return datetime.now(KST_TZ)


def is_trading_day(dt: datetime) -> bool:
    d = dt.astimezone(ET_TZ).date()
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def detect_session(dt: Optional[datetime] = None) -> Tuple[MarketSession, datetime]:
    """
    현재(또는 지정) 시각의 NYSE 세션 반환.

    [세션 경계 — ET 기준]
      00:00 ~ 04:00  → OVERNIGHT  (Yahoo "Overnight": PREPRE 상태)
      04:00 ~ 09:30  → PRE_MARKET
      09:30 ~ 16:00  → REGULAR    (조기마감일: 13:00)
      16:00 ~ 20:00  → AFTER_HOURS
      20:00 ~ 24:00  → OVERNIGHT  (Yahoo "Overnight": POSTPOST 상태)
      주말/공휴일    → CLOSED

    ※ OVERNIGHT 세션은 거래일 기준: 전일 20:00 ~ 당일 04:00
       비거래일(주말/공휴일)의 자정~04:00은 CLOSED로 분류
    """
    dt_et = dt.astimezone(ET_TZ) if dt else now_et()

    t = dt_et.hour * 60 + dt_et.minute  # 분 단위 시각

    # ── 전날 20:00 이후 ~ 오늘 04:00 이전: 자정 넘긴 overnight 처리 ──────────
    # 자정~04:00(분 기준 0~239)이면 "오늘" 또는 "어제"가 거래일인지 확인
    #
    # 수정 이유: 월요일 00:00~04:00 ET 에서 "어제"(일요일)는 비거래일이므로
    # 이전 로직이 CLOSED 를 반환하는 버그가 있었음.
    # 수정 로직: 오늘이 거래일이면 OVERNIGHT (Blue Ocean ATS 활성), 그 다음 어제 확인.
    if t < 240:  # 00:00 ~ 03:59
        if is_trading_day(dt_et):
            # 오늘이 거래일(월~금) → 전날 밤 ATS 연장 세션 (예: 월요일 자정~04:00)
            return MarketSession.OVERNIGHT, dt_et
        yesterday = (dt_et - timedelta(days=1)).astimezone(ET_TZ)
        if is_trading_day(yesterday):
            # 어제가 거래일 → 어제 밤 ATS 연장 세션 (예: 금요일 → 토요일 자정)
            return MarketSession.OVERNIGHT, dt_et
        # 어제·오늘 모두 비거래일 (예: 일요일 자정) → CLOSED
        return MarketSession.CLOSED, dt_et

    # 04:00 이후는 당일 거래일 여부 체크
    if not is_trading_day(dt_et):
        return MarketSession.CLOSED, dt_et

    reg_end = 780 if dt_et.date() in NYSE_EARLY_CLOSE else 960  # 13:00 or 16:00

    if   240  <= t < 570:      return MarketSession.PRE_MARKET,  dt_et  # 04:00~09:30
    elif 570  <= t < reg_end:  return MarketSession.REGULAR,     dt_et  # 09:30~16:00
    elif reg_end <= t < 1200:  return MarketSession.AFTER_HOURS, dt_et  # 16:00~20:00
    else:                      return MarketSession.OVERNIGHT,   dt_et  # 20:00~24:00


def session_info() -> Dict[str, str]:
    dt_e = now_et(); dt_k = now_kst()
    sess, _ = detect_session(dt_e)
    return {
        "kst_time":       dt_k.strftime("%Y-%m-%d %H:%M:%S KST"),
        "et_time":        dt_e.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "session":        sess.value,
        "session_ko":     sess.label_ko(),
        "is_trading_day": str(is_trading_day(dt_e)),
        "dst_active":     "EDT(서머타임)" if dt_e.tzname() == "EDT" else "EST(표준시)",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 공통 HTTP 클라이언트
# ──────────────────────────────────────────────────────────────────────────────
class _BaseClient:
    _min_interval: float = 1.0

    def __init__(self):
        self._last = 0.0

    def _wait(self):
        gap = self._min_interval - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)

    def _get(self, url: str, params: Dict, headers: Dict = None, label="API") -> Optional[Any]:
        self._wait()
        kw = dict(params=params, headers=headers or {}, timeout=REQUEST_TIMEOUT)
        try:
            r = requests.get(url, verify=True,  **kw); r.raise_for_status()
            self._last = time.monotonic(); return r.json()
        except requests.exceptions.SSLError:
            try:
                r = requests.get(url, verify=False, **kw); r.raise_for_status()
                self._last = time.monotonic(); return r.json()
            except Exception as e:
                logger.warning(f"[{label}] 요청 실패: {e}"); return None
        except Exception as e:
            logger.warning(f"[{label}] 요청 실패: {e}"); return None


# ──────────────────────────────────────────────────────────────────────────────
# Yahoo Finance Direct Client  ← 신규 추가
# ──────────────────────────────────────────────────────────────────────────────
class YahooFinanceDirectClient(_BaseClient):
    """
    Yahoo Finance v7/quote API 직접 호출 — 세션/Crumb 인증 포함.

    [인증 방식]
      Yahoo Finance v7 API는 crumb 없이 호출하면 401을 반환합니다.
      1. requests.Session으로 https://finance.yahoo.com/ 방문 → 쿠키 획득
      2. https://query2.finance.yahoo.com/v1/test/csrfToken 호출 → crumb 획득
      3. 이후 모든 v7 호출에 ?crumb=XXX 파라미터 포함
      4. 401 발생 시 crumb 갱신 후 1회 재시도

    [Overnight 감지 로직]
      1. marketState in ("PREPRE", "POSTPOST") → Yahoo "Overnight" 상태 확정
      2. postMarketPrice 존재 + postMarketTime이 8시간 이내 → Overnight 데이터 유효
      3. 위 조건 불충족 → None 반환 (상위 fallback으로)

    [캐시]
      crumb: 30분 / quote: cache_ttl 초 (기본 30초)
    """

    _CONSENT_URL = "https://finance.yahoo.com/"
    _CRUMB_URL   = "https://query2.finance.yahoo.com/v1/test/csrfToken"
    _QUOTE_URLS: List[str] = [
        "https://query2.finance.yahoo.com/v7/finance/quote",
        "https://query1.finance.yahoo.com/v7/finance/quote",
    ]
    _CRUMB_TTL = 1800  # 30분

    _HEADERS: Dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://finance.yahoo.com/",
        "Origin":          "https://finance.yahoo.com",
    }

    # 조회할 필드 목록 (Yahoo Finance v7 quote API)
    _FIELDS: str = ",".join([
        "regularMarketPrice",
        "regularMarketPreviousClose",
        "regularMarketTime",
        "regularMarketChange",
        "regularMarketChangePercent",
        # ── Overnight (Blue Ocean ATS, 8 PM~4 AM ET) ──────────────────────
        # overnightPrice=true 파라미터를 함께 전달해야 이 필드가 반환됨
        "overnightMarketPrice",        # Blue Ocean ATS 심야 연장 거래 최신가
        "overnightMarketTime",         # 해당 가격 타임스탬프
        "overnightMarketChange",       # 전일 정규장 종가 대비 변동액
        "overnightMarketChangePercent",# 전일 정규장 종가 대비 변동률
        # ── After-hours (POST, 4 PM~8 PM ET) ──────────────────────────────
        "postMarketPrice",             # After-hours 체결가
        "postMarketTime",
        "postMarketChange",
        "postMarketChangePercent",
        # ── Pre-market (PRE, 4 AM~9:30 AM ET) ────────────────────────────
        "preMarketPrice",
        "preMarketTime",
        "preMarketChange",
        "preMarketChangePercent",
        "marketState",               # PRE|REGULAR|POST|PREPRE|POSTPOST|CLOSED
        "shortName",
    ])
    _min_interval = 0.5

    def __init__(self, cache_ttl: int = 30):
        super().__init__()
        # 쿠키 유지 세션 (crumb 인증에 필수)
        self._sess = requests.Session()
        self._sess.headers.update(self._HEADERS)
        self._crumb:    Optional[str] = None
        self._crumb_at: float         = 0.0
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = cache_ttl

    def _ts(self, unix: Any) -> Optional[datetime]:
        """Unix timestamp → timezone-aware datetime (ET)."""
        if not unix:
            return None
        try:
            return datetime.fromtimestamp(float(unix), tz=ET_TZ)
        except Exception:
            return None

    def _refresh_crumb(self) -> bool:
        """
        crumb 갱신.

        [우선순위]
          1) yfinance 내부 YfData (SingletonMeta — 이미 쿠키/crumb 보유 시 즉시 반환)
             → yfinance가 관리하는 세션/쿠키를 그대로 재사용하므로 가장 신뢰도 높음
          2) 직접 HTTP: finance.yahoo.com 방문 → csrfToken 엔드포인트 호출 (백업)
        """
        # ── 방법 1: yfinance 내부 YfData 세션 공유 (권장) ──────────────────
        if _HAS_YFINANCE:
            try:
                from yfinance.data import YfData
                yfd_inst = YfData()          # SingletonMeta → 공유 인스턴스
                yfd_inst._get_cookie_and_crumb()
                if yfd_inst._crumb and yfd_inst._session:
                    self._crumb    = yfd_inst._crumb
                    self._crumb_at = time.monotonic()
                    self._sess     = yfd_inst._session  # crumb 설정된 세션 공유
                    logger.debug(
                        f"[YFDirect] yfinance crumb 획득 ({self._crumb[:8]}...)"
                    )
                    return True
            except Exception as e:
                logger.debug(f"[YFDirect] yfinance crumb 실패: {e}")

        # ── 방법 2: 직접 HTTP (백업) ────────────────────────────────────────
        try:
            self._sess.get(
                self._CONSENT_URL,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            r = self._sess.get(self._CRUMB_URL, timeout=REQUEST_TIMEOUT)
            if r.ok:
                try:
                    token = r.json().get("csrfToken", "")
                except Exception:
                    token = r.text.strip()
                if token and len(token) > 2:
                    self._crumb    = token
                    self._crumb_at = time.monotonic()
                    logger.debug(f"[YFDirect] 직접 crumb 획득 ({token[:8]}...)")
                    return True
            logger.warning(f"[YFDirect] csrfToken HTTP {r.status_code}")
        except Exception as e:
            logger.debug(f"[YFDirect] 직접 crumb 획득 실패: {e}")

        return False

    def _ensure_crumb(self) -> bool:
        """crumb이 없거나 만료(30분)되면 갱신 시도."""
        if self._crumb and (time.monotonic() - self._crumb_at) < self._CRUMB_TTL:
            return True
        return self._refresh_crumb()

    def fetch_raw_quote(self, ticker: str) -> Optional[dict]:
        """
        Yahoo Finance v7 quote API → 원시 quote dict (crumb 인증 포함).
        캐시 TTL 내 재호출은 캐시값 반환.
        """
        ck = ticker.upper()
        if ck in self._cache:
            cached_data, cached_at = self._cache[ck]
            if time.monotonic() - cached_at < self._cache_ttl:
                logger.debug(f"[YFDirect] {ticker}: 캐시 반환")
                return cached_data

        if not self._ensure_crumb():
            logger.warning("[YFDirect] crumb 없음 — YFDirect 스킵, fallback 진행")
            return None

        params = {
            "symbols":        ticker,
            "fields":         self._FIELDS,
            "crumb":          self._crumb,
            "overnightPrice": "true",   # overnightMarketPrice 필드 활성화
        }

        for url in self._QUOTE_URLS:
            self._wait()
            try:
                r = self._sess.get(url, params=params, timeout=REQUEST_TIMEOUT)

                if r.status_code == 401:
                    # crumb 만료 → 갱신 후 1회 재시도
                    logger.info("[YFDirect] 401 수신 → crumb 재발급 후 재시도")
                    self._crumb = None
                    if not self._refresh_crumb():
                        logger.warning("[YFDirect] crumb 재발급 실패 → 다음 fallback")
                        break
                    params["crumb"] = self._crumb
                    r = self._sess.get(url, params=params, timeout=REQUEST_TIMEOUT)

                r.raise_for_status()
                results = r.json().get("quoteResponse", {}).get("result", [])
                self._last = time.monotonic()
                if results:
                    q = results[0]
                    self._cache[ck] = (q, time.monotonic())
                    logger.debug(
                        f"[YFDirect] {ticker}: 조회 성공 "
                        f"(marketState={q.get('marketState', 'N/A')})"
                    )
                    return q
                logger.warning(f"[YFDirect] {ticker}: 응답 성공이나 result 비어 있음")
            except requests.exceptions.HTTPError as e:
                logger.warning(f"[YFDirect] {ticker} HTTP 오류 ({url}): {e}")
            except Exception as e:
                logger.debug(f"[YFDirect] {ticker} ({url}): {e}")

        logger.warning(f"[YFDirect] {ticker}: 모든 엔드포인트 실패")
        return None

    def get_overnight_price(
        self, ticker: str
    ) -> Optional[Tuple[float, float, Optional[datetime], str, str]]:
        """
        Yahoo Finance Overnight(Blue Ocean ATS) 가격 추출.

        반환: (price, prev_close, price_time, price_type, market_state)
              또는 None (데이터 없음·만료·비거래)

        [우선순위]
          ① overnightMarketPrice (Blue Ocean ATS 실제 체결가, 8 PM~4 AM ET)
             - overnightPrice=true 파라미터로 활성화
             - marketState 무관하게 8시간 이내 데이터이면 사용
          ② postMarketPrice      (애프터마켓 종가, PREPRE/POSTPOST 상태에서 fallback)
             - Blue Ocean ATS 데이터가 없을 때 보조 수단

        [타임스탬프 신선도 기준]
          overnightMarketTime 이 현재 시각으로부터 8시간 이내인 경우만 유효
        """
        q = self.fetch_raw_quote(ticker)
        if not q:
            return None

        prev_close   = float(q.get("regularMarketPreviousClose") or 0)
        market_state = q.get("marketState", "")

        # ── ① overnightMarketPrice (Blue Ocean ATS) ──────────────────────────
        # marketState 상관없이: 타임스탬프가 8시간 이내면 유효한 overnight 데이터
        ovn_price = q.get("overnightMarketPrice")
        ovn_time  = self._ts(q.get("overnightMarketTime"))

        if ovn_price and ovn_time:
            age_min = (now_et() - ovn_time).total_seconds() / 60
            if age_min <= 480:  # 8시간 이내
                logger.info(
                    f"[YFDirect] {ticker} Overnight(Blue Ocean ATS) ✓ "
                    f"marketState={market_state} | "
                    f"price={ovn_price} | "
                    f"time={ovn_time.strftime('%H:%M:%S %Z')} | "
                    f"{age_min:.0f}min ago"
                )
                return (
                    float(ovn_price),
                    prev_close,
                    ovn_time,
                    "overnight",
                    market_state,
                )
            logger.debug(
                f"[YFDirect] {ticker}: overnightMarketPrice 만료 "
                f"({age_min:.0f}min > 480min)"
            )

        # ── ② postMarketPrice fallback (PREPRE/POSTPOST 전용) ─────────────
        if market_state in _YF_OVERNIGHT_STATES:
            post_price = q.get("postMarketPrice")
            post_time  = self._ts(q.get("postMarketTime"))
            if post_price and post_time:
                age_min = (now_et() - post_time).total_seconds() / 60
                if age_min <= 480:
                    logger.info(
                        f"[YFDirect] {ticker} postMarketPrice fallback "
                        f"(overnightMarketPrice 없음) | "
                        f"price={post_price} | {age_min:.0f}min ago"
                    )
                    return (
                        float(post_price),
                        prev_close,
                        post_time,
                        "overnight",
                        market_state,
                    )

        logger.debug(
            f"[YFDirect] {ticker}: Overnight 가격 없음 "
            f"(marketState={market_state}, overnightPrice={ovn_price})"
        )
        return None

    def get_premarket_price(
        self, ticker: str
    ) -> Optional[Tuple[float, float, Optional[datetime], str, str]]:
        """
        Pre-market 가격 추출.

        [우선순위]
          ① preMarketPrice        : 04:00~09:30 ET 프리마켓 가격 (가장 최신)
          ② overnightMarketPrice  : Blue Ocean ATS 종료 직전 최종가 (4 AM 이후 유효)

        ※ marketState 가 정확히 "PRE" 가 아니어도 preMarketPrice 필드가 있으면 반환합니다.
           (세션 경계 시점에 marketState 가 아직 업데이트되지 않을 수 있음)

        반환: (price, prev_close, price_time, price_type, market_state) 또는 None
        """
        q = self.fetch_raw_quote(ticker)
        if not q:
            return None

        prev_close   = float(q.get("regularMarketPreviousClose") or 0)
        market_state = q.get("marketState", "")

        # ── ① preMarketPrice ─────────────────────────────────────────────────
        pre_price = q.get("preMarketPrice")
        pre_time  = self._ts(q.get("preMarketTime"))

        if pre_price and pre_time:
            age_min = (now_et() - pre_time).total_seconds() / 60
            if age_min <= 360:  # 6시간 이내 (04:00~09:30 ET 커버)
                logger.info(
                    f"[YFDirect] {ticker} Pre-market ✓ "
                    f"marketState={market_state} | "
                    f"price={pre_price} | "
                    f"time={pre_time.strftime('%H:%M:%S %Z')} | "
                    f"{age_min:.0f}min ago"
                )
                return float(pre_price), prev_close, pre_time, "pre_market", market_state

        # marketState 가 PRE 가 아니면 overnightMarketPrice fallback 도 스킵
        if market_state not in ("PRE", "PREPRE"):
            return None

        # ── ② overnightMarketPrice fallback (PRE 세션 직전 Blue Ocean ATS) ──
        ovn_price = q.get("overnightMarketPrice")
        ovn_time  = self._ts(q.get("overnightMarketTime"))
        if ovn_price and ovn_time:
            age_min = (now_et() - ovn_time).total_seconds() / 60
            if age_min <= 480:
                logger.info(
                    f"[YFDirect] {ticker} overnightMarketPrice fallback (PRE) "
                    f"price={ovn_price} | {age_min:.0f}min ago"
                )
                return float(ovn_price), prev_close, ovn_time, "overnight", market_state

        return None

    def get_postmarket_price(
        self, ticker: str
    ) -> Optional[Tuple[float, float, Optional[datetime], str, str]]:
        """
        After-hours 가격 추출 (marketState == "POST").

        반환: (price, prev_close, price_time, price_type, market_state)
              또는 None
        """
        q = self.fetch_raw_quote(ticker)
        if not q:
            return None

        prev_close   = float(q.get("regularMarketPreviousClose") or 0)
        market_state = q.get("marketState", "")

        if market_state not in ("POST", *_YF_OVERNIGHT_STATES):
            return None

        post_price = q.get("postMarketPrice")
        post_time  = self._ts(q.get("postMarketTime"))

        if post_price and post_time:
            age_min = (now_et() - post_time).total_seconds() / 60
            if age_min <= 240:  # 4시간 이내
                return float(post_price), prev_close, post_time, "post_market", market_state

        return None


# ──────────────────────────────────────────────────────────────────────────────
# 네이버 모바일 증권 해외주식 클라이언트  ← 신규 추가
# ──────────────────────────────────────────────────────────────────────────────
class NaverWorldStockClient(_BaseClient):
    """
    네이버 모바일 증권 해외주식 현재가 클라이언트.

    API 엔드포인트: https://api.stock.naver.com/stock/{reuters_code}/basic
    reuters_code 형식: 심볼 + 거래소 접미사 (예: AIIO.O, AAPL.O, TSLA.N)

    [거래소 접미사]
      .O = NASDAQ (NSQ)
      .N = NYSE   (NYQ/NYS)
      .A = AMEX   (ASE)
      접미사 없이 입력 시 .O → .N → .A 순으로 자동 탐색

    [네이버 현재가 표시 로직]
      marketStatus == "OPEN"
        → closePrice (정규장 실시간 체결가)
      overMarketPriceInfo.overMarketStatus == "OPEN"
        → overMarketPriceInfo.overPrice (프리/애프터마켓 가격)
      그 외 (장마감)
        → closePrice (마지막 정규장 종가)

    [세션 커버리지]
      PRE_MARKET  ✓  (overMarketPriceInfo.tradingSessionType == "PRE_MARKET")
      REGULAR     ✓  (marketStatus == "OPEN")
      AFTER_HOURS ✓  (overMarketPriceInfo.tradingSessionType == "AFTER_MARKET")
      CLOSED      ✓  (closePrice = 마지막 정규장 종가)
      OVERNIGHT   ✗  (Blue Ocean ATS 미지원 → 항상 None 반환)
    """

    BASE_URL = "https://api.stock.naver.com/stock"
    # 접미사 없이 입력된 경우 시도 순서 (NASDAQ이 가장 일반적)
    _SUFFIX_CANDIDATES = [".O", ".N", ".A", ".K"]
    _min_interval = 0.3

    _HEADERS: Dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Linux; Android 10) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Mobile Safari/537.36"
        ),
        "Referer": "https://m.stock.naver.com/",
        "Accept": "application/json",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }

    def __init__(self, cache_ttl: int = 15):
        super().__init__()
        self._sess = requests.Session()
        self._sess.headers.update(self._HEADERS)
        # 가격 데이터 캐시 {reuters_code → (data, monotonic_time)}
        self._data_cache: Dict[str, Tuple[Any, float]] = {}
        # Reuters code 해석 결과 캐시 {raw_ticker → resolved_reuters_code|None}
        self._code_cache: Dict[str, Optional[str]] = {}
        self._cache_ttl = cache_ttl  # 초 (장 중 빠른 갱신 위해 15초 기본)

    # ── 내부 API 호출 ──────────────────────────────────────────────────────────
    def _fetch_basic(self, reuters_code: str) -> Optional[dict]:
        """네이버 증권 /basic 엔드포인트 호출. 캐시 TTL 내 재호출은 캐시 반환."""
        ck = reuters_code.upper()
        cached = self._data_cache.get(ck)
        if cached and (time.monotonic() - cached[1]) < self._cache_ttl:
            return cached[0]

        self._wait()
        url = f"{self.BASE_URL}/{reuters_code}/basic"
        try:
            r = self._sess.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            self._last = time.monotonic()
            self._data_cache[ck] = (data, time.monotonic())
            logger.debug(
                f"[Naver] {reuters_code} 조회 성공 "
                f"(marketStatus={data.get('marketStatus', 'N/A')})"
            )
            return data
        except Exception as e:
            logger.debug(f"[Naver] {reuters_code} 조회 실패: {e}")
            return None

    def _resolve_reuters_code(self, ticker: str) -> Optional[str]:
        """
        티커 심볼 → 네이버 Reuters code 결정.

        - 접미사 포함(예: "AIIO.O") → 그대로 검증 후 반환
        - 접미사 없음(예: "AIIO")   → .O → .N → .A 순으로 시도, 첫 성공 반환
        결과는 _code_cache에 저장해 재호출 비용 제거.
        """
        t = ticker.upper()
        if t in self._code_cache:
            return self._code_cache[t]

        # 이미 접미사 포함된 경우
        if '.' in t:
            data = self._fetch_basic(t)
            code = t if (data and data.get("closePrice")) else None
            self._code_cache[t] = code
            return code

        # 접미사 없음 → 후보 순으로 시도
        for suffix in self._SUFFIX_CANDIDATES:
            code = f"{t}{suffix}"
            data = self._fetch_basic(code)
            if data and data.get("closePrice"):
                logger.info(f"[Naver] {t}: Reuters code 결정 → {code}")
                self._code_cache[t] = code
                return code

        logger.warning(f"[Naver] {t}: 매칭 Reuters code 없음 (스킵)")
        self._code_cache[t] = None
        return None

    # ── 공개 메서드 ───────────────────────────────────────────────────────────
    def get_price(
        self, ticker: str, session: MarketSession
    ) -> Optional[Tuple[float, float, Optional[datetime], str]]:
        """
        네이버 증권 기준 현재가 반환.

        [가격 선택 로직 — 네이버 UI와 동일]
          PRE_MARKET  : overMarketPriceInfo.overPrice  (tradingSessionType=PRE_MARKET)
          REGULAR     : closePrice                     (marketStatus=OPEN)
          AFTER_HOURS : overMarketPriceInfo.overPrice  (tradingSessionType=AFTER_MARKET)
          CLOSED      : closePrice                     (마지막 정규장 종가)
          OVERNIGHT   : None                           (네이버 미지원)

        Returns: (price, prev_close, price_time, price_type) 또는 None
        """
        # 네이버는 Overnight(Blue Ocean ATS 20:00~04:00 ET) 미지원
        if session == MarketSession.OVERNIGHT:
            return None

        reuters_code = self._resolve_reuters_code(ticker)
        if not reuters_code:
            return None

        # _resolve_reuters_code 내부에서 이미 fetch → 캐시 재사용
        data = self._fetch_basic(reuters_code)
        if not data:
            return None

        try:
            market_status = data.get("marketStatus", "CLOSE")

            # ── closePrice (정규장 체결가 / 마지막 종가) ────────────────────
            close_str = (data.get("closePrice") or "0").replace(",", "")
            close_price = float(close_str)
            if close_price <= 0:
                return None

            # 정규장 전일종가 역산
            # compareToPreviousClosePrice = closePrice와 전일종가의 절대 차이
            chg_str  = (data.get("compareToPreviousClosePrice") or "0").replace(",", "")
            chg_code = data.get("compareToPreviousPrice", {}).get("code", "3")
            chg_val  = float(chg_str) if chg_str else 0.0
            # code "2"=상승(RISING), "5"=하락(FALLING), "3"=보합(EVEN)
            if chg_code == "2":
                prev_close_regular = close_price - chg_val
            elif chg_code == "5":
                prev_close_regular = close_price + chg_val
            else:
                prev_close_regular = close_price
            if prev_close_regular <= 0:
                prev_close_regular = close_price

            # 정규장 마지막 체결 시각
            price_time: Optional[datetime] = None
            local_traded = data.get("localTradedAt")
            if local_traded:
                try:
                    price_time = datetime.fromisoformat(local_traded).astimezone(ET_TZ)
                except Exception:
                    pass

            # ── 장외 시장 정보 파싱 ──────────────────────────────────────────
            over = data.get("overMarketPriceInfo") or {}
            over_status   = over.get("overMarketStatus", "CLOSE")
            over_sess_type = over.get("tradingSessionType", "")

            # 장외 시장 활성 중인 경우 overPrice와 시각 추출
            over_price: float = 0.0
            over_time: Optional[datetime] = None
            if over_status == "OPEN":
                over_price_str = (over.get("overPrice") or "").replace(",", "")
                over_price = float(over_price_str) if over_price_str else 0.0
                over_local = over.get("localTradedAt")
                if over_local:
                    try:
                        over_time = datetime.fromisoformat(over_local).astimezone(ET_TZ)
                    except Exception:
                        pass

            # ── 세션별 가격 선택 ─────────────────────────────────────────────

            # 프리마켓: 네이버 tradingSessionType=PRE_MARKET 이어야 유효
            if session == MarketSession.PRE_MARKET:
                if over_status != "OPEN" or over_sess_type != "PRE_MARKET":
                    logger.debug(
                        f"[Naver] {ticker}: PRE_MARKET 요청이나 "
                        f"overMarketStatus={over_status} / type={over_sess_type} → None"
                    )
                    return None
                if over_price <= 0:
                    return None
                logger.info(
                    f"[Naver] {ticker}: PRE_MARKET | "
                    f"overPrice={over_price} | closePrice={close_price} | "
                    f"time={over_time.strftime('%H:%M %Z') if over_time else 'N/A'}"
                )
                # 프리마켓 prev_close = 오늘 정규장 종가(closePrice)
                return over_price, close_price, over_time, "pre_market"

            # 정규장: marketStatus=OPEN 이어야 유효
            elif session == MarketSession.REGULAR:
                if market_status != "OPEN":
                    logger.debug(
                        f"[Naver] {ticker}: REGULAR 요청이나 "
                        f"marketStatus={market_status} → None"
                    )
                    return None
                logger.info(
                    f"[Naver] {ticker}: REGULAR | "
                    f"closePrice={close_price} | "
                    f"time={price_time.strftime('%H:%M %Z') if price_time else 'N/A'}"
                )
                return close_price, prev_close_regular, price_time, "real_time"

            # 애프터마켓: tradingSessionType=AFTER_MARKET 이어야 유효
            elif session == MarketSession.AFTER_HOURS:
                valid_after = over_status == "OPEN" and over_sess_type in (
                    "AFTER_MARKET", "POST_MARKET", "AFTER_HOURS"
                )
                if not valid_after:
                    logger.debug(
                        f"[Naver] {ticker}: AFTER_HOURS 요청이나 "
                        f"overMarketStatus={over_status} / type={over_sess_type} → None"
                    )
                    return None
                if over_price <= 0:
                    return None
                logger.info(
                    f"[Naver] {ticker}: AFTER_HOURS | "
                    f"overPrice={over_price} | closePrice={close_price} | "
                    f"time={over_time.strftime('%H:%M %Z') if over_time else 'N/A'}"
                )
                # 애프터마켓 prev_close = 오늘 정규장 종가(closePrice)
                return over_price, close_price, over_time, "post_market"

            # 장마감: closePrice = 마지막 정규장 종가
            elif session == MarketSession.CLOSED:
                logger.info(
                    f"[Naver] {ticker}: CLOSED | closePrice={close_price}"
                )
                return close_price, prev_close_regular, price_time, "last_close"

        except Exception as e:
            logger.warning(f"[Naver] {ticker} 가격 파싱 실패: {e}")

        return None


# ──────────────────────────────────────────────────────────────────────────────
# yfinance 클라이언트 (Extended Hours 주력 소스)
# ──────────────────────────────────────────────────────────────────────────────
class YFinanceClient:
    """
    yfinance 기반 가격 조회.

    세션별 핵심 필드:
      PRE_MARKET  → info["preMarketPrice"]      + info["preMarketTime"]
      REGULAR     → info["currentPrice"]         + info["regularMarketTime"]
      AFTER_HOURS → info["postMarketPrice"]      + info["postMarketTime"]
      OVERNIGHT   → info["postMarketPrice"]      + info["postMarketTime"]  ← 신규
                    (타임스탬프로 Overnight 여부 검증)
      CLOSED      → info["previousClose"]        (전일 정규장 종가)

    특징:
    - Yahoo Finance / Cboe 데이터 기반 → 브로커 표시가와 가장 근접
    - 무료, 별도 API 키 불필요
    - postMarketPrice: 16:00~04:00 ET after-hours/overnight 실제 체결가 반영 가능
    """

    _min_interval = 0.5

    def __init__(self):
        self._last = 0.0

    def _wait(self):
        gap = self._min_interval - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)

    def _fetch_info(self, ticker: str) -> Optional[dict]:
        if not _HAS_YFINANCE:
            return None
        self._wait()
        try:
            info = yf.Ticker(ticker).info
            self._last = time.monotonic()
            return info
        except Exception as e:
            logger.warning(f"[yfinance] {ticker} info 조회 실패: {e}")
            return None

    def _ts(self, unix: Any) -> Optional[datetime]:
        if not unix:
            return None
        try:
            return datetime.fromtimestamp(float(unix), tz=ET_TZ)
        except Exception:
            return None

    def _validate_post_market(
        self, ticker: str, info: dict, max_age_min: int = 480
    ) -> Optional[Tuple[float, float, Optional[datetime], str]]:
        """
        postMarketPrice 유효성 검증 헬퍼.
        max_age_min 이내의 데이터만 반환, 만료된 경우 None.
        """
        prev_close = float(
            info.get("regularMarketPreviousClose")
            or info.get("previousClose")
            or 0
        )
        post_price = info.get("postMarketPrice")
        post_ts    = self._ts(info.get("postMarketTime"))

        if not post_price or not post_ts:
            return None

        age_min = (now_et() - post_ts).total_seconds() / 60
        if age_min > max_age_min:
            logger.info(
                f"[yfinance] {ticker}: postMarketPrice가 {age_min:.0f}분 전 데이터 "
                f"(기준 {max_age_min}분 초과) → 스킵"
            )
            return None

        return float(post_price), prev_close, post_ts, "post_market"

    def get_price(
        self, ticker: str, session: MarketSession
    ) -> Optional[Tuple[float, float, Optional[datetime], str]]:
        """
        Returns (price, prev_close, price_time, price_type) or None.

        ※ OVERNIGHT / AFTER_HOURS 에서 해당 세션 데이터가 없을 경우
           regularMarketPrice 로 폴백하지 않고 None 을 반환합니다.
           폴백은 USStockPriceFetcher.fetch() 의 EOD 체인에서 명시적으로 처리합니다.
        """
        info = self._fetch_info(ticker)
        if not info:
            return None

        prev_close = float(
            info.get("regularMarketPreviousClose")
            or info.get("previousClose")
            or 0
        )

        # ── PRE_MARKET ───────────────────────────────────────────────────────
        if session == MarketSession.PRE_MARKET:
            price = info.get("preMarketPrice")
            if not price:
                logger.debug(f"[yfinance] {ticker}: preMarketPrice=None")
                return None
            ts = self._ts(info.get("preMarketTime"))
            if ts:
                age = (now_et() - ts).total_seconds() / 60
                if age > 60:
                    logger.warning(
                        f"[yfinance] {ticker}: preMarketPrice가 {age:.0f}분 전 데이터"
                    )
            return float(price), prev_close, ts, "pre_market"

        # ── REGULAR ──────────────────────────────────────────────────────────
        elif session == MarketSession.REGULAR:
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if not price:
                return None
            ts = self._ts(info.get("regularMarketTime"))
            return float(price), prev_close, ts, "real_time"

        # ── AFTER_HOURS ──────────────────────────────────────────────────────
        elif session == MarketSession.AFTER_HOURS:
            validated = self._validate_post_market(ticker, info, max_age_min=240)
            if validated:
                return validated
            # 애프터마켓 데이터 없음 → None (regularMarketPrice 로 폴백 금지)
            return None

        # ── OVERNIGHT ────────────────────────────────────────────────────────
        # Yahoo marketState: POSTPOST(20:00~자정) / PREPRE(자정~04:00)
        elif session == MarketSession.OVERNIGHT:
            # 8시간 이내 postMarketPrice = Overnight 유효 데이터
            validated = self._validate_post_market(ticker, info, max_age_min=480)
            if validated:
                price, pc, ts, _ = validated
                logger.info(
                    f"[yfinance] {ticker}: Overnight postMarketPrice 확인 "
                    f"({ts.strftime('%H:%M:%S %Z') if ts else 'N/A'})"
                )
                return price, pc, ts, "overnight"
            # Overnight 데이터 없음 → None (regularMarketPrice 로 폴백 금지)
            logger.debug(f"[yfinance] {ticker}: Overnight postMarketPrice 없음/만료 → None")
            return None

        # ── CLOSED ───────────────────────────────────────────────────────────
        elif session == MarketSession.CLOSED:
            price = prev_close or info.get("regularMarketPrice")
            if not price:
                return None
            ts = self._ts(info.get("regularMarketTime"))
            return float(price), prev_close, ts, "last_close"

        return None

    def get_market_state_price(
        self, ticker: str
    ) -> Optional[Tuple[float, float, Optional[datetime], str, str]]:
        """
        yfinance info 의 실제 marketState 를 읽어 적절한 가격을 반환.

        시간 기반 세션 감지 없이 Yahoo Finance API 가 보고하는 marketState 를
        그대로 사용하기 때문에, 세션 경계 불일치 문제를 방지합니다.

        반환: (price, prev_close, ts, price_type, market_state) 또는 None

        price_type:
          "pre_market"  — PRE 세션 preMarketPrice
          "overnight"   — PREPRE / POSTPOST 세션 postMarketPrice
          "post_market" — POST 세션 postMarketPrice
          "real_time"   — REGULAR 세션 currentPrice
          "last_close"  — CLOSED 세션 previousClose
        """
        info = self._fetch_info(ticker)
        if not info:
            return None

        market_state = info.get("marketState", "REGULAR")
        prev_close = float(
            info.get("regularMarketPreviousClose")
            or info.get("previousClose")
            or 0
        )

        # ── Pre-Market (04:00~09:30 ET) ──────────────────────────────────────
        if market_state == "PRE":
            price = info.get("preMarketPrice")
            if price:
                ts = self._ts(info.get("preMarketTime"))
                logger.info(
                    f"[yfinance ms] {ticker}: PRE → preMarketPrice={price} "
                    f"({ts.strftime('%H:%M %Z') if ts else 'N/A'})"
                )
                return float(price), prev_close, ts, "pre_market", market_state
            return None

        # ── Overnight POSTPOST (20:00~자정 ET) ───────────────────────────────
        elif market_state == "POSTPOST":
            price = info.get("postMarketPrice")
            if price:
                ts = self._ts(info.get("postMarketTime"))
                if ts:
                    age_min = (now_et() - ts).total_seconds() / 60
                    if age_min <= 480:
                        logger.info(
                            f"[yfinance ms] {ticker}: POSTPOST → "
                            f"postMarketPrice={price} ({age_min:.0f}min ago)"
                        )
                        return float(price), prev_close, ts, "overnight", market_state
            return None

        # ── Overnight PREPRE (자정~04:00 ET) ─────────────────────────────────
        elif market_state == "PREPRE":
            price = info.get("postMarketPrice")
            if price:
                ts = self._ts(info.get("postMarketTime"))
                if ts:
                    age_min = (now_et() - ts).total_seconds() / 60
                    # 480분(8h) → 360분(6h)으로 축소: PREPRE 구간 최대 04:00 ET 까지이므로
                    # 전날 22:00 이후 postMarketPrice 만 유효로 인정 (04:00 - 6h = 22:00)
                    if age_min <= 360:
                        logger.info(
                            f"[yfinance ms] {ticker}: PREPRE → "
                            f"postMarketPrice={price} ({age_min:.0f}min ago)"
                        )
                        return float(price), prev_close, ts, "overnight", market_state
            return None

        # ── After-Hours POST (16:00~20:00 ET) ────────────────────────────────
        elif market_state == "POST":
            price = info.get("postMarketPrice")
            if price:
                ts = self._ts(info.get("postMarketTime"))
                if ts:
                    age_min = (now_et() - ts).total_seconds() / 60
                    if age_min <= 240:
                        return float(price), prev_close, ts, "post_market", market_state
            return None

        # ── Regular (09:30~16:00 ET) ─────────────────────────────────────────
        elif market_state == "REGULAR":
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if price:
                ts = self._ts(info.get("regularMarketTime"))
                return float(price), prev_close, ts, "real_time", market_state
            return None

        # ── Closed (주말·공휴일) ──────────────────────────────────────────────
        else:
            price = prev_close or info.get("regularMarketPrice")
            if price:
                ts = self._ts(info.get("regularMarketTime"))
                return float(price), prev_close, ts, "last_close", market_state
            return None

    def get_price_via_history(
        self, ticker: str
    ) -> Optional[Tuple[float, float, Optional[datetime], str]]:
        """
        yfinance Ticker.history(prepost=True) 기반 최신 가격 조회.

        정규장, 프리마켓, 애프터마켓, Blue Ocean ATS 오버나이트(yfinance 지원 시) 포함.
        marketState 판단 없이 가장 마지막 캔들 Close 가격을 반환합니다.

        반환: (price, prev_close, price_time, price_type) 또는 None
        """
        if not _HAS_YFINANCE:
            return None
        self._wait()
        try:
            t  = yf.Ticker(ticker)
            df = t.history(
                period="1d",
                interval="1m",
                prepost=True,
                auto_adjust=False,
            )
            if df.empty:
                logger.debug(f"[yfinance history] {ticker}: 빈 DataFrame")
                return None

            last  = df.iloc[-1]
            price = float(last["Close"])
            if price <= 0:
                return None

            # 타임스탬프 (DatetimeIndex → ET datetime)
            ts: Optional[datetime] = None
            try:
                ts_raw = last.name
                ts = ts_raw.to_pydatetime() if hasattr(ts_raw, "to_pydatetime") else None
                if ts:
                    ts = ts.replace(tzinfo=ET_TZ) if ts.tzinfo is None else ts.astimezone(ET_TZ)
            except Exception:
                pass

            # prev_close: fast_info 우선, 실패 시 info fallback
            prev_close = 0.0
            try:
                pc = t.fast_info.previous_close
                if pc and pc > 0:
                    prev_close = float(pc)
            except Exception:
                pass
            if not prev_close:
                try:
                    info = self._fetch_info(ticker)
                    if info:
                        prev_close = float(
                            info.get("regularMarketPreviousClose")
                            or info.get("previousClose")
                            or 0
                        )
                except Exception:
                    pass

            # 신선도 검사: 마지막 캔들이 120분(2시간) 이상 지났으면 스테일 데이터 폐기
            if ts:
                age_min = (now_et() - ts).total_seconds() / 60
                if age_min > 120:
                    logger.warning(
                        f"[yfinance history] {ticker}: 스테일 캔들 폐기 "
                        f"({age_min:.0f}min ago, ts={ts.strftime('%H:%M %Z')})"
                    )
                    return None

            self._last = time.monotonic()
            logger.info(
                f"[yfinance history] {ticker}: "
                f"price={price:.4f} | "
                f"time={ts.strftime('%H:%M:%S %Z') if ts else 'N/A'}"
            )
            return price, prev_close or price, ts, "extended_hours"
        except Exception as e:
            logger.warning(f"[yfinance history] {ticker}: {e}")
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Tiingo IEX 클라이언트
# ──────────────────────────────────────────────────────────────────────────────
class TiingoClient(_BaseClient):
    """
    Tiingo IEX 실시간 시세 클라이언트.

    필드 설명 (실측 기반):
      last     : IEX 거래소 마지막 체결가 (Extended에서 종종 None)
      tngoLast : Tiingo 집계 마지막가 (Extended 포함, yfinance와 ±0.3$ 수준)
      prevClose: 전일 정규장 종가 (신뢰도 높음)
      timestamp: 마지막 체결 시각 (ET 기준)
      open/high/low: 현재 세션 고가/저가 (Extended 포함)

    커버리지:
      정규장: IEX 실시간 (매우 정확)
      프리마켓: 08:00 ET 이후 부분 커버 (04:00~08:00 는 데이터 없을 수 있음)
      애프터마켓: 16:00~17:00 ET 부분 커버
      오버나이트: 거의 커버 안 됨 → yfinance 선행

    Rate limit: 500 req/hour (무료)
    """
    BASE = "https://api.tiingo.com"
    _min_interval = 0.5

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self._headers = {"Authorization": f"Token {api_key}",
                         "Content-Type": "application/json"}

    def iex(self, ticker: str) -> Optional[dict]:
        if not self.api_key:
            return None
        data = self._get(f"{self.BASE}/iex/{ticker.lower()}", {},
                         headers=self._headers, label="Tiingo")
        if isinstance(data, list):
            return data[0] if data else None
        return data

    def eod(self, ticker: str) -> Optional[dict]:
        """최근 EOD 종가 (장 마감 후 fallback)."""
        if not self.api_key:
            return None
        start = (date.today() - timedelta(days=7)).isoformat()
        data  = self._get(
            f"{self.BASE}/tiingo/daily/{ticker.lower()}/prices",
            {"startDate": start}, headers=self._headers, label="Tiingo EOD",
        )
        if not isinstance(data, list) or not data:
            return None
        return {"latest": data[-1], "prev": data[-2] if len(data) >= 2 else None}

    def _parse_ts(self, s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(ET_TZ)
        except Exception:
            return None

    def get_price(
        self, ticker: str, session: MarketSession
    ) -> Optional[Tuple[float, float, Optional[datetime], str]]:
        """
        Returns (price, prev_close, price_time, price_type) or None.

        정규장: last(IEX 실시간) 우선
        Extended: tngoLast 사용, 타임스탬프로 신선도 검증
        OVERNIGHT: Tiingo IEX 커버리지 거의 없음 → 빠르게 None 반환
        """
        q = self.iex(ticker)
        if not q:
            return None

        prev_close = float(q.get("prevClose") or 0)
        ts         = self._parse_ts(q.get("timestamp"))

        if session == MarketSession.REGULAR:
            price = q.get("last") or q.get("tngoLast")
            if not price:
                return None
            if ts:
                age = (now_et() - ts).total_seconds() / 60
                if age > 15:
                    logger.warning(f"[Tiingo] {ticker}: 정규장 데이터 {age:.0f}분 경과")
            return float(price), prev_close, ts, "real_time"

        elif session in (MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS):
            price = q.get("tngoLast") or q.get("last")
            if not price:
                return None
            if ts:
                age = (now_et() - ts).total_seconds() / 60
                if age > 60:
                    logger.info(
                        f"[Tiingo] {ticker}: Extended 데이터 {age:.0f}분 경과 "
                        f"(IEX 커버리지 외 구간일 수 있음) → yfinance가 더 정확"
                    )
                    return None
            return float(price), prev_close, ts, "extended"

        elif session == MarketSession.OVERNIGHT:
            # Overnight 세션에서 Tiingo 커버리지 매우 낮음
            # tngoLast가 있고 2시간 이내인 경우만 사용 (낮은 신뢰도)
            price = q.get("tngoLast") or q.get("last")
            if not price or not ts:
                return None
            age = (now_et() - ts).total_seconds() / 60
            if age > 120:
                logger.debug(f"[Tiingo] {ticker}: Overnight 데이터 만료 ({age:.0f}분)")
                return None
            logger.info(f"[Tiingo] {ticker}: Overnight tngoLast 사용 ({age:.0f}분 전, 신뢰도 낮음)")
            return float(price), prev_close, ts, "overnight"

        else:  # CLOSED
            data = self.eod(ticker)
            if not data:
                return None
            latest = data["latest"]
            prev   = data["prev"]
            price  = float(latest.get("adjClose") or latest.get("close") or 0)
            prev_c = float((prev.get("adjClose") or prev.get("close")) if prev else price)
            if not price:
                return None
            ts2 = self._parse_ts(latest.get("date"))
            return price, prev_c, ts2, "last_close"


# ──────────────────────────────────────────────────────────────────────────────
# Finnhub 클라이언트
# ──────────────────────────────────────────────────────────────────────────────
class FinnhubClient(_BaseClient):
    """
    Finnhub.io 무료 플랜 클라이언트.

    ⚠ 무료 플랜 실측 특성:
      - c 필드 = 마지막 정규장 종가 (Extended Hours 데이터 없음)
      - timestamp = 이전 거래일 16:00 ET (프리마켓/애프터마켓에서 3일 전 데이터)
      - → Extended/Overnight 세션에서는 사용 불가 (정규장 fallback 전용)
      - pc 필드 = 전전일 종가 (CLOSED 세션 참고용)

    정규장에서는 last trade 가격 제공 (약간의 지연 있을 수 있음)
    Rate limit: 60 req/min
    """
    BASE = "https://finnhub.io/api/v1"
    _min_interval = 1.05

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def quote(self, ticker: str) -> Optional[dict]:
        if not self.api_key:
            return None
        data = self._get(f"{self.BASE}/quote", {"symbol": ticker, "token": self.api_key},
                         label="Finnhub")
        if not data or not data.get("c"):
            return None
        return data

    def get_price(
        self, ticker: str, session: MarketSession
    ) -> Optional[Tuple[float, float, Optional[datetime], str]]:
        """
        Returns (price, prev_close, price_time, price_type) or None.

        ⚠ 핵심 제약:
          타임스탬프가 현재 거래일이 아닌 경우(= Extended/Closed 세션에서)
          Finnhub 데이터는 사용하지 않음.
          정규장 (REGULAR) 및 CLOSED 에서만 데이터 사용.
          OVERNIGHT / PRE_MARKET / AFTER_HOURS → None 반환.
        """
        q = self.quote(ticker)
        if not q:
            return None

        price      = float(q["c"])
        prev_close = float(q.get("pc") or price)
        ts: Optional[datetime] = None
        if q.get("t"):
            try:
                ts = datetime.fromtimestamp(q["t"], tz=ET_TZ)
            except Exception:
                pass

        if ts:
            now = now_et()
            days_old = (now.date() - ts.date()).days
            if days_old > 3:
                logger.debug(
                    f"[Finnhub] {ticker}: timestamp {ts.date()} → {days_old}일 전 데이터 "
                    f"(Extended 세션 — 사용 안 함)"
                )
                return None

        if session == MarketSession.REGULAR:
            return price, prev_close, ts, "real_time"
        elif session == MarketSession.CLOSED:
            return float(q.get("pc") or price), prev_close, ts, "last_close"
        else:
            # Extended / Overnight 세션: Finnhub 무료는 정확하지 않으므로 사용 안 함
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Alpha Vantage 클라이언트
# ──────────────────────────────────────────────────────────────────────────────
class AlphaVantageClient(_BaseClient):
    """
    Alpha Vantage 무료 플랜 클라이언트.

    무료에서 사용 가능:
      GLOBAL_QUOTE   → 최근 정규장 종가 (가격, 전일비, 거래량 등)
      TIME_SERIES_DAILY → 일봉 EOD 데이터

    유료 전용:
      TIME_SERIES_INTRADAY (extended_hours=true) → 무료 플랜에서 거부됨

    ⚠ 무료 한도: 25 req/day → 캐싱 필수, 마지막 fallback 전용

    Rate limit: 25/day → 15초 간격으로 보수적 사용
    """
    BASE = "https://www.alphavantage.co/query"
    _min_interval = 15.0

    def __init__(self, api_key: str, cache_ttl: int = 120):
        super().__init__()
        self.api_key   = api_key
        self._ttl      = cache_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}

    def _avget(self, params: Dict, key: str) -> Optional[dict]:
        if not self.api_key:
            return None
        if key in self._cache:
            d, at = self._cache[key]
            if time.monotonic() - at < self._ttl:
                return d
        params["apikey"] = self.api_key
        data = self._get(self.BASE, params, label="AlphaVantage")
        if not data:
            return None
        if "Error Message" in data:
            logger.warning(f"[AV] 에러: {data['Error Message']}"); return None
        if "Note" in data:
            logger.warning(f"[AV] Rate limit: {data['Note'][:60]}"); return None
        if "Information" in data:
            logger.debug("[AV] 프리미엄 전용 엔드포인트"); return None
        self._cache[key] = (data, time.monotonic())
        return data

    def global_quote(self, ticker: str) -> Optional[Tuple[float, float, Optional[datetime], str]]:
        """GLOBAL_QUOTE: 최근 정규장 종가."""
        data = self._avget({"function": "GLOBAL_QUOTE", "symbol": ticker}, f"gq:{ticker}")
        if not data:
            return None
        q     = data.get("Global Quote", {})
        price = q.get("05. price")
        prev  = q.get("08. previous close")
        day   = q.get("07. latest trading day")
        if not price:
            return None
        ts: Optional[datetime] = None
        try:
            ts = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=ET_TZ) if day else None
        except Exception:
            pass
        return float(price), float(prev or 0), ts, "last_close"

    def daily_close(self, ticker: str) -> Optional[Tuple[float, float, Optional[datetime], str]]:
        """TIME_SERIES_DAILY: EOD fallback."""
        data = self._avget(
            {"function": "TIME_SERIES_DAILY", "symbol": ticker, "outputsize": "compact"},
            f"daily:{ticker}",
        )
        if not data:
            return None
        series = data.get("Time Series (Daily)", {})
        if not series:
            return None
        keys   = sorted(series.keys(), reverse=True)
        latest = series[keys[0]]
        prev_d = series[keys[1]] if len(keys) > 1 else None
        price  = float(latest.get("4. close", 0))
        prev_c = float(prev_d["4. close"]) if prev_d else price
        ts: Optional[datetime] = None
        try:
            ts = datetime.strptime(keys[0], "%Y-%m-%d").replace(tzinfo=ET_TZ)
        except Exception:
            pass
        return price, prev_c, ts, "last_close"


# ──────────────────────────────────────────────────────────────────────────────
# 메인 오케스트레이터
# ──────────────────────────────────────────────────────────────────────────────
class USStockPriceFetcher:
    """
    세션 인식 멀티소스 미국 주가 수집기.
    Overnight(심야 연장 거래) 가격을 우선적으로 반환합니다.

    [실측 기반 API 우선순위]

    OVERNIGHT (20:00~04:00 ET / KST ~06:00~17:00)
      ← Yahoo Finance "Overnight" 레이블 구간
      1. YFDirect  postMarketPrice (marketState 명시 검증)  ← 가장 정확
      2. yfinance  postMarketPrice (타임스탬프 8시간 이내)
      3. Tiingo    tngoLast        (커버리지 낮음, 보조)
      4. 당일/전일 정규장 종가 fallback

    PRE_MARKET (04:00–09:30 ET / KST ~13:00–22:30)
      1. YFDirect  preMarketPrice  (marketState="PRE" 검증)
      2. yfinance  preMarketPrice
      3. Tiingo    tngoLast
      4. 전일종가 fallback

    REGULAR (09:30–16:00 ET / KST ~22:30–05:00)
      1. Tiingo IEX last
      2. yfinance  currentPrice
      3. Finnhub   c
      4. AV Global Quote

    AFTER_HOURS (16:00–20:00 ET / KST ~05:00–06:00)
      1. YFDirect  postMarketPrice (marketState="POST" 검증)
      2. yfinance  postMarketPrice
      3. Tiingo    tngoLast
      4. 당일종가 fallback

    CLOSED (주말 / NYSE 공휴일)
      1. yfinance  previousClose
      2. Finnhub   pc
      3. Tiingo    EOD
      4. AV Daily

    사용법:
        fetcher = USStockPriceFetcher()
        result  = fetcher.fetch("SHPH")
        print(result)
        print(result.to_dict())
    """

    def __init__(
        self,
        finnhub_key: str = FINNHUB_KEY,
        tiingo_key:  str = TIINGO_KEY,
        av_key:      str = AV_KEY,
    ):
        self.naver   = NaverWorldStockClient(cache_ttl=15)  # ← 네이버 증권 (최우선)
        self.yfd     = YahooFinanceDirectClient(cache_ttl=30)
        self.yf      = YFinanceClient()
        self.tiingo  = TiingoClient(tiingo_key)
        self.finnhub = FinnhubClient(finnhub_key)
        self.av      = AlphaVantageClient(av_key)

    # ── PriceResult 생성 헬퍼 ──────────────────────────────────────────────────
    def _build(
        self,
        ticker:  str,
        raw:     Tuple[float, float, Optional[datetime], str],
        session: MarketSession,
        source:  str,
        notes:   str = "",
    ) -> Optional["PriceResult"]:
        price, prev_close, ts, ptype = raw
        if price <= 0:
            return None
        if prev_close <= 0:
            prev_close = price
        change     = round(price - prev_close, 6)
        change_pct = round(change / prev_close * 100, 4)
        return PriceResult(
            ticker=ticker, price=round(price, 4), prev_close=round(prev_close, 4),
            change=change, change_pct=change_pct, session=session,
            price_type=ptype, source=source, price_time=ts, notes=notes,
        )

    def _try(
        self, fn, ticker: str, session: MarketSession, source: str, notes: str = ""
    ) -> Optional["PriceResult"]:
        try:
            raw = fn(ticker, session)
            if raw is None:
                return None
            return self._build(ticker, raw, session, source, notes)
        except Exception as e:
            logger.warning(f"[{source}] {ticker} 실패: {e}")
            return None

    def _try_yfd(
        self,
        fn,           # YFDirect의 메서드 (ticker만 인자로 받는 형태)
        ticker: str,
        session: MarketSession,
        source: str,
        notes: str = "",
    ) -> Optional["PriceResult"]:
        """
        YahooFinanceDirectClient 전용 _try 래퍼.
        반환값이 5-tuple (price, prev_close, ts, ptype, market_state) 이므로
        앞 4개만 추출하여 _build에 전달.
        """
        try:
            raw5 = fn(ticker)
            if raw5 is None:
                return None
            price, prev_close, ts, ptype, market_state = raw5
            raw4 = (price, prev_close, ts, ptype)
            result = self._build(ticker, raw4, session, source, notes)
            if result:
                result.notes = (
                    f"{notes} [Yahoo marketState={market_state}]"
                    if notes else f"Yahoo marketState={market_state}"
                ).strip()
            return result
        except Exception as e:
            logger.warning(f"[{source}] {ticker} 실패: {e}")
            return None

    def _first(self, *results: Optional["PriceResult"]) -> Optional["PriceResult"]:
        return next((r for r in results if r is not None and r.price > 0), None)

    # ── 공개 API ──────────────────────────────────────────────────────────────
    def fetch(
        self,
        ticker:      str,
        override_dt: Optional[datetime] = None,
    ) -> Optional["PriceResult"]:
        """
        `ticker`의 현재 시각에 맞는 최정확 미국 주가 반환.
        Overnight 세션에서는 Yahoo Finance postMarketPrice를 우선 반환합니다.
        """
        ticker  = ticker.upper().strip()
        session, dt_et = detect_session(override_dt)
        dt_kst  = dt_et.astimezone(KST_TZ)

        logger.info(
            f"[Fetcher] {ticker} | {session.label_ko()} | "
            f"ET {dt_et.strftime('%H:%M')} | KST {dt_kst.strftime('%H:%M')}"
        )

        # ── 1. OVERNIGHT (20:00~04:00 ET) ────────────────────────────────────
        # Yahoo Finance "Overnight" 레이블 구간 (Blue Ocean ATS 8PM~4AM ET)
        # 우선순위:
        #   ① YFDirect overnightMarketPrice (crumb, Blue Ocean ATS 실체결가)
        #   ② yfinance marketState 직접 읽기 (POSTPOST/PREPRE → postMarketPrice)
        #   ③ yfinance get_price (postMarketPrice 타임스탬프 8h 검증)
        #   ④ yfinance history(prepost=True) 마지막 캔들
        #   ⑤ Tiingo tngoLast (커버리지 낮음)
        #   → 모두 실패 시 정규장 종가 fallback
        if session == MarketSession.OVERNIGHT:
            logger.info(
                f"[Fetcher] {ticker}: 🌙 Overnight 세션 "
                f"— Yahoo Overnight(Blue Ocean ATS) 가격 우선 조회"
            )

            def _ms_overnight(t: str, _s: MarketSession):
                """yfinance marketState 기반 overnight 추출 래퍼."""
                raw5 = self.yf.get_market_state_price(t)
                if raw5 is None:
                    return None
                price, prev_close, ts, ptype, ms = raw5
                if ptype not in ("overnight", "post_market"):
                    return None  # regular / last_close 는 이 체인에서 무시
                return price, prev_close, ts, ptype

            result = self._first(
                # ① YFDirect: crumb + overnightMarketPrice (Blue Ocean ATS 실체결가)
                self._try_yfd(
                    self.yfd.get_overnight_price, ticker, session,
                    "Yahoo Finance Direct",
                    "overnightMarketPrice/postMarketPrice (marketState=PREPRE/POSTPOST)",
                ),
                # ② yfinance marketState 직접: POSTPOST/PREPRE → postMarketPrice
                self._try(
                    _ms_overnight, ticker, session, "yfinance(marketState)",
                    "marketState 기반 overnight — postMarketPrice",
                ),
                # ③ yfinance info: postMarketPrice + 타임스탬프 검증 (8h)
                self._try(
                    self.yf.get_price, ticker, session, "yfinance",
                    "postMarketPrice — Yahoo Finance Overnight 타임스탬프 검증",
                ),
                # ④ yfinance history(prepost=True): 마지막 Extended Hours 캔들
                self._try(
                    lambda t, _s: self.yf.get_price_via_history(t),
                    ticker, session, "yfinance(history)",
                    "history prepost=True — 최신 Extended Hours 캔들",
                ),
                # ⑤ Tiingo: 커버리지 낮음, 보조 수단
                self._try(
                    self.tiingo.get_price, ticker, session, "Tiingo IEX",
                    "tngoLast — Overnight 커버리지 낮음 (보조)",
                ),
            )
            if result:
                result.notes = result.notes or "🌙 Yahoo Overnight 가격"
                return result

            # 모든 Overnight 소스 실패 → 당일/전일 정규장 종가
            logger.info(
                f"[Fetcher] {ticker}: Overnight 데이터 없음 — 정규장 종가 fallback"
            )
            eod = self._first(
                self._try(self.yf.get_price, ticker, MarketSession.CLOSED, "yfinance"),
                self._try(self.finnhub.get_price, ticker, MarketSession.CLOSED, "Finnhub"),
                self._try(self.tiingo.get_price, ticker, MarketSession.CLOSED, "Tiingo EOD"),
                self._try(self.av.global_quote,  ticker, session, "AlphaVantage"),
            )
            if eod:
                eod.notes      = "Overnight 거래 없음 또는 데이터 미제공 — 가장 최근 정규장 종가"
                eod.price_type = "last_close"
                eod.session    = MarketSession.OVERNIGHT  # 세션 표기 유지
            return eod

        # ── 2. 프리마켓 (04:00–09:30 ET) ─────────────────────────────────────
        # 우선순위:
        #   ① YFDirect preMarketPrice (crumb, marketState=PRE 또는 preMarketPrice 필드 존재)
        #   ② yfinance marketState 직접 읽기 (PRE → preMarketPrice)
        #   ③ yfinance get_price (preMarketPrice)
        #   ④ yfinance history(prepost=True) 마지막 캔들
        #   ⑤ Tiingo tngoLast
        #   → 모두 실패 시 전일종가 fallback
        elif session == MarketSession.PRE_MARKET:

            def _ms_premarket(t: str, _s: MarketSession):
                """yfinance marketState 기반 pre-market 추출 래퍼."""
                raw5 = self.yf.get_market_state_price(t)
                if raw5 is None:
                    return None
                price, prev_close, ts, ptype, ms = raw5
                if ptype != "pre_market":
                    return None  # overnight / regular 등은 이 체인에서 무시
                return price, prev_close, ts, ptype

            result = self._first(
                # ① 네이버 증권: overMarketPriceInfo.overPrice (PRE_MARKET 세션 검증)
                self._try(
                    self.naver.get_price, ticker, session, "Naver 증권",
                    "overMarketPriceInfo.overPrice — 네이버 모바일 증권 프리마켓",
                ),
                # ② YFDirect: crumb + preMarketPrice 필드 존재 여부 우선 확인
                self._try_yfd(
                    self.yfd.get_premarket_price, ticker, session,
                    "Yahoo Finance Direct",
                    "preMarketPrice — YFDirect (marketState 완화 검증)",
                ),
                # ③ yfinance marketState 직접: PRE → preMarketPrice
                self._try(
                    _ms_premarket, ticker, session, "yfinance(marketState)",
                    "marketState=PRE 기반 preMarketPrice",
                ),
                # ④ yfinance info: preMarketPrice (세션 라우팅 기반)
                self._try(
                    self.yf.get_price, ticker, session, "yfinance",
                    "preMarketPrice — Yahoo Finance/Cboe 실시간",
                ),
                # ⑤ yfinance history(prepost=True): 마지막 Pre-market 캔들
                self._try(
                    lambda t, _s: self.yf.get_price_via_history(t),
                    ticker, session, "yfinance(history)",
                    "history prepost=True — 최신 Pre-market 캔들",
                ),
                # ⑥ Tiingo tngoLast
                self._try(
                    self.tiingo.get_price, ticker, session, "Tiingo IEX",
                    "tngoLast — IEX 집계 (yfinance ±0.3$ 수준)",
                ),
            )
            if result:
                return result
            # 모든 Pre-market 소스 실패 → 전일종가
            eod = self._first(
                self._try(self.yf.get_price, ticker, MarketSession.CLOSED, "yfinance"),
                self._try(self.tiingo.get_price, ticker, MarketSession.CLOSED, "Tiingo EOD"),
                self._try(self.av.global_quote,  ticker, session, "AlphaVantage"),
            )
            if eod:
                eod.notes      = "프리마켓 데이터 없음 (04:00 ET 이전 또는 거래 없음) — 전일종가 표시"
                eod.price_type = "last_close"
            return eod

        # ── 3. 정규장 (09:30–16:00 ET) ───────────────────────────────────────
        elif session == MarketSession.REGULAR:
            return self._first(
                # ① 네이버 증권: closePrice (marketStatus=OPEN 실시간)
                self._try(
                    self.naver.get_price, ticker, session, "Naver 증권",
                    "closePrice — 네이버 모바일 증권 정규장 실시간",
                ),
                self._try(
                    self.tiingo.get_price, ticker, session, "Tiingo IEX",
                    "IEX 실시간 last trade",
                ),
                self._try(
                    self.yf.get_price, ticker, session, "yfinance",
                    "currentPrice — Yahoo Finance 실시간",
                ),
                self._try(
                    self.finnhub.get_price, ticker, session, "Finnhub",
                    "last trade (약간 지연 가능)",
                ),
                self._try(self.av.global_quote, ticker, session, "AlphaVantage"),
            )

        # ── 4. 애프터마켓 (16:00–20:00 ET) ───────────────────────────────────
        elif session == MarketSession.AFTER_HOURS:
            result = self._first(
                # ① 네이버 증권: overMarketPriceInfo.overPrice (AFTER_MARKET 세션 검증)
                self._try(
                    self.naver.get_price, ticker, session, "Naver 증권",
                    "overMarketPriceInfo.overPrice — 네이버 모바일 증권 애프터마켓",
                ),
                # ② YFDirect: crumb 인증 후 marketState="POST" 명시 검증
                self._try_yfd(
                    self.yfd.get_postmarket_price, ticker, session,
                    "Yahoo Finance Direct",
                    "postMarketPrice — marketState=POST 검증",
                ),
                # ③ yfinance info: postMarketPrice
                self._try(
                    self.yf.get_price, ticker, session, "yfinance",
                    "postMarketPrice — Yahoo Finance/Cboe 실시간",
                ),
                # ④ yfinance history(prepost=True): 마지막 캔들 Close 가격
                self._try(
                    lambda t, _s: self.yf.get_price_via_history(t),
                    ticker, session, "yfinance(history)",
                    "history prepost=True — 최신 After-hours 캔들",
                ),
                # ⑤ Tiingo tngoLast (17:00 ET 이후 신뢰도 저하)
                self._try(
                    self.tiingo.get_price, ticker, session, "Tiingo IEX",
                    "tngoLast — IEX 집계 (17:00 ET 이후 신뢰도 낮음)",
                ),
            )
            if result:
                return result
            # 애프터마켓 데이터 없으면 당일 정규장 종가
            eod = self._first(
                self._try(self.yf.get_price, ticker, MarketSession.CLOSED, "yfinance"),
                self._try(self.tiingo.get_price, ticker, MarketSession.CLOSED, "Tiingo EOD"),
                self._try(self.av.global_quote, ticker, session, "AlphaVantage"),
            )
            if eod:
                eod.notes      = "애프터마켓 데이터 없음 — 당일 정규장 종가 표시"
                eod.price_type = "last_close"
            return eod

        # ── 5. 장마감 (주말 / NYSE 공휴일) ───────────────────────────────────
        elif session == MarketSession.CLOSED:
            result = self._first(
                # ① 네이버 증권: closePrice (마지막 정규장 종가)
                self._try(
                    self.naver.get_price, ticker, session, "Naver 증권",
                    "closePrice — 네이버 모바일 증권 마지막 정규장 종가",
                ),
                self._try(
                    self.yf.get_price, ticker, session, "yfinance",
                    "previousClose — 마지막 정규장 종가",
                ),
                self._try(
                    self.finnhub.get_price, ticker, session, "Finnhub",
                    "pc — 전일 정규장 종가",
                ),
                self._try(self.tiingo.get_price, ticker, session, "Tiingo EOD"),
                self._try(self.av.daily_close,   ticker, session, "AlphaVantage Daily"),
            )
            if result:
                result.notes = result.notes or "장마감 (주말/공휴일) — 가장 최근 정규장 종가"
            return result

        logger.error(f"[Fetcher] 모든 소스 실패: {ticker}")
        return None

    def fetch_batch(
        self,
        tickers:       List[str],
        delay_between: float = 0.0,
    ) -> Dict[str, Optional["PriceResult"]]:
        """여러 티커 일괄 조회."""
        results: Dict[str, Optional[PriceResult]] = {}
        for i, t in enumerate(tickers):
            results[t] = self.fetch(t)
            if delay_between > 0 and i < len(tickers) - 1:
                time.sleep(delay_between)
        return results


# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────────────────────
def kst_to_et(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST_TZ)
    return dt.astimezone(ET_TZ)

def et_to_kst(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("ET datetime은 timezone-aware여야 합니다")
    return dt.astimezone(KST_TZ)

def get_price(ticker: str, **kw) -> Optional[PriceResult]:
    """단일 티커 간편 조회 함수."""
    return USStockPriceFetcher(**kw).fetch(ticker)


# ──────────────────────────────────────────────────────────────────────────────
# CLI 진입점
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "US 주가 실시간 조회 — Overnight 지원\n"
            "(YFDirect Overnight → yfinance → Tiingo/Finnhub/AV 폴백)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python us_price_fetcher.py SHPH AAPL TSLA
  python us_price_fetcher.py MSFT --json
  python us_price_fetcher.py AAPL TSLA --delay 1
  python us_price_fetcher.py NVDA --debug
        """,
    )
    parser.add_argument("tickers", nargs="+", metavar="TICKER")
    parser.add_argument("--json",  action="store_true", help="JSON 출력")
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="티커 간 딜레이(초, AV 할당량 절약)",
    )
    parser.add_argument("--debug", action="store_true", help="DEBUG 로그 출력")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 현재 세션 정보 출력
    info = session_info()
    sep  = "=" * 60
    overnight_note = (
        "\n  ※ Overnight 세션: Yahoo Finance 'Overnight' 가격 우선 조회"
        if info["session"] == "overnight" else ""
    )
    print(f"\n{sep}")
    print(f"  KST   : {info['kst_time']}")
    print(f"  ET    : {info['et_time']}")
    print(f"  세션  : {info['session_ko']} ({info['session']}){overnight_note}")
    print(f"  거래일: {info['is_trading_day']}")
    print(f"  시간대: {info['dst_active']}")
    print(f"{sep}\n")

    fetcher = USStockPriceFetcher()

    if args.json:
        import json
        results = fetcher.fetch_batch(args.tickers, delay_between=args.delay)
        out = {t: (r.to_dict() if r else None) for t, r in results.items()}
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    else:
        for i, ticker in enumerate(args.tickers):
            r = fetcher.fetch(ticker)
            print(r if r else f"❌ [{ticker}] 모든 소스에서 조회 실패\n")
            if args.delay > 0 and i < len(args.tickers) - 1:
                time.sleep(args.delay)
