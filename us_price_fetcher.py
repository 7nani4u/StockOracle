#!/usr/bin/env python3
"""
us_price_fetcher.py
================================================================================
Multi-source U.S. stock real-time price fetcher.
KST 기준 언제 실행해도 항상 가장 최신의 정확한 미국 주가를 반환합니다.

[Market Sessions — ET 기준]
  Pre-market  : 04:00 – 09:29  (Extended hours, partial liquidity)
  Regular     : 09:30 – 15:59  (Primary session, full liquidity)
  After-hours : 16:00 – 19:59  (Extended hours, partial liquidity)
  Closed      : 20:00 – 03:59  (No trading, last close returned)

[KST 기준 — 서머타임(EDT) / 표준시(EST) 자동 전환]
  EDT(3~11월): Regular = 22:30 ~ 05:00 KST
  EST(11~3월): Regular = 23:30 ~ 06:00 KST

[API 우선순위 매트릭스]
  ┌─────────────┬──────────────────────────────────────────────────────────────┐
  │ Session     │ API 우선순위                                                 │
  ├─────────────┼──────────────────────────────────────────────────────────────┤
  │ REGULAR     │ Finnhub quote → Tiingo IEX → AV Global Quote                │
  │ PRE_MARKET  │ Tiingo IEX → AV Intraday(ext=true) → Finnhub → last_close   │
  │ AFTER_HOURS │ Tiingo IEX → AV Intraday(ext=true) → Finnhub → today_close  │
  │ CLOSED      │ Finnhub(pc) → Tiingo EOD → AV Daily                         │
  └─────────────┴──────────────────────────────────────────────────────────────┘

[API별 Extended Hours 커버리지]
  Finnhub   : 마지막 체결가 반환 (extended 구분 불명확, 무료 플랜 한계 있음)
  Tiingo IEX: IEX 거래소 데이터, 08:00–17:00 ET (부분 커버)
  AlphaVantage: extended_hours=true 시 04:00–20:00 ET 전체 커버 (가장 정확)

[Rate Limits]
  Finnhub   : 60 req/min  (free)
  Tiingo    : 500 req/hour (free)
  AlphaVantage: 25 req/day (free) → 보수적 사용 + 캐싱 필수

Dependencies:
  pip install requests
  (Python 3.9+: zoneinfo 내장 / 3.8: pip install backports.zoneinfo)
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
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # pip install backports.zoneinfo

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 설정 (환경변수 우선, 없으면 기본값)
# ──────────────────────────────────────────────────────────────────────────────
FINNHUB_KEY  = os.getenv("FINNHUB_API_KEY",   "d7lm0o9r01qm7o0cb440d7lm0o9r01qm7o0cb44g")
TIINGO_KEY   = os.getenv("TIINGO_API_KEY",    "12ebd1feef89b6728cc15808864b7402449a5637")
AV_KEY       = os.getenv("ALPHAVANTAGE_KEY",  "E0ODFSRNDU4P9HDU")

ET_TZ  = ZoneInfo("America/New_York")   # DST 자동 처리 (EDT/EST 전환)
KST_TZ = ZoneInfo("Asia/Seoul")

REQUEST_TIMEOUT = 8   # seconds per request


# ──────────────────────────────────────────────────────────────────────────────
# NYSE 휴장일 (2025–2026)
# 반드시 매년 갱신 필요 — 공식 소스: https://www.nyse.com/markets/hours-calendars
# ──────────────────────────────────────────────────────────────────────────────
NYSE_HOLIDAYS: frozenset[date] = frozenset({
    # 2025
    date(2025, 1,  1),   # New Year's Day
    date(2025, 1, 20),   # Martin Luther King Jr. Day
    date(2025, 2, 17),   # Presidents' Day
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 26),   # Memorial Day
    date(2025, 6, 19),   # Juneteenth
    date(2025, 7,  4),   # Independence Day
    date(2025, 9,  1),   # Labor Day
    date(2025, 11, 27),  # Thanksgiving Day
    date(2025, 12, 25),  # Christmas Day
    # 2025 half-days (정규장 13:00 ET 조기 종료): 필요시 별도 처리
    # date(2025, 11, 28),  # Day after Thanksgiving
    # date(2025, 12, 24),  # Christmas Eve

    # 2026
    date(2026, 1,  1),   # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4,  3),   # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7,  3),   # Independence Day (observed, July 4 = Saturday)
    date(2026, 9,  7),   # Labor Day
    date(2026, 11, 26),  # Thanksgiving Day
    date(2026, 12, 25),  # Christmas Day
})

# 조기 종료일 (13:00 ET에 정규장 마감)
NYSE_EARLY_CLOSE: frozenset[date] = frozenset({
    date(2025, 11, 28),  # Day after Thanksgiving 2025
    date(2025, 12, 24),  # Christmas Eve 2025
    date(2026, 11, 27),  # Day after Thanksgiving 2026
    date(2026, 12, 24),  # Christmas Eve 2026
})


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────────────────────────────────────
class MarketSession(Enum):
    PRE_MARKET  = "pre_market"    # 04:00–09:29 ET
    REGULAR     = "regular"       # 09:30–15:59 ET
    AFTER_HOURS = "after_hours"   # 16:00–19:59 ET
    CLOSED      = "closed"        # 20:00–03:59 ET / weekend / holiday

    @property
    def is_extended(self) -> bool:
        return self in (MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS)

    @property
    def is_active(self) -> bool:
        return self != MarketSession.CLOSED

    def label_ko(self) -> str:
        return {
            MarketSession.PRE_MARKET:  "프리마켓",
            MarketSession.REGULAR:     "정규장",
            MarketSession.AFTER_HOURS: "애프터마켓",
            MarketSession.CLOSED:      "장마감",
        }[self]


@dataclass
class PriceResult:
    """가격 조회 결과"""
    ticker:      str
    price:       float               # 현재가
    prev_close:  float               # 전일 종가
    change:      float               # 전일 대비 변동
    change_pct:  float               # 전일 대비 변동률 (%)
    session:     MarketSession       # 현재 세션
    price_type:  str                 # "real_time" | "extended" | "last_close"
    source:      str                 # 데이터 출처 API
    price_time:  Optional[datetime]  # 가격 생성 시각 (ET)
    fetch_time:  datetime = field(
        default_factory=lambda: datetime.now(KST_TZ)
    )
    notes: str = ""

    # ── 표시용 ────────────────────────────────────────────────────────────────
    def __str__(self) -> str:
        arrow  = "▲" if self.change >= 0 else "▼"
        sign   = "+" if self.change >= 0 else ""
        clr    = "\033[32m" if self.change >= 0 else "\033[31m"
        reset  = "\033[0m"
        t_str  = (
            self.price_time.strftime("%Y-%m-%d %H:%M:%S %Z")
            if self.price_time else "N/A"
        )
        lines = [
            f"┌─ {self.ticker} ─────────────────────────────",
            f"│  가격     : {clr}${self.price:,.4f}  "
            f"{arrow} {sign}{self.change:,.4f} ({sign}{self.change_pct:.2f}%){reset}",
            f"│  전일종가 : ${self.prev_close:,.4f}",
            f"│  세션     : {self.session.label_ko()} ({self.session.value})",
            f"│  가격유형 : {self.price_type}",
            f"│  출처     : {self.source}",
            f"│  가격시각 : {t_str}",
            f"│  조회시각 : {self.fetch_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        ]
        if self.notes:
            lines.append(f"│  비고     : {self.notes}")
        lines.append("└" + "─" * 44)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker":      self.ticker,
            "price":       self.price,
            "prev_close":  self.prev_close,
            "change":      self.change,
            "change_pct":  self.change_pct,
            "session":     self.session.value,
            "price_type":  self.price_type,
            "source":      self.source,
            "price_time":  self.price_time.isoformat() if self.price_time else None,
            "fetch_time":  self.fetch_time.isoformat(),
            "notes":       self.notes,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 세션 감지
# ──────────────────────────────────────────────────────────────────────────────
def now_et() -> datetime:
    """현재 시각을 ET(DST 자동 적용) 기준으로 반환."""
    return datetime.now(ET_TZ)


def now_kst() -> datetime:
    return datetime.now(KST_TZ)


def is_trading_day(dt: datetime) -> bool:
    """NYSE 거래일 여부 판단 (주말 + 공휴일 제외)."""
    d = dt.astimezone(ET_TZ).date()
    if d.weekday() >= 5:          # 토(5)/일(6)
        return False
    return d not in NYSE_HOLIDAYS


def get_regular_close_time(dt: datetime) -> int:
    """정규장 종료 시각 (ET 기준 분, 기본 960=16:00, 조기종료일 780=13:00)."""
    d = dt.astimezone(ET_TZ).date()
    return 780 if d in NYSE_EARLY_CLOSE else 960


def detect_session(dt: Optional[datetime] = None) -> Tuple[MarketSession, datetime]:
    """
    주어진(또는 현재) ET 시각을 분석해 시장 세션을 반환.

    Returns
    -------
    (MarketSession, dt_et)
        dt_et: 입력값이 있으면 ET로 변환, 없으면 현재 ET 시각
    """
    dt_et = (dt.astimezone(ET_TZ) if dt else now_et())

    if not is_trading_day(dt_et):
        return MarketSession.CLOSED, dt_et

    t            = dt_et.hour * 60 + dt_et.minute   # 자정 이후 경과 분
    regular_end  = get_regular_close_time(dt_et)

    if   240 <= t < 570:             # 04:00–09:29
        return MarketSession.PRE_MARKET, dt_et
    elif 570 <= t < regular_end:     # 09:30–15:59 (조기종료: ~12:59)
        return MarketSession.REGULAR, dt_et
    elif regular_end <= t < 1200:    # 16:00–19:59
        return MarketSession.AFTER_HOURS, dt_et
    else:                            # 00:00–03:59 / 20:00–23:59
        return MarketSession.CLOSED, dt_et


def session_info() -> Dict[str, str]:
    """현재 세션 정보를 사람이 읽기 쉬운 dict로 반환."""
    dt_e = now_et()
    dt_k = now_kst()
    sess, _ = detect_session(dt_e)
    return {
        "kst_time":      dt_k.strftime("%Y-%m-%d %H:%M:%S KST"),
        "et_time":       dt_e.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "session":       sess.value,
        "session_ko":    sess.label_ko(),
        "is_trading_day": str(is_trading_day(dt_e)),
        "dst_active":    "EDT(서머타임)" if dt_e.tzname() == "EDT" else "EST(표준시)",
    }


# ──────────────────────────────────────────────────────────────────────────────
# API 클라이언트
# ──────────────────────────────────────────────────────────────────────────────

class _BaseClient:
    """HTTP 요청 공통 로직 + 레이트 리미팅."""

    _min_interval: float = 1.0   # 서브클래스에서 오버라이드

    def __init__(self):
        self._last_call = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        gap = self._min_interval - elapsed
        if gap > 0:
            time.sleep(gap)

    def _get(
        self,
        url: str,
        params: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        label: str = "API",
    ) -> Optional[Any]:
        self._wait()
        try:
            resp = requests.get(
                url, params=params, headers=headers or {},
                timeout=REQUEST_TIMEOUT, verify=True,
            )
            resp.raise_for_status()
            self._last_call = time.monotonic()
            return resp.json()
        except requests.exceptions.SSLError:
            # SSL 검증 실패 시 verify=False로 재시도
            try:
                resp = requests.get(
                    url, params=params, headers=headers or {},
                    timeout=REQUEST_TIMEOUT, verify=False,
                )
                resp.raise_for_status()
                self._last_call = time.monotonic()
                return resp.json()
            except Exception as e:
                logger.warning(f"[{label}] {url} 요청 실패 (SSL 우회 후): {e}")
                return None
        except Exception as e:
            logger.warning(f"[{label}] {url} 요청 실패: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Finnhub
# ─────────────────────────────────────────────────────────────────────────────
class FinnhubClient(_BaseClient):
    """
    Finnhub.io 클라이언트.

    사용 엔드포인트:
    - /quote  : 현재가(c), 전일종가(pc), 타임스탬프(t)
                정규장 중 real-time, 비정규 시간엔 마지막 체결가 (free tier 한계)

    Rate limit: 60 req/min → 1 req/sec 안전 유지
    """
    BASE = "https://finnhub.io/api/v1"
    _min_interval = 1.05   # 60 req/min 안전 마진

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def _fget(self, endpoint: str, params: Dict) -> Optional[dict]:
        if not self.api_key:
            return None
        params["token"] = self.api_key
        return self._get(f"{self.BASE}{endpoint}", params, label="Finnhub")

    def quote(self, ticker: str) -> Optional[dict]:
        """
        /quote 응답 필드:
          c  - 현재가 (last trade price)
          h  - 당일 고가
          l  - 당일 저가
          o  - 당일 시가
          pc - 전일 종가 (항상 신뢰 가능)
          t  - 마지막 체결 Unix timestamp
          dp - 전일 대비 변동률 (%)
          d  - 전일 대비 변동액
        """
        data = self._fget("/quote", {"symbol": ticker})
        if not data:
            return None
        # c=0 이면 데이터 없음
        if not data.get("c"):
            return None
        return data

    def get_price(
        self, ticker: str, session: MarketSession
    ) -> Optional[Tuple[float, float, Optional[datetime]]]:
        q = self.quote(ticker)
        if not q:
            return None

        price      = float(q["c"])
        prev_close = float(q["pc"])
        ts: Optional[datetime] = None
        if q.get("t"):
            try:
                ts = datetime.fromtimestamp(q["t"], tz=ET_TZ)
            except Exception:
                pass

        # 정규장 중 타임스탬프가 15분 이상 오래됐으면 경고
        if ts and session == MarketSession.REGULAR:
            age = (datetime.now(ET_TZ) - ts).total_seconds() / 60
            if age > 15:
                logger.warning(
                    f"[Finnhub] {ticker}: 가격이 {age:.0f}분 전 데이터 (실시간 아닐 수 있음)"
                )
        return price, prev_close, ts


# ─────────────────────────────────────────────────────────────────────────────
# Tiingo
# ─────────────────────────────────────────────────────────────────────────────
class TiingoClient(_BaseClient):
    """
    Tiingo.com 클라이언트.

    사용 엔드포인트:
    - /iex/{ticker}                  : IEX 실시간 (pre/after 부분 커버)
    - /tiingo/daily/{ticker}/prices  : EOD 종가 (장 마감 후 fallback)

    IEX 거래소 extended hours 커버리지:
      Pre-market  : 08:00–09:30 ET (04:00은 미지원)
      After-hours : 16:00–17:00 ET (20:00은 미지원)
    → 04:00–08:00 / 17:00–20:00 구간은 AlphaVantage로 보완

    Rate limit: 500 req/hour (free) → 0.5s 간격으로 안전
    """
    BASE = "https://api.tiingo.com"
    _min_interval = 0.5

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self._headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Token {api_key}",
        }

    def _tget(self, path: str, params: Dict = None) -> Optional[Any]:
        if not self.api_key:
            return None
        return self._get(
            f"{self.BASE}{path}", params or {}, headers=self._headers, label="Tiingo"
        )

    def iex(self, ticker: str) -> Optional[dict]:
        """
        IEX 실시간 시세.
        응답 필드:
          last          - 마지막 체결가 (extended hours 포함)
          tngoLast      - Tiingo 통합 마지막가
          timestamp     - 체결 시각 (UTC ISO)
          quoteTimestamp- 시세 업데이트 시각
          prevClose     - 전일 종가
          open/high/low/mid
        """
        data = self._tget(f"/iex/{ticker.lower()}")
        if not data:
            return None
        # 응답이 리스트인 경우
        if isinstance(data, list):
            return data[0] if data else None
        return data

    def _parse_ts(self, ts_str: Optional[str]) -> Optional[datetime]:
        if not ts_str:
            return None
        try:
            return (
                datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                .astimezone(ET_TZ)
            )
        except Exception:
            return None

    def get_price(
        self, ticker: str, session: MarketSession
    ) -> Optional[Tuple[float, float, Optional[datetime]]]:
        q = self.iex(ticker)
        if not q:
            return None

        # last 우선, 없으면 tngoLast
        price = q.get("last") or q.get("tngoLast")
        if not price:
            return None

        prev_close = float(q.get("prevClose") or 0)
        ts = self._parse_ts(q.get("timestamp") or q.get("quoteTimestamp"))

        # extended 세션에서 가격 시각이 60분 이상 오래됐으면 데이터 신뢰도 낮음
        if ts and session.is_extended:
            age_min = (datetime.now(ET_TZ) - ts).total_seconds() / 60
            if age_min > 60:
                logger.warning(
                    f"[Tiingo IEX] {ticker}: extended 가격이 {age_min:.0f}분 경과 "
                    f"(IEX 커버리지 외 구간일 수 있음)"
                )
        return float(price), prev_close, ts

    def get_eod(
        self, ticker: str
    ) -> Optional[Tuple[float, float, Optional[datetime]]]:
        """최근 EOD 종가 (장 마감 이후 fallback)."""
        start = (date.today() - timedelta(days=7)).isoformat()
        data = self._tget(
            f"/tiingo/daily/{ticker.lower()}/prices",
            params={"startDate": start, "resampleFreq": "daily"},
        )
        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        latest = data[-1]
        close  = latest.get("adjClose") or latest.get("close")
        prev_r = data[-2] if len(data) >= 2 else None
        prev   = (prev_r.get("adjClose") or prev_r.get("close")) if prev_r else close
        ts = self._parse_ts(latest.get("date"))
        if not close:
            return None
        return float(close), float(prev or close), ts


# ─────────────────────────────────────────────────────────────────────────────
# Alpha Vantage
# ─────────────────────────────────────────────────────────────────────────────
class AlphaVantageClient(_BaseClient):
    """
    Alpha Vantage 클라이언트.

    사용 엔드포인트:
    - GLOBAL_QUOTE           : 최신 가격 (정규장 전용, extended 미지원)
    - TIME_SERIES_INTRADAY   : 1분봉 + extended_hours=true (04:00–20:00 ET 전체)
    - TIME_SERIES_DAILY      : 일봉 EOD 가격

    ⚠ Free tier: 25 req/day → 반드시 마지막 수단으로 사용 + 캐싱 적용

    캐시 전략:
    - 결과를 메모리에 캐싱 (기본 TTL: 90초)
    - 동일 티커 연속 요청 시 캐시 반환으로 일일 할당량 보존
    """
    BASE = "https://www.alphavantage.co/query"
    _min_interval = 12.0    # free: 25/day ≈ 1/~3500s; 여러 티커 고려해 12s

    def __init__(self, api_key: str, cache_ttl: int = 90):
        super().__init__()
        self.api_key   = api_key
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}

    def _avget(self, params: Dict, cache_key: str) -> Optional[dict]:
        if not self.api_key:
            return None

        # 캐시 확인
        if cache_key in self._cache:
            cached_data, cached_at = self._cache[cache_key]
            if time.monotonic() - cached_at < self._cache_ttl:
                logger.debug(f"[AlphaVantage] 캐시 반환: {cache_key}")
                return cached_data

        params["apikey"] = self.api_key
        data = self._get(self.BASE, params, label="AlphaVantage")
        if not data:
            return None

        # AV 에러 응답 처리
        if "Error Message" in data:
            logger.warning(f"[AlphaVantage] API 에러: {data['Error Message']}")
            return None
        if "Note" in data:
            # Rate limit 초과 메시지
            logger.warning(f"[AlphaVantage] Rate limit 경고: {data['Note']}")
            return None
        if "Information" in data:
            # 유료 플랜 전용 엔드포인트 접근 시 반환되는 안내 메시지
            # (WARNING → DEBUG로 하향: 정상적인 폴백 동작이므로 노이즈 아님)
            logger.debug(f"[AlphaVantage] 프리미엄 전용 엔드포인트 — 폴백으로 전환")
            return None

        self._cache[cache_key] = (data, time.monotonic())
        return data

    def global_quote(
        self, ticker: str
    ) -> Optional[Tuple[float, float, Optional[datetime]]]:
        """
        GLOBAL_QUOTE: 가장 최근 거래일 종가.
        주요 필드:
          '05. price'           - 최신 가격 (정규장 종가, extended 미반영)
          '08. previous close'  - 전일 종가
          '07. latest trading day'
        """
        data = self._avget(
            {"function": "GLOBAL_QUOTE", "symbol": ticker},
            cache_key=f"gq:{ticker}",
        )
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

        return float(price), float(prev or 0), ts

    def intraday_extended(
        self, ticker: str
    ) -> Optional[Tuple[float, float, Optional[datetime]]]:
        """
        TIME_SERIES_INTRADAY + extended_hours=true.

        ⚠ 유료 플랜 전용 엔드포인트 (무료 키로는 None 반환).
          → 무료 플랜: https://www.alphavantage.co/premium/
          → 실패 시 _avget이 None을 반환하고 상위 폴백 체인(Finnhub)으로 넘어감.

        AV extended hours 커버리지: 04:00–20:00 ET (Pre + After 전체)
        최신 1분봉 close 가격을 반환, prev_close는 GLOBAL_QUOTE에서 보완.
        """
        data = self._avget(
            {
                "function":       "TIME_SERIES_INTRADAY",
                "symbol":         ticker,
                "interval":       "1min",
                "extended_hours": "true",
                "outputsize":     "compact",
            },
            cache_key=f"eid:{ticker}",
        )
        if not data:
            return None

        series = data.get("Time Series (1min)", {})
        if not series:
            return None

        # 가장 최신 봉 선택 (사전순 내림차순 → 최신)
        latest_key = max(series.keys())
        bar        = series[latest_key]
        price      = float(bar.get("4. close", 0))
        if not price:
            return None

        # 타임스탬프: AV intraday는 ET 기준 (메타데이터 확인)
        ts: Optional[datetime] = None
        try:
            ts = datetime.strptime(latest_key, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ET_TZ)
        except Exception:
            pass

        # prev_close: GLOBAL_QUOTE에서 보완 (캐싱 덕분에 추가 API 호출 최소화)
        prev_close = 0.0
        gq = self.global_quote(ticker)
        if gq:
            prev_close = gq[1]

        return price, prev_close, ts

    def daily_close(
        self, ticker: str
    ) -> Optional[Tuple[float, float, Optional[datetime]]]:
        """TIME_SERIES_DAILY: EOD fallback."""
        data = self._avget(
            {
                "function":   "TIME_SERIES_DAILY",
                "symbol":     ticker,
                "outputsize": "compact",
            },
            cache_key=f"daily:{ticker}",
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

        return price, prev_c, ts


# ──────────────────────────────────────────────────────────────────────────────
# 메인 오케스트레이터
# ──────────────────────────────────────────────────────────────────────────────
class USStockPriceFetcher:
    """
    세션 인식 멀티소스 미국 주가 수집기.

    사용법:
        fetcher = USStockPriceFetcher(
            finnhub_key="...",
            tiingo_key="...",
            av_key="...",
        )
        result = fetcher.fetch("AAPL")
        print(result)
    """

    def __init__(
        self,
        finnhub_key: str = FINNHUB_KEY,
        tiingo_key:  str = TIINGO_KEY,
        av_key:      str = AV_KEY,
    ):
        self.finnhub = FinnhubClient(finnhub_key)
        self.tiingo  = TiingoClient(tiingo_key)
        self.av      = AlphaVantageClient(av_key)

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────────

    def _build(
        self,
        ticker:     str,
        price:      float,
        prev_close: float,
        session:    MarketSession,
        price_type: str,
        source:     str,
        ts:         Optional[datetime],
        notes:      str = "",
    ) -> Optional[PriceResult]:
        """PriceResult 생성 + 기본 유효성 검사."""
        if price <= 0:
            logger.warning(f"[{source}] {ticker}: 유효하지 않은 가격 {price}")
            return None
        # prev_close가 0인 경우 price로 대체 (변동률 계산 방지)
        if prev_close <= 0:
            prev_close = price
        change     = round(price - prev_close, 6)
        change_pct = round(change / prev_close * 100, 4)
        return PriceResult(
            ticker=ticker, price=round(price, 4), prev_close=round(prev_close, 4),
            change=change, change_pct=change_pct, session=session,
            price_type=price_type, source=source, price_time=ts, notes=notes,
        )

    def _from_finnhub(
        self, ticker: str, session: MarketSession
    ) -> Optional[PriceResult]:
        res = self.finnhub.get_price(ticker, session)
        if not res:
            return None
        price, prev_close, ts = res
        ptype = "real_time" if session == MarketSession.REGULAR else "last_trade"
        note  = "" if session == MarketSession.REGULAR else \
                "Finnhub 무료 플랜 — extended hours 데이터 제한 가능"
        return self._build(ticker, price, prev_close, session, ptype, "Finnhub", ts, note)

    def _from_tiingo(
        self, ticker: str, session: MarketSession
    ) -> Optional[PriceResult]:
        res = self.tiingo.get_price(ticker, session)
        if not res:
            return None
        price, prev_close, ts = res

        # IEX extended 커버리지 외 구간 감지 (04:00–08:00 ET, 17:00–20:00 ET)
        note = ""
        if ts and session.is_extended:
            age_min = (datetime.now(ET_TZ) - ts).total_seconds() / 60
            if age_min > 60:
                note = f"IEX 커버리지 외 구간 — 가격이 {age_min:.0f}분 오래됨"

        ptype = "real_time" if session == MarketSession.REGULAR else "extended"
        return self._build(ticker, price, prev_close, session, ptype, "Tiingo IEX", ts, note)

    def _from_av_intraday(
        self, ticker: str, session: MarketSession
    ) -> Optional[PriceResult]:
        """AV extended hours intraday (04:00–20:00 ET 전체 커버)."""
        res = self.av.intraday_extended(ticker)
        if not res:
            return None
        price, prev_close, ts = res
        return self._build(
            ticker, price, prev_close, session, "extended",
            "AlphaVantage Intraday(ext)", ts,
            "AV TIME_SERIES_INTRADAY extended_hours=true",
        )

    def _from_av_global(
        self, ticker: str, session: MarketSession
    ) -> Optional[PriceResult]:
        res = self.av.global_quote(ticker)
        if not res:
            return None
        price, prev_close, ts = res
        return self._build(
            ticker, price, prev_close, session, "real_time", "AlphaVantage Global", ts
        )

    def _from_tiingo_eod(
        self, ticker: str, session: MarketSession
    ) -> Optional[PriceResult]:
        res = self.tiingo.get_eod(ticker)
        if not res:
            return None
        price, prev_close, ts = res
        return self._build(
            ticker, price, prev_close, session, "last_close", "Tiingo EOD", ts
        )

    def _from_av_daily(
        self, ticker: str, session: MarketSession
    ) -> Optional[PriceResult]:
        res = self.av.daily_close(ticker)
        if not res:
            return None
        price, prev_close, ts = res
        return self._build(
            ticker, price, prev_close, session, "last_close", "AlphaVantage Daily", ts
        )

    def _first_valid(self, *results: Optional[PriceResult]) -> Optional[PriceResult]:
        """첫 번째 유효한 결과 반환 (None이 아니고 price > 0)."""
        for r in results:
            if r is not None and r.price > 0:
                return r
        return None

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def fetch(
        self,
        ticker: str,
        override_dt: Optional[datetime] = None,
    ) -> Optional[PriceResult]:
        """
        `ticker`의 현재 최정확 미국 주가를 반환.

        Parameters
        ----------
        ticker       : US 주식 티커 (예: "AAPL", "TSLA", "NVDA")
        override_dt  : 테스트용 시각 오버라이드 (None이면 현재 시각 사용)

        Returns
        -------
        PriceResult 또는 None (모든 소스 실패 시)

        세션별 API 우선순위
        -------------------
        REGULAR:     Finnhub → Tiingo IEX → AV Global
        PRE_MARKET:  Tiingo IEX → AV Intraday(ext) → Finnhub → EOD fallback
        AFTER_HOURS: Tiingo IEX → AV Intraday(ext) → Finnhub → EOD fallback
        CLOSED:      Finnhub(prev_close) → Tiingo EOD → AV Daily
        """
        ticker  = ticker.upper().strip()
        session, dt_et = detect_session(override_dt)

        logger.info(
            f"[Fetcher] {ticker} | {session.label_ko()} ({session.value}) | "
            f"ET {dt_et.strftime('%H:%M')} | "
            f"KST {dt_et.astimezone(KST_TZ).strftime('%H:%M')}"
        )

        # ── 정규장 (09:30–15:59 ET) ───────────────────────────────────────────
        if session == MarketSession.REGULAR:
            # Finnhub 실시간 → Tiingo IEX → AV Global
            result = self._first_valid(
                self._from_finnhub(ticker, session),
                self._from_tiingo(ticker, session),
                self._from_av_global(ticker, session),
            )
            if result:
                return result

        # ── 프리마켓 (04:00–09:29 ET) ─────────────────────────────────────────
        elif session == MarketSession.PRE_MARKET:
            # Tiingo IEX (08:00 ET 이후 커버)
            # → AV Intraday extended (04:00 ET부터 전체 커버, 가장 정확)
            # → Finnhub quote (extended 데이터 불안정)
            result = self._first_valid(
                self._from_tiingo(ticker, session),
                self._from_av_intraday(ticker, session),
                self._from_finnhub(ticker, session),
            )
            if result:
                return result
            # 모든 extended 소스 실패 → 전일 종가 반환
            eod = self._first_valid(
                self._from_tiingo_eod(ticker, session),
                self._from_av_daily(ticker, session),
            )
            if eod:
                eod.notes = "프리마켓 데이터 없음 — 전일 종가 표시"
                eod.price_type = "last_close"
                return eod

        # ── 애프터마켓 (16:00–19:59 ET) ──────────────────────────────────────
        elif session == MarketSession.AFTER_HOURS:
            # Tiingo IEX (17:00 ET까지 커버)
            # → AV Intraday extended (20:00 ET까지 전체 커버)
            # → Finnhub (마지막 체결가, extended 불안정)
            result = self._first_valid(
                self._from_tiingo(ticker, session),
                self._from_av_intraday(ticker, session),
                self._from_finnhub(ticker, session),
            )
            if result:
                return result
            eod = self._first_valid(
                self._from_tiingo_eod(ticker, session),
                self._from_av_daily(ticker, session),
            )
            if eod:
                eod.notes = "애프터마켓 데이터 없음 — 당일 종가 표시"
                eod.price_type = "last_close"
                return eod

        # ── 장 마감 (20:00–03:59 ET / 주말 / 공휴일) ─────────────────────────
        elif session == MarketSession.CLOSED:
            # 거래 없음 → 가장 최근 종가 반환
            # Finnhub pc(전일종가) 필드가 가장 안정적
            fh_raw = self.finnhub.quote(ticker)
            if fh_raw:
                # 장이 막 끝난 직후면 c가 당일 종가, 아니면 pc(전일)를 사용
                price = float(fh_raw.get("c") or fh_raw.get("pc") or 0)
                prev  = float(fh_raw.get("pc") or price)
                ts: Optional[datetime] = None
                if fh_raw.get("t"):
                    try:
                        ts = datetime.fromtimestamp(fh_raw["t"], tz=ET_TZ)
                    except Exception:
                        pass
                r = self._build(
                    ticker, price, prev, session, "last_close", "Finnhub",
                    ts, "장 마감 — 가장 최근 종가"
                )
                if r:
                    return r

            eod = self._first_valid(
                self._from_tiingo_eod(ticker, session),
                self._from_av_daily(ticker, session),
            )
            if eod:
                eod.notes = "장 마감 — EOD 종가"
                return eod

        logger.error(f"[Fetcher] 모든 소스 실패: {ticker}")
        return None

    def fetch_batch(
        self,
        tickers: List[str],
        delay_between: float = 0.0,
    ) -> Dict[str, Optional[PriceResult]]:
        """
        여러 티커 일괄 조회.

        Parameters
        ----------
        tickers        : 티커 리스트
        delay_between  : 티커 간 추가 딜레이 (초, AV 절약 목적)
        """
        results: Dict[str, Optional[PriceResult]] = {}
        for i, ticker in enumerate(tickers):
            results[ticker] = self.fetch(ticker)
            if delay_between > 0 and i < len(tickers) - 1:
                time.sleep(delay_between)
        return results


# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────────────────────

def kst_to_et(dt_kst: datetime) -> datetime:
    """KST → ET 변환 (DST 자동 처리)."""
    if dt_kst.tzinfo is None:
        dt_kst = dt_kst.replace(tzinfo=KST_TZ)
    return dt_kst.astimezone(ET_TZ)


def et_to_kst(dt_et: datetime) -> datetime:
    """ET → KST 변환."""
    if dt_et.tzinfo is None:
        raise ValueError("ET datetime은 timezone-aware여야 합니다")
    return dt_et.astimezone(KST_TZ)


def get_price(
    ticker: str,
    finnhub_key: str = FINNHUB_KEY,
    tiingo_key:  str = TIINGO_KEY,
    av_key:      str = AV_KEY,
) -> Optional[PriceResult]:
    """단일 종목 빠른 조회 래퍼."""
    return USStockPriceFetcher(finnhub_key, tiingo_key, av_key).fetch(ticker)


# ──────────────────────────────────────────────────────────────────────────────
# CLI 진입점
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys as _sys

    # Windows 터미널 한글 깨짐 방지: stdout/stderr를 UTF-8로 강제 설정
    if hasattr(_sys.stdout, "reconfigure"):
        try:
            _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="US 주가 실시간 조회 (Finnhub / Tiingo / AlphaVantage 폴백 체인)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python us_price_fetcher.py AAPL TSLA NVDA
  python us_price_fetcher.py MSFT --finnhub d1xxx --tiingo abc --av xyz
  FINNHUB_API_KEY=xxx TIINGO_API_KEY=yyy ALPHAVANTAGE_KEY=zzz python us_price_fetcher.py SPY
        """,
    )
    parser.add_argument("tickers", nargs="+", metavar="TICKER",
                        help="미국 주식 티커 (예: AAPL TSLA)")
    parser.add_argument("--finnhub",  default=FINNHUB_KEY,  metavar="KEY",
                        help="Finnhub API 키 (환경변수 FINNHUB_API_KEY 대체 가능)")
    parser.add_argument("--tiingo",   default=TIINGO_KEY,   metavar="KEY",
                        help="Tiingo API 키 (환경변수 TIINGO_API_KEY 대체 가능)")
    parser.add_argument("--av",       default=AV_KEY,       metavar="KEY",
                        help="Alpha Vantage API 키 (환경변수 ALPHAVANTAGE_KEY 대체 가능)")
    parser.add_argument("--json", action="store_true",
                        help="결과를 JSON으로 출력")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="티커 간 추가 딜레이 (초, AV 할당량 절약)")
    args = parser.parse_args()

    # ── 세션 정보 출력 ──────────────────────────────────────────────────────
    info = session_info()
    print("\n" + "=" * 60)
    print(f"  KST   : {info['kst_time']}")
    print(f"  ET    : {info['et_time']}")
    print(f"  세션  : {info['session_ko']} ({info['session']})")
    print(f"  거래일: {info['is_trading_day']}")
    print(f"  시간대: {info['dst_active']}")
    print("=" * 60 + "\n")

    fetcher = USStockPriceFetcher(
        finnhub_key=args.finnhub,
        tiingo_key=args.tiingo,
        av_key=args.av,
    )

    if args.json:
        import json
        results = fetcher.fetch_batch(args.tickers, delay_between=args.delay)
        out = {t: (r.to_dict() if r else None) for t, r in results.items()}
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    else:
        for ticker in args.tickers:
            result = fetcher.fetch(ticker)
            if result:
                print(result)
            else:
                print(f"❌ [{ticker}] 모든 소스에서 가격 조회 실패\n")
            if args.delay > 0 and ticker != args.tickers[-1]:
                time.sleep(args.delay)
