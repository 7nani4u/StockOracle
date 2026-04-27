#!/usr/bin/env python3
"""
us_price_fetcher.py  (StockOracle 프로젝트용)
================================================================================
KST 기준 언제 실행해도 항상 가장 최신·정확한 미국 주가를 반환합니다.

[실제 테스트로 확인된 각 API 특성]
  ┌───────────────┬────────────────────────────────────────────────────────────┐
  │ API           │ 특성                                                       │
  ├───────────────┼────────────────────────────────────────────────────────────┤
  │ yfinance      │ preMarketPrice / postMarketPrice 필드로 Extended 정확 지원 │
  │               │ → 가장 신뢰할 수 있는 무료 Extended Hours 소스             │
  ├───────────────┼────────────────────────────────────────────────────────────┤
  │ Tiingo IEX    │ tngoLast = IEX 집계 가격 (Extended 대략 ±0.3$ 오차)       │
  │               │ last 필드는 Extended 에서 None 반환 (IEX 거래 없을 때)     │
  ├───────────────┼────────────────────────────────────────────────────────────┤
  │ Finnhub (무료)│ c 필드 = 마지막 정규장 종가 고정 (Extended 미지원 확인)    │
  │               │ timestamp = 이전 거래일 16:00 ET (3일 전 데이터)           │
  │               │ → Extended Hours 에서는 사용 불가, 정규장 fallback 전용    │
  ├───────────────┼────────────────────────────────────────────────────────────┤
  │ AlphaVantage  │ GLOBAL_QUOTE = 정규장 종가 (무료 가능)                     │
  │               │ INTRADAY extended_hours = 유료 플랜 전용 (무료 미지원)     │
  └───────────────┴────────────────────────────────────────────────────────────┘

[세션별 API 우선순위 — 실측 기반 재설계]
  ┌──────────────┬──────────────────────────────────────────────────────────────┐
  │ Session      │ 우선순위                                                     │
  ├──────────────┼──────────────────────────────────────────────────────────────┤
  │ PRE_MARKET   │ yfinance(preMarketPrice) → Tiingo(tngoLast) → 전일종가       │
  │ REGULAR      │ Tiingo IEX(last) → yfinance(currentPrice) → Finnhub(c) → AV │
  │ AFTER_HOURS  │ yfinance(postMarketPrice) → Tiingo(tngoLast) → 당일종가      │
  │ CLOSED       │ yfinance(previousClose) → Finnhub(pc) → Tiingo EOD → AV     │
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
# NYSE 공휴일 / 조기 종료일 (2025–2026)
# ──────────────────────────────────────────────────────────────────────────────
NYSE_HOLIDAYS: frozenset[date] = frozenset({
    date(2025, 1, 1),  date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4),  date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    date(2026, 1, 1),  date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3),  date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
})
NYSE_EARLY_CLOSE: frozenset[date] = frozenset({
    date(2025, 11, 28), date(2025, 12, 24),
    date(2026, 11, 27), date(2026, 12, 24),
})


# ──────────────────────────────────────────────────────────────────────────────
# 세션 Enum + PriceResult 데이터 클래스
# ──────────────────────────────────────────────────────────────────────────────
class MarketSession(Enum):
    PRE_MARKET  = "pre_market"
    REGULAR     = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED      = "closed"

    @property
    def is_extended(self) -> bool:
        return self in (MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS)

    def label_ko(self) -> str:
        return {"pre_market": "프리마켓", "regular": "정규장",
                "after_hours": "애프터마켓", "closed": "장마감"}[self.value]


@dataclass
class PriceResult:
    ticker:      str
    price:       float          # 현재가 (세션에 맞는 최신가)
    prev_close:  float          # 전일 정규장 종가
    change:      float          # 전일 종가 대비 변동액
    change_pct:  float          # 전일 종가 대비 변동률 (%)
    session:     MarketSession
    price_type:  str            # "real_time" | "pre_market" | "post_market" | "last_close"
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
        lines = [
            f"┌─ {self.ticker} {'─'*40}",
            f"│  가격     : {clr}${self.price:>10,.4f}  "
            f"{arr} {sign}{self.change:,.4f} ({sign}{self.change_pct:.2f}%){RST}",
            f"│  전일종가 : ${self.prev_close:>10,.4f}",
            f"│  세션     : {self.session.label_ko()} ({self.session.value})",
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
            "ticker": self.ticker, "price": self.price,
            "prev_close": self.prev_close, "change": self.change,
            "change_pct": self.change_pct, "session": self.session.value,
            "price_type": self.price_type, "source": self.source,
            "price_time": self.price_time.isoformat() if self.price_time else None,
            "fetch_time": self.fetch_time.isoformat(), "notes": self.notes,
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
    """현재(또는 지정) 시각의 NYSE 세션 반환."""
    dt_et = dt.astimezone(ET_TZ) if dt else now_et()
    if not is_trading_day(dt_et):
        return MarketSession.CLOSED, dt_et
    t = dt_et.hour * 60 + dt_et.minute
    reg_end = 780 if dt_et.date() in NYSE_EARLY_CLOSE else 960
    if   240 <= t < 570:    return MarketSession.PRE_MARKET,  dt_et
    elif 570 <= t < reg_end: return MarketSession.REGULAR,     dt_et
    elif reg_end <= t < 1200: return MarketSession.AFTER_HOURS, dt_et
    else:                    return MarketSession.CLOSED,      dt_et


def session_info() -> Dict[str, str]:
    dt_e = now_et(); dt_k = now_kst()
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
# yfinance 클라이언트 (Extended Hours 주력 소스)
# ──────────────────────────────────────────────────────────────────────────────
class YFinanceClient:
    """
    yfinance 기반 가격 조회.

    세션별 핵심 필드:
      PRE_MARKET  → info["preMarketPrice"]      + info["preMarketTime"]
      REGULAR     → info["currentPrice"]         + info["regularMarketTime"]
      AFTER_HOURS → info["postMarketPrice"]      + info["postMarketTime"]
      CLOSED      → info["previousClose"]        (전일 정규장 종가)

    특징:
    - Yahoo Finance / Cboe 데이터 기반 → 브로커 표시가와 가장 근접
    - 무료, 별도 API 키 불필요
    - preMarketPrice: 04:00 ET 이후 pre-market 실제 체결가 반영
    - postMarketPrice: 16:00~20:00 ET after-hours 실제 체결가 반영
    """

    _min_interval = 0.5   # yfinance 자체 rate limit 대응

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

    def get_price(
        self, ticker: str, session: MarketSession
    ) -> Optional[Tuple[float, float, Optional[datetime], str]]:
        """
        Returns (price, prev_close, price_time, price_type) or None.
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
            # 데이터가 60분 이상 오래된 경우 경고 (프리마켓 중 거래 없을 수 있음)
            if ts:
                age = (now_et() - ts).total_seconds() / 60
                if age > 60:
                    logger.warning(f"[yfinance] {ticker}: preMarketPrice가 {age:.0f}분 전 데이터")
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
            price = info.get("postMarketPrice")
            if not price:
                # After-hours 데이터 없으면 당일 종가 반환
                price = info.get("regularMarketPrice")
                if not price:
                    return None
                ts = self._ts(info.get("regularMarketTime"))
                return float(price), prev_close, ts, "regular_close"
            ts = self._ts(info.get("postMarketTime"))
            return float(price), prev_close, ts, "post_market"

        # ── CLOSED ───────────────────────────────────────────────────────────
        elif session == MarketSession.CLOSED:
            # 전일 정규장 종가 반환
            price = prev_close or info.get("regularMarketPrice")
            if not price:
                return None
            ts = self._ts(info.get("regularMarketTime"))
            return float(price), prev_close, ts, "last_close"

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
        """
        q = self.iex(ticker)
        if not q:
            return None

        prev_close = float(q.get("prevClose") or 0)
        ts         = self._parse_ts(q.get("timestamp"))

        if session == MarketSession.REGULAR:
            # 정규장: last(IEX 실시간) 우선
            price = q.get("last") or q.get("tngoLast")
            if not price:
                return None
            # 15분 이상 오래된 정규장 데이터는 경고
            if ts:
                age = (now_et() - ts).total_seconds() / 60
                if age > 15:
                    logger.warning(f"[Tiingo] {ticker}: 정규장 데이터 {age:.0f}분 경과")
            return float(price), prev_close, ts, "real_time"

        elif session.is_extended:
            # Extended: tngoLast 사용 (last는 종종 None)
            price = q.get("tngoLast") or q.get("last")
            if not price:
                return None
            # 타임스탬프 신선도 검증 (60분 초과 → 신뢰도 낮음)
            if ts:
                age = (now_et() - ts).total_seconds() / 60
                if age > 60:
                    logger.info(
                        f"[Tiingo] {ticker}: Extended 데이터 {age:.0f}분 경과 "
                        f"(IEX 커버리지 외 구간일 수 있음) → yfinance가 더 정확"
                    )
                    return None   # 오래된 데이터는 스킵 → 상위 fallback 사용
            return float(price), prev_close, ts, "extended"

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
      - → Extended 세션에서는 사용 불가 (정규장 fallback 전용)
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
          Finnhub 데이터는 사용하지 않음 (이전 정규장 종가이므로 부정확).
          정규장 (REGULAR) 에서만 데이터 사용.
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

        # ── 타임스탬프 신선도 검증 ───────────────────────────────────────────
        # 무료 플랜: Extended/Closed 에서 timestamp = 이전 거래일 16:00 (수일 전)
        # 현재 거래일의 데이터인지 확인
        if ts:
            now = now_et()
            ts_date = ts.date()
            # 타임스탬프가 2거래일 이상 이전이면 스킵
            # (주말 고려: 금요일 종가라면 월~일 모두 허용)
            days_old = (now.date() - ts_date).days
            if days_old > 3:
                logger.debug(
                    f"[Finnhub] {ticker}: timestamp {ts_date} → {days_old}일 전 데이터 "
                    f"(Extended 세션 — 사용 안 함)"
                )
                return None

        if session == MarketSession.REGULAR:
            return price, prev_close, ts, "real_time"
        elif session == MarketSession.CLOSED:
            # CLOSED 에서는 pc(전일종가) 기준으로만 사용
            return float(q.get("pc") or price), prev_close, ts, "last_close"
        else:
            # Extended 세션: Finnhub 무료는 정확하지 않으므로 사용 안 함
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
            # 유료 전용 엔드포인트 접근 → 조용히 스킵
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

    [실측 기반 API 우선순위]

    PRE_MARKET (04:00–09:30 ET / KST ~17:00–22:30)
      1. yfinance preMarketPrice  ← 가장 정확 (Yahoo Finance/Cboe 실데이터)
      2. Tiingo tngoLast           ← 합리적 근사 (yfinance ±0.3$ 수준)
      3. 전일종가 fallback

    REGULAR (09:30–16:00 ET / KST ~22:30–05:00)
      1. Tiingo IEX last           ← IEX 실시간 (가장 빠름)
      2. yfinance currentPrice     ← Yahoo 실시간
      3. Finnhub c                 ← 약간 지연 가능
      4. AV Global Quote           ← 종가 수준 (마지막 fallback)

    AFTER_HOURS (16:00–20:00 ET / KST ~05:00–09:00)
      1. yfinance postMarketPrice  ← 가장 정확
      2. Tiingo tngoLast           ← 합리적 근사 (17:00 ET 이후 신뢰도 저하)
      3. 당일종가 fallback

    CLOSED (20:00–04:00 ET / 주말/공휴일)
      1. yfinance previousClose    ← 가장 최근 정규장 종가
      2. Finnhub pc                ← 신뢰도 높은 전일 종가
      3. Tiingo EOD
      4. AV Daily

    사용법:
        fetcher = USStockPriceFetcher()
        result  = fetcher.fetch("AAPL")
        print(result)
    """

    def __init__(
        self,
        finnhub_key: str = FINNHUB_KEY,
        tiingo_key:  str = TIINGO_KEY,
        av_key:      str = AV_KEY,
    ):
        self.yf      = YFinanceClient()
        self.tiingo  = TiingoClient(tiingo_key)
        self.finnhub = FinnhubClient(finnhub_key)
        self.av      = AlphaVantageClient(av_key)

    # ── PriceResult 생성 헬퍼 ──────────────────────────────────────────────────
    def _build(
        self,
        ticker: str,
        raw:    Tuple[float, float, Optional[datetime], str],
        session: MarketSession,
        source:  str,
        notes:   str = "",
    ) -> Optional[PriceResult]:
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

    def _try(self, fn, ticker, session, source, notes="") -> Optional[PriceResult]:
        try:
            raw = fn(ticker, session)
            if raw is None:
                return None
            return self._build(ticker, raw, session, source, notes)
        except Exception as e:
            logger.warning(f"[{source}] {ticker} 실패: {e}")
            return None

    def _first(self, *results: Optional[PriceResult]) -> Optional[PriceResult]:
        return next((r for r in results if r is not None and r.price > 0), None)

    # ── 공개 API ──────────────────────────────────────────────────────────────
    def fetch(
        self,
        ticker:      str,
        override_dt: Optional[datetime] = None,
    ) -> Optional[PriceResult]:
        """
        `ticker`의 현재 시각에 맞는 최정확 미국 주가 반환.

        세션별 우선순위는 클래스 독스트링 참조.
        """
        ticker  = ticker.upper().strip()
        session, dt_et = detect_session(override_dt)
        dt_kst = dt_et.astimezone(KST_TZ)

        logger.info(
            f"[Fetcher] {ticker} | {session.label_ko()} | "
            f"ET {dt_et.strftime('%H:%M')} | KST {dt_kst.strftime('%H:%M')}"
        )

        # ── 1. 프리마켓 (04:00–09:30 ET) ─────────────────────────────────────
        if session == MarketSession.PRE_MARKET:
            result = self._first(
                # yfinance preMarketPrice — 가장 정확 (Yahoo/Cboe 실데이터)
                self._try(self.yf.get_price, ticker, session, "yfinance",
                          "preMarketPrice — Yahoo Finance/Cboe 실시간"),
                # Tiingo tngoLast — yfinance 실패 시 합리적 근사
                self._try(self.tiingo.get_price, ticker, session, "Tiingo IEX",
                          "tngoLast — IEX 집계 (yfinance ±0.3$ 수준)"),
            )
            if result:
                return result
            # 모든 Extended 소스 실패 → 전일종가 반환
            eod = self._first(
                self._try(self.yf.get_price, ticker, MarketSession.CLOSED, "yfinance"),
                self._try(self.tiingo.get_price, ticker, MarketSession.CLOSED, "Tiingo EOD"),
                self._try(self.av.global_quote,  ticker, session, "AlphaVantage"),
            )
            if eod:
                eod.notes = "프리마켓 데이터 없음 (04:00 ET 이전 또는 거래 없음) — 전일종가 표시"
                eod.price_type = "last_close"
            return eod

        # ── 2. 정규장 (09:30–16:00 ET) ───────────────────────────────────────
        elif session == MarketSession.REGULAR:
            return self._first(
                # Tiingo IEX last — IEX 실시간 (가장 빠름)
                self._try(self.tiingo.get_price, ticker, session, "Tiingo IEX",
                          "IEX 실시간 last trade"),
                # yfinance currentPrice — Yahoo 실시간
                self._try(self.yf.get_price, ticker, session, "yfinance",
                          "currentPrice — Yahoo Finance 실시간"),
                # Finnhub c — 마지막 체결가 (약간 지연 가능)
                self._try(self.finnhub.get_price, ticker, session, "Finnhub",
                          "last trade (약간 지연 가능)"),
                # AV Global Quote — 마지막 fallback
                self._try(self.av.global_quote,  ticker, session, "AlphaVantage"),
            )

        # ── 3. 애프터마켓 (16:00–20:00 ET) ───────────────────────────────────
        elif session == MarketSession.AFTER_HOURS:
            result = self._first(
                # yfinance postMarketPrice — 가장 정확 (Yahoo/Cboe 실데이터)
                self._try(self.yf.get_price, ticker, session, "yfinance",
                          "postMarketPrice — Yahoo Finance/Cboe 실시간"),
                # Tiingo tngoLast — yfinance 실패 시 (17:00 ET 이후 신뢰도 저하)
                self._try(self.tiingo.get_price, ticker, session, "Tiingo IEX",
                          "tngoLast — IEX 집계 (17:00 ET 이후 신뢰도 낮음)"),
            )
            if result:
                return result
            # 애프터마켓 데이터 없으면 당일 정규장 종가
            eod = self._first(
                self._try(self.yf.get_price, ticker, MarketSession.CLOSED, "yfinance"),
                self._try(self.tiingo.get_price, ticker, MarketSession.CLOSED, "Tiingo EOD"),
                self._try(self.av.global_quote,  ticker, session, "AlphaVantage"),
            )
            if eod:
                eod.notes = "애프터마켓 데이터 없음 — 당일 정규장 종가 표시"
                eod.price_type = "last_close"
            return eod

        # ── 4. 장마감 (20:00–04:00 ET / 주말 / 공휴일) ───────────────────────
        elif session == MarketSession.CLOSED:
            result = self._first(
                # yfinance previousClose — 가장 최근 정규장 종가
                self._try(self.yf.get_price, ticker, session, "yfinance",
                          "previousClose — 마지막 정규장 종가"),
                # Finnhub pc — 전일종가 신뢰도 높음
                self._try(self.finnhub.get_price, ticker, session, "Finnhub",
                          "pc — 전일 정규장 종가"),
                # Tiingo EOD
                self._try(self.tiingo.get_price, ticker, session, "Tiingo EOD"),
                # AV Daily — 최종 fallback
                self._try(self.av.daily_close,   ticker, session, "AlphaVantage Daily"),
            )
            if result:
                result.notes = result.notes or "장마감 — 가장 최근 정규장 종가"
            return result

        logger.error(f"[Fetcher] 모든 소스 실패: {ticker}")
        return None

    def fetch_batch(
        self,
        tickers: List[str],
        delay_between: float = 0.0,
    ) -> Dict[str, Optional[PriceResult]]:
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
        description="US 주가 실시간 조회 (yfinance 주력 + Tiingo/Finnhub/AV 폴백)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python us_price_fetcher.py AAPL TSLA NVDA
  python us_price_fetcher.py MSFT --json
  python us_price_fetcher.py AAPL TSLA NVDA MSFT --delay 1
        """,
    )
    parser.add_argument("tickers", nargs="+", metavar="TICKER")
    parser.add_argument("--json",  action="store_true", help="JSON 출력")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="티커 간 딜레이(초, AV 할당량 절약)")
    parser.add_argument("--debug", action="store_true", help="DEBUG 로그 출력")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 현재 세션 정보 출력
    info = session_info()
    sep  = "=" * 60
    print(f"\n{sep}")
    print(f"  KST   : {info['kst_time']}")
    print(f"  ET    : {info['et_time']}")
    print(f"  세션  : {info['session_ko']} ({info['session']})")
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
