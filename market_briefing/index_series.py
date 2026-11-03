"""코스피·코스닥 일봉, 해외 지표 일봉, 지수 현재가 수집 — 주간 예측 전용 데이터 계층.

[출처 선택 근거 — 2026-11-04 01:00 KST 실측]
- 야후 `^KS11`·`^KQ11` 일봉은 11-03 종가가 9시간이 지나도 NaN 이었다(마지막 확정일 11-02).
  같은 시각 네이버 차트(fchart) 일봉에는 11-03 종가(5,648.79 / 704.39)가 있었고 응답은 0.3초,
  최대 3,000개(2014-08~)를 준다. → 국내 지수는 네이버를 1차, 야후를 폴백으로 쓴다.
- 해외 지표(^GSPC ^IXIC ^SOX ^VIX ^TNX)는 야후 일봉. 미국 장중에는 오늘 세션 막대가 진행 중
  값으로 들어오므로, 세션 날짜별 '확정 시각'(뉴욕 16:30)을 함께 돌려주고 호출자가 거른다.

모든 수집 함수는 (자료, 경고문|None) 을 돌려준다. 실패 시 마지막 정상 자료가 있으면 그것을
`stale=True` 로 표시해 돌려주고, 없으면 None — 임의 값으로 채우지 않는다.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
import threading
import time
from typing import Any, Callable

import requests

try:
    from zoneinfo import ZoneInfo
    _NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - tzdata 미설치 환경
    _NY = _dt.timezone(_dt.timedelta(hours=-5))

from market_briefing.krx_calendar import KST

KR_CODES = {"KOSPI": ("KOSPI", "^KS11"), "KOSDAQ": ("KOSDAQ", "^KQ11")}
GLOBAL_SYMBOLS = {
    "SPX": "^GSPC",   # S&P 500
    "NDX": "^IXIC",   # 나스닥 종합
    "SOX": "^SOX",    # 필라델피아 반도체
    "VIX": "^VIX",    # 변동성 지수
    "TNX": "^TNX",    # 미국 10년물 금리(%)
}
GLOBAL_LABELS = {
    "SPX": "S&P 500", "NDX": "나스닥", "SOX": "필라델피아 반도체", "VIX": "VIX", "TNX": "미국 10년물 금리",
}

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/138 Safari/537.36")
_NAVER_HEADERS = {"User-Agent": _UA, "Referer": "https://m.stock.naver.com/"}
_FCHART_URL = "https://fchart.stock.naver.com/sise.nhn?symbol={sym}&timeframe=day&count={count}&requestType=0"
_BASIC_URL = "https://m.stock.naver.com/api/index/{sym}/basic"

_CACHE: dict[str, tuple[float, Any]] = {}
_LAST_GOOD: dict[str, tuple[float, Any]] = {}
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
LAST_GOOD_MAX_AGE = 3 * 24 * 3600.0


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def cached_fetch(key: str, ttl: float, loader: Callable[[], Any], force: bool = False):
    """TTL 캐시 + 마지막 정상값 폴백. 반환 (data|None, note|None, stale: bool)."""
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit and not force and now - hit[0] < ttl:
        return hit[1], None, False
    with _lock_for(key):
        hit = _CACHE.get(key)
        if hit and not force and time.monotonic() - hit[0] < ttl:
            return hit[1], None, False
        try:
            data = loader()
            error = None if data is not None else "빈 응답"
        except Exception as exc:  # 네트워크·파싱 실패는 폴백으로 처리
            data, error = None, f"{type(exc).__name__}: {exc}"
        if data is not None:
            stamp = time.monotonic()
            _CACHE[key] = (stamp, data)
            _LAST_GOOD[key] = (stamp, data)
            return data, None, False
        good = _LAST_GOOD.get(key)
        if good and time.monotonic() - good[0] < LAST_GOOD_MAX_AGE:
            return good[1], f"{key} 수집 실패({error}) — 마지막 정상 자료 사용", True
        return None, f"{key} 수집 실패({error})", False


def clear_cache() -> None:
    _CACHE.clear()


# ── 국내 지수 일봉 ─────────────────────────────────────────────────────────

def parse_fchart(text: str) -> dict | None:
    """네이버 fchart XML → 날짜 오름차순 OHLCV. 종가가 0 이하·비정상인 행은 버린다."""
    rows = re.findall(r'data="([^"]+)"', text or "")
    by_date: dict[_dt.date, tuple] = {}
    for row in rows:
        parts = row.split("|")
        if len(parts) < 6:
            continue
        try:
            d = _dt.datetime.strptime(parts[0], "%Y%m%d").date()
            o, h, l, c = (float(x) for x in parts[1:5])
            v = float(parts[5]) if parts[5] not in ("", "null") else 0.0
        except ValueError:
            continue
        if not (c > 0 and math.isfinite(c)):
            continue
        o = o if o > 0 else c
        h = max(h, o, c) if h > 0 else max(o, c)
        l = min(l, o, c) if l > 0 else min(o, c)
        by_date[d] = (o, h, l, c, v)
    if not by_date:
        return None
    dates = sorted(by_date)
    return {
        "dates": dates,
        "open": [by_date[d][0] for d in dates],
        "high": [by_date[d][1] for d in dates],
        "low": [by_date[d][2] for d in dates],
        "close": [by_date[d][3] for d in dates],
        "volume": [by_date[d][4] for d in dates],
    }


def _naver_daily(code: str, count: int) -> dict | None:
    sym = KR_CODES[code][0]
    resp = requests.get(_FCHART_URL.format(sym=sym, count=count), headers=_NAVER_HEADERS, timeout=(3, 10))
    resp.raise_for_status()
    parsed = parse_fchart(resp.text)
    if parsed:
        parsed.update({"code": code, "source": "네이버 증권 일봉"})
    return parsed


def _frame_to_series(frame, ohlc: bool = True) -> dict | None:
    if frame is None or getattr(frame, "empty", True):
        return None
    frame = frame.dropna(subset=["Close"])     # 지수 마지막 행 종가 NaN 대응
    frame = frame[frame["Close"] > 0]
    if frame.empty:
        return None
    dates = [ts.date() for ts in frame.index]
    out: dict[str, Any] = {"dates": dates, "close": [float(v) for v in frame["Close"]]}
    tz = getattr(frame.index, "tz", None)
    out["tz"] = str(tz) if tz is not None else None
    if ohlc:
        for col, key in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Volume", "volume")):
            vals = frame[col] if col in frame.columns else frame["Close"]
            out[key] = [float(v) if v == v else 0.0 for v in vals]
    return out


def _yahoo_daily(symbol: str, period: str, ohlc: bool = True) -> dict | None:
    import yfinance as yf
    frame = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False, actions=False)
    return _frame_to_series(frame, ohlc=ohlc)


def fetch_kr_index_daily(code: str, count: int = 3000, ttl: float = 600.0, force: bool = False):
    """코스피/코스닥 일봉. 반환 (data|None, note|None, stale)."""
    def load():
        try:
            data = _naver_daily(code, count)
            if data and len(data["dates"]) >= 60:
                return data
        except Exception as exc:
            print(f"[index_series] 네이버 {code} 일봉 실패 → 야후 폴백: {exc}")
        data = _yahoo_daily(KR_CODES[code][1], "15y")
        if data:
            data.update({"code": code, "source": "야후 파이낸스 일봉(폴백)"})
        return data

    return cached_fetch(f"kr_daily:{code}:{count}", ttl, load, force=force)


def fetch_global_daily(key: str, period: str = "15y", ttl: float = 1800.0, force: bool = False):
    symbol = GLOBAL_SYMBOLS[key]

    def load():
        data = _yahoo_daily(symbol, period, ohlc=False)
        if data:
            data.update({"key": key, "symbol": symbol, "source": "야후 파이낸스 일봉"})
        return data

    return cached_fetch(f"global_daily:{key}:{period}", ttl, load, force=force)


def us_session_complete_at(d: _dt.date) -> _dt.datetime:
    """미국 세션일 d 의 막대가 확정됐다고 볼 시각(뉴욕 16:30 — 조기 폐장일도 보수적으로 같은 기준)."""
    return _dt.datetime.combine(d, _dt.time(16, 30), tzinfo=_NY)


def drop_incomplete_sessions(series: dict | None, now: _dt.datetime) -> tuple[dict | None, list[str]]:
    """now 기준 아직 마감되지 않은 미국 세션 막대를 제외한 사본과 제외된 날짜."""
    if not series:
        return series, []
    keep = [i for i, d in enumerate(series["dates"]) if us_session_complete_at(d) <= now]
    keep_set = set(keep)
    dropped = [d.isoformat() for i, d in enumerate(series["dates"]) if i not in keep_set]
    if not dropped:
        return series, []
    out = dict(series)
    for k, v in series.items():
        if isinstance(v, list) and len(v) == len(series["dates"]):
            out[k] = [v[i] for i in keep]
    return out, dropped


# ── 국내 지수 현재가 ───────────────────────────────────────────────────────

def _num(text: Any) -> float | None:
    try:
        v = float(str(text).replace(",", "").strip())
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def parse_index_basic(payload: dict) -> dict | None:
    """네이버 지수 basic 응답 → 현재가. compareToPreviousClosePrice 는 이미 부호를 포함한다."""
    if not isinstance(payload, dict):
        return None
    price = _num(payload.get("closePrice"))
    diff = _num(payload.get("compareToPreviousClosePrice"))
    pct = _num(payload.get("fluctuationsRatio"))
    if price is None or price <= 0 or diff is None:
        return None
    prev = price - diff
    if prev <= 0:
        return None
    traded_raw = str(payload.get("localTradedAt") or "")
    try:
        traded = _dt.datetime.fromisoformat(traded_raw)
        if traded.tzinfo is None:
            traded = traded.replace(tzinfo=KST)
    except ValueError:
        traded = None
    return {
        "price": price,
        "prev_close": round(prev, 4),
        "change_pct": pct if pct is not None else round(diff / prev * 100.0, 2),
        "market_status": str(payload.get("marketStatus") or ""),
        "traded_at": traded.isoformat() if traded else None,
        "source": "네이버 증권 지수 현재가",
    }


def fetch_kr_index_quote(code: str, ttl: float = 60.0, force: bool = False):
    sym = KR_CODES[code][0]

    def load():
        resp = requests.get(_BASIC_URL.format(sym=sym), headers=_NAVER_HEADERS, timeout=(3, 6))
        resp.raise_for_status()
        return parse_index_basic(resp.json())

    return cached_fetch(f"kr_quote:{code}", ttl, load, force=force)
