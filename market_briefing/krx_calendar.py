"""KRX(한국거래소) 거래일 달력과 정규장 시간.

[근거 — 2026-11-04 네이버 일봉 3,000개(2014-08~2026-11)와 대조]
- `holidays.KR` 기본(public) 범주만 쓰면 다음 휴장일이 빠진다.
  · 근로자의 날(5/1): `bank` 범주에만 있다 → categories=("public", "bank").
  · 연말 휴장일: 12월의 마지막 평일(12/31 이 주말이면 12/30·12/29).
  · 2026-07-17 제헌절: 2026년부터 공휴일로 다시 지정됐지만 holidays 0.91 에 없다.
- 과거 날짜는 실제 거래 기록(일봉 날짜)이 진실이다. 달력과 기록이 다르면
  `observed` 를 우선하고 `calendar_mismatches` 로 차이를 드러낸다.
- 예측하지 못한 임시 휴장·개장은 환경변수로 덮어쓴다(코드 수정 없이 운영 대응).
  STOCKORACLE_KRX_EXTRA_HOLIDAYS="2026-12-04,2027-01-02"
  STOCKORACLE_KRX_EXTRA_TRADING_DAYS="..."
  STOCKORACLE_KRX_LATE_OPEN_DAYS="2026-11-19"   (10:00 개장·16:30 마감, 수능일 등)

정규장: 09:00~15:30 KST. 새해 첫 거래일은 10:00 개장, 대학수학능력시험일은 10:00~16:30.
"""
from __future__ import annotations

import datetime as _dt
import functools
import os
from typing import Iterable

try:
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:  # tzdata 가 없는 환경
    KST = _dt.timezone(_dt.timedelta(hours=9))

REGULAR_OPEN = _dt.time(9, 0)
REGULAR_CLOSE = _dt.time(15, 30)
LATE_OPEN = _dt.time(10, 0)
LATE_CLOSE = _dt.time(16, 30)

# 대학수학능력시험일(개장 1시간 지연·마감 1시간 연장). 매년 교육부 공고로 확정된다.
_CSAT_DAYS = {
    _dt.date(2023, 11, 16),
    _dt.date(2024, 11, 14),
    _dt.date(2025, 11, 13),
    _dt.date(2026, 11, 19),
}

# 제헌절은 2008~2025년에는 공휴일이 아니었고 2026년부터 다시 공휴일이다(일봉 기록으로 확인).
_CONSTITUTION_DAY_FROM = 2026


def _parse_dates(raw: str) -> set[_dt.date]:
    out: set[_dt.date] = set()
    for token in (raw or "").replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(_dt.date.fromisoformat(token))
        except ValueError:
            continue
    return out


def _env_dates(name: str) -> set[_dt.date]:
    return _parse_dates(os.getenv(name, ""))


@functools.lru_cache(maxsize=32)
def _library_holidays(year: int) -> dict[_dt.date, str]:
    try:
        import holidays as _hol
    except Exception:
        return {}
    try:
        table = _hol.KR(years=[year], categories=("public", "bank"))
    except Exception:
        try:
            table = _hol.KR(years=[year])
        except Exception:
            return {}
    return {d: str(name) for d, name in table.items()}


def _last_weekday_of_year(year: int) -> _dt.date:
    d = _dt.date(year, 12, 31)
    while d.weekday() >= 5:
        d -= _dt.timedelta(days=1)
    return d


def holiday_name(d: _dt.date) -> str | None:
    """평일 휴장 사유. 거래일이면 None. 주말은 '주말'."""
    if isinstance(d, _dt.datetime):
        d = d.date()
    if d in _env_dates("STOCKORACLE_KRX_EXTRA_TRADING_DAYS"):
        return None
    if d.weekday() >= 5:
        return "주말"
    if d in _env_dates("STOCKORACLE_KRX_EXTRA_HOLIDAYS"):
        return "임시 휴장일"
    lib = _library_holidays(d.year)
    if d in lib:
        name = lib[d]
        return "근로자의 날" if (d.month == 5 and d.day == 1) else name
    if d.month == 5 and d.day == 1:
        return "근로자의 날"
    if d.month == 7 and d.day == 17 and d.year >= _CONSTITUTION_DAY_FROM:
        return "제헌절"
    if d == _last_weekday_of_year(d.year):
        return "연말 휴장일"
    return None


def is_trading_day(d: _dt.date, observed: set[_dt.date] | None = None,
                   observed_until: _dt.date | None = None) -> bool:
    """거래일 여부. observed(실제 일봉 날짜) 가 덮는 과거 구간에서는 기록이 우선한다."""
    if isinstance(d, _dt.datetime):
        d = d.date()
    if observed is not None and observed_until is not None and d <= observed_until:
        if observed and d >= min(observed):
            return d in observed
    return holiday_name(d) is None


def closure_reason(d: _dt.date, observed: set[_dt.date] | None = None,
                   observed_until: _dt.date | None = None) -> str | None:
    """휴장이면 사유 문구, 거래일이면 None."""
    if is_trading_day(d, observed, observed_until):
        return None
    name = holiday_name(d)
    if name:
        return name
    return "휴장(실제 거래 기록 없음)"


def trading_days(start: _dt.date, end: _dt.date, observed: set[_dt.date] | None = None,
                 observed_until: _dt.date | None = None) -> list[_dt.date]:
    out: list[_dt.date] = []
    cur = start
    while cur <= end:
        if is_trading_day(cur, observed, observed_until):
            out.append(cur)
        cur += _dt.timedelta(days=1)
    return out


def next_trading_day(d: _dt.date, observed: set[_dt.date] | None = None,
                     observed_until: _dt.date | None = None) -> _dt.date:
    cur = d + _dt.timedelta(days=1)
    for _ in range(40):
        if is_trading_day(cur, observed, observed_until):
            return cur
        cur += _dt.timedelta(days=1)
    return cur


def previous_trading_day(d: _dt.date, observed: set[_dt.date] | None = None,
                         observed_until: _dt.date | None = None) -> _dt.date:
    cur = d - _dt.timedelta(days=1)
    for _ in range(40):
        if is_trading_day(cur, observed, observed_until):
            return cur
        cur -= _dt.timedelta(days=1)
    return cur


def _is_first_trading_day_of_year(d: _dt.date) -> bool:
    cur = _dt.date(d.year, 1, 1)
    while cur < d:
        if is_trading_day(cur):
            return False
        cur += _dt.timedelta(days=1)
    return True


def session_hours(d: _dt.date) -> tuple[_dt.datetime, _dt.datetime]:
    """정규장 시작·종료 시각(KST, tz-aware)."""
    if isinstance(d, _dt.datetime):
        d = d.date()
    open_t, close_t = REGULAR_OPEN, REGULAR_CLOSE
    late_days = _CSAT_DAYS | _env_dates("STOCKORACLE_KRX_LATE_OPEN_DAYS")
    if d in late_days:
        open_t, close_t = LATE_OPEN, LATE_CLOSE
    elif _is_first_trading_day_of_year(d):
        open_t = LATE_OPEN
    return (_dt.datetime.combine(d, open_t, tzinfo=KST), _dt.datetime.combine(d, close_t, tzinfo=KST))


def calendar_mismatches(observed: Iterable[_dt.date], start: _dt.date | None = None,
                        end: _dt.date | None = None) -> list[dict]:
    """실제 거래 기록과 달력 규칙이 다른 평일 목록(검증·로그용)."""
    obs = set(observed)
    if not obs:
        return []
    start = start or min(obs)
    end = end or max(obs)
    out: list[dict] = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            rule_open = holiday_name(cur) is None
            data_open = cur in obs
            if rule_open != data_open:
                out.append({
                    "date": cur.isoformat(),
                    "calendar": "거래일" if rule_open else f"휴장({holiday_name(cur)})",
                    "observed": "거래 기록 있음" if data_open else "거래 기록 없음",
                })
        cur += _dt.timedelta(days=1)
    return out
