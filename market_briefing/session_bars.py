"""일봉 마지막 막대가 '장중 진행 중'인지 판정하고, 진행 중 막대의 거래량을 안전하게 다룬다.

정규장 중에 받은 일봉의 마지막 막대는 오늘 막대가 아직 만들어지는 중이다. 이 막대의 거래량은 그 시점까지의
누적값이라 **최종 거래량 이하**다. 이를 20일 평균과 곧바로 비교하면 같은 종목이 조회 시각만으로 '거래량 위축'으로
보인다(52종목 3,431개 시점 재계산: 장 30% 경과 시 NCS 평균 -11.3점·FWS +12.7점, 장 60% 경과 시 -7.5점·+8.3점,
`AUTO_YES` 6.1% → 1.4~2.8%). 가격이 아니라 시각이 점수를 바꾸는 것이다.

해결 원칙은 모델을 새로 가정하지 않는 것이다. 장중 시간대별 누적 비율(U자형 곡선)을 추정해 거래량을 부풀리지 않고,
진행 중 막대가 있으면 **직전 확정 막대**의 거래량 비율을 쓴다. 마감 후·휴장·장 시작 전처럼 마지막 막대가 이미
확정된 경우에는 기존과 똑같이 마지막 막대를 쓴다.

표준 라이브러리만 사용하고 네트워크 호출이 없다. 휴장일은 '마지막 막대 날짜 == 오늘' 비교로 걸러진다(휴장일에는
오늘 막대가 없다).
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Iterable, Sequence

# 거래소별 (시간대, 정규장 시작 분, 정규장 종료 분). 분은 현지 자정 기준.
_SESSIONS = {
    "KRX": ("Asia/Seoul", 9 * 60, 15 * 60 + 30),
    "US": ("America/New_York", 9 * 60 + 30, 16 * 60),
}
# 공급자가 마지막 막대를 확정해 내려주기까지의 지연 허용 시간(분). 이 시간이 지나면 확정 막대로 본다.
SETTLE_MINUTES = 20


def _market_key(market: Any) -> str:
    return "US" if str(market or "").strip().upper() == "US" else "KRX"


def _local_now(market: Any, now: _dt.datetime | None = None) -> _dt.datetime | None:
    zone_name = _SESSIONS[_market_key(market)][0]
    try:
        from zoneinfo import ZoneInfo
        zone = ZoneInfo(zone_name)
    except Exception:  # 시간대 데이터가 없는 환경 — 판정 불가
        return None
    current = now or _dt.datetime.now(zone)
    if current.tzinfo is None:
        current = current.replace(tzinfo=zone)
    return current.astimezone(zone)


def regular_session_open(market: Any, now: _dt.datetime | None = None) -> bool:
    """현지 시계 기준 평일 정규장 시간(종료 후 SETTLE_MINUTES 포함)인지. 공휴일은 판별하지 않는다."""
    local = _local_now(market, now)
    if local is None or local.weekday() >= 5:
        return False
    _, start, end = _SESSIONS[_market_key(market)]
    minutes = local.hour * 60 + local.minute
    return start <= minutes < end + SETTLE_MINUTES


def last_bar_in_progress(market: Any, last_bar_date: Any, now: _dt.datetime | None = None) -> bool:
    """일봉 마지막 막대가 오늘 장중 진행 중인 막대인지.

    정규장 시간이고 마지막 막대 날짜가 거래소 현지 오늘이면 True. 휴장일·장 시작 전·마감 후·주말이면 False.
    """
    local = _local_now(market, now)
    if local is None or not regular_session_open(market, local):
        return False
    return str(last_bar_date or "")[:10] == local.date().isoformat()


def market_session_in_progress(market: Any, now: _dt.datetime | None = None, is_trading_day: Any = None) -> bool:
    """종목별 마지막 막대 날짜를 모를 때 쓰는 시장 단위 판정(스캔 등).

    정규장 시간이고 오늘이 거래일이면 True. ``is_trading_day(date) -> bool`` 이 주어지면 공휴일을 걸러낸다.
    """
    local = _local_now(market, now)
    if local is None or not regular_session_open(market, local):
        return False
    if callable(is_trading_day):
        try:
            return bool(is_trading_day(local.date()))
        except Exception:
            return True
    return True


def _finite_positive_or_zero(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def completed_volume_ratio(volumes: Sequence[Any] | Iterable[Any], in_progress: bool = False,
                           lookback: int = 20) -> tuple[float | None, str]:
    """(거래량 비율, 기준) — 기준은 'live_bar' 또는 'completed_bar'.

    확정 막대면 ``volumes[-1] / mean(volumes[-lookback-1:-1])`` (기존 계산식 그대로).
    진행 중 막대면 한 칸 앞으로 옮긴 ``volumes[-2] / mean(volumes[-lookback-2:-2])`` 로 직전 확정 막대와 비교한다.
    표본이 모자라거나 평균이 0이면 (None, 기준)을 돌려준다.
    """
    series = [_finite_positive_or_zero(v) for v in (list(volumes) if volumes is not None else [])]
    shift = 1 if in_progress else 0
    basis = "completed_bar" if in_progress else "live_bar"
    end = len(series) - shift
    if end < lookback + 1:
        return None, basis
    window = [v for v in series[end - 1 - lookback:end - 1] if v is not None]
    current = series[end - 1]
    if current is None or len(window) < max(3, lookback // 2):
        return None, basis
    average = sum(window) / len(window)
    if average <= 0:
        return None, basis
    return current / average, basis


def reference_bar_index(in_progress: bool) -> int:
    """'직전 확정 막대'를 가리키는 음수 인덱스: 진행 중이면 -2, 이미 확정된 마지막 막대면 -1."""
    return -2 if in_progress else -1
