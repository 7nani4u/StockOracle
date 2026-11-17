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


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def settle_in_progress_volume(dd: dict, lookback: int = 20) -> tuple[dict, dict]:
    """진행 중인 마지막 일봉의 누적 거래량을 분석용 사본에서 '직전 확정 봉의 거래량'으로 바꾼다.

    위의 거래량 비율 함수는 진행 중이면 한 칸 앞 막대를 보지만, 예측 탭의 나머지 소비자(리스크·매수 구간·
    눌림목·목표가·예측 요약·OBV·매수 압력·거래대금)는 ``Volume[-1]`` 을 그대로 읽는다. 장중 누적값은 최종값
    이하라 같은 가격에서도 조회 시각만으로 돌파 확률·하락 위험 점수·단계 가격이 바뀐다(2026-11-17 재현:
    52종목, 누적 30% 대 100% 비교에서 시각 필드를 뺀 338개 출력 경로가 달라짐 — 돌파 확률 최대 29.8%p, 1차 단계 가격
    최대 9.7%, 하락 위험 점수 ±8점, 횡보 시나리오 +3%p).

    오늘 막대의 최종 거래량은 마감 전에는 알 수 없다. 넣을 값은 측정으로 골랐다(52종목 × 7개 날짜 = 364표본,
    장중 출력과 같은 날 마감 후 출력의 평균 차이, docs/intraday_volume_tick_audit_20261117.md):
    · 직전 확정 봉 거래량(이 함수)은 예전 동작(누적값 그대로)보다 14개 지표 모두에서, 장 30%·60% 경과 시점 모두
      마감 후 값에 가까웠다(30% 시점 돌파 확률 11.7→6.4%p, 하락 위험 점수 2.47→1.78, 2차 구간 하단 0.43→0.24%).
    · 직전 20봉 평균(비율 1.0)보다는 14개 중 12개에서 가까웠다(1차 단계 가격 0.290→0.241%, 균형 TP1 확률
      0.36→0.29%p). 거래량은 날마다 이어지는 경향이 있어 어제 값이 오늘 최종값의 더 나은 대리값이다. 어제 거래량과
      오늘 캔들 방향이 짝지어지는 우려는 측정상 손해가 없었다(하락 위험 점수 1.93→1.78). 평균보다 나빴던 것은
      돌파 확률(6.22→6.44%p)과 2차 단계 도달 확률(0.47→0.59%p)이다.
    · 둘 다 조회 시각과 무관하다. 지금까지 누적값을 하한으로 쓰는 변형(max)은 일부 지표에서 조금 더 가까웠지만
      시각에 따라 값이 다시 움직여(돌파 확률 30%↔60% 차이 0.76~2.56%p) 채택하지 않았다. 다른 네 소비자가 이미
      쓰는 '직전 확정 봉 기준' 원칙과 같고, 시간대별 누적 곡선 같은 추정 모형은 넣지 않는다.

    직전 봉 거래량이 없거나 0이면(거래정지일 등) 직전 ``lookback`` 개 확정 봉 평균을 쓴다. 가격·지표는 바꾸지
    않고, 거래량에서 파생된 OBV·BUY_PRESSURE 의 마지막 값만 ``add_indicators`` 와 같은 식으로 다시 계산한다.
    원본 dict·리스트는 수정하지 않는다(캐시 공유 객체).

    반환: (사본 dd, 메타). 적용 조건이 안 되면 (원본 dd, {"applied": False, ...}).
    """
    meta: dict = {"applied": False, "basis": "previous_completed_bar", "lookback": lookback}
    if not isinstance(dd, dict):
        meta["reason"] = "no_data"
        return dd, meta
    volumes = list(dd.get("Volume") or [])
    closes = list(dd.get("Close") or [])
    if len(volumes) < lookback + 2 or len(volumes) != len(closes):
        meta["reason"] = "insufficient_bars"
        return dd, meta
    # completed_volume_ratio 와 같은 창(직전 확정 봉 lookback 개, 0 포함). 누적 비율 표시와 대체값에만 쓴다.
    window = [v for v in (_finite_positive_or_zero(x) for x in volumes[-lookback - 1:-1]) if v is not None]
    mean_completed = sum(window) / len(window) if window else 0.0
    if len(window) < max(3, lookback // 2) or mean_completed <= 0:
        meta["reason"] = "insufficient_completed_volume"
        return dd, meta
    live = _finite_positive_or_zero(volumes[-1])
    previous = _finite_positive_or_zero(volumes[-2])
    if previous is not None and previous > 0:
        settled = volumes[-2]                       # 관측된 값 그대로(정수 거래량은 정수로)
    else:
        meta["basis"] = "completed_bar_mean"
        settled = float(round(mean_completed)) if mean_completed >= 1.0 else mean_completed

    out = dict(dd)
    new_volumes = volumes[:-1] + [settled]
    out["Volume"] = new_volumes

    c_last, c_prev = _finite(closes[-1]), _finite(closes[-2])
    obv = list(dd.get("OBV") or [])
    if len(obv) == len(volumes) and c_last is not None and c_prev is not None:
        obv_prev = _finite(obv[-2])
        if obv_prev is not None:
            direction = (c_last > c_prev) - (c_last < c_prev)
            obv[-1] = obv_prev + float(settled) * direction
            out["OBV"] = obv
    pressure = list(dd.get("BUY_PRESSURE") or [])
    if len(pressure) == len(volumes) and len(volumes) >= 15:
        up_sum = 0.0
        total = 0.0
        valid = True
        for i in range(len(volumes) - 14, len(volumes)):
            vol = _finite(new_volumes[i])
            cur, prev = _finite(closes[i]), _finite(closes[i - 1])
            if vol is None or vol <= 0:
                valid = False  # add_indicators 는 0 거래량을 NaN 으로 바꿔 창 전체를 결측 처리한다
                break
            total += vol
            if cur is not None and prev is not None and cur >= prev:
                up_sum += vol
        pressure[-1] = (up_sum / total * 100.0) if valid and total > 0 else None
        out["BUY_PRESSURE"] = pressure

    completed_ratio, _ = completed_volume_ratio(volumes, True, lookback)
    meta.update({
        "applied": True,
        "live_volume": live,
        "analysis_volume": settled,
        "live_ratio": (live / mean_completed) if (live is not None and mean_completed > 0) else None,
        "completed_ratio": completed_ratio,
    })
    return out, meta
