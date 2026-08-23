"""국내·미국 주식용 인과적 동적 RSI 연구 모듈.

현재 봉의 RSI를 밴드 학습 표본에서 제외하고, 과거 RSI 분포를 1차원
3-군집으로 나눠 동적 과매도·과매수 중심을 계산한다. 매수 후보는
상승 다이버전스와 RSI 50선 회복을 모두 요구한다. 이 모듈의 출력은
검증 전까지 기존 종합점수에 가중하지 않는 연구용 보조 근거다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DYNAMIC_RSI_RULE_VERSION = "dynamic-rsi-kmeans-divergence-v1"


@dataclass(frozen=True)
class DynamicRSIConfig:
    market: str
    lookback: int
    min_history: int
    band_smoothing: int
    lower_clip: tuple[float, float]
    upper_clip: tuple[float, float]
    pivot_span: int
    divergence_max_age: int
    setup_max_age: int
    min_rsi_divergence: float
    min_price_divergence_pct: float
    atr_stop_multiple: float
    max_hold_bars: int
    round_trip_cost_pct: float


_CONFIGS = {
    "KRX": DynamicRSIConfig(
        market="KRX", lookback=100, min_history=60, band_smoothing=5,
        lower_clip=(18.0, 45.0), upper_clip=(55.0, 82.0), pivot_span=2,
        divergence_max_age=24, setup_max_age=35, min_rsi_divergence=2.5,
        min_price_divergence_pct=0.40, atr_stop_multiple=2.2,
        max_hold_bars=40, round_trip_cost_pct=0.20,
    ),
    "US": DynamicRSIConfig(
        market="US", lookback=126, min_history=63, band_smoothing=5,
        lower_clip=(20.0, 44.0), upper_clip=(56.0, 80.0), pivot_span=2,
        divergence_max_age=25, setup_max_age=38, min_rsi_divergence=2.0,
        min_price_divergence_pct=0.25, atr_stop_multiple=1.8,
        max_hold_bars=42, round_trip_cost_pct=0.10,
    ),
}


def config_for_market(market: str) -> DynamicRSIConfig:
    return _CONFIGS["KRX" if str(market).upper() == "KRX" else "US"]


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.where(loss.ne(0), 100.0)


def _one_dimensional_centroids(values: Iterable[float]) -> Optional[np.ndarray]:
    sample = np.asarray(list(values), dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 12:
        return None
    centroids = np.quantile(sample, [0.15, 0.50, 0.85]).astype(float)
    if np.unique(np.round(centroids, 8)).size < 3:
        centroids = np.array([sample.min(), sample.mean(), sample.max()], dtype=float)
    for _ in range(20):
        labels = np.abs(sample[:, None] - centroids[None, :]).argmin(axis=1)
        updated = np.array([
            sample[labels == idx].mean() if np.any(labels == idx) else centroids[idx]
            for idx in range(3)
        ])
        updated.sort()
        if np.allclose(updated, centroids, atol=1e-7, rtol=0):
            centroids = updated
            break
        centroids = updated
    return np.sort(centroids)


def _causal_dynamic_bounds(rsi: pd.Series, config: DynamicRSIConfig) -> tuple[pd.Series, pd.Series]:
    values = rsi.to_numpy(dtype=float)
    lower = np.full(len(values), np.nan)
    upper = np.full(len(values), np.nan)
    # 현재 봉을 제외한 과거 표본만 사용해 밴드 이동에 의한 선행편향을 막는다.
    for index in range(config.min_history, len(values)):
        start = max(0, index - config.lookback)
        centroids = _one_dimensional_centroids(values[start:index])
        if centroids is not None:
            lower[index], upper[index] = centroids[0], centroids[-1]
    lower_s = pd.Series(lower, index=rsi.index).ewm(
        span=config.band_smoothing, adjust=False, min_periods=1
    ).mean().clip(*config.lower_clip)
    upper_s = pd.Series(upper, index=rsi.index).ewm(
        span=config.band_smoothing, adjust=False, min_periods=1
    ).mean().clip(*config.upper_clip)
    return lower_s, upper_s


def _confirmed_divergences(
    highs: np.ndarray,
    lows: np.ndarray,
    rsi: np.ndarray,
    config: DynamicRSIConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = min(len(highs), len(lows), len(rsi))
    bullish = np.zeros(size, dtype=bool)
    bearish = np.zeros(size, dtype=bool)
    confirmed_low = np.full(size, np.nan)
    confirmed_high = np.full(size, np.nan)
    low_pivots: List[int] = []
    high_pivots: List[int] = []
    span = config.pivot_span

    for confirmation_index in range(span * 2, size):
        pivot = confirmation_index - span
        left = slice(pivot - span, pivot)
        right = slice(pivot + 1, pivot + span + 1)
        if not np.isfinite(rsi[pivot]):
            continue

        if lows[pivot] < np.nanmin(lows[left]) and lows[pivot] < np.nanmin(lows[right]):
            if low_pivots:
                previous = low_pivots[-1]
                price_drop = (lows[previous] - lows[pivot]) / max(abs(lows[previous]), 1e-12) * 100
                rsi_rise = rsi[pivot] - rsi[previous]
                bullish[confirmation_index] = (
                    price_drop >= config.min_price_divergence_pct
                    and rsi_rise >= config.min_rsi_divergence
                )
            low_pivots.append(pivot)
            confirmed_low[confirmation_index] = lows[pivot]

        if highs[pivot] > np.nanmax(highs[left]) and highs[pivot] > np.nanmax(highs[right]):
            if high_pivots:
                previous = high_pivots[-1]
                price_rise = (highs[pivot] - highs[previous]) / max(abs(highs[previous]), 1e-12) * 100
                rsi_drop = rsi[previous] - rsi[pivot]
                bearish[confirmation_index] = (
                    price_rise >= config.min_price_divergence_pct
                    and rsi_drop >= config.min_rsi_divergence
                )
            high_pivots.append(pivot)
            confirmed_high[confirmation_index] = highs[pivot]

    return bullish, bearish, confirmed_low, confirmed_high


def _age(value: Optional[int], maximum: int) -> Optional[int]:
    if value is None:
        return None
    value += 1
    return value if value <= maximum else None


def add_dynamic_rsi_features(
    frame: pd.DataFrame,
    market: str = "US",
    config: Optional[DynamicRSIConfig] = None,
) -> pd.DataFrame:
    """OHLC DataFrame에 동적 RSI 밴드·다이버전스·롱 상태를 추가한다."""
    df = frame.copy()
    config = config or config_for_market(market)
    required = {"Close", "High", "Low"}
    if not required.issubset(df.columns):
        return df

    if "RSI" not in df.columns:
        df["RSI"] = _wilder_rsi(df["Close"])
    rsi = pd.to_numeric(df["RSI"], errors="coerce")
    lower, upper = _causal_dynamic_bounds(rsi, config)
    df["DRSI_Lower"] = lower
    df["DRSI_Upper"] = upper
    df["DRSI_Center"] = 50.0

    highs = pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
    rsi_values = rsi.to_numpy(dtype=float)
    bull_div, bear_div, pivot_lows, pivot_highs = _confirmed_divergences(
        highs, lows, rsi_values, config
    )
    df["DRSI_Bull_Div"] = bull_div.astype(int)
    df["DRSI_Bear_Div"] = bear_div.astype(int)

    atr = pd.to_numeric(df.get("ATR", pd.Series(index=df.index, dtype=float)), errors="coerce")
    if atr.isna().all():
        previous = pd.Series(closes, index=df.index).shift()
        true_range = pd.concat([
            pd.Series(highs - lows, index=df.index),
            (pd.Series(highs, index=df.index) - previous).abs(),
            (pd.Series(lows, index=df.index) - previous).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()
    atr_values = atr.to_numpy(dtype=float)

    signal = np.zeros(len(df), dtype=int)       # 1=롱 후보, -1=청산/하락 위험
    setup = np.zeros(len(df), dtype=int)        # 1=상승 준비, -1=하락 준비
    position = np.zeros(len(df), dtype=int)     # 연구용 종가 상태
    entry_price = np.full(len(df), np.nan)
    stop_price = np.full(len(df), np.nan)

    lower_age: Optional[int] = None
    upper_age: Optional[int] = None
    bull_age: Optional[int] = None
    bear_age: Optional[int] = None
    latest_swing_low = np.nan
    latest_swing_high = np.nan
    active_entry = np.nan
    active_stop = np.nan
    holding = 0
    holding_age = 0

    for index in range(len(df)):
        lower_age = _age(lower_age, config.setup_max_age)
        upper_age = _age(upper_age, config.setup_max_age)
        bull_age = _age(bull_age, config.divergence_max_age)
        bear_age = _age(bear_age, config.divergence_max_age)
        if np.isfinite(pivot_lows[index]):
            latest_swing_low = pivot_lows[index]
        if np.isfinite(pivot_highs[index]):
            latest_swing_high = pivot_highs[index]
        if bull_div[index]:
            bull_age = 0
        if bear_div[index]:
            bear_age = 0

        current_rsi = rsi_values[index]
        current_lower = lower.iloc[index]
        current_upper = upper.iloc[index]
        if not (np.isfinite(current_rsi) and np.isfinite(current_lower) and np.isfinite(current_upper)):
            position[index] = holding
            continue
        if current_rsi <= current_lower:
            lower_age = 0
        if current_rsi >= current_upper:
            upper_age = 0

        bullish_setup = lower_age is not None and bull_age is not None
        bearish_setup = upper_age is not None and bear_age is not None
        setup[index] = 1 if bullish_setup else (-1 if bearish_setup else 0)
        previous_rsi = rsi_values[index - 1] if index else np.nan
        crossed_up_50 = np.isfinite(previous_rsi) and previous_rsi <= 50 < current_rsi
        crossed_down_50 = np.isfinite(previous_rsi) and previous_rsi >= 50 > current_rsi

        if holding == 0 and bullish_setup and crossed_up_50:
            signal[index] = 1
            holding = 1
            active_entry = closes[index]
            atr_now = atr_values[index]
            atr_stop = closes[index] - config.atr_stop_multiple * atr_now if np.isfinite(atr_now) else np.nan
            swing_stop = latest_swing_low if np.isfinite(latest_swing_low) and latest_swing_low < closes[index] else np.nan
            candidates = [value for value in (atr_stop, swing_stop) if np.isfinite(value) and value > 0]
            active_stop = max(candidates) if candidates else np.nan
            holding_age = 0
            lower_age = bull_age = None
        elif holding == 0 and bearish_setup and crossed_down_50:
            # 국내·미국 현물 매수 사용자를 위한 신규 숏이 아니라 위험 경고다.
            signal[index] = -1
            upper_age = bear_age = None
        elif holding == 1:
            holding_age += 1
            previous_upper = upper.iloc[index - 1] if index else np.nan
            upper_reentry = (
                np.isfinite(previous_upper)
                and previous_rsi >= previous_upper
                and current_rsi < current_upper
            )
            stop_breached = np.isfinite(active_stop) and lows[index] <= active_stop
            timed_exit = holding_age >= config.max_hold_bars
            if stop_breached or upper_reentry or timed_exit or (bearish_setup and crossed_down_50):
                signal[index] = -1
                holding = 0
                active_entry = np.nan
                active_stop = np.nan
                holding_age = 0

        position[index] = holding
        entry_price[index] = active_entry
        stop_price[index] = active_stop

    df["DRSI_Setup"] = setup
    df["DRSI_Signal"] = signal
    df["DRSI_Position"] = position
    df["DRSI_Entry"] = entry_price
    df["DRSI_Stop"] = stop_price
    return df


def _last_number(dd: Dict[str, Any], key: str) -> Optional[float]:
    values = dd.get(key) or []
    if not values:
        return None
    try:
        value = float(values[-1])
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _last_event_age(dd: Dict[str, Any], key: str, predicate) -> Optional[int]:
    """마지막 조건 충족 봉이 현재 봉에서 몇 봉 전인지 반환한다."""
    values = dd.get(key) or []
    for age, value in enumerate(reversed(values)):
        try:
            if predicate(float(value)):
                return age
        except (TypeError, ValueError):
            continue
    return None


def _event_date(dd: Dict[str, Any], age: Optional[int]) -> Optional[str]:
    dates = dd.get("Date") or []
    if age is None or age < 0 or age >= len(dates):
        return None
    value = dates[-(age + 1)]
    return str(value) if value is not None else None


def _purchase_timing(
    dd: Dict[str, Any],
    config: DynamicRSIConfig,
    *,
    rsi: float,
    lower: float,
    upper: float,
    signal: int,
    setup: int,
    position: int,
) -> Dict[str, Any]:
    """동적 RSI 신호를 매수 실행 가능 시점과 대기 조건으로 변환한다.

    날짜를 임의 예측하지 않고, 종가로 확정되는 세 조건과 다음 거래일 시가
    체결 모델을 그대로 노출한다. ``valid_for_bars``는 적중 예상 기간이 아니라
    현재 설정이 만료되기 전까지의 최대 관찰 봉 수다.
    """
    lower_values = dd.get("DRSI_Lower") or []
    rsi_values = dd.get("RSI") or []
    pair_count = min(len(lower_values), len(rsi_values))
    lower_touch_age: Optional[int] = None
    for age in range(pair_count):
        try:
            rsi_value = float(rsi_values[-(age + 1)])
            lower_value = float(lower_values[-(age + 1)])
            if np.isfinite(rsi_value) and np.isfinite(lower_value) and rsi_value <= lower_value:
                lower_touch_age = age
                break
        except (TypeError, ValueError):
            continue

    bull_div_age = _last_event_age(dd, "DRSI_Bull_Div", lambda value: value == 1)
    bear_div_age = _last_event_age(dd, "DRSI_Bear_Div", lambda value: value == 1)
    entry_signal_age = _last_event_age(dd, "DRSI_Signal", lambda value: value == 1)
    exit_signal_age = _last_event_age(dd, "DRSI_Signal", lambda value: value == -1)
    lower_recent = lower_touch_age is not None and lower_touch_age <= config.setup_max_age
    bull_recent = bull_div_age is not None and bull_div_age <= config.divergence_max_age
    center_recovered = signal == 1 or position == 1
    conditions = [
        {
            "key": "dynamic_oversold",
            "label": "동적 과매도 구간 접촉",
            "met": lower_recent,
            "detail": (
                f"{lower_touch_age}봉 전 · RSI가 동적 하단 이하"
                + (" · 유효 기간 초과" if not lower_recent else "")
                if lower_touch_age is not None else "최근 동적 하단 접촉 없음"
            ),
            "date": _event_date(dd, lower_touch_age),
        },
        {
            "key": "bullish_divergence",
            "label": "확정 상승 다이버전스",
            "met": bull_recent,
            "detail": (
                f"{bull_div_age}봉 전 확정 · 피벗 확인 지연 포함"
                + (" · 유효 기간 초과" if not bull_recent else "")
                if bull_div_age is not None else "가격 저점·RSI 저점의 확정 다이버전스 없음"
            ),
            "date": _event_date(dd, bull_div_age),
        },
        {
            "key": "center_recovery",
            "label": "RSI 50 상향 회복",
            "met": center_recovered,
            "detail": (
                f"매수 신호 {entry_signal_age}봉 전 확정"
                if entry_signal_age is not None else f"현재 RSI {rsi:.1f} · 50 상향 돌파 대기"
            ),
            "date": _event_date(dd, entry_signal_age),
        },
    ]
    met_count = sum(1 for item in conditions if item["met"])
    remaining_candidates = []
    if lower_recent and lower_touch_age is not None:
        remaining_candidates.append(max(0, config.setup_max_age - lower_touch_age))
    if bull_recent and bull_div_age is not None:
        remaining_candidates.append(max(0, config.divergence_max_age - bull_div_age))
    valid_for_bars = min(remaining_candidates) if remaining_candidates else None

    close = _last_number(dd, "Close")
    signal_reference = _last_number(dd, "DRSI_Entry") if (signal == 1 or position == 1) else None
    reference_close = signal_reference if signal_reference is not None else close
    atr = _last_number(dd, "ATR")
    stop = _last_number(dd, "DRSI_Stop")
    max_chase_price = (
        reference_close + atr * 0.5
        if reference_close is not None and atr is not None and atr > 0 and (signal == 1 or position == 1)
        else None
    )
    market_note = (
        "KRX 정규장 시가와 VI·갭 상승 여부를 확인합니다."
        if config.market == "KRX"
        else "미국 정규장 시가 기준이며 프리마켓·애프터마켓 체결은 제외합니다."
    )

    if signal == 1:
        state, tone, label = "confirmed", "positive", "매수 확인 봉 발생"
        window = "다음 거래일 정규장 시가 이후"
        action = "시가가 과도하게 뜨지 않는지 확인한 뒤 소액 1차 분할 매수를 검토합니다."
        eligible = True
    elif position == 1:
        state, tone, label = "active", "positive", "매수 신호 이후 관리 구간"
        window = "신규 추격보다 기존 포지션 관리"
        action = "동적 RSI 신규 진입 시점은 지났습니다. 손절선과 동적 상단 재진입 실패를 우선 점검합니다."
        eligible = False
    elif signal == -1 or setup == -1 or (exit_signal_age == 0) or (
        bear_div_age is not None and bear_div_age <= config.divergence_max_age and rsi < 50
    ):
        state, tone, label = "risk", "negative", "매수 보류"
        window = "하락 위험 신호 해소 후 재평가"
        action = "신규 매수를 보류하고 RSI 50 회복과 상승 다이버전스의 새 조합을 기다립니다."
        eligible = False
    elif setup == 1 or (lower_recent and bull_recent):
        state, tone, label = "armed", "warning", "매수 준비 2/3"
        window = "RSI 50 상향 돌파 봉 확정 후 다음 거래일"
        action = "현재 봉 종가 기준 RSI 50 상향 돌파가 확정될 때까지 주문하지 않습니다."
        eligible = False
    elif rsi <= lower or lower_recent:
        state, tone, label = "watch", "neutral", "과매도 관찰 단계"
        window = "상승 다이버전스 확인 대기"
        action = "과매도만으로 매수하지 않고 더 낮은 가격 저점과 더 높은 RSI 저점이 확정되는지 관찰합니다."
        eligible = False
    else:
        state, tone, label = "neutral", "neutral", "조건 미충족 · 관망"
        window = "동적 과매도→다이버전스→50 회복 순서 대기"
        action = "세 조건이 순차적으로 충족되기 전에는 동적 RSI 근거의 매수 타이밍으로 해석하지 않습니다."
        eligible = False

    return {
        "state": state,
        "tone": tone,
        "label": label,
        "window": window,
        "action": action,
        "eligible_now": eligible,
        "conditions_met": met_count,
        "conditions_total": len(conditions),
        "conditions": conditions,
        "valid_for_bars": valid_for_bars,
        "signal_age_bars": entry_signal_age,
        "signal_date": _event_date(dd, entry_signal_age),
        "reference_close": reference_close,
        "max_chase_price": max_chase_price,
        "stop": stop,
        "execution_rule": "신호 봉 종가 확정 → 다음 거래일 정규장 시가 확인 → 최대 0.5 ATR 갭 한도 내 분할 검토",
        "market_note": market_note,
        "invalidation": "손절선 이탈 또는 하락 다이버전스와 RSI 50 하향 이탈 시 매수 시나리오 무효",
        "is_probability": False,
    }


def dynamic_rsi_snapshot(dd: Dict[str, Any], market: str = "US") -> Dict[str, Any]:
    config = config_for_market(market)
    rsi = _last_number(dd, "RSI")
    lower = _last_number(dd, "DRSI_Lower")
    upper = _last_number(dd, "DRSI_Upper")
    signal = int(_last_number(dd, "DRSI_Signal") or 0)
    setup = int(_last_number(dd, "DRSI_Setup") or 0)
    position = int(_last_number(dd, "DRSI_Position") or 0)
    if rsi is None or lower is None or upper is None:
        return {
            "available": False,
            "experimental": True,
            "rule_version": DYNAMIC_RSI_RULE_VERSION,
            "reason": f"동적 밴드 계산에 최소 {config.min_history + 1}봉이 필요합니다.",
            "purchase_timing": {
                "state": "unavailable", "tone": "neutral", "label": "계산 데이터 부족",
                "window": "과거 데이터 확보 후 재계산", "eligible_now": False,
                "conditions_met": 0, "conditions_total": 3, "conditions": [],
                "is_probability": False,
            },
        }
    zone = "과매수" if rsi >= upper else "과매도" if rsi <= lower else "정상 범위"
    label = "롱 진입 후보" if signal == 1 else "청산·하락 위험" if signal == -1 else "관망"
    setup_label = "상승 다이버전스 확인" if setup == 1 else "하락 다이버전스 확인" if setup == -1 else "확정 설정 없음"
    purchase_timing = _purchase_timing(
        dd, config, rsi=rsi, lower=lower, upper=upper,
        signal=signal, setup=setup, position=position,
    )
    return {
        "available": True,
        "experimental": True,
        "rule_version": DYNAMIC_RSI_RULE_VERSION,
        "method": "현재 봉 제외 과거 RSI 1D K-means 3군집 + 확정 피벗 다이버전스 + 50선",
        "market": config.market,
        "rsi": round(rsi, 2),
        "lower": round(lower, 2),
        "upper": round(upper, 2),
        "center": 50.0,
        "zone": zone,
        "setup": setup,
        "setup_label": setup_label,
        "signal": signal,
        "signal_label": label,
        "position": position,
        "entry": _last_number(dd, "DRSI_Entry"),
        "stop": _last_number(dd, "DRSI_Stop"),
        "purchase_timing": purchase_timing,
        "config": asdict(config),
        "disclaimer": "실험용 보조 신호이며 상승확률·수익 보장 수치가 아닙니다.",
    }


def dynamic_rsi_daily_snapshot(dd: Dict[str, Any], market: str = "US") -> Dict[str, Any]:
    """일봉 OHLC 사전에서 동적 RSI를 계산하고 일봉 메타데이터를 명시한다."""
    required = ("Open", "High", "Low", "Close")
    lengths = [len(dd.get(key) or []) for key in required]
    if not lengths or min(lengths) == 0:
        snapshot = dynamic_rsi_snapshot({}, market)
        snapshot.update({"timeframe": "1d", "timeframe_label": "일봉", "source_bars": 0})
        return snapshot

    size = min(lengths)
    columns: Dict[str, Any] = {
        key: list(dd.get(key) or [])[-size:] for key in required
    }
    volume = list(dd.get("Volume") or [])
    if len(volume) >= size:
        columns["Volume"] = volume[-size:]
    for key in ("RSI", "ATR"):
        values = list(dd.get(key) or [])
        if len(values) >= size:
            columns[key] = values[-size:]

    frame = add_dynamic_rsi_features(pd.DataFrame(columns), market=market)
    payload = frame.where(pd.notna(frame), None).to_dict(orient="list")
    dates = list(dd.get("Date") or [])
    if len(dates) >= size:
        payload["Date"] = dates[-size:]
    snapshot = dynamic_rsi_snapshot(payload, market)
    snapshot.update({
        "timeframe": "1d",
        "timeframe_label": "일봉",
        "source_bars": size,
        "as_of": str(dates[-1]) if dates else None,
    })
    return snapshot


def dynamic_rsi_signal_card(dd: Dict[str, Any], market: str = "US") -> Optional[Dict[str, Any]]:
    snapshot = dynamic_rsi_snapshot(dd, market)
    if not snapshot.get("available"):
        return None
    signal_code = snapshot["signal"]
    signal_label = "매수" if signal_code == 1 else "매도" if signal_code == -1 else "관망"
    state = snapshot["signal_label"] if signal_code else snapshot["zone"]
    desc = (
        f"RSI {snapshot['rsi']:.1f} · 동적 하단 {snapshot['lower']:.1f} / "
        f"상단 {snapshot['upper']:.1f} · {snapshot['setup_label']} · "
        "워크포워드 검증 전 종합점수 미반영"
    )
    return {
        "name": f"동적 RSI ({snapshot['market']}·실험)",
        "state": state,
        "signal": signal_label,
        "desc": desc,
        "value": f"{snapshot['rsi']:.1f} [{snapshot['lower']:.1f}~{snapshot['upper']:.1f}]",
        "context_only": True,
        "experimental": True,
    }


def backtest_dynamic_rsi(
    frame: pd.DataFrame,
    market: str = "US",
    config: Optional[DynamicRSIConfig] = None,
) -> Dict[str, Any]:
    """다음 봉 시가 체결·시장별 왕복비용을 적용한 롱 전용 검증 함수."""
    config = config or config_for_market(market)
    df = add_dynamic_rsi_features(frame, market, config)
    required = {"Open", "High", "Low", "Close", "DRSI_Signal", "DRSI_Stop"}
    if not required.issubset(df.columns) or len(df) < config.min_history + 5:
        return {"trades": [], "summary": {"trade_count": 0, "reason": "insufficient_data"}}

    trades: List[Dict[str, Any]] = []
    active: Optional[Dict[str, Any]] = None
    for index in range(len(df)):
        row = df.iloc[index]
        if active is not None and index >= active["entry_index"]:
            open_price = float(row["Open"])
            low_price = float(row["Low"])
            close_price = float(row["Close"])
            stop = active["stop_price"]
            exit_price = None
            exit_reason = None
            if np.isfinite(stop) and low_price <= stop:
                exit_price = min(open_price, stop) if open_price < stop else stop
                exit_reason = "stop"
            elif int(row["DRSI_Signal"]) == -1 and index + 1 < len(df):
                exit_price = float(df.iloc[index + 1]["Open"])
                exit_reason = "dynamic_exit_next_open"
            elif index - active["entry_index"] + 1 >= config.max_hold_bars:
                exit_price = close_price
                exit_reason = "max_hold"
            if exit_price is not None:
                gross = (exit_price / active["entry_price"] - 1) * 100
                net = gross - config.round_trip_cost_pct
                trades.append({
                    **active,
                    "exit_index": index if exit_reason != "dynamic_exit_next_open" else index + 1,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "gross_return_pct": gross,
                    "return_pct": net,
                    "win": net > 0,
                })
                active = None
                continue

        if active is None and int(row["DRSI_Signal"]) == 1 and index + 1 < len(df):
            entry_index = index + 1
            entry_price = float(df.iloc[entry_index]["Open"])
            stop = float(row["DRSI_Stop"]) if pd.notna(row["DRSI_Stop"]) else np.nan
            if np.isfinite(stop) and stop >= entry_price:
                stop = entry_price * (1 - config.round_trip_cost_pct / 100)
            active = {
                "signal_index": index,
                "entry_index": entry_index,
                "entry_price": entry_price,
                "stop_price": stop,
            }

    if active is not None:
        exit_price = float(df.iloc[-1]["Close"])
        gross = (exit_price / active["entry_price"] - 1) * 100
        net = gross - config.round_trip_cost_pct
        trades.append({
            **active, "exit_index": len(df) - 1, "exit_price": exit_price,
            "exit_reason": "end_of_data", "gross_return_pct": gross,
            "return_pct": net, "win": net > 0,
        })

    returns = [float(trade["return_pct"]) for trade in trades]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value <= 0]
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= 1 + value / 100
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, (equity / peak - 1) * 100)
    profit_factor = (
        sum(wins) / abs(sum(losses)) if losses and abs(sum(losses)) > 1e-12
        else (float("inf") if wins else 0.0)
    )
    summary = {
        "market": config.market,
        "rule_version": DYNAMIC_RSI_RULE_VERSION,
        "trade_count": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "average_return_pct": round(float(np.mean(returns)), 4) if returns else 0.0,
        "total_compounded_return_pct": round((equity - 1) * 100, 4),
        "profit_factor": None if np.isinf(profit_factor) else round(profit_factor, 4),
        "max_drawdown_pct": round(max_drawdown, 4),
        "round_trip_cost_pct": config.round_trip_cost_pct,
        "execution": "signal close -> next bar open; intrabar stop; long only",
    }
    return {"trades": trades, "summary": summary, "frame": df}
