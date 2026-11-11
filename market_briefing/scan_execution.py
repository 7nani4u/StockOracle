"""Scan execution diagnostics. Daily price touches never imply an order fill.

Pure, deterministic calculations shared by the scanner, position review and audit CLI.
Historical reference windows do not overlap and use only completed input bars. Their
frequencies are descriptive, not calibrated probabilities for the current signal.
"""
from __future__ import annotations

import math
from statistics import mean, median
from typing import Any

from .forecast_model import blended_daily_sigma, touch_probability_range, touch_time_window
from .session_bars import completed_volume_ratio

EXECUTION_VERSION = "2026-11-11.1"


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def bounded_number(value, name, low, high, *, integer=False):
    number = finite(value)
    if number is None or not low <= number <= high or (integer and number != int(number)):
        raise ValueError(f"{name}: {low}~{high} 범위의 {'정수' if integer else '유한한 숫자'}가 필요합니다")
    return int(number) if integer else number


def scan_options(params):
    """Validate request values before network calls; equity is in quote currency."""
    market = str(params.get("market", "KRX")).upper()
    if market not in ("KRX", "US"):
        raise ValueError("market: KRX 또는 US가 필요합니다")
    currency = "USD" if market == "US" else "KRW"
    if str(params.get("equity_currency", currency)).upper() != currency:
        raise ValueError(f"equity는 {currency} 기준이어야 합니다. 환전 후 자본을 입력하세요")
    return {
        "market": market, "currency": currency,
        "equity": bounded_number(params.get("equity", 10_000 if market == "US" else 100_000_000),
                                 "equity", 1, 1e12),
        "risk_pct": bounded_number(params.get("risk_pct", 1), "risk_pct", 0.01, 5),
        "horizon_days": bounded_number(params.get("horizon_days", 3), "horizon_days", 1, 60, integer=True),
        "participation_rate": bounded_number(params.get("participation_rate", 0.01),
                                             "participation_rate", 0.0001, 0.1),
        "exit_days": bounded_number(params.get("exit_days", 1), "exit_days", 1, 10, integer=True),
        "round_trip_bps": bounded_number(params.get("round_trip_bps", 20), "round_trip_bps", 0, 1000),
    }


def validate_ohlcv(opens, highs, lows, closes, volumes):
    """Reject misaligned, nonfinite or impossible bars without deleting sessions."""
    if any(x is None for x in (highs, lows, closes, volumes)):
        return ["OHLCV 누락"]
    n = len(closes)
    columns = (highs, lows, volumes) + ((opens,) if opens is not None else ())
    if n == 0 or any(len(col) != n for col in columns):
        return ["OHLCV 길이 불일치 또는 빈 데이터"]
    for i in range(n):
        c, h, l, v = (finite(x[i]) for x in (closes, highs, lows, volumes))
        o = finite(opens[i]) if opens is not None else c
        if any(x is None for x in (c, h, l, v, o)):
            return [f"{i}번 봉: 비유한 값 또는 결측"]
        if min(c, h, l, o) <= 0 or v < 0 or h < max(c, o, l) or l > min(c, o):
            return [f"{i}번 봉: 가격 범위 또는 거래량 오류"]
    return []


def aligned_history_frame(hist, min_bars=60):
    """Validate a history DataFrame, trimming only empty union-index edges."""
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if hist is None or hist.empty or any(c not in hist.columns for c in cols):
        return None
    if not hist.index.is_monotonic_increasing or hist.index.has_duplicates:
        return None
    present = hist[cols].notna().any(axis=1).to_numpy().nonzero()[0]
    if len(present) == 0:
        return None
    frame = hist.iloc[present[0]:present[-1]+1]
    if len(frame) < min_bars or validate_ohlcv(*(frame[c].tolist() for c in cols)):
        return None
    return frame


def liquidity_profile(closes, volumes, *, in_progress=False, shares=None,
                      participation_rate=0.01, exit_days=1, lookback=20):
    """A planning capacity at a chosen participation rate, not executable depth.

    min(mean, median) prevents one volume spike from dominating the estimate.
    An unfinished daily volume is excluded, never scaled up to a fictitious close.
    """
    participation_rate = bounded_number(participation_rate, "participation_rate", 0.0001, 0.1)
    exit_days = bounded_number(exit_days, "exit_days", 1, 10, integer=True)
    size = finite(shares)
    if shares is not None and (size is None or size < 0):
        raise ValueError("shares: 음수가 아닌 유한한 수량이 필요합니다")
    end = len(volumes) - int(bool(in_progress))
    start = max(0, end - lookback)
    pairs = [(finite(c), finite(v)) for c, v in zip(closes[start:end], volumes[start:end])]
    valid = len(closes) == len(volumes) and len(pairs) >= lookback and all(
        c is not None and c > 0 and v is not None and v >= 0 for c, v in pairs)
    ratio, basis = completed_volume_ratio(volumes, in_progress)
    out = {"available": valid, "volume_basis": basis, "observations": len(pairs),
           "participation_rate": participation_rate, "exit_days": exit_days,
           "requested_shares": size, "relative_volume": ratio,
           "fill_guaranteed": False, "spread_bps": None,
           "limitations": "일봉 거래량 기준 계획치. 실시간 호가·스프레드·거래정지·부분 체결은 확인되지 않음"}
    if not valid:
        return dict(out, status="unknown", reason="확정 거래량 20봉 또는 유효 가격 부족",
                    capacity_shares=None, estimated_exit_sessions=None, within_exit_budget=None)
    vs = [v for _, v in pairs]
    avg, med = mean(vs), median(vs)
    reference = min(avg, med)
    per_session = reference * participation_rate
    capacity = math.floor(per_session * exit_days)
    sessions = (size / per_session if per_session > 0 else None) if size is not None else None
    within = (size <= capacity) if size is not None else None
    status = "blocked" if capacity < 1 else "limited" if within is False else "estimated"
    return dict(out, status=status, reason=("거래량 기반 청산 여력 없음" if capacity < 1 else
                "설정한 청산 기간·거래참여율 초과" if within is False else "거래량 기준 계획 범위"),
                average_volume=round(avg, 2), median_volume=round(med, 2),
                average_turnover=round(mean(c * v for c, v in pairs), 2),
                reference_volume=round(reference, 2), capacity_shares=capacity,
                estimated_exit_sessions=None if sessions is None else round(sessions, 6),
                exit_sessions_ceil=None if sessions is None else math.ceil(sessions),
                participation_pct=None if size is None or reference <= 0 else size / reference * 100,
                within_exit_budget=within)


def historical_barrier_reference(opens, highs, lows, closes, *, target_ratio, stop_ratio,
                                 horizon_days, round_trip_bps=20, min_samples=30):
    """Replay fixed relative barriers on disjoint *past* paths.

    Entry is each window's starting close. A gap through the stop exits at the
    worse opening price. Unknown high/low ordering in a double-touch bar is
    counted explicitly and conservatively treated as stop first. Samples are
    unconditioned on the current strategy, volume or regime; no predictive claim.
    """
    h = bounded_number(horizon_days, "horizon_days", 1, 60, integer=True)
    cost = bounded_number(round_trip_bps, "round_trip_bps", 0, 1000) / 20_000
    if not (target_ratio > 1 and 0 < stop_ratio < 1):
        raise ValueError("목표는 기준가 위, 손절은 기준가 아래여야 합니다")
    counts = {"target_first": 0, "stop_first": 0, "timeout": 0, "ambiguous": 0}
    days_to_target, returns, target_touches = [], [], 0
    # Anchor from the end, leaving an unused prefix instead of overlapping outcomes.
    starts = range(len(closes) - 1 - h, -1, -h)
    for start in starts:
        entry = closes[start]
        target, stop = entry * target_ratio, entry * stop_ratio
        touched = any(highs[j] >= target for j in range(start + 1, start + h + 1))
        target_touches += int(touched)
        exit_price, outcome = closes[start + h], "timeout"
        for day, j in enumerate(range(start + 1, start + h + 1), 1):
            o = opens[j] if opens is not None else None
            if o is not None and o <= stop:
                outcome, exit_price = "stop_first", o
            elif o is not None and o >= target:
                outcome, exit_price = "target_first", target
            elif lows[j] <= stop:
                outcome, exit_price = "stop_first", stop
                counts["ambiguous"] += int(highs[j] >= target)
            elif highs[j] >= target:
                outcome, exit_price = "target_first", target
            else:
                continue
            if outcome == "target_first":
                days_to_target.append(day)
            break
        counts[outcome] += 1
        returns.append((exit_price * (1 - cost) / (entry * (1 + cost)) - 1) * 100)
    n = len(returns)
    available = n >= min_samples
    return {
        "available": available, "sample_count": n, "min_samples": min_samples,
        "counts": counts, "target_touches": target_touches, "non_overlapping": True,
        "target_before_stop_rate": counts["target_first"] / n if available else None,
        "stop_evaluation_basis": "장중 터치 보수적 대용치 (종가 손절 전략의 실제 성과가 아님)",
        "mean_net_return_pct": mean(returns) if available else None,
        "median_target_days": median(days_to_target) if available and days_to_target else None,
        "basis": "동일 종목 과거 비중첩 경로의 상대 목표·손절 참고 빈도 (현재 신호 적중 확률 아님)",
        "limitations": "추세·장세 조건 미일치, 표본 의존성·기업행동·실제 체결·시장 충격 미반영",
    }


def build_exit_review(opens, highs, lows, closes, volumes, *, targets, stop_price,
                      horizon_days=3, in_progress=False, shares=None, entry_price=None,
                      round_trip_bps=20, participation_rate=0.01, exit_days=1):
    """Forward review anchored at the quote, separately from re-entry waiting time."""
    h = bounded_number(horizon_days, "horizon_days", 1, 60, integer=True)
    bps = bounded_number(round_trip_bps, "round_trip_bps", 0, 1000)
    issues = validate_ohlcv(opens, highs, lows, closes, volumes)
    out = {"version": EXECUTION_VERSION, "available": False, "horizon_days": h,
           "basis": "현재 시세 이후 거래일", "fill_guaranteed": False,
           "validation_status": "model_assumption_only",
           "cost_assumption": {"round_trip_bps": bps, "measured": False,
                               "includes": "왕복 수수료·세금·스프레드·슬리피지 합계 가정, 실제 비용으로 변경 필요"},
           "targets": [], "warnings": list(issues)}
    if issues:
        return out
    quote = closes[-1]
    cut = len(closes) - int(bool(in_progress))
    completed = closes[:cut]
    if len(completed) < 21:
        out["warnings"].append("확정 일봉 21봉 미만 — 청산 모델 보류")
        return out
    tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
          for i in range(max(1, cut - 14), cut)]
    volatility = blended_daily_sigma(completed, completed[-1], mean(tr))
    sigma = volatility.get("sigma")
    out.update(available=sigma is not None, quote=quote, quote_basis="진행 중 일봉" if in_progress else "최근 일봉 종가",
               completed_bars=cut, sigma_daily=sigma, volatility_basis=volatility.get("basis"),
               liquidity=liquidity_profile(closes, volumes, in_progress=in_progress, shares=shares,
                   participation_rate=participation_rate, exit_days=exit_days))
    entry = finite(entry_price) if entry_price is not None else quote
    size = finite(shares)
    if entry is None or entry <= 0 or (shares is not None and (size is None or size <= 0)):
        raise ValueError("매수가와 보유 수량은 양수여야 합니다")
    cost = bps / 20_000
    out["position"] = {"entry_price": entry, "shares": size,
                       "gross_return_pct": (quote / entry - 1) * 100,
                       "gross_pnl": None if size is None else (quote - entry) * size,
                       "estimated_net_pnl": None if size is None else (quote * (1-cost) - entry * (1+cost)) * size,
                       "break_even_price": entry * (1+cost) / (1-cost)}
    stop = finite(stop_price)
    out["stop_price"] = stop
    out["stop_already_breached"] = stop is not None and quote <= stop
    if stop is None or stop <= 0:
        out["warnings"].append("유효 손절 기준 미확보 — 목표·손절 선후관계 계산 보류")
    elif stop >= quote:
        out["warnings"].append("현재가가 손절 기준 이하 — 기존 계획 재검토 필요")
    if sigma is None:
        return out
    for i, item in enumerate(targets or [], 1):
        price = finite(item.get("price") if isinstance(item, dict) else item)
        if price is None or price <= 0:
            continue
        reached = price <= quote
        band = ({"low": 1.0, "mid": 1.0, "high": 1.0} if reached else
                touch_probability_range(quote, price, sigma, h))
        timing = None if reached else touch_time_window(quote, price, sigma, h)
        ref = None
        if price > quote and stop is not None and 0 < stop < quote:
            ref = historical_barrier_reference(
                opens[:cut] if opens is not None else None, highs[:cut], lows[:cut], completed,
                target_ratio=price / quote, stop_ratio=stop / quote, horizon_days=h, round_trip_bps=bps)
        out["targets"].append({
            "label": item.get("label", f"{i}차 목표") if isinstance(item, dict) else f"{i}차 목표",
            "price": price, "already_reached": reached, "horizon_days": h,
            "touch_probability": band,
            "touch_basis": "무추세 변동성 터치 모델, 손절 선행·호가 체결 미반영",
            "days_if_reached": timing.get("days") if timing and timing.get("within_horizon") else None,
            "time_is_conditional": True, "historical_reference": ref,
            "net_return_pct": (price * (1-cost) / (entry * (1+cost)) - 1) * 100,
            "net_pnl": None if size is None else (price * (1-cost) - entry * (1+cost)) * size,
            "reward_risk": (price - quote) / (quote - stop) if stop and 0 < stop < quote < price else None,
        })
    out["warnings"].extend([
        "가격 도달은 매도 체결 보장이 아니며 거래량 증가만으로 상승·도달일을 예측하지 않음",
        "일봉 모델은 장중 즉시 매도 시각을 산출하지 않음. 기한 도래 시 자동 매도 없이 계획을 재검토",
        "과거 목표·손절 선후관계는 장중 터치 손절을 가정한 보수적 참고치이며 종가 손절 전략의 실적이 아님",
    ])
    return out
