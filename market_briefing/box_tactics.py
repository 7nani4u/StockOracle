"""Célia Dupont-style box execution rules, applied to daily OHLCV.

Verified source
---------------
YouTube ``OlMBqW3FBes`` ("1시간만에 80% 수익낸 프랑스 천재의 초간단 단타매매법",
reviewed 2026-10-24 from the Korean auto transcript, 444 caption lines).

Principles the video states
---------------------------
1. Trade only two situations: a working box (support/resistance hold) and the
   breakout of that box. Skip directionless, slow stretches (the video calls
   most of the chart untradeable).
2. The middle of the box is a liquidity magnet. When a lot of trading happened
   in the middle third, price tends to return to the midline after touching
   either edge. When the middle is empty, the edge more often gives way.
3. Speed matters. Slow approaches and slow rebounds fail; a fast move with
   momentum sustains.
4. Order-book pressure decides ambiguous candles. A bullish candle can still be
   a sell win at resistance, and a bearish candle can be a buy win at support.
   A close beyond a level with opposite pressure is a false breakout.
5. Execution: stop beyond the prior swing, take 50% at the box midline and 50%
   at the opposite edge; breakouts use a 1:3 risk/reward and add to the position
   ("불타기") only after a pressure-confirmed retest.

Honest limits
-------------
The video's "듀퐁 스탠다드" reads a live order book. Daily OHLCV cannot reproduce
bid/ask queues, so every pressure number here is a transparent OHLCV proxy
(close location, body, wicks, volume participation). It is never labelled as
real order-flow data. All numeric thresholds are supplementary TA rules, not
values stated in the video. No field is a calibrated probability.

The module is intentionally dependency-free and never raises for malformed
input: callers may attach the result to an existing prediction response.

Measured counter-evidence (2026-10-25, ``scripts/audit_box_tactics.py``)
-----------------------------------------------------------------------
On 265 walk-forward support tests from 52 tickers, price reached the box midline
before the stop 41.5% of the time (95% CI 35.7-47.5%).  The video's magnet rule
did not reproduce: strong middles reached it 33.9% (20/59), weak middles 49.4%
(42/85).  Speed (fast 39.3 / normal 38.7 / slow 44.4) and pressure (buy 37.0 /
balanced 45.4 / sell 41.5) did not separate either, and breakout holds were
78.3% / 72.7% / 80.0% by pressure.  Those are order-of-arrival rates on daily
bars with zero costs, not win rates.  The rules stay as *hypotheses with
explicit confluences*; no field is presented as a validated edge.

Two intuitively appealing changes were implemented, audited and then reverted:
allowing weak-magnet rebounds (24 plans, mean R +0.27, the 7 added plans -0.22R)
and hard-blocking plans below their structural break-even (15 plans, mean R
+0.05, the 9 removed plans +0.65R).  Both are reported instead of enforced.
"""
from __future__ import annotations

import math
from statistics import median

from .ohlcv import load as load_ohlcv

VERSION = "2026-10-26.dupond-box-v4"

VIDEO_EVIDENCE = {
    "url": "https://www.youtube.com/watch?v=OlMBqW3FBes",
    "title": "1시간만에 80% 수익낸 프랑스 천재의 초간단 단타매매법",
    "review_method": "ko_auto_transcript",
    "reviewed_on": "2026-10-24",
    "transcript_lines": 444,
    "verified_principles": [
        "trade_only_box_or_breakout",
        "box_midline_liquidity_magnet",
        "speed_filter",
        "order_flow_pressure",
        "midline_50_far_edge_50_exit",
        "one_to_three_risk_reward",
        "breakout_retest_pyramid",
    ],
    "supplementary_rules": [
        "OHLCV_pressure_proxy",
        "ATR_normalisation",
        "volume_median_participation",
        "numeric_thresholds_not_in_video",
        "single_add_50_percent_instead_of_video_two_adds",
        "breakout_split_exits_instead_of_video_final_entry_3R_full_exit",
        "stop_after_t1_policy",
    ],
    "limits": (
        "영상의 호가창 압력은 일봉 OHLCV로 직접 재현할 수 없어 종가 위치·몸통·꼬리·"
        "거래량 참여도로 근사한다. 수치 임계값과 ATR 배수는 영상에 명시되지 않은 "
        "일반 기술적 분석 보완 규칙이며, 어떤 값도 검증된 승률이 아니다."
    ),
}

AUDIT_SNAPSHOT = {
    "measured_on": "2026-10-25",
    "source": "scripts/audit_box_tactics.py",
    "report": "docs/backtests/box_tactics_audit_20261025.json",
    "method": "walk-forward, stride 3, horizon 10 bars, next-open fill, same-bar stop priority",
    "universe": {"tickers": 52, "snapshots": 19700, "support_tests": 265, "ready_plans": 17},
    "metric": (
        "지지 접촉 후 10봉 안에 박스 중간선(1차 50% 목표)에 먼저 닿은 비율. "
        "손절은 지지 하단 -0.35 ATR. 수익률이 아니라 도달 순서다."
    ),
    "overall": {"samples": 265, "rate": 0.4151, "ci": [0.3574, 0.4752]},
    "by_magnet": {
        "strong": {"samples": 59, "rate": 0.339, "ci": [0.2314, 0.4663],
                   "video_claim": "강한 중간 자석 → 중간선 회귀 우위",
                   "verdict": "not_reproduced"},
        "normal": {"samples": 121, "rate": 0.3967, "ci": [0.314, 0.4857]},
        "weak": {"samples": 85, "rate": 0.4941, "ci": [0.3904, 0.5983],
                 "video_claim": "약한 중간 자석 → 가장자리 이탈 우위",
                 "verdict": "not_reproduced"},
    },
    "by_speed": {
        "fast": {"samples": 117, "rate": 0.3932},
        "normal": {"samples": 31, "rate": 0.3871},
        "slow": {"samples": 117, "rate": 0.4444},
        "verdict": "no_separation",
    },
    "by_pressure": {
        "buy": {"samples": 92, "rate": 0.3696},
        "balanced": {"samples": 108, "rate": 0.4537},
        "sell": {"samples": 65, "rate": 0.4154},
        "verdict": "no_separation",
    },
    "by_regime": {
        "range_support": {"samples": 48, "rate": 0.3958},
        "range_wait": {"samples": 97, "rate": 0.299},
        "no_trade": {"samples": 35, "rate": 0.5714},
        "breakout_watch": {"samples": 85, "rate": 0.4941},
    },
    "breakout_hold": {
        "buy": {"samples": 23, "rate": 0.7826},
        "balanced": {"samples": 33, "rate": 0.7273},
        "sell": {"samples": 15, "rate": 0.8},
        "verdict": "no_separation",
    },
    "ready_plans": {"samples": 17, "mean_r": 0.4767, "win_rate": 0.3529,
                    "ci": [0.1731, 0.587], "exits": {"stop": 10, "target2": 4, "horizon": 3}},
    "rejected_changes": [
        {"change": "weak-magnet rebound allowed with pressure+speed",
         "plans": 24, "mean_r": 0.2745,
         "added": {"samples": 7, "mean_r": -0.2163},
         "decision": "reverted: keep the block"},
        {"change": "hard block below structural break-even",
         "plans": 15, "mean_r": 0.0483,
         "removed": {"samples": 9, "mean_r": 0.6515},
         "decision": "informational only: the filter removed the better plans"},
    ],
    "limits": (
        "일봉·비용 0·동일봉 손절 우선. 표본 265/17로 신뢰구간이 넓고 겹친다. "
        "어떤 비율도 확률이나 기대수익이 아니며, 이 감사는 규칙의 방향을 반박·확인하는 용도다."
    ),
}

_REGIME_LABELS = {
    "no_data": "자료 부족",
    "mode_breakdown": "박스 이탈 · 신규 매수 제외",
    "false_breakout": "가짜 돌파 · 신규 매수 제외",
    "breakout_watch": "중간 유동성 약함 · 돌파 감시",
    "breakout_unconfirmed": "돌파 확인 부족 · 관망",
    "breakout_confirmed": "돌파 확인 · 재시험 매수",
    "range_support": "박스 하단 지지 · 분할 매수",
    "range_wait": "박스 중간·상단 · 지지 재확인 대기",
    "no_trade": "매매 제외 구간",
    "trend_retest": "추세 전환 재시험",
}


def _finite(value):
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _clamp(value, low=0.0, high=1.0):
    value = _finite(value)
    if value is None:
        return low
    return max(low, min(high, value))


def _round(value, digits=4):
    value = _finite(value)
    return round(value, digits) if value is not None else None


def _norm_ratio(value, low, high):
    if high <= low:
        return 0.5
    return _clamp((value - low) / (high - low))


def _arrays(dd):
    """Return aligned OHLCV lists plus the envelope-repair report, or None."""
    if not isinstance(dd, dict):
        return None
    try:
        (op, hi, lo, cl, vol), quality = load_ohlcv(dd, window=140)
    except (ValueError, TypeError, AttributeError):
        return None
    dates = dd.get("dates", dd.get("Date"))
    if dates is not None and len(dates) != len(cl):
        dates = None
    return op, hi, lo, cl, vol, dates, quality


def _atr(hi, lo, cl, period=14):
    n = len(cl)
    if n < 2:
        return None
    period = max(1, min(int(period), n - 1))
    start = n - period
    trs = [max(hi[i] - lo[i], abs(hi[i] - cl[i - 1]), abs(lo[i] - cl[i - 1]))
           for i in range(start, n)]
    if not trs:
        return None
    value = sum(trs) / len(trs)
    return value if math.isfinite(value) and value > 0 else None


def _rows(op, hi, lo, cl, vol, dates, baseline):
    rows = []
    for i, (o, h, l, c, v) in enumerate(zip(op, hi, lo, cl, vol)):
        rng = h - l
        if rng > 0:
            clv = ((c - l) - (h - c)) / rng
            body_ratio = (c - o) / rng
            upper_wick = (h - max(o, c)) / rng
            lower_wick = (min(o, c) - l) / rng
        else:
            clv = body_ratio = upper_wick = lower_wick = 0.0
        pressure = 0.55 * clv + 0.30 * body_ratio + 0.15 * (lower_wick - upper_wick)
        pressure = _clamp(pressure, -1.0, 1.0)
        if rng > 0:
            close_position = (c - l) / rng
        else:
            close_position = 0.5
        close_zone = ("upper_third" if close_position >= 2 / 3
                      else "lower_third" if close_position <= 1 / 3 else "middle_third")
        conflict = None
        if c > o and pressure <= -0.12:
            conflict = "bullish_body_sell_pressure"
        elif c < o and pressure >= 0.12:
            conflict = "bearish_body_buy_pressure"
        participation = v / baseline if baseline and baseline > 0 else 1.0
        rows.append({
            "index": i,
            "date": str(dates[i]) if dates is not None else None,
            "open": o, "high": h, "low": l, "close": c, "volume": v,
            "range": rng,
            "clv": _round(clv, 4),
            "body_ratio": _round(body_ratio, 4),
            "upper_wick_ratio": _round(upper_wick, 4),
            "lower_wick_ratio": _round(lower_wick, 4),
            "close_zone": close_zone,
            "pressure": _round(pressure, 4),
            "pressure_label": ("buy" if pressure >= 0.15
                               else "sell" if pressure <= -0.15 else "balanced"),
            "conflict": conflict,
            "participation": _round(participation, 3),
        })
    return rows


def _volume_context(rows):
    if not rows:
        return {"ratio": None, "label": "확인 불가", "baseline": None}
    previous = [r["volume"] for r in rows[-21:-1] if r["volume"] > 0]
    baseline = median(previous) if previous else None
    last = rows[-1]["volume"]
    if not baseline or baseline <= 0:
        return {"ratio": None, "label": "확인 불가", "baseline": None}
    ratio = last / baseline
    label = "증가" if ratio >= 1.2 else "감소" if ratio <= 0.8 else "보통"
    return {"ratio": _round(ratio, 3), "label": label, "baseline": _round(baseline, 3)}


def level_pressure(rows, indices, level, atr, side):
    """Aggregate the OHLCV pressure proxy at one horizontal level.

    ``indices`` should be the box touch indices (plus recent bars); the function
    filters them again by proximity to ``level`` so a stale index cannot leak a
    far-away candle into the verdict.
    """
    level = _finite(level)
    atr = _finite(atr)
    if not rows or level is None or level <= 0 or atr is None or atr <= 0:
        return {"verdict": "unknown", "touches": 0, "net": None,
                "buy_share": None, "sell_share": None, "conflicts": 0,
                "implication": "압력 계산에 필요한 가격·ATR 부족", "label": "확인 불가"}
    tolerance = atr * 0.20
    recent = {r["index"] for r in rows[-5:]}
    wanted = {int(i) for i in (indices or []) if isinstance(i, (int, float))} | recent
    touches = []
    for row in rows:
        if row["index"] not in wanted:
            continue
        # The bar must actually intersect the level band. The old one-sided test
        # counted a candle entirely below support (or above resistance) as a
        # "touch" and leaked unrelated pressure into the verdict.
        if row["low"] <= level + tolerance and row["high"] >= level - tolerance:
            touches.append(row)
    if not touches:
        return {"verdict": "unknown", "touches": 0, "net": None,
                "buy_share": None, "sell_share": None, "conflicts": 0,
                "implication": "레벨 접촉 표본 없음", "label": "표본 없음"}
    weights = [max(0.25, min(2.0, r["participation"] or 1.0)) for r in touches]
    total_weight = sum(weights) or 1.0
    net = sum(r["pressure"] * w for r, w in zip(touches, weights)) / total_weight
    buy_share = sum(w for r, w in zip(touches, weights) if r["pressure"] >= 0.15) / total_weight
    sell_share = sum(w for r, w in zip(touches, weights) if r["pressure"] <= -0.15) / total_weight
    verdict = "buy" if net >= 0.12 else "sell" if net <= -0.12 else "balanced"
    conflicts = sum(1 for r in touches if r["conflict"])
    if side == "support":
        implication = {
            "buy": "지지 접촉에서 매수 우위: 반등 지속 가능성",
            "sell": "지지 접촉에서 매도 우위: 지지 이탈 위험",
            "balanced": "지지 접촉 압력 균형: 반등 확인 필요",
            "unknown": "지지 압력 확인 불가",
        }[verdict]
        label = {"buy": "매수 우위", "sell": "매도 우위",
                 "balanced": "균형", "unknown": "확인 불가"}[verdict]
    else:
        implication = {
            "buy": "저항 접촉에서 매수 우위: 돌파 압력",
            "sell": "저항 접촉에서 매도 우위: 저항 유지 가능성",
            "balanced": "저항 접촉 압력 균형: 방향 확인 필요",
            "unknown": "저항 압력 확인 불가",
        }[verdict]
        label = {"buy": "매수 우위", "sell": "매도 우위",
                 "balanced": "균형", "unknown": "확인 불가"}[verdict]
    last = touches[-1]
    return {
        "verdict": verdict,
        "label": label,
        "touches": len(touches),
        "net": _round(net, 4),
        "buy_share": _round(buy_share, 4),
        "sell_share": _round(sell_share, 4),
        "conflicts": conflicts,
        "last_touch_index": last["index"],
        "last_touch_date": last["date"],
        "last_touch_conflict": last["conflict"],
        "implication": implication,
    }


def midline_analysis(rows, box):
    """Volume/time concentration in the middle third of the box (magnet strength)."""
    structure = (box or {}).get("box") or {}
    low = _finite(structure.get("low"))
    high = _finite(structure.get("high"))
    if not rows or low is None or high is None or high <= low:
        return {
            "available": False, "price": None, "state": "unknown",
            "strength_label": "확인 불가", "score": None,
            "volume_share": None, "time_share": None,
            "note": "박스 상·하단이 없어 중간선 자석을 계산할 수 없음",
        }
    width = high - low
    mid = (low + high) / 2
    mid_low = low + width / 3
    mid_high = high - width / 3
    starts = []
    for side_name in ("support", "resistance"):
        value = ((box or {}).get(side_name) or {}).get("first_index")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            starts.append(int(value))
    start = min(starts) if starts else max(0, len(rows) - 80)
    window = rows[max(0, start):][-120:]
    if not window:
        window = rows[-80:]
    total_volume = 0.0
    mid_volume = 0.0
    mid_closes = 0
    previous_side = None
    crosses = 0
    for row in window:
        rng = row["range"]
        overlap = max(0.0, min(row["high"], mid_high) - max(row["low"], mid_low))
        share = overlap / rng if rng > 0 else 0.0
        total_volume += row["volume"]
        mid_volume += row["volume"] * share
        if mid_low <= row["close"] <= mid_high:
            mid_closes += 1
        side = 1 if row["close"] > mid else -1 if row["close"] < mid else 0
        if side and previous_side and side != previous_side:
            crosses += 1
        if side:
            previous_side = side
    volume_share = mid_volume / total_volume if total_volume > 0 else 0.0
    time_share = mid_closes / len(window) if window else 0.0
    expected = 1 / 3
    volume_ratio = volume_share / expected
    time_ratio = time_share / expected
    score = (0.55 * _norm_ratio(volume_ratio, 0.70, 1.60)
             + 0.30 * _norm_ratio(time_ratio, 0.60, 1.70)
             + 0.15 * _clamp(crosses / 6.0))
    state = "strong" if score >= 0.60 else "weak" if score <= 0.30 else "normal"
    strength_label = {"strong": "강함", "normal": "보통", "weak": "약함"}[state]
    if state == "strong":
        note = ("영상 기준: 중간 1/3 거래 비중이 높으면 중간선이 자석처럼 작동한다. "
                "2026-10-25 감사에서는 강한 자석의 중간선 도달률 우위가 재현되지 않았다.")
    elif state == "weak":
        note = ("영상 기준: 중간 1/3 거래 비중이 낮으면 가장자리 이탈이 우위다. "
                "2026-10-25 감사에서는 오히려 중간선 도달률이 가장 높았다(49.4%).")
    else:
        note = "중간 1/3 거래 비중이 평범함: 중간선 회귀와 이탈 중 방향 확인 필요"
    return {
        "available": True,
        "price": _round(mid, 4),
        "mid_low": _round(mid_low, 4),
        "mid_high": _round(mid_high, 4),
        "state": state,
        "strength_label": strength_label,
        "score": _round(score, 4),
        "volume_share": _round(volume_share, 4),
        "volume_ratio": _round(volume_ratio, 3),
        "time_share": _round(time_share, 4),
        "time_ratio": _round(time_ratio, 3),
        "mid_crosses": crosses,
        "sample_bars": len(window),
        "window_start_index": window[0]["index"] if window else None,
        "note": note,
        "basis": "박스 첫 접촉 이후 봉의 중간 1/3에 배분된 거래량 비중(봉 범위 겹침 비례)과 종가 체류 비율",
    }


def speed_analysis(rows, atr):
    """ATR-normalised 3-bar speed versus the box-window baseline."""
    atr = _finite(atr)
    if not rows or len(rows) < 6 or atr is None or atr <= 0:
        return {"available": False, "state": "unknown", "label": "확인 불가",
                "speed": None, "ratio": None, "direction": "flat",
                "last_bar_atr": None, "note": "속도 계산에 필요한 봉·ATR 부족"}
    closes = [r["close"] for r in rows]
    k = 3
    move = closes[-1] - closes[-1 - k]
    speed = abs(move) / (atr * k)
    # The current 3-bar window must not be part of its own baseline: with few
    # bars the old loop made ratio exactly 1.0 and the speed filter silent.
    baseline_values = []
    for i in range(max(k, len(closes) - 60), len(closes) - k):
        baseline_values.append(abs(closes[i] - closes[i - k]) / (atr * k))
    if baseline_values:
        baseline = median(baseline_values)
    else:
        one_bar = [abs(closes[i] - closes[i - 1]) / atr
                   for i in range(max(1, len(closes) - 20), len(closes))]
        baseline = median(one_bar) if one_bar else speed
    ratio = speed / baseline if baseline and baseline > 0 else 1.0
    last_bar_atr = abs(closes[-1] - closes[-2]) / atr
    state = "fast" if ratio >= 1.25 or last_bar_atr >= 1.0 else "slow" if ratio <= 0.70 or last_bar_atr <= 0.20 else "normal"
    label = {"fast": "빠름", "normal": "보통", "slow": "느림"}[state]
    direction = "up" if move > atr * 0.10 else "down" if move < -atr * 0.10 else "flat"
    note = {
        "fast": "최근 3봉 이동이 기준보다 빠름: 추세 지속·반등 신뢰도 보강",
        "normal": "최근 3봉 이동이 기준 수준: 방향 확인 필요",
        "slow": "최근 3봉 이동이 느림: 영상 기준상 느린 반등·접근은 실패 가능성",
    }[state]
    return {
        "available": True,
        "state": state,
        "label": label,
        "speed_atr_per_bar": _round(speed, 3),
        "baseline_atr_per_bar": _round(baseline, 3),
        "ratio": _round(ratio, 3),
        "last_bar_atr": _round(last_bar_atr, 3),
        "direction": direction,
        "sample_bars": len(baseline_values),
        "note": note,
        "basis": "3봉 종가 이동 ÷ ATR ÷ 3, 현재 3봉을 제외한 직전 최대 60봉 중앙값 대비",
    }


def _regime(key, why, watch=None, tradeable=False, **extra):
    out = {
        "key": key,
        "label": _REGIME_LABELS.get(key, key),
        "tradeable": bool(tradeable),
        "why": [w for w in why if w],
        "watch": [w for w in watch or [] if w],
    }
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def magnet_evidence(state):
    """Return the measured record for one magnet state, or None."""
    bucket = (AUDIT_SNAPSHOT["by_magnet"] or {}).get(state)
    if not isinstance(bucket, dict):
        return None
    return {
        "state": state,
        "samples": bucket.get("samples"),
        "midline_rate": bucket.get("rate"),
        "ci": bucket.get("ci"),
        "video_claim": bucket.get("video_claim"),
        "verdict": bucket.get("verdict") or "unmeasured",
        "measured_on": AUDIT_SNAPSHOT["measured_on"],
    }


def classify_regime(box, mid, speed, support_pressure, resistance_pressure,
                    volume_context, current, atr, trend_entry_eligible=False):
    """Map the video's two tradeable situations to the current box state."""
    state = (box or {}).get("state")
    support = (box or {}).get("support") or {}
    resistance = (box or {}).get("resistance") or {}
    atr = _finite(atr) or 0.0
    if state in (None, "insufficient_data"):
        return _regime("no_data", ["박스 판정에 필요한 봉·가격 자료 부족"],
                       ["일봉이 더 쌓인 뒤 다시 확인"])
    if state == "no_box":
        if trend_entry_eligible:
            return _regime("trend_retest",
                           ["수평 박스는 없지만 추세 채널 전환·재시험 구간이 유효"],
                           ["재시험 종가가 채널 안에서 유지되는지 확인"], True)
        return _regime("no_trade",
                       ["수평 박스도, 유효한 추세 재시험도 없음",
                        "영상 기준: 방향 없는 구간은 매매 제외"],
                       ["지지·저항 반복 접촉 또는 채널 전환 확인"])
    if state in ("breakdown_pending", "breakdown_confirmed"):
        return _regime("mode_breakdown",
                       ["박스 하단 이탈: 신규 매수 근거가 사라짐"],
                       ["이탈 후 재회복 종가가 나오기 전까지 대기"])
    if state == "false_breakout":
        return _regime("false_breakout",
                       ["상단을 찍고 되돌아온 가짜 돌파",
                        "영상 기준: 레벨 밖 종가에도 반대 압력이면 돌파 실패"],
                       ["저항 위 종가 유지 + 매수 압력 + 거래량 재확인"])
    tolerance = atr * 0.15
    width = (resistance.get("low", 0) - support.get("high", 0)) if support and resistance else 0
    width_atr = width / atr if atr > 0 else 0
    if state == "breakout_pending":
        return _regime("breakout_watch",
                       ["종가가 박스 상단 밖에 있으나 확인 봉·압력이 부족"],
                       ["확인 종가 2개, 매수 압력, 거래량 1.2배 동시 확인"])
    if state == "breakout_confirmed":
        pressure_ok = resistance_pressure.get("verdict") == "buy"
        volume_ok = (volume_context.get("ratio") or 0) >= 1.05
        speed_ok = speed.get("state") == "fast"
        if pressure_ok and (volume_ok or speed_ok):
            return _regime(
                "breakout_confirmed",
                ["저항 위 종가 확인 + 매수 압력 우위",
                 "거래량 또는 속도가 돌파를 보강"],
                ["깨진 저항의 재시험 지지 여부", "재시험에서 매도 압력으로 바뀌면 철회"],
                True)
        return _regime(
            "breakout_unconfirmed",
            ["저항 밖 종가는 있으나 매수 압력·거래량·속도 중 확인이 부족",
             "영상 기준: 머리만 내민 돌파는 추격 금지"],
            ["재시험 지지 + 매수 압력 + 거래량 1.2배"],
            False)
    if width_atr and width_atr < 2.5:
        return _regime("no_trade",
                       [f"박스 폭 {width_atr:.1f} ATR: 손익비를 확보하기 어려운 좁은 구간"],
                       ["폭 확대 또는 박스 이탈 후 재형성"])
    near_support = current is None or current <= (support.get("high") or current) + atr * 1.5
    slow = speed.get("state") == "slow"
    mid_state = (mid or {}).get("state")
    support_setup = state == "support_rebound" or (state == "range" and near_support)
    support_confluence = (support_setup
                          and support_pressure.get("verdict") == "buy"
                          and not slow)
    # A weak middle third stays a no-new-long state. The video's reason ("the
    # edge gives way") did NOT reproduce on the midline metric (85 weak tests
    # reached the midline 49.4% vs 33.9% for strong), but the decision-level
    # metric did: the 7 weak-magnet plans that this block suppresses averaged
    # -0.22R with a 28.6% win rate. Removing a risk control needs
    # decision-level evidence, not a proxy that disagrees with it - so the
    # block stays and both measurements are attached to the verdict.
    if mid_state == "weak":
        return _regime(
            "breakout_watch",
            ["중간 1/3 거래 비중이 낮아 중간선 자석이 약함",
             "영상 기준: 중간이 비면 가장자리 회귀보다 이탈 지속 우위",
             "감사: 중간선 도달률은 49.4%(n=85)로 낮지 않았지만, 같은 구조의 실행 플랜은 평균 -0.22R(n=7)라 신규 매수 보류"],
            ["가장자리 종가 이탈 + 같은 방향 압력 + 거래량 증가"],
            False,
            magnet_state=mid_state,
            magnet_evidence=magnet_evidence(mid_state))
    magnet_why = {
        "strong": "중간선 자석이 강함: 영상 기준 중간선 회귀 우위",
        "weak": "약한 중간 자석: 영상 기준 이탈 우위이나 감사에서는 중간선 도달률이 더 높았음",
        "normal": "중간선 자석 보통: 회귀와 이탈 중 방향 확인 필요",
    }.get(mid_state)
    if state == "support_rebound":
        if support_pressure.get("verdict") == "sell":
            return _regime(
                "no_trade",
                ["지지 접촉 캔들에서 매도 압력 우위",
                 "영상 기준: 양봉처럼 보여도 매도가 이기면 반등 신뢰 낮음"],
                ["지지 위 종가 유지 + 매수 압력 전환"])
        if slow:
            return _regime(
                "no_trade",
                ["지지 반등 속도가 기준 이하",
                 "영상 기준: 느린 반등은 지지가 무너지기 쉬움"],
                ["3봉 이동 속도 회복 + 거래량 증가"])
        return _regime(
            "range_support",
            ["지지 접촉 후 상승 종가", magnet_why],
            ["중간선 도달 전 지지 하단 이탈 시 철회"],
            True,
            magnet_state=mid_state,
            magnet_evidence=magnet_evidence(mid_state))
    if state == "resistance_rejection":
        return _regime("range_wait",
                       ["저항 재하락: 신규 매수 구간이 아님",
                        "보유자는 중간선 50% 익절 규칙을 우선"],
                       ["지지 재확인 + 매수 압력 + 속도 회복"],
                       False)
    if state == "range":
        if support_confluence:
            return _regime(
                "range_support",
                ["현재가가 지지 부근이고 매수 압력 우위", magnet_why],
                ["지지 하단 종가 이탈 시 철회"],
                True,
                magnet_state=mid_state,
                magnet_evidence=magnet_evidence(mid_state))
        return _regime("range_wait",
                       ["현재가가 지지에서 멀거나 압력·속도 확인이 부족",
                        "영상 기준: 박스 중간 추격 금지"],
                       ["지지 재터치 + 매수 압력 + 속도 확인"],
                       False)
    return _regime("no_trade", [f"해석하지 않는 박스 상태: {state}"],
                   ["상태 재확인"])


def _plan_geometry(first_rr, measured=True):
    """Break-even hit rate implied by a 50:50 plan's first target.

    If half is taken at ``first_rr`` and the stop moves to entry afterwards, the
    remaining half risks 0R, so E[R] = p * 0.5 * rr - (1 - p) * 1 and break-even
    is p = 1 / (1 + rr/2).  Comparing that with the only hit rate we measured
    (41.5% on 265 support tests) turns "is the magnet strong?" into a question
    with an answer: a narrow box cannot pay for its own stop.
    """
    rr = _finite(first_rr)
    if rr is None or rr <= 0:
        return {"first_target_rr": None, "breakeven_rate": None, "measured_rate": None,
                "measured_samples": None, "ok": False,
                "assumption": "1차 50% 익절 후 손절을 진입가로 이동(잔여 50% 0R 보수 가정)",
                "note": "1차 목표 R을 계산할 수 없어 손익분기 확인 불가"}
    breakeven = 1.0 / (1.0 + 0.5 * rr)
    base = {
        "first_target_rr": _round(rr, 3),
        "breakeven_rate": _round(breakeven, 4),
        "assumption": "1차 50% 익절 후 손절을 진입가로 이동(잔여 50% 0R 보수 가정), 비용 0",
    }
    if not measured:
        return {**base, "measured_rate": None, "measured_samples": None, "ok": None,
                "note": (f"1차 목표 {rr:.2f}R → 손익분기 도달률 {breakeven:.0%}. "
                         "이 구조의 도달률 표본이 없어 손익분기만 표시")}
    rate = AUDIT_SNAPSHOT["overall"]["rate"]
    samples = AUDIT_SNAPSHOT["overall"]["samples"]
    ok = rate >= breakeven
    head = f"1차 목표 {rr:.2f}R → 손익분기 도달률 {breakeven:.0%}. 감사 표본 {rate:.1%}(n={samples})"
    note = (head + "가 이를 넘는다." if ok else
            head + "가 이에 미달한다. 이 구조는 중간선 도달만으로 손절 비용을 회수하지 못하므로, "
                   "더 넓은 박스나 더 가까운 1차 목표가 아니면 기대값 근거가 약하다. "
                   "감사 도달률은 접촉 봉 종가 기준, 진입은 지지 밴드 중앙 기준이라 근사 비교다.")
    # Informational only: the 9 plans this test would have removed averaged
    # +0.65R in the 2026-10-25 run, so it is displayed, not enforced.
    return {**base, "measured_rate": rate, "measured_samples": samples, "ok": ok, "note": note}


def _target(price, low, high, pct, label, reason):
    return {
        "price": _round(price, 4),
        "low": _round(low, 4),
        "high": _round(high, 4),
        "pct": pct,
        "label": label,
        "reason": reason,
    }


def _empty_plan(reason):
    return {
        "status": "none",
        "side": "none",
        "tradeable": False,
        "entry": None,
        "stop": None,
        "targets": [],
        "risk_reward": None,
        "rr_to_final": None,
        "pyramid": None,
        "invalidation": [reason] if reason else [],
        "exit_rule": "진입 없음 · 청산 계획 없음",
    }


def build_plan(regime, box, mid, speed, support_pressure, resistance_pressure,
               current, atr, trend_entry=None, rows=None, volume_context=None):
    """Build the 50:50 exit plan and the 1:3 breakout plan."""
    key = regime.get("key")
    atr = _finite(atr) or 0.0
    current = _finite(current)
    support = (box or {}).get("support") or {}
    resistance = (box or {}).get("resistance") or {}
    mid_price = _finite((mid or {}).get("price"))
    if key in {"no_data", "no_trade", "mode_breakdown", "false_breakout",
               "breakout_watch", "breakout_unconfirmed", "range_wait"}:
        reason = (regime.get("why") or ["실행 조건 미충족"])[0]
        return _empty_plan(reason)
    if key == "trend_retest":
        entry = trend_entry or {}
        low = _finite(entry.get("low"))
        high = _finite(entry.get("high"))
        if low is None or high is None or high <= low or atr <= 0:
            return _empty_plan("추세 재시험 진입 범위 없음")
        entry_mid = (low + high) / 2
        stop = low - atr * 0.60
        risk = entry_mid - stop
        t1 = _target(entry_mid + risk, entry_mid + risk * 0.9, entry_mid + risk * 1.1,
                     50, "1차 50% · 1R", "추세 재시험 후 첫 수익 실현")
        t2 = _target(entry_mid + risk * 3, entry_mid + risk * 2.8, entry_mid + risk * 3.2,
                     50, "2차 50% · 3R", "영상 기준 1:3 손익비")
        status = "ready" if current is not None and low - atr * 0.3 <= current <= high + atr * 0.5 else "wait"
        return {
            "status": status, "side": "long",
            "tradeable": status == "ready",
            "entry": {"low": _round(low, 4), "high": _round(high, 4),
                      "mid": _round(entry_mid, 4),
                      "condition": entry.get("reason") or "추세 채널 재시험 종가 확인"},
            "stop": {"price": _round(stop, 4), "basis": "재시험 구간 하단 - 0.6 ATR"},
            "stop_after_t1": _round(entry_mid, 4),
            "targets": [t1, t2],
            "risk_reward": 2.0,
            "rr_to_final": 3.0,
            "pyramid": None,
            "geometry": _plan_geometry(1.0, measured=False),
            "invalidation": ["재시험 구간 종가 이탈", "전환 채널 붕괴"],
            "exit_rule": "1차 1R 50% · 2차 3R 50%",
        }
    if key == "breakout_confirmed":
        level_low = _finite(resistance.get("low")) or _finite(resistance.get("center"))
        level_high = _finite(resistance.get("high")) or level_low
        if level_low is None or atr <= 0:
            return _empty_plan("돌파 저항 가격 없음")
        entry_low = level_low
        entry_high = level_high + atr * 0.25
        entry_mid = (entry_low + entry_high) / 2
        stop = level_low - atr * 0.60
        risk = entry_mid - stop
        if risk <= 0:
            return _empty_plan("돌파 재시험 손절 폭 계산 불가")
        t1 = _target(entry_mid + risk * 1.5, entry_mid + risk * 1.35, entry_mid + risk * 1.65,
                     50, "1차 50% · 1.5R", "돌파 후 재시험 지지 확인 시 절반 청산")
        t2 = _target(entry_mid + risk * 3.0, entry_mid + risk * 2.8, entry_mid + risk * 3.2,
                     50, "2차 50% · 3R", "영상 기준 1:3 손익비 최종 목표")
        quote_in_zone = current is not None and entry_low - atr * 0.5 <= current <= entry_high + atr * 0.5
        latest = rows[-1] if rows else None
        retest_now = bool(latest and latest["low"] <= entry_high and latest["close"] >= entry_low)
        pressure_now = (resistance_pressure or {}).get("verdict") == "buy"
        volume_now = ((volume_context or {}).get("ratio") or 0) >= 1.2
        entry_checks = [
            {"key": "quote", "label": "현재 시세가 진입 구간", "ok": quote_in_zone},
            {"key": "retest", "label": "확정 일봉 재시험 지지", "ok": retest_now},
            {"key": "pressure", "label": "재시험 매수 압력", "ok": pressure_now},
        ]
        status = "ready" if all(check["ok"] for check in entry_checks) else "wait"
        pyramid_ready = status == "ready" and volume_now
        pyramid_checks = [
            {"key": "retest", "label": "깨진 저항 재시험", "ok": retest_now},
            {"key": "pressure", "label": "재시험 매수 압력", "ok": pressure_now},
            {"key": "volume", "label": "거래량 1.2배", "ok": volume_now},
        ]
        return {
            "status": status, "side": "long",
            "tradeable": status == "ready",
            "entry_checks": entry_checks,
            "entry": {"low": _round(entry_low, 4), "high": _round(entry_high, 4),
                      "mid": _round(entry_mid, 4),
                      "condition": "깨진 저항의 재시험에서 종가 지지 + 매수 압력 우위 확인"},
            "stop": {"price": _round(stop, 4), "basis": "깨진 저항 - 0.6 ATR"},
            "stop_after_t1": _round(entry_mid, 4),
            "targets": [t1, t2],
            "risk_reward": 2.25,
            "rr_to_final": 3.0,
            "geometry": _plan_geometry(1.5, measured=False),
            "pyramid": {
                "enabled": True,
                "ready": pyramid_ready,
                "checks": pyramid_checks,
                "add_zone": [_round(entry_low, 4), _round(entry_high, 4)],
                "max_adds": 1,
                "size_pct_of_first": 50,
                "condition": "재시험 지지 유지 + 매수 압력 우위 + 거래량 1.2배 이상일 때만 1회 추가",
                "stop_after_add": _round(entry_mid, 4),
                "note": "불타기는 확인된 재시험에서만. 손실 확대·매도 압력 전환 시 추가 금지",
            },
            "invalidation": ["재시험 종가가 깨진 저항 아래로 복귀", "재시험 매도 압력 우위"],
            "exit_rule": "1차 1.5R 50% · 2차 3R 50%",
        }
    if key == "range_support":
        s_low = _finite(support.get("low"))
        s_high = _finite(support.get("high"))
        if s_low is None or s_high is None or mid_price is None or atr <= 0:
            return _empty_plan("지지·중간선 가격 없음")
        entry_low, entry_high = s_low, s_high
        entry_mid = (entry_low + entry_high) / 2
        stop = s_low - atr * 0.35
        risk = entry_mid - stop
        if risk <= 0:
            return _empty_plan("지지 손절 폭 계산 불가")
        r_low = _finite(resistance.get("low")) or mid_price + (mid_price - s_low)
        r_high = _finite(resistance.get("high")) or r_low
        far_mid = (r_low + r_high) / 2
        t1 = _target(mid_price, mid_price, mid_price, 50,
                     "1차 50% · 박스 중간선", "중간 1/3 유동성 자석 회귀 목표")
        t2 = _target(far_mid, r_low, r_high, 50,
                     "2차 50% · 반대편 저항", "영상 기준: 남은 절반은 반대편 가장자리")
        rr_mid = (mid_price - entry_mid) / risk
        rr_far = (far_mid - entry_mid) / risk
        weighted_rr = rr_mid * 0.5 + rr_far * 0.5
        status = "ready" if current is not None and current <= entry_high + atr * 0.4 and current >= stop else "wait"
        if current is not None and current > entry_high + atr * 0.4:
            status = "wait"
        mid_state = (mid or {}).get("state")
        magnet = {
            "state": mid_state,
            "caution": mid_state == "weak",
            "note": (
                "중간 유동성 약함: 영상 기준 가장자리 이탈 우위. 감사(2026-10-25)에서는 "
                "중간선 도달률이 오히려 높았으므로 자석 보너스 없이 압력·속도 확인만으로 판단."
                if mid_state == "weak" else
                "중간선 자석은 영상 규칙의 참고값이며 검증된 확률이 아님."
            ),
            "evidence": magnet_evidence(mid_state),
        }
        geometry = _plan_geometry(rr_mid, status)
        return {
            "status": status, "side": "long",
            "tradeable": status == "ready",
            "entry": {"low": _round(entry_low, 4), "high": _round(entry_high, 4),
                      "mid": _round(entry_mid, 4),
                      "condition": "지지 영역 종가 지지 + 매수 압력 우위 + 느리지 않은 반등"},
            "stop": {"price": _round(stop, 4), "basis": "지지 하단 - 0.35 ATR"},
            "stop_after_t1": _round(entry_mid, 4),
            "targets": [t1, t2],
            "risk_reward": _round(weighted_rr, 2),
            "rr_to_final": _round(rr_far, 2),
            "pyramid": None,
            "magnet": magnet,
            "geometry": geometry,
            "invalidation": ["지지 하단 종가 이탈", "중간선 도달 전 매도 압력 우위 전환"],
            "exit_rule": "중간선 50% · 반대편 저항 50%",
        }
    return _empty_plan("실행 플랜 미정의 국면")


def reversal_analysis(box, mid, speed, support_pressure, resistance_pressure,
                      current, atr, latest=None):
    """The video's other half: rejection / false-breakout risk scenario.

    This is *not* a sell or short order recommendation: the app is long-only.
    It tells a holder why a new long entry is unsafe and where the down scenario
    would first pause (midline) and then complete (support).
    """
    atr = _finite(atr) or 0.0
    current = _finite(current)
    state = (box or {}).get("state")
    resistance = (box or {}).get("resistance") or {}
    support = (box or {}).get("support") or {}
    r_low, r_high = _finite(resistance.get("low")), _finite(resistance.get("high"))
    s_low = _finite(support.get("low"))
    s_center = _finite(support.get("center")) or s_low
    mid_price = _finite((mid or {}).get("price"))
    base = {
        "available": False, "state": "unavailable", "label": "확인 불가",
        "side": "short_reference", "action": "hold_ok",
        "reason": "저항·중간선·ATR 자료 부족", "evidence": [],
        "distribution_score": None, "note": "하락 시나리오는 보유자 위험 안내이며 매도 주문 지시가 아님",
    }
    if (state in (None, "insufficient_data", "no_box", "no_data") or atr <= 0
            or r_low is None or r_high is None or mid_price is None
            or s_center is None or mid_price >= r_low):
        return base
    sell_win = (resistance_pressure or {}).get("verdict") == "sell"
    conflict = (latest or {}).get("conflict") == "bullish_body_sell_pressure"
    failed = state == "false_breakout"
    rejection = state == "resistance_rejection"
    slow = (speed or {}).get("state") == "slow"
    near_resistance = current is not None and current >= r_low - atr * 0.75
    net = _finite((resistance_pressure or {}).get("net"))
    score = (0.40 * (1.0 if sell_win else 0.0)
             + 0.25 * (1.0 if conflict else 0.0)
             + 0.25 * (1.0 if failed else 0.0)
             + 0.20 * (1.0 if rejection else 0.0)
             - 0.15 * (1.0 if slow else 0.0))
    score = _clamp(score)
    reasons = []
    if failed:
        reasons.append("저항 위로 머리만 내민 뒤 종가가 되돌아온 가짜 돌파")
    if rejection:
        reasons.append("저항 재터치 후 하락 종가")
    if sell_win:
        reasons.append(f"저항 접촉에서 매도 압력 우위(프록시 net {net:+.2f})" if net is not None
                       else "저항 접촉에서 매도 압력 우위(프록시)")
    if conflict:
        reasons.append("양봉인데 종가 위치·꼬리가 매도 우위를 가리킴")
    if slow:
        reasons.append("최근 이동 속도가 느려 하락 지속 근거는 약함")
    confirmed = near_resistance and ((failed and (sell_win or conflict))
                                     or (rejection and sell_win and not slow))
    watch = near_resistance and (sell_win or conflict) and current is not None and current >= mid_price
    status = "confirmed" if confirmed else "watch" if watch else "none"
    label = {"confirmed": "하락 시나리오 확인 · 신규 매수 회피",
             "watch": "저항 압력 감시 · 추격 매수 금지",
             "none": "하락 시나리오 근거 없음"}[status]
    result = {
        "available": True,
        "state": status,
        "label": label,
        "side": "short_reference",
        "action": "avoid_new_long" if status in {"confirmed", "watch"} else "hold_ok",
        "reason": reasons[0] if reasons else "저항 매도 우위·가짜 돌파 근거 없음",
        "evidence": reasons,
        "distribution_score": _round(score, 3),
        "level": {"low": _round(r_low, 4), "high": _round(r_high, 4)},
        "note": "하락 시나리오는 보유자 위험 안내이며 매도 주문 지시가 아님. 호가창이 아닌 OHLCV 프록시.",
    }
    if status == "none":
        return result
    entry_low, entry_high = r_low, r_high
    entry_mid = (entry_low + entry_high) / 2
    stop = r_high + atr * 0.50
    risk = stop - entry_mid
    if risk <= 0 or current is None:
        result.update(state="unavailable", label="확인 불가", action="hold_ok",
                      reason="손절 폭 또는 현재가 계산 불가")
        return result
    t1 = _target(mid_price, mid_price, mid_price, 50,
                 "1차 50% · 박스 중간선", "하락 시 중간선 자석에서 절반 청산 참고")
    t2 = _target(s_center, s_low, s_center, 50,
                 "2차 50% · 박스 하단 지지", "하락 시 반대편 지지에서 잔여 청산 참고")
    rr1 = (entry_mid - mid_price) / risk
    rr2 = (entry_mid - s_center) / risk
    result.update(
        entry={"low": _round(entry_low, 4), "high": _round(entry_high, 4),
               "mid": _round(entry_mid, 4), "condition": "저항 재시험 실패 종가 확인"},
        stop={"price": _round(stop, 4), "basis": "직전 스윙 고점 + 0.5 ATR"},
        targets=[t1, t2],
        risk_reward=_round(rr1 * 0.5 + rr2 * 0.5, 2) if rr2 > 0 else _round(rr1, 2),
        rr_to_final=_round(rr2, 2),
        invoked_at=[_round(current, 4), _round(entry_low, 4), _round(entry_high, 4)] if current else None,
    )
    return result


def rule_check(rows, box, atr, horizon=10):
    """Causal, in-sample check of the two rules on the same box window.

    Outcomes are measured only after each signal bar's close. The current box
    levels are used as fixed references, so this is a rule sanity check, not a
    walk-forward backtest and not a probability.
    """
    support = (box or {}).get("support") or {}
    resistance = (box or {}).get("resistance") or {}
    s_low = _finite(support.get("low"))
    s_high = _finite(support.get("high"))
    r_low = _finite(resistance.get("low"))
    r_high = _finite(resistance.get("high"))
    mid_price = _finite(((box or {}).get("midline")))
    if mid_price is None and s_low is not None and r_high is not None:
        mid_price = (s_low + r_high) / 2
    atr = _finite(atr)
    if not rows or atr is None or atr <= 0 or mid_price is None:
        return {"available": False, "sample": 0, "note": "규칙 점검에 필요한 가격 부족"}
    start = max(1, len(rows) - 60)
    stop = s_low - atr * 0.35 if s_low is not None else None
    support_touches = 0
    midline_first = 0
    stop_first = 0
    ambiguous = 0
    last_touch = -99
    for i in range(start, len(rows) - 1):
        row = rows[i]
        if s_low is None or s_high is None or stop is None:
            break
        touched = row["low"] <= s_high + atr * 0.15 and row["close"] >= s_low
        if not touched or i - last_touch < 3:
            continue
        support_touches += 1
        last_touch = i
        for j in range(i + 1, min(len(rows), i + 1 + horizon)):
            hit_mid = rows[j]["high"] >= mid_price
            hit_stop = rows[j]["low"] <= stop
            if hit_mid and hit_stop:
                ambiguous += 1
                break
            if hit_mid:
                midline_first += 1
                break
            if hit_stop:
                stop_first += 1
                break
    breakout_touches = 0
    breakout_holds = 0
    if r_low is not None and r_high is not None:
        last_break = -99
        for i in range(start, len(rows) - 4):
            if i - last_break < 3:
                continue
            if rows[i]["close"] <= r_high + atr * 0.10:
                continue
            breakout_touches += 1
            last_break = i
            held = all(rows[j]["close"] >= r_low for j in range(i + 1, i + 4))
            if held:
                breakout_holds += 1
    resolved = midline_first + stop_first
    midline_rate = midline_first / resolved if resolved else None
    breakout_rate = breakout_holds / breakout_touches if breakout_touches else None
    return {
        "available": support_touches > 0 or breakout_touches > 0,
        "sample": support_touches,
        "horizon_bars": horizon,
        "support_touches": support_touches,
        "midline_first": midline_first,
        "stop_first": stop_first,
        "ambiguous": ambiguous,
        "midline_rate": _round(midline_rate, 4),
        "breakout_touches": breakout_touches,
        "breakout_holds": breakout_holds,
        "breakout_hold_rate": _round(breakout_rate, 4),
        "note": (
            "같은 박스 기준으로 신호 봉 종가 이후만 채점한 표본 점검. "
            "박스 레벨은 현재 값을 고정 참조하므로 워크포워드 백테스트가 아니며, "
            "표본이 적으면 비율을 신뢰하지 말 것. 확률 아님."
        ),
    }


def _checklist(regime, mid, speed, support_pressure, resistance_pressure,
               volume_context, plan):
    items = []
    items.append({"key": "regime", "label": "매매 국면",
                  "value": regime.get("label"),
                  "ok": bool(regime.get("tradeable")),
                  "detail": (regime.get("why") or [""])[0]})
    mid_state = (mid or {}).get("state")
    items.append({"key": "mid", "label": "중간선 자석",
                  "value": f"{(mid or {}).get('strength_label', '확인 불가')}"
                           + (f" · 중간 거래량 {((mid or {}).get('volume_share') or 0) * 100:.0f}%"
                              if (mid or {}).get("available") else ""),
                  "ok": mid_state in {"strong", "normal"},
                  "detail": (mid or {}).get("note")})
    items.append({"key": "speed", "label": "속도",
                  "value": f"{(speed or {}).get('label', '확인 불가')}"
                           + (f" · {((speed or {}).get('ratio') or 0):.2f}x" if (speed or {}).get("available") else ""),
                  "ok": (speed or {}).get("state") in {"fast", "normal"},
                  "detail": (speed or {}).get("note")})
    if regime.get("key") == "breakout_confirmed":
        verdict = (resistance_pressure or {}).get("verdict")
        label = (resistance_pressure or {}).get("label")
        detail = (resistance_pressure or {}).get("implication")
    else:
        verdict = (support_pressure or {}).get("verdict")
        label = (support_pressure or {}).get("label")
        detail = (support_pressure or {}).get("implication")
    items.append({"key": "pressure", "label": "압력(프록시)",
                  "value": label or "확인 불가",
                  "ok": verdict == "buy",
                  "detail": detail})
    ratio = (volume_context or {}).get("ratio")
    items.append({"key": "volume", "label": "거래량 확인",
                  "value": (f"{ratio:.2f}배" if ratio is not None else "확인 불가"),
                  "ok": ratio is not None and ratio >= 1.0,
                  "detail": "최근 봉 거래량 ÷ 직전 20봉 중앙값"})
    items.append({"key": "exit", "label": "분할 청산",
                  "value": (plan or {}).get("exit_rule") or "진입 없음",
                  "ok": bool((plan or {}).get("targets")),
                  "detail": "영상 기준: 중간선 50% + 반대편 50%, 돌파는 1:3"})
    geometry = (plan or {}).get("geometry") or {}
    if geometry.get("breakeven_rate") is not None:
        measured = geometry.get("measured_rate")
        items.append({
            "key": "geometry",
            "label": "손익분기 구조",
            "value": (f"1차 {geometry.get('first_target_rr')}R · 필요 {geometry.get('breakeven_rate'):.0%}"
                      + (f" / 감사 {measured:.1%}" if measured is not None else "")),
            "ok": geometry.get("ok") is True,
            "detail": geometry.get("note"),
        })
    return items


def _flow_context(flow):
    if not isinstance(flow, dict) or not flow.get("ok"):
        reason = (flow or {}).get("reason") if isinstance(flow, dict) else None
        return {"available": False, "reason": reason or "수급 데이터 없음"}
    foreign = _finite(flow.get("외국인")) or 0.0
    inst = _finite(flow.get("기관")) or 0.0
    pension = _finite(flow.get("연기금")) or 0.0
    net = foreign + inst
    side = "buy" if net > 0 else "sell" if net < 0 else "neutral"
    label = {"buy": "순매수 우위", "sell": "순매도 우위", "neutral": "중립"}[side]
    return {
        "available": True,
        "date": flow.get("date"),
        "foreign": _round(foreign, 0),
        "institution": _round(inst, 0),
        "pension": _round(pension, 0),
        "net": _round(net, 0),
        "side": side,
        "label": label,
        "note": "KRX 투자자별 순매수 주식 수 · 호가창 압력이 아닌 실제 수급 보조 지표",
    }


def analyze_box_tactics(dd, box, trend=None, current=None, flow=None):
    """Return the video-derived execution plan attached to a box response."""
    try:
        arrays = _arrays(dd)
        if arrays is None:
            return _placeholder("유효한 OHLCV 없음")
        op, hi, lo, cl, vol, dates, data_quality = arrays
        box = box if isinstance(box, dict) else {}
        atr = _finite(box.get("atr")) or _atr(hi, lo, cl)
        if atr is None or atr <= 0:
            return _placeholder("ATR 계산 불가")
        baseline_values = [v for v in vol[-20:] if v > 0]
        baseline = median(baseline_values) if baseline_values else 1.0
        rows = _rows(op, hi, lo, cl, vol, dates, baseline)
        volume_context = _volume_context(rows)
        mid = midline_analysis(rows, box)
        speed = speed_analysis(rows, atr)
        support = box.get("support") or {}
        resistance = box.get("resistance") or {}
        support_pressure = level_pressure(
            rows, support.get("touch_indices"), support.get("center"), atr, "support")
        resistance_pressure = level_pressure(
            rows, resistance.get("touch_indices"), resistance.get("center"), atr, "resistance")
        trend_entry = (trend or {}).get("entry_zone") or {}
        regime = classify_regime(
            box, mid, speed, support_pressure, resistance_pressure,
            volume_context, current, atr, trend_entry.get("eligible", False))
        plan = build_plan(regime, box, mid, speed, support_pressure,
                          resistance_pressure, current, atr, trend_entry,
                          rows=rows, volume_context=volume_context)
        reversal = reversal_analysis(box, mid, speed, support_pressure, resistance_pressure,
                                     current, atr, rows[-1] if rows else None)
        check = rule_check(rows, {**box, "midline": mid.get("price")}, atr)
        flow_context = _flow_context(flow)
        checklist = _checklist(regime, mid, speed, support_pressure,
                               resistance_pressure, volume_context, plan)
        if reversal.get("state") == "confirmed":
            checklist.append({"key": "reversal", "label": "하락 시나리오",
                              "value": reversal.get("label"), "ok": False,
                              "detail": reversal.get("reason")})
        met = sum(1 for item in checklist if item.get("ok"))
        summary = _summary(regime, plan)
        return {
            "version": VERSION,
            "video_evidence": VIDEO_EVIDENCE,
            "audit": AUDIT_SNAPSHOT,
            "data_quality": data_quality,
            "regime": regime,
            "midline": mid,
            "speed": speed,
            "pressure": {
                "support": support_pressure,
                "resistance": resistance_pressure,
                "latest_bar": rows[-1] if rows else None,
                "note": "OHLCV 프록시: 거래량 가중 종가 위치·몸통·꼬리. 실제 호가창 아님",
            },
            "volume": volume_context,
            "flow": flow_context,
            "plan": plan,
            "reversal": reversal,
            "checklist": checklist,
            "conditions_met": met,
            "conditions_total": len(checklist),
            "rule_check": check,
            "summary": summary,
            "limits": VIDEO_EVIDENCE["limits"],
        }
    except Exception as exc:  # never break a prediction response
        return _placeholder(f"박스 실행 규칙 계산 실패: {type(exc).__name__}")


def _price_text(value):
    value = _finite(value)
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


def _summary(regime, plan):
    key = (plan or {}).get("status")
    if key == "ready" and (plan or {}).get("entry"):
        entry = plan["entry"]
        warning = ""
        if (plan.get("geometry") or {}).get("ok") is False:
            warning = " · 손익분기 미달 주의(정보 표시, 진입 차단 아님)"
        return (f"{regime.get('label')} · 진입 {_price_text(entry['low'])}~{_price_text(entry['high'])}, "
                f"{(plan.get('exit_rule') or '')}{warning}")
    if key == "wait":
        missing = [item["label"] for item in plan.get("entry_checks") or [] if not item.get("ok")]
        if missing:
            return f"{regime.get('label')} · 진입 대기: " + " · ".join(missing) + " 확인 필요"
        return f"{regime.get('label')} · 가격이 진입 구간에 오면 실행, 지금은 추격 금지"
    why = (regime.get("why") or ["실행 조건 미충족"])[0]
    return f"{regime.get('label')} · {why}"


def _placeholder(reason):
    regime = _regime("no_data", [reason], ["일봉·가격 자료 확인"])
    mid = {"available": False, "price": None, "state": "unknown",
           "strength_label": "확인 불가", "score": None,
           "volume_share": None, "time_share": None, "note": reason}
    speed = {"available": False, "state": "unknown", "label": "확인 불가",
             "speed_atr_per_bar": None, "ratio": None, "direction": "flat",
             "note": reason}
    return {
        "version": VERSION,
        "video_evidence": VIDEO_EVIDENCE,
        "audit": AUDIT_SNAPSHOT,
        "data_quality": None,
        "regime": regime,
        "midline": mid,
        "speed": speed,
        "pressure": {"support": {"verdict": "unknown", "label": "확인 불가"},
                     "resistance": {"verdict": "unknown", "label": "확인 불가"},
                     "latest_bar": None,
                     "note": "OHLCV 프록시: 실제 호가창 아님"},
        "volume": {"ratio": None, "label": "확인 불가"},
        "flow": {"available": False, "reason": "수급 데이터 없음"},
        "plan": _empty_plan(reason),
        "reversal": {**reversal_analysis({}, None, None, None, None, None, None),
                     "reason": reason},
        "checklist": [],
        "conditions_met": 0,
        "conditions_total": 0,
        "rule_check": {"available": False, "sample": 0, "note": reason},
        "summary": f"자료 부족 · {reason}",
        "limits": VIDEO_EVIDENCE["limits"],
    }
