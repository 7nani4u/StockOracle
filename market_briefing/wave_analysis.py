"""N-wave direction and historical wave hit-rate (pure calculation).

Screenshot-2 contract: prediction tab shows
  예측 방향 | 상승 (N자)
  파동 적중률 | 48.9% (437/893)

This module provides both values from OHLCV arrays only (no network),
causally (only completed bars up to each signal point are used).

- N-wave: last 3 confirmed fractal pivots.
  - [Low, High, Low] with higher second low  -> bullish N (상승 N자)
  - [High, Low, High] with lower second high -> bearish N (하락 역N자)
  - [Low, High, Low] without higher low      -> weak bullish / 박스권
  - otherwise                                -> 중립/박스권
- Hit-rate: walk the same detector over history; a signal is a "hit"
  when the predicted direction is realized within the forward window
  (default 10 bars, ±1% move).  Pooled over up+down signals so the
  denominator matches the "(hits/total)" display.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clean(values: Sequence[Any]) -> list[float | None]:
    out: list[float | None] = []
    for v in values or []:
        f = _finite(v)
        out.append(f if f is not None and f > 0 else None)
    return out


def find_swing_pivots(
    highs: Sequence[Any],
    lows: Sequence[Any],
    window: int = 2,
) -> list[dict]:
    """Confirmed fractal pivots (needs `window` bars on each side)."""
    h = _clean(highs)
    l = _clean(lows)
    n = min(len(h), len(l))
    pivots: list[dict] = []
    if n < window * 2 + 1:
        return pivots
    for i in range(window, n - window):
        hi = h[i]
        lo = l[i]
        if hi is not None:
            neigh = [h[j] for j in range(i - window, i + window + 1) if h[j] is not None]
            if neigh and hi >= max(neigh):
                # strict extreme: must be strictly greater than at least one side
                # to avoid flat plateaus producing double pivots
                left = [h[j] for j in range(i - window, i) if h[j] is not None]
                right = [h[j] for j in range(i + 1, i + window + 1) if h[j] is not None]
                if (left and hi > min(left)) or (right and hi > min(right)):
                    # avoid consecutive same-type pivots: keep the more extreme
                    if pivots and pivots[-1]["kind"] == "high":
                        if hi > pivots[-1]["price"]:
                            pivots[-1] = {"kind": "high", "index": i, "price": hi}
                        continue
                    pivots.append({"kind": "high", "index": i, "price": hi})
                    continue
        if lo is not None:
            neigh = [l[j] for j in range(i - window, i + window + 1) if l[j] is not None]
            if neigh and lo <= min(neigh):
                left = [l[j] for j in range(i - window, i) if l[j] is not None]
                right = [l[j] for j in range(i + 1, i + window + 1) if l[j] is not None]
                if (left and lo < max(left)) or (right and lo < max(right)):
                    if pivots and pivots[-1]["kind"] == "low":
                        if lo < pivots[-1]["price"]:
                            pivots[-1] = {"kind": "low", "index": i, "price": lo}
                        continue
                    pivots.append({"kind": "low", "index": i, "price": lo})
    # enforce alternation (drop weaker of two consecutive same-kind)
    filtered: list[dict] = []
    for p in pivots:
        if filtered and filtered[-1]["kind"] == p["kind"]:
            continue
        filtered.append(p)
    return filtered


def classify_n_wave(pivots: Sequence[dict], current_price: float | None = None) -> dict:
    """Classify the latest N-wave shape from the last 3 pivots."""
    recent = list(pivots or [])[-3:]
    if len(recent) < 3:
        return {
            "pattern": "판단 보류",
            "pattern_short": "—",
            "direction": "중립",
            "direction_key": "neutral",
            "reason": "확정 스윙 고점·저점이 3개 미만이라 N자 구조를 판단할 수 없습니다.",
        }
    kinds = [p["kind"] for p in recent]
    prices = [float(p["price"]) for p in recent]
    if kinds == ["low", "high", "low"]:
        higher_low = prices[2] > prices[0]
        if higher_low:
            return {
                "pattern": "N자",
                "pattern_short": "N자",
                "direction": "상승",
                "direction_key": "up",
                "reason": f"저점 {prices[0]:g} → 고점 {prices[1]:g} → 더 높은 저점 {prices[2]:g}의 상승 N자",
            }
        return {
            "pattern": "박스권",
            "pattern_short": "박스권",
            "direction": "중립",
            "direction_key": "neutral",
            "reason": "저점-고점-저점이지만 저점이 낮아져 상승 N자로 보지 않습니다.",
        }
    if kinds == ["high", "low", "high"]:
        lower_high = prices[2] < prices[0]
        if lower_high:
            return {
                "pattern": "역N자",
                "pattern_short": "역N자",
                "direction": "하락",
                "direction_key": "down",
                "reason": f"고점 {prices[0]:g} → 저점 {prices[1]:g} → 더 낮은 고점 {prices[2]:g}의 하락 역N자",
            }
        return {
            "pattern": "박스권",
            "pattern_short": "박스권",
            "direction": "중립",
            "direction_key": "neutral",
            "reason": "고점-저점-고점이지만 고점이 높아져 하락 역N자로 보지 않습니다.",
        }
    # W / M need 5 pivots; with only 3 of mixed shape fall back to momentum of last leg
    if len(recent) == 3:
        if prices[2] > prices[0] and kinds[2] == "high":
            return {
                "pattern": "상승 추세",
                "pattern_short": "추세",
                "direction": "상승",
                "direction_key": "up",
                "reason": "최근 고점이 높아지는 상승 추세 구간입니다.",
            }
        if prices[2] < prices[0] and kinds[2] == "low":
            return {
                "pattern": "하락 추세",
                "pattern_short": "추세",
                "direction": "하락",
                "direction_key": "down",
                "reason": "최근 저점이 낮아지는 하락 추세 구간입니다.",
            }
    return {
        "pattern": "박스권",
        "pattern_short": "박스권",
        "direction": "중립",
        "direction_key": "neutral",
        "reason": "최근 3개 스윙이 N자·역N자 구조가 아니라 방향을 단정하지 않습니다.",
    }


def _signal_at(pivots: Sequence[dict]) -> str | None:
    """Direction signal ('up'/'down') implied by the last 3 pivots, or None."""
    cls = classify_n_wave(pivots)
    key = cls.get("direction_key")
    return key if key in ("up", "down") else None


def wave_hit_rate(
    closes: Sequence[Any],
    highs: Sequence[Any] | None = None,
    lows: Sequence[Any] | None = None,
    *,
    forward_bars: int = 10,
    threshold_pct: float = 1.0,
    min_pivots_for_signal: int = 3,
) -> dict:
    """Historical hit-rate of the N-wave direction signal (causal walk).

    For every past bar `t` (with enough history + forward window), the same
    fractal detector runs on data up to `t` only; if it emits up/down, the
    signal is a hit when the max/min close over the next `forward_bars`
    moves >= `threshold_pct` in the predicted direction.
    """
    c = _clean(closes)
    n = len(c)
    h = _clean(highs) if highs is not None else list(c)
    l = _clean(lows) if lows is not None else list(c)
    # align lengths
    m = min(len(c), len(h), len(l))
    c, h, l = c[:m], h[:m], l[:m]
    fwd = max(1, int(forward_bars))
    thr = abs(float(threshold_pct)) / 100.0
    hits = 0
    total = 0
    up_hits = up_total = down_hits = down_total = 0
    if m < 12:
        return {
            "hit_rate_pct": None, "hits": 0, "total": 0,
            "up": {"hits": 0, "total": 0}, "down": {"hits": 0, "total": 0},
            "forward_bars": fwd, "threshold_pct": float(threshold_pct),
            "note": "과거 파동 표본이 부족해 적중률을 산정하지 못했습니다.",
        }
    # walk: need at least ~10 bars of history for pivots + fwd bars ahead
    for t in range(10, m - fwd):
        entry = c[t]
        if entry is None or entry <= 0:
            continue
        window = 2
        # pivots strictly before/at t (exclude the in-progress bar at t+1..)
        pivots = find_swing_pivots(h[: t + 1], l[: t + 1], window=window)
        if len(pivots) < min_pivots_for_signal:
            continue
        sig = _signal_at(pivots)
        if sig is None:
            continue
        future = [v for v in c[t + 1: t + 1 + fwd] if v is not None and v > 0]
        if not future:
            continue
        total += 1
        if sig == "up":
            up_total += 1
            if max(future) >= entry * (1.0 + thr):
                hits += 1
                up_hits += 1
        else:
            down_total += 1
            if min(future) <= entry * (1.0 - thr):
                hits += 1
                down_hits += 1
    rate = round(hits / total * 100.0, 1) if total else None
    return {
        "hit_rate_pct": rate, "hits": hits, "total": total,
        "up": {"hits": up_hits, "total": up_total,
               "rate": round(up_hits / up_total * 100.0, 1) if up_total else None},
        "down": {"hits": down_hits, "total": down_total,
                 "rate": round(down_hits / down_total * 100.0, 1) if down_total else None},
        "forward_bars": fwd, "threshold_pct": float(threshold_pct),
        "note": (f"과거 N자·역N자 신호 {total}건 중 {hits}건 적중(앞 {fwd}봉·{threshold_pct:g}% 기준)"
                 if total else "과거 파동 표본이 부족해 적중률을 산정하지 못했습니다."),
    }


def analyze_wave(
    closes: Sequence[Any],
    highs: Sequence[Any] | None = None,
    lows: Sequence[Any] | None = None,
    *,
    forward_bars: int = 10,
    threshold_pct: float = 1.0,
) -> dict:
    """Current N-wave direction + pooled historical hit-rate (never raises)."""
    try:
        c = _clean(closes)
        h = _clean(highs) if highs is not None else list(c)
        l = _clean(lows) if lows is not None else list(c)
        m = min(len(c), len(h), len(l))
        c, h, l = c[:m], h[:m], l[:m]
        pivots = find_swing_pivots(h, l, window=2)
        current = c[-1] if c and c[-1] else None
        cls = classify_n_wave(pivots, current)
        stat = wave_hit_rate(c, h, l, forward_bars=forward_bars, threshold_pct=threshold_pct)
        direction = cls.get("direction") or "중립"
        pattern_short = cls.get("pattern_short") or "—"
        label = f"{direction} ({pattern_short})" if pattern_short not in ("—",) else direction
        return {
            "direction": direction,
            "direction_key": cls.get("direction_key") or "neutral",
            "pattern": cls.get("pattern") or "판단 보류",
            "pattern_short": pattern_short,
            "label": label,
            "reason": cls.get("reason") or "",
            "hit_rate_pct": stat.get("hit_rate_pct"),
            "hits": int(stat.get("hits") or 0),
            "total": int(stat.get("total") or 0),
            "up": stat.get("up") or {},
            "down": stat.get("down") or {},
            "forward_bars": stat.get("forward_bars") or forward_bars,
            "threshold_pct": float(stat.get("threshold_pct") if stat.get("threshold_pct") is not None else threshold_pct),
            "note": stat.get("note") or "",
            "pivot_count": len(pivots),
        }
    except Exception as exc:  # never break the prediction tab
        return {
            "direction": "중립", "direction_key": "neutral",
            "pattern": "판단 보류", "pattern_short": "—", "label": "중립",
            "reason": f"파동 분석 중 오류({type(exc).__name__})로 중립 처리합니다.",
            "hit_rate_pct": None, "hits": 0, "total": 0,
            "up": {}, "down": {}, "forward_bars": forward_bars,
            "threshold_pct": float(threshold_pct), "note": "파동 표본 부족",
            "pivot_count": 0,
        }
