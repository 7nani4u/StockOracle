"""Causal box-sequence detector (Darvas 3-bar rule, symmetric) for the box execution plan.

Why this module exists
----------------------
YouTube ``NOIOQP-7PPM`` ("모르면 평생 호구 되는 박스권 추세 전환 비밀", 79 s, uploaded
2026-10-08, re-checked 2026-11-05 from the Korean auto captions and key frames 0:06-1:17)
describes price as a *chain of boxes*: falling boxes that break down one after another
(0:32-0:41 "이탈 박스 이탈 박스"), a box that "no longer falls", then a box that steps up
(0:43). The place where the box character changes is the inflection; if the upward step
fails and price falls back, that is the stop (0:45-0:47 diagram labels "손절").

The existing horizontal engine (``box_range``) needs one box that contains 80 % of the last
80 closes, so trending stocks almost never qualify (2026-11-05 replay: 99 % of 5,458
walk-forward snapshots had no executable plan). This module finds the *latest* boxes
instead, so the plan can say where the current box is and whether the chain is falling,
flat or rising.

Numeric rules are supplements, not video rules
----------------------------------------------
The video gives no formula. The box edges follow Nicolas Darvas' published rule (a high is
a box top once the next three bars do not exceed it; the bottom is the lowest low after the
top that the next three bars do not undercut), mirrored for falling markets. ATR tolerances
are engineering choices documented in ``BoxSequenceConfig``. Only completed bars may be
passed in; the detector never reads beyond the last element, so a prefix gives the same
boxes it would have given on that day.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .ohlcv import FLOAT_NOISE_TOL, extract as extract_ohlcv, load as load_ohlcv
from .box_volume import build_volume_evidence

VERSION = "2026-11-11.box-sequence-v3-volume"

# An OHLC row whose high/low disagree with its open/close by more than float noise but less
# than this share of price is almost always a partial dividend adjustment (seen on 47 rows of
# 52 cached tickers, 0.1-2.2 %). Larger gaps are more likely a wrong print and stay rejected.
MAX_ENVELOPE_GAP = 0.05


@dataclass(frozen=True)
class BoxSequenceConfig:
    lookback: int = 160            # bars scanned (about 8 months of daily bars)
    atr_period: int = 14
    confirm_bars: int = 3          # Darvas: an extreme holds for three later bars
    min_box_bars: int = 5          # a box spans at least five bars from its anchor
    max_forming_atr: float = 6.0   # a forming range wider than this is a trend leg, re-anchor it
    break_tol_atr: float = 0.10    # close beyond an edge by more than this ends the box
    relation_tol_atr: float = 0.50 # box-to-box level comparison tolerance
    touch_tol_atr: float = 0.25    # bar extreme within this of an edge counts as a touch


def _finite(value):
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _atr_series(hi, lo, cl, period, origin=0):
    """Trailing simple ATR; ``atr[i]`` only uses bars from ``origin`` up to ``i``.

    ``origin`` > 0 starts a fresh regime after a one-day jump of 35 % or more, so the jump bar's
    gap does not inflate the tolerances of the next ``period`` bars.
    """
    out = []
    trs = []
    for i in range(len(cl)):
        if i < origin:
            out.append(max(hi[i] - lo[i], cl[i] * 1e-6, 1e-12))
            continue
        if i == 0 or i == origin:
            tr = hi[i] - lo[i]
        else:
            tr = max(hi[i] - lo[i], abs(hi[i] - cl[i - 1]), abs(lo[i] - cl[i - 1]))
        trs.append(tr)
        window = trs[-period:]
        value = sum(window) / len(window)
        out.append(value if value > 0 else max(cl[i] * 1e-6, 1e-12))
    return out


def _sanitize_envelope(dd, window, dates=None):
    """Use only the body (open/close) of bars whose high/low contradict it by a small margin.

    ``ohlcv.load`` rejects the whole series for one such bar, which blocked 13 % of all
    walk-forward snapshots in the 52-ticker audit. Widening the wick to the close would invent a
    fake extreme (and a fake box edge); replacing the bar by its body never goes beyond a price
    that was actually observed. Gaps above ``MAX_ENVELOPE_GAP`` still raise.
    """
    op, hi, lo, cl, vol = extract_ohlcv(dd)
    n = len(cl)
    changed = []
    for i in range(max(0, n - int(window)), n):
        o, h, l, c = op[i], hi[i], lo[i], cl[i]
        if not all(math.isfinite(x) for x in (o, h, l, c)) or min(o, h, l, c) <= 0:
            continue  # load_ohlcv reports it with the exact reason
        top, bottom = max(o, c, l), min(o, c, h)
        gap = max((top - h) / top if h < top else 0.0, (l - bottom) / l if l > bottom else 0.0)
        if gap <= FLOAT_NOISE_TOL:
            continue
        label = str(dates[i])[:10] if dates is not None and i < len(dates) else f"{i + 1}번째"
        if gap > MAX_ENVELOPE_GAP:
            raise ValueError(f"{label} 봉의 고·저가가 시·종가와 {gap:.1%} 어긋남(5% 초과, 데이터 오류)")
        hi[i], lo[i] = max(o, c), min(o, c)
        changed.append({"index": i, "date": label, "gap_pct": round(gap * 100, 3)})
    return {"Open": op, "High": hi, "Low": lo, "Close": cl, "Volume": vol}, changed


def _relation(prev, box, tol):
    if prev is None:
        return "first"
    if box["bottom"] > prev["bottom"] + tol and box["top"] > prev["top"] + tol:
        return "up"
    if box["top"] < prev["top"] - tol and box["bottom"] < prev["bottom"] - tol:
        return "down"
    return "sideways"


def _touches(hi, lo, start, stop, top, bottom, tol, spacing=2):
    top_idx, bottom_idx = [], []
    for i in range(start, stop + 1):
        if hi[i] >= top - tol and (not top_idx or i - top_idx[-1] >= spacing):
            top_idx.append(i)
        if lo[i] <= bottom + tol and (not bottom_idx or i - bottom_idx[-1] >= spacing):
            bottom_idx.append(i)
    return top_idx, bottom_idx


def scan_boxes(op, hi, lo, cl, cfg=None, start=0, atr_origin=0):
    """Run the Darvas-style state machine over ``[start, len)``.

    Returns ``(boxes, forming, atr)`` where ``boxes`` are confirmed boxes in time order
    (the last one may still be active: ``exit`` is ``None``) and ``forming`` describes
    the unconfirmed range after the last exit, if any.
    """
    cfg = cfg or BoxSequenceConfig()
    n = len(cl)
    atr = _atr_series(hi, lo, cl, cfg.atr_period, origin=atr_origin)
    boxes = []
    if n - start < cfg.min_box_bars + cfg.confirm_bars:
        return boxes, None, atr
    mode = "up" if cl[min(n - 1, start + cfg.confirm_bars)] >= cl[start] else "down"
    f_start = start
    f_top, f_ti = hi[start], start
    f_bot, f_bi = lo[start], start
    active = None
    entered_by = None
    for j in range(start + 1, n):
        if active is not None:
            tol = cfg.break_tol_atr * active["atr"]
            if cl[j] > active["top"] + tol or cl[j] < active["bottom"] - tol:
                active["exit"] = "up" if cl[j] > active["top"] else "down"
                active["exit_index"] = j
                active["end_index"] = j - 1
                boxes.append(active)
                entered_by = active["exit"]
                mode = active["exit"]
                active = None
                f_start = j
                f_top, f_ti = hi[j], j
                f_bot, f_bi = lo[j], j
            else:
                active["end_index"] = j
            continue
        if mode == "up":
            if hi[j] > f_top:
                f_top, f_ti = hi[j], j
                f_bot, f_bi = lo[j], j
                f_start = j
            elif lo[j] < f_bot:
                f_bot, f_bi = lo[j], j
        else:
            if lo[j] < f_bot:
                f_bot, f_bi = lo[j], j
                f_top, f_ti = hi[j], j
                f_start = j
            elif hi[j] > f_top:
                f_top, f_ti = hi[j], j
        if f_top - f_bot > cfg.max_forming_atr * atr[j]:
            # A range this wide is a trend leg, not a box: restart from the newest extreme.
            if f_ti >= f_bi:
                mode, f_start = "up", f_ti
                f_bot = min(lo[f_ti:j + 1])
                f_bi = f_ti + lo[f_ti:j + 1].index(f_bot)
            else:
                mode, f_start = "down", f_bi
                f_top = max(hi[f_bi:j + 1])
                f_ti = f_bi + hi[f_bi:j + 1].index(f_top)
        if (j - f_ti >= cfg.confirm_bars and j - f_bi >= cfg.confirm_bars
                and j - f_start + 1 >= cfg.min_box_bars and f_top > f_bot):
            active = {
                "start_index": f_start, "confirm_index": j, "end_index": j,
                "top": f_top, "bottom": f_bot, "top_index": f_ti, "bottom_index": f_bi,
                "atr": atr[j], "entered_by": entered_by, "exit": None, "exit_index": None,
            }
    if active is not None:
        boxes.append(active)
        forming = None
    elif boxes:
        last_exit = boxes[-1]["exit_index"]
        forming = {
            "start_index": f_start, "after_exit": boxes[-1]["exit"], "exit_index": last_exit,
            "top": f_top, "bottom": f_bot, "top_index": f_ti, "bottom_index": f_bi,
            "bars": n - f_start, "top_age": n - 1 - f_ti, "bottom_age": n - 1 - f_bi,
        }
    else:
        forming = {
            "start_index": f_start, "after_exit": None, "exit_index": None,
            "top": f_top, "bottom": f_bot, "top_index": f_ti, "bottom_index": f_bi,
            "bars": n - f_start, "top_age": n - 1 - f_ti, "bottom_age": n - 1 - f_bi,
        }
    previous = None
    for box in boxes:
        tol = cfg.relation_tol_atr * box["atr"]
        box["relation"] = _relation(previous, box, tol)
        top_idx, bottom_idx = _touches(hi, lo, box["start_index"], box["end_index"],
                                       box["top"], box["bottom"], cfg.touch_tol_atr * box["atr"])
        box["top_touches"] = len(top_idx)
        box["bottom_touches"] = len(bottom_idx)
        box["bars"] = box["end_index"] - box["start_index"] + 1
        box["height"] = box["top"] - box["bottom"]
        box["height_atr"] = box["height"] / box["atr"] if box["atr"] > 0 else None
        box["mid"] = (box["top"] + box["bottom"]) / 2
        previous = box
    return boxes, forming, atr


def _staircase(boxes):
    """Describe the last few box-to-box steps in the video's vocabulary."""
    relations = [b["relation"] for b in boxes if b.get("relation") not in (None, "first")]
    last = relations[-1] if relations else None
    before = relations[-2] if len(relations) >= 2 else None
    if last is None:
        return {"key": "unknown", "label": "박스 1개 · 계단 방향 판단 불가", "relations": relations[-4:]}
    if last == "down":
        streak = 0
        for rel in reversed(relations):
            if rel != "down":
                break
            streak += 1
        return {"key": "descending", "label": f"하락 박스 {streak + 1}단 계단", "relations": relations[-4:],
                "streak": streak}
    if last == "up":
        if before == "down" or (before == "sideways" and len(relations) >= 3 and relations[-3] == "down"):
            return {"key": "turn_up", "label": "하락 계단 → 상승 박스 전환", "relations": relations[-4:]}
        return {"key": "ascending", "label": "상승 박스 계단", "relations": relations[-4:]}
    if before == "down":
        return {"key": "base", "label": "하락 후 더 이상 빠지지 않는 박스", "relations": relations[-4:]}
    if before == "up":
        return {"key": "pause_after_rise", "label": "상승 후 횡보 박스", "relations": relations[-4:]}
    return {"key": "sideways", "label": "횡보 박스", "relations": relations[-4:]}


def analyze_box_sequence(dd, config=None):
    """Return the latest confirmed boxes, the live state and data-quality notes.

    ``dd`` must hold completed bars only. Never raises for malformed input.
    """
    cfg = config if isinstance(config, BoxSequenceConfig) else BoxSequenceConfig(**(config or {}))
    needed = cfg.atr_period + cfg.min_box_bars + cfg.confirm_bars
    result = {"version": VERSION, "status": "insufficient_data", "boxes": [], "reference": None,
              "previous": None, "forming": None, "staircase": None, "atr": None,
              "last_index": None, "last_date": None, "data_quality": None,
              "bars_available": None, "bars_required": needed, "restart_after_jump": None,
              "warnings": [], "config": asdict(cfg)}
    dates = dd.get("dates", dd.get("Date")) if isinstance(dd, dict) else None
    window = cfg.lookback + cfg.atr_period
    try:
        clean, body_only = _sanitize_envelope(dd, window, dates)
        (op, hi, lo, cl, vol), quality = load_ohlcv(clean, window=window)
    except (ValueError, TypeError, AttributeError) as exc:
        result.update(status="invalid_data", warnings=["유효하지 않은 OHLCV: " + str(exc)])
        return result
    quality = dict(quality)
    quality["body_only_rows"] = len(body_only)
    quality["body_only"] = body_only[-5:]
    result["data_quality"] = quality
    if body_only:
        shown = ", ".join(f"{b['date']}({b['gap_pct']}%)" for b in body_only[-3:])
        result["warnings"].append(
            f"일봉 {len(body_only)}개의 고·저가가 시·종가와 어긋나(배당 보정 등) 해당 봉은 시·종가(몸통)만 사용: {shown}")
    if dates is not None and len(dates) != len(cl):
        dates = None
    n = len(cl)
    start = max(0, n - cfg.lookback)
    jumps = []
    for i in range(max(1, start), n):
        change = cl[i] / cl[i - 1] - 1 if cl[i - 1] > 0 else 0.0
        if abs(change) >= 0.35:
            jumps.append({"index": i, "date": str(dates[i])[:10] if dates is not None else None,
                          "change_pct": round(change * 100, 2)})
    atr_origin = 0
    if jumps:
        # A one-day move of 35 %+ (split left unadjusted, takeover, limit-up run) is a new regime:
        # boxes are built only from the jump bar onward instead of blocking the plan for 160 bars.
        atr_origin = jumps[-1]["index"]
        start = max(start, atr_origin)
        result["restart_after_jump"] = jumps[-1]
        result["warnings"].append(
            f"{jumps[-1]['date'] or '최근'} 하루 {jumps[-1]['change_pct']}% 종가 급변: 그 이후 일봉만으로 박스를 계산")
    result["bars_available"] = n - start
    if n - start < needed:
        result["warnings"].insert(0, f"박스 산출에 필요한 완료 일봉 부족({n - start}/{needed}개)")
        return result
    boxes, forming, atr = scan_boxes(op, hi, lo, cl, cfg, start=start, atr_origin=atr_origin)

    def stamp(item):
        if item is None:
            return None
        out = dict(item)
        if dates is not None:
            for key in ("start_index", "confirm_index", "end_index", "exit_index", "top_index", "bottom_index"):
                idx = out.get(key)
                if isinstance(idx, int) and 0 <= idx < n:
                    out[key.replace("_index", "_date")] = str(dates[idx])[:10]
        for key in ("top", "bottom", "mid", "height", "atr"):
            if out.get(key) is not None:
                out[key] = round(float(out[key]), 6)
        if out.get("height_atr") is not None:
            out["height_atr"] = round(float(out["height_atr"]), 3)
        return out

    result["atr"] = atr[-1]
    result["last_index"] = n - 1
    result["last_date"] = str(dates[-1])[:10] if dates is not None else None
    result["last_close"] = cl[-1]
    result["prev_close"] = cl[-2] if n >= 2 else None
    result["last_bar"] = {"open": op[-1], "high": hi[-1], "low": lo[-1], "close": cl[-1], "volume": vol[-1]}
    base = [v for v in vol[max(0, n - 21):n - 1] if v > 0]
    base.sort()
    median_volume = base[len(base) // 2] if base else None
    result["volume_ratio"] = (vol[-1] / median_volume) if median_volume else None
    result["recent_closes"] = cl[-6:]
    result["recent_lows"] = lo[-6:]
    result["recent_highs"] = hi[-6:]
    result["closes_for_sigma"] = cl[max(0, n - 260):]
    result["boxes"] = [stamp(b) for b in boxes[-6:]]
    result["reference"] = result["boxes"][-1] if result["boxes"] else None
    result["volume_evidence"] = build_volume_evidence(
        op, hi, lo, cl, vol, result["reference"], dates=dates, origin=atr_origin)
    if boxes and boxes[-1].get("exit_index") is not None:
        k = boxes[-1]["exit_index"]
        # Bars from the exit bar to the latest completed bar: used to count confirmation
        # closes and retests after a breakout without re-reading the raw arrays.
        result["post_exit"] = {"closes": cl[k:][-60:], "lows": lo[k:][-60:], "highs": hi[k:][-60:],
                               "bars": n - k}
    else:
        result["post_exit"] = None
    result["previous"] = result["boxes"][-2] if len(result["boxes"]) >= 2 else None
    result["forming"] = stamp(forming)
    result["staircase"] = _staircase(boxes)
    result["price_jumps"] = jumps
    result["status"] = "valid" if boxes else "no_box"
    if not boxes:
        result["warnings"].append("확정된 박스 없음: 3봉 고점·저점 유지 조건을 아직 충족하지 못함")
    return result
