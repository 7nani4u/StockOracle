"""Box execution plan summary for the 7-stage scan table.

The scan table shows five plan fields per stock: 1st target (box middle), 2nd target (box top),
stop, re-entry candidate and expected wait.  All five are read from the same
``box_plan.build_execution_plan`` result the prediction tab renders, so the numbers in the two
places can never disagree.

A stock is listed only when the plan can print all five fields ("complete").  That is a display
guarantee, not a trading edge: the 2026-11-05 walk-forward audit (``docs/box_execution_plan_20261105.md``)
found no proven timing edge for the plan, so completeness is never used to rank stocks above others.

Why box-level targets only
--------------------------
Plans built after a breakout (``measured_50``/``measured_100`` targets) put both targets above the
box top, so "1st target = box middle" and "2nd target = box top" would be false for them.  Those stocks
are counted as excluded instead of being shown under a wrong heading.

Why the targets must sit above the current price
-----------------------------------------------
The plan's box levels are anchored to a *pull-back entry* near the box bottom: ``box_plan._targets``
keeps every level above the planned entry, not above the live quote.  Next to a 현재가 column that
reads as a target the stock has already passed.  2026-11-07 live scan, plans that were otherwise
complete: KRX 13 of 28 and US 37 of 74 had the 1st target (box middle) at or below the current price,
by three routes:

* ``new_base`` (box broke down, a new range is forming): in a down leg the forming range is
  [lowest low, highest high since that low] (``box_sequence.scan_boxes``), so a stock that has
  bounced more than half-way sits above the range middle by construction;
* ``reclaim`` (broken-down box, price back inside or above the old box): the old middle/top are
  levels the price has already crossed;
* ``range_support`` / ``new_box_above`` (failed breakout, upper new box): same effect.

A level at or below the current price is not a target, so a stock is "complete" only when both box
targets are strictly above the price shown in the table.  The prediction tab keeps the full plan
(entry zone first, targets measured from that entry); this is a scan-table listing rule, not a
change of the plan's numbers, ranking or probabilities.

The module is pure Python and never raises: a failure becomes an incomplete summary with a reason.
"""
from __future__ import annotations

import math

from .box_plan import build_execution_plan
from .box_sequence import analyze_box_sequence

SCAN_PLAN_VERSION = "2026-11-07.scan-box-plan-v2"

BOX_ROLES = ("box_mid", "box_top")

# Short captions for the re-entry candidate; the plan's own label stays in ``label``.
_ENTRY_CAPTION = {
    "new_base": "새 바닥 박스 하단",
    "reclaim": "이탈 실패 확인 후 하단",
    "forming_box": "형성 중 박스 하단",
    "range_support": "박스 하단 지지 재확인",
    "new_box_above": "위쪽 새 박스 하단",
}

INCOMPLETE_REASONS = {
    "no_data": "박스 실행 플랜을 만들 데이터가 부족",
    "measured_targets": "돌파 이후 구간이라 목표가 박스 중간·상단이 아님(측정 이동)",
    "target_mid": "1차 목표(박스 중간) 없음",
    "target_top": "2차 목표(박스 상단) 없음",
    "target_mid_passed": "1차 목표(박스 중간)가 현재가 이하 — 이미 지나간 가격",
    "target_top_passed": "2차 목표(박스 상단)가 현재가 이하 — 이미 박스 상단 위",
    "quote": "현재가를 알 수 없어 목표가 현재가 위인지 확인 불가",
    "stop": "손절 기준 없음",
    "reentry": "재진입 후보 가격 없음",
    "wait": "예상 기간 산출 불가",
    "error": "박스 실행 플랜 계산 실패",
}


def _num(value):
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _pct(price, base):
    price, base = _num(price), _num(base)
    if price is None or base is None or base <= 0:
        return None
    return round((price / base - 1) * 100, 2)


def _target(prices, role):
    for item in prices.get("targets") or []:
        if item.get("role") == role and _num(item.get("price")) is not None:
            return item
    return None


def _target_view(item, quote):
    if not item:
        return None
    return {"price": _num(item.get("price")), "pct_from_quote": _pct(item.get("price"), quote),
            "pct_from_entry": item.get("pct_from_entry"), "rr": item.get("rr"),
            "label": item.get("label"), "basis": item.get("basis")}


def _targets_not_above_quote(target1, target2, quote):
    """Missing-reason keys when a box target is not strictly above the current price.

    Only one key is returned: a box top at or below the price already implies the middle is too, so
    the more informative ``target_top_passed`` wins.  Nothing is checked when no target exists (the
    ``target_mid`` / ``target_top`` keys already say so).  A tick-rounded target equal to the quote is
    "not above" — the target would be reached the moment the plan is read.
    """
    if not (target1 or target2):
        return []
    if quote is None:
        return ["quote"]
    if target2 and target2["price"] <= quote:
        return ["target_top_passed"]
    if target1 and target1["price"] <= quote:
        return ["target_mid_passed"]
    return []


def summarize_wait(timing):
    """Two short lines for the table cell plus the long text for the tooltip.

    ``text`` is the minimum wait that does not depend on price (completed bars needed for a new box
    or a confirming close); ``sub`` adds the price-touch estimate.  The touch numbers are the
    driftless volatility model (not a direction forecast).  ``None`` when nothing can be said.
    """
    if not isinstance(timing, dict):
        return None
    touch = timing.get("touch") if isinstance(timing.get("touch"), dict) else {}
    bars = _num(timing.get("confirm_bars"))
    days = touch.get("days_if_reached")
    horizon = touch.get("horizon_days")
    p_horizon = _num((touch.get("p_horizon") or {}).get("mid"))
    bars_text = None
    if timing.get("immediate") is True:
        bars_text = "대기 없음"
    elif bars is not None:
        bars_text = "다음 봉 마감 확인" if bars <= 0 else f"최소 {int(bars)}거래일"
    reach_text = None
    if isinstance(days, (list, tuple)) and len(days) == 2 and None not in days:
        reach_text = (f"가격 도달 {days[0]}거래일째" if days[0] == days[1]
                      else f"가격 도달 {days[0]}~{days[1]}거래일째")
    prob_text = None
    if p_horizon is not None and horizon:
        prob_text = f"{int(horizon)}거래일 내 도달 가능성 약 {int(p_horizon)}%"
    main = bars_text or reach_text
    if not main:
        return None
    rest = [part for part in ((reach_text if bars_text else None), prob_text) if part]
    return {
        "text": main,
        "sub": " · ".join(rest) or None,
        "min_bars": None if bars is None else int(bars),
        "immediate": timing.get("immediate") is True,
        "days_if_reached": list(days) if reach_text else None,
        "touch_pct": None if p_horizon is None else int(p_horizon),
        "horizon_days": None if not horizon else int(horizon),
        "detail": (str(timing.get("estimate_label") or "").strip()
                   + " 방향을 예측한 값이 아니라 변동성만으로 본 추정입니다.").strip(),
        # Table cells: the same engine numbers, split into two short columns (예상 기간 · 도달 확률).
        "period": _period_view(timing, bars, days),
        "probability": _probability_view(touch, horizon),
    }


def _period_view(timing, bars, days):
    """Short text for the 예상 기간 column, from numbers the engine already computed.

    ``min~max일`` is the first-touch day window (25~75th percentile of the driftless model); when no
    window exists the only real values left are the engine's own minimum wait (completed bars needed)
    or "대기 없음".  Nothing is invented: no window, no range.  Days are trading days.
    """
    if isinstance(days, (list, tuple)) and len(days) == 2 and None not in days:
        low, high = int(days[0]), int(days[1])
        return {"text": f"{low}일" if low == high else f"{low}~{high}일", "kind": "window",
                "min_days": low, "max_days": high}
    if timing.get("immediate") is True:
        return {"text": "대기 없음", "kind": "immediate", "min_days": 0, "max_days": 0}
    if bars is not None:
        if bars <= 0:
            return {"text": "다음 봉 마감", "kind": "next_bar", "min_days": 0, "max_days": None}
        return {"text": f"최소 {int(bars)}일", "kind": "min_bars", "min_days": int(bars), "max_days": None}
    return None


def _probability_view(touch, horizon):
    """``low~high%`` (one decimal) for the 도달 확률 column, or ``None`` when the model gave none.

    The range is the engine's own volatility-error band around the touch probability
    (``touch_probability_range``); a collapsed band (low == high, e.g. certain touch) shows the single
    value.  ``level`` is the price the probability refers to; ``level_label`` (set by
    ``summarize_plan``) names it.  It is the touch price of the plan's timing, not the 1st target.
    """
    if not touch:
        return None
    band = touch.get("p_horizon_exact") or touch.get("p_horizon")
    if not isinstance(band, dict):
        return None
    mid = _num(band.get("mid"))
    low = _num(band.get("low")) if _num(band.get("low")) is not None else mid
    high = _num(band.get("high")) if _num(band.get("high")) is not None else mid
    if low is None or high is None:
        return None
    low, high = min(low, high), max(low, high)
    text = f"{low:.1f}%" if round(low, 1) == round(high, 1) else f"{low:.1f}~{high:.1f}%"
    return {"text": text, "low": low, "mid": mid, "high": high,
            "horizon_days": None if not horizon else int(horizon), "level": _num(touch.get("level")),
            "level_label": None}


def _level_label(plan, level):
    """Name the price the touch probability refers to (entry-zone top, a trigger price …)."""
    level = _num(level)
    if level is None:
        return None
    prices = plan.get("prices") or {}
    candidates = [((prices.get("entry") or {}).get("high"), "재진입 후보 구간 상단"),
                  ((prices.get("trigger") or {}).get("price"), (prices.get("trigger") or {}).get("label")),
                  (((plan.get("alternative") or {}).get("trigger") or {}).get("price"),
                   ((plan.get("alternative") or {}).get("trigger") or {}).get("label"))]
    for price, label in candidates:
        price = _num(price)
        if price is not None and label and abs(price - level) <= max(abs(level) * 1e-9, 1e-12):
            return label
    return "기준가"


def _entry_view(entry, quote):
    if not isinstance(entry, dict):
        return None
    low, high = _num(entry.get("low")), _num(entry.get("high"))
    if low is None or high is None:
        return None
    kind = entry.get("kind")
    label = str(entry.get("label") or "")
    caption = _ENTRY_CAPTION.get(kind) or ("박스 하단 지지" if not kind else label.split("·")[-1].strip())
    return {"low": low, "high": high, "pct_from_quote": _pct(high, quote), "label": label,
            "kind": kind, "caption": caption, "conditional": bool(entry.get("conditional")),
            "condition": entry.get("condition") or None, "basis": entry.get("basis") or None}


def summarize_plan(plan, quote):
    """Reduce one execution plan to the five scan fields and say whether all five exist.

    "Complete" also requires both box targets to be above ``quote`` (see the module docstring).
    """
    plan = plan if isinstance(plan, dict) else {}
    verdict = plan.get("verdict") or {}
    prices = plan.get("prices") or {}
    box = plan.get("box") or None
    out = {
        "version": SCAN_PLAN_VERSION, "complete": False, "missing": [], "reason": "",
        "verdict": {"key": verdict.get("key"), "icon": verdict.get("icon"), "label": verdict.get("label"),
                    "headline": verdict.get("headline")},
        "stage": (plan.get("stage") or {}).get("label"),
        "strategy": (plan.get("strategy") or {}).get("label"),
        "quote": quote,
        "box": ({"top": box.get("top"), "bottom": box.get("bottom"), "mid": box.get("mid"),
                 "position_label": box.get("position_label"), "forming": bool(box.get("forming"))}
                if isinstance(box, dict) else None),
        "target_basis": None,
        "target1": None, "target2": None, "stop": None, "reentry": None, "wait": None,
    }
    if not plan.get("available") or verdict.get("key") in (None, "unavailable"):
        out["missing"] = ["no_data"]
        out["reason"] = INCOMPLETE_REASONS["no_data"] + (f" — {verdict.get('headline')}" if verdict.get("headline") else "")
        return out
    roles = [t.get("role") for t in prices.get("targets") or []]
    out["target_basis"] = ("measured" if any(str(r).startswith("measured") for r in roles if r)
                           else "box" if any(r in BOX_ROLES for r in roles) else None)
    out["target1"] = _target_view(_target(prices, "box_mid"), quote)
    out["target2"] = _target_view(_target(prices, "box_top"), quote)
    stop = prices.get("stop") if _num((prices.get("stop") or {}).get("price")) is not None else prices.get("holder_stop")
    if _num((stop or {}).get("price")) is not None:
        out["stop"] = {"price": _num(stop["price"]), "pct_from_quote": _pct(stop["price"], quote),
                       "pct_from_entry": stop.get("pct_from_entry"), "basis": stop.get("basis")}
    out["reentry"] = _entry_view(prices.get("entry"), quote)
    out["wait"] = summarize_wait(plan.get("timing"))
    if out["wait"] and out["wait"].get("probability"):
        prob = out["wait"]["probability"]
        prob["level_label"] = _level_label(plan, prob.get("level"))
    missing = []
    if out["target_basis"] == "measured":
        missing.append("measured_targets")
    else:
        if out["target1"] is None:
            missing.append("target_mid")
        if out["target2"] is None:
            missing.append("target_top")
        missing.extend(_targets_not_above_quote(out["target1"], out["target2"], _num(quote)))
    if out["stop"] is None:
        missing.append("stop")
    if out["reentry"] is None:
        missing.append("reentry")
    if out["wait"] is None:
        missing.append("wait")
    out["missing"] = missing
    out["complete"] = not missing
    out["reason"] = "" if not missing else " · ".join(INCOMPLETE_REASONS[m] for m in missing)
    return out


def _incomplete(key, detail=""):
    return {"version": SCAN_PLAN_VERSION, "complete": False, "missing": [key],
            "reason": INCOMPLETE_REASONS[key] + (f" ({detail})" if detail else ""),
            "verdict": {}, "target1": None, "target2": None, "stop": None, "reentry": None,
            "wait": None, "box": None, "target_basis": None}


def build_scan_box_plan(opens, highs, lows, closes, volumes, *, dates=None, market="KRX",
                        in_progress=False, now=None, params=None):
    """Execution-plan summary for one stock from its daily OHLCV (oldest → newest).

    ``in_progress`` drops the last bar (today's unfinished session) from the box analysis and uses its
    close as the live quote, exactly as ``box_prediction.apply_box_prediction`` does.  Never raises.
    """
    try:
        if opens is None or not closes:
            return _incomplete("no_data", "시가 또는 종가 없음")
        n = len(closes)
        cut = n - 1 if in_progress else n
        if cut < 1:
            return _incomplete("no_data", "완료된 일봉 없음")
        quote = _num(closes[-1])
        data = {"Open": list(opens)[:cut], "High": list(highs)[:cut], "Low": list(lows)[:cut],
                "Close": list(closes)[:cut], "Volume": list(volumes)[:cut]}
        if dates is not None and len(dates) == n:
            data["dates"] = list(dates)[:cut]
        seq = analyze_box_sequence(data)
        plan = build_execution_plan(seq, current=quote, market=market, now=now, params=params)
        return summarize_plan(plan, quote)
    except Exception as exc:  # the scan must survive one bad ticker
        return _incomplete("error", type(exc).__name__)


def attach_box_plans(cands, snap_map, *, market="KRX", dates_map=None, in_progress=None, now=None):
    """Attach ``box_plan`` to every candidate dict and return the completeness statistics.

    ``in_progress`` is an optional ``ticker -> bool`` callable (is the last bar still forming).
    """
    stats = {"version": SCAN_PLAN_VERSION, "evaluated": 0, "complete": 0, "incomplete": 0, "excluded": {}}
    for cd in cands or []:
        snap = (snap_map or {}).get(cd.get("ticker"))
        if snap is None:
            cd["box_plan"] = _incomplete("no_data", "스냅샷 없음")
        else:
            try:
                running = bool(in_progress(cd["ticker"])) if in_progress else False
            except Exception:
                running = False
            cd["box_plan"] = build_scan_box_plan(
                snap.opens, snap.highs, snap.lows, snap.closes, snap.volumes,
                dates=(dates_map or {}).get(cd.get("ticker")), market=market,
                in_progress=running, now=now)
        stats["evaluated"] += 1
        if cd["box_plan"]["complete"]:
            stats["complete"] += 1
        else:
            stats["incomplete"] += 1
            for key in cd["box_plan"].get("missing") or ["no_data"]:
                stats["excluded"][key] = stats["excluded"].get(key, 0) + 1
    return stats
