"""Measured rectangle move and overlays on existing TP ranges.

The height projection is a conditional TA supplement, not a video rule,
observed resistance, return guarantee, or calibrated probability.
"""
from copy import deepcopy
import math


def _bounds(value):
    try:
        if value is None or len(value) != 2:
            return None
        lo, hi = map(float, value)
        return [lo, hi] if math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi else None
    except (ValueError, TypeError, OverflowError):
        return None


def measured_box_move(box):
    support, resistance = box.get("support"), box.get("resistance")
    inactive = box.get("state") in {"breakdown_pending", "breakdown_confirmed", "false_breakout"}
    if not support or not resistance or inactive:
        return None
    s = _bounds([support["low"], support["high"]])
    r = _bounds([resistance["low"], resistance["high"]])
    if not s or not r or r[0] <= s[1]:
        return None
    height = float(resistance["center"]) - float(support["center"])
    if not math.isfinite(height) or height <= 0:
        return None
    return {"low": 2*r[0]-s[1], "high": 2*r[1]-s[0],
            "center": float(resistance["center"])+height, "height": height,
            "eligible": box.get("state") == "breakout_confirmed",
            "source": "measured_move_estimate",
            "reason": "박스 높이 측정 목표 = 저항 + (저항−지지); 상단 돌파 확인 후 조건부 참고, 상승 보장 아님"}


def attach_box_exit_levels(risk, original, box, current):
    measured = measured_box_move(box)
    box["measured_move"] = measured
    invalidated = box.get("state") in {"breakdown_pending", "breakdown_confirmed", "breakout_confirmed"}
    for key, strategy in (("conservative", "conservative"), ("balanced", "neutral"), ("aggressive", "aggressive")):
        card = risk[key]
        plan = box["targets"][strategy]
        observed = _bounds([plan.get("low"), plan.get("high")]) if plan.get("price_source") != "legacy_estimate" else None
        old = original.get(key) or {}
        old_levels = old.get("tp_levels") or []
        valid_old = len(old_levels) == 5 and all(_bounds(lv.get("price_range") or [lv.get("price"),lv.get("price")]) for lv in old_levels)
        if valid_old and not invalidated:
            preserved = deepcopy(old)
            preserved.update(entry_eligible=False, entry_status=card.get("entry_status"), expected_value_pct=None,
                             box_target_range=observed, box_target_basis=plan["reason"],
                             price_source=card.get("price_source", "existing_tp_model"))
            risk[key] = card = preserved
        elif observed:
            low, high = observed
            card["tp_levels"] = [{"price": low+(high-low)*(i+.5)/5,
                                    "price_range": [low+(high-low)*i/5, low+(high-low)*(i+1)/5],
                                    "basis": "박스 목표 구간 내부 분할; 독립 저항 5개 아님",
                                    "prob_pct": None, "avg_days": None} for i in range(5)]
        annotations = []
        if observed:
            annotations.append({"low": observed[0], "high": observed[1],
                                "label": {"conservative":"박스 저항 전 청산", "balanced":"박스 상단 저항 청산", "aggressive":"돌파 후 다음 저항 청산"}[key],
                                "reason": plan["reason"], "eligible": plan.get("eligible",False)})
        if key == "aggressive" and measured and current and measured["high"] > current:
            annotations.append(dict(measured, label="박스 높이 조건부 목표"))
        if not card.get("tp_levels") and key == "aggressive" and measured and current and measured["high"] > current:
            low, high = measured["low"], measured["high"]
            card["tp_levels"] = [{"price":low+(high-low)*(i+.5)/5,
                                   "price_range":[low+(high-low)*i/5,low+(high-low)*(i+1)/5],
                                   "basis":"조건부 측정 목표 내부 분할; 관측 저항 아님",
                                   "prob_pct":None,"avg_days":None} for i in range(5)]
        unmatched = []
        for annotation in annotations:
            matched = []
            for i, lv in enumerate(card.get("tp_levels") or []):
                bounds = _bounds(lv.get("price_range") or [lv.get("price"),lv.get("price")])
                if not bounds:
                    continue
                low, high = max(bounds[0],annotation["low"]), min(bounds[1],annotation["high"])
                if low <= high:
                    lv.setdefault("box_exit_annotations", []).append({**annotation,"full_range":[annotation["low"],annotation["high"]],"low":low,"high":high})
                    matched.append(f"TP{i+1}")
            if not matched:
                unmatched.append(annotation)
        card["box_exit_unmatched"] = unmatched
    box["box_tp_display"] = True
