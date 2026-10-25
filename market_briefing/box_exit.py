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
        card = risk.setdefault(key, {})
        plan = (box.get("targets") or {}).get(strategy) or {}
        observed = _bounds([plan.get("low"), plan.get("high")]) if plan.get("price_source") != "legacy_estimate" else None
        old = (original or {}).get(key) or {}
        old_levels = old.get("tp_levels") or []
        valid_old = len(old_levels) == 5 and all(_bounds(lv.get("price_range") or [lv.get("price"), lv.get("price")]) for lv in old_levels)
        # 원본 5단이 있으면 박스 상태와 무관하게 항상 유지한다(기존에는 돌파·이탈 때 버렸다).
        # 박스 정보는 덧붙임 주석·범위로만 추가한다.
        if valid_old:
            preserved = deepcopy(old)
            # 기존 카드는 box_prediction에서 이미 원본을 유지하고 있으므로,
            # 여기서는 TP를 덮지 않고 박스 주석 필드만 보존·병합한다.
            preserved["box_target_range"] = observed
            preserved["box_target_basis"] = plan.get("reason")
            preserved["price_source"] = card.get("price_source") or preserved.get("price_source") or "existing_tp_model"
            # entry_eligible/expected_value는 box_prediction이 정한 값을 우선한다(있을 때만).
            if card.get("entry_status"):
                preserved["entry_status"] = card.get("entry_status")
            if "entry_eligible" in card:
                preserved["entry_eligible"] = card.get("entry_eligible")
            # card에 이미 있는 박스 주석이 있으면 유지
            for extra in ("box_exit_unmatched", "box_exit_annotations"):
                if extra in card and extra not in preserved:
                    preserved[extra] = card[extra]
            risk[key] = card = preserved
        elif observed and not card.get("tp_levels"):
            low, high = observed
            card["tp_levels"] = [{"price": low+(high-low)*(i+.5)/5,
                                    "price_range": [low+(high-low)*i/5, low+(high-low)*(i+1)/5],
                                    "basis": "박스 목표 구간 내부 분할; 독립 저항 5개 아님",
                                    "prob_pct": None, "avg_days": None} for i in range(5)]
        # 그래도 TP가 비어 있으면 원본 target/tp_range에서 5단을 복원한다.
        # 어떤 종목에서도 TP표가 비어 "산정 보류"만 남지 않게 하는 최후 보장이다.
        if not card.get("tp_levels"):
            fallback_range = (_bounds(card.get("tp_range")) or _bounds(card.get("target"))
                              or _bounds((old.get("tp_range") or old.get("target") or [None, None]))
                              or (observed if observed else None))
            if fallback_range and current:
                low, high = fallback_range
                if high > current and low > 0:
                    card["tp_levels"] = [{"price": low+(high-low)*(i+.5)/5,
                                            "price_range": [low+(high-low)*i/5, low+(high-low)*(i+1)/5],
                                            "basis": "기존 목표 범위 유지 분할; 박스 미확인 시에도 표시",
                                            "prob_pct": None, "avg_days": None} for i in range(5)]
                    if not card.get("tp_range"):
                        card["tp_range"] = list(fallback_range)
                    if not card.get("target"):
                        card["target"] = list(fallback_range)
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
        # ── 영상 OlMBqW3FBes: 박스 중간선 50% + 반대편 50% 분할 청산 ──
        # 기존 TP1~TP5 행은 절대 수정하지 않고, 중립 카드에 별도 청산 메모만 붙인다.
        midline = _bounds([box.get("midline"), box.get("midline")])
        range_like = box.get("state") in {"range", "support_rebound", "resistance_rejection"}
        if key == "balanced" and midline and current and range_like and midline[0] > current:
            far = observed or _bounds([plan.get("low"), plan.get("high")])
            card["box_midline_exit"] = {
                "price": midline[0],
                "pct": 50,
                "label": "박스 중간선 1차 익절(50%)",
                "reason": "영상 사례: 중간 1/3 거래가 많을수록 중간선이 자석처럼 작동한다. 남은 50%는 반대편 가장자리에서 청산.",
                "source": "video_OlMBqW3FBes",
                "eligible": bool(far and far[1] > midline[0]),
            }
            card["box_exit_split"] = [
                {"low": midline[0], "high": midline[0], "pct": 50,
                 "label": "중간선 50%", "reason": "자석 회귀 1차 목표"},
                {"low": (far or [None, None])[0], "high": (far or [None, None])[1], "pct": 50,
                 "label": "반대편 50%", "reason": "박스 반대편 저항/지지에서 잔여 청산"},
            ]
    box["box_tp_display"] = True
