"""Apply one closed-bar box plan to the prediction response contract."""
from copy import deepcopy
import math
from .box_range import analyze_box_range
from .trend_boxes import analyze_trend_boxes


def apply_box_prediction(response, dd, *, in_progress=False):
    source = {key: list(value) for key, value in dd.items()}
    if in_progress:
        source = {key: value[:-1] for key, value in source.items()}
    box = analyze_box_range(source)
    box["indicator_basis"] = "completed_daily_bars"
    box["last_bar_date"] = (source.get("Date") or source.get("dates") or [None])[-1]
    box["video_verified"] = True
    box["video_evidence"] = {
        "url": "https://www.youtube.com/watch?v=NOIOQP-7PPM",
        "review_method": "ko_auto_transcript_and_keyframes",
        "reviewed_on": "2026-10-13",
        "verified_principles": ["descending_flat_ascending_boxes", "segment_transition", "multiple_scales", "failed_upward_transition_risk"],
        "supplementary_rules": ["ATR", "OLS_channel", "numeric_thresholds", "body_wick_zones", "volume", "close_confirmation"],
        "limits": "자동 자막과 주요 시점 도식 확인; 영상은 수치 기준과 몸통·꼬리 선택 규칙을 명시하지 않음",
    }
    trend = analyze_trend_boxes(source)
    box["trend_boxes"] = trend
    current = response.get("last_close")
    try:
        current = float(current)
        if not math.isfinite(current) or current <= 0:
            current = None
    except (TypeError, ValueError, OverflowError):
        current = None
    transition = trend["transition"]
    short_channel = (trend["scales"].get("short") or {}).get("current") or {}
    trend_risk = transition["state"] in {"failed_upward_reversal", "downward_reversal"} or short_channel.get("direction") == "descending"
    if box["state"] in {"no_box", "insufficient_data"} and trend["entry_zone"].get("eligible"):
        box["entries"]["main"] = dict(trend["entry_zone"])
        box["entries"]["main"]["reason"] += " · 추세 회귀 채널의 전환·재시험 보완 기준"
    box["evidence"].extend(trend["evidence"])
    if trend_risk or box["state"] == "false_breakout":
        for plan in box["entries"].values():
            plan["eligible"] = False
            plan["reason"] += " · 상승 전환 실패/하락 박스: 위험 관리 우선"
    if trend.get("scale_conflict"):
        box["entries"]["main"]["eligible"] = False
        box["entries"]["main"]["reason"] += " · 작은 상승 박스와 큰 하락 박스 충돌: 본 진입 대기"
    for plan in box["targets"].values():
        if plan.get("high") is not None and current is not None and plan["high"] <= current:
            plan.update(low=None, high=None, eligible=False, reason=plan["reason"] + " · 현재 시세가 목표를 지나 다음 저항 재평가 필요")
    # A historical rebound does not authorize chasing a distant realtime quote.
    for plan in box["entries"].values():
        if plan.get("eligible") and current is not None and plan.get("high") is not None:
            if current > plan["high"] + (box.get("atr") or trend.get("atr") or 0) or current < plan["low"]:
                plan["eligible"] = False
                plan["reason"] += " · 현재 시세가 진입 구간을 벗어나 재시험 대기"
    if current is None:
        for plan in box["entries"].values():
            plan["eligible"] = False
            plan["reason"] += " · 유효한 현재 시세 미확보"
    prior = response.get("buy_price") or {}
    if ((response.get("prediction_outlook") or {}).get("decision") or {}).get("key") == "caution" or ((prior.get("strategy_rec") or {}).get("action_key")) == "wait_breakdown":
        for plan in box["entries"].values():
            plan["eligible"] = False
            plan["reason"] += " · 기존 종합 위험 판단에 따라 신규 진입 보류"
    response["box_range"] = box
    response.setdefault("chart_data", {})["box_range"] = box
    buy = deepcopy(prior)
    buy["box_range"] = box
    for family, key in (("aggressive_bands", "exploration"), ("recommended_bands", "main")):
        plan = box["entries"][key]
        buy[family] = [{"band": "A", "range": [plan["low"], plan["high"]] if plan["low"] is not None else None,
                        "entry_eligible": plan["eligible"], "basis": plan["reason"], "steps": [],
                        "evidence_class": "observed_box" if plan["low"] is not None else "insufficient"}]
    for family, key in (("aggressive", "exploration"), ("recommended", "main")):
        plan = box["entries"][key]
        buy[family] = {"range": [plan["low"], plan["high"]] if plan["low"] is not None else None,
                       "pct": None, "basis": plan["reason"], "interpretation": plan["reason"]}
    restrictive = trend_risk or box["state"] in {"breakdown_pending", "breakdown_confirmed", "false_breakout"}
    prior_action = (prior.get("strategy_rec") or {}).get("action_key")
    action = "wait_breakdown" if restrictive or prior_action == "wait_breakdown" else "box_entry" if any(p["eligible"] for p in box["entries"].values()) else "wait_support"
    buy["strategy_rec"] = {"action_key": action, "rationale": box["evidence"]}
    response["buy_price"] = buy
    risk = response.setdefault("risk_scenarios", {})
    for key, target_key in (("conservative", "conservative"), ("balanced", "neutral"), ("aggressive", "aggressive")):
        plan = box["targets"][target_key]
        card = risk.setdefault(key, {})
        card.setdefault("label", {"conservative": "보수적", "balanced": "중립적", "aggressive": "공격적"}[key])
        card.setdefault("icon", {"conservative": "🛡️", "balanced": "⚖️", "aggressive": "🚀"}[key])
        target = [plan["low"], plan["high"]] if plan["low"] is not None else None
        card.update(target=target, tp_range=target, tp_levels=[], target_basis=[plan["reason"]],
                    target_confidence_pct=None, breakout_probability_pct=None, expected_value_pct=None,
                    rr_ratio=None, target_eligible=plan["eligible"], entry_eligible=False, entry_status=plan["reason"])
        card["return"] = round((target[1] / current - 1) * 100, 2) if target and current else None
        if target:
            card["tp_levels"] = [{"price": target[1], "price_range": target, "basis": plan["reason"]}]
    pa = response.get("pullback_analysis")
    if isinstance(pa, dict):
        pa["target_main"] = box["targets"]["neutral"]["high"]
        pa["target_ext"] = box["targets"]["aggressive"]["high"]
        for key in ("rr_main", "rr_ext", "target_main_confidence_pct", "target_ext_confidence_pct"):
            pa[key] = None
    outlook = response.get("prediction_outlook")
    if isinstance(outlook, dict):
        levels = outlook.setdefault("levels", {})
        for key in ("support", "resistance"):
            zone = box.get(key)
            levels[key] = zone["center"] if zone else None
            levels[key + "_label"] = "관측 박스 " + key if zone else "박스 근거 부족"
            levels[key + "_gap_pct"] = round((levels[key] / current - 1) * 100, 2) if zone and current else None
        levels["warning_zone"] = [box["support"]["low"], box["support"]["high"]] if box.get("support") else []
        outlook["box_range"] = box
        outlook["box_execution_status"] = action
        outlook["forecast_basis_note"] = "통계적 가격 분포와 영상 기반 박스 전환의 실행 조건은 별도; 예측 분포가 진입 허용을 뜻하지 않음"
        decision = outlook.setdefault("decision", {})
        if action in {"wait_breakdown", "wait_support"}:
            outlook["pre_box_decision"] = deepcopy(decision)
            decision.update(key="caution" if action == "wait_breakdown" else "watch",
                            label="박스 전환 실패·신규 진입 보류" if action == "wait_breakdown" else "박스 진입 확인 대기",
                            tone="negative" if action == "wait_breakdown" else "neutral",
                            summary="확정 봉의 박스 방향·반등·재시험 조건을 먼저 확인합니다.",
                            confidence=None, confidence_interval=None, tp_confidence=None,
                            confidence_note="박스 전환은 규칙 기반 판단이며 검증된 승률이 아님")
    response["box_learning_status"] = "confirmation-aware replay not calibrated; legacy band outcomes excluded"
    return response
