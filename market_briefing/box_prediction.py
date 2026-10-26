"""Apply one closed-bar box plan to the prediction response contract."""
from copy import deepcopy
import math
from .box_range import analyze_box_range
from .trend_boxes import analyze_trend_boxes
from .box_exit import attach_box_exit_levels
from .box_tactics import analyze_box_tactics


def _valid_range(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        low, high = map(float, value)
    except (TypeError, ValueError, OverflowError):
        return None
    return [low, high] if math.isfinite(low) and math.isfinite(high) and 0 < low <= high else None


def apply_box_prediction(response, dd, *, in_progress=False):
    original_buy = deepcopy(response.get("buy_price") or {})
    original_risk = deepcopy(response.get("risk_scenarios") or {})
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
    # ── 영상 OlMBqW3FBes 적용: 박스 중간선 자석·속도·OHLCV 압력 프록시 ──
    # 영상의 호가창은 일봉으로 재현할 수 없으므로 압력은 근사치로만 표기한다.
    tactics = analyze_box_tactics(source, box, trend, current, flow=response.get("investor_flow"))
    box["tactics"] = tactics
    box["tactics_version"] = tactics.get("version")
    box["tactics_data_quality"] = tactics.get("data_quality")
    regime = tactics.get("regime") or {}
    regime_key = regime.get("key")
    if regime_key == "breakout_confirmed":
        entry = (tactics.get("plan") or {}).get("entry") or {}
        low, high = entry.get("low"), entry.get("high")
        if low is not None and high is not None:
            status = (tactics.get("plan") or {}).get("status")
            tradeable = (tactics.get("plan") or {}).get("tradeable")
            box["entries"]["main"] = {
                "low": low, "high": high, "eligible": status == "ready" and tradeable is not False,
                "reason": "돌파 확인 후 재시험 매수 · " + str(entry.get("condition") or ""),
                "price_source": "breakout_retest",
            }
            box["entries"]["exploration"] = {
                "low": low, "high": high, "eligible": False,
                "reason": "돌파 확인 구간: 추격 대신 재시험 1회만 허용",
                "price_source": "breakout_retest",
            }
    reversal = tactics.get("reversal") or {}
    box["reversal"] = reversal
    if reversal.get("state") == "confirmed":
        for plan in box["entries"].values():
            plan["eligible"] = False
            plan["reason"] += " · 실행 규칙: 저항 매도 우위·가짜 돌파 하락 시나리오 — 신규 매수 회피"
    midline = (tactics.get("midline") or {}).get("price")
    if midline is not None:
        box["midline"] = midline
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
    # ── 영상 실행 규칙 게이팅: 느린 흐름·약한 중간 자석·반대 압력·가짜 돌파는
    #    가격을 지우지 않고 "진입 불가 사유"만 덧붙인다(무조건 표시 계약 유지).
    regime = tactics.get("regime") or {}
    regime_key = regime.get("key")
    non_tradeable = {"no_data", "no_trade", "mode_breakdown", "false_breakout",
                     "breakout_watch", "breakout_unconfirmed", "range_wait"}
    if regime_key in non_tradeable:
        tag = " · 실행 규칙: " + str(regime.get("label") or "관망") + " — " + str((regime.get("why") or ["조건 미충족"])[0])
        for plan in box["entries"].values():
            if plan.get("eligible"):
                plan["eligible"] = False
                plan["reason"] += tag
    # Final risk gates also govern the raw execution card shown in the UI.
    execution = tactics.get("plan") or {}
    allowed = (box["entries"]["main"].get("eligible")
               if regime_key in {"breakout_confirmed", "trend_retest"}
               else any(p.get("eligible") for p in box["entries"].values()))
    if not allowed and execution.get("status") == "ready":
        execution.update(status="wait", tradeable=False)
        reasons = list(dict.fromkeys(p.get("reason", "진입 보류") for p in box["entries"].values()))
        execution["risk_gate_reasons"] = reasons
        execution["invalidation"] = list(execution.get("invalidation") or []) + reasons
        tactics["summary"] = "진입 보류 · " + " / ".join(reasons)
        if execution.get("pyramid"):
            execution["pyramid"].update(ready=False, enabled=False)
    box["execution_regime"] = regime_key
    # Preserve existing computed prices as explicitly conditional estimates.
    # A missing box must not erase the application's independent pricing model.
    invalidated = box["state"] in {"breakdown_pending", "breakdown_confirmed", "breakout_confirmed"}
    for key, family in (("exploration", "aggressive"), ("main", "recommended")):
        plan = box["entries"][key]
        plan.setdefault("price_source", "observed_box" if box.get("support") else "trend_retest" if plan["low"] is not None else "unavailable")
        reference = _valid_range((original_buy.get(family) or {}).get("range"))
        if plan["low"] is None and reference and current is not None and not invalidated:
            plan.update(low=reference[0], high=reference[1], eligible=False, price_source="legacy_estimate",
                        reason="박스 미확인 · 기존 지표/변동성 계산 참고 가격 · 지지·반등 확인 후 검토")
    for key, family in (("conservative", "conservative"), ("neutral", "balanced"), ("aggressive", "aggressive")):
        plan = box["targets"][key]
        plan["price_source"] = "observed_box" if plan["low"] is not None else "unavailable"
        card = original_risk.get(family) or {}
        reference = _valid_range(card.get("tp_range")) or _valid_range(card.get("target"))
        if plan["low"] is None and reference and current is not None and reference[0] > current and not invalidated:
            plan.update(low=reference[0], high=reference[1], eligible=False, price_source="legacy_estimate",
                        reason="기존 지표/변동성 기반 조건부 매도 추정 · 관측 박스 저항 목표와 구분" + (" · 상단 돌파 확인 후 검토" if key == "aggressive" else ""))
    box["legacy_buy_display"] = all(p.get("price_source") == "legacy_estimate" for p in box["entries"].values())
    box["legacy_sell_display"] = all(p.get("price_source") == "legacy_estimate" for p in box["targets"].values())
    if box["legacy_buy_display"] or box["legacy_sell_display"]:
        box["evidence"].append("박스 미확인 시 기존 계산 가격을 조건부 참고 구간으로 표시; 즉시 진입 신호 아님")
    response["box_range"] = box
    response.setdefault("chart_data", {})["box_range"] = box
    buy = deepcopy(prior)
    buy["box_range"] = box
    for family, key in (("aggressive_bands", "exploration"), ("recommended_bands", "main")):
        plan = box["entries"][key]
        if original_buy.get(family):
            buy[family] = deepcopy(original_buy[family])
            box_bounds = _valid_range([plan["low"], plan["high"]]) if plan.get("price_source") != "legacy_estimate" else None
            for band in buy[family]:
                band["strategy_label"] = "밴드 " + str(band.get("band") or "")
                band["entry_eligible"] = bool(plan["eligible"])
                band["box_entry_range"] = box_bounds
                band["box_entry_note"] = plan["reason"]
                band["box_entry_label"] = "박스 지지 탐색" if key == "exploration" else "지지 반등 본 진입"
                for step in (band.get("steps") or []) if box_bounds else []:
                    step_bounds = _valid_range(step.get("price_range"))
                    overlap = None
                    if box_bounds and step_bounds:
                        low, high = max(box_bounds[0], step_bounds[0]), min(box_bounds[1], step_bounds[1])
                        if low <= high:
                            overlap = [low, high]
                    step["box_entry_overlap"] = overlap
                    step["box_entry_label"] = band["box_entry_label"] if overlap else None
                    step["box_entry_note"] = plan["reason"] if overlap else None
                matched_steps = [step.get("label") for step in band.get("steps") or [] if step.get("box_entry_overlap")]
                band["box_entry_steps"] = matched_steps
                band["box_entry_matches"] = bool(matched_steps)
                if not matched_steps:
                    band["box_entry_range"] = None
                    band["box_entry_note"] = None
                    band["box_entry_label"] = None
                if plan.get("price_source") == "legacy_estimate":
                    band["entry_eligible"] = False
                    band["price_source"] = "legacy_estimate"
                    band["confirm_note"] = plan["reason"]
            continue
        bounds = _valid_range([plan["low"], plan["high"]])
        steps = []
        if bounds:
            low, high = bounds
            width = (high - low) / 5
            for index in range(5):
                step_high = high - width * index
                step_low = low if index == 4 else high - width * (index + 1)
                steps.append({
                    "label": f"TP{index + 1}", "price": (step_low + step_high) / 2,
                    "price_range": [step_low, step_high],
                    "decline_pct_range": [(1-step_high/current)*100, (1-step_low/current)*100] if current else None,
                    "decline_pct": (1-(step_low+step_high)/2/current)*100 if current else None,
                    "basis": plan["reason"],
                    "price_range_basis": "관측 진입 구간 내부를 상단→하단 5단계로 분할; 독립 지지선 또는 매도 목표가 아님",
                    "probability_label": "미산정", "period_label": "확인 후 진입",
                    "allocation_pct": 0, "entry_eligible": plan["eligible"],
                })
        buy[family] = [{
            "band": "A", "range": bounds, "entry_eligible": plan["eligible"],
            "basis": plan["reason"], "steps": steps, "price_source": plan["price_source"],
            "strategy_label": "박스 지지 탐색" if key == "exploration" else "지지 반등 본 진입",
            "strategy_desc": plan["reason"], "confirm_note": plan["reason"],
            "entry_role": "소액 탐색 진입" if key == "exploration" else "반등 확인 후 본 진입",
            "allocation_pct": 0,
            "evidence_label": "관측 박스 구간" if box.get("support") else "추세 재시험 구간" if bounds else "자료 부족",
            "evidence_note": plan["reason"], "evidence_level": "structure" if bounds else "unavailable",
            "range_order_basis": "TP1~TP5는 계산된 진입 가격대의 상단부터 하단까지 분할 표시",
            "evidence_class": "observed_box" if bounds else "insufficient",
        }]
    buy["box_stage_display"] = True
    buy["box_tactics"] = tactics
    for family, key in (("aggressive", "exploration"), ("recommended", "main")):
        plan = box["entries"][key]
        if plan.get("price_source") == "legacy_estimate":
            buy[family] = deepcopy(original_buy[family])
            buy[family].update(price_source="legacy_estimate", entry_eligible=False)
            buy[family]["basis"] = plan["reason"] + " · " + str(buy[family].get("basis") or "")
            continue
        buy[family] = {"range": [plan["low"], plan["high"]] if plan["low"] is not None else None,
                       "pct": None, "basis": plan["reason"], "interpretation": plan["reason"]}
    restrictive = trend_risk or box["state"] in {"breakdown_pending", "breakdown_confirmed", "false_breakout"} or reversal.get("state") == "confirmed"
    prior_action = (prior.get("strategy_rec") or {}).get("action_key")
    action = "wait_breakdown" if restrictive or prior_action == "wait_breakdown" else "box_entry" if any(p["eligible"] for p in box["entries"].values()) else "wait_support"
    buy["strategy_rec"] = {**(prior.get("strategy_rec") or {}), "action_key": action, "rationale": box["evidence"]}
    response["buy_price"] = buy
    risk = response.setdefault("risk_scenarios", {})
    # ── 리스크 3종(보수/중립/공격)은 어떤 종목·어떤 박스 상태에서도 항상 숫자로 표시한다.
    # 기존에는 관측 박스가 없거나(돌파·이탈·미형성) 현재가가 박스 목표를 지났을 때
    # calc_risk가 만든 target/tp_levels/rr/return을 None·[]으로 지워
    # "가격 표시 보류"/"산정 보류"만 남겼다. 이제 원본 가격을 절대 지우지 않고
    # 박스 정보는 overlay注釈(box_target_range/basis)로만 덧붙인다.
    _LABELS = {"conservative": "보수적", "balanced": "중립적", "aggressive": "공격적"}
    _ICONS = {"conservative": "🛡️", "balanced": "⚖️", "aggressive": "🚀"}
    _fallback_market = str(response.get("market") or "KRX").upper()
    if _fallback_market not in ("KRX", "US"):
        _fallback_market = "KRX"

    def _fallback_risk_card(key: str) -> dict:
        if current is None:
            return {
                "label": _LABELS[key], "icon": _ICONS[key],
                "target": None, "tp_range": None, "tp_levels": [], "stop": None,
                "stop_pct": None, "return": None, "rr_ratio": None,
                "target_basis": ["유효한 현재 시세 미확보 — 가격 계산 보류"],
                "target_confidence_pct": None, "expected_value_pct": None,
                "target_eligible": False, "entry_eligible": False,
                "entry_status": "유효한 현재 시세 미확보", "price_source": "unavailable",
            }
        base = current
        atr_fb = base * 0.02
        tgt_hi = base + atr_fb * {"conservative": 1.2, "balanced": 2.5, "aggressive": 4.5}[key]
        tgt_lo = base + atr_fb * {"conservative": 0.8, "balanced": 1.8, "aggressive": 3.2}[key]
        stp_hi = base - atr_fb * 0.8
        stp_lo = base - atr_fb * 1.2
        levels = []
        for i in range(5):
            frac = (i + 0.5) / 5.0
            px = tgt_lo + (tgt_hi - tgt_lo) * frac
            levels.append({
                "price": px, "price_range": [tgt_lo + (tgt_hi - tgt_lo) * i / 5,
                                             tgt_lo + (tgt_hi - tgt_lo) * (i + 1) / 5],
                "basis": "박스·지표 근거 부족 시 ATR 기준 참고 범위",
                "prob_pct": None, "avg_days": None,
            })
        return {
            "label": _LABELS[key], "icon": _ICONS[key],
            "target": [tgt_lo, tgt_hi], "tp_range": [tgt_lo, tgt_hi],
            "tp_levels": levels, "stop": [stp_lo, stp_hi],
            "stop_pct": round((sum([stp_lo, stp_hi]) / 2 / base - 1) * 100, 2),
            "return": round((tgt_hi / base - 1) * 100, 2),
            "rr_ratio": 1.0,
            "target_basis": ["관측 박스·지표 근거 부족 — ATR 기준 참고 범위(조건 확인 후 검토)"],
            "target_confidence_pct": None, "expected_value_pct": None,
            "target_eligible": False, "entry_eligible": False,
            "entry_status": "박스 미확인 · ATR 참고 범위 · 지지·반등 확인 후 검토",
            "price_source": "fallback_estimate",
        }

    for key, target_key in (("conservative", "conservative"), ("balanced", "neutral"), ("aggressive", "aggressive")):
        plan = box["targets"][target_key]
        orig = deepcopy(original_risk.get(key) or {})
        observed = _valid_range([plan.get("low"), plan.get("high")]) if plan.get("price_source") != "legacy_estimate" else None
        if not orig or not _valid_range(orig.get("tp_range") or orig.get("target")):
            if observed and current:
                orig = _fallback_risk_card(key)
                # 관측 박스가 있으면 그 범위를 우선 사용하되 숫자는 반드시 유지
                orig.update(target=list(observed), tp_range=list(observed),
                            target_basis=[plan.get("reason") or "관측 박스 목표"],
                            target_eligible=bool(plan.get("eligible")),
                            entry_eligible=False, entry_status=plan.get("reason") or orig.get("entry_status"),
                            price_source="observed_box")
                orig["tp_levels"] = [{"price": observed[1], "price_range": list(observed),
                                      "basis": plan.get("reason") or "관측 박스 목표"}]
                if current:
                    try:
                        orig["return"] = round((observed[1] / current - 1) * 100, 2)
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
            elif orig and _valid_range(orig.get("target") or orig.get("tp_range")):
                pass  # 부분 원본이라도 유지
            else:
                orig = _fallback_risk_card(key)
                if plan.get("reason"):
                    orig["target_basis"] = [plan.get("reason")] + list(orig.get("target_basis") or [])
                    orig["entry_status"] = plan.get("reason")
            risk[key] = orig
            # 박스 overlay 필드는 항상 기록
            risk[key].setdefault("label", _LABELS[key])
            risk[key].setdefault("icon", _ICONS[key])
            risk[key]["box_target_range"] = observed
            risk[key]["box_target_basis"] = plan.get("reason")
            continue
        # 원본이 있으면: 가격·손절·수익·R/R·TP 5단을 그대로 유지하고 박스만 덧붙인다.
        card = risk.setdefault(key, {})
        card.update(deepcopy(orig))
        card.setdefault("label", _LABELS[key])
        card.setdefault("icon", _ICONS[key])
        card["box_target_range"] = observed
        card["box_target_basis"] = plan.get("reason")
        card["box_target_eligible"] = bool(plan.get("eligible")) if observed else False
        if observed:
            basis = list(card.get("target_basis") or [])
            tag = "박스 관측: " + str(plan.get("reason") or "")
            if tag.strip() and tag not in basis:
                basis.append(tag)
                card["target_basis"] = basis
        else:
            # 박스 미확인·돌파·이탈이어도 가격은 유지. 진입만 보수적으로 보류 표기.
            card["price_source"] = plan.get("price_source") or card.get("price_source") or "no_box_preserved"
            if plan.get("reason"):
                basis = list(card.get("target_basis") or [])
                if plan["reason"] not in basis:
                    card["target_basis"] = [plan["reason"]] + basis
                # 기존 진입 판정을 덮지 않고 박스 사유만 덧붙인다: entry_eligible은 원본 유지가
                # 원칙이나, 박스 이탈 시 무리한 진입 권유를 막기 위해 False로만 좁힌다(True->False만 허용).
                if card.get("entry_eligible") is True and box.get("state") in {
                        "breakdown_pending", "breakdown_confirmed", "breakout_confirmed", "false_breakout"}:
                    card["entry_eligible"] = False
                    card["entry_status"] = plan.get("reason") or card.get("entry_status")
    attach_box_exit_levels(risk, original_risk, box, current)
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
            if not zone:
                continue
            levels[key] = zone["center"]
            levels[key + "_label"] = "관측 박스 " + key if zone else "박스 근거 부족"
            levels[key + "_gap_pct"] = round((levels[key] / current - 1) * 100, 2) if zone and current else None
        if box.get("support"):
            levels["warning_zone"] = [box["support"]["low"], box["support"]["high"]]
        outlook["box_range"] = box
        outlook["box_tactics"] = tactics
        outlook["box_reversal"] = reversal
        outlook["box_execution_status"] = action
        outlook["box_tactics_status"] = regime_key
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
