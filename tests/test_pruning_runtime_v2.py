from market_briefing import technique_prune as pruning


def rules(status="DROP", validated=False, exclusions=None):
    return {"version": 2, "rules": {"vcp": {"US": {"status": status,
        "validated": validated, "exclusions": exclusions or []}}}}


def test_unvalidated_drop_is_research_only():
    assert pruning.technique_allowed("vcp", "US", rules=rules())[0]
    assert not pruning.technique_evidence("vcp", "US", rules())["publishable"]


def test_validated_drop_blocks():
    assert not pruning.technique_allowed("vcp", "US", rules=rules(validated=True))[0]


def test_malformed_exclusion_cannot_be_wildcard():
    document = rules("PRUNED_KEEP", True, [{}, {"condition": "missing", "value": None},
                                          {"condition": "state", "value": None}])
    assert pruning.technique_allowed("vcp", "US", {}, document)[0]


def test_validated_exact_exclusion_and_explicit_empty(monkeypatch):
    document = rules("PRUNED_KEEP", True, [{"condition": "state", "value": "weak"}])
    assert not pruning.technique_allowed("vcp", "US", {"state": "weak"}, document)[0]
    assert pruning.technique_allowed("vcp", "US", {}, document)[0]
    monkeypatch.setattr(pruning, "load_rules", lambda: rules(validated=True))
    assert pruning.technique_allowed("vcp", "US", {}, {})[0]
    assert pruning.technique_status("vcp", "US", {}) == "UNKNOWN"


def test_report_only_publishes_validated_retained():
    doc = {"version": 2, "rules": {"vcp": {"US": {"status": "KEEP", "validated": True},
                                             "KRX": {"status": "INSUFFICIENT", "validated": False}}}}
    assert len(pruning.pruning_report(doc)["validated_techniques"]) == 1


def test_scan_and_research_share_condition_mapper():
    from market_briefing.scan_engine import momentum_conditions, vcp_conditions
    assert momentum_conditions is pruning.momentum_conditions
    assert vcp_conditions is pruning.vcp_conditions


def test_indicator_gate_removes_aggregate_buy_contribution(monkeypatch):
    from api.index import calc_indicator_signals
    monkeypatch.setattr(pruning, "load_rules", lambda: {})
    data = {"Close": [100.0] * 60, "RSI": [20.0] * 60}
    raw = calc_indicator_signals(data, market="US")
    doc = {"version": 2, "rules": {"indicator_rsi": {"US": {"status": "DROP", "validated": True}}}}
    monkeypatch.setattr(pruning, "load_rules", lambda: doc)
    gated = calc_indicator_signals(data, market="US")
    assert raw["signals"]["rsi"]["signal"] == "매수"
    assert gated["signals"]["rsi"]["signal"] == "관망"
    assert gated["summary"]["weighted_score"] < raw["summary"]["weighted_score"]


def test_strict_only_blocks_missing_and_unvalidated():
    doc = rules("INSUFFICIENT", False)
    doc["strict_validated_only"] = True
    assert not pruning.technique_allowed("vcp", "US", rules=doc)[0]
    assert not pruning.technique_allowed("missing", "KRX", rules=doc)[0]
    doc["rules"]["vcp"]["US"] = {"status": "KEEP", "validated": True}
    assert pruning.technique_allowed("vcp", "US", rules=doc)[0]


def test_context_is_causal_and_volume_denominator_excludes_current():
    c = [100 + i for i in range(60)]
    h = [x+1 for x in c]; l = [x-1 for x in c]
    v = [100.0]*59 + [150.0]
    result = pruning.context_conditions(c,h,l,v)
    assert result == {"trend_context": "UP", "volatility_context": "<2%", "volume_context": ">=1.5x"}
    assert pruning.context_conditions(c[:59],h[:59],l[:59],v[:59]) == {}
    assert pruning.context_conditions(c,h[:-1],l,v) == {}
    c[-1] = float("nan")
    assert pruning.context_conditions(c,h,l,v) == {}


def test_signal_mappers_preserve_shared_context():
    context = {"trend_context": "DOWN", "volume_context": "<0.8x"}
    for mapper in (pruning.hybrid_conditions, pruning.pattern_conditions, pruning.leader_conditions,
                   pruning.momentum_conditions, pruning.vcp_conditions):
        assert mapper({"context_conditions": context})["trend_context"] == "DOWN"


def test_briefing_three_signal_buy_cannot_bypass_strict_gate(monkeypatch):
    from market_briefing import stock_analyzer
    monkeypatch.setattr(pruning, "load_rules", lambda: {"version": 2, "strict_validated_only": True, "rules": {}})
    monkeypatch.setattr(stock_analyzer, "_HYBRID_AVAILABLE", False)
    result = stock_analyzer.analyze_stock({"code": "AAPL", "market": "US",
        "news": [{"impact": "positive"}], "overnight_signal": {"direction": "up"},
        "history": {"pos_52w_pct": 10}})
    assert result["raw_recommendation"] == "strong_buy"
    assert result["recommendation"] == "hold"
    assert result["confidence"] == "low"
    assert result["recommendation_validation"]["publishable"] is False


def test_external_hybrid_enrichment_gates_action_but_raw_research_can_opt_out(monkeypatch):
    from market_briefing import stock_analyzer
    monkeypatch.setattr(pruning, "load_rules", lambda: {"version": 2, "strict_validated_only": True, "rules": {}})
    monkeypatch.setattr(stock_analyzer, "_HYBRID_AVAILABLE", True)
    monkeypatch.setattr(stock_analyzer, "compute_hybrid_score", lambda **kwargs: {"action": "AUTO_YES", "ncs": 80})
    values = [100.0]*60
    gated = stock_analyzer.enrich_with_hybrid(values, values, values, values, market="US")
    assert gated["action"] == "WAIT"
    assert gated["raw_action"] == "AUTO_YES"
    assert stock_analyzer.enrich_with_hybrid(values, apply_pruning=False)["action"] == "AUTO_YES"


def test_evidence_is_compact_and_report_exposes_strict_flag():
    doc = rules("KEEP", True)
    doc["strict_validated_only"] = True
    node = doc["rules"]["vcp"]["US"]
    node.update(reason="validated", min_trades=20, steps=[{"detail": "x"*10000}],
                rejections=[{"detail": "y"*10000}]*3, baseline={"trades": 20},
                retained={"trades": 20})
    evidence = pruning.technique_evidence("vcp", "US", doc)["evidence"]
    assert "steps" not in evidence and "rejections" not in evidence
    assert evidence["steps_count"] == 1
    assert evidence["rejections_count"] == 3
    assert evidence["baseline"] == {"trades": 20}
    assert pruning.pruning_report(doc)["strict_validated_only"] is True


import pytest

@pytest.mark.parametrize("technique,function_name,stage,market", [
    ("leader_reversal", "apply_leader_promotion", "BREAKOUT", "US"),
    ("momentum_persistence", "apply_momentum_promotion", "PASS", "KRX"),
    ("vcp", "apply_vcp_promotion", "PASS", "KRX"),
])
def test_strict_runtime_prevents_direct_promotion_bypass(monkeypatch, technique, function_name, stage, market):
    from market_briefing import scan_engine
    document = {"version": 2, "strict_validated_only": True,
                "rules": {technique: {market: {"status": "INSUFFICIENT", "validated": False}}}}
    monkeypatch.setattr(pruning, "load_rules", lambda: document)
    candidate = {"ticker": "X", "price": 100., "status": "FAR", "sleeve": "CORE"}
    info = {"stage": stage, "entry_trigger": 100., "stop_price": 90., "available": True}
    assert getattr(scan_engine, function_name)([candidate], {"X": info}, market=market) == {"promoted": 0}
    assert candidate["status"] == "FAR"


def test_longterm_route_preserves_financial_ranking_but_marks_unvalidated_research(monkeypatch):
    from api import index
    monkeypatch.setattr(pruning, "load_rules", lambda: {"version": 2, "strict_validated_only": True, "rules": {}})
    raw = {"items": [{"ticker": "TEST", "score": 88, "peter_lynch": {"eligible": True}}]}
    monkeypatch.setattr(index, "fetch_kr_longterm_reco", lambda: raw)
    monkeypatch.setattr(index, "fetch_us_longterm_reco", lambda: raw)
    for market, endpoint in (("KRX", "/api/kr/longterm"), ("US", "/api/us/longterm")):
        response = index.route(endpoint, {})
        item = response["items"][0]
        assert item["score"] == 88
        assert item["actionable"] is False
        assert item["peter_lynch"]["eligible"] is True
        assert item["validation"]["market"] == market
        assert item["validation"]["validated"] is False
        assert response["actionable_count"] == 0
    charm = index._with_financial_validation({"smart_score": 90}, "investment_charm", "US")
    assert charm["smart_score"] == 90 and charm["actionable"] is False
    assert "validation" not in raw["items"][0]


def test_longterm_invalid_provider_payload_fails_without_buy_recommendation():
    from api.index import _validated_longterm_payload
    result = _validated_longterm_payload("invalid", "US")
    assert result["items"] == [] and result["actionable_count"] == 0
    assert result["error"]
