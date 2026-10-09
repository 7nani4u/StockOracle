from market_briefing import probability_calibration as calibration
from market_briefing import technique_prune


def test_rule_changes_cannot_reuse_old_score_calibration(monkeypatch):
    monkeypatch.setattr(technique_prune, "load_rules", lambda: {"version": 2, "strict_validated_only": True, "rules": {}})
    monkeypatch.setattr(calibration, "load_calibration", lambda: {"slope": .8, "intercept_at_half": {"US": .55}})
    result = calibration.calibrate_direction_probability(90, "US")
    assert result["prob_up"] == 55.
    assert result["slope"] == 0.
    assert result["rule_basis_compatible"] is False
    assert result["method"] == "historical_market_prior_rule_set_changed"
