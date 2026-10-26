"""Final prediction risk gates must dominate a raw breakout candidate."""
from copy import deepcopy

import pytest

from market_briefing import box_prediction
from market_briefing.box_tactics import build_plan


def _breakout(monkeypatch, block):
    box = {
        "state": "breakout_confirmed", "atr": 1.0, "evidence": [],
        "support": {"low": 90.0, "high": 91.0, "center": 90.5},
        "resistance": {"low": 100.0, "high": 101.0, "center": 100.5},
        "entries": {key: {"low": 100.0, "high": 101.0, "eligible": True,
                          "reason": "observed candidate"}
                    for key in ("main", "exploration")},
        "targets": {key: {"low": 105.0, "high": 106.0, "eligible": True,
                          "reason": "observed target"}
                    for key in ("conservative", "neutral", "aggressive")},
    }
    trend = {
        "transition": {"state": "downward_reversal" if block == "trend_risk" else "none"},
        "scales": {"short": {"current": {"direction": "ascending"}}},
        "entry_zone": {"eligible": False}, "evidence": [],
        "scale_conflict": block == "scale_conflict", "atr": 1.0,
    }
    tactics = {
        "version": "test", "regime": {"key": "breakout_confirmed", "tradeable": True},
        "plan": {"status": "ready", "tradeable": True, "side": "long",
                 "entry": {"low": 100.0, "high": 101.0, "mid": 100.5},
                 "targets": [], "pyramid": {"enabled": True, "ready": True}},
        "reversal": {"state": "none"}, "midline": {"price": 95.5},
    }
    monkeypatch.setattr(box_prediction, "analyze_box_range", lambda *_: deepcopy(box))
    monkeypatch.setattr(box_prediction, "analyze_trend_boxes", lambda *_: deepcopy(trend))
    monkeypatch.setattr(box_prediction, "analyze_box_tactics", lambda *_, **kw: deepcopy(tactics))
    response = {"last_close": None if block == "missing_quote" else
               110.0 if block == "outside_band" else 100.5,
               "prediction_outlook": {"decision": {"key": "caution" if block == "caution" else "buy"}}}
    return box_prediction.apply_box_prediction(response, {})


@pytest.mark.parametrize("block", ["caution", "trend_risk", "scale_conflict", "missing_quote", "outside_band"])
def test_breakout_cannot_reenable_blocked_main_entry(monkeypatch, block):
    out = _breakout(monkeypatch, block)
    assert out["box_range"]["entries"]["main"]["eligible"] is False


@pytest.mark.parametrize("block", ["caution", "trend_risk", "scale_conflict", "missing_quote", "outside_band"])
def test_raw_tactics_cannot_advertise_execution_after_final_risk_block(monkeypatch, block):
    out = _breakout(monkeypatch, block)
    plan = out["box_range"]["tactics"]["plan"]
    assert plan["tradeable"] is False
    assert plan["status"] != "ready"
    assert not (plan.get("pyramid") or {}).get("ready")
    assert "진입 보류" in out["box_range"]["tactics"]["summary"]


def test_unblocked_breakout_stays_eligible(monkeypatch):
    out = _breakout(monkeypatch, "none")
    assert out["box_range"]["entries"]["main"]["eligible"] is True
    assert out["box_range"]["tactics"]["plan"]["tradeable"] is True


def test_trend_retest_reports_weighted_reward_separately_from_final_target():
    plan = build_plan({"key": "trend_retest"}, {}, {}, {}, {}, {}, 100.5, 1.0,
                      trend_entry={"low": 100.0, "high": 101.0})
    assert plan["risk_reward"] == pytest.approx(2.0)
    assert plan["rr_to_final"] == pytest.approx(3.0)


@pytest.mark.parametrize("quote", [None, float("nan"), float("inf"), -1, "bad"])
def test_missing_quote_does_not_fabricate_100_based_target(quote):
    out = box_prediction.apply_box_prediction({"last_close": quote}, {})
    for key in ("conservative", "balanced", "aggressive"):
        card = out["risk_scenarios"][key]
        assert card.get("target") is None
        assert card.get("tp_range") is None
        assert card.get("stop") is None
        assert not card.get("tp_levels")
        assert card["entry_eligible"] is False
