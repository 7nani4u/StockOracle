import numpy as np
import pandas as pd

from scripts import backtest_target_price_accuracy as backtest


def _frame():
    size = 150
    index = pd.date_range("2025-01-02", periods=size, freq="B")
    data = pd.DataFrame({
        "Open": np.full(size, 100.0),
        "High": np.full(size, 101.0),
        "Low": np.full(size, 99.0),
        "Close": np.full(size, 100.0),
        "Volume": np.full(size, 2_000_000.0),
        "ATR": np.full(size, 2.0),
        "RSI": np.full(size, 55.0),
        "MACD": np.full(size, 1.0),
        "Signal_Line": np.full(size, 0.5),
        "ADX": np.full(size, 25.0),
        "DI_Plus": np.full(size, 25.0),
        "DI_Minus": np.full(size, 15.0),
        "MA20": np.full(size, 99.0),
        "MA60": np.full(size, 98.0),
        "MA120": np.full(size, 97.0),
    }, index=index)
    # Both target and stop touch on the first day after the signal.
    data.iloc[141, data.columns.get_loc("High")] = 110.0
    data.iloc[141, data.columns.get_loc("Low")] = 90.0
    return data


def test_walk_forward_uses_conservative_stop_first_order(monkeypatch):
    scenario = {
        "target": [108.0, 108.0],
        "stop": [95.0, 95.0],
        "target_confidence_pct": 70.0,
        "tp_levels": [{"avg_days": 5.0}, {"avg_days": 10.0}],
    }
    monkeypatch.setattr(backtest, "calc_risk", lambda *args, **kwargs: {
        "conservative": scenario, "balanced": scenario, "aggressive": scenario,
    })
    monkeypatch.setattr(backtest, "calc_pullback_analysis", lambda *args, **kwargs: None)

    records = backtest.eval_one_signal("TEST", "US", _frame(), 140)

    assert len(records) == 3
    assert all(record.actionable for record in records)
    assert all(record.stop_first for record in records)
    assert not any(record.hit for record in records)
    assert all(record.realized_return_pct < 0 for record in records)

    summary = backtest.summarize(records)
    conservative = summary["overall"]["US|conservative"]
    assert conservative["hit_rate_pct"] == 0.0
    assert conservative["stop_first_rate_pct"] == 100.0
    assert conservative["brier_score"] is not None


def test_walk_forward_excludes_scenarios_marked_as_entry_hold(monkeypatch):
    scenario = {
        "target": [108.0, 108.0],
        "stop": [95.0, 95.0],
        "target_confidence_pct": 70.0,
        "entry_eligible": False,
        "tp_levels": [{"avg_days": 5.0}, {"avg_days": 10.0}],
    }
    monkeypatch.setattr(backtest, "calc_risk", lambda *args, **kwargs: {
        "conservative": scenario, "balanced": scenario, "aggressive": scenario,
    })
    monkeypatch.setattr(backtest, "calc_pullback_analysis", lambda *args, **kwargs: None)

    records = backtest.eval_one_signal("TEST", "US", _frame(), 140)

    assert len(records) == 3
    assert not any(record.actionable for record in records)
    assert not any(record.strategy_eligible for record in records)


def test_stratification_covers_cheap_and_expensive_price_ranges():
    assert backtest._price_bucket(4_000, "KRX") == "under_5k"
    assert backtest._price_bucket(700_000, "KRX") == "over_500k"
    assert backtest._price_bucket(8, "US") == "under_10"
    assert backtest._price_bucket(800, "US") == "over_500"
