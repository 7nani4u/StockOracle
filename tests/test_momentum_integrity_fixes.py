import numpy as np
import pandas as pd
import pytest
from market_briefing.momentum_persistence import detect_momentum_persistence, find_surges
from market_briefing.scan_engine import apply_momentum_promotion
from scripts.backtest_momentum_persistence import evaluate_events, summarize, _load


def series():
    return [100.] * 15 + [120., 115., 114., 113.]


def test_exact_twenty_percent_is_inclusive():
    assert find_surges([100., 120.])[0]["pct"] == 20.
    assert detect_momentum_persistence(series())["stage"] == "PASS"


def test_missing_row_never_compresses_trading_days():
    c = [100.] * 10 + [None, 130., 120., 120., 120.]
    assert detect_momentum_persistence(c)["available"] is False


def test_aligned_arrays_and_numpy_supported():
    c = series()
    assert detect_momentum_persistence(c, highs=c[:-1])["available"] is False
    assert detect_momentum_persistence(np.array(c))["stage"] == "PASS"


def test_intraday_third_close_not_confirmed():
    r = detect_momentum_persistence(series(), in_progress=True)
    assert r["stage"] == "WAIT"
    assert r["conditions"]["persistence"]["passed"] is False
    assert r["in_progress_excluded"] is True


def test_unknown_lows_not_claimed_as_strict_observation():
    assert detect_momentum_persistence(series())["strict_low_break"] is None


def test_confirmation_risk_levels_do_not_use_future_bars():
    c = series()
    base = detect_momentum_persistence(c)
    later = detect_momentum_persistence(c + [130.])
    assert base["stop_price"] == later["stop_price"]
    assert base["entry_trigger"] == later["entry_trigger"]


def test_expiry_includes_confirmation_day():
    assert detect_momentum_persistence(series() + [113.] * 4)["stage"] == "PASS"
    assert detect_momentum_persistence(series() + [113.] * 5)["stage"] == "NONE"


def test_collapse_after_confirmation_cannot_promote_ready():
    c = series() + [90.]
    mo = detect_momentum_persistence(c)
    assert mo["stage"] == "PASS"  # historical condition remains true
    assert mo["entry_eligible"] is False
    candidates = [{"ticker": "X", "price": 90., "status": "FAR"}]
    assert apply_momentum_promotion(candidates, {"X": mo}) == {"promoted": 0}
    assert candidates[0]["status"] == "FAR"


@pytest.mark.parametrize("kwargs", [{"hold_days": 0}, {"retain_frac": 2}, {"expiry": 0}, {"surge_min_pct": float("nan")}])
def test_bad_configuration_fails_closed(kwargs):
    assert detect_momentum_persistence(series(), **kwargs)["available"] is False


def test_backtest_next_open_costs_and_short_horizons():
    c = series() + [114.] * 6
    d = pd.DataFrame({"c": c, "o": c})
    d.loc[19, "o"] = 112.
    rows = evaluate_events(d, "US", "X")
    assert len(rows) == 1 and rows[0]["H"] == 5
    assert rows[0]["entry_index"] == 19
    assert rows[0]["net"] == pytest.approx(114 / 112 - 1 - .001)
    assert summarize(rows)["events"] == 1
    assert summarize([])["events"] == 0


def test_backtest_missing_confirmation_rejected():
    c = series() + [114.] * 21
    c[17] = float("nan")
    assert evaluate_events(pd.DataFrame({"c": c, "o": c}), "US", "X") == []


def test_loader_rejects_unsorted_and_duplicate_dates(tmp_path):
    path = tmp_path / "X.csv"
    pd.DataFrame({"Date": ["2026-01-01"] * 10, "Close": [100.] * 10, "Open": [100.] * 10}).to_csv(path, index=False)
    assert _load(path) is None
