"""Regression tests for the 2026-10-25 data-quality and audit fixes.

Each test pins one concrete failure found while reviewing the video layer:

1. a 1e-11 float-noise high used to disable the whole box engine;
2. a 5% bad low must still be rejected instead of being widened into a fake wick;
3. validation is window-scoped, so an old bad row cannot poison new windows;
4. level pressure only counts bars that actually intersect the level band;
5. the speed baseline excludes the current 3-bar window;
6. the magnet window starts at the box's first touch, not at the last 80 bars;
7. the weak-magnet rule no longer blocks a support rebound with confluence;
8. the down-side (reversal) half and pyramid readiness are gated on evidence.
"""
import math

import pytest

from market_briefing import ohlcv
from market_briefing.box_tactics import (
    AUDIT_SNAPSHOT,
    _checklist,
    analyze_box_tactics,
    build_plan,
    level_pressure,
    midline_analysis,
    reversal_analysis,
    speed_analysis,
)
from market_briefing.box_prediction import apply_box_prediction


def frame(closes, volumes=None, spread=0.5):
    n = len(closes)
    return {
        "opens": list(closes),
        "closes": list(closes),
        "highs": [c + spread for c in closes],
        "lows": [c - spread for c in closes],
        "volumes": list(volumes) if volumes is not None else [1000] * n,
    }


def box(state="range", low=89.5, high=110.5, atr=1.0,
        support_touches=(), resistance_touches=(), first_index=None):
    support = {"low": low, "high": low + 1.0, "center": low + 0.5,
               "touch_indices": list(support_touches)}
    resistance = {"low": high - 1.0, "high": high, "center": high - 0.5,
                  "touch_indices": list(resistance_touches)}
    if first_index is not None:
        support["first_index"] = first_index
        resistance["first_index"] = first_index + 2
    return {"state": state, "atr": atr, "box": {"low": low, "high": high},
            "support": support, "resistance": resistance}


# ── 1-3. OHLCV envelope handling ────────────────────────────────────────────

def test_float_noise_envelope_is_repaired_and_reported():
    dd = frame([100.0] * 40)
    dd["highs"][-1] = dd["closes"][-1] - 1e-11   # high below close
    dd["lows"][-2] = dd["closes"][-2] + 1e-12    # low above close

    tactics = analyze_box_tactics(dd, box(), current=100.0)
    quality = tactics["data_quality"]
    assert quality["repaired_rows"] == 2
    assert quality["quality"] == "repaired"
    assert quality["max_gap_pct"] <= ohlcv.FLOAT_NOISE_TOL
    # closes are never rewritten
    assert quality["policy"].startswith("읽는 창만")


def test_large_envelope_error_is_still_rejected():
    dd = frame([100.0] * 40)
    dd["highs"][-1] = 94.0   # 6% below the close
    with pytest.raises(ValueError):
        ohlcv.load(dd, window=140)
    tactics = analyze_box_tactics(dd, box(), current=100.0)
    assert tactics["regime"]["key"] == "no_data"
    assert tactics["plan"]["status"] == "none"


def test_validation_is_window_scoped_not_full_history():
    closes = [100 + 5 * math.sin(i * math.pi / 5) for i in range(220)]
    dd = frame(closes)
    dd["highs"][5] = dd["closes"][5] * 0.9   # ancient 10% error, outside the 140-bar window

    quality = ohlcv.load(dd, window=140)[1]
    assert quality["validated_rows"] == 140
    assert quality["window_start"] == 80
    assert quality["quality"] == "ok"

    # The same row inside the window is refused.
    dd["highs"][-5] = dd["closes"][-5] * 0.9
    with pytest.raises(ValueError):
        ohlcv.load(dd, window=140)


# ── 4-6. Pressure, speed and magnet windows ─────────────────────────────────

def _row(index, low, high, pressure, participation=1.0, conflict=None):
    return {"index": index, "date": f"d{index}", "low": low, "high": high,
            "pressure": pressure, "participation": participation,
            "conflict": conflict}


def test_level_pressure_ignores_bars_outside_the_level_band():
    rows = [
        _row(0, 80.0, 90.0, -0.90),        # far below the level, heavy sell
        _row(1, 99.5, 100.8, 0.40),
        _row(2, 100.0, 101.0, 0.35),
        _row(3, 104.0, 112.0, -0.80),      # far above the level, heavy sell
        _row(4, 100.2, 100.9, 0.30),
    ]
    result = level_pressure(rows, [0, 1, 2, 3, 4], level=100.0, atr=1.0, side="support")
    assert result["touches"] == 3
    assert result["verdict"] == "buy"
    assert result["sell_share"] == 0.0

    # A level with no intersecting bar is reported as "no sample", not as a verdict.
    empty = level_pressure(rows[:1], [0], level=100.0, atr=1.0, side="support")
    assert empty["touches"] == 0
    assert empty["verdict"] == "unknown"


def test_speed_baseline_excludes_current_window_and_is_reported():
    closes = [100, 100.1, 100, 100.1, 100, 100.1, 100, 100.1, 100.3, 101.0]
    rows = [{"index": i, "close": c} for i, c in enumerate(closes)]
    speed = speed_analysis(rows, atr=1.0)
    assert speed["available"] is True
    assert speed["sample_bars"] == 4          # 60-bar baseline window, not self-referential
    assert speed["state"] == "fast"
    assert speed["ratio"] > 1.5
    assert "제외" in speed["basis"]


def test_midline_window_starts_at_first_box_touch():
    closes = [100.0] * 60 + [90.2] * 110
    volumes = [9000] * 60 + [100] * 110
    dd = frame(closes, volumes, spread=0.4)
    tactics = analyze_box_tactics(dd, box(first_index=60), current=90.2)
    mid = tactics["midline"]
    # 110 bars since the first touch, so the 120-bar cap does not trim it and
    # the old "last 80 bars" default would have started at 90 instead.
    assert mid["window_start_index"] == 60
    assert mid["sample_bars"] == 110
    assert mid["state"] == "weak"


# ── 7. The weak-magnet gate now requires confluence, not the label ──────────

def _rebound_series():
    closes = [90.2] * 76
    opens = [90.2] * 76
    highs = [90.6] * 76
    lows = [89.6] * 76
    volumes = [1000] * 76
    tail = [(89.9, 90.0, 90.3, 89.5, 1500),
            (90.0, 90.5, 90.7, 89.9, 1500),
            (90.5, 90.9, 91.0, 90.4, 1800),
            (90.9, 91.6, 91.7, 90.8, 2000)]
    for o, c, h, l, v in tail:
        opens.append(o)
        closes.append(c)
        highs.append(h)
        lows.append(l)
        volumes.append(v)
    return {"opens": opens, "closes": closes, "highs": highs,
            "lows": lows, "volumes": volumes}


def test_weak_magnet_support_rebound_stays_blocked_on_plan_evidence():
    # The midline metric alone said "allow this" (49.4% weak vs 33.9% strong),
    # but the 7 plans the block suppresses averaged -0.22R. A risk control is
    # not loosened on a proxy metric that disagrees with the decision metric.
    dd = _rebound_series()
    b = box("support_rebound", support_touches=(10, 20, 30, 40, 76, 77))
    tactics = analyze_box_tactics(dd, b, current=90.5)

    assert tactics["midline"]["state"] == "weak"
    assert tactics["pressure"]["support"]["verdict"] == "buy"
    assert tactics["speed"]["state"] != "slow"
    assert tactics["regime"]["key"] == "breakout_watch"
    assert tactics["regime"]["tradeable"] is False
    assert tactics["regime"]["magnet_state"] == "weak"
    assert tactics["regime"]["magnet_evidence"]["verdict"] == "not_reproduced"
    assert any("-0.22R" in reason for reason in tactics["regime"]["why"])
    assert tactics["plan"]["status"] == "none"


def test_weak_magnet_without_confluence_stays_breakout_watch():
    dd = frame([90.2] * 78 + [90.0, 89.8], volumes=[1000] * 78 + [900, 900], spread=0.4)
    b = box("support_rebound", support_touches=(10, 20, 30, 40))
    tactics = analyze_box_tactics(dd, b, current=89.8)
    assert tactics["midline"]["state"] == "weak"
    assert tactics["regime"]["key"] in {"breakout_watch", "no_trade"}
    assert tactics["regime"]["tradeable"] is False
    assert tactics["plan"]["status"] == "none"


# ── 8. Down-side half and pyramid readiness ────────────────────────────────

def test_reversal_confirms_false_breakout_with_sell_pressure():
    latest = {"index": 10, "conflict": "bullish_body_sell_pressure"}
    result = reversal_analysis(
        box("false_breakout"), {"price": 99.5}, {"state": "normal"},
        {"verdict": "balanced"}, {"verdict": "sell", "net": -0.42},
        current=109.6, atr=1.0, latest=latest)
    assert result["state"] == "confirmed"
    assert result["action"] == "avoid_new_long"
    assert result["stop"]["price"] > 110.5
    assert [t["pct"] for t in result["targets"]] == [50, 50]
    assert result["targets"][0]["price"] == 99.5        # midline first
    assert "매도 주문 지시가 아님" in result["note"]


def test_reversal_stays_quiet_without_evidence():
    result = reversal_analysis(
        box("range"), {"price": 99.5}, {"state": "fast"},
        {"verdict": "buy"}, {"verdict": "buy", "net": 0.4},
        current=100.0, atr=1.0, latest={"conflict": None})
    assert result["state"] == "none"
    assert result["action"] == "hold_ok"


def test_pyramid_requires_retest_pressure_and_volume():
    rows = [{"index": 0, "low": 109.4, "close": 110.1}]
    regime = {"key": "breakout_confirmed"}
    ready = build_plan(regime, box("breakout_confirmed"), {"price": 99.5},
                       {"state": "fast"}, {"verdict": "buy"},
                       {"verdict": "buy"}, current=110.1, atr=1.0,
                       rows=rows, volume_context={"ratio": 1.6})
    assert ready["pyramid"]["ready"] is True
    assert all(check["ok"] for check in ready["pyramid"]["checks"])

    thin = build_plan(regime, box("breakout_confirmed"), {"price": 99.5},
                      {"state": "fast"}, {"verdict": "buy"},
                      {"verdict": "buy"}, current=110.1, atr=1.0,
                      rows=rows, volume_context={"ratio": 1.05})
    assert thin["pyramid"]["ready"] is False
    keys = {check["key"]: check["ok"] for check in thin["pyramid"]["checks"]}
    assert keys == {"retest": True, "pressure": True, "volume": False}


# ── 9. Audit snapshot is exposed and self-consistent ────────────────────────

def test_geometry_flags_a_plan_that_cannot_pay_for_its_stop():
    # 5 ATR wide box: passes the "wider than 2.5 ATR" filter, but the first
    # target is only 2.35R away and needs a 46% hit rate against a measured
    # 41.5%. The flag is displayed but must NOT block: the 9 plans this test
    # would have removed averaged +0.65R in the 2026-10-25 run.
    tight = {"state": "range", "atr": 1.0, "box": {"low": 97.0, "high": 102.0},
             "support": {"low": 97.0, "high": 98.0, "center": 97.5, "touch_indices": [10]},
             "resistance": {"low": 101.0, "high": 102.0, "center": 101.5, "touch_indices": [20]}}
    plan = build_plan({"key": "range_support"}, tight, {"price": 99.5, "state": "normal"},
                      {"state": "fast"}, {"verdict": "buy"}, {"verdict": "sell"},
                      current=97.6, atr=1.0)
    geometry = plan["geometry"]
    assert plan["status"] == "ready"
    assert geometry["first_target_rr"] == pytest.approx(2.353, abs=1e-3)
    assert geometry["breakeven_rate"] > 0.45
    assert geometry["measured_rate"] == 0.4151
    assert geometry["ok"] is False
    assert "미달" in geometry["note"]
    assert plan["tradeable"] is True          # informational, not a gate
    assert any("손익분기" in str(item.get("label")) and item["ok"] is False
               for item in _checklist({"key": "range_support"}, {"price": 99.5},
                                      {"state": "fast"}, {"verdict": "buy"},
                                      {"verdict": "sell"}, {}, plan))

    # A wide box clears the same structural test without any new threshold.
    wide = {"state": "range", "atr": 1.0, "box": {"low": 89.5, "high": 110.5},
            "support": {"low": 89.5, "high": 90.5, "center": 90.0, "touch_indices": [10]},
            "resistance": {"low": 109.5, "high": 110.5, "center": 110.0, "touch_indices": [20]}}
    wide_plan = build_plan({"key": "range_support"}, wide, {"price": 100.0, "state": "strong"},
                           {"state": "fast"}, {"verdict": "buy"}, {"verdict": "sell"},
                           current=90.0, atr=1.0)
    assert wide_plan["geometry"]["breakeven_rate"] < 0.4151
    assert wide_plan["geometry"]["ok"] is True
    assert wide_plan["tradeable"] is True


def test_breakout_geometry_reports_breakeven_without_borrowed_rate():
    plan = build_plan({"key": "breakout_confirmed"}, box("breakout_confirmed"),
                      {"price": 100.0}, {"state": "fast"}, {"verdict": "buy"},
                      {"verdict": "buy"}, current=110.0, atr=1.0,
                      rows=[{"index": 0, "low": 109.4, "close": 110.1}],
                      volume_context={"ratio": 1.5})
    geometry = plan["geometry"]
    assert geometry["first_target_rr"] == 1.5
    assert geometry["breakeven_rate"] == pytest.approx(0.5714, abs=1e-3)
    assert geometry["measured_rate"] is None   # no comparable measured rate
    assert geometry["ok"] is None
    assert plan["tradeable"] is (plan["status"] == "ready")
    assert plan["stop_after_t1"] == plan["entry"]["mid"]


def test_audit_snapshot_is_exposed_with_samples_and_limits():
    tactics = analyze_box_tactics(frame([100.0] * 40), box(), current=100.0)
    audit = tactics["audit"]
    assert audit["measured_on"] == "2026-10-25"
    assert audit["overall"]["samples"] == 265
    assert audit["by_magnet"]["strong"]["samples"] + audit["by_magnet"]["normal"]["samples"] \
        + audit["by_magnet"]["weak"]["samples"] == 265
    assert audit["by_magnet"]["strong"]["verdict"] == "not_reproduced"
    assert audit["by_speed"]["verdict"] == "no_separation"
    assert "확률" in audit["limits"]
    # the sample actually reproduced the module constant
    assert AUDIT_SNAPSHOT["overall"]["rate"] == 0.4151


def test_prediction_exposes_reversal_and_data_quality():
    closes = [100 + 4 * math.sin(i * math.pi / 6) for i in range(80)]
    out = apply_box_prediction(
        {"last_close": closes[-1], "prediction_outlook": {"decision": {"key": "watch"}, "levels": {}}},
        frame(closes))
    tactics = out["box_range"]["tactics"]
    assert "reversal" in tactics
    assert out["box_range"]["reversal"] is tactics["reversal"]
    assert out["prediction_outlook"]["box_reversal"] is tactics["reversal"]
    assert out["box_range"]["tactics_data_quality"] == tactics["data_quality"]
