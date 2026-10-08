"""매수 구간 감사 스크립트의 불변식 판정·요약·합성 데이터 경로 테스트 (네트워크 없음)."""

import numpy as np
import pandas as pd

from api import index as ix
from scripts import audit_entry_bands as audit


def _band(name, low, high, prices, probabilities=None, available=True):
    probabilities = probabilities or [90.0, 80.0, 70.0, 60.0, 50.0][:len(prices)]
    steps = []
    for index, (price, probability) in enumerate(zip(prices, probabilities)):
        steps.append({"price": price, "price_range": [price - 0.4, price + 0.4], "reach_probability_pct": probability,
                      "days_min": 1 + index, "days_max": 2 + index})
    return {"band": name, "is_available": available, "range": [low, high], "steps": steps}


def _good_family():
    return [
        _band("A", 108.0, 112.0, [112.0, 111.0, 110.0, 109.0, 108.0]),
        _band("B", 102.0, 106.0, [106.0, 105.0, 104.0, 103.0, 102.0]),
        _band("C", 96.0, 100.0, [100.0, 99.0, 98.0, 97.0, 96.0]),
    ]


def test_check_family_accepts_a_valid_family():
    assert audit.check_family(_good_family()) == []


def test_check_family_names_each_broken_invariant():
    duplicated = _good_family()
    duplicated[0]["steps"][4]["price"] = duplicated[0]["steps"][3]["price"]
    assert "dup_step_price" in audit.check_family(duplicated)

    outside = _good_family()
    outside[1]["steps"][4]["price"] = 90.0
    assert "step_outside_band" in audit.check_family(outside)

    unordered = _good_family()
    unordered[2]["range"] = [102.5, 107.0]
    assert "band_order" in audit.check_family(unordered)

    withheld = _good_family()
    withheld[0] = _band("A", 108.0, 112.0, [], available=False)
    assert "withheld" in audit.check_family(withheld)

    short = _good_family()
    short[1]["steps"] = short[1]["steps"][:3]
    assert "step_count" in audit.check_family(short)

    missing = _good_family()
    missing[0]["steps"][2]["reach_probability_pct"] = None
    assert "prob_missing" in audit.check_family(missing)

    bad_days = _good_family()
    bad_days[0]["steps"][0]["days_max"] = 45
    assert "days_invalid" in audit.check_family(bad_days)

    assert "band_count" in audit.check_family(_good_family()[:2])


def test_summarize_reports_calibration_and_period_coverage():
    rng = np.random.default_rng(3)
    step_rows, sample_rows = [], []
    for sample in range(300):
        date = f"2025-{1 + sample % 12:02d}-{1 + sample % 27:02d}"
        sample_rows.append({"ticker": "T", "market": "US", "date": f"{date}#{sample}", "bars": 252, "context": "x", "atr_pct": 2.0,
                            "agg_problems": "", "rec_problems": "", "separated": True, "relaxed": False})
        for family in ("agg", "rec"):
            for band in "ABC":
                for step in range(1, 6):
                    p = float(rng.uniform(5, 95))
                    touched = int(rng.uniform() < p / 100.0)      # 확률대로 일어나는 이상적인 표본
                    step_rows.append({"ticker": "T", "market": "US", "date": f"{date}#{sample}", "family": family, "band": band,
                                      "step": step, "evidence": "structure", "p": p, "p_low": p - 5, "p_high": p + 5,
                                      "reference_p": p, "days_min": 3, "days_max": 12, "complete": True, "touched": touched,
                                      "first_day": int(rng.integers(1, 30)) if touched else None})
    summary = audit.summarize(step_rows, sample_rows)

    assert summary["samples"] == 300 and summary["steps_per_sample"] == {30: 300}
    assert summary["invariants"]["samples_with_agg_problem"] == 0 and summary["invariants"]["samples_not_separated"] == 0
    reach = summary["reach_probability"]
    assert reach["brier"] < reach["brier_constant"]
    assert reach["auc"] > 0.7
    # 이상적으로 보정된 표본이므로 신뢰도 곡선이 대각선에 붙는다
    assert all(abs(row["predicted_pct"] - row["realized_pct"]) < 8 for row in reach["reliability"] if row["n"] > 500)
    assert 0.0 <= summary["expected_period"]["inside_window"] <= 1.0
    assert set(summary["by_band"]) == {f"{f}{b}" for f in ("agg", "rec") for b in "ABC"}


def _synthetic_prices(n=420, seed=4):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(100_000, 400_000, n).astype(float)
    index = pd.bdate_range(end="2026-09-30", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=index)


def test_synthetic_walk_forward_returns_full_valid_bands_for_every_sample():
    prices = _synthetic_prices()
    steps, samples = audit.evaluate_ticker(ix, "SYNTH", prices, stride=40)

    assert samples, "평가 시점이 하나도 없습니다"
    assert all(row["agg_problems"] == "" and row["rec_problems"] == "" for row in samples)
    assert all(row["separated"] for row in samples)
    assert len(steps) == 30 * len(samples)
    assert all(row["p"] is not None and 0.0 <= row["p"] <= 100.0 for row in steps)
    complete = [row for row in steps if row["complete"]]
    assert complete and all(row["touched"] in (0, 1) for row in complete)
    summary = audit.summarize(steps, samples)
    assert summary["reach_probability"]["brier"] < 0.30


def test_short_history_walk_forward_still_returns_both_families():
    prices = _synthetic_prices(n=120, seed=8)
    steps, samples = audit.evaluate_ticker(ix, "NEWLIST", prices, stride=15, min_bars=25)

    assert samples and min(row["bars"] for row in samples) <= 40
    assert all(row["agg_problems"] == "" and row["rec_problems"] == "" for row in samples)
    assert len(steps) == 30 * len(samples)
