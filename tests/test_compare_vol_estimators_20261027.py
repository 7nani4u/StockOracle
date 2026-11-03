"""compare_vol_estimators 하네스 검증: 기준 후보가 서비스의 blended_daily_sigma 를 그대로 재현해야 비교가 의미 있다."""
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import compare_vol_estimators as cve  # noqa: E402
from market_briefing import forecast_model as fm  # noqa: E402


def _ohlc(n=400, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    return o, h, l, c


def test_baseline_reproduces_production_blended_sigma():
    o, h, l, c = _ohlc()
    frame = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c})
    atr = cve.wilder_atr(frame)
    for t in (260, 300, 399):
        got = cve.candidate_sigmas(o, h, l, c, atr, t)["baseline"]
        window = list(c[t - cve.WINDOW + 1:t + 1])
        want = fm.blended_daily_sigma(window, float(c[t]), float(atr[t]), True)["sigma"]
        assert got == pytest.approx(want, rel=1e-12)


def test_candidates_are_causal():
    """t 이후 봉을 바꿔도 시점 t 의 후보 σ 는 변하지 않는다."""
    o, h, l, c = _ohlc()
    frame = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c})
    atr = cve.wilder_atr(frame)
    before = cve.candidate_sigmas(o, h, l, c, atr, 300)
    o2, h2, l2, c2 = (a.copy() for a in (o, h, l, c))
    o2[301:] *= 3
    h2[301:] *= 3
    l2[301:] *= 3
    c2[301:] *= 3
    atr2 = cve.wilder_atr(pd.DataFrame({"Open": o2, "High": h2, "Low": l2, "Close": c2}))
    after = cve.candidate_sigmas(o2, h2, l2, c2, atr2, 300)
    assert before == after


def test_split_purges_training_targets_that_cross_the_boundary():
    samples = pd.DataFrame({
        "ticker": ["A"] * 4, "market": ["US"] * 4,
        "date": pd.to_datetime(["2024-05-20", "2024-06-20", "2024-07-10", "2024-08-01"]),
        "end_date": pd.to_datetime(["2024-06-21", "2024-07-22", "2024-08-12", "2024-09-03"]),
        "ret": [0.01, -0.02, 0.03, 0.0], "fwd_daily_sigma": [0.01] * 4, "s::baseline": [0.01] * 4,
    })
    train = samples[samples["end_date"] <= cve.SPLIT]
    test = samples[samples["date"] > cve.SPLIT]
    assert list(train["date"].dt.strftime("%m-%d")) == ["05-20"]
    assert set(train["date"]).isdisjoint(set(test["date"]))
    assert train["end_date"].max() < test["date"].min()
