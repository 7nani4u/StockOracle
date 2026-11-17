"""순위 감사 도구(scripts/audit_scan_ranking.py)가 미래 정보를 쓰지 않고 통계를 바르게 내는지 고정한다.

감사 결과를 믿으려면 도구 자체가 먼저 검증돼야 한다: 특징은 t 시점까지만 보고, 목표값은 t+1 시가 진입을
따르며, 장벽 판정은 같은 봉에서 손절을 우선하고, IC 는 날짜별 교차단면에서 계산돼야 한다.
"""

import importlib.util
import math
import os

import numpy as np
import pandas as pd
import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "audit_scan_ranking.py")
_spec = importlib.util.spec_from_file_location("audit_scan_ranking", _PATH)
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def _frame(n=320, seed=3, drift=0.0004):
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(8e5, 3e6, n).astype(float)
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"Date": dates, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol})


def test_extra_features_use_only_data_up_to_each_row():
    df = _frame()
    bench = _frame(seed=9)
    base = A.extra_features(df, bench)
    cut = 250
    future = df.copy()
    shock = np.linspace(1.0, 3.0, len(df) - cut - 1)
    for col in ("Open", "High", "Low", "Close"):
        future.loc[cut + 1:, col] = future.loc[cut + 1:, col].to_numpy() * shock
    future.loc[cut + 1:, "Volume"] = future.loc[cut + 1:, "Volume"].to_numpy() * 7
    bench_future = bench.copy()
    bench_future.loc[cut + 1:, "Close"] = bench_future.loc[cut + 1:, "Close"].to_numpy() * 2
    changed = A.extra_features(future, bench_future)
    pd.testing.assert_frame_equal(base.iloc[:cut + 1], changed.iloc[:cut + 1], check_exact=False, rtol=1e-12, atol=1e-12)
    assert not base.iloc[cut + 1:].equals(changed.iloc[cut + 1:])     # 미래 구간은 실제로 달라진다(검사가 비어 있지 않음)


def test_forward_targets_enter_at_next_open_and_exit_at_horizon_close():
    df = _frame(60)
    tg = A.forward_targets(df)
    i = 10
    expected = df.Close.iloc[i + 5] / df.Open.iloc[i + 1] - 1
    assert tg["fwd5"][i] == pytest.approx(expected)
    assert tg["fwd20"][i] == pytest.approx(df.Close.iloc[i + 20] / df.Open.iloc[i + 1] - 1)
    assert np.isnan(tg["fwd5"][len(df) - 5]) and not np.isnan(tg["fwd5"][len(df) - 6])
    assert tg["mae20"][i] == pytest.approx(df.Low.iloc[i + 1:i + 21].min() / df.Open.iloc[i + 1] - 1)


def test_barrier_label_is_conservative_and_requires_full_future():
    n = 40
    o = np.full(n, 100.0)
    h = np.full(n, 100.5)
    l = np.full(n, 99.5)
    assert A.barrier_label(o, h, l, 5, atr=2.0, k=1.5, days=20) == 0.0
    h2, l2 = h.copy(), l.copy()
    h2[8] = 103.5                       # 진입 100 + 1.5×2 = 103 이상
    assert A.barrier_label(o, h2, l2, 5, atr=2.0, k=1.5, days=20) == 1.0
    l2[8] = 96.5                        # 같은 봉에서 손절선(97)도 닿음 → 손절 우선
    assert A.barrier_label(o, h2, l2, 5, atr=2.0, k=1.5, days=20) == -1.0
    assert math.isnan(A.barrier_label(o, h, l, 30, atr=2.0, k=1.5, days=20))     # 미래 20봉이 없다
    assert math.isnan(A.barrier_label(o, h, l, 5, atr=0.0, k=1.5, days=20))


def _panel(rho: float, dates=60, names=40, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2024-01-02", periods=dates):
        score = rng.normal(size=names)
        noise = rng.normal(size=names)
        target = rho * score + math.sqrt(1 - rho ** 2) * noise
        for k in range(names):
            rows.append({"market": "US", "date": d, "ticker": f"T{k}", "score": score[k], "xr10": target[k]})
    return pd.DataFrame(rows)


def test_rank_ic_is_computed_per_date_and_recovers_known_signal():
    strong = A.ic_summary(_panel(0.30), "score", "xr10", 10, 3)
    none = A.ic_summary(_panel(0.0), "score", "xr10", 10, 3)
    assert 0.2 < strong["ic"] < 0.4 and strong["t_nw"] > 5
    assert abs(none["ic"]) < 0.06 and abs(none["t_nw"]) < 3
    assert strong["n"] == 60                                    # 날짜 수만큼의 교차단면


def test_partial_residual_removes_the_information_already_in_the_control():
    panel = _panel(0.0)
    rng = np.random.default_rng(1)
    panel["ncs_base"] = panel.score + rng.normal(0, 0.05, len(panel))      # 통제 변수 ≈ 점수
    panel = A.partial_residual(panel, "score", "ncs_base", "res")
    ic_res = A.ic_summary(panel.assign(res=panel.res), "res", "xr10", 10, 3)
    assert abs(ic_res["ic"]) < 0.06
    # 잔차는 날짜별로 통제 변수의 순위(백분위)와 직교한다(최소제곱 잔차의 정의)
    g = panel.dropna(subset=["res"]).groupby("date").apply(
        lambda x: np.corrcoef(x.res, x.ncs_base.rank(pct=True))[0, 1], include_groups=False)
    assert g.abs().max() < 1e-9


def test_newey_west_t_grows_with_signal_and_handles_short_series():
    assert math.isnan(A.newey_west_t([0.1, 0.2], 3))
    steady = [0.05 + 0.01 * ((-1) ** k) for k in range(200)]
    assert A.newey_west_t(steady, 3) > 10
    assert abs(A.newey_west_t(np.random.default_rng(0).normal(0, 0.1, 400), 3)) < 3.5
