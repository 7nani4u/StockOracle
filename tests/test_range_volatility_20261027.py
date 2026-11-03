"""범위 기반 변동성 추정기(Parkinson·Garman-Klass·Rogers-Satchell·Yang-Zhang) 순수 함수 테스트.

공식 출처: MQL5 Articles 23830(Riccio, 2026-10-01)·23454(Gbadebo, 2026-08-05). 두 글이 서로 일치하는
정의를 손으로 계산한 값과 대조하고, 시뮬레이션·불변성·결측 처리를 확인한다.
"""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from market_briefing import range_volatility as rv

LN2 = math.log(2.0)


def _bar(o, h, l, c):
    return {"o": [o], "h": [h], "l": [l], "c": [c]}


def test_parkinson_single_bar_matches_formula():
    x = math.log(105 / 95)
    assert rv.variance("parkinson", [100], [105], [95], [101], window=1, min_obs=1) == pytest.approx(x * x / (4 * LN2))


def test_garman_klass_single_bar_matches_formula():
    x, c = math.log(105 / 95), math.log(101 / 100)
    expected = 0.5 * x * x - (2 * LN2 - 1) * c * c
    assert rv.variance("garman_klass", [100], [105], [95], [101], window=1, min_obs=1) == pytest.approx(expected)


def test_rogers_satchell_single_bar_matches_formula():
    o, h, l, c = 100.0, 105.0, 95.0, 101.0
    u, d, cc = math.log(h / o), math.log(l / o), math.log(c / o)
    expected = u * (u - cc) + d * (d - cc)
    assert rv.variance("rogers_satchell", [o], [h], [l], [c], window=1, min_obs=1) == pytest.approx(expected)


def test_yang_zhang_matches_manual_computation():
    rng = np.random.default_rng(7)
    n = 20
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n + 1)))
    op = close[:-1] * np.exp(rng.normal(0, 0.004, n))
    cl = close[1:]
    hi = np.maximum(op, cl) * np.exp(np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(op, cl) * np.exp(-np.abs(rng.normal(0, 0.003, n)))
    prev = close[:-1]
    o_ret = np.log(op / prev)
    c_ret = np.log(cl / op)
    rs = np.log(hi / cl) * np.log(hi / op) + np.log(lo / cl) * np.log(lo / op)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    manual = o_ret.var(ddof=1) + k * c_ret.var(ddof=1) + (1 - k) * rs.mean()
    # 첫 봉의 전일 종가를 시리즈 맨 앞에 붙여 같은 n 개 구간을 만든다.
    got = rv.variance("yang_zhang", np.r_[prev[0], op], np.r_[prev[0], hi], np.r_[prev[0], lo], np.r_[prev[0], cl],
                      window=n, min_obs=n)
    # 위 입력은 맨 앞 가짜 봉(O=H=L=C=prev[0])을 포함하므로 마지막 n 개 봉과 그 직전 종가가 정확히 manual 과 같다.
    assert got == pytest.approx(manual, rel=1e-12)


def _simulate(sigma_daily, days, steps=400, seed=1, gap_sigma=0.0):
    """일중 400스텝 브라운 경로로 OHLC 를 만든다. 진짜 일간 분산 = sigma_daily² (+ 갭 분산)."""
    rng = np.random.default_rng(seed)
    o, h, l, c = [], [], [], []
    price = 100.0
    for _ in range(days):
        price *= math.exp(rng.normal(0, gap_sigma)) if gap_sigma else 1.0
        path = price * np.exp(np.cumsum(rng.normal(0, sigma_daily / math.sqrt(steps), steps)))
        o.append(price)
        h.append(max(path.max(), price))
        l.append(min(path.min(), price))
        c.append(path[-1])
        price = path[-1]
    return o, h, l, c


@pytest.mark.parametrize("method", ["parkinson", "garman_klass", "rogers_satchell", "yang_zhang"])
def test_estimators_recover_true_sigma_without_gaps(method):
    o, h, l, c = _simulate(0.02, 3000)
    var = rv.variance(method, o, h, l, c, window=3000, min_obs=100)
    # 일중 이산 표본화로 극값이 약간 낮게 잡히므로 5% 안에서 일치하면 된다.
    assert math.sqrt(var) == pytest.approx(0.02, rel=0.05)


def test_yang_zhang_sees_overnight_gaps_but_intrabar_estimators_do_not():
    o, h, l, c = _simulate(0.015, 3000, gap_sigma=0.01, seed=3)
    true_sigma = math.sqrt(0.015 ** 2 + 0.01 ** 2)
    yz = math.sqrt(rv.variance("yang_zhang", o, h, l, c, window=3000, min_obs=100))
    gk = math.sqrt(rv.variance("garman_klass", o, h, l, c, window=3000, min_obs=100))
    assert yz == pytest.approx(true_sigma, rel=0.06)
    assert gk < true_sigma * 0.92  # 갭이 빠져 낮게 나온다


@pytest.mark.parametrize("method", rv.METHODS)
def test_scale_invariance(method):
    o, h, l, c = _simulate(0.02, 80, seed=5)
    a = rv.variance(method, o, h, l, c, window=60, min_obs=20)
    b = rv.variance(method, [x * 1234.5 for x in o], [x * 1234.5 for x in h], [x * 1234.5 for x in l],
                    [x * 1234.5 for x in c], window=60, min_obs=20)
    assert a == pytest.approx(b, rel=1e-9)


def test_invalid_bars_are_skipped_not_joined():
    o, h, l, c = _simulate(0.02, 60, seed=9)
    clean = rv.variance("garman_klass", o, h, l, c, window=60, min_obs=20)
    o2, h2, l2, c2 = list(o), list(h), list(l), list(c)
    o2[30], h2[30], l2[30], c2[30] = float("nan"), float("nan"), float("nan"), float("nan")
    h2[40], l2[40] = l[40], h[40]  # high < low
    c2[50] = -1.0
    dirty = rv.variance("garman_klass", o2, h2, l2, c2, window=60, min_obs=20)
    assert dirty is not None and dirty >= 0
    assert dirty != clean
    # 세 봉을 뺀 나머지만으로 계산한 값과 같아야 한다(오류 봉을 이웃과 이어 붙이지 않는다).
    keep = [i for i in range(60) if i not in (30, 40, 50)]
    manual = rv.variance("garman_klass", [o[i] for i in keep], [h[i] for i in keep], [l[i] for i in keep],
                         [c[i] for i in keep], window=60, min_obs=20)
    assert dirty == pytest.approx(manual, rel=1e-12)


def test_yang_zhang_overnight_term_only_uses_adjacent_valid_bars():
    o, h, l, c = _simulate(0.02, 60, seed=11)
    o[30] = h[30] = l[30] = c[30] = float("nan")
    # 31번째 봉의 전일 종가는 결측이므로 그 봉의 갭은 쓰지 않는다 — 예외 없이 유한한 값이어야 한다.
    value = rv.variance("yang_zhang", o, h, l, c, window=60, min_obs=20)
    assert value is not None and math.isfinite(value) and value > 0


def test_insufficient_or_bad_input_returns_none():
    o, h, l, c = _simulate(0.02, 5)
    assert rv.variance("parkinson", o, h, l, c, window=60, min_obs=20) is None
    assert rv.variance("unknown", o, h, l, c, window=5, min_obs=2) is None
    assert rv.variance("parkinson", [], [], [], [], window=5, min_obs=2) is None
    assert rv.variance("parkinson", o, h, l, c[:-1], window=5, min_obs=2) is None  # 길이 불일치


def test_prefix_truncation_does_not_change_past_estimate():
    """미래 봉을 붙여도 과거 시점 추정은 변하지 않는다(인과성)."""
    o, h, l, c = _simulate(0.02, 120, seed=13)
    for method in rv.METHODS:
        past = rv.variance(method, o[:100], h[:100], l[:100], c[:100], window=60, min_obs=20)
        series = rv.rolling_sigma(method, o, h, l, c, window=60, min_obs=20)
        assert series[99] == pytest.approx(math.sqrt(past), rel=1e-12)
        assert all(math.isnan(x) for x in series[:19])


def test_daily_sigma_wrapper_is_square_root_of_variance():
    o, h, l, c = _simulate(0.02, 100, seed=17)
    assert rv.daily_sigma("yang_zhang", o, h, l, c, window=60, min_obs=20) == pytest.approx(
        math.sqrt(rv.variance("yang_zhang", o, h, l, c, window=60, min_obs=20)))
    assert rv.daily_sigma("yang_zhang", o[:3], h[:3], l[:3], c[:3], window=60, min_obs=20) is None
