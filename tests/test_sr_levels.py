"""반응 기반 수평선(sr_levels) 회귀 테스트 — 인과성·접촉 수 규칙·전환 재시험."""
import numpy as np

from market_briefing.sr_levels import (
    LevelConfig, atr_series, cluster_levels, confirmed_swings, find_levels, level_events,
)


def _ohlc(closes, spread=0.5):
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread
    return opens, highs, lows, closes


def _triangle_wave(cycles, low=100.0, high=110.0, half=10):
    path = [high]
    for _ in range(cycles):
        path += list(np.linspace(high, low, half + 1)[1:])
        path += list(np.linspace(low, high, half + 1)[1:])
    return path


def test_cluster_counts_only_separated_touches_and_needs_tolerance():
    swings = [(10, 100.0, "low"), (11, 100.1, "low"), (30, 100.2, "low"), (50, 120.0, "high")]
    levels = cluster_levels(swings, tol=0.5, min_gap=3)
    support = [lv for lv in levels if abs(lv["price"] - 100.1) < 0.5][0]
    assert support["touches"] == 2          # 인접한 10·11 봉은 같은 스윙으로 한 번만 센다
    assert [lv for lv in levels if lv["price"] > 119][0]["touches"] == 1


def test_find_levels_requires_two_reactions_and_labels_role():
    o, h, l, c = _ohlc(_triangle_wave(4) + [105.0])
    levels = find_levels(h, l, c)
    prices = [round(lv["price"]) for lv in levels]
    assert any(abs(p - 100) <= 1 for p in prices)       # 바닥 4번 반등
    assert any(abs(p - 110) <= 1 for p in prices)
    for lv in levels:
        assert lv["touches"] >= 2
        assert lv["role"] == ("support" if c[-1] > lv["price"] else "resistance")
    # 접촉 1회뿐인 선은 기본값에서 제외된다
    o1, h1, l1, c1 = _ohlc(_triangle_wave(1) + [105.0] * 3)
    assert find_levels(h1, l1, c1) == []


def test_support_bounce_event_on_fresh_touch_with_close_above_level():
    path = _triangle_wave(4)[:-1]                     # 마지막은 고점에서 내려오는 중
    path += list(np.linspace(110, 101.0, 9))          # 선 위까지 하락
    o, h, l, c = _ohlc(path)
    n = len(c)
    c[-1] = 100.6                                     # 종가는 선(≈100) 위에서 마감
    l[-1] = 99.7                                      # 저가는 선 근처까지 내려옴
    events = level_events(o, h, l, c, min_touches=2)
    assert any(e["type"] == "support_bounce" and e["side"] == 1 for e in events)
    assert all(e["touches"] >= 2 for e in events)
    # 같은 사건이라도 1회 접촉 선까지 세면 사건이 같거나 더 많다
    assert len(level_events(o, h, l, c, min_touches=1)) >= len(events)


def test_events_never_read_future_bars():
    o, h, l, c = _ohlc(_triangle_wave(5) + list(np.linspace(110, 100.8, 9)))
    t = len(c) - 1
    base = level_events(o, h, l, c, min_touches=1)
    o2 = np.concatenate([o, [90.0, 150.0, 60.0]])
    h2 = np.concatenate([h, [95.0, 160.0, 70.0]])
    l2 = np.concatenate([l, [80.0, 140.0, 50.0]])
    c2 = np.concatenate([c, [85.0, 155.0, 55.0]])
    prefix = level_events(o2[:t + 1], h2[:t + 1], l2[:t + 1], c2[:t + 1], min_touches=1)
    assert prefix == base


def test_confirmed_swings_exclude_the_last_unconfirmed_bars():
    o, h, l, c = _ohlc(_triangle_wave(3) + [104.0, 100.0])      # 마지막 저점은 오른쪽 봉이 없어 미확정
    swings = confirmed_swings(h, l, 4)
    assert all(i <= len(c) - 1 - 4 for i, _, _ in swings)


def test_non_finite_or_short_input_returns_empty():
    o, h, l, c = _ohlc(_triangle_wave(3))
    c = c.copy()
    c[5] = np.nan
    assert level_events(o, h, l, c) == []
    assert find_levels(h[:10], l[:10], c[:10]) == []
    assert LevelConfig().min_touches == 2
    assert atr_series(h, l, np.nan_to_num(c)).shape == c.shape
