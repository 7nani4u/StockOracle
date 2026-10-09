# -*- coding: utf-8 -*-
"""모멘텀 지속(급등+3일 절반수성) 판정기 테스트. 점수 미반영 연구용."""
import math

from market_briefing.momentum_persistence import (
    detect_momentum_persistence,
    find_surges,
    SURGE_MIN_PCT,
)


def _ohlc(closes, spread=0.005, vol=1_000_000):
    return (
        [c * (1 + spread) for c in closes],
        [c * (1 - spread) for c in closes],
        [vol] * len(closes),
    )


def _pass_series():
    # 100 → 100 → 125 급등(+25%) → 120, 118, 119 (중간값 112.5 상회, ATR용 17봉)
    closes = [100.0] * 12 + [100.0, 125.0, 120.0, 118.0, 119.0]
    assert len(closes) == 17
    return closes


def test_surge_day_reports_surge_stage():
    closes = [100.0] * 9 + [100.0, 125.0]
    highs, lows, vols = _ohlc(closes)
    r = detect_momentum_persistence(closes, highs, lows, vols)
    assert r["available"] is True
    assert r["stage"] == "SURGE"
    assert r["surge_pct"] == 25.0
    assert r["midpoint"] == 112.5


def test_three_day_hold_reports_pass():
    closes = _pass_series()
    highs, lows, vols = _ohlc(closes)
    r = detect_momentum_persistence(closes, highs, lows, vols)
    assert r["stage"] == "PASS"
    assert r["conditions"]["persistence"]["passed"] is True
    assert r["min_hold_close"] == 118.0
    assert r["retain_ratio"] == round((118.0 - 100.0) / 25.0, 3)
    assert r["entry_trigger"] == 119.0
    assert r["stop_price"] is not None and r["stop_price"] < r["midpoint"]


def test_break_below_midpoint_reports_fail():
    closes = [100.0] * 9 + [100.0, 125.0, 120.0, 110.0, 119.0]  # 2일차 110 < 112.5
    highs, lows, vols = _ohlc(closes)
    r = detect_momentum_persistence(closes, highs, lows, vols)
    assert r["stage"] == "FAIL"
    assert r["conditions"]["persistence"]["passed"] is False


def test_wait_stage_before_three_days():
    closes = [100.0] * 9 + [100.0, 125.0, 120.0]
    highs, lows, vols = _ohlc(closes)
    r = detect_momentum_persistence(closes, highs, lows, vols)
    assert r["stage"] == "WAIT"
    assert r["bars_since_surge"] == 1


def test_no_surge_is_none():
    closes = [100.0 + i * 0.3 for i in range(30)]  # 완만한 상승, 급등 없음
    highs, lows, vols = _ohlc(closes)
    r = detect_momentum_persistence(closes, highs, lows, vols)
    assert r["stage"] == "NONE"


def test_below_threshold_is_none():
    closes = [100.0] * 10 + [100.0, 115.0, 114.0, 113.0, 114.0]  # +15%는 미달
    highs, lows, vols = _ohlc(closes)
    r = detect_momentum_persistence(closes, highs, lows, vols)
    assert r["stage"] == "NONE"


def test_old_surge_expires():
    closes = _pass_series() + [120.0] * 10  # 확인 후 10봉 경과
    highs, lows, vols = _ohlc(closes)
    r = detect_momentum_persistence(closes, highs, lows, vols)
    assert r["stage"] == "NONE"


def test_volume_filter_rejects_thin_surge():
    closes = _pass_series()
    highs, lows, _ = _ohlc(closes)
    vols = [1_000_000] * 13 + [500_000] + [1_000_000] * 3  # 급등일 거래량 급감
    r = detect_momentum_persistence(closes, highs, lows, vols, require_volume_ratio=2.0)
    assert r["stage"] == "NONE"
    assert r["reason"] == "거래량 동반 부족"


def test_strict_low_break_is_reported():
    # 종가는 수성하나 저점이 중간값을 깬 경우 참고용으로 기록
    closes = _pass_series()
    highs = [c * 1.005 for c in closes]
    lows = [c * 0.995 for c in closes]
    lows[-2] = 100.0  # 2일차 저점 이탈
    vols = [1_000_000] * len(closes)
    r = detect_momentum_persistence(closes, highs, lows, vols)
    assert r["stage"] == "PASS"  # 종가 기준은 수성
    assert r["strict_low_break"] is True


def test_insufficient_data_is_unavailable():
    r = detect_momentum_persistence([100.0] * 5)
    assert r["available"] is False
    assert r["stage"] == "NONE"


def test_never_raises_on_garbage():
    r = detect_momentum_persistence([], None, None, None)
    assert r["stage"] == "NONE"
    r = detect_momentum_persistence([float("nan")] * 20, None, None, None)
    assert r["stage"] in ("NONE",)
    assert find_surges([100.0, 130.0]) == [{"index": 1, "pct": 30.0}]
    assert find_surges([100.0, 110.0]) == []
    assert SURGE_MIN_PCT == 20.0
