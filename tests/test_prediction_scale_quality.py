import copy

import pytest

from api.index import (
    _calibrate_target_probability,
    _krx_tick_size,
    _prediction_quality_profile,
    _round_market_price,
    calc_probability,
    calc_risk,
)
from tests.test_buy_entry_steps import _calculate, _sample_buy_dd


PRICE_COLUMNS = {
    "Open", "High", "Low", "Close", "ATR", "MA20", "MA60", "MA120",
    "EMA20", "BB_Middle", "BB_Lower", "BB_Upper", "MACD", "Signal_Line",
}


def _scaled_dd(scale: float):
    data = copy.deepcopy(_sample_buy_dd(size=260))
    for key in PRICE_COLUMNS:
        if key in data:
            data[key] = [None if value is None else value * scale for value in data[key]]
    data["Volume"] = [max(1.0, value / scale) for value in data["Volume"]]
    return data


def test_probability_is_price_scale_invariant_when_turnover_is_equal():
    cheap = _scaled_dd(40.0)
    expensive = _scaled_dd(5_000.0)

    cheap_probability = calc_probability(65, cheap, "KRX")[0]
    expensive_probability = calc_probability(65, expensive, "KRX")[0]

    assert cheap_probability == pytest.approx(expensive_probability, abs=0.2)
    assert _prediction_quality_profile(cheap, "KRX")["price_bucket"] == "under_5k"
    assert _prediction_quality_profile(expensive, "KRX")["price_bucket"] == "over_500k"


def test_low_turnover_shrinks_probability_toward_neutral():
    liquid = _scaled_dd(1_000.0)
    liquid["Volume"] = [1_000_000.0] * len(liquid["Volume"])
    thin = copy.deepcopy(liquid)
    thin["Volume"] = [1.0] * len(thin["Volume"])

    liquid_probability = calc_probability(70, liquid, "KRX")[0]
    thin_probability = calc_probability(70, thin, "KRX")[0]

    assert abs(thin_probability - 50.0) < abs(liquid_probability - 50.0)
    assert _prediction_quality_profile(thin, "KRX")["warnings"]


def test_krx_actionable_prices_use_valid_ticks_across_price_buckets():
    for value in (1_999.4, 5_005.0, 55_555.0, 250_250.0, 650_500.0):
        rounded = _round_market_price(value, "KRX")
        assert rounded % _krx_tick_size(rounded, "KRX") == 0

    result = _calculate(_scaled_dd(1_000.0))
    for family in ("aggressive_bands", "recommended_bands"):
        for band in result[family]:
            for price in band["range"]:
                assert price % _krx_tick_size(price, "KRX") == 0
            for step in band["steps"]:
                assert step["price"] % _krx_tick_size(step["price"], "KRX") == 0


def test_us_sub_dollar_and_regular_prices_use_market_ticks():
    assert _round_market_price(0.123456, "US") == 0.1235
    assert _round_market_price(123.456, "US") == 123.46


def test_target_probability_uses_stop_first_market_profile_priors():
    normal_quality = {"reliability": 1.0, "confidence_cap": 92.0, "atr_pct": 2.0}

    assert _calibrate_target_probability(71.5, "KRX", "balanced", normal_quality) == 47.9
    assert _calibrate_target_probability(53.2, "US", "balanced", normal_quality) == 27.9
    assert _calibrate_target_probability(74.4, "US", "pullback_main", normal_quality) == 34.8


def test_qualified_us_entry_uses_conditional_stop_first_prior():
    quality = {"reliability": 1.0, "confidence_cap": 92.0, "atr_pct": 2.0}

    standard = _calibrate_target_probability(53.2, "US", "balanced", quality)
    qualified = _calibrate_target_probability(
        53.2, "US", "balanced", quality, qualified_us_entry=True,
    )

    assert standard == 27.9
    assert qualified == 34.2


def test_high_volatility_reduces_profiles_with_observed_overconfidence():
    normal_quality = {"reliability": 1.0, "confidence_cap": 92.0, "atr_pct": 3.0}
    high_vol_quality = {"reliability": 1.0, "confidence_cap": 92.0, "atr_pct": 5.0}

    normal = _calibrate_target_probability(56.9, "KRX", "aggressive", normal_quality)
    high_vol = _calibrate_target_probability(56.9, "KRX", "aggressive", high_vol_quality)

    assert normal == 41.5
    assert high_vol == 35.5


def test_risk_target_probabilities_respect_walk_forward_caps():
    dd = _scaled_dd(100.0)
    price = dd["Close"][-1]
    atr = dd["ATR"][-1]

    krx = calc_risk(price, atr, "KRX", dd)
    us = calc_risk(price, atr, "US", dd)

    assert max(level["prob_pct"] for level in krx["balanced"]["tp_levels"]) <= 62.0
    assert max(level["prob_pct"] for level in us["balanced"]["tp_levels"]) <= 42.0
    assert max(level["prob_pct"] for level in us["aggressive"]["tp_levels"]) <= 38.0


def test_us_risk_scenarios_include_entry_status_and_cost_adjusted_expectancy():
    dd = _scaled_dd(100.0)
    result = calc_risk(dd["Close"][-1], dd["ATR"][-1], "US", dd)

    for profile in ("balanced", "aggressive"):
        scenario = result[profile]
        assert isinstance(scenario["entry_eligible"], bool)
        assert isinstance(scenario["entry_status"], str)
        assert isinstance(scenario["expected_value_pct"], float)
