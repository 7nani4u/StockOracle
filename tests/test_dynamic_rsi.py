import numpy as np
import pandas as pd

from market_briefing.dynamic_rsi import (
    add_dynamic_rsi_features,
    backtest_dynamic_rsi,
    config_for_market,
    dynamic_rsi_daily_snapshot,
    dynamic_rsi_signal_card,
    dynamic_rsi_snapshot,
)


def _strategy_frame() -> pd.DataFrame:
    size = 100
    close = np.full(size, 100.0)
    high = np.full(size, 101.0)
    low = np.full(size, 99.0)
    open_ = np.full(size, 100.0)
    rsi = np.resize(np.array([35.0, 50.0, 65.0]), size)

    # 첫 번째 저점 뒤 더 낮은 가격 저점·더 높은 RSI 저점(상승 다이버전스).
    close[65], low[65], high[65], rsi[65] = 91.0, 90.0, 92.0, 25.0
    close[66:69], low[66:69], high[66:69], rsi[66:69] = 94.0, 93.0, 95.0, 38.0
    close[72], low[72], high[72], rsi[72] = 89.0, 88.0, 90.0, 30.0
    close[73:76], low[73:76], high[73:76], rsi[73:76] = 92.0, 91.0, 93.0, 45.0
    # 50선 회복으로 다음 봉 시가 롱 후보 발생.
    close[76], open_[76], high[76], low[76], rsi[76] = 96.0, 94.0, 97.0, 93.0, 55.0
    open_[77], close[77], high[77], low[77], rsi[77] = 97.0, 99.0, 100.0, 96.0, 60.0
    # 동적 상단 진입 후 정상 범위 복귀로 청산.
    close[80], high[80], low[80], rsi[80] = 108.0, 109.0, 107.0, 80.0
    close[81], high[81], low[81], rsi[81] = 106.0, 108.0, 105.0, 55.0
    open_[82], close[82], high[82], low[82], rsi[82] = 105.0, 104.0, 106.0, 103.0, 50.0

    frame = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": np.full(size, 1_000_000.0), "RSI": rsi,
        "ATR": np.full(size, 2.0),
    })
    return frame


def test_market_profiles_are_explicit_and_cost_aware():
    krx = config_for_market("KRX")
    us = config_for_market("US")
    assert krx.market == "KRX"
    assert us.market == "US"
    assert krx.round_trip_cost_pct > us.round_trip_cost_pct
    assert krx.atr_stop_multiple > us.atr_stop_multiple
    assert krx.lookback != us.lookback


def test_dynamic_bounds_are_causal_and_ignore_future_changes():
    rng = np.random.default_rng(20260824)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, 220))))
    frame = pd.DataFrame({
        "Open": close.shift().fillna(close.iloc[0]),
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": 1_000_000,
    })
    first = add_dynamic_rsi_features(frame, market="US")
    changed = frame.copy()
    changed.loc[180:, "Close"] *= 1.7
    changed.loc[180:, "High"] *= 1.7
    changed.loc[180:, "Low"] *= 1.7
    second = add_dynamic_rsi_features(changed, market="US")
    pd.testing.assert_series_equal(
        first.loc[:179, "DRSI_Lower"], second.loc[:179, "DRSI_Lower"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        first.loc[:179, "DRSI_Upper"], second.loc[:179, "DRSI_Upper"],
        check_names=False,
    )


def test_bullish_divergence_and_50_cross_create_long_candidate():
    enriched = add_dynamic_rsi_features(_strategy_frame(), market="US")
    assert enriched.loc[74, "DRSI_Bull_Div"] == 1
    assert enriched.loc[76, "DRSI_Signal"] == 1
    assert enriched.loc[76, "DRSI_Position"] == 1
    assert enriched.loc[81, "DRSI_Signal"] == -1
    assert enriched.loc[81, "DRSI_Position"] == 0


def test_snapshot_and_card_are_explicitly_experimental():
    enriched = add_dynamic_rsi_features(_strategy_frame(), market="KRX")
    dd = enriched.where(pd.notna(enriched), None).to_dict(orient="list")
    snapshot = dynamic_rsi_snapshot(dd, market="KRX")
    card = dynamic_rsi_signal_card(dd, market="KRX")
    assert snapshot["available"] is True
    assert snapshot["experimental"] is True
    assert snapshot["market"] == "KRX"
    assert snapshot["purchase_timing"]["is_probability"] is False
    assert card is not None and card["context_only"] is True


def test_purchase_timing_waits_for_three_conditions_then_uses_next_open():
    enriched = add_dynamic_rsi_features(_strategy_frame(), market="US")

    armed_frame = enriched.iloc[:76]
    armed_dd = armed_frame.where(pd.notna(armed_frame), None).to_dict(orient="list")
    armed = dynamic_rsi_snapshot(armed_dd, market="US")["purchase_timing"]
    assert armed["state"] == "armed"
    assert armed["conditions_met"] == 2
    assert armed["eligible_now"] is False
    assert "RSI 50" in armed["window"]

    confirmed_frame = enriched.iloc[:77]
    confirmed_dd = confirmed_frame.where(pd.notna(confirmed_frame), None).to_dict(orient="list")
    confirmed = dynamic_rsi_snapshot(confirmed_dd, market="US")["purchase_timing"]
    assert confirmed["state"] == "confirmed"
    assert confirmed["conditions_met"] == 3
    assert confirmed["eligible_now"] is True
    assert "다음 거래일" in confirmed["window"]
    assert confirmed["reference_close"] == confirmed_frame.iloc[-1]["Close"]
    assert confirmed["max_chase_price"] == confirmed["reference_close"] + 1.0
    assert "프리마켓" in confirmed["market_note"]


def test_purchase_timing_does_not_treat_active_position_as_fresh_entry():
    enriched = add_dynamic_rsi_features(_strategy_frame(), market="KRX")
    active_frame = enriched.iloc[:78]
    active_dd = active_frame.where(pd.notna(active_frame), None).to_dict(orient="list")
    timing = dynamic_rsi_snapshot(active_dd, market="KRX")["purchase_timing"]

    assert timing["state"] == "active"
    assert timing["eligible_now"] is False
    assert timing["reference_close"] == enriched.iloc[76]["Close"]
    assert "VI" in timing["market_note"]


def test_daily_snapshot_is_independent_of_selected_chart_timeframe():
    frame = _strategy_frame()
    dd = frame.where(pd.notna(frame), None).to_dict(orient="list")
    dd["Date"] = pd.date_range("2026-01-01", periods=len(frame), freq="B").strftime("%Y-%m-%d").tolist()

    snapshot = dynamic_rsi_daily_snapshot(dd, market="US")

    assert snapshot["available"] is True
    assert snapshot["timeframe"] == "1d"
    assert snapshot["timeframe_label"] == "일봉"
    assert snapshot["source_bars"] == len(frame)
    assert snapshot["as_of"] == dd["Date"][-1]


def test_backtest_uses_next_open_and_market_cost():
    frame = _strategy_frame()
    result = backtest_dynamic_rsi(frame, market="US")
    assert result["summary"]["trade_count"] == 1
    trade = result["trades"][0]
    assert trade["entry_index"] == 77
    assert trade["entry_price"] == frame.loc[77, "Open"]
    assert trade["exit_reason"] == "dynamic_exit_next_open"
    assert trade["exit_price"] == frame.loc[82, "Open"]
    assert trade["return_pct"] < trade["gross_return_pct"]
