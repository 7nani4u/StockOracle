"""SMMA·Fractal 실시간 공통 규칙과 백테스트 체결의 동등성 검증."""

import pandas as pd

import scripts.backtest_arty_smma_fractal as backtest_module
from api.index import (
    _arty_evaluate_entry,
    _arty_evaluate_setup,
    _arty_validation_verdict,
    _prepare_arty_series,
    _smma_values,
)
from scripts.backtest_arty_smma_fractal import (
    KRX_TICKERS,
    US_TICKERS,
    StrategyConfig,
    backtest_config,
    cost_sensitivity,
    resolve_symbol,
)


def _historical_setup_frame(size=520, pivot=300):
    closes = [100 + 0.03 * index + 0.00025 * index * index for index in range(size)]
    opens = [value - 0.15 for value in closes]
    highs = [value + 0.5 for value in closes]
    lows = [value - 0.5 for value in closes]
    lows[pivot] = _smma_values(closes, 21)[pivot] - 0.05
    volumes = [100_000 + (index % 5) * 1_000 for index in range(size)]
    frame = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=pd.date_range("2024-01-02", periods=size, freq="B"),
    )
    series = _prepare_arty_series(frame.to_dict(orient="list"))
    for column in ("ATR", "SMMA21", "SMMA50", "SMMA200"):
        frame[column] = series[column]
    frame.attrs["arty_series"] = series
    return frame, series


def test_backtest_uses_same_setup_and_entry_rules_as_live_engine():
    pivot = 300
    confirmed = pivot + 2
    entry_index = confirmed + 1
    frame, series = _historical_setup_frame(pivot=pivot)
    config = StrategyConfig(
        retest_line=21,
        entry_delay=1,
        atr_tolerance=0.35,
        pullback_volume_max=None,
        rebound_volume_min=None,
        smma200_slope_min_pct20=None,
    )
    payload = {
        "retest_line": config.retest_line,
        "entry_delay": config.entry_delay,
        "atr_tolerance": config.atr_tolerance,
        "pullback_volume_max": config.pullback_volume_max,
        "rebound_volume_min": config.rebound_volume_min,
        "smma200_slope_min_pct20": config.smma200_slope_min_pct20,
    }

    setup = _arty_evaluate_setup(series, pivot, confirmed, payload)
    entry = _arty_evaluate_entry(
        series, pivot, entry_index, series["Open"][entry_index], payload
    )
    trades = backtest_config("TEST", "US", frame, config)

    assert setup["passed"] is True
    assert entry["passed"] is True
    matching = [
        trade for trade in trades
        if trade.signal_date == str(frame.index[confirmed].date())
    ]
    assert len(matching) == 1
    assert matching[0].entry_index == entry_index
    assert matching[0].entry_price == round(series["Open"][entry_index], 6)


def test_confirmation_close_is_labelled_as_non_slippage_benchmark():
    frame, _ = _historical_setup_frame()
    config = StrategyConfig(
        retest_line=21,
        entry_delay=0,
        atr_tolerance=0.35,
        pullback_volume_max=None,
        rebound_volume_min=None,
        smma200_slope_min_pct20=None,
    )
    trades = backtest_config("TEST", "US", frame, config)

    assert config.entry_label == "확정 봉 종가(비실행 비교)"
    assert all(trade.open_gap_vs_confirm_close_pct == 0 for trade in trades)


def test_retired_and_renamed_symbols_are_normalized_in_current_universe():
    assert resolve_symbol("091990.KQ")[0] is None
    assert resolve_symbol("MMC")[0] == "MRSH"
    assert "091990.KQ" not in KRX_TICKERS
    assert "042660.KS" in KRX_TICKERS
    assert "MMC" not in US_TICKERS
    assert "MRSH" in US_TICKERS


def test_validation_requires_latest_rolling_and_data_quality():
    latest = {
        "trades": 32,
        "avg_r_multiple": 0.4,
        "profit_factor": 1.4,
        "median_return_pct": 1.0,
        "avg_r_95ci_low": 0.1,
    }
    rolling = {
        "folds": [{}, {}, {}, {}],
        "aggregate": {
            "trades": 50,
            "avg_r_multiple": 0.3,
            "profit_factor": 1.3,
            "avg_r_95ci_low": 0.05,
        },
        "positive_avg_r_fold_ratio": 0.75,
    }

    assert _arty_validation_verdict(latest, rolling, True)[0] == "accepted"
    assert _arty_validation_verdict(latest, rolling, False)[0] == "rejected"
    assert _arty_validation_verdict(latest, {}, True)[0] == "provisional"


def test_extra_execution_cost_stress_never_improves_average_return():
    frame, _ = _historical_setup_frame()
    config = StrategyConfig(21, 1, 0.35, None, None, None)
    trades = backtest_config("TEST", "US", frame, config)
    results = cost_sensitivity(trades, "US")

    assert results["0.00"]["trades"] == len(trades)
    assert results["0.50"]["avg_return_pct"] <= results["0.00"]["avg_return_pct"]


def test_download_uses_secondary_provider_when_primary_is_empty(monkeypatch):
    fallback, _ = _historical_setup_frame(size=500, pivot=300)
    fallback["Adj Close"] = fallback["Close"]
    fallback["Dividends"] = 0.0
    fallback["Stock Splits"] = 0.0
    monkeypatch.setattr(
        backtest_module, "_download_yfinance_history", lambda ticker, period: pd.DataFrame()
    )
    monkeypatch.setattr(
        backtest_module, "_download_fdr_history", lambda ticker, period: fallback
    )

    frame, raw = backtest_module.download_one("TEST", "3y")

    assert len(frame) == 500
    assert len(raw) == 500
    assert frame.attrs["provider"] == "FinanceDataReader"
    assert frame.attrs["download_meta"]["attempts"][0]["rows"] == 0
