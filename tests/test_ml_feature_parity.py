"""Training and live LightGBM feature calculations must remain unit-compatible."""

import numpy as np
import pandas as pd
import pytest

from market_briefing.ml_features import FEATURE_COLS, compute_feature_vector, engineer_ticker_features


def test_market_aware_features_match_training_and_inference_units():
    size = 180
    dates = pd.date_range("2025-01-01", periods=size, freq="B")
    close = 100 + np.linspace(0, 18, size) + np.sin(np.arange(size) / 5) * 2
    raw = pd.DataFrame({
        "date": dates,
        "open": close * 0.998,
        "high": close * 1.012,
        "low": close * 0.988,
        "close": close,
        "volume": 1_000_000 + (np.arange(size) % 9) * 10_000,
    })
    market_returns = pd.Series(np.sin(np.arange(size) / 11) * 0.4)
    index = pd.DataFrame({
        "date": dates,
        "KRX_NIFTY_return": market_returns,
        "KRX_BANKNIFTY_return": market_returns / 2,
        "KRX_NIFTY_cum20": market_returns.rolling(20).sum(),
        "India_VIX": 16.0,
    })

    trained = engineer_ticker_features(raw, market="KRX", index_df=index).iloc[-15]
    # The last 14 rows have no forward label; compare the final label-bearing row.
    row_pos = trained.name
    cache = {
        "NIFTY_return": float(index.loc[row_pos, "KRX_NIFTY_return"]),
        "BANKNIFTY_return": float(index.loc[row_pos, "KRX_BANKNIFTY_return"]),
        "NIFTY_cum20": float(index.loc[row_pos, "KRX_NIFTY_cum20"]),
        "India_VIX": 16.0,
    }
    live = compute_feature_vector(
        raw["close"].iloc[:row_pos + 1].tolist(), raw["high"].iloc[:row_pos + 1].tolist(),
        raw["low"].iloc[:row_pos + 1].tolist(), raw["volume"].iloc[:row_pos + 1].tolist(),
        market="KRX", index_cache=cache,
    )

    assert set(FEATURE_COLS).issubset(live)
    for name in ("MACD", "MACD_signal", "MACD_hist", "market_return_20d", "trend_spread_20_50", "price_return_1d", "volatility_rank_60", "market_is_krx"):
        assert live[name] == pytest.approx(float(trained[name]), abs=1e-8)


def test_feature_labels_keep_neutral_observations_for_live_decision_parity():
    dates = pd.date_range("2025-01-01", periods=160, freq="B")
    close = np.full(160, 100.0)
    raw = pd.DataFrame({"date": dates, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1_000_000})

    featured = engineer_ticker_features(raw, market="US")

    assert set(featured["label"].dropna().unique()) == {2.0}
