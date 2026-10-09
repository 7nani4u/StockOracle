import numpy as np
import pandas as pd
import pytest

from scripts import train_ml_model as training
from market_briefing.ml_features import FEATURE_COLS


def feature_frame():
    dates = pd.bdate_range("2024-01-01", periods=100).delete([10, 11, 12, 13])
    frame = pd.DataFrame({column: np.ones(len(dates)) for column in FEATURE_COLS})
    frame["date"] = dates
    frame["ticker"] = "AAPL"
    frame["label"] = np.arange(len(dates)) % 2
    frame["label_end_date"] = frame["date"].shift(-14)
    return frame


def test_split_survives_removed_index_and_purges_actual_label_dates():
    frame = feature_frame()
    frame.loc[int((len(frame) - 1) * .8), FEATURE_COLS[0]] = np.nan
    _, _, _, _, train, test, split_date = training._time_split(frame)
    assert not train.empty
    assert (train.label_end_date < test.date.min()).all()
    assert train.date.max() < split_date
    assert (test.date >= split_date).all()


def test_small_split_never_falls_back_to_unpurged_rows():
    frame = feature_frame().iloc[:20].copy()
    frame["label_end_date"] = frame.date + pd.Timedelta(days=100)
    with pytest.raises(ValueError, match="purged"):
        training._time_split(frame)


@pytest.mark.parametrize("valid,synthetic,known", [(False, False, True), (True, True, True), (True, False, False)])
def test_publication_requires_validation_and_real_provenance(valid, synthetic, known):
    with pytest.raises(RuntimeError, match="publication rejected"):
        training._assert_publishable(valid, synthetic, known)
    training._assert_publishable(valid, synthetic, known, allow_unvalidated=True)


def test_real_validated_model_is_publishable():
    training._assert_publishable(True, False, True)


def test_index_accumulation_uses_own_sessions_before_calendar_union(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=35)
    kr_dates = dates.delete(23)
    def download(symbol, **kwargs):
        selected = kr_dates if symbol in ("^KS11", "069500.KS", "^KQ11") else dates
        return pd.DataFrame({"Close": 100 * 1.01 ** np.arange(len(selected))}, index=selected)
    monkeypatch.setattr(training, "_HAS_YFINANCE", True)
    monkeypatch.setattr(training.yf, "download", download)
    indices = training._fetch_index_df("2024-01-01", "2024-03-01").set_index("date")
    assert indices.loc[dates[-1], "KRX_NIFTY_cum20"] == pytest.approx(20.0)
    assert indices.loc[dates[-1], "US_NIFTY_cum20"] == pytest.approx(20.0)


@pytest.mark.parametrize("primary_rows", [35, 1, 0])
def test_krx_benchmark_matches_inference_with_short_history_fallback(monkeypatch, primary_rows):
    from market_briefing.ml_predictor import INDEX_SYMBOLS
    dates = pd.bdate_range("2024-01-01", periods=35)
    requested = []
    def download(symbol, **kwargs):
        requested.append(symbol)
        selected = dates[:primary_rows] if symbol == INDEX_SYMBOLS["KRX"]["market"] else dates
        growth = 1.02 if symbol == INDEX_SYMBOLS["KRX"]["market"] else 1.01
        return pd.DataFrame({"Close": 100 * growth ** np.arange(len(selected))}, index=selected)
    monkeypatch.setattr(training, "_HAS_YFINANCE", True)
    monkeypatch.setattr(training.yf, "download", download)
    indices = training._fetch_index_df("2024-01-01", "2024-03-01").set_index("date")
    assert requested[0] == INDEX_SYMBOLS["KRX"]["market"]
    assert ("069500.KS" in requested) == (primary_rows < 21)
    assert indices.loc[dates[-1], "KRX_NIFTY_cum20"] == pytest.approx(40 if primary_rows >= 21 else 20)


def test_feature_build_preserves_raw_synthetic_provenance():
    raw = training._synthetic_ohlcv("AAPL", days=180, seed=42)
    raw["is_synthetic"] = True
    features = training._build_features_from_raw(raw)
    assert not features.empty
    assert features["is_synthetic"].eq(True).all()
    assert features["label_end_date"].notna().all()
