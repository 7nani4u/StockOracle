import numpy as np
import pandas as pd
import pytest
from market_briefing.ml_features import walk_forward_splits, FORWARD_DAYS
from market_briefing.ml_evaluate import time_based_split, walk_forward_evaluate
from market_briefing import ml_predictor


def frame():
    dates = pd.bdate_range("2024-01-01", periods=180).delete([50, 51, 52, 53])
    parts = []
    for ticker in ("A", "B"):
        d = pd.DataFrame({"date": dates, "ticker": ticker, "x": np.arange(len(dates)),
                          "label": np.arange(len(dates)) % 2})
        d["label_end_date"] = d.date.shift(-FORWARD_DAYS)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_date_groups_and_label_horizons_are_purged_without_test_overlap():
    folds = walk_forward_splits(frame())
    seen = set()
    assert len(folds) == 5
    for train, test in folds:
        assert train.label_end_date.max() < test.date.min()
        assert set(train.date).isdisjoint(set(test.date))
        assert seen.isdisjoint(set(test.index))
        seen.update(test.index)


def test_holdout_keeps_same_date_symbols_together():
    d = frame()
    d.loc[10, "x"] = np.nan
    _, _, _, _, train_dates, test_dates = time_based_split(d, feature_cols=["x"])
    assert train_dates.max() < test_dates.min()
    assert set(train_dates).isdisjoint(set(test_dates))


def test_failed_fold_fit_never_scores_prefit_model():
    class UnsafeModel:
        def fit(self, *args):
            raise RuntimeError("unusable")
        def predict_proba(self, *args):
            pytest.fail("Prefit model was evaluated")
    report = walk_forward_evaluate(frame(), UnsafeModel(), feature_cols=["x"])
    assert report["aggregate"] == {}
    assert all("Fold training failed" in f["error"] for f in report["folds"])


def test_predictor_accepts_numpy_and_pandas_sequences(monkeypatch):
    def predict(ticker, closes, highs, lows, volumes, market):
        assert len(closes) == 100 and isinstance(closes, list)
        return {"ok": True}
    monkeypatch.setattr(ml_predictor, "predict_from_ohlcv", predict)
    c = np.arange(100.) + 100
    assert ml_predictor.predict_direction({"ticker": "AAPL", "closes": c,
        "highs": pd.Series(c + 1), "lows": c - 1, "volumes": np.ones(100)}) == {"ok": True}


@pytest.mark.parametrize("metadata", [
    {"validation": {"passed": False}},
    {"data_provenance": {"synthetic_present": True}},
    {"data_provenance": {"known": False}},
    {"experimental_override": True},
])
def test_runtime_rejects_unvalidated_artifacts_before_loading(tmp_path, monkeypatch, metadata):
    import json
    (tmp_path / ml_predictor.METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(ml_predictor, "_find_model_dir", lambda: tmp_path)
    monkeypatch.setattr(ml_predictor, "_MODEL", None)
    monkeypatch.setattr(ml_predictor, "_MODEL_AVAILABLE", False)
    monkeypatch.setattr(ml_predictor, "_MODEL_TYPE", "none")
    monkeypatch.setattr(ml_predictor, "_MODEL_META", {})
    monkeypatch.setattr(ml_predictor, "_FEATURE_COLS", [])
    monkeypatch.setattr(ml_predictor, "_CALIB_PARAMS", None)
    monkeypatch.setattr(ml_predictor, "_ACTION_CONFIDENCE_MIN", .5)
    assert ml_predictor.load_model(force_reload=True) is False
    assert not ml_predictor.is_model_available()
