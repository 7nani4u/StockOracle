import json

import numpy as np
import pytest

from market_briefing import ml_predictor as ml


@pytest.mark.parametrize('end', [90, 100, 110])
def test_short_history_probabilities_sum_to_one(end):
    closes = np.linspace(100, end, 30).tolist()
    result = ml.predict_from_ohlcv('TEST', closes, closes, closes, [100] * 30)
    assert result['prob_up'] + result['prob_down'] == pytest.approx(1)


@pytest.mark.parametrize('metadata', [None, {}, [], {'validation': True},
    {'validation': {'passed': True}},
    {'validation': {'passed': True}, 'data_provenance': {'known': True}},
])
def test_missing_or_malformed_validation_never_loads_model(tmp_path, monkeypatch, metadata):
    if metadata is not None:
        (tmp_path / ml.METADATA_FILENAME).write_text(json.dumps(metadata), encoding='utf-8')
    monkeypatch.setattr(ml, '_find_model_dir', lambda: tmp_path)
    monkeypatch.setattr(ml, '_MODEL_AVAILABLE', False)
    monkeypatch.setattr(ml, '_MODEL', None)
    monkeypatch.setattr(ml, '_MODEL_TYPE', 'none')
    monkeypatch.setattr(ml, '_MODEL_META', {})
    monkeypatch.setattr(ml, '_FEATURE_COLS', [])
    monkeypatch.setattr(ml, '_CALIB_PARAMS', None)
    monkeypatch.setattr(ml, '_ACTION_CONFIDENCE_MIN', .5)
    monkeypatch.setattr(ml, '_load_columns', lambda _: pytest.fail('Unvalidated artifact read'))
    assert ml.load_model(force_reload=True) is False


def prepare_model(monkeypatch, probabilities, classes):
    class Model:
        classes_ = np.array(classes)
        def predict_proba(self, X):
            return np.array([probabilities])
    monkeypatch.setattr(ml, '_MODEL', Model())
    monkeypatch.setattr(ml, '_MODEL_AVAILABLE', True)
    monkeypatch.setattr(ml, '_MODEL_TYPE', 'sklearn')
    monkeypatch.setattr(ml, '_FEATURE_COLS', ['x'])
    monkeypatch.setattr(ml, '_CALIB_PARAMS', None)
    monkeypatch.setattr(ml, '_ACTION_CONFIDENCE_MIN', .5)
    monkeypatch.setattr(ml, 'compute_feature_vector', lambda *a, **k: {'x': 1})


def test_neutral_dominant_model_does_not_issue_up_action(monkeypatch):
    prepare_model(monkeypatch, [.01, .09, .90], [0, 1, 2])
    result = ml.predict_from_ohlcv('TEST', [100] * 80, [101] * 80, [99] * 80,
                                 [1000] * 80, index_cache={})
    assert result['direction'] == 'NEUTRAL'
    assert result['prob_neutral'] == .90
    assert result['probability_basis'] == 'directional_conditional'
    assert result['class_probabilities'] == {'DOWN': .01, 'UP': .09, 'NEUTRAL': .90}


def test_probability_columns_follow_model_class_order(monkeypatch):
    prepare_model(monkeypatch, [.8, .2], [1, 0])
    assert ml._predict_proba([1]) == .8
