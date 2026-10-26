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


@pytest.fixture
def isolated_model_state(monkeypatch):
    # load_model mutates module state: restore it after each artifact experiment.
    for name, value in {'_MODEL': None, '_MODEL_AVAILABLE': False, '_MODEL_TYPE': 'none',
                        '_MODEL_META': {}, '_FEATURE_COLS': [], '_CALIB_PARAMS': None,
                        '_ACTION_CONFIDENCE_MIN': .5}.items():
        monkeypatch.setattr(ml, name, value)


@pytest.fixture
def approved_artifact(tmp_path, monkeypatch, isolated_model_state):
    metadata = {'validation': {'passed': True},
                'data_provenance': {'known': True, 'synthetic_present': False},
                'action_threshold': {'confidence_min': .62},
                'calibration': {'params': {'a': -1, 'b': 0}}}
    (tmp_path / ml.METADATA_FILENAME).write_text(json.dumps(metadata), encoding='utf-8')
    (tmp_path / ml.COLUMNS_FILENAME).write_text('["x"]', encoding='utf-8')
    monkeypatch.setattr(ml, '_find_model_dir', lambda: tmp_path)
    return tmp_path


@pytest.mark.parametrize('loader', ['lightgbm', 'sklearn'])
def test_explicitly_approved_real_artifact_loads_and_preserves_calibration(
        approved_artifact, monkeypatch, loader):
    (approved_artifact / ml.MODEL_FILENAME).touch()
    model = object()
    monkeypatch.setattr(ml, '_try_load_lightgbm',
                        lambda _: (model, loader) if loader == 'lightgbm' else (None, None))
    monkeypatch.setattr(ml, '_try_load_sklearn', lambda _: (model, loader))
    assert ml.load_model(force_reload=True)
    assert ml.is_model_available() and ml._MODEL is model
    assert ml.get_feature_columns() == ['x']
    assert ml._CALIB_PARAMS == {'a': -1., 'b': 0.}
    assert ml._ACTION_CONFIDENCE_MIN == .62
    # An already loaded artifact must not be read or deserialized again.
    monkeypatch.setattr(ml, '_find_model_dir', lambda: pytest.fail('Unnecessary model reload'))
    assert ml.load_model()


@pytest.mark.parametrize('present', [False, True])
def test_approved_but_missing_or_unreadable_binary_is_unavailable(
        approved_artifact, monkeypatch, present):
    if present:
        (approved_artifact / ml.MODEL_FILENAME).touch()
    monkeypatch.setattr(ml, '_try_load_lightgbm', lambda _: (None, None))
    monkeypatch.setattr(ml, '_try_load_sklearn', lambda _: (None, None))
    assert ml.load_model(force_reload=True) is False
    assert ml.is_model_available() is False
    assert ml.get_model_metadata()['validation']['passed'] is True


def test_approved_json_booster_loads_without_pickle(approved_artifact, monkeypatch):
    import sys
    from types import SimpleNamespace
    (approved_artifact / ml.MODEL_JSON_FILENAME).touch()
    model = object()
    monkeypatch.setitem(sys.modules, 'lightgbm', SimpleNamespace(Booster=lambda **_: model))
    assert ml.load_model(force_reload=True)
    assert ml._MODEL is model and ml._MODEL_TYPE == 'lightgbm'


@pytest.mark.parametrize('raw,expected', [([[.05, .15, .8]], .75), ([.8], .8), ([2.], .880797)])
def test_booster_probabilities_and_binary_margins_are_decoded_once(monkeypatch, raw, expected):
    calls = []
    class Booster:
        def predict(self, X):
            calls.append(X.copy())
            return np.array(raw)
    monkeypatch.setattr(ml, '_MODEL', Booster())
    monkeypatch.setattr(ml, '_MODEL_TYPE', 'lightgbm')
    conditional, classes = ml._predict_probability_details([1])
    assert conditional == pytest.approx(expected, abs=1e-6)
    assert len(calls) == 1
    assert sum(classes.values()) == pytest.approx(1)


@pytest.mark.parametrize('raw,classes', [
    ([.2, .8], [0, 1]),                 # missing sample dimension
    ([[.2, .8], [.2, .8]], [0, 1]),     # extra samples
    ([[np.nan, .8]], [0, 1]),
    ([[-.2, 1.2]], [0, 1]),
    ([[.2, .7]], [0, 1]),               # mass does not sum to one
    ([[.2, .8]], [0, 0]),               # duplicate classes
    ([[.2, .8]], [0, 3]),               # unknown label
    ([[.2, .8]], [0]),                  # class/column mismatch
])
def test_invalid_model_probabilities_fail_to_neutral_without_class_claims(monkeypatch, raw, classes):
    class Model:
        classes_ = np.array(classes)
        def predict_proba(self, X):
            return np.array(raw)
    monkeypatch.setattr(ml, '_MODEL', Model())
    assert ml._predict_probability_details([1]) == (.5, None)


def live_result():
    return ml.predict_from_ohlcv('TEST', [100] * 80, [101] * 80, [99] * 80,
                                 [1000] * 80, index_cache={})


@pytest.mark.parametrize('probabilities,classes,direction,conditional,neutral', [
    ([.8, .2], [1, 0], 'UP', .8, 0.),
    ([.1, .8, .1], [2, 0, 1], 'DOWN', 1 / 9, .1),
    ([1.], [0], 'DOWN', 0., 0.),
    ([1.], [1], 'UP', 1., 0.),
    ([1.], [2], 'NEUTRAL', .5, 1.),
    ([0., 0., 1.], [0, 1, 2], 'NEUTRAL', .5, 1.),
])
def test_action_keeps_unconditional_class_mass_and_conditional_direction_separate(
        monkeypatch, probabilities, classes, direction, conditional, neutral):
    prepare_model(monkeypatch, probabilities, classes)
    result = live_result()
    assert result['direction'] == direction
    assert result['prob_up_raw'] == pytest.approx(conditional, abs=.001)
    assert result['prob_up'] + result['prob_down'] == pytest.approx(1)
    assert result['prob_neutral'] == neutral
    assert result['probability_basis'] == 'directional_conditional'
    assert result['fallback'] is False


def test_public_inference_reuses_one_prediction_for_both_probability_views(monkeypatch):
    prepare_model(monkeypatch, [.1, .7, .2], [0, 1, 2])
    calls = []
    original = ml._MODEL.predict_proba
    monkeypatch.setattr(ml._MODEL, 'predict_proba', lambda X: (calls.append(1), original(X))[1])
    result = live_result()
    assert calls == [1]
    assert result['class_probabilities'] == {'DOWN': .1, 'UP': .7, 'NEUTRAL': .2}


def test_calibration_applies_to_directional_odds_without_rewriting_class_mass(monkeypatch):
    prepare_model(monkeypatch, [.1, .7, .2], [0, 1, 2])
    monkeypatch.setattr(ml, '_CALIB_PARAMS', {'a': 0., 'b': 0.})
    monkeypatch.setattr(ml, '_ACTION_CONFIDENCE_MIN', .62)
    result = live_result()
    assert result['prob_up_raw'] == .875
    assert result['prob_up'] == .5
    assert result['confidence'] == .5
    assert result['direction'] == 'NEUTRAL'
    assert result['class_probabilities'] == {'DOWN': .1, 'UP': .7, 'NEUTRAL': .2}


def test_valid_binary_confidence_obeys_volatility_shrink_and_action_threshold(monkeypatch):
    prepare_model(monkeypatch, [.2, .8], [0, 1])
    monkeypatch.setattr(ml, 'compute_feature_vector', lambda *a, **k: {'x': 1, 'ATR_14': 6})
    result = live_result()
    assert result['direction'] == 'UP'
    assert result['confidence'] == .77
    assert result['prob_up'] == .8
    monkeypatch.setattr(ml, '_ACTION_CONFIDENCE_MIN', .78)
    assert live_result()['direction'] == 'NEUTRAL'


def test_missing_model_uses_explicit_heuristic_basis_without_network(monkeypatch, isolated_model_state):
    monkeypatch.setattr(ml, 'load_model', lambda: False)
    monkeypatch.setattr(ml, 'compute_feature_vector', lambda *a, **k: {'RSI_14': 25, 'MACD_hist': 1})
    result = live_result()
    assert result['fallback'] is True
    assert result['probability_basis'] == 'heuristic'
    assert result['prob_neutral'] is None and result['class_probabilities'] is None
    assert result['prob_up'] + result['prob_down'] == pytest.approx(1)


def test_no_model_directory_returns_unavailable_and_creates_local_training_destination(
        tmp_path, monkeypatch, isolated_model_state):
    monkeypatch.setattr(ml, '_find_model_dir', lambda: None)
    monkeypatch.setattr(ml, '_PROJECT_ROOT', tmp_path)
    assert ml.load_model() is False
    assert not ml.is_model_available()
    assert (tmp_path / 'models').is_dir()


def test_failed_json_booster_deserialization_cannot_mark_model_available(approved_artifact, monkeypatch):
    import sys
    from types import SimpleNamespace
    (approved_artifact / ml.MODEL_JSON_FILENAME).touch()
    def reject(**kwargs):
        raise ValueError('Corrupt model artifact')
    monkeypatch.setitem(sys.modules, 'lightgbm', SimpleNamespace(Booster=reject))
    assert ml.load_model(force_reload=True) is False
    assert not ml.is_model_available()


@pytest.mark.parametrize('prediction', [0, 1])
def test_legacy_predict_only_estimator_remains_explicit_without_class_probabilities(monkeypatch, prediction):
    class Model:
        def predict(self, X):
            return np.array([prediction])
    monkeypatch.setattr(ml, '_MODEL', Model())
    monkeypatch.setattr(ml, '_MODEL_TYPE', 'sklearn')
    assert ml._predict_probability_details([1]) == (.75 if prediction else .25, None)


def test_decision_function_estimator_respects_reversed_class_order(monkeypatch):
    class Model:
        classes_ = np.array([1, 0])
        def decision_function(self, X):
            return np.array([np.log(4)])  # P(class 0)=.8, so P(UP)=.2
    monkeypatch.setattr(ml, '_MODEL', Model())
    monkeypatch.setattr(ml, '_MODEL_TYPE', 'sklearn')
    conditional, classes = ml._predict_probability_details([1])
    assert conditional == pytest.approx(.2)
    assert classes == pytest.approx({'UP': .2, 'DOWN': .8, 'NEUTRAL': 0})


def test_no_estimator_returns_neutral_without_fabricated_class_probabilities(monkeypatch):
    monkeypatch.setattr(ml, '_MODEL', None)
    assert ml._predict_probability_details([1]) == (.5, None)
