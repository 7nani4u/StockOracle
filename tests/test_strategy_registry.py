"""Registry invariants: raw rule parity, honest inventory, causal alignment."""
import pytest

from market_briefing.strategy_registry import (
    TECHNIQUES, strategy_inventory, candle_id, _api_functions, generate_signals,
)


def flat(n):
    return {'opens': [100.] * n, 'highs': [102.] * n,
            'lows': [98.] * n, 'closes': [100.] * n, 'volumes': [1000.] * n}


def test_inventory_does_not_fabricate_fundamental_or_ml_backtests():
    rows = {r['technique']: r for r in strategy_inventory()}
    assert len(TECHNIQUES) == len(set(TECHNIQUES))
    assert 'indicator_stoch14' in TECHNIQUES
    assert 'pattern_double_bottom' in TECHNIQUES
    assert candle_id('Three Outside Up') in TECHNIQUES
    for key in ('ml_direction', 'garp', 'news', 'forecast'):
        assert rows[key]['status'] == 'NOT_BACKTESTABLE_WITH_OHLCV'


def test_benchmark_cannot_use_future_suffix():
    with pytest.raises(ValueError, match='aligned'):
        list(generate_signals('X', 'US', flat(15), [100.] * 20))


def test_alignment_is_never_repaired_by_dropping_rows():
    data = flat(15)
    data['lows'].pop()
    with pytest.raises(ValueError, match='lengths'):
        list(generate_signals('X', 'US', data))


def test_confirmed_fractal_uses_right_hand_confirmation_bars():
    detector = _api_functions()['_confirmed_williams_fractals']
    assert not detector([10.] * 5, [9., 8., 5., 8.])['lower']
    result = detector([10.] * 5, [9., 8., 5., 8., 9.])['lower']
    assert result == [{'index': 2, 'price': 5., 'confirmed_index': 4}]


def test_adapter_does_not_use_future_entry_prices():
    prefix = flat(151)
    baseline = list(generate_signals('X', 'US', prefix, start_index=150))
    extended = flat(152)
    extended['opens'][-1] = 10000.
    extended['highs'][-1] = 11000.
    extended['lows'][-1] = 9000.
    extended['closes'][-1] = 10000.
    replay = list(generate_signals('X', 'US', extended, start_index=150))
    assert baseline
    assert baseline == [s for s in replay if s['signal_index'] == 150]


def test_candle_detector_reuses_project_hammer_rule():
    dd = {'Open':[10.,10.,10.,10.,8.], 'High':[11.,11.,11.,10.,8.55],
          'Low':[9.,9.,9.,8.,6.], 'Close':[10.,10.,10.,9.,8.5]}
    patterns = _api_functions()['detect_patterns'](dd)
    assert any(p['name'].endswith('Hammer') and p['direction'] == '상승' for p in patterns)


def test_existing_prune_rules_do_not_enter_raw_indicator_baseline():
    import pandas as pd
    from unittest.mock import patch
    api = _api_functions()
    frame = pd.DataFrame({'Open':[100.]*125,'High':[102.]*125,'Low':[98.]*125,
                          'Close':[100.]*125,'Volume':[1000.]*125})
    dd=api['add_indicators'](frame,'US').to_dict('list')
    with patch('market_briefing.technique_prune.technique_allowed', side_effect=AssertionError('raw adapter invoked pruning')):
        signals=api['calc_indicator_signals'](dd,'US')
    assert 'ma' in signals


def test_dynamic_rsi_uses_actual_rsi_feature_not_missing_drsi_alias(monkeypatch):
    import pandas as pd
    from market_briefing import dynamic_rsi
    from market_briefing.strategy_registry import generate_signals
    def features(frame, *args, **kwargs):
        out = frame.copy()
        out["RSI"] = 55.
        out["DRSI_Stop"] = 95.
        out["DRSI_Signal"] = 0
        out.loc[60, "DRSI_Signal"] = 1
        return out
    monkeypatch.setattr(dynamic_rsi, "add_dynamic_rsi_features", features)
    data = {"opens": [100.] * 65, "highs": [101.] * 65, "lows": [99.] * 65,
            "closes": [100.] * 65, "volumes": [1000.] * 65}
    signals = list(generate_signals("X", "US", data, start_index=60))
    signal = next(s for s in signals if s["technique"] == "dynamic_rsi")
    assert signal["conditions"]["rsi_bucket"] == ">=45"
