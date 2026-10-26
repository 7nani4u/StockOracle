import numpy as np
import pandas as pd
import pytest

from market_briefing.rsi_research import laguerre_rsi, valid_segments, entry_signals, paired_bootstrap, trade_summary


def frame(closes):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({'Open': c, 'High': c+1, 'Low': c-1, 'Close': c, 'Volume': 1000},
                        index=pd.bdate_range('2022-01-03', periods=len(c)))


def test_gamma_zero_is_three_price_difference_ratio():
    result = laguerre_rsi(pd.Series([100., 102., 101., 104.]), gamma=0, warmup=0)
    assert result.iloc[-1] == pytest.approx(100 * 5 / 6)


def test_flat_history_is_neutral_not_an_overbought_signal():
    result = laguerre_rsi(pd.Series([100.] * 200))
    assert result.dropna().eq(50).all()


@pytest.mark.parametrize('gamma', [-1, 1, float('nan'), float('inf')])
def test_invalid_gamma_is_rejected(gamma):
    with pytest.raises(ValueError):
        laguerre_rsi(pd.Series([100.] * 20), gamma=gamma)


def test_laguerre_is_causal_scale_invariant_and_bounded():
    c = pd.Series(100 + np.sin(np.arange(220) / 4) * 5)
    full = laguerre_rsi(c)
    pd.testing.assert_series_equal(full.iloc[:150], laguerre_rsi(c.iloc[:150]))
    np.testing.assert_allclose(full, laguerre_rsi(c * 1000), atol=1e-9, equal_nan=True)
    assert full.dropna().between(0, 100).all()


def test_missing_bar_resets_state_without_compressing_dates():
    c = pd.Series([100.] * 80 + [np.nan] + [101.] * 80)
    out = laguerre_rsi(c, warmup=20)
    assert len(out) == len(c)
    assert out.iloc[80:100].isna().all()
    assert out.iloc[-1] == 50


def test_bad_bar_splits_history_and_noise_only_repairs_envelope():
    data = frame([100.] * 400)
    data.iloc[200, data.columns.get_loc('High')] = 90
    data.iloc[10, data.columns.get_loc('High')] = 100 - 1e-10
    segments, quality = valid_segments(data)
    assert quality['invalid_rows'] == 1
    assert quality['repaired_rows'] == 1
    assert [len(s) for s in segments] == [200, 199]
    assert segments[1].index[0] == data.index[201]


def test_signal_uses_completed_cross_and_no_future_rows():
    data = frame([100.] * 150)
    oscillator = pd.Series([40.] * 130 + [10., 25.] + [40.] * 18, index=data.index)
    full = entry_signals(data, oscillator, threshold=20)
    assert [s['signal_index'] for s in full] == [131]
    assert full == entry_signals(data.iloc[:140], oscillator.iloc[:140], threshold=20)
    assert full[0]['stop_price'] < 100 < full[0]['target_price']


def test_bootstrap_is_paired_reproducible_and_keeps_zero_trade_symbols():
    a = paired_bootstrap([2, 3, 0], [1, 2, 0], samples=200)
    assert a == paired_bootstrap([2, 3, 0], [1, 2, 0], samples=200)
    assert a['symbols'] == 3
    assert a['mean_difference_pct'] == pytest.approx(2/3)


def test_summary_keeps_inactive_symbols_and_cost_stress_reduces_return():
    trades = [{'ticker': 'A', 'entry_date': '2022-01-01', 'return_pct': 10},
              {'ticker': 'A', 'entry_date': '2022-02-01', 'return_pct': -5}]
    result = trade_summary(trades, ['A', 'B'])
    assert result['by_symbol']['A'] == pytest.approx(4.5)
    assert result['by_symbol']['B'] == 0
    assert result['equal_symbol_return_pct'] == pytest.approx(2.25)
    assert result['mean_completed_trade_drawdown_pct'] == pytest.approx(-2.5)
    assert trade_summary(trades, ['A', 'B'], extra_cost=.3)['equal_symbol_return_pct'] < 2.25


def test_selection_uses_supplied_training_scores_and_has_no_test_access():
    from scripts.compare_rsi_laguerre import select_gamma
    assert select_gamma({'gamma.3': .1, 'gamma.5': .2, 'gamma.7': -.1}) == 'gamma.5'
    with pytest.raises(ValueError):
        select_gamma({'bad': float('nan')})


def test_higher_returns_alone_cannot_pass_adoption():
    from scripts.compare_rsi_laguerre import adoption_checks
    candidate = {'trades': 10, 'mean_trade_return_pct': 2, 'profit_factor': 2,
                 'mean_completed_trade_drawdown_pct': -30}
    base = {'mean_completed_trade_drawdown_pct': -10}
    out = adoption_checks(candidate, base, {'mean_difference_pct': 10, 'ci95': [-1, 21]}, 1, -2)
    assert out['test_improved'] is True
    assert out['test_minimum_30_trades'] is False
    assert out['paired_ci_positive'] is False
    assert out['drawdown_not_worse'] is False
    assert out['stress_improved'] is False


def test_dynamic_oscillator_substitution_only_changes_input_not_config():
    from market_briefing.dynamic_rsi import add_dynamic_rsi_features
    data = frame(100 + np.sin(np.arange(200)/4)*5)
    oscillator = laguerre_rsi(data.Close)
    actual = entry_signals(data, oscillator, dynamic=True)
    expected = add_dynamic_rsi_features(data.assign(RSI=oscillator), market='US')
    trigger = np.flatnonzero(expected.DRSI_Signal.eq(1).to_numpy())
    assert [s['signal_index'] for s in actual] == [int(i) for i in trigger if i >= 126]


def test_offline_cli_comparison_serializes_both_markets_without_runtime_change(tmp_path):
    import json
    from scripts.compare_rsi_laguerre import run
    cache = tmp_path/'cache'
    cache.mkdir()
    data = frame(100 + np.sin(np.arange(180)/4)*5)
    data.index.name = 'Date'
    data.to_csv(cache/'000000_KS.csv')
    data.to_csv(cache/'TEST.csv')
    output = tmp_path/'result.json'
    report = run(cache, output)
    saved = json.loads(output.read_text(encoding='utf-8'))
    assert saved['runtime_changed'] is False
    assert report['replacement_supported_by_this_test'] is False
    assert saved['symbols'] == {'KRX': ['000000_KS'], 'US': ['TEST']}
    assert len(saved['summaries']) == 11
    assert output.with_suffix('.trades.csv').exists()
    for split in saved['splits']:
        for market in ('KRX', 'US'):
            assert 'wilder' in saved['summaries']
            assert saved['summaries']['wilder'][split][market]['by_symbol']
