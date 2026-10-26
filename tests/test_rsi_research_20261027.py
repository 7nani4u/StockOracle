import numpy as np
import pandas as pd
import pytest

from market_briefing.rsi_research import laguerre_rsi, valid_segments, entry_signals, paired_bootstrap


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
