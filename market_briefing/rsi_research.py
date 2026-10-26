"""Offline RSI replacement experiments; never changes runtime indicators.

Laguerre recurrence is independently implemented from its mathematical definition.
All oscillator outputs use 0..100; no adaptive gamma or future-price fitting.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def laguerre_rsi(close: pd.Series, gamma: float = .5, warmup: int = 64) -> pd.Series:
    if not math.isfinite(gamma) or not 0 <= gamma < 1 or warmup < 0:
        raise ValueError('gamma must be finite in [0,1); warmup must be nonnegative')
    output = np.full(len(close), np.nan)
    state, count = None, 0
    for i, raw in enumerate(pd.to_numeric(close, errors='coerce')):
        if not math.isfinite(raw) or raw <= 0:
            state, count = None, 0
            continue
        if state is None:
            state = (raw, raw, raw, raw)
        old0, old1, old2, old3 = state
        l0 = (1-gamma) * raw + gamma * old0
        l1 = -gamma * l0 + old0 + gamma * old1
        l2 = -gamma * l1 + old1 + gamma * old2
        l3 = -gamma * l2 + old2 + gamma * old3
        differences = (l0-l1, l1-l2, l2-l3)
        up = sum(max(0., x) for x in differences)
        total = sum(abs(x) for x in differences)
        count += 1
        if count > warmup:
            output[i] = min(100., max(0., 100 * up / total)) if total > abs(raw) * 1e-14 else 50.
        state = (l0, l1, l2, l3)
    return pd.Series(output, index=close.index, name=close.name)


def valid_segments(frame: pd.DataFrame) -> tuple[list[pd.DataFrame], dict]:
    """Preserve bad-row boundaries; never stitch dates across missing prices."""
    keys = ['Open', 'High', 'Low', 'Close', 'Volume']
    d = frame[keys].apply(pd.to_numeric, errors='coerce').copy()
    if d.index.has_duplicates or not d.index.is_monotonic_increasing:
        raise ValueError('Daily dates must be unique and increasing')
    numeric = np.isfinite(d.to_numpy()).all(axis=1) & (d[keys[:4]] > 0).all(axis=1) & (d.Volume >= 0)
    high_needed = d[['Open', 'Close', 'Low']].max(axis=1)
    low_needed = d[['Open', 'Close', 'High']].min(axis=1)
    high_gap = (high_needed-d.High).clip(lower=0) / high_needed
    low_gap = (d.Low-low_needed).clip(lower=0) / low_needed
    valid = numeric & (high_gap <= 1e-6) & (low_gap <= 1e-6)
    repair = valid & ((high_gap > 0) | (low_gap > 0))
    d.loc[valid, 'High'] = d.loc[valid, 'High'].combine(high_needed[valid], max)
    d.loc[valid, 'Low'] = d.loc[valid, 'Low'].combine(low_needed[valid], min)
    segments, start = [], None
    for i, ok in enumerate(valid.to_numpy()):
        if ok and start is None:
            start = i
        if start is not None and (not ok or i == len(d)-1):
            end = i if not ok else i+1
            segments.append(d.iloc[start:end].copy())
            start = None
    return segments, {'rows': len(d), 'invalid_rows': int((~valid).sum()),
                      'repaired_rows': int(repair.sum()), 'segments': len(segments)}


def current_rsi(close: pd.Series) -> pd.Series:
    """Exactly the current API's RSI14 calculation, including zero-loss NaNs."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.)).ewm(alpha=1/14, adjust=False).mean()
    return 100 - 100/(1 + gain/loss.replace(0, np.nan))


def entry_signals(frame: pd.DataFrame, oscillator: pd.Series, threshold: float = 30,
                  dynamic: bool = False, market: str = 'US') -> list[dict]:
    if len(frame) != len(oscillator) or not frame.index.equals(oscillator.index):
        raise ValueError('Oscillator must align with daily frame')
    previous = frame.Close.shift(1)
    tr = pd.concat([frame.High-frame.Low, (frame.High-previous).abs(),
                    (frame.Low-previous).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    if dynamic:
        from .dynamic_rsi import add_dynamic_rsi_features
        enriched = frame.copy()
        enriched['RSI'] = oscillator
        enriched = add_dynamic_rsi_features(enriched, market=market)
        trigger = enriched.DRSI_Signal.eq(1)
    else:
        trigger = (oscillator.shift(1) <= threshold) & (oscillator > threshold)
    rows = []
    for i in np.flatnonzero(trigger.to_numpy()):
        if i < 126 or not math.isfinite(atr.iloc[i]) or atr.iloc[i] <= 0:
            continue
        close, a = float(frame.Close.iloc[i]), float(atr.iloc[i])
        if close - 2*a <= 0:
            continue
        rows.append({'signal_index': int(i), 'technique': 'oscillator_entry',
                     'stop_price': close-2*a, 'target_price': close+4*a, 'atr': a})
    return rows


def paired_bootstrap(candidate, baseline, samples: int = 2000, seed: int = 20261027) -> dict:
    a, b = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite aligned symbol-level results required')
    if not len(a):
        return {'symbols': 0, 'mean_difference_pct': None, 'ci95': None}
    differences = a-b
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(a), size=(samples, len(a)))].mean(axis=1)
    return {'symbols': len(a), 'mean_difference_pct': float(differences.mean()),
            'ci95': [float(x) for x in np.quantile(means, [.025, .975])]}


def trade_summary(trades: list[dict], symbols: list[str], extra_cost: float = 0.) -> dict:
    returns = [float(t['return_pct'])-extra_cost for t in trades]
    gains, losses = sum(max(0, x) for x in returns), -sum(min(0, x) for x in returns)
    by_symbol, drawdowns = {}, []
    for symbol in symbols:
        equity = peak = 1.
        dd = 0.
        for trade in sorted((t for t in trades if t['ticker'] == symbol), key=lambda t: t['entry_date']):
            equity *= max(1e-12, 1+(trade['return_pct']-extra_cost)/100)
            peak = max(peak, equity)
            dd = min(dd, (equity/peak-1)*100)
        by_symbol[symbol] = (equity-1)*100
        drawdowns.append(dd)
    return {'trades': len(trades), 'win_rate_pct': 100*sum(x>0 for x in returns)/len(returns) if returns else None,
            'mean_trade_return_pct': float(np.mean(returns)) if returns else None,
            'profit_factor': gains/losses if losses > 0 else None,
            'equal_symbol_return_pct': float(np.mean(list(by_symbol.values()))) if symbols else None,
            'mean_completed_trade_drawdown_pct': float(np.mean(drawdowns)) if symbols else None,
            'training_log_equity': float(np.mean([math.log(max(1e-12, 1+x/100)) for x in by_symbol.values()])) if symbols else -1e6,
            'by_symbol': by_symbol}
