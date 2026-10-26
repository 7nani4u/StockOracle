"""Reproduce frozen RSI/Laguerre entry experiments without changing runtime defaults."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_briefing.rsi_research import (
    laguerre_rsi, valid_segments, current_rsi, entry_signals, paired_bootstrap, trade_summary,
)
from market_briefing.dynamic_rsi import _wilder_rsi
from market_briefing.research_data import frame_to_ohlcv
from market_briefing.trade_replay import replay_signals

SPLITS = {'train': ('2022-01-01', '2024-01-01'),
          'validation': ('2024-01-01', '2025-01-01'),
          'test': ('2025-01-01', '2026-10-27')}
GAMMAS = (.3, .5, .7)


def select_gamma(training_scores: dict[str, float]) -> str:
    """Caller supplies TRAIN scores only. Stable name ordering breaks ties."""
    if not training_scores or not all(np.isfinite(v) for v in training_scores.values()):
        raise ValueError('Finite training scores required')
    return max(sorted(training_scores), key=lambda k: training_scores[k])


def adoption_checks(candidate: dict, baseline: dict, paired: dict,
                    validation_difference: float, stressed_difference: float) -> dict:
    ci = paired.get('ci95')
    return {
        'validation_improved': validation_difference > 0,
        'test_improved': (paired.get('mean_difference_pct') or 0) > 0,
        'paired_ci_positive': bool(ci and ci[0] > 0),
        'test_minimum_30_trades': candidate['trades'] >= 30,
        'test_profitable': (candidate.get('mean_trade_return_pct') or 0) > 0 and (candidate.get('profit_factor') or 0) > 1,
        'drawdown_not_worse': candidate['mean_completed_trade_drawdown_pct'] >= baseline['mean_completed_trade_drawdown_pct'],
        'stress_improved': stressed_difference > 0,
    }


def run(cache: Path, out: Path) -> dict:
    variants = {'wilder': ('fixed', None, 30), 'dynamic_wilder': ('dynamic', None, 50)}
    for g in GAMMAS:
        variants[f'laguerre_g{g}'] = ('fixed', g, 20)
        variants[f'laguerre_matched_g{g}'] = ('fixed', g, 30)
        variants[f'dynamic_laguerre_g{g}'] = ('dynamic', g, 50)
    symbols, manifest, trades, rejects = {'KRX': [], 'US': []}, [], [], Counter()
    for path in sorted(cache.glob('*.csv')):
        symbol = path.stem
        market = 'KRX' if symbol.endswith(('_KS', '_KQ')) else 'US'
        raw = pd.read_csv(path)
        raw.index = pd.to_datetime(raw['Date'].astype(str).str[:10], errors='raise')
        segments, quality = valid_segments(raw)
        eligible = [s for s in segments if len(s) >= 150]
        record = {'symbol': symbol, 'market': market, **quality,
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'first_date': str(raw.index[0].date()), 'last_date': str(raw.index[-1].date()),
                  'eligible_segments': len(eligible),
                  'eligible_signal_bars': sum(max(0, len(s)-126) for s in eligible),
                  'source': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)}
        manifest.append(record)
        if not eligible:
            continue
        symbols[market].append(symbol)
        for segment in eligible:
            oscillators = {None: current_rsi(segment.Close),
                           **{g: laguerre_rsi(segment.Close, g) for g in GAMMAS}}
            ohl = frame_to_ohlcv(segment)
            for name, (track, gamma, threshold) in variants.items():
                osc = _wilder_rsi(segment.Close) if name == 'dynamic_wilder' else oscillators[gamma]
                signals = entry_signals(segment, osc, threshold, dynamic=track == 'dynamic', market=market)
                for split, (start, end) in SPLITS.items():
                    rows, rejected = replay_signals(signals, ohl, symbol, market, start, end,
                        cost_pct=.2 if market == 'KRX' else .1, slippage_pct=.1, max_hold=20)
                    rejects.update({f'{name}:{split}:{k}': v for k, v in rejected.items()})
                    for row in rows:
                        row.update(variant=name, track=track, split=split)
                    trades.extend(rows)
        print(f'[rsi] {symbol}: {len(eligible)} segments, {quality["invalid_rows"]} invalid rows', flush=True)
    if not symbols['KRX'] or not symbols['US']:
        raise ValueError('Both KRX and US histories required for this protocol')
    summaries = {}
    for name in variants:
        summaries[name] = {}
        for split in SPLITS:
            summaries[name][split] = {}
            for market in ('KRX', 'US', 'ALL'):
                names = symbols[market] if market != 'ALL' else symbols['KRX'] + symbols['US']
                rows = [t for t in trades if t['variant'] == name and t['split'] == split and (market == 'ALL' or t['market'] == market)]
                summaries[name][split][market] = trade_summary(rows, names)
    decisions = {}
    for track, prefix, baseline in [('fixed', 'laguerre_g', 'wilder'),
                                     ('dynamic', 'dynamic_laguerre_g', 'dynamic_wilder')]:
        selected = select_gamma({n: summaries[n]['train']['ALL']['training_log_equity']
                                 for n in variants if n.startswith(prefix)})
        markets = {}
        for market in ('KRX', 'US'):
            a, b = summaries[selected]['test'][market], summaries[baseline]['test'][market]
            names = symbols[market]
            paired = paired_bootstrap([a['by_symbol'][s] for s in names], [b['by_symbol'][s] for s in names])
            val_difference = (summaries[selected]['validation'][market]['equal_symbol_return_pct']
                              - summaries[baseline]['validation'][market]['equal_symbol_return_pct'])
            stress = {}
            for name in (selected, baseline):
                subset = [t for t in trades if t['variant'] == name and t['split'] == 'test' and t['market'] == market]
                stress[name] = trade_summary(subset, names, extra_cost=.3)
            stress_difference = stress[selected]['equal_symbol_return_pct']-stress[baseline]['equal_symbol_return_pct']
            checks = adoption_checks(a, b, paired, val_difference, stress_difference)
            markets[market] = {'paired': paired, 'validation_difference_pct': val_difference,
                               'stress_difference_pct': stress_difference, 'checks': checks,
                               'passed': all(checks.values())}
        decisions[track] = {'selected_on_train_only': selected, 'baseline': baseline, 'markets': markets,
                            'passed': all(v['passed'] for v in markets.values())}
    passed = all(d['passed'] for d in decisions.values())
    report = {'protocol': 'docs/rsi_comparison_protocol_20261027.md', 'protocol_sha256': hashlib.sha256((ROOT/'docs/rsi_comparison_protocol_20261027.md').read_bytes()).hexdigest(),
              'as_of': '2026-10-27', 'splits': SPLITS, 'gamma_grid': GAMMAS, 'default_gamma': .5,
              'manifest': manifest, 'symbols': symbols, 'summaries': summaries, 'decisions': decisions,
              'replacement_supported_by_this_test': passed, 'runtime_changed': False,
              'rejections': dict(rejects),
              'limits': ['Historical survivor-selected cache, not fresh exchange data.',
                         'Adjusted-price provenance/corporate actions not independently reverified.',
                         'Invalid rows split segments and reset warmup; dates are never compressed.',
                         'Controlled entry comparison, not native strategy exits, portfolio or full service.',
                         'Final data previously used elsewhere in project research; not globally untouched.',
                         'Symbol bootstrap is descriptive; symbols and overlapping macro regimes are dependent.',
                         'Completed-trade equity drawdown, not daily mark-to-market drawdown.',
                         'A winning historical comparison alone cannot authorize replacement of trained ML RSI features.']}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(trades).to_csv(out.with_suffix('.trades.csv'), index=False, encoding='utf-8')
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT/'datasets/audit_cache')
    parser.add_argument('--out', type=Path, default=ROOT/'docs/backtests/rsi_laguerre_comparison_20261027.json')
    args = parser.parse_args(argv)
    report = run(args.cache, args.out)
    print(json.dumps({'decisions': report['decisions'], 'replacement_supported_by_this_test': report['replacement_supported_by_this_test']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
