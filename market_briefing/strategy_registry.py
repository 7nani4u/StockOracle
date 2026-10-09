"""Inventory and causal, ungated signal adapters for the common trade experiment.

Signals describe decisions at a completed close. Execution and gap rejection belong
to the simulator; no adapter reads the next open or uses fitted pruning rules.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
import ast
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .pattern_engine import RETAINED_PATTERN_REGISTRY
from .technique_prune import bucket, hybrid_conditions, pattern_conditions, drsi_conditions, leader_conditions

CANDLE_NAMES = ('Hammer', 'Marubozu', 'Bullish Engulfing', 'Bullish Harami',
                'Harami Cross', 'Piercing Line', 'Morning Star', 'Three White Soldiers',
                'Three Inside Up', 'Three Outside Up', 'Rising Three Methods',
                'Abandoned Baby Bull', 'Hikkake Bull', 'Mat Hold')
INDICATOR_NAMES = ('rsi', 'macd', 'ma', 'bb', 'adx', 'obv', 'stoch14', 'aroon', 'buy_pressure', 'psar')

def candle_id(name: str) -> str:
    return 'candle_' + re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')

TECHNIQUES = ('hybrid_breakout', 'pattern_breakout', 'dynamic_rsi', 'leader_reversal',
              'momentum_persistence', 'vcp', 'arty_smma_fractal') + tuple(
    'pattern_' + k for k, v in RETAINED_PATTERN_REGISTRY.items() if v['direction'] != 'bearish'
) + tuple(candle_id(x) for x in CANDLE_NAMES) + tuple('indicator_' + x for x in INDICATOR_NAMES)

def strategy_inventory() -> List[Dict[str, Any]]:
    rows = [{'technique': x, 'status': 'BACKTESTABLE', 'source': 'causal OHLCV adapter',
             'direction': 'long', 'exit_policy': 'common ATR stop/target unless structural stop supplied'} for x in TECHNIQUES]
    for name, reason in {
        'ml_direction': 'Historical point-in-time trained models required; current model replay leaks training history.',
        'garp': 'Point-in-time financial statements and publication dates required.',
        'news': 'Timestamped historical news and evidence snapshots required.',
        'three_signal_matrix': 'Timestamped historical news and price-position snapshots required.',
        'forecast': 'Historical model versions and training cutoffs required.',
        'sector_flow': 'Historical constituent and institutional flow snapshots required.',
        'investment_charm': 'Point-in-time fundamentals, peers and weights required.',
        'portfolio_allocation': 'Portfolio construction is not a standalone entry technique.',
        'support_resistance_fibonacci': 'Price zones alone specify no independent executable entry rule.',
        'atr': 'Volatility measure; no directional entry rule.',
        'bearish_patterns': 'Sell warnings are not executable long entries; shorting requires borrow data.',
    }.items():
        rows.append({'technique': name, 'status': 'NOT_BACKTESTABLE_WITH_OHLCV', 'reason': reason})
    return rows

def candle_conditions(pattern: Dict[str, Any]) -> Dict[str, str]:
    return {'confidence': bucket(pattern.get('conf'), [80, 90], ['low', 'medium', 'high'])}

def indicator_conditions(info: Dict[str, Any]) -> Dict[str, str]:
    return {'state': str(info.get('state') or 'na')}

def arty_conditions(info: Dict[str, Any]) -> Dict[str, str]:
    return {'retest_line': str(info.get('retest_line', 21)),
            'slope200': bucket(info.get('slope200_pct20'), [0, .1, .5], ['negative', 'flat', 'positive', 'strong'])}

@lru_cache(maxsize=1)
def _api_functions() -> Dict[str, Any]:
    """Load only pure functions from the monolithic API, without its network/app imports.

    The original function AST is reused, keeping indicator/candle/Arty rules identical.
    Missing dependencies raise visibly instead of silently fabricating zero signals.
    """
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'api' / 'index.py').read_text(encoding='utf-8'))
    names = {'detect_patterns', 'add_indicators', 'calc_indicator_signals', '_smma_values',
             '_confirmed_williams_fractals', '_arty_atr_values', '_prepare_arty_series',
             '_arty_volume_ratio_at', '_arty_trend_state_at', '_arty_retest_at',
             '_arty_evaluate_setup', '_arty_evaluate_entry', '_arty_fallback_config'}
    from .dynamic_rsi import add_dynamic_rsi_features
    scope = {'np': np, 'pd': pd, 'math': math, 'Dict': Dict, 'List': List, 'Any': Any,
             'Optional': Optional, 'add_dynamic_rsi_features': add_dynamic_rsi_features,
             'dynamic_rsi_signal_card': lambda *a, **kw: None,
             '_fmt_plain_price': lambda x, market='US': f'{x:.2f}'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for function in functions:
        if function.name == 'calc_indicator_signals':
            # Stop before runtime pruning/aggregate presentation. Both are
            # intentionally excluded from raw individual-technique experiments.
            cutoff = next(i for i,node in enumerate(function.body)
                          if (isinstance(node, ast.ImportFrom) and node.module == 'market_briefing.technique_prune')
                          or any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id == 'classify_market_state' for x in ast.walk(node)))
            function.body = function.body[:cutoff] + [ast.Return(value=ast.Name(id='signals',ctx=ast.Load()))]
    selected = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(selected, '<StockOracle pure API adapters>', 'exec'), scope)
    return scope

def _atr(ohl, t):
    if t < 14:
        return None
    h, l, c = ohl['highs'], ohl['lows'], ohl['closes']
    value = sum(max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(t-13, t+1))/14
    return value if math.isfinite(value) and value > 0 else None

def generate_signals(ticker: str, market: str, ohl: Dict[str, Any], benchmark=None, *, start_index: int = 14):
    """Yield all raw strategy decisions with aligned causal benchmark prefixes.

    A benchmark must already be date-aligned to the stock. Unaligned length is
    rejected, never repaired by suffix slicing (which would import future dates).
    """
    from .hybrid_signals import compute_hybrid_score
    from .leader_reversal import detect_leader_reversal
    from .momentum_persistence import detect_momentum_persistence
    from .vcp import detect_vcp
    from .dynamic_rsi import add_dynamic_rsi_features, config_for_market
    from .pattern_engine import PatternEngine, PatternEngineOptions
    from .technique_prune import momentum_conditions, vcp_conditions, context_conditions
    keys = ('opens', 'highs', 'lows', 'closes', 'volumes')
    n = len(ohl['closes'])
    if any(len(ohl[k]) != n for k in keys):
        raise ValueError('OHLCV lengths must match')
    if benchmark is not None and len(benchmark) != n:
        raise ValueError('Benchmark must be aligned by stock trading date')
    api = _api_functions()
    frame = pd.DataFrame({k: ohl[v] for k,v in zip(('Open','High','Low','Close','Volume'), keys)})
    enriched = api['add_indicators'](frame.copy(), market)
    dynamic = add_dynamic_rsi_features(frame.copy(), market, config_for_market(market))
    last_keys = set()
    for t in range(max(14, start_index), n):
        # Invalid rows retain their position and invalidate a prefix rather than
        # compressing the confirmation period by dropping missing trading days.
        if not all(math.isfinite(float(ohl[k][t])) for k in keys):
            continue
        p = {k: list(ohl[k][:t+1]) for k in keys}
        if not all(math.isfinite(float(x)) for k in keys for x in p[k]):
            continue
        close, atr = p['closes'][-1], _atr(ohl, t)
        if not atr or close <= 0:
            continue
        bench = list(benchmark[:t+1]) if benchmark is not None else []
        candidates = []
        def add(technique, stop=None, conditions=None, target=None, identity=None):
            stop = close-atr if stop is None else stop
            if stop is None or not math.isfinite(float(stop)) or not 0 < float(stop) < close:
                return
            candidates.append((identity or technique, {'ticker':ticker, 'market':market,
                'technique':technique, 'signal_index':t, 'stop_price':float(stop),
                'target_price':target, 'atr':atr, 'conditions':{**context_conditions(p['closes'],p['highs'],p['lows'],p['volumes']), **(conditions or {})}}))
        if t >= 60:
            hs = compute_hybrid_score(p['closes'],p['highs'],p['lows'],p['volumes'],open_prices=p['opens'],bench_closes=bench)
            if hs.get('action') == 'AUTO_YES': add('hybrid_breakout',hs.get('stop_price'),hybrid_conditions(hs))
            for technique, detector, mapper in [('momentum_persistence',detect_momentum_persistence,momentum_conditions),('vcp',detect_vcp,vcp_conditions)]:
                r = detector(p['closes'],p['highs'],p['lows'],p['volumes'])
                if r.get('stage') == 'PASS' and r.get('entry_eligible',True):
                    add(technique,r.get('stop_price'),mapper(r),identity=(technique,r.get('surge_index', r.get('pivot_index')),r.get('pivot')))
        if t >= 130:
            r=detect_leader_reversal(p['closes'],p['highs'],p['lows'],bench_closes=bench)
            if r.get('stage') == 'BREAKOUT': add('leader_reversal',r.get('stop_price'),leader_conditions(r))
        if t >= 60:
            row=dynamic.iloc[t]
            if row.get('DRSI_Signal',0) == 1:
                stop=row.get('DRSI_Stop')
                add('dynamic_rsi',stop,drsi_conditions(market,row.get('RSI'),(close-stop)/close*100))
        if t >= 150:
            patterns=PatternEngine(p['opens'],p['highs'],p['lows'],p['closes'],p['volumes'],options=PatternEngineOptions(timeframe='1D',include_forming=False)).detect()
            for r in patterns:
                if r.get('signal') == '매수' and r.get('pattern_status') == 'confirmed':
                    pid = r['id'].split(':')[0]
                    # Engine IDs are stable type IDs (underscore suffix coordinates
                    # are not included in the registry key).
                    pid=next((x for x in RETAINED_PATTERN_REGISTRY if r['id'].startswith(x)),pid)
                    for name in ('pattern_breakout','pattern_'+pid):
                        if name in TECHNIQUES: add(name,r.get('invalidation_price'),pattern_conditions(r),identity=(name,r.get('start_index'),r.get('end_index')))
        dd={k: enriched[k].iloc[:t+1].where(pd.notna(enriched[k].iloc[:t+1]),None).tolist() for k in enriched.columns}
        # NaN indicators are unavailable, not ordinary numerical signals.
        dd={k:[None if isinstance(x,float) and not math.isfinite(x) else x for x in v] for k,v in dd.items()}
        for name,r in api['calc_indicator_signals'](dd,market).items():
            if name in INDICATOR_NAMES and isinstance(r,dict) and r.get('signal') == '매수': add('indicator_'+name,conditions=indicator_conditions(r))
        for r in api['detect_patterns'](dd):
            if r.get('direction') == '상승':
                name=next((x for x in CANDLE_NAMES if r['name'].endswith(x)),None)
                if name: add(candle_id(name),min(p['lows'][-5:]),candle_conditions(r))
        if t >= 219:
            series=api['_prepare_arty_series'](dd)
            fractals=api['_confirmed_williams_fractals'](p['highs'],p['lows'])
            cfg=api['_arty_fallback_config'](market)
            for fractal in fractals['lower'][-1:]:
                if fractal['confirmed_index'] != t: continue
                setup=api['_arty_evaluate_setup'](series,fractal['index'],t,cfg)
                entry=api['_arty_evaluate_entry'](series,fractal['index'],t,close,cfg)
                if setup['passed'] and entry['passed']:
                    add('arty_smma_fractal',entry['stop'],arty_conditions(setup['trend']))
        current={key for key,_ in candidates}
        for key,signal in candidates:
            if key not in last_keys: yield signal
        last_keys=current

iter_signals = generate_signals
