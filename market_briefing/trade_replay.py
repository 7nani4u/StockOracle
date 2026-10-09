"""Uniform causal OHLCV trade replay for every registered long strategy.

One position per (symbol, technique). Signal at completed close, next open fill;
20 holding sessions, gaps at open, stop first on ambiguous daily candles.
No portfolio aggregate or fabricated 20 repetitions.
"""
from __future__ import annotations
import math
from collections import Counter
from typing import Any


def replay_signals(signals: list[dict], ohl: dict, ticker: str, market: str,
                   start_date: str, end_date: str, cost_pct: float,
                   slippage_pct: float = .1, max_hold: int = 20) -> tuple[list[dict], dict]:
    if max_hold < 1 or any(not math.isfinite(v) or v < 0 for v in (cost_pct, slippage_pct)):
        raise ValueError("Invalid replay parameters")
    keys = ("dates", "opens", "highs", "lows", "closes", "volumes")
    if len({len(ohl[k]) for k in keys}) != 1:
        raise ValueError("Misaligned OHLCV")
    n = len(ohl["dates"])
    last_exit: dict[str, int] = {}
    seen = set()
    rows, rejected = [], Counter()
    for signal in sorted(signals, key=lambda s: (s["signal_index"], s["technique"])):
        technique, index = signal["technique"], int(signal["signal_index"])
        if index < 0 or index + 1 >= n:
            rejected["no_next_session"] += 1
            continue
        entry_index = index + 1
        date = ohl["dates"][entry_index]
        if not start_date <= date < end_date:
            continue
        identity = (technique, index)
        if identity in seen:
            rejected["duplicate_signal"] += 1
            continue
        seen.add(identity)
        if entry_index <= last_exit.get(technique, -1):
            rejected["overlapping_position"] += 1
            continue
        raw_entry = float(ohl["opens"][entry_index])
        entry = raw_entry * (1 + slippage_pct / 100)
        stop = float(signal["stop_price"])
        atr = float(signal.get("atr") or 0)
        target = float(signal.get("target_price") or (entry + 2 * atr))
        if not all(math.isfinite(v) and v > 0 for v in (entry, stop, target)) or not stop < raw_entry or not target > entry:
            rejected["gap_or_invalid_levels"] += 1
            continue
        last = min(n - 1, entry_index + max_hold - 1)
        exit_index, raw_exit, reason = last, None, "timeout"
        ambiguous = False
        for j in range(entry_index, last + 1):
            op, hi, lo = (float(ohl[k][j]) for k in ("opens", "highs", "lows"))
            if j > entry_index and op <= stop:
                exit_index, raw_exit, reason = j, op, "stop_gap"
                break
            if j > entry_index and op >= target:
                exit_index, raw_exit, reason = j, target, "target_gap_limit"
                break
            if lo <= stop:
                ambiguous = hi >= target
                exit_index, raw_exit, reason = j, stop, "stop_first" if ambiguous else "stop"
                break
            if hi >= target:
                exit_index, raw_exit, reason = j, target, "target"
                break
        if raw_exit is None:
            # Do not count an unfinished trade as a fully realized timeout.
            if last - entry_index + 1 < max_hold:
                rejected["right_censored"] += 1
                last_exit[technique] = n - 1  # unfinished position still occupies the slot
                continue
            raw_exit = float(ohl["closes"][last])
        if ohl["dates"][exit_index] >= end_date:
            rejected["exit_outside_window"] += 1
            continue
        exit_price = raw_exit * (1 - slippage_pct / 100)
        net_pct = (exit_price / entry - 1) * 100 - cost_pct
        rows.append({"ticker": ticker, "market": market, "technique": technique,
                     "signal_index": index, "signal_date": ohl["dates"][index],
                     "entry_index": entry_index, "entry_date": date,
                     "entry_price": entry, "raw_entry_price": raw_entry,
                     "stop_price": stop, "target_price": target,
                     "exit_index": exit_index, "exit_date": ohl["dates"][exit_index],
                     "exit_price": exit_price, "raw_exit_price": raw_exit,
                     "exit_reason": reason, "ambiguous_daily_bar": ambiguous,
                     "hold_sessions": exit_index - entry_index + 1,
                     "gross_return_pct": (raw_exit / raw_entry - 1) * 100,
                     "return_pct": net_pct, "net_return": net_pct / 100,
                     "win": net_pct > 0,
                     "cost_round_trip_pct": cost_pct, "slippage_each_side_pct": slippage_pct,
                     "conditions": dict(signal.get("conditions") or {})})
        last_exit[technique] = exit_index
    return rows, dict(rejected)
