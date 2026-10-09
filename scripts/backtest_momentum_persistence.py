#!/usr/bin/env python3
"""Offline event study: confirm three completed closes, enter next session open.

Returns are horizon-specific, net of assumed round-trip costs and entry slippage.
This is an overlapping event study, not a portfolio backtest or stop-loss simulation.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from market_briefing.momentum_persistence import HOLD_DAYS, RETAIN_FRAC, SURGE_MIN_PCT, find_surges
CACHE_DIR = REPO_ROOT / "datasets" / "audit_cache"
COSTS = {"KRX": 0.2, "US": 0.1}  # assumed TOTAL round-trip percent; not statutory rates


def _market_of(name: str) -> str:
    stem = Path(name).stem
    return "KRX" if (stem.endswith((".KS", ".KQ")) or "_KS" in stem or "_KQ" in stem
                     or (len(stem) >= 6 and stem[:6].isdigit())) else "US"


def _load(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).lower() for c in df.columns]
        if "close" not in df:
            return None
        if "date" in df:
            dates = pd.to_datetime(df["date"], utc=True, errors="coerce")
            if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
                return None
        # Keep gaps in position: dropping NaN would compress the trading-day sequence.
        out = pd.DataFrame({"c": pd.to_numeric(df["close"], errors="coerce"),
                            "o": pd.to_numeric(df["open"], errors="coerce") if "open" in df else float("nan")})
        return out if len(out) >= HOLD_DAYS + 3 else None
    except (ValueError, OSError, pd.errors.ParserError):
        return None


def evaluate_events(data: pd.DataFrame, market: str, ticker: str, surge_min_pct: float = SURGE_MIN_PCT,
                    entry_mode: str = "next_open", slippage_pct: float = 0.0) -> list[dict]:
    if entry_mode not in {"next_open", "confirmation_close"}:
        raise ValueError("Unknown entry mode")
    if not math.isfinite(surge_min_pct) or surge_min_pct <= 0:
        raise ValueError("Surge threshold must be finite and positive")
    if not math.isfinite(slippage_pct) or slippage_pct < 0:
        raise ValueError("Slippage must be finite and nonnegative")
    closes = data["c"].tolist()
    rows = []
    for event in find_surges(closes, surge_min_pct):
        i = event["index"]
        confirm = i + HOLD_DAYS
        if confirm >= len(data):
            continue
        window = closes[i + 1:confirm + 1]
        if not all(math.isfinite(x) and x > 0 for x in window):
            continue
        mid = closes[i - 1] + (closes[i] - closes[i - 1]) * RETAIN_FRAC
        passed = min(window) >= mid
        entry_index = confirm + 1 if entry_mode == "next_open" else confirm
        if entry_index >= len(data):
            continue
        entry = float(data.iloc[entry_index]["o" if entry_mode == "next_open" else "c"])
        if not math.isfinite(entry) or entry <= 0:
            continue
        fill = entry * (1 + slippage_pct / 100)
        for horizon in (5, 10, 20):
            # H sessions from confirmation: next session is the first holding session.
            exit_index = confirm + horizon
            if exit_index >= len(data):
                continue
            exit_price = closes[exit_index]
            if not math.isfinite(exit_price) or exit_price <= 0:
                continue
            gross = exit_price / entry - 1
            net = exit_price / fill - 1 - COSTS[market] / 100
            rows.append({"market": market, "pass": passed, "H": horizon,
                         "gross": gross, "net": net, "ticker": ticker,
                         "surge_index": i, "confirmation_index": confirm,
                         "entry_index": entry_index, "entry": entry,
                         "exit_index": exit_index, "midpoint": mid,
                         "gap_below_midpoint": entry < mid,
                         "surge": event["pct"] / 100})
    return rows


def summarize(rows: list[dict], **metadata) -> dict:
    result = {"events": len({(r["ticker"], r["surge_index"]) for r in rows}),
              "event_horizon_observations": len(rows), "costs_round_trip_pct": COSTS,
              "limitations": ["Selected cached universe; survivorship bias possible",
                              "Overlapping events; no portfolio or stop-loss simulation",
                              "Costs/slippage are assumptions; no executable quote guarantees",
                              "Corporate action adjustment provenance must be verified"], **metadata}
    by = {}
    for market in ("KRX", "US", "ALL"):
        for passed in (True, False):
            for horizon in (5, 10, 20):
                group = [r for r in rows if r["H"] == horizon and r["pass"] == passed
                         and (market == "ALL" or r["market"] == market)]
                g = pd.Series([r["gross"] for r in group], dtype=float)
                n = pd.Series([r["net"] for r in group], dtype=float)
                by[f"{market}_{'PASS' if passed else 'FAIL'}_{horizon}d"] = {
                    "n": len(group), "gross_mean_pct": float(g.mean() * 100) if group else None,
                    "gross_med_pct": float(g.median() * 100) if group else None,
                    "net_mean_pct": float(n.mean() * 100) if group else None,
                    "pos_rate_pct": float((n > 0).mean() * 100) if group else None}
    result["by_group"] = by
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="Always uses local cache only")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--surge", type=float, default=SURGE_MIN_PCT)
    ap.add_argument("--entry", choices=("next_open", "confirmation_close"), default="next_open")
    ap.add_argument("--slippage-pct", type=float, default=0.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not math.isfinite(args.surge) or args.surge <= 0 or not math.isfinite(args.slippage_pct) or args.slippage_pct < 0:
        ap.error("surge must be positive; slippage must be nonnegative and finite")
    files = sorted(args.cache_dir.glob("*.csv"))
    if not files:
        ap.error("No cached CSV files found")
    rows, loaded, skipped = [], 0, []
    for path in files:
        data = _load(path)
        if data is None:
            skipped.append(path.name)
            continue
        loaded += 1
        rows.extend(evaluate_events(data, _market_of(path.name), path.stem,
                                    args.surge, args.entry, args.slippage_pct))
    summary = summarize(rows, surge_min_pct=args.surge, hold_days=HOLD_DAYS, retain_frac=RETAIN_FRAC,
                        entry_mode=args.entry, slippage_pct=args.slippage_pct,
                        files_loaded=loaded, files_skipped=skipped,
                        same_close_execution_assumption=args.entry == "confirmation_close")
    encoded = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    print(encoded)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
