#!/usr/bin/env python3
"""One-year, same-policy subtractive experiment; records every raw trade.

Run python scripts/optimize_techniques.py --as-of 2026-10-09 --publish
Use --offline after collection; missing/rejected data is always disclosed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import math
from zoneinfo import ZoneInfo
from pathlib import Path
import shutil
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from market_briefing.research_data import collect_histories, frame_to_ohlcv, aligned_benchmark
from market_briefing.trade_replay import replay_signals
from market_briefing.subtractive_validation import evaluate_technique
from scripts.prune_loss_conditions import DEFAULT_TICKERS


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _symbol_replay(task):
    from market_briefing.strategy_registry import generate_signals
    ticker, market, ohl, bench, start, end, cost, slip = task
    first = next((i for i, date in enumerate(ohl["dates"]) if date >= start), len(ohl["dates"]))
    # Previous completed close may trigger entry on the first evaluated session.
    signals = list(generate_signals(ticker, market, ohl, bench or None, start_index=max(14, first - 1)))
    rows, rejected = replay_signals(signals, ohl, ticker, market, start, end, cost, slip)
    return ticker, rows, rejected, len(signals)


def _split_for(row, train_end, validation_end):
    entry, exit_date = row["entry_date"], row["exit_date"]
    split = "train" if entry <= train_end else "validation" if entry <= validation_end else "final"
    cutoff = train_end if split == "train" else validation_end if split == "validation" else None
    return "purged_boundary" if cutoff and exit_date > cutoff else split



def write_summary_artifacts(report, rows, out_dir):
    from market_briefing.subtractive_validation import trade_metrics
    technique_rows, condition_groups = [], {}
    for tech, markets in report["rules"].items():
        for market, node in markets.items():
            final = node.get("retained", {}).get("final", {})
            technique_rows.append({"technique": tech, "market": market, "status": node["status"],
                "validated": node.get("validated", False),
                "train_trades": node.get("baseline", {}).get("train", {}).get("count", 0),
                "validation_trades": node.get("baseline", {}).get("validation", {}).get("count", 0),
                "final_trades": final.get("count", 0),
                "final_win_rate_pct": final["win_rate"] * 100 if final.get("win_rate") is not None else None,
                "final_net_expectancy_pct": final["net_expectancy"] * 100 if final.get("net_expectancy") is not None else None,
                "final_profit_factor": final.get("profit_factor"),
                "exclusions": json.dumps(node.get("exclusions", []), ensure_ascii=False), "reason": node.get("reason")})
    for row in rows:
        split = _split_for(row, report["params"]["train_end"], report["params"]["validation_end"])
        if split == "purged_boundary":
            continue
        for key, value in row["conditions"].items():
            identity = (row["technique"], row["market"], str(key), str(value), split)
            condition_groups.setdefault(identity, []).append(row)
    conditions = []
    for (tech, market, condition, value, split), group in sorted(condition_groups.items()):
        stats = trade_metrics(group)
        conditions.append({"technique": tech, "market": market, "condition": condition, "value": value,
            "split": split, "trades": stats["count"], "losses": stats["losses"],
            "loss_rate_pct": stats["loss_rate"] * 100,
            "net_expectancy_pct": stats["net_expectancy"] * 100,
            "minimum_20_met": stats["count"] >= report["params"]["min_trades"]})
    for name, records in (("technique_summary.csv", technique_rows), ("condition_losses.csv", conditions)):
        target = out_dir / name
        with target.open("w", newline="", encoding="utf-8-sig") as handle:
            if records:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)


def run(args):
    from market_briefing.strategy_registry import TECHNIQUES, strategy_inventory
    end = pd.Timestamp(args.as_of or datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat())
    start = end - pd.DateOffset(years=1)
    if (args.min_trades < 20 or args.workers < 1 or not 0 <= args.slippage_pct < 100
            or any(not math.isfinite(value) or value < 0 for value in
                   (args.slippage_pct, args.krx_cost_pct, args.us_cost_pct))):
        raise ValueError("min-trades >=20, workers >=1 and 0<=slippage<100 required")
    start_date, end_date = start.date().isoformat(), end.date().isoformat()
    span = (end - start).days
    train_end = (start + pd.Timedelta(days=int(span * .6) - 1)).date().isoformat()
    validation_end = (start + pd.Timedelta(days=int(span * .8) - 1)).date().isoformat()
    if args.tickers:
        tickers = list(dict.fromkeys(t.strip().upper() for t in args.tickers.split(",") if t.strip()))
    else:
        tickers = [t for groups in DEFAULT_TICKERS.values() for names in groups.values() for t in names]
    if args.market != "ALL":
        tickers = [t for t in tickers if ("KRX" if t.endswith((".KS", ".KQ")) else "US") == args.market]
    if getattr(args, "limit", 0):
        limited = []
        for market in ("KRX", "US"):
            limited.extend([t for t in tickers if ("KRX" if t.endswith((".KS", ".KQ")) else "US") == market][:args.limit])
        tickers = limited
    if not tickers:
        raise ValueError("No tickers selected")
    cache = args.cache_dir or ROOT / "datasets" / "technique_research_cache"
    print(f"[research] {start_date} <= entry/exit < {end_date}; {len(tickers)} symbols; {len(TECHNIQUES)} techniques", flush=True)
    data, evidence = collect_histories(tickers + ["^KS11", "SPY"], end_date, cache,
        offline=args.offline, fallback_dirs=[ROOT / "datasets" / "audit_cache", ROOT / "datasets" / "prune_cache"], workers=args.workers)
    # Recover explicit adjustment provenance from our own immutable collection manifest for offline use.
    manifest_path = cache / "collection_manifest.json"
    if args.offline and manifest_path.exists():
        manifest = {r["symbol"]: r for r in json.loads(manifest_path.read_text(encoding="utf-8"))}
        for record in evidence:
            prior = manifest.get(record["symbol"], {})
            if record.get("source", "").startswith("cache:") and record.get("last_date") == prior.get("last_date"):
                record["adjustment_provenance"] = prior.get("adjustment_provenance", record.get("adjustment_provenance"))
                record["collection_source"] = prior.get("source")
                record["warmup_trimmed_through"] = prior.get("warmup_trimmed_through")
    else:
        _atomic_json(manifest_path, evidence)
    costs = {"KRX": args.krx_cost_pct, "US": args.us_cost_pct}
    tasks, omissions = [], []
    for ticker in tickers:
        market = "KRX" if ticker.endswith((".KS", ".KQ")) else "US"
        frame = data.get(ticker)
        benchmark = data.get("^KS11" if market == "KRX" else "SPY")
        if frame is None or benchmark is None:
            omissions.append({"ticker": ticker, "reason": "invalid_missing_symbol_or_benchmark"})
            continue
        if frame.index[0] > start or len(frame[frame.index < start]) < 60:
            omissions.append({"ticker": ticker, "reason": "insufficient_preyear_warmup"})
            continue
        tasks.append((ticker, market, frame_to_ohlcv(frame), aligned_benchmark(frame, benchmark),
                      start_date, end_date, costs[market], args.slippage_pct))
    rows, errors, rejections = [], [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(_symbol_replay, task): task[0] for task in tasks}
        for future in as_completed(pending):
            ticker = pending[future]
            try:
                ticker, trades, rejected, signals = future.result()
                rows.extend(trades)
                rejections[ticker] = rejected
                print(f"[research] {ticker}: {signals} signals, {len(trades)} completed trades", flush=True)
            except Exception as exc:
                errors.append({"ticker": ticker, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[research] {ticker}: FAILED {exc}", flush=True)
    rows.sort(key=lambda row: (row["market"], row["technique"], row["entry_date"], row["ticker"]))
    inventory = strategy_inventory()
    rules = {}
    selected_markets = ("KRX", "US") if args.market == "ALL" else (args.market,)
    for item in inventory:
        tech = item["technique"]
        rules[tech] = {}
        for market in selected_markets:
            trades = [r for r in rows if r["market"] == market and r["technique"] == tech]
            if item["status"] != "BACKTESTABLE":
                node = {"status": "NOT_BACKTESTABLE_WITH_OHLCV", "validated": False, "exclusions": [], "reason": item["reason"]}
            else:
                node = evaluate_technique(trades, train_end, validation_end, args.min_trades)
            rules[tech][market] = node
    used_symbols = {task[0] for task in tasks}
    # Record implementation/data hashes so the same date alone cannot masquerade as the same experiment.
    implementation_files = ["market_briefing/strategy_registry.py", "market_briefing/trade_replay.py",
        "market_briefing/subtractive_validation.py", "market_briefing/research_data.py",
        "market_briefing/technique_prune.py", "api/index.py", "scripts/optimize_techniques.py"]
    implementation_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in implementation_files}
    for record in evidence:
        path = cache / (record["symbol"].replace(".", "_") + ".csv")
        if path.exists():
            record["cache_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_ok = all(r.get("fresh") and r.get("adjustment_provenance") == "explicit_adjusted"
                    for r in evidence if r["symbol"] in used_symbols | {"^KS11", "SPY"})
    reproducible = bool(tasks) and source_ok and not errors
    if not reproducible:
        for markets in rules.values():
            for node in markets.values():
                node["validated"] = False
                node["publication_blocked_reason"] = "Incomplete, stale, unknown-adjustment data or adapter failure"
    report = {"version": 2, "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": end_date, "start_date": start_date, "end_date": end_date,
        "data_status": "FRESH_PARTIAL_UNIVERSE" if reproducible and omissions else "FRESH" if reproducible else "UNVALIDATED_DATA",
        "registry": inventory, "rules": rules, "collection": evidence,
        "strict_validated_only": bool(args.publish and reproducible),
        "implementation_sha256": implementation_hashes,
        "selected_symbols": tickers, "tested_symbols": sorted(used_symbols),
        "omissions": omissions, "adapter_errors": errors, "execution_rejections": rejections,
        "params": {"min_trades": args.min_trades, "min_affected_per_split": args.min_trades,
          "split": "60/20/20 calendar dates with crossing exits purged", "train_end": train_end,
          "validation_end": validation_end, "max_hold_sessions": 20,
          "entry": "next_open", "stop": "signal structural stop or 1ATR", "target": "2ATR unless structural target",
          "cost_round_trip_pct": costs, "slippage_each_side_pct": args.slippage_pct,
          "same_bar": "stop_first", "overlap": "one position per symbol and technique",
          "scope": "selected contemporary universe; not all listings or survivorship-free"},
        "total_raw_completed_trades": len(rows),
        "status_counts": dict(Counter(n["status"] for ms in rules.values() for n in ms.values())),
        "published": bool(args.publish and reproducible),
        "limitations": ["Event study; no shared-capital portfolio simulation", "Daily intrabar execution order unknown; conservative stop first",
          "Observed losses do not prove causality or future predictive advantage", "Present-day selected universe; survivorship and multiple-testing risks",
          "ML/news/fundamental historical point-in-time data not available; no fabricated OHLCV proxy results"]}
    out_dir = args.out_dir or ROOT / "docs" / "backtests" / f"technique_validation_{end_date.replace('-', '')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = out_dir / "all_trades.csv"
    fields = list(rows[0]) if rows else ["ticker", "market", "technique", "entry_date", "exit_date", "return_pct", "conditions"]
    fields += ["split", "matches_frozen_exclusion", "runtime_status"]
    with ledger.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            node = rules[row["technique"]][row["market"]]
            match = any(row["conditions"].get(e["condition"]) == e["value"] for e in node.get("exclusions", []))
            writer.writerow({**row, "conditions": json.dumps(row["conditions"], ensure_ascii=False),
                "split": _split_for(row, train_end, validation_end), "matches_frozen_exclusion": match,
                "runtime_status": node["status"]})
    write_summary_artifacts(report, rows, out_dir)
    _atomic_json(out_dir / "report.json", report)
    if args.publish:
        if not reproducible:
            raise RuntimeError(f"Publication blocked; inspect {out_dir / 'report.json'}")
        active = ROOT / "models" / "technique_prune.json"
        if active.exists():
            archive = out_dir / "previous_active_rules.json"
            if not archive.exists():
                shutil.copy2(active, archive)
        _atomic_json(active, report)
    print(f"[research] raw trades={len(rows)}, statuses={report['status_counts']}, published={report['published']}", flush=True)
    print(f"[research] report={out_dir / 'report.json'} ledger={ledger}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="", help="Exclusive end date, YYYY-MM-DD; default local today")
    parser.add_argument("--tickers", default="")
    parser.add_argument("--full", action="store_true", help="Compatibility alias; all selected techniques run by default")
    parser.add_argument("--limit", type=int, default=0, help="Optional per-market symbol limit, zero means all selected")
    parser.add_argument("--market", choices=("ALL", "KRX", "US"), default="ALL")
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--publish", action="store_true", help="Atomically activate only validated findings")
    parser.add_argument("--slippage-pct", type=float, default=.1, help="Each-side assumed slippage percentage")
    parser.add_argument("--krx-cost-pct", type=float, default=.2)
    parser.add_argument("--us-cost-pct", type=float, default=.1)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except Exception as error:
        print(f"[research] ERROR {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
