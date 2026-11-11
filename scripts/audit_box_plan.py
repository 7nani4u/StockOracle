"""Walk-forward audit of the 🧭 box execution plan (``market_briefing.box_plan``).

At every completed daily bar the plan is rebuilt from that bar's prefix only (no future
rows), exactly as the service would have shown it after the close.  A ``매수`` verdict
places a one-day limit order at the top of the buy zone for the next session:

* fill at the next open if it is inside the zone, at the zone top if the bar trades down
  to it, otherwise the order expires (missed);
* an open at or below the stop cancels the order (the plan is already invalid);
* exits: half at the first target, the stop moves to the entry, the rest at the final
  target; the stop wins when a bar touches both; gaps fill at the open; 20-bar time exit;
* round-trip cost 0.25% (KRX: commission + sell tax + slippage) or 0.10% (US);
* one open position per ticker and variant.

Variants switch one rule at a time so each rule is judged by its own trades:
``service`` (defaults), ``allow_descending_support``, ``no_min_height`` and
``no_signal`` (price condition only).  A matched control buys random bars of the same
ticker with the same stop/target distances (in %) to show what the geometry earns without
the box timing.

Usage::

    python scripts/audit_box_plan.py --out docs/backtests/box_plan_audit_20261105.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from market_briefing.box_plan import VERSION as PLAN_VERSION, build_execution_plan, six_questions  # noqa: E402
from market_briefing.box_sequence import VERSION as SEQ_VERSION, analyze_box_sequence  # noqa: E402
from market_briefing.trend_boxes import analyze_trend_boxes  # noqa: E402
from scripts.audit_box_tactics import CACHE_DIR, as_dd, load_prices  # noqa: E402

HORIZON = 20
VARIANTS = {
    "service": {},
    "allow_descending_support": {"block_descending_support": False},
    "no_min_height": {"min_height_atr": 0.0},
    "no_signal": {"require_signal": False},
}


def wilson(k, n, z=1.96):
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def simulate(dd, i, entry_high, stop, targets, cost):
    """Simulate the next-session limit order and its exits. Returns a trade dict or None."""
    op, hi, lo, cl = dd["Open"], dd["High"], dd["Low"], dd["Close"]
    n = len(cl)
    f = i + 1
    if f >= n or not targets:
        return {"status": "no_bar"}
    if op[f] <= entry_high:
        if op[f] <= stop:
            return {"status": "gap_below_stop"}
        entry, at_open = op[f], True
    elif lo[f] <= entry_high:
        entry, at_open = entry_high, False
    else:
        return {"status": "missed"}
    risk = entry - stop
    if risk <= 0:
        return {"status": "bad_geometry"}
    t1, t2 = targets[0], targets[-1]
    remaining, realised = 1.0, 0.0
    cur_stop = stop
    took_t1 = len(targets) == 1
    exit_reason = "horizon"
    last = min(n - 1, f + HORIZON - 1)
    k = f
    for k in range(f, last + 1):
        if k > f:
            if op[k] <= cur_stop:
                realised += remaining * op[k]
                remaining, exit_reason = 0.0, "stop_gap"
                break
            target = t2 if took_t1 else t1
            if op[k] >= target:
                if took_t1:
                    realised += remaining * op[k]
                    remaining, exit_reason = 0.0, "target2_gap"
                    break
                realised += 0.5 * op[k]
                remaining, took_t1, cur_stop = 0.5, True, max(cur_stop, entry)
        if lo[k] <= cur_stop:
            realised += remaining * cur_stop
            remaining = 0.0
            exit_reason = "breakeven" if took_t1 else "stop"
            break
        if k == f and not at_open:
            continue  # intraday limit fill: the bar's high may have printed before the fill
        if not took_t1 and hi[k] >= t1:
            if len(targets) == 1:
                realised += remaining * t1
                remaining, exit_reason = 0.0, "target1"
                break
            realised += 0.5 * t1
            remaining, took_t1, cur_stop = 0.5, True, max(cur_stop, entry)
        if took_t1 and remaining > 0 and hi[k] >= t2:
            realised += remaining * t2
            remaining, exit_reason = 0.0, "target2"
            break
    if remaining > 0:
        realised += remaining * cl[k]
    r = (realised - entry - cost * entry) / risk
    return {"status": "filled", "entry": entry, "risk_pct": risk / entry * 100, "r": r,
            "exit": exit_reason, "bars": k - f + 1, "fill_index": f, "exit_index": k,
            "return_pct": (realised / entry - 1 - cost) * 100}


def summarise(trades):
    out = {}
    groups = defaultdict(list)
    for t in trades:
        groups["all"].append(t)
        groups["strategy:" + t["strategy"]].append(t)
        groups["relation:" + str(t["relation"])].append(t)
        groups["big_box:" + str(t["big_box"])].append(t)
        groups["height:" + ("lt2.5" if t["height_atr"] < 2.5 else "ge2.5")].append(t)
        groups["market:" + t["market"]].append(t)
    for key, items in sorted(groups.items()):
        rs = [t["r"] for t in items]
        n = len(rs)
        wins = sum(1 for r in rs if r > 0)
        mean = sum(rs) / n
        sd = (sum((r - mean) ** 2 for r in rs) / (n - 1)) ** 0.5 if n > 1 else None
        exits = defaultdict(int)
        for t in items:
            exits[t["exit"]] += 1
        out[key] = {
            "n": n, "win_rate": round(wins / n, 4), "win_ci": wilson(wins, n),
            "mean_r": round(mean, 4),
            "mean_r_ci": [round(mean - 1.96 * sd / n ** 0.5, 4), round(mean + 1.96 * sd / n ** 0.5, 4)] if sd else None,
            "median_r": round(sorted(rs)[n // 2], 4),
            "mean_return_pct": round(sum(t["return_pct"] for t in items) / n, 3),
            "mean_bars": round(sum(t["bars"] for t in items) / n, 1),
            "exits": dict(exits),
        }
    return out


def run(files, stride, seed):
    rng = random.Random(seed)
    trades = {name: [] for name in VARIANTS}
    orders = {name: defaultdict(int) for name in VARIANTS}
    verdicts = defaultdict(int)
    controls = []
    snapshots = 0
    coverage = {"snapshots": 0, "entry_price_shown": 0, "all_six_answered": 0, "data_side_unavailable": 0,
                "market_side_unanswered": defaultdict(int), "unavailable_reasons": defaultdict(int),
                "entry_kinds": defaultdict(int)}
    for path in files:
        frame = load_prices(path)
        if frame is None:
            continue
        frame = frame.dropna()
        dd = as_dd(frame)
        ticker = os.path.basename(path)[:-4]
        market = "KRX" if ticker.endswith(("_KS", "_KQ")) else "US"
        cost = 0.0025 if market == "KRX" else 0.0010
        n = len(dd["Close"])
        busy_until = {name: -1 for name in VARIANTS}
        for i in range(260, n - 1, stride):
            prefix = {k: v[:i + 1] for k, v in dd.items()}
            seq = analyze_box_sequence(prefix)
            trend = analyze_trend_boxes(prefix)
            snapshots += 1
            for name, override in VARIANTS.items():
                plan = build_execution_plan(seq, current=prefix["Close"][-1], market=market, trend=trend,
                                            now=prefix["Date"][-1], params=override)
                if name == "service":
                    verdicts[plan["verdict"]["key"] + "/" + plan["stage"]["key"]] += 1
                    sq = six_questions(plan)
                    entry = (plan.get("prices") or {}).get("entry") or {}
                    coverage["snapshots"] += 1
                    coverage["entry_price_shown"] += entry.get("low") is not None
                    coverage["all_six_answered"] += bool(sq["all"])
                    coverage["entry_kinds"][entry.get("kind") or ("conditional" if entry.get("conditional") else "direct")
                                            if entry.get("low") is not None else "none"] += 1
                    if sq["data_side_unavailable"]:
                        coverage["data_side_unavailable"] += 1
                        coverage["unavailable_reasons"][plan["verdict"]["headline"].split("—", 1)[-1].strip()[:40]] += 1
                    else:
                        for key, ok in sq["answers"].items():
                            if not ok:
                                coverage["market_side_unanswered"][key] += 1
                if plan["verdict"]["key"] != "buy" or i <= busy_until[name]:
                    continue
                prices = plan.get("prices") or {}
                entry = prices.get("entry") or {}
                stop = (prices.get("stop") or {}).get("price")
                targets = [t["price"] for t in prices.get("targets") or []]
                if entry.get("high") is None or stop is None or not targets:
                    continue
                sim = simulate(dd, i, entry["high"], stop, targets, cost)
                orders[name][sim["status"]] += 1
                if sim["status"] != "filled":
                    continue
                busy_until[name] = sim["exit_index"]
                box = plan.get("box") or {}
                record = {
                    "ticker": ticker, "market": market, "date": prefix["Date"][-1],
                    "strategy": plan["strategy"]["key"], "relation": box.get("relation"),
                    "stair": (plan.get("staircase") or {}).get("key"),
                    "big_box": (plan.get("big_box") or {}).get("direction"),
                    "height_atr": box.get("height_atr") or 0.0, "planned_rr": prices.get("rr"),
                    **{k: sim[k] for k in ("entry", "risk_pct", "r", "exit", "bars", "return_pct")},
                }
                trades[name].append(record)
                if name == "service":
                    stop_pct = stop / sim["entry"] - 1
                    target_pcts = [t / sim["entry"] - 1 for t in targets]
                    for _ in range(5):
                        b = rng.randrange(260, n - 2)
                        if abs(b - i) <= HORIZON:
                            continue
                        e = dd["Open"][b + 1]
                        ctrl = simulate(dd, b, e, e * (1 + stop_pct), [e * (1 + x) for x in target_pcts], cost)
                        if ctrl["status"] == "filled":
                            controls.append({"ticker": ticker, "market": market, "strategy": record["strategy"],
                                             "relation": record["relation"], "big_box": record["big_box"],
                                             "height_atr": record["height_atr"], **ctrl})
    report = {
        "plan_version": PLAN_VERSION, "sequence_version": SEQ_VERSION,
        "universe": {"tickers": len(files), "snapshots": snapshots, "stride": stride, "horizon_bars": HORIZON},
        "service_verdicts": dict(verdicts),
        "six_question_coverage": {k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in coverage.items()},
        "orders": {k: dict(v) for k, v in orders.items()},
        "variants": {name: summarise(items) for name, items in trades.items()},
        "matched_random_control": summarise(controls) if controls else None,
        "method": ("walk-forward prefix per completed bar; next-session limit at zone top; stop priority on "
                   "same-bar touch; 50% at T1 then stop to entry; 20-bar time exit; cost 0.25% KRX / 0.10% US"),
        "limits": ["daily bars only; intraday order of high/low is unknown",
                   "cached 52-ticker universe (2021-2026) is survivorship-biased toward current large caps",
                   "R multiples are realised outcomes on this sample, not probabilities"],
    }
    return report, trades, controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "backtests", "box_plan_audit.json"))
    parser.add_argument("--trades-csv", default="")
    args = parser.parse_args()
    files = sorted(os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith(".csv"))
    if args.tickers:
        wanted = {t.strip() for t in args.tickers.split(",") if t.strip()}
        files = [f for f in files if os.path.basename(f)[:-4] in wanted]
    report, trades, _ = run(files, args.stride, args.seed)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    if args.trades_csv:
        import csv
        with open(args.trades_csv, "w", encoding="utf-8", newline="") as handle:
            rows = [dict(variant=name, **t) for name, items in trades.items() for t in items]
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps({k: report[k] for k in ("universe", "orders")}, ensure_ascii=False))
    for name, summary in report["variants"].items():
        print(name, json.dumps(summary.get("all"), ensure_ascii=False))
    print("control", json.dumps((report["matched_random_control"] or {}).get("all"), ensure_ascii=False))


if __name__ == "__main__":
    main()
