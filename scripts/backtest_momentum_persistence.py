#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""backtest_momentum_persistence.py — 급등+3일 절반수성 오프라인 백테스트.

`datasets/audit_cache` 52종목 일봉에서 +20% 급등 → 3일 종가 절반수성(PASS/FAIL)을
찾아 진입(확인일 종가) 후 5/10/20일 전방수익률을 비용 차감 전후로 집계한다.
생존편향·상승표본 한계는 audit와 동일. 점수 반영 전 근거용.

Usage:
  python scripts/backtest_momentum_persistence.py --offline
  python scripts/backtest_momentum_persistence.py --offline --out /tmp/mom.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from market_briefing.momentum_persistence import HOLD_DAYS, RETAIN_FRAC, SURGE_MIN_PCT

CACHE_DIR = os.path.join(REPO_ROOT, "datasets", "audit_cache")
COSTS = {"KRX": 0.2, "US": 0.1}  # technique_prune과 동일 왕복 가정(%)


def _market_of(name: str) -> str:
    stem = Path(name).stem
    if stem.endswith((".KS", ".KQ")) or "_KS" in stem or "_KQ" in stem:
        return "KRX"
    if len(stem) >= 6 and stem[:6].isdigit():
        return "KRX"
    return "US"


def _load(f: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(f)
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    if "close" not in cols:
        return None

    def col(*names):
        for n in names:
            if n in cols:
                return pd.to_numeric(df[cols[n]], errors="coerce")
        return None

    c = col("close")
    h = col("high") if col("high") is not None else c
    l = col("low") if col("low") is not None else c
    out = pd.DataFrame({"c": c, "h": h, "l": l}).dropna().reset_index(drop=True)
    return out if len(out) >= 30 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="캐시만 사용")
    ap.add_argument("--surge", type=float, default=SURGE_MIN_PCT)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    files = sorted(Path(CACHE_DIR).glob("*.csv"))
    if not files:
        print("audit_cache 없음. 먼저 audit용 일봉을 수집하세요.")
        raise SystemExit(1)

    rows = []
    for f in files:
        d = _load(f)
        if d is None:
            continue
        mkt = _market_of(f.name)
        closes = d["c"].tolist()
        lows = d["l"].tolist()
        for i in range(1, len(closes) - HOLD_DAYS - 20):
            if closes[i - 1] <= 0:
                continue
            surge = closes[i] / closes[i - 1] - 1
            if surge < args.surge / 100.0:
                continue
            base, top = closes[i - 1], closes[i]
            mid = base + (top - base) * RETAIN_FRAC
            nxt = closes[i + 1:i + 1 + HOLD_DAYS]
            if len(nxt) < HOLD_DAYS:
                continue
            passed = min(nxt) >= mid
            entry = closes[i + HOLD_DAYS]
            for H in (5, 10, 20):
                if i + HOLD_DAYS + H >= len(closes) or entry <= 0:
                    continue
                gross = closes[i + HOLD_DAYS + H] / entry - 1
                cost = COSTS[mkt] / 100.0 * 2  # 왕복
                rows.append({"market": mkt, "pass": bool(passed), "H": H,
                             "gross": gross, "net": gross - cost,
                             "surge": surge, "ticker": f.stem})

    df = pd.DataFrame(rows)
    summary = {"events": int(len(df) // 3) if len(df) else 0, "bars_note": "52종목 생존표본",
               "surge_min_pct": args.surge, "hold_days": HOLD_DAYS, "retain_frac": RETAIN_FRAC,
               "costs": COSTS}
    by = {}
    for (mkt, passed, H), g in df.groupby(["market", "pass", "H"]):
        by[f"{mkt}_{'PASS' if passed else 'FAIL'}_{H}d"] = {
            "n": int(len(g)),
            "gross_mean_pct": round(float(g.gross.mean() * 100), 2),
            "gross_med_pct": round(float(g.gross.median() * 100), 2),
            "net_mean_pct": round(float(g.net.mean() * 100), 2),
            "pos_rate_pct": round(float((g.net > 0).mean() * 100), 1),
        }
    for H in (5, 10, 20):
        for passed in (True, False):
            g = df[(df.H == H) & (df["pass"] == passed)]
            key = f"ALL_{'PASS' if passed else 'FAIL'}_{H}d"
            by[key] = {"n": int(len(g)),
                       "gross_mean_pct": round(float(g.gross.mean() * 100), 2) if len(g) else None,
                       "gross_med_pct": round(float(g.gross.median() * 100), 2) if len(g) else None,
                       "net_mean_pct": round(float(g.net.mean() * 100), 2) if len(g) else None,
                       "pos_rate_pct": round(float((g.net > 0).mean() * 100), 1) if len(g) else None}
    summary["by_group"] = by
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
