#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""backtest_vcp.py — VCP 피벗 돌파 오프라인 백테스트.

`datasets/audit_cache` 52종목 일봉에서 신선 돌파(PASS + bars_since_cross==0)를 찾아
진입(다음 봉 종가) 후 5/10/20일 전방수익률을 비용 차감 전후로 집계한다.
생존편향·상승표본 한계는 audit와 동일. 점수 반영 전 근거용.

Usage:
  python scripts/backtest_vcp.py --offline
  python scripts/backtest_vcp.py --offline --out /tmp/vcp.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from market_briefing.vcp import detect_vcp

CACHE_DIR = os.path.join(REPO_ROOT, "datasets", "audit_cache")
COSTS = {"KRX": 0.2, "US": 0.1}


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
    v = col("volume")
    out = pd.DataFrame({"c": c, "h": h, "l": l,
                        "v": v if v is not None else 0.0}).dropna().reset_index(drop=True)
    return out if len(out) >= 100 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="캐시만 사용")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    files = sorted(Path(CACHE_DIR).glob("*.csv"))
    if not files:
        print("audit_cache 없음.")
        raise SystemExit(1)

    rows = []
    for f in files:
        d = _load(f)
        if d is None:
            continue
        mkt = _market_of(f.name)
        C, H, L = d["c"].tolist(), d["h"].tolist(), d["l"].tolist()
        V = d["v"].tolist() if "v" in d else [0.0] * len(C)
        for t in range(70, len(C) - 25):
            r = detect_vcp(C[:t + 1], H[:t + 1], L[:t + 1], V[:t + 1], "BULLISH")
            if r.get("stage") != "PASS" or r.get("bars_since_cross") != 0:
                continue
            entry = C[t + 1]
            if entry <= 0:
                continue
            for Hh in (5, 10, 20):
                if t + 1 + Hh >= len(C):
                    continue
                gross = C[t + 1 + Hh] / entry - 1
                rows.append({"market": mkt, "risk": bool(r["false_breakout_risk"]),
                             "H": Hh, "gross": gross,
                             "net": gross - COSTS[mkt] / 100.0 * 2})

    df = pd.DataFrame(rows)
    summary = {"events": int(len(df) // 3) if len(df) else 0,
               "note": "52종목 생존표본, BULLISH 가정",
               "costs": COSTS, "by_group": {}}
    for H in (5, 10, 20):
        for risk in (False, True):
            g = df[(df.H == H) & (df.risk == risk)]
            summary["by_group"][f"ALL_risk{risk}_{H}d"] = {
                "n": int(len(g)),
                "gross_mean_pct": round(float(g.gross.mean() * 100), 2) if len(g) else None,
                "gross_med_pct": round(float(g.gross.median() * 100), 2) if len(g) else None,
                "net_mean_pct": round(float(g.net.mean() * 100), 2) if len(g) else None,
                "pos_rate_pct": round(float((g.net > 0).mean() * 100), 1) if len(g) else None,
            }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
