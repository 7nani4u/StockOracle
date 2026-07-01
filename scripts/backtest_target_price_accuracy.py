#!/usr/bin/env python3
"""
Backtest the target-price logic added to api/index.py in this session:
  - calc_risk(): 보수적 / 중립적 / 공격적 목표가 (target range midpoint)
  - calc_pullback_analysis(): target_main (1차 정밀 목표가) / target_ext (2차 목표)

For each signal date over the trailing ~1 year of real KRX + US daily data,
this calls the *actual* production functions (imported from api.index, not a
reimplementation), then checks whether the future High price reaches the
predicted target within a fixed forward window. It reports:
  - hit rate (accuracy) per profile
  - calibration: predicted confidence bucket vs. actual hit rate in that bucket
  - median days-to-hit vs. the model's own avg_days estimate (risk profiles only)

Writes JSON + CSV under docs/backtests/.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.index import add_indicators, calc_risk, calc_pullback_analysis, _US_RECO_UNIVERSE  # noqa: E402

KRX_TICKERS = [
    "005930.KS", "000660.KS", "373220.KS", "207940.KS", "005380.KS",
    "000270.KS", "068270.KS", "105560.KS", "055550.KS", "005490.KS",
    "035420.KS", "035720.KS", "051910.KS", "006400.KS", "012330.KS",
    "028260.KS", "012450.KS", "034020.KS", "247540.KQ", "196170.KQ",
    "009150.KS", "009540.KS", "015760.KS", "033780.KS", "086790.KS",
    "003670.KS", "259960.KS", "086520.KQ", "028300.KQ", "145020.KQ",
]

FORWARD_WINDOW = 90   # trading days to check for target touch
SIGNAL_STEP = 15      # non-overlapping-ish spacing between signal dates
LOOKBACK_MIN = 140    # min bars needed before a signal for stable indicators


def us_tickers(n: int = 30) -> List[str]:
    uni = [t for t in dict.fromkeys(_US_RECO_UNIVERSE) if t not in {"SPY", "QQQ", "DIA", "IWM"}]
    return uni[:n]


@dataclass
class Record:
    ticker: str
    market: str
    kind: str          # "risk" or "pullback"
    profile: str        # conservative / balanced / aggressive / pullback_main / pullback_ext
    signal_date: str
    price: float
    target: float
    return_pct_needed: float
    predicted_confidence: Optional[float]
    predicted_days: Optional[float]
    hit: bool
    days_to_hit: Optional[int]


def _flatten(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            raw = raw[ticker]
        elif ticker in raw.columns.get_level_values(-1):
            raw = raw.xs(ticker, level=-1, axis=1)
    raw = raw.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    return raw[keep].dropna(subset=["Open", "High", "Low", "Close"])


def download_one(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True,
                           progress=False, threads=False, timeout=20)
        df = _flatten(raw, ticker)
        if len(df) < 300:
            return pd.DataFrame()
        df = add_indicators(df.copy())
        return df.dropna(subset=["ATR", "MA20", "MA60", "MACD", "Signal_Line", "RSI"])
    except Exception as e:
        print(f"  download failed: {e}")
        return pd.DataFrame()


_DD_COLS = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "Signal_Line",
            "ADX", "DI_Plus", "DI_Minus", "BB_Upper", "BB_Lower", "BB_Middle",
            "MA20", "MA60", "MA120", "ATR", "OBV", "EMA20", "EMA50"]


def to_dd(hist: pd.DataFrame) -> Dict[str, list]:
    return {c: hist[c].tolist() for c in _DD_COLS if c in hist.columns}


def eval_one_signal(ticker: str, market: str, df: pd.DataFrame, i: int) -> List[Record]:
    hist = df.iloc[:i + 1]
    future = df.iloc[i + 1:i + 1 + FORWARD_WINDOW]
    if future.empty:
        return []
    price = float(df["Close"].iloc[i])
    atr = float(df["ATR"].iloc[i])
    if not price or not atr or math.isnan(atr):
        return []
    dd = to_dd(hist)
    highs = future["High"].to_numpy(dtype=float)
    sdate = str(df.index[i].date())

    def _check(target: float) -> tuple[bool, Optional[int]]:
        touched = highs >= target
        if not touched.any():
            return False, None
        return True, int(np.argmax(touched)) + 1

    out: List[Record] = []
    try:
        risk = calc_risk(price, atr, market=market, dd=dd)
    except Exception:
        risk = None
    if risk:
        for key, label in [("conservative", "conservative"), ("balanced", "balanced"), ("aggressive", "aggressive")]:
            sc = risk.get(key) or {}
            tgt_range = sc.get("target")
            if not tgt_range:
                continue
            tgt = (tgt_range[0] + tgt_range[1]) / 2
            hit, days = _check(tgt)
            tp = (sc.get("tp_levels") or [None, None])[1]
            out.append(Record(ticker, market, "risk", label, sdate, price, tgt,
                               round((tgt - price) / price * 100, 2),
                               sc.get("target_confidence_pct"),
                               tp["avg_days"] if tp else None,
                               hit, days))
    try:
        pb = calc_pullback_analysis(dd, price, atr, score=50.0, market=market)
    except Exception:
        pb = None
    if pb:
        for tkey, ckey, label in [("target_main", "target_main_confidence_pct", "pullback_main"),
                                   ("target_ext", "target_ext_confidence_pct", "pullback_ext")]:
            tgt = pb.get(tkey)
            if tgt is None:
                continue
            hit, days = _check(float(tgt))
            out.append(Record(ticker, market, "pullback", label, sdate, price, float(tgt),
                               round((float(tgt) - price) / price * 100, 2),
                               pb.get(ckey), None, hit, days))
    return out


def run_market(tickers: List[str], market: str) -> List[Record]:
    records: List[Record] = []
    for n, ticker in enumerate(tickers, 1):
        print(f"[{market}] {n}/{len(tickers)} {ticker}", flush=True)
        df = download_one(ticker)
        if df.empty:
            continue
        n_bars = len(df)
        start = max(LOOKBACK_MIN, n_bars - 252 - FORWARD_WINDOW)
        end = n_bars - FORWARD_WINDOW
        if end <= start:
            continue
        for i in range(start, end, SIGNAL_STEP):
            records.extend(eval_one_signal(ticker, market, df, i))
        time.sleep(0.05)
    return records


def summarize(records: List[Record]) -> Dict:
    df = pd.DataFrame([asdict(r) for r in records])
    if df.empty:
        return {}
    out: Dict = {"overall": {}, "by_confidence_bucket": {}}

    for (market, profile), g in df.groupby(["market", "profile"]):
        out["overall"][f"{market}|{profile}"] = {
            "n_signals": int(len(g)),
            "hit_rate_pct": round(g["hit"].mean() * 100, 1),
            "avg_return_needed_pct": round(g["return_pct_needed"].mean(), 2),
            "median_predicted_confidence_pct": round(g["predicted_confidence"].dropna().median(), 1) if g["predicted_confidence"].notna().any() else None,
            "median_days_to_hit": float(g.loc[g["hit"], "days_to_hit"].median()) if g["hit"].any() else None,
            "median_predicted_days": round(g["predicted_days"].dropna().median(), 1) if g["predicted_days"].notna().any() else None,
        }

    def bucket(p):
        if pd.isna(p):
            return None
        if p < 35:
            return "low(<35%)"
        if p < 60:
            return "mid(35-60%)"
        return "high(60%+)"

    df["conf_bucket"] = df["predicted_confidence"].apply(bucket)
    for profile, g in df.dropna(subset=["conf_bucket"]).groupby("profile"):
        for b, gb in g.groupby("conf_bucket"):
            out["by_confidence_bucket"][f"{profile}|{b}"] = {
                "n_signals": int(len(gb)),
                "avg_predicted_confidence_pct": round(gb["predicted_confidence"].mean(), 1),
                "actual_hit_rate_pct": round(gb["hit"].mean() * 100, 1),
                "calibration_gap_pp": round(gb["hit"].mean() * 100 - gb["predicted_confidence"].mean(), 1),
            }
    return out


def main() -> None:
    outdir = ROOT / "docs" / "backtests"
    outdir.mkdir(parents=True, exist_ok=True)
    krx = KRX_TICKERS
    us = us_tickers(30)
    records = run_market(krx, "KRX") + run_market(us, "US")
    rows_df = pd.DataFrame([asdict(r) for r in records])
    rows_df.to_csv(outdir / "target_price_backtest_trades.csv", index=False, encoding="utf-8-sig")
    result = {
        "period": f"~2y daily data, signals every {SIGNAL_STEP} trading days over the trailing ~1y, fixed {FORWARD_WINDOW}-trading-day forward window per signal",
        "krx_tickers": krx,
        "us_tickers": us,
        "summary": summarize(records),
    }
    (outdir / "target_price_backtest_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
