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
import argparse
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

try:
    from scripts.backtest_prediction_zones import KRX_FALLBACK_100
    KRX_STRATIFIED = list(dict.fromkeys(KRX_TICKERS + KRX_FALLBACK_100))
except ImportError:
    KRX_STRATIFIED = KRX_TICKERS

US_STRATIFIED_EXTRA = [
    "F", "SOFI", "SNAP", "PFE", "T", "NU", "RIVN", "LCID",
    "BKNG", "AZO", "NVR", "MELI", "ORLY", "MTD",
]

FORWARD_WINDOW = 90   # trading days to check for target touch
SIGNAL_STEP = 15      # non-overlapping-ish spacing between signal dates
LOOKBACK_MIN = 140    # min bars needed before a signal for stable indicators


def us_tickers(n: int = 30) -> List[str]:
    uni = [
        t for t in dict.fromkeys(US_STRATIFIED_EXTRA + list(_US_RECO_UNIVERSE))
        if t not in {"SPY", "QQQ", "DIA", "IWM"}
    ]
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
    stop_first: bool
    actionable: bool
    realized_return_pct: Optional[float]
    price_bucket: str
    liquidity_bucket: str
    volatility_bucket: str
    average_turnover: float
    atr_pct: float
    next_open_gap_pct: float
    trend_score: Optional[int] = None
    downside_risk_score: Optional[float] = None
    breakout_probability_pct: Optional[float] = None
    volatility_expanding: Optional[bool] = None
    strategy_eligible: bool = True


def _price_bucket(price: float, market: str) -> str:
    if market == "KRX":
        if price < 5_000: return "under_5k"
        if price < 20_000: return "5k_20k"
        if price < 100_000: return "20k_100k"
        if price < 500_000: return "100k_500k"
        return "over_500k"
    if price < 10: return "under_10"
    if price < 50: return "10_50"
    if price < 200: return "50_200"
    if price < 500: return "200_500"
    return "over_500"


def _liquidity_bucket(turnover: float, market: str) -> str:
    if market == "KRX":
        if turnover < 1_000_000_000: return "thin"
        if turnover < 10_000_000_000: return "normal"
        return "liquid"
    if turnover < 5_000_000: return "thin"
    if turnover < 50_000_000: return "normal"
    return "liquid"


def _volatility_bucket(atr_pct: float, market: str) -> str:
    low, high = ((1.5, 4.0) if market == "KRX" else (1.0, 3.0))
    return "low" if atr_pct < low else "high" if atr_pct > high else "normal"


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
    payload = {c: hist[c].tolist() for c in _DD_COLS if c in hist.columns}
    payload["Date"] = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in hist.index]
    return payload


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
    sdate = str(df.index[i].date())
    next_open = float(future["Open"].iloc[0])
    gap_pct = (next_open / price - 1.0) * 100.0 if price > 0 else 0.0
    turnover = float((hist["Close"].tail(20) * hist["Volume"].tail(20)).mean())
    atr_pct = atr / price * 100.0
    price_bucket = _price_bucket(price, market)
    liquidity_bucket = _liquidity_bucket(turnover, market)
    volatility_bucket = _volatility_bucket(atr_pct, market)
    round_trip_cost = 0.0040 if market == "KRX" else 0.0015

    def _check(target: float, stop: float, strategy_eligible: bool = True) -> tuple[bool, Optional[int], bool, bool, Optional[float]]:
        actionable = bool(strategy_eligible and stop < next_open < target)
        if not actionable:
            return False, None, False, False, None
        for day, (_, bar) in enumerate(future.iterrows(), 1):
            stop_touched = float(bar["Low"]) <= stop
            target_touched = float(bar["High"]) >= target
            # Daily bars cannot reveal intraday order; use the conservative stop-first rule.
            if stop_touched:
                realized = (stop / next_open - 1.0 - round_trip_cost) * 100.0
                return False, None, True, True, round(realized, 2)
            if target_touched:
                realized = (target / next_open - 1.0 - round_trip_cost) * 100.0
                return True, day, False, True, round(realized, 2)
        exit_price = float(future["Close"].iloc[-1])
        realized = (exit_price / next_open - 1.0 - round_trip_cost) * 100.0
        return False, None, False, True, round(realized, 2)

    def _record(kind: str, profile: str, target: float, stop: float,
                confidence: Optional[float], predicted_days: Optional[float],
                context: Optional[Dict] = None,
                strategy_eligible: bool = True) -> Record:
        hit, days, stop_first, actionable, realized = _check(target, stop, strategy_eligible)
        context = context or {}
        return Record(
            ticker, market, kind, profile, sdate, price, target,
            round((target - price) / price * 100, 2), confidence,
            predicted_days, hit, days, stop_first, actionable, realized,
            price_bucket, liquidity_bucket, volatility_bucket,
            round(turnover, 2), round(atr_pct, 3), round(gap_pct, 3),
            context.get("trend_score"),
            context.get("downside_risk_score"),
            context.get("breakout_probability_pct"),
            context.get("volatility_expanding"),
            strategy_eligible,
        )

    out: List[Record] = []
    try:
        risk = calc_risk(price, atr, market=market, dd=dd)
    except Exception:
        risk = None
    if risk:
        context = risk.get("model_context") or {}
        for key, label in [("conservative", "conservative"), ("balanced", "balanced"), ("aggressive", "aggressive")]:
            sc = risk.get(key) or {}
            tgt_range = sc.get("target")
            if not tgt_range:
                continue
            tgt = (tgt_range[0] + tgt_range[1]) / 2
            stop_range = sc.get("stop") or [price - atr * 1.5, price - atr * 1.5]
            stop = (float(stop_range[0]) + float(stop_range[1])) / 2.0
            tp = (sc.get("tp_levels") or [None, None])[1]
            out.append(_record(
                "risk", label, tgt, stop, sc.get("target_confidence_pct"),
                tp["avg_days"] if tp else None, context,
                bool(sc.get("entry_eligible", True)),
            ))
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
            stop = float(pb.get("stop_loss") or (price - atr * 1.5))
            out.append(_record("pullback", label, float(tgt), stop, pb.get(ckey), None))
    return out


def run_market(tickers: List[str], market: str, period: str = "2y") -> List[Record]:
    records: List[Record] = []
    for n, ticker in enumerate(tickers, 1):
        print(f"[{market}] {n}/{len(tickers)} {ticker}", flush=True)
        df = download_one(ticker, period=period)
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
    out: Dict = {
        "overall": {}, "by_confidence_bucket": {},
        "by_price_bucket": {}, "by_liquidity_bucket": {},
        "by_volatility_bucket": {},
    }

    def metrics(group: pd.DataFrame) -> Dict:
        actionable = group[group["actionable"]]
        result = {
            "n_signals": int(len(group)),
            "n_actionable": int(len(actionable)),
            "actionable_rate_pct": round(len(actionable) / len(group) * 100, 1) if len(group) else 0.0,
            "hit_rate_pct": round(actionable["hit"].mean() * 100, 1) if len(actionable) else None,
            "stop_first_rate_pct": round(actionable["stop_first"].mean() * 100, 1) if len(actionable) else None,
            "avg_realized_return_pct": round(actionable["realized_return_pct"].dropna().mean(), 2) if actionable["realized_return_pct"].notna().any() else None,
            "avg_return_needed_pct": round(actionable["return_pct_needed"].mean(), 2) if len(actionable) else None,
            "median_days_to_hit": float(actionable.loc[actionable["hit"], "days_to_hit"].median()) if actionable["hit"].any() else None,
            "median_predicted_days": round(actionable["predicted_days"].dropna().median(), 1) if actionable["predicted_days"].notna().any() else None,
        }
        calibrated = actionable.dropna(subset=["predicted_confidence"])
        if len(calibrated):
            predicted = calibrated["predicted_confidence"].astype(float) / 100.0
            actual = calibrated["hit"].astype(float)
            result["median_predicted_confidence_pct"] = round(calibrated["predicted_confidence"].median(), 1)
            result["brier_score"] = round(float(np.mean((predicted - actual) ** 2)), 4)
            result["calibration_gap_pp"] = round(float(actual.mean() * 100.0 - predicted.mean() * 100.0), 1)
        else:
            result["median_predicted_confidence_pct"] = None
            result["brier_score"] = None
            result["calibration_gap_pp"] = None
        return result

    for (market, profile), g in df.groupby(["market", "profile"]):
        out["overall"][f"{market}|{profile}"] = metrics(g)

    def bucket(p):
        if pd.isna(p):
            return None
        if p < 35:
            return "low(<35%)"
        if p < 60:
            return "mid(35-60%)"
        return "high(60%+)"

    actionable_df = df[df["actionable"]].copy()
    actionable_df["conf_bucket"] = actionable_df["predicted_confidence"].apply(bucket)
    for profile, g in actionable_df.dropna(subset=["conf_bucket"]).groupby("profile"):
        for b, gb in g.groupby("conf_bucket"):
            out["by_confidence_bucket"][f"{profile}|{b}"] = {
                "n_signals": int(len(gb)),
                "avg_predicted_confidence_pct": round(gb["predicted_confidence"].mean(), 1),
                "actual_hit_rate_pct": round(gb["hit"].mean() * 100, 1),
                "calibration_gap_pp": round(gb["hit"].mean() * 100 - gb["predicted_confidence"].mean(), 1),
            }

    dimensions = {
        "by_price_bucket": "price_bucket",
        "by_liquidity_bucket": "liquidity_bucket",
        "by_volatility_bucket": "volatility_bucket",
    }
    for output_key, column in dimensions.items():
        for (market, profile, label), group in df.groupby(["market", "profile", column]):
            out[output_key][f"{market}|{profile}|{label}"] = metrics(group)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified walk-forward target-price validation")
    parser.add_argument("--krx-limit", type=int, default=len(KRX_STRATIFIED))
    parser.add_argument("--us-limit", type=int, default=100)
    parser.add_argument("--period", default="2y")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    outdir = ROOT / "docs" / "backtests"
    krx = KRX_STRATIFIED[:max(0, args.krx_limit)]
    us = us_tickers(max(0, args.us_limit))
    records = run_market(krx, "KRX", args.period) + run_market(us, "US", args.period)
    rows_df = pd.DataFrame([asdict(r) for r in records])
    result = {
        "period": f"{args.period} adjusted daily data, signals every {SIGNAL_STEP} trading days over the trailing ~1y, fixed {FORWARD_WINDOW}-trading-day forward window per signal",
        "methodology": {
            "execution": "signal close, entry at next trading-day open",
            "outcome_order": "conservative stop-first when target and stop touch in the same daily bar",
            "round_trip_cost": {"KRX": "0.40%", "US": "0.15%"},
            "strata": ["price_bucket", "20d_turnover_liquidity", "atr_percent_volatility"],
            "corporate_actions": "auto_adjust=True OHLC",
            "bias_note": "current listings only; delisted symbols are not included, so survivorship bias remains",
        },
        "krx_tickers": krx,
        "us_tickers": us,
        "summary": summarize(records),
    }
    if not args.no_write:
        outdir.mkdir(parents=True, exist_ok=True)
        rows_df.to_csv(
            outdir / "target_price_backtest_trades.csv",
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )
        with (outdir / "target_price_backtest_summary.json").open(
            "w", encoding="utf-8", newline="\n",
        ) as summary_file:
            summary_file.write(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
