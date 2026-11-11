"""Chronological, non-overlapping audit of scan exit diagnostics, without fitting.

This evaluates a fixed +1 ATR target / -1 ATR stop at each completed close. It
does NOT evaluate the scanner's stock selection or calibrate a trading signal.
The holdout constant baseline is estimated on the earlier half only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from market_briefing.scan_execution import aligned_history_frame, build_exit_review, historical_barrier_reference

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "datasets" / "scan_execution_cache"


def audit_frame(frame, horizons=(3, 5, 10), round_trip_bps=20):
    frame = aligned_history_frame(frame)
    if frame is None:
        return {"available": False, "reason": "유효·정렬 OHLCV 60봉 미확보"}
    o, hi, lo, c, v = (frame[key].tolist() for key in ("Open", "High", "Low", "Close", "Volume"))
    reports = {}
    for horizon in horizons:
        rows = []
        for start in range(60, len(c) - horizon, horizon):
            atr = mean(max(hi[i]-lo[i], abs(hi[i]-c[i-1]), abs(lo[i]-c[i-1]))
                       for i in range(start-13, start+1))
            target, stop = c[start] + atr, c[start] - atr
            if stop <= 0:
                continue
            # Prefix-only service calculation; the outcome suffix is never an input.
            review = build_exit_review(o[:start+1], hi[:start+1], lo[:start+1], c[:start+1], v[:start+1],
                targets=[target], stop_price=stop, horizon_days=horizon, round_trip_bps=round_trip_bps)
            t = (review.get("targets") or [None])[0]
            if not t or not t.get("touch_probability"):
                continue
            end = start + horizon + 1
            actual = historical_barrier_reference(o[start:end], hi[start:end], lo[start:end], c[start:end],
                target_ratio=target/c[start], stop_ratio=stop/c[start], horizon_days=horizon,
                round_trip_bps=round_trip_bps, min_samples=1)
            touch_days = [j for j in range(1, horizon+1) if hi[start+j] >= target]
            rows.append({"date": str(frame.index[start].date()),
                "p": t["touch_probability"]["mid"], "touch": int(bool(touch_days)),
                "target_first": actual["counts"]["target_first"],
                "stop_first": actual["counts"]["stop_first"],
                "ambiguous": actual["counts"]["ambiguous"],
                "net_return": actual["mean_net_return_pct"],
                "window": t["days_if_reached"], "first_day": touch_days[0] if touch_days else None})
        split = len(rows) // 2
        train, holdout = rows[:split], rows[split:]
        if not train or not holdout:
            reports[str(horizon)] = {"available": False, "sample_count": len(rows)}
            continue
        baseline = mean(r["touch"] for r in train)
        touched = [r for r in holdout if r["touch"] and r["window"]]
        reports[str(horizon)] = {
            "available": len(train) >= 10 and len(holdout) >= 10,
            "train_count": len(train), "holdout_count": len(holdout), "non_overlapping": True,
            "train_end": train[-1]["date"], "holdout_start": holdout[0]["date"],
            "holdout_end": holdout[-1]["date"], "baseline_train_touch_rate": baseline,
            "touch_brier": mean((r["p"]-r["touch"])**2 for r in holdout),
            "constant_baseline_brier": mean((baseline-r["touch"])**2 for r in holdout),
            "mean_model_touch_probability": mean(r["p"] for r in holdout),
            "actual_touch_rate": mean(r["touch"] for r in holdout),
            "actual_target_before_stop_rate": mean(r["target_first"] for r in holdout),
            "stop_first_rate": mean(r["stop_first"] for r in holdout),
            "ambiguous_bars": sum(r["ambiguous"] for r in holdout),
            "mean_net_return_pct": mean(r["net_return"] for r in holdout),
            "conditional_window_coverage": (mean(r["window"][0] <= r["first_day"] <= r["window"][1]
                                                  for r in touched) if touched else None),
        }
    return {"available": True, "bars": len(frame), "first_date": str(frame.index[0].date()),
            "last_date": str(frame.index[-1].date()), "horizons": reports}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default="GRAB,SOFI,005930.KS")
    parser.add_argument("--as-of", required=True, help="Completed observations strictly before this local date")
    parser.add_argument("--years", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--horizons", default="3,5,10")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    horizons = tuple(int(h) for h in args.horizons.split(","))
    if any(not 1 <= h <= 60 for h in horizons):
        parser.error("horizons must be between 1 and 60")
    asof = pd.Timestamp(args.as_of).normalize()
    CACHE.mkdir(parents=True, exist_ok=True)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "as_of_exclusive": str(asof.date()),
              "method": "completed-close +1 ATR target / -1 ATR stop, chronological 50/50 split; no fitting",
              "cost_round_trip_bps": 20, "years": args.years, "prediction_skill_established": False,
              "limitations": ["선정 종목·신호의 매매 성과 검증이 아님", "일봉 터치와 실제 체결은 다름",
                              "표본·장세 한계, 거래비용 가정, 호가 깊이와 시장 충격 미반영"], "tickers": {}}
    for ticker in (t.strip().upper() for t in args.tickers.split(",") if t.strip()):
        if not all(ch.isalnum() or ch in ".^-=" for ch in ticker):
            parser.error("invalid ticker")
        path = CACHE / f"{ticker}_{args.years}y_{asof.date()}.csv"
        try:
            if args.offline:
                frame = pd.read_csv(path, index_col=0, parse_dates=True)
                frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None)
            else:
                frame = yf.Ticker(ticker).history(start=str((asof-pd.DateOffset(years=args.years)).date()),
                    end=str(asof.date()), auto_adjust=True, timeout=15)
                if frame is not None and not frame.empty:
                    frame = frame.copy()
                    frame.index = pd.to_datetime([s.date().isoformat() for s in frame.index])
                    frame.to_csv(path)
            frame = frame.loc[frame.index < asof]
            result = audit_frame(frame, horizons)
            result["source"] = "Yahoo Finance via yfinance, auto_adjust=True"
            result["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            report["tickers"][ticker] = result
        except Exception as exc:
            report["tickers"][ticker] = {"available": False, "reason": str(exc)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if all(r["available"] for r in report["tickers"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
