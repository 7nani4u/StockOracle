#!/usr/bin/env python3
"""동적 RSI를 국내·미국 개별 종목에서 인과적으로 검증하는 CLI.

예:
  python scripts/backtest_dynamic_rsi.py --market KRX --tickers 005930.KS 000660.KS
  python scripts/backtest_dynamic_rsi.py --market US --tickers AAPL MSFT NVDA --period 5y
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from market_briefing.dynamic_rsi import backtest_dynamic_rsi  # noqa: E402


def _market_for(ticker: str, requested: str) -> str:
    if requested != "AUTO":
        return requested
    return "KRX" if ticker.upper().endswith((".KS", ".KQ")) else "US"


def _adjusted_history(ticker: str, period: str) -> pd.DataFrame:
    raw = yf.Ticker(ticker).history(
        period=period, interval="1d", auto_adjust=False,
        actions=True, repair=True, timeout=20,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [column[0] if isinstance(column, tuple) else column for column in raw.columns]
    raw = raw.rename(columns=str.title).dropna(subset=["Open", "High", "Low", "Close"])
    adjusted_close = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
    ratio = (adjusted_close / raw["Close"].replace(0, np.nan)).fillna(1.0)
    adjusted = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    for column in ("Open", "High", "Low", "Close"):
        adjusted[column] = adjusted[column] * ratio
    return adjusted


def run(tickers: list[str], market: str, period: str) -> Dict:
    results = {}
    for ticker in dict.fromkeys(value.strip().upper() for value in tickers if value.strip()):
        ticker_market = _market_for(ticker, market)
        try:
            history = _adjusted_history(ticker, period)
            if history.empty:
                results[ticker] = {"error": "no_history", "market": ticker_market}
                continue
            backtest = backtest_dynamic_rsi(history, market=ticker_market)
            results[ticker] = {
                "market": ticker_market,
                "first_date": str(history.index[0])[:10],
                "last_date": str(history.index[-1])[:10],
                "bars": len(history),
                "summary": backtest["summary"],
            }
        except Exception as exc:
            results[ticker] = {"error": str(exc), "market": ticker_market}
    return {
        "period": period,
        "tickers": results,
        "warning": (
            "과거 데이터의 다음 봉 시가·장중 손절·시장별 왕복비용을 적용한 연구 결과입니다. "
            "미래 수익률이나 상승확률을 보장하지 않습니다."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--market", choices=("KRX", "US", "AUTO"), default="AUTO")
    parser.add_argument("--period", default="5y", choices=("1y", "2y", "5y", "10y", "max"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run(args.tickers, args.market, args.period)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if all("error" not in item for item in payload["tickers"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
