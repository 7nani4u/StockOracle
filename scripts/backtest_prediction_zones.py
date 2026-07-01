#!/usr/bin/env python3
"""
Backtest StockOracle prediction zones with real OHLCV data.

The script compares:
  - baseline: pre-conservative shallow ATR bands
  - current: downside-risk adjusted bands mirroring api/index.py
      primary   = aggressive band A (ATR depth + base offset + depth_shift,
                  severe max-clamp, strong-support top clamp)
      secondary = recommended band B (VWAP/BB/MA20 anchor chain + strong-support
                  clamp + price ceiling), NOT a pure ATR multiple

Outcome evaluation mirrors the production tracker in api/index.py:
  - fixed 20 trading-day window from the signal date (entry included)
  - entry = close of the first day whose low touches the band top,
    clamped into the band
  - stop/target multipliers 1.35/1.15 (primary), 1.75/1.65 (secondary),
    identical for both logics

It writes JSON and CSV artifacts under docs/backtests/.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.index import _US_RECO_UNIVERSE, add_indicators  # noqa: E402


KRX_FALLBACK_100 = [
    "005930.KS","000660.KS","373220.KS","207940.KS","005380.KS","000270.KS","068270.KS","105560.KS","055550.KS","005490.KS",
    "035420.KS","035720.KS","051910.KS","006400.KS","012330.KS","028260.KS","012450.KS","034020.KS","247540.KQ","196170.KQ",
    "009150.KS","009540.KS","015760.KS","033780.KS","086790.KS","003670.KS","259960.KS","086520.KQ","028300.KQ","145020.KQ",
    "263750.KQ","293490.KQ","112040.KQ","041510.KQ","011200.KS","010140.KS","058470.KQ","035900.KQ","214150.KQ","357780.KQ",
    "095340.KQ","140860.KQ","098460.KQ","222800.KQ","240810.KQ","178320.KQ","032830.KS","066570.KS","000810.KS","316140.KS",
    "018260.KS","096770.KS","090430.KS","011070.KS","086280.KS","251270.KS","034730.KS","003550.KS","010130.KS","024110.KS",
    "030200.KS","017670.KS","352820.KS","326030.KS","138040.KS","000100.KS","018880.KS","267260.KS","047810.KS","071050.KS",
    "004020.KS","005830.KS","000720.KS","010950.KS","161390.KS","021240.KS","307950.KS","036570.KS","128940.KS","271560.KS",
    "039490.KS","078930.KS","008770.KS","180640.KS","006800.KS","071320.KS","005940.KS","272210.KS","004990.KS","023530.KS",
    "383220.KS","402340.KS","403870.KS","278470.KS","348370.KQ","277810.KQ","039030.KQ","067310.KQ","253450.KQ","091990.KQ",
]


@dataclass
class TradeResult:
    ticker: str
    market: str
    logic: str
    zone_type: str
    signal_date: str
    entry_date: str | None
    entry_price: float | None
    low_band: float
    high_band: float
    risk_score: int
    risk_level: str
    hold_signal: bool
    bounce_success: bool | None
    stop_hit: bool | None
    extra_drop: bool | None
    return_pct: float | None
    max_drawdown_pct: float | None
    vol_ratio: float
    atr_pct: float


def fetch_krx_100() -> List[str]:
    """Fetch KRX listing names, then use the curated fallback when market suffix is unknown."""
    # Yahoo needs KS/KQ suffixes. KRX download does not reliably include market section here,
    # so use the curated 100-ticker list for deterministic backtests.
    return KRX_FALLBACK_100[:100]


def us_100() -> List[str]:
    return [t for t in list(dict.fromkeys(_US_RECO_UNIVERSE)) if t not in {"SPY", "QQQ", "DIA", "IWM"}][:100]


def _flatten_download(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance may use either Price/Ticker or Ticker/Price layout.
        if ticker in raw.columns.get_level_values(0):
            raw = raw[ticker]
        elif ticker in raw.columns.get_level_values(-1):
            raw = raw.xs(ticker, level=-1, axis=1)
    raw = raw.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    out = raw[keep].dropna(subset=["Open", "High", "Low", "Close"])
    return out


def download_one(ticker: str, period: str = "3y") -> pd.DataFrame:
    try:
        # auto_adjust=True matches every yfinance call in api/index.py; unadjusted
        # prices would create fake gap_down signals on dividend ex-dates.
        raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True,
                          progress=False, threads=False, timeout=20)
        df = _flatten_download(raw, ticker)
        if len(df) < 220:
            return pd.DataFrame()
        return add_indicators(df.copy()).dropna(subset=["ATR", "MA20", "MA60", "MACD", "Signal_Line", "RSI"])
    except Exception:
        return pd.DataFrame()


def downside_risk(row: pd.Series, hist: pd.DataFrame, market: str) -> Tuple[int, str, float, bool, float | None]:
    """Mirror the downside score in api/index.py calc_buy_price.

    event_risk / learning_adjustment points are not reproducible offline and are
    excluded, so production scores can run higher than these.
    """
    price = float(row["Close"])
    atr = float(row["ATR"])
    ma20, ma60, ma120 = float(row.get("MA20", np.nan)), float(row.get("MA60", np.nan)), float(row.get("MA120", np.nan))
    rsi = float(row.get("RSI", 50))
    macd, sig = float(row.get("MACD", 0)), float(row.get("Signal_Line", 0))
    adx = float(row.get("ADX", 20)) if not pd.isna(row.get("ADX", np.nan)) else 20
    dip = float(row.get("DI_Plus", 0)) if not pd.isna(row.get("DI_Plus", np.nan)) else 0
    dim = float(row.get("DI_Minus", 0)) if not pd.isna(row.get("DI_Minus", np.nan)) else 0
    lows = hist["Low"].tail(21).iloc[:-1]
    recent_low_break = len(lows) >= 20 and price < float(lows.min())
    lows30 = sorted(float(x) for x in hist["Low"].tail(30) if x > 0)
    support_zone = float(np.mean(lows30[:5])) if len(lows30) >= 5 else price * 0.95
    lows20 = sorted(float(x) for x in hist["Low"].tail(20) if x > 0)
    strong_support = float(np.mean(lows20[:3])) if len(lows20) >= 3 else support_zone
    support_break = bool(strong_support and price < strong_support)
    atrs = hist["ATR"].dropna()
    vol_trend_expanding = False
    if len(atrs) >= 20:
        prev = float(atrs.iloc[-20:-5].mean())
        vol_trend_expanding = prev > 0 and float(atrs.iloc[-5:].mean()) / prev > 1.3
    vols = hist["Volume"].dropna()
    heavy_sell = False
    vol_ratio = 1.0
    if len(vols) >= 20 and len(hist) >= 2:
        avg = float(vols.iloc[-20:].mean())
        vol_ratio = float(vols.iloc[-1] / avg) if avg > 0 else 1.0
        heavy_sell = avg > 0 and vols.iloc[-1] >= avg * 1.35 and hist["Close"].iloc[-1] < hist["Close"].iloc[-2]
    ma20_slope = 0.0
    if len(hist) >= 6 and hist["MA20"].iloc[-6] > 0:
        ma20_slope = (hist["MA20"].iloc[-1] - hist["MA20"].iloc[-6]) / hist["MA20"].iloc[-6] * 100
    gap_down = len(hist) >= 2 and hist["Open"].iloc[-1] < hist["Close"].iloc[-2] * 0.985 and hist["Close"].iloc[-1] <= hist["Open"].iloc[-1] * 1.01

    score = 0
    if not math.isnan(ma20) and price < ma20: score += 10
    if not math.isnan(ma60) and price < ma60: score += 12
    if not math.isnan(ma120) and price < ma120: score += 8
    if ma20_slope < -0.6: score += 8
    if macd <= sig: score += 10
    if adx >= 25 and dim > dip: score += 16
    if rsi < 40: score += 8
    elif rsi > 70: score += 8
    atr_pct = atr / price * 100 if price > 0 else 0
    if market == "KRX":
        if atr_pct >= 5.5: score += 14
        elif atr_pct >= 4.0: score += 7
    else:
        if atr_pct >= 3.2: score += 14
        elif atr_pct >= 2.4: score += 7
    if recent_low_break: score += 16
    if support_break: score += 12
    if heavy_sell: score += 10
    if vol_ratio < 0.70 and not math.isnan(ma20) and price < ma20:
        score += 8
    if gap_down: score += 10
    if vol_trend_expanding: score += 8

    if score >= 68:
        return min(score, 100), "severe", 1.05, True, strong_support
    if score >= 50:
        return min(score, 100), "high", 0.75, True, strong_support
    if score >= 30:
        return min(score, 100), "medium", 0.45, False, strong_support
    return min(score, 100), "low", 0.20, False, strong_support


def old_bands(price: float, atr: float, market: str) -> Dict[str, Tuple[float, float]]:
    bw = 1.18 if market == "KRX" else 0.92
    zones = {
        "primary": (0.25, 0.65),
        "secondary": (0.65, 1.10),
    }
    out = {}
    for name, (k1, k2) in zones.items():
        center = price - ((k1 + k2) / 2) * atr
        hw = (k2 - k1) * atr * bw * 0.5
        out[name] = (center - hw, center + hw)
    return out


def new_bands(row: pd.Series, hist: pd.DataFrame, market: str, risk_level: str,
              depth_shift: float, strong_support: float | None) -> Dict[str, Tuple[float, float]]:
    """Mirror the bands tracked by the production outcome logger in api/index.py:
    primary = aggressive band A, secondary = recommended band B."""
    price = float(row["Close"])
    atr = float(row["ATR"])
    bw = 1.18 if market == "KRX" else 0.92
    base_offset = 0.20 if market == "KRX" else 0.15
    # Depth-capped ATR + price floor, same as production: crashed stocks can have
    # ATR worth tens of percent of price, which would push bands below zero.
    atr_d = min(atr, price * 0.15)
    band_floor = price * 0.10

    def floor_band(lo: float, hi: float) -> Tuple[float, float]:
        hi = max(hi, band_floor + atr_d * 0.05)
        lo = max(min(lo, hi - atr_d * 0.05), band_floor)
        return lo, hi

    # ── primary: aggressive band A ──
    k1 = 0.25 + base_offset + depth_shift
    k2 = 0.65 + base_offset + depth_shift
    if risk_level == "severe":
        # Production clamps with max(); the half-width keeps the base k spread (0.40).
        k1 = max(k1, 1.55)
        k2 = max(k2, 2.15)
    center = price - ((k1 + k2) / 2) * atr_d
    hw = (0.65 - 0.25) * atr_d * bw * 0.5
    p_lo, p_hi = center - hw, center + hw
    if risk_level in ("high", "severe") and strong_support:
        p_hi = min(p_hi, strong_support - atr_d * 0.10)
        p_lo = min(p_lo, p_hi - atr_d * 0.20)
    p_lo, p_hi = floor_band(p_lo, p_hi)

    # ── secondary: recommended band B (anchor chain, not a pure ATR multiple) ──
    bb_m = float(row["BB_Middle"]) if not pd.isna(row.get("BB_Middle", np.nan)) else None
    bb_l = float(row["BB_Lower"]) if not pd.isna(row.get("BB_Lower", np.nan)) else None
    if bb_l is not None and bb_l <= 0:
        bb_l = None  # production drops a non-positive BB lower as a support anchor
    ma20 = float(row["MA20"]) if not pd.isna(row.get("MA20", np.nan)) else None
    closes20 = hist["Close"].tail(20).to_numpy(dtype=float)
    vols20 = hist["Volume"].tail(20).to_numpy(dtype=float)
    vwap = None
    if len(closes20) >= 20 and np.nansum(vols20) > 0:
        vwap = float(np.average(closes20, weights=np.nan_to_num(vols20)))
    if not vwap:
        vwap = price - 0.875 * atr_d
    anc_a_raw = (vwap + bb_m) / 2 if bb_m is not None else vwap
    anc_a = min(anc_a_raw, price - atr_d * (0.55 + base_offset + depth_shift))
    if bb_l is not None and ma20 is not None:
        anc_b_raw = (bb_l + ma20) / 2
    elif bb_l is not None:
        anc_b_raw = bb_l
    else:
        anc_b_raw = price - 1.10 * atr_d
    # Production chains B off the pre-support-clamp anchor A.
    anc_b = min(anc_b_raw, anc_a - atr_d * (0.35 + depth_shift * 0.45))
    if risk_level in ("high", "severe") and strong_support:
        anc_b = min(anc_b, strong_support - atr_d * (0.75 + depth_shift * 0.30))
    hw_b = atr_d * 0.35 * bw
    ceiling = price - atr_d * 0.05
    s_hi = min(anc_b + hw_b, ceiling)
    s_lo = min(anc_b - hw_b, s_hi - atr_d * 0.05)
    s_lo, s_hi = floor_band(s_lo, s_hi)

    return {"primary": (p_lo, p_hi), "secondary": (s_lo, s_hi)}


def eval_band(ticker: str, market: str, logic: str, zone_type: str, df: pd.DataFrame,
              i: int, band: Tuple[float, float], risk_score: int, risk_level: str,
              hold_signal: bool) -> TradeResult:
    row = df.iloc[i]
    price, atr = float(row["Close"]), float(row["ATR"])
    low_band, high_band = band
    # Fixed 20-bar window from the signal date, same as the production outcome
    # tracker — evaluation never extends past signal+20 even when entry is late.
    future = df.iloc[i + 1:i + 21]
    touched = future["Low"].to_numpy(dtype=float) <= high_band
    if not touched.any():
        return TradeResult(ticker, market, logic, zone_type, str(df.index[i].date()), None, None,
                           low_band, high_band, risk_score, risk_level, hold_signal,
                           None, None, None, None, None, _vol_ratio(df.iloc[:i + 1]), atr / price * 100)
    entry_pos = int(np.argmax(touched))
    # Production enters at the close of the touch day, clamped into the band.
    entry_price = min(high_band, max(low_band, float(future["Close"].iloc[entry_pos])))
    after = future.iloc[entry_pos:]
    # Same fixed multipliers as api/index.py for both logics, so stop_hit_rate
    # differences come from band placement only.
    stop = entry_price - atr * (1.35 if zone_type == "primary" else 1.75)
    target = entry_price + atr * (1.15 if zone_type == "primary" else 1.65)
    min_low = float(after["Low"].min())
    stop_hit = bool((after["Low"] <= stop).any())
    bounce_success = bool((after["High"] >= target).any())
    extra_drop = min_low <= entry_price - atr * 0.75
    final_close = float(after["Close"].iloc[-1])
    ret = (final_close - entry_price) / entry_price * 100
    mdd = (min_low - entry_price) / entry_price * 100
    return TradeResult(ticker, market, logic, zone_type, str(df.index[i].date()), str(future.index[entry_pos].date()),
                       entry_price, low_band, high_band, risk_score, risk_level, hold_signal,
                       bounce_success, stop_hit, extra_drop, ret, mdd, _vol_ratio(df.iloc[:i + 1]), atr / price * 100)


def _vol_ratio(hist: pd.DataFrame) -> float:
    if len(hist) < 20:
        return 1.0
    avg = float(hist["Volume"].tail(20).mean())
    return round(float(hist["Volume"].iloc[-1] / avg), 3) if avg > 0 else 1.0


def run_market(tickers: List[str], market: str) -> List[TradeResult]:
    rows: List[TradeResult] = []
    for n, ticker in enumerate(tickers, 1):
        print(f"[{market}] {n}/{len(tickers)} {ticker}", flush=True)
        df = download_one(ticker)
        if df.empty or len(df) < 260:
            continue
        # 21-bar step > 20-bar forward window, so samples never overlap.
        signal_indices = list(range(max(140, len(df) - 190), len(df) - 21, 21))
        for i in signal_indices:
            hist = df.iloc[:i + 1]
            row = df.iloc[i]
            price, atr = float(row["Close"]), float(row["ATR"])
            if not price or not atr or math.isnan(atr):
                continue
            rs, rl, shift, hold, ssup = downside_risk(row, hist, market)
            for logic, bands in [
                ("baseline", old_bands(price, atr, market)),
                ("current", new_bands(row, hist, market, rl, shift, ssup)),
            ]:
                for zt, band in bands.items():
                    rows.append(eval_band(ticker, market, logic, zt, df, i, band, rs, rl, hold))
        time.sleep(0.05)
    return rows


def summarize(rows: List[TradeResult]) -> Dict:
    df = pd.DataFrame([asdict(r) for r in rows])
    if df.empty:
        return {}
    entered = df[df["entry_date"].notna()].copy()
    summary = {}
    for keys, g in entered.groupby(["market", "logic", "zone_type"]):
        summary["|".join(keys)] = {
            "signals": int(len(df[(df["market"] == keys[0]) & (df["logic"] == keys[1]) & (df["zone_type"] == keys[2])])),
            "entries": int(len(g)),
            "entry_rate": round(len(g) / max(1, len(df[(df["market"] == keys[0]) & (df["logic"] == keys[1]) & (df["zone_type"] == keys[2])])) * 100, 2),
            "bounce_success_rate": round(g["bounce_success"].mean() * 100, 2),
            "stop_hit_rate": round(g["stop_hit"].mean() * 100, 2),
            "extra_drop_rate": round(g["extra_drop"].mean() * 100, 2),
            "avg_return_pct": round(g["return_pct"].mean(), 3),
            "avg_mdd_pct": round(g["max_drawdown_pct"].mean(), 3),
            "median_atr_pct": round(g["atr_pct"].median(), 3),
            "median_vol_ratio": round(g["vol_ratio"].median(), 3),
        }
    for market, g in df.groupby("market"):
        # One row per signal (rows are duplicated across logic × zone), and the
        # gate stats only describe the current logic.
        per_signal = g[(g["logic"] == "current") & (g["zone_type"] == "primary")]
        cur_entered = g[(g["logic"] == "current") & g["entry_date"].notna()]
        high_entered = cur_entered[cur_entered["risk_score"] >= 55]
        summary[f"{market}|risk_gate"] = {
            "risk_samples": int(len(per_signal)),
            "hold_signal_rate": round(per_signal["hold_signal"].mean() * 100, 2),
            "high_risk_extra_drop_rate": round(high_entered["extra_drop"].mean() * 100, 2) if len(high_entered) else None,
            # 68 matches the severe gate in downside_risk.
            "severe_samples": int(len(per_signal[per_signal["risk_score"] >= 68])),
        }
    return summary


def main() -> None:
    outdir = ROOT / "docs" / "backtests"
    outdir.mkdir(parents=True, exist_ok=True)
    krx = fetch_krx_100()
    us = us_100()
    all_rows = run_market(krx, "KRX") + run_market(us, "US")
    rows_df = pd.DataFrame([asdict(r) for r in all_rows])
    rows_df.to_csv(outdir / "prediction_zone_backtest_trades.csv", index=False, encoding="utf-8-sig")
    result = {
        "period": "3y daily data, signals every 21 trading days (non-overlapping), fixed 20 trading-day forward window from each signal date",
        "methodology_notes": [
            "Bands and outcome rules mirror api/index.py: primary = aggressive band A, secondary = recommended band B; entry = close of the first touch day clamped into the band; stops 1.35/1.75 ATR for both logics.",
            "event_risk and learning_adjustment points are not reproducible offline and are excluded, so production downside scores can run higher than the ones here.",
            "Universe is the current constituent list applied to past data — results carry survivorship bias.",
        ],
        "krx_tickers": krx,
        "us_tickers": us,
        "summary": summarize(all_rows),
    }
    (outdir / "prediction_zone_backtest_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
