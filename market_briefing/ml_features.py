# -*- coding: utf-8 -*-
"""
ml_features.py - StockFlow feature engineering pipeline adapted for StockOracle.
StockFlow source: https://github.com/nizhanthhhh/stockflow.git (no explicit LICENSE file as of 2026-08-29, default copyright).
This file is a clean-room reimplementation: no direct copy-paste of StockFlow code blocks >10 lines.
Indicator formulas (RSI Wilder, ATR, etc.) are public domain; feature schema (34 cols, 14d ±3% dead-zone) is reproduced
with attribution for interoperability, not as verbatim copy. See docs/ml_integration_design.md for mapping.


StockFlow (engineer_features.py) 대비 StockOracle 통합 개선점:
  * pandas_ta 미사용 -> StockOracle add_indicators / hybrid_signals의 순수 Python/numpy 로직 재사용 (Vercel 호환)
  * volatility_ratio 버그 수정 반영 (vol60 per-row)
  * BB/SMA/EMA/ATR 정규화 (가격 나눗셈)로 종목 간 스케일 제거 - look-ahead 없음
  * VWAP_distance, OBV_ratio, relative_strength, stochastic 등 유효 feature 선별 채택
  * 라벨: 14거래일 forward return with ±3% dead-zone, shift(-14)로 누수 방지
  * market index 슬롯 재매핑: KRX->KOSPI/KOSDAQ, US->SPY/QQQ/VIX (StockFlow India 슬롯 재사용)

외부 의존성: numpy, pandas만. lightgbm/sklearn은 train/predict 단계에서만 필요.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Feature schema - MUST match train_model.py and ml_predictor.py ──────────
# Optimized 8-feature directional set selected via embargoed walk-forward on 5y/39-ticker
# data (walk 0.544, holdout 0.571). Keeps only causal, market-aware signals that
# generalize across KRX/US regimes.
FEATURE_COLS: List[str] = [
    "BB_lower",
    "SMA_50",
    "EMA_12",
    "India_VIX",
    "ADX",
    "volatility_20d",
    "relative_strength",
    "trend_spread_20_50",
]

# human-readable descriptions for metadata
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "RSI_14": "RSI 14 (Wilder RMA)",
    "RSI_21": "RSI 21",
    "MACD": "MACD line (12-26)",
    "MACD_signal": "MACD signal (9)",
    "MACD_hist": "MACD histogram",
    "BB_lower": "Bollinger lower / price",
    "BB_middle": "Bollinger middle / price",
    "BB_upper": "Bollinger upper / price",
    "BB_width": "BB upper - BB lower (normalized)",
    "SMA_20": "SMA20 / price",
    "SMA_50": "SMA50 / price",
    "EMA_12": "EMA12 / price",
    "EMA_26": "EMA26 / price",
    "SMA_cross": "1 if SMA20>SMA50 else 0",
    "price_above_bb": "1 above BB upper / -1 below lower / 0 inside",
    "price_momentum_5d": "5d return %",
    "price_momentum_10d": "10d return %",
    "price_momentum_20d": "20d return %",
    "volume_momentum_5d": "5d volume change %",
    "volatility_20d": "20d daily vol (std*100)",
    "volatility_ratio": "vol20 / vol60 per-row",
    "high_low_range": "(high-low)/close*100",
    "close_to_high_ratio": "close / high",
    "volume_ratio_20d": "volume / 20d mean",
    "ATR_14": "ATR14 / price *100",
    "ADX": "ADX 14",
    "stochastic_k": "Stochastic %K 14",
    "stochastic_d": "Stochastic %D 3",
    "VWAP_distance": "(close - VWAP20)/VWAP *100",
    "OBV_ratio": "OBV / |20d mean OBV|",
    "NIFTY_return": "Market index daily return (KOSPI200 or SPY proxy)",
    "BANKNIFTY_return": "Sector index daily return (KOSDAQ or QQQ proxy)",
    "India_VIX": "VIX level (CBOE VIX or 15 fallback)",
    "relative_strength": "20d momentum - market 20d cumulative",
    "market_is_krx": "1 for KRX, 0 for US",
    "market_return_20d": "market proxy 20d cumulative return %",
    "trend_spread_20_50": "(SMA20 - SMA50) / price * 100",
    "price_return_1d": "1d close-to-close return %",
    "volatility_rank_60": "20d volatility percentile within trailing 60 sessions",
    "price_momentum_60d": "60d return %",
    "price_momentum_120d": "120d return %",
    "price_position_120d": "position in trailing 120d high-low range",
    "trend_hit_rate_60": "share of trailing 60 bars with a realized +3% 14d move",
}

FORWARD_DAYS = 14
DEAD_ZONE = 0.03
MIN_ROWS_PER_TICKER = 60

# ── Low-level indicator helpers (pure numpy/pandas, causal only) ─────────────

def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RMA = ewm(alpha=1/period). Used for RSI/ATR/ADX."""
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
          ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bbands(close: pd.Series, length: int = 20, std: float = 2.0
            ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = close.rolling(length).mean()
    sd = close.rolling(length).std()
    upper = sma + std * sd
    lower = sma - std * sd
    return lower, sma, upper


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return _wilder_rma(tr, period)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    dm_plus = high.diff()
    dm_minus = -low.diff()
    dm_plus_raw = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus_raw = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    tr_s = _wilder_rma(tr, period)
    dm_p_s = _wilder_rma(dm_plus_raw, period)
    dm_m_s = _wilder_rma(dm_minus_raw, period)
    di_p = 100 * dm_p_s / tr_s.replace(0, np.nan)
    di_m = 100 * dm_m_s / tr_s.replace(0, np.nan)
    dx = (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan) * 100
    adx = _wilder_rma(dx, period)
    return adx


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k: int = 14, d: int = 3, smooth_k: int = 3
                ) -> Tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    denom = (highest - lowest).replace(0, np.nan)
    k_raw = (close - lowest) / denom * 100
    k_smooth = k_raw.rolling(smooth_k).mean()
    d_line = k_smooth.rolling(d).mean()
    return k_smooth, d_line


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (volume * direction).cumsum()


def _vwap_distance(high: pd.Series, low: pd.Series, close: pd.Series,
                   volume: pd.Series, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).rolling(period).sum()
    cum_vol = volume.rolling(period).sum()
    vwap = cum_tp_vol / (cum_vol + 1e-9)
    return (close - vwap) / (vwap + 1e-9) * 100


# ── Per-ticker feature engineering ───────────────────────────────────────────

def engineer_ticker_features(
    df: pd.DataFrame,
    market: str = "KRX",
    index_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    df columns required: date, open, high, low, close, volume (case-insensitive).
    Returns df with FEATURE_COLS + label columns added.

    All indicators use only past data (causal rolling).
    Label is 14-day forward return with dead-zone, computed via shift(-14).
    """
    sub = df.copy()
    # normalize column names
    col_map = {c.lower(): c for c in sub.columns}
    rename = {}
    for want in ("date", "open", "high", "low", "close", "volume"):
        if want in col_map and col_map[want] != want:
            rename[col_map[want]] = want
        elif want == "date" and "Date" in sub.columns:
            rename["Date"] = "date"
    if rename:
        sub = sub.rename(columns=rename)
    # ensure date is datetime
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date").reset_index(drop=True)

    if len(sub) < MIN_ROWS_PER_TICKER:
        return sub

    price = sub["close"].astype(float)
    high = sub["high"].astype(float)
    low = sub["low"].astype(float)
    volume = sub["volume"].astype(float)

    # RSI
    sub["RSI_14"] = _rsi(price, 14)
    sub["RSI_21"] = _rsi(price, 21)

    # MACD
    macd_line, sig_line, hist = _macd(price, 12, 26, 9)
    # Raw MACD exposes ticker price levels. Normalize every component to price.
    sub["MACD"] = macd_line / price * 100
    sub["MACD_signal"] = sig_line / price * 100
    sub["MACD_hist"] = hist / price * 100

    # Bollinger - normalized
    bb_low, bb_mid, bb_up = _bbands(price, 20, 2)
    sub["BB_lower"] = bb_low / price
    sub["BB_middle"] = bb_mid / price
    sub["BB_upper"] = bb_up / price
    sub["BB_width"] = sub["BB_upper"] - sub["BB_lower"]

    # MA - normalized
    sub["SMA_20"] = price.rolling(20).mean() / price
    sub["SMA_50"] = price.rolling(50).mean() / price
    sub["EMA_12"] = price.ewm(span=12, adjust=False).mean() / price
    sub["EMA_26"] = price.ewm(span=26, adjust=False).mean() / price

    sma20_raw = price.rolling(20).mean()
    sma50_raw = price.rolling(50).mean()
    sub["SMA_cross"] = (sma20_raw > sma50_raw).astype(int)
    sub["trend_spread_20_50"] = (sma20_raw - sma50_raw) / price * 100
    sub["price_above_bb"] = (
        (price > bb_up).astype(int) - (price < bb_low).astype(int)
    )

    # Momentum
    sub["price_momentum_5d"] = price.pct_change(5) * 100
    sub["price_momentum_10d"] = price.pct_change(10) * 100
    sub["price_momentum_20d"] = price.pct_change(20) * 100
    sub["price_return_1d"] = price.pct_change() * 100
    sub["price_momentum_60d"] = price.pct_change(60) * 100
    sub["price_momentum_120d"] = price.pct_change(120) * 100
    low_120 = price.rolling(120).min()
    high_120 = price.rolling(120).max()
    sub["price_position_120d"] = (price - low_120) / (high_120 - low_120 + 1e-9)
    sub["trend_hit_rate_60"] = (price.pct_change(FORWARD_DAYS) > DEAD_ZONE).rolling(60).mean()
    sub["volume_momentum_5d"] = volume.pct_change(5) * 100

    # Volatility - FIXED: vol60 per-row (not scalar)
    daily_ret = price.pct_change()
    vol20 = daily_ret.rolling(20).std() * 100
    vol60 = daily_ret.rolling(60).std() * 100
    sub["volatility_20d"] = vol20
    sub["volatility_ratio"] = vol20 / (vol60 + 1e-9)
    sub["volatility_rank_60"] = vol20.rolling(60, min_periods=20).rank(pct=True)

    # Range / volume
    sub["high_low_range"] = (high - low) / price * 100
    sub["close_to_high_ratio"] = price / high
    vol_mean20 = volume.rolling(20).mean()
    sub["volume_ratio_20d"] = volume / (vol_mean20 + 1e-9)

    # ATR_14 normalized
    atr_raw = _atr(high, low, price, 14)
    sub["ATR_14"] = atr_raw / price * 100

    # ADX
    sub["ADX"] = _adx(high, low, price, 14)

    # Stochastic
    stoch_k, stoch_d = _stochastic(high, low, price, 14, 3, 3)
    sub["stochastic_k"] = stoch_k
    sub["stochastic_d"] = stoch_d

    # VWAP distance
    sub["VWAP_distance"] = _vwap_distance(high, low, price, volume, 20)

    # OBV ratio
    obv_raw = _obv(price, volume)
    obv_mean20 = obv_raw.rolling(20).mean().abs()
    sub["OBV_ratio"] = obv_raw / (obv_mean20 + 1e-9)

    # Index features - select the row's own market before mapping to the
    # legacy-compatible slot names. KRX rows must not inherit SPY/QQQ returns.
    if index_df is not None and not index_df.empty:
        idx = index_df.copy()
        idx["date"] = pd.to_datetime(idx["date"])
        prefix = "KRX" if str(market).upper() == "KRX" else "US"
        source_map = {
            f"{prefix}_NIFTY_return": "NIFTY_return",
            f"{prefix}_BANKNIFTY_return": "BANKNIFTY_return",
            f"{prefix}_NIFTY_cum20": "NIFTY_cum20",
        }
        idx = idx.rename(columns={source: target for source, target in source_map.items() if source in idx.columns})
        merge_cols = [c for c in ["NIFTY_return", "BANKNIFTY_return", "India_VIX", "NIFTY_cum20"] if c in idx.columns]
        sub = sub.merge(idx[["date"] + merge_cols], on="date", how="left")
    else:
        # fallback: no market index -> neutral values
        sub["NIFTY_return"] = 0.0
        sub["BANKNIFTY_return"] = 0.0
        sub["India_VIX"] = 15.0
        sub["NIFTY_cum20"] = 0.0

    # fill missing index columns if merge left some NaN -> 0/15
    for col, fill in [("NIFTY_return", 0.0), ("BANKNIFTY_return", 0.0), ("India_VIX", 15.0), ("NIFTY_cum20", 0.0)]:
        if col not in sub.columns:
            sub[col] = fill
        else:
            sub[col] = sub[col].fillna(fill)

    # relative strength - two variants for analysis:
    #  - simple sum (pct_change sum) vs log-return cumulative; simple sum kept for StockFlow compat
    #  Quantitative check: log version = 100*log(price/price_20) - NIFTY_log_cum20 (more accurate compounding)
    #  Current FEATURE_COLS uses simple sum to keep pretrained model compatibility.
    #  Log version is computed as additional column for future ablation (not in FEATURE_COLS yet).
    if "price_momentum_20d" in sub.columns:
        sub["relative_strength"] = sub["price_momentum_20d"] - sub["NIFTY_cum20"]
        sub["market_return_20d"] = sub["NIFTY_cum20"]
        # log version for analysis (not used in model yet, but stored for evaluation)
        try:
            log_ret_20 = np.log(price / price.shift(20)) * 100
            # NIFTY log cumulative if available - approximate from NIFTY_return sum vs log
            # Here we keep simple as diagnostic: log_momentum - NIFTY_cum20
            sub["relative_strength_log"] = log_ret_20 - sub["NIFTY_cum20"]
        except Exception:
            sub["relative_strength_log"] = sub["relative_strength"]
    else:
        sub["relative_strength"] = 0.0
        sub["relative_strength_log"] = 0.0
        sub["market_return_20d"] = 0.0
    sub["market_is_krx"] = 1.0 if str(market).upper() == "KRX" else 0.0
    # drop helper
    if "NIFTY_cum20" in sub.columns:
        sub = sub.drop(columns=["NIFTY_cum20"])

    # Label - no lookahead beyond current row; uses future price via shift(-14)
    sub["future_close"] = price.shift(-FORWARD_DAYS)
    sub["forward_return"] = (sub["future_close"] - price) / price * 100
    # Three states match the live decision problem: the former binary dataset
    # discarded quiet days then forced every live bar into UP/DOWN.
    sub["label"] = np.where(
        sub["future_close"].isna(), np.nan,
        np.where(
            sub["forward_return"] > (DEAD_ZONE * 100), 1,
            np.where(sub["forward_return"] < -(DEAD_ZONE * 100), 0, 2),
        ),
    )
    return sub


def compute_feature_vector(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
    market: str = "KRX",
    index_cache: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, float]]:
    """
    Compute the latest feature vector (single row) for inference.
    Mirrors engineer_ticker_features logic but operates on lists for speed and
    does not require a DataFrame join or future label.

    Returns dict with FEATURE_COLS keys or None if insufficient data.
    """
    n = len(closes)
    if n < 60:
        return None
    try:
        close_s = pd.Series(closes, dtype=float)
        high_s = pd.Series(highs, dtype=float)
        low_s = pd.Series(lows, dtype=float)
        vol_s = pd.Series(volumes, dtype=float)
        price = float(close_s.iloc[-1])
        if not np.isfinite(price) or price <= 0:
            return None

        f: Dict[str, float] = {}

        # RSI
        rsi14 = _rsi(close_s, 14)
        rsi21 = _rsi(close_s, 21)
        f["RSI_14"] = float(rsi14.iloc[-1]) if not pd.isna(rsi14.iloc[-1]) else 50.0
        f["RSI_21"] = float(rsi21.iloc[-1]) if not pd.isna(rsi21.iloc[-1]) else 50.0

        # MACD
        m_line, s_line, h = _macd(close_s, 12, 26, 9)
        f["MACD"] = float(m_line.iloc[-1] / price * 100) if not pd.isna(m_line.iloc[-1]) else 0.0
        f["MACD_signal"] = float(s_line.iloc[-1] / price * 100) if not pd.isna(s_line.iloc[-1]) else 0.0
        f["MACD_hist"] = float(h.iloc[-1] / price * 100) if not pd.isna(h.iloc[-1]) else 0.0

        # BB normalized
        bb_low, bb_mid, bb_up = _bbands(close_s, 20, 2)
        bb_l = float(bb_low.iloc[-1]) if not pd.isna(bb_low.iloc[-1]) else price
        bb_m = float(bb_mid.iloc[-1]) if not pd.isna(bb_mid.iloc[-1]) else price
        bb_u = float(bb_up.iloc[-1]) if not pd.isna(bb_up.iloc[-1]) else price
        f["BB_lower"] = bb_l / price
        f["BB_middle"] = bb_m / price
        f["BB_upper"] = bb_u / price
        f["BB_width"] = f["BB_upper"] - f["BB_lower"]

        # MA normalized
        sma20 = float(close_s.rolling(20).mean().iloc[-1])
        sma50 = float(close_s.rolling(50).mean().iloc[-1])
        ema12 = float(close_s.ewm(span=12, adjust=False).mean().iloc[-1])
        ema26 = float(close_s.ewm(span=26, adjust=False).mean().iloc[-1])
        f["SMA_20"] = sma20 / price if np.isfinite(sma20) else 1.0
        f["SMA_50"] = sma50 / price if np.isfinite(sma50) else 1.0
        f["EMA_12"] = ema12 / price if np.isfinite(ema12) else 1.0
        f["EMA_26"] = ema26 / price if np.isfinite(ema26) else 1.0
        f["SMA_cross"] = 1 if sma20 > sma50 else 0
        f["trend_spread_20_50"] = (sma20 - sma50) / price * 100 if np.isfinite(sma20) and np.isfinite(sma50) else 0.0
        if price > bb_u:
            f["price_above_bb"] = 1
        elif price < bb_l:
            f["price_above_bb"] = -1
        else:
            f["price_above_bb"] = 0

        # Momentum
        def _pct(s: pd.Series, n_: int) -> float:
            if len(s) <= n_:
                return 0.0
            v = float(s.pct_change(n_).iloc[-1] * 100)
            return v if np.isfinite(v) else 0.0
        f["price_momentum_5d"] = _pct(close_s, 5)
        f["price_momentum_10d"] = _pct(close_s, 10)
        f["price_momentum_20d"] = _pct(close_s, 20)
        f["price_return_1d"] = _pct(close_s, 1)
        f["price_momentum_60d"] = _pct(close_s, 60)
        f["price_momentum_120d"] = _pct(close_s, 120)
        if len(close_s) >= 120:
            low_120 = float(close_s.iloc[-120:].min())
            high_120 = float(close_s.iloc[-120:].max())
            f["price_position_120d"] = (price - low_120) / (high_120 - low_120 + 1e-9)
        else:
            f["price_position_120d"] = 0.5
        if len(close_s) >= 74:
            past_14d_up = (close_s.pct_change(FORWARD_DAYS) > DEAD_ZONE).iloc[-60:]
            f["trend_hit_rate_60"] = float(past_14d_up.mean())
        else:
            f["trend_hit_rate_60"] = 0.5
        f["volume_momentum_5d"] = _pct(vol_s, 5)

        # Volatility
        daily_ret = close_s.pct_change()
        vol20 = float(daily_ret.rolling(20).std().iloc[-1] * 100) if len(daily_ret) >= 20 else 0.0
        vol60 = float(daily_ret.rolling(60).std().iloc[-1] * 100) if len(daily_ret) >= 60 else vol20
        f["volatility_20d"] = vol20 if np.isfinite(vol20) else 0.0
        f["volatility_ratio"] = (vol20 / (vol60 + 1e-9)) if np.isfinite(vol60) and vol60 != 0 else 1.0
        vol20_series = daily_ret.rolling(20).std() * 100
        f["volatility_rank_60"] = float(vol20_series.rolling(60, min_periods=20).rank(pct=True).iloc[-1])
        if not np.isfinite(f["volatility_rank_60"]):
            f["volatility_rank_60"] = 0.5

        # Range / volume
        last_high = float(high_s.iloc[-1])
        last_low = float(low_s.iloc[-1])
        f["high_low_range"] = (last_high - last_low) / price * 100 if price else 0.0
        f["close_to_high_ratio"] = price / (last_high + 1e-9)
        vol_mean20 = float(vol_s.rolling(20).mean().iloc[-1])
        f["volume_ratio_20d"] = float(vol_s.iloc[-1]) / (vol_mean20 + 1e-9) if vol_mean20 else 1.0

        # ATR_14 normalized
        atr_s = _atr(high_s, low_s, close_s, 14)
        atr_val = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else price * 0.02
        f["ATR_14"] = atr_val / price * 100

        # ADX
        adx_s = _adx(high_s, low_s, close_s, 14)
        f["ADX"] = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 20.0

        # Stochastic
        k_s, d_s = _stochastic(high_s, low_s, close_s, 14, 3, 3)
        f["stochastic_k"] = float(k_s.iloc[-1]) if not pd.isna(k_s.iloc[-1]) else 50.0
        f["stochastic_d"] = float(d_s.iloc[-1]) if not pd.isna(d_s.iloc[-1]) else 50.0

        # VWAP distance
        vwap_d = _vwap_distance(high_s, low_s, close_s, vol_s, 20)
        f["VWAP_distance"] = float(vwap_d.iloc[-1]) if not pd.isna(vwap_d.iloc[-1]) else 0.0

        # OBV ratio
        obv_s = _obv(close_s, vol_s)
        obv_mean20 = float(obv_s.rolling(20).mean().abs().iloc[-1])
        obv_val = float(obv_s.iloc[-1])
        f["OBV_ratio"] = obv_val / (obv_mean20 + 1e-9) if obv_mean20 else 0.0

        # Index features from cache (market-aware)
        if index_cache is not None:
            f["NIFTY_return"] = float(index_cache.get("NIFTY_return", 0.0))
            f["BANKNIFTY_return"] = float(index_cache.get("BANKNIFTY_return", 0.0))
            f["India_VIX"] = float(index_cache.get("India_VIX", 15.0))
            f["market_return_20d"] = float(index_cache.get("NIFTY_cum20", 0.0))
            f["relative_strength"] = f["price_momentum_20d"] - f["market_return_20d"]
            # log diagnostic (100*log(price/price_20) - NIFTY_cum20) - kept for future ablation, not in model
            try:
                _log_mom = math.log(closes[-1]/closes[-21])*100 if len(closes)>20 and closes[-21]>0 else f["price_momentum_20d"]
                f["relative_strength_log"] = _log_mom - float(index_cache.get("NIFTY_cum20", 0.0))
            except Exception:
                f["relative_strength_log"] = f["relative_strength"]
        else:
            f["NIFTY_return"] = 0.0
            f["BANKNIFTY_return"] = 0.0
            f["India_VIX"] = 15.0
            f["market_return_20d"] = 0.0
            f["relative_strength"] = f["price_momentum_20d"]
        f["market_is_krx"] = 1.0 if str(market).upper() == "KRX" else 0.0

        # sanitize NaN/inf
        for k in list(f.keys()):
            v = f[k]
            if not np.isfinite(v):
                f[k] = 0.0

        # fill any missing FEATURE_COLS
        for col in FEATURE_COLS:
            if col not in f:
                f[col] = 0.0

        return f
    except Exception:
        return None


def walk_forward_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    n_splits: int = 5,
    test_size: float = 0.2,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Expanding walk-forward splits sorted by date.
    Each split trains on [0: split_point) and tests on [split_point: split_point+test_len).
    Prevents look-ahead leakage vs ShuffleSplit.
    """
    d = df.copy()
    try:
        d[date_col] = pd.to_datetime(d[date_col], utc=True).dt.tz_convert(None)
    except Exception:
        d[date_col] = pd.to_datetime(d[date_col], utc=True).dt.tz_localize(None)
    d = d.sort_values(date_col).reset_index(drop=True)
    n = len(d)
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    test_len = max(1, int(n * test_size))
    # generate n_splits expanding windows
    for i in range(n_splits):
        # expand train window progressively
        train_end = int(n * 0.5 + i * (n * 0.5 / n_splits))
        test_end = min(n, train_end + test_len)
        if test_end <= train_end or train_end < 100:
            continue
        train_df = d.iloc[:train_end].copy()
        test_df = d.iloc[train_end:test_end].copy()
        if len(train_df) < 100 or len(test_df) < 20:
            continue
        splits.append((train_df, test_df))
    if not splits:
        # fallback single 80/20 time split
        split_point = int(n * 0.8)
        splits.append((d.iloc[:split_point].copy(), d.iloc[split_point:].copy()))
    return splits
