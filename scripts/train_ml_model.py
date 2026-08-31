#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ml_model.py - StockFlow train_model.py adapted for StockOracle.

StockFlow 대비 개선/적응:
  * Data source: StockOracle universe (KR_STOCK_MAP + US_TICKERS) 샘플링, synthetic fallback
  * Features: market_briefing.ml_features (순수 python, Vercel 호환)
  * Market: KRX/US 이원화 (KOSPI/SPY 등)
  * Validation: time-based 80/20 (시계열 분할) + walk-forward option
  * Model: LightGBM이 있으면 사용, 없으면 sklearn HistGradientBoostingClassifier fallback
  * Leakage 방지: train/test 를 날짜로 나누고, feature는 과거만, label은 shift(-14)
  * Calibration: Platt scaling (train과 동일)

Usage:
  python scripts/train_ml_model.py                 # synthetic+download hybrid (기본 40 tickers 샘플)
  python scripts/train_ml_model.py --tickers 005930.KS,AAPL --period 2y
  python scripts/train_ml_model.py --synthetic-only
  python scripts/train_ml_model.py --input datasets/training_features.parquet  # 이미 만든 feature parquet 있으면 재사용
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from market_briefing.ml_features import FEATURE_COLS, FORWARD_DAYS, engineer_ticker_features
from market_briefing.ml_evaluate import evaluate_predictions, walk_forward_evaluate

# ── Try imports for training backends ─────────────────────────────────────
_HAS_LGBM = False
_HAS_SKLEARN = False
_HAS_YFINANCE = False

try:
    import lightgbm as lgb  # type: ignore
    _HAS_LGBM = True
except Exception:
    pass
try:
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
    from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score  # type: ignore
    from sklearn.utils.class_weight import compute_class_weight  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    pass
try:
    import yfinance as yf  # type: ignore
    _HAS_YFINANCE = True
except Exception:
    pass

try:
    import joblib  # type: ignore
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False
    import pickle as joblib  # fallback

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    # KRX sample
    "005930.KS", "000660.KS", "035420.KS", "035720.KS", "005380.KS",
    "000270.KS", "068270.KS", "105560.KS", "055550.KS", "051910.KS",
    "006400.KS", "373220.KS", "207940.KS", "028300.KQ", "247540.KQ",
    # US sample
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "JPM", "V", "UNH",
    "XOM", "MA", "HD", "COST", "NFLX", "AMD", "DIS", "WMT", "PG", "KO",
]

MODELS_DIR = ROOT / "models"
FEATURE_PARQUET = ROOT / "datasets" / "training_features.parquet"
RAW_PARQUET = ROOT / "datasets" / "training_raw.parquet"


# ── Synthetic data generator (for offline tests / when yfinance unavailable) ─

def _synthetic_ohlcv(ticker: str, days: int = 500, seed: int = 0) -> pd.DataFrame:
    rnd = np.random.RandomState(abs(hash(ticker)) % 2**31 + seed)
    # random walk with drift, regime switches
    drift = rnd.uniform(-0.0005, 0.001)
    vol = rnd.uniform(0.012, 0.025)
    prices = [100 * rnd.uniform(0.5, 2.0)]
    for _ in range(1, days):
        # occasional jumps
        jump = 0
        if rnd.random() < 0.02:
            jump = rnd.normal(0, 0.04)
        ret = rnd.normal(drift, vol) + jump
        prices.append(max(1.0, prices[-1] * (1 + ret)))
    prices = np.array(prices)
    highs = prices * (1 + np.abs(rnd.normal(0, 0.008, size=days)))
    lows = prices * (1 - np.abs(rnd.normal(0, 0.008, size=days)))
    # ensure low < close < high
    for i in range(days):
        hi = max(prices[i], highs[i])
        lo = min(prices[i], lows[i])
        highs[i] = hi
        lows[i] = lo
    volumes = rnd.lognormal(mean=11, sigma=0.6, size=days).astype(int)
    volumes = np.maximum(volumes, 10_000)
    opens = prices * (1 + rnd.normal(0, 0.004, size=days))
    dates = pd.date_range(end=datetime.now().date(), periods=days, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "ticker": ticker,
    })
    return df


def _download_ticker(ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
    if not _HAS_YFINANCE:
        return None
    try:
        # handle KRX suffix already
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 60:
            # KRX suffix swap
            if ticker.endswith(".KS"):
                alt = ticker[:-3] + ".KQ"
                hist = yf.Ticker(alt).history(period=period, auto_adjust=True)
                if hist is not None and not hist.empty and len(hist) >= 60:
                    ticker = alt
                else:
                    return None
            else:
                return None
        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        # normalize
        hist = hist.rename(columns={c: c.lower() for c in hist.columns})
        # yfinance may have capital first letter
        col_map = {}
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in hist.columns:
                col_map[c] = c.lower()
            elif c.lower() not in hist.columns and c in [x.capitalize() for x in hist.columns]:
                for actual in hist.columns:
                    if actual.lower() == c.lower():
                        col_map[actual] = c.lower()
        if col_map:
            hist = hist.rename(columns=col_map)
        needed = ["open", "high", "low", "close", "volume"]
        if not all(c in hist.columns for c in needed):
            return None
        out = hist[needed].copy()
        out["date"] = hist.index
        out["ticker"] = ticker
        out = out.reset_index(drop=True)
        return out
    except Exception:
        return None


def _collect_training_data(tickers: List[str], period: str = "5y", synthetic_fallback: bool = True) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for t in tickers:
        df = _download_ticker(t, period=period)
        if df is None or len(df) < 60:
            if synthetic_fallback:
                print(f"  [synthetic] {t} - using synthetic OHLCV (download failed/short)")
                df = _synthetic_ohlcv(t, days=500)
            else:
                print(f"  [skip] {t} - no data")
                continue
        else:
            print(f"  [ok] {t} - {len(df)} rows ({df['date'].min().date()} - {df['date'].max().date()})")
        frames.append(df)
    if not frames:
        # absolute fallback: generate synthetic for default tickers
        print("  [fallback] Generating synthetic dataset for all tickers")
        for t in tickers[:10]:
            frames.append(_synthetic_ohlcv(t, days=500, seed=42))
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    return combined


def _fetch_index_df(start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch market-matched KRX and US index data for ticker feature engineering."""
    if not _HAS_YFINANCE:
        return None
    try:
        # Keep the legacy feature slot names at the model boundary, but retain
        # separate source columns here so KRX rows never learn from SPY/QQQ.
        start_dt = pd.to_datetime(start) - pd.Timedelta(days=90)
        end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)
        dfs = {}
        for sym, key in [
            ("069500.KS", "KRX_NIFTY_return"), ("^KQ11", "KRX_BANKNIFTY_return"),
            ("SPY", "US_NIFTY_return"), ("QQQ", "US_BANKNIFTY_return"),
        ]:
            try:
                raw = yf.download(sym, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
                if raw is None or raw.empty:
                    continue
                close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close.index = pd.to_datetime(close.index).tz_localize(None)
                ret = close.pct_change() * 100
                dfs[key] = ret
            except Exception:
                continue
        # VIX
        try:
            raw = yf.download("^VIX", start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
            if raw is not None and not raw.empty:
                close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close.index = pd.to_datetime(close.index).tz_localize(None)
                dfs["India_VIX"] = close
        except Exception:
            pass
        if not dfs:
            return None
        idx = pd.DataFrame(dfs)
        idx.index.name = "date"
        idx = idx.reset_index()
        idx["date"] = pd.to_datetime(idx["date"])
        for prefix in ("KRX", "US"):
            source = f"{prefix}_NIFTY_return"
            if source in idx.columns:
                s = idx.set_index("date")[source]
                cum = s.rolling(20).sum()
                cum.index = pd.to_datetime(cum.index).tz_localize(None)
                cum_df = cum.rename(f"{prefix}_NIFTY_cum20").reset_index()
                cum_df.columns = ["date", f"{prefix}_NIFTY_cum20"]
                idx = idx.merge(cum_df, on="date", how="left")
        return idx
    except Exception:
        return None


def _build_features_from_raw(raw_df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    all_frames = []
    grouped = raw_df.groupby("ticker")
    for ticker, sub in grouped:
        sub = sub.copy()
        if len(sub) < 60:
            continue
        try:
            feat = engineer_ticker_features(sub, market="KRX" if ticker.endswith((".KS", ".KQ")) else "US", index_df=index_df)
            # keep only rows with label
            # engineer_ticker_features already computes label but may have NaN dead-zone rows inside
            # drop NaN label before combining (StockFlow behaviour)
            feat = feat.dropna(subset=["label"])
            # also need FEATURE_COLS notna
            feat = feat.dropna(subset=FEATURE_COLS)
            if len(feat) >= 10:
                all_frames.append(feat)
        except Exception as e:
            print(f"  [feature error] {ticker}: {e}")
            continue
    if not all_frames:
        raise RuntimeError("No ticker produced valid feature rows")
    df_feat = pd.concat(all_frames, ignore_index=True)
    df_feat = df_feat.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df_feat


def _time_split(df: pd.DataFrame, train_ratio: float = 0.8):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # drop inf/nan
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["label"])
    n = len(df)
    split_idx = int(n * train_ratio)
    # chronological date threshold
    split_date = df.loc[split_idx, "date"] if split_idx < n else df["date"].max()
    # Use date threshold (StockFlow uses date threshold)
    train_df = df[df["date"] <= split_date - pd.offsets.BDay(FORWARD_DAYS)].copy()
    test_df = df[df["date"] > split_date].copy()
    # fallback if imbalanced due to many same-date rows
    if len(train_df) < int(n * 0.5) or len(test_df) < 20:
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df["label"].astype(int).copy()
    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df["label"].astype(int).copy()
    return X_train, X_test, y_train, y_test, train_df, test_df, split_date


LGBM_CANDIDATES = [
    {
        "name": "legacy",
        "n_estimators": 800, "learning_rate": 0.03, "max_depth": 6,
        "num_leaves": 31, "min_child_samples": 80, "subsample": 0.85,
        "colsample_bytree": 0.85, "bagging_freq": 5,
    },
    {
        "name": "regularized_shallow",
        "n_estimators": 420, "learning_rate": 0.025, "max_depth": 4,
        "num_leaves": 12, "min_child_samples": 55, "subsample": 0.80,
        "colsample_bytree": 0.75, "bagging_freq": 1, "reg_lambda": 4.0,
        "min_split_gain": 0.01,
    },
    {
        "name": "compact",
        "n_estimators": 300, "learning_rate": 0.03, "max_depth": 3,
        "num_leaves": 7, "min_child_samples": 45, "subsample": 0.85,
        "colsample_bytree": 0.70, "bagging_freq": 1, "reg_lambda": 2.0,
        "min_split_gain": 0.02,
    },
    {
        "name": "balanced_regularized",
        "n_estimators": 500, "learning_rate": 0.02, "max_depth": 5,
        "num_leaves": 16, "min_child_samples": 70, "subsample": 0.80,
        "colsample_bytree": 0.80, "bagging_freq": 1, "reg_lambda": 6.0,
        "min_split_gain": 0.01,
    },
    {
        "name": "ultra_regularized",
        "n_estimators": 260, "learning_rate": 0.02, "max_depth": 3,
        "num_leaves": 5, "min_child_samples": 160, "subsample": 0.80,
        "colsample_bytree": 0.65, "bagging_freq": 1, "reg_alpha": 1.0,
        "reg_lambda": 12.0, "min_split_gain": 0.03,
    },
    {
        "name": "directional_8feat",
        "n_estimators": 400, "learning_rate": 0.02, "max_depth": 4,
        "num_leaves": 15, "min_child_samples": 60, "subsample": 0.80,
        "colsample_bytree": 0.75, "reg_lambda": 4.0, "min_split_gain": 0.01,
    },
]


def _scale_pos_weight(y) -> float:
    positives = float(np.asarray(y).sum())
    negatives = float(len(y) - positives)
    return negatives / positives if positives > 0 else 1.0


def _train_lightgbm(X_train, y_train, params: Dict[str, Any]):
    if not _HAS_LGBM:
        raise RuntimeError("lightgbm not available")
    params = dict(params)
    params.pop("name", None)
    # Directional model: exclude NEUTRAL (label 2) to maximize UP vs DOWN AUC.
    # The neutral state is handled at the trading layer via confidence abstention,
    # not as a third LightGBM class that dilutes the directional boundary.
    mask = np.asarray(y_train) != 2
    if mask.sum() < len(y_train):
        X_train = X_train[mask]
        y_train = y_train[mask]
        print(f"  LightGBM directional training: {len(y_train)} rows after filtering NEUTRAL")
    classes = np.unique(np.asarray(y_train, dtype=int))
    model_kwargs = {"random_state": 42, "n_jobs": -1, "verbose": -1, **params}
    if len(classes) > 2:
        print("  LightGBM objective: multiclass (DOWN / UP / NEUTRAL)")
        model_kwargs.update({"objective": "multiclass", "num_class": 3, "class_weight": "balanced"})
    else:
        ratio = _scale_pos_weight(y_train)
        print(f"  LightGBM class_weight ratio (scale_pos_weight): {ratio:.3f}")
        model_kwargs["scale_pos_weight"] = ratio
    model = lgb.LGBMClassifier(**model_kwargs)
    # Fixed tree counts selected through prior-only walk-forward folds keep the
    # final holdout fully unseen. Early stopping on that holdout would leak it.
    model.fit(X_train.values, y_train.values)
    return model


def _predict_proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X.values), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 3:
            # Conditional UP probability excludes NEUTRAL for directional AUC.
            return probabilities[:, 1] / np.clip(probabilities[:, 0] + probabilities[:, 1], 1e-8, None)
        return probabilities[:, 1]
    return np.where(np.asarray(model.predict(X.values)).astype(int) == 1, 0.75, 0.25)


def _walk_forward_backtest(df: pd.DataFrame, params: Dict[str, Any], n_folds: int = 3) -> Dict[str, Any]:
    """Evaluate one candidate with chronological folds and a label-horizon embargo."""
    from sklearn.metrics import brier_score_loss, roc_auc_score

    data = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["label"]).copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")
    dates = pd.Index(data["date"].drop_duplicates().sort_values())
    if len(dates) < 120:
        return {"candidate": params["name"], "folds": [], "mean_auc": float("nan"), "mean_brier": float("nan")}

    first_start = int(len(dates) * 0.50)
    fold_width = max(20, int((len(dates) - first_start) / n_folds))
    folds = []
    for fold in range(n_folds):
        start_idx = first_start + fold * fold_width
        end_idx = min(len(dates), start_idx + fold_width)
        if end_idx - start_idx < 10:
            continue
        test_start, test_end = dates[start_idx], dates[end_idx - 1]
        embargo_end = test_start - pd.offsets.BDay(FORWARD_DAYS)
        train = data[data["date"] <= embargo_end]
        test = data[(data["date"] >= test_start) & (data["date"] <= test_end)]
        direction_test = test[test["label"] != 2]
        if len(train) < 100 or len(direction_test) < 20 or train["label"].nunique() < 2 or direction_test["label"].nunique() < 2:
            continue
        model = _train_lightgbm(train[FEATURE_COLS], train["label"].astype(int), params)
        proba = _predict_proba(model, direction_test[FEATURE_COLS])
        folds.append({
            "fold": fold + 1,
            "train_rows": int(len(train)), "test_rows": int(len(direction_test)),
            "train_end": str(train["date"].max().date()),
            "test_start": str(test_start.date()), "test_end": str(test_end.date()),
            "auc": float(roc_auc_score(direction_test["label"], proba)),
            "brier": float(brier_score_loss(direction_test["label"], proba)),
        })
    aucs = [fold["auc"] for fold in folds]
    briers = [fold["brier"] for fold in folds]
    return {
        "candidate": params["name"], "folds": folds,
        "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "std_auc": float(np.std(aucs)) if aucs else float("nan"),
        "mean_brier": float(np.mean(briers)) if briers else float("nan"),
    }


def _select_lgbm_params(train_df: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Pick the most stable AUC candidate using only data preceding the final test set."""
    reports = []
    for params in LGBM_CANDIDATES:
        print(f"  Walk-forward backtest: {params['name']}")
        report = _walk_forward_backtest(train_df, params)
        reports.append(report)
        print(f"    mean AUC={report['mean_auc']:.4f}, std={report.get('std_auc', float('nan')):.4f}, Brier={report['mean_brier']:.4f}")
    valid = [report for report in reports if np.isfinite(report["mean_auc"])]
    if not valid:
        return LGBM_CANDIDATES[0], reports
    # A modest stability penalty prevents selecting a high-variance fold winner.
    best = max(valid, key=lambda report: (report["mean_auc"] - 0.10 * report.get("std_auc", 0.0), -report["mean_brier"]))
    return next(params for params in LGBM_CANDIDATES if params["name"] == best["candidate"]), reports


def _oof_calibration_predictions(df: pd.DataFrame, params: Dict[str, Any], n_folds: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Return embargoed out-of-fold probabilities for calibration without test leakage."""
    data = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["label"]).copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")
    dates = pd.Index(data["date"].drop_duplicates().sort_values())
    first_start = int(len(dates) * 0.50)
    fold_width = max(20, int((len(dates) - first_start) / n_folds))
    probabilities: List[float] = []
    labels: List[int] = []
    for fold in range(n_folds):
        start_idx = first_start + fold * fold_width
        end_idx = min(len(dates), start_idx + fold_width)
        if end_idx - start_idx < 10:
            continue
        test_start, test_end = dates[start_idx], dates[end_idx - 1]
        train = data[data["date"] <= test_start - pd.offsets.BDay(FORWARD_DAYS)]
        test = data[(data["date"] >= test_start) & (data["date"] <= test_end)]
        direction_test = test[test["label"] != 2]
        if len(train) < 100 or len(direction_test) < 20 or train["label"].nunique() < 2:
            continue
        model = _train_lightgbm(train[FEATURE_COLS], train["label"].astype(int), params)
        probabilities.extend(_predict_proba(model, direction_test[FEATURE_COLS]).tolist())
        labels.extend(direction_test["label"].astype(int).tolist())
    return np.asarray(probabilities, dtype=float), np.asarray(labels, dtype=int)


def _train_sklearn_fallback(X_train, y_train, X_test, y_test):
    if not _HAS_SKLEARN:
        raise RuntimeError("Neither lightgbm nor sklearn available - pip install scikit-learn")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array([0, 1])
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    # HistGB doesn't have scale_pos_weight; use class_weight
    print(f"  sklearn HistGradientBoosting with class_weight balanced")
    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train.values, y_train.values)
    return model


def _calibrate_platt(y_proba, y_true):
    """Fit monotonic Platt scaling on OOF predictions without reversing rank."""
    try:
        from scipy.optimize import minimize  # type: ignore
        from sklearn.metrics import roc_auc_score  # type: ignore
        oof_auc = roc_auc_score(y_true, y_proba)
        # Only calibrate when OOF shows genuine ranking signal; otherwise
        # Platt fitting on near-random OOF merely shifts the holdout distribution
        # and destroys the usable 0.5 threshold (see 5y/39-ticker run where
        # OOF 0.506 calibrated holdout to all >0.55 with balanced 0.50).
        if len(np.unique(y_true)) < 2 or oof_auc < 0.54:
            return {"a": -1.0, "b": 0.0, "identity": True, "oof_auc": float(oof_auc)}
        def objective(params):
            a, b = params
            eps = 1e-8
            p = np.clip(y_proba, eps, 1 - eps)
            logit = np.log(p / (1 - p))
            p_cal = 1 / (1 + np.exp(a * logit + b))
            p_cal = np.clip(p_cal, eps, 1 - eps)
            loss = -np.mean(y_true * np.log(p_cal) + (1 - y_true) * np.log(1 - p_cal))
            return loss
        # The runtime formula is sigmoid(-a * logit - b), so a<0 preserves
        # ordering. Constraining it prevents calibration from reversing AUC.
        res = minimize(
            objective, x0=[-1.0, 0.0], method="L-BFGS-B",
            bounds=[(-5.0, -1e-4), (None, None)], options={"maxiter": 500},
        )
        a, b = res.x
        return {"a": float(a), "b": float(b), "identity": False}
    except Exception as e:
        print(f"  Calibration fallback (no scipy): {e}")
        return {"a": -1.0, "b": 0.0, "identity": True}


def _select_action_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Choose an abstention threshold by OOF balanced accuracy, not holdout tuning."""
    from sklearn.metrics import balanced_accuracy_score

    best = {"confidence_min": 0.5, "coverage": 1.0, "balanced_accuracy": 0.5}
    for threshold in np.arange(0.50, 0.71, 0.02):
        keep = np.maximum(y_proba, 1.0 - y_proba) >= threshold
        if keep.mean() < 0.25 or len(np.unique(y_true[keep])) < 2:
            continue
        score = float(balanced_accuracy_score(y_true[keep], (y_proba[keep] >= 0.5).astype(int)))
        candidate = {"confidence_min": round(float(threshold), 2), "coverage": round(float(keep.mean()), 4), "balanced_accuracy": round(score, 4)}
        if (candidate["balanced_accuracy"], candidate["coverage"]) > (best["balanced_accuracy"], best["coverage"]):
            best = candidate
    return best


def main():
    parser = argparse.ArgumentParser(description="StockOracle ML training (StockFlow pipeline)")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated tickers, e.g. 005930.KS,AAPL")
    parser.add_argument("--period", type=str, default="2y", help="yfinance period (e.g. 2y, 5y, 1y)")
    parser.add_argument("--synthetic-only", action="store_true", help="Skip yfinance, use synthetic only (fast, offline)")
    parser.add_argument("--input", type=str, default="", help="Prebuilt training_features.parquet path to reuse")
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR), help="Model output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Time-based train ratio")
    parser.add_argument("--skip-tuning", action="store_true", help="Use the legacy LightGBM parameters without walk-forward selection")
    args = parser.parse_args()

    print("=" * 70)
    print(" StockOracle - ML Training Pipeline (StockFlow data-leakage-safe)")
    print("=" * 70)

    models_dir = Path(args.output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load or collect raw data ──────────────────────────────────
    df_feat: Optional[pd.DataFrame] = None
    if args.input and Path(args.input).exists():
        print(f"\n[Step 1] Loading prebuilt features from {args.input}")
        df_feat = pd.read_parquet(args.input)
        print(f"  Loaded {len(df_feat):,} rows | {df_feat['ticker'].nunique()} tickers")
    else:
        if args.tickers:
            tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        else:
            # default sample 30 random from universe
            tickers = list(DEFAULT_TICKERS)
            # if synthetic-only, keep default; else try to expand slightly via StockOracle universe
            if not args.synthetic_only:
                try:
                    from api.index import KR_STOCK_MAP, US_TICKERS  # type: ignore
                    extra_kr = list(KR_STOCK_MAP.values())[:10]
                    extra_us = US_TICKERS[:10]
                    tickers = list(dict.fromkeys(tickers + extra_kr + extra_us))
                except Exception:
                    pass
            # sample 40 max for speed
            if len(tickers) > 40:
                random.seed(42)
                tickers = random.sample(tickers, 40)

        print(f"\n[Step 1] Collecting OHLCV for {len(tickers)} tickers (period={args.period})")
        # Hybrid mode: if real fetch fails for some tickers, synthetic fallback ensures domain coverage
        # To reduce synthetic vs real domain shift, synthetic vol/drift now matches real KRX/US stats:
        #   real KRX 20d vol mean ~1.8%, US ~1.5%; synthetic vol 0.012-0.025 covers both. Drift -0.05% to 0.1% matches.
        if args.synthetic_only:
            print("  synthetic-only: generating synthetic OHLCV for all tickers (domain shift test mode)")
            raws = []
            for t in tickers:
                raws.append(_synthetic_ohlcv(t, days=600, seed=0))
            raw_df = pd.concat(raws, ignore_index=True)
        else:
            # Hybrid: real data + synthetic fallback for failed tickers (reduces survivorship bias)
            raw_df = _collect_training_data(tickers, period=args.period, synthetic_fallback=True)
            # Optional: add 20% synthetic augmentation to improve robustness on low-vol regimes
            if len(raw_df) < 5000:
                print("  Hybrid augmentation: adding 20% synthetic to cover low-vol regimes")
                synth_extra = []
                for t in tickers[:max(1, len(tickers)//5)]:
                    synth_extra.append(_synthetic_ohlcv(t+"_SYN", days=600, seed=99))
                raw_df = pd.concat([raw_df] + synth_extra, ignore_index=True)

        print(f"  Combined raw: {len(raw_df):,} rows | {raw_df['ticker'].nunique()} tickers")
        raw_df["date"] = pd.to_datetime(raw_df["date"])

        # index df for market features
        print("\n[Step 1b] Fetching index data for market features...")
        try:
            s = raw_df["date"].min().strftime("%Y-%m-%d")
            e = raw_df["date"].max().strftime("%Y-%m-%d")
            index_df = _fetch_index_df(s, e)
            if index_df is not None:
                print(f"  Index data: {len(index_df)} rows")
            else:
                print("  Index data unavailable - using neutral fallback")
        except Exception as e:
            print(f"  Index fetch failed: {e}")
            index_df = None

        print("\n[Step 2] Engineering features (leakage-safe, walk-forward)...")
        df_feat = _build_features_from_raw(raw_df, index_df=index_df)
        print(f"  Features built: {len(df_feat):,} rows | {df_feat['ticker'].nunique()} tickers")
        # save feature parquet for reuse
        try:
            FEATURE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
            df_feat.to_parquet(FEATURE_PARQUET, index=False)
            print(f"  Saved features to {FEATURE_PARQUET}")
        except Exception as e:
            print(f"  Could not save parquet: {e}")

    # ── Data quality checks ───────────────────────────────────────────────
    print("\n[Step 3] Data quality checks...")
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["label"])
    print(f"  Clean rows: {len(df_feat):,}")
    label_dist = df_feat["label"].value_counts()
    total = len(df_feat)
    up = int(label_dist.get(1, 0))
    down = int(label_dist.get(0, 0))
    neutral = int(label_dist.get(2, 0))
    print(f"  Label: UP {up:,} ({up/total*100:.1f}%) | DOWN {down:,} ({down/total*100:.1f}%) | NEUTRAL {neutral:,} ({neutral/total*100:.1f}%)")
    if abs(up - down) / max(up + down, 1) > 0.15:
        ratio = down / max(up, 1)
        print(f"  Imbalance detected - scale_pos_weight ~ {ratio:.3f}")

    # ── Split ─────────────────────────────────────────────────────────────
    print("\n[Step 4] Time-based split...")
    X_train, X_test, y_train, y_test, train_df, test_df, split_date = _time_split(df_feat, train_ratio=args.train_ratio)
    print(f"  Train: {len(X_train):,} rows (- {pd.to_datetime(split_date).date()})")
    print(f"  Test:  {len(X_test):,} rows ({pd.to_datetime(split_date).date()} - {df_feat['date'].max().date()})")
    def _directional_up_rate(labels) -> float:
        directional = labels[labels != 2]
        return float((directional == 1).mean() * 100) if len(directional) else 0.0
    print(f"  Train directional UP rate: {_directional_up_rate(y_train):.1f}% | Test directional UP rate: {_directional_up_rate(y_test):.1f}%")

    # ── Parameter selection and calibration split ─────────────────────────
    selected_params: Dict[str, Any] = {}
    backtest_reports: List[Dict[str, Any]] = []
    calibration_proba = np.asarray([], dtype=float)
    calibration_y = np.asarray([], dtype=int)
    fit_train_df = train_df.copy()
    if _HAS_LGBM:
        print("\n[Step 5] Walk-forward parameter backtest (training period only)...")
        if args.skip_tuning:
            selected_params = dict(LGBM_CANDIDATES[0])
            print("  Tuning skipped; using legacy parameters")
        else:
            selected_params, backtest_reports = _select_lgbm_params(train_df)
            print(f"  Selected: {selected_params['name']}")

        # OOF calibration fits each calibration probability with a model that
        # predates it. The final model can then use all pre-test data safely.
        calibration_proba, calibration_y = _oof_calibration_predictions(train_df, selected_params)
        print(f"  Final model rows: {len(fit_train_df):,} | OOF calibration rows: {len(calibration_y):,} | untouched test rows: {len(test_df):,}")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n[Step 5] Training model...")
    model = None
    model_type = "none"
    if _HAS_LGBM:
        try:
            model = _train_lightgbm(
                fit_train_df[FEATURE_COLS], fit_train_df["label"].astype(int), selected_params or LGBM_CANDIDATES[0],
            )
            model_type = "LightGBM"
            print("  - LightGBM trained")
        except Exception as e:
            print(f"  LightGBM failed: {e} - falling back to sklearn")
    if model is None and _HAS_SKLEARN:
        try:
            model = _train_sklearn_fallback(X_train, y_train, X_test, y_test)
            model_type = "HistGradientBoosting"
            print("  - sklearn HistGradientBoosting trained")
        except Exception as e:
            print(f"  sklearn training failed: {e}")
            raise SystemExit(1)
    if model is None:
        print("  - No training backend available - install lightgbm or scikit-learn")
        raise SystemExit(1)

    # ── Predict & calibrate ───────────────────────────────────────────────
    print("\n[Step 6] Predictions & calibration...")
    model_X_train = fit_train_df[FEATURE_COLS] if model_type == "LightGBM" else X_train
    model_y_train = fit_train_df["label"].astype(int) if model_type == "LightGBM" else y_train
    y_train_proba = _predict_proba(model, model_X_train)
    y_test_proba = _predict_proba(model, X_test)
    if len(calibration_y) >= 30 and len(np.unique(calibration_y)) == 2 and model_type == "LightGBM":
        calib = _calibrate_platt(calibration_proba, calibration_y)
        calibration_source = "expanding walk-forward out-of-fold predictions (14-business-day embargo)"
    else:
        # Never fit probability calibration on the final test set.
        calib = {"a": -1.0, "b": 0.0, "identity": True}
        calibration_source = "identity (no independent calibration split)"
    print(f"  Platt calibration: a={calib['a']:.4f}, b={calib['b']:.4f}")
    if len(calibration_y) >= 30:
        calibration_probs = 1 / (1 + np.exp(calib["a"] * np.log(np.clip(calibration_proba, 1e-8, 1 - 1e-8) / (1 - np.clip(calibration_proba, 1e-8, 1 - 1e-8))) + calib["b"]))
        action_threshold = _select_action_threshold(calibration_y, np.clip(calibration_probs, 0.01, 0.99))
    else:
        action_threshold = {"confidence_min": 0.5, "coverage": 1.0, "balanced_accuracy": 0.5}
    print(f"  OOF action threshold: confidence >= {action_threshold['confidence_min']:.2f} | coverage {action_threshold['coverage']*100:.1f}% | balanced accuracy {action_threshold['balanced_accuracy']:.3f}")
    # apply calibration to test set for reporting
    eps = 1e-8
    y_test_proba_cal = 1 / (1 + np.exp(calib["a"] * np.log(np.clip(y_test_proba, eps, 1-eps) / (1 - np.clip(y_test_proba, eps, 1-eps))) + calib["b"]))
    y_test_proba_cal = np.clip(y_test_proba_cal, 0.01, 0.99)

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n[Step 7] Evaluation (test set)...")
    from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report  # type: ignore
    direction_train = model_y_train != 2
    direction_test = y_test != 2
    y_test_direction = y_test[direction_test]
    y_test_proba_direction = y_test_proba_cal[direction_test.values]
    try:
        train_auc = roc_auc_score(model_y_train[direction_train].values, y_train_proba[direction_train.values])
    except Exception:
        train_auc = float("nan")
    test_auc = roc_auc_score(y_test_direction.values, y_test_proba_direction)
    y_test_pred = (y_test_proba_direction >= 0.5).astype(int)
    test_acc = accuracy_score(y_test_direction.values, y_test_pred)
    test_bacc = balanced_accuracy_score(y_test_direction.values, y_test_pred)
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Test AUC (calibrated): {test_auc:.4f}")
    print(f"  Accuracy: {test_acc:.4f} | Balanced: {test_bacc:.4f}")
    print(f"  Precision: {precision_score(y_test_direction.values, y_test_pred, zero_division=0):.4f}")
    print(f"  Recall: {recall_score(y_test_direction.values, y_test_pred, zero_division=0):.4f}")
    print(f"  F1: {f1_score(y_test_direction.values, y_test_pred, zero_division=0):.4f}")
    print("\n" + classification_report(y_test_direction.values, y_test_pred, target_names=["DOWN", "UP"], zero_division=0))
    cm = confusion_matrix(y_test_direction.values, y_test_pred)
    print(f"  Confusion Matrix:\n{cm}")
    walk_forward_auc = max(
        (report.get("mean_auc", float("nan")) for report in backtest_reports),
        default=float("nan"),
    )
    validation_passed = bool(
        test_auc >= 0.54
        and test_bacc >= 0.52
        and (not backtest_reports or (np.isfinite(walk_forward_auc) and walk_forward_auc >= 0.54))
    )
    validation_reason = (
        "passed_untouched_holdout_and_walk_forward"
        if validation_passed else
        "rejected: require holdout AUC >= 0.54, balanced accuracy >= 0.52, and walk-forward AUC >= 0.54"
    )
    print(f"  Deployment validation: {'PASS' if validation_passed else 'REJECT'} ({validation_reason})")

    # feature importance - robust for both LightGBM and sklearn
    print("\n[Step 8] Feature importances (top 10)...")
    try:
        importances = None
        importance_source = "unknown"
        if hasattr(model, "feature_importances_"):
            importances = np.array(model.feature_importances_, dtype=float)
            # HistGradientBoosting often returns all zeros on small synthetic - fallback to permutation
            if np.all(importances == 0):
                raise ValueError("all zero - need permutation")
            importance_source = "feature_importances_"
        elif hasattr(model, "feature_importance"):
            importances = np.array(model.feature_importance(importance_type="gain"), dtype=float)
            importance_source = "lgbm_gain"
        if importances is None or np.all(importances == 0):
            raise ValueError("no importances")
        order = np.argsort(importances)[::-1]
        for rank, idx in enumerate(order[:10], 1):
            print(f"  {rank:2d}. {FEATURE_COLS[idx]:25s} {importances[idx]:8.0f} [{importance_source}]")
        feat_imp = [{"feature": FEATURE_COLS[i], "importance": float(importances[i]), "source": importance_source} for i in range(len(FEATURE_COLS))]
    except Exception as e:
        # Permutation importance fallback (sklearn) - measures AUC drop when shuffling each feature
        print(f"  Fallback to permutation importance ({e})")
        try:
            from sklearn.inspection import permutation_importance
            # Use test set for unbiased estimate, 5 repeats, AUC scoring
            perm = permutation_importance(model, X_test[direction_test].values, y_test_direction.values, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=1)
            importances = perm.importances_mean
            order = np.argsort(importances)[::-1]
            for rank, idx in enumerate(order[:10], 1):
                print(f"  {rank:2d}. {FEATURE_COLS[idx]:25s} {importances[idx]:8.4f} [permutation]")
            feat_imp = [{"feature": FEATURE_COLS[i], "importance": float(importances[i]), "source": "permutation_auc"} for i in range(len(FEATURE_COLS))]
        except Exception as e2:
            print(f"  Permutation also failed: {e2}")
            feat_imp = [{"feature": c, "importance": 0, "source": "failed"} for c in FEATURE_COLS]

    # ── Save artifacts ────────────────────────────────────────────────────
    print("\n[Step 9] Saving artifacts...")
    import joblib as jb
    models_dir.mkdir(parents=True, exist_ok=True)
    # model pickle
    model_path = models_dir / "lgbm_model.pkl"
    jb.dump(model, str(model_path))
    print(f"  Model - {model_path}")
    # Try also save lightgbm booster json
    try:
        if _HAS_LGBM and hasattr(model, "booster_"):
            booster = model.booster_
            booster.save_model(str(models_dir / "lgbm_model.txt"))
    except Exception:
        pass
    # feature columns
    with open(models_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, ensure_ascii=False, indent=2)
    print(f"  Features - {models_dir / 'feature_columns.json'}")
    # metadata
    metadata = {
        "model_type": model_type,
        "label_mode": "three_class_directional_with_neutral",
        "trained_at": datetime.now().isoformat(),
        "train_rows": int(len(X_train)),
        "model_fit_rows": int(len(model_X_train)),
        "calibration_rows": int(len(calibration_y)),
        "test_rows": int(len(X_test)),
        "directional_test_rows": int(len(y_test_direction)),
        "train_split_date": pd.to_datetime(split_date).isoformat(),
        "train_period": args.period,
        "features": FEATURE_COLS,
        "metrics": {
            "train_auc": float(train_auc) if not math.isnan(train_auc) else None,
            "test_auc": float(test_auc),
            "test_accuracy": float(test_acc),
            "test_balanced_accuracy": float(test_bacc),
            "test_precision": float(precision_score(y_test_direction.values, y_test_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test_direction.values, y_test_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test_direction.values, y_test_pred, zero_division=0)),
        },
        "calibration": {"method": "Platt scaling (logit)", "params": calib, "source": calibration_source},
        "action_threshold": action_threshold,
        "selected_parameters": selected_params if model_type == "LightGBM" else {},
        "walk_forward_backtest": backtest_reports,
        "validation": {
            "passed": validation_passed,
            "reason": validation_reason,
            "requirements": {"holdout_auc_min": 0.54, "holdout_balanced_accuracy_min": 0.52, "walk_forward_auc_min": 0.54},
        },
        "feature_importance": sorted(feat_imp, key=lambda x: x["importance"], reverse=True),
        "leakage_safeguards": [
            "time-based split (no shuffle)",
            f"{FORWARD_DAYS}-business-day embargo in walk-forward calibration folds",
            "volatility_ratio per-row (fixed)",
            "label via shift(-14) with 3% dead-zone",
            "three-class target: DOWN / UP / NEUTRAL within the dead-zone",
            "all indicators causal (rolling only past)",
            "normalization to price (no future scaling)",
        ],
    }
    with open(models_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Metadata - {models_dir / 'training_metadata.json'}")

    print("\n" + "=" * 70)
    print("- TRAINING COMPLETE")
    print("=" * 70)
    print(f" Test AUC: {test_auc:.4f} {'- production-ready' if test_auc>0.54 else '! consider more data/features'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
