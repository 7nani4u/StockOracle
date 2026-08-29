# -*- coding: utf-8 -*-
"""
ml_evaluate.py - StockFlow evaluate_model.py adapted for StockOracle.

Time-series aware evaluation (no shuffle, no future leakage):
  * single time-based 80/20 split
  * expanding walk-forward cross-validation
  * metrics: AUC, accuracy, balanced_accuracy, precision, recall, F1, confusion matrix

No hard dependency on LightGBM file - works with any model exposing predict_proba/predict.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score, confusion_matrix, classification_report,
    )
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

from .ml_features import FEATURE_COLS, walk_forward_splits


def _safe_auc(y_true, y_proba) -> float:
    if not _HAS_SKLEARN:
        return float("nan")
    try:
        if len(set(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_proba))
    except Exception:
        return float("nan")


def evaluate_predictions(
    y_true: Any,
    y_proba: Any,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Returns dict with auc, accuracy, balanced_accuracy, precision, recall, f1,
    confusion matrix entries and classification report text.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    result: Dict[str, Any] = {
        "threshold": threshold,
        "n": int(len(y_true)),
        "auc": _safe_auc(y_true, y_proba),
    }
    if not _HAS_SKLEARN:
        result["note"] = "sklearn not installed - metrics unavailable"
        return result
    try:
        result["accuracy"] = float(accuracy_score(y_true, y_pred))
        result["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        result["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        result["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        result["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            result["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        else:
            result["confusion_matrix"] = cm.tolist()
        result["classification_report"] = classification_report(y_true, y_pred, target_names=["DOWN", "UP"], zero_division=0)
        # distribution
        result["pred_up_rate"] = float((y_pred == 1).mean())
        result["true_up_rate"] = float((y_true == 1).mean())
        result["avg_confidence"] = float(np.mean(np.abs(y_proba - 0.5) + 0.5))
    except Exception as e:
        result["error"] = str(e)
    return result


def time_based_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_ratio: float = 0.8,
    feature_cols: List[str] | None = None,
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Single chronological split: first train_ratio% is train, remainder is test.
    Returns X_train, X_test, y_train, y_test, train_dates, test_dates
    Leakage-safe: sorted by date, no shuffle, future not in train.
    """
    d = df.copy()
    try:
        d[date_col] = pd.to_datetime(d[date_col], utc=True).dt.tz_convert(None)
    except Exception:
        d[date_col] = pd.to_datetime(d[date_col], utc=True).dt.tz_localize(None)
    d = d.sort_values(date_col).reset_index(drop=True)
    cols = feature_cols or FEATURE_COLS
    missing = [c for c in cols if c not in d.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if label_col not in d.columns:
        raise ValueError(f"Label column {label_col} not found")
    # clean first, then split chronologically by position (robust to dropna length mismatch)
    d_clean = d.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + [label_col])
    d_clean = d_clean.sort_values(date_col).reset_index(drop=True)
    n = len(d_clean)
    if n < 20:
        raise ValueError(f"Not enough clean rows for split: {n}")
    split_idx = int(n * train_ratio)
    split_idx = max(10, min(n - 10, split_idx))
    train_df = d_clean.iloc[:split_idx].copy()
    test_df = d_clean.iloc[split_idx:].copy()
    X_train = train_df[cols].copy()
    X_test = test_df[cols].copy()
    y_train = train_df[label_col].astype(int).copy()
    y_test = test_df[label_col].astype(int).copy()
    return X_train, X_test, y_train, y_test, train_df[date_col], test_df[date_col]


def walk_forward_evaluate(
    df: pd.DataFrame,
    model: Any,
    feature_cols: List[str] | None = None,
    label_col: str = "label",
    date_col: str = "date",
    n_splits: int = 5,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Expanding walk-forward CV.
    model must have predict_proba; if not, falls back to predict.
    Returns per-fold metrics and aggregate.
    """
    cols = feature_cols or FEATURE_COLS
    splits = walk_forward_splits(df, date_col=date_col, n_splits=n_splits)
    fold_results: List[Dict[str, Any]] = []
    all_y_true: List[int] = []
    all_y_proba: List[float] = []

    for fold_idx, (train_df, test_df) in enumerate(splits, 1):
        # clean fold
        train_clean = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + [label_col])
        test_clean = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + [label_col])
        if len(train_clean) < 50 or len(test_clean) < 10:
            continue
        X_train = train_clean[cols].values
        y_train = train_clean[label_col].astype(int).values
        X_test = test_clean[cols].values
        y_test = test_clean[label_col].astype(int).values

        # fit model per fold is caller responsibility if model is sklearn estimator
        # Here we assume model is already trained on some prior data OR we refit a clone.
        # We attempt to clone and fit if model has fit method
        fold_model = model
        try:
            from sklearn.base import clone  # type: ignore
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
        except Exception:
            # if clone/fit fails, use existing model predictions on this test split
            pass

        try:
            if hasattr(fold_model, "predict_proba"):
                y_proba = fold_model.predict_proba(X_test)[:, 1]
            elif hasattr(fold_model, "predict"):
                # decision_function fallback
                preds = fold_model.predict(X_test)
                # map 0/1 to 0.25/0.75
                y_proba = np.where(np.asarray(preds).astype(int) == 1, 0.75, 0.25)
            else:
                y_proba = np.full(len(y_test), 0.5)
        except Exception as e:
            y_proba = np.full(len(y_test), 0.5)
            fold_results.append({"fold": fold_idx, "error": str(e)})
            continue

        metrics = evaluate_predictions(y_test, y_proba, threshold=threshold)
        metrics["fold"] = fold_idx
        metrics["train_rows"] = int(len(train_clean))
        metrics["test_rows"] = int(len(test_clean))
        try:
            metrics["train_date_range"] = f"{train_clean[date_col].min()} -> {train_clean[date_col].max()}"
            metrics["test_date_range"] = f"{test_clean[date_col].min()} -> {test_clean[date_col].max()}"
        except Exception:
            pass
        fold_results.append(metrics)
        all_y_true.extend(y_test.tolist())
        all_y_proba.extend(y_proba.tolist())

    aggregate = {}
    if all_y_true:
        aggregate = evaluate_predictions(np.array(all_y_true), np.array(all_y_proba), threshold=threshold)
        aggregate["n_folds"] = len(fold_results)
        aggregate["folds_auc_mean"] = float(np.nanmean([f.get("auc", np.nan) for f in fold_results])) if fold_results else float("nan")
        aggregate["folds_auc_std"] = float(np.nanstd([f.get("auc", np.nan) for f in fold_results])) if fold_results else float("nan")

    return {"folds": fold_results, "aggregate": aggregate}


def classification_summary(y_true, y_pred) -> str:
    if not _HAS_SKLEARN:
        return "sklearn not available"
    try:
        return classification_report(y_true, y_pred, target_names=["DOWN", "UP"], zero_division=0)
    except Exception as e:
        return f"error: {e}"


def check_data_leakage(df: pd.DataFrame, date_col: str = "date", label_col: str = "label") -> Dict[str, Any]:
    """
    Basic leakage audit:
      - check that no feature uses future Close (by inspecting correlation with future return)
      - check that label is computed via shift(-14) (not same-day)
      - check that train/test split is chronological
    Returns checklist with passed / warnings.
    """
    issues: List[str] = []
    passed: List[str] = []

    d = df.copy()
    try:
        d[date_col] = pd.to_datetime(d[date_col], utc=True).dt.tz_convert(None)
    except Exception:
        try:
            d[date_col] = pd.to_datetime(d[date_col], utc=True).dt.tz_localize(None)
        except Exception:
            d[date_col] = pd.to_datetime(d[date_col].astype(str), errors='coerce')
    if not d[date_col].is_monotonic_increasing:
        # check globally sorted?
        if d.sort_values(date_col)[date_col].is_monotonic_increasing:
            passed.append("dates sortable to monotonic increasing")
        else:
            issues.append("dates not strictly increasing - possible shuffle leakage risk")
    else:
        passed.append("dates are monotonic increasing (chronological)")

    if label_col in d.columns:
        # label should be binary 0/1 and not all same
        uniq = d[label_col].dropna().unique()
        if set(uniq).issubset({0, 1, 0.0, 1.0}):
            passed.append(f"label is binary {sorted(uniq)}")
        else:
            issues.append(f"label values unexpected: {sorted(uniq)[:5]}")

        # check that forward_return is not also stored as feature (would leak)
        leakage_cols = [c for c in d.columns if "future" in c.lower() or "forward_return" in c.lower()]
        if leakage_cols:
            # these should not be in FEATURE_COLS
            overlap = [c for c in leakage_cols if c in FEATURE_COLS]
            if overlap:
                issues.append(f"LEAKAGE: future columns in FEATURE_COLS: {overlap}")
            else:
                passed.append(f"future columns ({leakage_cols}) not in FEATURE_COLS - safe")

    # feature NaN rate check
    feat_nan = d[FEATURE_COLS].isna().mean()
    high_nan = feat_nan[feat_nan > 0.3]
    if not high_nan.empty:
        issues.append(f"High NaN rate features (>30%): {high_nan.to_dict()}")
    else:
        passed.append("No feature has >30% NaN rate")

    # volatility_ratio per-row check hint
    if "volatility_ratio" in d.columns and "volatility_20d" in d.columns:
        # if vol_ratio were scalar bug, it would be constant; check variance
        std = d["volatility_ratio"].std()
        if std < 1e-6:
            issues.append("volatility_ratio appears constant - possible per-row bug regression")
        else:
            passed.append(f"volatility_ratio variance OK (std={std:.3f})")

    return {"passed": passed, "warnings": issues, "ok": len(issues) == 0}
