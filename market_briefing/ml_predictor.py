# -*- coding: utf-8 -*-
"""
ml_predictor.py - StockFlow LightGBM price-direction predictor adapted for StockOracle.

StockFlow ml_predictor.py와의 차이 및 StockOracle 통합 설계:
  * 기존 StockFlow: INDIA/US 이원화 (NIFTY/BANKNIFTY vs SPY/QQQ/VIX)
    -> StockOracle: KRX/US 이원화 (KOSPI200/KOSDAQ vs SPY/QQQ/VIX). 슬롯명은 호환성 때문에
      동일 유지 (NIFTY_return 등) - pretrained StockFlow 모델을 그대로 로드 가능.
  * VIX 캐시: StockOracle은 yfinance TTL 캐시와 동일 패턴 사용, 과도한 재요청 방지.
  * pandas_ta 미사용 -> ml_features.compute_feature_vector 재사용 (순수 numpy/pandas)
  * 모델 로딩: lightgbm이 없어도 sklearn.ensemble.HistGradientBoostingClassifier fallback으로 동작.
  * Leakage 방지: inference는 현재봉 close까지만 사용, label(미래 14일) 미사용.

Public API:
  predict_direction(metrics: dict) -> dict
  predict_from_ohlcv(ticker: str, closes, highs, lows, volumes, market) -> dict
  load_model() -> bool
  is_model_available() -> bool

metrics dict expectation:
  {ticker, market: "KRX"|"US", closes_20d/highs_20d/lows_20d/volumes_20d 등은
   선택이며, 없으면 yfinance로 6개월치 자동 조회 시도 (실패 시 graceful fallback).}
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except Exception:
    _HAS_YFINANCE = False
    yf = None

from .ml_features import FEATURE_COLS, compute_feature_vector

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
# StockOracle 루트 기준: StockOracle/models/*.pkl, *.json
# 로컬 개발/배포 공통으로 동작하도록 여러 후보 탐색
_THIS_DIR = Path(__file__).resolve().parent  # market_briefing/
_PROJECT_ROOT = _THIS_DIR.parent  # StockOracle/
CANDIDATE_MODEL_DIRS = [
    _PROJECT_ROOT / "models",
    _PROJECT_ROOT / "ml_models",
    Path.cwd() / "models",
    Path.cwd() / "StockOracle" / "models",
]

# StockFlow pretrained fallback
STOCKFLOW_MODEL_DIR = Path(r"C:\Users\Administrator\Documents\trae_projects\stockflow\models")

MODEL_FILENAME = "lgbm_model.pkl"
MODEL_JSON_FILENAME = "lgbm_model.json"
COLUMNS_FILENAME = "feature_columns.json"
METADATA_FILENAME = "training_metadata.json"

# ── Module-level state ─────────────────────────────────────────────────────
_MODEL: Any = None
_FEATURE_COLS: List[str] = []
_CALIB_PARAMS: Optional[Dict[str, float]] = None  # {"a":..., "b":...} Platt scaling
_MODEL_AVAILABLE: bool = False
_MODEL_TYPE: str = "none"  # lightgbm | sklearn | none
_MODEL_META: Dict[str, Any] = {}

# ── Index cache (market-aware) ────────────────────────────────────────────
# 한 세션에서 지수 데이터는 한 번만 조회, TTL 30분
_INDEX_CACHE: Dict[str, Dict[str, float]] = {}
_INDEX_CACHE_TS: Dict[str, float] = {}
_INDEX_TTL = 1800.0

# StockFlow India vs US 슬롯 매핑 유지
#   india 슬롯: NIFTY_return(India) / BANKNIFTY_return / India_VIX
#   us 슬롯   : SPY / QQQ / VIX (동일 슬롯명 재사용)
# StockOracle KRX/US 매핑:
#   KRX: KOSPI 200 (^KS200 or 069500.KS proxy, fallback ^KS11) / KOSDAQ (^KQ11) / VIX 15 fallback
#   US : SPY / QQQ / ^VIX

INDEX_SYMBOLS = {
    "KRX": {"market": "^KS200", "fallback_market": "^KS11", "sector": "^KQ11", "vix": "^VIX"},
    # ^KS200이 yfinance에서 조회 불가 시 ^KS11 fallback. VIX는 KRX 전용 없어 ^VIX 또는 15.0
    "US": {"market": "SPY", "sector": "QQQ", "vix": "^VIX"},
}


def _find_model_dir() -> Optional[Path]:
    for d in CANDIDATE_MODEL_DIRS:
        if d.exists() and (d / COLUMNS_FILENAME).exists():
            return d
    # StockFlow pretrained fallback
    if (STOCKFLOW_MODEL_DIR / COLUMNS_FILENAME).exists():
        return STOCKFLOW_MODEL_DIR
    # any existing models dir even without columns (will be created by training)
    for d in CANDIDATE_MODEL_DIRS:
        if d.exists():
            return d
    return None


def _try_load_lightgbm(model_path: Path):
    try:
        import lightgbm as lgb  # type: ignore
        import joblib  # type: ignore
        m = joblib.load(str(model_path))
        return m, "lightgbm"
    except Exception as e:
        logger.debug(f"[ML] LightGBM load failed: {e}")
        return None, None


def _try_load_sklearn(model_path: Path):
    try:
        import joblib  # type: ignore
        m = joblib.load(str(model_path))
        # sklearn HistGradientBoosting also uses joblib pickle
        return m, "sklearn"
    except Exception as e:
        logger.debug(f"[ML] sklearn load failed: {e}")
        return None, None


def _load_columns(model_dir: Path) -> List[str]:
    try:
        with open(model_dir / COLUMNS_FILENAME, encoding="utf-8") as f:
            cols = json.load(f)
            if isinstance(cols, list) and cols:
                return cols
    except Exception as e:
        logger.debug(f"[ML] columns load failed: {e}")
    return list(FEATURE_COLS)


def _load_calib_params(model_dir: Path) -> Optional[Dict[str, float]]:
    try:
        meta_path = model_dir / METADATA_FILENAME
        if not meta_path.exists():
            return None
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
            calib = meta.get("calibration") or {}
            params = calib.get("params") or {}
            if "a" in params and "b" in params:
                return {"a": float(params["a"]), "b": float(params["b"])}
    except Exception:
        pass
    return None


def load_model(force_reload: bool = False) -> bool:
    """
    탐색 경로에서 모델과 메타데이터를 로드. 성공 시 True.
    lightgbm -> sklearn 순으로 시도. 둘 다 실패 시 dummy baseline (0.5)로 fallback.
    """
    global _MODEL, _FEATURE_COLS, _CALIB_PARAMS, _MODEL_AVAILABLE, _MODEL_TYPE, _MODEL_META
    if _MODEL_AVAILABLE and not force_reload and _MODEL is not None:
        return True

    model_dir = _find_model_dir()
    if model_dir is None:
        # create StockOracle/models for future training
        default_dir = _PROJECT_ROOT / "models"
        try:
            default_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        logger.warning("[ML] No model directory found - will use heuristic fallback until training")
        _MODEL_AVAILABLE = False
        _MODEL_TYPE = "none"
        _FEATURE_COLS = list(FEATURE_COLS)
        return False

    # columns
    _FEATURE_COLS = _load_columns(model_dir)

    # calibration
    _CALIB_PARAMS = _load_calib_params(model_dir)

    # model binary
    pkl_path = model_dir / MODEL_FILENAME
    # prefer .pkl if exists, else try .json(booster)
    model = None
    mtype = None
    if pkl_path.exists():
        model, mtype = _try_load_lightgbm(pkl_path)
        if model is None:
            model, mtype = _try_load_sklearn(pkl_path)
    # also try lightgbm json booster
    json_path = model_dir / MODEL_JSON_FILENAME
    if model is None and json_path.exists():
        try:
            import lightgbm as lgb  # type: ignore
            m = lgb.Booster(model_file=str(json_path))
            model, mtype = m, "lightgbm"
        except Exception as e:
            logger.debug(f"[ML] json booster load failed: {e}")

    if model is None:
        logger.warning(f"[ML] Model file not found or failed to load in {model_dir}")
        _MODEL_AVAILABLE = False
        _MODEL_TYPE = "none"
        # dummy는 유지하되 is_model_available()=False로 알림
        _MODEL = None
        try:
            with open(model_dir / METADATA_FILENAME, encoding="utf-8") as f:
                _MODEL_META = json.load(f)
        except Exception:
            _MODEL_META = {}
        return False

    _MODEL = model
    _MODEL_TYPE = mtype or "unknown"
    _MODEL_AVAILABLE = True
    try:
        with open(model_dir / METADATA_FILENAME, encoding="utf-8") as f:
            _MODEL_META = json.load(f)
    except Exception:
        _MODEL_META = {"model_type": _MODEL_TYPE, "features": _FEATURE_COLS}
    logger.info(f"[ML] Model loaded ({_MODEL_TYPE}) - {len(_FEATURE_COLS)} features from {model_dir}")
    return True


def is_model_available() -> bool:
    return _MODEL_AVAILABLE and _MODEL is not None


def get_feature_columns() -> List[str]:
    if _FEATURE_COLS:
        return list(_FEATURE_COLS)
    return list(FEATURE_COLS)


def get_model_metadata() -> Dict[str, Any]:
    return dict(_MODEL_META)


# ── Index helpers ─────────────────────────────────────────────────────────

def _fetch_index_return(symbol: str, period: str = "3mo") -> tuple[float, float]:
    """Return (daily_return_pct, cum20_pct) for symbol. Falls back to 0."""
    if not _HAS_YFINANCE:
        return 0.0, 0.0
    try:
        raw = yf.download(symbol, period=period, auto_adjust=True, progress=False, timeout=8)
        if raw is None or raw.empty:
            return 0.0, 0.0
        # handle MultiIndex vs Series
        if isinstance(raw.columns, pd.MultiIndex):
            # take Close level
            try:
                close = raw.xs("Close", axis=1, level=0).iloc[:, 0]
            except Exception:
                close = raw["Close"].iloc[:, 0] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0]
        else:
            close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        close = pd.Series(close).dropna()
        if len(close) < 2:
            return 0.0, 0.0
        daily = float(close.pct_change().iloc[-1] * 100) if len(close) >= 2 else 0.0
        cum20 = float(close.pct_change().rolling(20).sum().iloc[-1]) if len(close) >= 20 else float(close.pct_change().iloc[-1] * 100)
        if not np.isfinite(daily):
            daily = 0.0
        if not np.isfinite(cum20):
            cum20 = 0.0
        return daily, cum20
    except Exception as e:
        logger.debug(f"[ML] index fetch {symbol} failed: {e}")
        return 0.0, 0.0


def _fetch_vix(symbol: str = "^VIX") -> float:
    if not _HAS_YFINANCE:
        return 15.0
    try:
        raw = yf.download(symbol, period="5d", auto_adjust=True, progress=False, timeout=8)
        if raw is None or raw.empty:
            return 15.0
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                close = raw.xs("Close", axis=1, level=0).iloc[:, 0]
            except Exception:
                close = raw.iloc[:, 0]
        else:
            close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        close = pd.Series(close).dropna()
        if close.empty:
            return 15.0
        v = float(close.iloc[-1])
        return v if np.isfinite(v) and v > 0 else 15.0
    except Exception:
        return 15.0


def _get_index_cache(market: str) -> Dict[str, float]:
    """
    Returns dict: {NIFTY_return, BANKNIFTY_return, India_VIX, NIFTY_cum20}
    Caches per market (KRX/US) for 30min.
    슬롯명은 StockFlow 호환 유지.
    """
    mk = "KRX" if market.upper() == "KRX" else "US"
    now = time.monotonic()
    if mk in _INDEX_CACHE and (now - _INDEX_CACHE_TS.get(mk, 0)) < _INDEX_TTL:
        return _INDEX_CACHE[mk]
    symbols = INDEX_SYMBOLS[mk]
    result: Dict[str, float] = {"NIFTY_return": 0.0, "BANKNIFTY_return": 0.0, "India_VIX": 15.0, "NIFTY_cum20": 0.0}
    if mk == "KRX":
        # try ^KS200 first, fallback ^KS11
        daily, cum20 = _fetch_index_return(symbols["market"])
        if daily == 0 and cum20 == 0:
            daily, cum20 = _fetch_index_return(symbols.get("fallback_market") or "^KS11")
        result["NIFTY_return"] = daily
        result["NIFTY_cum20"] = cum20
        d2, _ = _fetch_index_return(symbols["sector"])
        result["BANKNIFTY_return"] = d2
        # KRX has no native VIX - try CBOE VIX, else 15
        result["India_VIX"] = _fetch_vix(symbols["vix"])
    else:
        daily, cum20 = _fetch_index_return(symbols["market"])
        result["NIFTY_return"] = daily
        result["NIFTY_cum20"] = cum20
        d2, _ = _fetch_index_return(symbols["sector"])
        result["BANKNIFTY_return"] = d2
        result["India_VIX"] = _fetch_vix(symbols["vix"])
    _INDEX_CACHE[mk] = result
    _INDEX_CACHE_TS[mk] = now
    logger.info(f"[ML] {mk} index cache refreshed: NIFTY {result['NIFTY_return']:.2f}% VIX {result['India_VIX']:.1f}")
    return result


def invalidate_index_cache() -> None:
    _INDEX_CACHE.clear()
    _INDEX_CACHE_TS.clear()


# ── Inference helpers ─────────────────────────────────────────────────────

def _fetch_ohlcv(ticker: str, market: str = "KRX") -> Optional[pd.DataFrame]:
    """yfinance에서 6개월 일봉을 조회. 실패 시 None."""
    if not _HAS_YFINANCE:
        return None
    sym = ticker.strip().upper()
    # KRX: 6자리 숫자 -> .KS suffix (fallback .KQ)
    if market == "KRX" and sym.isdigit() and len(sym) == 6:
        sym = f"{sym}.KS"
    try:
        hist = yf.Ticker(sym).history(period="6mo", auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 60:
            # KRX: try KQ fallback
            if market == "KRX" and sym.endswith(".KS"):
                alt = sym[:-3] + ".KQ"
                hist = yf.Ticker(alt).history(period="6mo", auto_adjust=True)
                if hist is not None and len(hist) >= 60:
                    sym = alt
                else:
                    return None
            else:
                return None
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        # normalize columns
        needed = {}
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            for variant in [c, c.lower(), c.upper()]:
                if variant in hist.columns:
                    needed[c] = variant
                    break
        if len(needed) < 5:
            return None
        out = pd.DataFrame({
            "Open": hist[needed["Open"]].astype(float),
            "High": hist[needed["High"]].astype(float),
            "Low": hist[needed["Low"]].astype(float),
            "Close": hist[needed["Close"]].astype(float),
            "Volume": hist[needed["Volume"]].astype(float),
        })
        out.index = hist.index
        out = out.reset_index().rename(columns={"index": "date", "Date": "date"})
        if "date" not in out.columns:
            out = out.reset_index()
            if "index" in out.columns:
                out = out.rename(columns={"index": "date"})
        return out
    except Exception as e:
        logger.debug(f"[ML] OHLCV fetch {ticker} failed: {e}")
        return None


def _sigmoid_calibrate(proba: float, calib: Dict[str, float]) -> float:
    """Apply Platt scaling stored in metadata (a, b). prob -> calibrated."""
    try:
        a = float(calib["a"])
        b = float(calib["b"])
        # logit = log(p/(1-p))
        eps = 1e-8
        p = min(max(proba, eps), 1 - eps)
        logit = math.log(p / (1 - p))
        calibrated = 1 / (1 + math.exp(a * logit + b))
        return float(min(max(calibrated, 0.01), 0.99))
    except Exception:
        return proba


def _predict_proba(feature_vec: List[float]) -> float:
    """Run model.predict_proba and return prob of class 1 (UP). Handles lightgbm booster vs sklearn."""
    if _MODEL is None:
        return 0.5
    X = np.array(feature_vec, dtype=float).reshape(1, -1)
    # handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        if _MODEL_TYPE == "lightgbm":
            # lightgbm Booster has different API
            try:
                # LGBMClassifier has predict_proba
                proba = _MODEL.predict_proba(X)
                if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                    return float(proba[0, 1])
                # fallback: Booster predict returns margin
                raw = _MODEL.predict(X)
                # if raw in [0,1] already probs
                if isinstance(raw, np.ndarray):
                    v = float(raw[0])
                    return v if 0 <= v <= 1 else 1 / (1 + math.exp(-v))
                return float(raw)
            except AttributeError:
                # pure Booster
                import lightgbm as lgb  # type: ignore
                raw = _MODEL.predict(X)
                v = float(raw[0]) if isinstance(raw, np.ndarray) else float(raw)
                # if booster trained as binary, raw is prob already
                return v if 0 <= v <= 1 else 1 / (1 + math.exp(-v))
        else:
            # sklearn or generic joblib
            if hasattr(_MODEL, "predict_proba"):
                proba = _MODEL.predict_proba(X)
                return float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
            elif hasattr(_MODEL, "decision_function"):
                score = float(_MODEL.decision_function(X)[0])
                return 1 / (1 + math.exp(-score))
            else:
                pred = _MODEL.predict(X)
                return 0.75 if int(pred[0]) == 1 else 0.25
    except Exception as e:
        logger.warning(f"[ML] predict_proba failed: {e}")
        return 0.5


def _heuristic_proba_fallback(feature_dict: Dict[str, float]) -> float:
    """
    모델 없을 때 사용하는 휴리스틱 baseline.
    StockOracle 기존 analyze_score와 유사하게 RSI/MACD/ADX/MA 정렬을 경량 집계.
    Returns prob_up in [0.3, 0.7].
    """
    try:
        rsi14 = feature_dict.get("RSI_14", 50)
        macd_hist = feature_dict.get("MACD_hist", 0)
        adx = feature_dict.get("ADX", 20)
        sma_cross = feature_dict.get("SMA_cross", 0)
        stoch_k = feature_dict.get("stochastic_k", 50)
        mom20 = feature_dict.get("price_momentum_20d", 0)

        score = 0.0
        # RSI 30-70 neutral, <30 oversold bullish, >70 overbought bearish
        if rsi14 < 30:
            score += 0.12
        elif rsi14 < 40:
            score += 0.06
        elif rsi14 > 70:
            score -= 0.12
        elif rsi14 > 60:
            score -= 0.06
        # MACD hist
        if macd_hist > 0:
            score += 0.08
        else:
            score -= 0.08
        # ADX + trend
        if adx > 25 and sma_cross == 1:
            score += 0.10
        elif adx < 15:
            score -= 0.04
        # Stochastic
        if stoch_k < 20:
            score += 0.07
        elif stoch_k > 80:
            score -= 0.07
        # Momentum
        if mom20 > 3:
            score += 0.08
        elif mom20 < -3:
            score -= 0.08
        elif mom20 > 0:
            score += 0.03
        else:
            score -= 0.03

        prob = 0.5 + score
        return float(min(max(prob, 0.32), 0.68))
    except Exception:
        return 0.5


# ── Public API ────────────────────────────────────────────────────────────

def predict_from_ohlcv(
    ticker: str,
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
    market: str = "KRX",
    opens: Optional[List[float]] = None,
    index_cache: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    OHLCV 리스트에서 직접 예측 (yfinance 재조회 없음).
    StockFlow 방식과 달리 KRX/US 모두 지원하며 동일 feature 스키마 사용.
    """
    fallback: Dict[str, Any] = {
        "direction": "NEUTRAL",
        "confidence": 0.5,
        "prob_up": 0.5,
        "prob_down": 0.5,
        "horizon": "14d",
        "market": market,
        "model_available": is_model_available(),
        "model_type": _MODEL_TYPE,
        "ticker": ticker.upper(),
        "fallback": True,
        "reason": "insufficient_data" if len(closes) < 60 else "model_not_loaded",
    }
    if len(closes) < 60 or len(highs) < 60 or len(lows) < 60 or len(volumes) < 60:
        # short history -> neutral with heuristic adjustment if possible
        if len(closes) >= 20:
            # use simple heuristic on available closes
            try:
                # quick momentum proxy
                mom = (closes[-1] / closes[0] - 1) * 100 if closes[0] else 0
                if mom > 2:
                    fallback["direction"] = "UP"
                    fallback["confidence"] = 0.54
                    fallback["prob_up"] = 0.54
                elif mom < -2:
                    fallback["direction"] = "DOWN"
                    fallback["confidence"] = 0.54
                    fallback["prob_up"] = 0.46
            except Exception:
                pass
        return fallback

    # lazy load model on first call
    if not is_model_available():
        load_model()

    idx_cache = index_cache if index_cache is not None else _get_index_cache(market)
    feat = compute_feature_vector(closes, highs, lows, volumes, market=market, index_cache=idx_cache)
    if feat is None:
        return fallback

    cols = get_feature_columns()
    feature_vec = []
    for col in cols:
        v = feat.get(col, 0.0)
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            v = 0.0
        feature_vec.append(float(v))

    if is_model_available() and _MODEL is not None:
        prob_up_raw = _predict_proba(feature_vec)
        # calibration
        if _CALIB_PARAMS is not None:
            prob_up = _sigmoid_calibrate(prob_up_raw, _CALIB_PARAMS)
        else:
            prob_up = prob_up_raw
    else:
        prob_up = _heuristic_proba_fallback(feat)
        prob_up_raw = prob_up

    prob_up = float(min(max(prob_up, 0.01), 0.99))
    prob_down = 1.0 - prob_up
    direction = "UP" if prob_up >= 0.5 else "DOWN"
    # confidence = max(prob_up, prob_down) but shrink toward 0.5 if near threshold
    confidence = float(max(prob_up, prob_down))
    # apply small shrinks for low-volume / high-volatility uncertainty
    try:
        atr_pct = abs(feat.get("ATR_14", 2.0))
        if atr_pct > 5.0:
            # high volatility -> shrink confidence toward 0.5 by 10%
            confidence = 0.5 + (confidence - 0.5) * 0.9
    except Exception:
        pass

    return {
        "direction": direction,
        "confidence": round(float(confidence), 3),
        "prob_up": round(float(prob_up), 3),
        "prob_up_raw": round(float(prob_up_raw), 3) if 'prob_up_raw' in locals() else round(float(prob_up), 3),
        "prob_down": round(float(prob_down), 3),
        "horizon": "14d",
        "market": market,
        "model_available": is_model_available(),
        "model_type": _MODEL_TYPE,
        "ticker": ticker.upper(),
        "fallback": not is_model_available(),
        "feature_count": len(feature_vec),
        "atr_pct": round(float(feat.get("ATR_14", 0.0)), 2),
        "rsi_14": round(float(feat.get("RSI_14", 50.0)), 1),
    }


def predict_direction(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    StockFlow-compatible entry: metrics dict에서 ticker/market/price_history 등을 추출해 예측.

    Supports:
      metrics = {
        "ticker": "005930.KS" or "AAPL",
        "market": "KRX"|"US" (optional, auto-inferred),
        "closes": [...], "highs": [...], "lows": [...], "volumes": [...],
        # or legacy StockFlow keys:
        "price_history": [...], "high_history": [...], "low_history": [...], "volume_history": [...]
        # or dd dict: {"Close": [...], "High": [...], ...}
      }
    """
    ticker = str(metrics.get("ticker") or metrics.get("symbol") or "").strip().upper()
    if not ticker:
        return {"direction": "NEUTRAL", "confidence": 0.5, "prob_up": 0.5, "horizon": "14d", "fallback": True, "reason": "ticker_missing"}

    # infer market
    market = str(metrics.get("market") or "").upper()
    if market not in ("KRX", "US"):
        if ticker.endswith((".KS", ".KQ")) or (ticker.isdigit() and len(ticker) == 6):
            market = "KRX"
        elif "." not in ticker and ticker.isupper() and len(ticker) <= 5:
            # could be US; also check KR_STOCK_MAP? assume US
            market = "US"
        else:
            market = "KRX" if ticker.endswith(".KS") or ticker.endswith(".KQ") else "US"

    # extract OHLCV from various possible keys
    closes = metrics.get("closes") or metrics.get("price_history") or metrics.get("Close") or []
    highs = metrics.get("highs") or metrics.get("high_history") or metrics.get("High") or []
    lows = metrics.get("lows") or metrics.get("low_history") or metrics.get("Low") or []
    volumes = metrics.get("volumes") or metrics.get("volume_history") or metrics.get("Volume") or []

    # if dd dict with list values directly
    if not closes and isinstance(metrics.get("dd"), dict):
        dd = metrics["dd"]
        closes = dd.get("Close") or dd.get("close") or []
        highs = dd.get("High") or dd.get("high") or []
        lows = dd.get("Low") or dd.get("low") or []
        volumes = dd.get("Volume") or dd.get("volume") or []

    # if DataFrame-like dict with lists
    if isinstance(closes, np.ndarray):
        closes = closes.tolist()
    if isinstance(highs, np.ndarray):
        highs = highs.tolist()
    if isinstance(lows, np.ndarray):
        lows = lows.tolist()
    if isinstance(volumes, np.ndarray):
        volumes = volumes.tolist()

    # if we have enough data, use direct path
    if len(closes) >= 60 and len(highs) >= 60 and len(lows) >= 60 and len(volumes) >= 60:
        return predict_from_ohlcv(ticker, closes, highs, lows, volumes, market=market)

    # try yfinance fetch
    ohlcv = _fetch_ohlcv(ticker, market=market)
    if ohlcv is not None and len(ohlcv) >= 60:
        closes = ohlcv["Close"].tolist() if "Close" in ohlcv.columns else ohlcv["close"].tolist()
        highs = ohlcv["High"].tolist() if "High" in ohlcv.columns else ohlcv["high"].tolist()
        lows = ohlcv["Low"].tolist() if "Low" in ohlcv.columns else ohlcv["low"].tolist()
        volumes = ohlcv["Volume"].tolist() if "Volume" in ohlcv.columns else ohlcv["volume"].tolist()
        return predict_from_ohlcv(ticker, closes, highs, lows, volumes, market=market)

    # last fallback: use whatever closes we have (may be <60)
    if closes and len(closes) >= 20:
        # pad highs/lows/volumes if missing
        if not highs or len(highs) != len(closes):
            highs = list(closes)
        if not lows or len(lows) != len(closes):
            lows = list(closes)
        if not volumes or len(volumes) != len(closes):
            volumes = [100000.0] * len(closes)
        return predict_from_ohlcv(ticker, closes, highs, lows, volumes, market=market)

    return {
        "direction": "NEUTRAL",
        "confidence": 0.5,
        "prob_up": 0.5,
        "prob_down": 0.5,
        "horizon": "14d",
        "market": market,
        "model_available": is_model_available(),
        "ticker": ticker,
        "fallback": True,
        "reason": "insufficient_data_and_fetch_failed",
    }


# ── Auto-load on import (best-effort) ────────────────────────────────────
try:
    load_model()
except Exception:
    pass
