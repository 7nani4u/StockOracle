"""audit_prediction_layers.py — /api/stock 예측 체인의 층별 워크포워드 감사.

무엇을 보나
  각 종목·각 과거 시점 t 에서 서비스와 같은 입력(최근 252봉 + 인과적 지표)으로 route('/api/stock')의
  점수·확률·목표가·예측구간 계산을 다시 수행하고, t 이후 22거래일에 실제로 일어난 결과와 비교한다.
    · 방향 확률(prob_l0 → 상관 보정 후 prob_final → 시나리오 up/(up+down)): AUC·Brier·보정 기울기
    · 목표가 하단 터치율 대비 변동성 터치 확률의 보정
    · P10~P90 / P05~P95 예측구간의 실제 포함률
    · 목표가 범위의 실현 최고가 포함률(상관 보정 전/후)
    · 의사결정 라벨(조건부/관망/주의)별 사후 상승률과 하락폭

범위와 한계 (중요)
  · route 의 글루 로직을 이 파일이 다시 조립한다. route 의 보정 순서를 바꾸면 이 파일도 함께 고쳐야 한다.
  · 오프라인에서 재현할 수 없는 단계는 건너뛴다: 투자자 수급, ML, 뉴스·거시·섹터·실적, 실시간 시세.
    따라서 결과는 '기술적 입력만으로 만든 체인'의 성능이며, 건너뛴 입력이 추가로 주는 정보는 검증되지 않는다.
  · 표본의 22거래일 구간은 겹치므로 신뢰구간은 (종목, 월) 블록 부트스트랩으로 구한다.
  · 학습 로그(prediction_learning.jsonl)는 건드리지 않는다(STOCKORACLE_PREDICTION_LOG 를 임시 경로로 고정).

사용 예
    python scripts/audit_prediction_layers.py                      # 캐시 사용, 없으면 5년 일봉 수집
    python scripts/audit_prediction_layers.py --offline --stride 6  # 캐시만
    python scripts/audit_prediction_layers.py --tickers AAPL,005930.KS --stride 3
    python scripts/audit_prediction_layers.py --out /tmp/audit.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# 감사가 운영 학습 이력을 오염시키지 않도록 import 전에 임시 경로로 고정한다.
os.environ.setdefault("STOCKORACLE_PREDICTION_LOG", os.path.join(tempfile.gettempdir(), "stockoracle_audit_learning.jsonl"))

import numpy as np
import pandas as pd

HORIZON = 22            # 선행 평가 거래일 (예측 탭 '1개월' 과 같은 값)
WINDOW = 252            # route 가 쓰는 일봉 길이
CACHE_DIR = os.path.join(REPO_ROOT, "datasets", "audit_cache")
CACHE_TTL_S = 24 * 3600
DEFAULT_OUT = os.path.join(REPO_ROOT, "docs", "backtests", "prediction_layer_audit_summary.json")

# 대형·중형·소형 혼합 기본 유니버스 (technique_prune 과 같은 종목군)
DEFAULT_TICKERS = {
    "KRX": ["005930.KS", "000660.KS", "005380.KS", "000270.KS", "035420.KS", "035720.KS", "051910.KS", "068270.KS",
            "105560.KS", "012330.KS", "009150.KS", "011200.KS", "010140.KS", "086790.KS", "033780.KS", "015760.KS",
            "259960.KS", "003670.KS", "058470.KQ", "214150.KQ", "095340.KQ", "140860.KQ", "357780.KQ", "222800.KQ",
            "240810.KQ", "035900.KQ"],
    "US": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "AVGO", "JPM", "V", "UNH", "RBLX", "DKNG", "HOOD", "NET",
           "DDOG", "ROKU", "MU", "DELL", "SOFI", "RIVN", "IONQ", "RKLB", "PINS", "U", "LMND", "AFRM"],
}
REGIME_INDEX = {"KS": "^KS11", "KQ": "^KQ11", "US": "^GSPC"}

_NO_LEARNING = {"depth_extra": 0.0, "hold_score_delta": 0, "allocation_scale": 1.0, "sample_n": 0,
                "applied": False, "reason": "audit"}
_NO_FLOW = {"ok": False, "reason": "offline audit"}
_NO_EVENT = {"score": 0, "level": "low", "reasons": ["offline audit"], "days_to_earnings": None}


# ── 지표(순수 함수, 단위 테스트 대상) ──────────────────────────────────────

def auc(y, p) -> float:
    """순위 기반 AUC(동점은 평균 순위). 한쪽 클래스만 있으면 NaN."""
    y = np.asarray(y)
    ranks = pd.Series(np.asarray(p, float)).rank(method="average").to_numpy()
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def calibration_slope(y, p) -> float:
    """결과를 예측확률에 OLS 회귀한 기울기: 1 = 보정됨, 0 = 정보 없음."""
    x = np.asarray(p, float)
    x = x - x.mean()
    yy = np.asarray(y, float)
    denom = float((x * x).sum())
    return float((x * (yy - yy.mean())).sum() / denom) if denom > 1e-12 else float("nan")


def block_bootstrap_ci(frame: pd.DataFrame, stat, n: int = 200, seed: int = 11) -> Tuple[float, float]:
    """(종목, 월) 블록 재표집으로 stat(frame) 의 95% 구간."""
    rng = np.random.default_rng(seed)
    blocks = frame.assign(_m=frame["date"].astype(str).str[:7]).groupby(["ticker", "_m"]).indices
    keys = list(blocks)
    values = []
    for _ in range(n):
        picks = rng.choice(len(keys), len(keys), replace=True)
        sample = frame.iloc[np.concatenate([blocks[keys[i]] for i in picks])]
        values.append(stat(sample))
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


# ── 데이터 계층 ───────────────────────────────────────────────────────────

def _cache_path(ticker: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, ticker.replace("^", "_idx_").replace(".", "_") + ".csv")


def load_prices(ticker: str, offline: bool = False) -> Optional[pd.DataFrame]:
    """5년 일봉(auto_adjust). 신선한 캐시 우선, --offline 이면 캐시만. 종가 NaN 행은 제거한다."""
    path = _cache_path(ticker)
    fresh = os.path.exists(path) and time.time() - os.path.getmtime(path) < CACHE_TTL_S
    frame = None
    if os.path.exists(path) and (offline or fresh):
        try:
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
        except (OSError, ValueError):
            frame = None
    if frame is None and not offline:
        try:
            import yfinance as yf
            frame = yf.Ticker(ticker).history(period="5y", interval="1d", auto_adjust=True)
            if frame is not None and not frame.empty:
                frame[["Open", "High", "Low", "Close", "Volume"]].to_csv(path)
        except Exception as exc:  # 네트워크 일시 장애는 해당 종목만 건너뛴다
            print(f"[audit] {ticker} 수집 실패 ({type(exc).__name__})", flush=True)
            frame = None
    if frame is None or frame.empty or "Close" not in frame.columns:
        return None
    frame = frame[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
    # 거래소 현지 날짜만 쓴다(UTC 로 바꾸면 KRX 일봉이 하루 앞당겨진다).
    frame.index = pd.to_datetime(pd.Index([str(value)[:10] for value in frame.index]))
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def regime_series(prices: pd.DataFrame, classify) -> pd.Series:
    """각 날짜의 지수 레짐(서비스와 같은 classify_index_regime 규칙, 마지막 행까지 인과적)."""
    close = prices["Close"]
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    return pd.Series([classify(c, a, b) if not math.isnan(b) else "NEUTRAL" for c, a, b in zip(close, ma60, ma120)],
                     index=close.index)


# ── route 글루 재현 ───────────────────────────────────────────────────────

def _as_dd(frame: pd.DataFrame) -> Dict[str, list]:
    dd: Dict[str, list] = {"Date": frame.index.strftime("%Y-%m-%d").tolist()}
    for column in frame.columns:
        dd[column] = [float(v) if isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v)) else None
                      for v in frame[column].tolist()]
    return dd


def evaluate_ticker(ix, ticker: str, prices: pd.DataFrame, regimes: pd.Series, stride: int,
                    max_rows: int = 0) -> List[Dict[str, Any]]:
    """한 종목의 모든 평가 시점에 대해 예측 체인을 실행하고 선행 결과와 함께 한 행씩 반환한다."""
    from market_briefing.confidence_engine import build_signal_confidence
    from market_briefing.correlation_engine import correlate_and_narrow
    from market_briefing.stock_analyzer import enrich_with_hybrid

    market = "KRX" if ticker.endswith((".KS", ".KQ")) else "US"
    frame = ix.add_indicators(prices.copy(), market=market)
    full = _as_dd(frame)
    closes = frame["Close"].to_numpy(float)
    highs = frame["High"].to_numpy(float)
    lows = frame["Low"].to_numpy(float)
    regime_at = regimes.reindex(frame.index, method="ffill").fillna("NEUTRAL").tolist()
    rows: List[Dict[str, Any]] = []
    for t in range(WINDOW, len(frame) - HORIZON, stride):
        dd = {key: values[t - WINDOW + 1:t + 1] for key, values in full.items()}
        last, prev = float(closes[t]), float(closes[t - 1])
        pct = (last - prev) / prev * 100.0
        atr = dd["ATR"][-1]
        if atr is None or not math.isfinite(atr) or atr <= 0:
            atr = last * 0.02
        regime = regime_at[t]
        try:
            score, _steps, patterns, geo, ai = ix.analyze_score(dd, market, "1y")
            raw_score = score
            for pattern in geo:
                signal = str(pattern.get("signal") or "")
                direction = pattern.get("direction") or ("상승" if signal.startswith("매수") else "하락" if signal.startswith("매도") else "중립")
                patterns.append({**pattern, "direction": direction,
                                 "conf": int(round(float(pattern.get("completion_score") or pattern.get("conf") or 0)))})
            patterns = ix._deduplicate_pattern_types(patterns)
            hybrid = enrich_with_hybrid(
                closes=[float(c) for c in dd["Close"] if c is not None],
                highs=[float(c) for c in dd["High"] if c is not None],
                lows=[float(c) for c in dd["Low"] if c is not None],
                volumes=[float(c) for c in dd["Volume"] if c is not None],
                open_prices=[float(c) for c in dd["Open"] if c is not None],
            )
            score = ix.finalize_rule_score(score, regime=regime, debt_healthy=None, flow_adjust=0, hybrid=hybrid)["score"]
            prob_up, prob_down = ix.calc_probability(score, dd, market)
            prob_l0 = prob_up
            indicators = ix.calc_indicator_signals(dd, market=market)
            pivots = ix.calc_pivot_points(dd)
            weekly = ix.build_weekly_analysis_context(dd, market)
            buy = ix.calc_buy_price(dd, last, atr, score, indicators, market, "1y", _NO_EVENT, _NO_LEARNING, regime,
                                    prev, pct, arty_dd=dd, weekly_context=weekly)
            ncs = float(hybrid["ncs"]) if "ncs" in hybrid and "error" not in hybrid else None
            valid = [c for c in dd["Close"] if c is not None]
            pct5 = ((valid[-1] - valid[-6]) / valid[-6] * 100.0) if len(valid) >= 6 and valid[-6] else None
            confidence = build_signal_confidence(
                technical_score=float(score), ai_score=ncs,
                market_score={"BULL": 70.0, "NEUTRAL": 50.0, "BEAR": 30.0}.get(regime, 50.0), symbol=ticker,
                market=market, stock_pct5d=pct5, news_items=None, history_confidence=None,
                include_macro=False, include_sector=False, include_earnings=False)
            target = ix.calc_target_price(dd, last, atr, "1mo", market, weekly_context=weekly)
            target_base = dict(target) if target else {}
            target = ix._apply_signal_confidence_to_target(target, confidence, last, atr, market)
            target = ix._apply_learning_adjustment_to_target(target, _NO_LEARNING)
            if target:
                target = ix._normalize_target_output(target, last, market)
            pullback = ix.calc_pullback_analysis(dd, last, atr, score, market, target)
            volumes = [v for v in dd["Volume"][-21:] if v is not None]
            volume_ratio = (volumes[-1] / (sum(volumes[:-1]) / max(1, len(volumes) - 1))) if len(volumes) >= 21 and sum(volumes[:-1]) > 0 else 1.0
            macd_gap = float((indicators or {}).get("macd", 0) or 0) - float((indicators or {}).get("signal_line", 0) or 0)
            corr = correlate_and_narrow(
                symbol=ticker, market=market, dd=dd, last_price=last, atr=atr, score=score, prob_up_base=prob_up,
                prob_down_base=prob_down, target_price=target, signal_confidence=confidence, indicator_signals=indicators,
                candlestick_patterns=patterns, pullback_analysis=pullback, investor_flow=_NO_FLOW, ml_prediction=None,
                regime=regime, pct_change=pct, volume_ratio=volume_ratio, candle_up=bool(last >= prev),
                rsi=float((indicators or {}).get("rsi", 50) or 50), macd_gap=macd_gap, event_risk=_NO_EVENT)
            if isinstance(corr, dict):
                prob_up = float(corr.get("prob_up_corr", prob_up))
                prob_down = float(corr.get("prob_down_corr", prob_down))
                narrowed = corr.get("target_narrowed")
                if narrowed and target is not None:
                    target = dict(target, min_price=narrowed.get("min_price", target.get("min_price")),
                                  max_price=narrowed.get("max_price", target.get("max_price")))
                    target = ix._normalize_target_output(target, last, market)
                    try:
                        pullback = ix.calc_pullback_analysis(dd, last, atr, score, market, target)
                    except Exception:
                        pass
                if corr.get("confidence_corr") is not None and confidence is not None:
                    confidence = dict(confidence, confidence=corr["confidence_corr"],
                                      confidence_interval=corr.get("confidence_interval_corr"))
            outlook = ix.build_prediction_outlook(
                symbol=ticker, market=market, dd=dd, last_price=last, prev_close=prev, pct_change=pct, atr=atr,
                regime=regime, score=score, prob_up=prob_up, prob_down=prob_down, pivot_points=pivots,
                indicator_signals=indicators, buy_price=buy, target_price=target, pullback_analysis=pullback,
                signal_confidence=confidence, investor_flow=_NO_FLOW, ai_strategy=ai, candlestick_patterns=patterns,
                naver=None, us_enriched=None, toss_industry=None, event_risk=_NO_EVENT, period="1mo",
                atr_is_observed=True, dynamic_rsi=None, quote_date=None, data_warnings=[])
        except Exception as exc:  # 한 시점의 실패가 감사 전체를 멈추지 않게 한다
            print(f"[audit] {ticker} {dd['Date'][-1]} 건너뜀: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            continue
        scenarios = {s["key"]: s for s in outlook.get("scenarios", [])}
        forecast = outlook.get("forecast") or {}
        decision = outlook.get("decision") or {}
        p10_p90 = forecast.get("range_p10_p90") or [None, None]
        p05_p95 = forecast.get("range_p05_p95") or [None, None]
        upside = scenarios.get("upside") or {}
        future_high = highs[t + 1:t + 1 + HORIZON]
        future_low = lows[t + 1:t + 1 + HORIZON]
        rows.append({
            "ticker": ticker, "market": market, "date": dd["Date"][-1], "regime": regime, "close": last,
            "raw_score": raw_score, "score": score, "ncs": ncs, "fws": hybrid.get("fws"),
            "hyb_regime": hybrid.get("regime"), "hyb_action": hybrid.get("action"),
            "prob_l0": prob_l0, "prob_final": prob_up,
            "up_prob": upside.get("probability"), "down_prob": (scenarios.get("downside") or {}).get("probability"),
            "decision": decision.get("key"), "agreement": ((corr or {}).get("correlation") or {}).get("agreement"),
            "narrow_factor": ((corr or {}).get("target_narrowed") or {}).get("factor"),
            "tgt_min_base": target_base.get("min_price"), "tgt_max_base": target_base.get("max_price"),
            "tgt_min_final": (target or {}).get("min_price"), "tgt_max_final": (target or {}).get("max_price"),
            "fc_exp_ret": forecast.get("expected_return_pct"), "fc_lo": p10_p90[0], "fc_hi": p10_p90[1],
            "fc_lo95": p05_p95[0], "fc_hi95": p05_p95[1],
            "up_touch": upside.get("touch_probability"),
            "up_lo": (upside.get("price_range") or [None, None])[0],
            "fwd_ret": float(closes[t + HORIZON] / last - 1.0), "fwd_max": float(future_high.max() / last - 1.0),
            "fwd_min": float(future_low.min() / last - 1.0),
        })
        if max_rows and len(rows) >= max_rows:
            break
    return rows


# ── 요약 ──────────────────────────────────────────────────────────────────

def _prob_block(frame: pd.DataFrame, column: str) -> Dict[str, float]:
    y = frame["up"].to_numpy()
    p = frame[column].to_numpy(float) / 100.0
    base = float(y.mean())
    lo, hi = block_bootstrap_ci(frame, lambda s: auc(s["up"], s[column]), n=120)
    bs, bs0 = brier(y, p), brier(y, np.full_like(p, base))
    return {"mean_pred": round(float(p.mean()), 4), "base_rate": round(base, 4), "auc": round(auc(y, p), 4),
            "auc_ci95": [round(lo, 4), round(hi, 4)], "brier": round(bs, 4), "brier_skill": round(1 - bs / bs0, 4),
            "calibration_slope": round(calibration_slope(y, p), 4)}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"rows": 0}
    frame["up"] = (frame["fwd_ret"] > 0).astype(int)
    frame["up_share"] = frame["up_prob"] / (frame["up_prob"] + frame["down_prob"]).replace(0, np.nan) * 100.0
    frame = frame.dropna(subset=["prob_l0", "prob_final", "up_share"])
    out: Dict[str, Any] = {
        "rows": int(len(frame)), "tickers": int(frame["ticker"].nunique()),
        "period": [str(frame["date"].min()), str(frame["date"].max())], "horizon_sessions": HORIZON,
        "base_up_rate": round(float(frame["up"].mean()), 4),
    }
    for name, subset in (("all", frame), ("KRX", frame[frame.market == "KRX"]), ("US", frame[frame.market == "US"])):
        if len(subset) > 200:
            out[f"direction_{name}"] = {col: _prob_block(subset, col) for col in ("prob_l0", "prob_final", "up_share")}
    out["score_auc"] = {"raw": round(auc(frame["up"], frame["raw_score"]), 4), "final": round(auc(frame["up"], frame["score"]), 4)}
    out["hybrid"] = {
        "regime_counts": frame["hyb_regime"].value_counts().to_dict(),
        "action_share": frame["hyb_action"].value_counts(normalize=True).round(4).to_dict(),
        "fws_min": round(float(frame["fws"].min()), 2),
        "ncs_auc": round(auc(frame["up"], frame["ncs"].fillna(frame["ncs"].mean())), 4),
    }
    bear = frame[frame.regime == "BEAR"]
    if len(bear):
        out["bear_cap_violation_share"] = round(float((bear["score"] > 40).mean()), 4)
    close_next = frame["close"] * (1 + frame["fwd_ret"])
    coverage = {}
    for label, lo, hi, nominal in (("p10_p90", "fc_lo", "fc_hi", 0.80), ("p05_p95", "fc_lo95", "fc_hi95", 0.90)):
        sub = frame.dropna(subset=[lo, hi])
        inside = ((close_next.loc[sub.index] >= sub[lo]) & (close_next.loc[sub.index] <= sub[hi])).mean()
        coverage[label] = {"coverage": round(float(inside), 4), "nominal": nominal}
    out["forecast_band_coverage"] = coverage
    touch = frame.dropna(subset=["up_lo", "up_touch"]).copy()
    if len(touch):
        touch["touched"] = (touch["close"] * (1 + touch["fwd_max"]) >= touch["up_lo"]).astype(int)
        pred = touch["up_touch"].to_numpy(float) / 100.0
        out["upside_touch_calibration"] = {
            "mean_pred": round(float(pred.mean()), 4), "realized": round(float(touch["touched"].mean()), 4),
            "auc": round(auc(touch["touched"], pred), 4), "brier": round(brier(touch["touched"], pred), 4),
            "brier_constant": round(brier(touch["touched"], np.full(len(touch), touch["touched"].mean())), 4)}
    tr = frame.dropna(subset=["tgt_min_base", "tgt_max_base", "tgt_min_final", "tgt_max_final"])
    if len(tr):
        top = tr["close"] * (1 + tr["fwd_max"])
        out["target_range_inside_share"] = {
            "base": round(float(((top >= tr.tgt_min_base) & (top <= tr.tgt_max_base)).mean()), 4),
            "final": round(float(((top >= tr.tgt_min_final) & (top <= tr.tgt_max_final)).mean()), 4)}
    out["agreement"] = {"median": round(float(frame["agreement"].median()), 4),
                        "share_at_narrow_floor": round(float((frame["narrow_factor"].round(2) == 0.55).mean()), 4)}
    out["decision_outcomes"] = {
        key: {"n": int(len(sub)), "up_rate": round(float(sub["up"].mean()), 4), "mean_fwd_ret_pct": round(float(sub["fwd_ret"].mean() * 100), 3),
              "p_drawdown_8pct": round(float((sub["fwd_min"] < -0.08).mean()), 4)}
        for key, sub in frame.groupby("decision") if len(sub) >= 50}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride", type=int, default=6, help="평가 시점 간격(거래일), 기본 6")
    parser.add_argument("--tickers", default="", help="쉼표 구분 종목, 기본은 대형·중형·소형 52종목")
    parser.add_argument("--limit", type=int, default=0, help="종목 수 제한(빠른 확인용)")
    parser.add_argument("--offline", action="store_true", help="캐시만 사용")
    parser.add_argument("--out", default=DEFAULT_OUT, help="요약 JSON 경로")
    parser.add_argument("--rows-csv", default="", help="행 단위 결과를 저장할 CSV 경로(선택)")
    args = parser.parse_args(argv)

    from api import index as ix

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] or DEFAULT_TICKERS["KRX"] + DEFAULT_TICKERS["US"]
    if args.limit:
        tickers = tickers[:args.limit]
    regimes = {}
    for key, symbol in REGIME_INDEX.items():
        prices = load_prices(symbol, args.offline)
        if prices is None:
            print(f"[audit] 지수 {symbol} 없음 — 해당 시장 레짐은 NEUTRAL 로 처리", flush=True)
            continue
        regimes[key] = regime_series(prices, ix.classify_index_regime)

    rows: List[Dict[str, Any]] = []
    for number, ticker in enumerate(tickers, 1):
        prices = load_prices(ticker, args.offline)
        if prices is None or len(prices) < WINDOW + HORIZON + 5:
            print(f"[audit] {ticker} 건너뜀 (일봉 부족/없음)", flush=True)
            continue
        market = "KRX" if ticker.endswith((".KS", ".KQ")) else "US"
        key = ("KQ" if ticker.endswith(".KQ") else "KS") if market == "KRX" else "US"
        series = regimes.get(key, pd.Series("NEUTRAL", index=prices.index))
        rows.extend(evaluate_ticker(ix, ticker, prices, series, args.stride))
        print(f"[audit] {number}/{len(tickers)} {ticker} 누적 {len(rows)}행", flush=True)

    summary = summarize(rows)
    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["stride"] = args.stride
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    if args.rows_csv:
        pd.DataFrame(rows).to_csv(args.rows_csv, index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:6000])
    print(f"\n요약 저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
