"""prune_loss_conditions.py — 기법별 손실조건 제거(Subtractive Pruning) 검증.

핵심(수익 기법을 새로 찾지 않고 손실 조건을 지워 남긴다):
  1) 같은 조건의 매매를 기법·시장별로 최소 20번 반복 재현한다.
  2) 진입 자리·손절 자리·결과를 전부 기록한다.
  3) 손실이 가장 자주 발생한 조건을 찾아낸다.
  4) 그 조건부터 하나씩 제외하고 손절율이 개선될 때만 유지한다.

대상 기법(전부 OHLCV 배열 기반, 기존 모듈 재사용):
  hybrid_breakout / pattern_breakout / dynamic_rsi / leader_reversal / ml_direction

데이터: 최근 1년 일봉(국내 KRX + 미국 US 각각). yfinance 결과를
``datasets/prune_cache/`` 에 CSV로 캐시해 재실행을 오프라인(--offline) 으로
돌릴 수 있다. 기존 백테스트 스크립트들이 네트워크 필수였던 문제를 피한다.

사용 예:
    python scripts/prune_loss_conditions.py --limit 3
    python scripts/prune_loss_conditions.py --full --offline
    python scripts/prune_loss_conditions.py --tickers 005930.KS,000660.KS --market KRX
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

# ── 검증 상수 ─────────────────────────────────────────────────────────────
MIN_TRADES = 20        # 기법·시장별 최소 반복 매매 수
MIN_SUPPORT = 5        # 손실조건 후보 최소 표본
MAX_ITERS = 10         # 조건 제거 최대 반복
MAX_HOLD_BARS = 20     # 보유 상한 (일봉)
TP_ATR_MULT = 2.0      # 통일 목표가 = 진입 + 2.0 * ATR14
COST_PCT = {"KRX": 0.20, "US": 0.10}  # 왕복 비용 (dynamic_rsi와 동일 기준)
EXPECT_TOL_PCT = 0.10  # 제거 수락 시 허용되는 기대값 하락 한도 (%p)
DROP_WINRATE = 40.0    # 이하 + 기대값<=0 이면 기법 전체 제외 후보

CACHE_DIR = os.path.join(REPO_ROOT, "datasets", "prune_cache")
RULES_OUT = os.path.join(REPO_ROOT, "models", "technique_prune.json")
TRADES_OUT = os.path.join(REPO_ROOT, "docs", "backtests", "technique_prune_trades.csv")

DEFAULT_TICKERS = {
    "KRX": {
        "LARGE": [
            "005930.KS", "000660.KS", "005380.KS", "000270.KS", "035420.KS",
            "035720.KS", "051910.KS", "068270.KS", "105560.KS", "012330.KS",
        ],
        "MID": [
            "009150.KS", "011200.KS", "010140.KS", "086790.KS", "033780.KS",
            "015760.KS", "259960.KS", "003670.KS",
        ],
        "SMALL": [
            "058470.KQ", "214150.KQ", "095340.KQ", "140860.KQ", "357780.KQ",
            "222800.KQ", "240810.KQ", "035900.KQ",
        ],
    },
    "US": {
        "LARGE": [
            "AAPL", "MSFT", "NVDA", "AMZN", "META",
            "TSLA", "AVGO", "JPM", "V", "UNH",
        ],
        "MID": [
            "RBLX", "DKNG", "HOOD", "NET", "DDOG", "ROKU", "MU", "DELL",
        ],
        "SMALL": [
            "SOFI", "RIVN", "IONQ", "RKLB", "PINS", "U", "LMND", "AFRM",
        ],
    },
}
TIERS = ("LARGE", "MID", "SMALL")
BENCH = {"KRX": "^KS11", "US": "SPY"}  # Yahoo ^KS200 은 1행만 반환해 KRX 벤치 유무 조건이 항상 비었다

TECHNIQUES = (
    "hybrid_breakout",
    "pattern_breakout",
    "dynamic_rsi",
    "leader_reversal",
    "ml_direction",
)

_NEUTRAL_INDEX = {
    "NIFTY_return": 0.0, "BANKNIFTY_return": 0.0, "India_VIX": 15.0,
    "NIFTY_cum20": 0.0, "market_return_1d": 0.0, "sector_return_1d": 0.0,
    "volatility_index": 15.0, "market_return_20d": 0.0,
}


# ── 데이터 계층 (캐시 + 오프라인) ──────────────────────────────────────────

def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name.replace(".", "_") + ".csv")


def _frame_to_lists(frame: pd.DataFrame) -> Optional[Dict[str, List[float]]]:
    try:
        cols = {c.lower(): c for c in frame.columns}
        need = {"open": cols.get("open"), "high": cols.get("high"),
                "low": cols.get("low"), "close": cols.get("close"),
                "volume": cols.get("volume")}
        if any(v is None for v in need.values()):
            return None
        sub = frame[[need["open"], need["high"], need["low"],
                      need["close"], need["volume"]]].dropna()
        if len(sub) < 150:
            return None
        return {
            "opens": [float(v) for v in sub.iloc[:, 0]],
            "highs": [float(v) for v in sub.iloc[:, 1]],
            "lows": [float(v) for v in sub.iloc[:, 2]],
            "closes": [float(v) for v in sub.iloc[:, 3]],
            "volumes": [float(v) for v in sub.iloc[:, 4]],
        }
    except (TypeError, ValueError, IndexError):
        return None


CACHE_TTL_S = 24 * 3600  # 1일 경과 캐시는 재수집 (stale 방지)


def _cache_fresh(path: str, ttl: float = CACHE_TTL_S) -> bool:
    try:
        return time.time() - os.path.getmtime(path) < ttl
    except OSError:
        return False


def fetch_ohlcv(ticker: str, offline: bool = False) -> Optional[Dict[str, List[float]]]:
    """1년 일봉 조회. 신선한 캐시 우선, 만료 시 재수집, --offline이면 캐시만."""
    path = _cache_path(ticker)
    if os.path.exists(path) and (offline or _cache_fresh(path)):
        try:
            cached = pd.read_csv(path)
            parsed = _frame_to_lists(cached)
            if parsed is not None:
                return parsed
        except (OSError, ValueError):
            pass
    if offline:
        return None
    try:
        import yfinance as yf
        frame = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if frame is None or frame.empty:
            return None
        frame.to_csv(path)
        return _frame_to_lists(frame)
    except Exception as e:
        print(f"[prune] {ticker} 수집 실패 ({type(e).__name__})", flush=True)
        return None


# ── 지표 헬퍼 ─────────────────────────────────────────────────────────────

def atr_at(highs: List[float], lows: List[float], closes: List[float],
           idx: int, period: int = 14) -> Optional[float]:
    """idx 봉까지의 period-ATR (인과적, idx 이전 데이터만 사용)."""
    try:
        if idx < period:
            return None
        trs = []
        for i in range(idx - period + 1, idx + 1):
            h, l, pc = highs[i], lows[i], closes[i - 1]
            if not all(math.isfinite(v) and v > 0 for v in (h, l, pc)):
                return None
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        atr = sum(trs) / len(trs)
        return atr if math.isfinite(atr) and atr > 0 else None
    except (TypeError, IndexError):
        return None


from market_briefing.technique_prune import (  # 단일 조건 원천 (실전 게이트와 공유)
    bucket as _bucket,
    drsi_conditions,
    hybrid_conditions,
    leader_conditions,
    ml_conditions,
    pattern_conditions,
)


# ── 신호 생성 (기법별, 전부 인과적 prefix만 사용) ──────────────────────────

Signal = Dict[str, Any]


def _base_signal(ticker: str, market: str, technique: str, sig_idx: int,
                 ohl: Dict[str, List[float]], stop: float,
                 target: float, conditions: Dict[str, Any]) -> Optional[Signal]:
    n = len(ohl["closes"])
    entry_idx = sig_idx + 1
    if entry_idx >= n:
        return None
    try:
        entry = float(ohl["opens"][entry_idx])
        stop_f, tgt_f = float(stop), float(target)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(entry) and math.isfinite(stop_f)
            and math.isfinite(tgt_f)):
        return None
    if entry <= 0 or stop_f >= entry or tgt_f <= entry:
        return None
    return {
        "ticker": ticker, "market": market, "technique": technique,
        "signal_index": sig_idx, "entry_index": entry_idx,
        "entry_price": entry, "stop_price": stop_f, "target_price": tgt_f,
        "conditions": dict(conditions),
    }


def gen_hybrid_breakout(ticker: str, market: str, ohl: Dict[str, List[float]],
                        bench: List[float]) -> List[Signal]:
    from market_briefing.hybrid_signals import compute_hybrid_score
    closes, highs, lows = ohl["closes"], ohl["highs"], ohl["lows"]
    volumes, opens = ohl["volumes"], ohl["opens"]
    n, out = len(closes), []
    for t in range(60, n - 1):
        try:
            hs = compute_hybrid_score(
                closes[:t + 1], highs[:t + 1], lows[:t + 1], volumes[:t + 1],
                open_prices=opens[:t + 1],
                bench_closes=bench[max(0, len(bench) - (t + 1)):],
            )
        except Exception:
            continue
        if hs.get("action") != "AUTO_YES":
            continue
        atr = atr_at(highs, lows, closes, t)
        stop, tgt_base = hs.get("stop_price"), hs.get("entry_trigger")
        if stop is None or atr is None:
            continue
        try:
            entry = float(opens[t + 1])
        except (TypeError, IndexError):
            continue
        target = entry + TP_ATR_MULT * atr
        conds = hybrid_conditions(hs)
        sig = _base_signal(ticker, market, "hybrid_breakout", t, ohl,
                           float(stop), target, conds)
        if sig:
            out.append(sig)
    return out


def gen_pattern_breakout(ticker: str, market: str,
                         ohl: Dict[str, List[float]]) -> List[Signal]:
    from market_briefing.pattern_engine import PatternEngine, PatternEngineOptions
    closes, highs, lows = ohl["closes"], ohl["highs"], ohl["lows"]
    opens, volumes = ohl["opens"], ohl["volumes"]
    n, out = len(closes), []
    for t in range(150, n - 1, 5):
        try:
            eng = PatternEngine(
                opens[:t + 1], highs[:t + 1], lows[:t + 1], closes[:t + 1],
                volumes[:t + 1],
                options=PatternEngineOptions(
                    timeframe="1D", tolerance_mode="hybrid",
                    include_forming=False),
            )
            pats = eng.detect()
        except Exception:
            continue
        bull = [p for p in pats
                if p.get("direction_code") == "bullish"
                and p.get("pattern_status") == "confirmed"
                and p.get("signal") == "매수"]
        if not bull:
            continue
        p = bull[0]
        inv, tgt = p.get("invalidation_price"), p.get("pattern_target_price")
        atr = atr_at(highs, lows, closes, t)
        if inv is None or atr is None:
            continue
        try:
            entry = float(opens[t + 1])
        except (TypeError, IndexError):
            continue
        if not (math.isfinite(float(inv)) and float(inv) < entry):
            continue
        try:
            tgt_f = float(tgt)
            if not math.isfinite(tgt_f) or tgt_f <= entry:
                tgt_f = entry + TP_ATR_MULT * atr
        except (TypeError, ValueError):
            tgt_f = entry + TP_ATR_MULT * atr
        conds = pattern_conditions(p)
        sig = _base_signal(ticker, market, "pattern_breakout", t, ohl,
                           float(inv), tgt_f, conds)
        if sig:
            out.append(sig)
    return out


def gen_dynamic_rsi(ticker: str, market: str,
                    ohl: Dict[str, List[float]]) -> List[Signal]:
    from market_briefing.dynamic_rsi import add_dynamic_rsi_features, config_for_market
    cfg = config_for_market(market)
    frame = pd.DataFrame({
        "Open": ohl["opens"], "High": ohl["highs"], "Low": ohl["lows"],
        "Close": ohl["closes"], "Volume": ohl["volumes"],
    })
    try:
        df = add_dynamic_rsi_features(frame, market, cfg)
    except Exception:
        return []
    if "DRSI_Signal" not in df.columns or "DRSI_Stop" not in df.columns:
        return []
    rsi_col = "DRSI" if "DRSI" in df.columns else None
    n, out = len(df), []
    for i in range(cfg.min_history, n - 1):
        try:
            if int(df.iloc[i]["DRSI_Signal"]) != 1:
                continue
            stop = float(df.iloc[i]["DRSI_Stop"])
        except (TypeError, ValueError, IndexError):
            continue
        atr = atr_at(ohl["highs"], ohl["lows"], ohl["closes"], i)
        if atr is None:
            continue
        try:
            entry = float(ohl["opens"][i + 1])
        except IndexError:
            continue
        if not math.isfinite(stop) or stop >= entry:
            stop = entry * (1 - cfg.round_trip_cost_pct / 100)
        target = entry + TP_ATR_MULT * atr
        rsi_v = None
        if rsi_col:
            try:
                rsi_v = float(df.iloc[i][rsi_col])
            except (TypeError, ValueError):
                rsi_v = None
        dist = (entry - stop) / entry * 100 if entry > 0 else None
        conds = drsi_conditions(market, rsi_v, dist)
        sig = _base_signal(ticker, market, "dynamic_rsi", i, ohl,
                           stop, target, conds)
        if sig:
            out.append(sig)
    return out


def gen_leader_reversal(ticker: str, market: str, ohl: Dict[str, List[float]],
                        bench: List[float]) -> List[Signal]:
    from market_briefing.leader_reversal import detect_leader_reversal
    closes, highs, lows = ohl["closes"], ohl["highs"], ohl["lows"]
    opens = ohl["opens"]
    n, out = len(closes), []
    for t in range(130, n - 1, 2):
        try:
            r = detect_leader_reversal(
                closes[:t + 1], highs[:t + 1], lows[:t + 1],
                bench_closes=bench[max(0, len(bench) - (t + 1)):],
            )
        except Exception:
            continue
        if not r.get("available") or r.get("stage") != "BREAKOUT":
            continue
        stop = r.get("stop_price")
        atr = atr_at(highs, lows, closes, t)
        if stop is None or atr is None:
            continue
        try:
            entry = float(opens[t + 1])
            stop_f = float(stop)
        except (TypeError, ValueError, IndexError):
            continue
        if stop_f >= entry:
            continue
        target = entry + TP_ATR_MULT * atr
        conds = leader_conditions(r)
        sig = _base_signal(ticker, market, "leader_reversal", t, ohl,
                           stop_f, target, conds)
        if sig:
            out.append(sig)
    return out


def gen_ml_direction(ticker: str, market: str,
                     ohl: Dict[str, List[float]]) -> List[Signal]:
    from market_briefing.ml_predictor import predict_from_ohlcv
    closes, highs, lows = ohl["closes"], ohl["highs"], ohl["lows"]
    opens, volumes = ohl["opens"], ohl["volumes"]
    n, out = len(closes), []
    for t in range(60, n - 1, 5):
        try:
            pr = predict_from_ohlcv(
                ticker, closes[:t + 1], highs[:t + 1], lows[:t + 1],
                volumes[:t + 1], market=market, opens=opens[:t + 1],
                index_cache=dict(_NEUTRAL_INDEX),
            )
        except Exception:
            continue
        if pr.get("direction") != "UP":
            continue
        try:
            conf = float(pr.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if conf < 0.60:
            continue
        atr = atr_at(highs, lows, closes, t)
        if atr is None:
            continue
        try:
            entry = float(opens[t + 1])
        except IndexError:
            continue
        stop = entry - 1.5 * atr
        target = entry + 1.5 * atr
        conds = ml_conditions(pr)
        sig = _base_signal(ticker, market, "ml_direction", t, ohl,
                           stop, target, conds)
        if sig:
            out.append(sig)
    return out


GENERATORS = {
    "hybrid_breakout": gen_hybrid_breakout,
    "pattern_breakout": gen_pattern_breakout,
    "dynamic_rsi": gen_dynamic_rsi,
    "leader_reversal": gen_leader_reversal,
    "ml_direction": gen_ml_direction,
}


# ── 통일 매매 시뮬레이터 (진입·손절 전부 기록, 동시 도달은 손절 우선) ──────

def simulate_trades(signals: List[Signal], ohl_by_ticker: Dict[str, Dict[str, List[float]]],
                    cost_pct: float, max_hold: int = MAX_HOLD_BARS) -> List[Dict[str, Any]]:
    """신호 → 매매 재현. 조건 충족 신호가 겹쳐도 각각 독립 기록한다."""
    trades: List[Dict[str, Any]] = []
    for s in signals:
        ohl = ohl_by_ticker.get(s["ticker"])
        if ohl is None:
            continue
        n = len(ohl["closes"])
        ei = int(s["entry_index"])
        if ei >= n:
            continue
        entry = float(s["entry_price"])
        stop = float(s["stop_price"])
        target = float(s["target_price"])
        if not (entry > 0 and stop < entry and target > entry):
            continue
        last = min(n - 1, ei + max_hold)
        exit_price: Optional[float] = None
        exit_idx = last
        reason = "timeout"
        for j in range(ei, last + 1):
            try:
                lo = float(ohl["lows"][j])
                hi = float(ohl["highs"][j])
            except (TypeError, IndexError):
                continue
            hit_sl = lo <= stop
            hit_tp = hi >= target
            if hit_sl and hit_tp:
                exit_price, exit_idx, reason = stop, j, "stop_first"
                break
            if hit_sl:
                exit_price, exit_idx, reason = stop, j, "stop"
                break
            if hit_tp:
                exit_price, exit_idx, reason = target, j, "target"
                break
        if exit_price is None:
            try:
                exit_price = float(ohl["closes"][last])
            except (TypeError, IndexError):
                continue
        gross = (exit_price / entry - 1.0) * 100.0
        net = gross - cost_pct
        trades.append({
            "ticker": s["ticker"], "market": s["market"],
            "technique": s["technique"],
            "signal_index": s["signal_index"], "entry_index": ei,
            "entry_price": round(entry, 4), "stop_price": round(stop, 4),
            "target_price": round(target, 4),
            "exit_index": exit_idx, "exit_price": round(float(exit_price), 4),
            "exit_reason": reason,
            "gross_return_pct": round(gross, 3), "return_pct": round(net, 3),
            "win": bool(net > 0),
            "conditions": dict(s.get("conditions") or {}),
        })
    return trades


def summarize_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "wins": 0, "winrate": 0.0, "avg_return_pct": 0.0,
                "total_return_pct": 0.0, "expectancy_pct": 0.0,
                "loss_rate": 0.0, "profit_factor": 0.0}
    wins = sum(1 for t in trades if t["win"])
    rets = [float(t["return_pct"]) for t in trades]
    gp = sum(r for r in rets if r > 0)
    gl = -sum(r for r in rets if r < 0)
    return {
        "trades": n, "wins": wins,
        "winrate": round(wins / n * 100, 2),
        "avg_return_pct": round(sum(rets) / n, 3),
        "total_return_pct": round(sum(rets), 2),
        "expectancy_pct": round(sum(rets) / n, 3),
        "loss_rate": round((n - wins) / n * 100, 2),
        "profit_factor": round(gp / gl, 3) if gl > 0 else (float("inf") if gp > 0 else 0.0),
    }


# ── 손실 귀인 + 조건 제거 루프 ─────────────────────────────────────────────

def loss_attribution(trades: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, int]]:
    """손실 매매를 조건값별로 집계. {(조건, 값): {losses, total}}."""
    totals: Dict[Tuple[str, str], int] = {}
    losses: Dict[Tuple[str, str], int] = {}
    for t in trades:
        conds = t.get("conditions") or {}
        for k, v in conds.items():
            key = (str(k), str(v))
            totals[key] = totals.get(key, 0) + 1
            if not t["win"]:
                losses[key] = losses.get(key, 0) + 1
    return {k: {"losses": losses.get(k, 0), "total": totals[k]}
            for k in totals}


def best_exclusion(trades: List[Dict[str, Any]],
                   min_support: int = MIN_SUPPORT) -> Optional[Dict[str, Any]]:
    """손실이 가장 자주 발생한 조건 1개. (손실건수 최대, 동률이면 손실율 최대)."""
    stats = summarize_trades(trades)
    overall = stats["loss_rate"]
    attr = loss_attribution(trades)
    best: Optional[Dict[str, Any]] = None
    for (key, val), info in attr.items():
        if info["total"] < min_support or info["losses"] == 0:
            continue
        rate = info["losses"] / info["total"] * 100
        if rate <= overall:
            continue
        cand = {"condition": key, "value": val, "losses": info["losses"],
                "total": info["total"], "loss_rate": round(rate, 2)}
        if (best is None or cand["losses"] > best["losses"]
                or (cand["losses"] == best["losses"]
                    and cand["loss_rate"] > best["loss_rate"])):
            best = cand
    return best


def filter_signals(signals: List[Signal],
                   exclusions: List[Dict[str, Any]]) -> List[Signal]:
    """제외 규칙과 1개라도 일치하는 신호를 버린다."""
    kept = []
    for s in signals:
        conds = s.get("conditions") or {}
        if any(str(conds.get(r["condition"])) == str(r["value"])
               for r in exclusions):
            continue
        kept.append(s)
    return kept


def prune_technique(signals: List[Signal],
                    ohl_by_ticker: Dict[str, Dict[str, List[float]]],
                    cost_pct: float,
                    min_trades: int = MIN_TRADES,
                    max_iters: int = MAX_ITERS) -> Dict[str, Any]:
    """조건 제거 루프. 손실율 개선 + 최소매매수 + 기대값 하한을 모두 만족할 때만 유지."""
    base_trades = simulate_trades(signals, ohl_by_ticker, cost_pct)
    base = summarize_trades(base_trades)
    result: Dict[str, Any] = {
        "baseline": base, "steps": [], "exclusions": [],
        "pruned": base, "verdict": "KEEP", "note": "",
        "kept_trades": base_trades,
    }
    if base["trades"] < min_trades:
        result["verdict"] = "INSUFFICIENT"
        result["note"] = f"매매 {base['trades']}건으로 최소 {min_trades}건 미달"
        return result
    exclusions: List[Dict[str, Any]] = []
    cur_trades, cur = base_trades, base
    for _ in range(max_iters):
        cand = best_exclusion(cur_trades)
        if cand is None:
            break
        trial_excl = exclusions + [cand]
        trial_trades = simulate_trades(filter_signals(signals, trial_excl),
                                       ohl_by_ticker, cost_pct)
        trial = summarize_trades(trial_trades)
        if trial["trades"] < min_trades:
            break
        improved = trial["loss_rate"] < cur["loss_rate"] - 1e-9
        guarded = trial["expectancy_pct"] >= base["expectancy_pct"] - EXPECT_TOL_PCT
        if improved and guarded:
            exclusions = trial_excl
            cur_trades, cur = trial_trades, trial
            result["steps"].append({
                "removed": {"condition": cand["condition"], "value": cand["value"]},
                "support": {"losses": cand["losses"], "total": cand["total"],
                            "loss_rate": cand["loss_rate"]},
                "after": trial,
            })
        else:
            break
    result["exclusions"] = [
        {"condition": e["condition"], "value": e["value"]} for e in exclusions]
    result["pruned"] = cur
    result["kept_trades"] = cur_trades
    if cur["expectancy_pct"] <= 0:
        result["verdict"] = "DROP"
        result["note"] = "제거 후에도 기대값 <= 0"
    elif exclusions:
        result["verdict"] = "PRUNED_KEEP"
    return result


# ── 실행 ───────────────────────────────────────────────────────────────────

def tier_of_ticker(ticker: str, market: str) -> str:
    """티커의 시총 티어. 기본 유니버스에 없으면(-tickers 지정) MID 취급."""
    for tier in TIERS:
        if ticker in DEFAULT_TICKERS.get(market, {}).get(tier, []):
            return tier
    return "MID"


def _resolve_tickers(args: argparse.Namespace) -> Dict[str, List[str]]:
    if args.tickers:
        out = {"KRX": [], "US": []}
        for tok in args.tickers.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.upper().endswith((".KS", ".KQ")):
                out["KRX"].append(tok)
            else:
                out["US"].append(tok.upper())
        picked = {k: v for k, v in out.items() if v} or {"KRX": [], "US": []}
        if args.market in ("KRX", "US"):
            picked = {args.market: picked.get(args.market, [])}
        return picked
    tiers = [t.upper() for t in (args.tiers or "LARGE,MID,SMALL").split(",")]
    tiers = [t for t in tiers if t in TIERS] or list(TIERS)
    markets = {}
    for market in ("KRX", "US"):
        picked: List[str] = []
        for tier in tiers:
            names = list(DEFAULT_TICKERS[market][tier])
            if args.limit:
                names = names[:args.limit]
            picked.extend(names)
        markets[market] = picked
    if args.market in ("KRX", "US"):
        markets = {args.market: markets.get(args.market, [])}
    return markets


def run(args: argparse.Namespace) -> Dict[str, Any]:
    tickers_by_market = _resolve_tickers(args)
    min_trades = args.min_trades
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "markets": {}, "rules": {},
        "params": {"min_trades": min_trades, "max_iters": args.max_iters,
                   "max_hold": MAX_HOLD_BARS, "tp_atr": TP_ATR_MULT,
                   "costs": COST_PCT},
    }
    all_trades: List[Dict[str, Any]] = []
    for market, tickers in tickers_by_market.items():
        ohl_by_ticker: Dict[str, Dict[str, List[float]]] = {}
        bench: List[float] = []
        for tk in tickers:
            ohl = fetch_ohlcv(tk, args.offline)
            if ohl is None:
                print(f"[prune] {tk}: 데이터 없음 — 제외", flush=True)
                continue
            ohl_by_ticker[tk] = ohl
        b = fetch_ohlcv(BENCH[market], args.offline)
        if b is not None:
            bench = b["closes"]
        print(f"[prune] {market}: {len(ohl_by_ticker)}종목 확보", flush=True)
        market_node: Dict[str, Any] = {}
        for tech in TECHNIQUES:
            if args.technique and tech != args.technique:
                continue
            gen = GENERATORS[tech]
            signals: List[Signal] = []
            t0 = time.time()
            for tk, ohl in ohl_by_ticker.items():
                try:
                    if tech in ("hybrid_breakout", "leader_reversal"):
                        fresh = gen(tk, market, ohl, bench)
                    else:
                        fresh = gen(tk, market, ohl)
                    tier = tier_of_ticker(tk, market)
                    for s in fresh:
                        s["conditions"]["cap_tier"] = tier
                    signals.extend(fresh)
                except Exception as e:
                    print(f"[prune] {tech}/{tk} 신호 실패 ({type(e).__name__})",
                          flush=True)
            print(f"[prune] {market}/{tech}: 신호 {len(signals)}개 "
                  f"({time.time() - t0:.1f}s)", flush=True)
            res = prune_technique(signals, ohl_by_ticker, COST_PCT[market],
                                  min_trades=min_trades,
                                  max_iters=args.max_iters)
            market_node[tech] = {k: v for k, v in res.items()
                                 if k != "kept_trades"}
            all_trades.extend(res.get("kept_trades") or [])
            print(f"[prune] {market}/{tech}: {res['verdict']} "
                  f"기준 {res['baseline']['trades']}건 손절율 "
                  f"{res['baseline']['loss_rate']}% → 제거 후 "
                  f"{res['pruned']['loss_rate']}% "
                  f"(제외 {len(res['exclusions'])}조건)", flush=True)
            report["rules"].setdefault(tech, {})[market] = {
                "status": res["verdict"],
                "exclusions": res["exclusions"],
                "baseline": res["baseline"],
                "pruned": res["pruned"],
                "steps": res["steps"],
            }
        report["markets"][market] = market_node
    # ── 산출물 저장 ──
    os.makedirs(os.path.dirname(RULES_OUT), exist_ok=True)
    with open(RULES_OUT, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "generated_at": report["generated_at"],
                   "rules": report["rules"], "params": report["params"]},
                  f, ensure_ascii=False, indent=2)
    if all_trades:
        os.makedirs(os.path.dirname(TRADES_OUT), exist_ok=True)
        with open(TRADES_OUT, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ticker", "market", "technique", "signal_index",
                        "entry_index", "entry_price", "stop_price",
                        "target_price", "exit_index", "exit_price",
                        "exit_reason", "return_pct", "win", "conditions"])
            for t in all_trades:
                w.writerow([t["ticker"], t["market"], t["technique"],
                            t["signal_index"], t["entry_index"],
                            t["entry_price"], t["stop_price"],
                            t["target_price"], t["exit_index"],
                            t["exit_price"], t["exit_reason"],
                            t["return_pct"], int(t["win"]),
                            json.dumps(t["conditions"], ensure_ascii=False)])
    print(f"[prune] 규칙 저장: {RULES_OUT}", flush=True)
    print(f"[prune] 매매 기록: {TRADES_OUT} ({len(all_trades)}건)", flush=True)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="기법별 손실조건 제거 검증")
    ap.add_argument("--tickers", default="",
                    help="쉼표 구분 티커 (KRX는 .KS/.KQ 접미사)")
    ap.add_argument("--market", default="ALL", choices=["ALL", "KRX", "US"])
    ap.add_argument("--limit", type=int, default=0,
                    help="티어별 종목 수 상한 (0=티어 전체)")
    ap.add_argument("--tiers", default="LARGE,MID,SMALL",
                    help="검증 티어 선택 (예: LARGE,MID)")
    ap.add_argument("--technique", default="",
                    choices=[""] + list(TECHNIQUES))
    ap.add_argument("--min-trades", type=int, default=MIN_TRADES)
    ap.add_argument("--max-iters", type=int, default=MAX_ITERS)
    ap.add_argument("--offline", action="store_true",
                    help="캐시된 CSV만 사용 (네트워크 차단)")
    args = ap.parse_args(argv)
    try:
        run(args)
        return 0
    except Exception as e:
        print(f"[prune] 실패: {e}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
