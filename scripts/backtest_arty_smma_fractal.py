#!/usr/bin/env python3
"""SMMA 21·50·200 + Williams Fractal 실데이터 1년 워크포워드 백테스트.

검증 항목:
1. 프랙탈 확정 봉 종가와 다음 1·3·5봉 시가 진입 비교
2. SMMA21/SMMA50 재시험 분리
3. 시장별 ATR 접촉 허용폭 탐색
4. 거래량 감소 눌림 + 거래량 증가 확인 필터
5. SMMA200 횡보장 기울기 필터
6. 배당·분할 전후 조정주가 연속성 감사

최근 252거래일을 앞 168거래일 학습, 뒤 84거래일 검증으로 나눈다.
지표 워밍업에는 그 이전 데이터만 사용하며, 학습 거래는 보유 종료일까지
검증 구간에 침범하지 않게 제한한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import FinanceDataReader as fdr
except ImportError:  # 선택적 보조 공급자
    fdr = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.index import (  # noqa: E402
    ARTY_STRATEGY_RULE_VERSION,
    _US_RECO_UNIVERSE,
    _arty_evaluate_entry,
    _arty_evaluate_setup,
    _prepare_arty_series,
    _confirmed_williams_fractals,
    _smma_values,
)
from scripts.backtest_prediction_zones import KRX_FALLBACK_100  # noqa: E402


KRX_TICKERS = list(dict.fromkeys(KRX_FALLBACK_100))
US_TICKERS = [
    ticker
    for ticker in dict.fromkeys(_US_RECO_UNIVERSE)
    if ticker not in {"SPY", "QQQ", "DIA", "IWM"}
][:100]

SYMBOL_LINEAGE = {
    "091990.KQ": {
        "status": "merged_delisted",
        "effective_date": "2024-01-12",
        "successor": "068270.KS",
        "action": "exclude",
        "reason": "셀트리온헬스케어가 셀트리온에 합병되어 별도 상장 이력이 종료됨",
    },
    "MMC": {
        "status": "renamed",
        "effective_date": "2026-01-14",
        "successor": "MRSH",
        "action": "replace",
        "reason": "Marsh McLennan의 NYSE 심볼이 MRSH로 변경됨",
    },
}

ENTRY_DELAYS = (0, 1, 3, 5)  # 0=확정 봉 종가, 나머지=확정 후 N봉 시가
RETEST_LINES = (21, 50)
ATR_TOLERANCES = (0.15, 0.25, 0.35, 0.50)
PULLBACK_VOLUME_MAX = (None, 1.00, 0.85)
REBOUND_VOLUME_MIN = (None, 1.00, 1.15)
SMMA200_SLOPE_MIN = (None, 0.00, 0.10)
SMMA200_SLOPE_SENSITIVITY = (
    -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
)
MAX_HOLD_BARS = 20
RISK_REWARD = 2.0
TRAIN_BARS = 168
TEST_BARS = 84
MIN_TRAIN_TRADES = 30
ROUND_TRIP_COST_PCT = {"KRX": 0.20, "US": 0.10}
SLIPPAGE_STRESS_PCT = (0.00, 0.10, 0.25, 0.50)
QUARTERLY_TRAIN_BARS = 189
QUARTERLY_TEST_BARS = 63
MONTHLY_TRAIN_BARS = 231
MONTHLY_TEST_BARS = 21
MARKET_BENCHMARKS = {"KRX": "^KS11", "US": "SPY"}
MIN_HISTORY_ROWS = 470


@dataclass(frozen=True)
class StrategyConfig:
    retest_line: int
    entry_delay: int
    atr_tolerance: float
    pullback_volume_max: Optional[float]
    rebound_volume_min: Optional[float]
    smma200_slope_min_pct20: Optional[float]

    @property
    def key(self) -> str:
        return (
            f"L{self.retest_line}|D{self.entry_delay}|ATR{self.atr_tolerance:.2f}|"
            f"PV{self.pullback_volume_max}|RV{self.rebound_volume_min}|"
            f"S{self.smma200_slope_min_pct20}"
        )

    @property
    def entry_label(self) -> str:
        return (
            "확정 봉 종가(비실행 비교)"
            if self.entry_delay == 0
            else f"확정 후 {self.entry_delay}봉 시가"
        )


@dataclass
class Trade:
    ticker: str
    market: str
    split: str
    config_key: str
    retest_line: int
    entry_delay: int
    signal_date: str
    entry_date: str
    exit_date: str
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    exit_reason: str
    gross_return_pct: float
    return_pct: float
    r_multiple: float
    win: bool
    open_gap_vs_confirm_close_pct: float
    pullback_volume_ratio: float
    rebound_volume_ratio: float
    smma200_slope_pct20: float
    atr_tolerance: float
    average_turnover_20: float
    market_regime: str
    earnings_distance_bars: Optional[int]


def config_grid() -> List[StrategyConfig]:
    return [
        StrategyConfig(line, delay, tolerance, pullback, rebound, slope)
        for line in RETEST_LINES
        for delay in ENTRY_DELAYS
        for tolerance in ATR_TOLERANCES
        for pullback in PULLBACK_VOLUME_MAX
        for rebound in REBOUND_VOLUME_MIN
        for slope in SMMA200_SLOPE_MIN
    ]


def strategy_config_hash(configs: List[StrategyConfig]) -> str:
    payload = {
        "rule_version": ARTY_STRATEGY_RULE_VERSION,
        "configs": [asdict(config) for config in configs],
        "max_hold_bars": MAX_HOLD_BARS,
        "risk_reward": RISK_REWARD,
        "train_bars": TRAIN_BARS,
        "test_bars": TEST_BARS,
        "minimum_train_trades": MIN_TRAIN_TRADES,
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        "slippage_stress_pct": SLIPPAGE_STRESS_PCT,
        "quarterly_walk_forward": {
            "train_bars": QUARTERLY_TRAIN_BARS,
            "test_bars": QUARTERLY_TEST_BARS,
        },
        "smma200_slope_sensitivity": SMMA200_SLOPE_SENSITIVITY,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_history(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """비조정 원본과 Adj Close 비율로 만든 조정 OHLCV를 반환한다."""
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] if isinstance(col, tuple) else col for col in raw.columns]
    raw = raw.rename(columns=str.title).copy()
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in raw.columns for col in required):
        return pd.DataFrame(), pd.DataFrame()
    raw = raw.dropna(subset=["Open", "High", "Low", "Close"])
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    adj_close = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
    ratio = (adj_close / raw["Close"].replace(0, np.nan)).fillna(1.0)
    adjusted = raw[required].copy()
    for col in ("Open", "High", "Low", "Close"):
        adjusted[col] = raw[col] * ratio
    adjusted["Volume"] = raw["Volume"]
    return raw, adjusted


def _period_start(period: str) -> str:
    years = int(period[:-1]) if period.endswith("y") and period[:-1].isdigit() else 3
    return str((pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=years, days=10)).date())


def _download_yfinance_history(ticker: str, period: str) -> pd.DataFrame:
    quote = yf.Ticker(ticker)
    raw = quote.history(
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=True,
        timeout=20,
    )
    if raw is not None and len(raw) >= MIN_HISTORY_ROWS:
        return raw
    # period 조회가 신규/변경 심볼에서 실패할 때 명시적 시작일로 한 번 재시도한다.
    return quote.history(
        start=_period_start(period),
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=True,
        timeout=20,
    )


def _download_fdr_history(ticker: str, period: str) -> pd.DataFrame:
    if fdr is None:
        return pd.DataFrame()
    provider_ticker = ticker.split(".", 1)[0] if ticker.endswith((".KS", ".KQ")) else ticker
    frame = fdr.DataReader(provider_ticker, _period_start(period))
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.rename(columns=str.title).copy()
    if "Adj Close" not in frame.columns:
        frame["Adj Close"] = frame["Close"]
    if "Dividends" not in frame.columns:
        frame["Dividends"] = 0.0
    if "Stock Splits" not in frame.columns:
        frame["Stock Splits"] = 0.0
    return frame


def _attach_strategy_series(
    adjusted: pd.DataFrame,
    *,
    provider: str,
    requested_ticker: str,
    resolved_ticker: str,
) -> pd.DataFrame:
    adjusted = adjusted.copy()
    strategy_series = _prepare_arty_series(
        {
            column: adjusted[column].tolist()
            for column in ("Open", "High", "Low", "Close", "Volume")
        }
    )
    for column in ("ATR", "SMMA21", "SMMA50", "SMMA200"):
        adjusted[column] = strategy_series[column]
    adjusted.attrs.update({
        "arty_series": strategy_series,
        "provider": provider,
        "requested_ticker": requested_ticker,
        "resolved_ticker": resolved_ticker,
        "earnings_dates": [],
        "benchmark_regime_by_date": {},
    })
    return adjusted


def resolve_symbol(ticker: str) -> Tuple[Optional[str], Optional[Dict]]:
    lineage = SYMBOL_LINEAGE.get(ticker)
    if not lineage:
        return ticker, None
    if lineage["action"] == "exclude":
        return None, dict(lineage)
    return str(lineage["successor"]), dict(lineage)


def download_one(ticker: str, period: str = "3y") -> Tuple[pd.DataFrame, pd.DataFrame]:
    resolved, lineage = resolve_symbol(ticker)
    if resolved is None:
        empty = pd.DataFrame()
        empty.attrs["download_meta"] = {
            "requested_ticker": ticker,
            "resolved_ticker": None,
            "provider": None,
            "lineage": lineage,
            "failure_reason": "retired_symbol_excluded",
        }
        return empty, empty

    attempts = []
    for provider, loader in (
        ("Yahoo Finance via yfinance", _download_yfinance_history),
        ("FinanceDataReader", _download_fdr_history),
    ):
        try:
            raw = loader(resolved, period)
            source, adjusted = _normalize_history(raw)
            attempts.append({"provider": provider, "rows": len(adjusted)})
            if len(adjusted) < MIN_HISTORY_ROWS:
                continue
            adjusted = _attach_strategy_series(
                adjusted,
                provider=provider,
                requested_ticker=ticker,
                resolved_ticker=resolved,
            )
            meta = {
                "requested_ticker": ticker,
                "resolved_ticker": resolved,
                "provider": provider,
                "lineage": lineage,
                "attempts": attempts,
                "failure_reason": None,
            }
            adjusted.attrs["download_meta"] = meta
            source.attrs["download_meta"] = meta
            return adjusted, source
        except Exception as exc:
            attempts.append({
                "provider": provider,
                "rows": 0,
                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            })

    empty = pd.DataFrame()
    empty.attrs["download_meta"] = {
        "requested_ticker": ticker,
        "resolved_ticker": resolved,
        "provider": None,
        "lineage": lineage,
        "attempts": attempts,
        "failure_reason": "insufficient_history",
    }
    print(f"  download failed: {ticker} -> {resolved}: {attempts}", flush=True)
    return empty, empty


def fetch_earnings_dates(ticker: str) -> List[str]:
    try:
        dates = yf.Ticker(ticker).get_earnings_dates(limit=32)
        if dates is None or dates.empty:
            return []
        normalized = pd.DatetimeIndex(dates.index)
        if normalized.tz is not None:
            normalized = normalized.tz_convert(None)
        return sorted({str(value.date()) for value in normalized})
    except Exception:
        return []


def _fdr_close_series(ticker: str, period: str) -> pd.Series:
    raw = _download_fdr_history(ticker, period)
    if raw.empty or "Close" not in raw:
        return pd.Series(dtype=float)
    series = raw["Close"].astype(float).copy()
    series.index = pd.DatetimeIndex(series.index).tz_localize(None)
    return series


def cross_validate_history(
    ticker: str,
    raw: pd.DataFrame,
    period: str,
) -> Dict:
    if raw.empty or "Close" not in raw:
        return {"ticker": ticker, "status": "primary_missing"}
    secondary = _fdr_close_series(ticker, period)
    primary = raw["Close"].astype(float).copy()
    primary.index = pd.DatetimeIndex(primary.index).tz_localize(None)
    joined = pd.concat(
        [primary.rename("primary"), secondary.rename("secondary")],
        axis=1,
        join="inner",
    ).dropna().tail(120)
    if len(joined) < 20:
        return {
            "ticker": ticker,
            "status": "insufficient_overlap",
            "overlap_rows": len(joined),
        }
    differences = (
        (joined["primary"] / joined["secondary"].replace(0, np.nan) - 1).abs() * 100
    ).dropna()
    median_diff = float(differences.median()) if len(differences) else None
    return {
        "ticker": ticker,
        "status": (
            "passed"
            if median_diff is not None and median_diff <= 1.0
            else "mismatch"
        ),
        "overlap_rows": len(differences),
        "median_abs_close_diff_pct": (
            round(median_diff, 4) if median_diff is not None else None
        ),
        "max_abs_close_diff_pct": (
            round(float(differences.max()), 4) if len(differences) else None
        ),
    }


def benchmark_regime_by_date(df: pd.DataFrame) -> Dict[str, str]:
    if df.empty:
        return {}
    regimes = {}
    for index in range(len(df)):
        smma200 = df["SMMA200"].iloc[index]
        if pd.isna(smma200) or index < 20:
            continue
        old = df["SMMA200"].iloc[index - 20]
        slope = (float(smma200) / float(old) - 1) * 100 if float(old) > 0 else 0.0
        close = float(df["Close"].iloc[index])
        if close > float(smma200) and slope > 0.10:
            regime = "BULL"
        elif close < float(smma200) and slope < -0.10:
            regime = "BEAR"
        else:
            regime = "SIDEWAYS"
        regimes[str(pd.Timestamp(df.index[index]).date())] = regime
    return regimes


def _volume_ratio(df: pd.DataFrame, index: int) -> float:
    if index < 20:
        return 1.0
    average = float(df["Volume"].iloc[index - 20:index].mean())
    return float(df["Volume"].iloc[index] / average) if average > 0 else 1.0


def _smma200_slope(df: pd.DataFrame, index: int) -> float:
    if index < 20:
        return 0.0
    old = float(df["SMMA200"].iloc[index - 20])
    return ((float(df["SMMA200"].iloc[index]) - old) / old * 100.0) if old > 0 else 0.0


def _earnings_distance_bars(df: pd.DataFrame, entry_index: int) -> Optional[int]:
    earnings_dates = df.attrs.get("earnings_dates") or []
    if not earnings_dates:
        return None
    index_dates = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    entry_date = index_dates[entry_index]
    distances = []
    for raw_date in earnings_dates:
        event_date = pd.Timestamp(raw_date).normalize()
        event_position = int(index_dates.searchsorted(event_date, side="left"))
        event_position = min(max(event_position, 0), len(index_dates) - 1)
        distances.append(event_position - entry_index)
    return min(distances, key=lambda value: abs(value)) if distances else None


def _simulate_trade(
    ticker: str,
    market: str,
    split: str,
    config: StrategyConfig,
    df: pd.DataFrame,
    pivot_index: int,
    confirmed_index: int,
    allowed_end: int,
    entry_evaluation: Dict,
) -> Optional[Trade]:
    entry_index = confirmed_index if config.entry_delay == 0 else confirmed_index + config.entry_delay
    if entry_index >= allowed_end or entry_index >= len(df) - 1:
        return None
    confirm_close = float(df["Close"].iloc[confirmed_index])
    entry_price = (
        confirm_close if config.entry_delay == 0
        else float(df["Open"].iloc[entry_index])
    )
    stop = float(entry_evaluation["stop"])
    # 손절 거리와 추격 제한은 실시간과 공유하는 진입 평가에서 판정 완료됐다.
    target = float(entry_evaluation["target"])
    last_index = min(entry_index + MAX_HOLD_BARS, allowed_end - 1, len(df) - 1)
    if last_index < entry_index:
        return None
    exit_price = float(df["Close"].iloc[last_index])
    exit_index = last_index
    exit_reason = "time"
    start_index = entry_index + 1 if config.entry_delay == 0 else entry_index
    for bar_index in range(start_index, last_index + 1):
        row = df.iloc[bar_index]
        bar_open = float(row["Open"])
        bar_low = float(row["Low"])
        bar_high = float(row["High"])
        if bar_open <= stop:
            exit_price, exit_index, exit_reason = bar_open, bar_index, "stop_gap"
            break
        if bar_open >= target:
            exit_price, exit_index, exit_reason = bar_open, bar_index, "target_gap"
            break
        # 일봉 안에서 손절·목표가 모두 닿으면 시간 순서를 알 수 없으므로 손절 우선.
        if bar_low <= stop:
            exit_price, exit_index, exit_reason = stop, bar_index, "stop"
            break
        if bar_high >= target:
            exit_price, exit_index, exit_reason = target, bar_index, "target"
            break

    gross_return = (exit_price - entry_price) / entry_price * 100.0
    net_return = gross_return - ROUND_TRIP_COST_PCT[market]
    risk_amount = entry_price - stop
    net_cost_amount = entry_price * ROUND_TRIP_COST_PCT[market] / 100.0
    r_multiple = ((exit_price - entry_price) - net_cost_amount) / risk_amount
    turnover_start = max(0, entry_index - 20)
    turnover = (
        df["Close"].iloc[turnover_start:entry_index].astype(float)
        * df["Volume"].iloc[turnover_start:entry_index].astype(float)
    )
    entry_date_key = str(pd.Timestamp(df.index[entry_index]).date())
    regime_lookup = df.attrs.get("benchmark_regime_by_date") or {}
    return Trade(
        ticker=ticker,
        market=market,
        split=split,
        config_key=config.key,
        retest_line=config.retest_line,
        entry_delay=config.entry_delay,
        signal_date=str(df.index[confirmed_index].date()),
        entry_date=str(df.index[entry_index].date()),
        exit_date=str(df.index[exit_index].date()),
        entry_index=entry_index,
        exit_index=exit_index,
        entry_price=round(entry_price, 6),
        exit_price=round(exit_price, 6),
        stop_price=round(stop, 6),
        target_price=round(target, 6),
        exit_reason=exit_reason,
        gross_return_pct=round(gross_return, 6),
        return_pct=round(net_return, 6),
        r_multiple=round(r_multiple, 6),
        win=net_return > 0,
        open_gap_vs_confirm_close_pct=round(
            (entry_price - confirm_close) / confirm_close * 100.0, 6
        ),
        pullback_volume_ratio=round(_volume_ratio(df, pivot_index), 6),
        rebound_volume_ratio=round(_volume_ratio(df, confirmed_index), 6),
        smma200_slope_pct20=round(_smma200_slope(df, confirmed_index), 6),
        atr_tolerance=config.atr_tolerance,
        average_turnover_20=(
            round(float(turnover.mean()), 4) if len(turnover) else 0.0
        ),
        market_regime=regime_lookup.get(entry_date_key, "UNKNOWN"),
        earnings_distance_bars=_earnings_distance_bars(df, entry_index),
    )


def backtest_config(
    ticker: str,
    market: str,
    df: pd.DataFrame,
    config: StrategyConfig,
    *,
    train_bars: int = TRAIN_BARS,
    test_bars: int = TEST_BARS,
) -> List[Trade]:
    series = df.attrs.get("arty_series")
    if not series:
        series = _prepare_arty_series({
            column: df[column].tolist()
            for column in ("Open", "High", "Low", "Close", "Volume")
        })
    highs = series["High"]
    lows = series["Low"]
    lower_fractals = _confirmed_williams_fractals(highs, lows)["lower"]
    year_start = max(220, len(df) - (train_bars + test_bars))
    split_index = len(df) - test_bars
    trades: List[Trade] = []
    next_allowed = {"train": year_start, "test": split_index}

    for fractal in lower_fractals:
        pivot = int(fractal["index"])
        confirmed = int(fractal["confirmed_index"])
        if confirmed < year_start:
            continue
        split = "train" if confirmed < split_index else "test"
        allowed_start = year_start if split == "train" else split_index
        allowed_end = split_index if split == "train" else len(df)
        if confirmed < allowed_start or confirmed >= allowed_end - MAX_HOLD_BARS:
            continue
        if confirmed < next_allowed[split]:
            continue

        config_payload = asdict(config)
        setup = _arty_evaluate_setup(
            series, pivot, confirmed, config_payload
        )
        if not setup["passed"]:
            continue
        entry_index = (
            confirmed if config.entry_delay == 0
            else confirmed + config.entry_delay
        )
        if entry_index >= allowed_end or entry_index >= len(df) - 1:
            continue
        entry_price = (
            float(series["Close"][entry_index])
            if config.entry_delay == 0
            else float(series["Open"][entry_index])
        )
        entry_evaluation = _arty_evaluate_entry(
            series, pivot, entry_index, entry_price, config_payload
        )
        if not entry_evaluation["passed"]:
            continue

        trade = _simulate_trade(
            ticker,
            market,
            split,
            config,
            df,
            pivot,
            confirmed,
            allowed_end,
            entry_evaluation,
        )
        if trade:
            trades.append(trade)
            next_allowed[split] = trade.exit_index + 1
    return trades


def summarize_trades(trades: Iterable[Trade]) -> Dict:
    rows = list(trades)
    if not rows:
        return {
            "trades": 0,
            "win_rate_pct": None,
            "avg_return_pct": None,
            "median_return_pct": None,
            "avg_r_multiple": None,
            "median_r_multiple": None,
            "avg_r_95ci_low": None,
            "avg_r_95ci_high": None,
            "profit_factor": None,
            "return_std_pct": None,
            "largest_profit_share_pct": None,
            "avg_open_gap_vs_confirm_close_pct": None,
            "target_rate_pct": None,
            "stop_rate_pct": None,
        }
    returns = np.array([row.return_pct for row in rows], dtype=float)
    r_values = np.array([row.r_multiple for row in rows], dtype=float)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else None
    r_std = float(np.std(r_values, ddof=1)) if len(r_values) > 1 else 0.0
    r_margin = 1.96 * r_std / math.sqrt(len(r_values)) if r_values.size else 0.0
    positive_total = float(wins.sum()) if len(wins) else 0.0
    largest_profit_share = (
        float(wins.max() / positive_total * 100.0)
        if positive_total > 0 and len(wins) else None
    )
    return {
        "trades": len(rows),
        "win_rate_pct": round(float((returns > 0).mean() * 100.0), 2),
        "avg_return_pct": round(float(returns.mean()), 4),
        "median_return_pct": round(float(np.median(returns)), 4),
        "avg_r_multiple": round(float(r_values.mean()), 4),
        "median_r_multiple": round(float(np.median(r_values)), 4),
        "avg_r_95ci_low": round(float(r_values.mean() - r_margin), 4),
        "avg_r_95ci_high": round(float(r_values.mean() + r_margin), 4),
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "return_std_pct": round(float(np.std(returns, ddof=1)), 4)
        if len(returns) > 1 else 0.0,
        "largest_profit_share_pct": (
            round(largest_profit_share, 2)
            if largest_profit_share is not None else None
        ),
        "avg_open_gap_vs_confirm_close_pct": round(
            float(np.mean([row.open_gap_vs_confirm_close_pct for row in rows])), 4
        ),
        "target_rate_pct": round(
            sum(row.exit_reason.startswith("target") for row in rows) / len(rows) * 100.0, 2
        ),
        "stop_rate_pct": round(
            sum(row.exit_reason.startswith("stop") for row in rows) / len(rows) * 100.0, 2
        ),
    }


def objective(summary: Dict) -> float:
    count = int(summary.get("trades") or 0)
    if count < MIN_TRAIN_TRADES:
        return -999.0
    avg_r_ci_low_raw = summary.get("avg_r_95ci_low")
    median_r_raw = summary.get("median_r_multiple")
    avg_r_ci_low = float(avg_r_ci_low_raw) if avg_r_ci_low_raw is not None else -9.0
    median_r = float(median_r_raw) if median_r_raw is not None else -9.0
    pf = float(summary.get("profit_factor") or 0.0)
    win_rate = float(summary.get("win_rate_pct") or 0.0) / 100.0
    concentration = float(summary.get("largest_profit_share_pct") or 100.0)
    concentration_penalty = max(0.0, concentration - 35.0) / 100.0
    return (
        avg_r_ci_low
        + median_r * 0.15
        + min(pf, 2.0) * 0.05
        + win_rate * 0.03
        + math.log1p(count) * 0.01
        - concentration_penalty * 0.10
    )


def optimize_market(
    market: str,
    configs: List[StrategyConfig],
    trades_by_config: Dict[str, List[Trade]],
) -> Dict:
    scored = []
    for config in configs:
        rows = trades_by_config.get(config.key, [])
        train_summary = summarize_trades(row for row in rows if row.split == "train")
        test_summary = summarize_trades(row for row in rows if row.split == "test")
        scored.append({
            "config": config,
            "objective": objective(train_summary),
            "train": train_summary,
            "test": test_summary,
        })
    eligible = [row for row in scored if row["objective"] > -900]
    eligible.sort(key=lambda row: row["objective"], reverse=True)
    executable = [row for row in eligible if row["config"].entry_delay >= 1]
    executable_fallback = [row for row in scored if row["config"].entry_delay >= 1]
    best = (
        executable[0]
        if executable
        else max(executable_fallback, key=lambda row: row["train"]["trades"])
    )

    # 요인 비교는 최적 설정을 기준으로 한 번에 하나의 변수만 바꾼다.
    # 각 수준마다 나머지 변수까지 다시 최적화하면 지연·재시험선 등의 순수 효과가
    # 섞이므로, 그런 조건부 최적표는 비교 근거로 사용하지 않는다.
    scored_by_key = {row["config"].key: row for row in scored}
    factor_variants = {
        "entry_delay": [
            (str(value), replace(best["config"], entry_delay=value))
            for value in ENTRY_DELAYS
        ],
        "retest_line": [
            (str(value), replace(best["config"], retest_line=value))
            for value in RETEST_LINES
        ],
        "atr_tolerance": [
            (str(value), replace(best["config"], atr_tolerance=value))
            for value in ATR_TOLERANCES
        ],
        "volume_filter": [
            (
                f"{pullback}|{rebound}",
                replace(
                    best["config"],
                    pullback_volume_max=pullback,
                    rebound_volume_min=rebound,
                ),
            )
            for pullback in PULLBACK_VOLUME_MAX
            for rebound in REBOUND_VOLUME_MIN
        ],
        "smma200_slope": [
            (
                str(value),
                replace(best["config"], smma200_slope_min_pct20=value),
            )
            for value in SMMA200_SLOPE_MIN
        ],
    }
    factor_results = {}
    for factor, variants in factor_variants.items():
        factor_results[factor] = {}
        for value, config in variants:
            chosen = scored_by_key[config.key]
            factor_results[factor][value] = {
                "selected_config": asdict(config),
                "train": chosen["train"],
                "test": chosen["test"],
                "train_objective": round(chosen["objective"], 6),
                "comparison_mode": "one_factor_at_a_time",
            }
    return {
        "market": market,
        "best_config": asdict(best["config"]),
        "best_config_key": best["config"].key,
        "train_objective": round(best["objective"], 6),
        "train": best["train"],
        "test": best["test"],
        "top_10": [
            {
                "config": asdict(row["config"]),
                "train_objective": round(row["objective"], 6),
                "train": row["train"],
                "test": row["test"],
            }
            for row in eligible[:10]
        ],
        "factor_analysis": factor_results,
    }


def _slice_history(df: pd.DataFrame, end: int) -> pd.DataFrame:
    sliced = df.iloc[:end].copy()
    preserved = {
        key: value
        for key, value in df.attrs.items()
        if key != "arty_series"
    }
    sliced.attrs.update(preserved)
    return sliced


def walk_forward_analysis(
    market: str,
    frames: Dict[str, pd.DataFrame],
    configs: List[StrategyConfig],
    *,
    folds: int,
    frequency: str,
) -> Dict:
    if frequency == "monthly":
        train_bars, test_bars = MONTHLY_TRAIN_BARS, MONTHLY_TEST_BARS
    else:
        train_bars, test_bars = QUARTERLY_TRAIN_BARS, QUARTERLY_TEST_BARS
    fold_results = []
    selected_test_trades: List[Trade] = []
    for fold_number in range(folds):
        print(
            f"[{market}] {frequency} walk-forward "
            f"{fold_number + 1}/{folds}",
            flush=True,
        )
        offset = (folds - fold_number - 1) * test_bars
        trades_by_config: Dict[str, List[Trade]] = {
            config.key: [] for config in configs
        }
        for ticker, frame in frames.items():
            end = len(frame) - offset
            if end < 220 + train_bars + test_bars:
                continue
            sliced = _slice_history(frame, end)
            for config in configs:
                trades_by_config[config.key].extend(
                    backtest_config(
                        ticker,
                        market,
                        sliced,
                        config,
                        train_bars=train_bars,
                        test_bars=test_bars,
                    )
                )
        optimized = optimize_market(market, configs, trades_by_config)
        chosen = [
            row
            for row in trades_by_config.get(optimized["best_config_key"], [])
            if row.split == "test"
        ]
        selected_test_trades.extend(chosen)
        fold_results.append({
            "fold": fold_number + 1,
            "selected_config": optimized["best_config"],
            "train": optimized["train"],
            "test": optimized["test"],
            "test_start": min((row.signal_date for row in chosen), default=None),
            "test_end": max((row.exit_date for row in chosen), default=None),
        })
    aggregate = summarize_trades(selected_test_trades)
    positive_folds = sum(
        float((fold["test"] or {}).get("avg_r_multiple") or 0) > 0
        for fold in fold_results
    )
    return {
        "mode": f"{frequency}_walk_forward_reoptimization",
        "selection_is_independent_by_fold": True,
        "folds": fold_results,
        "aggregate": aggregate,
        "positive_avg_r_folds": positive_folds,
        "positive_avg_r_fold_ratio": (
            round(positive_folds / len(fold_results), 4) if fold_results else None
        ),
        "train_bars_per_fold": train_bars,
        "test_bars_per_fold": test_bars,
    }


def cost_sensitivity(rows: List[Trade], market: str) -> Dict:
    results = {}
    base_cost = ROUND_TRIP_COST_PCT[market]
    for extra_cost in SLIPPAGE_STRESS_PCT:
        returns = np.array(
            [row.gross_return_pct - base_cost - extra_cost for row in rows],
            dtype=float,
        )
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        results[f"{extra_cost:.2f}"] = {
            "extra_round_trip_cost_pct": extra_cost,
            "trades": len(rows),
            "avg_return_pct": (
                round(float(returns.mean()), 4) if len(returns) else None
            ),
            "win_rate_pct": (
                round(float((returns > 0).mean() * 100), 2)
                if len(returns) else None
            ),
            "profit_factor": (
                round(float(wins.sum() / abs(losses.sum())), 4)
                if len(losses) and losses.sum() else None
            ),
        }
    return results


def _grouped_trade_summaries(rows: List[Trade], attribute: str) -> Dict:
    groups: Dict[str, List[Trade]] = {}
    for row in rows:
        groups.setdefault(str(getattr(row, attribute)), []).append(row)
    return {key: summarize_trades(value) for key, value in sorted(groups.items())}


def extended_diagnostics(rows: List[Trade], market: str) -> Dict:
    test_rows = [row for row in rows if row.split == "test"]
    turnovers = np.array(
        [row.average_turnover_20 for row in test_rows if row.average_turnover_20 > 0],
        dtype=float,
    )
    liquidity = {"low": [], "mid": [], "high": []}
    if len(turnovers):
        lower, upper = np.quantile(turnovers, [1 / 3, 2 / 3])
        for row in test_rows:
            if row.average_turnover_20 <= lower:
                liquidity["low"].append(row)
            elif row.average_turnover_20 <= upper:
                liquidity["mid"].append(row)
            else:
                liquidity["high"].append(row)
    earnings = {"within_5_bars": [], "outside_5_bars": [], "unavailable": []}
    for row in test_rows:
        distance = row.earnings_distance_bars
        key = (
            "unavailable"
            if distance is None
            else "within_5_bars"
            if abs(distance) <= 5
            else "outside_5_bars"
        )
        earnings[key].append(row)
    return {
        "market_regime": _grouped_trade_summaries(test_rows, "market_regime"),
        "liquidity_by_20d_turnover_tercile": {
            key: summarize_trades(value) for key, value in liquidity.items()
        },
        "earnings_proximity": {
            key: summarize_trades(value) for key, value in earnings.items()
        },
        "cost_sensitivity": cost_sensitivity(test_rows, market),
        "limitations": {
            "liquidity": (
                "과거 시가총액 대신 진입 전 20거래일 평균 거래대금의 시장 내 "
                "삼분위수를 사용한다."
            ),
            "slippage": (
                "호가 체결 데이터가 없어 실제 슬리피지가 아니라 왕복 추가 비용 "
                "0·0.10·0.25·0.50% 스트레스 시나리오다."
            ),
            "earnings": (
                "yfinance가 반환한 과거/예정 실적일을 거래일 인덱스에 맞춰 "
                "가장 가까운 이벤트까지의 봉 수로 변환한다."
            ),
        },
    }


def slope_sensitivity_analysis(
    market: str,
    frames: Dict[str, pd.DataFrame],
    best_config: StrategyConfig,
) -> Dict:
    results = {}
    for threshold in SMMA200_SLOPE_SENSITIVITY:
        config = replace(
            best_config,
            smma200_slope_min_pct20=threshold,
        )
        rows = [
            trade
            for ticker, frame in frames.items()
            for trade in backtest_config(ticker, market, frame, config)
        ]
        results[f"{threshold:.2f}"] = {
            "selected_config": asdict(config),
            "train": summarize_trades(row for row in rows if row.split == "train"),
            "test": summarize_trades(row for row in rows if row.split == "test"),
            "comparison_mode": "one_factor_at_a_time_fine_grid",
        }
    return results


def audit_corporate_actions(
    ticker: str,
    market: str,
    raw: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> List[Dict]:
    if raw.empty or adjusted.empty:
        return []
    events = []
    dividends = raw.get("Dividends", pd.Series(0.0, index=raw.index)).fillna(0.0)
    splits = raw.get("Stock Splits", pd.Series(0.0, index=raw.index)).fillna(0.0)
    raw_smma200 = _smma_values(raw["Close"].astype(float).tolist(), 200)
    adjusted_smma200 = _smma_values(adjusted["Close"].astype(float).tolist(), 200)
    for index in raw.index[dividends.ne(0) | splits.ne(0)]:
        position = raw.index.get_loc(index)
        if not isinstance(position, (int, np.integer)) or position < 1:
            continue
        previous = raw.index[position - 1]
        raw_gap = (float(raw.loc[index, "Open"]) / float(raw.loc[previous, "Close"]) - 1) * 100
        if index not in adjusted.index or previous not in adjusted.index:
            continue
        adjusted_gap = (
            float(adjusted.loc[index, "Open"]) / float(adjusted.loc[previous, "Close"]) - 1
        ) * 100
        raw_smma_change = adjusted_smma_change = None
        if (
            position < len(raw_smma200)
            and raw_smma200[position] is not None
            and raw_smma200[position - 1] is not None
            and adjusted_smma200[position] is not None
            and adjusted_smma200[position - 1] is not None
        ):
            raw_smma_change = (
                float(raw_smma200[position]) / float(raw_smma200[position - 1]) - 1
            ) * 100
            adjusted_smma_change = (
                float(adjusted_smma200[position])
                / float(adjusted_smma200[position - 1]) - 1
            ) * 100
        events.append({
            "ticker": ticker,
            "market": market,
            "date": str(pd.Timestamp(index).date()),
            "dividend": float(dividends.loc[index]),
            "split_ratio": float(splits.loc[index]),
            "raw_overnight_gap_pct": round(raw_gap, 4),
            "adjusted_overnight_gap_pct": round(adjusted_gap, 4),
            "absolute_gap_reduction_pct_point": round(abs(raw_gap) - abs(adjusted_gap), 4),
            "raw_smma200_change_pct": (
                round(raw_smma_change, 6) if raw_smma_change is not None else None
            ),
            "adjusted_smma200_change_pct": (
                round(adjusted_smma_change, 6)
                if adjusted_smma_change is not None else None
            ),
            "smma200_jump_reduction_pct_point": (
                round(abs(raw_smma_change) - abs(adjusted_smma_change), 6)
                if raw_smma_change is not None and adjusted_smma_change is not None
                else None
            ),
        })
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="시장별 종목 수")
    parser.add_argument("--period", default="3y", help="워밍업 포함 다운로드 기간")
    parser.add_argument(
        "--walk-forward-frequency",
        choices=("quarterly", "monthly"),
        default="quarterly",
    )
    parser.add_argument("--walk-forward-folds", type=int, default=4)
    parser.add_argument("--cross-check-count", type=int, default=10)
    parser.add_argument("--skip-earnings-audit", action="store_true")
    args = parser.parse_args()

    configs = config_grid()
    all_trades: List[Trade] = []
    all_actions: List[Dict] = []
    market_results = {}
    download_status = {}
    data_end = {}
    universe_hashes = {}
    cross_provider_checks = {}
    earnings_coverage = {}

    for market, universe in (("KRX", KRX_TICKERS), ("US", US_TICKERS)):
        tickers = universe[:max(1, args.limit)]
        universe_hashes[market] = hashlib.sha256(
            "\n".join(tickers).encode("utf-8")
        ).hexdigest()
        market_trades: Dict[str, List[Trade]] = {config.key: [] for config in configs}
        success = []
        failures = []
        failure_details = []
        last_dates = []
        frames: Dict[str, pd.DataFrame] = {}
        raw_frames: Dict[str, pd.DataFrame] = {}
        provider_counts: Dict[str, int] = {}
        for number, ticker in enumerate(tickers, 1):
            print(f"[{market}] {number}/{len(tickers)} {ticker}", flush=True)
            df, raw = download_one(ticker, args.period)
            if df.empty:
                failures.append(ticker)
                failure_details.append(
                    dict(
                        df.attrs.get("download_meta")
                        or raw.attrs.get("download_meta")
                        or {"requested_ticker": ticker, "failure_reason": "unknown"}
                    )
                )
                continue
            success.append(ticker)
            frames[ticker] = df
            raw_frames[ticker] = raw
            last_dates.append(str(pd.Timestamp(df.index[-1]).date()))
            provider = str(df.attrs.get("provider") or "unknown")
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            _, adjusted_for_audit = _normalize_history(raw)
            all_actions.extend(audit_corporate_actions(
                ticker, market, raw, adjusted_for_audit
            ))

        earnings_dates_by_ticker: Dict[str, List[str]] = {}
        if not args.skip_earnings_audit and frames:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(
                        fetch_earnings_dates,
                        str(frame.attrs.get("resolved_ticker") or ticker),
                    ): ticker
                    for ticker, frame in frames.items()
                }
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        earnings_dates_by_ticker[ticker] = future.result()
                    except Exception:
                        earnings_dates_by_ticker[ticker] = []
        for ticker, frame in frames.items():
            frame.attrs["earnings_dates"] = earnings_dates_by_ticker.get(ticker, [])
        earnings_coverage[market] = {
            "requested": len(frames),
            "available": sum(bool(value) for value in earnings_dates_by_ticker.values()),
            "skipped": bool(args.skip_earnings_audit),
        }

        benchmark_ticker = MARKET_BENCHMARKS[market]
        benchmark_frame, _ = download_one(benchmark_ticker, args.period)
        regime_lookup = benchmark_regime_by_date(benchmark_frame)
        for frame in frames.values():
            frame.attrs["benchmark_regime_by_date"] = regime_lookup

        check_count = min(max(0, args.cross_check_count), len(success))
        if check_count:
            positions = sorted(set(
                int(value)
                for value in np.linspace(0, len(success) - 1, check_count)
            ))
            check_tickers = [success[position] for position in positions]
        else:
            check_tickers = []
        cross_provider_checks[market] = [
            cross_validate_history(ticker, raw_frames[ticker], args.period)
            for ticker in check_tickers
        ]

        for ticker, df in frames.items():
            for config in configs:
                rows = backtest_config(ticker, market, df, config)
                market_trades[config.key].extend(rows)
                all_trades.extend(rows)
            time.sleep(0.01)
        download_status[market] = {
            "requested": len(tickers),
            "success": len(success),
            "failed": failures,
            "failure_details": failure_details,
            "tickers": success,
            "provider_counts": provider_counts,
        }
        data_end[market] = max(last_dates) if last_dates else None
        optimized = optimize_market(market, configs, market_trades)
        selected_rows = market_trades.get(optimized["best_config_key"], [])
        optimized["extended_diagnostics"] = extended_diagnostics(
            selected_rows, market
        )
        optimized["extended_diagnostics"]["smma200_slope_sensitivity"] = (
            slope_sensitivity_analysis(
                market,
                frames,
                StrategyConfig(**optimized["best_config"]),
            )
        )
        optimized["walk_forward"] = walk_forward_analysis(
            market,
            frames,
            configs,
            folds=max(1, args.walk_forward_folds),
            frequency=args.walk_forward_frequency,
        )
        market_results[market] = optimized

    def _event_group_summary(rows: List[Dict]) -> Dict:
        audited = [
            row for row in rows
            if row["smma200_jump_reduction_pct_point"] is not None
        ]
        return {
            "events": len(rows),
            "median_abs_gap_reduction_pct_point": (
                round(float(np.median([
                    row["absolute_gap_reduction_pct_point"] for row in rows
                ])), 4) if rows else None
            ),
            "smma200_audited_events": len(audited),
            "median_smma200_jump_reduction_pct_point": (
                round(float(np.median([
                    row["smma200_jump_reduction_pct_point"] for row in audited
                ])), 6) if audited else None
            ),
        }

    dividend_rows = [row for row in all_actions if bool(row["dividend"])]
    split_rows = [row for row in all_actions if bool(row["split_ratio"])]
    action_summary = {
        "events": len(all_actions),
        "dividend_events": len(dividend_rows),
        "split_events": len(split_rows),
        "median_abs_gap_reduction_pct_point": (
            round(float(np.median([
                row["absolute_gap_reduction_pct_point"] for row in all_actions
            ])), 4)
            if all_actions else None
        ),
        "smma200_audited_events": sum(
            row["smma200_jump_reduction_pct_point"] is not None for row in all_actions
        ),
        "median_smma200_jump_reduction_pct_point": (
            round(float(np.median([
                row["smma200_jump_reduction_pct_point"]
                for row in all_actions
                if row["smma200_jump_reduction_pct_point"] is not None
            ])), 6)
            if any(
                row["smma200_jump_reduction_pct_point"] is not None
                for row in all_actions
            )
            else None
        ),
        "dividends": _event_group_summary(dividend_rows),
        "splits": _event_group_summary(split_rows),
        "note": (
            "Adj Close 비율로 OHLC 전체를 조정하며, 배당과 분할을 분리 감사한다. "
            "Yahoo 원시 Close가 이미 분할 정규화된 사례가 있어 결과를 "
            "모든 이벤트의 연속성 보장으로 해석하지 않는다."
        ),
    }
    gap_reduction = action_summary["median_abs_gap_reduction_pct_point"]
    smma_reduction = action_summary["median_smma200_jump_reduction_pct_point"]
    action_summary["continuity_verdict"] = (
        f"가격 갭 중앙값은 {'개선' if gap_reduction is not None and gap_reduction > 0 else '개선 미확인'}, "
        f"SMMA200 연속성은 {'개선' if smma_reduction is not None and smma_reduction > 0 else '일률적 개선 미확인'}"
    )

    outdir = ROOT / "docs" / "backtests"
    outdir.mkdir(parents=True, exist_ok=True)
    selected_keys = {
        market: result["best_config_key"] for market, result in market_results.items()
    }
    selected_trades = [
        row for row in all_trades if row.config_key == selected_keys.get(row.market)
    ]
    pd.DataFrame([asdict(row) for row in selected_trades]).to_csv(
        outdir / "arty_smma_fractal_selected_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(all_actions).to_csv(
        outdir / "arty_smma_fractal_corporate_actions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result = {
        "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "strategy_rule_version": ARTY_STRATEGY_RULE_VERSION,
        "strategy_config_hash": strategy_config_hash(configs),
        "universe_hashes": universe_hashes,
        "data_end": data_end,
        "data_provider": {
            "primary": "Yahoo Finance via yfinance",
            "fallback": (
                "FinanceDataReader" if fdr is not None else "unavailable"
            ),
            "yfinance_version": getattr(yf, "__version__", None),
            "auto_adjust": False,
            "repair": True,
            "ohlc_adjustment": "Adj Close / Close ratio",
            "cross_provider_checks": cross_provider_checks,
        },
        "symbol_lineage": SYMBOL_LINEAGE,
        "earnings_calendar_coverage": earnings_coverage,
        "method": "SMMA 21·50·200 + confirmed 5-bar Williams Fractal",
        "evaluation_window": (
            "종목별 최근 252거래일: 앞 168일 학습, 뒤 84일 검증. "
            "지표 계산은 이전 데이터로 워밍업하며 최대 20거래일 보유."
        ),
        "execution": {
            "entry_delays": list(ENTRY_DELAYS),
            "entry_delay_zero": "프랙탈 확정 봉 종가(비실행 비교 기준)",
            "entry_delay_zero_executable": False,
            "other_delays": "프랙탈 확정 후 N봉 시가",
            "open_gap_is_actual_slippage": False,
            "risk_reward": RISK_REWARD,
            "same_bar_collision": "손절 우선",
            "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
            "slippage_stress_pct": list(SLIPPAGE_STRESS_PCT),
            "walk_forward_frequency": args.walk_forward_frequency,
            "walk_forward_folds": max(1, args.walk_forward_folds),
        },
        "bias_limits": [
            (
                "합병·심볼 변경 계보는 정규화했지만 시점별 지수 구성종목 원장이 "
                "없어 현재 구성종목 생존편향은 완전히 제거되지 않는다."
            ),
            "일봉 OHLC만으로 같은 봉 안의 손절·목표 도달 순서를 알 수 없어 손절 우선 처리했다.",
            "1년 표본의 학습·검증 분할이므로 장기 시장 체제 전체를 대표하지 않는다.",
            "학습 최적값의 검증 성과가 낮으면 운영에는 보수적 기본값을 유지해야 한다.",
            (
                "거래대금은 과거 유동성 대용치이며 과거 시가총액·호가 체결 "
                "데이터를 대신하지 않는다."
            ),
        ],
        "grid_size_per_market": len(configs),
        "download_status": download_status,
        "markets": market_results,
        "corporate_action_audit": action_summary,
    }
    (outdir / "arty_smma_fractal_backtest_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "download_status": download_status,
        "markets": {
            market: {
                "best_config": value["best_config"],
                "train": value["train"],
                "test": value["test"],
            }
            for market, value in market_results.items()
        },
        "corporate_action_audit": action_summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
