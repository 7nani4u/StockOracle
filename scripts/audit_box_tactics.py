"""Walk-forward audit of the box tactics layer (video OlMBqW3FBes).

For every completed daily bar in ``datasets/audit_cache`` the audit rebuilds what
the service would have known at that close, then scores only what happened next:

* support test  -> did price reach the box midline before the stop?
* breakout      -> did the breakout hold above the broken resistance for 3 bars?
* ready plan    -> realized R multiple with next-open fill and 50:50 exits

It also records what the rule *rejected* (regime ``range_wait`` / weak magnet /
sell pressure) so the filter's value can be compared with its cost.

This is a rule audit, not a probability calibration and not investment advice.
Daily bars are coarser than the intraday market the video trades; every number is
reported as a sample rate with a Wilson interval, never as a win guarantee.

Usage
-----
python scripts/audit_box_tactics.py --offline --stride 5
python scripts/audit_box_tactics.py --offline --tickers 005930.KS,AAPL --stride 3
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from market_briefing.box_range import analyze_box_range  # noqa: E402
from market_briefing.box_tactics import analyze_box_tactics, VERSION  # noqa: E402
from market_briefing.trend_boxes import analyze_trend_boxes  # noqa: E402

CACHE_DIR = os.path.join(REPO_ROOT, "datasets", "audit_cache")
HORIZON = 10


def load_prices(path: str) -> Optional[pd.DataFrame]:
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except (OSError, ValueError):
        return None
    columns = ["Open", "High", "Low", "Close", "Volume"]
    if frame.empty or not set(columns).issubset(frame.columns):
        return None
    frame = frame[columns].apply(pd.to_numeric, errors="coerce").copy()
    frame.index = pd.to_datetime(pd.Index([str(v)[:10] for v in frame.index]), errors="coerce")
    frame = frame[~frame.index.isna()]
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def as_dd(frame: pd.DataFrame) -> Dict[str, list]:
    dd: Dict[str, list] = {"Date": frame.index.strftime("%Y-%m-%d").tolist()}
    for column in ("Open", "High", "Low", "Close", "Volume"):
        dd[column] = [float(v) for v in frame[column].tolist()]
    return dd


def wilson(success: int, total: int, z: float = 1.96) -> Optional[List[float]]:
    if total <= 0:
        return None
    p = success / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


class Bucket:
    def __init__(self) -> None:
        self.n = 0
        self.first = 0
        self.stop = 0
        self.none = 0
        self.ambiguous = 0
        self.rs: List[float] = []

    def add(self, first: bool, stop: bool, ambiguous: bool = False) -> None:
        self.n += 1
        if ambiguous:
            self.ambiguous += 1
        if first:
            self.first += 1
        elif stop:
            self.stop += 1
        else:
            self.none += 1

    def as_dict(self) -> Dict[str, Any]:
        resolved = self.first + self.stop
        return {
            "samples": self.n,
            "midline_first": self.first,
            "stop_first": self.stop,
            "unresolved": self.none,
            "ambiguous": self.ambiguous,
            "midline_rate": round(self.first / self.n, 4) if self.n else None,
            "midline_rate_ci": wilson(self.first, self.n),
            "resolved_only_rate": round(self.first / resolved, 4) if resolved else None,
            "mean_r": round(sum(self.rs) / len(self.rs), 4) if self.rs else None,
        }


def support_event(frame: pd.DataFrame, i: int, box: Dict[str, Any],
                  tactics: Dict[str, Any], horizon: int) -> Optional[Dict[str, Any]]:
    support = box.get("support") or {}
    mid = tactics.get("midline") or {}
    s_low, s_high = support.get("low"), support.get("high")
    mid_price = mid.get("price")
    atr = box.get("atr")
    if None in (s_low, s_high, mid_price, atr) or atr <= 0:
        return None
    row = frame.iloc[i]
    if not (float(row["Low"]) <= s_high + atr * 0.15 and float(row["Close"]) >= s_low):
        return None
    stop = s_low - atr * 0.35
    end = min(len(frame), i + 1 + horizon)
    if i < 0 or horizon < 1 or i + horizon >= len(frame):
        return None
    status = {"first": False, "stop": False, "ambiguous": False}
    for j in range(i + 1, end):
        if not _valid_bar(frame.iloc[j]):
            return None
        opening = float(frame.iloc[j]["Open"])
        if opening >= mid_price:
            status["first"] = True
            break
        if opening <= stop:
            status["stop"] = True
            break
        hit_mid = float(frame.iloc[j]["High"]) >= mid_price
        hit_stop = float(frame.iloc[j]["Low"]) <= stop
        if hit_mid and hit_stop:
            status["ambiguous"] = True
            status["stop"] = True  # conservative same-bar resolution
            break
        if hit_mid:
            status["first"] = True
            break
        if hit_stop:
            status["stop"] = True
            break
    return {
        "outcome": "midline_first" if status["first"] else "stop_first" if status["stop"] else "unresolved",
        "ambiguous": status["ambiguous"],
        "midline_state": mid.get("state"),
        "midline_score": mid.get("score"),
        "speed_state": (tactics.get("speed") or {}).get("state"),
        "pressure": ((tactics.get("pressure") or {}).get("support") or {}).get("verdict"),
        "regime": (tactics.get("regime") or {}).get("key"),
        "volume_ratio": (tactics.get("volume") or {}).get("ratio"),
    }


def breakout_event(frame: pd.DataFrame, i: int, box: Dict[str, Any],
                   tactics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    resistance = box.get("resistance") or {}
    r_low, r_high = resistance.get("low"), resistance.get("high")
    atr = box.get("atr")
    if None in (r_low, r_high, atr) or atr <= 0:
        return None
    close = float(frame.iloc[i]["Close"])
    prev = float(frame.iloc[i - 1]["Close"]) if i else None
    if prev is None or not (prev <= r_high + atr * 0.10 and close > r_high + atr * 0.10):
        return None
    if i + 3 >= len(frame):
        return None
    if not all(_valid_bar(frame.iloc[j]) for j in range(i + 1, i + 4)):
        return None
    held = all(float(frame.iloc[j]["Close"]) >= r_low for j in range(i + 1, i + 4))
    return {
        "held": bool(held),
        "pressure": ((tactics.get("pressure") or {}).get("resistance") or {}).get("verdict"),
        "midline_state": (tactics.get("midline") or {}).get("state"),
        "speed_state": (tactics.get("speed") or {}).get("state"),
        "volume_ratio": (tactics.get("volume") or {}).get("ratio"),
        "regime": (tactics.get("regime") or {}).get("key"),
    }


def _valid_bar(row) -> bool:
    try:
        op, hi, lo, cl = (float(row[k]) for k in ("Open", "High", "Low", "Close"))
        return (all(math.isfinite(v) and v > 0 for v in (op, hi, lo, cl))
                and hi >= max(op, cl, lo) and lo <= min(op, cl))
    except (KeyError, TypeError, ValueError):
        return False


def plan_event(frame: pd.DataFrame, i: int, tactics: Dict[str, Any],
               horizon: int, *, cost_pct: float = 0.0,
               slippage_pct: float = 0.0) -> Optional[Dict[str, Any]]:
    """Next-open long execution; target percentages refer to original size.

    Stop gaps execute at the open. Profit limits receive no favourable gap
    improvement. Intrabar ambiguity resolves stop first. A post-T1 stop change
    takes effect on the NEXT bar because daily bars cannot establish its timing.
    """
    if (horizon < 1 or i < 0 or i + 1 >= len(frame)
            or any(not math.isfinite(v) or v < 0 or v >= 100
                   for v in (cost_pct, slippage_pct))):
        return None
    plan = tactics.get("plan") or {}
    if (plan.get("status") != "ready" or plan.get("side") != "long"
            or plan.get("tradeable") is False):
        return None
    try:
        entry = plan.get("entry") or {}
        stop = float((plan.get("stop") or {})["price"])
        entry_mid = float(entry["mid"])
        entry_cap = float(entry.get("high") or entry_mid)
        targets = sorted((float(t["price"]), float(t.get("pct", 100 / len(plan["targets"]))) / 100)
                         for t in plan["targets"])
        moved_stop = plan.get("stop_after_t1")
        moved_stop = float(moved_stop) if moved_stop is not None else None
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if (not targets or not all(math.isfinite(v) and v > 0
                               for v in (stop, entry_mid, entry_cap))
            or not math.isclose(sum(weight for _, weight in targets), 1.0, abs_tol=1e-6)
            or any(not math.isfinite(price) or not math.isfinite(weight) or weight <= 0
                   for price, weight in targets)
            or (moved_stop is not None and (not math.isfinite(moved_stop) or moved_stop <= 0))):
        return None
    if not _valid_bar(frame.iloc[i + 1]):
        return None
    raw_fill = float(frame.iloc[i + 1]["Open"])
    fill = raw_fill * (1 + slippage_pct / 100)
    if raw_fill <= stop or raw_fill > entry_cap or targets[0][0] <= fill:
        return None
    risk = fill - stop
    proceeds, remaining, next_target = 0.0, 1.0, 0
    end = min(len(frame), i + 1 + horizon)

    def result(reason, j):
        profit = proceeds - fill - fill * cost_pct / 100
        return {"r": round(profit / risk, 4), "exit": reason,
                "fill_price": round(fill, 6), "initial_risk": round(risk, 6),
                "net_return_pct": round(profit / fill * 100, 6),
                "entry_index": i + 1, "exit_index": j,
                "cost_round_trip_pct": cost_pct, "slippage_each_side_pct": slippage_pct,
                "midline_state": (tactics.get("midline") or {}).get("state")}

    for j in range(i + 1, end):
        row = frame.iloc[j]
        if not _valid_bar(row):
            return None
        opening, low, high = (float(row[k]) for k in ("Open", "Low", "High"))
        if opening <= stop:
            proceeds += remaining * opening * (1 - slippage_pct / 100)
            return result("stop_gap", j)
        # At the open, marketable profit limits precede later intrabar lows.
        while next_target < len(targets) and opening >= targets[next_target][0]:
            price, weight = targets[next_target]
            proceeds += weight * price * (1 - slippage_pct / 100)
            remaining -= weight
            next_target += 1
        if remaining <= 1e-8:
            return result("target2" if len(targets) > 1 else "target", j)
        if low <= stop:
            proceeds += remaining * stop * (1 - slippage_pct / 100)
            return result("stop", j)
        while next_target < len(targets) and high >= targets[next_target][0]:
            price, weight = targets[next_target]
            proceeds += weight * price * (1 - slippage_pct / 100)
            remaining -= weight
            next_target += 1
        if remaining <= 1e-8:
            return result("target2" if len(targets) > 1 else "target", j)
        if next_target > 0 and moved_stop is not None:
            stop = max(stop, moved_stop)
    if end - i - 1 < horizon:
        return None  # right-censored: no realized timeout exists yet
    proceeds += remaining * float(frame.iloc[end - 1]["Close"]) * (1 - slippage_pct / 100)
    return result("horizon", end - 1)


def audit_ticker(ticker: str, frame: pd.DataFrame, stride: int, *,
                 cost_pct: float = 0.0, slippage_pct: float = 0.0) -> Dict[str, Any]:
    out = {"support": [], "breakout": [], "plans": [], "states": Counter(), "regimes": Counter()}
    last_exit = -1
    for i in range(90, len(frame) - HORIZON - 2, stride):
        prefix = frame.iloc[: i + 1]
        dd = as_dd(prefix)
        box = analyze_box_range(dd)
        out["states"][box.get("state")] += 1
        if box.get("state") in (None, "no_box", "insufficient_data"):
            continue
        trend = analyze_trend_boxes(dd)
        tactics = analyze_box_tactics(dd, box, trend, current=float(prefix["Close"].iloc[-1]))
        out["regimes"][(tactics.get("regime") or {}).get("key")] += 1
        event = support_event(frame, i, box, tactics, HORIZON)
        if event:
            out["support"].append(event)
        event = breakout_event(frame, i, box, tactics)
        if event:
            out["breakout"].append(event)
        event = (plan_event(frame, i, tactics, HORIZON, cost_pct=cost_pct, slippage_pct=slippage_pct)
                 if i + 1 > last_exit else None)
        if event:
            event["ticker"] = ticker
            out["plans"].append(event)
            last_exit = event["exit_index"]
    return out


def rate_block(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    buckets: Dict[str, Bucket] = defaultdict(Bucket)
    for row in rows:
        buckets[str(row.get(key))].add(row["outcome"] == "midline_first",
                                        row["outcome"] == "stop_first", row["ambiguous"])
    return {k: v.as_dict() for k, v in sorted(buckets.items())}


def summarise(rows: Dict[str, List[Dict[str, Any]]], snapshots: int, tickers: int,
              states: Counter, regimes: Counter) -> Dict[str, Any]:
    support = rows["support"]
    all_bucket = Bucket()
    for row in support:
        all_bucket.add(row["outcome"] == "midline_first", row["outcome"] == "stop_first", row["ambiguous"])
    plans = rows["plans"]
    plan_rs = [p["r"] for p in plans]
    plan_bucket = Bucket()
    for p in plans:
        plan_bucket.rs.append(p["r"])
    breakout_bucket = defaultdict(lambda: [0, 0])
    for row in rows["breakout"]:
        breakout_bucket[str(row.get("pressure"))][0] += int(row["held"])
        breakout_bucket[str(row.get("pressure"))][1] += 1
    return {
        "snapshots": snapshots,
        "tickers": tickers,
        "state_counts": dict(states),
        "regime_counts": dict(regimes),
        "support_tests": all_bucket.as_dict(),
        "support_by_magnet": rate_block(support, "midline_state"),
        "support_by_speed": rate_block(support, "speed_state"),
        "support_by_pressure": rate_block(support, "pressure"),
        "support_by_regime": rate_block(support, "regime"),
        "breakout_hold": {
            k: {"held": v[0], "samples": v[1],
                "rate": round(v[0] / v[1], 4) if v[1] else None,
                "rate_ci": wilson(v[0], v[1])}
            for k, v in sorted(breakout_bucket.items())
        },
        "ready_plans": {
            "samples": len(plans),
            "mean_r": round(sum(plan_rs) / len(plan_rs), 4) if plan_rs else None,
            "win_rate": round(sum(1 for r in plan_rs if r > 0) / len(plan_rs), 4) if plan_rs else None,
            "win_rate_ci": wilson(sum(1 for r in plan_rs if r > 0), len(plan_rs)),
            "exit_counts": dict(Counter(p["exit"] for p in plans)),
            "by_magnet": {
                k: {
                    "samples": len(v),
                    "mean_r": round(sum(x["r"] for x in v) / len(v), 4),
                    "win_rate": round(sum(1 for x in v if x["r"] > 0) / len(v), 4),
                }
                for k, v in sorted({m: [p for p in plans if p.get("midline_state") == m]
                                    for m in {p.get("midline_state") for p in plans}}.items())
                if v
            },
        },
        "limitations": [
            "일봉 완료 창만 사용하며 신호 봉 종가 이후만 채점한다.",
            "실제 다음 시가와 지정 비용·슬리피지로 계산; 종목별 동시 포지션은 하나다.",
            "시가 갭을 먼저 처리하고 이후 같은 봉의 목표·손절 경합은 손절 우선이다.",
            "지지 도달률 분모에 기간 내 미도달을 포함; resolved_only_rate는 별도다.",
            "겹치는 지지 관측은 독립 표본이 아니므로 Wilson 구간은 기술적 참고다.",
            "표본 비율은 확률 보정이 아니며, 표본이 적은 칸은 신뢰구간이 넓다.",
            "영상은 분단위 차트를 다루고 이 감사는 일봉이므로 시간 규모가 다르다.",
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "backtests", "box_tactics_audit.json"))
    parser.add_argument("--offline", action="store_true", help="캐시만 사용(현재 기본 동작)")
    parser.add_argument("--cost-pct", type=float, default=0.0, help="왕복 비용(%%); 시장별 실제 세금·수수료로 지정")
    parser.add_argument("--slippage-pct", type=float, default=0.0, help="편도 슬리피지(%%)")
    args = parser.parse_args(argv)
    if args.stride < 1 or any(not math.isfinite(v) or not 0 <= v < 100 for v in (args.cost_pct, args.slippage_pct)):
        parser.error("stride must be positive and execution costs must be finite in [0, 100)")

    paths = sorted(glob.glob(os.path.join(CACHE_DIR, "*.csv")))
    wanted = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if wanted:
        keys = {t.replace(".", "_").replace("^", "_idx_") for t in wanted}
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in keys]
    if not paths:
        print("[audit] 캐시 없음: datasets/audit_cache")
        return 1

    rows: Dict[str, List[Dict[str, Any]]] = {"support": [], "breakout": [], "plans": []}
    states: Counter = Counter()
    regimes: Counter = Counter()
    snapshots = 0
    processed = 0
    for path in paths:
        frame = load_prices(path)
        if frame is None or len(frame) < 200:
            continue
        ticker = os.path.splitext(os.path.basename(path))[0]
        processed += 1
        print(f"[audit] {ticker} ({len(frame)} bars)", flush=True)
        result = audit_ticker(ticker, frame, args.stride,
                              cost_pct=args.cost_pct, slippage_pct=args.slippage_pct)
        snapshots += sum(result["states"].values())
        states.update(result["states"])
        regimes.update(result["regimes"])
        for key in rows:
            rows[key].extend(result[key])

    if not processed:
        print("[audit] 유효한 200봉 이상 캐시 없음")
        return 1
    summary = summarise(rows, snapshots, processed, states, regimes)
    summary.update({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stride": args.stride,
        "horizon_bars": HORIZON,
        "cache_dir": os.path.relpath(CACHE_DIR, REPO_ROOT),
        "module_version": VERSION,
        "audit_version": "2026-10-26.execution-v2",
        "execution": {"cost_round_trip_pct": args.cost_pct, "slippage_each_side_pct": args.slippage_pct,
                      "fill_basis": "next_open", "positions_per_ticker": 1,
                      "stop_move_timing": "next_bar", "pyramiding_simulated": False},
        "decision_basis": "raw_box_tactics_rules; excludes final service trend/quote/overall-risk gates",
        "source": "scripts/audit_box_tactics.py",
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in ("snapshots", "tickers", "support_tests",
                                              "support_by_magnet", "breakout_hold", "ready_plans")},
                     ensure_ascii=False, indent=2))
    print(f"[audit] saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
