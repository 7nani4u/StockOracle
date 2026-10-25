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
from market_briefing.box_tactics import analyze_box_tactics  # noqa: E402
from market_briefing.trend_boxes import analyze_trend_boxes  # noqa: E402

CACHE_DIR = os.path.join(REPO_ROOT, "datasets", "audit_cache")
HORIZON = 10


def load_prices(path: str) -> Optional[pd.DataFrame]:
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except (OSError, ValueError):
        return None
    if frame is None or frame.empty or "Close" not in frame.columns:
        return None
    frame = frame[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
    frame.index = pd.to_datetime(pd.Index([str(v)[:10] for v in frame.index]))
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
            "midline_rate": round(self.first / resolved, 4) if resolved else None,
            "midline_rate_ci": wilson(self.first, resolved),
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
    if i + 1 >= len(frame):
        return None
    status = {"first": False, "stop": False, "ambiguous": False}
    for j in range(i + 1, end):
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
    if not status["first"] and not status["stop"]:
        return None
    return {
        "outcome": "midline_first" if status["first"] else "stop_first",
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
    if i + 4 >= len(frame):
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


def plan_event(frame: pd.DataFrame, i: int, tactics: Dict[str, Any],
               horizon: int) -> Optional[Dict[str, Any]]:
    plan = tactics.get("plan") or {}
    if plan.get("status") != "ready" or plan.get("side") != "long":
        return None
    entry, stop = plan.get("entry") or {}, (plan.get("stop") or {}).get("price")
    targets = plan.get("targets") or []
    if stop is None or not targets or i + 1 >= len(frame):
        return None
    entry_mid = entry.get("mid")
    if entry_mid is None or entry_mid <= stop:
        return None
    nxt = frame.iloc[i + 1]
    fill = float(nxt["Open"])
    if fill <= stop or fill > float(entry.get("high") or entry_mid):
        return None  # no chase, no gap-through-stop fill
    risk = entry_mid - stop
    if risk <= 0:
        return None
    t1 = float(targets[0]["price"])
    t2 = float(targets[-1]["price"])
    end = min(len(frame), i + 1 + horizon)
    realised = 0.0
    remaining = 1.0
    for j in range(i + 1, end):
        low, high = float(frame.iloc[j]["Low"]), float(frame.iloc[j]["High"])
        if remaining > 0 and low <= stop:
            realised += remaining * ((stop - entry_mid) / risk)
            return {"r": round(realised, 4), "exit": "stop", "midline_state": (tactics.get("midline") or {}).get("state")}
        if remaining > 0.5 and high >= t1:
            realised += 0.5 * ((t1 - entry_mid) / risk)
            remaining -= 0.5
        if remaining > 0 and t2 > t1 and high >= t2:
            realised += remaining * ((t2 - entry_mid) / risk)
            return {"r": round(realised, 4), "exit": "target2", "midline_state": (tactics.get("midline") or {}).get("state")}
    last = float(frame.iloc[end - 1]["Close"])
    realised += remaining * ((last - entry_mid) / risk)
    return {"r": round(realised, 4), "exit": "horizon", "midline_state": (tactics.get("midline") or {}).get("state")}


def audit_ticker(ticker: str, frame: pd.DataFrame, stride: int) -> Dict[str, Any]:
    out = {"support": [], "breakout": [], "plans": [], "states": Counter(), "regimes": Counter()}
    atr = None
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
        event = plan_event(frame, i, tactics, HORIZON)
        if event:
            event["ticker"] = ticker
            out["plans"].append(event)
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
            "다음 거래일 시가 체결을 가정하고 수수료·세금·슬리피지는 0으로 둔다.",
            "같은 봉에서 목표와 손절이 겹치면 손절로 보수 처리한다.",
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
    args = parser.parse_args(argv)

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
    for path in paths:
        frame = load_prices(path)
        if frame is None or len(frame) < 200:
            continue
        ticker = os.path.splitext(os.path.basename(path))[0]
        print(f"[audit] {ticker} ({len(frame)} bars)", flush=True)
        result = audit_ticker(ticker, frame, max(1, args.stride))
        snapshots += sum(result["states"].values())
        states.update(result["states"])
        regimes.update(result["regimes"])
        for key in rows:
            rows[key].extend(result[key])

    summary = summarise(rows, snapshots, len(paths), states, regimes)
    summary.update({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stride": args.stride,
        "horizon_bars": HORIZON,
        "cache_dir": os.path.relpath(CACHE_DIR, REPO_ROOT),
        "module_version": (analyze_box_tactics(as_dd(load_prices(paths[0]).iloc[:120]), {"state": "insufficient_data"})
                           .get("version")),
        "source": "scripts/audit_box_tactics.py",
    })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in ("snapshots", "tickers", "support_tests",
                                              "support_by_magnet", "breakout_hold", "ready_plans")},
                     ensure_ascii=False, indent=2))
    print(f"[audit] saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
