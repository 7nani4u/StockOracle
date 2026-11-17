"""audit_intraday_volume.py — 장중(진행 중 일봉) 거래량이 예측 탭을 흔드는지, 넣을 거래량 값은 무엇이 나은지 감사.

무엇을 보나
  route('/api/stock') 전체를 오프라인으로 다시 실행한다. 네트워크는 전송 계층(requests·feedparser·미국 실시간
  시세 수집기)에서 끊고, 가격 이력만 datasets/audit_cache 의 일봉 CSV 를 가짜 yf.Ticker 로 공급해 실제
  fetch_stock_data → add_indicators → route 경로를 그대로 탄다. 마지막 봉을 '장중 진행 중'으로 표시하고
  그 봉의 거래량을 최종값의 f 배(장 30%·60% 경과의 대용)로 줄인 응답을, 같은 날 마감 후 응답(진행 중 아님,
  최종 거래량)과 비교한다.
    · 시각 불변성: f=0.3 과 f=0.6 응답의 수치 차이(0 이어야 함)
    · 마감 후 근접도: 장중 응답과 마감 후 응답의 평균 절대 차이(작을수록 장중 화면이 마감 후 화면에 가깝다)
  추정값 후보: old(누적값 그대로, 예전 동작) · carry(직전 확정 봉, 운영) · neutral(직전 20봉 평균) ·
  lb_neutral / lb_carry(지금까지 누적값을 하한으로 쓰는 max 변형).

범위와 한계
  · 실제 장중 거래량 곡선이 아니라 '최종값의 f 배'로 흉내 낸다(장중 누적은 U자형이라 실제 30% 시각의 누적
    비율은 이보다 조금 높다). 가격·고저는 마감 값 그대로라 장중 고저 변화는 따로 봐야 한다(--range-frac).
  · 수급·뉴스·실시간 시세·ML·실적 일정 등 네트워크 입력은 빠진다(모두 실패 경로로 처리). 같은 입력을
    old·carry 등에 똑같이 주므로 비교에는 영향이 없다.
  · 학습 로그는 건드리지 않는다(STOCKORACLE_PREDICTION_LOG 를 임시 경로로 고정).

사용 예
    python scripts/audit_intraday_volume.py run OUT.json --asof 2026-09-15 --frac 0.3 --estimator carry
    python scripts/audit_intraday_volume.py grid --out-dir /tmp/iv --workers 8      # 7개 날짜 × 6개 구성
    python scripts/audit_intraday_volume.py analyze --out-dir /tmp/iv --json docs/backtests/intraday_volume_audit_20261117.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import math
import os
import statistics as st
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "datasets", "audit_cache")
DATES = ["2025-03-14", "2025-06-13", "2025-09-15", "2025-12-12", "2026-03-13", "2026-06-15", "2026-09-15"]
# (이름, f, 진행 중 여부, 추정값)
CONFIGS = [
    ("eod", 1.0, False, "carry"),
    ("old030", 0.3, True, "old"), ("old060", 0.6, True, "old"),
    ("carry030", 0.3, True, "carry"), ("carry060", 0.6, True, "carry"),
    ("neutral030", 0.3, True, "neutral"), ("neutral060", 0.6, True, "neutral"),
    ("lb_neutral030", 0.3, True, "lb_neutral"), ("lb_neutral060", 0.6, True, "lb_neutral"),
    ("lb_carry030", 0.3, True, "lb_carry"), ("lb_carry060", 0.6, True, "lb_carry"),
]
METRICS = {
    "breakout_prob_pp": (["risk_scenarios", "breakout_probability_pct"], "abs"),
    "downside_risk_pts": (["buy_price", "downside_risk", "score"], "abs"),
    "risk_card_downside_pts": (["risk_scenarios", "downside_risk_score"], "abs"),
    "cons_TP1_prob_pp": (["risk_scenarios", "conservative", "tp_levels", 0, "prob_pct"], "abs"),
    "bal_TP1_prob_pp": (["risk_scenarios", "balanced", "tp_levels", 0, "prob_pct"], "abs"),
    "agg_TP1_prob_pp": (["risk_scenarios", "aggressive", "tp_levels", 0, "prob_pct"], "abs"),
    "bal_EV_pp": (["risk_scenarios", "balanced", "expected_value_pct"], "abs"),
    "scenario_up_pp": (["prediction_outlook", "forecast", "scenario_probabilities", "up"], "abs"),
    "scenario_side_pp": (["prediction_outlook", "forecast", "scenario_probabilities", "sideways"], "abs"),
    "forecast_base_pct": (["prediction_outlook", "forecast", "base_price"], "rel"),
    "A_step1_price_pct": (["buy_price", "aggressive_bands", 0, "steps", 0, "price"], "rel"),
    "B_step1_price_pct": (["buy_price", "recommended_bands", 0, "steps", 0, "price"], "rel"),
    "B_range_lo_pct": (["buy_price", "recommended_bands", 0, "range", 0], "rel"),
    "B_step_reach_pp": (["buy_price", "recommended_bands", 0, "steps", 0, "reach_probability_pct"], "abs"),
}


# ── 오프라인 route 실행 ───────────────────────────────────────────────────────

def _install_offline(asof: str, frac: float, range_frac: float, in_progress: bool, estimator: str):
    """네트워크를 끊고 캐시 일봉을 공급하는 가짜 yf.Ticker 를 심은 뒤 api.index 모듈을 돌려준다."""
    os.environ.setdefault("STOCKORACLE_PREDICTION_LOG",
                          os.path.join(tempfile.gettempdir(), "stockoracle_intraday_audit_learning.jsonl"))
    os.environ["STOCKORACLE_INPROGRESS_VOLUME_SETTLE"] = "0" if estimator == "old" else "1"
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import pandas as pd
    import requests

    def _no_net(*_a, **_k):
        raise requests.ConnectionError("offline audit: network disabled")

    requests.get = requests.post = requests.put = _no_net
    requests.Session.request = _no_net
    try:
        import urllib.request as _ur
        _ur.urlopen = _no_net
    except Exception:
        pass
    from api import index as ix

    frames: dict = {}

    def _frame(sym):
        if sym not in frames:
            path = os.path.join(CACHE_DIR, sym.replace(".", "_") + ".csv")
            df = pd.read_csv(path)
            tz = "Asia/Seoul" if sym.endswith((".KS", ".KQ")) else "America/New_York"
            df.index = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(tz)
            frames[sym] = df.drop(columns=["Date"])
        return frames[sym]

    class FakeTicker:
        def __init__(self, sym, *a, **k):
            self.ticker = sym
            self.info = {}
            self.calendar = None
            self.news = []
            self.fast_info = type("FI", (), {})()

        def history(self, period="1y", interval="1d", auto_adjust=True, **_):
            if interval != "1d":
                return pd.DataFrame()
            try:
                df = _frame(self.ticker)
            except FileNotFoundError:
                return pd.DataFrame()
            df = df[df.index.strftime("%Y-%m-%d") <= asof]
            n = {"1mo": 23, "3mo": 64, "6mo": 127, "1y": 253, "2y": 505, "5y": 1262, "10y": 2600}.get(period, 505)
            df = df.tail(n).copy()
            valid = df.index[df["Close"].notna()]
            if len(valid):
                i = valid[-1]
                if frac != 1.0:
                    df.loc[i, "Volume"] = float(df.loc[i, "Volume"]) * frac
                if range_frac != 1.0:
                    o, c, h, l = (float(df.loc[i, k]) for k in ("Open", "Close", "High", "Low"))
                    df.loc[i, "High"] = max(o, c) + range_frac * (h - max(o, c))
                    df.loc[i, "Low"] = min(o, c) - range_frac * (min(o, c) - l)
            return df

    ix.yf.Ticker = FakeTicker
    ix.yf.download = lambda *a, **k: pd.DataFrame()
    ix.yf.Search = lambda *a, **k: type("S", (), {"quotes": []})()
    try:
        import feedparser
        feedparser.parse = lambda *a, **k: type("F", (), {"entries": []})()
    except ImportError:
        pass
    ix.resolve_ticker = lambda raw: (str(raw).upper(), "KRX" if str(raw).upper().endswith((".KS", ".KQ")) else "US",
                                     str(raw).upper())
    ix.get_krx_security_status = lambda code: {"analysis_blocked": False, "listed_date": "", "status_labels": [],
                                               "warnings": []}
    ix._get_us_price_fetcher = lambda: None
    if in_progress:
        ix._bar_in_progress = lambda *a, **k: True

    if estimator in ("neutral", "lb_neutral", "lb_carry"):
        import market_briefing.session_bars as sb
        production = sb.settle_in_progress_volume

        def _variant(dd, lookback=20):
            out, meta = production(dd, lookback)
            if not meta.get("applied"):
                return out, meta
            vols, closes = list(dd["Volume"]), list(dd["Close"])
            f = sb._finite
            live, prev = f(vols[-1]) or 0.0, f(vols[-2]) or 0.0
            window = [v for v in (f(x) for x in vols[-lookback - 1:-1]) if v is not None and v >= 0]
            neutral = float(round(sum(window) / len(window)))
            target = {"neutral": neutral, "lb_neutral": max(live, neutral), "lb_carry": max(live, prev)}[estimator]
            new_vols = vols[:-1] + [target]
            out = dict(out, Volume=new_vols)
            obv = list(dd.get("OBV") or [])
            if len(obv) == len(vols) and f(obv[-2]) is not None and f(closes[-1]) is not None and f(closes[-2]) is not None:
                c1, c0 = f(closes[-1]), f(closes[-2])
                obv[-1] = f(obv[-2]) + target * ((c1 > c0) - (c1 < c0))
                out["OBV"] = obv
            bp = list(dd.get("BUY_PRESSURE") or [])
            if len(bp) == len(vols):
                up = tot = 0.0
                ok = True
                for i in range(len(vols) - 14, len(vols)):
                    v = f(new_vols[i])
                    if v is None or v <= 0:
                        ok = False
                        break
                    tot += v
                    a, b = f(closes[i]), f(closes[i - 1])
                    if a is not None and b is not None and a >= b:
                        up += v
                bp[-1] = up / tot * 100 if ok and tot > 0 else None
                out["BUY_PRESSURE"] = bp
            return out, dict(meta, analysis_volume=target, estimator=estimator)

        sb.settle_in_progress_volume = _variant
    return ix


def _clean(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return float(f"{obj:.10g}") if math.isfinite(obj) else str(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def cmd_run(args) -> int:
    ix = _install_offline(args.asof, args.frac, args.range_frac, not args.completed, args.estimator)
    tickers = args.tickers.split(",") if args.tickers else sorted(
        f[:-4].replace("_", ".") for f in os.listdir(CACHE_DIR) if f.endswith(".csv"))
    results, failures = {}, []
    for ticker in tickers:
        ix._CACHE.clear()
        try:
            resp = ix.route("/api/stock", {"ticker": ticker})
        except Exception as exc:  # 안정성: route 는 예외를 던지면 안 된다
            resp = {"__exception__": f"{type(exc).__name__}: {exc}"}
        if "__exception__" in resp or "error" in resp:
            failures.append(ticker)
        results[ticker] = _clean(resp)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"asof": args.asof, "frac": args.frac, "range_frac": args.range_frac,
                   "estimator": args.estimator, "in_progress": not args.completed, "results": results},
                  fh, ensure_ascii=False)
    print(f"[intraday-audit] {args.asof} {args.estimator} f={args.frac} tickers={len(results)} failures={failures}")
    return 1 if failures else 0


def cmd_grid(args) -> int:
    os.makedirs(args.out_dir, exist_ok=True)

    def job(date, cfg):
        name, frac, in_progress, est = cfg
        out = os.path.join(args.out_dir, f"{name}_{date}.json")
        if os.path.exists(out):
            return out, 0
        cmd = [sys.executable, os.path.abspath(__file__), "run", out, "--asof", date, "--frac", str(frac),
               "--estimator", est] + ([] if in_progress else ["--completed"])
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        with open(out + ".log", "w", encoding="utf-8") as log:
            return out, subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

    jobs = [(d, c) for d in (args.dates.split(",") if args.dates else DATES) for c in CONFIGS]
    started = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for out, rc in pool.map(lambda j: job(*j), jobs):
            print(rc, os.path.basename(out), flush=True)
    print(f"[intraday-audit] grid done in {time.time() - started:.0f}s")
    return 0


def _get(resp, path):
    cur = resp
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and isinstance(key, int):
            cur = cur[key] if len(cur) > key else None
        else:
            return None
        if cur is None:
            return None
    try:
        value = float(cur)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _dist(a, b, kind):
    if a is None or b is None:
        return None
    if kind == "rel":
        return abs(a - b) / abs(b) * 100 if b else None
    return abs(a - b)


def cmd_analyze(args) -> int:
    def load(name, date):
        path = os.path.join(args.out_dir, f"{name}_{date}.json")
        return json.load(open(path, encoding="utf-8"))["results"] if os.path.exists(path) else None

    estimators = ["old", "carry", "neutral", "lb_neutral", "lb_carry"]
    near = {e: {m: {"30": [], "60": []} for m in METRICS} for e in estimators}
    drift = {e: {m: [] for m in METRICS} for e in estimators}
    dates_used = []
    for date in DATES:
        eod = load("eod", date)
        runs = {e: (load(f"{e}030", date), load(f"{e}060", date)) for e in estimators}
        if eod is None or any(a is None or b is None for a, b in runs.values()):
            continue
        dates_used.append(date)
        for e, (r30, r60) in runs.items():
            for ticker, base in eod.items():
                if ticker not in r30 or ticker not in r60:
                    continue
                for metric, (path, kind) in METRICS.items():
                    ev, v30, v60 = _get(base, path), _get(r30[ticker], path), _get(r60[ticker], path)
                    d30, d60, dd = _dist(v30, ev, kind), _dist(v60, ev, kind), _dist(v30, v60, kind)
                    if d30 is None or d60 is None:
                        continue
                    near[e][metric]["30"].append(d30)
                    near[e][metric]["60"].append(d60)
                    if dd is not None:
                        drift[e][metric].append(dd)
    if not dates_used:
        print("[intraday-audit] 분석할 날짜가 없습니다 — 먼저 grid 를 실행하세요")
        return 1
    summary = {
        "dates": dates_used,
        "samples_per_cell": len(near["old"][next(iter(METRICS))]["30"]),
        "metric_definitions": {m: {"path": p, "kind": k} for m, (p, k) in METRICS.items()},
        "mean_abs_gap_to_end_of_day": {
            e: {m: {"session_30pct": round(st.mean(v["30"]), 4), "session_60pct": round(st.mean(v["60"]), 4)}
                for m, v in near[e].items() if v["30"]} for e in estimators},
        "mean_abs_drift_30_vs_60": {
            e: {m: round(st.mean(v), 4) for m, v in drift[e].items() if v} for e in estimators},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:8000])
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"\n요약 저장: {args.json}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="한 날짜·한 구성의 52종목 오프라인 route 응답")
    run.add_argument("out")
    run.add_argument("--asof", required=True)
    run.add_argument("--frac", type=float, default=1.0, help="마지막 봉 거래량 = 최종값 × frac")
    run.add_argument("--range-frac", type=float, default=1.0, help="마지막 봉 고저 범위 = 몸통 + frac × 꼬리")
    run.add_argument("--estimator", default="carry", choices=["old", "carry", "neutral", "lb_neutral", "lb_carry"])
    run.add_argument("--completed", action="store_true", help="마지막 봉을 확정 봉으로 둔다(마감 후 기준)")
    run.add_argument("--tickers", default="")
    grid = sub.add_parser("grid", help="날짜 × 구성 전체를 병렬 실행")
    grid.add_argument("--out-dir", default=os.path.join(tempfile.gettempdir(), "stockoracle_intraday_audit"))
    grid.add_argument("--workers", type=int, default=6)
    grid.add_argument("--dates", default="")
    ana = sub.add_parser("analyze", help="마감 후 근접도·시각 불변성 요약")
    ana.add_argument("--out-dir", default=os.path.join(tempfile.gettempdir(), "stockoracle_intraday_audit"))
    ana.add_argument("--json", default="")
    args = parser.parse_args(argv)
    return {"run": cmd_run, "grid": cmd_grid, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
