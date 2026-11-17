#!/usr/bin/env python
"""차트 패턴 엔진 인과(causal) 사건 연구 — 가이드 도표의 주장과 실제 성과를 대조한다.

왜 만들었나
-----------
``models/technique_prune.json`` 의 패턴 기법은 전부 INSUFFICIENT 다(1년·50종목, 검증 구간 표본 0~10건).
그 연구는 5봉마다 표본을 뽑고 *'confirmed' 상태인 첫 패턴*을 집었기 때문에 돌파 몇 주 뒤의 낡은 패턴도 신호로
섞였다. 이 도구는 (1) 패턴이 처음 관측된 봉을 사건으로 삼고 (2) 10년 이력·300여 종목으로 표본을 키우고
(3) 같은 손절·목표 폭을 가진 무작위 시점과 비교해 '패턴 타이밍' 자체의 효과만 본다.

사전 고정한 규칙(결과를 보기 전에 정함)
--------------------------------------
* 신호 봉 t: PatternEngine 을 t 종가까지의 250봉 창(실서비스 1y 일봉과 같은 길이)에 돌려 얻은 첫 'confirmed' 관측.
  진입은 t+1 시가, 청산은 H 거래일 뒤 종가(H=5·10·20, 1차 지표는 H=10).
* 개발 구간: 진입일 2021-12-31 까지. 검증 구간: 2022-01-01 부터. 채택 판단은 개발 구간에서 하고 검증에서 확인한다.
* 1차 지표: 날짜별 교차단면 평균을 뺀 초과수익률(방향 보정: 하락 패턴은 부호 반전). 표준오차는 (시장, 연월) 군집.
* 구조 거래: 손절=패턴 무효화가, 목표=패턴 목표가, 최대 20거래일, 같은 봉 손절 우선, 갭 이탈은 시가 체결.
  비용: KRX 왕복 0.2%+슬리피지 편도 0.1%, US 0.1%+0.1% (기존 prune 감사와 동일).
* 대조군: 같은 종목·같은 구간의 무작위 20개 시점에 같은 손절·목표 폭(%)을 적용.
* 채택 기준(패턴 그룹별): 개발·검증 모두 평균 초과수익 > 0, 풀링 군집 t ≥ 2.5(13개 그룹 다중검정 보정), 표본 ≥ 100.

한계: 유니버스는 2026-11 기준 현재 종목이라 생존 편향이 있고, 이벤트는 시점·종목별로 군집한다.

사용법::

    python scripts/audit_chart_patterns.py fetch            # 유니버스 구성·10년 일봉 수집
    python scripts/audit_chart_patterns.py extract          # 패턴 사건 추출(다중 프로세스)
    python scripts/audit_chart_patterns.py analyze          # 통계표 JSON 작성
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CACHE = os.path.join(ROOT, "datasets", "pattern_study_cache")
FETCH_START = "2014-06-01"
WINDOW = 250
WARM = 250
SPLIT = "2022-01-01"
HORIZONS = (5, 10, 20)
PRIMARY_H = 10
MAX_HOLD = 20
CONTROLS = 20
WINSOR = 0.5            # 수익률은 ±50% 로 자른다(2017-18 QUBT 가격 이력 오류 같은 자료 결함·극단 종목 방어)
MIN_RISK = 0.005        # 손절 폭 0.5% 미만은 비용(0.3~0.4%)과 같은 크기라 R 배수가 폭주한다 → 구조 거래에서 제외
AGES = (0, 3, 10, 20)
COST = {"KRX": 0.002 + 0.002, "US": 0.001 + 0.002}      # 왕복 비용 + 편도 0.1% 슬리피지 ×2
CLAIMED = {   # 첨부 이미지(가이드)가 말한 '신뢰도'와 대응 패턴
    "double_top:bearish": {"claim_pct": 100, "label": "쌍봉"},
    "bearish_consolidation:bearish": {"claim_pct": 80, "label": "하락깃발"},
    "bullish_consolidation:bullish": {"claim_pct": 80, "label": "상승깃발"},
    "inverse_head_shoulders:bullish": {"claim_pct": 65, "label": "역삼중(역헤드앤숄더)"},
    "ascending_triangle:bullish": {"claim_pct": 100, "label": "상승삼각형"},
}
RANKING_BUY = ["inverse_head_shoulders:bullish", "double_bottom:bullish", "bullish_consolidation:bullish",
               "ascending_triangle:bullish", "triple_bottom:bullish"]
RANKING_SELL = ["double_top:bearish", "bearish_consolidation:bearish", "triple_top:bearish"]


# ───────────────────────── 유니버스·수집 ─────────────────────────

def build_universe() -> dict:
    """앱의 스캔·추천·스마트스코어 유니버스와 기존 연구 캐시를 합친다(생존 편향 있음)."""
    kr: set[str] = set()
    us: set[str] = set()
    src = open(os.path.join(ROOT, "api", "index.py"), encoding="utf-8", newline="").read()

    def block(name: str, closer: str) -> str:
        i = src.find(name + " = [")
        j = src.find(closer, i)
        return src[i:j]

    kr.update(re.findall(r'\("(\d{6}\.K[SQ])"', block("_SCAN_KRX_UNIVERSE", "\n            ]")))
    us.update(re.findall(r'\("([A-Z][A-Z0-9.\-]{0,5})",\s*"', block("_SCAN_US_UNIVERSE", "\n            ]")))
    us.update(re.findall(r'"([A-Z][A-Z0-9.\-]*)"', block("_US_RECO_UNIVERSE", "]")))
    smart = json.load(open(os.path.join(ROOT, "models", "smart_score_universe.json"), encoding="utf-8"))
    for market, node in smart["markets"].items():
        for rec in node["records"]:
            (kr if market == "KRX" else us).add(rec["ticker"])
    research = os.path.join(ROOT, "datasets", "technique_research_cache")
    if os.path.isdir(research):
        for name in os.listdir(research):
            stem = name[:-4] if name.endswith(".csv") else name
            if re.fullmatch(r"\d{6}_K[SQ]", stem):
                kr.add(stem.replace("_", "."))
    kr = {t for t in kr if re.fullmatch(r"\d{6}\.K[SQ]", t)}
    us = {t.replace(".", "-") if t.count(".") else t for t in us}
    return {"KRX": sorted(kr), "US": sorted(us)}


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE, ticker.replace("^", "_") + ".csv")


def fetch(limit: int | None = None) -> None:
    import yfinance as yf
    os.makedirs(CACHE, exist_ok=True)
    uni = build_universe()
    json.dump(uni, open(os.path.join(CACHE, "_universe.json"), "w", encoding="utf-8"), indent=1)
    tickers = [t for market in uni.values() for t in market] + ["^KS11", "SPY"]
    if limit:
        tickers = tickers[:limit]
    todo = [t for t in tickers if not os.path.exists(_cache_path(t))]
    print(f"universe KRX={len(uni['KRX'])} US={len(uni['US'])}; to download {len(todo)}", flush=True)
    failed = []
    for i in range(0, len(todo), 25):
        chunk = todo[i:i + 25]
        try:
            raw = yf.download(chunk, start=FETCH_START, interval="1d", auto_adjust=True,
                              group_by="ticker", progress=False, threads=False)
        except Exception as exc:  # 네트워크 오류는 종목 단위로 재시도
            print("chunk failed", chunk[:2], exc, flush=True)
            failed.extend(chunk)
            continue
        for t in chunk:
            try:
                df = raw[t] if len(chunk) > 1 else raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if len(df) < 300:
                    failed.append(t)
                    continue
                df = df.copy()
                df.index = pd.to_datetime(df.index).tz_localize(None).strftime("%Y-%m-%d")
                df.index.name = "Date"
                df.to_csv(_cache_path(t))
            except Exception:
                failed.append(t)
        print(f"  {min(i + 25, len(todo))}/{len(todo)} downloaded (failed so far {len(failed)})", flush=True)
    print("failed:", failed)


def load_frame(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df[(df.Close > 0) & (df.Open > 0) & (df.High >= df.Low) & (df.Volume >= 0)]
    df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    return df if len(df) >= WARM + 60 else None


# ───────────────────────── 사건 추출 ─────────────────────────

def _confirmation_without_volume_gate(closes, volumes, neckline, *, direction, after_index, atr, cfg):
    """엔진의 ``_confirmation`` 에서 '거래량 1.0배 미만이면 확정 보류' 한 줄만 뺀 변형(실험용)."""
    from market_briefing import pattern_engine as pe
    needed = int(cfg["breakout_confirm_bars"])
    min_atr = float(cfg["min_breakout_atr"])
    for i in range(max(after_index + 1, needed - 1), len(closes)):
        held = True
        for j in range(i - needed + 1, i + 1):
            line = pe._line_value(int(neckline["start_index"]), float(neckline["start_price"]),
                                  int(neckline["end_index"]), float(neckline["end_price"]), j)
            buffer = (atr[j] if j < len(atr) else 0.0) * min_atr
            if direction == "bullish" and not (closes[j] > line + buffer):
                held = False
            if direction == "bearish" and not (closes[j] < line - buffer):
                held = False
        if held:
            return {"confirmed": True, "index": i, "volume_ratio": pe._volume_ratio(volumes, i, int(cfg["volume_period"]))}
    return {"confirmed": False, "index": None, "volume_ratio": None}


def _extract_worker(args):
    ticker, market, dates, o, h, l, c, v, variant = args
    from market_briefing import pattern_engine as pe
    if variant == "novol":
        pe._confirmation = _confirmation_without_volume_gate
    n = len(c)
    inst: dict = {}
    options = pe.PatternEngineOptions(timeframe="1D", tolerance_mode="hybrid", include_forming=False)
    for t in range(WARM, n - 1):
        lo = max(0, t - WINDOW + 1)
        eng = pe.PatternEngine(o[lo:t + 1], h[lo:t + 1], l[lo:t + 1], c[lo:t + 1], v[lo:t + 1], options=options)
        for p in eng.detect():
            if p.get("pattern_status") != "confirmed":
                continue
            nl = p.get("neckline") or {}
            bo = nl.get("breakout_index")
            if bo is None:
                continue
            abs_bo = lo + int(bo)
            key = (p["id"], p["direction_code"], abs_bo)
            rec = inst.get(key)
            if rec is None:
                line_at_bo = pe._line_value(int(nl["start_index"]), float(nl["start_price"]),
                                            int(nl["end_index"]), float(nl["end_price"]), int(bo))
                pts = p.get("points") or []
                rec = inst[key] = {
                    "ticker": ticker, "market": market, "t": t, "date": str(dates[t]), "id": p["id"],
                    "dir": p["direction_code"], "family": p.get("family"), "abs_bo": abs_bo,
                    "age0": t - abs_bo, "completion": float(p.get("completion_score") or 0.0),
                    "vol_component": (p.get("completion_components") or {}).get("volume"),
                    "target": p.get("pattern_target_price"), "invalidation": p.get("invalidation_price"),
                    "neck_at_bo": float(line_at_bo), "n_points": len(pts),
                    "formation_bars": int(p.get("end_index", 0)) - int(p.get("start_index", 0)),
                    "seen": [],
                }
            rec["seen"].append(t)
    return list(inst.values())


def extract(variant: str = "baseline", limit: int | None = None, workers: int | None = None) -> None:
    uni = json.load(open(os.path.join(CACHE, "_universe.json"), encoding="utf-8"))
    tasks = []
    for market in ("KRX", "US"):
        for ticker in uni[market]:
            df = load_frame(ticker)
            if df is None:
                continue
            tasks.append((ticker, market, df.Date.tolist(), df.Open.to_numpy(float), df.High.to_numpy(float),
                          df.Low.to_numpy(float), df.Close.to_numpy(float), df.Volume.to_numpy(float), variant))
    if limit:
        tasks = tasks[:limit]
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    print(f"extract[{variant}] {len(tasks)} symbols, {workers} workers", flush=True)
    out, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, rows in enumerate(pool.map(_extract_worker, tasks, chunksize=1), 1):
            out.extend(rows)
            if k % 20 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} symbols, {len(out)} instances, {time.time() - t0:.0f}s", flush=True)
    path = os.path.join(CACHE, f"_instances_{variant}.pkl")
    pickle.dump(out, open(path, "wb"))
    print("saved", path, len(out), flush=True)


# ───────────────────────── 성과 계산 ─────────────────────────

def simulate(o, h, l, c, e: int, side: int, stop: float, target: float | None, max_hold: int = MAX_HOLD):
    """진입 e(시가). side=+1 롱, -1 숏. 같은 봉에서는 손절 우선, 시가 갭은 시가 체결. (청산가, 사유, 보유봉) 반환."""
    n = len(c)
    last = min(n - 1, e + max_hold - 1)
    for j in range(e, last + 1):
        if side > 0:
            if o[j] <= stop:
                return o[j], "stop", j - e + 1
            if l[j] <= stop:
                return stop, "stop", j - e + 1
            if target is not None:
                if o[j] >= target:
                    return o[j], "target", j - e + 1
                if h[j] >= target:
                    return target, "target", j - e + 1
        else:
            if o[j] >= stop:
                return o[j], "stop", j - e + 1
            if h[j] >= stop:
                return stop, "stop", j - e + 1
            if target is not None:
                if o[j] <= target:
                    return o[j], "target", j - e + 1
                if l[j] <= target:
                    return target, "target", j - e + 1
    return c[last], "time", last - e + 1


def cluster_stats(values, clusters, scale: float = 1.0) -> dict:
    x = np.asarray(values, dtype=float)
    ok = np.isfinite(x)
    x = x[ok]
    cl = np.asarray(clusters)[ok]
    n = len(x)
    if n < 2:
        return {"n": int(n), "mean": None, "se": None, "t": None, "median": None, "pos_rate": None}
    m = float(x.mean())
    dev = pd.Series(x - m).groupby(cl).sum()
    se = math.sqrt(float((dev ** 2).sum())) / n
    return {"n": int(n), "mean": round(m * scale, 4), "se": round(se * scale, 4),
            "t": round(m / se, 2) if se > 0 else None, "median": round(float(np.median(x)) * scale, 4),
            "pos_rate": round(float((x > 0).mean()), 4)}


def prepare():
    """유니버스·일봉·날짜별 교차단면 평균 수익률(시장별)을 읽는다."""
    uni = json.load(open(os.path.join(CACHE, "_universe.json"), encoding="utf-8"))
    frames = {t: load_frame(t) for m in uni.values() for t in m}
    frames = {t: f for t, f in frames.items() if f is not None}
    arr = {t: tuple(f[k].to_numpy(float) for k in ("Open", "High", "Low", "Close")) + (f.Date.tolist(),)
           for t, f in frames.items()}
    xs = {}
    for market in ("KRX", "US"):
        opens = pd.DataFrame({t: frames[t].set_index("Date").Open for t in uni[market] if t in frames})
        closes = pd.DataFrame({t: frames[t].set_index("Date").Close for t in uni[market] if t in frames})
        opens, closes = opens.sort_index(), closes.sort_index()
        per_h = {}
        for hh in HORIZONS:
            r = closes.shift(-(hh - 1)) / opens - 1.0
            cnt = r.notna().sum(axis=1)
            per_h[hh] = r.mean(axis=1).where(cnt >= 20)
        xs[market] = per_h
    return uni, frames, arr, xs


def analyze(variant: str = "baseline", out_path: str | None = None) -> dict:
    insts = pickle.load(open(os.path.join(CACHE, f"_instances_{variant}.pkl"), "rb"))
    uni, frames, arr, xs = prepare()

    rows = []
    rng_seed = zlib.crc32(variant.encode())
    for k, ins in enumerate(insts):
        o, h, l, c, dates = arr[ins["ticker"]]
        n = len(c)
        side = 1 if ins["dir"] == "bullish" else -1
        seen = set(ins["seen"])
        atr_t = float(np.mean(np.maximum(h[ins["t"] - 13:ins["t"] + 1] - l[ins["t"] - 13:ins["t"] + 1],
                                         np.abs(h[ins["t"] - 13:ins["t"] + 1] - c[ins["t"] - 14:ins["t"]]))))
        for age in AGES:
            ta = ins["t"] + age
            if ta not in seen:
                continue
            e = ta + 1
            if e >= n:
                continue
            row = {"i": k, "ticker": ins["ticker"], "market": ins["market"], "id": ins["id"], "dir": ins["dir"],
                   "group": f"{ins['id']}:{ins['dir']}", "age": age, "date": dates[ta],
                   "split": "dev" if dates[e] < SPLIT else "val", "completion": ins["completion"],
                   "age0": ins["age0"], "vol_component": ins["vol_component"], "formation_bars": ins["formation_bars"]}
            for hh in HORIZONS:
                if e + hh - 1 < n:
                    gross = float(np.clip(c[e + hh - 1] / o[e] - 1.0, -WINSOR, WINSOR))
                    base = xs[ins["market"]][hh].get(dates[e], np.nan)
                    row[f"r{hh}"] = gross
                    row[f"x{hh}"] = gross - base if np.isfinite(base) else np.nan
            if age == 0:
                stop, tgt = ins["invalidation"], ins["target"]
                entry = o[e]
                ok = stop is not None and np.isfinite(stop) and ((stop < entry) if side > 0 else (stop > entry))
                if ok and abs(entry - stop) / entry < MIN_RISK:
                    ok = False
                if ok:
                    tgt_ok = tgt is not None and np.isfinite(tgt) and ((tgt > entry) if side > 0 else (tgt < entry))
                    tgt_v = float(tgt) if tgt_ok else None
                    cost = COST[ins["market"]]
                    risk = abs(entry - stop) / entry
                    px, why, held = simulate(o, h, l, c, e, side, float(stop), tgt_v)
                    net = float(np.clip(side * (px / entry - 1.0) - cost, -WINSOR, WINSOR))
                    row.update({"has_geo": True, "stop_pct": risk, "tgt_pct": (abs(tgt_v - entry) / entry if tgt_v else None),
                                "R": net / risk, "net": net, "exit": why, "held": held})
                    if tgt_v is not None:
                        rng = np.random.default_rng(rng_seed + k)
                        pool = [j for j in range(WARM, n - 1 - MAX_HOLD)
                                if abs(j - ins["t"]) > 20 and ((dates[j + 1] < SPLIT) == (dates[e] < SPLIT))]
                        if len(pool) >= CONTROLS:
                            picks = rng.choice(pool, size=CONTROLS, replace=False)
                            ctrl = []
                            for j in picks:
                                ce = j + 1
                                ent = o[ce]
                                tp = abs(tgt_v - entry) / entry
                                stp = ent * (1 - side * risk)
                                tg = ent * (1 + side * tp)
                                cpx, _, _ = simulate(o, h, l, c, ce, side, stp, tg)
                                ctrl.append(float(np.clip(side * (cpx / ent - 1.0) - cost, -WINSOR, WINSOR)))
                            row["ctrl_net"] = float(np.mean(ctrl))
                            row["ctrl_R"] = row["ctrl_net"] / risk
            rows.append(row)
    df = pd.DataFrame(rows)
    df["cluster"] = df.market + "|" + df.date.str[:7]

    report: dict = {"variant": variant, "split": SPLIT, "universe": {m: len(v) for m, v in uni.items()},
                    "instances": len(insts), "rows": int(len(df)), "groups": {}, "age": {}, "claims": {}}
    sgn = lambda g: 1.0 if g.endswith("bullish") else -1.0       # noqa: E731  하락 패턴은 부호 반전
    for group, sub in df[df.age == 0].groupby("group"):
        entry = {}
        for split, part in [("all", sub), ("dev", sub[sub.split == "dev"]), ("val", sub[sub.split == "val"])]:
            cell = {}
            for hh in HORIZONS:
                cell[f"x{hh}_bps"] = cluster_stats(part[f"x{hh}"] * sgn(group), part.cluster, 1e4)
            g = part[part.get("has_geo", False) == True] if "has_geo" in part else part.iloc[0:0]   # noqa: E712
            if len(g):
                cell["structural"] = {"R": cluster_stats(g.R, g.cluster),
                                      "stop_rate": round(float((g.exit == "stop").mean()), 4),
                                      "target_rate": round(float((g.exit == "target").mean()), 4),
                                      "time_rate": round(float((g.exit == "time").mean()), 4),
                                      "median_held": float(g.held.median())}
                gc = g.dropna(subset=["ctrl_net"]) if "ctrl_net" in g else g.iloc[0:0]
                if len(gc):
                    cell["structural"]["ctrl_R_mean"] = round(float(gc.ctrl_R.mean()), 4)
                    cell["structural"]["net_bps"] = cluster_stats(gc.net, gc.cluster, 1e4)
                    cell["structural"]["paired_net_diff_bps"] = cluster_stats(gc.net - gc.ctrl_net, gc.cluster, 1e4)
            cell["n_events"] = int(len(part))
            entry[split] = cell
        entry["age0_median_bars"] = float(sub.age0.median()) if len(sub) else None
        report["groups"][group] = entry
    for group, sub in df.groupby("group"):
        report["age"][group] = {
            str(a): cluster_stats(s[f"x{PRIMARY_H}"] * sgn(group), s.cluster, 1e4)
            for a, s in sub.groupby("age")}
    # 완성도 구간(<85, 85+)
    comp = {}
    for group, sub in df[df.age == 0].groupby("group"):
        sub = sub.assign(b=np.where(sub.completion >= 85, "85+", "<85"))
        comp[group] = {b: cluster_stats(s[f"x{PRIMARY_H}"] * sgn(group), s.cluster, 1e4) for b, s in sub.groupby("b")}
    report["completion_buckets"] = comp
    for group, claim in CLAIMED.items():
        sub = df[(df.age == 0) & (df.group == group)]
        if not len(sub):
            continue
        d = sub[f"x{PRIMARY_H}"] * sgn(group)
        report["claims"][group] = {
            **claim, "n": int(d.notna().sum()),
            "measured_direction_hit_pct": round(float((d > 0).mean() * 100), 1),
            "measured_gross_win_pct": round(float(((sub[f"r{PRIMARY_H}"] * sgn(group)) > 0).mean() * 100), 1),
        }
    out_path = out_path or os.path.join(ROOT, "docs", "backtests", f"chart_pattern_audit_{variant}.json")
    json.dump(report, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    df.to_pickle(os.path.join(CACHE, f"_rows_{variant}.pkl"))
    print("saved", out_path)
    return report


# ───────────────────────── 수평선(지지·저항 전환) 사건 연구 ─────────────────────────
# 사전 고정: 사건은 market_briefing.sr_levels.level_events(min_touches=1) 가 마지막 봉에서 내는 것을 그대로 쓴다.
# 1차 비교는 접촉 수 구간(1회 / 2회 / 3회+)과 유형(일반 접촉 / 전환 재시험)이다. '2회 이상' 규칙이 1회 접촉보다
# 나은지(도식 주장)가 핵심 질문이다. 구조 거래: 손절=선 ∓ 0.5 ATR, 목표=2R, 최대 20거래일, 대조군은 같은 폭.

LEVEL_TARGET_R = 2.0


def _levels_worker(args):
    ticker, market, dates, o, h, l, c = args
    from market_briefing.sr_levels import LevelConfig, level_events
    cfg = LevelConfig()
    n = len(c)
    out = []
    for t in range(WARM, n - 1):
        lo = max(0, t - WINDOW + 1)
        for ev in level_events(o[lo:t + 1], h[lo:t + 1], l[lo:t + 1], c[lo:t + 1], cfg, min_touches=1):
            out.append({"ticker": ticker, "market": market, "t": t, "date": str(dates[t]), **ev})
    return out


def extract_levels(limit: int | None = None, workers: int | None = None) -> None:
    uni = json.load(open(os.path.join(CACHE, "_universe.json"), encoding="utf-8"))
    tasks = []
    for market in ("KRX", "US"):
        for ticker in uni[market]:
            df = load_frame(ticker)
            if df is None:
                continue
            tasks.append((ticker, market, df.Date.tolist(), df.Open.to_numpy(float), df.High.to_numpy(float),
                          df.Low.to_numpy(float), df.Close.to_numpy(float)))
    if limit:
        tasks = tasks[:limit]
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    print(f"levels: {len(tasks)} symbols, {workers} workers", flush=True)
    out, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, rows in enumerate(pool.map(_levels_worker, tasks, chunksize=1), 1):
            out.extend(rows)
            if k % 40 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} symbols, {len(out)} events, {time.time() - t0:.0f}s", flush=True)
    pickle.dump(out, open(os.path.join(CACHE, "_level_events.pkl"), "wb"))


def analyze_levels(out_path: str | None = None) -> dict:
    events = pickle.load(open(os.path.join(CACHE, "_level_events.pkl"), "rb"))
    uni, frames, arr, xs = prepare()
    # 같은 종목·유형·가격대(±1%) 사건은 10봉 안에 한 번만 센다
    events.sort(key=lambda e: (e["ticker"], e["type"], e["t"]))
    kept, last = [], {}
    for e in events:
        key = (e["ticker"], e["type"])
        prev = last.get(key)
        if prev and e["t"] - prev["t"] < 10 and abs(e["level"] / prev["level"] - 1) <= 0.01:
            continue
        last[key] = e
        kept.append(e)
    rows = []
    for k, ev in enumerate(kept):
        o, h, l, c, dates = arr[ev["ticker"]]
        n, t, side = len(c), ev["t"], ev["side"]
        e = t + 1
        if e >= n:
            continue
        entry = o[e]
        stop = ev["stop_ref"]
        if not ((stop < entry) if side > 0 else (stop > entry)):
            continue
        risk = abs(entry - stop) / entry
        if risk < MIN_RISK or risk > 0.25:
            continue
        tgt = entry * (1 + side * LEVEL_TARGET_R * risk)
        cost = COST[ev["market"]]
        row = {"market": ev["market"], "ticker": ev["ticker"], "date": dates[t], "type": ev["type"],
               "touches": ev["touches"], "tb": "1" if ev["touches"] == 1 else "2" if ev["touches"] == 2 else "3+",
               "side": side, "split": "dev" if dates[e] < SPLIT else "val", "flip_age": ev["flip_age"]}
        for hh in HORIZONS:
            if e + hh - 1 < n:
                gross = float(np.clip(c[e + hh - 1] / o[e] - 1.0, -WINSOR, WINSOR))
                base = xs[ev["market"]][hh].get(dates[e], np.nan)
                row[f"x{hh}"] = (gross - base) * side if np.isfinite(base) else np.nan
        px, why, held = simulate(o, h, l, c, e, side, float(stop), float(tgt))
        net = float(np.clip(side * (px / entry - 1.0) - cost, -WINSOR, WINSOR))
        row.update({"R": net / risk, "net": net, "exit": why})
        rng = np.random.default_rng(zlib.crc32(b"levels") + k)
        pool = [j for j in range(WARM, n - 1 - MAX_HOLD)
                if abs(j - t) > 20 and ((dates[j + 1] < SPLIT) == (dates[e] < SPLIT))]
        if len(pool) >= CONTROLS:
            ctrl = []
            for j in rng.choice(pool, size=CONTROLS, replace=False):
                ent = o[j + 1]
                cpx, _, _ = simulate(o, h, l, c, j + 1, side, ent * (1 - side * risk), ent * (1 + side * LEVEL_TARGET_R * risk))
                ctrl.append(float(np.clip(side * (cpx / ent - 1.0) - cost, -WINSOR, WINSOR)))
            row["ctrl_net"] = float(np.mean(ctrl))
            row["ctrl_R"] = row["ctrl_net"] / risk
        rows.append(row)
    df = pd.DataFrame(rows)
    df["cluster"] = df.market + "|" + df.date.str[:7]
    report = {"events_raw": len(events), "events_deduped": len(kept), "rows": int(len(df)), "cells": {}}
    for (typ, tb), sub in df.groupby(["type", "tb"]):
        cell = {}
        for split, part in [("all", sub), ("dev", sub[sub.split == "dev"]), ("val", sub[sub.split == "val"])]:
            gc = part.dropna(subset=["ctrl_net"])
            cell[split] = {
                "x10_bps": cluster_stats(part["x10"], part.cluster, 1e4),
                "x5_bps": cluster_stats(part["x5"], part.cluster, 1e4),
                "R": cluster_stats(part.R, part.cluster),
                "ctrl_R_mean": round(float(gc.ctrl_R.mean()), 4) if len(gc) else None,
                "net_bps": cluster_stats(part.net, part.cluster, 1e4),
                "paired_net_diff_bps": cluster_stats(gc.net - gc.ctrl_net, gc.cluster, 1e4) if len(gc) else None,
                "target_rate": round(float((part.exit == "target").mean()), 4) if len(part) else None,
                "stop_rate": round(float((part.exit == "stop").mean()), 4) if len(part) else None,
            }
        report["cells"][f"{typ}|touches={tb}"] = cell
    # 접촉 수 효과: 같은 유형 안에서 (2회+) − (1회) 의 초과수익 차이
    diff = {}
    for typ, sub in df.groupby("type"):
        multi, single = sub[sub.tb != "1"], sub[sub.tb == "1"]
        if len(multi) and len(single):
            diff[typ] = {"multi_x10_bps": round(float(multi.x10.mean() * 1e4), 1),
                         "single_x10_bps": round(float(single.x10.mean() * 1e4), 1),
                         "multi_n": int(len(multi)), "single_n": int(len(single))}
    report["touch_count_effect"] = diff
    out_path = out_path or os.path.join(ROOT, "docs", "backtests", "sr_level_audit_20261117.json")
    json.dump(report, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    df.to_pickle(os.path.join(CACHE, "_rows_levels.pkl"))
    print("saved", out_path)
    return report


# ───────────────────────── 모멘텀 맞춤 강건성 검사 ─────────────────────────
# 사후 점검(사전 고정 아님): 하락 사건(깨진 지지 재시험 등)은 하락 추세에서만 생겨 '추세 지속' 효과와 섞일 수 있다.
# 신호 시점의 60봉 수익률 교차단면 순위(5분위)별 무조건 평균 초과수익을 사건 초과수익에서 빼 본다.

def momentum_matched(out_path: str | None = None) -> dict:
    uni, frames, arr, xs = prepare()
    base: dict = {}
    rank5: dict = {}
    for market in ("KRX", "US"):
        cl = pd.DataFrame({t: frames[t].set_index("Date").Close for t in uni[market] if t in frames}).sort_index()
        op = pd.DataFrame({t: frames[t].set_index("Date").Open for t in uni[market] if t in frames}).sort_index()
        m60 = cl / cl.shift(60) - 1.0
        q = m60.rank(axis=1, pct=True)
        rank5[market] = (np.ceil(q * 5).clip(1, 5)).where(q.notna())
        fwd = (cl.shift(-PRIMARY_H) / op.shift(-1) - 1.0).clip(-WINSOR, WINSOR)
        ex = fwd.sub(fwd.mean(axis=1), axis=0)
        entry_date = pd.Series(cl.index, index=cl.index).shift(-1)
        long = pd.DataFrame({"q": rank5[market].stack(), "x": ex.stack()}).dropna().reset_index()
        long.columns = ["date", "ticker", "q", "x"]
        long["entry"] = long.date.map(entry_date)
        long["split"] = np.where(long.entry < SPLIT, "dev", "val")
        base[market] = long.groupby(["split", "q"]).x.mean().to_dict()
    out: dict = {"momentum_quintile_baseline_bps": {m: {f"{k[0]}|q{int(k[1])}": round(v * 1e4, 1) for k, v in d.items()}
                                                  for m, d in base.items()}, "groups": {}}
    sources = []
    for name, path, unsign in (("pattern", f"_rows_baseline.pkl", False), ("level", "_rows_levels.pkl", True)):
        df = pd.read_pickle(os.path.join(CACHE, path))
        if name == "pattern":
            df = df[df.age == 0].copy()
            df["raw"] = df.x10
            df["sign"] = np.where(df.dir == "bullish", 1.0, -1.0)
            df["key"] = df.group
        else:
            df = df.copy()
            df["raw"] = df.x10 * df.side
            df["sign"] = df.side.astype(float)
            df["key"] = df.type + "|touches=" + df.tb
        q = []
        for r in df.itertuples():
            try:
                q.append(rank5[r.market].at[r.date, r.ticker])
            except KeyError:
                q.append(np.nan)
        df["q"] = q
        df = df.dropna(subset=["q", "raw"])
        df["b"] = [base[m].get((s, qq), np.nan) for m, s, qq in zip(df.market, df.split, df.q)]
        df["matched"] = (df.raw - df.b) * df.sign
        df["plain"] = df.raw * df.sign
        sources.append(df)
        for key, sub in df.groupby("key"):
            out["groups"][key] = {
                "plain_bps": cluster_stats(sub.plain, sub.cluster, 1e4),
                "momentum_matched_bps": cluster_stats(sub.matched, sub.cluster, 1e4),
                "dev_matched_bps": cluster_stats(sub[sub.split == "dev"].matched, sub[sub.split == "dev"].cluster, 1e4),
                "val_matched_bps": cluster_stats(sub[sub.split == "val"].matched, sub[sub.split == "val"].cluster, 1e4),
                "mean_momentum_quintile": round(float(sub.q.mean()), 2),
            }
    out_path = out_path or os.path.join(ROOT, "docs", "backtests", "momentum_matched_audit_20261117.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("saved", out_path)
    return out


# ───────────────────────── 런타임 근거 파일 ─────────────────────────
# 판정(2단계): ① 사전 고정 기준 — 개발·검증 평균이 모두 신호 방향으로 양(+), 풀링 군집 t ≥ 2.5, 구간별 n ≥ 100.
# ② 실제로 검정한 모든 셀(패턴 14 + 수평선 12)에 대한 Holm 보정 p < 0.05 와 모멘텀 맞춤 t ≥ 2.0.
# ①만 통과하면 'no_evidence'(경계선). ①②를 모두 통과해야 'supported_weak'. 풀링 t ≤ -2.0 이고 두 구간 모두 음(-)이며
# Holm p < 0.05 이면 'contradicted'(신호와 반대). 나머지는 'no_evidence'. 어떤 판정도 점수 가중치를 만들지 않는다(표시용).

def _holm(pvals: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m, prev, out = len(ordered), 0.0, {}
    for i, (k, p) in enumerate(ordered):
        prev = min(1.0, max(prev, (m - i) * p))
        out[k] = prev
    return out


def export_evidence(path: str | None = None) -> dict:
    root_bt = os.path.join(ROOT, "docs", "backtests")
    pat = json.load(open(os.path.join(root_bt, "chart_pattern_audit_baseline.json"), encoding="utf-8"))
    lev = json.load(open(os.path.join(root_bt, "sr_level_audit_20261117.json"), encoding="utf-8"))
    mm = json.load(open(os.path.join(root_bt, "momentum_matched_audit_20261117.json"), encoding="utf-8"))
    cells: dict[str, dict] = {}
    for key, e in pat["groups"].items():
        cells[key] = {"kind": "pattern", "all": e["all"]["x10_bps"], "dev": e["dev"]["x10_bps"], "val": e["val"]["x10_bps"],
                      "paired": (e["all"].get("structural") or {}).get("paired_net_diff_bps")}
    for key, e in lev["cells"].items():
        cells[key] = {"kind": "level", "all": e["all"]["x10_bps"], "dev": e["dev"]["x10_bps"], "val": e["val"]["x10_bps"],
                      "paired": e["all"].get("paired_net_diff_bps")}
    pvals = {k: math.erfc(abs(c["all"]["t"]) / math.sqrt(2)) for k, c in cells.items() if c["all"].get("t") is not None}
    holm = _holm(pvals)
    out_cells = {}
    for key, c in cells.items():
        a, d, v = c["all"], c["dev"], c["val"]
        rule1 = (a["t"] is not None and d["mean"] is not None and v["mean"] is not None and d["mean"] > 0 and v["mean"] > 0
                 and a["t"] >= 2.5 and d["n"] >= 100 and v["n"] >= 100)
        t_matched = ((mm["groups"].get(key) or {}).get("momentum_matched_bps") or {}).get("t")
        rule2 = holm.get(key, 1.0) < 0.05 and t_matched is not None and t_matched >= 2.0
        if rule1 and rule2:
            verdict = "supported_weak"
        elif (a["t"] is not None and a["t"] <= -2.0 and d["mean"] < 0 and v["mean"] < 0
              and holm.get(key, 1.0) < 0.05):
            verdict = "contradicted"
        else:
            verdict = "no_evidence"
        direction = ("bullish" if key.endswith(":bullish") or key.startswith(("support_bounce", "flip_support"))
                     else "bearish")
        out_cells[key] = {
            "kind": c["kind"], "direction": direction, "n": a["n"], "edge_bps": a["mean"], "t": a["t"], "dev_bps": d["mean"], "val_bps": v["mean"],
            "holm_p": round(holm.get(key, 1.0), 4), "momentum_matched_t": t_matched,
            "paired_vs_random_timing_bps": (c["paired"] or {}).get("mean"), "rule1_pre_registered": bool(rule1),
            "verdict": verdict,
            "direction_hit_pct": round(float(pat["claims"][key]["measured_direction_hit_pct"]), 1) if key in pat.get("claims", {}) else None,
        }
    doc = {
        "version": 1, "as_of": "2026-11-17", "split": SPLIT,
        "universe": pat["universe"], "horizon_sessions": PRIMARY_H,
        "method": ("348종목·2017~2026 일봉, 첫 관측 다음 시가 진입, 10거래일 보유, 같은 날짜 교차단면 평균을 뺀 초과수익, "
                   "표준오차는 (시장,연월) 군집. edge_bps 는 신호 방향 기준(양수=신호가 맞음). "
                   "표시용 근거이며 점수·확률에 가중치를 주지 않는다."),
        "limitations": ["현재 상장 종목 기준(생존 편향)", "일봉 종가 기준, 장중 체결 미반영", "26개 셀을 동시에 검정(Holm 보정 적용)"],
        "cells": out_cells,
    }
    path = path or os.path.join(ROOT, "models", "pattern_evidence.json")
    json.dump(doc, open(path, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1, default=float)
    print("saved", path)
    for k, v in sorted(out_cells.items(), key=lambda kv: kv[1]["holm_p"]):
        print(f"  {k:34s} edge{v['edge_bps']:+7.1f}bp t{v['t']:+5.2f} holm_p={v['holm_p']:.3f} rule1={v['rule1_pre_registered']!s:5s} -> {v['verdict']}")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fetch", "extract", "analyze", "levels", "levels-analyze", "momentum-matched",
                                      "export-evidence"])
    ap.add_argument("--variant", default="baseline", choices=["baseline", "novol"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.stage == "fetch":
        fetch(a.limit)
    elif a.stage == "extract":
        extract(a.variant, a.limit, a.workers)
    elif a.stage == "levels":
        extract_levels(a.limit, a.workers)
    elif a.stage == "levels-analyze":
        analyze_levels(a.out)
    elif a.stage == "momentum-matched":
        momentum_matched(a.out)
    elif a.stage == "export-evidence":
        export_evidence(a.out)
    else:
        analyze(a.variant, a.out)


if __name__ == "__main__":
    main()
