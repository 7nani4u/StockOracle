#!/usr/bin/env python
"""7단계 스캔 엔진 순위 품질 감사 — 시점별(point-in-time) 교차단면 평가.

무엇을 재는가
-------------
스캔은 매일 같은 유니버스 안에서 종목에 점수를 매겨 순서를 정한다. 그래서 평가 단위는 '날짜별 교차단면'이다:
날짜 t 에 그 시점까지의 일봉만으로 실제 파이프라인(``build_snapshot_from_ohlcv`` → ``run_full_scan``)을 돌려
점수를 얻고, 그 점수가 **t 이후** 수익률의 순서를 맞추는지를 본다(순위상관 IC, 상위/하위 분위 격차, 삼중 장벽 승패 AUC).

누출 방지
---------
* 특징은 t 종가까지의 250봉 창(실서비스의 yfinance 1y 일봉과 같은 길이)과 t 까지의 벤치마크만 쓴다.
* 진입은 t+1 시가, 청산은 t+H 종가다(신호를 본 뒤 다음 시가에 체결). 목표값은 특징 계산에 쓰이지 않는다.
* 날짜별 평균을 빼는 교차단면 초과수익률은 목표값에만 쓴다.
* 개발(~2024-12-31)과 검증(2025-01-01~) 구간을 나눠 부호·크기가 유지되는지 본다. 수식에 학습 파라미터가
  없어 '재학습'은 없지만, 변형(ablation)의 채택은 개발 구간 결과로만 정하고 검증 구간에서 재확인한다.
* ML 비교(LightGBM)는 확장창 walk-forward: 검증 연도 시작 전 (H + 엄바고) 거래일 이전 행만 학습한다.

한계(보고서에도 기록)
---------------------
* 유니버스는 2026-11 기준 유동성으로 고른 현재 종목 목록이라 생존·선정 편향이 있다(과거 시점의 실제 유니버스가 아님).
* QMJ 품질 등급은 시점별 재무제표가 없어 재현할 수 없다 → 모든 종목을 'unknown'으로 두고 기술 점수만 평가한다.
* 박스 실행 플랜 완성 여부는 종목·시점마다 플랜 엔진을 돌려야 해 이 도구에 포함하지 않았다(별도 감사 존재).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HORIZONS = (5, 10, 20)
WINDOW = 250                 # 실서비스 1y 일봉과 같은 길이
WARM = 260                   # 252봉 모멘텀까지 계산할 수 있는 최소 이력
SPLIT = "2025-01-01"         # 개발/검증 경계
BARRIER_ATR = 1.5            # 스캔의 손절 배수(ATR_STOP_MULTIPLIER)와 같은 장벽 폭
BARRIER_DAYS = 20
ROUND_TRIP_COST = {"KRX": 0.0025, "US": 0.0010}   # 기존 박스 감사와 같은 왕복 비용 가정
STATE_SCORE = {"READY": 100.0, "WATCH": 70.0, "WAIT_PULLBACK": 60.0, "COOLDOWN": 20.0,
               "EARNINGS_BLOCK": 10.0, "FAR": 0.0}

# ───────────────────────── 자료 ─────────────────────────

_DATA: dict = {}


def _clean(df: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df[(df.Close > 0) & (df.Open > 0) & (df.High >= df.Low) & (df.Volume >= 0)]
    df = df[df.Date <= cutoff].drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    return df


def load_cache(cache_dir: str, cutoff: str) -> dict:
    uni = json.load(open(os.path.join(cache_dir, "_universe.json"), encoding="utf-8"))

    def read(sym: str) -> pd.DataFrame:
        path = os.path.join(cache_dir, sym.replace("^", "_") + ".csv")
        return _clean(pd.read_csv(path, parse_dates=["Date"]), cutoff)

    out = {"stocks": {"KRX": {}, "US": {}}, "bench": {"KRX": read("^KS11"), "US": read("SPY")}, "vix": read("^VIX")}
    for market, rows in uni.items():
        for ticker, name, tier in rows:
            try:
                df = read(ticker)
            except FileNotFoundError:
                continue
            if len(df) >= WARM + 30:
                out["stocks"][market][ticker] = df
                out.setdefault("tier", {})[ticker] = tier
    return out


# ───────────────────────── 특징(벡터화, 모두 인과적) ─────────────────────────

def extra_features(df: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    """t 까지의 값만 쓰는 롤링 특징. 행 i 는 날짜 df.Date[i] 의 종가 시점에 알 수 있는 값이다."""
    c, h, l, o, v = (df[k].astype(float) for k in ("Close", "High", "Low", "Open", "Volume"))
    ret1 = c.pct_change()
    out = pd.DataFrame(index=df.index)
    for k in (5, 20, 60, 120):
        out[f"ret{k}"] = c / c.shift(k) - 1
    out["mom12_1"] = c.shift(21) / c.shift(252) - 1
    b = bench.set_index("Date")["Close"].astype(float)
    bc = b.reindex(df.Date, method="ffill").to_numpy()
    bser = pd.Series(bc, index=df.index)
    for k in (20, 63, 126):
        out[f"rs{k}_x"] = (c / c.shift(k) - bser / bser.shift(k)) * 100.0
    prior20 = v.shift(1).rolling(20).mean()
    out["rvol20"] = v / prior20.replace(0, np.nan)
    out["rvol5_20"] = v.rolling(5).mean() / v.shift(5).rolling(20).mean().replace(0, np.nan)
    out["vtrend10_50"] = np.log(v.rolling(10).mean().replace(0, np.nan) / v.rolling(50).mean().replace(0, np.nan))
    obv = (np.sign(c.diff()).fillna(0.0) * v).cumsum()
    out["obv_slope20"] = (obv - obv.shift(20)) / (20.0 * v.rolling(20).mean().replace(0, np.nan))
    rng = (h - l).replace(0, np.nan)
    mfm = (((c - l) - (h - c)) / rng).fillna(0.0)
    out["cmf20"] = (mfm * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    tp = (h + l + c) / 3.0
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0.0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(14).sum()
    out["mfi14"] = 100.0 - 100.0 / (1.0 + pos / neg.replace(0, np.nan))
    out["dvol20"] = np.log((c * v).rolling(20).mean().replace(0, np.nan))
    out["amihud20"] = np.log1p((ret1.abs() / (c * v).replace(0, np.nan)).rolling(20).mean() * 1e9)
    up = v.where(ret1 > 0, 0.0).rolling(20).sum()
    dn = v.where(ret1 < 0, 0.0).rolling(20).sum()
    out["updown20"] = np.log((up + 1.0) / (dn + 1.0))
    out["rv20"] = ret1.rolling(20).std() * math.sqrt(252)
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    out["bbw20"] = 4.0 * sd20 / ma20
    out["pctb20"] = (c - (ma20 - 2 * sd20)) / (4.0 * sd20).replace(0, np.nan)
    lo20, hi20 = l.rolling(20).min(), h.rolling(20).max()
    out["don20pos"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    out["dma50"] = c / c.rolling(50).mean() - 1
    out["dma200"] = c / c.rolling(200).mean() - 1
    out["gap1"] = o / c.shift() - 1
    out["body1"] = c / o - 1
    out["clv1"] = ((c - l) - (h - c)) / rng
    return out


def forward_targets(df: pd.DataFrame, atr_series: np.ndarray | None = None) -> dict:
    """t+1 시가 진입 기준 목표값(배열, 길이 n). 미래가 모자라면 NaN."""
    o = df.Open.to_numpy(float)
    h = df.High.to_numpy(float)
    l = df.Low.to_numpy(float)
    c = df.Close.to_numpy(float)
    n = len(df)
    out = {}
    entry = np.full(n, np.nan)
    entry[:-1] = o[1:]
    for H in HORIZONS:
        r = np.full(n, np.nan)
        idx = np.arange(n - H)
        r[idx] = c[idx + H] / entry[idx] - 1.0
        out[f"fwd{H}"] = r
    mae = np.full(n, np.nan)
    for i in range(n - BARRIER_DAYS - 1):
        mae[i] = l[i + 1:i + 1 + BARRIER_DAYS].min() / entry[i] - 1.0
    out["mae20"] = mae
    return out


def barrier_label(o, h, l, i: int, atr: float, k: float = BARRIER_ATR, days: int = BARRIER_DAYS) -> float:
    """t+1 시가에서 ±k·ATR 장벽 중 먼저 닿는 쪽. 같은 봉에서 둘 다 닿으면 손절 우선(보수적). 없으면 0, 미래 부족이면 NaN."""
    n = len(o)
    if i + 1 + days > n or not np.isfinite(atr) or atr <= 0:
        return float("nan")
    entry = o[i + 1]
    up, dn = entry + k * atr, entry - k * atr
    for j in range(i + 1, i + 1 + days):
        hit_dn = l[j] <= dn
        hit_up = h[j] >= up
        if hit_dn:
            return -1.0
        if hit_up:
            return 1.0
    return 0.0


# ───────────────────────── 날짜별 파이프라인 실행 ─────────────────────────

def _init_worker(cache_dir: str, cutoff: str, gate: str):
    from market_briefing import technique_prune
    if gate == "off":                       # 규칙 파일이 없으면 모든 기법을 허용(fail-open) — 7단계 논리 자체를 본다
        technique_prune._RULES_PATH_OVERRIDE = os.path.join(cache_dir, "__no_rules__.json")
    _DATA.update(load_cache(cache_dir, cutoff))
    _DATA["extras"] = {}
    _DATA["targets"] = {}
    _DATA["lists"] = {}
    _DATA["pos"] = {}
    for market, stocks in _DATA["stocks"].items():
        for t, df in stocks.items():
            _DATA["extras"][t] = extra_features(df, _DATA["bench"][market])
            _DATA["targets"][t] = forward_targets(df)
            _DATA["lists"][t] = {k: df[k].astype(float).tolist() for k in ("Close", "High", "Low", "Volume", "Open")}
            _DATA["pos"][t] = {d: i for i, d in enumerate(df.Date.tolist())}
    vix = _DATA["vix"]
    _DATA["vix_s"] = vix.set_index("Date")["Close"].astype(float)


def _regime_at(market: str, t: pd.Timestamp, vix_mode: str):
    from market_briefing.hybrid_signals import compute_regime, detect_vol_regime, _calc_adx, _calc_ma, _calc_atr
    from market_briefing.dual_score_v2 import REGIME_SIDEWAYS, VOL_NORMAL
    bench = _DATA["bench"][market]
    end = int(bench.Date.searchsorted(t, side="right"))
    sl = bench.iloc[max(0, end - WINDOW):end]
    bc, bh, bl = sl.Close.tolist(), sl.High.tolist(), sl.Low.tolist()
    regime, vol_regime = REGIME_SIDEWAYS, VOL_NORMAL
    if len(bc) >= 20:
        bn = len(bc)
        adx_d = _calc_adx(bh[-50:], bl[-50:], bc[-50:]) if bn >= 30 else None
        ma200 = _calc_ma(bc, 200) if bn >= 200 else None
        vix_v = None
        if vix_mode == "both" or (vix_mode == "us" and market == "US"):
            vs = _DATA["vix_s"]
            k = int(vs.index.searchsorted(t, side="right"))
            if k > 0:
                vix_v = float(vs.iloc[k - 1])
        regime = compute_regime(bc[-1], ma200, adx_d, vix_v)["regime"]
        atr_v = _calc_atr(bh[-20:], bl[-20:], bc[-20:])
        if atr_v and bc[-1] > 0:
            vol_regime = detect_vol_regime(atr_v / bc[-1] * 100)
    return regime, vol_regime, bc


def process_dates(job):
    market, dates, vix_mode = job
    from market_briefing.scan_engine import StockUniverse, run_full_scan, build_snapshot_from_ohlcv
    rows = []
    tickers = list(_DATA["stocks"][market])
    for t in dates:
        t = pd.Timestamp(t)
        regime, vol_regime, bc = _regime_at(market, t, vix_mode)
        universe, snaps, meta = [], {}, {}
        for tk in tickers:
            i = _DATA["pos"][tk].get(t)
            if i is None or i < WARM:
                continue
            lst = _DATA["lists"][tk]
            a = i - WINDOW + 1
            snaps[tk] = build_snapshot_from_ohlcv(
                tk, lst["Close"][a:i + 1], lst["High"][a:i + 1], lst["Low"][a:i + 1], lst["Volume"][a:i + 1],
                lst["Open"][a:i + 1], bench_closes=bc)
            universe.append(StockUniverse(tk, tk, "CORE", currency="KRW" if market == "KRX" else "USD"))
            meta[tk] = i
        if len(universe) < 15:
            continue
        res = run_full_scan(universe, snaps, {}, regime=regime, vol_regime=vol_regime, scan_mode="FULL")
        for c in res.candidates:
            tk, i = c.ticker, meta[c.ticker]
            s = snaps[tk]
            fd = c.filter_detail
            row = {"market": market, "date": t, "ticker": tk, "regime": regime, "vol_regime": vol_regime,
                   "status": c.status, "raw_pass": bool(fd.get("passes_all")),
                   "f_ma200": bool(fd.get("price_above_ma200")), "f_adx": bool(fd.get("adx_above_20")),
                   "f_di": bool(fd.get("plus_di_above_minus")), "f_atr": bool(fd.get("atr_pct_below_cap")),
                   "f_eff": bool(fd.get("efficiency_above_30")),
                   "atr_spike": fd.get("atr_spike_action") == "SOFT_CAP",
                   "chase_fail": not c.passes_anti_chase, "pullback": bool(c.pullback_signal),
                   "bqs": c.bqs, "fws": c.fws, "ncs": c.ncs, "rank_score": c.rank_score,
                   "dist": c.distance_pct, "price": c.price,
                   "adx": s.adx, "plus_di": s.plus_di, "minus_di": s.minus_di, "atr_pct": s.atr_pct,
                   "vol_ratio": s.vol_ratio, "rs_full": s.rs_vs_bench, "eff": s.efficiency,
                   "hurst": s.hurst if s.hurst is not None else np.nan, "bis": s.bis_score,
                   "ext_atr": (c.price - c.entry_trigger) / s.atr if s.atr > 0 else np.nan}
            row.update({k: v for k, v in c.bqs_components.items() if k != "BQS"})
            row.update({k: v for k, v in c.fws_components.items() if k != "FWS"})
            ex = _DATA["extras"][tk].iloc[i]
            row.update({k: float(ex[k]) for k in ex.index})
            tg = _DATA["targets"][tk]
            for k, arr in tg.items():
                row[k] = float(arr[i])
            lst = _DATA["lists"][tk]
            row["barrier"] = barrier_label(lst["Open"], lst["High"], lst["Low"], i, s.atr)
            rows.append(row)
    return rows


def build_rows(args) -> pd.DataFrame:
    data = load_cache(args.cache, args.cutoff)
    jobs = []
    for market in ("KRX", "US"):
        cal = data["bench"][market].Date
        cal = cal.iloc[WARM::args.step]
        # 마지막 H 거래일은 목표값이 없으므로 제외해도 되지만 특징은 남겨 둔다(분석에서 NaN 제거).
        dates = cal.tolist()
        chunk = max(1, len(dates) // (args.workers * 3))
        for k in range(0, len(dates), chunk):
            jobs.append((market, dates[k:k + chunk], args.vix))
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.cache, args.cutoff, args.gate)) as ex:
        for k, part in enumerate(ex.map(process_dates, jobs)):
            rows.extend(part)
            if k % 10 == 0:
                print(f"  chunk {k + 1}/{len(jobs)}  rows={len(rows)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["period"] = np.where(df.date < pd.Timestamp(SPLIT), "dev", "test")
    return df


# ───────────────────────── 통계 ─────────────────────────

def newey_west_t(x, lag: int) -> float:
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    n = len(x)
    if n < 8:
        return float("nan")
    e = x - x.mean()
    s = float(e @ e) / n
    for l in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - l / (lag + 1.0)) * float(e[l:] @ e[:-l]) / n
    se = math.sqrt(max(s, 1e-18) / n)
    return float(x.mean() / se)


def _rank_ic_by_group(df: pd.DataFrame, feat: str, tgt: str) -> pd.Series:
    sub = df[["market", "date", feat, tgt]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    g = sub.groupby(["market", "date"])
    rf = g[feat].rank(method="average")
    rt = g[tgt].rank(method="average")
    sub = sub.assign(_rf=rf, _rt=rt)

    def corr(x):
        if len(x) < 15 or x._rf.nunique() < 3:
            return np.nan
        return float(np.corrcoef(x._rf, x._rt)[0, 1])
    return sub.groupby(["market", "date"]).apply(corr, include_groups=False).dropna()


def ic_summary(df: pd.DataFrame, feat: str, tgt: str, H: int, step: int) -> dict:
    s = _rank_ic_by_group(df, feat, tgt)
    if s.empty:
        return {"n": 0}
    lag = int(math.ceil(H / step)) + 1
    out = {"n": int(len(s)), "ic": round(float(s.mean()), 4), "t_nw": round(newey_west_t(s.values, lag), 2),
           "pos": round(float((s > 0).mean()), 3)}
    for per in ("dev", "test"):
        sp = s[s.index.get_level_values("date") < pd.Timestamp(SPLIT)] if per == "dev" else \
            s[s.index.get_level_values("date") >= pd.Timestamp(SPLIT)]
        out[per] = {"n": int(len(sp)), "ic": round(float(sp.mean()), 4) if len(sp) else None,
                    "t_nw": round(newey_west_t(sp.values, lag), 2) if len(sp) else None}
    for mk in ("KRX", "US"):
        sm = s[s.index.get_level_values("market") == mk]
        out[mk] = {"ic": round(float(sm.mean()), 4) if len(sm) else None,
                   "t_nw": round(newey_west_t(sm.values, lag), 2) if len(sm) else None}
    return out


def paired_delta(df: pd.DataFrame, a: str, b: str, tgt: str, H: int, step: int) -> dict:
    sa, sb = _rank_ic_by_group(df, a, tgt), _rank_ic_by_group(df, b, tgt)
    d = (sa - sb).dropna()
    lag = int(math.ceil(H / step)) + 1
    out = {"n": int(len(d)), "delta_ic": round(float(d.mean()), 4) if len(d) else None,
           "t_nw": round(newey_west_t(d.values, lag), 2) if len(d) else None}
    for per in ("dev", "test"):
        m = d.index.get_level_values("date") < pd.Timestamp(SPLIT) if per == "dev" else \
            d.index.get_level_values("date") >= pd.Timestamp(SPLIT)
        sp = d[m]
        out[per] = {"delta_ic": round(float(sp.mean()), 4) if len(sp) else None,
                    "t_nw": round(newey_west_t(sp.values, lag), 2) if len(sp) else None}
    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for H in HORIZONS:
        df[f"xr{H}"] = df[f"fwd{H}"] - df.groupby(["market", "date"])[f"fwd{H}"].transform("mean")
    df["xmae20"] = df["mae20"] - df.groupby(["market", "date"])["mae20"].transform("mean")
    return df


def partial_residual(df: pd.DataFrame, feat: str, control: str, name: str) -> pd.DataFrame:
    """날짜별 교차단면에서 feat 의 순위를 control 순위에 회귀한 잔차(이미 점수가 가진 정보를 뺀 몫)."""
    sub = df[["market", "date", feat, control]].dropna()
    g = sub.groupby(["market", "date"])
    rf, rc = g[feat].rank(pct=True), g[control].rank(pct=True)
    sub = sub.assign(_rf=rf, _rc=rc)

    def resid(x):
        c = x._rc - x._rc.mean()
        den = float((c * c).sum())
        beta = float((c * (x._rf - x._rf.mean())).sum() / den) if den > 0 else 0.0
        return x._rf - x._rf.mean() - beta * c
    r = sub.groupby(["market", "date"], group_keys=False).apply(resid, include_groups=False)
    df = df.copy()
    df[name] = np.nan
    df.loc[r.index, name] = r
    return df


def quintile_spread(df: pd.DataFrame, score: str, tgt: str, q: int = 5) -> dict:
    sub = df[["market", "date", score, tgt]].dropna()
    g = sub.groupby(["market", "date"])
    sub = sub.assign(_q=g[score].transform(lambda s: pd.qcut(s.rank(method="first"), q, labels=False)))
    top = sub[sub._q == q - 1].groupby(["market", "date"])[tgt].mean()
    bot = sub[sub._q == 0].groupby(["market", "date"])[tgt].mean()
    spread = (top - bot).dropna()
    return {"top_mean_excess": round(float(top.mean()), 5), "bottom_mean_excess": round(float(bot.mean()), 5),
            "spread": round(float(spread.mean()), 5), "n_dates": int(len(spread))}


def auc_by_block(df: pd.DataFrame, score: str, label: str = "barrier", seed: int = 7) -> dict:
    from sklearn.metrics import roc_auc_score
    sub = df[["market", "date", score, label]].dropna()
    sub = sub[sub[label] != 0]
    if sub[label].nunique() < 2:
        return {}
    sub = sub.assign(_p=sub.groupby(["market", "date"])[score].rank(pct=True), _y=(sub[label] > 0).astype(int))
    auc = roc_auc_score(sub._y, sub._p)
    sub["_m"] = sub.date.dt.to_period("M").astype(str)
    months = sub._m.unique()
    rng = np.random.default_rng(seed)
    boots = []
    grouped = {m: g for m, g in sub.groupby("_m")}
    for _ in range(300):
        pick = rng.choice(months, len(months))
        b = pd.concat([grouped[m] for m in pick])
        if b._y.nunique() == 2:
            boots.append(roc_auc_score(b._y, b._p))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    out = {"auc": round(float(auc), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": int(len(sub)),
           "win_rate": round(float(sub._y.mean()), 4)}
    for per, m in (("dev", sub.date < pd.Timestamp(SPLIT)), ("test", sub.date >= pd.Timestamp(SPLIT))):
        sp = sub[m]
        out[per] = round(float(roc_auc_score(sp._y, sp._p)), 4) if sp._y.nunique() == 2 else None
    return out


# ───────────────────────── 분석 ─────────────────────────

def ncs_variant(df: pd.DataFrame, drop_vol_bonus=False, drop_fws_vol=False, drop_bis=False,
                drop_hurst=False, drop_ext=False, rs_col: str | None = None) -> pd.Series:
    """NCS = clamp(BQS' - 0.8·FWS' + 10) 를 구성 요소를 빼거나 바꿔 다시 계산(패널티는 이 평가에서 0)."""
    bqs = df["bqs_trend"] + df["bqs_direction"] + df["bqs_volatility"] + df["bqs_proximity"] + df["bqs_tailwind"] \
        + df["bqs_rs"] + df["bqs_vol_bonus"] + df["bqs_weekly_adx"] + df["bqs_bis"] + df["bqs_hurst"]
    fws = df["fws_volume"] + df["fws_extension"] + df["fws_marginal_trend"] + df["fws_vol_shock"] + df["fws_regime_instability"]
    if drop_vol_bonus:
        bqs = bqs - df["bqs_vol_bonus"]
    if drop_bis:
        bqs = bqs - df["bqs_bis"]
    if drop_hurst:
        bqs = bqs - df["bqs_hurst"]
    if drop_fws_vol:
        fws = fws - df["fws_volume"]
    if drop_ext:
        fws = fws - df["fws_extension"]
    if rs_col:
        bqs = bqs - df["bqs_rs"] + 15.0 * ((df[rs_col] + 5.0) / 20.0).clip(0.0, 1.0)
    return (bqs.clip(0, 100) - 0.8 * fws.clip(0, 100) + 10.0).clip(0, 100)


VOLUME_CANDIDATES = ["rvol20", "rvol5_20", "vtrend10_50", "obv_slope20", "cmf20", "mfi14", "dvol20", "amihud20", "updown20"]
PRICE_CANDIDATES = ["ret5", "ret20", "ret60", "mom12_1", "rs20_x", "rs63_x", "rs126_x", "rs_full", "rv20", "bbw20",
                    "pctb20", "don20pos", "dma50", "dma200", "gap1", "body1", "clv1", "eff", "hurst", "adx", "atr_pct"]


def analyze(df: pd.DataFrame, step: int) -> dict:
    df = add_targets(df)
    df["state_shipped"] = df.status.map(lambda s: 70.0 if s == "READY" else STATE_SCORE.get(s, 0.0))
    df["state_raw"] = df.status.map(lambda s: STATE_SCORE.get(s, 0.0))
    df["composite_noqmj"] = 0.35 * df.state_shipped + 0.40 * df.ncs
    df["composite_noqmj_raw"] = 0.35 * df.state_raw + 0.40 * df.ncs
    df["neg_fws"] = -df.fws
    df["raw_pass_f"] = df.raw_pass.astype(float)
    res: dict = {"meta": {"rows": int(len(df)), "dates": int(df.date.nunique()),
                          "tickers": int(df.ticker.nunique()), "split": SPLIT, "step": step,
                          "date_range": [str(df.date.min())[:10], str(df.date.max())[:10]],
                          "n_test_dates": int(df[df.period == "test"].date.nunique())}}

    # ① 단계별 통과 깔때기 (날짜별 평균 개수가 아니라 전체 비율)
    fun = {}
    for mk in ("KRX", "US"):
        d = df[df.market == mk]
        fun[mk] = {"rows": int(len(d)),
                   "price_above_ma200": round(float(d.f_ma200.mean()), 4), "adx_ge_20": round(float(d.f_adx.mean()), 4),
                   "plus_di_gt_minus": round(float(d.f_di.mean()), 4), "atr_below_cap": round(float(d.f_atr.mean()), 4),
                   "efficiency_ge_30": round(float(d.f_eff.mean()), 4), "all_six": round(float(d.raw_pass.mean()), 4),
                   "status_share": {k: round(float(v), 4) for k, v in d.status.value_counts(normalize=True).items()},
                   "chase_fail": round(float(d.chase_fail.mean()), 4), "atr_spike_soft_cap": round(float(d.atr_spike.mean()), 4),
                   "pullback_triggered": round(float(d.pullback.mean()), 5)}
    res["funnel"] = fun

    # ② 현재 점수들의 순위 예측력
    base_scores = ["bqs", "ncs", "composite_noqmj", "composite_noqmj_raw", "neg_fws", "rank_score", "state_shipped",
                   "raw_pass_f", "bqs_trend", "bqs_direction", "bqs_volatility", "bqs_proximity", "bqs_rs",
                   "bqs_vol_bonus", "bqs_bis", "bqs_hurst", "fws_volume", "fws_extension", "fws_marginal_trend", "fws_vol_shock"]
    ic = {}
    for sc in base_scores:
        ic[sc] = {f"xr{H}": ic_summary(df, sc, f"xr{H}", H, step) for H in HORIZONS}
    res["ic_current"] = ic

    # ③ 삼중 장벽 AUC · 분위 격차
    res["barrier_auc"] = {sc: auc_by_block(df, sc) for sc in ("bqs", "ncs", "composite_noqmj", "neg_fws", "rs_full", "rs63_x", "mom12_1")}
    res["quintile"] = {sc: {f"xr{H}": quintile_spread(df, sc, f"xr{H}") for H in HORIZONS}
                       for sc in ("ncs", "composite_noqmj", "rs63_x", "mom12_1")}

    # ④ 상태·2단계 통과의 조건부 초과수익률
    cond = {}
    for H in HORIZONS:
        t = f"xr{H}"
        cond[t] = {"status": {k: {"n": int(len(g)), "mean_excess_pct": round(float(g[t].mean() * 100), 3)}
                              for k, g in df.groupby("status") if g[t].notna().sum() > 30},
                   "raw_pass": {str(k): {"n": int(len(g)), "mean_excess_pct": round(float(g[t].mean() * 100), 3)}
                                for k, g in df.groupby("raw_pass") if g[t].notna().sum() > 30}}
        for k in ("f_ma200", "f_adx", "f_di", "f_eff", "f_atr"):
            cond[t][k] = {str(v): {"n": int(len(g)), "mean_excess_pct": round(float(g[t].mean() * 100), 3)}
                          for v, g in df.groupby(k) if g[t].notna().sum() > 30}
    res["conditional_excess"] = cond

    # ⑤ 거래량 구성요소 ablation (NCS 에서 빼거나 바꿨을 때 IC 변화)
    df["ncs_base"] = ncs_variant(df)
    variants = {"no_vol_bonus": dict(drop_vol_bonus=True), "no_fws_volume": dict(drop_fws_vol=True),
                "no_bis": dict(drop_bis=True), "no_volume_all": dict(drop_vol_bonus=True, drop_fws_vol=True, drop_bis=True),
                "no_hurst": dict(drop_hurst=True), "no_extension": dict(drop_ext=True),
                "rs63": dict(rs_col="rs63_x"), "rs126": dict(rs_col="rs126_x"), "rs20": dict(rs_col="rs20_x")}
    abl = {}
    for name, kw in variants.items():
        df[f"ncs_{name}"] = ncs_variant(df, **kw)
        abl[name] = {f"xr{H}": {"ic": ic_summary(df, f"ncs_{name}", f"xr{H}", H, step),
                                "delta_vs_base": paired_delta(df, f"ncs_{name}", "ncs_base", f"xr{H}", H, step)}
                     for H in HORIZONS}
    abl["baseline"] = {f"xr{H}": {"ic": ic_summary(df, "ncs_base", f"xr{H}", H, step)} for H in HORIZONS}
    res["ablation"] = abl

    # ⑥ 후보 특징: 단독 IC + NCS 통제 후 추가 IC
    cand = {}
    for ft in VOLUME_CANDIDATES + PRICE_CANDIDATES:
        if ft not in df:
            continue
        df = partial_residual(df, ft, "ncs_base", f"res_{ft}")
        cand[ft] = {"group": "volume" if ft in VOLUME_CANDIDATES else "price",
                    **{f"xr{H}": {"alone": ic_summary(df, ft, f"xr{H}", H, step),
                                  "incremental_after_ncs": ic_summary(df, f"res_{ft}", f"xr{H}", H, step)}
                       for H in (10, 20)}}
    res["candidates"] = cand
    return res, df


def adoption_flag(entry: dict, horizon: str = "xr20") -> str:
    """사전 고정한 채택 규칙: 개발·검증 구간 모두 같은 부호이고 검증 구간 |t_NW| ≥ 2 이며 시장 두 곳 부호 일치."""
    inc = entry[horizon]["incremental_after_ncs"]
    if not inc.get("n"):
        return "no_data"
    dev, test = inc["dev"]["ic"], inc["test"]["ic"]
    if dev is None or test is None or dev == 0 or test == 0:
        return "no_data"
    same_sign = (dev > 0) == (test > 0)
    krx, us = inc["KRX"]["ic"], inc["US"]["ic"]
    markets_agree = krx is not None and us is not None and (krx > 0) == (us > 0) == (test > 0)
    strong = inc["test"]["t_nw"] is not None and abs(inc["test"]["t_nw"]) >= 2.0
    if same_sign and strong and markets_agree:
        return "consistent_positive" if test > 0 else "consistent_negative"
    return "not_robust"


# ───────────────────────── ML walk-forward(기준선 대비) ─────────────────────────

def ml_walk_forward(df: pd.DataFrame, step: int, horizon: int = 10, embargo_days: int = 5) -> dict:
    import lightgbm as lgb
    feats = ["adx", "plus_di", "minus_di", "atr_pct", "vol_ratio", "rs_full", "eff", "hurst", "bis", "dist", "ext_atr",
             "bqs", "fws", "ncs", "rank_score"] + [c for c in VOLUME_CANDIDATES + PRICE_CANDIDATES if c in df]
    feats = list(dict.fromkeys(feats))
    tgt = f"xr{horizon}"
    data = df.dropna(subset=[tgt]).copy()
    data["rk"] = data.groupby(["market", "date"])[tgt].rank(pct=True)     # 교차단면 순위 목표
    data["mkt"] = (data.market == "US").astype(int)
    feats_m = feats + ["mkt"]
    years = [2024, 2025, 2026]
    out = {"horizon": horizon, "features": feats_m, "folds": []}
    pred_parts = []
    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        end = pd.Timestamp(f"{y + 1}-01-01")
        cutoff = start - pd.tseries.offsets.BDay(horizon + embargo_days)
        train = data[data.date < cutoff]
        test = data[(data.date >= start) & (data.date < end)]
        if len(train) < 5000 or test.empty:
            continue
        model = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                                  subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                                  random_state=7, verbose=-1, n_jobs=4)
        model.fit(train[feats_m], train["rk"])
        t2 = test.copy()
        t2["ml_pred"] = model.predict(test[feats_m])
        pred_parts.append(t2)
        out["folds"].append({"test_year": y, "train_rows": int(len(train)), "test_rows": int(len(test))})
    if not pred_parts:
        return out
    pred = pd.concat(pred_parts)
    out["oos_rows"] = int(len(pred))
    out["ml_ic"] = ic_summary(pred, "ml_pred", tgt, horizon, step)
    out["ncs_ic_same_rows"] = ic_summary(pred, "ncs", tgt, horizon, step)
    out["composite_ic_same_rows"] = ic_summary(pred, "composite_noqmj", tgt, horizon, step) if "composite_noqmj" in pred else None
    out["ml_minus_ncs"] = paired_delta(pred, "ml_pred", "ncs", tgt, horizon, step)
    out["ml_quintile"] = quintile_spread(pred, "ml_pred", tgt)
    out["ncs_quintile"] = quintile_spread(pred, "ncs", tgt)
    return out


# ───────────────────────── 실행 ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=os.path.join(ROOT, "tmp", "scan_eval_cache_20261112"))
    ap.add_argument("--cutoff", default="2026-11-11")
    ap.add_argument("--step", type=int, default=3, help="날짜 표본 간격(거래일)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gate", choices=("off", "on"), default="off", help="off=기법 검증 게이트 없이 7단계 논리만 평가")
    ap.add_argument("--vix", choices=("both", "us", "none"), default="both", help="레짐에 VIX 를 쓰는 시장(현재 서비스는 both)")
    ap.add_argument("--rows", default=os.path.join(ROOT, "tmp", "scan_eval_rows_20261112.pkl"))
    ap.add_argument("--reuse", action="store_true", help="저장된 행을 재사용(분석만 다시 실행)")
    ap.add_argument("--ml", action="store_true", help="LightGBM walk-forward 비교 포함")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "backtests", "scan_ranking_audit_20261112.json"))
    args = ap.parse_args()

    if args.reuse and os.path.exists(args.rows):
        df = pd.read_pickle(args.rows)
        print(f"rows reused: {len(df)}")
    else:
        t0 = time.time()
        df = build_rows(args)
        os.makedirs(os.path.dirname(args.rows), exist_ok=True)
        df.to_pickle(args.rows)
        print(f"rows built: {len(df)} in {time.time() - t0:.0f}s")

    res, df = analyze(df, args.step)
    for ft, entry in res["candidates"].items():
        entry["adoption_rule_xr20"] = adoption_flag(entry, "xr20")
        entry["adoption_rule_xr10"] = adoption_flag(entry, "xr10")
    if args.ml:
        res["ml_walk_forward"] = ml_walk_forward(df, args.step)
    h = hashlib.sha256()
    for f in sorted(os.listdir(args.cache)):
        if f.endswith(".csv"):
            h.update(f.encode())
            h.update(str(os.path.getsize(os.path.join(args.cache, f))).encode())
    res["meta"].update({"cutoff": args.cutoff, "gate": args.gate, "vix_mode": args.vix, "window": WINDOW,
                        "barrier": {"atr_mult": BARRIER_ATR, "days": BARRIER_DAYS},
                        "cache_fingerprint": h.hexdigest()[:16], "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                        "cost_round_trip": ROUND_TRIP_COST,
                        "caveats": ["universe selected with 2026-11 liquidity (survivorship/selection bias)",
                                    "QMJ quality tier not reproducible point-in-time (held constant)",
                                    "box plan completeness not evaluated here"]})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=lambda o: None if o is None else str(o))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
