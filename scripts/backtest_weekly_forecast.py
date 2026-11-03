"""코스피·코스닥 거래일 방향 예측 walk-forward 백테스트 (사전 고정 프로토콜).

프로토콜(실행 전 고정 — docs/weekly_forecast_20261104.md 와 동일)
- 데이터: 네이버 일봉 3,000개(2014-08~) + 야후 해외 일봉(^GSPC ^IXIC ^SOX ^VIX ^TNX).
- 확장 창 walk-forward, 21거래일마다 재학습, 최소 학습 750행. 학습 표본은 예측 앵커 이전에
  정답이 확정된 행만(kr_direction_model.train_mask).
- 개발 구간(대상일 ~2021-12-31)에서만 후보를 고른다: 지수·예측 거리(h)별 Brier 최소.
  기준 모델(학습 기저 상승률)도 후보다 — 아무것도 이기지 못하면 기준 모델을 쓴다.
- 보류 구간(2022-01-01~)은 선택이 끝난 뒤 한 번만 보고한다. 보류 구간 결과로 다시 고르지 않는다.
- 판단 유보 범위는 |p-0.5| <= 0.05 로 고정.

사용:
  python scripts/backtest_weekly_forecast.py            # 네트워크 수집 후 캐시 저장
  python scripts/backtest_weekly_forecast.py --offline  # datasets/weekly_forecast_cache 만 사용
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from market_briefing import index_series as ix  # noqa: E402
from market_briefing import kr_direction_model as km  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "datasets", "weekly_forecast_cache")
OUT_JSON = os.path.join(ROOT, "docs", "backtests", "weekly_forecast_backtest_20261104.json")
DEV_END = dt.date(2021, 12, 31)
HOLD_BAND = 0.05
REFIT = 21
MIN_TRAIN = 750


def _ser(obj):
    if isinstance(obj, dict):
        return {k: _ser(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ser(v) for v in obj]
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    return obj


def _deser(obj):
    out = dict(obj)
    out["dates"] = [dt.date.fromisoformat(d) for d in obj["dates"]]
    return out


def load_data(offline: bool) -> tuple[dict, dict]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    kr, gl = {}, {}
    for code in ("KOSPI", "KOSDAQ"):
        path = os.path.join(CACHE_DIR, f"{code}.json")
        if offline:
            kr[code] = _deser(json.load(open(path, encoding="utf-8")))
            continue
        data, note, _ = ix.fetch_kr_index_daily(code, count=3000, force=True)
        if data is None:
            raise SystemExit(f"{code} 수집 실패: {note}")
        json.dump(_ser(data), open(path, "w", encoding="utf-8"))
        kr[code] = data
    for key in ix.GLOBAL_SYMBOLS:
        path = os.path.join(CACHE_DIR, f"{key}.json")
        if offline:
            gl[key] = _deser(json.load(open(path, encoding="utf-8")))
            continue
        data, note, _ = ix.fetch_global_daily(key, period="15y", force=True)
        if data is None:
            raise SystemExit(f"{key} 수집 실패: {note}")
        # 백테스트 시점에 미확정인 미국 막대는 제외(확정 시각 기준)
        data, _ = ix.drop_incomplete_sessions(data, dt.datetime.now(dt.timezone.utc))
        json.dump(_ser(data), open(path, "w", encoding="utf-8"))
        gl[key] = data
    return kr, gl


def _candidates(h: int):
    """(이름, 특징 집합, 모델 생성자)"""
    out = []
    sets = ["overnight_full", "overnight_compact", "kr_only"] if h == 1 else ["kr_only"]
    for fs in sets:
        for alpha in (3.0, 30.0, 300.0):
            for prob in ("empirical", "gauss"):
                out.append((f"ridge[{fs},a={alpha:g},{prob}]", fs,
                            (lambda a=alpha, pr=prob: km.RidgeResidualModel(alpha=a, prob=pr))))
        if fs != "overnight_compact":
            out.append((f"logit[{fs},a=30]", fs, (lambda: km.LogisticModel(alpha=30.0))))
    return out


def _segment_masks(ds: km.Dataset, preds_ok: np.ndarray):
    dates = np.asarray(ds.target_dates)
    dev = preds_ok & np.asarray([d <= DEV_END for d in dates])
    hold = preds_ok & np.asarray([d > DEV_END for d in dates])
    return dev, hold


def evaluate_index(code: str, kr: dict, gl: dict, log) -> dict:
    frame = km.kr_anchor_frame(kr)
    result: dict = {"index": code, "horizons": {}}
    for h in (1, 2, 3, 4, 5):
        t0 = time.time()
        datasets = {fs: km.build_dataset(kr, gl, horizon=h, feature_set=fs, frame=frame)
                    for fs in (["overnight_full", "overnight_compact", "kr_only"] if h == 1 else ["kr_only"])}
        # 모든 후보가 같은 대상일 집합에서 비교되도록 공통 대상일로 맞춘다
        common = set.intersection(*[set(d.target_dates) for d in datasets.values()])
        for fs, d in list(datasets.items()):
            keep = np.asarray([t in common for t in d.target_dates])
            datasets[fs] = _subset(d, keep)
        ref = datasets["kr_only"]
        start_row = 0
        base = km.baseline_walk_forward(ref, "base", start_row, REFIT, MIN_TRAIN)
        ok = np.isfinite(base["p"])
        dev, hold = _segment_masks(ref, ok)
        rows = {}

        def score(name, pred, ds):
            mu = pred["zhat"] * ds.sigma
            lo = pred.get("q10")
            hi = pred.get("q90")
            entry = {}
            for seg, m in (("development", dev), ("holdout", hold)):
                p = pred["p"].copy()
                p[~m] = np.nan
                entry[seg] = km.direction_metrics(p, ds.y, base["p"], HOLD_BAND)
                entry[seg]["returns"] = km.return_metrics(
                    np.where(m, mu, np.nan), ds.ret,
                    None if lo is None else lo * ds.sigma, None if hi is None else hi * ds.sigma)
            rows[name] = entry

        score("baseline:base_rate", base, ref)
        persist = km.baseline_walk_forward(ref, "conditional", start_row, REFIT, MIN_TRAIN,
                                           condition=ref.X[:, ref.feature_names.index("kr_r1")])
        score("baseline:persistence", persist, ref)
        if h == 1:
            dsf = datasets["overnight_full"]
            us = km.baseline_walk_forward(dsf, "conditional", start_row, REFIT, MIN_TRAIN,
                                          condition=dsf.X[:, dsf.feature_names.index("on_spx")])
            score("baseline:us_direction", us, dsf)
        preds_store = {"baseline:base_rate": (base, ref)}
        for name, fs, factory in _candidates(h):
            pred = km.walk_forward(datasets[fs], factory, start_row, REFIT, MIN_TRAIN)
            score(name, pred, datasets[fs])
            preds_store[name] = (pred, datasets[fs])
        # 개발 구간 Brier 최소 후보 선택(기준 모델 포함)
        def dev_brier(name):
            return rows[name]["development"].get("brier") or 9.0
        eligible = [n for n in rows if not n.startswith("baseline:") or n == "baseline:base_rate"]
        selected = min(eligible, key=dev_brier)
        sel_pred, sel_ds = preds_store[selected]
        hold_p = sel_pred["p"].copy()
        hold_p[~hold] = np.nan
        detail = {
            "by_phase": km.sliced_metrics(hold_p, sel_ds.y, sel_ds.phase, base["p"]),
            "by_high_vol": km.sliced_metrics(hold_p, sel_ds.y,
                                             ["변동성 확대" if v else "평상" for v in sel_ds.high_vol], base["p"]),
            "by_weekday": km.sliced_metrics(hold_p, sel_ds.y,
                                            ["월화수목금토일"[t.weekday()] for t in sel_ds.target_dates], base["p"]),
            "by_year": km.sliced_metrics(hold_p, sel_ds.y, [t.year for t in sel_ds.target_dates], base["p"]),
        }
        result["horizons"][str(h)] = {
            "n_rows": len(ref),
            "development_range": _range(ref, dev),
            "holdout_range": _range(ref, hold),
            "selected": selected,
            "candidates": rows,
            "selected_holdout_detail": detail,
        }
        log(f"{code} h={h} rows={len(ref)} selected={selected} "
            f"dev brier={rows[selected]['development'].get('brier')} "
            f"hold acc={rows[selected]['holdout'].get('accuracy')} "
            f"base acc={rows['baseline:base_rate']['holdout'].get('majority_accuracy')} "
            f"hold brier={rows[selected]['holdout'].get('brier')} vs base {rows['baseline:base_rate']['holdout'].get('brier')} "
            f"({time.time()-t0:.1f}s)")
    return result


def _range(ds, m):
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        return None
    return [ds.target_dates[idx[0]].isoformat(), ds.target_dates[idx[-1]].isoformat(), int(len(idx))]


def _subset(ds: km.Dataset, keep: np.ndarray) -> km.Dataset:
    sel = np.flatnonzero(keep)
    return km.Dataset(
        feature_names=ds.feature_names, X=ds.X[sel], z=ds.z[sel], y=ds.y[sel], ret=ds.ret[sel],
        sigma=ds.sigma[sel], target_pos=ds.target_pos[sel], anchor_pos=ds.anchor_pos[sel],
        target_dates=[ds.target_dates[i] for i in sel], anchor_dates=[ds.anchor_dates[i] for i in sel],
        phase=[ds.phase[i] for i in sel], high_vol=ds.high_vol[sel], horizon=ds.horizon, meta=ds.meta)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args(argv)
    kr, gl = load_data(args.offline)
    lines: list[str] = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    summary = {
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),
        "protocol": {
            "walk_forward": f"expanding, refit every {REFIT} sessions, min train {MIN_TRAIN}",
            "development_end": DEV_END.isoformat(),
            "selection": "per index & horizon: minimum development Brier among candidates incl. base rate",
            "hold_band": HOLD_BAND,
        },
        "data": {code: {"first": kr[code]["dates"][0].isoformat(), "last": kr[code]["dates"][-1].isoformat(),
                        "n": len(kr[code]["dates"]), "source": kr[code].get("source")} for code in kr},
        "globals": {k: {"first": v["dates"][0].isoformat(), "last": v["dates"][-1].isoformat(),
                        "n": len(v["dates"])} for k, v in gl.items()},
        "indexes": {},
    }
    for code in ("KOSPI", "KOSDAQ"):
        summary["indexes"][code] = evaluate_index(code, kr[code], gl, log)
    summary["log"] = lines
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
