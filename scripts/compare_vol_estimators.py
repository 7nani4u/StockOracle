"""compare_vol_estimators.py — 일간 σ 추정기(현행 ATR 결합 vs 범위 기반) 사전 고정 워크포워드 비교.

질문: 현행 blended_daily_sigma 의 'ATR/1.596' 항(또는 실현 변동성 전체)을 Parkinson·Garman-Klass·
Rogers-Satchell·Yang-Zhang(market_briefing/range_volatility.py)으로 바꾸면 22거래일 선행 로그수익률의
분포 예측이 더 나아지는가?

사전 고정 프로토콜 (결과를 보기 전에 정한 값 — 바꾸려면 새 파일/새 날짜로 기록한다)
  · 자료: datasets/audit_cache 52종목(KRX 26·US 26) 일봉, NaN 행 제거. 추정 시점은 stride 5, 직전 252봉만 사용.
  · 목표: 시점 t 종가 → t+22 거래일 종가의 로그수익률. 예측 분포: N(0, (σ√22)²) — 서비스의 범위/도달 계산과 같은 가정.
  · 후보: 기준(현행) + {parkinson, garman_klass, rogers_satchell, yang_zhang} × 창 {20, 60} × 방식
          {atr_leg: ATR 항만 교체, short_leg: 실현변동성+ATR 짧은 구간 전체 교체}. 장기(251일) 혼합 50%·하한 0.002 는 동일.
  · 분할: 학습 = 목표 종료일 ≤ 2024-06-30 (여기서만 설정을 고른다), 검증 = 시점 > 2024-06-30.
          학습 구간 선행 목표가 검증 시점과 겹치지 않도록 학습은 목표 종료일 기준으로 자른다(purge).
  · 지표: 평균 음의 로그우도(NLL, 낮을수록 좋음), P10~P90·P05~P95 포함률(명목 80/90%), 예측 σ 대비 이후 22일 실현 σ 비율 중앙값.
  · 불확실성: (종목, 월) 블록 부트스트랩 2,000회, 시드 20261027, 기준 대비 쌍 차이의 95% 구간.
  · 채택 기준(모두 충족해야 서비스 기본값 변경 후보): 학습에서 고른 단일 설정이 검증의 KRX·US 각각에서
          ① ΔNLL<0 ② 부트스트랩 95% 구간 상한<0 ③ 학습 ΔNLL<0(방향 일치) ④ P10~P90 포함률의 80%와의 거리가 기준보다 0.01 넘게 나빠지지 않음.
  · 검증 구간 결과로 설정을 다시 고르지 않는다. 모든 후보의 검증 성적도 JSON 에 남긴다.

한계: 정규·무추세 가정의 분포 점수이며 거래 수익성이나 방향 정확도가 아니다. 종목 간 상관·같은 달의 시장 충격은
월 블록으로만 부분 반영된다. 캐시는 다른 감사에도 쓰였다(완전한 미사용 자료 아님).

사용: python scripts/compare_vol_estimators.py [--stride 5] [--out docs/backtests/vol_estimator_comparison_20261027.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from market_briefing import forecast_model as fm  # noqa: E402
from market_briefing import range_volatility as rv  # noqa: E402

CACHE = ROOT / "datasets" / "audit_cache"
DEFAULT_OUT = ROOT / "docs" / "backtests" / "vol_estimator_comparison_20261027.json"
HORIZON = 22
WINDOW = 252
SPLIT = pd.Timestamp("2024-06-30")
LONG_WEIGHT = 0.5
FLOOR = 0.002
WINDOWS = (20, 60)
MODES = ("atr_leg", "short_leg")
Z10, Z05 = fm.Z_P90, fm.Z_P95
SEED = 20261027
BOOT = 2000


def load_prices(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    return frame


def wilder_atr(frame: pd.DataFrame) -> np.ndarray:
    """서비스(api/index.py add_indicators)와 같은 Wilder ATR14: alpha=1/14, adjust=False, min_periods=14."""
    prev = frame["Close"].shift(1)
    tr = pd.concat([frame["High"] - frame["Low"], (frame["High"] - prev).abs(), (frame["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy(float)


def _log_sigma(closes: np.ndarray, window: int, min_obs: int) -> float | None:
    value, _ = fm.daily_log_volatility(list(closes), window=window, min_obs=min_obs)
    return value


def candidate_sigmas(o, h, l, c, atr, t: int) -> Dict[str, float | None]:
    """시점 t(포함)까지의 봉만 써서 모든 후보 일간 σ 를 계산한다."""
    begin = t - WINDOW + 1
    closes = c[begin:t + 1]
    realized = _log_sigma(closes, 60, 15)
    long_run = _log_sigma(closes, fm.LONG_RUN_WINDOW, fm.LONG_RUN_MIN_OBS)
    last = float(c[t])
    atr_sigma = (atr[t] / last / fm.ATR_TO_SIGMA) if np.isfinite(atr[t]) and atr[t] > 0 else None

    def finish(short: float | None) -> float | None:
        if short is None:
            return None
        sigma = short
        if long_run is not None and realized is not None:
            sigma = math.sqrt((1 - LONG_WEIGHT) * short ** 2 + LONG_WEIGHT * long_run ** 2)
        return max(sigma, FLOOR)

    out: Dict[str, float | None] = {}
    base_short = None
    if realized is not None and atr_sigma is not None:
        base_short = math.sqrt(0.5 * realized ** 2 + 0.5 * atr_sigma ** 2)
    elif realized is not None:
        base_short = realized
    out["baseline"] = finish(base_short)
    for method in rv.METHODS:
        for window in WINDOWS:
            est = rv.daily_sigma(method, o[begin:t + 1], h[begin:t + 1], l[begin:t + 1], c[begin:t + 1],
                                 window=window, min_obs=max(10, window // 2))
            if est is None or realized is None:
                out[f"{method}_{window}_atr_leg"] = None
                out[f"{method}_{window}_short_leg"] = None
                continue
            out[f"{method}_{window}_atr_leg"] = finish(math.sqrt(0.5 * realized ** 2 + 0.5 * est ** 2))
            out[f"{method}_{window}_short_leg"] = finish(est)
    return out


def build_samples(stride: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(str(CACHE / "*.csv"))):
        ticker = os.path.basename(path)[:-4]
        market = "KRX" if ticker.endswith(("_KS", "_KQ")) else "US"
        frame = load_prices(path)
        if len(frame) < WINDOW + HORIZON + 5:
            continue
        o, h, l, c = (frame[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
        dates = frame["Date"].to_numpy()
        atr = wilder_atr(frame)
        for t in range(WINDOW - 1, len(frame) - HORIZON, stride):
            sig = candidate_sigmas(o, h, l, c, atr, t)
            if sig["baseline"] is None:
                continue
            fwd = c[t + 1:t + 1 + HORIZON]
            daily = np.diff(np.log(np.r_[c[t], fwd]))
            row = {"ticker": ticker, "market": market, "date": pd.Timestamp(dates[t]),
                   "end_date": pd.Timestamp(dates[t + HORIZON]), "ret": math.log(c[t + HORIZON] / c[t]),
                   "fwd_daily_sigma": float(daily.std(ddof=1))}
            row.update({f"s::{k}": v for k, v in sig.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def nll(ret: np.ndarray, sigma_daily: np.ndarray) -> np.ndarray:
    s = sigma_daily * math.sqrt(HORIZON)
    return 0.5 * np.log(2 * math.pi * s ** 2) + 0.5 * (ret / s) ** 2


def score(frame: pd.DataFrame, name: str) -> Dict[str, float]:
    sub = frame.dropna(subset=[f"s::{name}"])
    sigma = sub[f"s::{name}"].to_numpy(float)
    ret = sub["ret"].to_numpy(float)
    s = sigma * math.sqrt(HORIZON)
    return {
        "n": int(len(sub)), "nll": float(nll(ret, sigma).mean()),
        "cov_p10_p90": float((np.abs(ret) <= Z10 * s).mean()), "cov_p05_p95": float((np.abs(ret) <= Z05 * s).mean()),
        "median_vol_ratio": float(np.median(sub["fwd_daily_sigma"].to_numpy(float) / sigma)),
    }


def paired_delta(frame: pd.DataFrame, name: str, rng: np.random.Generator) -> Dict[str, float]:
    """기준 대비 ΔNLL(후보−기준)과 (종목, 월) 블록 부트스트랩 95% 구간. 두 값이 모두 있는 시점만 쓴다."""
    sub = frame.dropna(subset=[f"s::{name}", "s::baseline"]).copy()
    if sub.empty:
        return {"n": 0, "delta_nll": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    ret = sub["ret"].to_numpy(float)
    diff = nll(ret, sub[f"s::{name}"].to_numpy(float)) - nll(ret, sub["s::baseline"].to_numpy(float))
    sub["diff"] = diff
    sub["block"] = sub["ticker"] + "|" + sub["date"].dt.strftime("%Y-%m")
    grouped = sub.groupby("block")["diff"].agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    k = len(grouped)
    draws = rng.integers(0, k, size=(BOOT, k))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"n": int(len(sub)), "delta_nll": float(diff.mean()),
            "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5))}


def analyze(samples: pd.DataFrame) -> Dict[str, Any]:
    names = [c[3:] for c in samples.columns if c.startswith("s::")]
    train = samples[samples["end_date"] <= SPLIT]
    test = samples[samples["date"] > SPLIT]
    # 학습: 두 시장을 합친 평균 NLL 이 가장 낮은 후보(기준 포함)만 고른다.
    train_scores = {n: score(train, n) for n in names}
    common = train.dropna(subset=[f"s::{n}" for n in names])
    pooled = {n: float(nll(common["ret"].to_numpy(float), common[f"s::{n}"].to_numpy(float)).mean()) for n in names}
    selected = min(pooled, key=pooled.get)
    rng = np.random.default_rng(SEED)
    report: Dict[str, Any] = {"selected_on_train": selected, "train_pooled_nll_common_rows": pooled,
                              "n_common_train_rows": int(len(common)), "candidates": {}}
    for name in names:
        entry: Dict[str, Any] = {"train": train_scores[name]}
        for market in ("KRX", "US"):
            for label, part in (("train", train), ("test", test)):
                sub = part[part["market"] == market]
                entry[f"{label}_{market}"] = {**score(sub, name), "paired_vs_baseline": paired_delta(sub, name, rng)}
        report["candidates"][name] = entry
    # 채택 판정은 학습에서 고른 설정에만 적용한다.
    verdict: Dict[str, Any] = {"selected": selected, "per_market": {}}
    ok_all = selected != "baseline"
    for market in ("KRX", "US"):
        cand = report["candidates"][selected]
        base = report["candidates"]["baseline"]
        t_pair = cand[f"test_{market}"]["paired_vs_baseline"]
        tr_pair = cand[f"train_{market}"]["paired_vs_baseline"]
        cov_c = abs(cand[f"test_{market}"]["cov_p10_p90"] - 0.80)
        cov_b = abs(base[f"test_{market}"]["cov_p10_p90"] - 0.80)
        checks = {"test_delta_negative": t_pair["delta_nll"] < 0, "test_ci_upper_negative": t_pair["ci_high"] < 0,
                  "train_delta_negative": tr_pair["delta_nll"] < 0, "coverage_not_worse_by_0.01": cov_c <= cov_b + 0.01}
        verdict["per_market"][market] = {"checks": checks, "test_delta_nll": t_pair["delta_nll"],
                                         "test_ci": [t_pair["ci_low"], t_pair["ci_high"]]}
        ok_all = ok_all and all(checks.values())
    verdict["adopt_as_default_candidate"] = bool(ok_all)
    report["verdict"] = verdict
    report["split"] = {"train_target_end_on_or_before": str(SPLIT.date()), "test_start_after": str(SPLIT.date()),
                       "n_train": int(len(train)), "n_test": int(len(test))}
    return report


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--samples-csv", default="", help="시점별 후보 σ·목표 CSV 저장 경로(선택)")
    args = parser.parse_args(argv)
    try:  # Windows 콘솔(cp949)에서도 한글 요약이 깨지지 않게 한다
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    samples = build_samples(args.stride)
    if samples.empty:
        print("캐시에서 표본을 만들지 못했다:", CACHE)
        return 2
    if args.samples_csv:
        samples.to_csv(args.samples_csv, index=False)
    report = analyze(samples)
    report["meta"] = {"protocol": __doc__.split("사전 고정 프로토콜")[0].strip().splitlines()[0],
                      "horizon_days": HORIZON, "stride": args.stride, "tickers": int(samples["ticker"].nunique()),
                      "date_min": str(samples["date"].min().date()), "date_max": str(samples["date"].max().date()),
                      "bootstrap": BOOT, "seed": SEED}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    verdict = report["verdict"]
    print(f"학습에서 고른 설정: {verdict['selected']}  채택 후보 여부: {verdict['adopt_as_default_candidate']}")
    for market, info in verdict["per_market"].items():
        print(f"  {market}: 검증 ΔNLL={info['test_delta_nll']:+.4f} 95%={info['test_ci'][0]:+.4f}~{info['test_ci'][1]:+.4f} {info['checks']}")
    base = report["candidates"]["baseline"]
    print("기준 검증:", {m: {k: round(v, 4) for k, v in base[f'test_{m}'].items() if isinstance(v, float)} for m in ("KRX", "US")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
