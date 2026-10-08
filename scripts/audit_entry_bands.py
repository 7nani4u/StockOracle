"""audit_entry_bands.py — 예측 탭 1차·2차 매수 구간의 워크포워드 감사.

무엇을 보나
  각 종목·각 과거 시점 t 에서 서비스와 같은 입력(최근 252봉 + 인과적 지표)으로 calc_buy_price 를 다시 실행하고
    1. 불변식: 1차(aggressive)·2차(recommended) 가족이 항상 3밴드×5단계로 나오는지, 단계 가격이 서로 다르고
       밴드 안에서 내림차순인지, 밴드 간 순서(A>B>C)와 2차<1차 분리가 지켜지는지
    2. 단계별 도달 확률 보정: 표시 확률(reach_probability_pct)과 이후 30거래일 안에 저가가 단계 가격에 닿은
       실제 빈도의 Brier·AUC·신뢰도 곡선
    3. 예상 기간: 닿은 경우 첫 도달일이 [days_min, days_max] 에 들어온 비율
  을 확인한다.

범위와 한계
  · route 의 글루(지표 계산 → 점수 → calc_buy_price)를 같은 함수로 재현하지만 투자자 수급·뉴스·실시간 시세는 없다.
  · 같은 종목의 인접 시점 결과는 겹치므로 종목 단위 부트스트랩 없이 점추정만 낸다(표본 19만 단계).
  · 학습 로그는 건드리지 않는다(STOCKORACLE_PREDICTION_LOG 를 임시 경로로 고정).

사용 예
    python scripts/audit_entry_bands.py --offline --stride 8
    # 병렬: 8조각으로 나눠 실행한 뒤 합쳐서 요약
    python scripts/audit_entry_bands.py --offline --shard 0/8 --rows-csv /tmp/entry_0.csv --out /tmp/entry_0.json   # ... 7까지
    python scripts/audit_entry_bands.py --merge "/tmp/entry_?.csv"
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("STOCKORACLE_PREDICTION_LOG", os.path.join(tempfile.gettempdir(), "stockoracle_audit_learning.jsonl"))

import numpy as np
import pandas as pd

from scripts.audit_prediction_layers import (DEFAULT_TICKERS, WINDOW, _NO_EVENT, _NO_LEARNING, _as_dd, auc, brier,
                                              load_prices)

HORIZON = 30            # 단계 도달 확률·기간의 기준 기간(거래일) — calc_buy_price 의 _STEP_HORIZON_DAYS 와 같다
DEFAULT_OUT = os.path.join(REPO_ROOT, "docs", "backtests", "entry_band_audit_summary.json")
BINS = [-0.1, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100.1]


# ── 불변식 ────────────────────────────────────────────────────────────────

def check_family(bands: List[Dict[str, Any]]) -> List[str]:
    """한 가족(A/B/C)이 어긴 불변식 이름 목록. 비어 있으면 모두 지킨 것이다."""
    problems: List[str] = []
    if [band.get("band") for band in bands] != ["A", "B", "C"]:
        problems.append("band_count")
    for band in bands:
        steps = band.get("steps") or []
        if not band.get("is_available") or not band.get("range"):
            problems.append("withheld")
            continue
        if len(steps) != 5:
            problems.append("step_count")
            continue
        low, high = band["range"]
        prices = [step["price"] for step in steps]
        if len(set(prices)) != len(prices):
            problems.append("dup_step_price")
        if any(not (low - 1e-9 <= price <= high + 1e-9) for price in prices):
            problems.append("step_outside_band")
        if prices != sorted(prices, reverse=True):
            problems.append("step_unordered")
        if any(a["price_range"][0] < b["price_range"][1] for a, b in zip(steps, steps[1:])):
            problems.append("step_range_overlap")
        probabilities = [step.get("reach_probability_pct") for step in steps]
        if any(value is None for value in probabilities):
            problems.append("prob_missing")
        elif probabilities != sorted(probabilities, reverse=True):
            problems.append("prob_not_monotonic")
        days = [(step.get("days_min"), step.get("days_max")) for step in steps]
        if any(a is None or b is None or not (1 <= a <= b <= HORIZON) for a, b in days):
            problems.append("days_invalid")
    for upper, lower in zip(bands, bands[1:]):
        if upper.get("range") and lower.get("range"):
            if not (upper["range"][0] > lower["range"][0] and upper["range"][1] > lower["range"][1]):
                problems.append("band_order")
            if any(not (a["price"] > b["price"]) for a, b in zip(upper.get("steps") or [], lower.get("steps") or [])):
                problems.append("cross_band_step_order")
    return sorted(set(problems))


# ── route 글루 재현 ───────────────────────────────────────────────────────

def evaluate_ticker(ix, ticker: str, prices: pd.DataFrame, stride: int, min_bars: int = 0
                    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(단계 행, 시점 행). 단계 행에는 표시 확률·기간과 선행 30거래일의 실제 도달 결과가 함께 들어간다."""
    market = "KRX" if ticker.endswith((".KS", ".KQ")) else "US"
    frame = ix.add_indicators(prices.copy(), market=market)
    full = _as_dd(frame)
    closes = frame["Close"].to_numpy(float)
    lows = frame["Low"].to_numpy(float)
    helpers = ix._load_forecast_helpers()
    step_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    start = min_bars if min_bars else WINDOW
    for t in range(start, len(frame) - 1, stride):
        begin = max(0, t - WINDOW + 1)
        dd = {key: values[begin:t + 1] for key, values in full.items()}
        last, prev = float(closes[t]), float(closes[t - 1])
        atr = dd["ATR"][-1]
        if atr is None or not math.isfinite(atr) or atr <= 0:
            atr = last * 0.02
        try:
            score = ix.analyze_score(dd, market, "1y")[0]
            indicators = ix.calc_indicator_signals(dd, market=market)
            weekly = ix.build_weekly_analysis_context(dd, market)
            buy = ix.calc_buy_price(dd, last, atr, score, indicators, market, "1y", _NO_EVENT, _NO_LEARNING, "NEUTRAL",
                                    prev, (last - prev) / prev * 100.0, arty_dd=dd, weekly_context=weekly)
        except Exception as exc:  # 한 시점의 실패가 감사 전체를 멈추지 않게 한다
            print(f"[entry-audit] {ticker} {dd['Date'][-1]} 건너뜀: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            continue
        aggressive, recommended = buy["aggressive_bands"], buy["recommended_bands"]
        structure = buy["price_structure"]
        separated = bool(aggressive and recommended) and (
            max(b["range"][1] for b in recommended) < min(b["range"][0] for b in aggressive))
        sample_rows.append({
            "ticker": ticker, "market": market, "date": dd["Date"][-1], "bars": len(dd["Close"]),
            "context": buy["strategy_rec"]["context"], "atr_pct": buy["atr_pct"],
            "agg_problems": "|".join(check_family(aggressive)), "rec_problems": "|".join(check_family(recommended)),
            "separated": separated,
            "relaxed": bool((structure.get("ordering_relaxed") or {}).get("aggressive")
                            or (structure.get("ordering_relaxed") or {}).get("recommended")),
        })
        sigma = (helpers["blended_daily_sigma"](list(dd["Close"]), last, atr, True) or {}).get("sigma")
        future = lows[t + 1:t + 1 + HORIZON]
        complete = len(future) >= HORIZON
        for family, bands in (("agg", aggressive), ("rec", recommended)):
            for band in bands:
                for index, step in enumerate(band.get("steps") or []):
                    price = float(step["price"])
                    hits = np.nonzero(future <= price)[0] if complete else np.array([], dtype=int)
                    reference = helpers["touch_probability"](last, price, sigma, HORIZON) if sigma else None
                    step_rows.append({
                        "ticker": ticker, "market": market, "date": dd["Date"][-1], "family": family, "band": band["band"],
                        "step": index + 1, "evidence": band.get("evidence_level"),
                        "p": step.get("reach_probability_pct"), "p_low": step.get("probability_low_pct"),
                        "p_high": step.get("probability_high_pct"), "reference_p": None if reference is None else reference * 100.0,
                        "days_min": step.get("days_min"), "days_max": step.get("days_max"),
                        "complete": complete, "touched": int(hits.size > 0) if complete else None,
                        "first_day": int(hits[0]) + 1 if hits.size else None,
                    })
    return step_rows, sample_rows


# ── 요약 ──────────────────────────────────────────────────────────────────

def _reliability(frame: pd.DataFrame, column: str) -> List[Dict[str, float]]:
    cuts = pd.cut(frame[column], bins=BINS)
    table = frame.groupby(cuts, observed=True).agg(n=("touched", "size"), predicted=(column, "mean"), realized=("touched", "mean"))
    return [{"bin": str(index), "n": int(row.n), "predicted_pct": round(float(row.predicted), 1),
             "realized_pct": round(float(row.realized) * 100.0, 1)} for index, row in table.iterrows()]


def summarize(step_rows: List[Dict[str, Any]], sample_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = pd.DataFrame(step_rows)
    samples = pd.DataFrame(sample_rows)
    if samples.empty:
        return {"samples": 0}
    out: Dict[str, Any] = {
        "samples": int(len(samples)), "tickers": int(samples["ticker"].nunique()),
        "period": [str(samples["date"].min()), str(samples["date"].max())], "horizon_sessions": HORIZON,
        "invariants": {
            "samples_with_agg_problem": int((samples["agg_problems"] != "").sum()),
            "samples_with_rec_problem": int((samples["rec_problems"] != "").sum()),
            "samples_not_separated": int((~samples["separated"].astype(bool)).sum()),
            "samples_order_relaxed": int(samples["relaxed"].astype(bool).sum()),
            "problem_counts": dict(Counter(name for text in samples["agg_problems"].tolist() + samples["rec_problems"].tolist()
                                           for name in str(text).split("|") if name and name != "nan")),
        },
        "steps_per_sample": steps.groupby(["ticker", "date"]).size().value_counts().to_dict(),
        "evidence_mix": steps.drop_duplicates(["ticker", "date", "family", "band"])["evidence"].value_counts(normalize=True).round(4).to_dict(),
    }
    done = steps[steps["complete"] == True].copy()  # noqa: E712
    if done.empty:
        return out
    done["touched"] = done["touched"].astype(int)
    y = done["touched"].to_numpy()
    p = done["p"].to_numpy(float) / 100.0
    base = float(y.mean())
    out["reach_probability"] = {
        "n": int(len(done)), "realized_touch_rate": round(base, 4), "mean_predicted": round(float(p.mean()), 4),
        "brier": round(brier(y, p), 4), "brier_constant": round(brier(y, np.full_like(p, base)), 4),
        "auc": round(auc(y, p), 4), "reliability": _reliability(done, "p"),
    }
    reference = done.dropna(subset=["reference_p"])
    if len(reference):
        ry = reference["touched"].to_numpy()
        rp = reference["reference_p"].to_numpy(float) / 100.0
        out["reach_probability"]["reference_zero_drift_model"] = {
            "brier": round(brier(ry, rp), 4), "auc": round(auc(ry, rp), 4)}
    by_depth = done.groupby(["family", "band"]).agg(n=("touched", "size"), predicted=("p", "mean"), realized=("touched", "mean"))
    out["by_band"] = {f"{family}{band}": {"n": int(row.n), "predicted_pct": round(float(row.predicted), 1),
                                          "realized_pct": round(float(row.realized) * 100.0, 1)}
                      for (family, band), row in by_depth.iterrows()}
    touched = done[(done["touched"] == 1) & done["first_day"].notna() & done["days_min"].notna()]
    if len(touched):
        inside = (touched["first_day"] >= touched["days_min"]) & (touched["first_day"] <= touched["days_max"])
        out["expected_period"] = {
            "n_touched": int(len(touched)), "inside_window": round(float(inside.mean()), 4),
            "before_window": round(float((touched["first_day"] < touched["days_min"]).mean()), 4),
            "after_window": round(float((touched["first_day"] > touched["days_max"]).mean()), 4),
            "mean_window_width_days": round(float((touched["days_max"] - touched["days_min"]).mean()), 2),
            "median_first_day": float(touched["first_day"].median()),
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride", type=int, default=8, help="평가 시점 간격(거래일), 기본 8")
    parser.add_argument("--tickers", default="", help="쉼표 구분 종목, 기본은 대형·중형·소형 52종목")
    parser.add_argument("--limit", type=int, default=0, help="종목 수 제한(빠른 확인용)")
    parser.add_argument("--min-bars", type=int, default=0, help="이 봉 수부터 평가(짧은 이력·신규상장 점검용), 기본은 252")
    parser.add_argument("--shard", default="0/1", help="i/n — 종목을 n 조각으로 나눠 i 번째만 실행(병렬용)")
    parser.add_argument("--offline", action="store_true", help="캐시만 사용")
    parser.add_argument("--out", default=DEFAULT_OUT, help="요약 JSON 경로")
    parser.add_argument("--rows-csv", default="", help="단계 행 결과 CSV 경로(선택). 시점 행은 같은 이름의 .samples.csv 로 함께 저장")
    parser.add_argument("--merge", default="", help="평가 없이 --rows-csv 조각들(glob)을 합쳐 요약만 만든다")
    args = parser.parse_args(argv)

    if args.merge:
        step_files = sorted(f for f in glob.glob(args.merge) if not f.endswith(".samples.csv"))
        if not step_files:
            print(f"[entry-audit] 합칠 파일이 없습니다: {args.merge}")
            return 1
        step_rows = pd.concat([pd.read_csv(f) for f in step_files], ignore_index=True).to_dict("records")
        sample_rows = pd.concat([pd.read_csv(f[:-4] + ".samples.csv") for f in step_files], ignore_index=True).fillna("").to_dict("records")
        summary = summarize(step_rows, sample_rows)
        summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        summary["merged_files"] = len(step_files)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2)[:6000])
        print(f"\n요약 저장: {args.out}")
        return 0

    from api import index as ix

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] or DEFAULT_TICKERS["KRX"] + DEFAULT_TICKERS["US"]
    if args.limit:
        tickers = tickers[:args.limit]
    shard, shards = (int(part) for part in args.shard.split("/"))
    tickers = tickers[shard::shards]

    step_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    for number, ticker in enumerate(tickers, 1):
        prices = load_prices(ticker, args.offline)
        needed = (args.min_bars or WINDOW) + 5
        if prices is None or len(prices) < needed:
            print(f"[entry-audit] {ticker} 건너뜀 (일봉 부족/없음)", flush=True)
            continue
        steps, samples = evaluate_ticker(ix, ticker, prices, args.stride, args.min_bars)
        step_rows.extend(steps)
        sample_rows.extend(samples)
        print(f"[entry-audit] {number}/{len(tickers)} {ticker} 누적 시점 {len(sample_rows)}", flush=True)

    summary = summarize(step_rows, sample_rows)
    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["stride"] = args.stride
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    if args.rows_csv:
        pd.DataFrame(step_rows).to_csv(args.rows_csv, index=False)
        pd.DataFrame(sample_rows).to_csv(args.rows_csv[:-4] + ".samples.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:6000])
    print(f"\n요약 저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
