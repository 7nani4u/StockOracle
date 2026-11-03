"""fit_probability_calibration.py — 방향 확률 보정 파일(models/probability_calibration.json) 생성.

입력은 ``scripts/audit_prediction_layers.py --rows-csv`` 가 만든 행 단위 결과다. 각 행은 과거 시점 t 의
표시 상승 확률(``prob_final``, 상관 보정 후·시나리오 전)과 이후 22거래일의 실제 결과(``fwd_ret``)를 담는다.

무엇을 하나
  1. 시장 합산 기울기와 시장별 절편을 전체 표본으로 추정한다(market_briefing.probability_calibration).
  2. 시간 분할(앞 구간 학습 → 뒤 구간 검증)로 표본 밖 Brier·ECE 를 계산해 보정 전·기저율 상수와 비교한다.
  3. 기울기의 95% 구간을 (종목, 월) 블록 부트스트랩으로 구한다(22거래일 구간이 겹치므로).
  4. 위 내용을 models/probability_calibration.json 에 기록한다. 기울기 구간이 0 을 포함하면 노트에 명시한다.

사용 예
    python scripts/audit_prediction_layers.py --offline --stride 6 --rows-csv /tmp/audit_rows.csv
    python scripts/fit_probability_calibration.py --rows-csv /tmp/audit_rows.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

from market_briefing.probability_calibration import fit_direction_calibration

DEFAULT_OUT = os.path.join(REPO_ROOT, "models", "probability_calibration.json")
PROB_COLUMN = "prob_final"


def _brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def _ece(y, p, bins: int = 10) -> float:
    frame = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    frame["q"] = pd.qcut(frame["p"], bins, duplicates="drop")
    grouped = frame.groupby("q", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return float((grouped["n"] * (grouped["p"] - grouped["y"]).abs()).sum() / grouped["n"].sum())


def _apply(fit: dict, probs_pct, markets) -> np.ndarray:
    slope = fit["slope"]
    intercepts = fit["intercept_at_half"]
    return np.array([
        min(0.95, max(0.05, intercepts.get("KRX" if str(m) == "KRX" else "US", intercepts["ALL"])
                      + slope * (float(p) / 100.0 - 0.5)))
        for p, m in zip(probs_pct, markets)])


def build_calibration(rows: pd.DataFrame, split: str = "", boot: int = 300, seed: int = 7) -> dict:
    frame = rows.dropna(subset=[PROB_COLUMN, "fwd_ret"]).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["up"] = (frame["fwd_ret"] > 0).astype(int)
    fit = fit_direction_calibration(frame[PROB_COLUMN], frame["up"], frame["market"])

    dates = np.sort(frame["date"].unique())
    split_date = pd.Timestamp(split) if split else pd.Timestamp(dates[int(len(dates) * 0.45)])
    train, test = frame[frame["date"] < split_date], frame[frame["date"] >= split_date]
    oos = {"split": str(split_date.date()), "train_rows": int(len(train)), "test_rows": int(len(test))}
    if len(train) > 200 and len(test) > 200:
        train_fit = fit_direction_calibration(train[PROB_COLUMN], train["up"], train["market"])
        calibrated = _apply(train_fit, test[PROB_COLUMN], test["market"])
        raw = test[PROB_COLUMN].to_numpy(float) / 100.0
        const = np.array([train_fit["base_up_rate"].get("KRX" if m == "KRX" else "US", train_fit["base_up_rate"]["ALL"])
                          for m in test["market"]])
        y = test["up"].to_numpy()
        oos.update({
            "train_slope": train_fit["slope"],
            "brier_raw": round(_brier(y, raw), 4), "brier_calibrated": round(_brier(y, calibrated), 4),
            "brier_base_rate": round(_brier(y, const), 4),
            "ece_raw": round(_ece(y, raw), 4), "ece_calibrated": round(_ece(y, calibrated), 4),
        })

    rng = np.random.default_rng(seed)
    blocks = frame.assign(_m=frame["date"].astype(str).str[:7]).groupby(["ticker", "_m"]).indices
    keys = list(blocks)
    slopes = []
    for _ in range(boot):
        picks = rng.choice(len(keys), len(keys), replace=True)
        sample = frame.iloc[np.concatenate([blocks[keys[i]] for i in picks])]
        slopes.append(fit_direction_calibration(sample[PROB_COLUMN], sample["up"], sample["market"])["slope"])
    low, high = (float(np.percentile(slopes, q)) for q in (2.5, 97.5))

    notes = [
        "p_cal = intercept_at_half[market] + slope * (p_raw - 0.5), clip_pct 로 제한. 입력은 상관 보정 후 prob_up.",
        "시장별 기울기는 표본 밖에서 부호가 뒤집혀 채택하지 않고 시장 합산 기울기 + 시장별 기저율 절편을 쓴다.",
    ]
    if low <= 0.0 <= high:
        notes.append("기울기의 95% 구간이 0 을 포함한다: 규칙 점수가 22거래일 방향을 구분한다는 증거가 없고, "
                     "보정 후 확률은 시장별 기저 상승률 부근에 모인다.")
    return {
        "version": time.strftime("%Y-%m-%d"),
        "target": "close(t+22 sessions) > close(t)",
        "horizon_sessions": 22,
        "input": f"route prob_up after correlation adjustments ({PROB_COLUMN})",
        "method": "linear_shrink_market_intercept",
        "slope": fit["slope"],
        "intercept_at_half": fit["intercept_at_half"],
        "clip_pct": [5.0, 95.0],
        "slope_bootstrap_ci95": [round(low, 4), round(high, 4)],
        "fitted_on": {
            "rows": fit["rows"], "tickers": int(frame["ticker"].nunique()),
            "period": [str(frame["date"].min().date()), str(frame["date"].max().date())],
            "base_up_rate": fit["base_up_rate"],
        },
        "oos": oos,
        "notes": notes,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows-csv", required=True, help="audit_prediction_layers.py --rows-csv 결과")
    parser.add_argument("--out", default=DEFAULT_OUT, help="보정 JSON 경로")
    parser.add_argument("--split", default="", help="학습/검증 분할일(YYYY-MM-DD), 비우면 날짜 분포의 45%% 지점")
    args = parser.parse_args(argv)

    rows = pd.read_csv(args.rows_csv)
    payload = build_calibration(rows, split=args.split)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
