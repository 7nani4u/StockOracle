"""장중 수정 예측(나우캐스트) 검증 — 야후 5분봉 60거래일 × 코스피·코스닥.

규칙(실행 전 고정): 날짜 순으로 앞 절반 표본에서 scale ∈ {0.6,0.8,1.0,1.2,1.4} 중 Brier 최소를 고르고,
뒤 절반에서 한 번만 보고한다. 비교 기준: 현재 부호(r_now>0 이면 상승) 와 장 시작 전 확률(없음 → 0.5).
확인 시점: 09:30 10:00 11:00 12:00 13:00 14:00 14:30 (해당 5분봉 종가 = 그 시각 지수).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from market_briefing import kr_direction_model as km  # noqa: E402
from market_briefing import krx_calendar as kc  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "datasets", "weekly_forecast_cache")
OUT_JSON = os.path.join(ROOT, "docs", "backtests", "weekly_forecast_intraday_20261104.json")
CHECKPOINTS = [(9, 30), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (14, 30)]
SCALES = (0.6, 0.8, 1.0, 1.2, 1.4)


def _load_daily(code):
    raw = json.load(open(os.path.join(CACHE_DIR, f"{code}.json"), encoding="utf-8"))
    raw["dates"] = [dt.date.fromisoformat(d) for d in raw["dates"]]
    return raw


def collect(code: str, symbol: str):
    import yfinance as yf
    daily = _load_daily(code)
    pos = {d: i for i, d in enumerate(daily["dates"])}
    frame = yf.Ticker(symbol).history(period="60d", interval="5m")
    frame = frame.dropna(subset=["Close"])
    rows = []
    for day, grp in frame.groupby(frame.index.date):
        if day not in pos or pos[day] < 60:
            continue
        i = pos[day]
        prev_close = daily["close"][i - 1]
        final = np.log(daily["close"][i] / prev_close)
        if final == 0:
            continue
        hist = {k: (v[:i] if isinstance(v, list) else v) for k, v in daily.items()}
        sig = km.session_sigma(hist)
        open_t, close_t = kc.session_hours(day)
        total = (close_t - open_t).total_seconds() / 60.0
        for hh, mm in CHECKPOINTS:
            stamp = dt.datetime.combine(day, dt.time(hh, mm), tzinfo=kc.KST)
            bar_start = stamp - dt.timedelta(minutes=5)
            sel = grp[(grp.index >= bar_start) & (grp.index < stamp)]
            if sel.empty or stamp <= open_t:
                continue
            r_now = float(np.log(sel["Close"].iloc[-1] / prev_close))
            left = (close_t - stamp).total_seconds() / 60.0
            rows.append({"date": day, "cp": f"{hh:02d}:{mm:02d}", "r_now": r_now, "left": left,
                         "total": total, "sigma": sig, "y": 1.0 if final > 0 else 0.0})
    return rows


def evaluate(rows, scale):
    p = np.asarray([km.intraday_prob_up(r["r_now"], r["sigma"], r["left"], r["total"], scale)["p"] for r in rows])
    y = np.asarray([r["y"] for r in rows])
    return p, y


def main() -> int:
    out = {"generated_at": dt.datetime.now(kc.KST).isoformat(), "checkpoints": [f"{h:02d}:{m:02d}" for h, m in CHECKPOINTS],
           "indexes": {}}
    for code, symbol in (("KOSPI", "^KS11"), ("KOSDAQ", "^KQ11")):
        rows = collect(code, symbol)
        days = sorted({r["date"] for r in rows})
        cut = days[len(days) // 2]
        first = [r for r in rows if r["date"] < cut]
        second = [r for r in rows if r["date"] >= cut]
        fit = {}
        for s in SCALES:
            p, y = evaluate(first, s)
            fit[s] = float(np.mean((p - y) ** 2))
        scale = min(fit, key=fit.get)
        p, y = evaluate(second, scale)
        sign_p = np.asarray([1.0 if r["r_now"] > 0 else 0.0 for r in second])
        per_cp = {}
        for cp in out["checkpoints"]:
            m = np.asarray([r["cp"] == cp for r in second])
            if m.sum() < 5:
                continue
            per_cp[cp] = {
                "n": int(m.sum()),
                "brier": round(float(np.mean((p[m] - y[m]) ** 2)), 4),
                "accuracy": round(float(((p[m] >= 0.5) == (y[m] == 1)).mean()), 4),
                "sign_rule_accuracy": round(float((sign_p[m] == y[m]).mean()), 4),
                "brier_coinflip": 0.25,
            }
        out["indexes"][code] = {
            "days": len(days), "fit_days": [days[0].isoformat(), (cut - dt.timedelta(days=1)).isoformat()],
            "test_days": [cut.isoformat(), days[-1].isoformat()],
            "scale_fit_brier": {str(k): round(v, 4) for k, v in fit.items()},
            "selected_scale": scale,
            "test_brier": round(float(np.mean((p - y) ** 2)), 4),
            "test_calibration": km.calibration_bins(p, y),
            "per_checkpoint": per_cp,
        }
        print(code, "scale", scale, "test brier", out["indexes"][code]["test_brier"], per_cp)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
