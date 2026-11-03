"""층별 워크포워드 감사 스크립트의 지표·합성 데이터 경로 테스트 (네트워크 없음)."""

import math

import numpy as np
import pandas as pd
import pytest

from api import index as ix
from scripts import audit_prediction_layers as audit


def test_auc_matches_known_values():
    assert audit.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert audit.auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert audit.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5          # 동점은 0.5
    assert math.isnan(audit.auc([1, 1, 1], [0.2, 0.3, 0.4]))              # 한쪽 클래스뿐


def test_brier_and_calibration_slope_distinguish_calibrated_from_uninformative():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20_000)
    calibrated = (rng.uniform(size=p.size) < p).astype(int)
    assert audit.calibration_slope(calibrated, p) == pytest.approx(1.0, abs=0.05)
    informationless = (rng.uniform(size=p.size) < 0.55).astype(int)       # 확률과 무관한 결과
    assert abs(audit.calibration_slope(informationless, p)) < 0.05
    assert audit.brier(calibrated, p) < audit.brier(informationless, p)


def test_block_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    rows = []
    for ticker in ("A", "B", "C", "D"):
        for month in range(1, 9):
            for day in range(10):
                score = rng.uniform()
                rows.append({"ticker": ticker, "date": f"2025-{month:02d}-{day + 1:02d}", "p": score,
                             "up": int(rng.uniform() < 0.3 + 0.4 * score)})
    frame = pd.DataFrame(rows)
    point = audit.auc(frame["up"], frame["p"])
    low, high = audit.block_bootstrap_ci(frame, lambda s: audit.auc(s["up"], s["p"]), n=80)
    assert low < point < high and 0.5 < low


def test_regime_series_uses_the_service_classifier():
    idx = pd.bdate_range("2025-01-01", periods=300)
    down = pd.DataFrame({"Close": np.concatenate([np.linspace(100, 130, 160), np.linspace(130, 90, 140)])}, index=idx)
    labels = audit.regime_series(down, ix.classify_index_regime)
    assert labels.iloc[50] == "NEUTRAL"        # MA120 이전
    assert labels.iloc[-1] == "BEAR"
    up = pd.DataFrame({"Close": np.linspace(100, 200, 300)}, index=idx)
    assert audit.regime_series(up, ix.classify_index_regime).iloc[-1] == "BULL"


def test_classify_index_regime_handles_nan_inputs():
    assert ix.classify_index_regime(float("nan"), 100.0, 110.0) == "NEUTRAL"
    assert ix.classify_index_regime(90.0, 95.0, 100.0) == "BEAR"
    assert ix.classify_index_regime(120.0, 110.0, 100.0) == "BULL"
    assert ix.classify_index_regime(105.0, 95.0, 100.0) == "NEUTRAL"


def _synthetic_prices(n=330, seed=5):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    index = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(900_000, 1_300_000, n).astype(float)}, index=index)


def test_evaluate_ticker_replays_the_chain_on_synthetic_prices_without_network():
    prices = _synthetic_prices()
    regimes = pd.Series("NEUTRAL", index=prices.index)
    rows = audit.evaluate_ticker(ix, "SYN", prices, regimes, stride=25)
    assert len(rows) >= 2
    first = rows[0]
    for key in ("prob_l0", "prob_final", "up_prob", "down_prob", "ncs", "fws", "fc_lo", "fc_hi", "fwd_ret", "fwd_max", "fwd_min"):
        assert key in first and first[key] is not None
    assert first["fc_lo"] < first["close"] < first["fc_hi"]
    assert first["hyb_regime"] == "SIDEWAYS"                       # 벤치마크 없는 하이브리드 레짐은 중립
    assert first["fwd_min"] <= first["fwd_ret"] <= first["fwd_max"]
    summary = audit.summarize(rows)
    assert summary["rows"] == len(rows)
    assert 0.0 <= summary["forecast_band_coverage"]["p10_p90"]["coverage"] <= 1.0


def test_load_prices_rejects_missing_cache_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "CACHE_DIR", str(tmp_path))
    assert audit.load_prices("NOPE", offline=True) is None


def test_load_prices_keeps_exchange_local_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "CACHE_DIR", str(tmp_path))
    csv = tmp_path / "005930_KS.csv"
    csv.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2026-01-05 00:00:00+09:00,1,2,0.5,1.5,100\n"
        "2026-01-06 00:00:00+09:00,1,2,0.5,,100\n"        # 종가 NaN 행은 제거
        "2026-01-07 00:00:00+09:00,1,2,0.5,1.7,100\n", encoding="utf-8")
    frame = audit.load_prices("005930.KS", offline=True)
    assert [d.strftime("%Y-%m-%d") for d in frame.index] == ["2026-01-05", "2026-01-07"]
