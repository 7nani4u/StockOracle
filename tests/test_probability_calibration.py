"""방향 확률 보정·레짐 라벨 분리 계약 검증."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from api import index as ix
from market_briefing import probability_calibration as pc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 보정 모듈 ────────────────────────────────────────────────────────────────

_PAYLOAD = {
    "slope": 0.04,
    "intercept_at_half": {"ALL": 0.5665, "KRX": 0.5475, "US": 0.5848},
    "clip_pct": [5.0, 95.0],
    "horizon_sessions": 22,
    "method": "linear_shrink_market_intercept",
    "fitted_on": {"rows": 8368, "period": ["2022-10-07", "2026-09-02"], "base_up_rate": {"KRX": 0.5465, "US": 0.5864}},
}


def test_calibration_pulls_wide_raw_probabilities_to_the_market_base_rate():
    low = pc.calibrate_direction_probability(10.0, "KRX", _PAYLOAD)
    high = pc.calibrate_direction_probability(91.0, "KRX", _PAYLOAD)
    us = pc.calibrate_direction_probability(50.0, "US", _PAYLOAD)

    assert low["applied"] and high["applied"]
    # 10~91% 로 퍼지던 값이 시장 기저 상승률(KRX 약 55%) 부근의 좁은 범위에 모인다
    assert 52.0 <= low["prob_up"] < high["prob_up"] <= 58.0
    assert high["prob_up"] - low["prob_up"] < 4.0
    assert us["prob_up"] == pytest.approx(58.5, abs=0.1)
    for result in (low, high, us):
        assert result["prob_up"] + result["prob_down"] == pytest.approx(100.0, abs=0.11)
    # 원래 값은 신호 점수로 보존된다
    assert low["raw_prob_up"] == 10.0 and high["raw_prob_up"] == 91.0
    assert high["horizon_sessions"] == 22 and "보정" in high["note"]


def test_calibration_is_monotonic_and_clipped():
    values = [pc.calibrate_direction_probability(p, "US", _PAYLOAD)["prob_up"] for p in range(0, 101, 5)]
    assert values == sorted(values)
    steep = dict(_PAYLOAD, slope=5.0)
    assert pc.calibrate_direction_probability(100.0, "US", steep)["prob_up"] == 95.0
    assert pc.calibrate_direction_probability(0.0, "US", steep)["prob_up"] == 5.0


def test_calibration_fails_open_without_a_file_and_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "_PATH_OVERRIDE", str(tmp_path / "missing.json"))
    pc._CACHE.clear()
    result = pc.calibrate_direction_probability(73.0, "KRX")
    assert result["applied"] is False and result["method"] == "identity"
    assert result["prob_up"] == 73.0 and result["prob_down"] == 27.0

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(pc, "_PATH_OVERRIDE", str(broken))
    pc._CACHE.clear()
    assert pc.calibrate_direction_probability(73.0, "US")["applied"] is False

    assert pc.calibrate_direction_probability(None, "KRX")["prob_up"] is None
    assert pc.calibrate_direction_probability("abc", "KRX")["applied"] is False
    assert pc.calibrate_direction_probability(float("nan"), "KRX")["prob_up"] is None
    assert pc.calibrate_direction_probability(55.0, "KRX", {"slope": "x"})["applied"] is False


def test_fit_recovers_slope_and_market_base_rates():
    rng = np.random.default_rng(5)
    n = 6000
    raw = rng.uniform(10, 90, n)
    markets = np.where(rng.random(n) < 0.5, "KRX", "US")
    base = np.where(markets == "KRX", 0.52, 0.60)
    prob = np.clip(base + 0.30 * (raw / 100.0 - 0.5), 0.02, 0.98)
    went_up = (rng.random(n) < prob).astype(int)

    fit = pc.fit_direction_calibration(raw, went_up, markets)

    assert fit["slope"] == pytest.approx(0.30, abs=0.06)
    assert fit["intercept_at_half"]["KRX"] == pytest.approx(0.52, abs=0.03)
    assert fit["intercept_at_half"]["US"] == pytest.approx(0.60, abs=0.03)
    assert fit["rows"] == n
    with pytest.raises(ValueError):
        pc.fit_direction_calibration([], [], [])


def test_uninformative_scores_calibrate_to_the_base_rate_with_a_flat_slope():
    rng = np.random.default_rng(9)
    raw = rng.uniform(5, 95, 8000)
    went_up = (rng.random(8000) < 0.57).astype(int)   # 점수와 무관한 결과
    fit = pc.fit_direction_calibration(raw, went_up, ["KRX"] * 8000)
    assert abs(fit["slope"]) < 0.06
    calibrated = [pc.calibrate_direction_probability(p, "KRX", dict(fit, clip_pct=[5, 95]))["prob_up"] for p in (10, 50, 90)]
    assert max(calibrated) - min(calibrated) < 4.0
    assert all(53.0 <= value <= 61.0 for value in calibrated)


# ── 배포되는 보정 파일 ───────────────────────────────────────────────────────

def test_shipped_calibration_file_is_valid_and_documents_its_evidence():
    path = os.path.join(REPO_ROOT, "models", "probability_calibration.json")
    payload = json.load(open(path, encoding="utf-8"))

    assert payload["horizon_sessions"] == 22
    assert set(payload["intercept_at_half"]) >= {"ALL", "KRX", "US"}
    assert abs(payload["slope"]) < 0.3
    low, high = payload["slope_bootstrap_ci95"]
    assert low < high
    assert payload["fitted_on"]["rows"] >= 5000 and payload["fitted_on"]["tickers"] >= 40
    oos = payload["oos"]
    # 표본 밖에서 보정 전보다 확실히 낫고(Brier·ECE), 기저율 상수와 비슷하다
    assert oos["brier_calibrated"] < oos["brier_raw"] - 0.02
    assert oos["ece_calibrated"] < oos["ece_raw"] / 2
    assert oos["brier_calibrated"] <= oos["brier_base_rate"] + 0.003
    # 모듈이 이 파일로 실제 보정한다
    pc._CACHE.clear()
    result = pc.calibrate_direction_probability(85.0, "US")
    assert result["applied"] is True and 50.0 < result["prob_up"] < 65.0


# ── 라우트·화면 계약 ─────────────────────────────────────────────────────────

def test_route_calibrates_after_correlation_and_before_the_outlook():
    source = open(os.path.join(REPO_ROOT, "api", "index.py"), encoding="utf-8").read()
    corr_pos = source.index('prob_up = float(_corr.get("prob_up_corr", prob_up))')
    cal_pos = source.index("calibrate_direction_probability(prob_up, market)")
    outlook_pos = source.index("prediction_outlook = build_prediction_outlook(")
    response_pos = source.index('"prob_up_raw": prob_up_raw')
    assert corr_pos < cal_pos < outlook_pos < response_pos


def test_probability_ui_separates_calibrated_probability_from_raw_signal_score():
    html = ix.HTML
    assert "d.probability_calibration" in html
    assert "신호 점수 ▲" in html and "(보정 전)" in html
    assert "const probWord = probCalibrated ? '확률' : '점수'" in html


# ── 레짐 라벨 분리 ───────────────────────────────────────────────────────────

def test_index_structure_fact_is_labelled_as_the_index_not_the_stock():
    kospi = ix.index_structure_fact("KRX", "005930.KS", "BEAR")
    kosdaq = ix.index_structure_fact("KRX", "247540.KQ", "BULL")
    spx = ix.index_structure_fact("US", "AAPL", "NEUTRAL")

    assert kospi["label"] == "KOSPI 중기 구조" and kospi["value"] == "하락 구조" and kospi["tone"] == "negative"
    assert kosdaq["label"] == "KOSDAQ 중기 구조" and kosdaq["value"] == "상승 구조" and kosdaq["tone"] == "positive"
    assert spx["label"] == "S&P 500 중기 구조" and spx["value"] == "혼조"
    assert "종목 자체의 추세가 아님" in kospi["detail"]


def test_three_regime_concepts_get_distinct_names_and_conflicts_are_explained():
    market_hybrid = {"regime": "BULLISH", "regime_detail": {"ma200_available": True}}
    stock_hybrid = {"regime": "SIDEWAYS", "regime_detail": {"ma200_available": False}}

    conflict = ix.build_regime_layers("KRX", "005930.KS", "BEAR", market_hybrid)
    assert conflict["index_structure"]["value"] == "하락 구조"
    assert conflict["long_regime"]["value"] == "강세장"
    assert conflict["stock_trend"] is None
    assert conflict["index_structure"]["label"] != conflict["long_regime"]["label"]
    assert "서로 다른 기준" in conflict["note"]

    single = ix.build_regime_layers("US", "AAPL", "BULL", stock_hybrid)
    assert single["long_regime"] is None
    assert single["stock_trend"]["value"] == "방향 불명"
    assert "시장 기준 미적용" in single["stock_trend"]["label"]
    assert single["note"] is None

    assert ix.build_regime_layers("KRX", "005930.KS", "NEUTRAL", {"error": "x"})["long_regime"] is None
    assert ix.build_regime_layers("KRX", "005930.KS", "NEUTRAL", None)["index_structure"]["value"] == "혼조"


def test_outlook_market_facts_never_call_the_index_structure_a_stock_structure():
    source = open(os.path.join(REPO_ROOT, "api", "index.py"), encoding="utf-8").read()
    assert '"label": "종목 중기 구조"' not in source
    assert "해당 종목의 60일·120일 가격 구조 기준" not in source
    # 두 아웃룩(정식·축소)이 같은 사실 카드 생성 함수를 쓴다
    assert source.count("index_structure_fact(market, symbol, regime)") >= 3


def test_hybrid_card_and_scan_header_name_the_regime_they_show():
    html = ix.HTML
    assert "시장 장기 국면" in html          # 스캔 머리말·벤치마크가 있는 하이브리드 국면
    assert "종목 추세" in html               # 벤치마크 없이 종목 ADX·DI 로만 낸 국면
    assert "const regPrefix = marketBased ? '시장 장기 국면' : '종목 추세'" in html
    assert "60·120일선 하락 구조" in open(os.path.join(REPO_ROOT, "api", "index.py"), encoding="utf-8").read()
