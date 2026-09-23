"""Regression checks for missing forecast inputs and prediction-tab rendering."""

import json
import math
import shutil
import subprocess

import pytest

from api.index import HTML, build_prediction_outlook
from market_briefing import forecast_model


def _outlook_kwargs():
    closes = [100.0 + i * 0.2 for i in range(80)]
    return {
        "symbol": "AAPL", "market": "US",
        "dd": {
            "Close": closes,
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Volume": [100_000] * 80,
            "RSI": [55.0] * 80,
            "MACD": [1.0] * 80,
            "Signal_Line": [0.5] * 80,
            "MA20": closes.copy(),
            "MA60": closes.copy(),
        },
        "last_price": closes[-1], "prev_close": closes[-2],
        "pct_change": 0.18, "atr": 2.0, "regime": "BULL", "score": 65,
        "prob_up": 60, "prob_down": 20,
        "pivot_points": {"classic": {"S1": closes[-1] - 2, "R1": closes[-1] + 2}},
        "indicator_signals": {}, "buy_price": {"strategy_rec": {"action_key": "split_buy"}},
        "target_price": {"min_price": closes[-1] + 3, "max_price": closes[-1] + 6},
        "pullback_analysis": {"stop_loss": closes[-1] - 4},
        "signal_confidence": {"confidence": 65}, "investor_flow": None,
        "ai_strategy": None, "candlestick_patterns": [], "naver": None,
        "us_enriched": None, "toss_industry": None, "event_risk": None,
    }


def test_latest_missing_indicator_does_not_reuse_yesterdays_value():
    kwargs = _outlook_kwargs()
    baseline = build_prediction_outlook(**kwargs)
    kwargs["dd"]["RSI"][-1] = None
    kwargs["dd"]["MACD"][-1] = float("nan")
    result = build_prediction_outlook(**kwargs)

    trend = next(row for row in result["status"] if row["key"] == "trend")
    assert "RSI 미확보" in trend["detail"]
    assert result["decision"]["confidence"] < baseline["decision"]["confidence"]
    assert any("기술지표 미확보" in gap for gap in result["market_context"]["data_gaps"])


def test_missing_current_moving_averages_reduce_forecast_confidence():
    kwargs = _outlook_kwargs()
    baseline = build_prediction_outlook(**kwargs)
    kwargs["dd"]["MA20"][-1] = None
    kwargs["dd"]["MA60"][-1] = None
    result = build_prediction_outlook(**kwargs)

    trend = next(row for row in result["status"] if row["key"] == "trend")
    assert "MA20 미확보" in trend["detail"]
    assert "MA60 미확보" in trend["detail"]
    assert result["decision"]["confidence"] <= baseline["decision"]["confidence"] - 4
    assert any("MA20 또는 MA60 미확보" in gap for gap in result["market_context"]["data_gaps"])


def test_volatility_does_not_join_prices_across_missing_bar():
    closes = [100.0] * 20 + [None] + [200.0] * 20
    sigma, observed = forecast_model.daily_log_volatility(closes)
    assert observed == 38
    assert sigma == 0.0
    blended = forecast_model.blended_daily_sigma(closes, 200.0, atr=None)
    assert blended["sigma"] == 0.002
    assert math.isfinite(blended["sigma"])


def test_forecast_tab_distinguishes_missing_values_and_clears_old_risk():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for the browser renderer regression check")
    source = "function renderPredictionSections" + HTML.split("function renderPredictionSections", 1)[1].split(
        "function renderTechnicalSignals", 1
    )[0]
    payload = {
        "data_quality": {"status": "데이터 부족", "last_bar_date": "2026-09-18",
                         "history_bars": 18, "source": "테스트 일봉", "warnings": ["표본 부족"]},
        "prediction_outlook": {
            "decision": {"key": "watch", "label": "관망", "tone": "neutral"},
            "forecast": {"status": "limited", "base_price": 101, "current_price": 100,
                         "range_p10_p90": [90, 110], "range_return_pct": [-10, 10],
                         "scenario_probabilities": {"up": None, "sideways": 20, "down": 30},
                         "sigma_horizon_pct": None},
            "dynamic_rsi": {"available": True, "lower": None, "rsi": None, "upper": None,
                            "purchase_timing": {"state": "watch", "conditions": []}},
            "scenarios": [], "market_context": {"facts": [], "data_gaps": []},
        },
        "risk_scenarios": {"provisional": True, "conservative": {
            "label": "보수적", "icon": "", "desc": "", "target": [98, 104],
            "stop_pct": -2, "rr_ratio": 1, "return": 3,
            "tp_levels": [{"price": 103, "return_pct": None, "prob_pct": None,
                           "avg_days": None}],
        }},
    }
    script = """
const vm = require('vm');
const elements = {};
const sandbox = {
  document: {getElementById(id) { return elements[id] ||= {innerHTML:'', textContent:''}; }},
  _isFiniteNumber(v) { return (typeof v === 'number' || (typeof v === 'string' && v.trim() !== '')) && Number.isFinite(Number(v)); },
  _escPrediction(v) { return String(v == null ? '' : v); },
  _predictionTone() { return '#8b949e'; },
  _predictionLiveFacts() { return []; },
  fmt(v) { return v == null ? '—' : String(v); },
};
vm.createContext(sandbox);
vm.runInContext(SOURCE, sandbox);
const data = DATA;
sandbox.renderForecast(data, true);
const first = {overview: elements['prediction-overview-section'].innerHTML,
               risk: elements['risk-grid'].innerHTML};
data.risk_scenarios = null;
data.buy_price = {};
sandbox.renderForecast(data, true);
console.log(JSON.stringify({first, clearedRisk: elements['risk-grid'].innerHTML,
                            emptyEntry: elements['buy-price-section'].innerHTML}));
""".replace("SOURCE", json.dumps(source, ensure_ascii=False)).replace(
        "DATA", json.dumps(payload, ensure_ascii=False)
    )
    completed = subprocess.run([node, "-"], input=script, text=True, encoding="utf-8",
                               capture_output=True, check=True, timeout=20)
    rendered = json.loads(completed.stdout)
    assert "데이터 상태: 데이터 부족" not in rendered["first"]["overview"]
    assert "RSI 미확보" in rendered["first"]["overview"]
    assert "상승 미산정" in rendered["first"]["overview"]
    assert "1σ ±미산정" in rendered["first"]["overview"]
    assert "가능성 산정 보류" in rendered["first"]["risk"]
    assert "기간 산정 보류" in rendered["first"]["risk"]
    assert "리스크 계산에 필요한" in rendered["clearedRisk"]
    assert "진입 가격 구간을 산정할" in rendered["emptyEntry"]
