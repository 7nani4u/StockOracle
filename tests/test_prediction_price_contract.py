"""현재가 앵커와 규칙 기반 진단 출력 계약 회귀 테스트."""

import json
import inspect
import shutil
import subprocess

import pytest

from api.index import HTML, route
from market_briefing.prediction_contract import (
    apply_prediction_price_contract,
    build_price_anchor,
)


def test_price_anchor_is_propagated_to_every_prediction_section():
    anchor = build_price_anchor(
        current_price=110.0,
        previous_close=100.0,
        market="US",
        source="nasdaq_basic",
        session="프리마켓",
        quote_date="2026-09-23",
        quote_time="08:42 ET",
        last_bar_date="2026-09-22",
        is_extended=True,
        applied_to_chart=False,
    )
    response = {
        "last_close": 110.0,
        "buy_price": {"current": 100.0},
        "risk_scenarios": {"conservative": {"target": [112.0, 115.0]}},
        "target_price": {"min_price": 112.0, "max_price": 118.0},
        "prediction_outlook": {
            "decision": {"key": "watch"},
            "forecast": {"status": "ok", "current_price": 100.0},
            "market_context": {"data_gaps": []},
        },
    }

    result = apply_prediction_price_contract(response, anchor)

    assert result["price_anchor"]["current_price"] == 110.0
    assert result["buy_price"]["current"] == 110.0
    assert result["buy_price"]["price_anchor"]["anchor_id"] == anchor["anchor_id"]
    assert result["risk_scenarios"]["price_anchor"]["anchor_id"] == anchor["anchor_id"]
    assert result["target_price"]["price_anchor"]["anchor_id"] == anchor["anchor_id"]
    assert result["prediction_outlook"]["price_anchor"]["anchor_id"] == anchor["anchor_id"]
    assert result["prediction_outlook"]["forecast"]["current_price"] == 110.0
    assert result["prediction_price_contract"]["status"] == "corrected"
    assert set(result["prediction_price_contract"]["mismatches"]) == {
        "buy_price.current",
        "prediction_outlook.forecast.current_price",
    }
    assert result["prediction_outlook"]["forecast"]["status"] == "limited"
    assert any(
        "현재가 앵커" in warning
        for warning in result["prediction_outlook"]["market_context"]["data_gaps"]
    )


def test_price_anchor_distinguishes_live_quote_from_confirmed_indicator_bar():
    anchor = build_price_anchor(
        current_price=215.25,
        previous_close=210.0,
        market="US",
        source="yahoo",
        session="애프터마켓",
        quote_date="2026-09-23",
        quote_time="18:12 ET",
        last_bar_date="2026-09-23",
        is_extended=True,
        applied_to_chart=False,
    )

    assert anchor["price_basis"] == "현재 시세"
    assert anchor["indicator_basis"] == "2026-09-23 확정 일봉"
    assert anchor["mixed_time_basis"] is True
    assert anchor["change_pct"] == 2.5


def test_daily_history_fallback_is_not_labeled_as_live_quote():
    anchor = build_price_anchor(
        current_price=70_500.0,
        previous_close=70_000.0,
        market="KRX",
        source="yfinance_daily",
        session="장 마감",
        quote_date="2026-09-22",
        quote_time=None,
        last_bar_date="2026-09-22",
        is_extended=False,
        applied_to_chart=True,
    )

    assert anchor["price_basis"] == "최근 확정 종가"
    assert anchor["mixed_time_basis"] is False


def test_route_finalizes_current_price_before_rule_analysis():
    source = inspect.getsource(route)

    anchor_pos = source.index("_price_anchor = build_price_anchor")
    analysis_pos = source.index("score, steps, patterns, geo_patterns, ai_strategy = analyze_score")
    risk_pos = source.index("risk             = calc_risk")

    assert anchor_pos < analysis_pos < risk_pos


def test_same_session_current_bar_refreshes_price_dependent_indicators():
    from api.index import _refresh_current_bar_indicators

    closes = [80.0 + i * 0.25 for i in range(80)]
    dd = {
        "Date": [f"2026-06-{(i % 28) + 1:02d}" for i in range(80)],
        "Open": closes.copy(),
        "High": [value + 1.0 for value in closes],
        "Low": [value - 1.0 for value in closes],
        "Close": closes.copy(),
        "Volume": [100_000.0] * 80,
        "EMA20": [None] * 79 + [90.0],
        "RSI": [None] * 79 + [50.0],
    }

    refreshed = _refresh_current_bar_indicators(dd, market="US")

    assert refreshed is not dd
    assert refreshed["Close"][-1] == closes[-1]
    assert refreshed["EMA20"][-1] != 90.0
    assert refreshed["RSI"][-1] != 50.0
    assert len(refreshed["EMA20"]) == len(closes)


def test_prediction_tab_renders_price_anchor_before_forecast_details():
    renderer = HTML.split("function renderPredictionSections", 1)[1].split(
        "function renderForecast", 1
    )[0]

    assert 'id="prediction-price-anchor"' in renderer
    assert "가격 기준" in renderer
    assert "기술지표 기준" in renderer
    assert "priceAnchorHtml" in renderer
    assert renderer.index("priceAnchorHtml") < renderer.index("forecastHtml")


def test_rule_diagnosis_uses_decision_first_order_with_named_sections():
    renderer = HTML.split("function renderTechnicalDiagnosis", 1)[1].split(
        "function renderInvestorFlow", 1
    )[0]
    expected = [
        'data-diagnosis-section="decision"',
        'data-diagnosis-section="current"',
        'data-diagnosis-section="technical"',
        'data-diagnosis-section="flow"',
        'data-diagnosis-section="risk"',
        'data-diagnosis-section="fundamental"',
        'data-diagnosis-section="interpretation"',
    ]

    positions = [renderer.index(marker) for marker in expected]
    assert positions == sorted(positions)
    assert "규칙 기반 종합 해석" in renderer
    assert "AI가 생성한 종합 진단" not in renderer


def test_rule_diagnosis_semantically_deduplicates_summary_lines():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for the diagnosis dedupe regression check")
    source = "function _diagnosisEvidenceKey" + HTML.split(
        "function _diagnosisEvidenceKey", 1
    )[1].split("function renderTechnicalDiagnosis", 1)[0]
    script = """
const vm = require('vm');
const sandbox = {};
vm.createContext(sandbox);
vm.runInContext(SOURCE, sandbox);
const result = sandbox.dedupeDiagnosisLines([
  '[투자자 수급] 외국인 +12,000주 / 기관 +5,000주',
  '외국인 +12,000주 · 기관 +5,000주',
  '[시장 상태] 대표지수 상승 구조(BULL)',
  '대표지수 상승 구조(BULL)',
  '부채비율 150% 초과 — 재무 레버리지 위험 확인 필요'
]);
console.log(JSON.stringify(result));
""".replace("SOURCE", json.dumps(source, ensure_ascii=False))
    completed = subprocess.run(
        [node, "-"], input=script, text=True, encoding="utf-8",
        capture_output=True, check=True, timeout=20,
    )
    result = json.loads(completed.stdout)

    assert len(result) == 3
    assert sum("외국인" in line for line in result) == 1
    assert sum("대표지수" in line for line in result) == 1
