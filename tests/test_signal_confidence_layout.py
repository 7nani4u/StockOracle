"""분석 결과 상단 정보 순서와 신호 신뢰도 카드 UI 회귀 테스트."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from api.index import HTML


SOURCE = Path(__file__).parents[1].joinpath("api", "index.py").read_text(encoding="utf-8")


def test_result_summary_order_follows_context_before_interpretation():
    result = HTML.split('<div id="state-result"', 1)[1].split('<div class="tabs" id="result-tabs">', 1)[0]

    metrics = result.index('<div class="metrics-grid">')
    fundamentals = result.index('id="r-naver-fund"')
    confidence = result.index('id="signal-confidence-card"')
    assert metrics < fundamentals < confidence


def test_signal_confidence_layout_restores_compact_context_chips():
    renderer = SOURCE.split("function renderSignalConfidence", 1)[1].split(
        "function renderResult", 1
    )[0]

    for text in (
        "종합 신뢰도", "현재 해석", "거시 ", "뉴스감정 ",
        "신뢰도 제한·변동 요인", "상승확률이나 적중확률이 아닙니다",
    ):
        assert text in renderer
    for data_key in (
        "macro_regime", "sector_relative",
        "days_to_earnings", "sentiment", "cap_reasons", "confidence_interval",
    ):
        assert data_key in renderer
    assert "환경·이벤트 보정" not in renderer
    assert "방향 점수 구성" not in renderer
    assert "signal-confidence-factor-grid" not in renderer
    assert "signal-confidence-source-grid" not in renderer


def test_confidence_meter_is_accessible_and_explains_the_neutral_baseline():
    renderer = SOURCE.split("function renderSignalConfidence", 1)[1].split(
        "function renderResult", 1
    )[0]

    assert 'role="meter"' in renderer
    assert 'aria-valuemin="0"' in renderer
    assert 'aria-valuemax="100"' in renderer
    assert "50점은 중립 기준" in renderer


def test_confidence_layout_has_desktop_and_mobile_reading_order():
    assert ".signal-confidence-overview{display:grid;grid-template-columns:" in HTML
    assert ".signal-confidence-chips{display:flex;flex-wrap:wrap" in HTML
    assert ".signal-confidence-chip{" in HTML
    assert ".signal-confidence-reason-grid{display:grid;grid-template-columns:repeat(2" in HTML
    assert ".signal-confidence-overview{grid-template-columns:1fr}" in HTML
    assert ".signal-confidence-reason-grid{grid-template-columns:1fr}" in HTML
