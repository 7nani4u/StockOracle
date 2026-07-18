"""📋 KRX 탭의 정보 구조와 데이터 재사용 경로 회귀 테스트."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from api.index import HTML


SOURCE = Path(__file__).parents[1].joinpath("api", "index.py").read_text(encoding="utf-8")


def test_krx_tab_prioritizes_actionable_conditional_analysis():
    krx_html = HTML.split('<div id="tab-evening"', 1)[1].split("<!-- 스크리너 페이지 -->", 1)[0]

    summary = krx_html.index("KRX 종합 판단")
    state = krx_html.index("현재 상태와 핵심 가격")
    scenarios = krx_html.index("조건부 KRX 시나리오")
    context = krx_html.index("한국 시장·업종·수급")
    assert summary < state < scenarios < context

    for element_id in (
        "krx-decision",
        "krx-current-price",
        "krx-leading-scenario",
        "krx-stop-price",
        "krx-state-grid",
        "krx-level-grid",
        "krx-scenario-grid",
        "krx-market-context",
        "krx-signal-breakdown",
        "krx-risk-list",
    ):
        assert f'id="{element_id}"' in krx_html


def test_krx_tab_does_not_present_same_snapshot_as_historical_validation():
    krx_html = HTML.split('<div id="tab-evening"', 1)[1].split("<!-- 스크리너 페이지 -->", 1)[0]

    for removed_label in ("예측 vs 실제 비교", "아침 예측", "실제 종가 등락", "장 마감(15:30)"):
        assert removed_label not in krx_html
    assert "확정 예측이나 투자 권유가 아닌" in krx_html


def test_krx_tab_reuses_main_analysis_and_selects_the_correct_board():
    loader = SOURCE.split("async function loadKrxAnalysis", 1)[1].split(
        "function _krxSetText", 1
    )[0]

    assert loader.index("renderKrxAnalysis(null") < loader.index("await fetch(")
    assert "markets=${market}" in loader
    assert "evening=1" not in loader
    assert "krxBoard === 'KOSDAQ'" in loader
    assert "endsWith('.KQ') ? 'KOSDAQ' : 'KOSPI'" in SOURCE


def test_krx_layout_has_desktop_and_mobile_grid_breakpoints():
    assert ".krx-scenario-grid{display:grid;grid-template-columns:repeat(3" in HTML
    assert "@media(max-width:900px)" in HTML
    assert ".krx-scenario-grid{grid-template-columns:1fr}" in HTML
    assert "@media(max-width:600px)" in HTML
    assert ".krx-summary-grid,.krx-state-grid,.krx-level-grid,.krx-context-grid,.krx-risk-list{grid-template-columns:1fr}" in HTML


def test_krx_tab_only_merges_observed_supplement_signals_and_refreshes_late_data():
    renderer = SOURCE.split("function renderKrxAnalysis", 1)[1].split(
        "// ══════════════════════════════════════════════════════\n// 🔔 알림 시스템", 1
    )[0]

    assert "stocks.find(s => String(s.code) === String(krxCode)) || null" in renderer
    assert "if (overnight && typeof overnight === 'object' && overnight.direction)" in renderer
    assert "20거래일 평균의 ${supplementVolumeRatio.toFixed(2)}배" in renderer
    assert "history.pos_52w_pct" in renderer
    assert "미확보 데이터는 확정 판단에서 제외" in renderer
    assert SOURCE.count("refreshKrxAnalysisFromCache();") >= 3


def test_krx_entry_cards_follow_wait_or_caution_decision_without_layout_change():
    renderer = SOURCE.split("function renderKrxAnalysis", 1)[1].split(
        "// ══════════════════════════════════════════════════════\n// 🔔 알림 시스템", 1
    )[0]

    assert "decision.key === 'caution' ? 'negative'" in renderer
    assert "현재 판단은 매수 보류 · 가격은 재평가용 대기 구간" in SOURCE
    assert "최근접 지지 후보" in renderer
    assert "최근접 저항 후보" in renderer
