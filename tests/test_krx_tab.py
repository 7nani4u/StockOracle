"""국내·해외 공통 동종업계 전망 탭의 회귀 테스트."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from api.index import HTML


SOURCE = Path(__file__).parents[1].joinpath("api", "index.py").read_text(encoding="utf-8")


def _peer_tab_html():
    return HTML.split('<div id="tab-evening"', 1)[1].split("<!-- 스크리너 페이지 -->", 1)[0]


def test_tab_is_renamed_and_contains_simple_probability_graphs():
    tab_html = _peer_tab_html()

    assert "🏭 동종업계 전망" in HTML
    assert "📋 KRX" not in HTML
    for element_id in (
        "peer-up-prob", "peer-down-prob", "peer-up-bar", "peer-down-bar",
        "peer-balance-up", "peer-balance-down", "peer-metrics", "peer-list",
        "peer-selected-comparison", "peer-data-scope",
    ):
        assert f'id="{element_id}"' in tab_html


def test_tab_is_loaded_for_both_krx_and_foreign_analysis_results():
    analyze_tail = SOURCE.split("// 국내·해외 공통: 동종기업 모멘텀", 1)[1].split("} catch(e)", 1)[0]

    assert "eveningTabBtn.style.display = ''" in analyze_tail
    assert "loadPeerIndustryOutlook(d)" in analyze_tail
    assert "d.market === 'KRX'" not in analyze_tail
    assert "style.display = 'none'" not in analyze_tail


def test_peer_data_uses_a_dedicated_lazy_endpoint_and_industry_metadata():
    loader = SOURCE.split("async function loadPeerIndustryOutlook", 1)[1].split(
        "function refreshPeerIndustryTabFromCache", 1
    )[0]

    assert "analysisData.toss_industry" in loader
    assert "analysisData.naver" in loader
    assert "fetch('/api/peer-outlook?'" in loader
    assert "ticker=" in loader and "market=" in loader and "industry=" in loader
    assert 'if path == "/api/peer-outlook"' in SOURCE


def test_probability_bars_and_peer_rows_have_responsive_styles():
    assert ".peer-prob-grid{display:grid;grid-template-columns:repeat(2" in HTML
    assert ".peer-row{display:grid" in HTML
    assert "@media(max-width:600px)" in HTML
    assert ".peer-prob-grid{grid-template-columns:1fr}" in HTML
    assert "upBar.style.width = up + '%'" in SOURCE
    assert "downBar.style.width = down + '%'" in SOURCE


def test_renderer_discloses_relative_estimate_and_data_limitations():
    renderer = SOURCE.split("function renderPeerIndustryOutlook", 1)[1].split(
        "// 🔔 알림 시스템", 1
    )[0]

    assert "상승 상대 가능성" in renderer
    assert "업계 평균보다" in renderer
    assert "확정적인 주가 예측이나 투자 권유가 아니며" in renderer
    assert "업종 분류 및 데이터 제공 범위" in renderer
    assert SOURCE.count("refreshPeerIndustryTabFromCache();") >= 3


def test_peer_company_names_are_clickable_and_never_fall_back_to_ticker():
    renderer = SOURCE.split("function renderPeerIndustryOutlook", 1)[1].split(
        "// 🔔 알림 시스템", 1
    )[0]

    assert 'class="peer-name peer-name-button"' in renderer
    assert "openStockDetail(" in renderer
    assert "displayName" in renderer
    assert "peer.name || peer.ticker" not in renderer
    assert ".peer-name-button:hover" in HTML


def test_obsolete_krx_scenario_renderer_is_removed():
    for obsolete in (
        "loadKrxAnalysis", "renderKrxAnalysis", "refreshKrxAnalysisFromCache",
        "krx-scenario-grid", "KRX 종합 판단", "조건부 KRX 시나리오",
    ):
        assert obsolete not in SOURCE
