"""AI 진단 탭 패턴 중복 제거·구분선 회귀 테스트."""

from api.index import HTML, _deduplicate_pattern_types


def test_pattern_list_keeps_only_first_item_per_pattern_type():
    patterns = [
        {"id": "inverse_head_shoulders", "name": "역헤드앤숄더", "end_index": 120},
        {"id": "triple_top", "name": "삼중 천장", "end_index": 118},
        {"id": "inverse_head_shoulders", "name": "역헤드앤숄더", "end_index": 90},
        {"name": "역헤드앤숄더", "end_index": 85},
        {"name": "📏 Marubozu", "end_index": 80},
        {"name": "  📏   Marubozu  ", "end_index": 70},
    ]

    result = _deduplicate_pattern_types(patterns)

    assert [(item.get("id"), item["end_index"]) for item in result] == [
        ("inverse_head_shoulders", 120),
        ("triple_top", 118),
        (None, 80),
    ]


def test_candle_pattern_section_uses_visible_gray_divider():
    assert ".step-patterns{" in HTML
    assert "padding-top:12px;border-top:1px solid #30363d" in HTML


def test_incomplete_charm_data_keeps_smart_score_status_and_existing_diagnosis():
    assert "const hasCharmPayload = charm && typeof charm === 'object'" in HTML
    assert "const hasUsableCharm = charm && charm.smart_score != null" in HTML
    assert "Number(charm.available_count) >= 3" in HTML


def test_smart_score_and_diagnosis_follow_one_reading_flow():
    assert "renderTechnicalDiagnosis(d, isKrx, diagEl, fundamentalHtml);" in HTML
    assert "function renderTechnicalDiagnosis(d, isKrx, diagEl)" in HTML
    assert "hasCompleteRadarData = subScoresForRadar.every" in HTML


def test_ai_diagnosis_numbered_section_titles_are_removed_and_uses_card_layout():
    renderer = HTML.split("function renderTechnicalDiagnosis", 1)[1].split(
        "function renderInvestorFlow", 1
    )[0]

    for numbered in (
        "1. 현재 상태", "2. 핵심 데이터·근거", "3. 기술적 분석", "4. 수급 분석",
        "5. 기술·수급 종합 해석", "6. 위험 및 확인 요소", "7. AI 종합 진단", "8. 최종 판단",
    ):
        assert numbered not in renderer
    # 흐름 설명 문구는 섹션 타이틀에서 제거되고 카드 본문 구조만 유지된다
    assert 'class="diag-grade-row"' in renderer
    assert 'id="flow-rec-badge"' in renderer


def test_technical_and_supply_conclusions_are_combined_once_after_their_evidence():
    renderer = HTML.split("function renderTechnicalDiagnosis", 1)[1].split(
        "function renderInvestorFlow", 1
    )[0]

    # 기술·수급 종합 해석 카드는 별도 블록으로 중복 노출하지 않는다
    assert "기술·수급 신호 엇갈림" not in renderer
    assert "기술·수급 ${techDirection} 신호 일치" not in renderer
    assert "기술과 수급은 각각의 근거 섹션에서는 따로 읽고" not in renderer
    assert "거래량 확인" in renderer
    assert "교차 지표 정합" in renderer


def test_diagnosis_badge_omits_composite_score_but_keeps_signal_consistency():
    renderer = HTML.split("function renderInvestorFlow", 1)[1].split(
        "function resetPeerIndustryTab", 1
    )[0]

    assert "`${recLbl} · ${confText}`" in renderer
    assert "종합 ${effScore.toFixed(1)}점" not in renderer


def test_longterm_cards_render_strict_garp_criteria_for_both_markets():
    assert "function renderLongtermGarp(assessment)" in HTML
    assert "GARP 5/5 통과" in HTML
    assert HTML.count("var garpHtml = renderLongtermGarp(it.peter_lynch);") == 2


def test_charm_score_label_and_disclosure_remain_visible():
    assert "스마트스코어" in HTML
    assert "비교 유니버스 상위" in HTML
    # 비교 유니버스 고지 문구 중 중복 설명은 제거된다
    assert "전체 상장사 순위가 아닌 실제 재무 데이터가 수집된 비교 유니버스 기준" not in HTML
    assert "사업독점력은 시장점유율이 아닌" in HTML
