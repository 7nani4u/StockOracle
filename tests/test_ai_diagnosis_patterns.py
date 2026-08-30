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


def test_ai_diagnosis_orders_evidence_before_interpretation_and_final_judgment():
    renderer = HTML.split("function renderTechnicalDiagnosis", 1)[1].split(
        "function renderInvestorFlow", 1
    )[0]

    labels = (
        "현재 상태", "핵심 데이터·근거", "기술적 분석", "수급 분석",
        "기술·수급 종합 해석", "위험 및 확인 요소", "AI 종합 진단", "최종 판단",
    )
    positions = [renderer.index(label) for label in labels]
    assert positions == sorted(positions)


def test_technical_and_supply_conclusions_are_combined_once_after_their_evidence():
    renderer = HTML.split("function renderTechnicalDiagnosis", 1)[1].split(
        "function renderInvestorFlow", 1
    )[0]

    assert "기술과 수급은 각각의 근거 섹션에서는 따로 읽고" in renderer
    assert "기술·수급 신호 엇갈림" in renderer
    assert "기술·수급 ${techDirection} 신호 일치" in renderer
    assert "거래량 확인" in renderer
    assert "교차 지표 정합" in renderer


def test_charm_score_label_and_disclosure_remain_visible():
    assert "스마트스코어" in HTML
    assert "비교 유니버스 상위" in HTML
    assert "전체 상장사 순위가 아닌 실제 재무 데이터가 수집된 비교 유니버스 기준" in HTML
