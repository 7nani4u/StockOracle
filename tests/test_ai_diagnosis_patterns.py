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


def test_smart_score_and_existing_diagnosis_render_together():
    assert 'id="legacy-tech-diagnosis"' in HTML
    assert "renderTechnicalDiagnosis(d, isKrx, legacyEl);" in HTML
    assert "function renderTechnicalDiagnosis(d, isKrx, diagEl)" in HTML
    assert "hasCompleteRadarData = subScoresForRadar.every" in HTML


def test_charm_score_label_and_disclosure_remain_visible():
    assert "스마트스코어" in HTML
    assert "비교 유니버스 상위" in HTML
    assert "전체 상장사 순위가 아닌 실제 재무 데이터가 수집된 비교 유니버스 기준" in HTML
