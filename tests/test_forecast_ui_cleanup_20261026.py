from api.index import HTML


def test_requested_forecast_panels_and_legacy_sentence_are_removed():
    assert '📊 기술적 현재 상태' not in HTML
    assert '신뢰도 제한·변동 요인' not in HTML
    assert 'id="prediction-forecast-summary"' not in HTML
    assert '박스권 미확인으로 기존 지표·변동성 기반 참고 가격을 표시합니다.' not in HTML
    renderer = HTML.split('function renderPredictionSections', 1)[1].split('function renderForecast', 1)[0]
    assert 'prediction-status-section' not in renderer
    assert '${waveHtml}' in renderer  # independent wave analysis remains visible
