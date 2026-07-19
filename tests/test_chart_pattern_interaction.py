"""가격 차트 패턴 오버레이의 hover·touch 노출 회귀 테스트."""

from pathlib import Path

from api.index import HTML


SOURCE = Path(__file__).parents[1].joinpath("api", "index.py").read_text(encoding="utf-8")


def test_pattern_overlays_are_bound_as_hover_touch_interaction():
    helper = SOURCE.split("function _bindInteractivePatternOverlays", 1)[1].split(
        "function renderCharts", 1
    )[0]

    assert "let patternsVisible = false" in helper
    assert "priceEl.dataset.patternsVisible = 'false'" in helper
    assert "addEventListener('pointermove', onPointerMove)" in helper
    assert "addEventListener('pointerdown', onPointerDown)" in helper
    assert "addEventListener('pointerleave', onPointerLeave)" in helper
    assert "!priceEl.contains(event.target)" in helper
    assert "setTimeout(hidePatterns, 450)" in helper


def test_hiding_patterns_removes_every_overlay_artifact():
    helper = SOURCE.split("function _bindInteractivePatternOverlays", 1)[1].split(
        "function renderCharts", 1
    )[0]

    assert "candleSeries.setMarkers([])" in helper
    assert "candleSeries.removePriceLine(line)" in helper
    assert "chart.removeSeries(series)" in helper
    assert "removeEventListener('pointermove', onPointerMove)" in helper
    assert "Object.values(chartCleanupHandlers)" in SOURCE


def test_chart_payload_declares_hover_touch_mode():
    assert '"interaction_mode": "hover_touch"' in SOURCE
    assert "_bindInteractivePatternOverlays(" in HTML
