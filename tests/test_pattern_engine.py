import math
from pathlib import Path
import re

import numpy as np
import pytest

from market_briefing.pattern_engine import (
    PATTERN_REFACTOR_AUDIT,
    PatternEngine,
    PatternEngineOptions,
    build_pattern_overlays,
    calculate_tolerance,
    deduplicate_patterns,
    integrate_pattern_targets,
    resolve_timeframe,
)


def _ohlcv(points, n=66, *, zero_volume=False):
    x = np.arange(n)
    close = np.interp(x, [p[0] for p in points], [p[1] for p in points])
    volume = np.zeros(n) if zero_volume else np.full(n, 1_000.0)
    if not zero_volume:
        volume[-6:] = 1_800.0
    return {
        "Date": [f"2026-01-{(i % 28) + 1:02d}T{i:02d}:00:00" for i in x],
        "Open": (close - 0.1).tolist(),
        "High": (close + 0.5).tolist(),
        "Low": (close - 0.5).tolist(),
        "Close": close.tolist(),
        "Volume": volume.tolist(),
    }


def _inverse_hs(mode="confirmed"):
    points = [(0, 105), (10, 100), (20, 110), (30, 90), (40, 111), (50, 101)]
    if mode == "confirmed":
        points += [(58, 109), (62, 114), (65, 116)]
    elif mode == "awaiting":
        points += [(65, 109)]
    elif mode == "invalidated":
        points += [(55, 106), (60, 96), (65, 95)]
    elif mode == "forming":
        return _ohlcv(points[:-1] + [(47, 104)], n=48)
    return _ohlcv(points)


def _head_shoulders():
    return _ohlcv([
        (0, 105), (10, 110), (20, 100), (30, 120), (40, 99),
        (50, 109), (58, 101), (62, 95), (65, 94),
    ])


def _find(patterns, pattern_id):
    return next(p for p in patterns if p["id"] == pattern_id and len(p["points"]) == 5)


def test_fixed_tolerance_boundary_is_inclusive():
    exact = calculate_tolerance(100.0, 101.5, atr_value=0.0, reference_price=100.0, mode="fixed")
    outside = calculate_tolerance(100.0, 101.5001, atr_value=0.0, reference_price=100.0, mode="fixed")
    assert exact["passed"] is True
    assert exact["basis"] == "fixed_1.5pct"
    assert outside["passed"] is False


def test_atr_adaptive_tolerance():
    result = calculate_tolerance(
        100.0, 101.9, atr_value=2.0, reference_price=100.0,
        mode="atr", atr_multiplier=1.0,
    )
    assert result["passed"] is True
    assert result["tolerance"] == pytest.approx(0.02)
    assert result["basis"] == "atr_adaptive"


def test_hybrid_tolerance_is_clamped_to_min_and_max():
    minimum = calculate_tolerance(
        100.0, 100.7, atr_value=0.0, reference_price=100.0,
        mode="hybrid", fixed_tolerance=0.001,
        min_tolerance=0.008, max_tolerance=0.045,
    )
    maximum = calculate_tolerance(
        100.0, 104.4, atr_value=20.0, reference_price=100.0,
        mode="hybrid", atr_multiplier=1.0,
        min_tolerance=0.008, max_tolerance=0.045,
    )
    assert minimum["tolerance"] == pytest.approx(0.008)
    assert maximum["tolerance"] == pytest.approx(0.045)
    assert maximum["passed"] is True


@pytest.mark.parametrize(
    "period",
    ["1d", "3d", "1wk", "2wk", "1mo", "6mo", "1y", "2y", "5y"],
)
def test_supported_ui_periods_reuse_timeframe_specific_detection(period):
    engine = PatternEngine.from_mapping(_inverse_hs("confirmed"), timeframe=period)
    pattern = _find(engine.detect(), "inverse_head_shoulders")
    assert pattern["timeframe"] == resolve_timeframe(period)
    assert pattern["pattern_status"] == "confirmed"


def test_analysis_period_dropdown_matches_backend_allowlist():
    source = (Path(__file__).resolve().parents[1] / "api" / "index.py").read_text(encoding="utf-8")
    select = re.search(r'<select id="period-select"[^>]*>(.*?)</select>', source, re.S)
    assert select is not None
    options = re.findall(r'<option value="([^"]+)"(?: selected)?>([^<]+)</option>', select.group(1))
    expected = [
        ("1d", "초단기 (1일)"), ("3d", "초단기 (3일)"),
        ("1wk", "초단기 (1주)"), ("2wk", "단기 (2주)"),
        ("1mo", "단기 (1개월)"), ("6mo", "6개월"),
        ("1y", "1년"), ("2y", "2년"), ("5y", "5년"),
    ]
    assert options == expected
    allowlist_line = re.search(r'^VALID_PERIODS\s*=\s*\{([^}]*)\}', source, re.M)
    assert allowlist_line is not None
    allowlist = set(re.findall(r'"([^"]+)"', allowlist_line.group(1)))
    assert allowlist == {value for value, _ in expected}


def test_e1_to_e5_coordinates_are_stable():
    pattern = _find(PatternEngine.from_mapping(_inverse_hs()).detect(), "inverse_head_shoulders")
    assert [p["label"] for p in pattern["points"]] == ["E1", "E2", "E3", "E4", "E5"]
    assert [p["index"] for p in pattern["points"]] == [10, 20, 30, 40, 50]
    assert all(p["confirmed_index"] > p["index"] for p in pattern["points"])


def test_neckline_uses_e2_and_e4():
    pattern = _find(PatternEngine.from_mapping(_inverse_hs()).detect(), "inverse_head_shoulders")
    neckline = pattern["neckline"]
    assert neckline["start_index"] == pattern["points"][1]["index"]
    assert neckline["end_index"] == pattern["points"][3]["index"]
    assert neckline["breakout_index"] is not None


def test_status_transitions_from_waiting_to_confirmed():
    waiting = _find(PatternEngine.from_mapping(_inverse_hs("awaiting")).detect(), "inverse_head_shoulders")
    confirmed = _find(PatternEngine.from_mapping(_inverse_hs("confirmed")).detect(), "inverse_head_shoulders")
    assert waiting["pattern_status"] == "awaiting_breakout"
    assert waiting["score_eligible"] is False
    assert confirmed["pattern_status"] == "confirmed"
    assert confirmed["score_eligible"] is True


def test_forming_pattern_has_completion_and_next_pivot():
    patterns = PatternEngine.from_mapping(_inverse_hs("forming")).detect()
    forming = next(p for p in patterns if p["id"] == "inverse_head_shoulders" and len(p["points"]) == 4)
    assert forming["pattern_status"] == "forming"
    assert 0 < forming["completion_score"] < 85
    assert forming["next_pivot_type"] == "low"
    assert len(forming["expected_completion_price_range"]) == 2


def test_pattern_can_be_invalidated_after_structure_completion():
    pattern = _find(PatternEngine.from_mapping(_inverse_hs("invalidated")).detect(), "inverse_head_shoulders")
    assert pattern["pattern_status"] == "invalidated"
    assert pattern["score_eligible"] is False


def test_head_shoulders_target_uses_structure_height():
    pattern = _find(PatternEngine.from_mapping(_inverse_hs()).detect(), "inverse_head_shoulders")
    assert pattern["pattern_target_price"] > pattern["neckline"]["breakout_price"]
    assert pattern["pattern_target_low"] < pattern["pattern_target_price"] < pattern["pattern_target_high"]
    assert pattern["target_method"] == "head_to_neckline_height_projection"


def test_pattern_target_merges_with_nearest_existing_tp_without_overwrite():
    pattern = {
        "name": "테스트 역헤드앤숄더", "direction_code": "bullish",
        "pattern_status": "confirmed", "completion_score": 90,
        "pattern_target_price": 110.2,
    }
    source_levels = [{"price": 110.0, "prob_pct": 60.0, "prob_low_pct": 55.0, "prob_high_pct": 65.0,
                      "sources": [{"source": "atr_scenario"}], "source_count": 1}]
    result = integrate_pattern_targets(
        {"conservative": {"tp_levels": source_levels}, "balanced": {"tp_levels": []}, "aggressive": {"tp_levels": []}},
        [pattern], current_price=100.0, atr_value=2.0,
    )
    level = result["scenarios"]["conservative"]["tp_levels"][0]
    assert level["price"] == 110.0
    assert level["pattern_confluence"] is True
    assert level["source_count"] == 2
    assert level["prob_pct"] == 62.0
    assert result["accepted"][0]["tp_index"] == 1


def test_duplicate_patterns_keep_specific_complete_representative():
    base = {
        "family": "head_shoulders", "direction_code": "bullish",
        "start_index": 10, "end_index": 50, "completion_score": 90,
        "pattern_status": "confirmed", "neckline": {"breakout_price": 111},
    }
    complete = {**base, "name": "역헤드앤숄더", "points": [{"index": i} for i in range(5)]}
    forming = {**base, "name": "복합 역헤드앤숄더", "end_index": 46,
               "completion_score": 60, "pattern_status": "forming", "points": [{"index": i} for i in range(4)]}
    result = deduplicate_patterns([forming, complete])
    assert len(result) == 1
    assert result[0]["name"] == "역헤드앤숄더"
    assert "복합 역헤드앤숄더" in result[0]["related_patterns"]


def test_bullish_and_bearish_head_shoulders_are_symmetric():
    bullish = _find(PatternEngine.from_mapping(_inverse_hs()).detect(), "inverse_head_shoulders")
    bearish = _find(PatternEngine.from_mapping(_head_shoulders()).detect(), "head_shoulders_top")
    assert bullish["direction_code"] == "bullish"
    assert bearish["direction_code"] == "bearish"
    assert bullish["signal"] == "매수"
    assert bearish["signal"] == "매도"
    assert bullish["pattern_target_price"] > bullish["neckline"]["breakout_price"]
    assert bearish["pattern_target_price"] < bearish["neckline"]["breakout_price"]


def test_overlay_reuses_exact_detection_coordinates():
    pattern = _find(PatternEngine.from_mapping(_inverse_hs()).detect(), "inverse_head_shoulders")
    overlay = build_pattern_overlays([pattern])[0]
    assert overlay["points"] == pattern["points"]
    assert [p["index"] for p in overlay["connector"]] == [p["index"] for p in pattern["points"]]
    assert overlay["neckline"] == pattern["neckline"]
    assert overlay["target"]["price"] == pattern["pattern_target_price"]


def test_flag_pennant_rectangle_aliases_share_one_consolidation_detector():
    data = _ohlcv([
        (0, 100), (23, 100), (31, 110), (32, 109.5),
        (35, 109.0), (38, 108.8), (39, 112.0),
    ], n=40)
    patterns = PatternEngine.from_mapping(data, timeframe="1D").detect()
    pattern = next(p for p in patterns if p["id"] == "bullish_consolidation")
    assert pattern["family"] == "continuation_consolidation"
    assert pattern["pattern_status"] == "confirmed"
    assert pattern["target_method"] == "flagpole_projection"
    assert len(pattern["points"]) == 4


def test_insufficient_data_returns_empty_result():
    data = {"Date": [0, 1], "Open": [1, 1], "High": [2, 2], "Low": [0, 0],
            "Close": [1, 1], "Volume": [1, 1]}
    assert PatternEngine.from_mapping(data).detect() == []


def test_nan_infinity_and_zero_volume_do_not_crash():
    data = _inverse_hs()
    data["High"][5] = float("inf")
    data["Low"][6] = float("nan")
    data["Volume"] = [0.0] * len(data["Close"])
    patterns = PatternEngine.from_mapping(data).detect()
    assert isinstance(patterns, list)
    assert all(math.isfinite(float(p["completion_score"])) for p in patterns)


def test_invalid_timeframe_and_tolerance_mode_are_rejected():
    with pytest.raises(ValueError):
        resolve_timeframe("2H")
    with pytest.raises(ValueError):
        PatternEngineOptions(timeframe="1D", tolerance_mode="loose").validate()


def test_audit_covers_all_174_candidates_once():
    assert sum(row["count"] for row in PATTERN_REFACTOR_AUDIT) == 174
    cursor = 1
    for row in PATTERN_REFACTOR_AUDIT:
        assert row["range"][0] == cursor
        assert row["range"][1] - row["range"][0] + 1 == row["count"]
        cursor = row["range"][1] + 1
    assert cursor == 175
