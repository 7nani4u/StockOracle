"""Long-term recommendations must degrade safely around strict GARP data."""

import pandas as pd

from api.index import (
    HTML,
    _has_valid_latest_ohlcv,
    _longterm_garp_rank,
    calc_event_risk,
    validate_financial_health,
)


def test_garp_qualified_candidate_outranks_a_higher_technical_only_score():
    qualified = {"eligible": True, "passed_count": 5}
    incomplete = {"eligible": False, "passed_count": 4}

    assert _longterm_garp_rank(qualified, 55) > _longterm_garp_rank(incomplete, 90)


def test_latest_ohlcv_validation_rejects_missing_or_non_finite_price_data():
    valid = pd.DataFrame({"Close": [10.0], "High": [11.0], "Low": [9.0], "Volume": [100.0]})
    invalid = pd.DataFrame({"Close": [float("nan")], "High": [11.0], "Low": [9.0], "Volume": [100.0]})

    assert _has_valid_latest_ohlcv(valid) is True
    assert _has_valid_latest_ohlcv(invalid) is False


def test_financial_health_preserves_yahoo_percentage_units_and_missing_state():
    assert validate_financial_health({"debtToEquity": 3.868}) is True
    assert validate_financial_health({"debtToEquity": 1.5}) is True
    assert validate_financial_health({"debtToEquity": 151}) is False
    assert validate_financial_health({}) is None


def test_event_risk_requires_clinical_context_for_fda_headlines():
    generic = calc_event_risk("AAPL", "US", [{"title": "FDA policy update for app stores"}], [], {"days_to_earnings": 99})
    clinical = calc_event_risk("BIOT", "US", [{"title": "FDA clinical trial hold for new drug"}], [], {"days_to_earnings": 99})

    assert generic["score"] == 0
    assert clinical["score"] == 14


def test_longterm_ui_discloses_garp_sources_and_daily_close_timestamp():
    assert "출처:" in HTML
    assert "price_as_of" in HTML
    assert "최근 일봉 종가" in HTML
    assert "연속 4개 사업연도 EPS가 확인되면" in HTML
    assert "'/api/kr/longterm' + (force ? '?refresh=1' : '')" in HTML
    assert "'/api/us/longterm' + (force ? '?refresh=1' : '')" in HTML
