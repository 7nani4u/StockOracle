"""Peter Lynch-style GARP screen must be strict about reported EPS history."""

import pandas as pd

from market_briefing.peter_lynch import build_peter_lynch_assessment


def _annual_income(eps):
    return pd.DataFrame(
        [eps],
        index=["Diluted EPS"],
        columns=pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]),
    )


def test_peter_lynch_screen_passes_only_all_five_article_thresholds():
    result = build_peter_lynch_assessment(
        {"trailingPE": 18.0, "debtToEquity": 65.0, "marketCap": 5_000_000_000},
        {}, "US", _annual_income([1.0, 1.25, 1.5625, 1.953125]),
    )

    assert result["eligible"] is True
    assert result["status"] == "pass"
    assert result["passed_count"] == 5
    assert result["metrics"]["eps_cagr_3y_pct"] == 25.0
    assert result["metrics"]["peg"] == 0.72


def test_peter_lynch_screen_never_substitutes_missing_eps_growth_or_peg():
    result = build_peter_lynch_assessment(
        {"trailingPE": 18.0, "debtToEquity": 65.0, "marketCap": 5_000_000_000, "pegRatio": 0.7},
        {}, "US", None,
    )

    by_key = {criterion["key"]: criterion for criterion in result["criteria"]}
    assert result["eligible"] is False
    assert result["status"] == "incomplete"
    assert by_key["eps_cagr_3y"]["status"] == "unavailable"
    assert by_key["peg"]["value"] is None
    assert by_key["peg"]["status"] == "unavailable"


def test_peter_lynch_screen_requires_positive_eps_for_cagr():
    result = build_peter_lynch_assessment(
        {"trailingPE": 18.0, "debtToEquity": 65.0, "marketCap": 5_000_000_000},
        {}, "US", _annual_income([-1.0, 0.5, 1.0, 2.0]),
    )

    by_key = {criterion["key"]: criterion for criterion in result["criteria"]}
    assert by_key["eps_cagr_3y"]["passed"] is None
    assert "양수" in by_key["eps_cagr_3y"]["unavailable_reason"]


def test_peter_lynch_screen_uses_krx_naver_per_and_market_cap():
    result = build_peter_lynch_assessment(
        {"debtToEquity": 60.0},
        {"per": 20.0, "market_cap_raw": 10_000_000_000_000},
        "KRX", _annual_income([1.0, 1.3, 1.69, 2.197]),
    )

    assert result["eligible"] is True
    assert result["metrics"]["market_cap"] == 10_000_000_000_000
