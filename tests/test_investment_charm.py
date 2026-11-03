"""Investment charm scoring must be deterministic and safe with incomplete fundamentals."""

from market_briefing.investment_charm import compute_charm_scores, get_key_metrics


def test_charm_scores_use_available_fundamentals_without_fake_rankings():
    info = {
        "revenueGrowth": 0.20,
        "earningsGrowth": 0.18,
        "grossMargins": 0.62,
        "operatingMargins": 0.24,
        "profitMargins": 0.18,
        "returnOnEquity": 0.21,
        "returnOnAssets": 0.10,
        "debtToEquity": 45,
        "currentRatio": 1.9,
        "quickRatio": 1.3,
        "operatingCashflow": 2_000,
        "freeCashflow": 1_200,
        "marketCap": 20_000,
        "totalRevenue": 8_000,
        "netIncomeToCommon": 1_000,
    }

    result = compute_charm_scores(info, {}, "US")

    assert result["smart_score"] is not None
    assert result["available_count"] == 5
    assert all(score is not None for score in result["sub_scores"].values())
    assert result["overall_rank"] is None
    assert result["overall_percentile"] is None
    assert result["industry_rank"] is None
    assert result["industry_percentile"] is None
    assert result["universe_size"] is None


def test_charm_metrics_preserve_negative_per_and_missing_values():
    metrics = get_key_metrics(
        {"trailingPE": -4.5, "returnOnEquity": float("nan"), "dividendYield": 0},
        {},
        "KRX",
    )

    assert metrics["per_str"] == "-4.50"
    assert metrics["psr_str"] == "N/A"
    assert metrics["roe_str"] == "N/A"
    assert metrics["dy_str"] == "0.00%"


def test_key_metrics_supports_all_six_values_for_us_and_krx():
    us = get_key_metrics(
        {
            "marketCap": 1_520_324_000_000,
            "trailingPE": 14.98,
            "priceToBook": 3.02,
            "priceToSalesTrailing12Months": 3.52,
            "returnOnEquity": 0.3079,
            "dividendYield": 0.58,
        },
        {},
        "US",
    )
    krx = get_key_metrics(
        {"marketCap": 10_000_000_000, "priceToSalesTrailing12Months": 1.2},
        {"market_cap_raw": 1_520_324_000_000_000, "per": 14.98, "pbr": 3.02, "roe": 30.79},
        "KRX",
    )

    assert us["market_cap_str"] == "$1.52T"
    assert us["pbr_str"] == "3.02"
    assert us["psr_str"] == "3.52"
    assert us["roe_str"] == "30.79%"
    assert us["dy_str"] == "0.58%"
    assert krx["market_cap_str"] == "1,520조 3,240억"
    assert krx["per_str"] == "14.98"
    assert krx["pbr_str"] == "3.02"


def test_charm_score_requires_three_available_axes():
    result = compute_charm_scores({"returnOnEquity": 0.15}, {}, "US")

    assert result["smart_score"] is None
    assert result["available_count"] < 3
