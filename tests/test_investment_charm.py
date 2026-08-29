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


def test_charm_score_requires_three_available_axes():
    result = compute_charm_scores({"returnOnEquity": 0.15}, {}, "US")

    assert result["smart_score"] is None
    assert result["available_count"] < 3
