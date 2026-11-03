"""Smart Score comparison-universe ranks must remain deterministic and disclosed."""

from datetime import datetime, timedelta, timezone

from market_briefing.charm_ranking import enrich_charm_with_ranks


NOW = datetime(2026, 8, 29, tzinfo=timezone.utc)


def _record(ticker, score, industry="Software"):
    return {
        "ticker": ticker,
        "smart_score": score,
        "available_count": 5,
        "industry": industry,
        "sub_scores": {
            "growth": score,
            "monopoly": score,
            "safety": score,
            "profitability": score,
            "cash": score,
        },
    }


def _snapshot(records, generated_at=NOW):
    return {
        "universe_id": "smart-score-coverage-v1",
        "generated_at": generated_at.isoformat(),
        "score_model": "investment_charm.v1",
        "markets": {
            "US": {
                "coverage_label": "US 주요 상장사 재무 비교 유니버스",
                "catalog_count": len(records),
                "records": records,
            }
        },
    }


def test_ranking_uses_competition_ties_and_live_target_replacement():
    records = [_record("AAA", 90), _record("BBB", 90), _record("CCC", 80)]
    records.extend(_record(f"X{index}", 70 - index, "Hardware") for index in range(7))
    charm = {"smart_score": 90, "available_count": 5, "industry": "Software", "sub_scores": _record("", 90)["sub_scores"]}

    result = enrich_charm_with_ranks(charm, "AAA", "US", _snapshot(records), NOW)

    assert result["overall_rank"] == 1
    assert result["overall_percentile"] == 10.0
    assert result["universe_size"] == 10
    assert result["industry_rank"] == 1
    assert result["industry_percentile"] == 33.33
    assert result["sub_percentiles"]["growth"] == 10.0
    assert result["ranking_meta"]["status"] == "fresh"


def test_ranking_excludes_unscorable_candidates_and_requires_target_score():
    records = [_record(f"X{index}", 80 - index) for index in range(10)]
    records.append({"ticker": "BAD", "smart_score": None, "available_count": 0})
    charm = {"smart_score": None, "available_count": 2, "sub_scores": {}}

    result = enrich_charm_with_ranks(charm, "BAD", "US", _snapshot(records), NOW)

    assert result["overall_rank"] is None
    assert result["universe_size"] is None
    assert result["ranking_meta"]["status"] == "target_unscorable"


def test_stale_snapshot_does_not_publish_ranks():
    records = [_record(f"X{index}", 80 - index) for index in range(10)]
    charm = {"smart_score": 75, "available_count": 5, "industry": "Software", "sub_scores": _record("", 75)["sub_scores"]}
    stale_at = NOW - timedelta(days=9)

    result = enrich_charm_with_ranks(charm, "TARGET", "US", _snapshot(records, stale_at), NOW)

    assert result["overall_rank"] is None
    assert result["ranking_meta"]["status"] == "stale"
