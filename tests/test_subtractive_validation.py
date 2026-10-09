import pytest

from market_briefing.subtractive_validation import evaluate_technique, trade_metrics


def rows(day, good=20, bad=20):
    return [{"entry_date": day, "exit_date": day, "net_return": .04,
             "conditions": {"regime": "good"}} for _ in range(good)] + [
        {"entry_date": day, "exit_date": day, "net_return": -.02,
         "conditions": {"regime": "bad"}} for _ in range(bad)]


def evaluate(events):
    return evaluate_technique(events, "2025-04-01", "2025-08-01")


def test_frozen_pruning_and_metrics():
    node = evaluate(rows("2025-03-01") + rows("2025-07-01") + rows("2025-10-01"))
    assert node["status"] == "PRUNED_KEEP"
    assert node["validated"] is True
    assert node["excluded_conditions"] == [{"condition": "regime", "value": "bad"}]
    assert node["retained"]["final"]["count"] == 20
    assert node["baseline"]["final"]["event_net_return_sum"] == pytest.approx(.4)


def test_final_cannot_select_conditions():
    base = rows("2025-03-01") + rows("2025-07-01")
    first = evaluate(base + rows("2025-10-01"))
    final = rows("2025-10-01")
    for row in final:
        row["net_return"] *= -1
    second = evaluate(base + final)
    assert first["excluded_conditions"] == second["excluded_conditions"]
    assert first["development_candidates"] == second["development_candidates"]
    assert second["status"] == "INCONCLUSIVE"


def test_boundary_crossings_are_purged():
    node = evaluate([{"entry_date": "2025-04-01", "exit_date": "2025-04-02",
                      "net_return": .1, "conditions": {}}])
    assert len(node["purged_trades"]) == 1
    assert node["baseline"]["train"]["count"] == 0
    assert node["status"] == "INSUFFICIENT"


def test_calibration_requires_twenty_affected():
    node = evaluate(rows("2025-03-01") + rows("2025-07-01", bad=19) + rows("2025-10-01"))
    assert node["excluded_conditions"] == []
    assert any(item["reason"] == "insufficient_affected_validation" for item in node["rejections"])


def test_drop_requires_negative_each_split():
    events = rows("2025-03-01", good=0) + rows("2025-07-01", good=0) + rows("2025-10-01", good=0)
    assert evaluate(events)["status"] == "DROP"
    assert evaluate(events)["validated"] is True
    assert evaluate(events[:-1])["status"] == "INSUFFICIENT"


def test_final_exclusion_requires_twenty_affected_without_adaptation():
    node = evaluate(rows("2025-03-01") + rows("2025-07-01") + rows("2025-10-01", bad=19))
    assert node["status"] == "INSUFFICIENT"
    assert node["validated"] is False
    assert node["exclusions"] == [{"condition": "regime", "value": "bad"}]


def test_invalid_and_empty_metrics():
    assert trade_metrics([])["win_rate_wilson_95"] == [None, None]
    node = evaluate([{"entry_date": "2025-03-01", "exit_date": "2025-03-02", "net_return": float("nan")}])
    assert len(node["invalid_trades"]) == 1
    with pytest.raises(ValueError):
        evaluate_technique([], "2025-04-01", "2025-08-01", 19)
