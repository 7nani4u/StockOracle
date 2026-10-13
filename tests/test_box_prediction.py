from copy import deepcopy
from market_briefing.box_prediction import apply_box_prediction


def test_empty_data_withholds_all_prices():
    response = {"last_close": 100, "buy_price": {}, "risk_scenarios": {}}
    out = apply_box_prediction(response, {k: [] for k in ("Open", "High", "Low", "Close", "Volume")})
    assert out["box_range"]["state"] == "insufficient_data"
    assert all(out["risk_scenarios"][key]["target"] is None for key in ("conservative", "balanced", "aggressive"))
    assert out["buy_price"]["recommended_bands"][0]["range"] is None


def test_forming_bar_excluded_and_source_immutable():
    dd = {"Open": [100]*30, "High": [101]*30, "Low": [99]*30, "Close": [100]*30, "Volume": [100]*30, "Date": [str(i) for i in range(30)]}
    before = deepcopy(dd)
    out = apply_box_prediction({"last_close": 100}, dd, in_progress=True)
    assert dd == before
    assert out["box_range"]["last_bar_date"] == "28"
    assert out["chart_data"]["box_range"] is out["box_range"]
    assert out["box_range"]["video_verified"] is True
    assert out["box_range"]["video_evidence"]["review_method"] == "ko_auto_transcript_and_keyframes"


def test_shared_ranges_rebound_risk_and_targets():
    from tests.test_box_range import series
    dd = series((95, 96, 97))
    response = {"last_close": 97, "buy_price": {}, "pullback_analysis": {"target_ext": 1000},
                "prediction_outlook": {"decision": {"key": "caution"}, "levels": {"support": 1}}}
    out = apply_box_prediction(response, dd)
    box = out["box_range"]
    assert not box["entries"]["main"]["eligible"]
    assert out["buy_price"]["recommended"]["range"] == [box["entries"]["main"]["low"], box["entries"]["main"]["high"]]
    assert out["risk_scenarios"]["balanced"]["target"] == [box["resistance"]["low"], box["resistance"]["high"]]
    assert out["pullback_analysis"]["target_ext"] is None
    assert out["prediction_outlook"]["levels"]["support"] == box["support"]["center"]


def test_breakdown_withholds_old_sell_targets():
    from tests.test_box_range import series
    out = apply_box_prediction({"last_close": 91}, series((93, 92, 91)))
    assert out["risk_scenarios"]["balanced"]["target"] is None
    assert not any(p["eligible"] for p in out["box_range"]["entries"].values())


def test_failed_breakout_blocks_entries_and_reconciles_outlook():
    from tests.test_box_range import series
    out = apply_box_prediction({"last_close": 95, "prediction_outlook": {"decision": {"key": "buy"}}}, series((108, 100, 95)))
    assert out["box_range"]["state"] == "false_breakout"
    assert not any(p["eligible"] for p in out["box_range"]["entries"].values())
    assert out["buy_price"]["strategy_rec"]["action_key"] == "wait_breakdown"
    assert out["prediction_outlook"]["decision"]["key"] == "caution"
    assert out["prediction_outlook"]["pre_box_decision"]["key"] == "buy"


def test_realtime_quote_above_target_withholds_and_invalid_quote_blocks_entry():
    from tests.test_box_range import series
    out = apply_box_prediction({"last_close": 120}, series())
    assert all(card["target"] is None for card in [out["risk_scenarios"][k] for k in ("conservative", "balanced", "aggressive")])
    for price in (None, float("nan"), float("inf"), -1, "bad"):
        out = apply_box_prediction({"last_close": price}, series((95, 96, 97)))
        assert not any(p["eligible"] for p in out["box_range"]["entries"].values())


def test_video_upward_retest_enters_without_horizontal_box():
    from tests.test_trend_boxes import bars
    values = [104-.2*i for i in range(20)] + [99+.11*i for i in range(20)] + [101.2,101.3]
    out = apply_box_prediction({"last_close": 101.3}, bars(values))
    box = out["box_range"]
    assert box["state"] == "no_box"
    assert box["trend_boxes"]["transition"]["state"] == "upward_reversal"
    assert box["entries"]["main"]["eligible"]
    assert box["entries"]["main"]["low"] == box["trend_boxes"]["entry_zone"]["low"]
    assert out["risk_scenarios"]["aggressive"]["target"] is None


def test_large_downward_box_blocks_small_upward_entry():
    from tests.test_trend_boxes import bars
    values = [180-.8*i for i in range(80)] + [104-.2*i for i in range(20)] + [99+.11*i for i in range(20)] + [101.2,101.3]
    out = apply_box_prediction({"last_close": 101.3}, bars(values))
    assert out["box_range"]["trend_boxes"]["scale_conflict"]
    assert not out["box_range"]["entries"]["main"]["eligible"]
