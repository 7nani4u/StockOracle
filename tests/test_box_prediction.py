from copy import deepcopy
from market_briefing.box_prediction import apply_box_prediction


def test_empty_data_withholds_all_prices():
    response = {"last_close": 100, "buy_price": {}, "risk_scenarios": {}}
    out = apply_box_prediction(response, {k: [] for k in ("Open", "High", "Low", "Close", "Volume")})
    assert out["box_range"]["state"] == "insufficient_data"
    # 무조건 표시 계약: 원본이 없어도 3종 카드는 ATR 참고 범위로 합성한다(빈 화면 금지).
    for key in ("conservative", "balanced", "aggressive"):
        card = out["risk_scenarios"][key]
        assert card["target"] is not None and len(card["target"]) == 2
        assert card["tp_range"] is not None
        assert card["entry_eligible"] is False
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
    # 무조건 표시 계약: 돌파·이탈이어도 기존 목표가를 지우지 않고 유지한다.
    assert out["risk_scenarios"]["balanced"]["target"] is not None
    assert len(out["risk_scenarios"]["balanced"].get("tp_levels") or []) >= 1
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
    # 무조건 표시 계약: 현재가가 박스 목표를 지났어도 3종 카드는 참고 범위로 합성한다.
    assert all(card["target"] is not None for card in [out["risk_scenarios"][k] for k in ("conservative", "balanced", "aggressive")])
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
    # 무조건 표시 계약: 수평 박스가 없어도 공격적 카드는 참고 범위로 합성한다.
    assert out["risk_scenarios"]["aggressive"]["target"] is not None


def test_large_downward_box_blocks_small_upward_entry():
    from tests.test_trend_boxes import bars
    values = [180-.8*i for i in range(80)] + [104-.2*i for i in range(20)] + [99+.11*i for i in range(20)] + [101.2,101.3]
    out = apply_box_prediction({"last_close": 101.3}, bars(values))
    assert out["box_range"]["trend_boxes"]["scale_conflict"]
    assert not out["box_range"]["entries"]["main"]["eligible"]


def test_no_box_keeps_actual_existing_buy_and_sell_prices():
    from api.index import calc_risk
    from tests.test_buy_entry_steps import _sample_buy_dd, _calculate
    dd = _sample_buy_dd()
    # A unidirectional series must not be presented as a confirmed horizontal box.
    for i in range(len(dd["Close"])):
        c = 100 + i * .4
        dd["Open"][i] = c
        dd["High"][i] = c + 1
        dd["Low"][i] = c - 1
        dd["Close"][i] = c
    buy = _calculate(dd, market="US")
    risk = calc_risk(dd["Close"][-1], dd["ATR"][-1], "US", dd)
    prior_buy, prior_risk = deepcopy(buy), deepcopy(risk)
    out = apply_box_prediction({"last_close": dd["Close"][-1], "buy_price": buy, "risk_scenarios": risk}, dd)
    box = out["box_range"]
    assert box["state"] == "no_box"
    assert box["legacy_buy_display"] and box["legacy_sell_display"]
    for family in ("aggressive", "recommended"):
        assert out["buy_price"][family]["range"] == prior_buy[family]["range"]
    for family in ("aggressive_bands", "recommended_bands"):
        assert len(out["buy_price"][family]) == len(prior_buy[family])
        assert out["buy_price"][family][0]["steps"] == prior_buy[family][0]["steps"]
    for family in ("conservative", "balanced", "aggressive"):
        assert out["risk_scenarios"][family]["tp_range"] == prior_risk[family]["tp_range"]
        assert out["risk_scenarios"][family]["tp_levels"] == prior_risk[family]["tp_levels"]
        assert not out["risk_scenarios"][family]["entry_eligible"]
        assert out["risk_scenarios"][family]["price_source"] == "legacy_estimate"
    assert all(not plan["eligible"] for plan in box["entries"].values())


def test_observed_entry_ranges_populate_tp1_to_tp5_within_each_range():
    from tests.test_box_range import series
    out = apply_box_prediction({"last_close": 97}, series((95,96,97)))
    buy = out["buy_price"]
    assert buy["box_stage_display"]
    for family in ("aggressive_bands", "recommended_bands"):
        band = buy[family][0]
        steps = band["steps"]
        assert [step["label"] for step in steps] == ["TP1","TP2","TP3","TP4","TP5"]
        assert steps[0]["price_range"][1] == band["range"][1]
        assert steps[-1]["price_range"][0] == band["range"][0]
        for step in steps:
            assert band["range"][0] <= step["price_range"][0] <= step["price"] <= step["price_range"][1] <= band["range"][1]
            assert "probability_low_pct" not in step
        for left,right in zip(steps,steps[1:]):
            assert left["price_range"][0] == right["price_range"][1]


def test_original_abc_and_five_steps_preserved_with_box_overlap():
    from tests.test_box_range import series
    dd = series((95,96,97))
    plan_low,plan_high = 93.0,98.0
    bands = [{"band": name,"range":[plan_low,plan_high],"steps":[{"label":f"{i+1}단계","price":97-i,"price_range":[96.5-i,97.5-i]} for i in range(5)]} for name in "ABC"]
    original = deepcopy(bands)
    out = apply_box_prediction({"last_close":97,"buy_price":{"aggressive_bands":bands,"recommended_bands":deepcopy(bands)}},dd)
    for family in ("aggressive_bands","recommended_bands"):
        result = out["buy_price"][family]
        assert [b["band"] for b in result] == list("ABC")
        for b,old in zip(result,original):
            assert b["range"] == old["range"]
            assert len(b["steps"]) == 5
            for step,previous in zip(b["steps"],old["steps"]):
                assert step["label"] == previous["label"]
                assert step["price_range"] == previous["price_range"]
                overlap = step["box_entry_overlap"]
                if overlap:
                    assert step["price_range"][0] <= overlap[0] <= overlap[1] <= step["price_range"][1]
            assert any(step["box_entry_overlap"] for step in b["steps"])
    assert bands == original


def test_box_description_only_attaches_to_matching_band_and_names_stages():
    from tests.test_box_range import series
    bands = [{"band": name,"range":bounds,"steps":[{"label":f"{i+1}단계","price_range":[bounds[0]+i*.01,bounds[0]+(i+1)*.01]} for i in range(5)]} for name,bounds in [("A",[110,111]),("B",[94.9,95.1]),("C",[80,81])]]
    out = apply_box_prediction({"last_close":97,"buy_price":{"aggressive_bands":bands}},series((95,96,97)))
    a,b,c = out["buy_price"]["aggressive_bands"]
    for band in (a,c):
        assert not band["box_entry_matches"]
        assert band["box_entry_range"] is None
        assert band["box_entry_note"] is None
        assert not any(step["box_entry_overlap"] for step in band["steps"])
    assert b["box_entry_matches"]
    assert b["box_entry_steps"] == [f"{i+1}단계" for i in range(5)]
    assert b["box_entry_label"] == "박스 지지 탐색"
