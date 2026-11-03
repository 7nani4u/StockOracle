import math

from market_briefing.box_tactics import analyze_box_tactics
from market_briefing.box_prediction import apply_box_prediction


def make_dd(prices, volumes=None, spread=0.2, opens=None, highs=None, lows=None):
    n = len(prices)
    volumes = list(volumes) if volumes is not None else [1000] * n
    opens = list(opens) if opens is not None else list(prices)
    highs = list(highs) if highs is not None else [p + spread for p in prices]
    lows = list(lows) if lows is not None else [p - spread for p in prices]
    return {"opens": opens, "closes": list(prices), "highs": highs,
            "lows": lows, "volumes": volumes}


def make_box(state="range", low=89.5, high=110.5, support_touches=None,
             resistance_touches=None, atr=1.0):
    return {
        "state": state,
        "atr": atr,
        "box": {"low": low, "high": high},
        "support": {"low": low, "high": low + 1.0, "center": (low + low + 1.0) / 2,
                    "touch_indices": list(support_touches or [])},
        "resistance": {"low": high - 1.0, "high": high, "center": (high - 1.0 + high) / 2,
                       "touch_indices": list(resistance_touches or [])},
    }


def series(tail=(100, 100, 100)):
    closes = [100 + 5 * math.sin(i * math.pi / 4) for i in range(64)] + list(tail)
    return {"opens": list(closes), "closes": list(closes),
            "highs": [c + .5 for c in closes], "lows": [c - .5 for c in closes],
            "volumes": [1000] * len(closes)}


def test_ohlcv_pressure_flags_conflicting_candles():
    prices = [100.0] * 30
    dd = make_dd(prices, spread=0.2)
    # 양봉 몸통이지만 종가가 범위 하단 + 긴 위꼬리 → 매도 우위
    dd["opens"][-1], dd["closes"][-1] = 100.0, 100.6
    dd["highs"][-1], dd["lows"][-1] = 102.0, 99.9
    tactics = analyze_box_tactics(dd, make_box(), current=100.6)
    latest = tactics["pressure"]["latest_bar"]
    assert latest["conflict"] == "bullish_body_sell_pressure"
    assert latest["pressure"] < 0

    # 음봉 몸통이지만 종가가 범위 상단 + 긴 아래꼬리 → 매수 우위
    dd["opens"][-1], dd["closes"][-1] = 100.0, 99.4
    dd["highs"][-1], dd["lows"][-1] = 100.2, 97.5
    tactics = analyze_box_tactics(dd, make_box(), current=99.4)
    latest = tactics["pressure"]["latest_bar"]
    assert latest["conflict"] == "bearish_body_buy_pressure"
    assert latest["pressure"] > 0


def test_midline_magnet_strength_tracks_middle_volume():
    prices = [100.2, 109.8] * 20 + [100.0] * 20
    volumes = [100] * 40 + [5000] * 20
    strong = analyze_box_tactics(make_dd(prices, volumes), make_box(), current=100.0)
    assert strong["midline"]["state"] == "strong"
    assert strong["midline"]["volume_share"] > 0.5

    edges = analyze_box_tactics(make_dd([90.2, 109.8] * 30), make_box(), current=90.2)
    assert edges["midline"]["state"] == "weak"
    assert edges["midline"]["volume_share"] < 0.2


def test_speed_separates_fast_and_slow_moves():
    quiet = [100.0, 100.02] * 18
    fast = analyze_box_tactics(make_dd(quiet + [99.6, 100.0, 100.7], spread=0.05),
                               make_box(), current=100.7)
    assert fast["speed"]["state"] == "fast"
    slow = analyze_box_tactics(make_dd(quiet + [100.0, 100.02, 100.04], spread=0.05),
                               make_box(), current=100.04)
    assert slow["speed"]["state"] == "slow"


def test_malformed_input_returns_placeholder_without_raising():
    tactics = analyze_box_tactics({}, {"state": "insufficient_data"}, None, None)
    assert tactics["regime"]["key"] == "no_data"
    assert tactics["plan"]["status"] == "none"
    assert tactics["version"]
    assert tactics["rule_check"]["available"] is False


def test_range_plan_splits_midline_and_opposite_edge():
    prices = [100.0] * 67
    volumes = [5000] * 67
    opens, highs, lows = list(prices), [p + .5 for p in prices], [p - .5 for p in prices]
    # 지지 접촉: 긴 아래꼬리 + 상단 종가 = 매수 압력
    for i in (10, 20, 30, 40):
        opens[i], prices[i] = 89.8, 90.4
        highs[i], lows[i], volumes[i] = 90.6, 89.5, 1500
    # 저항 접촉: 상단 꼬리 + 하단 종가 = 매도 압력
    for i in (50, 55, 60):
        opens[i], prices[i] = 110.2, 109.6
        highs[i], lows[i], volumes[i] = 110.6, 109.4, 800
    # 마지막 3봉은 조용한 기준 대비 빠른 지지 반등
    for idx, (o, c) in zip((64, 65, 66), ((89.4, 89.6), (89.6, 90.0), (90.0, 90.3))):
        opens[idx], prices[idx] = o, c
        highs[idx], lows[idx], volumes[idx] = c + .15, o - .1, 1200
    dd = make_dd(prices, volumes, spread=0.05, opens=opens, highs=highs, lows=lows)
    tactics = analyze_box_tactics(dd, make_box(support_touches=(10, 20, 30, 40),
                                               resistance_touches=(50, 55, 60)),
                                  current=90.3)
    assert tactics["regime"]["key"] == "range_support"
    assert tactics["pressure"]["support"]["verdict"] == "buy"
    plan = tactics["plan"]
    assert plan["status"] == "ready"
    assert [t["pct"] for t in plan["targets"]] == [50, 50]
    assert plan["targets"][0]["price"] == 100.0
    assert 109.4 <= plan["targets"][1]["price"] <= 110.6
    assert plan["stop"]["price"] < 89.5
    assert plan["exit_rule"] == "중간선 50% · 반대편 저항 50%"


def test_breakout_plan_uses_three_r_and_pyramid_after_retest():
    prices = [100.0] * 60 + [111.0, 112.0, 113.0, 113.5, 114.0]
    volumes = [1000] * 60 + [3000] * 5
    opens = list(prices)
    highs = [p + .2 for p in prices]
    lows = [p - .2 for p in prices]
    for i in (10, 20, 30):
        opens[i], prices[i] = 109.8, 110.4
        highs[i], lows[i], volumes[i] = 110.6, 109.7, 1200
    dd = make_dd(prices, volumes, spread=0.2, opens=opens, highs=highs, lows=lows)
    tactics = analyze_box_tactics(dd, make_box("breakout_confirmed",
                                               resistance_touches=(10, 20, 30)),
                                  current=114.0)
    assert tactics["regime"]["key"] == "breakout_confirmed"
    plan = tactics["plan"]
    assert plan["rr_to_final"] == 3.0
    assert [t["pct"] for t in plan["targets"]] == [50, 50]
    assert plan["pyramid"]["enabled"] is True
    assert plan["entry"]["low"] == 109.5
    assert plan["stop"]["price"] < 109.5


def test_rule_check_scores_only_after_signal_close():
    prices = [100.0] * 8 + [90.0, 92.0, 95.0, 98.0, 100.0, 101.0, 100.0, 100.0,
                             90.0, 89.0, 88.0, 88.0, 88.0, 88.0, 88.0, 88.0,
                             100.0, 100.0, 100.0, 100.0]
    tactics = analyze_box_tactics(make_dd(prices, spread=0.3), make_box(), current=100.0)
    check = tactics["rule_check"]
    assert check["support_touches"] >= 2
    assert check["midline_first"] >= 1
    assert check["stop_first"] >= 1
    assert "확률 아님" in check["note"]


def test_prediction_attaches_tactics_and_gates_weak_middle():
    out = apply_box_prediction(
        {"last_close": 97, "prediction_outlook": {"decision": {"key": "watch"}, "levels": {}}},
        series((95, 96, 97)))
    box = out["box_range"]
    tactics = box["tactics"]
    assert tactics["version"]
    assert box["execution_regime"] == tactics["regime"]["key"]
    assert box["midline"] == tactics["midline"]["price"]
    assert out["prediction_outlook"]["box_tactics"] is tactics
    assert out["buy_price"]["box_tactics"] is tactics
    assert not any(plan["eligible"] for plan in box["entries"].values())
    assert "실행 규칙" in box["entries"]["exploration"]["reason"]
    assert out["risk_scenarios"]["balanced"]["box_midline_exit"]["pct"] == 50
    assert out["risk_scenarios"]["balanced"]["box_exit_split"][0]["pct"] == 50


def test_pressure_never_claims_real_order_book():
    tactics = analyze_box_tactics(make_dd([100.0] * 30), make_box(), current=100.0)
    assert "프록시" in tactics["pressure"]["note"]
    assert "실제 호가창 아님" in tactics["pressure"]["note"] or "호가창" in tactics["limits"]
    assert tactics["video_evidence"]["review_method"] == "ko_auto_transcript"
