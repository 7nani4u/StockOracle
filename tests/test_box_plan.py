"""🧭 박스 실행 플랜 — box sequence detector, decision scenarios, ticks, API attachment, rendering."""
import json
import math
import subprocess

import pytest

from market_briefing.box_plan import build_execution_plan, round_tick, six_questions, tick_size, _pj
from market_briefing.box_sequence import analyze_box_sequence, scan_boxes

NOW = "2026-11-05"


# ── synthetic OHLCV ────────────────────────────────────────────────────────────

def bars_from_closes(closes, spread=0.6, start="2026-01-01"):
    from datetime import date, timedelta
    d0 = date.fromisoformat(start)
    dd = {"Date": [], "Open": [], "High": [], "Low": [], "Close": [], "Volume": []}
    prev = closes[0]
    for i, c in enumerate(closes):
        o = prev
        dd["Date"].append((d0 + timedelta(days=i)).isoformat())
        dd["Open"].append(o)
        dd["High"].append(max(o, c) + spread)
        dd["Low"].append(min(o, c) - spread)
        dd["Close"].append(c)
        dd["Volume"].append(1000.0)
        prev = c
    return dd


def oscillate(low, high, n, period=8):
    mid, amp = (low + high) / 2, (high - low) / 2
    return [mid + amp * math.sin(2 * math.pi * i / period) for i in range(n)]


def staircase_down():
    closes = []
    for level in (200, 185, 170, 155):
        closes += oscillate(level - 5, level + 5, 24)
    return closes


# ── box sequence detector ──────────────────────────────────────────────────────

def test_descending_staircase_is_recognised():
    seq = analyze_box_sequence(bars_from_closes([210.0] * 30 + staircase_down()))
    assert seq["status"] == "valid"
    relations = [b["relation"] for b in seq["boxes"][1:]]
    assert relations and relations.count("down") >= 2
    assert seq["staircase"]["key"] == "descending"


def test_flat_box_then_breakout_exits_up():
    closes = [100.0] * 20 + oscillate(95, 105, 48) + [108, 111, 113, 114, 115]
    seq = analyze_box_sequence(bars_from_closes(closes))
    ref = seq["reference"]
    assert ref["exit"] == "up"
    assert ref["top"] < 108
    assert seq["post_exit"]["closes"][-1] == 115


def test_confirmed_boxes_do_not_change_when_future_bars_arrive():
    closes = [100.0] * 20 + staircase_down() + oscillate(150, 162, 40)
    dd = bars_from_closes(closes)
    cut = len(closes) - 25
    early = analyze_box_sequence({k: v[:cut] for k, v in dd.items()}, {"lookback": 400})
    late = analyze_box_sequence(dd, {"lookback": 400})
    later_by_start = {(b["start_index"], b["confirm_index"]): b for b in late["boxes"]}
    checked = 0
    for box in early["boxes"]:
        key = (box["start_index"], box["confirm_index"])
        if key in later_by_start:
            checked += 1
            assert later_by_start[key]["top"] == box["top"]
            assert later_by_start[key]["bottom"] == box["bottom"]
    assert checked >= 2


def test_inconsistent_bar_uses_body_only_instead_of_blocking_the_plan():
    closes = [100.0] * 20 + oscillate(95, 105, 80)
    dd = bars_from_closes(closes)
    i = len(closes) - 40
    dd["High"][i] = dd["Close"][i] * 0.99      # dividend-adjustment style mismatch (about 1 %)
    seq = analyze_box_sequence(dd)
    assert seq["status"] in {"valid", "no_box"}
    assert seq["data_quality"]["body_only_rows"] == 1 and any("몸통" in w for w in seq["warnings"])
    dd["High"][i] = dd["Close"][i] * 0.90      # 10 % off is a wrong print: still rejected, with the date
    bad = analyze_box_sequence(dd)
    assert bad["status"] == "invalid_data" and dd["Date"][i] in bad["warnings"][0] and "5%" in bad["warnings"][0]


def test_one_day_jump_restarts_the_box_search_after_the_gap():
    closes = [100.0] * 60 + [60.0] + oscillate(55, 65, 60)
    seq = analyze_box_sequence(bars_from_closes(closes))
    assert seq["restart_after_jump"]["change_pct"] == -40.0
    assert seq["status"] == "valid" and seq["reference"]["top"] < 70            # only post-jump bars
    assert seq["atr"] < 5                                                       # gap bar not inflating ATR
    assert any("급변" in w for w in seq["warnings"])
    short = analyze_box_sequence(bars_from_closes([100.0] * 60 + [60.0] + [61.0, 60.5, 61.5, 60.8, 61.2, 60.9, 61.1, 60.7, 61.3]))
    assert short["status"] == "insufficient_data" and short["bars_available"] == 10 and short["bars_required"] == 22
    plan = plan_for({**short, "last_close": 61.3, "last_date": short["last_date"] or "2026-11-04"})
    assert plan["verdict"]["key"] == "unavailable" and "급변 이후 일봉이 10개뿐" in plan["verdict"]["headline"]
    assert "12거래일" in plan["when"]


def test_short_or_invalid_history_is_reported_not_raised():
    assert analyze_box_sequence(bars_from_closes([100.0] * 10))["status"] == "insufficient_data"
    assert analyze_box_sequence({"Close": [1, 2]})["status"] == "invalid_data"
    boxes, forming, _ = scan_boxes([1.0] * 5, [1.0] * 5, [1.0] * 5, [1.0] * 5)
    assert boxes == [] and forming is None


# ── decision scenarios (hand-built sequences) ──────────────────────────────────

FORMING_AFTER_BREAKDOWN = {"top": 47600.0, "bottom": 46000.0, "top_age": 1, "bottom_age": 0, "bars": 4,
                           "start_date": "2026-11-01", "after_exit": "down"}


def make_seq(top=55000.0, bottom=48000.0, atr=1000.0, relation="sideways", exit=None,
             last_close=52000.0, prev_close=51800.0, last_bar=None, post_exit=None,
             last_date="2026-11-04", status="valid", boxes_before=(), forming=None):
    ref = {"start_index": 100, "confirm_index": 108, "end_index": 140, "start_date": "2026-09-01",
           "top": top, "bottom": bottom, "mid": (top + bottom) / 2, "height": top - bottom,
           "height_atr": (top - bottom) / atr, "atr": atr, "bars": 41, "top_touches": 3,
           "bottom_touches": 3, "relation": relation, "exit": exit,
           "exit_index": 141 if exit else None}
    bar = last_bar or {"open": prev_close, "high": max(prev_close, last_close) + 200,
                       "low": min(prev_close, last_close) - 200, "close": last_close, "volume": 1000}
    stair = {"down": "descending", "up": "ascending"}.get(relation, "sideways")
    closes = [last_close * (1 + 0.01 * math.sin(i)) for i in range(80)]
    return {"status": status, "reference": ref if status == "valid" else None,
            "boxes": list(boxes_before) + ([ref] if status == "valid" else []),
            "previous": None, "forming": forming, "staircase": {"key": stair, "label": stair},
            "bars_available": 160, "bars_required": 22,
            "atr": atr, "last_index": 160, "last_date": last_date, "last_close": last_close,
            "prev_close": prev_close, "last_bar": bar, "post_exit": post_exit,
            "closes_for_sigma": closes, "price_jumps": [], "warnings": []}


def plan_for(seq, current=None, market="KRX", **kw):
    return build_execution_plan(seq, current=current, market=market, now=NOW, **kw)


def assert_on_tick(price, market="KRX"):
    tick = tick_size(price, market)
    assert abs(price / tick - round(price / tick)) < 1e-9, (price, tick)


def test_support_buy_when_bottom_touched_and_close_rises():
    seq = make_seq(last_close=48400, prev_close=48100,
                   last_bar={"open": 48100, "high": 48600, "low": 47900, "close": 48400, "volume": 1})
    plan = plan_for(seq, current=48400)
    assert plan["verdict"]["key"] == "buy" and plan["stage"]["key"] == "ready"
    assert plan["strategy"]["key"] == "support"
    pr = plan["prices"]
    assert pr["entry"]["low"] == 48000 and pr["entry"]["high"] == 48500
    assert pr["stop"]["price"] == 47650
    assert [t["price"] for t in pr["targets"]] == [51500, 55000]
    ref = (48000 + 48500) / 2
    assert pr["rr"] == round((55000 - ref) / (ref - 47650), 2)
    for value in (pr["entry"]["low"], pr["entry"]["high"], pr["stop"]["price"], *[t["price"] for t in pr["targets"]]):
        assert_on_tick(value)
    assert all(c["ok"] for c in plan["conditions"])
    assert plan["timing"]["immediate"] is True
    assert_particles_ok(plan)


def assert_particles_ok(plan):
    import re
    text = " ".join(str(plan.get(k) or "") for k in ("action", "when", "cancel")) + " ".join(plan.get("invalidation") or [])
    assert not re.search(r"원(는|를|가|로) ", text), text


@pytest.mark.parametrize("kwargs,current", [
    ({"last_close": 51000, "prev_close": 51100}, 51000),
    ({"relation": "down"}, 52000),
    ({"exit": "down", "last_close": 46500, "prev_close": 47000}, 46500),                      # no forming range at all
    ({"exit": "down", "last_close": 46500, "prev_close": 47000, "forming": FORMING_AFTER_BREAKDOWN}, 46500),
    ({"exit": "down", "last_close": 47200, "prev_close": 47900, "forming": FORMING_AFTER_BREAKDOWN}, 48600),  # intraday recovery
    ({"exit": "down", "last_close": 48300, "prev_close": 47000, "forming": FORMING_AFTER_BREAKDOWN}, 48300),  # back inside
    ({"status": "no_box", "forming": {"top": 53000.0, "bottom": 49000.0, "top_age": 1, "bottom_age": 4, "bars": 9}}, 52000),
    ({"exit": "up", "last_close": 54000, "prev_close": 55800,
      "post_exit": {"closes": [55800, 54000], "lows": [55100, 53800], "highs": [56000, 55900], "bars": 2}}, 54000),
    ({"exit": "up", "last_close": 58500, "prev_close": 57000,
      "post_exit": {"closes": [55800, 57000, 58500], "lows": [55100, 56500, 57800], "highs": [56000, 57300, 58800], "bars": 3}}, 58500),
    ({"exit": "up", "last_close": 62500, "prev_close": 59000, "forming": {"top": 63000.0, "bottom": 60500.0, "top_age": 0, "bottom_age": 2, "bars": 3},
      "post_exit": {"closes": [56000, 59000, 62500], "lows": [55200, 56000, 59000], "highs": [56500, 59500, 63000], "bars": 3}}, 62500),
    ({"top": 50000, "bottom": 48000, "last_close": 48300, "prev_close": 48100}, 48300),
    ({"last_date": "2026-10-01"}, 52000),
])
def test_every_scenario_has_six_answers_and_correct_particles(kwargs, current):
    plan = plan_for(make_seq(**kwargs), current=current)
    assert plan["verdict"]["label"] and plan["action"] and plan["when"] and plan["cancel"]
    assert plan["invalidation"]
    assert_particles_ok(plan)
    sq = six_questions(plan)
    assert sq["all"], sq
    if not sq["data_side_unavailable"]:
        # Every market-side state names a price zone, a stop and a target — never a bare "산출 불가".
        entry = plan["prices"]["entry"]
        assert entry and entry["low"] is not None and plan["prices"]["entry_unavailable"] is None
        assert plan["prices"]["stop"]["price"] and plan["prices"]["targets"]


def test_middle_of_box_waits_for_price_with_touch_estimate():
    plan = plan_for(make_seq(last_close=51000, prev_close=51100), current=51000)
    assert plan["verdict"]["key"] == "conditional" and plan["stage"]["key"] == "price_wait"
    touch = plan["timing"]["touch"]
    assert touch["level"] == 48500 and touch["distance_pct"] < 0
    assert 0 <= touch["p5"]["mid"] <= touch["p_horizon"]["mid"] <= 100
    assert "방향을 예측한 값이 아니" in plan["when"]
    assert "아래" in plan["conditions"][0]["detail"]


def test_descending_box_is_not_bought_at_the_bottom():
    seq = make_seq(relation="down", last_close=48400, prev_close=48100,
                   last_bar={"open": 48100, "high": 48600, "low": 47900, "close": 48400, "volume": 1})
    plan = plan_for(seq, current=48400)
    assert plan["verdict"]["key"] == "watch"
    assert plan["strategy"]["key"] == "breakout"
    assert plan["prices"]["trigger"]["price"] == round_tick(55000 + 100, "KRX", "ceil")
    assert "하락" in plan["action"]


def test_narrow_box_skips_support_and_explains_why():
    seq = make_seq(top=50000, bottom=48000, atr=1000, last_close=48300, prev_close=48100)
    plan = plan_for(seq, current=48300)
    assert plan["verdict"]["key"] == "watch"
    assert plan["strategy"]["key"] == "breakout"
    assert any("2.5 ATR" in w for w in plan["warnings"])
    assert "하단 매수는 하지 않습니다" in plan["action"]


def test_breakdown_excludes_but_still_gives_conditional_reentry_prices():
    seq = make_seq(exit="down", last_close=46500, prev_close=47000, relation="up", forming=FORMING_AFTER_BREAKDOWN)
    plan = plan_for(seq, current=46500)
    assert plan["verdict"]["key"] == "exclude" and plan["stage"]["key"] == "cancelled"
    pr = plan["prices"]
    entry = pr["entry"]
    # Primary = new base box from the forming range (video: a box that stops falling), clearly conditional.
    assert entry["kind"] == "new_base" and entry["conditional"] is True
    assert (entry["low"], entry["high"]) == (46000, 46400)
    assert pr["stop"]["price"] == 45650 and [t["price"] for t in pr["targets"]] == [46800, 47600]
    assert pr["rr"] == round((47600 - 46200) / (46200 - 45650), 2)
    assert pr["holder_stop"]["price"] == 47650           # existing box bottom - 0.35 ATR for current holders
    assert pr["entry_unavailable"] is None
    assert plan["alternative"]["key"] == "reclaim"       # failed-breakdown reclaim as the alternative
    assert plan["alternative"]["entry"]["low"] == 48000 and plan["alternative"]["stop"]["price"] == 47400
    assert plan["timing"]["confirm_bars"] == 3 and "최소 3거래일" in plan["when"]
    assert "신규 매수는 하지 않습니다" in plan["action"] and "손절" in plan["action"]
    assert six_questions(plan)["all"] is True
    assert_particles_ok(plan)


def test_breakdown_back_inside_old_box_prefers_reclaim_candidate():
    seq = make_seq(exit="down", last_close=48200, prev_close=47000, relation="up", forming=FORMING_AFTER_BREAKDOWN)
    plan = plan_for(seq, current=48200)
    assert plan["verdict"]["key"] == "watch"
    assert plan["prices"]["entry"]["kind"] == "reclaim" and plan["alternative"]["key"] == "new_base"
    assert plan["prices"]["entry"]["low"] == 48000 and plan["prices"]["stop"]["price"] == 47400
    assert six_questions(plan)["all"] is True


def test_quote_far_below_the_reentry_zone_is_flagged():
    plan = plan_for(make_seq(exit="down", last_close=46500, prev_close=47000, forming=FORMING_AFTER_BREAKDOWN), current=43000)
    entry = plan["prices"]["entry"]
    assert "새 저점 진행 중" not in entry["condition"]            # said once, as a warning
    assert sum("새 저점이 진행 중" in w for w in plan["warnings"]) == 1
    assert six_questions(plan)["all"] is True


def test_narrow_forming_base_is_flagged_not_hidden():
    forming = {"top": 46300.0, "bottom": 46000.0, "top_age": 0, "bottom_age": 0, "bars": 2, "start_date": "2026-11-03"}
    plan = plan_for(make_seq(exit="down", last_close=46200, prev_close=47000, forming=forming), current=46200)
    assert plan["prices"]["entry"]["conditional"] is True
    assert "2.5 ATR 미만" in plan["prices"]["entry"]["condition"]
    assert "손익비 1:1 미만" in plan["action"]
    assert six_questions(plan)["all"] is True


def test_intraday_recovery_after_breakdown_is_watch_not_buy():
    # Completed close broke the bottom; the live quote is back above it (seen live on AAPL 2026-11-05).
    seq = make_seq(exit="down", last_close=47200, prev_close=47900, relation="up")
    plan = plan_for(seq, current=48600)
    assert plan["verdict"]["key"] == "watch" and plan["stage"]["key"] == "signal_wait"
    # The price zone is a conditional reclaim candidate, never an order: confirmation needs a close.
    assert plan["prices"]["entry"]["kind"] == "reclaim" and plan["prices"]["entry"]["conditional"] is True
    assert plan["timing"]["immediate"] is False
    assert "장중 회복" in plan["verdict"]["headline"] and "종가" in plan["when"]
    assert_particles_ok(plan)


def test_intraday_drop_below_breakout_stop_waits_for_close():
    post = {"closes": [55800, 56500, 55600], "lows": [55100, 55900, 55200], "highs": [56000, 56900, 56100], "bars": 3}
    seq = make_seq(exit="up", last_close=55600, prev_close=56500, post_exit=post)
    plan = plan_for(seq, current=54200)
    assert plan["verdict"]["key"] == "watch" and plan["stage"]["key"] == "signal_wait"
    assert "손절 기준 아래" in plan["verdict"]["headline"]


def test_first_breakout_close_waits_for_confirmation():
    seq = make_seq(exit="up", last_close=55600, prev_close=54800,
                   post_exit={"closes": [55600], "lows": [54900], "highs": [55800], "bars": 1})
    plan = plan_for(seq, current=55600)
    assert plan["verdict"]["key"] == "conditional" and plan["stage"]["key"] == "signal_wait"
    assert plan["timing"]["confirm_bars"] == 1


def test_confirmed_breakout_retest_is_a_buy():
    post = {"closes": [55800, 56500, 55300], "lows": [55000, 55900, 55050], "highs": [56000, 56900, 56100], "bars": 3}
    seq = make_seq(exit="up", last_close=55300, prev_close=56500, post_exit=post,
                   last_bar={"open": 56400, "high": 56500, "low": 55050, "close": 55300, "volume": 1})
    plan = plan_for(seq, current=55300)
    assert plan["verdict"]["key"] == "buy" and plan["strategy"]["key"] == "retest"
    pr = plan["prices"]
    assert pr["entry"]["low"] == 55000 and pr["entry"]["high"] == 55500
    assert pr["stop"]["price"] == 54400
    assert pr["targets"][-1]["price"] == 62000  # top + box height (measured move)
    assert pr["stop"]["price"] < pr["entry"]["low"] < pr["targets"][0]["price"]


def test_extended_breakout_waits_for_retest_not_chase():
    post = {"closes": [55800, 57000, 58500], "lows": [55100, 56500, 57800], "highs": [56000, 57300, 58800], "bars": 3}
    seq = make_seq(exit="up", last_close=58500, prev_close=57000, post_exit=post)
    plan = plan_for(seq, current=58500)
    assert plan["verdict"]["key"] == "conditional" and plan["stage"]["key"] == "price_wait"
    assert "추격" in plan["action"]


def test_failed_breakout_cancels_plan():
    post = {"closes": [55800, 54000], "lows": [55100, 53800], "highs": [56000, 55900], "bars": 2}
    seq = make_seq(exit="up", last_close=54000, prev_close=55800, post_exit=post)
    plan = plan_for(seq, current=54000)
    assert plan["verdict"]["key"] == "exclude" and plan["stage"]["key"] == "cancelled"
    assert "실패" in plan["verdict"]["headline"]
    # Still inside the old range: the old bottom is a conditional support candidate, not a buy.
    entry = plan["prices"]["entry"]
    assert entry["kind"] == "range_support" and entry["conditional"] is True
    assert (entry["low"], entry["high"]) == (48000, 48500) and plan["prices"]["entry_unavailable"] is None
    assert plan["prices"]["holder_stop"]["price"] == 47650 and "손절" in plan["cancel"]
    assert six_questions(plan)["all"] is True


def test_failed_breakout_that_also_loses_the_old_bottom_uses_breakdown_plan():
    post = {"closes": [55800, 47000], "lows": [55100, 46800], "highs": [56000, 55900], "bars": 2}
    seq = make_seq(exit="up", last_close=47000, prev_close=55800, post_exit=post, forming=FORMING_AFTER_BREAKDOWN)
    plan = plan_for(seq, current=47000)
    assert plan["verdict"]["key"] == "exclude" and plan["prices"]["entry"]["kind"] == "new_base"
    assert six_questions(plan)["all"] is True


def test_measured_move_reached_is_watch():
    post = {"closes": [56000, 59000, 62500], "lows": [55200, 56000, 59000], "highs": [56500, 59500, 63000], "bars": 3}
    seq = make_seq(exit="up", last_close=62500, prev_close=59000, post_exit=post)
    plan = plan_for(seq, current=62500)
    assert plan["verdict"]["key"] == "watch"


def test_no_box_shows_forming_range_prices_and_bars_to_confirm():
    forming = {"top": 53000.0, "bottom": 49000.0, "top_age": 1, "bottom_age": 4, "bars": 9, "start_date": "2026-10-20"}
    plan = plan_for(make_seq(status="no_box", forming=forming), current=52000)
    assert plan["verdict"]["key"] == "watch" and plan["box"]["forming"] is True
    entry = plan["prices"]["entry"]
    assert entry["kind"] == "forming_box" and entry["conditional"] is True and (entry["low"], entry["high"]) == (49000, 49500)
    assert plan["timing"]["confirm_bars"] == 2 and "최소 2거래일" in plan["when"]
    assert plan["prices"]["stop"]["price"] == 48650 and plan["prices"]["targets"]
    assert six_questions(plan)["all"] is True


def test_data_side_unavailable_names_the_reason_and_what_to_check():
    stale = plan_for(make_seq(last_date="2026-10-01"), current=52000)
    assert stale["verdict"]["key"] == "unavailable" and stale["prices"]["entry"] is None
    assert "지연" in stale["prices"]["entry_unavailable"] and "최신 일봉" in stale["when"]
    gap = plan_for(make_seq(), current=90000)
    assert gap["verdict"]["key"] == "unavailable" and "수정주가" in gap["when"]
    new_listing = plan_for({"status": "insufficient_data", "warnings": ["박스 산출에 필요한 완료 일봉 부족(9/22개)"],
                            "bars_available": 9, "bars_required": 22})
    assert new_listing["verdict"]["key"] == "unavailable"
    assert "일봉이 9개뿐" in new_listing["verdict"]["headline"] and "13거래일 뒤" in new_listing["when"]
    broken = plan_for({"status": "invalid_data", "warnings": ["유효하지 않은 OHLCV: 2024-01-15 봉의 고·저가가 시·종가와 5.7% 어긋남"]})
    assert broken["verdict"]["key"] == "unavailable" and "원본 시세" in broken["when"]
    for plan in (stale, gap, new_listing, broken):
        assert six_questions(plan)["all"] is True and six_questions(plan)["data_side_unavailable"] is True
        assert "조건 미충족" not in plan["prices"]["entry_unavailable"]


def test_prior_caution_never_upgrades_to_buy():
    seq = make_seq(last_close=48400, prev_close=48100,
                   last_bar={"open": 48100, "high": 48600, "low": 47900, "close": 48400, "volume": 1})
    plan = plan_for(seq, current=48400, risk={"prior_decision": "caution"})
    assert plan["verdict"]["key"] == "conditional"
    assert plan["conditions"][2]["ok"] is False


def test_us_prices_use_cent_ticks():
    seq = make_seq(top=110.0, bottom=100.0, atr=2.0, last_close=100.6, prev_close=100.2,
                   last_bar={"open": 100.2, "high": 101.0, "low": 99.9, "close": 100.6, "volume": 1})
    plan = plan_for(seq, current=100.6, market="US")
    pr = plan["prices"]
    assert pr["entry"] == {**pr["entry"], "low": 100.0, "high": 101.0}
    assert pr["stop"]["price"] == 99.3
    assert "$" in plan["action"]


@pytest.mark.parametrize("price,tick", [(1999, 1), (4990, 5), (19990, 10), (49950, 50),
                                        (199900, 100), (425000, 500), (1773000, 1000)])
def test_krx_ticks_follow_2023_unified_table(price, tick):
    assert tick_size(price, "KRX") == tick


def test_korean_particles_follow_the_price_reading():
    assert _pj(48000, "KRX", "을") == "48,000원을"
    assert _pj(48000, "KRX", "은") == "48,000원은"
    assert _pj(48000, "KRX", "이") == "48,000원이"
    assert _pj(48000, "KRX", "으로") == "48,000원으로"
    assert _pj(12.5, "US", "은") == "$12.50은"
    assert _pj(12.0, "US", "을") == "$12.00을"


# ── service attachment and rendering ───────────────────────────────────────────

def test_apply_box_prediction_attaches_serialisable_plan():
    from market_briefing.box_prediction import apply_box_prediction
    closes = [100.0] * 30 + oscillate(95, 108, 120)
    dd = bars_from_closes(closes)
    response = {"last_close": closes[-1], "market": "US", "buy_price": {}, "risk_scenarios": {},
                "prediction_outlook": {"decision": {"key": "watch"}}}
    out = apply_box_prediction(response, dd)
    plan = out["box_range"]["execution_plan"]
    assert plan["verdict"]["key"] in {"buy", "conditional", "watch", "exclude", "unavailable"}
    assert out["box_range"]["box_sequence"]["status"] == "valid"
    assert "closes_for_sigma" not in out["box_range"]["box_sequence"]
    json.dumps(out["box_range"], allow_nan=False)


def _render(data):
    from api.index import HTML
    source = 'function renderPredictionSections' + HTML.split('function renderPredictionSections', 1)[1].split('function renderTechnicalSignals', 1)[0]
    script = r'''
const vm = require('vm');
const elements = {};
const context = {
 document: {getElementById(id) {return elements[id] ||= {innerHTML:'',textContent:'',hidden:false};}},
 _isFiniteNumber(v) {return v != null && v !== '' && Number.isFinite(Number(v));},
 _escPrediction(v) {return String(v ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;');}, _predictionTone() {return '#aaa';},
 fmt(v) {return Number(v).toLocaleString('en-US') + '원';}, fmtRange(a,b) {return `${Number(a).toLocaleString('en-US')}~${Number(b).toLocaleString('en-US')}원`;}
};
vm.createContext(context); vm.runInContext(SOURCE, context);
context.renderForecast(DATA, true);
console.log(JSON.stringify({plan: elements['box-execution-section'].innerHTML, hidden: elements['box-execution-section'].hidden}));
'''.replace('SOURCE', json.dumps(source)).replace('DATA', json.dumps(data))
    result = subprocess.run(['node', '-'], input=script, text=True, encoding='utf-8',
                            capture_output=True, check=True, timeout=20)
    return json.loads(result.stdout)


def test_decision_card_puts_conclusion_first_and_folds_details():
    seq = make_seq(last_close=51000, prev_close=51100)
    plan = plan_for(seq, current=51000)
    data = {'prediction_outlook': {'decision': {'key': 'watch'}},
            'box_range': {'execution_plan': plan, 'box_sequence': {'boxes': [seq['reference']]},
                          'tactics': {'regime': {'key': 'range_wait', 'label': '대기', 'why': ['x']},
                                      'plan': {'status': 'none'}, 'summary': '레거시'}},
            'buy_price': {'box_stage_display': True}, 'risk_scenarios': None}
    rendered = _render(data)
    html = rendered['plan']
    assert rendered['hidden'] is False
    assert html.count('🧭 박스 실행 플랜') == 1
    assert '조건부 매수' in html and '가격 도달 대기' in html
    head, details = html.split('<details', 1)
    for label in ('현재가', '박스 하단', '박스 상단', '1차 매수 후보', '매수 조건', '대기 기준',
                  '예상 대기 기간', '손절 기준', '1차 목표 · 박스 중간', '2차 목표 · 박스 상단', '예상 손익비', '매수 취소 조건',
                  '💡 지금 무엇을 해야 하나?', '⏳ 언제 매수할 수 있나?', '⚠️ 언제 계획을 취소하나?'):
        assert label in head, label
    assert '48,000~48,500원' in head and '47,650원' in head
    assert '영상에서 확인한 원칙' in details
    # The old midline-magnet/pressure/speed reference card is no longer part of the detail fold.
    for gone in ('참고: 중간선 자석', '매매 국면', '중간선 자석', '압력(프록시)', '분할 청산',
                 '수평 박스도, 유효한 추세 재시험도 없음', '일봉 OHLCV 프록시', '영상 기준: 방향 없는 구간은 매매 제외'):
        assert gone not in html, gone


def test_response_without_execution_plan_renders_nothing():
    # An old/partial deployment sends only `tactics`. The midline-magnet/pressure/speed card is gone:
    # the section stays hidden instead of falling back to it.
    tactics = {'regime': {'key': 'no_trade', 'label': '매매 제외 구간', 'why': ['수평 박스도, 유효한 추세 재시험도 없음']},
               'midline': {'available': False, 'strength_label': '확인 불가', 'state': 'unknown'},
               'speed': {'available': True, 'label': '느림', 'ratio': 0.08, 'state': 'slow'},
               'pressure': {'support': {'label': '확인 불가', 'verdict': 'unknown'},
                            'resistance': {'label': '확인 불가', 'verdict': 'unknown'}},
               'volume': {'ratio': 1.61, 'label': '증가'}, 'plan': {'status': 'none'}, 'summary': '레거시'}
    rendered = _render({'prediction_outlook': {}, 'box_range': {'tactics': tactics},
                        'buy_price': {'box_stage_display': True}, 'risk_scenarios': None})
    assert rendered['plan'].strip() == '' and rendered['hidden'] is True


def test_decision_card_shows_conditional_reentry_price_after_breakdown():
    plan = plan_for(make_seq(exit="down", last_close=46500, prev_close=47000, forming=FORMING_AFTER_BREAKDOWN), current=46500)
    data = {'prediction_outlook': {}, 'box_range': {'execution_plan': plan}, 'buy_price': {'box_stage_display': True},
            'risk_scenarios': None}
    html = _render(data)['plan']
    head = html.split('<details', 1)[0]
    assert '매수 제외' in head and 'bxp-cond">조건부<' in head
    assert '46,000~46,400원' in head and '재진입 후보(조건부)' in head
    assert '45,650원' in head and '이미 보유 중이라면' in head and '47,650원' in head
    assert '산출 불가' not in head and '조건 미충족' not in head


def test_decision_card_tags_every_non_buy_price_zone_as_conditional():
    watch = plan_for(make_seq(relation="down", last_close=52000, prev_close=51800), current=52000)
    buy = plan_for(make_seq(last_close=48400, prev_close=48100,
                            last_bar={"open": 48100, "high": 48600, "low": 47900, "close": 48400, "volume": 1}), current=48400)
    assert watch["verdict"]["key"] == "watch" and buy["verdict"]["key"] == "buy"
    base = {'prediction_outlook': {}, 'buy_price': {'box_stage_display': True}, 'risk_scenarios': None}
    assert 'bxp-cond">조건부<' in _render({**base, 'box_range': {'execution_plan': watch}})['plan']
    assert 'bxp-cond">조건부<' not in _render({**base, 'box_range': {'execution_plan': buy}})['plan']


def test_decision_card_names_the_data_problem_when_data_is_unusable():
    plan = plan_for(make_seq(last_date="2026-10-01"), current=52000)
    html = _render({'prediction_outlook': {}, 'box_range': {'execution_plan': plan}, 'buy_price': {'box_stage_display': True},
                    'risk_scenarios': None})['plan']
    assert '산출 불가' in html and '지연' in html and '조건 미충족' not in html


def test_no_scenario_renders_the_bare_unconditional_message():
    for kwargs, current in [({"exit": "down", "last_close": 46500, "prev_close": 47000}, 46500),
                            ({"status": "no_box"}, 52000), ({"top": 50000, "bottom": 48000, "last_close": 48300, "prev_close": 48100}, 48300),
                            ({"exit": "up", "last_close": 54000, "prev_close": 55800,
                              "post_exit": {"closes": [55800, 54000], "lows": [55100, 53800], "highs": [56000, 55900], "bars": 2}}, 54000)]:
        plan = plan_for(make_seq(**kwargs), current=current)
        html = _render({'prediction_outlook': {}, 'box_range': {'execution_plan': plan}, 'buy_price': {'box_stage_display': True},
                        'risk_scenarios': None})['plan']
        assert '산출 불가 — 조건 미충족' not in html and '매수 가격 산출 안 함' not in html
