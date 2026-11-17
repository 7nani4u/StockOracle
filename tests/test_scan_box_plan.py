"""7단계 스캔의 🧭 박스 실행 플랜 요약(scan_box_plan) — 완성 조건·선정·가격 상한·유니버스 불변식."""
import math
import re
from datetime import date, timedelta
from pathlib import Path

import pytest

from market_briefing.box_plan import build_execution_plan
from market_briefing.scan_box_plan import (
    INCOMPLETE_REASONS, attach_box_plans, build_scan_box_plan, summarize_plan, summarize_wait)
from market_briefing.scan_engine import (
    SCAN_COLLECT_CAP_US_FULL, SCAN_US_MAX_PRICE, is_scan_price_eligible)

NOW = "2026-11-05"


# ── synthetic OHLCV ────────────────────────────────────────────────────────────

def oscillate(low, high, n, period=8):
    mid, amp = (low + high) / 2, (high - low) / 2
    return [mid + amp * math.sin(2 * math.pi * i / period) for i in range(n)]


def ohlcv(closes, spread=0.6, start="2026-06-01"):
    d0 = date.fromisoformat(start)
    opens, highs, lows, vols, dates = [], [], [], [], []
    prev = closes[0]
    for i, c in enumerate(closes):
        opens.append(prev)
        highs.append(max(prev, c) + spread)
        lows.append(min(prev, c) - spread)
        vols.append(1000.0)
        dates.append((d0 + timedelta(days=i)).isoformat())
        prev = c
    return opens, highs, lows, list(closes), vols, dates


def plan_of(closes, market="US", **kw):
    o, h, l, c, v, d = ohlcv(closes)
    return build_scan_box_plan(o, h, l, c, v, dates=d, market=market, now=d[-1], **kw)


IN_BOX = [100.0] * 20 + oscillate(95, 105, 56)
BREAKOUT = [100.0] * 20 + oscillate(95, 105, 48) + [108, 111, 113, 114, 115]
BREAKDOWN = [100.0] * 20 + oscillate(95, 105, 48) + [93, 90, 88, 87, 86, 85]


# ── completeness ───────────────────────────────────────────────────────────────

def test_in_box_plan_prints_all_five_fields_in_order():
    plan = plan_of(IN_BOX)
    assert plan["complete"] is True and plan["missing"] == [] and plan["reason"] == ""
    t1, t2, stop, entry = plan["target1"], plan["target2"], plan["stop"], plan["reentry"]
    assert t1["label"].endswith("박스 중간") and t2["label"].endswith("박스 상단")
    assert stop["price"] < entry["low"] <= entry["high"] < t1["price"] < t2["price"]
    assert plan["wait"]["text"] and plan["verdict"]["key"]


def test_breakdown_plan_gives_reentry_candidate_with_condition():
    plan = plan_of(BREAKDOWN)
    assert plan["complete"] is True
    entry = plan["reentry"]
    assert entry["conditional"] is True and entry["caption"] == "새 바닥 박스 하단" and entry["condition"]
    assert plan["wait"]["min_bars"] and plan["wait"]["text"].startswith("최소 ")
    assert plan["stop"]["price"] < entry["low"]


def test_breakout_plan_is_not_listed_because_targets_are_measured_moves():
    plan = plan_of(BREAKOUT)
    assert plan["complete"] is False
    assert plan["target_basis"] == "measured" and plan["missing"] == ["measured_targets"]
    assert plan["target1"] is None and plan["target2"] is None   # 박스 중간·상단 자리에 측정 이동을 넣지 않는다
    assert "측정 이동" in plan["reason"]


def test_short_history_and_garbage_are_incomplete_not_errors():
    short = plan_of([100.0] * 10)
    assert short["complete"] is False and short["missing"] == ["no_data"]
    assert build_scan_box_plan([], [], [], [], [])["complete"] is False
    bad = build_scan_box_plan(None, None, None, [1, 2, 3], None)
    assert bad["complete"] is False and bad["missing"] == ["no_data"]
    nan = build_scan_box_plan([float("nan")] * 60, [1.0] * 60, [1.0] * 60, [float("nan")] * 60, [1.0] * 60)
    assert nan["complete"] is False
    for plan in (short, bad, nan):
        assert set(plan["missing"]) <= set(INCOMPLETE_REASONS)


def test_every_missing_key_has_a_reason_text():
    for key, text in INCOMPLETE_REASONS.items():
        assert text, key
    partial = summarize_plan(build_execution_plan({"status": "insufficient_data", "warnings": ["x"]}), 100.0)
    assert partial["complete"] is False and partial["reason"].startswith(INCOMPLETE_REASONS["no_data"])


def test_summary_never_reports_complete_without_a_wait_estimate():
    seq_plan = build_execution_plan(
        {"status": "valid", "reference": None, "boxes": [], "forming": None, "last_close": 100.0,
         "atr": 2.0, "last_date": NOW, "staircase": None, "warnings": []}, current=100.0, now=NOW)
    summary = summarize_plan(seq_plan, 100.0)
    assert summary["complete"] is False


# ── targets must be above the current price (2026-11-07) ────────────────────────

_BOX = [100.0] * 20 + oscillate(95, 105, 48)                     # box 95~105 (middle 100, top ~105.6)
RECLAIM_BELOW_MID = _BOX + [93, 90, 88, 92, 96, 97]              # broke down, back inside under the middle
RECLAIM_ABOVE_MID = _BOX + [93, 90, 88, 92, 97, 101, 102]        # back inside above the middle
REBOUND_ABOVE_TOP = _BOX + [93, 90, 88, 92, 99, 104, 108, 110]   # broke down, then rose above the old top
BREAKDOWN_BOUNCE = _BOX + [93, 90, 88, 87, 86, 85, 88, 91, 93]  # new range forming, bounced above its middle


def _prices(plan):
    return plan["quote"], plan["target1"]["price"], plan["target2"]["price"]


def test_reclaim_with_price_above_the_old_middle_is_not_listed():
    ok = plan_of(RECLAIM_BELOW_MID)
    assert ok["complete"] is True
    quote, t1, t2 = _prices(ok)
    assert quote < t1 < t2                       # 현재가 < 1차 목표 < 2차 목표

    passed = plan_of(RECLAIM_ABOVE_MID)
    assert passed["complete"] is False and passed["missing"] == ["target_mid_passed"]
    assert passed["quote"] > passed["target1"]["price"]                # 엔진 값은 그대로(1차 목표 100 < 현재가 102)
    assert passed["target2"]["price"] > passed["quote"]
    assert INCOMPLETE_REASONS["target_mid_passed"] in passed["reason"]


def test_price_above_the_old_box_top_reports_the_top_not_the_middle():
    plan = plan_of(REBOUND_ABOVE_TOP)
    assert plan["complete"] is False and plan["missing"] == ["target_top_passed"]   # 하나만 보고한다
    assert plan["target2"]["price"] <= plan["quote"]
    assert INCOMPLETE_REASONS["target_top_passed"] in plan["reason"]


def test_new_range_bounced_above_its_middle_is_not_listed():
    # 하락 구간의 형성 범위는 [최저점, 그 뒤 최고점] 이라 많이 반등한 종목은 범위 중간 위에 있다
    plan = plan_of(BREAKDOWN_BOUNCE)
    assert plan["reentry"]["kind"] == "new_base"
    assert plan["complete"] is False and plan["missing"] == ["target_mid_passed"]


def test_every_listed_plan_has_both_targets_strictly_above_the_price():
    series = [IN_BOX, BREAKDOWN, RECLAIM_BELOW_MID, RECLAIM_ABOVE_MID, REBOUND_ABOVE_TOP, BREAKDOWN_BOUNCE,
              IN_BOX + [96.0, 95.5], IN_BOX + [99.0], IN_BOX + [103.0]]
    listed = 0
    for closes in series:
        plan = plan_of(closes)
        if plan["complete"]:
            listed += 1
            quote, t1, t2 = _prices(plan)
            assert quote < t1 < t2, closes[-3:]
    assert listed >= 3          # 불변식이 비어 있는 집합 위에서 통과하는 일이 없도록


def test_target_equal_to_the_price_is_not_above_it():
    plan = {"available": True, "verdict": {"key": "watch"}, "strategy": {"label": "x"}, "stage": {"label": "y"},
            "box": {"top": 105.0, "bottom": 95.0, "mid": 100.0},
            "prices": {"targets": [{"price": 100.0, "role": "box_mid", "label": "1차 목표 · 박스 중간"},
                                   {"price": 105.0, "role": "box_top", "label": "2차 목표 · 박스 상단"}],
                       "stop": {"price": 90.0}, "entry": {"low": 95.0, "high": 97.0, "kind": "reclaim"}},
            "timing": {"immediate": False, "confirm_bars": 3, "estimate_label": "x"}}
    assert summarize_plan(plan, 100.0)["missing"] == ["target_mid_passed"]
    assert summarize_plan(plan, 99.99)["complete"] is True
    assert summarize_plan(plan, 105.0)["missing"] == ["target_top_passed"]
    unknown = summarize_plan(plan, None)
    assert unknown["complete"] is False and unknown["missing"] == ["quote"]
    assert set(unknown["missing"]) <= set(INCOMPLETE_REASONS)


def test_passed_targets_are_counted_in_the_exclusion_statistics():
    snaps = {"A": _Snap(RECLAIM_BELOW_MID), "B": _Snap(RECLAIM_ABOVE_MID), "C": _Snap(REBOUND_ABOVE_TOP),
             "D": _Snap(BREAKDOWN_BOUNCE)}
    cands = [{"ticker": t} for t in "ABCD"]
    stats = attach_box_plans(cands, snaps, market="US", now=NOW)
    assert stats["complete"] == 1 and stats["excluded"] == {"target_mid_passed": 2, "target_top_passed": 1}


# ── in-progress bar ────────────────────────────────────────────────────────────

def test_in_progress_bar_is_excluded_from_the_box_but_is_the_live_quote():
    base = IN_BOX + [96.0, 95.5]
    done = plan_of(base)
    live = plan_of(base + [97.0], in_progress=True)
    assert done["complete"] and live["complete"]
    assert live["quote"] == 97.0 and done["quote"] == 95.5
    # 같은 완료 봉으로 계산하므로 박스·목표·손절은 같다
    for key in ("target1", "target2", "stop"):
        assert live[key]["price"] == done[key]["price"]
    # 진행 중 봉의 극단값이 박스 경계를 바꾸지 못한다
    spike = plan_of(base + [160.0], in_progress=True)
    assert spike["complete"] is False   # 현재가와 완료 봉 종가가 35% 넘게 어긋나면 계산하지 않는다


# ── wait text ──────────────────────────────────────────────────────────────────

def test_wait_text_variants():
    assert summarize_wait(None) is None
    assert summarize_wait({"estimate_label": "x"}) is None                       # 말할 근거가 없으면 None
    now = summarize_wait({"immediate": True, "confirm_bars": 0, "estimate_label": "대기 없음"})
    assert now["text"] == "대기 없음" and now["immediate"] is True
    nxt = summarize_wait({"immediate": False, "confirm_bars": 0, "estimate_label": "a"})
    assert nxt["text"] == "다음 봉 마감 확인"
    touch = {"p_horizon": {"mid": 62.4}, "horizon_days": 20, "days_if_reached": [2, 6]}
    both = summarize_wait({"confirm_bars": 3, "touch": touch, "estimate_label": "b"})
    assert both["text"] == "최소 3거래일"
    assert both["sub"] == "가격 도달 2~6거래일째 · 20거래일 내 도달 가능성 약 62%"
    assert both["touch_pct"] == 62 and "방향을 예측한 값이 아니라" in both["detail"]
    same = summarize_wait({"confirm_bars": 1, "touch": {**touch, "days_if_reached": [3, 3]}})
    assert same["sub"].startswith("가격 도달 3거래일째")
    only_touch = summarize_wait({"touch": touch})
    assert only_touch["text"] == "가격 도달 2~6거래일째"


# ── 예상 기간 · 도달 확률 (2026-11-07 표 형식) ───────────────────────────────────────

def test_period_is_the_engine_day_window_or_a_stated_minimum_never_invented():
    touch = {"p_horizon": {"low": 40, "mid": 50, "high": 60}, "horizon_days": 20, "days_if_reached": [1, 8]}
    assert summarize_wait({"confirm_bars": 3, "touch": touch})["period"] == {
        "text": "1~8일", "kind": "window", "min_days": 1, "max_days": 8}
    one = summarize_wait({"confirm_bars": 3, "touch": {**touch, "days_if_reached": [5, 5]}})["period"]
    assert one["text"] == "5일"                                  # 최소=최대면 범위를 만들지 않는다
    no_window = summarize_wait({"confirm_bars": 3, "touch": {**touch, "days_if_reached": None}})["period"]
    assert no_window == {"text": "최소 3일", "kind": "min_bars", "min_days": 3, "max_days": None}
    assert summarize_wait({"confirm_bars": 0})["period"]["text"] == "다음 봉 마감"
    assert summarize_wait({"immediate": True, "confirm_bars": 0})["period"]["text"] == "대기 없음"
    # 기간 창도, 최소 대기도 없고 확률만 있으면 기간은 비어 있다(→ 화면 '산출 불가'), 게이트(wait None)는 그대로
    assert summarize_wait({"touch": {"p_horizon": {"mid": 50}, "horizon_days": 20}}) is None


def test_probability_is_low_to_high_with_one_decimal_and_one_percent_sign():
    timing = {"confirm_bars": 1, "touch": {
        "p_horizon": {"low": 73, "mid": 78, "high": 82}, "p_horizon_exact": {"low": 73.46, "mid": 77.9, "high": 82.24},
        "horizon_days": 20, "days_if_reached": [1, 8], "level": 9.7}}
    prob = summarize_wait(timing)["probability"]
    assert prob["text"] == "73.5~82.2%" and (prob["low"], prob["high"]) == (73.46, 82.24)   # 반올림은 표시에서만
    assert prob["text"].count("%") == 1 and prob["horizon_days"] == 20 and prob["level"] == 9.7
    # 정수 반올림 값만 있으면 그 실제 값을 소수 첫째 자리로 쓴다(없는 소수를 만들지 않음)
    legacy = {**timing, "touch": {**timing["touch"], "p_horizon_exact": None}}
    assert summarize_wait(legacy)["probability"]["text"] == "73.0~82.0%"
    # 범위가 한 점으로 모이면(거의 확실) 단일 값
    sure = {**timing, "touch": {**timing["touch"], "p_horizon_exact": {"low": 100.0, "mid": 100.0, "high": 100.0}}}
    assert summarize_wait(sure)["probability"]["text"] == "100.0%"
    # 하한이 없으면 중앙값 하나만(범위를 지어내지 않는다)
    mid_only = {**timing, "touch": {**timing["touch"], "p_horizon_exact": {"mid": 64.25}}}
    assert summarize_wait(mid_only)["probability"]["text"] == "64.2%"
    # 확률 근거가 없으면 None
    assert summarize_wait({"confirm_bars": 2})["probability"] is None
    assert summarize_wait({"confirm_bars": 2, "touch": {"horizon_days": 20}})["probability"] is None


def test_touch_model_exposes_one_decimal_probabilities_without_changing_the_integers():
    from market_briefing.box_plan import _timing_touch
    closes = oscillate(95, 105, 120)
    touch = _timing_touch(97.0, 101.0, closes, 1.5, 20)
    # 기존 정수 값은 그대로이고(예측 탭이 쓴다), 새 값은 같은 확률의 소수 첫째 자리 표현이다
    assert all(touch["p_horizon"][k] == round(touch["p_horizon_exact"][k]) or
               abs(touch["p_horizon"][k] - touch["p_horizon_exact"][k]) <= 0.5 for k in ("low", "mid", "high"))
    assert all(isinstance(v, float) and round(v, 1) == v for v in touch["p_horizon_exact"].values())
    assert touch["p_horizon_exact"]["low"] <= touch["p_horizon_exact"]["mid"] <= touch["p_horizon_exact"]["high"]


def test_probability_names_the_price_it_refers_to_not_the_first_target():
    plan = plan_of(BREAKDOWN)
    prob = plan["wait"]["probability"]
    assert prob and prob["level"] is not None and prob["level_label"]
    assert prob["level_label"] in {"재진입 후보 구간 상단", "이탈 실패 확인가", "돌파 확인 가격", "돌파 확인가(잠정)", "기준가"}
    assert prob["level"] != plan["target1"]["price"]
    assert plan["wait"]["period"]["text"] and plan["wait"]["period"]["kind"] in {"window", "min_bars", "next_bar", "immediate"}


# ── selection statistics ───────────────────────────────────────────────────────

class _Snap:
    def __init__(self, closes):
        o, h, l, c, v, _ = ohlcv(closes)
        self.opens, self.highs, self.lows, self.closes, self.volumes = o, h, l, c, v


def test_attach_box_plans_counts_complete_and_excluded():
    snaps = {"IN": _Snap(IN_BOX), "UP": _Snap(BREAKOUT), "DOWN": _Snap(BREAKDOWN), "TINY": _Snap([100.0] * 10)}
    cands = [{"ticker": t} for t in ("IN", "UP", "DOWN", "TINY", "GONE")]
    stats = attach_box_plans(cands, snaps, market="US", now=NOW)
    assert stats["evaluated"] == 5 and stats["complete"] == 2 and stats["incomplete"] == 3
    assert stats["excluded"] == {"measured_targets": 1, "no_data": 2}
    assert [bool(c["box_plan"]["complete"]) for c in cands] == [True, False, True, False, False]


def test_attach_box_plans_uses_the_in_progress_callable():
    snaps = {"IN": _Snap(IN_BOX + [96.0, 95.5, 97.0])}
    cands = [{"ticker": "IN"}]
    attach_box_plans(cands, snaps, market="US", in_progress=lambda t: True, now=NOW)
    assert cands[0]["box_plan"]["quote"] == 97.0
    boom = [{"ticker": "IN"}]
    attach_box_plans(boom, snaps, market="US", in_progress=lambda t: 1 / 0, now=NOW)   # 판정 실패는 완료 봉으로 취급
    assert boom[0]["box_plan"]["complete"] is True


# ── US price cap ───────────────────────────────────────────────────────────────

def test_us_price_cap_is_23_dollars_inclusive():
    assert SCAN_US_MAX_PRICE == 23.0
    assert is_scan_price_eligible("US", 23.0) and is_scan_price_eligible("US", 0.5)
    assert not is_scan_price_eligible("US", 23.01) and not is_scan_price_eligible("US", 70.0)
    assert not is_scan_price_eligible("US", 0) and not is_scan_price_eligible("US", float("nan"))
    assert is_scan_price_eligible("KRX", 1_000_000)   # 한국은 상한 없음


def _universe(name):
    source = (Path(__file__).resolve().parents[1] / "api" / "index.py").read_text(encoding="utf-8")
    start = source.index(f"{name} = [")
    end = source.index("\n            ]", start)
    return re.findall(r'\("([^"]+)",\s*"([^"]*)",\s*"(\w+)"\)', source[start:end])


def test_us_universe_is_unique_complete_and_fits_the_collect_cap():
    rows = _universe("_SCAN_US_UNIVERSE")
    tickers = [t for t, _, _ in rows]
    assert len(tickers) == len(set(tickers))
    assert all(name.strip() for _, name, _ in rows)
    assert {tier for _, _, tier in rows} <= {"LARGE", "MID", "SMALL"}
    # 수집 상한이 목록보다 작으면 뒤쪽 종목이 조용히 잘린다
    assert len(tickers) <= SCAN_COLLECT_CAP_US_FULL
    # $23 이하가 나올 수 없는 초대형주는 수집하지 않는다(2026-11-06 종가 $69 초과)
    for expensive in ("AAPL", "NVDA", "MSFT", "LLY", "COST", "MU"):
        assert expensive not in tickers
    # 거래 종료·합병된 종목은 빠져 있다
    for dead in ("PARA", "WBD", "EA"):
        assert dead not in tickers


def test_krx_universe_fits_its_collect_cap():
    rows = _universe("_SCAN_KRX_UNIVERSE")
    tickers = [t for t, _, _ in rows]
    assert len(tickers) == len(set(tickers))
    from api.index import SCAN_COLLECT_CAP_FULL
    assert len(tickers) <= SCAN_COLLECT_CAP_FULL


# ── 3차 목표 = 예측 탭 ⚖️ 중립적 TP2 의 가장 높은 목표 가격 (2026-11-07) ────────────────

from market_briefing.scan_box_plan import apply_target3_gate, summarize_target3  # noqa: E402


def _balanced(levels):
    return {"label": "중립적", "tp_levels": levels}


def test_third_target_is_the_highest_price_of_the_neutral_tp2_range():
    card = _balanced([
        {"price": 14.0, "price_range": [13.8, 14.2]},
        {"price": 15.27, "price_range": [15.19, 15.35], "return_pct": 10.2, "prob_pct": 61.0,
         "days_min": 4, "days_max": 9},
        {"price": 16.0, "price_range": [15.7, 16.3]}])
    t3 = summarize_target3(card, 13.85, 14.54)
    assert t3["price"] == 15.35 == t3["high"] and t3["low"] == 15.19      # 범위 중 가장 높은 가격, TP1·TP3 이 아님
    assert t3["level"] == "TP2" and t3["scenario"] == "중립적" and t3["days"] == [4, 9]
    assert t3["pct_from_quote"] == round((15.35 / 13.85 - 1) * 100, 2) and t3["not_above_target2"] is False
    # 범위가 뒤집혀 들어와도 가장 높은 값
    assert summarize_target3(_balanced([{}, {"price_range": [15.35, 15.19]}]), 13.85)["price"] == 15.35


def test_third_target_below_or_equal_to_the_second_is_flagged():
    card = _balanced([{"price": 1}, {"price_range": [22.39, 22.61]}])
    below = summarize_target3(card, 21.48, 23.0)
    assert below["price"] == 22.61 and below["not_above_target2"] is True     # 값은 그대로, 플래그만 켠다(제외는 게이트가)
    assert summarize_target3(card, 21.48, 22.61)["not_above_target2"] is True   # 같아도 '높지 않음'
    assert summarize_target3(card, 21.48, 22.60)["not_above_target2"] is False
    assert summarize_target3(card, 21.48, None)["not_above_target2"] is False   # 2차 목표를 모르면 단정하지 않는다


def test_third_target_falls_back_to_the_level_price_and_never_raises():
    assert summarize_target3(_balanced([{}, {"price": 5.5}]), 5.0)["price"] == 5.5
    for bad in (None, {}, "x", 7, {"tp_levels": None}, {"tp_levels": [{"price": 1}]},
                {"tp_levels": [{}, None]}, {"tp_levels": [{}, {"price_range": ["a", None], "price": None}]},
                {"tp_levels": [{}, {"price_range": [float("nan"), float("inf")]}]}):
        assert summarize_target3(bad, 10.0, 11.0) is None, bad
    assert summarize_target3(_balanced([{}, {"price": 5.5}]), None)["pct_from_quote"] is None


def test_summarize_plan_leaves_the_third_target_to_the_server_and_the_gate():
    plan = plan_of(IN_BOX)
    assert plan["target3"] is None            # summarize_plan 은 비워 두고, 서버가 선정 전에 채운다
    assert plan["complete"] is True and "target3" not in plan["missing"]


def _listed(price, flag, quote=10.0):
    plan = {"complete": True, "missing": [], "reason": "", "quote": quote,
            "target2": {"price": 11.0}, "target3": {"price": price, "not_above_target2": flag} if price else None}
    return {"ticker": f"T{price}{flag}", "box_plan": plan}


def test_gate_drops_stocks_whose_third_target_is_missing_or_not_above_the_second():
    ok, low, none = _listed(12.0, False), _listed(10.9, True), _listed(None, False)
    done = {"complete": False, "missing": ["stop"], "reason": "x", "target3": None}
    already = {"ticker": "X", "box_plan": dict(done)}
    stats = {"evaluated": 5, "complete": 3, "incomplete": 2, "excluded": {"stop": 1, "measured_targets": 1}}
    cands = [ok, low, none, already, {"ticker": "NOPLAN"}]
    assert apply_target3_gate(cands, stats) == 2
    assert ok["box_plan"]["complete"] is True and ok["box_plan"]["missing"] == []
    assert low["box_plan"]["complete"] is False and low["box_plan"]["missing"] == ["target3_passed"]
    assert none["box_plan"]["complete"] is False and none["box_plan"]["missing"] == ["target3"]
    assert INCOMPLETE_REASONS["target3_passed"] in low["box_plan"]["reason"]
    assert already["box_plan"]["missing"] == ["stop"]                         # 이미 미완성인 종목은 건드리지 않는다
    # 통계도 같이 맞춘다: 완성 3 → 1, 미완성 2 → 4, 사유별 건수
    assert stats == {"evaluated": 5, "complete": 1, "incomplete": 4,
                     "excluded": {"stop": 1, "measured_targets": 1, "target3_passed": 1, "target3": 1}}
    assert apply_target3_gate(cands, stats) == 0                              # 두 번 돌려도 같은 결과
    assert stats["complete"] == 1 and stats["incomplete"] == 4
    assert apply_target3_gate(None) == 0 and apply_target3_gate([{"box_plan": None}, {}], {}) == 0


def test_gate_keeps_stocks_that_were_skipped_for_time_instead_of_emptying_the_list():
    slow = _listed(None, False)
    slow["box_plan"]["target3_unchecked"] = True        # 시간 예산 초과로 계산을 건너뜀 — 2차 이하로 확인된 것이 아니다
    failed = _listed(None, False)                       # 계산했는데 못 구함 → 제외
    stats = {"evaluated": 2, "complete": 2, "incomplete": 0, "excluded": {}}
    assert apply_target3_gate([slow, failed], stats) == 1
    assert slow["box_plan"]["complete"] is True and failed["box_plan"]["complete"] is False
    assert stats["complete"] == 1 and stats["target3_unchecked"] == 1 and stats["excluded"] == {"target3": 1}


def test_gate_makes_listed_stocks_a_strict_ladder_from_real_plans():
    # 합성 가격 열로 만든 실제 플랜 + 실제 calc_risk 3차 목표: 남는 종목은 현재가 < 1차 < 2차 < 3차
    from api.index import _scan_attach_target3

    series = {"A": IN_BOX + [96.0, 95.5], "B": RECLAIM_BELOW_MID, "C": BREAKDOWN, "D": IN_BOX + [97.0]}
    snaps = {k: _Snap(v) for k, v in series.items()}
    cands = [{"ticker": k, "box_plan": plan_of(v)} for k, v in series.items()]
    pool = [cd for cd in cands if cd["box_plan"]["complete"]]
    assert len(pool) >= 3
    assert _scan_attach_target3(pool, snaps, "US") == len(pool)
    stats = {"evaluated": len(cands), "complete": len(pool), "incomplete": len(cands) - len(pool), "excluded": {}}
    apply_target3_gate(cands, stats)
    kept = [cd for cd in cands if cd["box_plan"]["complete"]]
    assert stats["complete"] == len(kept) and 0 < len(kept) < len(pool)   # 남는 종목도 빠지는 종목도 있다(공집합 통과 방지)
    for cd in kept:
        bp = cd["box_plan"]
        assert bp["quote"] < bp["target1"]["price"] < bp["target2"]["price"] < bp["target3"]["price"]


def test_scan_attaches_the_tab_calc_risk_neutral_tp2_to_plan_cards():
    import pandas as pd
    from api.index import _scan_attach_target3, add_indicators, calc_risk

    series = IN_BOX + [96.0, 95.5]
    snap = _Snap(series)
    shown = {"ticker": "IN", "box_plan": plan_of(series)}
    other = {"ticker": "GONE", "box_plan": plan_of(series)}                       # 스냅샷 없음
    plain = {"ticker": "IN"}                                                      # 플랜 없음 → 건드리지 않음
    done = _scan_attach_target3([shown, other, plain], {"IN": snap}, "US")
    assert done == 1 and other["box_plan"]["target3"] is None and "box_plan" not in plain

    # 같은 계산(예측 탭 calc_risk)의 중립적 TP2 목표 가격 범위 중 가장 높은 가격과 일치해야 한다
    frame = add_indicators(pd.DataFrame({"Open": snap.opens, "High": snap.highs, "Low": snap.lows,
                                         "Close": snap.closes, "Volume": snap.volumes}), market="US")
    dd = frame.where(pd.notna(frame), other=None).to_dict(orient="list")
    tp2 = calc_risk(snap.closes[-1], dd["ATR"][-1], "US", dd)["balanced"]["tp_levels"][1]
    t3 = shown["box_plan"]["target3"]
    assert t3["price"] == max(tp2["price_range"]) and t3["low"] == min(tp2["price_range"])
    assert t3["price"] > shown["box_plan"]["quote"]
    assert shown["box_plan"]["complete"] is True      # 완성 여부·다른 항목은 그대로


def test_scan_target3_failure_is_isolated_per_stock():
    from api.index import _scan_attach_target3

    class _Broken:
        opens = highs = lows = closes = volumes = None

    good = {"ticker": "IN", "box_plan": plan_of(IN_BOX)}
    bad = {"ticker": "BAD", "box_plan": plan_of(IN_BOX)}
    noopen = {"ticker": "NOOPEN", "box_plan": plan_of(IN_BOX)}
    no_open_snap = _Snap(IN_BOX)
    no_open_snap.opens = None
    done = _scan_attach_target3([good, bad, noopen], {"IN": _Snap(IN_BOX), "BAD": _Broken(), "NOOPEN": no_open_snap}, "US")
    assert done == 1 and good["box_plan"]["target3"] and bad["box_plan"]["target3"] is None and noopen["box_plan"]["target3"] is None


def test_scan_target3_skips_remaining_stocks_after_the_deadline():
    import time
    from api.index import _scan_attach_target3

    snaps = {"IN": _Snap(IN_BOX)}
    late = {"ticker": "IN", "box_plan": plan_of(IN_BOX)}
    assert _scan_attach_target3([late], snaps, "US", deadline=time.time() - 1.0) == 0
    assert late["box_plan"]["target3"] is None and late["box_plan"]["target3_unchecked"] is True   # 못 구했다고 표시
    on_time = {"ticker": "IN", "box_plan": plan_of(IN_BOX)}
    assert _scan_attach_target3([on_time], snaps, "US", deadline=time.time() + 60.0) == 1
    assert on_time["box_plan"]["target3"]["price"] > 0 and "target3_unchecked" not in on_time["box_plan"]
