"""Volume must confirm the original box event without rewriting its geometry."""
from copy import deepcopy

import pytest

from market_briefing.box_sequence import analyze_box_sequence
from market_briefing.box_volume import assess_box_volume, build_volume_evidence
from market_briefing.box_volume import allows_box_entry
from market_briefing.box_plan import build_execution_plan
from tests.test_box_plan import make_seq


def evidence(volumes, *, index=25, close_location=.75):
    n=len(volumes)
    lows=[9.]*n; highs=[11.]*n
    closes=[10.]*n
    closes[-1]=9+2*close_location
    return build_volume_evidence([10.]*n,highs,lows,closes,volumes,
        {"exit":"up","exit_index":index},dates=[f"bar-{i}" for i in range(n)])


def test_confirmation_uses_pre_event_mean_and_keeps_the_breakout_event():
    volumes=[100.]*25+[1000.,200.,30.,30.]
    p=evidence(volumes)
    assert p["breakout"]["initial_ratio"]==10
    assert p["breakout"]["confirmation_ratio"]==6
    assert p["pullback_ratio"]==.05
    assert p["latest_ratio"]<1
    assert assess_box_volume(p,"retest","breakout")["passed"] is True
    later=evidence(volumes+[50.,60.,5000.])
    assert later["breakout"]["confirmation_ratio"]==6


def test_large_retest_volume_does_not_rewrite_an_unconfirmed_original_breakout():
    p=evidence([100.]*25+[20.,20.,300.])
    assert p["latest_ratio"]>1
    assert p["breakout"]["confirmation_ratio"]==.2
    assert assess_box_volume(p,"retest","breakout")["passed"] is False


def test_first_breakout_bar_does_not_read_a_future_confirmation_bar():
    one=evidence([100.]*25+[200.])
    two=evidence([100.]*25+[200.,400.])
    assert one["breakout"]["confirmation_bars"]==1
    assert one["breakout"]["confirmation_ratio"]==2
    assert two["breakout"]["confirmation_bars"]==2
    assert two["breakout"]["confirmation_ratio"]==3


def test_money_flow_uses_volume_weights_instead_of_an_unweighted_price_average():
    profile=evidence([100.]*29+[200.],close_location=.75)
    assert profile["money_flow_5"]==pytest.approx(1/6)


@pytest.mark.parametrize("volumes",[[0.]*30,[100.]*29+[float("nan")],
                                    [100.]*29+[-1.],[100.]*19])
def test_bad_volume_is_unknown_and_never_converted_to_positive_confirmation(volumes):
    p=evidence(volumes)
    assert p["available"] is False
    assert assess_box_volume(p,"retest","breakout")["passed"] is False


def test_zero_current_volume_cannot_confirm_even_when_breakout_was_large():
    p=evidence([100.]*25+[1000.,1000.,0.])
    assert p["available"]
    assert assess_box_volume(p,"retest","breakout")["status"]=="unavailable"


def test_support_and_breakout_are_evaluated_as_different_volume_events():
    p=evidence([100.]*25+[1000.,1000.,50.],close_location=.9)
    assert assess_box_volume(p,"retest","balanced")["passed"] is True
    assert assess_box_volume(p,"support","balanced")["passed"] is False
    assert assess_box_volume(p,"support","breakout")["status"]=="not_applicable"


def test_legacy_sequence_does_not_fabricate_a_volume_confirmation():
    p=assess_box_volume(None,"retest","breakout")
    assert p["passed"] is None and p["status"]=="not_supplied"
    with pytest.raises(ValueError):
        assess_box_volume({},"retest","typo")


def test_volume_does_not_move_box_edges_or_mutate_source_prices():
    import math
    closes=[100.+3*math.sin(i/2) for i in range(100)]
    raw={"Open":closes[:],"High":[c+.5 for c in closes],"Low":[c-.5 for c in closes],
         "Close":closes[:],"Volume":[1000.]*len(closes)}
    original=deepcopy(raw)
    normal=analyze_box_sequence(raw)
    assert raw==original
    spiked=deepcopy(raw)
    spiked["Volume"][-1]=100_000.
    burst=analyze_box_sequence(spiked)
    assert normal["boxes"]==burst["boxes"]
    assert normal["forming"]==burst["forming"]
    assert burst["volume_evidence"]["latest_ratio"]==100
    assert normal["volume_evidence"]["latest_ratio"]==1


def test_policies_can_be_disabled_for_exact_before_after_replay():
    p=evidence([100.]*25+[10.,10.,10.])
    assert assess_box_volume(p,"retest","off")["passed"] is True
    assert assess_box_volume(p,"retest","breakout")["passed"] is False


def retest_sequence(volume):
    seq=make_seq(exit="up",last_close=55300,prev_close=55800,
        post_exit={"closes":[55800,56200,55800,55300],"lows":[55100,55500,55000,55100],
                   "highs":[56100,56500,56000,55600],"bars":4})
    seq["volume_evidence"]=volume
    return seq


def test_weak_breakout_becomes_conditional_without_changing_any_prices():
    profile=evidence([100.]*25+[20.,20.,300.])
    seq=retest_sequence(profile)
    before=deepcopy(seq)
    raw=build_execution_plan(seq,current=55300,market="KRX",now="2026-11-04",params={"volume_policy":"off"})
    checked=build_execution_plan(seq,current=55300,market="KRX",now="2026-11-04")
    assert raw["verdict"]["key"]=="buy"
    assert checked["verdict"]["key"]=="conditional" and not checked["timing"]["immediate"]
    for key in ("stop","targets","trigger","rr"):
        assert checked["prices"][key]==raw["prices"][key]
    assert [checked["prices"]["entry"][k] for k in ("low","high")]==[raw["prices"]["entry"][k] for k in ("low","high")]
    assert checked["prices"]["entry"]["conditional"] and checked["prices"]["entry"]["condition"]
    assert "거래량" in checked["action"] and "동반한 새로운" in checked["when"]
    assert seq==before


def test_confirmed_breakout_stays_buy_even_with_a_quiet_retest():
    seq=retest_sequence(evidence([100.]*25+[500.,500.,20.]))
    checked=build_execution_plan(seq,current=55300,market="KRX",now="2026-11-04")
    assert checked["verdict"]["key"]=="buy"
    assert checked["volume_confirmation"]["status"]=="confirmed"
    assert all(c["ok"] for c in checked["conditions"])


def test_support_volume_remains_evidence_without_a_failed_universal_threshold():
    seq=make_seq(last_close=48400,prev_close=48100,
        last_bar={"open":48100,"high":48600,"low":47900,"close":48400,"volume":10})
    seq["volume_evidence"]=evidence([100.]*25+[10.,10.,10.])
    checked=build_execution_plan(seq,current=48400,market="KRX",now="2026-11-04")
    assert checked["verdict"]["key"]=="buy"
    assert checked["volume_confirmation"]["status"]=="not_applicable"


def test_real_zero_volume_cannot_be_an_immediate_buy_but_legacy_input_stays_readable():
    seq=retest_sequence(evidence([100.]*25+[500.,500.,0.]))
    checked=build_execution_plan(seq,current=55300,market="KRX",now="2026-11-04")
    assert checked["verdict"]["key"]=="conditional"
    assert checked["volume_confirmation"]["status"]=="unavailable"
    del seq["volume_evidence"]
    legacy=build_execution_plan(seq,current=55300,market="KRX",now="2026-11-04")
    assert legacy["verdict"]["key"]=="buy" and legacy["volume_confirmation"]["status"]=="not_supplied"


def test_volume_confirmation_cannot_override_an_existing_failed_price_plan():
    seq=make_seq(exit="up",last_close=47000,prev_close=55800,
        post_exit={"closes":[55800,47000],"lows":[55100,46800],"highs":[56000,55900],"bars":2})
    seq["volume_evidence"]=evidence([100.]*25+[1000.,1000.,1000.])
    checked=build_execution_plan(seq,current=47000,market="KRX",now="2026-11-04")
    assert checked["verdict"]["key"]=="exclude"


def test_service_and_replay_agree_when_original_event_history_is_missing():
    volumes=[100.]*50
    profile=build_volume_evidence([10.]*50,[11.]*50,[9.]*50,[10.]*50,volumes,
        {"exit":"up","exit_index":30},origin=20)
    checked=assess_box_volume(profile,"retest","breakout")
    assert checked["passed"] is None and not allows_box_entry(checked)
    seq=retest_sequence(profile)
    plan=build_execution_plan(seq,current=55300,market="KRX",now="2026-11-04")
    assert plan["verdict"]["key"]=="conditional"
