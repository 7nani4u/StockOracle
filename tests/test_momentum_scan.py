# -*- coding: utf-8 -*-
"""7단계 스캔 모멘텀 지속 편입 테스트. 점수 미반영·승격만 검증."""
from market_briefing.scan_engine import (
    apply_momentum_promotion,
    momentum_conditions,
)
from market_briefing.technique_prune import technique_allowed


def _pass_mo(entry=119.0, stop=110.0):
    return {"available": True, "stage": "PASS", "stage_label": "절반 수성",
            "surge_pct": 25.0, "entry_trigger": entry, "stop_price": stop}


def _cand(**over):
    base = {
        "ticker": "TEST", "price": 119.0, "status": "FAR",
        "entry_trigger": 130.0, "stop_price": 115.0, "distance_pct": 9.0,
        "sleeve": "CORE", "passes_tech_filters": True,
        "shares": 10.0, "risk_amount": 100.0, "risk_pct": 1.0, "total_cost": 500.0,
    }
    base.update(over)
    return base


def test_promotion_sets_ready_with_momentum_levels():
    cands = [_cand()]
    out = apply_momentum_promotion(cands, {"TEST": _pass_mo()}, 10_000_000.0, 1.0)
    assert out == {"promoted": 1}
    cd = cands[0]
    assert cd["status"] == "READY"
    assert cd["status_source"] == "momentum_persistence"
    assert cd["entry_trigger"] == 119.0
    assert cd["stop_price"] == 110.0
    assert cd["distance_pct"] == 0.0
    assert cd["orig_status"] == "FAR"
    assert cd["orig_entry_trigger"] == 130.0
    # 사이징이 모멘텀 기준(119-110=9)으로 재계산됨
    assert cd["shares"] == round(10_000_000.0 * 0.01 / 9.0, 4)


def test_promotion_skips_non_pass_earnings_and_leader_rows():
    cands = [
        _cand(ticker="A", status="WATCH"),
        _cand(ticker="B", status="FAR"),
        _cand(ticker="C", status="EARNINGS_BLOCK"),
        _cand(ticker="D", status="FAR", status_source="leader_reversal"),
    ]
    mo = {
        "A": {"available": True, "stage": "WAIT"},
        "B": {"available": False, "stage": "NONE"},
        "C": _pass_mo(),
        "D": _pass_mo(),
    }
    out = apply_momentum_promotion(cands, mo)
    assert out == {"promoted": 0}
    assert [c["status"] for c in cands] == ["WATCH", "FAR", "EARNINGS_BLOCK", "FAR"]
    assert all("orig_status" not in c for c in cands)


def test_promotion_rejects_invalid_levels():
    # 손절 >= 진입 → 승격 안 됨
    cands = [_cand(ticker="X")]
    out = apply_momentum_promotion(cands, {"X": _pass_mo(entry=110.0, stop=110.0)})
    assert out == {"promoted": 0}
    assert cands[0]["status"] == "FAR"


def test_promotion_never_raises():
    assert apply_momentum_promotion(None, None) == {"promoted": 0}
    assert apply_momentum_promotion([None, "x", {}], {}) == {"promoted": 0}


def test_prune_gate_allows_momentum_without_rules():
    ok, _ = technique_allowed("momentum_persistence", "KRX", momentum_conditions(_pass_mo()))
    assert ok is True
    ok, _ = technique_allowed("momentum_persistence", "US", momentum_conditions(_pass_mo()))
    assert ok is True
    conds = momentum_conditions(_pass_mo())
    assert conds["surge_bucket"] == "25-40%"
