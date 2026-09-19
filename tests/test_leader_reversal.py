# -*- coding: utf-8 -*-
"""리더주 반전매매 신호 + 미국 $70 상한 필터 + 진입준비 승격 검증."""
import math

from market_briefing.leader_reversal import detect_leader_reversal
from market_briefing.scan_engine import (
    SCAN_US_MAX_PRICE,
    SCAN_COLLECT_CAP_US_FULL, SCAN_COLLECT_CAP_US_LITE,
    apply_leader_promotion, is_scan_price_eligible,
)


def _trend(n, start, step, jitter=0.0):
    return [start + i * step + (jitter if i % 2 else 0.0) for i in range(n)]


def _ohlc(closes, spread=0.01, vol=100_000):
    return (
        [c * (1 + spread) for c in closes],
        [c * (1 - spread) for c in closes],
        [vol] * len(closes),
    )


def _leader_breakout_series():
    """리더(벤치 초과) → -40% 조정 → 저점 후 횡보 → 상단 돌파."""
    # 0~129: 100 → 210 상승 (리더 구간)
    up = _trend(130, 100.0, 110.0 / 129)
    # 130~189: 210 → 126 하락 (-40%)
    down = [210 - (210 - 126) * (i + 1) / 60 for i in range(60)]
    # 190~229: 126 → 138 횡보 상승 (저점 126 유지)
    base = [126 + (138 - 126) * (i + 1) / 40 for i in range(40)]
    # 230~249: 138 → 140 횡보 후 마지막 봉 146으로 상단 돌파
    brk = [138 + (140 - 138) * (i + 1) / 19 for i in range(19)] + [146.0]
    closes = up + down + base + brk
    assert len(closes) == 250
    # 벤치: 120일간 +5% 수준 (리더 대비 열위), 최근 20봉 신저가 형성
    bench = _trend(210, 100.0, 5.0 / 209)
    bench += [b - (i + 1) * 0.15 for i, b in enumerate(bench[-1:] * 40)]
    bench = bench[:250]
    assert len(bench) == 250
    return closes, bench


def test_breakout_stage_when_all_conditions_met():
    closes, bench = _leader_breakout_series()
    highs, lows, _ = _ohlc(closes)
    # 돌파 봉: 마지막 종가가 직전 20봉 고점 위인지 확인 후 판정
    r = detect_leader_reversal(closes, highs, lows, bench)
    assert r["available"] is True
    conds = r["conditions"]
    assert conds["leader"]["passed"] is True
    assert conds["deep_correction"]["passed"] is True
    assert conds["no_new_low"]["passed"] is True
    assert conds["breakout"]["passed"] is True
    assert r["stage"] == "BREAKOUT"
    assert r["entry_trigger"] == r["range_high"]
    assert r["stop_price"] is not None and r["stop_price"] < r["trough"]
    assert r["drawdown_pct"] >= 30.0


def test_wait_breakout_without_range_break():
    closes, bench = _leader_breakout_series()
    # 마지막 20봉을 횡보 상단(≈138) 아래에 머물도록 교체 → 돌파 미충족
    closes = closes[:230] + [136.0] * 20
    assert len(closes) == 250
    highs, lows, _ = _ohlc(closes)
    r = detect_leader_reversal(closes, highs, lows, bench)
    assert r["available"] is True
    assert r["conditions"]["leader"]["passed"] is True
    assert r["conditions"]["deep_correction"]["passed"] is True
    assert r["conditions"]["breakout"]["passed"] is False
    assert r["stage"] in ("WAIT_BREAKOUT", "BASE_BUILDING")


def test_shallow_correction_is_not_leader_reversal():
    closes = _trend(250, 100.0, 0.4)  # 꾸준한 상승, 조정 없음
    bench = _trend(250, 100.0, 0.05)
    highs, lows, _ = _ohlc(closes)
    r = detect_leader_reversal(closes, highs, lows, bench)
    assert r["available"] is True
    assert r["conditions"]["deep_correction"]["passed"] is False
    assert r["stage"] == "NONE"


def test_laggard_vs_bench_is_rejected():
    # 종목은 횡보, 벤치는 강세 → 리더 아님
    closes = _trend(250, 100.0, 0.02)
    closes[130:] = [c * (1 - 0.35 * (i + 1) / 120) for i, c in enumerate(closes[130:])]
    bench = _trend(250, 100.0, 0.3)
    highs, lows, _ = _ohlc(closes)
    r = detect_leader_reversal(closes, highs, lows, bench)
    assert r["available"] is True
    assert r["conditions"]["leader"]["passed"] is False
    assert r["stage"] == "NONE"


def test_falling_knife_with_fresh_low_is_rejected():
    closes, bench = _leader_breakout_series()
    # 저점을 마지막 봉에 새로 만듦 (낙하 나이프)
    closes[-1] = min(closes) * 0.95
    highs, lows, _ = _ohlc(closes)
    r = detect_leader_reversal(closes, highs, lows, bench)
    assert r["conditions"]["no_new_low"]["passed"] is False
    assert r["stage"] != "BREAKOUT"


def test_insufficient_data_is_unavailable():
    r = detect_leader_reversal([100.0] * 60)
    assert r["available"] is False
    assert r["stage"] == "NONE"


def test_never_raises_on_garbage():
    r = detect_leader_reversal([], None, None, None)
    assert r["stage"] == "NONE"
    r = detect_leader_reversal([float("nan")] * 200, None, None, None)
    assert r["stage"] == "NONE"


def test_us_price_cap_boundary():
    assert SCAN_US_MAX_PRICE == 70.0
    assert is_scan_price_eligible("US", 70.0) is True
    assert is_scan_price_eligible("US", 69.99) is True
    assert is_scan_price_eligible("US", 70.01) is False
    assert is_scan_price_eligible("US", 250.0) is False
    assert is_scan_price_eligible("US", 0) is False
    assert is_scan_price_eligible("US", None) is False
    assert is_scan_price_eligible("US", float("nan")) is False
    # KRX는 제한 없음
    assert is_scan_price_eligible("KRX", 500_000) is True
    assert is_scan_price_eligible("KRX", 1_000_000) is True


def test_us_collect_caps_cover_full_universe():
    # 미국 전수 검토: 수집 상한이 확장 유니버스(112종목)를 덮어야 한다
    assert SCAN_COLLECT_CAP_US_FULL >= 112
    assert SCAN_COLLECT_CAP_US_LITE >= 24


def _promo_cand(**over):
    base = {
        "ticker": "TEST", "price": 40.0, "status": "FAR",
        "entry_trigger": 50.0, "stop_price": 38.0, "distance_pct": 25.0,
        "sleeve": "CORE", "passes_tech_filters": True,
        "shares": 10.0, "risk_amount": 100.0, "risk_pct": 1.0, "total_cost": 500.0,
    }
    base.update(over)
    return base


def _promo_leader(**over):
    base = {"available": True, "stage": "BREAKOUT", "stage_label": "리더 반전 돌파",
            "entry_trigger": 42.0, "stop_price": 36.0}
    base.update(over)
    return base


def test_promotion_sets_ready_with_leader_levels():
    cands = [_promo_cand()]
    out = apply_leader_promotion(cands, {"TEST": _promo_leader()}, 10_000_000.0, 1.0)
    assert out == {"promoted": 1}
    cd = cands[0]
    assert cd["status"] == "READY"
    assert cd["status_source"] == "leader_reversal"
    assert cd["entry_trigger"] == 42.0
    assert cd["stop_price"] == 36.0
    assert cd["distance_pct"] == round((42.0 - 40.0) / 40.0 * 100, 2)
    assert cd["orig_status"] == "FAR"
    assert cd["orig_entry_trigger"] == 50.0
    # 사이징이 리더 기준(42-36=6)으로 재계산됨: 10,000,000*1%/6
    assert cd["shares"] == round(10_000_000.0 * 0.01 / 6.0, 4)


def test_promotion_skips_non_breakout_and_earnings_block():
    cands = [
        _promo_cand(ticker="A", status="WATCH"),
        _promo_cand(ticker="B", status="FAR"),
        _promo_cand(ticker="C", status="EARNINGS_BLOCK"),
        _promo_cand(ticker="D", status="FAR"),
    ]
    leader = {
        "A": _promo_leader(stage="WAIT_BREAKOUT"),
        "B": {"available": False, "stage": "NONE"},
        "C": _promo_leader(),
        "D": _promo_leader(entry_trigger=None, stop_price=36.0),
    }
    out = apply_leader_promotion(cands, leader)
    assert out == {"promoted": 0}
    assert [c["status"] for c in cands] == ["WATCH", "FAR", "EARNINGS_BLOCK", "FAR"]
    assert all("status_source" not in c for c in cands)


def test_promotion_rejects_invalid_levels():
    cands = [_promo_cand(ticker="X")]
    # 손절 >= 진입 → 승격 안 됨
    out = apply_leader_promotion(cands, {"X": _promo_leader(entry_trigger=36.0, stop_price=36.0)})
    assert out == {"promoted": 0}
    assert cands[0]["status"] == "FAR"


def test_promotion_never_raises():
    out = apply_leader_promotion(None, None)
    assert out == {"promoted": 0}
    out = apply_leader_promotion([None, "x", {}], {})
    assert out == {"promoted": 0}
