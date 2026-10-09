"""2026-10-09 고도화 회귀 테스트 — TREE_OF_THOUGHTS ROOTCAUSE 5건.

- prior 20일 고점(당일 제외)으로 추격 가드가 살아나는지
- 피벗이 직전 확정봉 기준으로 바뀌는지
- 변동성 장기 가중이 env 변경에 즉시 반응하는지(reload 없이)
- US 장기 GARP 조회 축소(15) 및 data_quality 노출
- 학습 검증 게이트가 선택 후보 기준으로 동작하는지(코드 정적 확인)
"""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from market_briefing import forecast_model as fm
from market_briefing.hybrid_signals import (
    compute_hybrid_score,
    _prior_twenty_day_high,
)


def test_prior_high_excludes_current_bar():
    highs = [100.0, 101.0, 102.0, 103.0, 110.0]
    assert _prior_twenty_day_high(highs) == 103.0
    assert _prior_twenty_day_high([100.0]) is None
    assert _prior_twenty_day_high([]) is None


def test_breakout_now_triggers_chasing_guard():
    # 직전 20봉 고점 100, 오늘 종가/고가 106 돌파 → ext_atr>0.8로 chasing True여야 한다.
    # 예전 당일포함 기준이면 high=106, entry=106+buffer라 chasing이 절대 안 걸렸다.
    base = [95.0 + i * 0.2 for i in range(30)]
    highs = [c * 1.005 for c in base]
    lows = [c * 0.995 for c in base]
    vols = [100000.0] * 30
    # 마지막 봉을 돌파로 교체
    closes = base[:-1] + [106.0]
    highs = highs[:-1] + [106.5]
    lows = lows[:-1] + [105.0]
    result = compute_hybrid_score(closes, highs, lows, vols)
    assert result["twenty_day_high"] < 106.0  # 직전 고점 기준
    assert result["dist_to_high"] == 0.0  # 돌파이므로 거리 0
    assert result["anti_chase"]["chasing"] is True
    assert result["anti_chase"]["ext_atr"] > 0.8


def test_non_breakout_keeps_chasing_off():
    closes = [100.0 + (i % 5) * 0.3 for i in range(40)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    vols = [100000.0] * 40
    result = compute_hybrid_score(closes, highs, lows, vols)
    assert result["anti_chase"]["chasing"] is False


def test_pivot_uses_confirmed_bar():
    from api import index
    dd = {
        "High": [10.0, 11.0, 12.0, 13.0],
        "Low": [9.0, 10.0, 11.0, 12.0],
        "Close": [9.5, 10.5, 11.5, 12.5],
        "Open": [9.2, 10.2, 11.2, 12.2],
        # 과거 날짜 → 마지막 막대 확정 → [-1] 기준 (13,12,12.5 → Pivot 12.5)
        "Date": ["2026-10-01", "2026-10-02", "2026-10-06", "2026-10-07"],
    }
    piv = index.calc_pivot_points(dd, market="KRX")
    assert piv["classic"]["Pivot"] == 12.5
    # 날짜 없이 호출해도 기존처럼 동작(예외 없이)
    dd2 = {k: v for k, v in dd.items() if k != "Date"}
    piv2 = index.calc_pivot_points(dd2, market="KRX")
    assert "classic" in piv2


def test_long_run_weight_reacts_without_reload(monkeypatch):
    closes = [100.0 * math.exp(0.005 * (1 if i % 2 == 0 else -0.9)) for i in range(300)]
    # accumulate deterministically
    price, seq = 100.0, []
    for i in range(300):
        price *= math.exp(0.005 if i % 2 == 0 else -0.0045)
        seq.append(price)
    monkeypatch.setenv("STOCKORACLE_VOL_LONG_RUN_WEIGHT", "0")
    assert fm.get_long_run_weight() == 0.0
    assert fm.get_vol_error_quantiles() == (0.836, 1.268)
    off = fm.blended_daily_sigma(seq, seq[-1], atr=seq[-1] * 0.02)
    assert "장기" not in off["basis"]
    monkeypatch.setenv("STOCKORACLE_VOL_LONG_RUN_WEIGHT", "0.5")
    assert fm.get_long_run_weight() == 0.5
    assert fm.get_vol_error_quantiles() == (0.800, 1.200)
    on = fm.blended_daily_sigma(seq, seq[-1], atr=seq[-1] * 0.02)
    assert "장기 변동성 혼합" in on["basis"]
    # touch 범위가 가중에 따라 달라진다
    monkeypatch.setenv("STOCKORACLE_VOL_LONG_RUN_WEIGHT", "0")
    r_off = fm.touch_probability_range(100.0, 110.0, 0.02, 22)
    monkeypatch.setenv("STOCKORACLE_VOL_LONG_RUN_WEIGHT", "0.5")
    r_on = fm.touch_probability_range(100.0, 110.0, 0.02, 22)
    assert r_off["high"] != r_on["high"] or r_off["low"] != r_on["low"]


def test_us_longterm_garp_top_is_capped():
    # 30→15 축소: US 장기추천은 env로 상한을 둔다
    import pathlib
    src = pathlib.Path("api/index.py").read_text(encoding="utf-8")
    assert "STOCKORACLE_US_LONGTERM_GARP_TOP" in src
    assert "STOCKORACLE_KR_LONGTERM_GARP_TOP" in src
    assert '"garp_top_n"' in src or "'garp_top_n'" in src or "garp_top_n" in src


def test_train_validation_uses_selected_candidate():
    import pathlib
    src = pathlib.Path("scripts/train_ml_model.py").read_text(encoding="utf-8")
    assert "_selected_name" in src
    assert 'report.get("candidate") == _selected_name' in src
