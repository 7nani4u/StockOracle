"""2026-10-09 고도화 회귀 테스트 — 구조 분석 3항목 + 기존 고도화 유지.

- 추격 가드: 직전20봉 기준으로 표시는 살아나되 점수 반영은 기본 꺼짐
  (ext dead + 스캔 플래그 무관련 → 근거 없이 FWS에 넣지 않음)
- 피벗 [-2]·나머지 16곳 거래량: 영향 작아 그대로 둠(고정 행위 고정)
- 변동성 장기 가중 env 즉시반영·GARP 15·게이트 선택후보는 유지
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
    # 직전 20봉 고점 기준이면 돌파일에 ext_atr>0.8로 표시가 살아난다.
    # 예전 당일포함 기준이면 high=돌파고가, entry=그 위라 chasing이 절대 안 걸렸다(최대 -0.097).
    # 단 점수 반영은 기본 꺼짐(_chase_penalty_enabled=False → FWS 추격 0점 유지).
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
    assert result["chase_penalty_enabled"] is False
    assert result["ext_atr_for_score"] == 0.0


def test_chase_penalty_is_gated_by_env(monkeypatch):
    # STOCKORACLE_CHASE_PENALTY=1일 때만 FWS 추격 15/25점이 반영된다.
    monkeypatch.setenv("STOCKORACLE_CHASE_PENALTY", "1")
    base = [95.0 + i * 0.2 for i in range(30)]
    highs = [c * 1.005 for c in base]
    lows = [c * 0.995 for c in base]
    vols = [100000.0] * 30
    closes = base[:-1] + [106.0]
    highs = highs[:-1] + [106.5]
    lows = lows[:-1] + [105.0]
    result = compute_hybrid_score(closes, highs, lows, vols)
    assert result["chase_penalty_enabled"] is True
    assert result["ext_atr_for_score"] > 0.8
    assert result["fws"] >= 15.0  # 추격 패널티 반영
    monkeypatch.setenv("STOCKORACLE_CHASE_PENALTY", "0")
    result2 = compute_hybrid_score(closes, highs, lows, vols)
    assert result2["ext_atr_for_score"] == 0.0


def test_non_breakout_keeps_chasing_off():
    closes = [100.0 + (i % 5) * 0.3 for i in range(40)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    vols = [100000.0] * 40
    result = compute_hybrid_score(closes, highs, lows, vols)
    assert result["anti_chase"]["chasing"] is False


def test_pivot_stays_fixed_at_minus_two():
    # 피벗 [-2] 고정: 마감 후 하루 묵지만 영향 작아 그대로 둔다.
    from api import index
    dd = {
        "High": [10.0, 11.0, 12.0, 13.0],
        "Low": [9.0, 10.0, 11.0, 12.0],
        "Close": [9.5, 10.5, 11.5, 12.5],
        "Open": [9.2, 10.2, 11.2, 12.2],
        "Date": ["2026-10-01", "2026-10-02", "2026-10-06", "2026-10-07"],
    }
    piv = index.calc_pivot_points(dd, market="KRX")
    # [-2] 기준 (12,11,11.5 → Pivot 11.5). [-1]이면 12.5가 된다.
    assert piv["classic"]["Pivot"] == 11.5
    dd2 = {k: v for k, v in dd.items() if k != "Date"}
    piv2 = index.calc_pivot_points(dd2, market="KRX")
    assert piv2["classic"]["Pivot"] == 11.5


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
