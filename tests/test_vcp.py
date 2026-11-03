# -*- coding: utf-8 -*-
"""VCP(변동성 수축 패턴) 판정기 테스트. 점수 미반영 연구용."""
from market_briefing.vcp import detect_vcp


def _vcp_series():
    """상승 → 25% → 15% → 8% 수축 → 피벗 돌파."""
    c = [100 + i * 0.5 for i in range(60)]
    c += [150]
    c += [150 - i * (37.5 / 8) for i in range(1, 9)]      # →112.5 (-25%)
    c += [112.5 + i * (22.5 / 6) for i in range(1, 7)]    # →135
    c += [135 - i * (20.25 / 8) for i in range(1, 9)]     # →114.75 (-15%)
    c += [114.75 + i * (13.25 / 6) for i in range(1, 7)]  # →128
    c += [128 - i * (10.24 / 8) for i in range(1, 9)]     # →117.76 (-8%)
    c += [117.76 + i * (12.24 / 6) for i in range(1, 7)]  # →130 돌파
    h = [x * 1.005 for x in c]
    l = [x * 0.995 for x in c]
    v = [100_000] * len(c)
    v[-1] = 300_000
    return c, h, l, v


def test_breakout_reports_pass():
    c, h, l, v = _vcp_series()
    r = detect_vcp(c, h, l, v, "BULLISH")
    assert r["available"] is True
    assert r["stage"] == "PASS"
    assert r["depths_pct"][0] > r["depths_pct"][1] > r["depths_pct"][2]
    assert r["false_breakout_risk"] is False
    assert r["entry_trigger"] == r["pivot"]
    assert r["stop_price"] < r["pivot"]


def test_before_breakout_reports_contracting():
    c, h, l, v = _vcp_series()
    c2, h2, l2, v2 = c[:-3], h[:-3], l[:-3], v[:-3]  # 돌파 전 (피벗 아래)
    r = detect_vcp(c2, h2, l2, v2, "BULLISH")
    assert r["stage"] in ("CONTRACTING", "PASS")


def test_weak_market_flags_false_breakout_risk():
    c, h, l, v = _vcp_series()
    r = detect_vcp(c, h, l, v, "BEARISH")
    assert r["stage"] == "PASS"
    assert r["false_breakout_risk"] is True
    assert "약세장 돌파" in r["risk_reasons"]


def test_thin_volume_flags_false_breakout_risk():
    c, h, l, v = _vcp_series()
    v = [100_000] * len(c)  # 돌파봉 거래량 평범
    r = detect_vcp(c, h, l, v, "BULLISH")
    assert r["stage"] == "PASS"
    assert "거래량 미동반" in r["risk_reasons"]


def test_trend_without_contraction_is_none():
    c = [100.0 + i * 0.4 for i in range(120)]
    h = [x * 1.005 for x in c]
    l = [x * 0.995 for x in c]
    r = detect_vcp(c, h, l, None, "BULLISH")
    assert r["stage"] == "NONE"


def test_expanding_corrections_are_none():
    c = [100.0] * 60 + [130.0]
    c += [130 - i * (6.5 / 6) for i in range(1, 7)]    # -5%
    c += [123.5 + i * (6.5 / 6) for i in range(1, 7)]  # 반등
    c += [130 - i * (19.5 / 6) for i in range(1, 7)]   # -15% 확대
    c += [110.5 + i * (20.0 / 6) for i in range(1, 7)]
    h = [x * 1.005 for x in c]
    l = [x * 0.995 for x in c]
    r = detect_vcp(c, h, l, None, "BULLISH")
    assert r["stage"] == "NONE"


def test_insufficient_data_is_unavailable():
    r = detect_vcp([100.0] * 30)
    assert r["available"] is False
    assert r["stage"] == "NONE"


def test_never_raises_on_garbage():
    r = detect_vcp([], None, None, None)
    assert r["stage"] == "NONE"
    r = detect_vcp([float("nan")] * 100, None, None, None)
    assert r["stage"] == "NONE"
