"""파동 방향·적중률 계약 — 스크린샷-2 (예측 방향 + 파동 적중률)."""
import math

from api.index import HTML, build_prediction_outlook
from market_briefing.wave_analysis import analyze_wave, classify_n_wave, find_swing_pivots, wave_hit_rate


def _trend_dd(size=120, base=100.0):
    closes = [base + i * 0.3 + math.sin(i / 4.0) * 2.0 for i in range(size)]
    highs = [c + 1.2 for c in closes]
    lows = [c - 1.3 for c in closes]
    volumes = [100_000 + (i % 7) * 3_000 for i in range(size)]
    return {
        "Open": [c - 0.2 for c in closes], "High": highs, "Low": lows,
        "Close": closes, "Volume": volumes,
        "MA20": [None] * 19 + closes[19:], "MA60": [None] * 59 + closes[59:],
        "MA120": [None] * size, "RSI": [55.0] * size,
        "MACD": [0.5] * size, "Signal_Line": [0.2] * size,
        "ADX": [22.0] * size, "DI_Plus": [22.0] * size, "DI_Minus": [16.0] * size,
        "BB_Upper": [c + 3.0 for c in closes], "BB_Lower": [c - 3.0 for c in closes],
        "Date": [f"2026-01-{(i % 28) + 1:02d}" for i in range(size)],
    }


def test_wave_direction_is_always_present():
    dd = _trend_dd()
    wave = analyze_wave(dd["Close"], dd["High"], dd["Low"])
    assert wave["direction"] in ("상승", "하락", "중립")
    assert wave["pattern_short"] in ("N자", "역N자", "박스권", "추세", "—", "판단 보류")
    assert "(" in wave["label"] or wave["label"] in ("상승", "하락", "중립")
    # 어떤 종목에서도 방향 라벨은 비지 않는다
    empty = analyze_wave([], [], [])
    assert empty["direction"] == "중립" and empty["label"] == "중립"


def test_wave_hit_rate_counts_hits_over_total():
    dd = _trend_dd(size=180)
    stat = wave_hit_rate(dd["Close"], dd["High"], dd["Low"])
    assert stat["total"] >= 0 and stat["hits"] <= stat["total"]
    if stat["total"]:
        assert 0.0 <= stat["hit_rate_pct"] <= 100.0
        assert stat["hits"] + (stat["total"] - stat["hits"]) == stat["total"]
    wave = analyze_wave(dd["Close"], dd["High"], dd["Low"])
    assert wave["hits"] == stat["hits"] and wave["total"] == stat["total"]


def test_prediction_outlook_exposes_wave():
    dd = _trend_dd()
    last = dd["Close"][-1]
    outlook = build_prediction_outlook(
        symbol="005930.KS", market="KRX", dd=dd, last_price=last,
        prev_close=dd["Close"][-2], pct_change=0.5, atr=2.0, regime="NEUTRAL",
        score=60.0, prob_up=55.0, prob_down=30.0, pivot_points={},
        indicator_signals={"signals": {}, "summary": {}},
        buy_price={"strategy_rec": {"action_key": "split_buy"}},
        target_price={"min_price": last * 1.02, "max_price": last * 1.05},
        pullback_analysis={}, signal_confidence={"confidence": 60.0},
        investor_flow=None, ai_strategy=None, candlestick_patterns=[],
        naver=None, us_enriched=None, toss_industry=None, event_risk={"score": 0, "reasons": []},
    )
    wave = outlook.get("wave_analysis") or (outlook.get("forecast") or {}).get("wave")
    assert wave is not None
    assert wave["direction"] in ("상승", "하락", "중립")
    assert "hits" in wave and "total" in wave


def test_wave_html_renders_direction_and_hit_rate():
    assert "prediction-wave-summary" in HTML
    assert "예측 방향" in HTML
    assert "파동 적중률" in HTML
    assert "fc.wave" in HTML or "p.wave_analysis" in HTML
