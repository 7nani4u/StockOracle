"""예측 탭 가격 표시 계약 — 줄바꿈 금지·호가 단위·정수 거래일."""
import math

from api.index import HTML, _as_int_day_window, calc_risk


def _risk_dd(size=120, base=50000.0):
    closes = [base * (1.0 + 0.001 * i + 0.004 * math.sin(i / 5.0)) for i in range(size)]
    opens = [c * 0.999 for c in closes]
    highs = [c * 1.006 for c in closes]
    lows = [c * 0.994 for c in closes]
    volumes = [200_000 + (i % 7) * 10_000 for i in range(size)]
    tr = [h - l for h, l in zip(highs, lows)]
    atrs = [sum(tr[max(0, i - 13):i + 1]) / len(tr[max(0, i - 13):i + 1]) for i in range(size)]

    def sma(p):
        return [None if i + 1 < p else sum(closes[i - p + 1:i + 1]) / p for i in range(size)]

    ma20 = sma(20)
    return {
        "Open": opens, "High": highs, "Low": lows, "Close": closes,
        "Volume": volumes, "ATR": atrs, "MA20": ma20, "MA60": sma(60),
        "MA120": [None] * size, "EMA20": ma20,
        "RSI": [55.0] * size, "MACD": [0.5] * size, "Signal_Line": [0.2] * size,
        "ADX": [24.0] * size, "DI_Plus": [24.0] * size, "DI_Minus": [16.0] * size,
    }


def _krx_tick(price):
    p = float(price)
    if p < 2_000:
        return 1
    if p < 5_000:
        return 5
    if p < 20_000:
        return 10
    if p < 50_000:
        return 50
    return 100


def test_tp_days_are_integer_trading_days_with_strict_order():
    dd = _risk_dd()
    result = calc_risk(dd["Close"][-1], dd["ATR"][-1], "KRX", dd)
    for profile in ("conservative", "balanced", "aggressive"):
        for level in result[profile]["tp_levels"]:
            for key in ("days_min", "avg_days", "days_max"):
                value = level[key]
                assert isinstance(value, int) and not isinstance(value, bool), (profile, key, value)
            assert 1 <= level["days_min"] < level["avg_days"] < level["days_max"]


def test_provisional_tp_days_are_integer_trading_days():
    dd = _risk_dd(size=6, base=30000.0)
    result = calc_risk(dd["Close"][-1], 800.0, "KRX", dd)
    assert result["provisional"] is True
    for profile in ("conservative", "balanced", "aggressive"):
        for level in result[profile]["tp_levels"]:
            for key in ("days_min", "avg_days", "days_max"):
                assert isinstance(level[key], int), (profile, key, level[key])
            assert level["days_min"] < level["avg_days"] < level["days_max"]


def test_tp_prices_are_on_krx_ticks():
    dd = _risk_dd()
    result = calc_risk(dd["Close"][-1], dd["ATR"][-1], "KRX", dd)
    for profile in ("conservative", "balanced", "aggressive"):
        for level in result[profile]["tp_levels"]:
            for price in (level["price"], *level["price_range"]):
                tick = _krx_tick(price)
                assert abs(price / tick - round(price / tick)) < 1e-6, (profile, price)


def test_int_day_window_helper_keeps_order():
    lo, avg, hi = _as_int_day_window(25.3, 40.1, 66.2)
    assert (lo, avg, hi) == (25, 40, 67)
    assert lo < avg < hi
    lo, avg, hi = _as_int_day_window(1.0, 1.0, 2.0)
    assert lo < avg < hi and lo >= 1
    lo, avg, hi = _as_int_day_window("x", None, float("nan"))
    assert (lo, avg, hi) == (1, 2, 3)


def test_price_ranges_render_on_one_line_with_single_unit():
    assert "function fmtRange(lo, hi, isKrx)" in HTML
    assert HTML.count("fmtRange(") >= 8
    # 매수 단계 가격 셀 줄바꿈 금지
    assert ".buy-stage-price{font-size:10px;font-weight:900;color:#e6edf3;white-space:nowrap;text-align:right;line-height:1.25}" in HTML
    # 목표가 셀·청산 범위 줄바꿈 금지
    assert ".risk-tgt{color:#f85149;font-weight:700;white-space:nowrap}" in HTML
    # 가격 열 확장 (좁은 열이 '원' 고아 단어를 만든다)
    assert "minmax(132px,1.55fr)" in HTML
    assert "minmax(106px,1.35fr)" in HTML
