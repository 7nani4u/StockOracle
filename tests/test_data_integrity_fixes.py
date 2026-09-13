"""KRX/US 데이터 정합성 회귀 테스트 (전일 대비 부호, JSON 무결성, 이벤트 위험 오탐 등)."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from api import index


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code == 200

    def json(self):
        return self._payload


def _basic_payload(close, diff, code, traded_at="2026-09-11T16:10:21+09:00"):
    return {
        "closePrice": close,
        "compareToPreviousClosePrice": diff,
        "compareToPreviousPrice": {"code": code},
        "localTradedAt": traded_at,
        "marketStatus": "CLOSE",
    }


def test_naver_realtime_signed_decline_keeps_previous_close_above_price(monkeypatch):
    # 네이버는 하락폭을 부호 포함 문자열로 준다: 259,500원 / -9,500 / code 5(하락)
    monkeypatch.setattr(index.requests, "get",
                        lambda *a, **k: _FakeResponse(_basic_payload("259,500", "-9,500", "5")))
    index._CACHE.clear()
    result = index.fetch_naver_realtime("005930")

    assert float(result["price"]) == 259500
    assert float(result["prev_close"]) == 269000
    assert result["change_pct"] < 0
    assert result["trade_date"] == "2026-09-11"


def test_naver_realtime_handles_unsigned_rise_and_flat(monkeypatch):
    payloads = iter([
        _basic_payload("13,580", "3,130", "1"),   # 상한가(부호 없음)
        _basic_payload("50,000", "0", "3"),        # 보합
    ])
    monkeypatch.setattr(index.requests, "get", lambda *a, **k: _FakeResponse(next(payloads)))

    rise = index.fetch_naver_realtime("117670")
    flat = index.fetch_naver_realtime("000000")

    assert float(rise["prev_close"]) == 10450
    assert rise["change_pct"] > 0
    assert float(flat["prev_close"]) == 50000
    assert flat["change_pct"] == 0


def test_json_payload_never_contains_infinity_or_nan():
    import numpy as np
    import pandas as pd

    cleaned = index.replace_nan_with_none({
        "inf": float("inf"), "ninf": -np.inf, "nan": np.float64("nan"), "nat": pd.NaT,
        "nested": [1.5, float("inf"), {"x": np.float32("nan")}], "flag": np.bool_(True),
    })
    text = json.dumps(cleaned, allow_nan=False)
    assert "Infinity" not in text and "NaN" not in text
    assert cleaned["nested"] == [1.5, None, {"x": None}]
    assert cleaned["nat"] is None and cleaned["flag"] is True


def test_event_risk_ignores_neutral_disclosure_titles():
    neutral = index.calc_event_risk("005930.KS", "KRX", [], [
        {"title": "삼성전자(주) 기타 경영사항(자율공시)"},
        {"title": "삼성전자(주) 수시공시의무관련사항(공정공시)"},
        {"title": "삼성전자(주) 자기주식 취득 결정"},
        {"title": "반도체 시장조사 기관 전망"},
    ], {"days_to_earnings": 60})
    risky = index.calc_event_risk("000000.KQ", "KRX", [], [
        {"title": "(주)테스트 주주배정 유상증자 결정"},
    ], {"days_to_earnings": 60})
    assert neutral["score"] == 0
    assert risky["score"] >= 12 and any("희석" in r for r in risky["reasons"])


def test_long_term_band_lower_bound_stays_positive_and_meaningful():
    closes = [100.0 * (1.0 + (0.03 if i % 2 else -0.028)) ** (i % 5) for i in range(260)]
    dd = {"Close": closes, "High": [c * 1.03 for c in closes], "Low": [c * 0.97 for c in closes],
          "ATR": [6.0] * 260, "RSI": [50.0] * 260, "MA20": closes, "MA60": closes}
    rows = index._build_long_term_targets(dd, closes[-1], 6.0, "KRX", {}, {"confidence": 40,
                                          "confidence_interval": {"spread": 40}}, None, None, None, None)
    assert rows, "장기 범위가 생성되어야 한다"
    for row in rows:
        assert row["min_price"] > 0.01 * 100, row  # 로그 대칭 하단: '최저 1원' 같은 값 금지
        assert row["min_price"] < row["max_price"]
        assert "로그 구간" in row["band_basis"]
    # 변동성이 큰 종목의 10년 하단이 현재가보다 높게(사실상 손실 불가) 표시되면 과신이다.
    assert rows[-1]["min_price"] < closes[-1] < rows[-1]["max_price"]


def test_prediction_outlook_aligns_volume_rows_with_close_rows():
    closes = [100.0 + i * 0.3 for i in range(40)]
    volumes = [1000.0] * 40
    volumes[20] = None           # 중간 결측
    closes_with_gap = list(closes)
    closes_with_gap[10] = None   # 다른 위치의 종가 결측
    volumes[-1] = 3000.0
    dd = {"Open": closes_with_gap, "High": [c and c + 1 for c in closes_with_gap],
          "Low": [c and c - 1 for c in closes_with_gap], "Close": closes_with_gap, "Volume": volumes}
    result = index.build_prediction_outlook(
        symbol="TEST", market="US", dd=dd, last_price=closes[-1], prev_close=closes[-2], pct_change=0.3,
        atr=1.0, regime="NEUTRAL", score=55, prob_up=50, prob_down=40, pivot_points={},
        indicator_signals={}, buy_price={}, target_price={}, pullback_analysis={},
        signal_confidence=None, investor_flow=None, ai_strategy=None, candlestick_patterns=[],
        naver=None, us_enriched=None, toss_industry=None, event_risk=None,
    )
    volume = next(item for item in result["status"] if item["key"] == "volume")
    assert "3.00배" in volume["detail"]  # 결측 행을 건너뛰어도 마지막 거래량은 마지막 종가와 같은 행


def test_display_price_helpers_follow_market_units():
    assert index._fmt_plain_price(262800.0, "KRX") == "262,800"
    assert index._fmt_plain_price(331.8567, "US") == "331.86"
    assert index._fmt_plain_price(0.41234, "US") == "0.4123"
    assert index._fmt_market_price(245008.4, "KRX") == "245,008원"
    assert index._fmt_market_price(301.39, "US") == "$301.39"
    assert index._fmt_market_price(None, "US") == "미확보"
    assert index._disp_rnd("KRX", 2) == 0 and index._disp_rnd("US", 4) == 4


def test_ai_summary_is_resynced_to_final_score_and_market_prices():
    dd = {"EMA20": [261096.0], "BB_Upper": [280592.0], "BB_Lower": [245008.0], "ATR": [14508.9]}
    lines = index._ai_strategy_summary_lines(43, 259500, 261096, 280592, 245008, "KRX")
    lines += ["🔍 종합 판단 근거: 테스트"] + index._ai_scenario_lines(259500, 280592, 245008, 14508.9, "KRX")
    strategy = {"result": " | ".join(lines + ["[투자자 수급] 외국인 -1주"]), "core_line_count": len(lines)}
    synced = index._sync_ai_strategy_summary(strategy, 30, 43, dd, 259500.0, "KRX")
    parts = synced["result"].split(" | ")
    assert parts[0] == "[핵심 요약] SELL (매도 우위 / 리스크 관리)"
    assert "43점" in parts[1] and "30점" in parts[1]
    assert parts[-1] == "[투자자 수급] 외국인 -1주"      # 보정 단계에서 덧붙인 문구 보존
    assert "e+" not in synced["result"] and ".00" not in synced["result"]
    assert "ATR" in synced["result"] and "+5~10%" not in synced["result"]


def test_parse_naver_number_rejects_placeholders():
    assert index._parse_naver_number("11.64배") == 11.64
    assert index._parse_naver_number("46.71%") == 46.71
    assert index._parse_naver_number("-") is None
    assert index._parse_naver_number("N/A") is None
    assert index._parse_naver_number(float("inf")) is None
