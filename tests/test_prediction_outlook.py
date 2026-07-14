"""예측 탭 구조화 로직과 미국 급등 추천 가격 경계 검증."""

from api.index import HTML, _is_us_surge_price_eligible, build_prediction_outlook


def _sample_dd():
    closes = [100 + i * 0.45 for i in range(80)]
    opens = [c - 0.2 for c in closes]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.1 for c in closes]
    volumes = [100_000 + (i % 7) * 3_000 for i in range(79)] + [165_000]
    return {
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
        "MA20": [None] * 19 + [closes[max(0, i - 9)] for i in range(19, 80)],
        "MA60": [None] * 59 + [closes[max(0, i - 29)] for i in range(59, 80)],
        "MA120": [None] * 80,
        "RSI": [56.0] * 80,
        "MACD": [1.2] * 80,
        "Signal_Line": [0.8] * 80,
        "BB_Upper": [c + 3.0 for c in closes],
        "BB_Lower": [c - 3.0 for c in closes],
    }


def _base_kwargs(market="KRX", symbol="005930.KS", flags=None, signal_confidence=None):
    dd = _sample_dd()
    last = dd["Close"][-1]
    return {
        "symbol": symbol,
        "market": market,
        "dd": dd,
        "last_price": last,
        "prev_close": dd["Close"][-2],
        "pct_change": 0.34,
        "atr": 2.1,
        "regime": "BULL",
        "score": 68,
        "prob_up": 64,
        "prob_down": 36,
        "pivot_points": {"classic": {"S1": last - 1.4, "S2": last - 3.0, "R1": last + 1.6, "R2": last + 3.2}},
        "indicator_signals": {"summary": {"overall_label": "매수 우세"}},
        "buy_price": {"strategy_rec": {"action_key": "split_buy"}},
        "target_price": {"min_price": last + 3.0, "max_price": last + 6.0, "reach_probability": 67},
        "pullback_analysis": {
            "zones": {
                "core": {"low": last - 2.5, "high": last - 1.2},
                "defense": {"low": last - 5.0, "high": last - 3.5},
                "resistance": {"low": last + 1.5, "high": last + 3.0},
            },
            "stop_loss": last - 4.8,
            "sl_triggered": 0,
            "sl_conditions": [],
            "manipulation_flags": flags or [],
        },
        "signal_confidence": signal_confidence or {
            "confidence": 66,
            "confidence_interval": {"lower": 55, "upper": 76, "spread": 21},
            "macro_regime": {"regime": "Neutral", "components": {}},
        },
        "investor_flow": {"ok": True, "외국인": 125_000, "기관": 82_000},
        "ai_strategy": {"result": "[기술 분석] 상승 우위 | 거래량 확인 필요"},
        "candlestick_patterns": [{"name": "Hammer", "direction": "상승"}],
        "naver": {"industry": "반도체"} if market == "KRX" else None,
        "us_enriched": None,
        "toss_industry": None,
        "event_risk": {"score": 0, "reasons": []},
    }


def test_us_surge_price_filter_is_strictly_below_20():
    assert _is_us_surge_price_eligible(19.99)
    assert _is_us_surge_price_eligible("0.01")
    assert not _is_us_surge_price_eligible(20)
    assert not _is_us_surge_price_eligible(20.01)
    assert not _is_us_surge_price_eligible(0)
    assert not _is_us_surge_price_eligible(float("nan"))
    assert not _is_us_surge_price_eligible(None)


def test_prediction_outlook_builds_three_conditional_scenarios():
    flags = [{
        "pattern": "지지선 이탈 척 (손절 유도)",
        "desc": "장중 이탈 후 종가 회복",
        "action": "다음 봉 지지 확인",
    }]
    result = build_prediction_outlook(**_base_kwargs(flags=flags))

    assert result["decision"]["label"] == "조건부 분할 접근"
    assert len(result["status"]) == 5
    assert [s["key"] for s in result["scenarios"]] == ["upside", "sideways", "downside"]
    assert sum(s["probability"] for s in result["scenarios"]) == 100
    assert result["pattern_context"]["manipulation_detected"] is True
    assert "세력 흔들림" in result["decision"]["summary"]
    assert result["levels"]["support"] < result["levels"]["resistance"]
    assert any(f["label"] == "외국인·기관" for f in result["market_context"]["facts"])


def test_us_prediction_reuses_macro_sector_and_earnings_context():
    confidence = {
        "confidence": 61,
        "confidence_interval": {"lower": 49, "upper": 72, "spread": 23},
        "macro_regime": {
            "regime": "Transition",
            "components": {
                "sp500": {"pct5d": -1.25, "pct10d": 0.4},
                "vix": {"level": 23.1, "change5d_pct": 18.0},
                "dxy": {"pct5d": 1.2},
            },
        },
        "sector_relative": {
            "adjust": -6,
            "reason": "Financials 섹터 하락 속 매수 신호 — 신뢰 약화",
            "sector": {"name": "Financials", "etf": "XLF", "pct5d": -1.8},
        },
        "earnings": {"days_to_earnings": 4, "earnings_risk": True},
    }
    kwargs = _base_kwargs(market="US", symbol="SOFI", signal_confidence=confidence)
    kwargs["investor_flow"] = {"ok": False}
    kwargs["naver"] = None
    result = build_prediction_outlook(**kwargs)
    labels = {f["label"] for f in result["market_context"]["facts"]}

    assert {"S&P 500", "VIX", "달러", "섹터", "실적"}.issubset(labels)
    assert any("나스닥" in gap for gap in result["market_context"]["data_gaps"])
    assert any("정책금리" in gap for gap in result["market_context"]["data_gaps"])


def test_forecast_tab_sections_follow_decision_context_order():
    forecast_html = HTML.split('<div id="tab-forecast"', 1)[1].split('<!-- 뉴스 탭 -->', 1)[0]
    ordered_labels = [
        "🔮 핵심 판단과 현재 상태",
        "🌐 판단 근거 · 시장 흐름과 AI 진단",
        "🧭 조건부 예측 시나리오",
        "📈 목표 가격 범위",
        "🎯 현재가 기준 매수 전략",
        "🛡️ 리스크 관리 (ATR 기반)",
    ]
    positions = [forecast_html.index(label) for label in ordered_labels]

    assert positions == sorted(positions)
    assert forecast_html.count('id="ai-strategy-section"') == 1
    assert "분석 흐름" in forecast_html
    assert "AI 보조 해석 상세 보기" in forecast_html
