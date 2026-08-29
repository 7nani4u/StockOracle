"""예측 탭 매수 밴드의 5단계 가격·확률·기간 계약 검증."""

import math

from api.index import HTML, calc_buy_price


def _sample_buy_dd(size=180):
    closes = [100.0 + index * 0.08 + math.sin(index / 4.0) * 2.2 for index in range(size)]
    opens = [close - math.sin(index) * 0.25 for index, close in enumerate(closes)]
    highs = [close + 1.5 + (index % 3) * 0.08 for index, close in enumerate(closes)]
    lows = [close - 1.6 - (index % 4) * 0.07 for index, close in enumerate(closes)]
    volumes = [100_000 + (index % 11) * 4_500 for index in range(size)]
    true_ranges = [high - low for high, low in zip(highs, lows)]
    atrs = [
        sum(true_ranges[max(0, index - 13):index + 1])
        / len(true_ranges[max(0, index - 13):index + 1])
        for index in range(size)
    ]

    def sma(period):
        return [
            None if index + 1 < period else sum(closes[index - period + 1:index + 1]) / period
            for index in range(size)
        ]

    ma20 = sma(20)
    ma60 = sma(60)
    return {
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
        "ATR": atrs,
        "MA20": ma20,
        "MA60": ma60,
        "MA120": [None] * size,
        "EMA20": ma20,
        "BB_Middle": ma20,
        "BB_Lower": [None if value is None else value - 3.2 for value in ma20],
        "BB_Upper": [None if value is None else value + 3.2 for value in ma20],
        "RSI": [52.0] * size,
        "MACD": [0.3] * size,
        "Signal_Line": [0.1] * size,
        "ADX": [23.0] * size,
        "DI_Plus": [22.0] * size,
        "DI_Minus": [17.0] * size,
    }


def _calculate(dd):
    return calc_buy_price(
        dd=dd,
        last_price=dd["Close"][-1],
        atr=dd["ATR"][-1],
        score=65,
        indicator_signals={"signals": {}},
        market="KRX",
        period="1y",
        event_risk={"score": 8, "reasons": []},
        learning_adjustment={},
        market_regime="NEUTRAL",
        reference_prev_close=dd["Close"][-2],
        reference_pct_change=0.2,
    )


def test_all_buy_bands_expose_only_ordered_independent_structure_steps():
    result = _calculate(_sample_buy_dd())

    for family in ("aggressive_bands", "recommended_bands"):
        assert [band["band"] for band in result[family]] == ["A", "B", "C"]
        for band in result[family]:
            steps = band["steps"]
            if band.get("is_available") is False:
                assert steps == []
                assert band.get("availability_note")
                continue
            assert [step["label"] for step in steps] == [f"{index}단계" for index in range(1, len(steps) + 1)]
            assert 1 <= len(steps) <= 5
            assert all(band["range"][0] <= step["price"] <= band["range"][1] for step in steps)
            assert all(step["price_range"][0] <= step["price"] <= step["price_range"][1] for step in steps)
            assert steps[0]["price_range"][1] == band["range"][1]
            assert steps[-1]["price_range"][0] == band["range"][0]
            assert all(
                steps[index]["price_range"][0] >= steps[index + 1]["price_range"][1]
                for index in range(len(steps) - 1)
            )
            assert all(len(step["decline_pct_range"]) == 2 for step in steps)
            assert [step["price"] for step in steps] == sorted(
                (step["price"] for step in steps), reverse=True
            )
            assert len({step["price"] for step in steps}) == len(steps)
            assert [step["decline_pct"] for step in steps] == sorted(
                (step["decline_pct"] for step in steps), reverse=True
            )
            assert [step["allocation_pct"] for step in steps] == sorted(
                step["allocation_pct"] for step in steps
            )


def test_reach_probability_and_period_are_bounded_and_monotonic():
    result = _calculate(_sample_buy_dd())

    for family in ("aggressive_bands", "recommended_bands"):
        for band in result[family]:
            steps = band["steps"]
            probability_steps = [
                step for step in steps if step["reach_probability_pct"] is not None
            ]
            assert all(step["days_min"] is not None for step in probability_steps)
            assert all(step["days_max"] is not None for step in probability_steps)
            assert all(step["period_label"] is None for step in probability_steps)
            assert [step["probability_low_pct"] for step in probability_steps] == sorted(
                (step["probability_low_pct"] for step in probability_steps), reverse=True
            )
            assert [step["probability_high_pct"] for step in probability_steps] == sorted(
                (step["probability_high_pct"] for step in probability_steps), reverse=True
            )
            for step in probability_steps:
                assert 0 <= step["probability_low_pct"] <= step["reach_probability_pct"]
                assert step["reach_probability_pct"] <= step["probability_high_pct"] <= 100

            period_steps = [step for step in steps if step["days_min"] is not None]
            assert [step["days_min"] for step in period_steps] == sorted(
                step["days_min"] for step in period_steps
            )
            assert [step["days_max"] for step in period_steps] == sorted(
                step["days_max"] for step in period_steps
            )

    exploration_floor = min(band["range"][0] for band in result["aggressive_bands"])
    for recommended in result["recommended_bands"]:
        if recommended["is_available"]:
            assert recommended["steps"][0]["price"] < exploration_floor
        else:
            assert recommended["steps"] == []


def test_insufficient_history_never_fabricates_probability_or_period():
    result = _calculate(_sample_buy_dd(size=35))

    for family in ("aggressive_bands", "recommended_bands"):
        for band in result[family]:
            for step in band["steps"]:
                assert step["reach_probability_pct"] is None
                assert step["probability_label"] == "분석 데이터 부족"
                assert step["days_min"] is None
                assert step["days_max"] is None
                assert step["period_label"] == "기간 산정 불가"
                assert step["period_source"] is None
                assert step["period_note"] is None


def test_period_uses_atr_speed_model_when_probability_path_sample_is_short():
    # 44개 봉은 확률용 과거 경로 40건 기준에는 미달하지만, ATR·최근 속도 기반
    # 기간 추정에는 충분하다. 이 경우 확률은 보류해도 기간까지 비우지 않는다.
    result = _calculate(_sample_buy_dd(size=44))

    for family in ("aggressive_bands", "recommended_bands"):
        for band in result[family]:
            steps = band["steps"]
            assert all(step["reach_probability_pct"] is None for step in steps)
            assert all(step["probability_label"] == "분석 데이터 부족" for step in steps)
            assert all(step["period_source"] == "model" for step in steps)
            assert all(step["period_label"] is None for step in steps)
            assert all(1 <= step["days_min"] <= step["days_max"] <= 30 for step in steps)
            assert [step["days_min"] for step in steps] == sorted(
                step["days_min"] for step in steps
            )
            assert [step["days_max"] for step in steps] == sorted(
                step["days_max"] for step in steps
            )
            assert all("ATR 거리" in step["period_note"] for step in steps)


def test_sparse_empirical_hits_use_dynamic_model_period_instead_of_unavailable():
    dd = _sample_buy_dd()
    # 최근 변동성이 과거보다 급격히 확대된 종목을 재현한다. 깊은 밴드는
    # 과거 도달 사례가 최소 표본 수보다 적지만 가격·ATR·거래량 데이터는 충분하다.
    dd["ATR"][-1] = 25.0
    result = _calculate(dd)
    all_steps = [
        step
        for family in ("aggressive_bands", "recommended_bands")
        for band in result[family]
        for step in band["steps"]
    ]

    assert any(step["period_source"] == "model" for step in all_steps)
    assert all(step["days_min"] is not None for step in all_steps)
    assert all(step["days_max"] is not None for step in all_steps)
    assert all(1 <= step["days_min"] <= step["days_max"] <= 30 for step in all_steps)
    assert all(step["period_label"] is None for step in all_steps)
    assert all(
        step["period_note"] and "ATR 거리" in step["period_note"]
        for step in all_steps
        if step["period_source"] == "model"
    )

    for family in ("aggressive_bands", "recommended_bands"):
        for band in result[family]:
            assert [step["days_min"] for step in band["steps"]] == sorted(
                step["days_min"] for step in band["steps"]
            )
            assert [step["days_max"] for step in band["steps"]] == sorted(
                step["days_max"] for step in band["steps"]
            )


def test_forecast_band_ui_uses_aligned_five_column_stage_rows():
    for label in ("단계", "매수 가격 범위", "하락률", "도달 확률", "예상 기간"):
        assert f'role="columnheader">{label}</span>' in HTML
    assert "buy-stage-row" in HTML
    assert "buy-stage-price" in HTML
    assert "밴드 전체 매수 가격" in HTML
    assert "s.price_range" in HTML
    assert "분석 데이터 부족" in HTML
    assert "기간 산정 불가" in HTML
    assert "TP1" not in HTML.split("const renderBandCard", 1)[1].split(
        "const recBandsHtml", 1
    )[0]
