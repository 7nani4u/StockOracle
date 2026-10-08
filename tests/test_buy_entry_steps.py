"""예측 탭 매수 밴드의 차트 구조·가격·확률·기간 계약 검증."""

import copy
import math
import random

import pytest

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


def _calculate(dd, market="KRX", market_regime="NEUTRAL"):
    return calc_buy_price(
        dd=dd,
        last_price=dd["Close"][-1],
        atr=dd["ATR"][-1],
        score=65,
        indicator_signals={"signals": {}},
        market=market,
        period="1y",
        event_risk={"score": 8, "reasons": []},
        learning_adjustment={},
        market_regime=market_regime,
        reference_prev_close=dd["Close"][-2],
        reference_pct_change=0.2,
    )


def test_all_buy_bands_expose_only_ordered_independent_structure_steps():
    result = _calculate(_sample_buy_dd())

    for family in ("aggressive_bands", "recommended_bands"):
        assert [band["band"] for band in result[family]] == ["A", "B", "C"]
        assert [band["strategy_label"] for band in result[family]] == ["밴드 A", "밴드 B", "밴드 C"]
        for band in result[family]:
            steps = band["steps"]
            assert band["is_available"] is True
            assert band["availability_note"] is None
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

    for aggressive, recommended in zip(
        result["aggressive_bands"], result["recommended_bands"]
    ):
        assert recommended["steps"]
        assert recommended["steps"][0]["price"] <= aggressive["steps"][0]["price"]


def test_band_cards_are_strictly_lower_and_use_distinct_ranges():
    result = _calculate(_sample_buy_dd())

    for family in ("aggressive_bands", "recommended_bands"):
        bands = result[family]
        assert all(band["is_available"] for band in bands)
        assert len({tuple(band["range"]) for band in bands}) == 3
        widths = [band["range"][1] - band["range"][0] for band in bands]
        assert len(set(widths)) == 3

        for previous, current in zip(bands, bands[1:]):
            assert previous["range"][0] > current["range"][0]
            assert previous["range"][1] > current["range"][1]
            for previous_step, current_step in zip(previous["steps"], current["steps"]):
                assert previous_step["price"] > current_step["price"]
                assert previous_step["price_range"][0] > current_step["price_range"][0]
                assert previous_step["price_range"][1] > current_step["price_range"][1]


def test_chart_price_candidates_are_clustered_and_entry_families_do_not_overlap():
    result = _calculate(_sample_buy_dd())
    structure = result["price_structure"]
    sources = {
        source
        for cluster in structure["clusters"]
        for source in cluster["sources"]
    }

    assert structure["ready"] is True
    assert structure["candidate_count"] > structure["cluster_count"] >= 2
    assert {"ma20", "bb_lower", "volume_node", "swing_low"} <= sources
    assert structure["bollinger_width_pct"] is not None
    assert structure["recent_intraday_range_pct"] > 0
    assert structure["first_second_separated"] is True
    assert max(band["range"][1] for band in result["recommended_bands"]) < min(
        band["range"][0] for band in result["aggressive_bands"]
    )
    assert all(band["structure_note"] for band in result["aggressive_bands"])
    assert all(band["structure_note"] for band in result["recommended_bands"])


def test_market_regime_moves_entry_ranges_without_overriding_chart_structure():
    dd = _sample_buy_dd()
    bull = _calculate(dd, market_regime="BULL")
    neutral = _calculate(dd, market_regime="NEUTRAL")
    bear = _calculate(dd, market_regime="BEAR")

    assert bull["price_structure"]["market_weighting"] == "가까운 지지 가중"
    assert bear["price_structure"]["market_weighting"] == "깊은 지지 가중"
    assert bull["aggressive_bands"][0]["range"][0] >= neutral["aggressive_bands"][0]["range"][0]
    assert bear["aggressive_bands"][0]["range"][0] < neutral["aggressive_bands"][0]["range"][0]
    assert bear["recommended_bands"][0]["range"][1] < neutral["recommended_bands"][0]["range"][1]
    assert all(band["is_available"] for band in bear["recommended_bands"])


def test_volatility_expands_real_order_ranges_instead_of_using_fixed_percentages():
    base = _sample_buy_dd()

    def with_volatility(multiplier):
        dd = copy.deepcopy(base)
        for index, close in enumerate(dd["Close"]):
            dd["High"][index] = close + (dd["High"][index] - close) * multiplier
            dd["Low"][index] = close - (close - dd["Low"][index]) * multiplier
            dd["ATR"][index] *= multiplier
            if dd["BB_Middle"][index] is not None:
                middle = dd["BB_Middle"][index]
                dd["BB_Lower"][index] = middle - (middle - dd["BB_Lower"][index]) * multiplier
                dd["BB_Upper"][index] = middle + (dd["BB_Upper"][index] - middle) * multiplier
        return dd

    quiet = _calculate(with_volatility(0.45), market="US")
    volatile = _calculate(with_volatility(2.0), market="US")
    quiet_width = sum(band["range"][1] - band["range"][0] for band in quiet["aggressive_bands"])
    volatile_width = sum(band["range"][1] - band["range"][0] for band in volatile["aggressive_bands"])

    assert quiet["atr_pct"] < volatile["atr_pct"]
    assert quiet["price_structure"]["bollinger_width_pct"] < volatile["price_structure"]["bollinger_width_pct"]
    assert volatile_width > quiet_width * 2


def test_top_result_support_cards_are_removed_but_internal_support_contract_remains():
    assert "바로 아래 버팀목(단기 지지)" not in HTML
    assert "중기 버팀목 구간" not in HTML
    assert 'id="r-support-short"' not in HTML
    assert 'id="r-support-mid"' not in HTML

    result = _calculate(_sample_buy_dd())
    assert result["support_zone"] > 0
    assert result["fib"]["f382"] > 0


def test_very_short_history_still_returns_both_families_with_evidence_levels():
    # 12개 봉에서는 차트 구조가 완성되지 않았다(ready=False). 예전에는 두 구간의 가격과 단계를 지우고
    # '표시 보류'로 바꿨지만, 이제는 계산된 범위를 항상 표시하고 근거 등급으로 신뢰 수준을 밝힌다.
    result = _calculate(_sample_buy_dd(size=12))
    structure = result["price_structure"]

    assert structure["ready"] is False
    assert structure["display_policy"] == "always_display_with_evidence_level"
    for family in ("aggressive_bands", "recommended_bands"):
        assert [band["band"] for band in result[family]] == ["A", "B", "C"]
        for band in result[family]:
            assert band["is_available"] is True
            assert band["availability_note"] is None
            assert len(band["steps"]) == 5
            assert band["evidence_level"] in {"structure", "indicator", "volatility"}
            assert band["evidence_label"] and band["evidence_note"]
    assert max(band["range"][1] for band in result["recommended_bands"]) < min(
        band["range"][0] for band in result["aggressive_bands"]
    )
    for family_name, family in (
        ("aggressive", result["aggressive_bands"]),
        ("recommended", result["recommended_bands"]),
    ):
        counts = structure["evidence_summary"][family_name]
        assert sum(counts.values()) == 3
        assert counts == {
            level: sum(1 for band in family if band["evidence_level"] == level)
            for level in ("structure", "indicator", "volatility")
        }


@pytest.mark.parametrize("size", [12, 35, 44, 180])
def test_every_step_probability_and_period_come_from_the_touch_model(size):
    # 도달 확률·예상 기간은 과거 표본 길이와 무관하게 같은 무추세 변동성 터치 모델에서 나온다.
    # (52종목·2022-10~2026-09 워크포워드 검증: 기존 경험 경로+가산 방식 Brier 0.2139 → 터치 모델 0.1913)
    from market_briefing import forecast_model as fm

    dd = _sample_buy_dd(size=size)
    result = _calculate(dd)
    last = dd["Close"][-1]
    sigma = fm.blended_daily_sigma(dd["Close"], last, dd["ATR"][-1], True)["sigma"]

    for family in ("aggressive_bands", "recommended_bands"):
        for band in result[family]:
            previous_probability, previous_days = 100.0, (0, 0)
            for step in band["steps"]:
                assert step["probability_source"] == "touch_model"
                assert step["period_source"] == "touch_model"
                assert step["probability_label"] is None and step["period_label"] is None
                assert "무추세 변동성 모델" in step["probability_note"]
                expected = fm.touch_probability_range(last, step["price"], sigma, 30)
                # 같은 호가로 반올림되는 단계의 단조 보정만 허용한다(값은 모델보다 커질 수 없다).
                assert step["reach_probability_pct"] <= expected["mid"] * 100.0 + 0.06
                assert step["probability_low_pct"] <= step["reach_probability_pct"] <= step["probability_high_pct"]
                assert 0.0 <= step["probability_low_pct"] and step["probability_high_pct"] <= 100.0
                assert step["reach_probability_pct"] <= previous_probability
                previous_probability = step["reach_probability_pct"]
                assert 1 <= step["days_min"] <= step["days_max"] <= 30
                assert step["days_min"] >= previous_days[0] and step["days_max"] >= previous_days[1]
                previous_days = (step["days_min"], step["days_max"])
                assert "첫 도달일" in step["period_note"]
            first = band["steps"][0]
            expected_first = fm.touch_probability(last, first["price"], sigma, 30) * 100.0
            assert first["reach_probability_pct"] == pytest.approx(expected_first, abs=0.06)


def test_extreme_atr_still_gives_bounded_probabilities_and_periods():
    dd = _sample_buy_dd()
    # 최근 변동성이 과거보다 급격히 확대된 종목: 깊은 밴드의 확률은 낮아지고 기간은 30일 상한에 붙는다.
    dd["ATR"][-1] = 25.0
    result = _calculate(dd)
    all_steps = [
        step
        for family in ("aggressive_bands", "recommended_bands")
        for band in result[family]
        for step in band["steps"]
    ]

    assert all(step["probability_source"] == "touch_model" for step in all_steps)
    assert all(0.0 <= step["reach_probability_pct"] <= 100.0 for step in all_steps)
    assert all(1 <= step["days_min"] <= step["days_max"] <= 30 for step in all_steps)
    assert all(step["period_label"] is None for step in all_steps)
    for family in ("aggressive_bands", "recommended_bands"):
        for band in result[family]:
            assert [step["days_min"] for step in band["steps"]] == sorted(
                step["days_min"] for step in band["steps"]
            )
            assert [step["days_max"] for step in band["steps"]] == sorted(
                step["days_max"] for step in band["steps"]
            )


def test_deeper_steps_are_less_likely_and_slower_than_shallower_ones():
    result = _calculate(_sample_buy_dd())
    steps = [
        step
        for family in ("aggressive_bands", "recommended_bands")
        for band in result[family]
        for step in band["steps"]
    ]
    steps.sort(key=lambda step: step["price"], reverse=True)

    probabilities = [step["reach_probability_pct"] for step in steps]
    assert probabilities == sorted(probabilities, reverse=True)
    assert steps[0]["reach_probability_pct"] > 80.0 > 20.0 > steps[-1]["reach_probability_pct"]
    assert steps[0]["days_max"] <= steps[-1]["days_max"]


def test_forecast_band_ui_uses_aligned_five_column_stage_rows():
    for label in ("단계", "매수 가격 범위", "하락률", "도달 확률", "예상 기간"):
        assert f'role="columnheader">{label}</span>' in HTML
    assert "buy-stage-row" in HTML
    assert "buy-stage-price" in HTML
    assert "밴드 전체 매수 가격" in HTML
    assert "s.price_range" in HTML
    assert "분석 데이터 부족" in HTML
    assert "기간 산정 불가" in HTML
    assert "b.range_order_basis" in HTML
    assert "TP1" not in HTML.split("const renderBandCard", 1)[1].split(
        "const recBandsHtml", 1
    )[0]
    assert "1차 탐색 구간 · 소액 테스트" in HTML


# ── 어떤 종목이든 1차·2차 구간이 항상 유효하게 나오는지 확인하는 형태별 회귀 검증 ──────────────

def _dd_from_closes(closes, wick=0.015):
    """종가열만으로 calc_buy_price 입력(OHLCV + 지표)을 만든다. 저가주·급락·급등 같은 형태 검증용."""
    size = len(closes)
    opens = [close * (1.0 - 0.002 * math.sin(index)) for index, close in enumerate(closes)]
    highs = [max(open_, close) * (1.0 + wick * (1.0 + (index % 3) * 0.05))
             for index, (open_, close) in enumerate(zip(opens, closes))]
    lows = [min(open_, close) * (1.0 - wick * (1.0 + (index % 4) * 0.04))
            for index, (open_, close) in enumerate(zip(opens, closes))]
    volumes = [100_000 + (index % 11) * 4_500 for index in range(size)]
    true_ranges = [high - low for high, low in zip(highs, lows)]
    atrs = [
        sum(true_ranges[max(0, index - 13):index + 1]) / len(true_ranges[max(0, index - 13):index + 1])
        for index in range(size)
    ]

    def sma(period):
        return [
            None if index + 1 < period else sum(closes[index - period + 1:index + 1]) / period
            for index in range(size)
        ]

    def band_offset(index, multiple):
        window = closes[max(0, index - 19):index + 1]
        mean = sum(window) / len(window)
        return multiple * math.sqrt(sum((value - mean) ** 2 for value in window) / len(window))

    ma20 = sma(20)
    return {
        "Open": opens, "High": highs, "Low": lows, "Close": list(closes), "Volume": volumes, "ATR": atrs,
        "MA20": ma20, "MA60": sma(60), "MA120": sma(120), "EMA20": ma20, "BB_Middle": ma20,
        "BB_Lower": [None if value is None else value - band_offset(index, 2.0) for index, value in enumerate(ma20)],
        "BB_Upper": [None if value is None else value + band_offset(index, 2.0) for index, value in enumerate(ma20)],
        "RSI": [50.0] * size, "MACD": [0.0] * size, "Signal_Line": [0.0] * size,
        "ADX": [20.0] * size, "DI_Plus": [20.0] * size, "DI_Minus": [20.0] * size,
    }


def _shape_closes(kind, size, base):
    rng = random.Random(f"{kind}-{size}")
    if kind == "uptrend":
        return [base * (1.0 + 0.0025 * index) + base * 0.01 * math.sin(index / 3.0) for index in range(size)]
    if kind == "downtrend":
        return [base * (1.0 - 0.0018 * index) + base * 0.008 * math.sin(index / 3.0) for index in range(size)]
    if kind == "crash":  # 평탄하다가 5거래일에 -35%: 현재가 아래에 구조적 지지가 전혀 없다
        closes = [base] * size
        crash_at = max(1, size - 6)
        for index in range(crash_at, size):
            closes[index] = base * (1.0 - 0.07 * (index - crash_at + 1))
        return closes
    if kind == "spike":  # 평탄하다가 급등
        closes = [base] * size
        start = max(1, size - 4)
        for index in range(start, size):
            closes[index] = base * (1.0 + 0.15 * (index - start + 1))
        return closes
    if kind == "flat":
        return [base * (1.0 + 0.002 * math.sin(index)) for index in range(size)]
    if kind == "volatile":
        price, closes = base, []
        for _ in range(size):
            price *= 1.0 + rng.gauss(0.0, 0.04)
            closes.append(price)
        return closes
    raise ValueError(kind)


def _assert_family_contract(family):
    assert [band["band"] for band in family] == ["A", "B", "C"]
    for band in family:
        steps = band["steps"]
        assert band["is_available"] is True
        assert len(steps) == 5
        low, high = band["range"]
        assert 0 < low < high
        prices = [step["price"] for step in steps]
        assert len(set(prices)) == 5, f"밴드 {band['band']} 단계 가격 중복: {prices}"
        assert prices == sorted(prices, reverse=True)
        assert all(low <= price <= high for price in prices)
        assert all(
            steps[index]["price_range"][0] >= steps[index + 1]["price_range"][1]
            for index in range(4)
        )
        probabilities = [step["reach_probability_pct"] for step in steps]
        assert all(value is not None and 0.0 <= value <= 100.0 for value in probabilities)
        assert probabilities == sorted(probabilities, reverse=True)
        assert all(step["days_min"] is not None and 1 <= step["days_min"] <= step["days_max"] <= 30
                   for step in steps)
        assert band["evidence_level"] in {"structure", "indicator", "volatility"}
    for upper, lower in zip(family, family[1:]):
        assert upper["range"][0] > lower["range"][0] and upper["range"][1] > lower["range"][1]
        assert all(a["price"] > b["price"] for a, b in zip(upper["steps"], lower["steps"]))


@pytest.mark.parametrize("market,base", [("KRX", 1_250.0), ("KRX", 48_500.0), ("KRX", 650_000.0), ("US", 3.4), ("US", 187.5)])
@pytest.mark.parametrize("kind", ["uptrend", "downtrend", "crash", "spike", "flat", "volatile"])
def test_both_entry_families_are_always_valid_for_every_chart_shape(kind, market, base):
    for size in (12, 35, 80, 180):
        dd = _dd_from_closes(_shape_closes(kind, size, base))
        result = _calculate(dd, market=market)

        _assert_family_contract(result["aggressive_bands"])
        _assert_family_contract(result["recommended_bands"])
        # 2차 구간은 항상 1차 전체보다 아래에 있다
        assert max(band["range"][1] for band in result["recommended_bands"]) < min(
            band["range"][0] for band in result["aggressive_bands"]
        ), f"{kind}/{market}/{base}/size={size}: 2차가 1차와 겹침"
        # 현재가의 10% 하한 아래로는 내려가지 않는다
        floor_price = dd["Close"][-1] * 0.10
        assert all(
            band["range"][0] >= floor_price - 1e-9
            for family in ("aggressive_bands", "recommended_bands") for band in result[family]
        )


def test_narrow_low_priced_band_keeps_five_distinct_step_prices():
    # 호가 1원인 저가주에서 1차 밴드 폭이 4호가(112~116)일 때 4·5단계가 같은 112원으로 겹쳐
    # 검증이 실패했고, 그 결과 1차 구간 전체가 '표시 보류'가 되던 결함의 회귀 검증이다.
    dd = _sample_buy_dd()
    for regime in ("BULL", "NEUTRAL", "BEAR"):
        result = _calculate(dd, market_regime=regime)
        for family in ("aggressive_bands", "recommended_bands"):
            for band in result[family]:
                assert band["is_available"] is True
                prices = [step["price"] for step in band["steps"]]
                assert len(set(prices)) == 5, (regime, family, band["band"], prices)


def test_evidence_levels_reflect_how_many_independent_price_sources_overlap():
    result = _calculate(_sample_buy_dd())
    summary = result["price_structure"]["evidence_summary"]

    for family_name, family in (
        ("aggressive", result["aggressive_bands"]),
        ("recommended", result["recommended_bands"]),
    ):
        assert sum(summary[family_name].values()) == 3
        for band in family:
            sources = band["evidence_sources"]
            if band["evidence_level"] == "structure":
                assert len(sources) >= 2
            elif band["evidence_level"] == "indicator":
                assert len(sources) == 1
            else:
                assert sources == []
                assert "변동성" in band["evidence_note"]
            assert band["nearest_reference"] is None or band["nearest_reference"]["label"]
            probabilities = [step["reach_probability_pct"] for step in band["steps"]]
            low, high = band["reach_probability_range_pct"]
            assert low <= min(probabilities) and high >= max(probabilities)
            assert min(probabilities) <= band["reach_probability_mid_pct"] <= max(probabilities)


def test_entry_sections_always_use_fixed_titles_and_never_show_withheld_cards():
    assert "⚡ 1차 탐색 구간 · 소액 테스트" in HTML
    assert "📍 2차 매수 구간 · 본 진입" in HTML
    # 제목은 신규상장(관찰 전용)에서도 바뀌지 않고, 칩으로만 구분한다
    assert "⚡ 관찰 가격 구간 · 흐름 확인" not in HTML
    assert "관찰 전용</span>" in HTML
    # 옛 '표시 보류' 카드 문구가 더는 없어야 한다
    for withheld_text in ("가격 표시 보류", "탐색 범위 표시를 보류했습니다", "주 진입 범위 표시를 보류했습니다"):
        assert withheld_text not in HTML
    # 근거 등급과 추정 확률 표기
    assert "b.evidence_label" in HTML
    assert "b.nearest_reference" in HTML
    assert "b.reach_probability_range_pct" in HTML
    assert "probability_note" in HTML
