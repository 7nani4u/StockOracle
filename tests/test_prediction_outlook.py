"""예측 탭 구조화 로직과 미국 급등 추천 가격 경계 검증."""
import gzip
import io
import json
import sys
from types import SimpleNamespace

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import api.index as index_module
from market_briefing import confidence_engine as confidence_engine_module
from api.index import (
    HTML,
    ARTY_STRATEGY_RULE_VERSION,
    _KR_SURGE_VI_OBSERVATIONS,
    _kr_surge_vi_state,
    _parse_official_us_calendar,
    assess_confirmed_volume_recovery,
    _arty_atr_values,
    _build_us_ipo_risk,
    _confirmed_williams_fractals,
    _is_us_surge_price_eligible,
    _project_nasdaq_session_date,
    _smma_values,
    build_prediction_outlook,
    calc_arty_smma_fractal,
    calc_risk,
    calibrate_volume_recovery_threshold,
    compare_scalp_period_signals,
    fetch_arty_daily_data,
    fetch_execution_quality,
    fetch_scalp_period_comparison,
    get_market_session_calendar_status,
    get_krx_disclosure_cooldown,
    get_us_earnings_recommendation_block,
    recommend_scalp_analysis_period,
)


@pytest.fixture(autouse=True)
def _stable_official_us_calendar(monkeypatch):
    monkeypatch.setattr(index_module, "_fetch_us_official_calendar_reconciliation", lambda year: {
        "verified": True, "checked_at": "2026-08-24T00:00:00-04:00",
        "full_holidays": {}, "early_closes": {}, "discrepancies": {}, "sources": {},
    })
    monkeypatch.setattr(index_module, "_fetch_nasdaq_market_alert_status", lambda date_key: {
        "available": True, "emergency_closed": False, "early_close": False,
        "alerts": [], "checked_at": "2026-08-24T00:00:00-04:00",
    })


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


def _daily_liquidity(close, volume):
    return {"Close": [close] * 20, "Volume": [volume] * 20}


def _safe_scalp_context():
    return {
        "period_comparison": {
            "available": True, "conflict": False, "aligned": True,
            "verdict": "aligned", "verdict_label": "1일·3일 일치",
        },
        "volume_confirmation": {
            "available": True, "bar_confirmed": True, "recovered": True,
            "reason": "확정 봉 거래량 1.20배 · 회복 확인",
        },
        "execution_quality": {
            "available": True, "quality_ok": True,
            "spread_ok": True, "slippage_ok": True,
            "reason": "스프레드 0.05% · 예상 슬리피지 0.08% 통과",
        },
    }


def test_krx_scalp_period_uses_opening_time_turnover_and_atr():
    opening = recommend_scalp_analysis_period(
        "KRX", _daily_liquidity(100_000, 150_000), 100_000, 2_000,
        selected_period="3d",
        now=datetime(2026, 8, 24, 9, 10, tzinfo=ZoneInfo("Asia/Seoul")),
    )
    assert opening["recommended_period"] == "3d"
    assert opening["phase_key"] == "opening_noise"
    assert opening["liquidity_ok"] is True
    assert opening["atr_ok"] is True

    active = recommend_scalp_analysis_period(
        "KRX", _daily_liquidity(100_000, 150_000), 100_000, 2_000,
        selected_period="3d",
        now=datetime(2026, 8, 24, 10, 0, tzinfo=ZoneInfo("Asia/Seoul")),
        security_status={"vi": {"status": "normal", "active": False}},
        **_safe_scalp_context(),
    )
    assert active["recommended_period"] == "1d"
    assert active["recommended_interval"] == "5분봉"
    assert active["period_match"] is False
    assert active["entry_permission"] is False


def test_us_scalp_period_has_distinct_warmup_liquidity_and_atr_limits():
    early = recommend_scalp_analysis_period(
        "US", _daily_liquidity(100, 1_000_000), 100, 2,
        selected_period="3d",
        now=datetime(2026, 8, 24, 10, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    assert early["recommended_period"] == "3d"
    assert early["elapsed_minutes"] == 30

    active = recommend_scalp_analysis_period(
        "US", _daily_liquidity(100, 1_000_000), 100, 2,
        selected_period="1d",
        now=datetime(2026, 8, 24, 10, 30, tzinfo=ZoneInfo("America/New_York")),
        days_to_earnings=30,
        **_safe_scalp_context(),
    )
    assert active["recommended_period"] == "1d"
    assert active["period_match"] is True
    assert active["turnover_threshold"] == 50_000_000
    assert active["atr_range_pct"] == [0.8, 3.2]

    volatile = recommend_scalp_analysis_period(
        "US", _daily_liquidity(100, 1_000_000), 100, 4,
        selected_period="3d",
        now=datetime(2026, 8, 24, 10, 30, tzinfo=ZoneInfo("America/New_York")),
        days_to_earnings=30,
        **_safe_scalp_context(),
    )
    assert volatile["recommended_period"] == "3d"
    assert volatile["atr_ok"] is False

    premarket_response = recommend_scalp_analysis_period(
        "US", _daily_liquidity(100, 1_000_000), 100, 2,
        selected_period="3d", session_name="프리마켓",
        now=datetime(2026, 8, 24, 10, 30, tzinfo=ZoneInfo("America/New_York")),
        days_to_earnings=30,
        **_safe_scalp_context(),
    )
    assert premarket_response["recommended_period"] == "3d"
    assert premarket_response["phase_key"] == "reported_non_regular"


def test_vi_earnings_disclosure_and_opposite_signals_force_three_day():
    kr_now = datetime(2026, 8, 24, 10, 30, tzinfo=ZoneInfo("Asia/Seoul"))
    vi_block = recommend_scalp_analysis_period(
        "KRX", _daily_liquidity(100_000, 150_000), 100_000, 2_000,
        selected_period="1d", now=kr_now,
        security_status={"vi": {"status": "active", "label": "VI 발동", "active": True}},
        **_safe_scalp_context(),
    )
    assert vi_block["recommended_period"] == "3d"
    assert vi_block["vi_blocked"] is True

    us_now = datetime(2026, 8, 24, 10, 30, tzinfo=ZoneInfo("America/New_York"))
    earnings_block = recommend_scalp_analysis_period(
        "US", _daily_liquidity(100, 1_000_000), 100, 2,
        selected_period="1d", now=us_now, days_to_earnings=0,
        **_safe_scalp_context(),
    )
    assert earnings_block["recommended_period"] == "3d"
    assert earnings_block["earnings_today"] is True

    disclosure_block = recommend_scalp_analysis_period(
        "KRX", _daily_liquidity(100_000, 150_000), 100_000, 2_000,
        selected_period="1d", now=kr_now,
        disclosures=[{"title": "주요사항보고서", "date": "2026.08.24 10:05"}],
        **_safe_scalp_context(),
    )
    assert disclosure_block["recommended_period"] == "3d"
    assert disclosure_block["disclosure_cooldown"]["active"] is True

    opposite = _safe_scalp_context()
    opposite["period_comparison"] = {
        "available": True, "conflict": True, "aligned": False,
        "verdict": "opposite", "verdict_label": "1일·3일 반대 신호",
    }
    opposite_block = recommend_scalp_analysis_period(
        "KRX", _daily_liquidity(100_000, 150_000), 100_000, 2_000,
        selected_period="1d", now=kr_now, **opposite,
    )
    assert opposite_block["recommended_period"] == "3d"
    assert opposite_block["period_comparison"]["conflict"] is True


def test_unknown_vi_or_earnings_calendar_fails_closed_to_three_day():
    krx = recommend_scalp_analysis_period(
        "KRX", _daily_liquidity(100_000, 150_000), 100_000, 2_000,
        selected_period="1d",
        now=datetime(2026, 8, 24, 10, 30, tzinfo=ZoneInfo("Asia/Seoul")),
        **_safe_scalp_context(),
    )
    assert krx["recommended_period"] == "3d"
    assert krx["vi_check_available"] is False

    us = recommend_scalp_analysis_period(
        "US", _daily_liquidity(100, 1_000_000), 100, 2,
        selected_period="1d",
        now=datetime(2026, 8, 24, 10, 30, tzinfo=ZoneInfo("America/New_York")),
        **_safe_scalp_context(),
    )
    assert us["recommended_period"] == "3d"
    assert us["earnings_check_available"] is False


def test_after_close_earnings_blocks_the_next_trading_day_but_before_open_does_not_extend():
    after_close = {
        "earnings_date": "2026-08-21", "earnings_session": "after_close",
        "earnings_session_label": "장후 발표",
    }
    monday = get_us_earnings_recommendation_block(
        after_close,
        now=datetime(2026, 8, 24, 10, 30, tzinfo=ZoneInfo("America/New_York")),
    )
    assert monday["active"] is True
    assert monday["next_trading_date"] == "2026-08-24"
    assert monday["block_dates"] == ["2026-08-21", "2026-08-24"]

    before_open = {
        "earnings_date": "2026-08-24", "earnings_session": "before_open",
        "earnings_session_label": "장전 발표",
    }
    tuesday = get_us_earnings_recommendation_block(
        before_open,
        now=datetime(2026, 8, 25, 10, 30, tzinfo=ZoneInfo("America/New_York")),
    )
    assert tuesday["active"] is False
    assert tuesday["block_dates"] == ["2026-08-24"]


def test_earnings_provider_timestamp_is_classified_as_after_close(monkeypatch):
    event_at = pd.Timestamp.now(tz="America/New_York").normalize() + pd.Timedelta(days=2, hours=16)

    class FakeEarningsTicker:
        def get_earnings_dates(self, limit=8):
            return pd.DataFrame({"EPS Estimate": [1.0]}, index=pd.DatetimeIndex([event_at]))

        @property
        def calendar(self):
            return {}

    confidence_engine_module._CACHE.pop("earnings|EARNINGS-TIMING-TEST", None)
    monkeypatch.setattr(confidence_engine_module.yf, "Ticker", lambda symbol: FakeEarningsTicker())
    result = confidence_engine_module.get_earnings_proximity("EARNINGS-TIMING-TEST")
    assert result["earnings_session"] == "after_close"
    assert result["earnings_session_label"] == "장후 발표"
    assert result["earnings_datetime"].endswith("16:00:00-04:00") or "16:00:00-05:00" in result["earnings_datetime"]


def test_repeated_vi_activation_blocks_the_entire_observed_day(monkeypatch):
    code = "009999"
    _KR_SURGE_VI_OBSERVATIONS.pop(code, None)
    times = iter([
        datetime(2026, 8, 24, 10, 0, tzinfo=ZoneInfo("Asia/Seoul")),
        datetime(2026, 8, 24, 10, 3, tzinfo=ZoneInfo("Asia/Seoul")),
        datetime(2026, 8, 24, 10, 20, tzinfo=ZoneInfo("Asia/Seoul")),
    ])
    monkeypatch.setattr(index_module, "_kr_surge_now", lambda: next(times))
    active_row = {
        "tradeStopType": {"code": "VI", "name": "VI"},
        "tradableStatus": "tradable",
    }
    normal_row = {
        "tradeStopType": {"code": "1", "name": "TRADING"},
        "tradableStatus": "tradable",
    }
    first = _kr_surge_vi_state(code, active_row)
    _kr_surge_vi_state(code, normal_row)
    second = _kr_surge_vi_state(code, active_row)
    assert first["activation_count_today"] == 1
    assert first["day_blocked"] is False
    assert second["activation_count_today"] == 2
    assert second["day_blocked"] is True
    _KR_SURGE_VI_OBSERVATIONS.pop(code, None)


def test_volume_threshold_uses_stock_specific_walk_forward_distribution():
    closes = [100.0]
    volumes = []
    for index in range(260):
        ratio_boost = 1.45 if index % 7 == 0 else 0.90 + (index % 5) * 0.05
        volumes.append(1000.0 * ratio_boost)
        closes.append(closes[-1] * (1.002 if ratio_boost >= 1.4 else 0.9998))
    result = calibrate_volume_recovery_threshold({
        "Close": closes[:260], "Volume": volumes,
    })
    assert result["available"] is True
    assert 1.05 <= result["selected_ratio"] <= 1.50
    assert result["train_observations"] > result["validation_observations"]
    assert result["best"]["signals"] >= 8


def test_official_calendar_parsers_reconcile_closed_and_early_dates():
    nyse_html = """
    <table><tr><th>Holiday</th><th>2026</th></tr>
    <tr><td>Christmas Day</td><td>Friday, December 25</td></tr></table>
    <p>Each market will close early at 1:00 p.m. on Thursday, December 24, 2026.</p>
    """
    nasdaq_html = """
    <table><tr><th>2026</th><th>Holiday</th><th>Status</th></tr>
    <tr><td>December 24, 2026</td><td>Early Close</td><td>1:00 p.m.</td></tr>
    <tr><td>December 25, 2026</td><td>Christmas</td><td>Closed</td></tr></table>
    """
    nyse = _parse_official_us_calendar(nyse_html, "nyse", 2026)
    nasdaq = _parse_official_us_calendar(nasdaq_html, "nasdaq", 2026)
    assert set(nyse["full_holidays"]) == set(nasdaq["full_holidays"]) == {"2026-12-25"}
    assert set(nyse["early_closes"]) == set(nasdaq["early_closes"]) == {"2026-12-24"}


def test_official_emergency_market_alert_overrides_regular_calendar(monkeypatch):
    monkeypatch.setattr(index_module, "_fetch_nasdaq_market_alert_status", lambda date_key: {
        "available": True, "emergency_closed": True, "early_close": False,
        "alerts": [{"title": "U.S. equity market closed"}],
    })
    status = get_market_session_calendar_status(
        "US", datetime(2026, 8, 24, 10, 30, tzinfo=ZoneInfo("America/New_York")),
    )
    assert status["is_closed"] is True
    assert status["holiday_label"] == "Nasdaq 공식 긴급 휴장 경보"


def test_execution_quality_requires_fresh_real_bid_ask_and_limits_slippage(monkeypatch):
    now_epoch = 1_787_575_400
    monkeypatch.setattr(index_module.time, "time", lambda: now_epoch)

    class FakeTicker:
        @property
        def info(self):
            return {
                "bid": 99.95, "ask": 100.05, "bidSize": 100, "askSize": 120,
                "regularMarketTime": now_epoch - 30,
            }

    monkeypatch.setattr(index_module.yf, "Ticker", lambda symbol: FakeTicker())
    result = fetch_execution_quality(
        "EXECUTION-QUALITY-TEST", "US", 100.0, 100_000_000.0, 2.0,
    )
    assert result["available"] is True
    assert result["spread_pct"] == pytest.approx(0.1, abs=0.001)
    assert result["quality_ok"] is True
    assert result["estimated_slippage_pct"] <= result["slippage_limit_pct"]


def test_period_comparison_computes_both_horizons_independent_of_selection(monkeypatch):
    def fake_score(data, market, period):
        return (70 if period == "1d" else 30), None

    monkeypatch.setattr(index_module, "analyze_score", fake_score)
    one_day = {"Close": list(range(30))}
    three_day = {"Close": list(range(60))}
    result = compare_scalp_period_signals(one_day, three_day, "KRX")
    assert result["available"] is True
    assert result["one_day"]["direction"] == "up"
    assert result["three_day"]["direction"] == "down"
    assert result["conflict"] is True
    assert result["verdict"] == "opposite"


def test_period_comparison_fetch_reuses_five_minute_feed_for_confirmed_volume(monkeypatch):
    timezone = ZoneInfo("Asia/Seoul")
    ranges = [
        pd.date_range(
            datetime(2026, 8, day, 9, 0, tzinfo=timezone), periods=30, freq="5min",
        )
        for day in (19, 20, 21)
    ]
    index = ranges[0].append(ranges[1:])
    close = [100 + i * 0.1 for i in range(len(index))]
    frame = pd.DataFrame({
        "Open": [value - 0.1 for value in close],
        "High": [value + 0.2 for value in close],
        "Low": [value - 0.2 for value in close],
        "Close": close,
        "Volume": [100.0] * (len(index) - 1) + [150.0],
    }, index=index)

    class FakeTicker:
        def history(self, **kwargs):
            return frame

    monkeypatch.setattr(index_module.yf, "Ticker", lambda symbol: FakeTicker())
    monkeypatch.setattr(
        index_module, "analyze_score",
        lambda data, market, period: ((65 if period == "1d" else 60), None),
    )
    result = fetch_scalp_period_comparison("TEST-VOLUME-INDEPENDENT", "KRX")
    assert result["available"] is True
    assert result["volume_confirmation"]["bar_confirmed"] is True
    assert result["volume_confirmation"]["recovered"] is True
    assert result["volume_confirmation"]["interval_minutes"] == 5


def test_disclosure_without_time_keeps_same_day_in_cooldown():
    result = get_krx_disclosure_cooldown(
        [{"title": "공급계약", "date": "2026.08.24"}],
        now=datetime(2026, 8, 24, 14, 0, tzinfo=ZoneInfo("Asia/Seoul")),
    )
    assert result["active"] is True
    assert result["time_precision"] == "date"


def test_volume_recovery_waits_for_current_bar_close():
    timezone = ZoneInfo("Asia/Seoul")
    bar_start = datetime(2026, 8, 24, 10, 0, tzinfo=timezone)
    data = {
        "Date": [int((bar_start - timedelta(minutes=15 * i)).timestamp()) for i in range(20, -1, -1)],
        "Volume": [100] * 20 + [120],
    }
    pending = assess_confirmed_volume_recovery(
        data, "3d", "KRX", now=datetime(2026, 8, 24, 10, 10, tzinfo=timezone),
    )
    assert pending["bar_confirmed"] is False
    assert pending["recovered"] is False
    confirmed = assess_confirmed_volume_recovery(
        data, "3d", "KRX", now=datetime(2026, 8, 24, 10, 16, tzinfo=timezone),
    )
    assert confirmed["bar_confirmed"] is True
    assert confirmed["recovered"] is True


def test_market_calendar_cache_covers_krx_holiday_and_us_early_close():
    krx = get_market_session_calendar_status(
        "KRX", datetime(2026, 5, 1, 10, 0, tzinfo=ZoneInfo("Asia/Seoul")),
    )
    assert krx["is_closed"] is True
    assert krx["cache_ttl_seconds"] == 21600
    us = get_market_session_calendar_status(
        "US", datetime(2026, 11, 27, 14, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    assert us["is_early_close"] is True
    assert us["early_close_time"] == "13:00"


def test_scalp_period_ui_explains_reanalysis_and_no_buy_approval():
    assert 'id="scalp-period-recommendation"' in HTML
    assert 'id="scalp-entry-gate-section"' in HTML
    assert "applyScalpPeriodRecommendation()" in HTML
    assert "기간 추천은 매수 승인이 아닙니다." in HTML
    assert "3일 기본값과 동적 RSI만으로 매수하지 마세요." in HTML
    assert "VI·거래정지 단타 추천 차단" in HTML
    assert "미국 실적 일정 미확인: 보수적 대기" in HTML
    assert "미국 공식 달력 일일 교차검증 완료" in HTML
    assert "호가 스프레드·예상 슬리피지" in HTML
    assert "거래량 보정:" in HTML
    assert 'class="scalp-rec-details"' in HTML
    assert "상세 근거 보기" in HTML
    assert "⛔ 차단 사유 ${hardBlockers.length}개" in HTML
    assert "detailsWasOpen" in HTML
    assert "scalp-rec-chip" in HTML


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


def test_risk_scenarios_expose_only_ordered_independent_tp_estimate_ranges():
    dd = _sample_dd()
    for market in ("KRX", "US"):
        result = calc_risk(
            price=dd["Close"][-1],
            atr=2.1,
            market=market,
            dd=dd,
            event_risk={"score": 8, "level": "low", "reasons": []},
        )

        for key in ("conservative", "balanced", "aggressive"):
            scenario = result[key]
            levels = scenario["tp_levels"]
            assert 1 <= len(levels) <= 5
            assert [level["price"] for level in levels] == sorted(level["price"] for level in levels)
            assert scenario["tp_range"] == [levels[0]["price_range"][0], levels[-1]["price_range"][1]]
            probabilities = [level["prob_pct"] for level in levels]
            assert probabilities == sorted(probabilities, reverse=True)
            for level in levels:
                assert level["price_range"][0] <= level["price"] <= level["price_range"][1]
                assert level["prob_low_pct"] <= level["prob_pct"] <= level["prob_high_pct"]
                assert level["days_min"] <= level["avg_days"] <= level["days_max"]
                assert level["basis"]
            assert all(
                levels[index]["price_range"][1] < levels[index + 1]["price_range"][0]
                for index in range(len(levels) - 1)
            )
            assert "position_plan" not in scenario
            assert "failure_conditions" not in scenario

        assert result["conservative"]["tp_levels"][-1]["price"] < result["balanced"]["tp_levels"][0]["price"]
        assert result["balanced"]["tp_levels"][-1]["price"] < result["aggressive"]["tp_levels"][0]["price"]
        assert (
            result["conservative"]["tp_levels"][-1]["price_range"][1]
            < result["balanced"]["tp_levels"][0]["price_range"][0]
        )
        assert (
            result["balanced"]["tp_levels"][-1]["price_range"][1]
            < result["aggressive"]["tp_levels"][0]["price_range"][0]
        )


def test_removed_risk_card_sections_are_not_rendered():
    assert "분할 비중" not in HTML
    assert "최대 허용 손실" not in HTML
    assert "이 시나리오가 실패할 수 있는 조건" not in HTML
    assert "목표가 레벨별 도달 가능성" in HTML
    assert "예측 목표 가격 범위" not in HTML
    for column in ("목표 가격 범위", "수익률", "도달 가능성", "예상 거래일"):
        assert f'role="columnheader">{column}</span>' in HTML


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


def test_krx_outlook_uses_final_score_flow_and_selected_horizon():
    bullish = _base_kwargs()
    bullish["score"] = 78
    bullish["period"] = "3d"
    bullish_result = build_prediction_outlook(**bullish)

    bearish = _base_kwargs()
    bearish["score"] = 32
    bearish["period"] = "3d"
    bearish["investor_flow"] = {"ok": True, "외국인": -125_000, "기관": -82_000}
    bearish["pct_change"] = -3.0
    bearish_result = build_prediction_outlook(**bearish)

    bullish_up = next(s for s in bullish_result["scenarios"] if s["key"] == "upside")
    bearish_up = next(s for s in bearish_result["scenarios"] if s["key"] == "upside")
    assert bullish_up["probability"] > bearish_up["probability"]
    assert max(bullish_up["expected_days"]) <= 3
    assert "3거래일 분석 범위" in bullish_result["scenario_note"]
    assert "최근 20봉 평균" in " ".join(bullish_up["conditions"])
    assert any(check["label"] == "현재 충족" for check in bullish_up["checks"])
    assert bullish_result["levels"]["support_gap_pct"] < 0
    assert bullish_result["levels"]["resistance_gap_pct"] > 0


def test_krx_outlook_marks_missing_inputs_instead_of_claiming_normal_state():
    kwargs = _base_kwargs()
    kwargs["dd"] = {
        "Open": [100.0, 101.0],
        "High": [102.0, 103.0],
        "Low": [99.0, 100.0],
        "Close": [101.0, 102.0],
        "Volume": [],
    }
    kwargs["last_price"] = 102.0
    kwargs["prev_close"] = 101.0
    kwargs["atr"] = 2.04
    kwargs["atr_is_observed"] = False
    kwargs["investor_flow"] = {"ok": False}
    result = build_prediction_outlook(**kwargs)

    volume = next(item for item in result["status"] if item["key"] == "volume")
    volatility = next(item for item in result["status"] if item["key"] == "volatility")
    assert volume["value"] == "거래량 확인 필요"
    assert "시나리오 가중치에 반영하지 않음" in volume["detail"]
    assert volatility["detail"].startswith("대체 ATR")
    assert result["decision"]["confidence"] < 66
    assert any("기술지표 미확보" in gap for gap in result["market_context"]["data_gaps"])


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


def test_forecast_tab_keeps_only_actionable_sections_in_required_order():
    forecast_html = HTML.split('<div id="tab-forecast"', 1)[1].split('<!-- 뉴스 탭 -->', 1)[0]
    overview_pos = forecast_html.index("🔮 핵심 판단과 현재 상태")
    buy_pos = forecast_html.index('id="buy-price-section"')
    scenario_pos = forecast_html.index('id="prediction-scenarios-section"')
    risk_pos = forecast_html.index('id="risk-grid"')

    assert overview_pos < buy_pos < risk_pos < scenario_pos
    assert "분석 흐름" not in forecast_html
    assert "📈 목표 가격 범위" not in forecast_html
    assert 'id="target-price-section"' not in forecast_html
    assert 'id="ai-strategy-section"' not in forecast_html
    assert "AI 진단 탭의 추가 근거" not in forecast_html
    assert 'class="prediction-context-inline"' not in forecast_html
    assert 'id="prediction-context-section"' in forecast_html
    assert "시장·AI 판단 근거 상세 보기" not in forecast_html
    assert "<details" not in forecast_html
    assert "시장·업종·수급" in HTML
    assert "AI 진단 · 세력 흔들림 재활용" in HTML
    assert '<div class="card forecast-scenario-group">' in forecast_html


def test_forecast_overview_does_not_render_aggregate_confidence_card():
    renderer = HTML.split("function renderPredictionSections", 1)[1].split(
        "function renderForecast", 1
    )[0]

    assert 'class="prediction-confidence"' not in renderer
    assert 'class="prediction-confidence-head"' not in renderer
    assert 'class="prediction-confidence-value"' not in renderer
    assert 'class="prediction-confidence-range"' not in renderer
    assert "종합 신뢰도" not in renderer
    assert ".prediction-confidence{" not in HTML


def test_forecast_removes_current_condition_decision_card_and_risk_warnings():
    overview_renderer = HTML.split("function renderPredictionSections", 1)[1].split(
        "function renderForecast", 1
    )[0]
    forecast_renderer = HTML.split("function renderForecast", 1)[1].split(
        "function renderTechnicalSignals", 1
    )[0]

    assert "현재 조건에서의 대응 판단" not in overview_renderer
    assert 'class="prediction-decision"' not in overview_renderer
    assert 'role="status"' not in overview_renderer
    assert "1차 구간 이격 주의" not in forecast_renderer
    assert "추가 하락 위험:" not in forecast_renderer
    assert "bandDistanceHtml" not in forecast_renderer
    assert "downsideRiskHtml" not in forecast_renderer
    assert "buyRiskNotesEl.innerHTML = eventRiskHtml" in forecast_renderer
    assert "p.data_scope" not in overview_renderer
    assert ".prediction-context-inline{" not in HTML
    assert "현재 앱에서 확보 가능한 가격·거래량·기술지표·수급·시장 데이터 기준" not in HTML

    removed_decision_classes = (
        "prediction-decision",
        "prediction-kicker",
        "prediction-badges",
        "prediction-action",
        "prediction-chip",
        "prediction-summary",
    )
    for class_name in removed_decision_classes:
        assert f".{class_name}{{" not in HTML
        assert f'class="{class_name}"' not in overview_renderer


def test_us_balanced_tp2_uses_color_highlight_without_extra_copy():
    dd = _sample_dd()
    result = calc_risk(
        price=dd["Close"][-1],
        atr=2.1,
        market="US",
        dd=dd,
        event_risk={"score": 8, "level": "low", "reasons": []},
    )
    tp2 = result["balanced"]["tp_levels"][1]

    assert tp2["highlight_primary_exit"] is True
    assert result["balanced"]["tp_levels"][0]["highlight_primary_exit"] is False
    assert "display_label" not in tp2
    assert "action_guide" not in tp2
    assert "lv.highlight_primary_exit" in HTML
    assert "linear-gradient(90deg,#162a46,#111b2c)" in HTML


def test_forecast_renders_dynamic_rsi_purchase_timing_and_conditions():
    forecast_html = HTML.split('<div id="tab-forecast"', 1)[1].split('<!-- 뉴스 탭 -->', 1)[0]
    renderer = HTML.split("function renderPredictionSections", 1)[1].split(
        "function renderForecast", 1
    )[0]

    assert 'id="dynamic-rsi-purchase-timing"' in renderer
    assert "동적 RSI 구매 타이밍" in renderer
    assert "dynamic-rsi-condition-grid" in renderer
    assert "추격 제한 참고가" in renderer
    assert "이 단계 수는 상승확률이나 적중률이 아닙니다" in renderer
    assert forecast_html.index('id="prediction-overview-section"') < forecast_html.index('id="buy-price-section"')


def test_analysis_loader_has_timeout_cancellation_and_retry_path():
    analyzer = HTML.split("async function analyze", 1)[1].split("function setState", 1)[0]

    assert "new AbortController()" in analyzer
    assert "_ANALYSIS_TIMEOUT_MS" in analyzer
    assert "requestId !== _analysisRequestId" in analyzer
    assert "signal: controller.signal" in analyzer
    assert "finally" in analyzer
    assert 'id="analysis-retry-btn"' in HTML
    assert "function retryLastAnalysis" in HTML
    assert "function _startLoadingAnimation() {\n  _stopLoadingAnimation();" in HTML
    assert "_networkProfile()" in analyzer
    assert "&lite=1" in analyzer
    assert "75000" in analyzer


def test_lite_stock_response_reduces_chart_and_backtest_diagnostics():
    metric = {
        "trades": 30, "avg_return_pct": 1.2, "median_return_pct": 0.4,
        "profit_factor": 1.4, "unused_large_field": "x" * 1000,
    }
    payload = {
        "chart_data": {
            "dates": list(range(500)), "close": list(range(500)),
            "open": list(range(500)), "pattern_overlays": [{"x": 1}],
            "pattern_overlay_options": {"show_pattern_overlay": True},
        },
        "chart_patterns": list(range(20)), "candlestick_patterns": list(range(20)),
        "news": list(range(20)),
        "buy_price": {"arty_smma_fractal": {"backtest_validation": {
            "train": metric, "test": metric, "entry_test": {"1": metric},
            "retest_test": {"21": metric}, "atr_test": {"0.5": metric},
            "walk_forward": {"folds": [{"fold": i, "blob": "z" * 500} for i in range(4)], "aggregate": metric},
            "extended_diagnostics": {"market_regime": {"BULL": metric}},
            "data_quality": {"passed": True, "download": {"requested": 100, "success": 100, "tickers": list(range(100))},
                             "cross_provider_checks": [{"status": "passed", "blob": "q" * 500}]},
        }}},
    }

    before = len(json.dumps(payload))
    result = index_module._compact_stock_response(payload)
    after = len(json.dumps(result))

    assert result["response_meta"]["lite_mode"] is True
    assert len(result["chart_data"]["dates"]) <= 161
    assert result["chart_data"]["pattern_overlays"] == []
    assert result["buy_price"]["arty_smma_fractal"]["backtest_validation"]["train"].get("unused_large_field") is None
    assert after < before * 0.55


def test_send_uses_gzip_when_client_accepts_it():
    class Handler:
        path = "/api/stock"
        headers = {"Accept-Encoding": "gzip, deflate"}

        def __init__(self):
            self.output_headers = {}
            self.wfile = io.BytesIO()

        def send_response(self, status):
            self.status = status

        def send_header(self, key, value):
            self.output_headers[key] = value

        def end_headers(self):
            pass

    handler = Handler()
    index_module._send(handler, {"text": "모바일" * 2000})

    assert handler.output_headers["Content-Encoding"] == "gzip"
    decoded = json.loads(gzip.decompress(handler.wfile.getvalue()).decode("utf-8"))
    assert decoded["text"].startswith("모바일")


def test_daily_history_uses_fdr_when_yahoo_history_is_short(monkeypatch):
    short = index_module.pd.DataFrame(
        {"Open": range(60), "High": range(1, 61), "Low": range(60),
         "Close": range(1, 61), "Volume": [1000] * 60},
        index=index_module.pd.date_range("2026-01-01", periods=60, freq="B"),
    )
    full = index_module.pd.DataFrame(
        {"Open": range(260), "High": range(1, 261), "Low": range(260),
         "Close": range(1, 261), "Volume": [1000] * 260},
        index=index_module.pd.date_range("2025-01-01", periods=260, freq="B"),
    )

    class FakeTicker:
        def history(self, **_kwargs):
            return short

    monkeypatch.setattr(index_module.yf, "Ticker", lambda _symbol: FakeTicker())
    monkeypatch.setitem(sys.modules, "FinanceDataReader", SimpleNamespace(DataReader=lambda *_args: full))
    index_module._CACHE.clear()

    result = fetch_arty_daily_data("FDR-FALLBACK-TEST", "US")

    assert result is not None and len(result["Close"]) == 260
    assert result["_history_meta"]["provider"] == "FinanceDataReader"
    assert result["_history_meta"]["history_exhausted"] is False


def test_daily_provider_failure_is_not_mislabeled_as_short_listing():
    dd = {
        "Open": [], "High": [], "Low": [], "Close": [], "Date": [],
        "_history_meta": {"failure_reason": "daily_providers_unavailable", "attempts": []},
    }
    result = calc_arty_smma_fractal(dd, 100.0, market="US")

    assert result["status_key"] == "provider_error"
    assert result["status"] == "일봉 공급자 연결 실패"
    assert result["history_failure_reason"] == "daily_providers_unavailable"


def test_us_fundamentals_are_not_rendered_in_result_ui():
    assert 'id="r-us-fund"' not in HTML
    assert 'id="f-us-sector"' not in HTML
    assert "기업 펀더멘털 (Alpha Vantage)" not in HTML


def _arty_daily_dd(size=260, pivot_offset=3):
    closes = [100 + 0.03 * i + 0.00025 * i * i for i in range(size)]
    opens = [value - 0.15 for value in closes]
    highs = [value + 0.5 for value in closes]
    lows = [value - 0.5 for value in closes]
    if size >= 220:
        smma21 = _smma_values(closes, 21)
        pivot = size - pivot_offset
        lows[pivot] = smma21[pivot] - 0.05
    return {
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": [100_000 + (index % 5) * 1_000 for index in range(size)],
        "Date": [
            str(date(2025, 1, 1) + timedelta(days=index))
            for index in range(size)
        ],
    }


def _set_arty_profile(monkeypatch, market, config, verdict):
    profile = index_module.ARTY_SMMA_BACKTEST_EVIDENCE["markets"][market]
    monkeypatch.setitem(profile, "selected", dict(config))
    monkeypatch.setitem(profile, "verdict", verdict)
    monkeypatch.setitem(
        profile,
        "verdict_label",
        "검증 통과" if verdict == "accepted" else "검증 실패",
    )
    monkeypatch.setitem(profile, "production_applied", verdict == "accepted")


def test_smma_uses_sma_seed_then_wilder_recursive_smoothing():
    values = [float(i) for i in range(1, 9)]
    result = _smma_values(values, 3)

    assert result[:2] == [None, None]
    assert result[2] == 2.0
    assert result[3] == (2.0 * 2 + 4.0) / 3
    assert result[4] == (result[3] * 2 + 5.0) / 3


def test_williams_fractal_never_uses_unconfirmed_last_two_bars():
    highs = [10, 11, 15, 11, 10, 12, 20]
    lows = [8, 7, 6, 7, 8, 5, 4]
    result = _confirmed_williams_fractals(highs, lows)

    assert [item["index"] for item in result["upper"]] == [2]
    assert all(item["confirmed_index"] <= len(highs) - 1 for item in result["upper"] + result["lower"])
    assert all(item["index"] <= len(highs) - 3 for item in result["upper"] + result["lower"])


def test_arty_daily_atr_is_calculated_from_same_ohlc_series():
    dd = _arty_daily_dd()
    values = _arty_atr_values(dd["High"], dd["Low"], dd["Close"])

    assert values[:13] == [None] * 13
    assert values[-1] is not None
    assert values[-1] > 0


def test_arty_strategy_waits_until_configured_entry_bar(monkeypatch):
    config = {
        "retest_line": 21,
        "entry_delay": 1,
        "atr_tolerance": 0.35,
        "pullback_volume_max": None,
        "rebound_volume_min": None,
        "smma200_slope_min_pct20": None,
    }
    _set_arty_profile(monkeypatch, "US", config, "accepted")
    dd = _arty_daily_dd(pivot_offset=3)
    result = calc_arty_smma_fractal(dd, dd["Close"][-1], atr=0.001, market="US")

    assert result["available"] is True
    assert result["status_key"] == "entry_pending"
    assert result["entry_timing"]["state"] == "pending"
    assert result["entry_timing"]["bars_remaining"] == 1
    assert result["entry"] is None
    assert result["stop"] is None
    assert result["target"] is None


def test_arty_strategy_uses_delayed_open_and_blocks_rejected_market(monkeypatch):
    config = {
        "retest_line": 21,
        "entry_delay": 1,
        "atr_tolerance": 0.35,
        "pullback_volume_max": None,
        "rebound_volume_min": None,
        "smma200_slope_min_pct20": None,
    }
    dd = _arty_daily_dd(pivot_offset=4)

    _set_arty_profile(monkeypatch, "US", config, "accepted")
    accepted = calc_arty_smma_fractal(dd, dd["Close"][-1], atr=0.001, market="US")
    assert accepted["status_key"] == "confirmed"
    assert accepted["entry_timing"]["state"] == "eligible"
    assert accepted["entry"] == round(dd["Open"][-1], 4)
    assert accepted["stop"] < accepted["entry"] < accepted["target"]

    _set_arty_profile(monkeypatch, "US", config, "rejected")
    rejected = calc_arty_smma_fractal(dd, dd["Close"][-1], atr=9999, market="US")
    assert rejected["status_key"] == "technical_only"
    assert rejected["entry_timing"]["state"] == "technical_only"
    assert rejected["entry"] is None
    assert "검증 실패" in rejected["status"]


def test_arty_strategy_ignores_external_atr_and_matches_daily_rules(monkeypatch):
    config = {
        "retest_line": 21,
        "entry_delay": 1,
        "atr_tolerance": 0.35,
        "pullback_volume_max": None,
        "rebound_volume_min": None,
        "smma200_slope_min_pct20": None,
    }
    _set_arty_profile(monkeypatch, "US", config, "accepted")
    dd = _arty_daily_dd(pivot_offset=4)
    low_atr = calc_arty_smma_fractal(dd, dd["Close"][-1], atr=0.001, market="US")
    high_atr = calc_arty_smma_fractal(dd, dd["Close"][-1], atr=9999, market="US")

    assert low_atr["status_key"] == high_atr["status_key"]
    assert low_atr["retest"] == high_atr["retest"]
    assert low_atr["risk_distance"] == high_atr["risk_distance"]
    assert low_atr["atr_source"] == "동일 일봉 OHLC의 ATR14"
    assert low_atr["signal_conditions_match_backtest"] is True
    assert low_atr["strategy_rule_version"] == ARTY_STRATEGY_RULE_VERSION


def test_arty_strategy_confirms_fractal_without_lookahead(monkeypatch):
    config = {
        "retest_line": 21,
        "entry_delay": 1,
        "atr_tolerance": 0.35,
        "pullback_volume_max": None,
        "rebound_volume_min": None,
        "smma200_slope_min_pct20": None,
    }
    _set_arty_profile(monkeypatch, "US", config, "accepted")
    dd = _arty_daily_dd(pivot_offset=4)
    result = calc_arty_smma_fractal(dd, dd["Close"][-1], market="US")

    assert result["available"] is True
    assert result["status_key"] == "confirmed"
    assert result["alignment"] == "정배열"
    assert result["fan_expanding"] is True
    assert result["retest"]["confirmed"] is True
    assert result["retest"]["line"] == "SMMA21"
    assert result["retest"]["atr_tolerance"] == 0.35
    assert result["retests"]["21"]["confirmed"] is True
    assert result["retests"]["50"]["confirmed"] is False
    assert result["fractals"]["support"]["pivot_index"] == len(dd["Close"]) - 4
    assert result["fractals"]["support"]["confirmed_index"] == len(dd["Close"]) - 2
    assert result["stop"] < result["entry"] < result["target"]
    actual_rr = (result["target"] - result["entry"]) / (result["entry"] - result["stop"])
    assert abs(actual_rr - 2.0) < 0.001
    assert result["is_empirical_probability"] is False
    assert result["risk_distance"]["valid"] is True
    assert result["sideways_filter"]["filtered"] is False
    assert result["entry_timing"]["execution_model"] == "확정 후 1봉 시가"
    assert result["entry_timing"]["auto_order_scheduled"] is False
    assert result["backtest_validation"]["verdict"] == "accepted"


def test_arty_strategy_disables_signal_when_200_day_history_is_missing():
    dd = _arty_daily_dd(size=120)
    result = calc_arty_smma_fractal(dd, dd["Close"][-1], atr=1.5, market="KRX")

    assert result["available"] is False
    assert result["status_key"] == "insufficient"
    assert result["required_bars"] == 220
    assert result["observed_bars"] == 120
    assert result["missing_bars"] == 100
    assert result["condition_score_pct"] is None


def test_arty_recent_ipo_exposes_partial_observation_without_fabricating_smma200():
    dd = _arty_daily_dd(size=48)
    dd["_history_meta"] = {
        "provider": "Yahoo Finance via yfinance",
        "first_date": dd["Date"][0],
        "last_date": "2026-07-24",
        "requested_symbol": "EXYN",
        "resolved_symbol": "EXYN",
        "history_exhausted": True,
        "attempts": [
            {"symbol": "EXYN", "period": "2y", "rows": 48},
            {"symbol": "EXYN", "period": "max", "rows": 48},
        ],
    }

    result = calc_arty_smma_fractal(dd, dd["Close"][-1], market="US")

    assert result["available"] is False
    assert result["status"] == "신규상장 관찰 모드"
    assert result["observed_bars"] == 48
    assert result["missing_bars"] == 172
    assert result["history_completion_pct"] == 22
    assert result["history_exhausted"] is True
    assert result["partial_smma"]["21"] is not None
    assert result["partial_smma"]["50"] is None
    assert result["partial_smma"]["200"] is None
    assert result["estimated_ready_date"] == "2027-04-01"
    assert result["session_projection"]["sessions_needed"] == 172
    assert result["ipo_risk"]["ticker"] == "EXYN"


def test_nasdaq_220_bar_projection_excludes_scheduled_full_day_holidays():
    result = _project_nasdaq_session_date("2026-07-24", 172)

    assert result["date"] == "2027-04-01"
    assert result["sessions_needed"] == 172
    assert [row["date"] for row in result["scheduled_holidays_excluded"]] == [
        "2026-09-07",
        "2026-11-26",
        "2026-12-25",
        "2027-01-01",
        "2027-01-18",
        "2027-02-15",
        "2027-03-26",
    ]
    assert "2026-11-27" not in {
        row["date"] for row in result["scheduled_holidays_excluded"]
    }
    assert result["special_closure_warning"] is True


def test_exyn_ipo_risk_separates_resale_overhang_from_warrant_dilution():
    result = _build_us_ipo_risk("EXYN", 2.65, as_of="2026-07-26")

    assert result is not None
    assert result["risk_level"] == "high"
    assert result["risk_score_is_probability"] is False
    assert result["lockup_start_date"] == "2026-05-14"
    assert [
        (row["expiry_date"], row["days_remaining"])
        for row in result["lockups"]
    ] == [("2026-08-12", 17), ("2026-11-10", 107)]
    registration = result["registrations"][0]
    assert registration["effective_date"] == "2026-07-02"
    assert registration["registered_existing_shares"] == 3_658_564
    assert registration["new_dilution"] is False
    assert registration["supply_overhang_pct"] == 47.5
    assert result["shares_outstanding_reference"] == 7_707_460
    assert result["potential_warrant_shares"] == 3_242_852
    assert result["potential_warrant_dilution_pct"] == 42.1
    svb = next(row for row in result["warrants"] if row["label"] == "SVB 워런트")
    assert svb["dilution_counted"] is False
    assert all(row["in_the_money"] is False for row in result["warrants"])


def test_unverified_recent_ipo_does_not_fabricate_disclosure_metrics():
    result = _build_us_ipo_risk("UNVERIFIED", 5.0, as_of="2026-07-26")

    assert result is not None
    assert result["available"] is False
    assert "SEC 공시 프로필 미확보" in result["status"]
    assert "표시하지 않습니다" in result["warning"]


def test_arty_daily_fetch_retries_max_history_when_two_year_result_is_short(monkeypatch):
    short = index_module.pd.DataFrame(
        {
            "Open": range(120),
            "High": range(1, 121),
            "Low": range(120),
            "Close": range(1, 121),
            "Volume": [1000] * 120,
        },
        index=index_module.pd.date_range("2026-01-01", periods=120, freq="B"),
    )
    full = index_module.pd.DataFrame(
        {
            "Open": range(300),
            "High": range(1, 301),
            "Low": range(300),
            "Close": range(1, 301),
            "Volume": [1000] * 300,
        },
        index=index_module.pd.date_range("2025-01-01", periods=300, freq="B"),
    )
    calls = []

    class FakeTicker:
        def history(self, *, period, **_kwargs):
            calls.append(period)
            return short if period == "2y" else full

    monkeypatch.setattr(index_module.yf, "Ticker", lambda _symbol: FakeTicker())

    result = fetch_arty_daily_data("SHORT-HISTORY-TEST", "US")

    assert calls == ["2y", "max"]
    assert result is not None
    assert len(result["Close"]) == 300
    assert result["_history_meta"]["history_exhausted"] is False


def test_arty_evidence_matches_current_rule_version():
    evidence = index_module.ARTY_SMMA_BACKTEST_EVIDENCE
    assert evidence["strategy_rule_version"] == ARTY_STRATEGY_RULE_VERSION
    assert evidence["strategy_config_hash"]


def test_current_arty_evidence_is_diagnostic_only_for_both_markets():
    for market in ("KRX", "US"):
        profile = index_module.ARTY_SMMA_BACKTEST_EVIDENCE["markets"][market]
        assert profile["verdict"] == "inconclusive"
        assert profile["production_applied"] is False
        assert profile["verdict_label"] == "최신·롤링 결과 불일치"


def test_import_does_not_start_toss_prewarm_thread():
    assert index_module._prewarm_started is False


def test_forecast_renders_arty_smma_fractal_strategy_inside_buy_section():
    assert 'id="arty-smma-fractal-strategy"' in HTML
    assert 'id="arty-backtest-validation"' in HTML
    assert "SMMA·프랙탈 기술 조건" in HTML
    assert "확정 Williams Fractal(2봉 지연)" in HTML
    assert "① 종가/다음 시가" in HTML
    assert "② 재시험 분리" in HTML
    assert "③ 프랙탈 지연" in HTML
    assert "④ ATR 허용폭" in HTML
    assert "⑤ 거래량" in HTML
    assert "⑥ 횡보장" in HTML
    assert "⑦ 기업행위" in HTML
    assert "⑧ 분기 워크포워드" in HTML
    assert "⑨ 시장체제" in HTML
    assert "⑩ 민감도" in HTML
    assert "⑪ 데이터 품질" in HTML
    assert "나스닥 예정 전일 휴장 반영" in HTML
    assert "Nasdaq 거래 달력" in HTML
    assert 'id="ipo-supply-dilution-risk"' in HTML
    assert 'id="ipo-supply-dilution-risk-unverified"' in HTML
    assert "기존주식 재판매이므로 그 자체는 신주 희석이 아니며" in HTML
    assert "확인된 워런트 상한 시나리오" in HTML
    assert HTML.index('id="buy-price-section"') < HTML.index('id="arty-smma-fractal-strategy"')
