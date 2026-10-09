"""변동성 기반 예측 구간·도달 가능성·뉴스 정규화 순수 함수 테스트."""

from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from market_briefing import forecast_model as fm


def _gbm_like_closes(n=120, start=100.0, step=0.012):
    closes, price = [], start
    for i in range(n):
        price *= math.exp(step if i % 2 == 0 else -step * 0.9)
        closes.append(price)
    return closes


def test_blended_sigma_uses_realized_and_atr_and_has_floor():
    closes = _gbm_like_closes()
    result = fm.blended_daily_sigma(closes, closes[-1], atr=closes[-1] * 0.02)
    assert result["sigma"] is not None
    assert 0.005 < result["sigma"] < 0.03
    assert "실현 변동성" in result["basis"]

    flat = fm.blended_daily_sigma([100.0] * 40, 100.0, atr=None)
    assert flat["sigma"] == 0.002  # 0 변동성으로 범위가 붕괴하지 않도록 하한


def test_forecast_summary_is_ordered_positive_and_direction_consistent():
    up = fm.build_forecast_summary(last_price=100.0, sigma_daily=0.02, horizon_days=22, up_prob=60, down_prob=20)
    down = fm.build_forecast_summary(last_price=100.0, sigma_daily=0.02, horizon_days=22, up_prob=15, down_prob=55)
    flat = fm.build_forecast_summary(last_price=100.0, sigma_daily=0.02, horizon_days=22, up_prob=40, down_prob=38)

    for summary in (up, down, flat):
        lo90, hi90 = summary["range_p10_p90"]
        lo95, hi95 = summary["range_p05_p95"]
        assert 0 < lo95 < lo90 < summary["base_price"] < hi90 < hi95
    assert up["direction_key"] == "up" and up["expected_return_pct"] > 0
    assert down["direction_key"] == "down" and down["expected_return_pct"] < 0
    assert flat["direction_key"] == "neutral"
    # 방향 기울기는 최대 ±0.35σ√H 로 제한 — 기준가가 범위 중심에서 크게 벗어나지 않는다.
    assert abs(math.log(up["base_price"] / 100.0)) <= 0.35 * 0.02 * math.sqrt(22) + 1e-9


def test_touch_probability_and_day_window_follow_distance():
    near = fm.touch_probability(100.0, 101.0, 0.02, 22)
    far = fm.touch_probability(100.0, 130.0, 0.02, 22)
    assert near > 0.8 and far < 0.05
    window = fm.touch_day_window(100.0, 112.0, 0.016, 22)
    assert window["within_horizon"] is False
    assert window["days"][1] <= 22 and window["raw_days"][1] > 22
    close_window = fm.touch_day_window(100.0, 100.5, 0.016, 3)
    assert close_window["days"] == [1, 1]


def test_parse_news_datetime_supports_rss_iso_and_naver_formats():
    now = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    assert fm.parse_news_datetime("Fri, 11 Sep 2026 02:55:00 GMT", now).day == 11
    assert fm.parse_news_datetime("2026-09-11T20:02:10Z", now).hour == 20
    naver = fm.parse_news_datetime("2026.09.12 20:07", now)
    assert naver.astimezone(fm.KST).hour == 20
    short = fm.parse_news_datetime("09.11", now)
    assert short.astimezone(fm.KST).month == 9
    assert fm.parse_news_datetime("not a date", now) is None


def test_normalize_news_drops_stale_irrelevant_duplicates_and_sorts_latest_first():
    now = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    items = [
        {"title": "삼성전자 주가, 오늘 왜 하락했을까? - Investing.com", "published": "Fri, 11 Sep 2026 02:55:00 GMT"},
        {"title": "삼성전자 주가, 오늘 왜 하락했을까? - 다른매체", "published": "Fri, 11 Sep 2026 03:55:00 GMT"},
        {"title": "코스피 마감 시황", "published": "Sat, 12 Sep 2026 01:00:00 GMT"},
        {"title": "삼성전자 110조 주주환원", "published": "Sat, 01 Aug 2026 07:00:00 GMT"},
        {"title": "Samsung Drops 3.5% on chip contract news", "published": "Fri, 11 Sep 2026 20:02:10 +0000"},
    ]
    terms = fm.company_name_terms("삼성전자", "Samsung Electronics Co., Ltd.")
    kept, stats = fm.normalize_news_items(items, now=now, relevance_terms=terms,
                                          tickers=["005930"], require_relevance=True)
    titles = [item["title"] for item in kept]
    assert titles[0].startswith("Samsung Drops")          # 최신순
    assert len([t for t in titles if "왜 하락" in t]) == 1  # 매체만 다른 중복 제거
    assert "코스피 마감 시황" not in titles                  # 무관 기사 제외
    assert stats["stale_dropped"] == 1                       # 30일 초과 제외
    assert all(item["published_at"].endswith("Z") for item in kept)


def _outlook_kwargs(market="US", last=332.27, target=(371.17, 381.52), up=74, down=6, period="1mo"):
    from api import index

    closes = _gbm_like_closes(252, start=last / 1.4, step=0.011)
    scale = last / closes[-1]
    closes = [c * scale for c in closes]
    dd = {"Date": [f"2026-{(i // 21) % 12 + 1:02d}-{i % 20 + 1:02d}" for i in range(251)] + ["2026-09-11"],
          "Open": closes, "High": [c * 1.01 for c in closes], "Low": [c * 0.99 for c in closes],
          "Close": closes, "Volume": [1_000_000.0] * 252, "RSI": [62.8] * 252,
          "MACD": [1.0] * 252, "Signal_Line": [0.5] * 252, "MA20": closes, "MA60": closes}
    return index, dict(
        symbol="AAPL" if market == "US" else "005930.KS", market=market, dd=dd, last_price=last,
        prev_close=closes[-2], pct_change=1.75, atr=last * 0.024, regime="BULL", score=69,
        prob_up=77.7, prob_down=12.3,
        pivot_points={"classic": {"S1": last * 0.99, "R1": last * 1.004}},
        indicator_signals={}, buy_price={"strategy_rec": {"action_key": "split_buy"}},
        target_price={"min_price": target[0], "max_price": target[1], "reach_probability": 75.3},
        pullback_analysis={"stop_loss": last * 0.95}, signal_confidence={"confidence": 59},
        investor_flow=None, ai_strategy=None, candlestick_patterns=[], naver=None, us_enriched=None,
        toss_industry=None, event_risk=None, period=period,
    )


def test_outlook_caps_unrealistic_upside_and_reports_consistent_forecast():
    index, kwargs = _outlook_kwargs()
    result = index.build_prediction_outlook(**kwargs)
    upside = next(s for s in result["scenarios"] if s["key"] == "upside")
    forecast = result["forecast"]

    assert forecast["status"] in ("ok", "limited")
    assert forecast["horizon_days"] == 22
    lo90, hi90 = forecast["range_p10_p90"]
    assert 0 < lo90 < forecast["base_price"] < hi90
    assert upside["price_range"][1] <= forecast["range_p05_p95"][1] + 0.01
    assert forecast["upside_range_capped"] is True
    assert forecast["original_target_range"] == [371.17, 381.52]
    # 방향 라벨·기준가·기대수익률 부호가 서로 일치
    if forecast["direction_key"] == "up":
        assert forecast["expected_return_pct"] > 0 and forecast["base_price"] > kwargs["last_price"]
    assert forecast["target_date"] > "2026-09-11"
    assert max(upside["expected_days"]) <= 22
    assert result["decision"]["tp_confidence"] < 60   # +12% 목표를 22거래일 내 75% 로 표시하지 않는다


def test_outlook_marks_forecast_unavailable_for_short_history():
    index, kwargs = _outlook_kwargs()
    dd = kwargs["dd"]
    kwargs["dd"] = {k: v[-8:] for k, v in dd.items()}
    result = index.build_prediction_outlook(**kwargs)
    assert result["forecast"]["status"] == "unavailable"
    assert "base_price" not in result["forecast"]


def test_krx_target_date_skips_weekend_and_chuseok():
    from api import index

    projected = index._project_trading_date("KRX", "2026-09-23", 1)
    # 2026 추석 연휴(9/24~9/26)와 주말을 건너뛴 다음 거래일
    assert projected["date"] == "2026-09-28"
    us = index._project_trading_date("US", "2026-11-25", 1)
    assert us["date"] == "2026-11-27"   # 추수감사절(11/26) 제외


def test_naver_style_news_relevance_uses_summary_and_aliases():
    now = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    items = [
        {"title": "AI·반도체가 못 막은 중동 악재…코스피 7000선 무너졌다", "summary": "유가와 금리가 급등했다", "date": "2026.09.11 17:57"},
        {"title": "신형 아이폰 든 로제에 삼성 깜짝 등판", "summary": "삼성전자 공식 계정이 댓글을 남겼다", "date": "2026.09.12 17:57"},
        {"title": "삼전, 40만원 전엔 팔지 마라", "summary": "", "date": "2026.09.12 20:41"},
    ]
    terms = fm.company_name_terms("삼성전자") + ["삼전"]
    kept, stats = fm.normalize_news_items(items, now=now, relevance_terms=terms, tickers=["005930"],
                                          require_relevance=True, title_keys=("title", "summary"))
    assert [item["title"] for item in kept] == ["삼전, 40만원 전엔 팔지 마라", "신형 아이폰 든 로제에 삼성 깜짝 등판"]
    assert stats["irrelevant_dropped"] == 1


def test_relevance_uses_word_boundaries_for_short_tickers():
    assert fm.is_relevant_title("Ford (F) shares rise", [], ["F"]) is False  # 1글자 티커는 사용 안 함
    assert fm.is_relevant_title("AAPL stock slides", [], ["AAPL"]) is True
    assert fm.is_relevant_title("SNAPPY results", [], ["SNAP"]) is False


# ── 장기 변동성 혼합 (2026-10-09) ───────────────────────────────────────────────

def _vol_regime_closes(n_calm=190, n_hot=62, calm=0.005, hot=0.03, seed=3):
    import random
    rnd = random.Random(seed)
    price, closes = 100.0, []
    for i in range(n_calm + n_hot):
        sigma = calm if i < n_calm else hot
        price *= math.exp(rnd.gauss(0, sigma))
        closes.append(price)
    return closes


def test_long_run_volatility_is_mixed_in_half_when_history_is_long_enough():
    closes = _vol_regime_closes()
    result = fm.blended_daily_sigma(closes, closes[-1], atr=None)
    short, _ = fm.daily_log_volatility(closes)                           # 최근 60일: 급등한 변동성
    long_run, n_long = fm.daily_log_volatility(closes, window=fm.LONG_RUN_WINDOW, min_obs=fm.LONG_RUN_MIN_OBS)
    assert n_long >= fm.LONG_RUN_MIN_OBS and long_run < short
    expected = math.sqrt((1 - fm.LONG_RUN_WEIGHT) * short ** 2 + fm.LONG_RUN_WEIGHT * long_run ** 2)
    assert result["sigma"] == pytest.approx(expected)
    assert long_run < result["sigma"] < short                            # 단기 쏠림을 장기 수준으로 끌어당긴다
    assert "장기 변동성 혼합" in result["basis"] and result["long_run_observations"] == n_long


def test_short_history_keeps_the_short_window_estimate_only():
    closes = _vol_regime_closes(n_calm=60, n_hot=40)                     # 99개 수익률 < 120
    result = fm.blended_daily_sigma(closes, closes[-1], atr=None)
    short, _ = fm.daily_log_volatility(closes)
    assert result["sigma"] == pytest.approx(short)
    assert result["long_run"] is None and "장기" not in result["basis"]


def test_long_run_mix_applies_on_top_of_the_atr_blend():
    closes = _vol_regime_closes()
    atr = closes[-1] * 0.04
    with_atr = fm.blended_daily_sigma(closes, closes[-1], atr=atr)
    realized, _ = fm.daily_log_volatility(closes)
    atr_sigma = atr / closes[-1] / fm.ATR_TO_SIGMA
    short = math.sqrt(0.5 * realized ** 2 + 0.5 * atr_sigma ** 2)
    assert with_atr["sigma"] == pytest.approx(math.sqrt(0.5 * short ** 2 + 0.5 * with_atr["long_run"] ** 2))
    # ATR 를 관측하지 못한 경우(대체 ATR)에는 실현 변동성만 쓰는 기존 규칙 위에 같은 혼합을 적용한다.
    unobserved = fm.blended_daily_sigma(closes, closes[-1], atr=atr, atr_observed=False)
    assert unobserved["sigma"] == pytest.approx(math.sqrt(0.5 * realized ** 2 + 0.5 * with_atr["long_run"] ** 2))


def test_volatility_error_quantiles_describe_the_current_estimator():
    assert 0.7 < fm.VOL_ERROR_Q25 < 1.0 < fm.VOL_ERROR_Q75 < 1.4
    low = fm.touch_probability_range(100.0, 110.0, 0.02, 22)
    assert low["low"] < low["mid"] < low["high"]


def _stdlib_fallback_helpers(monkeypatch):
    import importlib.util
    from api import index as ix
    monkeypatch.setattr(ix, "_FORECAST_HELPERS", {})
    monkeypatch.setitem(sys.modules, "market_briefing.forecast_model", None)   # 패키지 import 실패
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *a, **k: None)  # 파일 직접 로드 실패
    helpers = ix._load_forecast_helpers()
    assert helpers["basis"] == "stdlib fallback"
    return helpers


def test_server_stdlib_fallback_matches_forecast_model(monkeypatch):
    """서버 내부 폴백 복제본이 패키지 구현과 어긋나면 import 실패 시에만 숫자가 달라진다 — 그 드리프트를 막는다."""
    import random
    rnd = random.Random(9)
    cases = []
    for n in (30, 80, 130, 252, 300):
        price, closes = 100.0, []
        for _ in range(n):
            price *= math.exp(rnd.gauss(0.0003, 0.018))
            closes.append(price)
        cases.append(closes)
    gapped = list(cases[3])
    gapped[100] = None
    gapped[180] = float("nan")
    cases.append(gapped)

    expected = [(fm.blended_daily_sigma(c, c[-1], c[-1] * 0.03, True), fm.blended_daily_sigma(c, c[-1], None, False))
                for c in cases]
    fallback = _stdlib_fallback_helpers(monkeypatch)
    for closes, (with_atr, without_atr) in zip(cases, expected):
        got = fallback["blended_daily_sigma"](closes, closes[-1], closes[-1] * 0.03, True)
        assert got["sigma"] == pytest.approx(with_atr["sigma"], rel=1e-9)
        assert got["long_run_observations"] == with_atr["long_run_observations"]
        got_plain = fallback["blended_daily_sigma"](closes, closes[-1], None, False)
        assert got_plain["sigma"] == pytest.approx(without_atr["sigma"], rel=1e-9)
    package_range = fm.touch_probability_range(100.0, 112.0, 0.02, 22)
    fallback_range = fallback["touch_probability_range"](100.0, 112.0, 0.02, 22)
    for key in ("low", "mid", "high"):
        assert fallback_range[key] == pytest.approx(package_range[key], rel=1e-12)


def test_long_run_weight_can_be_switched_off_from_the_environment(monkeypatch):
    """STOCKORACLE_VOL_LONG_RUN_WEIGHT=0 이면 예전 σ(단기 60일+ATR)와 예전 오차 분위 배율로 돌아간다."""
    import importlib
    closes = _vol_regime_closes()
    atr = closes[-1] * 0.03
    try:
        monkeypatch.setenv("STOCKORACLE_VOL_LONG_RUN_WEIGHT", "0")
        legacy = importlib.reload(fm)
        off = legacy.blended_daily_sigma(closes, closes[-1], atr=atr)
        realized, _ = legacy.daily_log_volatility(closes)
        atr_sigma = atr / closes[-1] / legacy.ATR_TO_SIGMA
        assert off["sigma"] == pytest.approx(math.sqrt(0.5 * realized ** 2 + 0.5 * atr_sigma ** 2))
        assert "장기 변동성" not in off["basis"]
        assert (legacy.VOL_ERROR_Q25, legacy.VOL_ERROR_Q75) == (0.836, 1.268)
        monkeypatch.setenv("STOCKORACLE_VOL_LONG_RUN_WEIGHT", "not-a-number")
        assert importlib.reload(fm).LONG_RUN_WEIGHT == 0.5          # 잘못된 값은 기본값
        monkeypatch.setenv("STOCKORACLE_VOL_LONG_RUN_WEIGHT", "7")
        assert importlib.reload(fm).LONG_RUN_WEIGHT == 1.0          # 범위 밖은 0~1 로 자른다
    finally:
        monkeypatch.delenv("STOCKORACLE_VOL_LONG_RUN_WEIGHT", raising=False)
        importlib.reload(fm)
    assert fm.LONG_RUN_WEIGHT == 0.5 and (fm.VOL_ERROR_Q25, fm.VOL_ERROR_Q75) == (0.800, 1.200)
