"""변동성 기반 예측 구간·도달 가능성·뉴스 정규화 순수 함수 테스트."""

from datetime import datetime, timezone
import math
from pathlib import Path
import sys

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
