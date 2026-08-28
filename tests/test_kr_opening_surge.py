"""국내 개장 급등 추천의 계산·API·화면 계약 회귀 테스트."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import api.index as api_index

from api.index import (
    HTML,
    _KR_SURGE_MAX_CHANGE_EXCLUSIVE,
    _fetch_kr_surge_quotes,
    _kr_surge_item,
    _kr_surge_score,
    _kr_surge_session,
    _kr_surge_exclusion_reasons,
    _kr_surge_performance_payload,
    _record_kr_surge_recommendations,
    _krx_tick_size,
)


def _quote(**overrides):
    base = {
        "ticker": "005930.KS",
        "code": "005930",
        "name": "삼성전자",
        "market": "KOSPI",
        "price": 72_000,
        "prev_close": 67_000,
        "change_pct": 7.46,
        "open": 69_000,
        "high": 73_000,
        "low": 68_500,
        "volume": 20_000_000,
        "turnover": 120_000_000_000,
        "market_cap": 400_000_000_000_000,
        "local_traded_at": "2026-08-13T10:05:00+09:00",
    }
    base.update(overrides)
    return base


def _analysis():
    return {
        "atr": 2_000,
        "rsi": {"v": 64.0},
        "macd": {"macd": 2.0, "signal": 1.0},
        "adx": {"adx": 31.0, "direction": "bullish"},
        "rs": {"rs20": 4.2},
        "squeeze": {"momentum": 1.0},
        "obv": {"trend": "accumulation"},
    }


def test_krx_tick_size_respects_kospi_and_kosdaq_price_bands():
    assert _krx_tick_size(1_999, "KOSPI") == 1
    assert _krx_tick_size(2_000, "KOSPI") == 5
    assert _krx_tick_size(5_000, "KOSPI") == 10
    assert _krx_tick_size(20_000, "KOSPI") == 50
    assert _krx_tick_size(50_000, "KOSPI") == 100
    assert _krx_tick_size(200_000, "KOSPI") == 500
    assert _krx_tick_size(500_000, "KOSPI") == 1_000
    assert _krx_tick_size(500_000, "KOSDAQ") == 100


def test_kr_surge_score_rewards_confirmed_liquidity_and_penalizes_chasing():
    base_score, base_parts = _kr_surge_score(_quote(), _analysis(), 1.0)
    strong_score, strong_parts = _kr_surge_score(
        _quote(turnover=350_000_000_000), _analysis(), 3.2
    )
    overheated_score, overheated_parts = _kr_surge_score(
        _quote(price=80_000, change_pct=22.0, high=80_500, low=68_500),
        {**_analysis(), "rsi": {"v": 86.0}},
        3.2,
    )

    assert strong_score > base_score
    assert strong_parts["거래대금"] > base_parts["거래대금"]
    assert strong_parts["상대거래량"] > base_parts["상대거래량"]
    assert overheated_parts["과열감점"] < 0
    assert overheated_score < sum(v for k, v in overheated_parts.items() if k != "과열감점")
    assert _kr_surge_score(_quote(change_pct=_KR_SURGE_MAX_CHANGE_EXCLUSIVE), _analysis(), 2.0)[0] == -1


def test_kr_surge_card_prices_are_ordered_tick_aligned_and_limit_capped():
    item = _kr_surge_item(_quote(), _analysis(), 2.1, "regular")

    assert item is not None
    assert item["entry_low"] <= item["entry_high"]
    assert item["stop_loss"] < item["price"] < item["target_price"]
    assert item["target_price"] <= item["upper_limit"]
    for key in ("entry_low", "entry_high", "target_price", "stop_loss", "upper_limit"):
        assert item[key] % _krx_tick_size(item[key], item["market"]) == 0
    assert item["risk_reward"] > 0
    assert item["score_breakdown"]
    assert item["reasons"]


def test_non_regular_session_is_explicitly_marked_as_reference_data():
    saturday = datetime(2026, 8, 15, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    session = _kr_surge_session(saturday)
    item = _kr_surge_item(_quote(), _analysis(), 1.5, session["session"])

    assert session["session"] == "closed"
    assert "직전 확정 시세" in session["session_label"]
    assert any("정규장 외 시세" in warning for warning in item["warning"])


def test_quote_batch_parser_supports_current_naver_polling_schema(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "datas": [{
                    "itemCode": "005930",
                    "stockExchangeType": {"code": "KS"},
                    "closePriceRaw": "72000",
                    "compareToPreviousClosePriceRaw": "5000",
                    "compareToPreviousPrice": {"code": "2", "name": "RISING"},
                    "fluctuationsRatioRaw": "7.46",
                    "openPriceRaw": "69000",
                    "highPriceRaw": "73000",
                    "lowPriceRaw": "68500",
                    "accumulatedTradingVolumeRaw": "20000000",
                    "accumulatedTradingValueRaw": "120000000000",
                    "marketValueFullRaw": "400000000000000",
                    "marketStatus": "OPEN",
                    "tradableStatus": "tradable",
                    "tradeStopType": {"name": "TRADING"},
                }]
            }

    monkeypatch.setattr("api.index.requests.get", lambda *args, **kwargs: Response())
    rows = _fetch_kr_surge_quotes(["005930.KS"])

    assert len(rows) == 1
    assert rows[0]["name"] == "삼성전자"
    assert rows[0]["price"] == 72_000
    assert rows[0]["prev_close"] == 67_000
    assert rows[0]["turnover"] == 120_000_000_000


def test_domestic_surge_navigation_page_renderer_and_mobile_layout_exist():
    required_contract = (
        'id="nav-kr-surge"',
        'id="page-kr-surge"',
        "/api/kr/opening-surge",
        "function loadKrSurge",
        "function renderKrSurgeCards",
        "score_breakdown",
        "진입 보류·주의 조건",
        "@media(max-width:620px)",
    )
    for marker in required_contract:
        assert marker in HTML


def test_vi_and_market_warning_are_hard_exclusion_reasons():
    quote = _quote(
        tradable_status="tradable",
        trade_status="TRADING",
        vi={"active": True, "cooldown_active": False},
    )
    reasons = _kr_surge_exclusion_reasons(
        quote,
        {"management": True, "warning": True, "short_overheat": True},
    )

    assert "VI 발동" in reasons
    assert "관리종목" in reasons
    assert "투자경고" in reasons
    assert "단기과열" in reasons


def test_only_intraday_investor_flow_contributes_to_supply_score():
    unobserved = _quote(supply={
        "foreigner": {"net_amount": 50_000_000_000, "intraday": False},
        "institution": {"net_amount": 40_000_000_000, "intraday": False},
    })
    observed = _quote(supply={
        "foreigner": {"net_amount": 50_000_000_000, "intraday": True},
        "institution": {"net_amount": 40_000_000_000, "intraday": True},
    })

    old_score, old_parts = _kr_surge_score(unobserved, _analysis(), 2.0)
    new_score, new_parts = _kr_surge_score(observed, _analysis(), 2.0)

    assert old_parts["외인·기관"] == 0
    assert new_parts["외인·기관"] == 12
    assert new_score > old_score


def test_performance_tracker_records_once_and_completes_due_checkpoint(monkeypatch, tmp_path):
    clock = {"now": datetime(2026, 8, 13, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))}
    monkeypatch.setattr(api_index, "_KR_SURGE_TRACKING_PATH", str(tmp_path / "tracking.json"))
    monkeypatch.setattr(api_index, "_kr_surge_now", lambda: clock["now"])
    item = _kr_surge_item(_quote(), _analysis(), 2.0, "regular")
    session = {"session": "regular", "as_of": clock["now"].isoformat()}

    first = _record_kr_surge_recommendations([item], session)
    second = _record_kr_surge_recommendations([item], session)
    assert first["recorded"] == 1
    assert second["recorded"] == 0

    clock["now"] += timedelta(minutes=31)
    monkeypatch.setattr(api_index, "_fetch_kr_surge_quotes", lambda tickers: [
        {**_quote(price=73_000), "tradable_status": "tradable", "trade_status": "TRADING"}
    ])
    payload = _kr_surge_performance_payload(update=True)
    event = payload["events"][0]

    assert event["checkpoints"]["m30"]["status"] == "completed"
    assert event["checkpoints"]["m30"]["return_pct"] == 1.39
    assert event["checkpoints"]["m60"]["status"] == "pending"
    assert payload["summary"]["m30"]["completed"] == 1


def test_performance_api_and_supply_risk_ui_contract_exist():
    for marker in (
        "/api/kr/opening-surge/performance",
        "VI·시장경보 하드 제외",
        "외국인 수급",
        "기관 수급",
        "시장 프로그램",
        "성과 자동 추적",
        "function loadKrSurgePerformance",
    ):
        assert marker in HTML or marker in api_index.route.__code__.co_consts


def test_external_short_overheat_feed_sets_coverage_and_exclusion(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, **kwargs):
        if url == "https://feed.example/short-overheat":
            return Response({"items": [{"code": "005930", "active": True}]})
        return Response([])

    monkeypatch.setenv("KR_SURGE_SHORT_OVERHEAT_URL", "https://feed.example/short-overheat")
    monkeypatch.setattr(api_index.requests, "get", fake_get)
    for key in list(api_index._CACHE):
        if key.startswith("_fetch_kr_surge_risk_flags|"):
            api_index._CACHE.pop(key, None)

    snapshot = api_index._fetch_kr_surge_risk_flags()

    assert snapshot["coverage"]["short_overheat"] is True
    assert snapshot["flags"]["005930"]["short_overheat"] is True
    assert snapshot["short_overheat_source"] == "외부 JSON 피드"
