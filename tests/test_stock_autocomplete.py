"""국내외 종목 자동완성 API와 입력 UI 회귀 테스트."""

from api import index


def test_krx_suggestions_match_partial_name_and_preserve_market_suffix(monkeypatch):
    monkeypatch.setattr(index, "_enrich_krx_suggestion_status", lambda items, _query: items)
    monkeypatch.setattr(index, "get_krx_security_universe", lambda: ())
    monkeypatch.setattr(index, "search_krx_security_remote", lambda _q: ())
    monkeypatch.setattr(index, "get_krx_code_map", lambda: ({
        "DB하이텍": "000990.KS",
        "테스트바이오": "123456.KQ",
    }, {
        "000990": "DB하이텍",
        "123456": "테스트바이오",
    }))

    names = [item["name"] for item in index.search_stock_suggestions("하이", 10)]
    assert names[:3] == ["SK하이닉스", "하이브", "DB하이텍"]

    ticker, market, company = index.resolve_ticker("테스트바이오")
    assert (ticker, market, company) == ("123456.KQ", "KRX", "테스트바이오")


def test_us_suggestions_search_company_name_and_filter_non_us_exchange(monkeypatch):
    class _FakeSearch:
        def __init__(self, *_args, **_kwargs):
            self.quotes = [
                {
                    "symbol": "AAPL", "quoteType": "EQUITY", "exchange": "NMS",
                    "longname": "Apple Inc.", "exchDisp": "NASDAQ",
                },
                {
                    "symbol": "APC.F", "quoteType": "EQUITY", "exchange": "FRA",
                    "longname": "Apple Inc. Frankfurt", "exchDisp": "Frankfurt",
                },
            ]

    monkeypatch.setattr(index, "get_krx_code_map", lambda: ({}, {}))
    monkeypatch.setattr(index, "get_krx_security_universe", lambda: ())
    monkeypatch.setattr(index, "search_krx_security_remote", lambda _q: ())
    monkeypatch.setattr(index.yf, "Search", _FakeSearch)

    items = index.search_stock_suggestions("Apple", 10)
    assert items == [{
        "name": "Apple Inc.",
        "ticker": "AAPL",
        "code": "AAPL",
        "market": "US",
        "exchange": "NASDAQ",
    }]


def test_suggestions_route_and_autocomplete_ui(monkeypatch):
    expected = [{"name": "Apple Inc.", "ticker": "AAPL", "code": "AAPL", "market": "US", "exchange": "NASDAQ"}]
    monkeypatch.setattr(index, "search_stock_suggestions", lambda query, limit: expected)

    assert index.route("/api/suggestions", {"q": "Apple", "limit": "5"}) == {"items": expected}
    for fragment in (
        'id="ticker-suggestions"',
        'aria-autocomplete="list"',
        "/api/suggestions?q=",
        "event.key === 'ArrowDown'",
        "analyze(item.ticker)",
        "button.addEventListener('click'",
        "input.blur()",
        "_selectedStockTicker = item.ticker",
        "tickerOverride || _selectedStockTicker || inputValue",
    ):
        assert fragment in index.HTML

    assert "button.addEventListener('pointerdown'" not in index.HTML


def test_mobile_refresh_reuses_canonical_symbol_instead_of_display_name():
    """정식 회사명이 입력창에 표시돼도 PTR은 분석 결과의 티커를 재사용해야 한다."""
    assert "analyze(currentData && currentData.symbol ? currentData.symbol : '')" in index.HTML


def test_krx_universe_merges_stocks_etfs_and_etns_by_code(monkeypatch):
    stock = index._build_krx_security_record("108860", "셀바스AI", "코스닥", "STOCK")
    etf = index._build_krx_security_record("122630", "KODEX 레버리지", "KOSPI", "ETF")
    etn = index._build_krx_security_record(
        "530107", "삼성 인버스 2X 코스닥150 선물 ETN", "KOSPI", "ETN"
    )
    monkeypatch.setattr(index, "_download_krx_corporations", lambda: [stock])
    monkeypatch.setattr(index, "_download_krx_exchange_products", lambda: [etf, etn])

    records = {item["code"]: item for item in index._fetch_krx_security_universe()}

    assert records["108860"]["ticker"] == "108860.KQ"
    assert records["122630"]["security_type"] == "ETF"
    assert records["122630"]["is_leveraged"] is True
    assert records["530107"]["security_type"] == "ETN"
    assert records["530107"]["is_inverse"] is True


def test_selvas_ai_code_uses_remote_market_recovery_instead_of_wrong_ks(monkeypatch):
    selvas = index._build_krx_security_record("108860", "셀바스AI", "KOSDAQ", "STOCK")
    monkeypatch.setattr(index, "get_krx_code_map", lambda: ({}, {}))
    monkeypatch.setattr(index, "search_krx_security_remote", lambda _q: (selvas,))

    assert index.resolve_ticker("108860") == ("108860.KQ", "KRX", "셀바스AI")
    assert index.resolve_ticker("셀바스AI") == ("108860.KQ", "KRX", "셀바스AI")


def test_leveraged_etf_is_searchable_with_product_metadata(monkeypatch):
    leveraged = index._build_krx_security_record(
        "122630", "KODEX 레버리지", "KOSPI", "ETF"
    )
    monkeypatch.setattr(
        index, "get_krx_code_map",
        lambda: ({"KODEX 레버리지": "122630.KS"}, {"122630": "KODEX 레버리지"}),
    )
    monkeypatch.setattr(index, "get_krx_security_universe", lambda: (leveraged,))
    monkeypatch.setattr(index, "search_krx_security_remote", lambda _q: ())
    monkeypatch.setattr(index, "_enrich_krx_suggestion_status", lambda items, _query: items)

    items = index.search_stock_suggestions("122630", 10)

    assert items[0] == {
        "name": "KODEX 레버리지",
        "ticker": "122630.KS",
        "code": "122630",
        "market": "KRX",
        "exchange": "ETF · KOSPI",
        "security_type": "ETF",
        "is_leveraged": True,
        "is_inverse": False,
    }


def test_unknown_numeric_code_is_not_silently_assigned_to_kospi(monkeypatch):
    monkeypatch.setattr(index, "get_krx_code_map", lambda: ({}, {}))
    monkeypatch.setattr(index, "search_krx_security_remote", lambda _q: ())

    assert index.resolve_ticker("999999") == (None, None, None)


def test_alphanumeric_krx_short_code_is_preserved_and_resolved(monkeypatch):
    product = index._build_krx_security_record(
        "0000D0", "테스트 신규 ETF", "KOSPI", "ETF"
    )
    monkeypatch.setattr(
        index, "get_krx_code_map",
        lambda: ({product["name"]: product["ticker"]}, {product["code"]: product["name"]}),
    )
    monkeypatch.setattr(index, "search_krx_security_remote", lambda _q: ())

    assert product["code"] == "0000D0"
    assert index.resolve_ticker("0000D0") == ("0000D0.KS", "KRX", "테스트 신규 ETF")
    assert index.resolve_ticker("0000D0.KS") == ("0000D0.KS", "KRX", "테스트 신규 ETF")


def test_security_status_policy_blocks_halt_and_delisting_but_allows_management():
    management = index._compose_krx_security_status(
        "111111", flags={"management": True}
    )
    halted = index._compose_krx_security_status(
        "222222", flags={"management": True, "trading_halt": True}
    )
    delisting = index._compose_krx_security_status(
        "333333", live_detail={"available": True, "delisting_scheduled": True}
    )

    assert management["analysis_blocked"] is False
    assert management["status_labels"] == ["관리종목"]
    assert halted["analysis_blocked"] is True
    assert "거래정지" in halted["block_reason"]
    assert delisting["analysis_blocked"] is True
    assert "상장폐지 예정" in delisting["block_reason"]


def test_latest_listing_maintenance_notice_overrides_older_delisting_decision():
    html = """
    <table>
      <tr><th>제목</th><th>정보제공</th><th>날짜</th></tr>
      <tr><td>테스트(주) 상장유지 결정</td><td>KOSCOM</td><td>2026.08.18</td></tr>
      <tr><td>테스트(주) (정정)상장폐지</td><td>KOSCOM</td><td>2026.08.17</td></tr>
    </table>
    """

    status = index._parse_krx_delisting_notice(html)

    assert status["scheduled"] is False
    assert "상장유지" in status["title"]


def test_halt_release_parser_accepts_only_current_or_future_disclosed_date():
    current = index._parse_krx_halt_release_content(
        "매매거래정지해제일 2026년08월18일 09:00 부터", today="2026-08-18"
    )
    stale = index._parse_krx_halt_release_content(
        "3. 해제일시 2026-08-14 - 4. 근거규정", today="2026-08-18"
    )

    assert current["halt_release_date"] == "2026-08-18"
    assert current["halt_release_time"] == "09:00"
    assert current["halt_release_basis"] == "공시 명시"
    assert stale["halt_release_date"] == ""
    assert stale["halt_release_label"] == "해제일 미정"


def test_halt_release_parser_preserves_indefinite_disclosure_condition():
    status = index._parse_krx_halt_release_content(
        "3.정지기간 가.정지일시 2026-06-08 - 나.만료일시 "
        "상장폐지결정 등 효력정지 가처분신청에 대한 법원의 결정 확인시까지 "
        "4.근거규정 코스닥시장규정"
    )

    assert status["halt_release_date"] == ""
    assert status["halt_release_label"] == "해제일 미정"
    assert "법원의 결정 확인시까지" in status["halt_release_condition"]
    assert status["halt_release_basis"] == "조건부·미정"


def test_halted_security_status_exposes_release_guidance():
    status = index._compose_krx_security_status(
        "222222",
        flags={"trading_halt": True},
        live_detail={
            "available": True,
            "halt_release_date": "2026-08-21",
            "halt_release_time": "09:00",
            "halt_release_label": "2026-08-21 09:00",
            "halt_release_basis": "공시 명시",
        },
    )

    assert status["halt_release_label"] == "2026-08-21 09:00"
    assert any("거래정지 예상 해제일: 2026-08-21 09:00" in warning
               for warning in status["warnings"])


def test_new_listing_history_warnings_follow_20_60_120_boundaries():
    dates = [day.date().isoformat() for day in index.pd.bdate_range("2026-01-02", periods=130)]

    under_20 = index._build_listing_history_status(
        dates[:19], dates[0], today=dates[-1]
    )
    under_60 = index._build_listing_history_status(
        dates[:20], dates[0], today=dates[-1]
    )
    under_120 = index._build_listing_history_status(
        dates[:60], dates[0], today=dates[-1]
    )
    sufficient = index._build_listing_history_status(
        dates[:120], dates[0], today=dates[-1]
    )

    assert (under_20["stage"], under_20["remaining_trading_days"]) == ("under_20", 1)
    assert (under_60["stage"], under_60["remaining_trading_days"]) == ("under_60", 40)
    assert (under_120["stage"], under_120["remaining_trading_days"]) == ("under_120", 60)
    assert sufficient["stage"] == "sufficient"
    assert sufficient["data_sufficient"] is True
    assert sufficient["warning"] == ""


def test_prediction_history_confidence_caps_follow_listing_stages():
    prices = [100.0 + day * 0.7 + (day % 6) * 0.15 for day in range(150)]

    under_20 = index._prediction_history_confidence_profile(prices[:19], 19)
    under_60 = index._prediction_history_confidence_profile(prices[:45], 45)
    under_120 = index._prediction_history_confidence_profile(prices[:90], 90)
    sufficient = index._prediction_history_confidence_profile(prices, 150)

    assert under_20["stage_confidence_cap"] == 45
    assert under_20["optimal_lookback_days"] is None
    assert under_60["stage_confidence_cap"] == 60
    assert under_120["stage_confidence_cap"] == 75
    assert sufficient["stage_confidence_cap"] == 90
    for profile in (under_60, under_120, sufficient):
        assert profile["optimal_lookback_days"] >= 20
        assert profile["optimal_lookback_days"] % 5 == 0
        assert profile["candidate_count"] > 1
        assert profile["confidence_cap"] <= profile["stage_confidence_cap"]


def test_optimal_prediction_window_is_not_limited_to_fixed_thresholds():
    import random

    rng = random.Random(3)
    prices = [100.0]
    for _ in range(179):
        prices.append(max(1.0, prices[-1] * (1.0 + rng.gauss(0.0005, 0.02))))

    profile = index._prediction_history_confidence_profile(prices, len(prices))

    assert profile["candidate_count"] > 3
    assert profile["optimal_lookback_days"] == 35
    assert profile["optimal_lookback_days"] not in {20, 60, 120}
    assert profile["validation_points"] >= 12


def test_history_confidence_cap_is_applied_to_point_and_interval():
    from market_briefing.confidence_engine import build_signal_confidence

    result = build_signal_confidence(
        technical_score=88, ai_score=86, sentiment_score=84, market_score=82,
        include_macro=False, include_sector=False, include_earnings=False,
        history_confidence={
            "available_trading_days": 42,
            "confidence_cap": 60,
            "optimal_lookback_days": 25,
            "cap_reason": "신규 상장 데이터 42거래일 · 최종 신뢰도 상한 60%",
        },
    )

    assert result["confidence"] <= 60
    assert result["confidence_interval"]["upper"] <= 60
    assert result["history_confidence"]["optimal_lookback_days"] == 25
    assert any("신규 상장 데이터 42거래일" in reason for reason in result["cap_reasons"])


def test_search_result_marks_management_halt_and_delisting(monkeypatch):
    record = index._build_krx_security_record("009310", "참엔지니어링", "KOSPI", "STOCK")
    monkeypatch.setattr(index, "get_krx_security_universe", lambda: (record,))
    monkeypatch.setattr(
        index, "_krx_status_directory",
        lambda: {
            "flags": {"009310": {"management": True, "trading_halt": True}},
            "details": {"009310": {"trading_halt_date": "20260720"}},
        },
    )
    monkeypatch.setattr(
        index, "_fetch_krx_security_detail_status",
        lambda _code: {
            "available": True, "management": True, "trading_halt": True,
            "delisting_scheduled": True, "newly_listed": False,
            "halt_release_label": "해제일 미정",
            "halt_release_condition": "법원 결정 확인시까지",
        },
    )

    enriched = index._enrich_krx_suggestion_status([{
        "name": "참엔지니어링", "ticker": "009310.KS", "code": "009310",
        "market": "KRX", "exchange": "KOSPI", "security_type": "STOCK",
    }], "009310")

    assert enriched[0]["management"] is True
    assert enriched[0]["trading_halt"] is True
    assert enriched[0]["delisting_scheduled"] is True
    assert enriched[0]["analysis_blocked"] is True
    assert enriched[0]["halt_release_label"] == "해제일 미정"


def test_new_listing_is_promoted_into_intraday_cache_immediately(monkeypatch):
    monkeypatch.setattr(index, "_KRX_UNIVERSE_CACHE", {
        "records": (), "updated_at": 123.0, "session_date": index._krx_session_date(),
    })
    record = index._build_krx_security_record("1234A0", "당일신규ETF", "KOSPI", "ETF")

    index._promote_new_krx_listing(record)

    cached = index._KRX_UNIVERSE_CACHE
    assert cached["records"][0]["code"] == "1234A0"
    assert cached["records"][0]["is_new_listing"] is True
    assert cached["updated_at"] == 0.0


def test_stock_route_blocks_restricted_krx_before_price_analysis(monkeypatch):
    monkeypatch.setattr(
        index, "resolve_ticker", lambda _raw: ("009310.KS", "KRX", "참엔지니어링")
    )
    monkeypatch.setattr(index, "get_krx_security_status", lambda _ticker: {
        "analysis_blocked": True,
        "block_reason": "상장폐지 예정 · 거래정지",
        "status_labels": ["상장폐지 예정", "거래정지"],
        "trading_halt": True,
        "halt_release_label": "해제일 미정",
        "halt_release_condition": "법원 결정 확인시까지",
    })
    monkeypatch.setattr(
        index, "fetch_stock_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("호출되면 안 됨")),
    )

    result = index.route("/api/stock", {"ticker": "009310", "period": "1y"})

    assert result["error_code"] == "KRX_SECURITY_BLOCKED"
    assert "분석을 차단" in result["error"]
    assert "해제일 미정" in result["error"]
    assert "법원 결정 확인시까지" in result["error"]


def test_security_status_badges_and_banner_exist_in_ui():
    for fragment in (
        "_stockSuggestionStatusTags",
        "상장폐지 예정",
        "거래정지",
        "관리종목",
        "신규상장",
        "해제일 미정",
        "item.analysis_blocked",
        'id="security-status-banner"',
        "d.security_status || {}",
    ):
        assert fragment in index.HTML
