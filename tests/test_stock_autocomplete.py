"""국내외 종목 자동완성 API와 입력 UI 회귀 테스트."""

from api import index


def test_krx_suggestions_match_partial_name_and_preserve_market_suffix(monkeypatch):
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
