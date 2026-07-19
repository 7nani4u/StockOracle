"""국내외 종목 자동완성 API와 입력 UI 회귀 테스트."""

from api import index


def test_krx_suggestions_match_partial_name_and_preserve_market_suffix(monkeypatch):
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
    ):
        assert fragment in index.HTML
