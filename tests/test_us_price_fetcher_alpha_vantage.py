from us_price_fetcher import AlphaVantageClient, MarketSession


def test_alpha_vantage_global_quote_accepts_orchestrator_session(monkeypatch):
    client = AlphaVantageClient("test")
    monkeypatch.setattr(
        client,
        "_avget",
        lambda _params, _key: {
            "Global Quote": {
                "05. price": "101.25",
                "08. previous close": "100.00",
                "07. latest trading day": "2026-07-24",
            }
        },
    )

    result = client.global_quote("AAPL", MarketSession.CLOSED)

    assert result is not None
    assert result[:2] == (101.25, 100.0)


def test_alpha_vantage_daily_close_accepts_orchestrator_session(monkeypatch):
    client = AlphaVantageClient("test")
    monkeypatch.setattr(
        client,
        "_avget",
        lambda _params, _key: {
            "Time Series (Daily)": {
                "2026-07-24": {"4. close": "101.25"},
                "2026-07-23": {"4. close": "100.00"},
            }
        },
    )

    result = client.daily_close("AAPL", MarketSession.CLOSED)

    assert result is not None
    assert result[:2] == (101.25, 100.0)
