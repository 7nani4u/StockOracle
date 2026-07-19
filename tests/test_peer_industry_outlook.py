"""동종업계 모멘텀 계산 및 API 경로 테스트."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from api import index


def test_peer_group_resolves_known_krx_and_us_symbols():
    krx_group, krx_members = index._resolve_peer_group("005930.KS", "KRX")
    us_group, us_members = index._resolve_peer_group("AAPL", "US")
    detailed_group, detailed_members = index._resolve_peer_group(
        "AAPL", "US", "Technology", "Consumer Electronics"
    )

    assert krx_group == "반도체"
    assert any(symbol == "000660.KS" for symbol, _ in krx_members)
    assert us_group == "Technology"
    assert any(symbol == "MSFT" for symbol, _ in us_members)
    assert detailed_group == "Consumer Electronics"
    assert any(symbol == "DELL" for symbol, _ in detailed_members)


def test_outlook_aggregates_peer_momentum_and_excludes_selected_stock(monkeypatch):
    symbols = ["TEST", "AAA", "BBB"]
    dates = pd.date_range("2026-01-01", periods=45, freq="B")
    raw = pd.DataFrame(
        np.column_stack([
            np.linspace(100, 118, len(dates)),
            np.linspace(80, 100, len(dates)),
            np.linspace(120, 105, len(dates)),
        ]),
        index=dates,
        columns=pd.MultiIndex.from_product([["Close"], symbols]),
    )

    monkeypatch.setattr(
        index, "_resolve_peer_group",
        lambda *args, **kwargs: ("Test Industry", [("TEST", "Selected"), ("AAA", "Peer A"), ("BBB", "Peer B")]),
    )
    monkeypatch.setattr(index.yf, "download", lambda *args, **kwargs: raw)
    index._CACHE.pop("build_peer_industry_outlook|('TEST', 'US', 'Selected', 'Test', 'Test Industry')|[]", None)

    result = index.build_peer_industry_outlook("TEST", "US", "Selected", "Test", "Test Industry")

    assert result["ok"] is True
    assert result["peer_count"] == 2
    assert {peer["ticker"] for peer in result["peers"]} == {"AAA", "BBB"}
    assert "TEST" not in {peer["ticker"] for peer in result["peers"]}
    assert result["up_probability"] + result["down_probability"] == 100
    assert result["selected"] is not None
    assert result["selected"]["rsi"] > 90
    assert result["relative_to_industry"] is not None


def test_peer_outlook_route_preserves_krx_and_us_market(monkeypatch):
    calls = []

    def fake_outlook(symbol, market, company, sector, industry):
        calls.append((symbol, market, company, sector, industry))
        return {"ok": True, "symbol": symbol, "market": market}

    monkeypatch.setattr(index, "build_peer_industry_outlook", fake_outlook)

    krx = index.route("/api/peer-outlook", {
        "ticker": "005930.KS", "market": "KRX", "company": "삼성전자",
        "sector": "반도체", "industry": "반도체 제조",
    })
    us = index.route("/api/peer-outlook", {
        "ticker": "AAPL", "market": "US", "company": "Apple",
        "sector": "Technology", "industry": "Consumer Electronics",
    })

    assert krx == {"ok": True, "symbol": "005930.KS", "market": "KRX"}
    assert us == {"ok": True, "symbol": "AAPL", "market": "US"}
    assert calls[0][:2] == ("005930.KS", "KRX")
    assert calls[1][:2] == ("AAPL", "US")
