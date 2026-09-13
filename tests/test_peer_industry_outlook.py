"""동종업계 모멘텀 계산 및 API 경로 테스트."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from api import index


def test_peer_group_resolves_known_krx_and_us_symbols():
    krx_group, krx_members, krx_src = index._resolve_peer_group("005930.KS", "KRX")
    us_group, us_members, us_src = index._resolve_peer_group("AAPL", "US")
    detailed_group, detailed_members, detailed_src = index._resolve_peer_group(
        "AAPL", "US", "Technology", "Consumer Electronics"
    )

    assert krx_group == "반도체"
    assert any(symbol == "000660.KS" for symbol, _ in krx_members)
    assert us_group == "Technology"
    assert any(symbol == "MSFT" for symbol, _ in us_members)
    assert detailed_group == "Consumer Electronics"
    assert any(symbol == "DELL" for symbol, _ in detailed_members)


def test_krx_ticker_fallback_is_resolved_to_company_name(monkeypatch):
    monkeypatch.setattr(
        index,
        "get_krx_code_map",
        lambda: ({}, {"000810": "삼성화재", "035250": "강원랜드"}),
    )

    assert index._resolve_peer_display_name("000810.KS", "000810.KS", "KRX") == "삼성화재"
    assert index._resolve_peer_display_name("035250.KS", "035250", "KRX") == "강원랜드"
    assert index._is_ticker_display_name("000810.KS", "000810.KS") is True
    assert index._is_ticker_display_name("삼성화재", "000810.KS") is False


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
        lambda *args, **kwargs: ("Test Industry", [("TEST", "Selected"), ("AAA", "Peer A"), ("BBB", "Peer B")], "static"),
    )
    monkeypatch.setattr(index.yf, "download", lambda *args, **kwargs: raw)
    monkeypatch.setattr(index, "_fetch_peer_fundamentals_batch", lambda tickers, market, timeout=12.0: {
        "TEST": {"available": True, "per": 30.0, "pbr": 5.0, "roe": 25.0, "revenue_growth": 12.0,
                 "operating_margin": 20.0, "market_cap": 3e9},
        "AAA": {"available": True, "per": 20.0, "pbr": 3.0, "roe": 15.0, "revenue_growth": 5.0,
                "operating_margin": 12.0, "market_cap": 5e9},
        "BBB": {"available": True, "per": -4.0, "pbr": 1.0, "roe": -2.0, "revenue_growth": None,
                "operating_margin": 4.0, "market_cap": 1e9},
    })
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
    # 음수 PER(적자)은 중앙값에서 제외하고, 결측 성장률은 임의로 채우지 않는다.
    assert result["peer_medians"]["per"] == {"median": 20.0, "count": 1}
    assert result["peer_medians"]["revenue_growth"] == {"median": 5.0, "count": 1}
    per_row = next(row for row in result["relative_valuation"] if row["key"] == "per")
    assert per_row["status"] == "higher"
    assert result["market_cap_rank"] == {"rank": 2, "total": 3}
    assert "확률 아님" in result["score_label"]


def test_outlook_never_returns_ticker_as_peer_display_name(monkeypatch):
    symbols = ["TEST.KS", "000810.KS", "035250.KS"]
    dates = pd.date_range("2026-01-01", periods=45, freq="B")
    raw = pd.DataFrame(
        np.column_stack([np.linspace(100, 110, len(dates)) for _ in symbols]),
        index=dates,
        columns=pd.MultiIndex.from_product([["Close"], symbols]),
    )
    monkeypatch.setattr(
        index, "_resolve_peer_group",
        lambda *args, **kwargs: ("서비스", [(ticker, ticker) for ticker in symbols], "static"),
    )
    monkeypatch.setattr(
        index, "_resolve_peer_display_name",
        lambda ticker, supplied, market: {"000810.KS": "삼성화재", "035250.KS": "강원랜드"}.get(ticker, ""),
    )
    monkeypatch.setattr(index.yf, "download", lambda *args, **kwargs: raw)
    monkeypatch.setattr(index, "_fetch_peer_fundamentals_batch", lambda tickers, market, timeout=12.0: {})
    cache_key = "build_peer_industry_outlook|('TEST.KS', 'KRX', '테스트', '서비스', '서비스')|[]"
    index._CACHE.pop(cache_key, None)

    result = index.build_peer_industry_outlook("TEST.KS", "KRX", "테스트", "서비스", "서비스")

    assert result["ok"] is True
    assert [peer["name"] for peer in result["peers"]] == ["삼성화재", "강원랜드"]
    assert all(peer["name"] != peer["ticker"] for peer in result["peers"])


def test_krx_outlook_prefers_naver_industry_peers(monkeypatch):
    symbols = ["123450.KS", "000660.KS", "042700.KS"]
    dates = pd.date_range("2026-01-01", periods=45, freq="B")
    raw = pd.DataFrame(
        np.column_stack([np.linspace(100, 110 + i, len(dates)) for i in range(len(symbols))]),
        index=dates,
        columns=pd.MultiIndex.from_product([["Close"], symbols]),
    )
    monkeypatch.setattr(index, "_naver_mobile_fundamentals_cached", lambda code, include_finance=True: {
        "ok": True, "industry_name": "반도체와반도체장비",
        "industry_peers": [{"ticker": "000660.KS", "name": "SK하이닉스"}, {"ticker": "042700.KS", "name": "한미반도체"}],
    })
    monkeypatch.setattr(index, "_resolve_peer_group", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fallback used")))
    monkeypatch.setattr(index.yf, "download", lambda *args, **kwargs: raw)
    monkeypatch.setattr(index, "_fetch_peer_fundamentals_batch", lambda tickers, market, timeout=12.0: {})
    index._CACHE.pop("build_peer_industry_outlook|('123450.KS', 'KRX', '테스트반도체', '', '')|[]", None)

    result = index.build_peer_industry_outlook("123450.KS", "KRX", "테스트반도체", "", "")

    assert result["ok"] is True
    assert result["group_source"] == "naver_industry"
    assert result["group_name"] == "반도체와반도체장비"
    assert [peer["name"] for peer in result["peers"]]
    assert result["fundamentals_available"] is False
    assert any("밸류에이션" in reason for reason in result["reasons"])


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
