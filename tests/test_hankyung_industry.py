"""한국경제 코스피 업종 및 상위 종목 정규화 테스트."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from api import index
from market_briefing import hankyung_industry as provider


def _reset_provider_cache(monkeypatch):
    monkeypatch.setattr(provider, "_SUMMARY_CACHE", (None, 0.0))
    monkeypatch.setattr(provider, "_TOP_STOCKS_CACHE", {})


def test_industry_summary_uses_industry_rate_not_average_of_top_stocks(monkeypatch):
    _reset_provider_cache(monkeypatch)
    monkeypatch.setattr(provider, "_authorized_get_json", lambda path: [{
        "hname": "제조",
        "upcode": "1027",
        "index_trader_delay": {"chgrate": "1.234", "timestamp": "2026-07-19 15:30:00"},
        "index_updown": {"totcount": "534"},
    }])

    result = provider.fetch_kospi_industry_summary(force=True)
    sector = result["sectors"][0]

    assert result["source"] == "한국경제 코스피 업종등락"
    assert sector["name"] == "제조"
    assert sector["avg_change_pct"] == 1.23
    assert sector["member_count"] == 534
    assert sector["stock_names"] == []
    assert sector["upcode"] == "1027"


def test_constituents_are_sorted_by_change_rate_and_limited_to_two(monkeypatch):
    _reset_provider_cache(monkeypatch)
    rows = [
        {"shcode": "000001", "shname": "보합", "stock_trader": {"chgrate": 0}},
        {"shcode": "000002", "shname": "두번째", "stock_trader": {"chgrate": 12.3}},
        {"shcode": "000003", "shname": "첫번째", "stock_trader": {"chgrate": 29.9}},
        {"shcode": "000004", "shname": "데이터없음", "stock_trader": {}},
    ]
    monkeypatch.setattr(provider, "_authorized_get_json", lambda path: rows)

    result = provider.fetch_industry_top_stocks("1027", limit=2, force=True)

    assert [(stock["name"], stock["change_pct"]) for stock in result["stocks"]] == [
        ("첫번째", 29.9),
        ("두번째", 12.3),
    ]


def test_public_bundle_token_extraction():
    script = 'config.headers.common.Authorization="Bearer ".concat("public-token-12345678901234567890")'
    assert provider._extract_public_token(script) == "public-token-12345678901234567890"


def test_sector_cards_lazy_load_top_two_stocks_on_expand():
    html = index.HTML

    assert "/api/market/sector-top-stocks?" in html
    assert "async function toggleSectorCard(el)" in html
    assert "data-loaded=" in html
    assert "상위 2종목" in html
    assert "(data.stocks || []).slice(0, 2)" in html


def test_all_static_fallback_candidates_are_kospi():
    assert all(item["market"] == "KOSPI" for item in index._SECTOR_DEFAULT_STOCKS)

