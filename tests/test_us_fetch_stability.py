"""KR/US 안정화 회귀 테스트 — info캐시, 네거티브캐시, REGULAR 폴백, 슬롯 별칭, 429 재시도."""

import time

import us_price_fetcher as upf
from us_price_fetcher import MarketSession, NaverWorldStockClient, YFinanceClient
from market_briefing.ml_features import _normalize_index_cache, compute_feature_vector


def test_yfinance_info_cached_30s(monkeypatch):
    calls = {"n": 0}

    class FakeTicker:
        def __init__(self, t):
            self.t = t

        @property
        def info(self):
            calls["n"] += 1
            return {"regularMarketPrice": 100.0, "previousClose": 99.0,
                    "regularMarketTime": 9999999999, "currentPrice": 100.0}

    monkeypatch.setattr(upf.yf, "Ticker", FakeTicker, raising=False)
    # _HAS_YFINANCE가 False인 환경 대비 강제
    monkeypatch.setattr(upf, "_HAS_YFINANCE", True)
    c = YFinanceClient()
    c._INFO_TTL = 30.0
    assert c._fetch_info("AAPL") is not None
    assert c._fetch_info("AAPL") is not None
    assert calls["n"] == 1  # 두 번째는 캐시


def test_naver_negative_cache_avoids_reprobing(monkeypatch):
    c = NaverWorldStockClient(cache_ttl=5)
    calls = {"n": 0}

    def fake_fetch(code):
        calls["n"] += 1
        return None

    monkeypatch.setattr(c, "_fetch_basic", fake_fetch)
    assert c._resolve_reuters_code("ZZZZNOPE") is None
    first = calls["n"]
    assert first >= 1
    assert c._resolve_reuters_code("ZZZZNOPE") is None
    assert calls["n"] == first  # 60초 네거티브 캐시로 재탐색 없음


def test_regular_fallback_to_eod(monkeypatch):
    from us_price_fetcher import USStockPriceFetcher
    f = USStockPriceFetcher(finnhub_key="", tiingo_key="", av_key="")
    monkeypatch.setattr(f.naver, "get_price", lambda t, s: None)
    monkeypatch.setattr(f.tiingo, "get_price", lambda t, s: None)
    monkeypatch.setattr(f.yf, "get_price",
                        lambda t, s: (98.0, 97.0, None, "last_close") if s == MarketSession.CLOSED else None)
    monkeypatch.setattr(f.finnhub, "get_price", lambda t, s: None)
    monkeypatch.setattr(f.av, "global_quote", lambda t, s: None)
    monkeypatch.setattr(f.av, "daily_close", lambda t, s: None)
    from datetime import datetime
    from zoneinfo import ZoneInfo
    _et = ZoneInfo("America/New_York")
    monkeypatch.setattr(upf, "detect_session",
                        lambda dt=None: (MarketSession.REGULAR, datetime(2026, 9, 10, 11, 0, tzinfo=_et)))
    try:
        res = f.fetch("AAPL")
    finally:
        import importlib
        importlib.reload(upf)
    assert res is not None
    assert res.price_type == "last_close"


def test_price_route_uses_canonical_us_ticker_not_display_name(monkeypatch):
    from types import SimpleNamespace
    from api import index

    calls = []

    class Fetcher:
        def fetch(self, ticker):
            calls.append(ticker)
            return SimpleNamespace(price=201.25, prev_close=200.0, price_type="regular")

    monkeypatch.setattr(index, "resolve_ticker", lambda _q: ("AAPL", "US", "애플"))
    monkeypatch.setattr(index, "_get_us_price_fetcher", lambda: Fetcher())

    result = index.route("/api/price", {"ticker": "AAPL", "market": "US"})

    assert calls == ["AAPL"]
    assert result["symbol"] == "AAPL"
    assert result["market"] == "US"


def test_index_cache_alias_normalization():
    canon = {"market_return_1d": 0.5, "sector_return_1d": 0.2,
             "volatility_index": 18.0, "market_return_20d": 3.0}
    norm = _normalize_index_cache(canon)
    assert norm["NIFTY_return"] == 0.5
    assert norm["BANKNIFTY_return"] == 0.2
    assert norm["India_VIX"] == 18.0
    assert norm["NIFTY_cum20"] == 3.0
    # legacy도 그대로 통과
    legacy = {"NIFTY_return": 0.1, "BANKNIFTY_return": 0.1, "India_VIX": 15.0, "NIFTY_cum20": 1.0}
    assert _normalize_index_cache(legacy)["NIFTY_return"] == 0.1
    assert _normalize_index_cache(None)["India_VIX"] == 15.0


def test_compute_vector_accepts_canonical_cache():
    n = 70
    closes = [100.0 + i * 0.1 for i in range(n)]
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    vols = [1_000_000.0] * n
    vec = compute_feature_vector(closes, highs, lows, vols, market="KRX",
                                 index_cache={"market_return_1d": 0.3, "sector_return_1d": 0.1,
                                              "volatility_index": 16.0, "market_return_20d": 2.0})
    assert vec is not None
    assert vec["market_return_1d"] == 0.3
    assert vec["volatility_index"] == 16.0


def test_get_with_retry_retries_429(monkeypatch):
    from market_briefing import data_fetcher as df
    calls = {"n": 0}

    class FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}

        def raise_for_status(self):
            return None

    def fake_get(url, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            class R429:
                status_code = 429
            return R429()
        return FakeResp()

    monkeypatch.setattr(df.requests, "get", fake_get)
    r = df._get_with_retry("http://x", {}, timeout=1, retries=2, backoff=0.01)
    assert r.status_code == 200
    assert calls["n"] == 2
