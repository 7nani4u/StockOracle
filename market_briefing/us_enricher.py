"""🇺🇸 US 주식 보강 데이터 — Finnhub / Tiingo / Alpha Vantage.

API 우선순위:
  1. Alpha Vantage  — 회사 개요(섹터, PER, PBR, EPS, 베타, 시총), 분기 실적
  2. Finnhub        — 최근 뉴스(7일), 뉴스 감성(불/베어 비율)
  3. Tiingo         — EOD 가격 보조 (AV/yfinance 없을 때만)
"""
from __future__ import annotations

import datetime
from typing import Any

import requests

FINNHUB_KEY = "d7lm0o9r01qm7o0cb440d7lm0o9r01qm7o0cb44g"
TIINGO_KEY  = "12ebd1feef89b6728cc15808864b7402449a5637"
AV_KEY      = "E0ODFSRNDU4P9HDU"

_TIMEOUT = 6  # seconds per request


def _get(url: str, headers: dict | None = None) -> Any:
    try:
        r = requests.get(url, headers=headers or {}, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── Finnhub ──────────────────────────────────────────────────────────────────

def fetch_finnhub_news(symbol: str, days: int = 7) -> list[dict]:
    """최근 N일 회사 뉴스 (최대 10건)."""
    to_date   = datetime.date.today()
    from_date = to_date - datetime.timedelta(days=days)
    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_KEY}"
    )
    data = _get(url)
    if not isinstance(data, list):
        return []
    out = []
    for item in data[:10]:
        ts = item.get("datetime", 0)
        try:
            date_str = datetime.datetime.utcfromtimestamp(ts).strftime("%Y.%m.%d %H:%M")
        except Exception:
            date_str = ""
        out.append({
            "title":   item.get("headline", ""),
            "link":    item.get("url", ""),
            "source":  item.get("source", ""),
            "date":    date_str,
            "summary": (item.get("summary") or "")[:200],
        })
    return out


def fetch_finnhub_sentiment(symbol: str) -> dict:
    """뉴스 감성 집계 (buzz + bullish/bearish 비율)."""
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}&token={FINNHUB_KEY}"
    data = _get(url)
    if not isinstance(data, dict):
        return {}
    buzz = data.get("buzz") or {}
    sent = data.get("sentiment") or {}
    return {
        "buzz_score":    buzz.get("buzz"),
        "article_count": buzz.get("articlesInLastWeek"),
        "bullish_pct":   sent.get("bullishPercent"),
        "bearish_pct":   sent.get("bearishPercent"),
        "sector_avg":    data.get("sectorAverageBullishPercent"),
    }


# ── Tiingo ────────────────────────────────────────────────────────────────────

def fetch_tiingo_price(symbol: str) -> dict:
    """최신 EOD 가격 데이터 (보조 소스)."""
    headers = {"Authorization": f"Token {TIINGO_KEY}"}
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    data = _get(url, headers=headers)
    if not isinstance(data, list) or not data:
        return {}
    latest = data[-1]
    return {
        "date":      (latest.get("date") or "")[:10],
        "close":     latest.get("close"),
        "open":      latest.get("open"),
        "high":      latest.get("high"),
        "low":       latest.get("low"),
        "volume":    latest.get("volume"),
        "adj_close": latest.get("adjClose"),
    }


# ── Alpha Vantage ─────────────────────────────────────────────────────────────

def fetch_alpha_overview(symbol: str) -> dict:
    """회사 개요: 섹터, 업종, PER, EPS, PBR, 시가총액 등."""
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=OVERVIEW&symbol={symbol}&apikey={AV_KEY}"
    )
    data = _get(url)
    if not isinstance(data, dict) or not data.get("Symbol"):
        return {}
    def _flt(v: Any) -> str | None:
        if v in (None, "None", "-", ""):
            return None
        return str(v)
    mc = data.get("MarketCapitalization")
    return {
        "sector":         _flt(data.get("Sector")),
        "industry":       _flt(data.get("Industry")),
        "market_cap":     mc if mc not in (None, "None", "-") else None,
        "per":            _flt(data.get("PERatio")),
        "pbr":            _flt(data.get("PriceToBookRatio")),
        "eps":            _flt(data.get("EPS")),
        "roe":            _flt(data.get("ReturnOnEquityTTM")),
        "profit_margin":  _flt(data.get("ProfitMargin")),
        "52w_high":       _flt(data.get("52WeekHigh")),
        "52w_low":        _flt(data.get("52WeekLow")),
        "beta":           _flt(data.get("Beta")),
        "description":    (data.get("Description") or "")[:300],
    }


def fetch_alpha_earnings(symbol: str) -> list[dict]:
    """최근 4분기 EPS 실적."""
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=EARNINGS&symbol={symbol}&apikey={AV_KEY}"
    )
    data = _get(url)
    if not isinstance(data, dict):
        return []
    quarterly = (data.get("quarterlyEarnings") or [])[:4]
    return [
        {
            "date":         q.get("fiscalDateEnding", ""),
            "reported":     q.get("reportedEPS"),
            "estimated":    q.get("estimatedEPS"),
            "surprise":     q.get("surprise"),
            "surprise_pct": q.get("surprisePercentage"),
        }
        for q in quarterly
    ]


# ── 통합 ──────────────────────────────────────────────────────────────────────

def fetch_us_enriched(symbol: str) -> dict:
    """US 종목 보강 데이터 통합 반환.

    Returns:
        {
          "news":      list[dict],   # Finnhub 뉴스 (최대 10건)
          "sentiment": dict,         # Finnhub 뉴스 감성
          "overview":  dict,         # Alpha Vantage 회사 개요
          "earnings":  list[dict],   # Alpha Vantage 분기 실적
          "tiingo":    dict,         # Tiingo EOD (보조)
        }
    """
    overview  = fetch_alpha_overview(symbol)
    news      = fetch_finnhub_news(symbol)
    sentiment = fetch_finnhub_sentiment(symbol)
    # Tiingo는 Alpha Vantage 가격 데이터가 없을 때만 호출
    tiingo    = fetch_tiingo_price(symbol) if not overview.get("per") else {}
    earnings  = fetch_alpha_earnings(symbol) if overview.get("sector") else []

    return {
        "news":      news,
        "sentiment": sentiment,
        "overview":  overview,
        "earnings":  earnings,
        "tiingo":    tiingo,
    }
