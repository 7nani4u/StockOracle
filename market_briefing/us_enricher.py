"""🇺🇸 US 주식 보강 데이터 — Finnhub / Tiingo / Alpha Vantage.

API 우선순위:
  1. Alpha Vantage  — 회사 개요(섹터, PER, PBR, EPS, 베타, 시총), 분기 실적
  2. Finnhub        — 최근 뉴스(7일), 뉴스 감성(불/베어 비율)
  3. Tiingo         — EOD 가격 보조 (AV/yfinance 없을 때만)
"""
from __future__ import annotations

import datetime
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

FINNHUB_KEY = "d7lm0o9r01qm7o0cb440d7lm0o9r01qm7o0cb44g"
TIINGO_KEY  = "12ebd1feef89b6728cc15808864b7402449a5637"
AV_KEY      = "E0ODFSRNDU4P9HDU"

_TIMEOUT = 6  # seconds per request
_US_ENRICH_CACHE: dict[str, tuple[dict, float]] = {}
_US_ENRICH_TTL = 300.0


def _get(url: str, headers: dict | None = None) -> Any:
    try:
        r = requests.get(url, headers=headers or {}, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


_HANGUL_RE = re.compile(r"[가-힣]")
_NEWS_ITEM_BREAK = "<<<SO_ITEM_BREAK>>>"
_NEWS_FIELD_BREAK = "<<<SO_FIELD_BREAK>>>"


def _translate_news_batch_ko(items: list[dict]) -> list[dict]:
    """영문 뉴스 제목·요약을 한 번의 경량 요청으로 한국어 표시문으로 만든다.

    번역 실패는 뉴스 수집 자체를 중단시키지 않는다. 결과는 상위
    ``fetch_us_enriched`` 5분 캐시에 함께 저장되므로 탭 전환이나 재렌더링 때
    번역 요청을 반복하지 않는다.
    """
    targets: list[tuple[int, str, str]] = []
    for index, item in enumerate(items):
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if title and not _HANGUL_RE.search(title):
            targets.append((index, title, summary))
        elif title:
            item["title_ko"] = title
            if summary and _HANGUL_RE.search(summary):
                item["summary_ko"] = summary
    if not targets:
        return items

    payload = f"\n{_NEWS_ITEM_BREAK}\n".join(
        f"{title} {_NEWS_FIELD_BREAK} {summary}" for _, title, summary in targets
    )
    try:
        response = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "auto", "tl": "ko", "dt": "t", "q": payload},
            timeout=4,
        )
        response.raise_for_status()
        data = response.json()
        translated = "".join(
            str(part[0]) for part in (data[0] if isinstance(data, list) and data else [])
            if isinstance(part, list) and part
        )
        chunks = re.split(re.escape(_NEWS_ITEM_BREAK), translated, flags=re.IGNORECASE)
        if len(chunks) != len(targets):
            return items
        for (index, _, _), chunk in zip(targets, chunks):
            fields = re.split(
                re.escape(_NEWS_FIELD_BREAK),
                chunk.strip(),
                maxsplit=1,
                flags=re.IGNORECASE,
            )
            title_ko = fields[0].strip()
            summary_ko = fields[1].strip() if len(fields) > 1 else ""
            if title_ko and _HANGUL_RE.search(title_ko):
                items[index]["title_ko"] = title_ko
                items[index]["translation_source"] = "자동 번역"
            if summary_ko and _HANGUL_RE.search(summary_ko):
                items[index]["summary_ko"] = summary_ko[:260]
    except Exception:
        pass
    return items


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
            published_at = datetime.datetime.fromtimestamp(
                ts, tz=datetime.timezone.utc
            ).isoformat().replace("+00:00", "Z")
            date_str = datetime.datetime.fromtimestamp(
                ts, tz=datetime.timezone.utc
            ).strftime("%Y.%m.%d %H:%M")
        except Exception:
            published_at = ""
            date_str = ""
        out.append({
            "title":   item.get("headline", ""),
            "original_title": item.get("headline", ""),
            "link":    item.get("url", ""),
            "source":  item.get("source", ""),
            "date":    date_str,
            "published_at": published_at,
            "summary": (item.get("summary") or "")[:200],
            "image_url": item.get("image") or "",
            "source_type": "finnhub",
        })
    return _translate_news_batch_ko(out)


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
    symbol = (symbol or "").upper().strip()
    now = time.time()
    cached = _US_ENRICH_CACHE.get(symbol)
    if cached and now - cached[1] < _US_ENRICH_TTL:
        return cached[0]

    overview: dict = {}
    news: list[dict] = []
    sentiment: dict = {}
    earnings: list[dict] = []
    tiingo: dict = {}

    try:
        with ThreadPoolExecutor(max_workers=3) as ex:
            fut_overview = ex.submit(fetch_alpha_overview, symbol)
            fut_news = ex.submit(fetch_finnhub_news, symbol)
            fut_sentiment = ex.submit(fetch_finnhub_sentiment, symbol)
            overview = fut_overview.result(timeout=_TIMEOUT + 1) or {}
            news = fut_news.result(timeout=_TIMEOUT + 1) or []
            sentiment = fut_sentiment.result(timeout=_TIMEOUT + 1) or {}
    except Exception:
        overview = overview or {}
        news = news or []
        sentiment = sentiment or {}

    followups = []
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            if overview.get("sector"):
                followups.append(("earnings", ex.submit(fetch_alpha_earnings, symbol)))
            # Tiingo는 가격 보조용이라 분석 정확도 핵심이 아니다. AV 개요가 비어 있을 때만 짧게 시도한다.
            if not overview.get("per") and not overview.get("sector"):
                followups.append(("tiingo", ex.submit(fetch_tiingo_price, symbol)))
            for key, fut in followups:
                try:
                    if key == "earnings":
                        earnings = fut.result(timeout=_TIMEOUT + 1) or []
                    elif key == "tiingo":
                        tiingo = fut.result(timeout=3) or {}
                except Exception:
                    pass
    except Exception:
        pass

    result = {
        "news":      news,
        "sentiment": sentiment,
        "overview":  overview,
        "earnings":  earnings,
        "tiingo":    tiingo,
    }
    _US_ENRICH_CACHE[symbol] = (result, now)
    if len(_US_ENRICH_CACHE) > 200:
        for k, (_, ts) in list(_US_ENRICH_CACHE.items()):
            if now - ts > _US_ENRICH_TTL:
                _US_ENRICH_CACHE.pop(k, None)
    return result
