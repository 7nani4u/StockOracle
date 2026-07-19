"""한국경제 코스피 업종등락 데이터를 업종 카드용으로 정규화한다.

공개 웹 화면이 사용하는 API를 호출하되, 화면 번들에서 현재 공개 Bearer 토큰을
찾아 사용한다. 토큰과 결과는 메모리 캐시해 외부 호출 수를 제한한다.
"""
from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse

import requests

from .labels import SECTOR_EMOJI


_BASE_URL = "https://markets.hankyung.com"
_INDUSTRY_PAGE = f"{_BASE_URL}/index-info/industry"
_TIMEOUT = (4, 15)
_TOKEN_TTL = 6 * 60 * 60
_SUMMARY_TTL = 5 * 60
_STOCKS_TTL = 3 * 60
_KST = timezone(timedelta(hours=9))

_LOCK = threading.RLock()
_TOKEN_CACHE: tuple[str | None, float] = (None, 0.0)
_SUMMARY_CACHE: tuple[dict | None, float] = (None, 0.0)
_TOP_STOCKS_CACHE: dict[str, tuple[dict, float]] = {}

_SESSION = requests.Session()
_SESSION.headers.update({
    "Accept": "application/json, text/plain, */*",
    "Referer": _INDUSTRY_PAGE,
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138 Safari/537.36"
    ),
})


def _as_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _emoji(name: str) -> str:
    return SECTOR_EMOJI.get(name) or next(
        (icon for key, icon in SECTOR_EMOJI.items() if key in name),
        "🏭",
    )


def _extract_public_token(script: str) -> str | None:
    patterns = (
        r'Authorization\s*=\s*"Bearer "\.concat\("([^"]{20,})"\)',
        r'Bearer\s+([A-Za-z0-9._~-]{20,})',
    )
    for pattern in patterns:
        match = re.search(pattern, script)
        if match:
            return match.group(1)
    return None


def _discover_public_token(force: bool = False) -> str:
    global _TOKEN_CACHE
    now = time.monotonic()
    with _LOCK:
        if not force and _TOKEN_CACHE[0] and _TOKEN_CACHE[1] > now:
            return _TOKEN_CACHE[0]

    page = _SESSION.get(_INDUSTRY_PAGE, timeout=_TIMEOUT)
    page.raise_for_status()
    sources = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', page.text, re.I)
    # Nuxt 공통 번들은 보통 문서 끝쪽에 있으므로 역순으로 확인한다.
    nuxt_urls = []
    for source in reversed(sources):
        url = urljoin(_BASE_URL, source)
        parsed = urlparse(url)
        if parsed.scheme == "https" and parsed.netloc == "markets.hankyung.com" and "/_nuxt/" in parsed.path:
            nuxt_urls.append(url)

    for url in nuxt_urls:
        response = _SESSION.get(url, timeout=_TIMEOUT)
        response.raise_for_status()
        token = _extract_public_token(response.text)
        if token:
            with _LOCK:
                _TOKEN_CACHE = (token, now + _TOKEN_TTL)
            return token
    raise RuntimeError("한국경제 공개 데이터 인증 정보를 찾지 못했습니다.")


def _authorized_get_json(path: str):
    token = _discover_public_token()
    url = urljoin(_BASE_URL, path)
    response = _SESSION.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT)
    if response.status_code in (401, 403):
        token = _discover_public_token(force=True)
        response = _SESSION.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_kospi_industry_summary(force: bool = False) -> dict:
    """코스피 업종 목록과 업종 자체 등락률을 반환한다."""
    global _SUMMARY_CACHE
    now = time.monotonic()
    with _LOCK:
        cached, expires = _SUMMARY_CACHE
        if not force and cached is not None and expires > now:
            return cached

    rows = _authorized_get_json("/api/v2/index/1/industries")
    sectors = []
    latest_timestamp = ""
    for row in rows if isinstance(rows, list) else []:
        name = str(row.get("hname") or "").strip()
        upcode = str(row.get("upcode") or "").strip()
        if not name or not upcode:
            continue
        trader = row.get("index_trader_delay") or row.get("index_trader") or {}
        pct = _as_float(trader.get("chgrate"))
        timestamp = str(trader.get("timestamp") or row.get("timestamp") or "")
        latest_timestamp = max(latest_timestamp, timestamp)
        direction = "up" if pct is not None and pct > 0 else "down" if pct is not None and pct < 0 else "flat"
        sectors.append({
            "name": name,
            "upcode": upcode,
            "emoji": _emoji(name),
            "stock_count": 2,
            "member_count": int((row.get("index_updown") or {}).get("totcount") or 0),
            "stock_names": [],
            "top_stocks": [],
            "price_direction": direction,
            "avg_change_pct": round(pct, 2) if pct is not None else None,
            "overnight": "neutral",
            "mood": "positive" if direction == "up" else "negative" if direction == "down" else "neutral",
            "top_news": [],
        })

    if not sectors:
        raise RuntimeError("한국경제 코스피 업종 데이터가 비어 있습니다.")
    result = {
        "generated_at": latest_timestamp or datetime.now(_KST).isoformat(),
        "source": "한국경제 코스피 업종등락",
        "source_url": _INDUSTRY_PAGE,
        "sectors": sectors,
    }
    with _LOCK:
        _SUMMARY_CACHE = (result, now + _SUMMARY_TTL)
    return result


def fetch_industry_top_stocks(upcode: str, limit: int = 2, force: bool = False) -> dict:
    """한 코스피 업종의 구성종목을 등락률 내림차순으로 정렬해 반환한다."""
    upcode = str(upcode or "").strip()
    if not re.fullmatch(r"\d{3,6}", upcode):
        raise ValueError("유효하지 않은 코스피 업종 코드입니다.")
    limit = max(1, min(int(limit), 10))
    now = time.monotonic()
    with _LOCK:
        cached = _TOP_STOCKS_CACHE.get(upcode)
        if not force and cached and cached[1] > now:
            return {**cached[0], "stocks": cached[0]["stocks"][:limit]}

    rows = _authorized_get_json(f"/api/v2/index/{upcode}/stocks")
    ranked = []
    latest_timestamp = ""
    for row in rows if isinstance(rows, list) else []:
        trader = row.get("stock_trader_delay") or row.get("stock_trader") or {}
        pct = _as_float(trader.get("chgrate"))
        name = str(row.get("shname") or "").strip()
        code = str(row.get("shcode") or "").strip()
        if pct is None or not name or not re.fullmatch(r"\d{6}", code):
            continue
        timestamp = str(trader.get("timestamp") or "")
        latest_timestamp = max(latest_timestamp, timestamp)
        ranked.append({
            "code": code,
            "name": name,
            "change_pct": round(pct, 2),
            "price": _as_float(trader.get("curprc")),
            "direction": "up" if pct > 0 else "down" if pct < 0 else "flat",
        })
    ranked.sort(key=lambda item: (-item["change_pct"], item["name"]))
    if not ranked:
        raise RuntimeError("해당 업종의 종목 등락률 데이터가 비어 있습니다.")

    result = {
        "upcode": upcode,
        "as_of": latest_timestamp or datetime.now(_KST).isoformat(),
        "source": "한국경제 코스피 업종등락",
        "source_url": _INDUSTRY_PAGE,
        "stocks": ranked,
    }
    with _LOCK:
        _TOP_STOCKS_CACHE[upcode] = (result, now + _STOCKS_TTL)
    return {**result, "stocks": ranked[:limit]}

