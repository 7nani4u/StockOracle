"""Deterministic news selection shared by collectors, diagnosis and the UI.

Render ``feed`` (including untranslated originals), but pass only ``evidence``
to diagnosis/event-risk consumers. Source-specific feeds do not prove relevance.
No network calls or translation are performed here.
"""

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
import math
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def _text(value):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(str(value or "")))).strip()


def _published(value, market, source_type):
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        if isinstance(value, (int, float)) or re.fullmatch(r"\d{10,13}(?:\.\d+)?", str(value)):
            stamp = float(value)
            if not math.isfinite(stamp):
                return None
            result = datetime.fromtimestamp(stamp / 1000 if stamp > 1e11 else stamp, timezone.utc)
        elif isinstance(value, datetime):
            result = value
        else:
            text = str(value).strip()
            text = re.sub(r"^(\d{4})\.(\d{2})\.(\d{2})\.?", r"\1-\2-\3", text)
            if re.fullmatch(r"\d{8}", text):
                text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
            try:
                result = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                result = parsedate_to_datetime(text)
        if result.tzinfo is None:
            # Naver/DART timestamps are local KST, not UTC.
            result = result.replace(tzinfo=timezone(timedelta(hours=9)) if (
                market == "KRX" or source_type in {"naver", "dart", "disclosure"}
            ) else timezone.utc)
        return result.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError, OSError, AttributeError):
        return None


def normalize_news_evidence(items, *, symbol, market, company="", aliases=(), now=None,
                            max_age_days=7, limit=40, source_status=None):
    """Return related display feed, recent dated evidence, and exclusion counts.

    ``source_status`` is optional collector-supplied availability metadata; an
    empty response alone is not evidence that a provider failed. Matching uses
    original headline/summary, never a feed's query or provider label. Ambiguous
    short US tickers require $TICKER, (TICKER), or exchange qualification.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    symbol = str(symbol or "").strip().upper()
    market = str(market or "").upper()
    ticker = symbol.split(".")[0] if market == "KRX" else symbol
    names = []
    for name in (company, *aliases):
        name = _text(name)
        name = re.sub(r"\s+(?:incorporated|inc\.?|corporation|corp\.?|co\.?|ltd\.?|plc)\s*$", "", name, flags=re.I).strip(" ,.")
        if len(name) >= 2 and name.casefold() not in {
            ticker.casefold(), symbol.casefold(), "unknown", "n/a", "none", "technology", "company",
        }:
            names.append(name)
    excluded = dict.fromkeys(("invalid", "unrelated", "duplicate", "future", "stale", "undated"), 0)
    candidates = []
    input_count = 0
    for raw in items or []:
        input_count += 1
        if not isinstance(raw, dict):
            excluded["invalid"] += 1
            continue
        item = dict(raw)
        title = _text(item.get("original_title") or item.get("title") or item.get("headline"))
        if not title:
            excluded["invalid"] += 1
            continue
        content = title + " " + _text(item.get("original_summary") or item.get("summary"))
        matched = ""
        for name in names:
            # Korean particles may follow company names; Latin names need token boundaries.
            pattern = re.escape(name)
            if re.search(r"[\uac00-\ud7a3]", name):
                pattern = pattern.replace(r"\ ", r"\s*")
            else:
                pattern = r"(?<!\w)" + pattern + r"(?!\w)"
            if re.search(pattern, content, re.I):
                matched = "company"
                break
        if not matched and ticker:
            pattern = r"(?<!\w)" + re.escape(ticker) + r"(?!\w)"
            if market != "KRX" and (len(ticker) <= 2 or ticker in {"ALL", "ARE", "FOR", "NOW", "ON", "IT", "CAT"}):
                pattern = r"(?:\$" + re.escape(ticker) + r"\b|\(" + re.escape(ticker) + r"\)|(?:NASDAQ|NYSE)\s*:\s*" + re.escape(ticker) + r"\b)"
            if re.search(pattern, content, re.I if market == "KRX" else 0):
                matched = "ticker"
        if not matched:
            excluded["unrelated"] += 1
            continue
        source_type = str(item.get("source_type") or "news")
        published = None
        for key in ("published_at", "published", "datetime", "date", "pubDate"):
            published = _published(item.get(key), market, source_type)
            if published is not None:
                break
        age_hours = (now - published).total_seconds() / 3600 if published else None
        if age_hours is not None and age_hours < 0:
            excluded["future"] += 1
            continue
        reason = "undated" if published is None else "stale" if age_hours > max_age_days * 24 else None
        link = str(item.get("link") or item.get("url") or "").strip()
        try:
            parsed = urlsplit(link)
            if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username:
                link = ""
            else:
                query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                         if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
                link = urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), urlencode(sorted(query)), ""))
        except ValueError:
            link = ""
        iso = published.isoformat().replace("+00:00", "Z") if published else None
        item.update(title=title, original_title=title, link=link, published=iso, published_at=iso,
                    age_hours=round(age_hours, 2) if age_hours is not None else None,
                    source_type=source_type, relatedness=matched, evidence_eligible=reason is None,
                    exclusion_reason=reason)
        candidates.append(item)
    # Prefer a dated, eligible copy over an undated/stale syndicated duplicate.
    candidates.sort(key=lambda n: (n["evidence_eligible"], n["published_at"] or ""), reverse=True)
    feed, titles, links = [], set(), set()
    for item in candidates:
        key = re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", item["title"]).casefold())
        if key in titles or (item["link"] and item["link"] in links):
            excluded["duplicate"] += 1
            continue
        titles.add(key)
        if item["link"]:
            links.add(item["link"])
        if item["exclusion_reason"]:
            excluded[item["exclusion_reason"]] += 1
        feed.append(item)
    feed = feed[:max(0, int(limit))]
    evidence = [item for item in feed if item["evidence_eligible"]]
    return {"feed": feed, "evidence": evidence, "status": {
        "state": "available" if evidence else "no_recent_related_news",
        "symbol": symbol, "input_count": input_count, "feed_count": len(feed),
        "evidence_count": len(evidence), "excluded": excluded,
        "max_age_days": max_age_days, "as_of": now.isoformat().replace("+00:00", "Z"),
        "sources": dict(source_status or {}), "language_filter_applied": False,
    }}
