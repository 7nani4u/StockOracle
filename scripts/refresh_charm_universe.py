#!/usr/bin/env python
"""Build the deployed Smart Score comparison-universe snapshot from live fundamentals."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yfinance as yf

from api.index import KR_STOCK_MAP, US_TICKERS
from market_briefing.investment_charm import compute_charm_scores


def _has_sufficient_coverage(records: list[Dict[str, Any]], catalog_count: int) -> bool:
    """Reject provider outages instead of publishing a thin comparison universe."""
    return len(records) >= max(10, int(catalog_count * 0.75 + 0.999))


def _record(market: str, ticker: str) -> Dict[str, Any] | None:
    try:
        info = yf.Ticker(ticker).info or {}
        charm = compute_charm_scores(info, {}, market)
        if charm.get("smart_score") is None or int(charm.get("available_count") or 0) < 3:
            return None
        industry = str(info.get("industry") or "").strip()
        return {
            "ticker": ticker,
            "smart_score": charm["smart_score"],
            "available_count": charm["available_count"],
            "sub_scores": charm.get("sub_scores") or {},
            "industry": industry,
            "sector": str(info.get("sector") or "").strip(),
            "source": "Yahoo Finance Ticker.info",
        }
    except Exception:
        return None


def _collect(market: str, tickers: Iterable[str], workers: int) -> tuple[list[Dict[str, Any]], int]:
    unique = list(dict.fromkeys(tickers))
    records: list[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_record, market, ticker): ticker for ticker in unique}
        for future in as_completed(futures):
            try:
                item = future.result()
            except Exception:
                item = None
            if item:
                records.append(item)
    return sorted(records, key=lambda item: item["ticker"]), len(unique)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "smart_score_universe.json")
    args = parser.parse_args()

    kr_records, kr_catalog_count = _collect("KRX", KR_STOCK_MAP.values(), max(1, args.workers))
    us_records, us_catalog_count = _collect("US", US_TICKERS, max(1, args.workers))
    coverage = {
        "KRX": (kr_records, kr_catalog_count),
        "US": (us_records, us_catalog_count),
    }
    failed = [
        f"{market} {len(records)}/{catalog_count}"
        for market, (records, catalog_count) in coverage.items()
        if not _has_sufficient_coverage(records, catalog_count)
    ]
    if failed:
        print("refusing to replace snapshot: insufficient coverage (" + ", ".join(failed) + ")")
        return 1
    snapshot = {
        "schema_version": 1,
        "universe_id": "smart-score-coverage-v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "score_model": "investment_charm.v1",
        "markets": {
            "KRX": {
                "coverage_label": "KRX 주요 상장사 재무 비교 유니버스",
                "catalog_count": kr_catalog_count,
                "records": kr_records,
            },
            "US": {
                "coverage_label": "US 주요 상장사 재무 비교 유니버스",
                "catalog_count": us_catalog_count,
                "records": us_records,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: KRX {len(kr_records)}/{kr_catalog_count}, US {len(us_records)}/{us_catalog_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
