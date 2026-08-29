# -*- coding: utf-8 -*-
"""Read-only ranking for the published Smart Score comparison universe."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


UNIVERSE_ID = "smart-score-coverage-v1"
MIN_OVERALL_POPULATION = 10
MIN_INDUSTRY_POPULATION = 3
MAX_SNAPSHOT_AGE_DAYS = 8
_SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "models" / "smart_score_universe.json"
_SNAPSHOT_CACHE: Optional[Dict[str, Any]] = None
_SNAPSHOT_MTIME_NS: Optional[int] = None


def _safe_score(value: Any) -> Optional[int]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return int(round(max(0, min(100, score))))


def _normalise_ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def _industry_key(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _parse_generated_at(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def load_charm_universe(path: Path = _SNAPSHOT_PATH) -> Optional[Dict[str, Any]]:
    """Load a versioned, offline-generated snapshot without request-time I/O."""
    global _SNAPSHOT_CACHE, _SNAPSHOT_MTIME_NS
    try:
        stat = path.stat()
        if _SNAPSHOT_CACHE is not None and _SNAPSHOT_MTIME_NS == stat.st_mtime_ns:
            return _SNAPSHOT_CACHE
        with path.open("r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("markets"), dict):
            return None
        _SNAPSHOT_CACHE = snapshot
        _SNAPSHOT_MTIME_NS = stat.st_mtime_ns
        return snapshot
    except (OSError, ValueError, TypeError):
        return None


def _rank(score: int, records: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    scores = [item["smart_score"] for item in records if item.get("smart_score") is not None]
    if not scores:
        return None
    rank = 1 + sum(item_score > score for item_score in scores)
    total = len(scores)
    return {"rank": rank, "percentile": round(rank / total * 100, 2), "population": total}


def _eligible_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    eligible: Dict[str, Dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        ticker = _normalise_ticker(item.get("ticker"))
        score = _safe_score(item.get("smart_score"))
        try:
            available_count = int(item.get("available_count") or 0)
        except (TypeError, ValueError):
            available_count = 0
        if ticker and score is not None and available_count >= 3:
            eligible[ticker] = {**item, "ticker": ticker, "smart_score": score}
    return eligible


def enrich_charm_with_ranks(
    charm: Dict[str, Any],
    symbol: str,
    market: str,
    snapshot: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Add disclosed comparison-universe ranks without inventing a full-market rank.

    The selected stock's live score replaces its snapshot score, while all peers retain
    the same snapshot date. The metadata exposes that distinction to the UI.
    """
    result = dict(charm or {})
    result.update({
        "overall_rank": None,
        "overall_rank_str": "N/A",
        "overall_percentile": None,
        "industry_rank": None,
        "industry_percentile": None,
        "universe_size": None,
    })
    score = _safe_score(result.get("smart_score"))
    try:
        available_count = int(result.get("available_count") or 0)
    except (TypeError, ValueError):
        available_count = 0
    if score is None or available_count < 3:
        result["ranking_meta"] = {
            "universe_id": UNIVERSE_ID,
            "status": "target_unscorable",
            "reason": "스마트스코어 계산 축이 3개 미만입니다.",
        }
        return result

    snapshot = snapshot if snapshot is not None else load_charm_universe()
    market = str(market or "").upper()
    if not snapshot:
        result["ranking_meta"] = {
            "universe_id": UNIVERSE_ID,
            "status": "unavailable",
            "reason": "비교 유니버스 스냅샷을 불러오지 못했습니다.",
        }
        return result

    generated_at = _parse_generated_at(snapshot.get("generated_at"))
    reference_now = now or datetime.now(timezone.utc)
    if generated_at is None or (reference_now - generated_at).total_seconds() > MAX_SNAPSHOT_AGE_DAYS * 86400:
        result["ranking_meta"] = {
            "universe_id": snapshot.get("universe_id") or UNIVERSE_ID,
            "status": "stale",
            "generated_at": snapshot.get("generated_at"),
            "reason": "비교 유니버스 기준일이 만료되었습니다.",
        }
        return result

    market_data = (snapshot.get("markets") or {}).get(market) or {}
    records = _eligible_records(market_data.get("records") or [])
    if len(records) < MIN_OVERALL_POPULATION:
        result["ranking_meta"] = {
            "universe_id": snapshot.get("universe_id") or UNIVERSE_ID,
            "status": "insufficient_population",
            "generated_at": snapshot.get("generated_at"),
            "eligible_scored_count": len(records),
            "reason": "순위를 계산할 비교 대상이 부족합니다.",
        }
        return result

    ticker = _normalise_ticker(symbol)
    prior_target = records.get(ticker) or {}
    rank_industry = str(prior_target.get("industry") or result.get("industry") or "").strip()
    rank_industry_key = _industry_key(rank_industry)
    live_target = {
        "ticker": ticker,
        "smart_score": score,
        "available_count": available_count,
        "industry": rank_industry,
        "sub_scores": result.get("sub_scores") or {},
    }
    records[ticker] = live_target
    ranked_records = list(records.values())
    overall = _rank(score, ranked_records)
    if overall:
        result["overall_rank"] = overall["rank"]
        result["overall_rank_str"] = f"{overall['rank']:,}위"
        result["overall_percentile"] = overall["percentile"]
        result["universe_size"] = overall["population"]

    industry_records = [
        item for item in ranked_records
        if rank_industry_key and _industry_key(item.get("industry")) == rank_industry_key
    ]
    if len(industry_records) >= MIN_INDUSTRY_POPULATION:
        industry = _rank(score, industry_records)
        if industry:
            result["industry_rank"] = industry["rank"]
            result["industry_percentile"] = industry["percentile"]

    sub_percentiles: Dict[str, Optional[float]] = {}
    for key, raw_score in (result.get("sub_scores") or {}).items():
        sub_score = _safe_score(raw_score)
        axis_records = []
        if sub_score is not None:
            for item in ranked_records:
                axis_score = _safe_score((item.get("sub_scores") or {}).get(key))
                if axis_score is not None:
                    axis_records.append({"smart_score": axis_score})
            if len(axis_records) >= MIN_OVERALL_POPULATION:
                axis_rank = _rank(sub_score, axis_records)
                sub_percentiles[key] = axis_rank["percentile"] if axis_rank else None
            else:
                sub_percentiles[key] = None
        else:
            sub_percentiles[key] = None
    result["sub_percentiles"] = sub_percentiles
    result["ranking_meta"] = {
        "universe_id": snapshot.get("universe_id") or UNIVERSE_ID,
        "status": "fresh",
        "coverage_label": market_data.get("coverage_label") or f"{market} 재무 비교 유니버스",
        "market": market,
        "generated_at": snapshot.get("generated_at"),
        "eligible_scored_count": len(ranked_records),
        "catalog_count": market_data.get("catalog_count"),
        "industry": rank_industry or None,
        "industry_eligible_count": len(industry_records),
        "target_included_live": True,
        "tie_policy": "competition_rank_on_published_smart_score",
        "score_model": snapshot.get("score_model"),
    }
    return result
