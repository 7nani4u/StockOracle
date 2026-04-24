"""🏭 섹터 흐름 — 산업/섹터별 방향성 분석 모듈.

출처: k-ant-daily/scripts/render.py (_normalize_sectors, _portfolio_snapshot)
     + labels.py (SECTOR_EMOJI) 에서 핵심 로직 추출.

입력:  종목 스냅샷 리스트 (data_fetcher.fetch_stock_list_snapshot() 반환값)
       각 종목에 "sector" 필드가 있어야 함

출력:
  build_sector_flow() → 섹터별 집계 + 방향성 + 뉴스 요약 dict
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from .labels import IMPACT_LABEL, SECTOR_EMOJI

KST = timezone(timedelta(hours=9))

# ── 헤드라인 감성 분류 (core_summary 와 동일 패턴, 섹터 맥락으로 재사용) ────

_POS = re.compile(
    r"(상승|급등|강세|반등|신고가|돌파|호실적|영업이익|수주|승인|허가|"
    r"배당|흑자|수혜|성장|호재|랠리|최대\s*실적|낙관)"
)
_NEG = re.compile(
    r"(하락|급락|약세|부진|적자|감소|제재|과징금|소송|경고|우려|손실|위기|"
    r"부담|둔화|논란|규제|조사|리스크)"
)


def _auto_impact(title: str) -> str:
    pos = bool(_POS.search(title))
    neg = bool(_NEG.search(title))
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return "neutral"


def _annotate(obj: dict) -> None:
    impact = obj.get("impact")
    if impact and "impact_label" not in obj:
        obj["impact_label"] = IMPACT_LABEL.get(impact, impact)


# ── 오버나이트 신호 집계 ─────────────────────────────────────────────────────

def _aggregate_overnight(stocks: list[dict]) -> str:
    """섹터 내 종목들의 overnight_signal direction → 섹터 오버나이트 방향."""
    counts = {"up": 0, "down": 0, "neutral": 0}
    for s in stocks:
        sig = s.get("overnight_signal") or {}
        d   = sig.get("direction") if isinstance(sig, dict) else str(sig)
        if d in counts:
            counts[d] += 1
    total = sum(counts.values())
    if not total:
        return "neutral"
    if counts["up"] / total > 0.6:
        return "up"
    if counts["down"] / total > 0.6:
        return "down"
    return "neutral"


# ── 종목 등락 집계 ───────────────────────────────────────────────────────────

def _aggregate_price_direction(stocks: list[dict]) -> str:
    """종목 quote 에서 오늘의 등락 방향 집계."""
    ups = downs = 0
    for s in stocks:
        d = (s.get("quote") or {}).get("direction", "")
        if d == "up":
            ups += 1
        elif d == "down":
            downs += 1
    if not ups and not downs:
        return "flat"
    if ups > downs:
        return "up"
    if downs > ups:
        return "down"
    return "flat"


def _average_change_pct(stocks: list[dict]) -> float | None:
    """섹터 평균 등락률 (quote.change_pct_num 기반)."""
    vals = []
    for s in stocks:
        pct = (s.get("quote") or {}).get("change_pct_num")
        if pct is not None:
            try:
                vals.append(float(pct))
            except (TypeError, ValueError):
                pass
    return round(sum(vals) / len(vals), 2) if vals else None


# ── 섹터 뉴스 수집 ───────────────────────────────────────────────────────────

def _collect_sector_news(stocks: list[dict], limit: int = 5) -> list[dict]:
    """섹터 내 모든 종목 뉴스를 모아 중복 제거 후 최신순 상위 N개 반환."""
    seen_titles: set[str] = set()
    all_news: list[dict] = []
    for s in stocks:
        for n in (s.get("news") or []):
            title = n.get("title", "")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            item = dict(n)
            item["stock_name"] = s.get("name", "")
            item["stock_code"] = s.get("code", "")
            if not item.get("impact"):
                item["impact"] = _auto_impact(title)
            _annotate(item)
            all_news.append(item)
    # 최신순 정렬 (date 문자열은 "YYYY.MM.DD HH:MM" 형식)
    all_news.sort(key=lambda n: n.get("date") or n.get("published_at") or "", reverse=True)
    return all_news[:limit]


# ── 공개 API ─────────────────────────────────────────────────────────────────

def build_sector_flow(stock_snapshots: list[dict]) -> dict:
    """🏭 섹터 흐름 빌드.

    Args:
        stock_snapshots: fetch_stock_list_snapshot() 반환값.
            각 항목에 "sector" 키가 있어야 함. 없으면 "기타" 버킷.

    Returns:
        {
          "generated_at": str,
          "sectors": [
            {
              "name":              str,       # 섹터명
              "emoji":             str,       # 이모지
              "stock_count":       int,
              "stock_names":       list[str],
              "price_direction":   "up"|"down"|"flat",
              "avg_change_pct":    float|None,
              "overnight":         "up"|"down"|"neutral",
              "mood":              "positive"|"neutral"|"negative",
              "top_news":          list[dict],
            },
            ...
          ],
          "portfolio_snapshot": {
            "total":    int,
            "up":       int,
            "down":     int,
            "flat":     int,
            "by_sector": dict[str, {"up":int,"down":int,"flat":int}]
          }
        }
    """
    # 섹터별 종목 그루핑
    by_sector: dict[str, list[dict]] = defaultdict(list)
    for s in stock_snapshots:
        sector = s.get("sector") or "기타"
        by_sector[sector].append(s)

    sectors_out = []
    for sector_name, stocks in sorted(by_sector.items()):
        price_dir   = _aggregate_price_direction(stocks)
        overnight   = _aggregate_overnight(stocks)
        avg_pct     = _average_change_pct(stocks)
        top_news    = _collect_sector_news(stocks)

        # 무드: 가격 방향 + 오버나이트 합산
        pos_signals = sum([
            1 if price_dir == "up" else 0,
            1 if overnight == "up" else 0,
        ])
        neg_signals = sum([
            1 if price_dir == "down" else 0,
            1 if overnight == "down" else 0,
        ])
        if pos_signals > neg_signals:
            mood = "positive"
        elif neg_signals > pos_signals:
            mood = "negative"
        else:
            mood = "neutral"

        # 이모지 폴백
        emoji = next(
            (emj for key, emj in SECTOR_EMOJI.items() if key in sector_name),
            "🏭"
        )

        sectors_out.append({
            "name":            sector_name,
            "emoji":           emoji,
            "stock_count":     len(stocks),
            "stock_names":     [s.get("name", s["code"]) for s in stocks],
            "price_direction": price_dir,
            "avg_change_pct":  avg_pct,
            "overnight":       overnight,
            "mood":            mood,
            "top_news":        top_news,
        })

    # 포트폴리오 스냅샷
    pf_total = pf_up = pf_down = pf_flat = 0
    by_sector_counts: dict[str, dict] = {}
    for s in stock_snapshots:
        pf_total += 1
        d = (s.get("quote") or {}).get("direction", "flat")
        if d == "up":
            pf_up += 1
        elif d == "down":
            pf_down += 1
        else:
            pf_flat += 1
        sec = s.get("sector") or "기타"
        cnt = by_sector_counts.setdefault(sec, {"up": 0, "down": 0, "flat": 0})
        cnt[d if d in cnt else "flat"] += 1

    return {
        "generated_at": datetime.now(KST).isoformat(),
        "sectors":      sectors_out,
        "portfolio_snapshot": {
            "total":     pf_total,
            "up":        pf_up,
            "down":      pf_down,
            "flat":      pf_flat,
            "by_sector": by_sector_counts,
        },
    }
