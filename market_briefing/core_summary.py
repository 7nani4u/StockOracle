"""⭐ 오늘의 핵심 — 거시 시장 요약 모듈.

출처: k-ant-daily/scripts/render.py (_normalize_macro, _normalize_top_stories,
     _normalize_mood_dashboard) 에서 핵심 로직 추출.

입력:  macro_context dict (data_fetcher.fetch_macro_context() 반환값)
출력:  build_core_summary() → 정규화된 시장 요약 dict
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from .labels import IMPACT_LABEL, CATEGORY_LABEL, MOOD_LABEL, MOOD_AXES

KST = timezone(timedelta(hours=9))

# ── 헤드라인 감성 키워드 ─────────────────────────────────────────────────────

_POS = re.compile(
    r"(상승|급등|강세|반등|신고가|돌파|호실적|영업이익\s*증가|수주|승인|허가|"
    r"자사주|배당\s*(인상|증가)|흑자|수혜|성장|호재|랠리|낙관|최대\s*실적)"
)
_NEG = re.compile(
    r"(하락|급락|약세|부진|적자|감소|제재|과징금|소송|경고|우려|손실|위기|"
    r"부담|둔화|논란|규제|조사|리스크)"
)


def _auto_impact(title: str) -> str:
    """헤드라인 제목에서 호재/악재/중립 추론."""
    pos = bool(_POS.search(title))
    neg = bool(_NEG.search(title))
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return "neutral"


def _annotate(obj: dict) -> None:
    """impact → impact_label 주입."""
    impact = obj.get("impact")
    if impact and "impact_label" not in obj:
        obj["impact_label"] = IMPACT_LABEL.get(impact, impact)


# ── 지수 / 지표 정규화 ───────────────────────────────────────────────────────

def _normalize_index_entry(entry: dict) -> dict:
    """단일 지수/지표 항목에 direction 추론 + impact_label 보강."""
    e = dict(entry)
    if "direction" not in e:
        change = e.get("change_pct") or e.get("change_abs") or ""
        s = str(change).strip()
        if s.startswith("-") or "▼" in s or "하락" in s:
            e["direction"] = "down"
        elif s.startswith("+") or "▲" in s or "상승" in s:
            e["direction"] = "up"
        else:
            e["direction"] = "flat"
    _annotate(e)
    return e


def _normalize_indices(indices: dict[str, Any]) -> dict[str, Any]:
    return {k: _normalize_index_entry(v) for k, v in indices.items() if isinstance(v, dict)}


def _normalize_overnight(overnight: list[dict]) -> list[dict]:
    """해외 시장 항목 리스트 정규화."""
    result = []
    for item in overnight:
        e = dict(item)
        if "direction" not in e:
            pct = e.get("change_pct") if isinstance(e.get("change_pct"), (int, float)) else None
            if pct is not None:
                e["direction"] = "up" if pct > 0 else ("down" if pct < 0 else "flat")
        _annotate(e)
        result.append(e)
    return result


# ── 뉴스 정규화 ──────────────────────────────────────────────────────────────

def _normalize_news_item(item: dict) -> dict:
    """개별 뉴스 항목 — impact 자동 추론 + category_label."""
    n = dict(item)
    if not n.get("impact"):
        n["impact"] = _auto_impact(n.get("title", ""))
    _annotate(n)
    if n.get("category") and "category_label" not in n:
        n["category_label"] = CATEGORY_LABEL.get(n["category"], n["category"])
    return n


# ── 시장 무드 판정 ───────────────────────────────────────────────────────────

def _derive_market_mood(indices: dict, overnight: list[dict]) -> str:
    """지수 방향성 + 해외 시장으로 전체 무드 추론.

    반환: "positive" / "neutral" / "negative"
    """
    scores: list[float] = []
    for idx in indices.values():
        d = idx.get("direction", "flat")
        scores.append(1.0 if d == "up" else (-1.0 if d == "down" else 0.0))
    # 해외: S&P500 과 나스닥에 가중치
    key_foreign = {"S&P 500", "나스닥", "^GSPC", "^IXIC"}
    for o in overnight:
        if any(k in (o.get("name", "") + o.get("symbol", "")) for k in key_foreign):
            d = o.get("direction", "flat")
            scores.append(1.5 if d == "up" else (-1.5 if d == "down" else 0.0))
    if not scores:
        return "neutral"
    avg = sum(scores) / len(scores)
    if avg > 0.3:
        return "positive"
    if avg < -0.3:
        return "negative"
    return "neutral"


def _derive_vix_signal(overnight: list[dict]) -> str | None:
    """VIX 수준으로 공포/안도 신호 반환. None if not found."""
    for o in overnight:
        if "VIX" in (o.get("symbol", "") + o.get("name", "")):
            try:
                val = float(str(o.get("value", "")).replace(",", ""))
                if val >= 30:
                    return "extreme_fear"
                if val >= 20:
                    return "fear"
                if val <= 13:
                    return "complacency"
                return "normal"
            except ValueError:
                pass
    return None


# ── 공개 API ─────────────────────────────────────────────────────────────────

def build_core_summary(macro_context: dict) -> dict:
    """⭐ 오늘의 핵심 — 거시 시장 요약 빌드.

    Args:
        macro_context: data_fetcher.fetch_macro_context() 반환값.
            {"indices", "fx", "overnight", "crypto_krw", "news", "generated_at"}

    Returns:
        {
          "generated_at":  str,
          "market_mood":   "positive"|"neutral"|"negative",
          "mood_label":    str,          # 우호 / 혼조 / 부담
          "vix_signal":    str|None,     # extreme_fear / fear / normal / complacency
          "indices":       dict,         # 정규화된 국내 지수
          "overnight":     list[dict],   # 정규화된 해외 시장
          "fx":            list[dict],   # 환율
          "crypto_krw":    dict,
          "top_news":      list[dict],   # 거시 뉴스 (impact 자동 분류)
          "news_summary":  {             # 호재/악재/중립 카운트
              "positive": int, "negative": int, "neutral": int
          },
          "mood_axes":     list,         # MOOD_AXES 메타 (UI용)
        }
    """
    indices  = _normalize_indices(macro_context.get("indices") or {})
    overnight = _normalize_overnight(macro_context.get("overnight") or [])
    news     = [_normalize_news_item(n) for n in (macro_context.get("news") or [])]

    market_mood = _derive_market_mood(indices, overnight)
    vix_signal  = _derive_vix_signal(overnight)

    news_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for n in news:
        news_counts[n.get("impact", "neutral")] = news_counts.get(n.get("impact", "neutral"), 0) + 1

    return {
        "generated_at": macro_context.get("generated_at", datetime.now(KST).isoformat()),
        "market_mood":  market_mood,
        "mood_label":   MOOD_LABEL.get(market_mood, market_mood),
        "vix_signal":   vix_signal,
        "indices":      indices,
        "overnight":    overnight,
        "fx":           macro_context.get("fx") or [],
        "crypto_krw":   macro_context.get("crypto_krw") or {},
        "top_news":     news,
        "news_summary": news_counts,
        "mood_axes":    MOOD_AXES,
    }
