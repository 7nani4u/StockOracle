"""Strict Peter Lynch-style GARP fundamental assessment.

This module implements the five thresholds stated in the supplied article.
It is an explainable stock-level screen, not a return forecast or a buy signal.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, Optional


US_MARKET_CAP_LIMIT = 10_000_000_000.0
KRX_MARKET_CAP_LIMIT = 14_000_000_000_000.0


def _number(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _debt_to_equity_pct(value: Any) -> Optional[float]:
    """Normalize Yahoo/Naver debt-to-equity values to percent."""
    amount = _number(value)
    if amount is None:
        return None
    return amount * 100.0 if 0 < amount < 1.0 else amount


def _criterion(
    key: str,
    label: str,
    value: Optional[float],
    threshold: str,
    passed: Optional[bool],
    source: str,
    unavailable_reason: str = "",
) -> Dict[str, Any]:
    status = "pass" if passed is True else "fail" if passed is False else "unavailable"
    return {
        "key": key,
        "label": label,
        "value": round(value, 4) if value is not None else None,
        "threshold": threshold,
        "passed": passed,
        "status": status,
        "source": source,
        "unavailable_reason": unavailable_reason,
    }


def _iter_eps_rows(annual_income: Any) -> Iterable[tuple[Any, Any]]:
    """Read an annual EPS row from a yfinance income-statement DataFrame."""
    if annual_income is None or not hasattr(annual_income, "index") or not hasattr(annual_income, "loc"):
        return ()
    labels = list(annual_income.index)
    normalized = {
        re.sub(r"[^a-z]", "", str(label).lower()): label
        for label in labels
    }
    row_label = None
    for candidate in ("dilutedeps", "dilutedearningspershare", "basiceps"):
        if candidate in normalized:
            row_label = normalized[candidate]
            break
    if row_label is None:
        return ()
    try:
        return tuple(annual_income.loc[row_label].items())
    except Exception:
        return ()


def extract_consecutive_annual_eps(annual_income: Any) -> list[Dict[str, Any]]:
    """Return the latest four consecutive reported annual EPS observations.

    A CAGR should not be derived from quarterly EPS or from years with a gap,
    because both can materially distort the GARP screen.
    """
    by_year: Dict[int, tuple[str, float]] = {}
    for date_value, eps_value in _iter_eps_rows(annual_income):
        year_match = re.search(r"(19|20)\d{2}", str(date_value))
        eps = _number(eps_value)
        if not year_match or eps is None:
            continue
        year = int(year_match.group(0))
        by_year[year] = (str(date_value)[:10], eps)
    years = sorted(by_year)
    if len(years) < 4:
        return []
    latest_four = years[-4:]
    if latest_four != list(range(latest_four[0], latest_four[0] + 4)):
        return []
    return [
        {"year": year, "fiscal_date": by_year[year][0], "eps": by_year[year][1]}
        for year in latest_four
    ]


def build_peter_lynch_assessment(
    info: Dict[str, Any],
    naver: Optional[Dict[str, Any]],
    market: str,
    annual_income: Any = None,
) -> Dict[str, Any]:
    """Evaluate the supplied article's five Peter Lynch-style thresholds.

    The PEG is intentionally calculated from the same reported three-year EPS
    CAGR used by the growth criterion. This avoids mixing historical growth
    with provider-specific forward-growth PEG definitions.
    """
    info = info or {}
    naver = naver or {}
    is_krx = str(market).upper() == "KRX"

    per = _number(naver.get("per")) if is_krx else _number(info.get("trailingPE"))
    per_source = "Naver Finance" if is_krx and per is not None else "Yahoo Finance"
    if per is None:
        per = _number(info.get("trailingPE")) or _number(info.get("forwardPE"))
        per_source = "Yahoo Finance"

    debt_raw = info.get("debtToEquity")
    debt_source = "Yahoo Finance"
    if debt_raw is None and is_krx:
        debt_raw = naver.get("debt")
        debt_source = "Naver Finance"
    debt_pct = _debt_to_equity_pct(debt_raw)

    market_cap = _number(naver.get("market_cap_raw")) if is_krx else _number(info.get("marketCap"))
    cap_source = "Naver Finance" if is_krx and market_cap is not None else "Yahoo Finance"
    if market_cap is None:
        market_cap = _number(info.get("marketCap"))
        cap_source = "Yahoo Finance"
    cap_limit = KRX_MARKET_CAP_LIMIT if is_krx else US_MARKET_CAP_LIMIT

    eps_history = extract_consecutive_annual_eps(annual_income)
    eps_cagr = None
    eps_reason = ""
    if len(eps_history) != 4:
        eps_reason = "연속 4개 사업연도 EPS가 필요합니다."
    elif any(point["eps"] <= 0 for point in eps_history):
        eps_reason = "EPS CAGR은 시작·종료 EPS가 모두 양수일 때만 계산합니다."
    else:
        start_eps = eps_history[0]["eps"]
        end_eps = eps_history[-1]["eps"]
        eps_cagr = ((end_eps / start_eps) ** (1.0 / 3.0) - 1.0) * 100.0

    peg = per / eps_cagr if per is not None and per > 0 and eps_cagr is not None and eps_cagr > 0 else None
    peg_reason = ""
    if peg is None:
        peg_reason = "양수 PER와 양수 3년 EPS CAGR이 모두 필요합니다."

    criteria = [
        _criterion("per", "PER", per, "0 < PER < 25배", None if per is None else 0 < per < 25, per_source, "PER 데이터 없음" if per is None else ""),
        _criterion("debt_to_equity", "부채비율", debt_pct, "부채비율 < 80%", None if debt_pct is None else debt_pct < 80, debt_source, "부채비율 데이터 없음" if debt_pct is None else ""),
        _criterion("eps_cagr_3y", "3년 EPS 연평균 성장률", eps_cagr, "20% <= CAGR <= 50%", None if eps_cagr is None else 20 <= eps_cagr <= 50, "Yahoo Finance 연간 손익계산서", eps_reason),
        _criterion("peg", "PEG", peg, "PEG < 1.5배", None if peg is None else peg < 1.5, "PER / 보고 EPS 3년 CAGR", peg_reason),
        _criterion("market_cap", "시가총액", market_cap, "< 14조원" if is_krx else "< $10B", None if market_cap is None else market_cap < cap_limit, cap_source, "시가총액 데이터 없음" if market_cap is None else ""),
    ]
    passed_count = sum(item["passed"] is True for item in criteria)
    unavailable_count = sum(item["passed"] is None for item in criteria)
    failed_count = sum(item["passed"] is False for item in criteria)
    eligible = passed_count == len(criteria)
    sector = str(info.get("sector") or info.get("industry") or naver.get("sector") or "")
    notes = [
        "원문(2025-04-29)의 5개 수치 기준을 엄격 적용한 개별 종목 점검입니다. 매수·수익 예측이 아닙니다.",
        "EPS CAGR과 PEG는 분기치·추정치가 아닌 보고된 연간 EPS 4개 연속 사업연도를 사용합니다.",
    ]
    if any(token in sector.lower() for token in ("bank", "financial", "insurance", "은행", "금융", "보험")):
        notes.append("금융업은 부채비율의 경제적 의미가 일반 산업과 다를 수 있어 별도 재무 분석이 필요합니다.")

    return {
        "strategy": "Peter Lynch-style GARP",
        "eligible": eligible,
        "status": "pass" if eligible else "incomplete" if unavailable_count else "fail",
        "passed_count": passed_count,
        "failed_count": failed_count,
        "unavailable_count": unavailable_count,
        "criteria": criteria,
        "metrics": {
            "per": round(per, 4) if per is not None else None,
            "debt_to_equity_pct": round(debt_pct, 4) if debt_pct is not None else None,
            "eps_cagr_3y_pct": round(eps_cagr, 4) if eps_cagr is not None else None,
            "peg": round(peg, 4) if peg is not None else None,
            "market_cap": round(market_cap, 2) if market_cap is not None else None,
        },
        "eps_history": [
            {**point, "eps": round(point["eps"], 4)}
            for point in eps_history
        ],
        "notes": notes,
    }
