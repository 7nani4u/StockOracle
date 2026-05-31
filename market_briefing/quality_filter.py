"""
quality_filter.py — AQR Quality Minus Junk (QMJ) 사전 필터

HybridTurtle-v6.0 src/lib/quality-filter.ts Python 이식.

3가지 기본 재무 지표로 종목을 사전 필터링:
  1. ROE > 10%          → +1 (수익성)
  2. 부채비율 < 1.5     → +1 (재무건전성)
  3. 매출성장률 > 0     → +1 (기본 모멘텀)

금융 섹터 대체:
  은행/보험/REIT는 D/E 높음 → 부채비율 대신 ROA > 1% 사용.

등급 체계:
  3/3 → high   (multiplier 1.0,  통과)
  2/3 → medium (multiplier 0.75, 통과)
  1/3 → low    (multiplier 0.0,  탈락)
  0/3 → junk   (multiplier 0.0,  탈락)
  모두 없음 → unknown (multiplier 0.5, 통과)

KRX/미국 모두 지원. yahoo-finance2 응답 구조에 맞게 파싱.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# 금융 섹터 — D/E 대신 ROA 사용
_FINANCIAL_SECTORS = frozenset({
    "Financial Services",
    "Real Estate",
    "금융",
    "부동산",
    "Finance",
})


@dataclass
class QualityFilterResult:
    """QMJ 필터 결과."""
    ticker:                   str
    pass_filter:              bool
    quality_tier:             str     # high / medium / low / junk / unknown
    quality_score:            float   # 0~3 (연속값)
    momentum_score_multiplier:float   # NCS에 곱할 배율
    roe:                      Optional[float] = None
    debt_to_equity:           Optional[float] = None
    revenue_growth:           Optional[float] = None
    return_on_assets:         Optional[float] = None
    is_financial_sector:      bool = False
    data_complete:            bool = False
    reason:                   str  = ""


def score_quality(
    ticker:          str,
    roe:             Optional[float],
    debt_to_equity:  Optional[float],
    revenue_growth:  Optional[float],
    return_on_assets:Optional[float] = None,
    sector:          Optional[str]   = None,
) -> QualityFilterResult:
    """QMJ 품질 점수 계산 (순수 함수, 네트워크 호출 없음).

    Args:
        ticker:           종목 코드
        roe:              자기자본이익률 (소수점, 0.10 = 10%)
        debt_to_equity:   부채비율 (소수점, 1.5 = 150%)
        revenue_growth:   매출성장률 (소수점, 0.05 = 5%)
        return_on_assets: 총자산이익률 (금융 섹터 대체, 소수점)
        sector:           섹터 문자열

    Returns:
        QualityFilterResult
    """
    is_financial = bool(sector and sector in _FINANCIAL_SECTORS)

    # 데이터 존재 여부
    roe_present     = roe is not None
    revenue_present = revenue_growth is not None
    third_present   = (return_on_assets is not None) if is_financial else (debt_to_equity is not None)
    missing_count   = sum(1 for p in [roe_present, revenue_present, third_present] if not p)

    # 모두 없으면 unknown
    if missing_count == 3:
        logger.warning("[QMJ] %s: 재무 데이터 전혀 없음 → unknown", ticker)
        return QualityFilterResult(
            ticker=ticker,
            pass_filter=True,
            quality_tier="unknown",
            quality_score=0.0,
            momentum_score_multiplier=0.5,
            roe=roe,
            debt_to_equity=debt_to_equity,
            revenue_growth=revenue_growth,
            return_on_assets=return_on_assets,
            is_financial_sector=is_financial,
            data_complete=False,
            reason="재무 데이터 없음 (unknown 처리)",
        )

    score = 0.0
    reasons = []

    # 지표 1: ROE > 10%
    if roe is None:
        score += 0.5
        reasons.append("ROE 없음 (+0.5)")
    elif roe > 0.10:
        score += 1.0
        reasons.append(f"ROE {roe*100:.1f}% > 10% (+1)")
    else:
        reasons.append(f"ROE {roe*100:.1f}% ≤ 10% (0)")

    # 지표 2: 매출성장률 > 0
    if revenue_growth is None:
        score += 0.5
        reasons.append("매출성장률 없음 (+0.5)")
    elif revenue_growth > 0:
        score += 1.0
        reasons.append(f"매출성장률 {revenue_growth*100:.1f}% > 0% (+1)")
    else:
        reasons.append(f"매출성장률 {revenue_growth*100:.1f}% ≤ 0% (0)")

    # 지표 3: D/E < 1.5 (또는 ROA > 1% for 금융)
    if is_financial:
        if return_on_assets is None:
            score += 0.5
            reasons.append("ROA 없음 (금융 섹터 대체, +0.5)")
        elif return_on_assets > 0.01:
            score += 1.0
            reasons.append(f"ROA {return_on_assets*100:.2f}% > 1% (+1)")
        else:
            reasons.append(f"ROA {return_on_assets*100:.2f}% ≤ 1% (0)")
    else:
        if debt_to_equity is None:
            score += 0.5
            reasons.append("부채비율 없음 (+0.5)")
        elif debt_to_equity < 1.5:
            score += 1.0
            reasons.append(f"D/E {debt_to_equity:.2f} < 1.5 (+1)")
        else:
            reasons.append(f"D/E {debt_to_equity:.2f} ≥ 1.5 (0)")

    rounded = round(score)
    data_complete = (missing_count == 0)

    # 등급 결정
    if rounded >= 3:
        tier, multiplier, pass_f = "high",   1.00, True
    elif rounded >= 2:
        tier, multiplier, pass_f = "medium", 0.75, True
    elif rounded >= 1:
        tier, multiplier, pass_f = "low",    0.00, False
    else:
        tier, multiplier, pass_f = "junk",   0.00, False

    return QualityFilterResult(
        ticker=ticker,
        pass_filter=pass_f,
        quality_tier=tier,
        quality_score=score,
        momentum_score_multiplier=multiplier,
        roe=roe,
        debt_to_equity=debt_to_equity,
        revenue_growth=revenue_growth,
        return_on_assets=return_on_assets,
        is_financial_sector=is_financial,
        data_complete=data_complete,
        reason=" | ".join(reasons),
    )


def get_conviction_bonus(quality_tier: str, is_converged: bool) -> float:
    """VolumeTurtle + HBME 신호 수렴 시 포지션 크기 보너스 (0.0~0.15).

    Args:
        quality_tier:  high / medium / low / junk / unknown
        is_converged:  두 신호가 동시에 발동 여부

    Returns:
        추가 배율 (0.0~0.15)
    """
    if not is_converged:
        return 0.0
    if quality_tier == "high":
        return 0.15
    if quality_tier == "medium":
        return 0.05
    return 0.0


# ── Yahoo Finance 재무 데이터 파싱 헬퍼 ─────────────────────────────────────

def _extract_fundamental_from_info(info: dict) -> dict:
    """yfinance Ticker.info dict에서 QMJ 재무 데이터 추출.

    Returns:
        {"roe", "debt_to_equity", "revenue_growth", "return_on_assets", "sector"}
    """
    def _to_float(v) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
            import math
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    # yfinance key 매핑
    roe             = _to_float(info.get("returnOnEquity"))
    debt_to_equity  = _to_float(info.get("debtToEquity"))
    revenue_growth  = _to_float(info.get("revenueGrowth"))
    roa             = _to_float(info.get("returnOnAssets"))
    sector          = info.get("sector") or info.get("sectorDisp")

    # D/E: yfinance는 %로 반환하는 경우가 있으므로 정규화
    # 보통 1.5 = 150% 표현 (소수점 기준). 값이 100 이상이면 / 100
    if debt_to_equity is not None and abs(debt_to_equity) > 10:
        debt_to_equity = debt_to_equity / 100.0

    return {
        "roe":             roe,
        "debt_to_equity":  debt_to_equity,
        "revenue_growth":  revenue_growth,
        "return_on_assets":roa,
        "sector":          sector,
    }


def get_quality_score_from_info(ticker: str, info: dict) -> QualityFilterResult:
    """yfinance Ticker.info에서 직접 QMJ 점수 계산.

    Usage:
        import yfinance as yf
        t = yf.Ticker("AAPL")
        result = get_quality_score_from_info("AAPL", t.info)

    Args:
        ticker: 종목 코드
        info:   yfinance Ticker.info dict

    Returns:
        QualityFilterResult
    """
    fundamentals = _extract_fundamental_from_info(info)
    return score_quality(
        ticker          = ticker,
        roe             = fundamentals["roe"],
        debt_to_equity  = fundamentals["debt_to_equity"],
        revenue_growth  = fundamentals["revenue_growth"],
        return_on_assets= fundamentals["return_on_assets"],
        sector          = fundamentals["sector"],
    )
