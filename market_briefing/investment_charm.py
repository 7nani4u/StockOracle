# -*- coding: utf-8 -*-
"""
investment_charm.py — ChoiceStock 투자매력(스마트스코어) 진단 모듈

StockOracle의 기존 재무/지표 데이터를 재사용하여 5개 축으로 투자매력을 0~100점으로 평가하고,
종합 스마트스코어와 레이더 차트 데이터를 생성.

5개 축 (ChoiceStock 참고, StockOracle 데이터로 재정의):
  1) 미래성장성  — revenueGrowth, earningsGrowth, net_profit_growth
  2) 사업독점력  — grossMargins, operatingMargins, profitMargins, ROE, marketCap (대체 지표)
  3) 재무안전성  — debtToEquity, currentRatio, quickRatio
  4) 수익성      — returnOnEquity, returnOnAssets, profitMargins, operatingMargins
  5) 현금창출력  — operatingCashflow / freeCashflow 기반

모든 계산은 하나의 모듈에서 관리, UI에는 흩어놓지 않음.
결측치가 많으면 N/A 반환, 임의 점수 생성 금지.
한국/미국 데이터 필드 차이, 음수/0나누기/NaN 안전 처리.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

# ── 유틸 ────────────────────────────────────────────────────────────────

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        if not math.isfinite(f):
            return None
        return f
    except Exception:
        return None

def _clip(v: float, lo: float = 0, hi: float = 100) -> float:
    return max(lo, min(hi, v))

def _to_percent(v: Optional[float], is_ratio: bool = False) -> Optional[float]:
    """ROE/ROA/margins 등은 yfinance에서 소수(0.3=30%)로 오고, Naver에서는 퍼센트(15.2)로 옴.
    is_ratio=True면 0.3→30으로 변환, 아니면 15.2 그대로.
    자동 판별: 0< v <1 이면 *100, 1<=v<100 이면 그대로, >100 이면 100으로 클립.
    """
    if v is None:
        return None
    if 0 < v < 1:
        return v * 100
    if 1 <= v < 100:
        return v
    if v >= 100:
        return 100
    if v <= 0:
        return v  # 음수는 그대로 (적자)
    return v

def _normalize_debt_pct(v: Optional[float]) -> Optional[float]:
    """debtToEquity 정규화: yfinance는 퍼센트(3.86) 또는 소수(0.038)로 혼재.
    <1.0 이면 ratio로 보고 *100, 1~1000 이면 그대로 퍼센트."""
    if v is None:
        return None
    if 0 < v < 1.0:
        return v * 100
    return v

def _score_to_stars(score: Optional[float]) -> Tuple[str, int]:
    """0~100 → 5단계 별점. N/A면 0개."""
    if score is None or not math.isfinite(score):
        return "☆☆☆☆☆", 0
    if score >= 80:
        return "★★★★★", 5
    if score >= 60:
        return "★★★★☆", 4
    if score >= 40:
        return "★★★☆☆", 3
    if score >= 20:
        return "★★☆☆☆", 2
    return "★☆☆☆☆", 1

# ── 5개 세부 진단 ─────────────────────────────────────────────────────

def score_growth(info: Dict[str, Any], naver: Dict[str, Any]) -> Optional[float]:
    """미래성장성: revenueGrowth, earningsGrowth, net_profit_growth 중 가용 데이터로 평가.
    각 성장률 30%+ → 90, 15%+ →75, 5%+ →60, 0%+ →45, -10%+ →30, 그외 15."""
    vals = []
    for key in ["revenueGrowth", "earningsGrowth"]:
        v = _safe_float(info.get(key))
        if v is not None:
            # yfinance는 소수(0.164=16.4%), naver는 퍼센트일 수도 있으나 info 우선
            # 0.01~1.5 범위면 소수로 보고 *100, 이미 퍼센트면 그대로
            if -5 < v < 5:  # 소수 범위
                v = v * 100
            vals.append(v)
    # naver net_profit_growth는 이미 퍼센트
    ng = _safe_float(naver.get("net_profit_growth"))
    if ng is not None:
        vals.append(ng)
    # naver에서 revenueGrowth/earningsGrowth가 없을 경우, info의 값이 유일
    if not vals:
        return None
    # 각 값을 0~100으로 변환
    scores = []
    for g in vals:
        if g >= 50:
            s = 95
        elif g >= 30:
            s = 85
        elif g >= 15:
            s = 75
        elif g >= 5:
            s = 60
        elif g >= 0:
            s = 45
        elif g >= -10:
            s = 30
        else:
            s = 15
        scores.append(s)
    # 가중 평균: earningsGrowth가 있으면 60% 가중, 아니면 균등
    if len(scores) == 1:
        return scores[0]
    # earningsGrowth가 첫 번째가 아닐 수도 있으니, 가장 높은 성장률에 가중
    # 단순 평균으로 처리
    return sum(scores) / len(scores)

def score_monopoly(info: Dict[str, Any], naver: Dict[str, Any], market_cap: Optional[float], market: str) -> Optional[float]:
    """사업독점력: 직접 점유율 데이터 없으므로 대체 지표로만 평가.
    grossMargins, operatingMargins, profitMargins, ROE, marketCap 기반.
    2개 이상 지표가 있어야 평가, 아니면 N/A."""
    gross = _safe_float(info.get("grossMargins"))
    opm = _safe_float(info.get("operatingMargins"))
    if opm is None:
        opm = _safe_float(naver.get("op_margin"))
        if opm is not None and opm > 1:  # naver는 퍼센트
            opm = opm / 100
    npm = _safe_float(info.get("profitMargins"))
    roe = _safe_float(info.get("returnOnEquity"))
    if roe is None:
        roe = _safe_float(naver.get("roe"))
        if roe is not None and roe > 1:
            roe = roe / 100

    # marketCap은 yfinance info의 marketCap 사용
    mc = _safe_float(market_cap) or _safe_float(info.get("marketCap"))
    # naver market_cap은 "1,234조 5,678" 같은 문자열이므로 파싱 필요하지만, 여기서는 yfinance만 사용

    vals = []
    # grossMargins
    if gross is not None:
        if gross > 0.6:
            vals.append(90)
        elif gross > 0.4:
            vals.append(75)
        elif gross > 0.2:
            vals.append(55)
        elif gross > 0.1:
            vals.append(35)
        else:
            vals.append(20)
    # operatingMargins
    if opm is not None:
        if opm > 0.4:
            vals.append(90)
        elif opm > 0.2:
            vals.append(75)
        elif opm > 0.1:
            vals.append(55)
        elif opm > 0.03:
            vals.append(35)
        else:
            vals.append(15)
    # profitMargins
    if npm is not None:
        if npm > 0.25:
            vals.append(90)
        elif npm > 0.15:
            vals.append(70)
        elif npm > 0.08:
            vals.append(50)
        elif npm > 0.03:
            vals.append(30)
        else:
            vals.append(10)
    # ROE
    if roe is not None:
        roe_pct = roe * 100 if 0 < roe < 10 else roe
        if roe_pct > 25:
            vals.append(90)
        elif roe_pct > 15:
            vals.append(75)
        elif roe_pct > 8:
            vals.append(55)
        elif roe_pct > 0:
            vals.append(30)
        else:
            vals.append(10)

    # 데이터가 2개 미만이면 신뢰성 낮아 N/A
    if len(vals) < 2:
        return None
    return sum(vals) / len(vals)

def score_safety(info: Dict[str, Any], naver: Dict[str, Any]) -> Optional[float]:
    """재무안전성: debtToEquity, currentRatio, quickRatio"""
    debt = _safe_float(info.get("debtToEquity"))
    if debt is None:
        debt = _safe_float(naver.get("debt"))
    debt_pct = _normalize_debt_pct(debt)
    cr = _safe_float(info.get("currentRatio"))
    qr = _safe_float(info.get("quickRatio"))

    vals = []
    if debt_pct is not None:
        if debt_pct < 30:
            vals.append(90)
        elif debt_pct < 70:
            vals.append(75)
        elif debt_pct < 120:
            vals.append(55)
        elif debt_pct < 180:
            vals.append(30)
        else:
            vals.append(10)
    if cr is not None:
        if cr > 2.5:
            vals.append(90)
        elif cr > 1.8:
            vals.append(75)
        elif cr > 1.2:
            vals.append(55)
        elif cr > 0.8:
            vals.append(30)
        else:
            vals.append(10)
    if qr is not None:
        if qr > 1.8:
            vals.append(90)
        elif qr > 1.2:
            vals.append(75)
        elif qr > 0.8:
            vals.append(55)
        elif qr > 0.5:
            vals.append(30)
        else:
            vals.append(10)

    if not vals:
        return None
    # debt가 가장 중요하므로 가중 50%
    if debt_pct is not None and len(vals) > 1:
        debt_score = vals[0]
        other_avg = sum(vals[1:]) / len(vals[1:]) if len(vals) > 1 else 50
        return debt_score * 0.6 + other_avg * 0.4
    return sum(vals) / len(vals)

def score_profitability(info: Dict[str, Any], naver: Dict[str, Any]) -> Optional[float]:
    """수익성: ROE, ROA, profitMargins, operatingMargins"""
    roe = _safe_float(info.get("returnOnEquity"))
    if roe is None:
        roe = _safe_float(naver.get("roe"))
        if roe is not None and roe > 1:
            roe = roe / 100
    roa = _safe_float(info.get("returnOnAssets"))
    pm = _safe_float(info.get("profitMargins"))
    opm = _safe_float(info.get("operatingMargins"))
    if opm is None:
        opm = _safe_float(naver.get("op_margin"))
        if opm is not None and opm > 1:
            opm = opm / 100

    vals = []
    if roe is not None:
        roe_pct = roe * 100 if 0 < roe < 10 else roe
        if roe_pct > 25:
            vals.append(95)
        elif roe_pct > 15:
            vals.append(80)
        elif roe_pct > 8:
            vals.append(60)
        elif roe_pct > 3:
            vals.append(35)
        elif roe_pct > 0:
            vals.append(20)
        else:
            vals.append(5)
    if roa is not None:
        roa_pct = roa * 100 if 0 < roa < 10 else roa
        if roa_pct > 15:
            vals.append(90)
        elif roa_pct > 8:
            vals.append(70)
        elif roa_pct > 3:
            vals.append(50)
        elif roa_pct > 0:
            vals.append(25)
        else:
            vals.append(5)
    if pm is not None:
        if pm > 0.25:
            vals.append(90)
        elif pm > 0.15:
            vals.append(70)
        elif pm > 0.08:
            vals.append(50)
        elif pm > 0.03:
            vals.append(30)
        else:
            vals.append(10)
    if opm is not None:
        if opm > 0.3:
            vals.append(85)
        elif opm > 0.15:
            vals.append(65)
        elif opm > 0.05:
            vals.append(45)
        elif opm > 0:
            vals.append(25)
        else:
            vals.append(5)

    if not vals:
        return None
    return sum(vals) / len(vals)

def score_cash(info: Dict[str, Any]) -> Optional[float]:
    """현금창출력: operatingCashflow, freeCashflow, marketCap, totalRevenue 기반
    - FCF yield = freeCashflow / marketCap
    - OCF margin = operatingCashflow / totalRevenue
    - FCF conversion = freeCashflow / netIncome
    """
    ocf = _safe_float(info.get("operatingCashflow"))
    fcf = _safe_float(info.get("freeCashflow"))
    mc = _safe_float(info.get("marketCap"))
    rev = _safe_float(info.get("totalRevenue"))
    net = _safe_float(info.get("netIncomeToCommon"))
    if net is None:
        net = _safe_float(info.get("netIncome"))

    vals = []
    # FCF yield
    if fcf is not None and mc is not None and mc != 0:
        y = fcf / mc * 100
        if y > 8:
            vals.append(90)
        elif y > 4:
            vals.append(75)
        elif y > 1:
            vals.append(55)
        elif y > 0:
            vals.append(35)
        elif y > -2:
            vals.append(20)
        else:
            vals.append(5)
    # OCF margin
    if ocf is not None and rev is not None and rev != 0:
        m = ocf / rev * 100
        if m > 30:
            vals.append(90)
        elif m > 20:
            vals.append(75)
        elif m > 10:
            vals.append(55)
        elif m > 5:
            vals.append(35)
        elif m > 0:
            vals.append(20)
        else:
            vals.append(5)
    # FCF conversion (FCF / netIncome)
    if fcf is not None and net is not None and net != 0:
        # net이 음수면 별도 처리
        if net > 0:
            conv = fcf / net
            if conv > 1.0:
                vals.append(90)
            elif conv > 0.7:
                vals.append(75)
            elif conv > 0.4:
                vals.append(55)
            elif conv > 0.1:
                vals.append(35)
            else:
                vals.append(15)
        else:
            # 적자企业인데 FCF 양수면 우수
            if fcf > 0:
                vals.append(60)
            else:
                vals.append(10)

    if not vals:
        return None
    return sum(vals) / len(vals)

# ── 핵심 지표 (PER/PSR/ROE/DY) 헬퍼 ───────────────────────────────────────

def get_key_metrics(info: Dict[str, Any], naver: Dict[str, Any], market: str) -> Dict[str, Any]:
    """PER, PSR, ROE, DY를 안전하게 추출. N/A 처리."""
    # PER: trailingPE 우선, 없으면 forwardPE, KRX는 naver per
    per = _safe_float(info.get("trailingPE"))
    if per is None:
        per = _safe_float(info.get("forwardPE"))
    if per is None:
        per = _safe_float(naver.get("per"))
    # PSR: priceToSalesTrailing12Months 또는 marketCap/totalRevenue
    psr = _safe_float(info.get("priceToSalesTrailing12Months"))
    if psr is None:
        mc = _safe_float(info.get("marketCap"))
        rev = _safe_float(info.get("totalRevenue"))
        if mc is not None and rev is not None and rev != 0:
            psr = mc / rev
    # ROE
    roe = _safe_float(info.get("returnOnEquity"))
    if roe is None:
        roe = _safe_float(naver.get("roe"))
        if roe is not None and roe > 1:
            roe = roe / 100
    if roe is not None and 0 < roe < 10:
        roe = roe * 100
    # DY
    dy = _safe_float(info.get("dividendYield"))
    if dy is None:
        dy = _safe_float(info.get("trailingAnnualDividendYield"))
        # trailingAnnualDividendYield는 항상 소수(0.0033)이므로 *100 필요
        if dy is not None and 0 < dy < 1:
            dy = dy * 100
    else:
        # dividendYield: KRX와 US 모두 yfinance에서 퍼센트(0.58, 0.34)로 오는 경우가 많음
        # 0.58은 0.58% 그대로, 0.0033처럼 <0.05인 경우만 *100
        if dy is not None and 0 < dy < 0.05:
            dy = dy * 100
        # 0.05~10 사이는 이미 퍼센트로 간주 (0.34 → 0.34%, 0.58 → 0.58%)
        # 1.48 같은 ROE와 달리 DY는 10% 넘으면 그대로 (이미 퍼센트)

    def fmt(v, is_pct=False):
        if v is None or not math.isfinite(v):
            return "N/A"
        if is_pct:
            return f"{v:.2f}%"
        return f"{v:.2f}"

    def fmt_or_na(v, fmt_str="{:.2f}"):
        if v is None or not math.isfinite(v):
            return "N/A"
        try:
            return fmt_str.format(v)
        except:
            return "N/A"

    return {
        "per": per, "per_str": fmt_or_na(per),
        "psr": psr, "psr_str": fmt_or_na(psr),
        "roe": roe, "roe_str": fmt_or_na(roe, "{:.2f}%") if roe is not None else "N/A",
        "dy": dy, "dy_str": fmt_or_na(dy, "{:.2f}%") if dy is not None else "N/A",
        "per_raw": per, "psr_raw": psr, "roe_raw": roe, "dy_raw": dy,
    }

# ── 종합 스마트스코어 ───────────────────────────────────────────────────

def compute_charm_scores(info: Dict[str, Any], naver: Dict[str, Any], market: str) -> Dict[str, Any]:
    """5개 세부 점수와 종합 스마트스코어를 계산.
    반환: {smart_score, sub_scores: {growth, monopoly, safety, profitability, cash}, details}
    각 sub_scores는 0~100 또는 None(N/A)
    smart_score는 가용한 sub_scores의 평균, 3개 미만이면 N/A
    """
    # market_cap for monopoly
    mc = _safe_float(info.get("marketCap"))

    s_growth = score_growth(info, naver)
    s_mono = score_monopoly(info, naver, mc, market)
    s_safety = score_safety(info, naver)
    s_profit = score_profitability(info, naver)
    s_cash = score_cash(info)

    sub = {
        "growth": s_growth,
        "monopoly": s_mono,
        "safety": s_safety,
        "profitability": s_profit,
        "cash": s_cash,
    }
    # 종합: 가용한 점수 평균, 3개 미만이면 N/A
    avail = [v for v in sub.values() if v is not None]
    if len(avail) >= 3:
        smart = sum(avail) / len(avail)
        # 클리핑 및 반올림
        smart = _clip(smart, 0, 100)
        smart = round(smart)
    else:
        smart = None

    # 별점 및 퍼센트 (전체/업종 순위는 데이터 부족으로 N/A)
    # 퍼센트는 점수 자체를 백분위로 근사하지 않고 N/A로 처리
    result = {
        "smart_score": smart,
        "smart_score_str": str(smart) if smart is not None else "N/A",
        "sub_scores": {k: (round(v) if v is not None else None) for k, v in sub.items()},
        "sub_scores_raw": sub,
        "stars": {k: _score_to_stars(v) for k, v in sub.items()},
        "overall_stars": _score_to_stars(smart),
        "available_count": len(avail),
        "total_count": 5,
        "is_reliable": len(avail) >= 4,
        # 순위/백분위는 전체 모집단이 없으므로 N/A
        "overall_rank": None,
        "overall_rank_str": "N/A",
        "overall_percentile": None,
        "industry_rank": None,
        "industry_percentile": None,
        "industry": (naver.get("industry") or info.get("industry") or info.get("sector") or "N/A"),
        "universe_size": None,
    }
    return result

def get_industry_name(info: Dict[str, Any], naver: Dict[str, Any]) -> str:
    for k in ["industry", "sector"]:
        v = naver.get(k) or info.get(k)
        if v:
            return str(v)
    return "N/A"
