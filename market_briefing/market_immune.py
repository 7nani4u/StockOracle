"""
market_immune.py — 시장 위험 면역 시스템

HybridTurtle-v6.0 의도 (TypeScript 미구현, 문서 기반 Python 구현):
  "현재 시장 상황과 과거 위기 비교" — SYSTEM-BREAKDOWN.md, TRADING-LOGIC.md

기능:
  1. VIX / 지수 ATR% 기반 변동성 충격 감지
  2. 지수 가격과 MA200 이격도 계산
  3. 과거 위기 패턴과 유사도 비교 (코사인 유사도)
  4. 복합 면역 레벨 산출 (IMMUNE/ALERT/CAUTION/CLEAR)
  5. Kill 스위치 권고

공개 API:
  MarketImmuneResult  — 결과 데이터 클래스
  MarketImmune        — 핵심 클래스
    assess()          — 현재 시장 상황 평가
    get_kill_switch_advice() — Kill 스위치 권고
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


# ── 과거 위기 프로필 ──────────────────────────────────────────────────────────
# 각 위기의 특징: vix_level, ma200_deviation_pct, daily_vol_pct, drawdown_pct
# 소스: 역사적 데이터 기반 근사값

CRISIS_PROFILES: List[Dict[str, Any]] = [
    {
        "name":    "COVID-19 (2020-03)",
        "vix":     82.0,
        "ma200_dev": -30.0,
        "vol_pct":   6.5,
        "drawdown": -34.0,
        "severity": "EXTREME",
    },
    {
        "name":    "GFC (2008-10)",
        "vix":     89.5,
        "ma200_dev": -35.0,
        "vol_pct":   7.0,
        "drawdown": -50.0,
        "severity": "EXTREME",
    },
    {
        "name":    "닷컴 붕괴 (2002-10)",
        "vix":     42.0,
        "ma200_dev": -25.0,
        "vol_pct":   4.0,
        "drawdown": -49.0,
        "severity": "SEVERE",
    },
    {
        "name":    "Flash Crash (2010-05)",
        "vix":     45.0,
        "ma200_dev": -12.0,
        "vol_pct":   5.5,
        "drawdown": -16.0,
        "severity": "SEVERE",
    },
    {
        "name":    "2018-12 급락",
        "vix":     36.0,
        "ma200_dev": -10.0,
        "vol_pct":   3.0,
        "drawdown": -20.0,
        "severity": "MODERATE",
    },
    {
        "name":    "2022 금리 인상 충격",
        "vix":     38.0,
        "ma200_dev": -18.0,
        "vol_pct":   3.5,
        "drawdown": -27.0,
        "severity": "MODERATE",
    },
    {
        "name":    "정상 조정 (일반적)",
        "vix":     25.0,
        "ma200_dev": -5.0,
        "vol_pct":   1.5,
        "drawdown": -8.0,
        "severity": "MILD",
    },
]

# ── 임계값 ────────────────────────────────────────────────────────────────────

VIX_EXTREME  = 40.0
VIX_SEVERE   = 30.0
VIX_MODERATE = 20.0

MA200_BEAR_PCT      = -10.0   # MA200 대비 -10% 이하 → 약세
MA200_CRITICAL_PCT  = -20.0   # -20% 이하 → 심각

VOL_PCT_HIGH  = 3.0   # ATR% > 3% → 고변동
VOL_PCT_SHOCK = 5.0   # ATR% > 5% → 충격

# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class CrisisMatch:
    """과거 위기 유사도 결과."""
    name:        str
    severity:    str
    similarity:  float   # 0~1 (코사인 유사도)
    vix:         float
    ma200_dev:   float
    vol_pct:     float
    drawdown:    float


@dataclass
class MarketImmuneResult:
    """시장 위험 면역 시스템 평가 결과."""
    immune_level:    str          # CLEAR / CAUTION / ALERT / IMMUNE (매수 금지)
    immune_score:    float        # 0~100 (높을수록 위험)
    vix_level:       Optional[float]
    ma200_deviation: Optional[float]
    vol_pct:         Optional[float]
    warnings:        List[str] = field(default_factory=list)
    kill_switch:     Dict[str, bool] = field(default_factory=dict)
    crisis_matches:  List[CrisisMatch] = field(default_factory=list)
    top_crisis:      Optional[CrisisMatch] = None
    generated_at:    str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "immune_level":    self.immune_level,
            "immune_score":    round(self.immune_score, 1),
            "vix_level":       self.vix_level,
            "ma200_deviation": round(self.ma200_deviation, 2) if self.ma200_deviation is not None else None,
            "vol_pct":         round(self.vol_pct, 2) if self.vol_pct is not None else None,
            "warnings":        self.warnings,
            "kill_switch":     self.kill_switch,
            "crisis_matches":  [
                {
                    "name":       c.name,
                    "severity":   c.severity,
                    "similarity": round(c.similarity * 100, 1),
                    "vix":        c.vix,
                    "ma200_dev":  c.ma200_dev,
                    "drawdown":   c.drawdown,
                }
                for c in self.crisis_matches[:3]
            ],
            "top_crisis": {
                "name":       self.top_crisis.name,
                "severity":   self.top_crisis.severity,
                "similarity": round(self.top_crisis.similarity * 100, 1),
                "drawdown":   self.top_crisis.drawdown,
            } if self.top_crisis else None,
            "generated_at": self.generated_at,
        }


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """코사인 유사도 (0~1)."""
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), 0.0, 1.0))


def _safe(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ── 핵심 클래스 ───────────────────────────────────────────────────────────────

class MarketImmune:
    """시장 위험 면역 시스템.

    SPY/KOSPI200 OHLCV + VIX 데이터를 받아 현재 시장 위험도를 평가하고,
    과거 위기 패턴과 유사도를 계산한다.
    """

    def assess(
        self,
        index_closes:  List[float],            # 지수 종가 (최소 20개 권장)
        index_highs:   List[float],
        index_lows:    List[float],
        vix:           Optional[float] = None,  # VIX 현재값 (미국 시장)
        adv_decline:   Optional[float] = None,  # A/D 비율
    ) -> MarketImmuneResult:
        """시장 위험도 평가.

        Args:
            index_closes:  지수 종가 리스트
            index_highs:   지수 고가 리스트
            index_lows:    지수 저가 리스트
            vix:           VIX 현재값
            adv_decline:   A/D 비율

        Returns:
            MarketImmuneResult
        """
        warnings: List[str] = []
        score    = 0.0

        n = len(index_closes)
        if n < 5:
            return MarketImmuneResult(
                immune_level = "CAUTION",
                immune_score = 50.0,
                vix_level    = vix,
                ma200_deviation = None,
                vol_pct      = None,
                warnings     = ["지수 데이터 부족 — 평가 제한적"],
                kill_switch  = {"disable_new_entries": False},
            )

        current = index_closes[-1]

        # ── 1. MA200 이격도 ──────────────────────────────────────────────────
        ma200_dev = None
        if n >= 200:
            ma200 = float(np.mean(index_closes[-200:]))
            ma200_dev = (current / ma200 - 1) * 100
        elif n >= 50:
            ma_n = float(np.mean(index_closes))
            ma200_dev = (current / ma_n - 1) * 100

        if ma200_dev is not None:
            if ma200_dev < MA200_CRITICAL_PCT:
                score += 35.0
                warnings.append(f"지수 MA200 이격 {ma200_dev:.1f}% — 심각한 하락장")
            elif ma200_dev < MA200_BEAR_PCT:
                score += 20.0
                warnings.append(f"지수 MA200 이격 {ma200_dev:.1f}% — 약세장 진입")

        # ── 2. VIX 점수 ──────────────────────────────────────────────────────
        if vix is not None:
            if vix >= VIX_EXTREME:
                score += 40.0
                warnings.append(f"VIX {vix:.0f} — 극단적 공포 (≥{VIX_EXTREME})")
            elif vix >= VIX_SEVERE:
                score += 25.0
                warnings.append(f"VIX {vix:.0f} — 심각한 공포 (≥{VIX_SEVERE})")
            elif vix >= VIX_MODERATE:
                score += 10.0
                warnings.append(f"VIX {vix:.0f} — 공포 지수 상승 (≥{VIX_MODERATE})")

        # ── 3. 현재 변동성 (ATR%) ────────────────────────────────────────────
        vol_pct = None
        if n >= 15 and len(index_highs) >= 15 and len(index_lows) >= 15:
            trs = []
            for i in range(1, min(15, n)):
                tr = max(
                    index_highs[-i] - index_lows[-i],
                    abs(index_highs[-i] - index_closes[-i-1]),
                    abs(index_lows[-i]  - index_closes[-i-1]),
                )
                trs.append(tr)
            atr = float(np.mean(trs)) if trs else 0.0
            vol_pct = atr / current * 100 if current > 0 else 0.0

            if vol_pct >= VOL_PCT_SHOCK:
                score += 20.0
                warnings.append(f"지수 ATR% {vol_pct:.1f}% — 변동성 충격")
            elif vol_pct >= VOL_PCT_HIGH:
                score += 10.0
                warnings.append(f"지수 ATR% {vol_pct:.1f}% — 고변동성")

        # ── 4. A/D 비율 ──────────────────────────────────────────────────────
        if adv_decline is not None and adv_decline < 0.7:
            score += 10.0
            warnings.append(f"A/D 비율 {adv_decline:.2f} — 하락 종목 압도")

        score = min(score, 100.0)

        # ── 5. 면역 레벨 결정 ────────────────────────────────────────────────
        if score >= 65:
            level = "IMMUNE"     # 신규 매수 전면 금지
        elif score >= 40:
            level = "ALERT"      # 신규 매수 극도 제한
        elif score >= 20:
            level = "CAUTION"    # 사이즈 축소 권고
        else:
            level = "CLEAR"

        # ── 6. Kill 스위치 권고 ──────────────────────────────────────────────
        kill_switch = {
            "disable_new_entries":        level in ("IMMUNE", "ALERT"),
            "disable_automated_entries":  level in ("IMMUNE", "ALERT", "CAUTION"),
            "reduce_position_size":       level in ("IMMUNE", "ALERT", "CAUTION"),
            "require_manual_confirm":     level == "ALERT",
        }

        # ── 7. 과거 위기 유사도 비교 ─────────────────────────────────────────
        current_vec = [
            _safe(vix, 15.0),
            _safe(ma200_dev, 0.0),
            _safe(vol_pct, 1.5),
        ]
        matches: List[CrisisMatch] = []
        for profile in CRISIS_PROFILES:
            crisis_vec = [
                profile["vix"],
                abs(profile["ma200_dev"]) * -1,  # 이미 음수
                profile["vol_pct"],
            ]
            sim = _cosine_similarity(current_vec, crisis_vec)
            matches.append(CrisisMatch(
                name       = profile["name"],
                severity   = profile["severity"],
                similarity = sim,
                vix        = profile["vix"],
                ma200_dev  = profile["ma200_dev"],
                vol_pct    = profile["vol_pct"],
                drawdown   = profile["drawdown"],
            ))

        matches.sort(key=lambda c: c.similarity, reverse=True)
        top = matches[0] if matches else None

        return MarketImmuneResult(
            immune_level    = level,
            immune_score    = round(score, 1),
            vix_level       = vix,
            ma200_deviation = ma200_dev,
            vol_pct         = vol_pct,
            warnings        = warnings,
            kill_switch     = kill_switch,
            crisis_matches  = matches,
            top_crisis      = top,
        )

    def quick_check(
        self,
        vix:           Optional[float] = None,
        ma200_dev_pct: Optional[float] = None,
        atr_pct:       Optional[float] = None,
    ) -> Dict[str, Any]:
        """빠른 위험 체크 (지수 OHLCV 없이 지표 직접 입력).

        Returns:
            {"immune_level": str, "immune_score": float, "warnings": list}
        """
        warnings: List[str] = []
        score = 0.0

        if vix is not None:
            if vix >= VIX_EXTREME:
                score += 40.0
                warnings.append(f"VIX {vix:.0f} — 극단적 공포")
            elif vix >= VIX_SEVERE:
                score += 25.0
                warnings.append(f"VIX {vix:.0f} — 심각한 공포")
            elif vix >= VIX_MODERATE:
                score += 10.0
                warnings.append(f"VIX {vix:.0f} — 공포 지수 상승")

        if ma200_dev_pct is not None:
            if ma200_dev_pct < MA200_CRITICAL_PCT:
                score += 35.0
                warnings.append(f"MA200 이격 {ma200_dev_pct:.1f}% — 심각한 하락장")
            elif ma200_dev_pct < MA200_BEAR_PCT:
                score += 20.0
                warnings.append(f"MA200 이격 {ma200_dev_pct:.1f}% — 약세장 진입")

        if atr_pct is not None:
            if atr_pct >= VOL_PCT_SHOCK:
                score += 20.0
                warnings.append(f"ATR% {atr_pct:.1f}% — 변동성 충격")
            elif atr_pct >= VOL_PCT_HIGH:
                score += 10.0
                warnings.append(f"ATR% {atr_pct:.1f}% — 고변동성")

        score = min(score, 100.0)

        if score >= 65:
            level = "IMMUNE"
        elif score >= 40:
            level = "ALERT"
        elif score >= 20:
            level = "CAUTION"
        else:
            level = "CLEAR"

        return {
            "immune_level":  level,
            "immune_score":  round(score, 1),
            "warnings":      warnings,
            "kill_switch": {
                "disable_new_entries": level in ("IMMUNE", "ALERT"),
                "reduce_position_size": level in ("IMMUNE", "ALERT", "CAUTION"),
            },
        }

    def get_kill_switch_advice(
        self,
        immune_level: str,
        has_open_positions: bool = False,
    ) -> Dict[str, Any]:
        """면역 레벨 기반 Kill 스위치 권고 문구."""
        advice = {
            "CLEAR": {
                "summary":    "정상 시장 — 정상 운영",
                "new_entries":"허용",
                "position_size": "정상",
                "action":     "평상시 전략 유지",
            },
            "CAUTION": {
                "summary":    "주의 단계 — 신중한 접근 권고",
                "new_entries":"제한 (고품질 종목만)",
                "position_size": "50% 축소",
                "action":     "NCS ≥ 70 & FWS ≤ 30인 종목만 진입",
            },
            "ALERT": {
                "summary":    "경보 단계 — 신규 진입 최소화",
                "new_entries":"강력 제한 (수동 확인 필수)",
                "position_size": "25% 이하",
                "action":     "기존 포지션 손절가 상향 조정, 신규 진입 자제",
            },
            "IMMUNE": {
                "summary":    "면역 발동 — 신규 매수 전면 금지",
                "new_entries":"전면 금지",
                "position_size": "0 (진입 없음)",
                "action":     "기존 포지션 보호에 집중, 현금 비중 확대",
            },
        }.get(immune_level, {
            "summary":    "알 수 없음",
            "new_entries":"평가 불가",
            "position_size": "정상",
            "action":     "추가 데이터 필요",
        })

        if has_open_positions and immune_level in ("ALERT", "IMMUNE"):
            advice["action"] += " | 오픈 포지션 손절가 즉시 확인 및 상향 조정"

        return advice
