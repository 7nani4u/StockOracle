"""
scan_engine.py — HybridTurtle 7단계 스캔 엔진 (Python 이식)

원본: src/lib/scan-engine.ts (HybridTurtle-v6.0)
추가: BQS/FWS/NCS v2 통합, QMJ 품질 필터, 풀백 컨티뉴에이션

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1 │ Universe        │ 분석 대상 종목 로드
Stage 2 │ TechFilters     │ MA200, ADX, DI, ATR%, 효율, Hurst 필터
Stage 3 │ Classification  │ READY / WATCH / FAR 상태 분류
Stage 4 │ Ranking (BQS)   │ BQS + Sleeve 우선순위 기반 점수
Stage 5 │ Risk Gates      │ 포지션 집중도·리스크 한도 검증
Stage 6 │ AntiChase       │ 추격 방지 + 풀백 컨티뉴에이션
Stage 7 │ Sizing & Final  │ NCS 기반 최종 정렬 + 포지션 사이징
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

StockOracle 기존 코드와의 연동:
  - hybrid_signals.py  → compute_hybrid_score() 결과를 Stage 4/7에서 사용
  - dual_score_v2.py   → BQS/FWS/NCS 계산
  - quality_filter.py  → QMJ 사전 필터
  - data_fetcher.py    → OHLCV 데이터 제공
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dual_score_v2 import (
    SnapshotRow, compute_bqs, compute_fws, compute_penalties, compute_ncs,
    action_note, calc_hurst_v2, compute_bis_from_candle,
    compute_chasing_flags, REGIME_BULLISH, REGIME_BEARISH, REGIME_SIDEWAYS,
    VOL_LOW, VOL_NORMAL, VOL_HIGH,
)
from .quality_filter import QualityFilterResult, score_quality
from .hybrid_signals import (
    compute_hybrid_score, compute_regime, detect_vol_regime,
    adaptive_atr_buffer, anti_chase_guard,
    REGIME_BULLISH as HT_BULLISH, REGIME_BEARISH as HT_BEARISH,
)

logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────────────────────

ATR_STOP_MULTIPLIER        = 1.5     # 손절 = 진입 - ATR × 1.5
ATR_VOLATILITY_CAP_ALL     = 8.0     # 모든 슬리브 ATR% 상한
ATR_VOLATILITY_CAP_HIGH    = 12.0    # HIGH_RISK 슬리브 ATR% 상한
ANTI_CHASE_EXT_ATR_HARD    = 0.8     # ext_ATR 초과 → WAIT_PULLBACK
PULLBACK_ZONE_ATR_FACTOR   = 0.25    # 풀백 구간 = anchor ± 0.25×ATR
FAILED_BREAKOUT_COOLDOWN   = 3       # 실패 브레이크아웃 쿨다운 (일)

SLEEVE_PRIORITY = {
    "CORE":      40,
    "ETF":       20,
    "HIGH_RISK": 10,
    "HEDGE":     5,
}

TIER_MULTIPLIER = {
    "high":    1.0,
    "medium":  0.75,
    "low":     0.0,
    "junk":    0.0,
    "unknown": 0.5,
}

# ── 시장별 가격 상한 (출력 필터) ─────────────────────────────────────────────
# 미국 스캔은 $70 초과 종목을 출력에서 제외한다 (수집 후 실측가 기준).
SCAN_US_MAX_PRICE = 70.0

# ── 시장별 수집 상한 (Vercel 60s 타임아웃 방지 + 시간 예산 하드가드 병행) ──
# 미국은 전수 검토를 위해 유니버스 전체를 수집 대상으로 둔다.
SCAN_COLLECT_CAP_US_FULL = 120
SCAN_COLLECT_CAP_US_LITE = 48

# ── 거래량-가격 괴리(매집 탐지) ────────────────────────────────────────────
# 최근 20거래일과 그 직전 60거래일을 분리해 비교한다. 최근 구간을 기준 거래량에
# 섞지 않아 급증 신호가 스스로 희석되는 것을 막는다.
VPD_RECENT_DAYS = 20
VPD_BASELINE_DAYS = 60
VPD_MIN_VOLUME_RATIO = 3.0
VPD_MAX_ABS_PRICE_CHANGE_PCT = 3.0


def detect_volume_price_divergence(
    closes: List[float],
    volumes: List[float],
    recent_days: int = VPD_RECENT_DAYS,
    baseline_days: int = VPD_BASELINE_DAYS,
    min_volume_ratio: float = VPD_MIN_VOLUME_RATIO,
    max_abs_price_change_pct: float = VPD_MAX_ABS_PRICE_CHANGE_PCT,
) -> Dict[str, Any]:
    """거래량은 급증했지만 가격은 정체된 종목을 탐지한다.

    ``recent_days``의 일평균 거래량을 바로 앞 ``baseline_days``의 일평균과
    비교하고, 최근 구간 시작 직전 종가 대비 최신 종가의 순변동을 측정한다.
    두 조건(거래량 비율 >= 기준, 절대 가격변동 <= 기준)을 모두 만족할 때만
    ``is_match``가 참이다. 매집의 인과관계를 확정하는 지표는 아니다.
    """
    result: Dict[str, Any] = {
        "available": False,
        "is_match": False,
        "recent_days": recent_days,
        "baseline_days": baseline_days,
        "min_volume_ratio": float(min_volume_ratio),
        "max_abs_price_change_pct": float(max_abs_price_change_pct),
        "recent_avg_volume": None,
        "baseline_avg_volume": None,
        "volume_ratio": None,
        "price_change_pct": None,
        "volume_condition": False,
        "price_condition": False,
        "reason": "",
    }

    if recent_days <= 0 or baseline_days <= 0:
        result["reason"] = "분석 기간은 양수여야 합니다"
        return result
    if min_volume_ratio <= 0 or max_abs_price_change_pct < 0:
        result["reason"] = "탐지 임계값이 올바르지 않습니다"
        return result

    required = recent_days + baseline_days
    n = min(len(closes or []), len(volumes or []))
    if n < required:
        result["reason"] = f"데이터 부족: 최소 {required}거래일 필요 (현재 {n}일)"
        return result

    try:
        aligned_closes = [float(v) for v in closes[-n:]]
        aligned_volumes = [float(v) for v in volumes[-n:]]
        close_anchor = aligned_closes[-(recent_days + 1)]
        close_latest = aligned_closes[-1]
        recent_volumes = aligned_volumes[-recent_days:]
        baseline_volumes = aligned_volumes[-required:-recent_days]
    except (TypeError, ValueError, IndexError):
        result["reason"] = "가격 또는 거래량 데이터 형식 오류"
        return result

    values = [close_anchor, close_latest, *recent_volumes, *baseline_volumes]
    if not all(math.isfinite(v) for v in values):
        result["reason"] = "가격 또는 거래량에 유효하지 않은 값이 있습니다"
        return result
    if close_anchor <= 0 or close_latest <= 0:
        result["reason"] = "가격은 양수여야 합니다"
        return result
    if any(v < 0 for v in recent_volumes) or any(v < 0 for v in baseline_volumes):
        result["reason"] = "거래량은 음수일 수 없습니다"
        return result

    recent_avg = float(np.mean(recent_volumes))
    baseline_avg = float(np.mean(baseline_volumes))
    if baseline_avg <= 0:
        result["reason"] = "기준 기간 평균 거래량이 0입니다"
        return result

    volume_ratio = recent_avg / baseline_avg
    price_change_pct = (close_latest / close_anchor - 1.0) * 100.0
    # 부동소수점 경계 오차로 정확히 3배/±3%인 값이 탈락하지 않게 한다.
    epsilon = 1e-12
    volume_condition = volume_ratio + epsilon >= min_volume_ratio
    price_condition = abs(price_change_pct) <= max_abs_price_change_pct + epsilon
    is_match = volume_condition and price_condition

    result.update({
        "available": True,
        "is_match": is_match,
        "recent_avg_volume": round(recent_avg, 2),
        "baseline_avg_volume": round(baseline_avg, 2),
        "volume_ratio": round(volume_ratio, 3),
        "price_change_pct": round(price_change_pct, 3),
        "volume_condition": volume_condition,
        "price_condition": price_condition,
        "reason": (
            "거래량 급증과 가격 정체 조건을 모두 충족"
            if is_match else
            "거래량 기준 미충족" if not volume_condition else
            "가격 변동폭 기준 미충족"
        ),
    })
    return result


# ── 거래량-가격 괴리 관찰(Tier W) ──────────────────────────────────────────
# Tier S(엄격: 최근 20일 평균 >= 직전 60일 평균의 3배 + 20일 가격 ±3%)는
# detect_volume_price_divergence()에 고정한다. 아래는 엄격식을 바꾸지 않고
# 0건인 화면을 설명하기 위한 보조 증거·롤링·관찰 등급이다.
VPD_ROLLING_LOOKBACK_DAYS = 60
VPD_ROLLING_STEP_DAYS = 5
VPD_WATCH_MIN_VOLUME_RATIO = 2.0
VPD_WATCH_MAX_PRICE_CHANGE_PCT = 6.0
VPD_WATCH_MIN_DAYS_GE_2X = 5
VPD_WATCH_MIN_TURNOVER_RATIO = 1.5
VPD_WATCH_MAX_CANDIDATES = 10
VPD_SPLIT_SINGLE_DAY_PCT = 30.0


def _vpd_trimmed_mean(values: List[float], trim_top: float = 0.10) -> float:
    """상위 trim_top 구간을 제외한 절사평균. 과거 스파이크 1개의 영향을 줄인다."""
    clean = [float(v) for v in values if math.isfinite(float(v)) and float(v) >= 0]
    if not clean:
        return 0.0
    clean.sort()
    cut = int(len(clean) * trim_top)
    kept = clean[:len(clean) - cut] if cut > 0 else clean
    if not kept:
        return 0.0
    return float(np.mean(kept))


def _vpd_atr14(highs: List[float], lows: List[float], closes: List[float]) -> Optional[float]:
    """최근 14개 True Range 평균. 부족하면 None."""
    try:
        if len(highs) < 15 or len(lows) < 15 or len(closes) < 15:
            return None
        h = [float(v) for v in highs[-15:]]
        l = [float(v) for v in lows[-15:]]
        c = [float(v) for v in closes[-15:]]
        if not all(math.isfinite(v) and v > 0 for v in h + l + c):
            return None
        trs = []
        for i in range(1, 15):
            trs.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
        atr = float(np.mean(trs))
        return atr if math.isfinite(atr) and atr > 0 else None
    except (TypeError, ValueError, IndexError):
        return None


def scan_vpd_rolling(
    closes: List[float],
    volumes: List[float],
    lookback_days: int = VPD_ROLLING_LOOKBACK_DAYS,
    step_days: int = VPD_ROLLING_STEP_DAYS,
) -> Dict[str, Any]:
    """최근 20일 고정 대신 과거 윈도우를 슬라이딩해 엄격식 충족 시점을 찾는다.

    각 윈도우는 엄격식과 동일하게 disjoint 20/60 + 시작-종료 순가격변동으로
    판정한다. 겹침 과다집계를 막기 위해 step_days 간격으로 평가한다.
    """
    out: Dict[str, Any] = {
        "available": False,
        "windows_evaluated": 0,
        "match_count": 0,
        "best_offset": None,
        "best_volume_ratio": None,
        "best_price_change_pct": None,
        "matches": [],
        "reason": "",
    }
    try:
        n = min(len(closes or []), len(volumes or []))
        required = VPD_RECENT_DAYS + VPD_BASELINE_DAYS
        if n < required:
            out["reason"] = f"데이터 부족: 최소 {required}거래일 필요 (현재 {n}일)"
            return out
        max_offset = min(int(lookback_days), n - required)
        if max_offset < 0:
            out["reason"] = "롤링 탐색 범위 없음"
            return out
        step = max(1, int(step_days))
        matches = []
        evaluated = 0
        for offset in range(0, max_offset + 1, step):
            end_exclusive = n - offset
            try:
                cc = [float(x) for x in closes[end_exclusive - required:end_exclusive]]
                vv = [float(x) for x in volumes[end_exclusive - required:end_exclusive]]
            except (TypeError, ValueError, IndexError):
                continue
            if len(cc) < required or len(vv) < required:
                continue
            if not all(math.isfinite(x) for x in cc + vv):
                continue
            if any(x < 0 for x in vv):
                continue
            try:
                anchor_c = cc[-(VPD_RECENT_DAYS + 1)]
                latest_c = cc[-1]
            except IndexError:
                continue
            if not (math.isfinite(anchor_c) and math.isfinite(latest_c)):
                continue
            if anchor_c <= 0 or latest_c <= 0:
                continue
            recent_v = vv[-VPD_RECENT_DAYS:]
            base_v = vv[:-VPD_RECENT_DAYS]
            base_avg = float(np.mean(base_v))
            if base_avg <= 0:
                continue
            ratio = float(np.mean(recent_v)) / base_avg
            price_chg = (latest_c / anchor_c - 1.0) * 100.0
            evaluated += 1
            if ratio + 1e-12 >= VPD_MIN_VOLUME_RATIO and abs(price_chg) <= VPD_MAX_ABS_PRICE_CHANGE_PCT + 1e-12:
                matches.append({
                    "end_offset": offset,
                    "volume_ratio": round(ratio, 3),
                    "price_change_pct": round(price_chg, 3),
                })
        matches.sort(key=lambda m: (-m["volume_ratio"], abs(m["price_change_pct"])))
        out.update({
            "available": True,
            "windows_evaluated": evaluated,
            "match_count": len(matches),
            "best_offset": matches[0]["end_offset"] if matches else None,
            "best_volume_ratio": matches[0]["volume_ratio"] if matches else None,
            "best_price_change_pct": matches[0]["price_change_pct"] if matches else None,
            "matches": matches[:12],
            "reason": "과거 윈도우 엄격식 충족" if matches else "최근 탐색범위 내 과거 엄격식 충족 없음",
        })
        return out
    except Exception:
        out["reason"] = "롤링 계산 실패"
        return out


def score_vpd_watch(volume_ratio: float, price_change_pct: float, days_ge_2x: int) -> Dict[str, float]:
    """관찰 등급용 괴리점수. 엄격 임계(3배·±3%)를 바꾸지 않고 근접도를 점수화한다."""
    try:
        ratio = float(volume_ratio)
        move = abs(float(price_change_pct))
    except (TypeError, ValueError):
        return {"volume_score": 0.0, "flatness_score": 0.0, "divergence_score": 0.0}
    # 3배=100, 2배=66.7, 1배=33.3 선형
    volume_score = max(0.0, min(100.0, ratio / 3.0 * 100.0))
    flatness_score = max(0.0, min(100.0, (1.0 - move / 9.0) * 100.0))
    try:
        persist_bonus = min(6.0, max(0.0, float(days_ge_2x) - VPD_WATCH_MIN_DAYS_GE_2X))
    except (TypeError, ValueError):
        persist_bonus = 0.0
    divergence = max(0.0, min(100.0, volume_score * 0.6 + flatness_score * 0.4 + persist_bonus))
    return {
        "volume_score": round(volume_score, 2),
        "flatness_score": round(flatness_score, 2),
        "divergence_score": round(divergence, 2),
    }


def build_vpd_watch_signal(
    closes: List[float],
    volumes: List[float],
    highs: Optional[List[float]] = None,
    lows: Optional[List[float]] = None,
    strict_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Tier W 관찰 신호. strict_result는 읽기 전용으로 사용하고 재정의하지 않는다."""
    out: Dict[str, Any] = {
        "available": False,
        "eligible": False,
        "reasons": [],
        "volume_ratio": None,
        "price_change_pct": None,
        "baseline_median": None,
        "median_ratio": None,
        "baseline_trimmed_avg": None,
        "trimmed_ratio": None,
        "turnover_ratio": None,
        "days_ge_2x": 0,
        "days_ge_3x": 0,
        "max_consec_ge_2x": 0,
        "range_pct": None,
        "max_up_dev_pct": None,
        "max_down_dev_pct": None,
        "range_to_atr": None,
        "obv_flow": None,
        "clv_avg": None,
        "up_down_vol_ratio": None,
        "zero_volume_days_recent": 0,
        "zero_volume_days_baseline": 0,
        "split_suspect": False,
        "split_reason": "",
        "price_discontinuity": False,
        "volume_score": 0.0,
        "flatness_score": 0.0,
        "divergence_score": 0.0,
        "rolling": {"available": False, "match_count": 0},
    }
    try:
        strict = strict_result or detect_volume_price_divergence(closes or [], volumes or [])
    except Exception:
        out["reasons"] = ["엄격식 계산 실패"]
        return out
    if not strict.get("available"):
        out["reasons"] = [str(strict.get("reason") or "엄격식 데이터 부족")]
        return out
    try:
        ratio = float(strict.get("volume_ratio"))
        move = float(strict.get("price_change_pct"))
    except (TypeError, ValueError):
        out["reasons"] = ["엄격식 수치 오류"]
        return out
    out["volume_ratio"] = ratio
    out["price_change_pct"] = move

    # 정렬된 꼬리 재구성 (날짜 정렬 보존)
    try:
        has_hl = highs is not None and lows is not None and len(highs) >= 20 and len(lows) >= 20
        n = min(len(closes), len(volumes), len(highs) if has_hl else len(closes), len(lows) if has_hl else len(closes))
        required = VPD_RECENT_DAYS + VPD_BASELINE_DAYS
        if n < required:
            out["reasons"] = [f"관찰 증거 부족: 최소 {required}거래일 필요 (현재 {n}일)"]
            return out
        cc = [float(v) for v in closes[-n:]]
        vv = [float(v) for v in volumes[-n:]]
        hh = [float(v) for v in highs[-n:]] if has_hl else []
        ll = [float(v) for v in lows[-n:]] if has_hl else []
        anchor = cc[-(VPD_RECENT_DAYS + 1)]
        recent_v = vv[-VPD_RECENT_DAYS:]
        base_v = vv[-required:-VPD_RECENT_DAYS]
        recent_c = cc[-VPD_RECENT_DAYS:]
    except (TypeError, ValueError, IndexError):
        out["reasons"] = ["관찰 증거 형식 오류"]
        return out

    try:
        base_avg = float(np.mean(base_v))
        base_med = float(np.median(base_v))
        base_trim = _vpd_trimmed_mean(base_v, 0.10)
        recent_avg = float(np.mean(recent_v))
        out["baseline_median"] = round(base_med, 2)
        out["median_ratio"] = round(recent_avg / base_med, 3) if base_med > 0 else None
        out["baseline_trimmed_avg"] = round(base_trim, 2)
        out["trimmed_ratio"] = round(recent_avg / base_trim, 3) if base_trim > 0 else None
    except Exception:
        pass

    # 지속성: 같은 분모(baseline 평균) 기준 일수 카운트
    try:
        denom = float(strict.get("baseline_avg_volume") or 0.0) or float(np.mean(base_v))
        ge2 = [1 if v >= denom * 2.0 else 0 for v in recent_v]
        ge3 = [1 if v >= denom * 3.0 else 0 for v in recent_v]
        out["days_ge_2x"] = int(sum(ge2))
        out["days_ge_3x"] = int(sum(ge3))
        best = cur = 0
        for flag in ge2:
            cur = cur + 1 if flag else 0
            best = max(best, cur)
        out["max_consec_ge_2x"] = int(best)
        out["zero_volume_days_recent"] = int(sum(1 for v in recent_v if v == 0))
        out["zero_volume_days_baseline"] = int(sum(1 for v in base_v if v == 0))
    except Exception:
        pass

    # 거래대금 비율 (분할 가드용: 주식수만 늘고 금액이 안 늘면 분할 의심)
    try:
        anchor_c = cc[-(VPD_RECENT_DAYS + 1)]
        rec_t = float(np.mean([p * v for p, v in zip(recent_c, recent_v)]))
        base_c = cc[-required:-VPD_RECENT_DAYS]
        base_t = float(np.mean([p * v for p, v in zip(base_c, base_v)]))
        out["turnover_ratio"] = round(rec_t / base_t, 3) if base_t > 0 else None
    except Exception:
        out["turnover_ratio"] = None

    # 가격 강건성: 종가 2점 외에 기간 고저폭 표시
    if has_hl:
        try:
            recent_h = hh[-VPD_RECENT_DAYS:]
            recent_l = ll[-VPD_RECENT_DAYS:]
            hi = max(recent_h)
            lo = min(recent_l)
            if anchor > 0 and all(math.isfinite(v) for v in [hi, lo]):
                out["range_pct"] = round((hi - lo) / anchor * 100.0, 2)
                out["max_up_dev_pct"] = round((hi / anchor - 1.0) * 100.0, 2)
                out["max_down_dev_pct"] = round((lo / anchor - 1.0) * 100.0, 2)
            atr = _vpd_atr14(hh, ll, cc)
            if atr and atr > 0 and out["range_pct"] is not None:
                out["range_to_atr"] = round((hi - lo) / atr, 2)
        except Exception:
            pass
        # 수급: OBV 흐름·CLV·상승/하락 거래량 비대칭
        try:
            prev = anchor
            obv = 0.0
            up_v = 0.0
            down_v = 0.0
            clvs = []
            for px, h, l, v in zip(recent_c, hh[-VPD_RECENT_DAYS:], ll[-VPD_RECENT_DAYS:], recent_v):
                if px > prev:
                    obv += v
                    up_v += v
                elif px < prev:
                    obv -= v
                    down_v += v
                prev = px
                if h > l and all(math.isfinite(x) for x in [h, l, px]):
                    clvs.append(((px - l) - (h - px)) / (h - l))
            tot = float(sum(recent_v))
            out["obv_flow"] = round(obv / tot, 3) if tot > 0 else 0.0
            out["clv_avg"] = round(float(np.mean(clvs)), 3) if clvs else None
            if down_v > 0:
                out["up_down_vol_ratio"] = round(up_v / down_v, 3)
            elif up_v > 0:
                out["up_down_vol_ratio"] = None
        except Exception:
            pass

    # 이벤트 가드
    try:
        tail_len = min(len(cc), required + VPD_ROLLING_LOOKBACK_DAYS)
        tail = cc[-tail_len:]
        disc = False
        for i in range(1, len(tail)):
            if tail[i - 1] > 0 and math.isfinite(tail[i]) and math.isfinite(tail[i - 1]):
                if abs(tail[i] / tail[i - 1] - 1.0) * 100.0 > VPD_SPLIT_SINGLE_DAY_PCT:
                    disc = True
                    break
        out["price_discontinuity"] = disc
        split = False
        reason = ""
        if out["turnover_ratio"] is not None and ratio >= 2.5 and out["turnover_ratio"] < VPD_WATCH_MIN_TURNOVER_RATIO:
            split = True
            reason = "거래량 배수는 높으나 거래대금 비율이 낮아 주식수 착시 가능"
        elif disc and ratio >= 2.0:
            split = True
            reason = "단일일 30% 초과 가격 불연속과 거래량 급증이 겹침"
        out["split_suspect"] = split
        out["split_reason"] = reason
    except Exception:
        pass

    # 롤링 (엄격식 과거 충족)
    try:
        out["rolling"] = scan_vpd_rolling(cc, vv)
    except Exception:
        out["rolling"] = {"available": False, "match_count": 0}

    scores = score_vpd_watch(ratio, move, int(out.get("days_ge_2x") or 0))
    # 수급 가산(최대 +4): 종가위치·OBV가 매수쪽일 때만
    try:
        bonus = 0.0
        if out.get("clv_avg") is not None and float(out["clv_avg"]) > 0.1:
            bonus += 2.0
        if out.get("obv_flow") is not None and float(out["obv_flow"]) > 0.1:
            bonus += 2.0
        scores["divergence_score"] = round(min(100.0, float(scores["divergence_score"]) + bonus), 2)
    except (TypeError, ValueError):
        pass
    out.update(scores)
    out["available"] = True

    # 관찰 적격 (엄격 충족분은 제외해 중복 표시 방지)
    reasons = []
    if strict.get("is_match"):
        reasons.append("엄격 일치분은 Tier S에 표시")
    if ratio < VPD_WATCH_MIN_VOLUME_RATIO:
        reasons.append("관찰 거래량 기준 미달(2배)")
    if abs(move) > VPD_WATCH_MAX_PRICE_CHANGE_PCT:
        reasons.append("관찰 가격 기준 초과(±6%)")
    if int(out.get("days_ge_2x") or 0) < VPD_WATCH_MIN_DAYS_GE_2X:
        reasons.append("지속일수 부족(2배 이상 5일 미만)")
    if out.get("turnover_ratio") is None or float(out["turnover_ratio"]) < VPD_WATCH_MIN_TURNOVER_RATIO:
        reasons.append("거래대금 뒷받침 부족(1.5배 미만)")
    if out.get("split_suspect"):
        reasons.append("분할·착시 의심")
    if not has_hl:
        reasons.append("고저 증거 없음")
    out["reasons"] = reasons
    out["eligible"] = len(reasons) == 0
    return out


def is_scan_price_eligible(market: str, price: Any) -> bool:
    """스캔 출력 가격 상한 판정. US는 $70 초과 제외, KRX는 제한 없음."""
    if str(market or "").upper() != "US":
        return True
    try:
        value = float(price)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(value) or value <= 0:
        return False
    return value <= SCAN_US_MAX_PRICE


def apply_leader_promotion(
    cands: List[Dict],
    leader_map: Dict[str, Dict] | None,
    equity: float = 10_000_000.0,
    risk_pct: float = 1.0,
) -> Dict[str, int]:
    """리더 반전 BREAKOUT 후보를 진입 준비(READY)로 승격한다.

    스캔 엔진의 자체 돌파 체계와 리더 반전 체계는 진입가·손절가가 다르므로,
    승격 시 진입 트리거·손절가·이격률·포지션 사이징을 리더 기준으로 교체하고
    원래 값은 orig_* 로 보존한다 (투명성). 점수(BQS/FWS/NCS)는 손대지 않는다.

    규칙:
      - leader stage == BREAKOUT 일 때만 승격
      - EARNINGS_BLOCK(실적 대기)은 하드 게이트라 승격 제외
      - 진입가·손절가가 유한 양수이고 손절 < 진입일 때만 승격
    절대 raise하지 않는다.
    """
    promoted = 0
    try:
        for cd in cands or []:
            try:
                if not isinstance(cd, dict):
                    continue
                if cd.get("status") == "EARNINGS_BLOCK":
                    continue
                lr = ((leader_map or {}).get(cd.get("ticker")) or {})
                if not isinstance(lr, dict) or lr.get("stage") != "BREAKOUT":
                    continue
                try:
                    entry_f = float(lr.get("entry_trigger"))
                    stop_f = float(lr.get("stop_price"))
                    price_f = float(cd.get("price"))
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(entry_f) and math.isfinite(stop_f)
                        and math.isfinite(price_f)):
                    continue
                if entry_f <= 0 or stop_f <= 0 or stop_f >= entry_f or price_f <= 0:
                    continue
                cd["orig_status"] = cd.get("status")
                cd["orig_entry_trigger"] = cd.get("entry_trigger")
                cd["orig_stop_price"] = cd.get("stop_price")
                cd["status"] = "READY"
                cd["status_source"] = "leader_reversal"
                cd["entry_trigger"] = round(entry_f, 4)
                cd["stop_price"] = round(stop_f, 4)
                cd["distance_pct"] = round((entry_f - price_f) / price_f * 100, 2)
                try:
                    sizing = calculate_position_size(
                        equity, entry_f, stop_f,
                        cd.get("sleeve") or "CORE", risk_pct)
                    cd["shares"] = sizing.get("shares")
                    cd["risk_amount"] = sizing.get("risk_amount")
                    cd["risk_pct"] = sizing.get("risk_pct")
                    cd["total_cost"] = sizing.get("total_cost")
                except Exception:
                    pass
                promoted += 1
            except Exception:
                continue
    except Exception:
        pass
    return {"promoted": promoted}


# ── 데이터 구조 ───────────────────────────────────────────────────────────────

@dataclass
class TechnicalSnapshot:
    """단일 종목의 기술적 스냅샷 (Stage 2 입력)."""
    ticker:         str
    closes:         List[float]   # 일별 종가 (오래된→최신)
    highs:          List[float]
    lows:           List[float]
    volumes:        List[float]
    opens:          Optional[List[float]] = None

    # 계산된 기술지표 (OHLCV에서 도출)
    current_price:  float = 0.0
    ma200:          float = 0.0
    adx:            float = 0.0
    plus_di:        float = 0.0
    minus_di:       float = 0.0
    atr:            float = 0.0
    atr_pct:        float = 0.0
    vol_ratio:      float = 1.0
    high_20d:       float = 0.0
    efficiency:     float = 0.0   # 추세 효율성 (0~100)
    rs_vs_bench:    float = 0.0   # 상대강도

    # 고급
    weekly_adx:     float = 0.0
    ema20:          float = 0.0
    hurst:          Optional[float] = None
    bis_score:      float = 0.0
    chasing_20:     bool  = False
    chasing_55:     bool  = False
    median_atr_14:  float = 0.0
    atr_spiking:    bool  = False
    atr_collapsing: bool  = False
    volume_price_divergence: Dict[str, Any] = field(default_factory=dict)
    volume_price_watch: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StockUniverse:
    """스캔 대상 종목 정보."""
    ticker:  str
    name:    str
    sleeve:  str      # CORE / ETF / HIGH_RISK / HEDGE
    sector:  str = ""
    cluster: str = ""
    super_cluster: str = ""
    currency:str = "KRW"

    # 클러스터 집중도
    cluster_exposure_pct:      float = 0.0
    super_cluster_exposure_pct:float = 0.0
    max_cluster_pct:           float = 25.0
    max_super_cluster_pct:     float = 40.0

    # 실적
    days_to_earnings: Optional[int] = None


@dataclass
class ScanCandidate:
    """스캔 결과 단일 후보."""
    ticker:      str
    name:        str
    sleeve:      str
    sector:      str
    cluster:     str

    price:       float
    entry_trigger: float
    stop_price:  float
    distance_pct: float

    status:      str   # READY / WATCH / FAR / WAIT_PULLBACK / COOLDOWN / EARNINGS_BLOCK

    # 점수
    rank_score:  float
    bqs:         float
    fws:         float
    ncs:         float
    adjusted_ncs:float
    action_note: str

    # 품질
    quality_tier: str
    quality_multiplier: float

    # 필터 통과 여부
    passes_tech_filters:   bool = True
    passes_risk_gates:     bool = True
    passes_anti_chase:     bool = True

    # 상세 결과
    bqs_components: Dict[str, float] = field(default_factory=dict)
    fws_components: Dict[str, float] = field(default_factory=dict)
    filter_detail:  Dict[str, Any]  = field(default_factory=dict)
    anti_chase_reason: str = ""
    pullback_signal: Optional[Dict] = None

    # 포지션 사이징
    shares:      Optional[float] = None
    risk_amount: Optional[float] = None
    risk_pct:    Optional[float] = None
    total_cost:  Optional[float] = None

    hurst_exponent: Optional[float] = None
    hurst_warn:     bool = False

    scan_mode: str = "FULL"
    volume_price_divergence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    """runFullScan() 최종 반환값."""
    regime:         str
    vol_regime:     str
    candidates:     List[ScanCandidate]
    ready_count:    int = 0
    watch_count:    int = 0
    far_count:      int = 0
    total_scanned:  int = 0
    passed_filters: int = 0
    passed_risk:    int = 0
    passed_anti_chase: int = 0
    scan_mode:      str = "FULL"
    generated_at:   str = field(default_factory=lambda: datetime.now().isoformat())


# ── Stage 2: 기술적 필터 ──────────────────────────────────────────────────────

def run_technical_filters(
    price:      float,
    snap:       TechnicalSnapshot,
    sleeve:     str,
    scan_mode:  str = "FULL",
) -> Dict[str, Any]:
    """Stage 2: 기술적 조건 필터.

    모든 조건이 True여야 passes_all=True.

    조건:
      1. price > MA200       (상승 추세)
      2. ADX ≥ 20            (추세 존재)
      3. +DI > -DI           (상승 방향)
      4. ATR% < 상한         (변동성 적정)
      5. data_quality        (기본 데이터 유효)
      6. efficiency ≥ 30     (추세 효율성)
    """
    atr_cap = ATR_VOLATILITY_CAP_HIGH if sleeve == "HIGH_RISK" else ATR_VOLATILITY_CAP_ALL

    filters = {
        "price_above_ma200":    price > snap.ma200 if snap.ma200 > 0 else False,
        "adx_above_20":         snap.adx >= 20.0,
        "plus_di_above_minus":  snap.plus_di > snap.minus_di,
        "atr_pct_below_cap":    snap.atr_pct < atr_cap,
        "data_quality":         snap.ma200 > 0 and snap.adx > 0,
        "efficiency_above_30":  snap.efficiency >= 30.0,
    }

    # Hurst 경고 (FULL 모드만, soft filter — 통과/탈락 미결정)
    hurst_warn = False
    if scan_mode == "FULL" and snap.hurst is not None:
        hurst_warn = snap.hurst < 0.5

    passes_all = all(filters.values())

    return {
        **filters,
        "hurst_warn":   hurst_warn,
        "passes_all":   passes_all,
    }


# ── Stage 3: 상태 분류 ────────────────────────────────────────────────────────

def classify_candidate(price: float, entry_trigger: float) -> str:
    """Stage 3: 후보 상태 분류.

    distance = (trigger - price) / price × 100
    ≤ 0% (이미 돌파): READY
    ≤ 2%: READY
    ≤ 3%: WATCH
    > 3%: FAR
    """
    if entry_trigger <= 0:
        return "FAR"
    distance = ((entry_trigger - price) / price) * 100
    if distance <= 2.0:
        return "READY"
    if distance <= 3.0:
        return "WATCH"
    return "FAR"


# ── Stage 4: 랭킹 (BQS 기반) ─────────────────────────────────────────────────

def rank_candidate(
    sleeve:  str,
    snap:    TechnicalSnapshot,
    status:  str,
    bqs:     float,
) -> float:
    """Stage 4: 후보 순위 점수 계산.

    BQS + Sleeve 우선순위 + 상태 보너스 + ATR·거래량 타이브레이커
    """
    score = float(SLEEVE_PRIORITY.get(sleeve, 0))

    if status == "READY":
        score += 30.0
    elif status in ("WATCH", "WAIT_PULLBACK"):
        score += 10.0

    # BQS 직접 반영 (0~100 → 0~50 스케일)
    score += bqs * 0.5

    # 보조 타이브레이커
    score += min(snap.adx, 50.0) * 0.3
    score += min(snap.vol_ratio, 3.0) * 5.0
    score += min(snap.efficiency, 100.0) * 0.2
    score += min(snap.rs_vs_bench, 100.0) * 0.1

    return round(score * 100) / 100


# ── Stage 5: 리스크 게이트 ───────────────────────────────────────────────────

def validate_risk_gates(
    candidate: StockUniverse,
    total_cost: float,
    risk_dollars: float,
    portfolio_equity: float,
    existing_positions: List[Dict],
) -> List[Dict[str, Any]]:
    """Stage 5: 리스크 한도 검증.

    검증 항목:
      1. 단일 포지션 최대 비중 (슬리브별)
      2. 섹터 집중도 한도
      3. 클러스터 집중도 한도
      4. 전체 오픈 리스크 한도
      5. 포지션 수 한도

    Returns:
        게이트별 결과 리스트 [{"name", "passed", "reason"}]
    """
    results = []

    # 슬리브별 단일 포지션 최대 비중
    max_position_pct = {
        "CORE":      15.0,
        "ETF":       20.0,
        "HIGH_RISK": 8.0,
        "HEDGE":     10.0,
    }.get(candidate.sleeve, 15.0)

    pos_pct = (total_cost / portfolio_equity * 100) if portfolio_equity > 0 else 0
    results.append({
        "name":   "position_size_cap",
        "passed": pos_pct <= max_position_pct,
        "reason": f"포지션 비중 {pos_pct:.1f}% / 상한 {max_position_pct:.0f}%",
    })

    # 섹터 집중도 (동일 섹터 기존 포지션)
    sector_val = sum(
        p.get("value", 0) for p in existing_positions
        if p.get("sector") == candidate.sector
    )
    sector_pct = ((sector_val + total_cost) / portfolio_equity * 100) if portfolio_equity > 0 else 0
    results.append({
        "name":   "sector_concentration",
        "passed": sector_pct <= 40.0,
        "reason": f"섹터 집중도 {sector_pct:.1f}% / 상한 40%",
    })

    # 전체 오픈 리스크 한도 (자본 대비 2%)
    total_open_risk = sum(p.get("risk_dollars", 0) for p in existing_positions) + risk_dollars
    open_risk_pct = (total_open_risk / portfolio_equity * 100) if portfolio_equity > 0 else 0
    results.append({
        "name":   "open_risk_cap",
        "passed": open_risk_pct <= 6.0,
        "reason": f"전체 오픈 리스크 {open_risk_pct:.1f}% / 상한 6%",
    })

    return results


# ── Stage 6: 추격 방지 + 풀백 컨티뉴에이션 ──────────────────────────────────

def check_anti_chase(
    price:          float,
    entry_trigger:  float,
    atr:            float,
    is_monday:      bool = False,
) -> Dict[str, Any]:
    """Stage 6a: ext_ATR 기반 추격 방지 가드.

    ext_ATR = (price - entry_trigger) / ATR
    > 0.8 → WAIT_PULLBACK 강제
    """
    if atr <= 0:
        return {"passed": True, "reason": "ATR 데이터 없음", "ext_atr": 0.0}

    ext_atr = (price - entry_trigger) / atr

    if ext_atr > ANTI_CHASE_EXT_ATR_HARD:
        return {
            "passed":   False,
            "reason":   f"WAIT_PULLBACK — ext_atr {ext_atr:.2f} > {ANTI_CHASE_EXT_ATR_HARD}",
            "ext_atr":  round(ext_atr, 3),
        }

    # 갭 추격 (퍼센트 기반)
    gap_pct = 3.0 if is_monday else 4.0
    pct_above = ((price / entry_trigger) - 1) * 100 if entry_trigger > 0 else 0
    if price >= entry_trigger and pct_above > gap_pct:
        return {
            "passed":   False,
            "reason":   f"갭 추격 {pct_above:.1f}% > {gap_pct}% 기준",
            "ext_atr":  round(ext_atr, 3),
        }

    return {
        "passed":   True,
        "reason":   f"OK — {ext_atr:.2f} ATR gap, {pct_above:.1f}% above trigger",
        "ext_atr":  round(ext_atr, 3),
    }


def check_pullback_continuation(
    price:  float,
    high20: float,
    ema20:  float,
    atr:    float,
    low:    float,
) -> Optional[Dict]:
    """Stage 6b: 풀백 컨티뉴에이션 진입 신호 (Mode B).

    WAIT_PULLBACK 상태에서만 평가.
    anchor = max(HH20, EMA20)
    구간  = anchor ± 0.25×ATR
    조건:
      - price가 구간 내 혹은 아래로 풀백했다가 close가 구간 상단 위로 회복

    Returns:
        진입 신호 dict 또는 None
    """
    anchor = max(high20, ema20)
    if anchor <= 0 or atr <= 0:
        return None

    zone_low  = anchor - PULLBACK_ZONE_ATR_FACTOR * atr
    zone_high = anchor + PULLBACK_ZONE_ATR_FACTOR * atr

    # low가 zone 안쪽 또는 아래에 찍혔고, 현재 close가 zone_high 위
    touched_zone = low <= zone_high
    close_above  = price >= zone_high

    if touched_zone and close_above:
        entry_price = zone_high
        stop_price  = zone_low - atr * 0.5
        return {
            "triggered":   True,
            "mode":        "PULLBACK_CONTINUATION",
            "anchor":      round(anchor, 4),
            "zone_low":    round(zone_low, 4),
            "zone_high":   round(zone_high, 4),
            "entry_price": round(entry_price, 4),
            "stop_price":  round(stop_price, 4),
            "reason":      f"풀백 구간({zone_low:.2f}~{zone_high:.2f}) 터치 후 close({price:.2f}) 회복",
        }
    return None


# ── Stage 7: 포지션 사이징 ───────────────────────────────────────────────────

def calculate_position_size(
    equity:        float,
    entry_price:   float,
    stop_price:    float,
    sleeve:        str,
    risk_pct:      float = 1.0,   # 자본 대비 리스크 비율 (%)
) -> Dict[str, float]:
    """Stage 7: ATR 기반 포지션 사이징.

    shares = (equity × risk_pct%) / (entry - stop)
    """
    if entry_price <= 0 or stop_price >= entry_price:
        return {"shares": 0, "risk_amount": 0, "risk_pct": 0, "total_cost": 0}

    risk_per_share = entry_price - stop_price
    risk_budget    = equity * (risk_pct / 100.0)
    shares         = risk_budget / risk_per_share

    # 슬리브별 최대 포지션 비중
    max_pct = {
        "CORE":      15.0,
        "ETF":       20.0,
        "HIGH_RISK": 8.0,
        "HEDGE":     10.0,
    }.get(sleeve, 15.0)
    max_cost = equity * (max_pct / 100.0)
    total_cost = shares * entry_price

    if total_cost > max_cost:
        shares    = max_cost / entry_price
        total_cost = max_cost

    risk_amount = shares * risk_per_share
    return {
        "shares":      round(shares, 4),
        "risk_amount": round(risk_amount, 2),
        "risk_pct":    round(risk_amount / equity * 100, 3) if equity > 0 else 0,
        "total_cost":  round(total_cost, 2),
    }


# ── 적응형 ATR 버퍼 (hybrid_signals.py 래핑) ─────────────────────────────────

def _compute_entry_trigger(
    high_20d:   float,
    atr:        float,
    atr_pct:    float,
    vol_regime: str = VOL_NORMAL,
) -> float:
    """적응형 ATR 버퍼를 사용해 진입 트리거 계산."""
    try:
        result = adaptive_atr_buffer(high_20d, atr, atr_pct, vol_regime)
        return result.get("entry_trigger", high_20d)
    except Exception:
        # fallback: 고정 10% 버퍼
        return high_20d + 0.10 * atr


# ── 기술지표 계산 헬퍼 ────────────────────────────────────────────────────────

def _calc_ma(prices: List[float], period: int) -> float:
    if len(prices) < period:
        return 0.0
    return float(np.mean(prices[-period:]))


def _calc_efficiency(closes: List[float], period: int = 20) -> float:
    """추세 효율성: 직선 거리 / 총 이동 거리 (0~100)."""
    if len(closes) < period:
        return 0.0
    seg = closes[-period:]
    total_path = sum(abs(seg[i] - seg[i-1]) for i in range(1, len(seg)))
    net_move   = abs(seg[-1] - seg[0])
    if total_path == 0:
        return 0.0
    return min(net_move / total_path * 100, 100.0)


def _calc_vol_ratio(volumes: List[float], period: int = 21) -> float:
    """현재 거래량 / 20일 평균 거래량."""
    if len(volumes) < period:
        return 1.0
    avg = float(np.mean(volumes[-period:-1]))
    if avg <= 0:
        return 1.0
    return round(volumes[-1] / avg, 3)


def _build_snapshot(
    snap_input: TechnicalSnapshot,
    universe:   StockUniverse,
    regime:     str,
    vol_regime: str,
    regime_stable: bool,
    dual_aligned:  bool,
) -> SnapshotRow:
    """TechnicalSnapshot + StockUniverse → SnapshotRow (dual_score_v2 입력)."""
    chasing_20, chasing_55 = compute_chasing_flags(snap_input.closes, snap_input.highs)

    return SnapshotRow(
        ticker=snap_input.ticker,
        sleeve=universe.sleeve,
        status="FAR",  # Stage 3에서 갱신됨
        close=snap_input.current_price,
        atr_14=snap_input.atr,
        atr_pct=snap_input.atr_pct,
        adx_14=snap_input.adx,
        plus_di=snap_input.plus_di,
        minus_di=snap_input.minus_di,
        vol_ratio=snap_input.vol_ratio,
        market_regime=regime,
        market_regime_stable=regime_stable,
        vol_regime=vol_regime,
        dual_regime_aligned=dual_aligned,
        distance_to_20d_high_pct=max(0.0, (snap_input.high_20d - snap_input.current_price) / snap_input.current_price * 100) if snap_input.current_price > 0 else 0.0,
        chasing_20_last5=chasing_20,
        chasing_55_last5=chasing_55,
        atr_spiking=snap_input.atr_spiking,
        atr_collapsing=snap_input.atr_collapsing,
        rs_vs_benchmark_pct=snap_input.rs_vs_bench,
        days_to_earnings=universe.days_to_earnings,
        cluster_name=universe.cluster,
        super_cluster_name=universe.super_cluster,
        cluster_exposure_pct=universe.cluster_exposure_pct,
        super_cluster_exposure_pct=universe.super_cluster_exposure_pct,
        max_cluster_pct=universe.max_cluster_pct,
        max_super_cluster_pct=universe.max_super_cluster_pct,
        weekly_adx=snap_input.weekly_adx,
        bis_score=snap_input.bis_score,
        hurst_exponent=snap_input.hurst or 0.0,
        quality_tier="unknown",
        momentum_score_multiplier=1.0,
    )


# ── 메인 스캔 파이프라인 ──────────────────────────────────────────────────────

def run_full_scan(
    universe:           List[StockUniverse],
    snap_map:           Dict[str, TechnicalSnapshot],   # ticker → TechnicalSnapshot
    quality_map:        Dict[str, QualityFilterResult], # ticker → QMJ 결과
    regime:             str  = REGIME_SIDEWAYS,
    vol_regime:         str  = VOL_NORMAL,
    regime_stable:      bool = True,
    dual_aligned:       bool = False,
    portfolio_equity:   float = 10_000_000.0,
    existing_positions: List[Dict] = None,
    risk_pct_per_trade: float = 1.0,
    scan_mode:          str  = "FULL",
    is_monday:          bool = False,
) -> ScanResult:
    """7단계 종목 스캔 파이프라인.

    Args:
        universe:           분석 대상 종목 리스트
        snap_map:           종목별 TechnicalSnapshot
        quality_map:        종목별 QMJ 품질 결과
        regime:             시장 레짐 (BULLISH/BEARISH/SIDEWAYS)
        vol_regime:         변동성 레짐 (LOW_VOL/NORMAL_VOL/HIGH_VOL)
        regime_stable:      레짐 안정 여부
        dual_aligned:       SPY + VWRL 양쪽 강세 정렬 여부
        portfolio_equity:   포트폴리오 자본 (원화)
        existing_positions: 기존 오픈 포지션 리스트
        risk_pct_per_trade: 거래당 리스크 비율 (%, 기본 1%)
        scan_mode:          FULL / CORE_LITE
        is_monday:          월요일 여부 (추격 방지 강화)

    Returns:
        ScanResult
    """
    if existing_positions is None:
        existing_positions = []

    is_core_lite = scan_mode == "CORE_LITE"
    candidates: List[ScanCandidate] = []

    for stock in universe:
        ticker = stock.ticker
        snap   = snap_map.get(ticker)
        if snap is None:
            logger.warning("[Scan] %s: 스냅샷 없음 — 건너뜀", ticker)
            continue

        price = snap.current_price
        if price <= 0:
            continue

        # ── Stage 1: QMJ 품질 필터 ──────────────────────────────────────────
        qmj = quality_map.get(ticker)
        if qmj and not qmj.pass_filter:
            logger.info("[Scan] %s: QMJ 탈락 (%s)", ticker, qmj.quality_tier)
            continue

        quality_tier = qmj.quality_tier if qmj else "unknown"
        momentum_mult = TIER_MULTIPLIER.get(quality_tier, 0.5)

        # ── Stage 2: 기술적 필터 ────────────────────────────────────────────
        filter_result = run_technical_filters(price, snap, stock.sleeve, scan_mode)
        passes_tech = filter_result["passes_all"]
        hurst_warn  = filter_result.get("hurst_warn", False)

        # ATR 스파이크: SOFT_CAP (READY→WATCH 강등)
        atr_spike_action = "NONE"
        median_atr = snap.median_atr_14
        if not is_core_lite and median_atr > 0 and snap.atr >= median_atr * 1.3:
            atr_spike_action = "SOFT_CAP"
            snap.atr_spiking = True

        # ── 적응형 ATR 버퍼 → 진입 트리거 계산 ─────────────────────────────
        entry_trigger = _compute_entry_trigger(snap.high_20d, snap.atr, snap.atr_pct, vol_regime)
        stop_price    = entry_trigger - snap.atr * ATR_STOP_MULTIPLIER
        distance_pct  = ((entry_trigger - price) / price) * 100 if price > 0 else 999.0

        # ── Stage 3: 상태 분류 ──────────────────────────────────────────────
        status = classify_candidate(price, entry_trigger)

        # ATR 스파이크 시 READY → WATCH 강등
        if atr_spike_action == "SOFT_CAP" and status == "READY":
            status = "WATCH"

        # 실적 블록
        if not is_core_lite and stock.days_to_earnings is not None and stock.days_to_earnings <= 2:
            status = "EARNINGS_BLOCK"
            passes_tech = False

        # ── SnapshotRow 구성 (BQS/FWS 계산용) ──────────────────────────────
        row = _build_snapshot(snap, stock, regime, vol_regime, regime_stable, dual_aligned)
        row.quality_tier              = quality_tier
        row.momentum_score_multiplier = momentum_mult
        row.status                    = status

        # ── Stage 4: BQS/FWS/NCS 계산 + 랭킹 ──────────────────────────────
        bqs_res = compute_bqs(row)
        fws_res = compute_fws(row)
        pen_res = compute_penalties(row)
        ncs_res = compute_ncs(bqs_res["BQS"], fws_res["FWS"], pen_res)

        bqs = bqs_res["BQS"]
        fws = fws_res["FWS"]
        ncs = ncs_res["NCS"]
        adj_ncs = min(100.0, ncs * momentum_mult)
        a_note  = action_note(fws, ncs, pen_res["EarningsPenalty"])

        rank_score = rank_candidate(stock.sleeve, snap, status, bqs)

        # ── Stage 5: 리스크 게이트 ──────────────────────────────────────────
        passes_risk   = True
        risk_gate_res = []
        sizing        = {}

        if passes_tech and status not in ("FAR", "EARNINGS_BLOCK"):
            # 임시 포지션 사이징 (리스크 계산용)
            sizing = calculate_position_size(
                portfolio_equity, entry_trigger, stop_price, stock.sleeve, risk_pct_per_trade
            )
            risk_gate_res = validate_risk_gates(
                stock,
                sizing.get("total_cost", 0),
                sizing.get("risk_amount", 0),
                portfolio_equity,
                existing_positions,
            )
            passes_risk = all(g["passed"] for g in risk_gate_res)

        # ── Stage 6: 추격 방지 ──────────────────────────────────────────────
        passes_chase    = True
        anti_chase_res  = {"passed": True, "reason": "미평가", "ext_atr": 0.0}
        pullback_signal = None

        if passes_tech and status not in ("FAR", "EARNINGS_BLOCK", "COOLDOWN") and not is_core_lite:
            anti_chase_res = check_anti_chase(price, entry_trigger, snap.atr, is_monday)

            if not anti_chase_res["passed"]:
                status = "WAIT_PULLBACK"

                # 풀백 컨티뉴에이션 확인
                pullback_signal = check_pullback_continuation(
                    price, snap.high_20d, snap.ema20, snap.atr,
                    snap.lows[-1] if snap.lows else price,
                )
                if pullback_signal and pullback_signal.get("triggered"):
                    entry_trigger = pullback_signal["entry_price"]
                    stop_price    = pullback_signal["stop_price"]
                    distance_pct  = ((entry_trigger - price) / price) * 100
                    status        = "READY"
                    anti_chase_res["passed"] = True
                    anti_chase_res["reason"] = f"PULLBACK_CONTINUATION — {pullback_signal['reason']}"
                    # 포지션 재계산
                    sizing = calculate_position_size(
                        portfolio_equity, entry_trigger, stop_price, stock.sleeve, risk_pct_per_trade
                    )
            passes_chase = anti_chase_res["passed"]

        # ── Stage 7: 최종 후보 구성 ─────────────────────────────────────────
        candidate = ScanCandidate(
            ticker        = ticker,
            name          = stock.name,
            sleeve        = stock.sleeve,
            sector        = stock.sector,
            cluster       = stock.cluster,
            price         = round(price, 4),
            entry_trigger = round(entry_trigger, 4),
            stop_price    = round(stop_price, 4),
            distance_pct  = round(distance_pct, 2),
            status        = status,
            rank_score    = rank_score,
            bqs           = bqs,
            fws           = fws,
            ncs           = ncs,
            adjusted_ncs  = round(adj_ncs, 2),
            action_note   = a_note,
            quality_tier  = quality_tier,
            quality_multiplier = momentum_mult,
            passes_tech_filters   = passes_tech,
            passes_risk_gates     = passes_risk,
            passes_anti_chase     = passes_chase,
            bqs_components = bqs_res,
            fws_components = fws_res,
            filter_detail  = {
                **filter_result,
                "atr_spike_action": atr_spike_action,
                "risk_gates":       risk_gate_res,
            },
            anti_chase_reason = anti_chase_res.get("reason", ""),
            pullback_signal   = pullback_signal,
            shares      = sizing.get("shares"),
            risk_amount = sizing.get("risk_amount"),
            risk_pct    = sizing.get("risk_pct"),
            total_cost  = sizing.get("total_cost"),
            hurst_exponent = snap.hurst,
            hurst_warn     = hurst_warn,
            scan_mode      = scan_mode,
            volume_price_divergence = dict(snap.volume_price_divergence or {}),
        )
        candidates.append(candidate)

    # ── 최종 정렬: AdjustedNCS 기반 (READY → WATCH → FAR 순서 유지) ─────────
    status_order = {
        "READY": 0, "WATCH": 1, "WAIT_PULLBACK": 1,
        "COOLDOWN": 2, "EARNINGS_BLOCK": 2, "FAR": 3,
    }

    def _sort_key(c: ScanCandidate):
        # 이미 트리거된 것(price >= entry) 최우선
        triggered = 1 if c.passes_tech_filters and c.price >= c.entry_trigger else 0
        return (-triggered, status_order.get(c.status, 3), -c.adjusted_ncs, -c.rank_score)

    candidates.sort(key=_sort_key)

    # 집계
    passed = [c for c in candidates if c.passes_tech_filters]
    return ScanResult(
        regime          = regime,
        vol_regime      = vol_regime,
        candidates      = candidates,
        ready_count     = sum(1 for c in passed if c.status == "READY"),
        watch_count     = sum(1 for c in passed if c.status in ("WATCH", "WAIT_PULLBACK")),
        far_count       = sum(1 for c in candidates if c.status == "FAR"),
        total_scanned   = len(universe),
        passed_filters  = len(passed),
        passed_risk     = sum(1 for c in passed if c.passes_risk_gates),
        passed_anti_chase = sum(1 for c in passed if c.passes_anti_chase),
        scan_mode       = scan_mode,
    )


# ── 데이터 헬퍼: OHLCV에서 TechnicalSnapshot 자동 생성 ───────────────────────

def build_snapshot_from_ohlcv(
    ticker:       str,
    closes:       List[float],
    highs:        List[float],
    lows:         List[float],
    volumes:      List[float],
    opens:        Optional[List[float]] = None,
    bench_closes: Optional[List[float]] = None,
    weekly_adx:   float = 0.0,
) -> TechnicalSnapshot:
    """OHLCV 데이터에서 TechnicalSnapshot 자동 계산.

    hybrid_signals.compute_hybrid_score()를 내부적으로 사용해
    ADX, ATR, 레짐 등 주요 지표를 계산한다.

    Args:
        ticker:       종목 코드
        closes:       일별 종가 리스트 (오래된→최신)
        highs/lows/volumes: 각 OHLCV
        opens:        시가 (BIS 계산용, 선택)
        bench_closes: 벤치마크 종가 (RS 계산용, 선택)
        weekly_adx:   주간 ADX (외부에서 주입, 없으면 0)

    Returns:
        TechnicalSnapshot
    """
    n = len(closes)
    if n < 20:
        logger.warning("[Snapshot] %s: 데이터 부족 (%d봉)", ticker, n)
        return TechnicalSnapshot(ticker=ticker, closes=closes, highs=highs, lows=lows, volumes=volumes)

    # hybrid_signals의 compute_hybrid_score로 지표 계산
    try:
        hs = compute_hybrid_score(
            closes        = closes,
            highs         = highs,
            lows          = lows,
            volumes       = volumes,
            open_prices   = opens,
            bench_closes  = bench_closes,
        )
    except Exception as e:
        logger.error("[Snapshot] %s: hybrid_score 계산 실패: %s", ticker, e)
        hs = {}

    current_price = closes[-1]
    atr     = hs.get("atr")     or 0.0
    atr_pct = hs.get("atr_percent") or 0.0
    adx     = hs.get("adx")     or 0.0
    plus_di = hs.get("plus_di") or 0.0
    minus_di= hs.get("minus_di")or 0.0
    high_20d= hs.get("twenty_day_high") or (max(highs[-20:]) if n >= 20 else current_price)

    # MA200
    ma200 = _calc_ma(closes, 200)

    # EMA20 (지수이동평균)
    ema20 = _calc_ema(closes, 20)

    # 추세 효율성
    efficiency = _calc_efficiency(closes)

    # 거래량 비율
    vol_ratio = _calc_vol_ratio(volumes)

    # 거래량-가격 괴리(매집 탐지 보조 신호) — 엄격식은 고정
    volume_price_divergence = detect_volume_price_divergence(closes, volumes)
    try:
        volume_price_watch = build_vpd_watch_signal(
            closes, volumes, highs, lows, volume_price_divergence,
        )
    except Exception:
        volume_price_watch = {"available": False, "eligible": False, "reasons": ["관찰 계산 실패"]}

    # RS vs 벤치마크
    rs = 0.0
    if bench_closes and len(bench_closes) >= 2 and n >= 2:
        t_ret = (closes[-1] / closes[0] - 1) * 100 if closes[0] > 0 else 0
        b_ret = (bench_closes[-1] / bench_closes[0] - 1) * 100 if bench_closes[0] > 0 else 0
        rs = round(t_ret - b_ret, 2)

    # Hurst 지수 (50봉 이상 필요)
    hurst = calc_hurst_v2(closes) if n >= 50 else None

    # BIS (최신 캔들)
    bis = 0.0
    if opens and len(opens) >= 1 and len(volumes) >= 11:
        avg_vol_10 = float(np.mean(volumes[-11:-1])) if len(volumes) >= 11 else 0
        bis = float(compute_bis_from_candle(
            open_p   = opens[-1],
            high_p   = highs[-1],
            low_p    = lows[-1],
            close_p  = closes[-1],
            volume   = volumes[-1],
            avg_vol_10d = avg_vol_10,
        ))

    # ATR 스파이크/붕괴 감지
    atr_spiking = hs.get("atr_spiking", False)
    atr_collapsing = hs.get("atr_collapsing", False)

    # 중앙값 ATR (스파이크 감지 기준)
    median_atr = 0.0
    if n >= 35:
        from .hybrid_signals import _calc_atr
        old_atr = _calc_atr(highs[-35:-20], lows[-35:-20], closes[-35:-20])
        if old_atr:
            median_atr = old_atr

    return TechnicalSnapshot(
        ticker          = ticker,
        closes          = closes,
        highs           = highs,
        lows            = lows,
        volumes         = volumes,
        opens           = opens,
        current_price   = current_price,
        ma200           = ma200,
        adx             = adx,
        plus_di         = plus_di,
        minus_di        = minus_di,
        atr             = atr,
        atr_pct         = atr_pct,
        vol_ratio       = vol_ratio,
        high_20d        = high_20d,
        efficiency      = efficiency,
        rs_vs_bench     = rs,
        weekly_adx      = weekly_adx,
        ema20           = ema20,
        hurst           = hurst,
        bis_score       = bis,
        median_atr_14   = median_atr,
        atr_spiking     = atr_spiking,
        atr_collapsing  = atr_collapsing,
        volume_price_divergence = volume_price_divergence,
        volume_price_watch = volume_price_watch,
    )


def _calc_ema(prices: List[float], period: int) -> float:
    """지수이동평균 계산."""
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    k = 2.0 / (period + 1)
    ema = float(np.mean(prices[:period]))
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
    return round(ema, 4)
