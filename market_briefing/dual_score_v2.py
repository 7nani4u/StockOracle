"""
dual_score_v2.py — HybridTurtle-v6.0 Dual Score Engine (Python 이식 v2)

BQS (Breakout Quality Score)  : 브레이크아웃 품질 점수  (0-100, 높을수록 좋음)
FWS (Fatal Weakness Score)    : 치명적 약점 점수        (0-100, 높을수록 위험)
NCS (Net Composite Score)     : 순복합 점수             (0-100, 높을수록 좋음)

v2 추가 사항 (hybrid_signals.py 대비):
  - BQS: 주간 ADX 보너스, Hurst 보너스(로그 수익률 R/S), BIS 전달 지원
  - BQS: calcDualRegimeScore — dual_aligned(SPY+VWRL 양쪽 강세 정렬) 반영
  - FWS: 추격 확장 리스크 (chasing_20_last5 / chasing_55_last5)
  - NCS: 클러스터 패널티, 슈퍼클러스터 패널티, 실적 패널티 (상한 40점)
  - QMJ 품질 스코어와 연동되는 momentum_multiplier 지원
  - actionNote() — AUTO_YES / Conditional / Auto-No 분류 + 실적 주의 텍스트

원본 TypeScript:
  src/lib/dual-score.ts  (HybridTurtle-v6.0)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── 상수 ──────────────────────────────────────────────────────────────────────

REGIME_BULLISH  = "BULLISH"
REGIME_BEARISH  = "BEARISH"
REGIME_SIDEWAYS = "SIDEWAYS"
REGIME_NEUTRAL  = "NEUTRAL"

VOL_LOW    = "LOW_VOL"
VOL_NORMAL = "NORMAL_VOL"
VOL_HIGH   = "HIGH_VOL"


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if not math.isfinite(v):
        return lo
    return max(lo, min(hi, v))


def _safe(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        n = float(v)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError):
        return default


def _bool(v, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _r2(v: float) -> float:
    return round(v * 100) / 100


# ── 스냅샷 행 타입 ─────────────────────────────────────────────────────────────

@dataclass
class SnapshotRow:
    """단일 종목 스냅샷 — BQS/FWS 계산에 필요한 모든 피처.

    None은 '데이터 없음'을 의미하며 각 sub-score 함수가 기본값으로 처리한다.
    """
    ticker: str = ""
    name:   str = ""
    sleeve: str = "CORE"       # CORE / ETF / HIGH_RISK / HEDGE
    status: str = "FAR"        # READY / WATCH / FAR / WAIT_PULLBACK / COOLDOWN

    # ── 가격 / 기술 지표 ──────────────────────────────────────────────────────
    close:    float = 0.0
    atr_14:   float = 0.0
    atr_pct:  float = 0.0      # ATR / close × 100 (%)
    adx_14:   float = 0.0
    plus_di:  float = 0.0
    minus_di: float = 0.0
    vol_ratio:float = 1.0      # 오늘 거래량 / 20일 평균 거래량
    dollar_vol_20: Optional[float] = None

    # ── 레짐 ──────────────────────────────────────────────────────────────────
    market_regime:        str  = REGIME_NEUTRAL
    market_regime_stable: bool = True
    vol_regime:           str  = VOL_NORMAL
    dual_regime_aligned:  bool = False   # SPY + VWRL 양쪽 강세 정렬 여부

    # ── 가격 구간 ─────────────────────────────────────────────────────────────
    high_20:                  Optional[float] = None
    high_55:                  Optional[float] = None
    distance_to_20d_high_pct: float = 0.0
    distance_to_55d_high_pct: Optional[float] = None
    entry_trigger:            float = 0.0
    stop_level:               float = 0.0

    # ── 추격 방지 플래그 ──────────────────────────────────────────────────────
    chasing_20_last5: bool = False   # 최근 5일 중 20일 고점 추격 여부
    chasing_55_last5: bool = False   # 최근 5일 중 55일 고점 추격 여부

    # ── ATR 상태 ──────────────────────────────────────────────────────────────
    atr_spiking:         bool = False
    atr_collapsing:      bool = False
    atr_compression_ratio: Optional[float] = None

    # ── 상대강도 ──────────────────────────────────────────────────────────────
    rs_vs_benchmark_pct: float = 0.0   # vs SPY/KOSPI200

    # ── 실적 ──────────────────────────────────────────────────────────────────
    days_to_earnings:    Optional[int]  = None
    earnings_in_next_5d: bool           = False

    # ── 클러스터 집중도 ───────────────────────────────────────────────────────
    cluster_name:              str   = ""
    super_cluster_name:        str   = ""
    cluster_exposure_pct:      float = 0.0
    super_cluster_exposure_pct:float = 0.0
    max_cluster_pct:           float = 0.0
    max_super_cluster_pct:     float = 0.0

    # ── 고급 지표 ─────────────────────────────────────────────────────────────
    weekly_adx:     float = 0.0    # 주간 ADX (MTF 확인용)
    bis_score:      float = 0.0    # Breakout Integrity Score (0~15)
    hurst_exponent: float = 0.0    # Hurst 지수 (0~1, >0.5=추세)

    # ── QMJ 품질 필터 결과 ────────────────────────────────────────────────────
    quality_tier:               str   = "unknown"   # high/medium/low/junk/unknown
    momentum_score_multiplier:  float = 1.0         # QMJ에서 주어지는 배율


# ── BQS 서브점수 ───────────────────────────────────────────────────────────────

def _bqs_trend(row: SnapshotRow) -> float:
    """추세 강도: ADX 기반 0~25점."""
    adx = _safe(row.adx_14)
    return 25.0 * _clamp((adx - 15.0) / 20.0, 0.0, 1.0)


def _bqs_direction(row: SnapshotRow) -> float:
    """방향 지배력: +DI − -DI 기반 0~10점."""
    spread = _safe(row.plus_di) - _safe(row.minus_di)
    return 10.0 * _clamp(spread / 25.0, 0.0, 1.0)


def _bqs_volatility(row: SnapshotRow) -> float:
    """변동성 건전성: ATR% 기반 0~15점.
    1% 미만 — 점진적 증가, 1~4% — 만점, 4~6% — 점진적 감소, 6% 초과 — 0점.
    """
    atr_pct = _safe(row.atr_pct)
    if atr_pct < 1.0:
        return 15.0 * _clamp(atr_pct / 1.0, 0.0, 1.0)
    if atr_pct <= 4.0:
        return 15.0
    if atr_pct <= 6.0:
        return 15.0 * _clamp(1.0 - (atr_pct - 4.0) / 2.0, 0.0, 1.0)
    return 0.0


def _bqs_proximity(row: SnapshotRow) -> float:
    """20일 고점 근접도: 0~15점. 거리 0% → 15점, 거리 3% → 0점."""
    d20 = row.distance_to_20d_high_pct
    d55 = row.distance_to_55d_high_pct
    dist = _safe(d20 if d20 is not None else d55)
    return 15.0 * _clamp(1.0 - dist / 3.0, 0.0, 1.0)


def calc_dual_regime_score(row: SnapshotRow) -> float:
    """이중 레짐 점수 (DRS): -10 ~ +20.

    BEARISH:  -10 (항상)
    SIDEWAYS:   0
    BULLISH:
        HIGH_VOL:            +10
        dual_aligned=True:
            LOW_VOL:         +20
            NORMAL_VOL:      +15
        dual_aligned=False:  +10
    """
    regime     = (row.market_regime or REGIME_NEUTRAL).upper()
    vol_regime = (row.vol_regime or VOL_NORMAL).upper()
    dual       = _bool(row.dual_regime_aligned, False)

    if regime == REGIME_BEARISH:
        return -10.0
    if regime in (REGIME_SIDEWAYS, REGIME_NEUTRAL):
        return 0.0
    if regime == REGIME_BULLISH:
        if vol_regime == VOL_HIGH:
            return 10.0
        if dual:
            return 20.0 if vol_regime == VOL_LOW else 15.0
        return 10.0
    return 0.0


def _bqs_rs(row: SnapshotRow) -> float:
    """상대강도: 벤치마크 대비 RS% 기반 0~15점."""
    rs = _safe(row.rs_vs_benchmark_pct)
    return 15.0 * _clamp((rs + 5.0) / 20.0, 0.0, 1.0)


def _bqs_vol_bonus(row: SnapshotRow) -> float:
    """거래량 보너스: vol_ratio > 1.2 시 최대 +5점."""
    vr = _safe(row.vol_ratio, 1.0)
    if vr > 1.2:
        return 5.0 * _clamp((vr - 1.2) / 0.6, 0.0, 1.0)
    return 0.0


def _bqs_weekly_adx(row: SnapshotRow) -> float:
    """주간 ADX 보너스: 상위 타임프레임 추세 확인.

    ≥ 30: +10 (강한 주간 추세)
    ≥ 25: +5  (보통)
    < 20: -5  (추세 없음)
    데이터 없음(0): 0 (중립)
    """
    w_adx = _safe(row.weekly_adx)
    if w_adx == 0:
        return 0.0
    if w_adx >= 30:
        return 10.0
    if w_adx >= 25:
        return 5.0
    if w_adx < 20:
        return -5.0
    return 0.0


def _bqs_hurst(row: SnapshotRow) -> float:
    """Hurst 지수 보너스: 추세 지속성 검증.

    H ≥ 0.7: +8  (강한 지속적 추세)
    H ≥ 0.6: +5  (보통)
    H ≥ 0.5: +2  (약한 추세)
    H < 0.5:  0  (평균회귀 — 보너스 없음)
    데이터 없음(0): 0 (중립)
    """
    h = _safe(row.hurst_exponent)
    if h == 0:
        return 0.0
    if h >= 0.7:
        return 8.0
    if h >= 0.6:
        return 5.0
    if h >= 0.5:
        return 2.0
    return 0.0


def compute_bqs(row: SnapshotRow) -> dict:
    """BQS: 브레이크아웃 품질 점수 (0~100).

    서브점수 합산 후 clamp(0,100) 적용.
    이론 범위: -15 ~ 148 → clamp 후 0~100.
    """
    trend    = _bqs_trend(row)
    direction= _bqs_direction(row)
    vol      = _bqs_volatility(row)
    prox     = _bqs_proximity(row)
    tailwind = calc_dual_regime_score(row)
    rs       = _bqs_rs(row)
    vol_b    = _bqs_vol_bonus(row)
    w_adx_b  = _bqs_weekly_adx(row)
    bis      = _safe(row.bis_score)           # 0~15, 브레이크아웃 무결성
    hurst    = _bqs_hurst(row)

    bqs = _clamp(trend + direction + vol + prox + tailwind + rs + vol_b + w_adx_b + bis + hurst)

    return {
        "bqs_trend":       _r2(trend),
        "bqs_direction":   _r2(direction),
        "bqs_volatility":  _r2(vol),
        "bqs_proximity":   _r2(prox),
        "bqs_tailwind":    _r2(tailwind),
        "bqs_rs":          _r2(rs),
        "bqs_vol_bonus":   _r2(vol_b),
        "bqs_weekly_adx":  _r2(w_adx_b),
        "bqs_bis":         _r2(bis),
        "bqs_hurst":       _r2(hurst),
        "BQS":             _r2(bqs),
    }


# ── FWS 서브점수 ───────────────────────────────────────────────────────────────

def _fws_volume(row: SnapshotRow) -> float:
    """거래량 리스크: vol_ratio 낮을수록 위험. 0~30점."""
    vr = _safe(row.vol_ratio, 1.0)
    return 30.0 * _clamp(1.0 - (vr - 0.6) / 0.6, 0.0, 1.0)


def _fws_extension(row: SnapshotRow) -> float:
    """추격 확장 리스크: 20일/55일 고점 추격 여부 기반.

    양쪽 모두: 25점
    한쪽만:    15점
    없음:       0점
    """
    c20 = _bool(row.chasing_20_last5)
    c55 = _bool(row.chasing_55_last5)
    if c20 and c55:
        return 25.0
    if c20 or c55:
        return 15.0
    return 0.0


def _fws_marginal_trend(row: SnapshotRow) -> float:
    """추세 한계 리스크: ADX가 낮을수록 추세가 취약. 0~10점."""
    adx = _safe(row.adx_14)
    if adx < 20:
        return 10.0
    if adx <= 25:
        return 7.0
    if adx <= 30:
        return 3.0
    return 0.0


def _fws_vol_shock(row: SnapshotRow) -> float:
    """변동성 충격 리스크: ATR 스파이크/붕괴 시 10점.

    OVERLAP-02: scan-engine이 이미 spiking 종목을 강등하므로
    스파이크 패널티를 20→10으로 축소(이중 감점 방지).
    """
    if _bool(row.atr_spiking):
        return 10.0
    if _bool(row.atr_collapsing):
        return 10.0
    return 0.0


def _fws_regime_instability(row: SnapshotRow) -> float:
    """레짐 불안정 리스크: 레짐이 불안정하면 10점."""
    return 0.0 if _bool(row.market_regime_stable, True) else 10.0


def compute_fws(row: SnapshotRow) -> dict:
    """FWS: 치명적 약점 점수 (0~100, 높을수록 위험)."""
    vol      = _fws_volume(row)
    ext      = _fws_extension(row)
    marginal = _fws_marginal_trend(row)
    shock    = _fws_vol_shock(row)
    regime   = _fws_regime_instability(row)

    fws = _clamp(vol + ext + marginal + shock + regime)

    return {
        "fws_volume":            _r2(vol),
        "fws_extension":         _r2(ext),
        "fws_marginal_trend":    _r2(marginal),
        "fws_vol_shock":         _r2(shock),
        "fws_regime_instability":_r2(regime),
        "FWS":                   _r2(fws),
    }


# ── 패널티 ────────────────────────────────────────────────────────────────────

def compute_earnings_penalty(row: SnapshotRow) -> float:
    """실적 패널티: 발표일 근접할수록 증가.

    ≤1일: 20점 / ≤3일: 15점 / ≤5일: 10점
    days_to_earnings 없고 earnings_in_next_5d=True: 12점
    """
    d = row.days_to_earnings
    if d is not None:
        try:
            days = int(d)
            if days <= 1:
                return 20.0
            if days <= 3:
                return 15.0
            if days <= 5:
                return 10.0
            return 0.0
        except (TypeError, ValueError):
            pass
    if _bool(row.earnings_in_next_5d):
        return 12.0
    return 0.0


def compute_cluster_penalty(row: SnapshotRow) -> float:
    """클러스터 집중도 패널티.

    exposure/max ≤ 0.8:  0점 (여유 있음)
    0.8 ~ 1.0:           선형 증가 0~20점
    > 1.0:               20 + 초과분 × 30점 (급격한 페널티)
    """
    exposure = _safe(row.cluster_exposure_pct)
    mx       = _safe(row.max_cluster_pct)
    if mx <= 0:
        return 0.0
    x = exposure / mx
    if x <= 0.8:
        return 0.0
    if x <= 1.0:
        return 20.0 * (x - 0.8) / 0.2
    return 20.0 + 30.0 * (x - 1.0)


def compute_super_cluster_penalty(row: SnapshotRow) -> float:
    """슈퍼클러스터 집중도 패널티 (클러스터보다 엄격).

    ≤ 0.8:  0점
    0.8~1.0: 0~25점
    > 1.0:   25 + 초과분 × 40점
    """
    exposure = _safe(row.super_cluster_exposure_pct)
    mx       = _safe(row.max_super_cluster_pct)
    if mx <= 0:
        return 0.0
    x = exposure / mx
    if x <= 0.8:
        return 0.0
    if x <= 1.0:
        return 25.0 * (x - 0.8) / 0.2
    return 25.0 + 40.0 * (x - 1.0)


def compute_penalties(row: SnapshotRow) -> dict:
    ep  = compute_earnings_penalty(row)
    cp  = compute_cluster_penalty(row)
    scp = compute_super_cluster_penalty(row)
    return {
        "EarningsPenalty":     _r2(ep),
        "ClusterPenalty":      _r2(cp),
        "SuperClusterPenalty": _r2(scp),
    }


# ── NCS ──────────────────────────────────────────────────────────────────────

def compute_ncs(bqs: float, fws: float, penalties: dict) -> dict:
    """NCS: 순복합 점수 (0~100).

    공식: NCS = clamp(BQS - 0.8*FWS + 10) - cappedPenalty
    패널티 총합 상한: 40점 (과도한 중복 패널티 방지)
    """
    base_ncs = _clamp(bqs - 0.8 * fws + 10.0)
    total_penalty = (
        _safe(penalties.get("EarningsPenalty", 0))
        + _safe(penalties.get("ClusterPenalty", 0))
        + _safe(penalties.get("SuperClusterPenalty", 0))
    )
    capped = min(total_penalty, 40.0)
    ncs = _clamp(base_ncs - capped)
    return {
        "BaseNCS": _r2(base_ncs),
        "NCS":     _r2(ncs),
    }


def action_note(fws: float, ncs: float, earnings_pen: float) -> str:
    """매매 액션 분류 + 설명 텍스트."""
    if fws > 65.0:
        cls = "Auto-No (취약 종목 — 진입 지양)"
    elif ncs >= 70.0 and fws <= 30.0:
        cls = "Auto-Yes (고품질 브레이크아웃)"
    else:
        cls = "Conditional (거래량·캔들 확인 필요)"

    if earnings_pen > 0:
        return f"{cls} | 실적 위험: -{int(earnings_pen)}점"
    return cls


# ── 전체 행 점수 계산 ──────────────────────────────────────────────────────────

def score_row(row: SnapshotRow) -> dict:
    """단일 SnapshotRow → 전체 BQS/FWS/NCS 결과 dict."""
    bqs_result = compute_bqs(row)
    fws_result = compute_fws(row)
    pen_result = compute_penalties(row)
    ncs_result = compute_ncs(bqs_result["BQS"], fws_result["FWS"], pen_result)
    note       = action_note(fws_result["FWS"], ncs_result["NCS"], pen_result["EarningsPenalty"])

    # QMJ 품질 배율 적용 — NCS를 multiplier로 조정
    multiplier = _safe(row.momentum_score_multiplier, 1.0)
    adjusted_ncs = _r2(_clamp(ncs_result["NCS"] * multiplier))

    return {
        # 기본 필드
        "ticker":  row.ticker,
        "name":    row.name,
        "sleeve":  row.sleeve,
        "status":  row.status,
        # BQS
        **bqs_result,
        # FWS
        **fws_result,
        # 패널티
        **pen_result,
        # NCS
        **ncs_result,
        "AdjustedNCS":        adjusted_ncs,
        "quality_tier":       row.quality_tier,
        "momentum_multiplier":multiplier,
        # 파생
        "di_spread":   _r2(_safe(row.plus_di) - _safe(row.minus_di)),
        "ActionNote":  note,
    }


def score_all(rows: list[SnapshotRow]) -> list[dict]:
    return [score_row(r) for r in rows]


# ── Hurst 지수 계산 (log-return R/S, TypeScript 이식) ─────────────────────────

def calc_hurst_v2(closes: list[float], min_bars: int = 50) -> Optional[float]:
    """Hurst 지수 계산 — 로그 수익률 R/S 분석.

    TypeScript hurst.ts의 Python 이식.
    H > 0.5: 추세 지속 / H ≈ 0.5: 랜덤 워크 / H < 0.5: 평균 회귀

    Args:
        closes: 일별 종가 리스트 (시간순, 오래된 것 → 최신)
        min_bars: 최소 필요 봉 수 (기본 50)

    Returns:
        Hurst 지수 (0~1) 또는 None
    """
    if len(closes) < min_bars:
        return None

    # 로그 수익률: ln(P[t] / P[t-1])
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            returns.append(math.log(closes[i] / closes[i - 1]))

    if len(returns) < min_bars - 1:
        return None

    n = len(returns)
    arr = np.array(returns)

    # 서브기간 크기: 8부터 시작해 1.5배씩 증가, n//2까지
    sizes = []
    s = 8
    while s <= n // 2:
        sizes.append(s)
        s = max(s + 1, int(s * 1.5))

    if len(sizes) < 3:
        return None

    log_sizes = []
    log_rs    = []

    for size in sizes:
        num_segs = n // size
        if num_segs < 1:
            continue
        rs_vals = []
        for seg_idx in range(num_segs):
            seg = arr[seg_idx * size : (seg_idx + 1) * size]
            rs  = _rescaled_range(seg)
            if rs is not None and rs > 0:
                rs_vals.append(rs)
        if rs_vals:
            log_sizes.append(math.log(size))
            log_rs.append(math.log(np.mean(rs_vals)))

    if len(log_sizes) < 3:
        return None

    slope = _linear_regression_slope(log_sizes, log_rs)
    return max(0.0, min(1.0, slope))


def _rescaled_range(segment: np.ndarray) -> Optional[float]:
    n = len(segment)
    if n < 2:
        return None
    mean    = np.mean(segment)
    std     = np.std(segment)
    if std == 0:
        return None
    cum_dev = np.cumsum(segment - mean)
    R       = np.max(cum_dev) - np.min(cum_dev)
    return R / std


def _linear_regression_slope(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    xs, ys, xys, x2s = sum(x), sum(y), sum(xi*yi for xi,yi in zip(x,y)), sum(xi**2 for xi in x)
    denom = n * x2s - xs**2
    if denom == 0:
        return 0.0
    return (n * xys - xs * ys) / denom


# ── BIS (Breakout Integrity Score) 보완 계산 ─────────────────────────────────

def compute_bis_from_candle(
    open_p:  float,
    high_p:  float,
    low_p:   float,
    close_p: float,
    volume:  float,
    avg_vol_10d: float,
) -> int:
    """BIS: 브레이크아웃 무결성 점수 (0~15).

    3가지 서브컴포넌트:
      1. 몸통/범위 비율 (0/+2/+5)
      2. 거래량 vs 10일 평균 (0/+2/+5)
      3. 종가 위치 고가권 (0/+2/+5)
    """
    score = 0
    candle_range = high_p - low_p

    # 1. 몸통 비율
    if candle_range > 0:
        body = abs(close_p - open_p)
        ratio = body / candle_range
        if ratio > 0.6:
            score += 5
        elif ratio >= 0.4:
            score += 2

    # 2. 거래량
    if avg_vol_10d and avg_vol_10d > 0:
        vr = volume / avg_vol_10d
        if vr > 1.5:
            score += 5
        elif vr >= 1.0:
            score += 2

    # 3. 종가 위치
    if candle_range > 0:
        pos = (close_p - low_p) / candle_range
        if pos >= 0.7:
            score += 5
        elif pos >= 0.3:
            score += 2

    return score


# ── chasing_last5 계산 헬퍼 ──────────────────────────────────────────────────

def compute_chasing_flags(
    closes:  list[float],
    highs:   list[float],
    lookback_days: int = 5,
    n_day_high:    int = 20,
) -> tuple[bool, bool]:
    """최근 N일 동안 고점 추격 여부 계산.

    Args:
        closes:        일별 종가 (오래된 것 → 최신)
        highs:         일별 고가
        lookback_days: 추격 판단 기간 (기본 5일)
        n_day_high:    고점 기준 기간 (기본 20일)

    Returns:
        (chasing_20_last5, chasing_55_last5)
    """
    if len(closes) < n_day_high + lookback_days:
        return False, False

    chasing_20 = False
    chasing_55 = False

    # 최근 lookback_days 각 날에 대해 고점 초과 여부 확인
    for i in range(lookback_days):
        idx = len(closes) - 1 - i
        if idx < n_day_high:
            continue

        close_i = closes[idx]

        # 20일 고점: idx 이전 20일
        high_20 = max(highs[max(0, idx - n_day_high):idx]) if idx >= n_day_high else 0
        if high_20 > 0 and close_i > high_20:
            chasing_20 = True

        # 55일 고점 (available)
        if idx >= 55:
            high_55 = max(highs[max(0, idx - 55):idx])
            if high_55 > 0 and close_i > high_55:
                chasing_55 = True

    return chasing_20, chasing_55
