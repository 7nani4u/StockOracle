"""HybridTurtle-v6.0 알고리즘을 StockOracle Python으로 이식한 예측 신호 모듈.

원본: HybridTurtle-v6.0 (TypeScript)
  - packages/signals/src/  : breakout.ts, trend.ts, math.ts, ranking.ts
  - src/lib/dual-score.ts  : BQS / FWS / NCS 복합 점수 엔진
  - src/lib/regime-detector.ts : 시장 레짐 감지
  - src/lib/breakout-integrity.ts : 브레이크아웃 품질 점수
  - src/lib/hurst.ts        : 허스트 지수
  - src/lib/modules/adaptive-atr-buffer.ts : 적응형 ATR 버퍼
  - src/lib/scan-guards.ts  : 추격 방지 가드

공개 API:
  compute_regime()          — 시장 레짐 감지 (BULLISH/BEARISH/SIDEWAYS)
  compute_bqs()             — BQS: 브레이크아웃 품질 점수 (0-100)
  compute_fws()             — FWS: 치명적 약점 점수 (0-100, 높을수록 위험)
  compute_ncs()             — NCS: 순복합 점수 = BQS - 0.8*FWS + 10, 클램프(0,100)
  compute_bis()             — BIS: 브레이크아웃 무결성 점수 (0-15)
  calc_hurst()              — 허스트 지수 (0-1, >0.5=추세, <0.5=평균회귀)
  adaptive_atr_buffer()     — 적응형 ATR 기반 진입 트리거 계산
  anti_chase_guard()        — 추격 방지 가드 (ext_atr 기반)
  compute_hybrid_score()    — 전체 복합 점수 계산 (단일 진입점)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# ── 상수 ─────────────────────────────────────────────────────────────────────

REGIME_BULLISH  = "BULLISH"
REGIME_BEARISH  = "BEARISH"
REGIME_SIDEWAYS = "SIDEWAYS"

_VOL_REGIME_LOW    = "LOW_VOL"
_VOL_REGIME_NORMAL = "NORMAL_VOL"
_VOL_REGIME_HIGH   = "HIGH_VOL"

# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _safe(v, default: float = 0.0) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _calc_atr(highs, lows, closes, period: int = 14) -> float | None:
    """14-period Wilder's ATR."""
    n = len(closes)
    if n < period + 1:
        return None
    trs = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = float(np.mean(trs[:period]))
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 6)


def _calc_adx(highs, lows, closes, period: int = 14) -> dict | None:
    """ADX + DI+ / DI- (Wilder's smoothing)."""
    n = len(closes)
    if n < period * 2:
        return None

    dm_plus_list, dm_minus_list, tr_list = [], [], []
    for i in range(1, n):
        h_diff = highs[i] - highs[i - 1]
        l_diff = lows[i - 1] - lows[i]
        dm_plus_list.append(h_diff if h_diff > l_diff and h_diff > 0 else 0.0)
        dm_minus_list.append(l_diff if l_diff > h_diff and l_diff > 0 else 0.0)
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        tr_list.append(tr)

    if len(tr_list) < period:
        return None

    def _smooth(lst):
        s = float(sum(lst[:period]))
        result = [s]
        for v in lst[period:]:
            s = s - s / period + v
            result.append(s)
        return result

    tr_s   = _smooth(tr_list)
    dmp_s  = _smooth(dm_plus_list)
    dmm_s  = _smooth(dm_minus_list)

    dx_list = []
    di_plus_list, di_minus_list = [], []
    for tr, dp, dm in zip(tr_s, dmp_s, dmm_s):
        if tr == 0:
            continue
        di_p = 100 * dp / tr
        di_m = 100 * dm / tr
        di_plus_list.append(di_p)
        di_minus_list.append(di_m)
        total = di_p + di_m
        dx_list.append(100 * abs(di_p - di_m) / total if total else 0.0)

    if len(dx_list) < period:
        return None

    adx = float(np.mean(dx_list[-period:]))
    return {
        "adx":      round(adx, 2),
        "plus_di":  round(di_plus_list[-1], 2) if di_plus_list else 0.0,
        "minus_di": round(di_minus_list[-1], 2) if di_minus_list else 0.0,
        "bullish":  di_plus_list[-1] > di_minus_list[-1] if (di_plus_list and di_minus_list) else False,
    }


def _calc_ma(prices, period: int) -> float | None:
    if len(prices) < period:
        return None
    return float(np.mean(prices[-period:]))


def _calc_rs(prices_ticker, prices_bench, period: int = 63) -> float | None:
    """상대강도: ticker 수익률 - benchmark 수익률 (63일 기준)."""
    n = min(len(prices_ticker), len(prices_bench), period + 1)
    if n < 2:
        return None
    t = prices_ticker[-n:]
    b = prices_bench[-n:]
    t_ret = (t[-1] / t[0] - 1) * 100
    b_ret = (b[-1] / b[0] - 1) * 100
    return round(t_ret - b_ret, 2)


# ── 허스트 지수 ───────────────────────────────────────────────────────────────

def calc_hurst(prices: list[float], min_lag: int = 2, max_lag: int = 20) -> float | None:
    """허스트 지수 계산 (R/S Analysis).

    반환:
      H > 0.55  → 추세 지속 (BQS 보너스 대상)
      H ~0.5    → 랜덤 워크
      H < 0.45  → 평균 회귀 (BQS 보너스 없음, FWS 주의)
    """
    arr = np.array(prices, dtype=float)
    n   = len(arr)
    if n < max_lag * 2:
        return None

    lags = range(min_lag, min(max_lag, n // 2))
    rs_vals, lag_vals = [], []

    for lag in lags:
        chunks = [arr[i:i + lag] for i in range(0, n - lag, lag)]
        if len(chunks) < 2:
            continue
        rs_chunk = []
        for chunk in chunks:
            mean_c = np.mean(chunk)
            dev    = np.cumsum(chunk - mean_c)
            r_range = float(np.max(dev) - np.min(dev))
            s_std   = float(np.std(chunk, ddof=0))
            if s_std > 0:
                rs_chunk.append(r_range / s_std)
        if rs_chunk:
            rs_vals.append(np.log(np.mean(rs_chunk)))
            lag_vals.append(np.log(lag))

    if len(lag_vals) < 3:
        return None

    slope, _ = np.polyfit(lag_vals, rs_vals, 1)
    return round(float(slope), 4)


# ── Breakout Integrity Score ─────────────────────────────────────────────────

def compute_bis(
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    volume: float,
    avg_volume_10d: float,
) -> int:
    """브레이크아웃 무결성 점수 (0-15).

    3 서브컴포넌트:
      1. 몸통/범위 비율  (0/+2/+5 pts)
      2. 거래량 vs 10일평균 (0/+2/+5 pts)
      3. 종가 위치 (0/+2/+5 pts)
    """
    score = 0
    candle_range = high_p - low_p

    # 1. 몸통/범위
    if candle_range > 0:
        body = abs(close_p - open_p)
        ratio = body / candle_range
        if ratio > 0.6:
            score += 5
        elif ratio >= 0.4:
            score += 2

    # 2. 거래량
    if avg_volume_10d and avg_volume_10d > 0:
        vol_ratio = volume / avg_volume_10d
        if vol_ratio > 1.5:
            score += 5
        elif vol_ratio >= 1.0:
            score += 2

    # 3. 종가 위치
    if candle_range > 0:
        pos = (close_p - low_p) / candle_range
        if pos >= 0.7:
            score += 5
        elif pos >= 0.3:
            score += 2

    return score


# ── 시장 레짐 감지 ────────────────────────────────────────────────────────────

def compute_regime(
    price: float,
    ma200: float,
    adx_data: dict | None,
    vix: float | None = None,
    advance_decline_ratio: float | None = None,
) -> dict:
    """5신호 포인트 기반 레짐 감지 (HybridTurtle §8).

    Args:
        price:    현재가 (SPY 또는 KOSPI200)
        ma200:    200일 이동평균
        adx_data: calc_adx() 반환 dict {"adx", "plus_di", "minus_di", "bullish"}
        vix:      VIX 현재값 (미국: 공포지수, 한국은 None 가능)
        advance_decline_ratio: 상승/하락 비율 (없으면 None)

    반환:
        {
          "regime":     "BULLISH" | "BEARISH" | "SIDEWAYS",
          "bull_pts":   int,
          "bear_pts":   int,
          "chop_band":  bool,   # MA200 ±2% 이내 → 강제 SIDEWAYS
          "vol_regime": "LOW_VOL" | "NORMAL_VOL" | "HIGH_VOL",
        }
    """
    bull_pts = 0
    bear_pts = 0

    # ① Price vs MA200
    if ma200 and ma200 > 0:
        if price > ma200:
            bull_pts += 3
        else:
            bear_pts += 3

    # ② ADX Trend Strength + DI Direction
    if adx_data:
        adx_val = adx_data.get("adx", 0) or 0
        plus_di  = adx_data.get("plus_di", 0) or 0
        minus_di = adx_data.get("minus_di", 0) or 0

        if adx_val >= 25:
            if plus_di > minus_di:
                bull_pts += 1
            else:
                bear_pts += 1

        if plus_di > minus_di:
            bull_pts += 2
        else:
            bear_pts += 2

    # ③ VIX (미국 시장 전용)
    if vix is not None:
        if vix < 20:
            bull_pts += 1
        elif vix >= 30:
            bear_pts += 1

    # ④ A/D Ratio (breadth)
    if advance_decline_ratio is not None:
        if advance_decline_ratio > 1.2:
            bull_pts += 1
        elif advance_decline_ratio < 0.8:
            bear_pts += 1

    # ⑤ CHOP Band (MA200 ±2%)
    chop_band = False
    if ma200 and ma200 > 0:
        upper = ma200 * 1.02
        lower = ma200 * 0.98
        if lower <= price <= upper:
            chop_band = True

    # 레짐 결정
    if chop_band:
        regime = REGIME_SIDEWAYS
    elif bull_pts >= 5:
        regime = REGIME_BULLISH
    elif bear_pts >= 5:
        regime = REGIME_BEARISH
    else:
        regime = REGIME_SIDEWAYS

    # 변동성 레짐 (ATR% 기반, 단순화)
    vol_regime = _VOL_REGIME_NORMAL

    return {
        "regime":    regime,
        "bull_pts":  bull_pts,
        "bear_pts":  bear_pts,
        "chop_band": chop_band,
        "vol_regime": vol_regime,
    }


def detect_vol_regime(atr_percent: float) -> str:
    """SPY/KOSPI200 ATR% 기반 변동성 레짐 분류."""
    if atr_percent < 1.0:
        return _VOL_REGIME_LOW
    if atr_percent <= 2.0:
        return _VOL_REGIME_NORMAL
    return _VOL_REGIME_HIGH


# ── 적응형 ATR 버퍼 ──────────────────────────────────────────────────────────

def adaptive_atr_buffer(
    twenty_day_high: float,
    atr: float,
    atr_percent: float,
    vol_regime: str = _VOL_REGIME_NORMAL,
) -> dict:
    """적응형 ATR 기반 진입 트리거 계산 (HybridTurtle Module 11b).

    ATR%가 낮을수록 버퍼 크게, 높을수록 작게 조정.
    변동성 레짐에 따른 추가 배율 적용.

    Returns:
        {
          "entry_trigger":  float,  # 적응형 진입 가격
          "stop_price":     float,  # 초기 스탑 (entry_trigger - 1.5*ATR)
          "buffer_pct":     float,  # 적용된 버퍼 비율
        }
    """
    # ATR% 기반 버퍼 비율 계산 (선형 보간)
    ATR_LOW  = 2.0
    ATR_HIGH = 6.0
    BUF_LOW  = 0.20   # ATR% ≤2% 시 버퍼 20%
    BUF_HIGH = 0.05   # ATR% ≥6% 시 버퍼 5%

    if atr_percent <= ATR_LOW:
        base_buffer = BUF_LOW
    elif atr_percent >= ATR_HIGH:
        base_buffer = BUF_HIGH
    else:
        t = (atr_percent - ATR_LOW) / (ATR_HIGH - ATR_LOW)
        base_buffer = BUF_LOW + t * (BUF_HIGH - BUF_LOW)

    # 변동성 레짐 배율
    vol_multiplier = {
        _VOL_REGIME_LOW:    0.8,
        _VOL_REGIME_NORMAL: 1.0,
        _VOL_REGIME_HIGH:   1.3,
    }.get(vol_regime, 1.0)

    scaled_buffer = base_buffer * vol_multiplier
    entry_trigger = twenty_day_high + scaled_buffer * atr
    stop_price    = entry_trigger - 1.5 * atr

    return {
        "entry_trigger": round(entry_trigger, 4),
        "stop_price":    round(stop_price, 4),
        "buffer_pct":    round(scaled_buffer * 100, 2),
    }


# ── 추격 방지 가드 ────────────────────────────────────────────────────────────

def anti_chase_guard(
    current_price: float,
    entry_trigger: float,
    atr: float,
    is_monday: bool = False,
) -> dict:
    """ext_atr 기반 추격 방지 가드 (HybridTurtle §3 / Stage 6b).

    Args:
        current_price:  현재가
        entry_trigger:  적응형 진입 트리거
        atr:            14일 ATR
        is_monday:      월요일 여부 (더 엄격한 기준 적용)

    Returns:
        {
          "ext_atr":     float,   # (price - trigger) / ATR
          "chasing":     bool,    # True = 추격 중
          "wait_pullback": bool,  # True = 풀백 대기 권장
          "reason":      str,
        }
    """
    if atr <= 0:
        return {"ext_atr": 0.0, "chasing": False, "wait_pullback": False, "reason": "ATR 데이터 없음"}

    ext_atr = (current_price - entry_trigger) / atr

    # ext_atr > 0.8 → WAIT_PULLBACK (전일 기준, 모든 요일)
    if ext_atr > 0.8:
        return {
            "ext_atr":      round(ext_atr, 3),
            "chasing":      True,
            "wait_pullback": True,
            "reason":       f"과도한 추격 (ext_atr={ext_atr:.2f}, 기준 0.8)",
        }

    # 월요일 갭 추격 방지 (더 엄격)
    gap_threshold_pct = 3.0 if is_monday else 4.0
    if current_price >= entry_trigger:
        pct_above = ((current_price / entry_trigger) - 1) * 100
        if pct_above > gap_threshold_pct:
            return {
                "ext_atr":      round(ext_atr, 3),
                "chasing":      True,
                "wait_pullback": False,
                "reason":       f"갭 추격 ({pct_above:.1f}% > {gap_threshold_pct}% 기준)",
            }

    return {
        "ext_atr":      round(ext_atr, 3),
        "chasing":      False,
        "wait_pullback": False,
        "reason":       "정상 진입 가능",
    }


# ── Dual Regime Score (DRS) ───────────────────────────────────────────────────

def _calc_dual_regime_score(regime: str, vol_regime: str, dual_aligned: bool = True) -> float:
    """DRS: 레짐 상태를 BQS 컴포넌트 점수로 변환 (−10 ~ +20).

    HybridTurtle dual-score.ts calcDualRegimeScore() 이식.
    """
    if regime == REGIME_BEARISH:
        return -10.0
    if regime == REGIME_SIDEWAYS:
        return 0.0

    # BULLISH 분기
    if vol_regime == _VOL_REGIME_HIGH:
        return 10.0
    if not dual_aligned:
        return 10.0
    if vol_regime == _VOL_REGIME_NORMAL:
        return 15.0
    # LOW_VOL + BULLISH + dual_aligned
    return 20.0


# ── BQS (Breakout Quality Score) ─────────────────────────────────────────────

def compute_bqs(
    adx:          float,
    plus_di:      float,
    minus_di:     float,
    atr_percent:  float,
    dist_to_high: float,        # 현재가에서 20일 고점까지 거리 (%)
    regime:       str,
    vol_regime:   str,
    rs_pct:       float = 0.0,  # 상대강도 (%)
    vol_ratio:    float = 1.0,  # 거래량 비율 (오늘/20일평균)
    hurst:        float | None = None,
    bis_score:    int   = 0,
    dual_aligned: bool  = True,
) -> float:
    """BQS: 브레이크아웃 품질 점수 (0-100).

    HybridTurtle dual-score.ts computeBQS() 이식.
    이론 범위: −15~148, clamp(0,100) 적용.
    """
    score = 0.0

    # 1. 추세 강도 (0-25): ADX
    score += 25.0 * _clamp((adx - 15.0) / 20.0, 0.0, 1.0)

    # 2. 방향 지배력 (0-10): +DI vs -DI
    score += 10.0 * _clamp((plus_di - minus_di) / 25.0, 0.0, 1.0)

    # 3. 변동성 건전성 (0-15): ATR%
    if atr_percent < 1.0:
        vol_health = 15.0 * _clamp(atr_percent / 1.0, 0.0, 1.0)
    elif atr_percent <= 4.0:
        vol_health = 15.0
    elif atr_percent <= 6.0:
        vol_health = 15.0 * (1.0 - (atr_percent - 4.0) / 2.0)
    else:
        vol_health = 0.0
    score += vol_health

    # 4. 근접성 (0-15): 20일 고점까지의 거리
    score += 15.0 * _clamp(1.0 - dist_to_high / 3.0, 0.0, 1.0)

    # 5. Dual Regime Score (−10 ~ +20)
    drs = _calc_dual_regime_score(regime, vol_regime, dual_aligned)
    score += drs

    # 6. 상대강도 점수 (0-15)
    score += 15.0 * _clamp((rs_pct + 5.0) / 20.0, 0.0, 1.0)

    # 7. 거래량 보너스 (0-5)
    if vol_ratio > 1.2:
        score += 5.0 * _clamp((vol_ratio - 1.2) / 0.6, 0.0, 1.0)

    # 8. 허스트 보너스 (0 ~ +8)
    if hurst is not None:
        if hurst >= 0.7:
            score += 8.0
        elif hurst >= 0.6:
            score += 5.0
        elif hurst >= 0.5:
            score += 2.0

    # 9. BIS 점수 (0-15)
    score += float(min(bis_score, 15))

    return round(_clamp(score), 2)


# ── FWS (Fatal Weakness Score) ────────────────────────────────────────────────

def compute_fws(
    vol_ratio:         float,
    ext_atr:           float,
    adx:               float,
    atr_spiking:       bool,
    atr_collapsing:    bool,
    regime_stable:     bool,
) -> float:
    """FWS: 치명적 약점 점수 (0-100, 높을수록 위험).

    HybridTurtle dual-score.ts computeFWS() 이식.
    """
    score = 0.0

    # 1. 거래량 리스크 (0-30): vol_ratio가 낮을수록 위험
    score += 30.0 * _clamp(1.0 - (vol_ratio - 0.6) / 0.6, 0.0, 1.0)

    # 2. 추격 리스크 (0-25): ext_atr 기반
    if ext_atr > 0.8:
        score += 25.0
    elif ext_atr > 0.4:
        score += 15.0

    # 3. 추세 한계 리스크 (0-10): ADX
    if adx < 20:
        score += 10.0
    elif adx < 25:
        score += 7.0
    elif adx < 30:
        score += 3.0

    # 4. 변동성 충격 리스크 (0-10): ATR 스파이크/붕괴
    if atr_spiking or atr_collapsing:
        score += 10.0

    # 5. 레짐 불안정 리스크 (0-10)
    if not regime_stable:
        score += 10.0

    return round(_clamp(score), 2)


# ── NCS (Net Composite Score) ─────────────────────────────────────────────────

def compute_ncs(
    bqs: float,
    fws: float,
    earnings_penalty:    float = 0.0,
    cluster_penalty:     float = 0.0,
    super_cluster_penalty: float = 0.0,
) -> float:
    """NCS: 순복합 점수 (0-100).

    NCS = clamp(BQS - 0.8*FWS + 10) - cappedPenalty
    총 패널티 최대 40으로 제한.
    """
    base_ncs     = _clamp(bqs - 0.8 * fws + 10.0)
    total_penalty = earnings_penalty + cluster_penalty + super_cluster_penalty
    capped_penalty = min(total_penalty, 40.0)
    return round(_clamp(base_ncs - capped_penalty), 2)


def ncs_action(ncs: float, fws: float) -> str:
    """NCS + FWS → 행동 분류.

    Returns:
      "AUTO_YES"    — NCS ≥70 AND FWS ≤30
      "AUTO_NO"     — FWS >65 (취약)
      "CONDITIONAL" — 그 외
    """
    if fws > 65.0:
        return "AUTO_NO"
    if ncs >= 70.0 and fws <= 30.0:
        return "AUTO_YES"
    return "CONDITIONAL"


# ── 종합 진입점 ───────────────────────────────────────────────────────────────

def compute_hybrid_score(
    closes:           list[float],
    highs:            list[float],
    lows:             list[float],
    volumes:          list[float],
    open_prices:      list[float] | None = None,
    bench_closes:     list[float] | None = None,  # SPY 또는 KOSPI200 종가
    bench_ma200:      float | None = None,         # 지수 200일 MA
    vix:              float | None = None,
    adv_decline:      float | None = None,
    earnings_days:    int   | None = None,          # 실적 발표까지 남은 일수
) -> dict:
    """단일 진입점: 종목의 HybridTurtle 복합 점수 전체 계산.

    Args:
        closes:       일별 종가 리스트 (최소 60개 권장)
        highs:        일별 고가 리스트
        lows:         일별 저가 리스트
        volumes:      일별 거래량 리스트
        open_prices:  일별 시가 리스트 (BIS용, 없어도 됨)
        bench_closes: 벤치마크(SPY/KOSPI200) 종가 (RS/레짐용)
        bench_ma200:  벤치마크 200일 MA (레짐용)
        vix:          VIX 현재값 (미국 시장)
        adv_decline:  A/D ratio (breadth)
        earnings_days: 실적 발표까지 남은 일 (패널티용)

    Returns:
        {
          "bqs": float,   # 0-100
          "fws": float,   # 0-100
          "ncs": float,   # 0-100
          "action": str,  # AUTO_YES / AUTO_NO / CONDITIONAL
          "regime": str,  # BULLISH / BEARISH / SIDEWAYS
          "vol_regime": str,
          "hurst": float | None,
          "bis_score": int,
          "adx": float,
          "plus_di": float,
          "minus_di": float,
          "atr": float | None,
          "atr_percent": float | None,
          "vol_ratio": float | None,
          "rs_pct": float | None,
          "twenty_day_high": float,
          "dist_to_high": float,
          "entry_trigger": float | None,
          "stop_price": float | None,
          "anti_chase": dict,
        }
    """
    n = len(closes)
    if n < 20:
        return {"bqs": 0.0, "fws": 50.0, "ncs": 0.0, "action": "AUTO_NO",
                "regime": REGIME_SIDEWAYS, "error": "데이터 부족 (최소 20개 필요)"}

    cur_price = closes[-1]

    # ADX / DI
    adx_data  = _calc_adx(highs, lows, closes)
    adx_val   = _safe(adx_data and adx_data.get("adx"), 0.0)
    plus_di   = _safe(adx_data and adx_data.get("plus_di"), 0.0)
    minus_di  = _safe(adx_data and adx_data.get("minus_di"), 0.0)

    # ATR
    atr = _calc_atr(highs, lows, closes)
    atr_pct = (atr / cur_price * 100) if (atr and cur_price) else None

    # 거래량 비율
    vol_ratio = None
    if len(volumes) >= 21:
        avg_vol = float(np.mean(volumes[-21:-1]))
        if avg_vol > 0:
            vol_ratio = round(volumes[-1] / avg_vol, 3)

    # ATR 스파이크/붕괴 감지
    atr_spiking    = False
    atr_collapsing = False
    if atr and len(closes) >= 35:
        old_atr = _calc_atr(highs[-35:-20], lows[-35:-20], closes[-35:-20])
        if old_atr and old_atr > 0:
            if atr >= old_atr * 1.3:
                atr_spiking = True
            if atr <= old_atr * 0.5:
                atr_collapsing = True

    # 20일 고점 + 거리
    high_20d = float(max(highs[-20:])) if len(highs) >= 20 else cur_price
    dist_to_high = max(0.0, (high_20d - cur_price) / cur_price * 100) if cur_price else 0.0

    # 허스트
    hurst = calc_hurst(closes) if n >= 40 else None

    # BIS (브레이크아웃 무결성)
    avg_vol_10d = float(np.mean(volumes[-11:-1])) if len(volumes) >= 11 else 0.0
    bis_score = 0
    if open_prices and len(open_prices) >= 1:
        bis_score = compute_bis(
            open_p=open_prices[-1],
            high_p=highs[-1],
            low_p=lows[-1],
            close_p=closes[-1],
            volume=volumes[-1] if volumes else 0.0,
            avg_volume_10d=avg_vol_10d,
        )

    # 상대강도
    rs_pct = None
    if bench_closes and len(bench_closes) >= 2:
        rs_pct = _calc_rs(closes, bench_closes)

    # MA200 (벤치마크)
    bench_price = bench_closes[-1] if bench_closes else None
    ma200       = bench_ma200 if bench_ma200 else (
        _calc_ma(bench_closes, 200) if bench_closes and len(bench_closes) >= 200 else None
    )

    # 레짐 감지
    regime_data  = compute_regime(
        price=bench_price or cur_price,
        ma200=ma200 or cur_price,
        adx_data=adx_data,
        vix=vix,
        advance_decline_ratio=adv_decline,
    )
    regime       = regime_data["regime"]
    vol_regime   = (detect_vol_regime(atr_pct) if atr_pct else _VOL_REGIME_NORMAL)
    regime_stable = not regime_data["chop_band"]

    # 적응형 ATR 버퍼
    entry_info = None
    if atr and atr_pct:
        entry_info = adaptive_atr_buffer(high_20d, atr, atr_pct, vol_regime)

    # 추격 방지 가드
    anti_chase = {"ext_atr": 0.0, "chasing": False, "wait_pullback": False, "reason": "ATR 없음"}
    if atr and entry_info:
        anti_chase = anti_chase_guard(cur_price, entry_info["entry_trigger"], atr)

    # 실적 패널티
    earnings_penalty = 0.0
    if earnings_days is not None:
        if earnings_days <= 1:
            earnings_penalty = 20.0
        elif earnings_days <= 3:
            earnings_penalty = 15.0
        elif earnings_days <= 5:
            earnings_penalty = 10.0

    # BQS / FWS / NCS
    bqs = compute_bqs(
        adx          = adx_val,
        plus_di      = plus_di,
        minus_di     = minus_di,
        atr_percent  = atr_pct or 3.0,
        dist_to_high = dist_to_high,
        regime       = regime,
        vol_regime   = vol_regime,
        rs_pct       = rs_pct or 0.0,
        vol_ratio    = vol_ratio or 1.0,
        hurst        = hurst,
        bis_score    = bis_score,
    )
    fws = compute_fws(
        vol_ratio      = vol_ratio or 1.0,
        ext_atr        = anti_chase["ext_atr"],
        adx            = adx_val,
        atr_spiking    = atr_spiking,
        atr_collapsing = atr_collapsing,
        regime_stable  = regime_stable,
    )
    ncs = compute_ncs(bqs, fws, earnings_penalty=earnings_penalty)
    action = ncs_action(ncs, fws)

    return {
        "bqs":            bqs,
        "fws":            fws,
        "ncs":            ncs,
        "action":         action,
        "regime":         regime,
        "vol_regime":     vol_regime,
        "regime_stable":  regime_stable,
        "hurst":          hurst,
        "bis_score":      bis_score,
        "adx":            round(adx_val, 2),
        "plus_di":        round(plus_di, 2),
        "minus_di":       round(minus_di, 2),
        "atr":            round(atr, 4) if atr else None,
        "atr_percent":    round(atr_pct, 2) if atr_pct else None,
        "atr_spiking":    atr_spiking,
        "atr_collapsing": atr_collapsing,
        "vol_ratio":      vol_ratio,
        "rs_pct":         rs_pct,
        "twenty_day_high":round(high_20d, 4),
        "dist_to_high":   round(dist_to_high, 2),
        "entry_trigger":  entry_info["entry_trigger"] if entry_info else None,
        "stop_price":     entry_info["stop_price"] if entry_info else None,
        "buffer_pct":     entry_info["buffer_pct"] if entry_info else None,
        "anti_chase":     anti_chase,
        "regime_detail":  regime_data,
    }
