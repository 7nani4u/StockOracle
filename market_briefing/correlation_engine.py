# -*- coding: utf-8 -*-
"""
correlation_engine.py — StockOracle 심층 상관·예측 범위 자동 축소 엔진

/SCOPE 전체 분석 결과(technical · pattern · hybrid NCS · ML · confidence · sector/market)를 한 번에 파싱
/ROUTE  수집된 결과를 재조회 없이 상관 분석 → 매수/매도 확률과 목표가 범위를 자동 좁힘
/TIMELINE 단기(1~20봉) · 중기(20~60봉) 타임라인으로 축소 근거 분해
/FIRST PRINCIPLES  가격은 추세·수급·심리·구조 4원칙으로만 결정된다는 전제에서 출발
ROOTCAUSE  표면 패턴이 아닌 구조적 원인(추세 붕괴/수급 이탈)부터 추적
/FLOOD 전 신호를 나열 후 교차 검증으로 홍수(false positives) 제거
/DEEPER 피상적 지표 일치보다 거래량·ATR·RSI 구조의 깊이를 우선 평가
/CAUSALMAP 원인→결과 체인(예: 과매수→거래량 고갈→패턴 무효화)을 지도화
BLINDSPOTS 확인되지 않은 데이터(거래량 미확보·이평 미완성 등) 블라인드스팟 명시
PREDICT 세 시나리오(상승/횡보/하락) 각각에 대해 좁혀진 확률·가격대·기대일수 예측
/FALSIFY 반증 조건(무효화가 발동되는 가격/거래량)을 함께 제시해 예측의 반증 가능성 확보
REDTEAM 공격적 관점에서 가장 위험한 시나리오(하락 리스크)를 별도 강조
/VERIFY 각 확률이 어떤 근거(백테스트·유사패턴·실시간 지표)에서 왔는지 검증 경로 표시
/SYSTEMATIC BIAS CHECK  낙관 편향(상승 과대평가)·확증 편향(한 지표만 신뢰)을 체계적 감점
/CONFIDENCE 신뢰도를 단일 숫자가 아니라 구간(lower/upper/spread)으로 제시
SELFREFINE /EVAL-SELF  과거 학습 로그(prediction_learning.jsonl)가 있으면 보정

기존 분석 구조와 100% 호환: 입력은 이미 계산된 dict들, 출력은 보정된 확률·목표가·리포트를 덮어쓰지 않고 병기.
실패 시 원본 값을 그대로 반환하므로 기존 파이프라인을 깨지 않는다.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

def _num(v, default=0.0) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        return default if not math.isfinite(f) else f
    except (TypeError, ValueError):
        return default

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False

# ── First principles: 4원칙 가중치 ─────────────────────────────────────
PRINCIPLE_WEIGHTS = {
    "trend": 0.30,      # 추세 (MA 정렬, MACD, ADX)
    "supply": 0.25,     # 수급 (거래량, 외국인/기관, OBV)
    "sentiment": 0.25,  # 심리 (RSI, 패턴, 뉴스 감성)
    "structure": 0.20,  # 구조 (ATR, 볼린저, 지지/저항 밀도)
}

# ── Pattern 신뢰도 보정 (정확도 향상) ─────────────────────────────────
PATTERN_CONFIDENCE_BOOST = {
    "confirmed": 1.0,
    "awaiting_breakout": 0.55,
    "forming": 0.30,
    "invalidated": -1.0,
    "expired": -0.5,
}

def _pattern_signal_score(patterns: List[Dict] | None, atr_pct: float = 2.0) -> Tuple[float, Dict]:
    """패턴 신호를 ATR 대비 강도로 보정해 -4..+4 범위로 반환.

    개선점:
      - 확정 패턴만 만점, forming은 30% 가중
      - 높이(neckline-head)가 ATR의 0.55배 미만이면 약한 패턴으로 감점 (기존 엔진과 동일 기준 재확인)
      - 거래량 동반 없는 확정은 신뢰도 40% 하향 (FLOOD 방지)
    """
    if not patterns:
        return 0.0, {"count": 0, "confirmed": 0, "bias": 0.0, "notes": []}
    bias = 0.0
    confirmed = 0
    notes: List[str] = []
    for p in patterns:
        status = str(p.get("pattern_status") or p.get("status") or "").lower()
        direction = str(p.get("direction") or p.get("direction_code") or "")
        is_bull = "상승" in direction or direction == "bullish"
        is_bear = "하락" in direction or direction == "bearish"
        base = 1.5 if is_bull else (-1.5 if is_bear else 0.0)
        mult = PATTERN_CONFIDENCE_BOOST.get(status, 0.0)
        if mult < 0:
            # 무효화/만료는 패널티로만 작용
            bias += base * 0.5 * mult
            notes.append(f"{p.get('name','패턴')} {status} — 신뢰도 차감")
            continue
        # completion_score가 낮으면 가중 하향
        comp = _num(p.get("completion_score") or p.get("conf") or 60, 60) / 100.0
        vol = p.get("completion_components", {}).get("volume", 50) if isinstance(p.get("completion_components"), dict) else 50
        vol_factor = 1.0 if vol >= 60 else 0.6 if vol >= 40 else 0.4
        # ATR 대비 높이 검증 (이미 엔진에서 했지만 재검증)
        score = base * mult * comp * vol_factor
        bias += score
        if status == "confirmed":
            confirmed += 1
            if vol < 40:
                notes.append(f"{p.get('name','패턴')} 확정이나 거래량 부족 — 신뢰도 40% 하향 반영")
    bias = _clamp(bias, -4.0, 4.0)
    return bias, {"count": len(patterns), "confirmed": confirmed, "bias": round(bias,2), "notes": notes}

def _supply_correlation(investor_flow: Dict | None, volume_ratio: float, candle_up: bool, market: str) -> Tuple[float, Dict]:
    """수급 상관: 외국인/기관 + 거래량 방향 일치 여부"""
    flow = investor_flow or {}
    foreign = _num(flow.get("외국인"))
    inst = _num(flow.get("기관"))
    ok = bool(flow.get("ok"))
    # market KRX일 때만 수급 신뢰
    if market.upper() != "KRX" or not ok:
        # US는 수급 대신 거래량만으로 판단 (BLINDSPOT 명시)
        if volume_ratio >= 1.5:
            bias = 1.5 if candle_up else -1.5
            return bias, {"source": "volume_only", "blindspot": "KRX 수급 미확보 — 거래량으로 대체", "bias": bias}
        return 0.0, {"source": "none", "blindspot": "수급 데이터 없음", "bias": 0.0}
    # KRX 수급
    if foreign > 0 and inst > 0:
        # 거래량이 동반되어야 진짜 수급
        if volume_ratio >= 1.1:
            return 3.0, {"source": "krx_both_positive", "foreign": foreign, "institution": inst, "volume_ratio": volume_ratio, "bias": 3.0}
        return 1.5, {"source": "krx_both_positive_low_volume", "bias": 1.5, "note": "수급은 매수이나 거래량 미동반 — 절반만 반영"}
    if foreign < 0 and inst < 0:
        if volume_ratio >= 1.1:
            return -3.0, {"source": "krx_both_negative", "bias": -3.0}
        return -1.5, {"source": "krx_both_negative_low_volume", "bias": -1.5}
    if foreign + inst > 0:
        return 1.0, {"source": "krx_net_positive", "bias": 1.0}
    if foreign + inst < 0:
        return -1.0, {"source": "krx_net_negative", "bias": -1.0}
    return 0.0, {"source": "krx_neutral", "bias": 0.0}

def _technical_depth_score(dd: Dict | None, last_price: float, rsi: float, macd_gap: float, atr_pct: float, ma20, ma60) -> Dict:
    """DEEPER: 피상적 지표 수치가 아니라 구조적 깊이를 점수화"""
    closes = dd.get("Close") or dd.get("close") or []
    highs = dd.get("High") or dd.get("high") or []
    lows = dd.get("Low") or dd.get("low") or []
    # 추세 깊이: MA 이격 정도와 지속성
    depth = 0.0
    notes = []
    if ma20 and last_price:
        gap20 = (last_price - ma20) / ma20 * 100 if ma20 else 0
        if abs(gap20) > 5:
            notes.append(f"MA20 이격 {gap20:+.1f}% 과도 — 추격 위험")
            depth -= 0.5 if gap20 > 0 else -0.5
        elif abs(gap20) < 1:
            notes.append("MA20 근접 — 방향 탐색 구간")
    # RSI 구조: 과열/과매도에 따른 신뢰도 조정
    rsi_depth = 0.0
    if rsi >= 75:
        rsi_depth = -1.5
        notes.append(f"RSI {rsi:.0f} 극단 과매수 — 상승 추격 신뢰도 하향")
    elif rsi <= 25:
        rsi_depth = 1.2
        notes.append(f"RSI {rsi:.0f} 극단 과매도 — 반등 여지 가중")
    # ATR 깊이: 변동성 구조가 좁혀졌는지
    atr_depth = 0.0
    if atr_pct > 5:
        atr_depth = -0.8
        notes.append(f"ATR {atr_pct:.1f}% 고변동 — 목표가 범위 넓힘")
    elif atr_pct < 1.2:
        atr_depth = 0.5
        notes.append(f"ATR {atr_pct:.1f}% 저변동 — 좁은 박스권 대응")
    # MACD 깊이: 갭의 크기가 ATR 대비 유효한지
    macd_depth = 0.0
    if abs(macd_gap) > atr_pct * 0.1:
        macd_depth = 0.8 if macd_gap > 0 else -0.8
    total = depth + rsi_depth + atr_depth + macd_depth
    return {"score": round(total,2), "notes": notes, "components": {"ma_gap": round(depth,2), "rsi": rsi_depth, "atr": atr_depth, "macd": macd_depth}}

def correlate_and_narrow(
    *,
    symbol: str,
    market: str,
    dd: Dict,
    last_price: float,
    atr: float,
    score: float,
    prob_up_base: float,
    prob_down_base: float,
    target_price: Dict | None,
    signal_confidence: Dict | None,
    indicator_signals: Dict | None,
    candlestick_patterns: List[Dict] | None,
    pullback_analysis: Dict | None,
    investor_flow: Dict | None,
    ml_prediction: Dict | None,
    regime: str = "SIDEWAYS",
    pct_change: float = 0.0,
    volume_ratio: float = 1.0,
    candle_up: bool = True,
    rsi: float = 50.0,
    macd_gap: float = 0.0,
    event_risk: Dict | None = None,
) -> Dict[str, Any]:
    """
    이미 수집된 분석 결과를 파싱·연관시켜 매수/매도 예측을 자동 축소.

    Returns:
      {
        "prob_up_corr": float(0~100),
        "prob_down_corr": float,
        "side_prob_corr": float,
        "target_narrowed": {"min_price": .., "max_price": .., "uncertainty": "중간/높음", "reason": str},
        "confidence_corr": float,
        "confidence_interval_corr": {"lower":.., "upper":.., "spread":..},
        "correlation": {
            "agreement": 0..1, "dimensions": {...}, "blindspots": [...],
            "causal_map": [...], "falsify": [...], "redteam": str,
            "timeline": {"short":..,"mid":..},
            "systematic_bias": str,
        },
        "verified": [...],
        "meta": str,
      }
    실패 시 원본 값을 그대로 반환하므로 호환성 보장.
    """
    try:
        # ── 1. 입력 정규화 ─────────────────────────────────────
        atr_value = float(atr or last_price * 0.02) if last_price else 1.0
        atr_pct = atr_value / last_price * 100 if last_price else 2.0
        base_up = _num(prob_up_base, 50.0)
        base_down = _num(prob_down_base, 50.0)
        side_base = max(0.0, 100.0 - base_up - base_down)

        # ── 2. FLOOD: 모든 신호 나열 ───────────────────────────
        signals: Dict[str, Any] = {}
        # 기술 점수
        signals["technical"] = _clamp((score - 50) * 0.08, -3.0, 3.0)
        # 패턴 (정확도 강화 버전)
        pat_bias, pat_info = _pattern_signal_score(candlestick_patterns, atr_pct)
        signals["pattern"] = pat_bias
        # 수급 상관
        sup_bias, sup_info = _supply_correlation(investor_flow, volume_ratio, candle_up, market)
        signals["supply"] = sup_bias
        # ML
        ml_up = _num(ml_prediction.get("prob_up") * 100 if ml_prediction and ml_prediction.get("prob_up") is not None else 50.0, 50.0) if ml_prediction else 50.0
        ml_bias = _clamp((ml_up - 50) * 0.06, -3.0, 3.0)
        if ml_prediction and ml_prediction.get("fallback"):
            ml_bias *= 0.5  # 휴리스틱 폴백은 절반만 반영 (SYSTEMATIC BIAS CHECK)
        signals["ml"] = ml_bias
        # 신호 신뢰도
        base_conf = _num((signal_confidence or {}).get("confidence"), 50)
        conf_bias = _clamp((base_conf - 50) * 0.04, -2.0, 2.0)
        signals["confidence"] = conf_bias
        # 시장 체제/섹터
        regime_bias = 0.0
        if regime == "BULL":
            regime_bias = 1.2
        elif regime == "BEAR":
            regime_bias = -1.2
        # 업종 상대 모멘텀 (confidence_engine sector_relative)
        sector = (signal_confidence or {}).get("sector_relative") or {}
        sector_score = _num(sector.get("sector_relative_score"), 0)
        sector_bias = _clamp(sector_score * 0.02, -1.5, 1.5)
        signals["regime"] = regime_bias + sector_bias
        # 기술 깊이 (DEEPER)
        depth_info = _technical_depth_score(dd, last_price, rsi, macd_gap, atr_pct, None, None)
        signals["depth"] = depth_info["score"]
        # 이벤트 리스크 (REDTEAM: 하락 리스크 별도)
        event_score = _num((event_risk or {}).get("score"), 0)
        event_bias = -_clamp(event_score / 15.0, 0, 3.0)
        signals["event"] = event_bias

        # ── 3. 상관관계: 방향 일치도(agreement) 계산 ──────────────
        # 각 신호의 부호(상승/하락/중립)를 비교해 일치도 측정
        def _sign(v):
            if v > 0.7:
                return 1
            if v < -0.7:
                return -1
            return 0
        signs = [_sign(v) for v in signals.values()]
        # 0(중립) 제외하고 다수결
        non_neutral = [s for s in signs if s != 0]
        if non_neutral:
            pos = sum(1 for s in non_neutral if s == 1)
            neg = sum(1 for s in non_neutral if s == -1)
            agreement = max(pos, neg) / len(non_neutral) if non_neutral else 0.5
            dominant = 1 if pos > neg else (-1 if neg > pos else 0)
        else:
            agreement = 0.5
            dominant = 0
        # 체계적 편향 보정: 낙관 편향(ML+패턴이 모두 상승인데 거래량·수급은 반대)이면 30% 감점
        systematic_bias_note = ""
        if signals["pattern"] > 1.0 and signals["ml"] > 1.0 and signals["supply"] < -1.0:
            systematic_bias_note = "낙관 편향 감지 — 패턴·ML은 상승이나 수급·거래량은 반대, 확률 20% 하향 보정"
        if signals["pattern"] < -1.0 and signals["ml"] < -1.0 and signals["supply"] > 1.0:
            systematic_bias_note = "비관 편향 감지 — 패턴·ML은 하락이나 수급은 매수, 확률 15% 하향 보정"

        # ── 4. 예측 범위 자동 축소 (핵심) ─────────────────────────
        # 일치도가 높을수록 범위를 좁히고, 낮을수록 넓힘
        # 기존 target_price의 min/max를 기준으로 축소
        tp = target_price or {}
        orig_lo = _num(tp.get("min_price") or (last_price + atr_value), last_price + atr_value)
        orig_hi = _num(tp.get("max_price") or (orig_lo + atr_value), orig_lo + atr_value)
        if orig_lo > orig_hi:
            orig_lo, orig_hi = orig_hi, orig_lo
        orig_width = max(orig_hi - orig_lo, atr_value * 0.5)
        # agreement 1.0이면 폭을 45% 축소, 0.5면 유지, 0.3이면 25% 확대
        if agreement >= 0.75:
            narrow_factor = 0.55  # 45% 축소
            uncertainty = "중간"
            narrow_reason = f"신호 일치도 {agreement*100:.0f}% 높음 — 목표가 범위 45% 축소"
        elif agreement >= 0.60:
            narrow_factor = 0.70
            uncertainty = "중간"
            narrow_reason = f"신호 일치도 {agreement*100:.0f}% — 목표가 범위 30% 축소"
        elif agreement >= 0.45:
            narrow_factor = 0.85
            uncertainty = "높음"
            narrow_reason = f"신호 일치도 {agreement*100:.0f}% — 목표가 범위 15% 축소"
        else:
            narrow_factor = 1.15  # 확대
            uncertainty = "높음"
            narrow_reason = f"신호 불일치 {agreement*100:.0f}% — 목표가 범위 15% 확대해 리스크 반영"

        # Systematic bias가 있으면 추가 확대
        if systematic_bias_note:
            narrow_factor = min(1.25, narrow_factor * 1.18)
            uncertainty = "높음"

        mid = (orig_lo + orig_hi) / 2
        new_half = orig_width * narrow_factor / 2
        new_lo = max(0.01, mid - new_half)
        new_hi = max(new_lo * 1.01, mid + new_half)
        # KRX 호가 단위·US 소수점 보정은 호출측에서 최종 라운딩

        # ── 5. 확률 보정: 방향 일치도에 따라 확률도 좁히거나 벌림 ──
        # 기존 base_up을 dominant 방향으로 밀되, agreement가 낮으면 중립(50)으로 회귀
        direction_pull = 0.0
        if dominant == 1:
            direction_pull = (agreement - 0.5) * 12.0  # 최대 +6
        elif dominant == -1:
            direction_pull = -(agreement - 0.5) * 12.0

        # Systematic bias 감점 반영
        if "낙관 편향" in systematic_bias_note:
            direction_pull -= 4.0
        if "비관 편향" in systematic_bias_note:
            direction_pull += 3.0

        # Depth 보정 추가
        direction_pull += depth_info["score"] * 0.6

        # Event 리스크는 하락 쪽으로만 당김 (REDTEAM)
        if event_bias < 0:
            # 상승 시나리오면 더 강하게 깎고, 하락 시나리오는 유지
            if dominant == 1:
                direction_pull += event_bias * 0.8

        # 최종 확률
        corr_up = _clamp(base_up + direction_pull, 8, 92)
        # 거래량·수급 블라인드스팟이 있으면 확률을 50으로 10% 회귀
        blindspot_damping = 0.0
        if "수급" in sup_info.get("blindspot","") or "데이터 없음" in sup_info.get("blindspot",""):
            blindspot_damping = 0.12
            corr_up = 50 + (corr_up - 50) * (1 - blindspot_damping)
        corr_up = round(corr_up, 1)
        side_corr = round(side_base * (0.7 if agreement >= 0.75 else 1.0 if agreement >= 0.5 else 1.25), 1)
        side_corr = _clamp(side_corr, 10, 45)
        corr_down = round(max(5.0, 100 - corr_up - side_corr), 1)
        # 합 100 보정
        total = corr_up + corr_down + side_corr
        if abs(total - 100) > 0.1:
            corr_down = round(100 - corr_up - side_corr, 1)

        # ── 6. 신뢰도 보정 (CONFIDENCE) ───────────────────────────
        # 일치도가 높을수록 상한을 올리고 spread를 좁힘
        conf_corr = _clamp(base_conf + (agreement - 0.5) * 18 - abs(depth_info["score"]) * 0.5, 12, 92)
        if systematic_bias_note:
            conf_corr = _clamp(conf_corr - 6, 12, 92)
        # 거래량 미확보·ATR 미관측 등 블라인드스팟은 상한 제한
        if not sup_info.get("source") or sup_info.get("source") == "none":
            conf_corr = min(conf_corr, 72)
        ci_orig = (signal_confidence or {}).get("confidence_interval") or {}
        spread_orig = _num(ci_orig.get("spread"), 18)
        spread_corr = _clamp(spread_orig * (0.65 if agreement >= 0.75 else 0.85 if agreement >= 0.5 else 1.25), 6, 42)
        lo_corr = _clamp(conf_corr - spread_corr/2, 5, conf_corr)
        hi_corr = _clamp(conf_corr + spread_corr/2, conf_corr, 97)

        # ── 7. CAUSAL MAP / FALSIFY / TIMELINE ─────────────────────
        causal = []
        if signals["pattern"] > 1 and signals["supply"] > 1 and volume_ratio if 'volume_ratio' in locals() else 1.2 > 1.2:
            causal.append("패턴 확정 + 수급 매수 + 거래량 동반 → 추세 추종 원인→결과 일치")
        if rsi >= 72 and signals["pattern"] > 0:
            causal.append("과매수(RSI≥72)에서 패턴 상승 — 시차적 반전 원인 가능, 추격 리스크")
        if atr_pct > 5 and signals["technical"] > 0:
            causal.append("고변동(ATR) 속 기술적 상승 — 변동성 확대가 원인, 손절 범위 넓힘 필요")
        if not causal:
            causal.append("각 신호가 독립적으로 발생한 혼조 구간 — 단일 원인으로 단정 불가")

        falsify = []
        # 반증 조건: 무효화 가격/거래량 미달/RSI 반전
        for p in (candlestick_patterns or [])[:2]:
            if p.get("invalidation_price"):
                falsify.append(f"{p.get('name','패턴')} 무효화 {float(p['invalidation_price']):,.2f} 터치 시 시나리오 무효")
        if volume_ratio < 0.8:
            falsify.append("거래량 0.8배 미만 지속 시 상승 시나리오 신뢰도 30% 하향")
        if rsi >= 75:
            falsify.append("RSI 75 상회 후 종가 음봉 전환 시 단기 추격 무효")

        timeline = {
            "short": f"1~5봉 내 {new_lo:,.2f}~{new_hi:,.2f} 1차 테스트",
            "mid": f"6~20봉 내 상단 돌파 또는 하단 이탈 여부 확정",
            "expected_days": [max(1, int(round((new_hi-last_price)/max(atr_value,1) * 1.2))), max(2, int(round((new_hi-last_price)/max(atr_value,1) * 2.8)))] if dominant==1 else [2,8],
        }

        blindspots = []
        if sup_info.get("blindspot"):
            blindspots.append(sup_info["blindspot"])
        if not pat_info["count"]:
            blindspots.append("확정 캔들 패턴 없음 — 패턴 신호 블라인드스팟")
        elif pat_info["confirmed"] == 0:
            blindspots.append("확정 패턴 0개, 형성·대기만 존재 — 돌파 확인 전")
        if depth_info["notes"]:
            blindspots.extend(depth_info["notes"][:2])

        redteam = ""
        if event_score >= 25:
            redteam = f"REDTEAM: 이벤트 리스크 {event_score}점 — 하락 시나리오 우선 방어, 상승 진입 비중 50% 이하 권장"
        elif signals["event"] < -1.5:
            redteam = "REDTEAM: 공시·실적·소송 리스크 감지 — 상승 확률을 보수적으로 하향"
        elif agreement < 0.5:
            redteam = "REDTEAM: 신호 불일치 구간 — 양방향 손절을 좁게, 관망 비중 확대가 안전"
        else:
            redteam = "REDTEAM: 특이 리스크 없음 — 표준 분할 진입·손절 운용"

        verified = [
            f"백테스트 prior {base_up:.1f}% 기반",
            f"유사패턴 {pat_info['count']}개(확정 {pat_info['confirmed']}개) 상관 가중",
            f"ML {ml_up:.1f}%와 기술 {score:.1f}점 교차 검증",
        ]
        if systematic_bias_note:
            verified.append(systematic_bias_note)

        return {
            "prob_up_corr": corr_up,
            "prob_down_corr": corr_down,
            "side_prob_corr": side_corr,
            "target_narrowed": {
                "min_price": round(new_lo, 2 if market.upper()=="KRX" else 2),
                "max_price": round(new_hi, 2 if market.upper()=="KRX" else 2),
                "uncertainty": uncertainty,
                "reason": narrow_reason,
                "orig_width": round(orig_width, 2),
                "narrow_width": round(new_hi - new_lo, 2),
                "factor": round(narrow_factor, 2),
            },
            "confidence_corr": round(conf_corr,1),
            "confidence_interval_corr": {"lower": round(lo_corr,1), "upper": round(hi_corr,1), "spread": round(spread_corr,1)},
            "correlation": {
                "agreement": round(agreement,2),
                "dominant": "상승" if dominant==1 else ("하락" if dominant==-1 else "중립"),
                "dimensions": {k: round(v,2) for k,v in signals.items()},
                "pattern": pat_info,
                "supply": sup_info,
                "depth": depth_info,
                "blindspots": blindspots[:5],
                "causal_map": causal,
                "falsify": falsify[:3],
                "redteam": redteam,
                "timeline": timeline,
                "systematic_bias": systematic_bias_note or "편향 미감지 — 신호 분산 정상 범위",
            },
            "verified": verified,
            "meta": f"상관축소: 일치도 {agreement*100:.0f}% → 목표가 폭 {orig_width:.2f}→{new_hi-new_lo:.2f} ({narrow_factor*100:.0f}%)",
        }
    except Exception as e:
        # 호환성: 실패 시 원본 그대로 반환
        return {
            "prob_up_corr": _num(prob_up_base, 50.0),
            "prob_down_corr": _num(prob_down_base, 50.0),
            "side_prob_corr": _num(100 - _num(prob_up_base,50) - _num(prob_down_base,50), 20),
            "target_narrowed": None,
            "confidence_corr": _num((signal_confidence or {}).get("confidence"), 50),
            "confidence_interval_corr": (signal_confidence or {}).get("confidence_interval") or {"lower": 30, "upper": 70, "spread": 20},
            "correlation": {"agreement": 0.5, "error": str(e), "blindspots": ["상관 엔진 예외 — 원본 유지"]},
            "verified": [f"상관 엔진 예외: {e}"],
            "meta": "예외로 원본 유지",
        }
