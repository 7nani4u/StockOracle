# -*- coding: utf-8 -*-
"""
vcp.py — 변동성 수축 패턴(VCP, Volatility Contraction Pattern) 판정기 (연구용, 점수 미반영).

영상 요약 기준, 종가·고가·저가만으로 계산하는 순수 함수:
  ① 고점 이후 출렁임(조정 깊이)이 점점 좁아지는 주식을 찾는다 (예: 25% → 15% → 8%).
  ② 마지막으로 좁아진 고점(피벗)을 종가로 뚫는 순간이 진입 후보(PASS).
  ③ 시장 약세·거래량 미동반·약한 마감은 가짜 돌파 위험으로 따로 표시한다.

단계:
  PASS         피벗 종가 돌파(최근 5봉 이내) → 후보 (진입=피벗, 손절=최종 수축 저점-0.5ATR)
  CONTRACTING  3단 수축 완성, 피벗 6% 아래 대기 — 돌파 미발생
  NONE         구조 없음 / 데이터 부족 / 만료 / 거래량 필터 미충족

주의: 확정 예측·매수 권유가 아니라 연구용 신호다. 점수·확률 가중에는 반영하지
않고, 스캔에서는 PASS + 가짜돌파 위험없음일 때만 READY 승격한다.
52종목 캐시 실측(신호일 종가 돌파 신선건만 94건): 5일 위험없음 -0.74% vs 위험 -2.30%,
10일 -1.03% vs -1.40%, 20일 +3.31% vs -1.74%. 위험 플래그는 분리되나 절대 우위가
비용 앞에서 증명되지 않아 INSUFFICIENT扱い. `scripts/backtest_vcp.py` 로 재현.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# ── 임계값 ─────────────────────────────────────────────────────────────────
VCP_WINDOW = 60              # 베이스 탐색 구간 (거래일)
VCP_MIN_CONTRACTIONS = 3     # 최소 수축 단수
VCP_FIRST_MIN_DEPTH = 0.12   # 첫 조정 최소 깊이 (12%)
VCP_LAST_MAX_DEPTH = 0.12    # 마지막 조정 최대 깊이 (12%)
VCP_MAX_DEPTH = 0.45         # 첫 조정 상한 (넘으면 붕괴로 보고 제외)
VCP_SHRINK_RATIO = 0.85      # 다음 조정은 이전의 85% 이하로 좁아져야 함
VCP_ADVANCE_MIN = 0.15       # 베이스 고점까지 사전 상승 최소폭 (15%)
VCP_PROXIMITY_PCT = 6.0      # CONTRACTING 인정 피벗 하방 허용폭 (%)
VCP_PASS_EXPIRY = 5          # 돌파 확인 유효 기간 (거래일)
VCP_FRACTAL_K = 2            # 스윙 판정 프랙탈 반경
MIN_BARS = 65                # 최소 봉 수 (60 + 여유)


def _fnum(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _clean(values: Any) -> List[float]:
    return [f for v in (values or []) for f in ([_fnum(v)] if _fnum(v) is not None else [])]


def _atr14(highs: List[float], lows: List[float], closes: List[float]) -> Optional[float]:
    n = min(len(highs), len(lows), len(closes))
    if n < 15:
        return None
    trs = []
    for i in range(n - 14, n):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs) if trs else None


def _swing_points(highs: List[float], lows: List[float], k: int = VCP_FRACTAL_K
                  ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """프랙탈 스윙 고점/저점 [(index, price)]."""
    n = len(highs)
    sh, sl = [], []
    for i in range(k, n - k):
        window_h = highs[i - k:i + k + 1]
        window_l = lows[i - k:i + k + 1]
        if highs[i] == max(window_h) and highs[i] > 0:
            sh.append((i, highs[i]))
        if lows[i] == min(window_l) and lows[i] > 0:
            sl.append((i, lows[i]))
    return sh, sl


def detect_vcp(
    closes: List[float],
    highs: List[float] | None = None,
    lows: List[float] | None = None,
    volumes: List[float] | None = None,
    market_regime: str = "UNKNOWN",
) -> Dict[str, Any]:
    """VCP 구조 판정. 절대 raise하지 않는다."""
    try:
        c = _clean(closes)
        if len(c) < MIN_BARS:
            return {"available": False, "stage": "NONE", "stage_label": "데이터 부족",
                    "reason": f"최소 {MIN_BARS}봉 필요(현재 {len(c)}봉)"}
        h = _clean(highs) if highs else list(c)
        l = _clean(lows) if lows else list(c)
        n = min(len(c), len(h), len(l))
        c, h, l = c[-n:], h[-n:], l[-n:]
        v = [_fnum(x) or 0.0 for x in (volumes or [])]
        v = (v[-n:] if len(v) >= n else [0.0] * n)
        last = c[-1]

        window = c[-VCP_WINDOW:]
        window_h = h[-VCP_WINDOW:]
        window_l = l[-VCP_WINDOW:]
        base_idx_win = window_h.index(max(window_h))
        base_high = window_h[base_idx_win]
        base_idx = n - VCP_WINDOW + base_idx_win
        advance = (base_high / window[0] - 1.0) if window[0] > 0 else 0.0

        sh, sl = _swing_points(h, l)
        # 베이스 고점 이후의 스윙만 본다 (베이스 고점 자체가 첫 스윙 High).
        post_sh = [(i, p) for i, p in sh if i > base_idx - VCP_FRACTAL_K]
        if post_sh and post_sh[0][0] <= base_idx:
            post_sh = [(base_idx, base_high)] + [(i, p) for i, p in post_sh if i > base_idx]
        else:
            post_sh = [(base_idx, base_high)] + post_sh
        post_sl = [(i, p) for i, p in sl if i > base_idx]

        # 고점→저점→고점 순서로 조정 깊이를 잰다.
        depths: List[float] = []
        for j in range(len(post_sh) - 1):
            hi, hp = post_sh[j]
            nhi = post_sh[j + 1][0]
            mids = [p for i, p in post_sl if hi < i < nhi]
            if not mids:
                seg = l[hi + 1:nhi] if nhi > hi + 1 else []
                if not seg:
                    break
                mids = [min(seg)]
            trough = min(mids)
            if hp > 0:
                depths.append((hp - trough) / hp)
        # 마지막 고점 이후(진행 중 구간) 조정도 포함한다.
        if post_sh:
            hi, hp = post_sh[-1]
            tail = l[hi + 1:] if hi + 1 < n else []
            if tail:
                trough = min(tail)
                if hp > 0:
                    depths.append((hp - trough) / hp)

        def _no(reason: str) -> Dict[str, Any]:
            return {"available": True, "stage": "NONE", "stage_label": "해당 없음",
                    "reason": reason, "contractions": len(depths),
                    "depths_pct": [round(d * 100, 1) for d in depths[:5]],
                    "advance_pct": round(advance * 100, 1)}

        if len(depths) < VCP_MIN_CONTRACTIONS:
            return _no(f"수축 {len(depths)}단 (기준 {VCP_MIN_CONTRACTIONS}단 이상)")
        depths = depths[-VCP_MIN_CONTRACTIONS:]
        if not all(depths[j] > depths[j + 1] for j in range(len(depths) - 1)):
            return _no("조정 깊이 비수축 (점점 좁아지지 않음)")
        if not all(depths[j + 1] / depths[j] <= VCP_SHRINK_RATIO + 1e-9 for j in range(len(depths) - 1) if depths[j] > 0):
            return _no("수축 비율 미달 (이전의 85% 이하로 좁아지지 않음)")
        if depths[0] < VCP_FIRST_MIN_DEPTH or depths[0] > VCP_MAX_DEPTH:
            return _no(f"첫 조정 {depths[0]*100:.1f}%가 범위(12~45%) 밖")
        if depths[-1] > VCP_LAST_MAX_DEPTH:
            return _no(f"마지막 조정 {depths[-1]*100:.1f}%가 12% 초과로 덜 조여짐")
        if advance < VCP_ADVANCE_MIN:
            return _no(f"사전 상승 {advance*100:.1f}%가 15% 미만 (베이스 아님)")

        pivot = post_sh[-1][1]
        pivot_idx = post_sh[-1][0]
        # 피벗 이후 첫 종가 돌파 위치
        cross_idx: Optional[int] = None
        for i in range(pivot_idx + 1, n):
            if c[i] > pivot:
                cross_idx = i
                break
        bars_since_cross = (n - 1 - cross_idx) if cross_idx is not None else None

        # 거래량 (돌파봉/직전 20일 평균)
        volume_ratio: Optional[float] = None
        ref_idx = cross_idx if cross_idx is not None else n - 1
        if ref_idx >= 5:
            win = [x for x in v[max(0, ref_idx - 20):ref_idx] if x > 0]
            if win and v[ref_idx] > 0:
                volume_ratio = v[ref_idx] / (sum(win) / len(win))

        # 가짜 돌파 위험 (셋째 요구사항)
        risks: List[str] = []
        if str(market_regime or "").upper() == "BEARISH":
            risks.append("약세장 돌파")
        if volume_ratio is not None and volume_ratio < 1.2:
            risks.append("거래량 미동반")
        rng = h[-1] - l[-1]
        if rng > 0 and (c[-1] - l[-1]) / rng < 0.5:
            risks.append("종가 하단 마감")
        body = abs(c[-1] - (c[-2] if n >= 2 else c[-1]))
        wick = h[-1] - max(c[-1], c[-2] if n >= 2 else c[-1])
        if body > 0 and wick / body > 2.0:
            risks.append("윗꼬리 우세")

        atr = _atr14(h, l, c)
        tail_low = min(l[pivot_idx + 1:]) if pivot_idx + 1 < n else l[-1]
        entry = pivot
        stop = (tail_low - 0.5 * atr) if atr else tail_low * 0.98
        if stop <= 0:
            stop = tail_low * 0.98

        common = {
            "available": True,
            "conditions": {
                "contraction": {"passed": True,
                                "detail": "조정 " + "→".join(f"{d*100:.0f}%" for d in depths)},
                "advance": {"passed": True, "detail": f"베이스까지 +{advance*100:.1f}%"},
            },
            "contractions": len(depths),
            "depths_pct": [round(d * 100, 1) for d in depths],
            "advance_pct": round(advance * 100, 1),
            "pivot": round(pivot, 4),
            "pivot_index": int(pivot_idx),
            "contraction_low": round(tail_low, 4),
            "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
            "false_breakout_risk": bool(risks),
            "risk_reasons": risks,
            "entry_trigger": round(entry, 4),
            "stop_price": round(stop, 4),
        }
        if cross_idx is not None and bars_since_cross is not None and bars_since_cross <= VCP_PASS_EXPIRY:
            return {**common, "stage": "PASS", "stage_label": "피벗 돌파",
                    "cross_index": int(cross_idx), "bars_since_cross": int(bars_since_cross),
                    "summary": (f"피벗 돌파 — 조정 "
                                + "→".join(f"{d*100:.0f}%" for d in depths)
                                + (f" · 가짜돌파 주의({', '.join(risks)})" if risks else ""))}
        if cross_idx is not None:
            return {**common, "stage": "NONE", "stage_label": "해당 없음",
                    "reason": f"돌파 후 {bars_since_cross}봉 경과로 만료"}
        dist_pct = (pivot - last) / pivot * 100.0 if pivot > 0 else 999.0
        if 0 <= dist_pct <= VCP_PROXIMITY_PCT:
            return {**common, "stage": "CONTRACTING", "stage_label": "수축 진행",
                    "dist_to_pivot_pct": round(dist_pct, 2),
                    "summary": (f"수축 진행 — 피벗 대비 -{dist_pct:.1f}%, 조정 "
                                + "→".join(f"{d*100:.0f}%" for d in depths))}
        return {**common, "stage": "NONE", "stage_label": "해당 없음",
                "reason": f"피벗 대비 -{dist_pct:.1f}%로 관측권 밖"}
    except Exception as e:
        return {"available": False, "stage": "NONE", "stage_label": "계산 실패",
                "reason": f"{type(e).__name__}: {e}"}
