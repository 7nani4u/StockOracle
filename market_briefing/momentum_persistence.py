# -*- coding: utf-8 -*-
"""
momentum_persistence.py — 급등 후 눌림 버팀(모멘텀 지속) 판정기 (연구용, 점수 미반영).

전략 정의 (영상 요약 기준, 종가 기준 순수 함수):
  ① 하루 +20% 이상 급등한 종목을 찾는다 (surge: C[T]/C[T-1]-1 >= 20%).
  ② 이후 3거래일 동안 상승분의 절반 이상을 지키면 매수 후보로 본다.
     - 상승분 G = C[T] - C[T-1], 중간값 M = C[T-1] + G/2 = (C[T]+C[T-1])/2.
     - 3일 종가가 모두 M 이상이면 PASS (CLOSE 기준, 기본).
     - 저점 기준 엄격판정(STRICT)은 참고용으로만 병기한다.

상태:
  SURGE  급등 당일 — 3일 관찰 필요
  WAIT   급등 후 1~2일차, 지금까지는 M 상회 — 대기
  PASS   급등+3일, 3일 종가 모두 M 이상 — 후보 (진입=확인일 종가, 손절=M 아래)
  FAIL   3일 안에 종가가 M 하회 — 탈락
  NONE   최근 구간에 급등 없음 / 데이터 부족 / 거래량 필터 미충족

주의: 확정 예측·매수 권유가 아니라 연구용 신호다. 점수·확률·스캔 가중에는
반영하지 않는다. 52종목 캐시 실측(2022-10~2026-09, 64,300봉):
  급등 95건(0.15%), PASS 76 / FAIL 19,
  PASS 진입 후 5일 평균 -0.04%·중앙값 -2.52%·상승률 42% (무작위 5일 +0.48%/53% 하회),
  10일 +4.73% vs FAIL +1.52%, 20일 +2.41% vs FAIL -0.84% (표본 작고 생존편향).
  즉 PASS/FAIL 분리는 있으나 절대 우위가 비용·슬리피지 앞에서 증명되지 않아
  INSUFFICIENT扱い. 비용(KRX 0.2%·US 0.1%+갭 슬리피지 1~3%)을 빼면 5일은 음수.
  `scripts/backtest_momentum_persistence.py` 로 재현.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# ── 임계값 ─────────────────────────────────────────────────────────────────
SURGE_MIN_PCT = 20.0   # ① 당일 급등 최소폭 (%)
HOLD_DAYS = 3          # ② 관찰 거래일 수
RETAIN_FRAC = 0.5      # ② 지켜야 하는 상승분 비율 (절반)
PASS_EXPIRY = 5        # PASS/FAIL 판정 유효 기간 (확인일 포함 거래일)
MIN_BARS = 8           # 최소 봉 수 (급등 1 + 여유)


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


def find_surges(closes: List[float], threshold_pct: float = SURGE_MIN_PCT) -> List[Dict[str, Any]]:
    """전체 구간에서 threshold 이상 당일 급등 목록 [{index, pct}]."""
    out: List[Dict[str, Any]] = []
    for i in range(1, len(closes)):
        base, top = closes[i - 1], closes[i]
        if base and base > 0 and top and top > 0:
            pct = (top / base - 1.0) * 100.0
            if pct >= threshold_pct:
                out.append({"index": i, "pct": round(pct, 2)})
    return out


def detect_momentum_persistence(
    closes: List[float],
    highs: List[float] | None = None,
    lows: List[float] | None = None,
    volumes: List[float] | None = None,
    surge_min_pct: float = SURGE_MIN_PCT,
    hold_days: int = HOLD_DAYS,
    retain_frac: float = RETAIN_FRAC,
    require_volume_ratio: float | None = None,
    expiry: int = PASS_EXPIRY,
) -> Dict[str, Any]:
    """최신 바 기준 모멘텀 지속 단계 판정. 절대 raise하지 않는다."""
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

        surges = find_surges(c, surge_min_pct)
        if not surges:
            return {"available": True, "stage": "NONE", "stage_label": "해당 없음",
                    "reason": f"최근 {n}봉에 +{surge_min_pct:.0f}% 이상 급등 없음"}
        # 가장 최근 급등을 기준으로 평가한다.
        surge_idx = surges[-1]["index"]
        surge_pct = surges[-1]["pct"]
        base, top = c[surge_idx - 1], c[surge_idx]
        midpoint = base + (top - base) * retain_frac
        bars_since_surge = (len(c) - 1) - surge_idx

        # 거래량 필터 (선택): 급등일 거래량/직전 20일 평균
        volume_ratio: Optional[float] = None
        if surge_idx >= 5:
            window = [x for x in v[max(0, surge_idx - 20):surge_idx] if x > 0]
            if window and v[surge_idx] > 0:
                volume_ratio = v[surge_idx] / (sum(window) / len(window))
        if require_volume_ratio is not None:
            if volume_ratio is None or volume_ratio < require_volume_ratio:
                return {"available": True, "stage": "NONE", "stage_label": "해당 없음",
                        "reason": "거래량 동반 부족",
                        "surge_index": surge_idx, "surge_pct": surge_pct,
                        "volume_ratio": volume_ratio}

        # 급등 후 경과별 종가·저점
        post_c = c[surge_idx + 1:]
        post_l = l[surge_idx + 1:]
        need = min(hold_days, len(post_c))
        broke_idx: Optional[int] = None
        for k, px in enumerate(post_c[:hold_days]):
            if px < midpoint:
                broke_idx = surge_idx + 1 + k
                break
        strict_broke = any(px < midpoint for px in post_l[:hold_days]) if post_l else None

        if bars_since_surge == 0:
            stage, label = "SURGE", "급등 당일"
        elif bars_since_surge < hold_days:
            if broke_idx is not None:
                stage, label = "FAIL", "절반 이탈"
            else:
                stage, label = "WAIT", f"관찰 {bars_since_surge}/{hold_days}일"
        else:
            # 3일 window 확정
            if broke_idx is not None:
                stage, label = "FAIL", "절반 이탈"
            else:
                stage, label = "PASS", "절반 수성"
            # 만료: 확인 후 expiry를 넘긴 옛 급등은 NONE으로 돌린다.
            if bars_since_surge > hold_days + expiry:
                return {"available": True, "stage": "NONE", "stage_label": "해당 없음",
                        "reason": f"급등 후 {hold_days + expiry}봉 경과로 만료",
                        "surge_index": surge_idx, "surge_pct": surge_pct,
                        "midpoint": round(midpoint, 4)}

        min_hold = min(post_c[:hold_days]) if len(post_c) >= hold_days else (min(post_c) if post_c else None)
        retain = ((min_hold - base) / (top - base)) if (min_hold is not None and top != base) else None
        max_dd = ((min(post_l[:hold_days]) - top) / top * 100.0) if len(post_l) >= hold_days and top > 0 else None

        atr = _atr14(h, l, c)
        entry = c[surge_idx + hold_days] if stage == "PASS" and len(c) > surge_idx + hold_days else (c[-1] if stage == "PASS" else None)
        stop = (midpoint - 0.5 * atr) if (atr and stage == "PASS") else None
        if stop is not None and stop <= 0:
            stop = midpoint * 0.99

        return {
            "available": True,
            "stage": stage,
            "stage_label": label,
            "conditions": {
                "surge": {"passed": True, "detail": f"당일 {surge_pct:+.1f}% (기준 +{surge_min_pct:.0f}%)"},
                "persistence": {"passed": stage in ("WAIT", "PASS"),
                                "detail": (f"{hold_days}일 종가 모두 중간값 상회" if stage == "PASS"
                                           else (f"중간값 하회 {broke_idx}봉" if stage == "FAIL"
                                                 else f"{bars_since_surge}/{hold_days}일 경과·이탈 없음"))},
            },
            "surge_index": int(surge_idx),
            "bars_since_surge": int(bars_since_surge),
            "surge_pct": round(surge_pct, 2),
            "base": round(base, 4),
            "top": round(top, 4),
            "midpoint": round(midpoint, 4),
            "retain_frac": retain_frac,
            "min_hold_close": round(min_hold, 4) if min_hold is not None else None,
            "retain_ratio": round(retain, 3) if retain is not None else None,
            "max_drawdown_from_top_pct": round(max_dd, 2) if max_dd is not None else None,
            "strict_low_break": bool(strict_broke) if strict_broke is not None else None,
            "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
            "entry_trigger": round(entry, 4) if entry else None,
            "stop_price": round(stop, 4) if stop else None,
            "summary": (
                f"{label} — 당일 {surge_pct:+.1f}%, 중간값 {midpoint:.2f} 대비 "
                f"{'수성' if stage in ('WAIT', 'PASS') else '이탈' if stage == 'FAIL' else '관찰 필요'}"
            ),
        }
    except Exception as e:
        return {"available": False, "stage": "NONE", "stage_label": "계산 실패",
                "reason": f"{type(e).__name__}: {e}"}
