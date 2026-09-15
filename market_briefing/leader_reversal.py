# -*- coding: utf-8 -*-
"""
leader_reversal.py — 상대강도 리더주 반전매매 신호 (미국 스캔 전용 보조 지표)

'하락장에서 아무 주식이나 싸다고 사는 게 아니라, 강했던 주식 중 가장 먼저
하락을 멈춘 주식'을 찾는 방식의 규칙 기반 판정기. 외부 호출 없이 OHLCV만으로
계산해 단위 테스트가 가능하도록 분리했다.

판정 4조건 (모두 종가·고가·저가 기준, 벤치마크=SPY):
  ① 리더주 확인   — 상승기(고점 기준 과거 약 180거래일→고점) 수익률이
                    벤치마크보다 +10%p 이상 우위
  ② 깊은 조정     — trailing 고점(최대 250봉) 대비 -30% 이상 하락한 상태
  ③ 신저가 중단   — 조정 저점이 최소 10거래일 전에 형성되고, 최근 20봉 저점이
                    그 저점보다 높음(하락 멈춤). 벤치가 같은 기간 신저가를
                    만들었으면 시장 역행 확인으로 가산
  ④ 횡보 상단 돌파 — 현재 종가가 최근 20봉 고점(횡보구간 상단) 위에서 마감

단계:
  BREAKOUT      ①+②+③+④  → 반전 진입 트리거 (entry=횡보 상단, stop=조정저점-0.5ATR)
  WAIT_BREAKOUT ①+②+③    → 돌파 대기 (entry=횡보 상단, 조건 동일)
  BASE_BUILDING ①+②      → 바닥 다지기 관찰
  NONE          그 외    → 해당 없음

주의: 확정 예측·매수 권유가 아니라 스캔 보조 신호다. 점수 가중에는 반영하지 않는다.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# ── 임계값 ─────────────────────────────────────────────────────────────────
LOOKBACK_RS = 120          # ① 상대강도 비교 구간 (거래일)
RS_LEADER_EDGE_PP = 10.0   # ① 리더 판정 최소 초과수익 (%p)
RS_ADVANCE_SPAN = 180      # ① 상승기 측정 span (고점 기준 과거 거래일)
RS_FALLBACK_MIN_RET = 30.0  # 벤치 없을 때 대체 기준 (상승기 자체 수익률 %)
PEAK_WINDOW = 250          # ② trailing 고점 탐색 최대 범위
DRAWDOWN_MIN_PCT = 30.0    # ② 최소 조정폭 (%)
TROUGH_MIN_AGO = 10        # ③ 저점 형성 후 최소 경과 (거래일)
RECENT_WINDOW = 20         # ③·④ 최근 관찰 구간 (거래일)
MIN_BARS = 130             # 전체 최소 봉 수 (120 + 여유)


def _fnum(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _clean(values: Any) -> List[float]:
    out: List[float] = []
    for v in values or []:
        f = _fnum(v)
        if f is not None and f > 0:
            out.append(f)
    return out


def _ret_pct(series: List[float], days: int) -> Optional[float]:
    """최근 days 거래일 수익률 (%)."""
    if len(series) < days + 1:
        return None
    base, last = series[-days - 1], series[-1]
    if base <= 0:
        return None
    return (last / base - 1.0) * 100.0


def _atr14(highs: List[float], lows: List[float], closes: List[float]) -> Optional[float]:
    n = min(len(highs), len(lows), len(closes))
    if n < 15:
        return None
    trs = []
    for i in range(n - 14, n):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs) if trs else None


def detect_leader_reversal(
    closes: List[float],
    highs: List[float] | None = None,
    lows: List[float] | None = None,
    bench_closes: List[float] | None = None,
) -> Dict[str, Any]:
    """리더주 반전 4조건 판정. 절대 raise하지 않는다."""
    try:
        c = _clean(closes)
        if len(c) < MIN_BARS:
            return {"available": False, "stage": "NONE", "stage_label": "데이터 부족",
                    "reason": f"최소 {MIN_BARS}봉 필요(현재 {len(c)}봉)"}
        h = _clean(highs) if highs else list(c)
        l = _clean(lows) if lows else list(c)
        # 길이가 어긋나면 꼬리 기준으로 정렬
        n = min(len(c), len(h), len(l))
        c, h, l = c[-n:], h[-n:], l[-n:]
        last = c[-1]

        # ── ① 리더주 확인: '상승장에서 시장보다 강했던' 종목 ──
        # 조정 진행 중에는 trailing 수익률이 당연히 낮으므로, 상승기(시작→고점)
        # 구간의 초과수익으로 판정한다. 고점 이후 기간이 짧으면 trailing 120일로 폴백.
        bench = _clean(bench_closes)
        peak_win = c[-min(PEAK_WINDOW, len(c)):]
        peak = max(peak_win)
        peak_idx = len(c) - 1 - c[::-1].index(peak)
        peak_ago = (len(c) - 1) - peak_idx
        start_ago = min(len(c) - 1, peak_ago + RS_ADVANCE_SPAN)
        start_idx = (len(c) - 1) - start_ago
        adv_bars = peak_idx - start_idx
        bench_available = len(bench) >= RECENT_WINDOW * 2 + 1
        rs_edge: Optional[float] = None
        leader = False
        if adv_bars >= 60 and c[start_idx] > 0:
            stock_adv = (peak / c[start_idx] - 1.0) * 100.0
            if bench_available:
                b_peak_idx = len(bench) - 1 - peak_ago
                b_start_idx = len(bench) - 1 - start_ago
                if b_peak_idx > b_start_idx >= 0 and bench[b_start_idx] > 0:
                    bench_adv = (bench[b_peak_idx] / bench[b_start_idx] - 1.0) * 100.0
                    rs_edge = stock_adv - bench_adv
                    leader = rs_edge >= RS_LEADER_EDGE_PP
                    leader_detail = (f"상승기 수익률 {stock_adv:+.1f}% vs 벤치 {bench_adv:+.1f}% "
                                     f"(초과 {rs_edge:+.1f}%p)")
                else:
                    leader = stock_adv >= RS_FALLBACK_MIN_RET
                    leader_detail = (f"벤치 구간 미확보 — 자체 상승기 수익률 {stock_adv:+.1f}% "
                                     f"(기준 +{RS_FALLBACK_MIN_RET:.0f}%)")
            else:
                leader = stock_adv >= RS_FALLBACK_MIN_RET
                leader_detail = (f"벤치 미확보 — 자체 상승기 수익률 {stock_adv:+.1f}% "
                                 f"(기준 +{RS_FALLBACK_MIN_RET:.0f}%)")
        else:
            # 고점이 너무 최근이면 trailing 120일 비교로 폴백
            stock_ret = _ret_pct(c, LOOKBACK_RS)
            bench_ret = _ret_pct(bench, LOOKBACK_RS) if len(bench) >= LOOKBACK_RS + 1 else None
            if bench_ret is not None and stock_ret is not None:
                rs_edge = stock_ret - bench_ret
                leader = rs_edge >= RS_LEADER_EDGE_PP
                leader_detail = (f"120일 수익률 {stock_ret:+.1f}% vs 벤치 {bench_ret:+.1f}% "
                                 f"(초과 {rs_edge:+.1f}%p)")
            else:
                leader = stock_ret is not None and stock_ret >= RS_FALLBACK_MIN_RET
                leader_detail = ("자체 120일 수익률 기준 판정(벤치 미확보)" if stock_ret is not None
                                 else "수익률 산출 불가")

        # ── ② 깊은 조정 ──
        drawdown = (peak - last) / peak * 100.0 if peak > 0 else 0.0
        deep = drawdown >= DRAWDOWN_MIN_PCT
        # 조정 저점: 고점 이후 최저 저점
        post_peak_lows = l[peak_idx:]
        trough = min(post_peak_lows) if post_peak_lows else last
        trough_ago = (len(c) - 1 - l[::-1].index(trough)) if trough in l else 0
        deep_detail = f"고점 대비 {drawdown:+.1f}% (기준 -{DRAWDOWN_MIN_PCT:.0f}% 이상)"

        # ── ③ 신저가 중단 ──
        recent_lows = l[-RECENT_WINDOW:]
        recent_low = min(recent_lows) if recent_lows else last
        trough_old_enough = trough_ago >= TROUGH_MIN_AGO
        higher_low = recent_low > trough
        no_new_low = bool(trough_old_enough and higher_low)
        # 시장 역행 확인: 벤치가 최근 20봉에 그 이전 20봉 저점보다 낮은 신저가
        market_confirm = False
        if len(bench) >= RECENT_WINDOW * 2 + 1:
            bench_recent_low = min(bench[-RECENT_WINDOW:])
            bench_prior_low = min(bench[-RECENT_WINDOW * 2:-RECENT_WINDOW])
            market_confirm = bool(bench_recent_low < bench_prior_low)
        no_new_low_detail = (
            f"조정저점 {trough_ago}봉 전 형성 · 최근 20봉 저점이 "
            f"{'저점 상회(하락 멈춤)' if higher_low else '저점 하회(진행 중)'}"
            + (" · 벤치 신저가 속 역행" if market_confirm else "")
        )

        # ── ④ 횡보 상단 돌파 ──
        # 당일봉을 제외한 직전 20봉 고점이 횡보구간 상단. 당일봉 포함 시
        # 종가가 당일 고점을 넘을 수 없어 돌파가 영원히 성립하지 않는다.
        _hist_highs = h[-RECENT_WINDOW - 1:-1] if len(h) > RECENT_WINDOW else h[:-1]
        range_high = max(_hist_highs) if _hist_highs else last
        breakout = bool(last > range_high)
        dist_pct = (last - range_high) / range_high * 100.0 if range_high > 0 else 0.0

        # ── 단계 판정 ──
        if leader and deep and no_new_low and breakout:
            stage, stage_label = "BREAKOUT", "리더 반전 돌파"
        elif leader and deep and no_new_low:
            stage, stage_label = "WAIT_BREAKOUT", "돌파 대기"
        elif leader and deep:
            stage, stage_label = "BASE_BUILDING", "바닥 다지기"
        else:
            stage, stage_label = "NONE", "해당 없음"

        atr = _atr14(h, l, c)
        entry = range_high if stage in ("BREAKOUT", "WAIT_BREAKOUT") else None
        stop = (trough - 0.5 * atr) if (atr and stage in ("BREAKOUT", "WAIT_BREAKOUT")) else None
        if stop is not None and stop <= 0:
            stop = trough * 0.98

        return {
            "available": True,
            "stage": stage,
            "stage_label": stage_label,
            "conditions": {
                "leader": {"passed": bool(leader), "detail": leader_detail},
                "deep_correction": {"passed": bool(deep), "detail": deep_detail},
                "no_new_low": {"passed": bool(no_new_low), "detail": no_new_low_detail},
                "breakout": {"passed": bool(breakout),
                             "detail": f"20봉 상단 대비 {dist_pct:+.2f}%"},
            },
            "rs_edge_pp": round(rs_edge, 2) if rs_edge is not None else None,
            "bench_available": bench_available,
            "drawdown_pct": round(drawdown, 2),
            "peak": round(peak, 4),
            "trough": round(trough, 4),
            "trough_ago": int(trough_ago),
            "range_high": round(range_high, 4),
            "market_confirm": bool(market_confirm),
            "entry_trigger": round(entry, 4) if entry else None,
            "stop_price": round(stop, 4) if stop else None,
            "summary": (
                f"{stage_label} — 120일 초과수익 "
                f"{f'{rs_edge:+.1f}%p' if rs_edge is not None else '미확보'}, "
                f"고점 대비 {drawdown:.1f}%, "
                f"{'횡보 상단 돌파' if breakout else '횡보 상단 하회'}"
            ),
        }
    except Exception as e:
        return {"available": False, "stage": "NONE", "stage_label": "계산 실패",
                "reason": f"{type(e).__name__}: {e}"}
