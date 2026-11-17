"""반응 기반 수평 지지·저항선(2회 이상 반응) 탐지와 지지↔저항 전환 사건.

첨부 도식 '수평선'의 요령은 "2번 이상 반등하는 곳에 선을 긋는다"이고, 뚫린 저항은 지지로, 뚫린
지지는 저항으로 바뀐다(전환)는 것이다. 클래식 피봇(``calc_pivot_points``)은 전일 고저종 공식이라
가격이 실제로 반응한 자리와 무관하고, 박스 엔진은 박스가 있을 때만 경계를 낸다. 이 모듈은 확정된
스윙 고점·저점만으로 '가격이 반응한 수평선'을 인과적으로 만든다.

인과성
------
* 피벗은 오른쪽 ``pivot_window`` 봉이 지난 뒤에만 쓴다(마지막 봉 이후 정보 없음).
* 모든 함수는 배열 마지막 시점까지만 읽는다. 접두(prefix) 배열에 돌려도 그 날 실서비스가 냈을 값과 같다.

한계: 수평선 후보·허용 폭(ATR 배수)은 공학적 선택이며 이 모듈 자체는 수익 근거를 주장하지 않는다.
근거는 ``scripts/audit_chart_patterns.py levels`` 의 사건 연구 결과로만 판단한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

VERSION = "2026-11-17.sr-levels-v1"


@dataclass(frozen=True)
class LevelConfig:
    pivot_window: int = 4          # 일봉 패턴 엔진과 같은 피벗 폭
    atr_period: int = 14
    lookback: int = 250            # 실서비스 1y 일봉과 같은 길이
    tol_atr: float = 0.5           # 같은 선으로 묶는 허용 폭(ATR 배수)
    min_touches: int = 2           # "2번 이상 반등하는 곳"
    min_touch_gap: int = 3         # 같은 스윙을 두 번 세지 않도록 접촉 간 최소 봉 수
    break_atr: float = 0.5         # 종가가 선을 이만큼 넘어야 돌파/이탈
    retest_atr: float = 0.25       # 재시험 접촉 허용 폭(ATR 배수)
    max_flip_age: int = 30         # 돌파 후 재시험까지 허용 봉 수
    fresh_bars: int = 3            # 직전 N봉은 선에서 떨어져 있어야 '새 접촉'


def _finite_arrays(*seqs: Sequence[Any], tail: int | None = None) -> list[np.ndarray] | None:
    """길이를 맞춘 float 배열. ``tail`` 이 있으면 마지막 ``tail`` 봉만 검사한다(먼 과거의 결측이 전체를 막지 않게)."""
    n = min(len(s) for s in seqs)
    if tail:
        n = min(n, int(tail))
    out = []
    for s in seqs:
        try:
            a = np.asarray(s[-n:] if n else [], dtype=float)
        except (TypeError, ValueError):
            return None
        out.append(a)
    if n < 20 or not all(np.isfinite(a).all() for a in out):
        return None
    return out


def atr_series(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    prev = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum.reduce([highs - lows, np.abs(highs - prev), np.abs(lows - prev)])
    csum = np.cumsum(np.concatenate([[0.0], tr]))
    idx = np.arange(1, len(tr) + 1)
    start = np.maximum(0, idx - period)
    return (csum[idx] - csum[start]) / (idx - start)


def confirmed_swings(highs: np.ndarray, lows: np.ndarray, window: int) -> list[tuple[int, float, str]]:
    """오른쪽 ``window`` 봉이 지나 확정된 스윙 고점·저점. 같은 값이면 왼쪽(앞선) 봉을 택한다."""
    n = len(highs)
    out: list[tuple[int, float, str]] = []
    for i in range(window, n - window):
        left_h, right_h = highs[i - window:i], highs[i + 1:i + window + 1]
        if highs[i] > left_h.max() and highs[i] >= right_h.max():
            out.append((i, float(highs[i]), "high"))
        left_l, right_l = lows[i - window:i], lows[i + 1:i + window + 1]
        if lows[i] < left_l.min() and lows[i] <= right_l.min():
            out.append((i, float(lows[i]), "low"))
    return out


def cluster_levels(swings: Sequence[tuple[int, float, str]], tol: float, min_gap: int) -> list[dict[str, Any]]:
    """가격 순으로 허용 폭 안의 스윙을 한 선으로 묶는다. 한 선의 접촉은 서로 ``min_gap`` 봉 이상 떨어져야 센다."""
    ordered = sorted(swings, key=lambda s: s[1])
    clusters: list[list[tuple[int, float, str]]] = []
    for sw in ordered:
        if clusters:
            mean = sum(p[1] for p in clusters[-1]) / len(clusters[-1])
            if sw[1] - mean <= tol:
                clusters[-1].append(sw)
                continue
        clusters.append([sw])
    levels = []
    for members in clusters:
        kept: list[tuple[int, float, str]] = []
        for sw in sorted(members, key=lambda s: s[0]):
            if kept and sw[0] - kept[-1][0] < min_gap:
                continue
            kept.append(sw)
        levels.append({
            "price": float(sum(p[1] for p in kept) / len(kept)),
            "touches": len(kept),
            "pivots": [(p[0], p[2]) for p in kept],
        })
    return levels


def find_levels(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
                config: LevelConfig | None = None) -> list[dict[str, Any]]:
    """마지막 봉 기준 수평선 목록(접촉 ``min_touches`` 이상). 가격 오름차순."""
    cfg = config or LevelConfig()
    arrs = _finite_arrays(highs, lows, closes, tail=cfg.lookback)
    if arrs is None:
        return []
    h, l, c = arrs
    atr = atr_series(h, l, c, cfg.atr_period)
    tol = cfg.tol_atr * float(atr[-1])
    if tol <= 0:
        return []
    levels = cluster_levels(confirmed_swings(h, l, cfg.pivot_window), tol, cfg.min_touch_gap)
    last = float(c[-1])
    out = []
    for lv in levels:
        if lv["touches"] < cfg.min_touches:
            continue
        lv = dict(lv)
        lv["role"] = "support" if last > lv["price"] else "resistance"
        lv["last_touch_index"] = max(i for i, _ in lv["pivots"])
        lv["bars_since_touch"] = len(c) - 1 - lv["last_touch_index"]
        out.append(lv)
    return sorted(out, key=lambda x: x["price"])


def level_events(opens: Sequence[float], highs: Sequence[float], lows: Sequence[float],
                 closes: Sequence[float], config: LevelConfig | None = None,
                 min_touches: int | None = None) -> list[dict[str, Any]]:
    """마지막 봉에서 발생한 수평선 사건.

    * ``support_bounce`` / ``resistance_rejection``: 선에서 떨어져 있던 가격이 선 근처를 건드리고 종가는
      선 바깥(위/아래)에 머문 새 접촉.
    * ``flip_support`` / ``flip_resistance``: 최근 ``max_flip_age`` 봉 안에 선을 ``break_atr`` 이상 종가로
      뚫은 뒤 한 번도 되돌아오지 않고 재시험(지지↔저항 전환 확인).
    ``min_touches`` 를 낮추면 연구용으로 1회 접촉 선도 사건으로 센다(접촉 수는 ``touches`` 에 기록).
    """
    cfg = config or LevelConfig()
    need = cfg.min_touches if min_touches is None else int(min_touches)
    arrs = _finite_arrays(opens, highs, lows, closes, tail=cfg.lookback)
    if arrs is None:
        return []
    o, h, l, c = arrs
    n = len(c)
    atr = atr_series(h, l, c, cfg.atr_period)
    a_t = float(atr[-1])
    if a_t <= 0 or n < cfg.pivot_window * 2 + cfg.fresh_bars + 3:
        return []
    swings = confirmed_swings(h, l, cfg.pivot_window)
    levels = cluster_levels(swings, cfg.tol_atr * a_t, cfg.min_touch_gap)
    t = n - 1
    events: list[dict[str, Any]] = []
    for lv in levels:
        if lv["touches"] < need:
            continue
        price = lv["price"]
        zone = cfg.retest_atr * a_t
        fresh_lo = float(np.min(l[t - cfg.fresh_bars:t]))
        fresh_hi = float(np.max(h[t - cfg.fresh_bars:t]))
        # ── 지지 쪽 접촉: 저가가 선 근처까지 내려왔지만 종가는 선 위
        support_touch = (l[t] <= price + zone and c[t] > price and c[t - 1] > price
                         and fresh_lo > price + zone and l[t] >= price - cfg.break_atr * a_t)
        resistance_touch = (h[t] >= price - zone and c[t] < price and c[t - 1] < price
                            and fresh_hi < price - zone and h[t] <= price + cfg.break_atr * a_t)
        if not (support_touch or resistance_touch):
            continue
        side = 1 if support_touch else -1
        kind = "support_bounce" if side > 0 else "resistance_rejection"
        flip_age = None
        pre_touches = lv["touches"]
        # 전환 확인: 선을 반대편에서 뚫은 뒤 재시험까지 되돌아오지 않았는가
        lo_bar = max(1, t - cfg.max_flip_age)
        for b in range(t - 2, lo_bar - 1, -1):
            if side > 0 and c[b] > price + cfg.break_atr * atr[b] and np.any(c[max(0, b - 10):b] <= price):
                if np.all(c[b + 1:t] > price - zone):
                    flip_age = t - b
                    pre_touches = sum(1 for i, _ in lv["pivots"] if i < b)
                break
            if side < 0 and c[b] < price - cfg.break_atr * atr[b] and np.any(c[max(0, b - 10):b] >= price):
                if np.all(c[b + 1:t] < price + zone):
                    flip_age = t - b
                    pre_touches = sum(1 for i, _ in lv["pivots"] if i < b)
                break
        if flip_age is not None:
            kind = "flip_support" if side > 0 else "flip_resistance"
            if pre_touches < need:
                continue
        events.append({
            "type": kind, "side": side, "level": float(price), "touches": int(pre_touches),
            "atr": a_t, "flip_age": flip_age, "stop_ref": float(price - side * cfg.break_atr * a_t),
        })
    return events


# ───────────────────────── 차트 표시용 ─────────────────────────

FLIP_LOOKBACK = 60      # '전환' 꼬리표: 최근 60봉 안에 선의 반대편 종가가 있었는가
SUPPORT_NOTE_KIND = "support_bounce"


def _role_label(level: dict[str, Any], flipped: bool) -> str:
    base = ("지지" if level["role"] == "support" else "저항") + f" {int(level['touches'])}회"
    return base + (" · 전환" if flipped else "")


def chart_levels(opens: Sequence[float], highs: Sequence[float], lows: Sequence[float],
                 closes: Sequence[float], config: LevelConfig | None = None,
                 max_per_side: int = 1) -> dict[str, Any]:
    """일봉 차트에 그릴 수평선(서술용)과 실측 근거 안내.

    * 현재가 위·아래에서 가까운 선을 각각 ``max_per_side`` 개까지만 고른다(차트가 복잡해지지 않게).
    * 선은 '가격이 반응한 자리'를 보여 줄 뿐 매수·매도 신호가 아니다.
    * ``warning`` 은 실측 근거가 ``supported_weak`` 로 판정된 사건(3회 이상 지지 이탈 후 재시험)이 마지막 봉에
      있을 때만 나온다. 근거 파일의 판정이 바뀌면 경고도 자동으로 사라진다.
    * 일봉 전용이다(근거가 일봉 사건 연구라서). 호출 쪽이 기간을 걸러야 한다.
    """
    from .pattern_evidence import evidence_text, level_evidence

    cfg = config or LevelConfig()
    empty = {"version": VERSION, "timeframe": "1D", "levels": [], "warning": None, "note": None}
    arrs = _finite_arrays(opens, highs, lows, closes, tail=cfg.lookback)
    if arrs is None:
        return empty
    o, h, l, c = arrs
    levels = find_levels(h, l, c, cfg)
    if not levels:
        return empty
    atr = atr_series(h, l, c, cfg.atr_period)
    last = float(c[-1])
    flip_start = max(0, len(c) - FLIP_LOOKBACK)

    def flipped(level: dict[str, Any]) -> bool:
        price = level["price"]
        window = c[flip_start:]
        gap = cfg.break_atr * atr[flip_start:]
        if level["role"] == "support":
            return bool(np.any(window < price - gap))
        return bool(np.any(window > price + gap))

    below = sorted((lv for lv in levels if lv["role"] == "support"), key=lambda lv: last - lv["price"])
    above = sorted((lv for lv in levels if lv["role"] == "resistance"), key=lambda lv: lv["price"] - last)
    shown = []
    for lv in below[:max_per_side] + above[:max_per_side]:
        flag = flipped(lv)
        shown.append({"price": round(float(lv["price"]), 4), "touches": int(lv["touches"]), "role": lv["role"],
                      "flipped": flag, "bars_since_touch": int(lv["bars_since_touch"]), "label": _role_label(lv, flag)})
    warning = None
    for ev in level_events(o, h, l, c, cfg, min_touches=3):
        if ev["type"] != "flip_resistance":
            continue
        evidence = level_evidence("flip_resistance", ev["touches"])
        if evidence and evidence.get("verdict") == "supported_weak":
            warning = {"type": "flip_resistance", "level": round(float(ev["level"]), 4), "touches": int(ev["touches"]),
                       "text": f"{int(ev['touches'])}회 지지하던 가격대를 이탈한 뒤 아래에서 재시험 중", "evidence_text": evidence_text(evidence)}
            break
    note = None
    support_ev = level_evidence(SUPPORT_NOTE_KIND, 3)
    if support_ev:
        note = ("수평선은 가격이 2회 이상 반응한 자리를 그린 참고선입니다. 선 근처 반등만으로 매수 근거로 삼기 어렵습니다 — "
                f"지지선 접촉 후 성과: {evidence_text(support_ev)}")
    return {"version": VERSION, "timeframe": "1D", "levels": shown, "warning": warning, "note": note}
