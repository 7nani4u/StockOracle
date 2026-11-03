"""범위 기반 일간 변동성 추정기: Parkinson · Garman-Klass · Rogers-Satchell · Yang-Zhang.

종가-종가 표준편차는 하루 안의 고가·저가 정보를 쓰지 않는다. 시가·고가·저가·종가(OHLC)를 모두 쓰는 추정기는
같은 표본 수에서 분산이 작다. 공식은 MQL5 Articles 23830(Riccio, 2026-10-01)과 23454(Gbadebo, 2026-08-05)의
정의를 파이썬으로 옮겼고, 두 글이 일치하는 부분만 사용했다(MQL5 코드는 가져오지 않았다).

    o=ln(O/전일C)  c=ln(C/O)  u=ln(H/O)  d=ln(L/O)  x=ln(H/L)
    Parkinson      (1/4n ln2) Σ x²
    Garman-Klass   (1/n) Σ [½x² − (2ln2−1)c²]
    Rogers-Satchell(1/n) Σ [u(u−c) + d(d−c)]
    Yang-Zhang     var(o) + k·var(c) + (1−k)·RS,  k = 0.34 / (1.34 + (n+1)/(n−1))   (var 은 n−1 분모)

결측·비정상 봉은 이웃과 이어 붙이지 않고 건너뛴다(OHLC 가 양수가 아니거나 H<L, O/C 가 [L,H] 밖이면 무효).
Yang-Zhang 의 전일 대비 갭은 바로 앞 봉도 유효할 때만 쓴다. 반환값은 모두 '비연율 일간 분산(σ²)'이다.
어떤 추정기가 더 정확한지는 데이터에 따라 달라진다 — 서비스 반영 여부는 scripts/compare_vol_estimators.py 의
사전 고정 검증 결과로 판단한다.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

METHODS = ("parkinson", "garman_klass", "rogers_satchell", "yang_zhang")
_LN2 = math.log(2.0)
# 부동소수 외피 오차(adjusted 가격에서 O/C 가 H/L 을 1e-11 정도 벗어나는 경우)는 무효로 보지 않는다.
_ENVELOPE_TOL = 1e-6


def _terms(opens: Iterable, highs: Iterable, lows: Iterable, closes: Iterable) -> Dict[str, np.ndarray] | None:
    """봉별 항을 한 번에 계산한다. 길이가 다르면 None."""
    try:
        o, h, l, c = (np.asarray(list(v), dtype=float) for v in (opens, highs, lows, closes))
    except (TypeError, ValueError):
        return None
    if not (len(o) == len(h) == len(l) == len(c)):
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
        positive = (o > 0) & (h > 0) & (l > 0) & (c > 0)
        ordered = h >= l * (1 - _ENVELOPE_TOL)
        inside = ((o <= h * (1 + _ENVELOPE_TOL)) & (o >= l * (1 - _ENVELOPE_TOL)) &
                  (c <= h * (1 + _ENVELOPE_TOL)) & (c >= l * (1 - _ENVELOPE_TOL)))
        valid = finite & positive & ordered & inside
        # 외피 오차는 H/L 을 O/C 까지 넓혀 흡수한다(가격을 지어내는 것이 아니라 산술적 보정).
        hh = np.where(valid, np.maximum(h, np.maximum(o, c)), 1.0)
        ll = np.where(valid, np.minimum(l, np.minimum(o, c)), 1.0)
        oo = np.where(valid, o, 1.0)
        cc = np.where(valid, c, 1.0)
        x = np.log(hh / ll)
        cr = np.log(cc / oo)
        u = np.log(hh / oo)
        d = np.log(ll / oo)
        prev_close = np.r_[np.nan, cc[:-1]]
        prev_valid = np.r_[False, valid[:-1]]
        gap_ok = valid & prev_valid
        gap = np.where(gap_ok, np.log(oo / np.where(gap_ok, prev_close, 1.0)), 0.0)
    return {
        "valid": valid, "gap_ok": gap_ok, "gap": gap, "cr": cr,
        "pk": x * x / (4.0 * _LN2),
        "gk": 0.5 * x * x - (2.0 * _LN2 - 1.0) * cr * cr,
        "rs": u * (u - cr) + d * (d - cr),
    }


def _variance_from_terms(method: str, t: Dict[str, np.ndarray], lo: int, hi: int, min_obs: int) -> float | None:
    valid = t["valid"][lo:hi]
    n = int(valid.sum())
    if n < max(1, min_obs):
        return None
    if method == "parkinson":
        return float(max(t["pk"][lo:hi][valid].mean(), 0.0))
    if method == "garman_klass":
        return float(max(t["gk"][lo:hi][valid].mean(), 0.0))
    if method == "rogers_satchell":
        return float(max(t["rs"][lo:hi][valid].mean(), 0.0))
    if method == "yang_zhang":
        gap_ok = t["gap_ok"][lo:hi]
        m = int(gap_ok.sum())
        if m < max(3, min_obs):
            return None
        gaps = t["gap"][lo:hi][gap_ok]
        cr = t["cr"][lo:hi][gap_ok]
        rs = t["rs"][lo:hi][gap_ok]
        k = 0.34 / (1.34 + (m + 1.0) / (m - 1.0))
        return float(max(gaps.var(ddof=1) + k * cr.var(ddof=1) + (1.0 - k) * rs.mean(), 0.0))
    return None


def variance(method: str, opens: Sequence, highs: Sequence, lows: Sequence, closes: Sequence,
             window: int = 20, min_obs: int = 10) -> float | None:
    """마지막 window 개 봉의 일간 분산. 유효한 봉이 min_obs 보다 적거나 입력이 잘못되면 None."""
    if method not in METHODS or window < 1:
        return None
    t = _terms(opens, highs, lows, closes)
    if t is None:
        return None
    size = len(t["valid"])
    if size == 0:
        return None
    return _variance_from_terms(method, t, max(0, size - window), size, min_obs)


def daily_sigma(method: str, opens: Sequence, highs: Sequence, lows: Sequence, closes: Sequence,
                window: int = 20, min_obs: int = 10) -> float | None:
    value = variance(method, opens, highs, lows, closes, window=window, min_obs=min_obs)
    return None if value is None else math.sqrt(value)


def rolling_sigma(method: str, opens: Sequence, highs: Sequence, lows: Sequence, closes: Sequence,
                  window: int = 20, min_obs: int = 10) -> List[float]:
    """각 시점 i 에서 i 이하 봉만 쓴 σ. 추정이 불가능한 시점은 NaN(인과적: 미래 봉을 쓰지 않는다)."""
    t = _terms(opens, highs, lows, closes)
    if t is None or method not in METHODS or window < 1:
        return []
    out: List[float] = []
    for i in range(len(t["valid"])):
        value = _variance_from_terms(method, t, max(0, i + 1 - window), i + 1, min_obs)
        out.append(math.nan if value is None else math.sqrt(value))
    return out
