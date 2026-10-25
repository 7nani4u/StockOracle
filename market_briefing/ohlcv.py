"""OHLCV envelope repair shared by the box/trend engines.

Two data-quality problems surfaced on the real audit cache (52 tickers, 5 years):

1. ``auto_adjust`` rows occasionally carry a high a few 1e-11 below the close, or
   a low a few 1e-11 above it.  A strict ``high < max(open, close, low)`` test
   rejected the *whole* series for that ticker.
2. Validation ran over the entire history, so one bad row in 2023 disabled the
   box engine for every later window, even though the analysis only reads the
   last 80-120 bars.

This module fixes both without inventing prices:

* only the window a caller actually reads is validated;
* float-representation noise (``<= 1e-6`` relative) is repaired by widening the
  bar envelope, which is arithmetic and cannot fabricate a direction;
* a larger inconsistency is still rejected, because widening a 5% bad low into
  a fake 5% wick would corrupt every later support/ATR calculation.

The report is attached to the analysis output so repaired data is never confused
with pristine data.
"""
from __future__ import annotations

import math

VERSION = "2026-10-25.ohlcv-window-v2"
FLOAT_NOISE_TOL = 1e-6


def extract(dd, keys=("opens", "highs", "lows", "closes", "volumes")):
    """Return five float arrays from a service-style dict, or raise ValueError."""
    if not isinstance(dd, dict):
        raise ValueError("OHLCV dict 아님")
    aliases = ("Open", "High", "Low", "Close", "Volume")
    out = []
    try:
        for key, alias in zip(keys, aliases):
            value = dd.get(key, dd.get(alias, []))
            out.append([float(x) for x in value])
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("숫자로 변환할 수 없는 OHLCV") from exc
    if not out[3]:
        raise ValueError("종가 배열 없음")
    if len({len(a) for a in out}) != 1:
        raise ValueError("OHLCV 배열 길이 불일치")
    return tuple(out)


def repair(op, hi, lo, cl, vol, window=None, tol=FLOAT_NOISE_TOL):
    """Validate/repair the last ``window`` rows; return (arrays, report).

    Raises ``ValueError`` when a row inside the window is unusable.  ``closes``
    is never modified; only high/low are widened, and only for float noise.
    """
    op, hi, lo, cl, vol = map(list, (op, hi, lo, cl, vol))
    n = len(cl)
    if n == 0:
        raise ValueError("빈 OHLCV")
    start = 0 if not window else max(0, n - int(window))
    repaired_high = repaired_low = repaired_volume = 0
    max_gap_pct = 0.0
    for i in range(start, n):
        o, h, l, c, v = op[i], hi[i], lo[i], cl[i], vol[i]
        if not all(math.isfinite(x) for x in (o, h, l, c, v)):
            raise ValueError(f"{i}번째 봉에 유한하지 않은 값")
        if min(o, h, l, c) <= 0:
            raise ValueError(f"{i}번째 봉 가격이 0 이하")
        if v < 0:
            vol[i] = 0.0
            repaired_volume += 1
        target_high = max(o, c, l)
        if h < target_high:
            gap = (target_high - h) / max(abs(target_high), 1e-12)
            if gap > tol:
                raise ValueError(
                    f"{i}번째 봉 고가가 시·종가·저가보다 {gap:.2e} 낮음: 데이터 오류")
            max_gap_pct = max(max_gap_pct, gap)
            hi[i] = target_high
            repaired_high += 1
        target_low = min(o, c, hi[i])
        if l > target_low:
            gap = (l - target_low) / max(abs(l), 1e-12)
            if gap > tol:
                raise ValueError(
                    f"{i}번째 봉 저가가 시·종가·고가보다 {gap:.2e} 높음: 데이터 오류")
            max_gap_pct = max(max_gap_pct, gap)
            lo[i] = target_low
            repaired_low += 1
        if lo[i] > hi[i]:
            lo[i], hi[i] = hi[i], lo[i]
    repaired = repaired_high + repaired_low
    validated = n - start
    report = {
        "version": VERSION,
        "rows": n,
        "validated_rows": validated,
        "window_start": start,
        "repaired_high": repaired_high,
        "repaired_low": repaired_low,
        "repaired_volume": repaired_volume,
        "repaired_rows": repaired,
        "repaired_ratio": round(repaired / validated, 6) if validated else 0.0,
        "max_gap_pct": round(max_gap_pct, 10),
        "quality": "ok" if not repaired and not repaired_volume else "repaired",
        "policy": "읽는 창만 검증; 1e-6 이하 부동소수 외피만 보정; 종가 불변; 큰 불일치는 거부",
    }
    return (op, hi, lo, cl, vol), report


def load(dd, window=None, keys=("opens", "highs", "lows", "closes", "volumes")):
    """extract + window-scoped repair in one call."""
    arrays = extract(dd, keys)
    fixed, report = repair(*arrays, window=window)
    return fixed, report
