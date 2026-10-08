"""probability_calibration.py — 방향(상승) 확률을 과거 실현 빈도에 맞추는 보정 계층.

규칙 점수에서 나온 상승 확률(``prob_up``)은 10~91%까지 퍼지지만, 52종목·2022-10~2026-09·8,368 시점의
워크포워드 검증에서 22거래일 뒤 종가가 현재가보다 높았던 실제 빈도와는 거의 무관했다
(AUC 0.508, 보정 기울기 0.005). 그대로 보여 주면 "80% 상승"이 실제로는 57% 안팎인 셈이라,
표시 확률을 실현 빈도에 맞춘 값으로 바꾸고 원래 값은 신호 점수(``prob_up_raw``)로 따로 남긴다.

모형은 일부러 단순하게 둔다.

    p_cal = intercept[market] + slope * (p_raw - 0.5)        (0.05 ~ 0.95 로 제한)

- ``slope`` 는 시장을 합쳐 한 값으로 추정한다. 시장별 기울기는 학습 구간과 검증 구간에서 부호가 뒤집혀
  (KRX -0.31 → 검증 AUC +0.56) 표본 밖에서 기저 상승률보다 나빴다. 블록 부트스트랩 95% 구간이 0 을
  포함하므로(-0.06~0.12) 보정 후 확률은 사실상 시장별 기저 상승률 부근에 모인다. 이것이 정직한 결과이며,
  방향 예측력이 검증되기 전까지 표시 확률이 기저율에서 크게 벗어나지 않게 하는 안전장치다.
- ``intercept`` 는 시장별 기저 상승률에서 역산한다(KRX 약 55%, US 약 58%).
- 보정 파일(``models/probability_calibration.json``)이 없거나 깨져도 raise 하지 않고 원래 값을 그대로
  돌려준다(fail-open, ``applied=False``).

보정 파일은 ``scripts/fit_probability_calibration.py`` 가 ``scripts/audit_prediction_layers.py`` 의 행 단위
결과(``--rows-csv``)로 다시 만든다.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Sequence

CALIBRATION_FILENAME = "probability_calibration.json"
CLIP_PCT = (5.0, 95.0)
_CACHE_TTL_S = 60.0
_CACHE: Dict[str, Any] = {}
_CACHE_TS: Dict[str, float] = {}
_PATH_OVERRIDE: str | None = None


def calibration_path() -> str:
    """보정 파일 경로. 테스트에서 ``_PATH_OVERRIDE`` 로 바꿀 수 있다."""
    if _PATH_OVERRIDE:
        return _PATH_OVERRIDE
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "models", CALIBRATION_FILENAME)


def load_calibration(path: str | None = None, ttl: float = _CACHE_TTL_S) -> Dict[str, Any]:
    """보정 JSON 로드(짧은 캐시). 없거나 깨지면 빈 dict."""
    target = path or calibration_path()
    now = time.monotonic()
    if target in _CACHE and (now - _CACHE_TS.get(target, 0.0)) < ttl:
        cached = _CACHE[target]
        return dict(cached) if isinstance(cached, dict) else {}
    try:
        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        result = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        result = {}
    _CACHE[target] = result
    _CACHE_TS[target] = now
    return dict(result)


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _market_key(market: Any) -> str:
    return "KRX" if str(market or "").upper() in {"KRX", "KR", "KOSPI", "KOSDAQ"} else "US"


def calibrate_direction_probability(prob_up_pct: Any, market: Any,
                                    calibration: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """상승 확률(%)을 실현 빈도에 맞춘 값으로 바꾼다. 절대 raise 하지 않는다.

    Returns:
        ``prob_up``/``prob_down``(보정 후 %), ``raw_prob_up``/``raw_prob_down``(보정 전 %), ``applied``,
        ``method``, ``base_rate_pct``, ``slope``, ``horizon_sessions``, ``note``.
    """
    raw = _finite(prob_up_pct)
    if raw is None:
        return {"prob_up": None, "prob_down": None, "raw_prob_up": None, "raw_prob_down": None,
                "applied": False, "method": "unavailable", "note": "상승 확률을 계산할 수 없습니다."}
    raw = min(100.0, max(0.0, raw))
    result: Dict[str, Any] = {
        "prob_up": round(raw, 1), "prob_down": round(100.0 - raw, 1),
        "raw_prob_up": round(raw, 1), "raw_prob_down": round(100.0 - raw, 1),
        "applied": False, "method": "identity",
        "note": "보정 파일이 없어 규칙 점수 기반 확률을 그대로 표시합니다.",
    }
    try:
        payload = calibration if calibration is not None else load_calibration()
        slope = _finite(payload.get("slope"))
        intercepts = payload.get("intercept_at_half") or {}
        intercept = _finite(intercepts.get(_market_key(market)), _finite(intercepts.get("ALL")))
        if slope is None or intercept is None:
            return result
        low, high = (float(v) for v in (payload.get("clip_pct") or CLIP_PCT))
        calibrated = min(high, max(low, (intercept + slope * (raw / 100.0 - 0.5)) * 100.0))
        horizon = int(payload.get("horizon_sessions") or 22)
        fitted = payload.get("fitted_on") or {}
        base_rate = _finite((fitted.get("base_up_rate") or {}).get(_market_key(market)))
        result.update({
            "prob_up": round(calibrated, 1), "prob_down": round(100.0 - calibrated, 1),
            "applied": True, "method": str(payload.get("method") or "linear_shrink"),
            "slope": round(slope, 4), "horizon_sessions": horizon,
            "base_rate_pct": round(base_rate * 100.0, 1) if base_rate is not None else None,
            "fitted_period": fitted.get("period"), "fitted_rows": fitted.get("rows"),
            "note": (f"규칙 점수의 상승 확률을 과거 {horizon}거래일 뒤 실제 상승 빈도에 맞춰 보정했습니다. "
                     "점수와 실제 방향의 관계가 약해(보정 기울기 ≈ 0) 시장별 기저 상승률 부근에 모입니다."),
        })
    except Exception as exc:  # 보정 실패가 분석 전체를 막지 않게 한다
        result["note"] = f"보정 실패로 원래 값을 표시합니다({type(exc).__name__})."
    return result


# ── 보정 파일 생성(순수 함수: 단위 테스트·스크립트 공용) ──────────────────────

def _ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    var = sum((a - mean_x) ** 2 for a in x)
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / var if var > 1e-12 else 0.0


def fit_direction_calibration(prob_up_pct: Iterable[float], went_up: Iterable[int], markets: Iterable[str]) -> Dict[str, Any]:
    """시장 합산 기울기 + 시장별 절편(=기저 상승률에서 역산)을 구한다.

    Args:
        prob_up_pct: 표시하던 상승 확률(0~100).
        went_up: 선행 ``horizon`` 거래일 뒤 종가가 현재가보다 높았으면 1.
        markets: 각 행의 시장(KRX/US).
    """
    p = [min(1.0, max(0.0, float(v) / 100.0)) for v in prob_up_pct]
    y = [int(v) for v in went_up]
    m = [_market_key(v) for v in markets]
    if not p or len(p) != len(y) or len(p) != len(m):
        raise ValueError("입력 길이가 같고 비어 있지 않아야 합니다.")
    slope = _ols_slope(p, y)
    intercepts: Dict[str, float] = {}
    base_rates: Dict[str, float] = {}
    groups: Dict[str, List[int]] = {"ALL": list(range(len(p)))}
    for index, key in enumerate(m):
        groups.setdefault(key, []).append(index)
    for key, idx in groups.items():
        base = sum(y[i] for i in idx) / len(idx)
        mean_p = sum(p[i] for i in idx) / len(idx)
        base_rates[key] = round(base, 4)
        intercepts[key] = round(base - slope * (mean_p - 0.5), 4)
    return {"slope": round(slope, 4), "intercept_at_half": intercepts, "base_up_rate": base_rates, "rows": len(p)}
