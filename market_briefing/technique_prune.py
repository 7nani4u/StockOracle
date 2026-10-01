"""technique_prune.py — 기법별 손실조건 제거 규칙 적용 모듈.

``scripts/prune_loss_conditions.py`` 가 최근 1년치 국내·미국 종목에서
기법별 매매를 반복 재현해 찾아낸 손실조건 제외 규칙
(``models/technique_prune.json``)을 스캔·분석 경로에서 읽기 위한
가벼운 조회 계층이다. 규칙 파일이 없거나 깨져도 절대 raise하지 않고
모든 기법을 허용한다 (fail-open).
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

TECHNIQUES = (
    "hybrid_breakout",
    "pattern_breakout",
    "dynamic_rsi",
    "leader_reversal",
    "ml_direction",
)


def bucket(value: Any, edges: List[float], labels: List[str],
           unknown: str = "na") -> str:
    """연속값을 구간 라벨로 변환. 절대 raise하지 않는다."""
    try:
        v = float(value)
        if not math.isfinite(v):
            return unknown
        for edge, label in zip(edges, labels):
            if v < edge:
                return label
        return labels[-1]
    except (TypeError, ValueError):
        return unknown


def hybrid_conditions(hs: Dict[str, Any]) -> Dict[str, str]:
    """prune 스크립트와 실전 게이트가 공유하는 hybrid 조건 키."""
    return {
        "ncs_bucket": bucket(hs.get("ncs"), [50, 60, 70],
                             ["<50", "50-60", "60-70", ">=70"]),
        "fws_bucket": bucket(hs.get("fws"), [20, 30, 50],
                             ["<=20", "20-30", "30-50", ">50"]),
        "regime": str(hs.get("regime") or "na"),
        "vol_regime": str(hs.get("vol_regime") or "na"),
        "adx_bucket": bucket(hs.get("adx"), [20, 30],
                             ["<20", "20-30", ">=30"]),
        "dist_bucket": bucket(hs.get("dist_to_high"), [1, 2],
                              ["<1%", "1-2%", ">=2%"]),
    }


def pattern_conditions(pattern: Dict[str, Any]) -> Dict[str, str]:
    return {
        "family": str(pattern.get("family") or "na"),
        "completion_bucket": bucket(pattern.get("completion_score"), [70, 85],
                                    ["<70", "70-85", ">=85"]),
        "tolerance_mode": str(pattern.get("tolerance_mode") or "hybrid"),
    }


def drsi_conditions(market: str, rsi_value: Any,
                    stop_dist_pct: Any) -> Dict[str, str]:
    return {
        "market": str(market or "na"),
        "rsi_bucket": bucket(rsi_value, [30, 45], ["<30", "30-45", ">=45"]),
        "stop_dist_bucket": bucket(stop_dist_pct, [1.5, 3.0],
                                   ["<1.5%", "1.5-3%", ">=3%"]),
    }


def leader_conditions(info: Dict[str, Any]) -> Dict[str, str]:
    try:
        drawdown = abs(float(info.get("drawdown_pct") or 0))
    except (TypeError, ValueError):
        drawdown = None
    return {
        "rs_edge_bucket": bucket(info.get("rs_edge_pp"), [15, 25],
                                 ["<15pp", "15-25pp", ">=25pp"]),
        "drawdown_bucket": bucket(drawdown, [40, 55],
                                  ["30-40%", "40-55%", ">=55%"]),
        "bench_flag": "bench" if info.get("market_confirm") else "no_bench",
    }


def ml_conditions(pred: Dict[str, Any]) -> Dict[str, str]:
    return {
        "conf_bucket": bucket(pred.get("confidence"), [0.65, 0.75],
                              ["0.60-0.65", "0.65-0.75", ">=0.75"]),
        "prob_up_bucket": bucket(pred.get("prob_up"), [0.60, 0.70],
                                 ["<0.60", "0.60-0.70", ">=0.70"]),
        "model_flag": "model" if pred.get("model_available") else "fallback",
    }

_RULES_PATH_OVERRIDE: str | None = None

RULES_TTL_S = 60.0  # 스캔당 수백 회 호출되므로 단기 캐시 (실전 게이트용)
_RULES_CACHE: Dict[str, Any] = {}
_RULES_CACHE_TS: Dict[str, float] = {}


def rules_path() -> str:
    """규칙 파일 경로. 테스트에서 오버라이드 가능하다."""
    if _RULES_PATH_OVERRIDE:
        return _RULES_PATH_OVERRIDE
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "models", "technique_prune.json")


def load_rules(path: str | None = None,
               ttl: float = RULES_TTL_S) -> Dict[str, Any]:
    """규칙 JSON 로드. 없거나 깨지면 빈 dict (fail-open)."""
    target = path or rules_path()
    now = time.monotonic()
    if target in _RULES_CACHE and (now - _RULES_CACHE_TS.get(target, 0.0)) < ttl:
        cached = _RULES_CACHE[target]
        return dict(cached) if isinstance(cached, dict) else {}
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        result = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        result = {}
    _RULES_CACHE[target] = result
    _RULES_CACHE_TS[target] = now
    return dict(result)


def _rules_for(rules: Dict[str, Any], technique: str, market: str) -> Dict[str, Any]:
    try:
        node = (rules.get("rules") or {}).get(technique) or {}
        return node.get(str(market).upper()) or {}
    except (AttributeError, TypeError):
        return {}


def technique_status(
    technique: str,
    market: str,
    rules: Dict[str, Any] | None = None,
) -> str:
    """기법 상태: KEEP / PRUNED_KEEP / DROP / INSUFFICIENT / UNKNOWN."""
    node = _rules_for(rules or load_rules(), technique, market)
    status = node.get("status") or "UNKNOWN"
    return str(status).upper()


def technique_allowed(
    technique: str,
    market: str,
    conditions: Dict[str, Any] | None = None,
    rules: Dict[str, Any] | None = None,
) -> Tuple[bool, str]:
    """손실조건 제외 규칙 판정. 절대 raise하지 않는다.

    Returns:
        (허용 여부, 사유). DROP 상태면 일괄 차단, exclusions의
        (condition == value) 일치 1개라도 있으면 차단한다.
    """
    try:
        node = _rules_for(rules or load_rules(), technique, market)
        if str(node.get("status") or "").upper() == "DROP":
            return False, f"{technique} 기법 전체 제외 (검증 기대값 <= 0)"
        conds = conditions or {}
        exclusions: List[Dict[str, Any]] = node.get("exclusions") or []
        for rule in exclusions:
            if not isinstance(rule, dict):
                continue
            key, val = rule.get("condition"), rule.get("value")
            if key is None or conds.get(key) == val:
                return False, f"손실조건 제외: {key}={val}"
        return True, "허용"
    except Exception as e:  # fail-open
        return True, f"판정 실패로 허용: {type(e).__name__}"
