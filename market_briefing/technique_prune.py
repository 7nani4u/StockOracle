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
    return {**_signal_context(hs),
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
    return {**_signal_context(pattern),
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
    return {**_signal_context(info),
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
_RULES_CACHE_TOKEN: Dict[str, str] = {}


def rules_path() -> str:
    """규칙 파일 경로. 테스트에서 오버라이드 가능하다."""
    if _RULES_PATH_OVERRIDE:
        return _RULES_PATH_OVERRIDE
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "models", "technique_prune.json")


def rules_cache_token(path: str | None = None) -> str:
    """Cheap artifact identity for immediate cache invalidation on activation."""
    target = path or rules_path()
    try:
        stat = os.stat(target)
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return "missing"


def load_rules(path: str | None = None,
               ttl: float = RULES_TTL_S) -> Dict[str, Any]:
    """규칙 JSON 로드. 없거나 깨지면 빈 dict (fail-open)."""
    target = path or rules_path()
    now = time.monotonic()
    token = rules_cache_token(target)
    if (target in _RULES_CACHE and _RULES_CACHE_TOKEN.get(target) == token
            and (now - _RULES_CACHE_TS.get(target, 0.0)) < ttl):
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
    _RULES_CACHE_TOKEN[target] = token
    return dict(result)


def _rules_for(rules: Dict[str, Any], technique: str, market: str) -> Dict[str, Any]:
    try:
        node = (rules.get("rules") or {}).get(technique) or {}
        market_node = node.get(str(market).upper())
        return market_node if isinstance(market_node, dict) else {}
    except (AttributeError, TypeError):
        return {}


def technique_status(
    technique: str,
    market: str,
    rules: Dict[str, Any] | None = None,
) -> str:
    """기법 상태: KEEP / PRUNED_KEEP / DROP / INSUFFICIENT / UNKNOWN."""
    node = _rules_for(load_rules() if rules is None else rules, technique, market)
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
        node = _rules_for(load_rules() if rules is None else rules, technique, market)
        document = load_rules() if rules is None else rules
        if document.get("version", document.get("schema_version", 1)) == 2 and (node.get("validated") is not True or str(node.get("status") or "").upper() not in ("KEEP", "PRUNED_KEEP", "DROP")):
            if document.get("strict_validated_only") is True:
                return False, "검증 부족: 매매 기여 제외"
            return True, "검증 부족: 연구 신호만 허용"
        if str(node.get("status") or "").upper() == "DROP":
            return False, f"{technique} 기법 전체 제외 (검증 기대값 <= 0)"
        conds = conditions or {}
        exclusions: List[Dict[str, Any]] = node.get("exclusions") or []
        for rule in exclusions:
            if not isinstance(rule, dict):
                continue
            key, val = rule.get("condition"), rule.get("value")
            if not isinstance(key, str) or not key or key not in conds or val is None or isinstance(val, (dict, list)):
                continue
            if conds[key] == val:
                return False, f"손실조건 제외: {key}={val}"
        return True, "허용"
    except Exception as e:  # fail-open
        return True, f"판정 실패로 허용: {type(e).__name__}"


def momentum_conditions(info: Dict[str, Any] | None) -> Dict[str, str]:
    info = info or {}
    return {**_signal_context(info),"surge_bucket": bucket(info.get("surge_pct"), [25.0, 40.0],
                                   ["20-25%", "25-40%", ">=40%"]),
            "stage": str(info.get("stage") or "na")}


def vcp_conditions(info: Dict[str, Any] | None) -> Dict[str, str]:
    info = info or {}
    depths = info.get("depths_pct") or []
    first = depths[0] if isinstance(depths, (list, tuple)) and depths else None
    return {**_signal_context(info),"contractions": str(info.get("contractions") or "na"),
            "first_depth_bucket": bucket(first, [20.0, 30.0], ["<20%", "20-30%", ">=30%"]),
            "risk_flag": "risky" if info.get("false_breakout_risk") else "clean",
            "stage": str(info.get("stage") or "na")}


def technique_evidence(technique: str, market: str, rules=None) -> Dict[str, Any]:
    """Expose research status separately from permission to retain raw signals."""
    document = load_rules() if rules is None else rules
    node = _rules_for(document, technique, market)
    status = technique_status(technique, market, document)
    validated = (document.get("version", document.get("schema_version")) == 2
                 and node.get("validated") is True)
    return {"technique": technique, "market": str(market).upper(),
            "status": status, "validated": validated,
            "publishable": validated and status in ("KEEP", "PRUNED_KEEP"),
            "as_of": document.get("as_of"),
            "evidence": {**{key: node[key] for key in (
                "reason", "min_trades", "split_cutoffs", "baseline", "retained",
                "exclusions", "final_affected_support") if key in node},
                "steps_count": len(node.get("steps") or []) if isinstance(node.get("steps"), list) else 0,
                "rejections_count": len(node.get("rejections") or []) if isinstance(node.get("rejections"), list) else 0}}


def pruning_report(rules=None) -> Dict[str, Any]:
    document = load_rules() if rules is None else rules
    nodes = document.get("rules") or {}
    if not isinstance(nodes, dict):
        nodes = {}
    evidence = [technique_evidence(tech, market, document)
                for tech, markets in nodes.items()
                if isinstance(markets, dict) for market in markets]
    return {"strict_validated_only": document.get("strict_validated_only", False),
            "version": document.get("version", document.get("schema_version")),
            "as_of": document.get("as_of"), "start_date": document.get("start_date"),
            "end_date": document.get("end_date"), "data_status": document.get("data_status"),
            "registry": document.get("registry", []), "techniques": evidence,
            "validated_techniques": [e for e in evidence if e["publishable"]]}


def context_conditions(closes, highs, lows, volumes) -> Dict[str, str]:
    """Fixed causal context buckets; requires 60 aligned, finite completed bars.

    Thresholds describe conditions and are not fitted to trading outcomes.
    The current volume is compared with the preceding 20 bars, excluding itself.
    """
    try:
        arrays = [list(values) for values in (closes, highs, lows, volumes)]
        if len({len(values) for values in arrays}) != 1 or len(arrays[0]) < 60:
            return {}
        c, h, l, v = [[float(x) for x in values[-60:]] for values in arrays]
        if not all(math.isfinite(x) for values in (c, h, l, v) for x in values):
            return {}
        if any(x <= 0 for x in c + h + l) or any(x < 0 for x in v):
            return {}
        if any(hi < lo for hi, lo in zip(h, l)):
            return {}
        ma20, ma60 = sum(c[-20:]) / 20, sum(c) / 60
        trend = "UP" if c[-1] > ma20 > ma60 else "DOWN" if c[-1] < ma20 < ma60 else "MIXED"
        atr14 = sum(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
                    for i in range(46, 60)) / 14
        result = {"trend_context": trend,
                  "volatility_context": bucket(atr14/c[-1]*100, [2, 5], ["<2%", "2-5%", ">=5%"])}
        prior_volume = sum(v[-21:-1]) / 20
        if prior_volume > 0:
            result["volume_context"] = bucket(v[-1]/prior_volume, [.8, 1.5],
                                               ["<0.8x", "0.8-1.5x", ">=1.5x"])
        return result
    except (TypeError, ValueError, OverflowError):
        return {}


def _signal_context(info) -> Dict[str, str]:
    context = info.get("context_conditions") if isinstance(info, dict) else None
    if not isinstance(context, dict):
        return {}
    return {key: value for key, value in context.items()
            if key in ("trend_context", "volatility_context", "volume_context")
            and isinstance(value, str)}
