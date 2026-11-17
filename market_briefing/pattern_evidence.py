"""패턴·수평선 실측 근거 조회(표시 전용).

``models/pattern_evidence.json`` 은 ``scripts/audit_chart_patterns.py export-evidence`` 가 348종목·10년
인과 사건 연구에서 만든다. 가이드 도표의 '신뢰도 100%·80%·65%' 같은 숫자 대신 **실제로 잰 값**을
패턴 카드·차트 안내에 붙이기 위한 얇은 조회 계층이다.

* 점수·확률·시나리오 가중치를 만들지 않는다. 표시 문구만 만든다.
* 파일이 없거나 깨져도 절대 예외를 던지지 않는다(fail-open → 근거 문구 없음).
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

_PATH_OVERRIDE: str | None = None
_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"path": None, "mtime": None, "doc": {}}

VERDICT_TEXT = {
    "supported_weak": "근거 약함(효과 작음)",
    "no_evidence": "통계적 근거 없음",
    "contradicted": "신호와 반대로 움직임",
}


def evidence_path() -> str:
    if _PATH_OVERRIDE:
        return _PATH_OVERRIDE
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "models", "pattern_evidence.json")


def load_evidence(path: str | None = None) -> dict[str, Any]:
    target = path or evidence_path()
    try:
        mtime = os.stat(target).st_mtime_ns
    except OSError:
        return {}
    with _LOCK:
        if _CACHE["path"] == target and _CACHE["mtime"] == mtime:
            return _CACHE["doc"]
        try:
            with open(target, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
            doc = doc if isinstance(doc, dict) and isinstance(doc.get("cells"), dict) else {}
        except (OSError, ValueError):
            doc = {}
        _CACHE.update(path=target, mtime=mtime, doc=doc)
        return doc


def _cell(key: str) -> dict[str, Any] | None:
    doc = load_evidence()
    cell = (doc.get("cells") or {}).get(key)
    if not isinstance(cell, dict):
        return None
    return {**cell, "horizon_sessions": doc.get("horizon_sessions"), "as_of": doc.get("as_of")}


def pattern_evidence(pattern_id: str | None, direction_code: str | None) -> dict[str, Any] | None:
    """차트 패턴(``double_top`` 등)과 신호 방향(``bullish``/``bearish``)의 실측 근거. 없으면 None."""
    if not pattern_id or direction_code not in ("bullish", "bearish"):
        return None
    return _cell(f"{pattern_id}:{direction_code}")


def touch_bucket(touches: int) -> str:
    return "1" if int(touches) <= 1 else "2" if int(touches) == 2 else "3+"


def level_evidence(kind: str, touches: int) -> dict[str, Any] | None:
    """수평선 사건(``support_bounce``·``flip_resistance`` 등)과 접촉 수 구간의 실측 근거. 없으면 None."""
    return _cell(f"{kind}|touches={touch_bucket(touches)}")


def evidence_text(ev: dict[str, Any] | None) -> str | None:
    """카드·차트 안내용 한 줄. 주가가 시장보다 실제로 얼마나 움직였는지(부호 그대로)를 보여 준다."""
    if not ev or ev.get("edge_bps") is None or not ev.get("n"):
        return None
    horizon = int(ev.get("horizon_sessions") or 10)
    # edge_bps 는 '신호 방향대로 움직였으면 +' 이므로, 하락 신호는 부호를 뒤집어야 실제 상대수익률이다.
    relative = float(ev["edge_bps"]) / 100.0 * (-1.0 if ev.get("direction") == "bearish" else 1.0)
    verdict = VERDICT_TEXT.get(str(ev.get("verdict")), "통계적 근거 없음")
    return (f"과거 10년 {int(ev['n']):,}건 · 이후 {horizon}거래일 시장 대비 평균 {relative:+.2f}%p · {verdict}")
