# -*- coding: utf-8 -*-
"""7단계 스캔 VCP 편입 테스트. 점수 미반영·승격만 검증."""
import json
import shutil
import subprocess

import pytest

from api.index import HTML
from market_briefing.scan_engine import (
    apply_vcp_promotion,
    vcp_conditions,
)
from market_briefing.technique_prune import technique_allowed


def _pass_vc(entry=130.0, stop=115.0, risky=False):
    return {"available": True, "stage": "PASS", "stage_label": "피벗 돌파",
            "false_breakout_risk": risky,
            "risk_reasons": ["약세장 돌파"] if risky else [],
            "entry_trigger": entry, "stop_price": stop}


def _cand(**over):
    base = {
        "ticker": "TEST", "price": 131.0, "status": "FAR",
        "entry_trigger": 140.0, "stop_price": 120.0, "distance_pct": 6.9,
        "sleeve": "CORE", "passes_tech_filters": True,
        "shares": 10.0, "risk_amount": 100.0, "risk_pct": 1.0, "total_cost": 500.0,
    }
    base.update(over)
    return base


def test_promotion_sets_ready_with_vcp_levels():
    cands = [_cand()]
    out = apply_vcp_promotion(cands, {"TEST": _pass_vc()}, 10_000_000.0, 1.0)
    assert out == {"promoted": 1}
    cd = cands[0]
    assert cd["status"] == "READY"
    assert cd["status_source"] == "vcp"
    assert cd["entry_trigger"] == 130.0
    assert cd["stop_price"] == 115.0
    assert cd["orig_status"] == "FAR"
    assert cd["shares"] == round(10_000_000.0 * 0.01 / 15.0, 4)


def test_risky_pass_is_not_promoted():
    cands = [_cand(ticker="R")]
    out = apply_vcp_promotion(cands, {"R": _pass_vc(risky=True)})
    assert out == {"promoted": 0}
    assert cands[0]["status"] == "FAR"
    assert "status_source" not in cands[0]


def test_promotion_skips_non_pass_earnings_and_prior_promotions():
    cands = [
        _cand(ticker="A", status="WATCH"),
        _cand(ticker="B", status="FAR"),
        _cand(ticker="C", status="EARNINGS_BLOCK"),
        _cand(ticker="D", status="READY", status_source="momentum_persistence"),
    ]
    mo = {
        "A": {"available": True, "stage": "CONTRACTING"},
        "B": {"available": False, "stage": "NONE"},
        "C": _pass_vc(),
        "D": _pass_vc(),
    }
    out = apply_vcp_promotion(cands, mo)
    assert out == {"promoted": 0}
    assert [c["status"] for c in cands] == ["WATCH", "FAR", "EARNINGS_BLOCK", "READY"]


def test_promotion_rejects_invalid_levels():
    cands = [_cand(ticker="X")]
    out = apply_vcp_promotion(cands, {"X": _pass_vc(entry=115.0, stop=115.0)})
    assert out == {"promoted": 0}
    assert cands[0]["status"] == "FAR"


def test_promotion_never_raises():
    assert apply_vcp_promotion(None, None) == {"promoted": 0}
    assert apply_vcp_promotion([None, "x", {}], {}) == {"promoted": 0}


def test_prune_gate_allows_vcp_without_rules():
    ok, _ = technique_allowed("vcp", "KRX", vcp_conditions(_pass_vc()))
    assert ok is True
    ok, _ = technique_allowed("vcp", "US", vcp_conditions(_pass_vc()))
    assert ok is True


def _scan_markup() -> str:
    return HTML.split('<div id="page-scan"', 1)[1].split(
        '<div id="page-immune"', 1
    )[0]


def test_scan_table_columns_reordered_without_chg_signal_quant():
    markup = _scan_markup()
    assert ">등락률<" not in markup
    assert ">신호<" not in markup
    assert ">퀀트 모멘텀 점수<" not in markup
    assert ">VCP 수축<" in markup
    order = ["종목", "상태", "현재가", "진입 트리거", "돌파 신뢰도",
             "치명적 약점", "순복합 점수", "카테고리", "리더 반전",
             "모멘텀 지속", "VCP 수축"]
    positions = [markup.index(f">{label}<") for label in order]
    assert positions == sorted(positions)


def test_scan_empty_message_spans_all_columns():
    # 빈 목록 메시지는 JS 렌더러에 있으므로 HTML 전체에서 확인 (12열)
    assert 'colspan="12"' in HTML
    assert "선정 종목 없음" in HTML


def test_vcp_badges_render_in_scan_table():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for the scan renderer regression check")
    payload = {
        "total_scanned": 3, "passed_filters": 3, "ready_count": 1,
        "watch_count": 0, "good_count": 3,
        "regime": "BULLISH", "vol_regime": "NORMAL_VOL",
        "candidates": [
            {"ticker": "VCP1", "name": "VCP Pass", "status": "READY",
             "status_source": "vcp", "price": 131.0, "entry_trigger": 130.0,
             "stop_price": 115.0, "bqs": 70.0, "fws": 20.0, "ncs": 68.0,
             "quality_tier": "high",
             "vcp": {"stage": "PASS", "false_breakout_risk": False,
                     "depths_pct": [25.0, 15.0, 8.0]}},
            {"ticker": "VCP2", "name": "VCP Risky", "status": "WATCH",
             "price": 120.0, "entry_trigger": 125.0, "stop_price": 110.0,
             "bqs": 50.0, "fws": 40.0, "ncs": 45.0, "quality_tier": "medium",
             "vcp": {"stage": "PASS", "false_breakout_risk": True,
                     "risk_reasons": ["약세장 돌파"]}},
            {"ticker": "VCP3", "name": "VCP Contracting", "status": "WATCH",
             "price": 118.0, "entry_trigger": 122.0, "stop_price": 108.0,
             "bqs": 55.0, "fws": 35.0, "ncs": 50.0, "quality_tier": "medium",
             "vcp": {"stage": "CONTRACTING"}},
        ],
    }
    script = """
const vm = require('vm');
const elements = {};
const sandbox = {
  document: {getElementById(id) { return elements[id] ||= {innerHTML:'', textContent:''}; }},
  fmtPrice(v) { return '$' + Number(v).toFixed(2); },
  fmtSymbol(v) { return String(v); },
  openStockDetail() {},
};
vm.createContext(sandbox);
vm.runInContext(SOURCE, sandbox);
sandbox.renderScanResult(DATA, 'US');
console.log(JSON.stringify({table: elements['scan-tbody'].innerHTML}));
""".replace("SOURCE", json.dumps(
        "function renderScanResult" + HTML.split(
            "function renderScanResult", 1)[1].split("// 🛡️ 시장 위험 면역 UI", 1)[0],
        ensure_ascii=False)).replace("DATA", json.dumps(payload, ensure_ascii=False))
    completed = subprocess.run(
        [node, "-"], input=script, text=True, encoding="utf-8",
        capture_output=True, check=True, timeout=20,
    )
    table = json.loads(completed.stdout)["table"]
    assert "VCP 돌파" in table
    assert "통과" in table and "주의" in table
    assert "수축중" in table


@pytest.fixture(autouse=True)
def raw_promotion_policy(monkeypatch):
    """These tests exercise raw promotion mechanics, independently of trained artifacts."""
    from market_briefing import technique_prune
    monkeypatch.setattr(technique_prune, "load_rules", lambda *args, **kwargs: {})
