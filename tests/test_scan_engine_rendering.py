"""7단계 스캔 엔진의 화면 전용 출력 계약 테스트."""

import json
import shutil
import subprocess

import pytest

from api.index import HTML


def _scan_markup() -> str:
    return HTML.split('<div id="page-scan"', 1)[1].split(
        '<div id="page-immune"', 1
    )[0]


def _renderer_source() -> str:
    return "function renderScanResult" + HTML.split(
        "function renderScanResult", 1
    )[1].split("// 🛡️ 시장 위험 면역 UI", 1)[0]


def test_scan_table_hides_stop_price_and_uses_plain_breakout_label():
    markup = _scan_markup()

    assert ">손절가<" not in markup
    assert ">브레이크아웃 품질<" not in markup
    assert ">돌파 신뢰도<" in markup


def test_leader_reversal_keeps_signal_but_hides_its_price_plan_and_metrics():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for the scan renderer regression check")
    payload = {
        "total_scanned": 1,
        "passed_filters": 1,
        "ready_count": 1,
        "watch_count": 0,
        "good_count": 1,
        "regime": "BULLISH",
        "vol_regime": "NORMAL_VOL",
        "candidates": [{
            "ticker": "LEAD",
            "name": "Leader Test",
            "status": "READY",
            "price": 10.0,
            "change_pct": 3.0,
            "category": "Technology",
            "analyst_signal": "매수",
            "entry_trigger": 11.17,
            "stop_price": 8.10,
            "bqs": 72.0,
            "fws": 20.0,
            "ncs": 70.0,
            "quality_tier": "high",
            "leader_reversal": {
                "stage": "BREAKOUT",
                "stage_label": "리더 반전 돌파",
                "summary": "테스트용 리더 반전",
                "rs_edge_pp": 232.2,
                "drawdown_pct": 35.1,
                "range_high": 10.22,
                "entry_trigger": 11.17,
                "stop_price": 8.10,
            },
        }],
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
console.log(JSON.stringify({
  table: elements['scan-tbody'].innerHTML,
  regime: elements['scan-regime-bar'].innerHTML,
}));
""".replace("SOURCE", json.dumps(_renderer_source(), ensure_ascii=False)).replace(
        "DATA", json.dumps(payload, ensure_ascii=False)
    )
    completed = subprocess.run(
        [node, "-"], input=script, text=True, encoding="utf-8",
        capture_output=True, check=True, timeout=20,
    )
    rendered = json.loads(completed.stdout)

    assert "🔥 돌파" in rendered["table"]
    assert "초과 +232.2%p" not in rendered["table"]
    assert "조정 35.1%" not in rendered["table"]
    assert "진입 $11.17" not in rendered["table"]
    assert "손절 $8.10" not in rendered["table"]
    assert "$8.10" not in rendered["table"]
    assert "진입/손절가" not in rendered["regime"]
