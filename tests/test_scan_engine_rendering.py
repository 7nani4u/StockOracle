"""7단계 스캔 엔진의 화면 전용 출력 계약 테스트."""

import json
import re
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
    assert ">돌파 점수<" in markup          # 2026-11-12: BQS 는 확률이 아닌 점수라 '신뢰도' 표기를 '점수'로 바꿨다
    assert ">돌파 신뢰도<" not in markup


BOX_PLAN = {
    "complete": True,
    "verdict": {"key": "watch", "headline": "관망 — 박스 하단 이탈 뒤 재진입 후보"},
    "target1": {"price": 10.5, "pct_from_quote": 5.0, "label": "1차 목표 · 박스 중간", "rr": 1.2,
                "basis": "박스 상·하단의 중간값"},
    "target2": {"price": 11.4, "pct_from_quote": 14.0, "label": "2차 목표 · 박스 상단", "rr": 2.1},
    "target3": {"price": 12.1, "low": 11.9, "high": 12.1, "level": "TP2", "scenario": "중립적",
                "return_pct": 21.0, "not_above_target2": False},
    "stop": {"price": 9.2, "pct_from_quote": -8.0, "basis": "박스 하단 − 0.35 ATR"},
    "reentry": {"low": 9.4, "high": 9.7, "pct_from_quote": -3.0, "caption": "새 바닥 박스 하단",
                "conditional": True, "condition": "새 저점 없이 3거래일 유지 → 박스 확정"},
    "wait": {"text": "최소 3거래일", "sub": "가격 도달 1~8거래일째 · 20거래일 내 도달 가능성 약 78%",
             "detail": "새 박스는 최소 3거래일 뒤에 확정될 수 있습니다.",
             "period": {"text": "1~8일", "kind": "window", "min_days": 1, "max_days": 8},
             "probability": {"text": "73.5~82.2%", "low": 73.5, "mid": 77.9, "high": 82.2, "horizon_days": 20,
                             "level": 9.7, "level_label": "재진입 후보 구간 상단"}},
}

REMOVED_LEGEND = ("리더 반전 읽는 법", "4조건 충족", "돌파 직전", "조정 진행 중",
                  "점수 미반영 보조 지표이며 매수 권유가 아닙니다")


def test_html_no_longer_contains_the_leader_reversal_legend():
    for text in REMOVED_LEGEND:
        assert text not in HTML, text


def test_box_plan_columns_replace_the_removed_signal_columns():
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
        "price_cap": {"market": "US", "max_price": 23.0, "filtered_out": 40},
        "plan_stats": {"evaluated": 30, "complete": 12, "displayed": 1,
                       "excluded": [{"key": "measured_targets", "label": "돌파 이후 구간", "count": 9}]},
        "leader_reversal_counts": {"BREAKOUT": 3, "WAIT_BREAKOUT": 2},
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
            "box_plan": BOX_PLAN,
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

    table, regime = rendered["table"], rendered["regime"]
    # 삭제된 열·안내문은 데이터가 와도 그리지 않는다
    for text in ("🔥", "👀", "🧱", "리더 반전", "초과 +232.2%p", "조정 35.1%", "$8.10"):
        assert text not in table, text
    for text in REMOVED_LEGEND + ("리더 반전", "🔥", "👀"):
        assert text not in regime, text
    assert "진입/손절가" not in regime
    # 박스 실행 플랜 항목 (스캔 자체의 손절가 $8.10 이 아니라 플랜의 손절 기준 $9.20)
    for text in ("$10.50", "$11.40", "$9.20", "$9.40 ~ $9.70", "새 바닥 박스 하단 · 조건부",
                 "1~8일", "73.5~82.2%", "후보 진입 기준"):
        assert text in table, text
    # 가격 칸의 현재가 대비 % 보조 문구는 더 이상 그리지 않는다
    for text in ("현재가 +5.0%", "현재가 -8.0%", "현재가 -3.0%", "현재가 +14.0%"):
        assert text not in table, text
    assert "최소 3거래일" not in table.replace(BOX_PLAN["wait"]["detail"], "")   # 본문에는 없고 툴팁에만 있다
    assert "박스 플랜 판단: 관망" in table                         # 판단은 툴팁으로만
    assert "$23 이하만 표시" in regime and "40종목 제외" in regime
    assert "박스 플랜 완성" in regime and "12/30" in regime
    assert "돌파 이후 구간 9종목" in regime                        # 빠진 사유는 툴팁


def test_empty_result_explains_the_box_plan_requirement():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for the scan renderer regression check")
    payload = {"total_scanned": 5, "good_count": 0, "candidates": [],
               "plan_stats": {"evaluated": 5, "complete": 0, "excluded": [
                   {"key": "no_data", "label": "플랜 데이터 부족", "count": 5}]}}
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
sandbox.renderScanResult(DATA, 'KRX');
console.log(JSON.stringify({table: elements['scan-tbody'].innerHTML}));
""".replace("SOURCE", json.dumps(_renderer_source(), ensure_ascii=False)).replace(
        "DATA", json.dumps(payload, ensure_ascii=False))
    completed = subprocess.run(
        [node, "-"], input=script, text=True, encoding="utf-8",
        capture_output=True, check=True, timeout=20,
    )
    table = json.loads(completed.stdout)["table"]
    assert 'colspan="12"' in table and "박스 실행 플랜" in table and "플랜 데이터 부족 5종목" in table
    assert "예상 기간·도달 확률" in table and "예상 대기" not in table


# ── 2026-11-07 표 순서·예상 기간·도달 확률 ────────────────────────────────────────

HEADERS = ["#", "종목", "카테고리", "현재가", "돌파 점수", "1차 목표", "2차 목표", "3차 목표",
           "손절 기준", "재진입 후보", "예상 기간", "도달 확률"]


def _render(payload, market="US"):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for the scan renderer regression check")
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
sandbox.renderScanResult(DATA, MARKET);
console.log(JSON.stringify({table: elements['scan-tbody'].innerHTML}));
""".replace("SOURCE", json.dumps(_renderer_source(), ensure_ascii=False)).replace(
        "DATA", json.dumps(payload, ensure_ascii=False)).replace("MARKET", json.dumps(market))
    completed = subprocess.run(
        [node, "-"], input=script, text=True, encoding="utf-8",
        capture_output=True, check=True, timeout=20,
    )
    return json.loads(completed.stdout)["table"]


def _cells(table_html):
    """Row → list of visible cell texts (the renderer never nests <td>)."""
    import re
    rows = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S)
        rows.append([re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c)).strip() for c in cells])
    return rows


def _candidate(ticker, name, price, trigger, bqs, category, plan):
    return {"ticker": ticker, "name": name, "status": "WATCH", "price": price, "entry_trigger": trigger,
            "bqs": bqs, "fws": 30.0, "ncs": 50.0, "category": category, "cap_tier": "MID", "box_plan": plan}


def test_table_headers_follow_the_investment_flow_order():
    headers = [h.strip() for h in re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", _scan_markup(), flags=re.S)]
    assert headers == HEADERS
    assert "예상 대기 시간" not in _scan_markup()


def test_each_cell_is_mapped_to_its_own_data_in_header_order():
    second = {**BOX_PLAN,
              "target1": {"price": 21.1, "label": "1차 목표 · 이전 박스 중간"},
              "target2": {"price": 22.2, "label": "2차 목표 · 이전 박스 상단"},
              "target3": {"price": 23.0, "low": 22.6, "high": 23.0, "return_pct": 12.2, "not_above_target2": False},
              "stop": {"price": 19.3, "basis": "x"},
              "reentry": {"low": 19.8, "high": 20.4, "caption": "이탈 실패 확인 후 하단", "conditional": True},
              "wait": {"text": "3~12일", "detail": "d",
                       "period": {"text": "3~12일", "kind": "window"},
                       "probability": {"text": "65.0~78.4%", "level": 20.4, "horizon_days": 20}}}
    payload = {"total_scanned": 2, "good_count": 2, "candidates": [
        _candidate("AAA", "Alpha", 10.0, 11.17, 72.0, "Technology", BOX_PLAN),
        _candidate("BBB", "Beta", 20.5, 21.0, 41.5, "Healthcare", second)]}
    rows = _cells(_render(payload))
    assert len(rows) == 2 and all(len(r) == len(HEADERS) for r in rows)
    first = dict(zip(HEADERS, rows[0]))
    assert first["#"] == "1" and "Alpha" in first["종목"] and first["카테고리"] == "기술"
    assert first["현재가"] == "$10.00" and first["돌파 점수"].startswith("72.0")
    assert first["1차 목표"] == "$10.50" and first["2차 목표"] == "$11.40" and first["3차 목표"] == "$12.10"
    assert first["손절 기준"].startswith("$9.20") and first["재진입 후보"].startswith("$9.40 ~ $9.70")
    assert first["예상 기간"] == "1~8일" and first["도달 확률"] == "73.5~82.2%"
    other = dict(zip(HEADERS, rows[1]))
    assert other["카테고리"] == "헬스케어" and other["현재가"] == "$20.50" and other["돌파 점수"].startswith("41.5")
    assert other["1차 목표"].startswith("$21.10") and other["2차 목표"].startswith("$22.20")
    assert other["3차 목표"] == "$23.00"       # 1<2<3차 순서로 오르는 종목만 서버가 내보내므로 값만 그린다
    assert other["손절 기준"].startswith("$19.30") and other["재진입 후보"].startswith("$19.80 ~ $20.40")
    assert other["예상 기간"] == "3~12일" and other["도달 확률"] == "65.0~78.4%"


def test_price_columns_show_no_percent_text_and_no_red_helper_lines():
    payload = {"total_scanned": 1, "good_count": 1,
               "candidates": [_candidate("AAA", "Alpha", 10.0, 11.17, 72.0, "Technology", BOX_PLAN)]}
    table = _render(payload)
    row = _cells(table)[0]
    for column in (5, 6, 7, 8, 9):                   # 1차·2차·3차 목표·손절 기준·재진입 후보
        assert "%" not in row[column] and "현재가" not in row[column], row[column]
    # 손절 가격 자체의 빨강은 그대로 — 지워지는 것은 현재가 대비 보조 줄뿐이다
    assert 'color:#f85149">$9.20' in table
    assert "color:#3fb950\">$10.50" in table and "color:#58a6ff\">$9.40 ~ $9.70" in table


def test_missing_period_or_probability_is_stated_not_invented():
    only_bars = {**BOX_PLAN, "wait": {"text": "최소 1거래일", "detail": "d",
                                      "period": {"text": "최소 1일", "kind": "min_bars"}, "probability": None}}
    nothing = {**BOX_PLAN, "wait": {"text": "x", "detail": "d", "period": None, "probability": None}}
    no_wait = {**BOX_PLAN, "wait": None}
    payload = {"total_scanned": 3, "good_count": 3, "candidates": [
        _candidate("AAA", "Alpha", 10.0, 11.0, 70.0, "Technology", only_bars),
        _candidate("BBB", "Beta", 10.0, 11.0, 70.0, "Technology", nothing),
        _candidate("CCC", "Gamma", 10.0, 11.0, 70.0, "Technology", no_wait)]}
    rows = [dict(zip(HEADERS, r)) for r in _cells(_render(payload))]
    assert rows[0]["예상 기간"] == "최소 1일" and rows[0]["도달 확률"] == "산출 불가"
    for row in rows[1:]:
        assert row["예상 기간"] == "산출 불가" and row["도달 확률"] == "산출 불가"
    assert all(len(r) == len(HEADERS) for r in rows)


def test_third_target_column_is_next_to_the_second_and_has_plain_header_names():
    markup = _scan_markup()
    headers = [h.strip() for h in re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", markup, flags=re.S)]
    assert headers[5:8] == ["1차 목표", "2차 목표", "3차 목표"]
    assert ">1차 목표(박스 중간)<" not in markup and ">2차 목표(박스 상단)<" not in markup   # 머리글 글자에서 괄호 설명 삭제(툴팁에는 남김)
    assert "TP2" in markup and "중립적" in markup                                          # 3차 목표 머리글 툴팁이 출처를 말한다


def test_third_target_missing_is_stated_not_invented():
    # 서버는 3차 목표가 없거나 2차 이하인 종목을 목록에서 빼므로 아래는 구버전 응답 방어용 표시다
    none_plan = {**BOX_PLAN, "target3": None}
    absent = {k: v for k, v in BOX_PLAN.items() if k != "target3"}         # 키 자체가 없는 경우
    payload = {"total_scanned": 2, "good_count": 2, "candidates": [
        _candidate("AAA", "Alpha", 10.0, 11.0, 70.0, "Technology", none_plan),
        _candidate("BBB", "Beta", 10.0, 11.0, 70.0, "Technology", absent)]}
    table = _render(payload)
    rows = [dict(zip(HEADERS, r)) for r in _cells(table)]
    assert rows[0]["3차 목표"] == "산출 불가" and rows[1]["3차 목표"] == "산출 불가"
    assert "2차 목표 이하" not in table
    assert all(len(r) == len(HEADERS) for r in rows)
    assert "수익률 +21%" in _render({"total_scanned": 1, "good_count": 1, "candidates": [
        _candidate("AAA", "Alpha", 10.0, 11.0, 70.0, "Technology", BOX_PLAN)]})   # 툴팁에 TP2 범위·수익률
