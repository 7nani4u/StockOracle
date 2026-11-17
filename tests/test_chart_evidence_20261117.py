"""2026-11-17 차트 패턴·수평선 실측 근거 — 근거 파일 정합성, 표시 문구, 화면·백엔드 정합성 회귀 테스트."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import api.index as ix
from market_briefing import pattern_evidence as pe
from market_briefing import sr_levels
from market_briefing.pattern_engine import PatternEngine, compatibility_patterns

SOURCE = Path(__file__).parents[1].joinpath("api", "index.py").read_text(encoding="utf-8")
EVIDENCE_PATH = Path(__file__).parents[1] / "models" / "pattern_evidence.json"


@pytest.fixture(autouse=True)
def _restore_evidence_path():
    saved = pe._PATH_OVERRIDE
    yield
    pe._PATH_OVERRIDE = saved
    pe._CACHE.update(path=None, mtime=None, doc={})


# ── 근거 조회 계층 ─────────────────────────────────────────────────────────────

def test_loader_is_fail_open_for_missing_and_corrupt_files(tmp_path):
    pe._PATH_OVERRIDE = str(tmp_path / "missing.json")
    assert pe.load_evidence() == {}
    assert pe.pattern_evidence("double_top", "bearish") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    pe._PATH_OVERRIDE = str(bad)
    assert pe.load_evidence() == {}
    assert pe.evidence_text(pe.level_evidence("support_bounce", 3)) is None


def test_lookup_rejects_neutral_and_unknown_patterns():
    assert pe.pattern_evidence("double_top", "neutral") is None
    assert pe.pattern_evidence("not_a_pattern", "bullish") is None
    assert pe.pattern_evidence(None, "bullish") is None
    assert pe.touch_bucket(1) == "1" and pe.touch_bucket(2) == "2" and pe.touch_bucket(7) == "3+"


def test_text_shows_real_relative_return_sign_for_each_direction():
    bearish = {"edge_bps": 38.2, "n": 14606, "direction": "bearish", "verdict": "supported_weak", "horizon_sessions": 10}
    bullish = {"edge_bps": -27.2, "n": 7952, "direction": "bullish", "verdict": "no_evidence", "horizon_sessions": 10}
    # 하락 신호가 맞았다(+38bp) = 주가가 시장보다 0.38%p 덜 올랐다 → 화면에는 -0.38%p
    assert "-0.38%p" in pe.evidence_text(bearish) and "근거 약함" in pe.evidence_text(bearish)
    assert "-0.27%p" in pe.evidence_text(bullish) and "통계적 근거 없음" in pe.evidence_text(bullish)
    assert pe.evidence_text({"edge_bps": None, "n": 5}) is None
    assert pe.evidence_text(None) is None


def test_shipped_evidence_file_follows_its_own_stated_rules():
    doc = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    cells = doc["cells"]
    assert len(cells) >= 20 and doc["horizon_sessions"] == 10
    ordered = sorted(cells.items(), key=lambda kv: math.erfc(abs(kv[1]["t"]) / math.sqrt(2)))
    prev, holm = 0.0, {}
    for i, (key, cell) in enumerate(ordered):
        prev = min(1.0, max(prev, (len(ordered) - i) * math.erfc(abs(cell["t"]) / math.sqrt(2))))
        holm[key] = prev
    for key, cell in cells.items():
        assert cell["holm_p"] == pytest.approx(holm[key], abs=1e-3), key
        rule1 = cell["dev_bps"] > 0 and cell["val_bps"] > 0 and cell["t"] >= 2.5
        assert cell["rule1_pre_registered"] == rule1, key
        if cell["verdict"] == "supported_weak":
            assert rule1 and cell["holm_p"] < 0.05 and (cell["momentum_matched_t"] or 0) >= 2.0, key
        if cell["verdict"] == "contradicted":
            assert cell["t"] <= -2.0 and cell["dev_bps"] < 0 and cell["val_bps"] < 0 and cell["holm_p"] < 0.05, key
        assert cell["direction"] in ("bullish", "bearish")


def test_no_guide_pattern_is_presented_as_reliable():
    """가이드 도표의 100%·80%·65% 신뢰도 주장은 실측에서 재현되지 않았다 — 근거 판정이 그것을 가리지 않는지 고정한다."""
    cells = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))["cells"]
    for key in ("double_top:bearish", "bearish_consolidation:bearish", "bullish_consolidation:bullish",
                "inverse_head_shoulders:bullish", "ascending_triangle:bullish"):
        assert cells[key]["verdict"] == "no_evidence", key
        assert cells[key]["n"] >= 1000, key
        assert 40.0 <= cells[key]["direction_hit_pct"] <= 60.0, key


# ── 수평선 차트 페이로드 ───────────────────────────────────────────────────────

def _wave_with_pullback(cycles=5):
    path = [110.0]
    for _ in range(cycles):
        path += list(np.linspace(110, 100, 11)[1:]) + list(np.linspace(100, 110, 11)[1:])
    path += list(np.linspace(110, 103, 8))
    closes = np.asarray(path, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return opens, np.maximum(opens, closes) + 0.5, np.minimum(opens, closes) - 0.5, closes


def test_chart_levels_shows_at_most_one_line_per_side_with_labels():
    o, h, l, c = _wave_with_pullback()
    payload = sr_levels.chart_levels(o, h, l, c)
    assert payload["timeframe"] == "1D"
    roles = [lv["role"] for lv in payload["levels"]]
    assert roles.count("support") <= 1 and roles.count("resistance") <= 1
    for lv in payload["levels"]:
        assert lv["touches"] >= 2 and lv["label"].startswith("지지" if lv["role"] == "support" else "저항")
    assert payload["note"] and "매수 근거로 삼기 어렵습니다" in payload["note"]


def test_flip_warning_appears_only_when_the_evidence_verdict_is_supported(monkeypatch):
    o, h, l, c = _wave_with_pullback()
    fake_event = [{"type": "flip_resistance", "side": -1, "level": 104.0, "touches": 4, "atr": 1.0,
                   "flip_age": 5, "stop_ref": 104.5}]
    monkeypatch.setattr(sr_levels, "level_events", lambda *a, **k: fake_event)
    cell = {"edge_bps": 38.2, "n": 14606, "direction": "bearish", "horizon_sessions": 10}
    monkeypatch.setattr(pe, "level_evidence", lambda kind, touches: {**cell, "verdict": "supported_weak"})
    warned = sr_levels.chart_levels(o, h, l, c)["warning"]
    assert warned and warned["type"] == "flip_resistance" and "-0.38%p" in warned["evidence_text"]
    monkeypatch.setattr(pe, "level_evidence", lambda kind, touches: {**cell, "verdict": "no_evidence"})
    assert sr_levels.chart_levels(o, h, l, c)["warning"] is None
    monkeypatch.setattr(pe, "level_evidence", lambda kind, touches: None)
    assert sr_levels.chart_levels(o, h, l, c)["warning"] is None


def test_chart_levels_is_empty_for_short_or_non_finite_input():
    o, h, l, c = _wave_with_pullback()
    assert sr_levels.chart_levels(o[:10], h[:10], l[:10], c[:10])["levels"] == []
    c2 = c.copy()
    c2[-3] = np.nan
    assert sr_levels.chart_levels(o, h, l, c2)["levels"] == []


def test_a_single_old_gap_does_not_disable_levels_for_long_histories():
    o, h, l, c = _wave_with_pullback(cycles=14)          # 289봉 > lookback(250)
    assert len(c) > sr_levels.LevelConfig().lookback
    pad = np.full(100, np.nan)
    long_o, long_h, long_l, long_c = (np.concatenate([pad, a]) for a in (o, h, l, c))
    assert sr_levels.chart_levels(long_o, long_h, long_l, long_c)["levels"]   # lookback 밖의 결측은 무시


# ── analyze_score: 확정 패턴에만 근거 문구 ──────────────────────────────────────

def _random_walk_dd(n=260, seed=5, market="US"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    frame = pd.DataFrame({"Open": close * 0.999, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                          "Volume": rng.integers(1_000_000, 2_000_000, n).astype(float)})
    enriched = ix.add_indicators(frame.copy(), market=market)
    dd = {"Date": [f"2026-01-{(i % 28) + 1:02d}" for i in range(len(enriched))]}
    for column in enriched.columns:
        dd[column] = [float(v) if np.isfinite(v) else None for v in enriched[column].to_numpy(float)]
    return dd


def _inverse_hs_patterns(mode):
    points = {"confirmed": [(0, 105), (10, 100), (20, 110), (30, 90), (40, 111), (50, 101), (58, 109), (62, 114), (65, 116)],
              "awaiting": [(0, 105), (10, 100), (20, 110), (30, 90), (40, 111), (50, 101), (65, 109)]}[mode]
    x = np.arange(66)
    close = np.interp(x, [p[0] for p in points], [p[1] for p in points])
    volume = np.full(66, 1_000.0)
    volume[-6:] = 1_800.0
    data = {"Date": list(range(66)), "Open": (close - 0.1).tolist(), "High": (close + 0.5).tolist(),
            "Low": (close - 0.5).tolist(), "Close": close.tolist(), "Volume": volume.tolist()}
    return compatibility_patterns(PatternEngine.from_mapping(data).detect())


def test_analyze_score_attaches_measured_evidence_only_to_confirmed_patterns(monkeypatch):
    dd = _random_walk_dd()
    confirmed = _inverse_hs_patterns("confirmed")
    assert any(p["pattern_status"] == "confirmed" for p in confirmed)
    monkeypatch.setattr(ix.ChartPatternAnalyzer, "detect_patterns", lambda self: [dict(p) for p in confirmed])
    _, _, _, geo, _ = ix.analyze_score(dd, "US", "1y")
    done = [p for p in geo if p["pattern_status"] == "confirmed" and p["id"] == "inverse_head_shoulders"]
    assert done and done[0]["evidence_text"].startswith("과거 10년")
    assert "통계적 근거 없음" in done[0]["evidence_text"]
    assert all("evidence_text" not in p for p in geo if p["pattern_status"] != "confirmed")


def test_analyze_score_survives_a_broken_evidence_file(monkeypatch, tmp_path):
    pe._PATH_OVERRIDE = str(tmp_path / "missing.json")
    confirmed = _inverse_hs_patterns("confirmed")
    monkeypatch.setattr(ix.ChartPatternAnalyzer, "detect_patterns", lambda self: [dict(p) for p in confirmed])
    _, _, _, geo, _ = ix.analyze_score(_random_walk_dd(), "US", "1y")
    assert geo and all("evidence_text" not in p for p in geo)


# ── 화면 정합성(소스 고정) ───────────────────────────────────────────────────────

def test_pattern_dimension_counts_only_what_the_backend_scores():
    block = SOURCE.split("// ── 5. 캔들·차트 패턴 신호", 1)[1].split("// ── 종합 등급 계산", 1)[0]
    assert "patterns.filter(p => p.score_eligible !== false && !p.pruned)" in block
    assert "scoredPatterns.filter(p => p.direction === '상승')" in block
    assert "patterns.filter(p => p.direction === '상승')" not in block      # 방향 라벨만으로 세던 이전 방식 제거
    assert "(i !== 4 || patternSignalActive)" in SOURCE                   # 근거 있는 패턴이 없으면 평균에서 제외
    assert "검증을 통과한 패턴 없음" in SOURCE


def test_pattern_cards_and_chart_show_the_measured_evidence_lines():
    assert "p.evidence_text ? `<div class=\"pattern-evid\">${p.evidence_text}</div>`" in SOURCE
    assert 'id="sr-level-note"' in SOURCE and "cd.sr_levels" in SOURCE
    assert "candleSeries.createPriceLine({\n        price: lv.price" in SOURCE.replace("\r\n", "\n")
    assert '"sr_levels": sr_levels_payload' in SOURCE
    # 근거가 일봉 사건 연구이므로 분봉 기간에는 수평선을 만들지 않고, 환경변수로 끌 수 있다.
    assert '_resolve_tf(period) == "1D"' in SOURCE and 'STOCKORACLE_SR_LEVELS' in SOURCE
    # 이전 요청(패턴 WAIT/NECK/FORM 라벨 제거)은 되돌리지 않는다.
    assert "차트 탭 호버 패턴 라벨 비활성화" in SOURCE
