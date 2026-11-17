"""7단계 스캔 엔진 정합성 감사(2026-11-12)의 회귀 테스트.

감사 전에는 `run_full_scan`·`build_snapshot_from_ohlcv`를 직접 검증하는 테스트가 하나도 없었다. 여기서는
(1) 엔진 함수 단위 결함 — 시장 라벨, 종목 단위 예외 격리, 진행 중 막대, 허스트 중복 계산, RS 창 정렬, 낡은 시세,
(2) 종합 점수의 품질 등급 중복 반영, (3) 네트워크 없이 `/api/scan` 전체를 돌려 응답 계약(화면이 읽는 필드)과
추가 투명성 필드를 고정한다.
"""

import json
import math
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from market_briefing import hybrid_signals, scan_engine, technique_prune
from market_briefing.dual_score_v2 import calc_hurst_v2
from market_briefing.scan_engine import (StockUniverse, TechnicalSnapshot, build_snapshot_from_ohlcv,
                                         run_full_scan, stale_quote_tickers)


# ───────────────────────── 합성 일봉 ─────────────────────────

def _ohlcv(seed=1, n=260, start=12.0, drift=0.0006, vol=0.018, volume=2_000_000):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vols = rng.integers(int(volume * 0.6), int(volume * 1.4), n).astype(float)
    return {"closes": close.tolist(), "highs": high.tolist(), "lows": low.tolist(), "volumes": vols.tolist(),
            "opens": open_.tolist()}


def _snapshot(ticker, seed, **kw):
    d = _ohlcv(seed, **kw)
    return build_snapshot_from_ohlcv(ticker, d["closes"], d["highs"], d["lows"], d["volumes"], d["opens"])


# ───────────────────────── (1) 시장 라벨 ─────────────────────────

def _rules_file(tmp_path, krx_status, us_status):
    def node(status):
        return {"status": status, "validated": True, "exclusions": []}
    doc = {"version": 2, "strict_validated_only": False,
           "rules": {"hybrid_breakout": {"KRX": node(krx_status), "US": node(us_status)}}}
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def test_run_full_scan_applies_the_technique_rules_of_the_requested_market(tmp_path, monkeypatch):
    monkeypatch.setattr(technique_prune, "_RULES_PATH_OVERRIDE", _rules_file(tmp_path, "KEEP", "DROP"))
    technique_prune._RULES_CACHE.clear()
    snaps = {"AAA": _snapshot("AAA", 1), "BBB": _snapshot("BBB", 2)}
    universe = [StockUniverse("AAA", "AAA", "CORE"), StockUniverse("BBB", "BBB", "CORE")]   # 통화 기본값 KRW

    as_us = run_full_scan(universe, snaps, {}, market="US")
    as_krx = run_full_scan(universe, snaps, {}, market="KRX")

    assert {c.filter_detail["technique_validation"]["market"] for c in as_us.candidates} == {"US"}
    assert all(c.filter_detail.get("pruned") for c in as_us.candidates)            # 미국 규칙(DROP)이 적용
    assert {c.filter_detail["technique_validation"]["market"] for c in as_krx.candidates} == {"KRX"}
    assert not any(c.filter_detail.get("pruned") for c in as_krx.candidates)       # 한국 규칙(KEEP)이 적용


def test_run_full_scan_without_market_falls_back_to_the_stock_currency(tmp_path, monkeypatch):
    monkeypatch.setattr(technique_prune, "_RULES_PATH_OVERRIDE", _rules_file(tmp_path, "KEEP", "DROP"))
    technique_prune._RULES_CACHE.clear()
    snaps = {"AAA": _snapshot("AAA", 1)}
    krw = run_full_scan([StockUniverse("AAA", "AAA", "CORE")], snaps, {})                      # 기존 호출 방식 그대로
    usd = run_full_scan([StockUniverse("AAA", "AAA", "CORE", currency="USD")], snaps, {})
    assert krw.candidates[0].filter_detail["technique_validation"]["market"] == "KRX"
    assert usd.candidates[0].filter_detail["technique_validation"]["market"] == "US"


# ───────────────────────── (1) 종목 단위 예외 격리 ─────────────────────────

def test_one_bad_snapshot_does_not_abort_the_whole_scan():
    good_a, good_b = _snapshot("GOOD1", 3), _snapshot("GOOD2", 4)
    bad = _snapshot("BAD", 5)
    bad.atr = float("nan")                         # 이전에는 rank_candidate 의 round(NaN) 이 전체 스캔을 중단시켰다
    inf = _snapshot("INF", 6)
    inf.efficiency = float("inf")
    snaps = {"GOOD1": good_a, "GOOD2": good_b, "BAD": bad, "INF": inf}
    universe = [StockUniverse(t, t, "CORE", currency="USD") for t in snaps]

    result = run_full_scan(universe, snaps, {}, market="US")

    assert {c.ticker for c in result.candidates} == {"GOOD1", "GOOD2"}
    assert sorted(result.skipped) == ["BAD", "INF"]
    assert result.total_scanned == 4               # 검토 대상 수는 그대로, 건너뛴 종목은 따로 보고


def test_missing_or_empty_snapshots_are_skipped_silently_not_reported_as_failures():
    snaps = {"AAA": _snapshot("AAA", 1), "EMPTY": TechnicalSnapshot("EMPTY", [], [], [], [])}
    universe = [StockUniverse("AAA", "AAA", "CORE"), StockUniverse("EMPTY", "EMPTY", "CORE"),
                StockUniverse("NOSNAP", "NOSNAP", "CORE")]
    result = run_full_scan(universe, snaps, {})
    assert [c.ticker for c in result.candidates] == ["AAA"]
    assert result.skipped == []                    # 데이터가 없는 것(가격 0)과 계산 실패는 구분한다


# ───────────────────────── (1) 진행 중 막대: 기법 게이트 맥락 ─────────────────────────

def test_gate_context_conditions_use_completed_bars_when_the_last_bar_is_in_progress():
    d = _ohlcv(8, 120, volume=1_000_000)
    d["volumes"][-1] = d["volumes"][-1] * 30          # 장중 누적 거래량이 아니라 '큰 값'이 들어온 마지막 막대
    snap = build_snapshot_from_ohlcv("X", d["closes"], d["highs"], d["lows"], d["volumes"], d["opens"],
                                     volume_in_progress=True)
    assert snap.bar_in_progress is True
    live = scan_engine._hybrid_gate_conditions(snap, ncs=50, fws=30, regime="BULLISH", vol_regime="NORMAL_VOL",
                                               distance_pct=1.0)
    expected = technique_prune.context_conditions(d["closes"][:-1], d["highs"][:-1], d["lows"][:-1], d["volumes"][:-1])
    for key in ("trend_context", "volatility_context", "volume_context"):
        assert live[key] == expected[key]
    # 확정된 막대였다면 같은 입력에서 거래량 맥락이 '>=1.5x' 가 된다(진행 중 처리가 실제로 결과를 바꾼다)
    done = build_snapshot_from_ohlcv("X", d["closes"], d["highs"], d["lows"], d["volumes"], d["opens"])
    assert scan_engine._hybrid_gate_conditions(done, ncs=50, fws=30, regime="BULLISH", vol_regime="NORMAL_VOL",
                                               distance_pct=1.0)["volume_context"] == ">=1.5x"
    assert live["volume_context"] != ">=1.5x"


# ───────────────────────── (1) 허스트 중복 계산 ─────────────────────────

def test_snapshot_computes_hurst_once_and_keeps_both_values_identical(monkeypatch):
    calls = []
    original = calc_hurst_v2

    def counting(closes, min_bars=50):
        calls.append(len(closes))
        return original(closes, min_bars)

    monkeypatch.setattr(scan_engine, "calc_hurst_v2", counting)
    monkeypatch.setattr(hybrid_signals, "_calc_hurst_returns", counting)
    d = _ohlcv(9)
    snap = build_snapshot_from_ohlcv("X", d["closes"], d["highs"], d["lows"], d["volumes"], d["opens"])

    assert len(calls) == 1                                           # 이전: 스냅샷마다 2번
    assert snap.hurst == pytest.approx(original(d["closes"]))
    assert snap.hybrid_signal["hurst"] == round(snap.hurst, 4)        # hybrid 쪽 값도 예전 계산(반올림)과 같다


def test_hybrid_score_without_a_hint_still_computes_hurst_itself():
    d = _ohlcv(9)
    alone = hybrid_signals.compute_hybrid_score(d["closes"], d["highs"], d["lows"], d["volumes"], d["opens"])
    hinted = hybrid_signals.compute_hybrid_score(d["closes"], d["highs"], d["lows"], d["volumes"], d["opens"],
                                                 hurst_value=calc_hurst_v2(d["closes"]))
    assert alone == hinted
    short = hybrid_signals.compute_hybrid_score(d["closes"][:40], d["highs"][:40], d["lows"][:40], d["volumes"][:40],
                                                d["opens"][:40], hurst_value=None)
    assert short["hurst"] is None


# ───────────────────────── (1) RS 창 정렬 ─────────────────────────

def test_relative_strength_compares_the_same_tail_window():
    stock = _ohlcv(10, 100)
    bench_close = [100.0 - 0.2 * k for k in range(150)] + [70.0 * (1 + 0.004 * k) for k in range(100)]   # 앞 150봉 하락(100→70), 마지막 100봉 +39.6%
    snap = build_snapshot_from_ohlcv("X", stock["closes"], stock["highs"], stock["lows"], stock["volumes"],
                                     stock["opens"], bench_closes=bench_close)
    stock_ret = (stock["closes"][-1] / stock["closes"][0] - 1) * 100
    bench_ret = (bench_close[-1] / bench_close[-100] - 1) * 100
    assert snap.rs_vs_bench == pytest.approx(round(stock_ret - bench_ret, 2))
    # 이전 계산(벤치마크 전체 250봉)과는 다른 값이어야 의미가 있다
    legacy = round(stock_ret - (bench_close[-1] / bench_close[0] - 1) * 100, 2)
    assert snap.rs_vs_bench != legacy


def test_relative_strength_is_unchanged_when_both_windows_have_the_same_length():
    stock = _ohlcv(11, 250)
    bench = _ohlcv(12, 250)["closes"]
    snap = build_snapshot_from_ohlcv("X", stock["closes"], stock["highs"], stock["lows"], stock["volumes"],
                                     stock["opens"], bench_closes=bench)
    legacy = round((stock["closes"][-1] / stock["closes"][0] - 1) * 100 - (bench[-1] / bench[0] - 1) * 100, 2)
    assert snap.rs_vs_bench == legacy
    none = build_snapshot_from_ohlcv("X", stock["closes"], stock["highs"], stock["lows"], stock["volumes"], stock["opens"])
    assert none.rs_vs_bench == 0.0


# ───────────────────────── (1) 낡은 시세 ─────────────────────────

def test_stale_quote_tickers_flags_only_symbols_lagging_the_market_reference_day():
    dates = {f"T{k}": "2026-11-11" for k in range(6)}
    dates.update({"HALTED": "2026-11-02", "LAG2": "2026-11-09", "BOGUS_FUTURE": "2026-12-31", "BROKEN": "n/a"})
    stale = stale_quote_tickers(dates)
    assert set(stale) == {"HALTED"}                                   # 9일 지연만. 2일 지연·잘못된 단독 미래 날짜·읽을 수 없는 값은 제외
    assert stale["HALTED"] == {"last_bar": "2026-11-02", "lag_days": 9, "reference": "2026-11-11"}


def test_market_wide_holiday_or_tiny_universe_never_marks_symbols_stale():
    assert stale_quote_tickers({f"T{k}": "2026-10-02" for k in range(8)}) == {}            # 모두 같은 마지막 날짜(연휴)
    assert stale_quote_tickers({"ONLY": "2026-11-11"}) == {}
    assert stale_quote_tickers({}) == {} and stale_quote_tickers(None) == {}
    assert stale_quote_tickers({"A": "2026-11-11", "B": "2026-10-01"}) == {"B": {
        "last_bar": "2026-10-01", "lag_days": 41, "reference": "2026-11-11"}}               # 2종목뿐이면 최신 날짜가 기준


# ───────────────────────── (2) 종합 점수의 품질 등급 중복 반영 ─────────────────────────

def _cand(**kw):
    base = {"status": "WATCH", "ncs": 60.0, "adjusted_ncs": 60.0, "quality_tier": "high", "quality_multiplier": 1.0}
    base.update(kw)
    return base


def test_composite_counts_the_quality_tier_once(monkeypatch):
    from api import index
    monkeypatch.delenv("STOCKORACLE_SCAN_NCS_QMJ_ADJUST", raising=False)
    high = _cand()
    unknown = _cand(adjusted_ncs=30.0, quality_tier="unknown", quality_multiplier=0.5)     # 정보 조회 실패: NCS 가 절반이 되던 경우
    assert index.scan_ncs_score(unknown) == 60.0                                          # NCS 축에는 품질 배율이 곱해지지 않는다
    gap = index.scan_composite_score(high) - index.scan_composite_score(unknown)
    # 남는 차이는 세 번째 축(품질)의 가중 25%×(100−50)점 하나뿐이다
    assert gap == pytest.approx(index.SCAN_W_MOMENTUM * (100.0 - 50.0), abs=0.01)


def test_composite_legacy_switch_restores_the_multiplied_score(monkeypatch):
    from api import index
    unknown = _cand(adjusted_ncs=30.0, quality_tier="unknown", quality_multiplier=0.5)
    monkeypatch.setenv("STOCKORACLE_SCAN_NCS_QMJ_ADJUST", "1")
    assert index.scan_ncs_qmj_adjust_enabled() is True
    assert index.scan_ncs_score(unknown) == 30.0
    legacy_gap = index.scan_composite_score(_cand()) - index.scan_composite_score(unknown)
    monkeypatch.delenv("STOCKORACLE_SCAN_NCS_QMJ_ADJUST")
    assert legacy_gap > index.scan_composite_score(_cand()) - index.scan_composite_score(unknown) + 5   # 예전엔 격차가 훨씬 컸다


def test_scan_ncs_score_tolerates_missing_fields():
    from api import index
    assert index.scan_ncs_score({}) == 0.0
    assert index.scan_ncs_score({"ncs": 0.0, "adjusted_ncs": 40.0}) == 0.0                 # 0.0 도 유효값
    assert index.scan_ncs_score({"adjusted_ncs": 42.0}) == 42.0                            # NCS 가 없으면 있는 값을 쓴다


# ───────────────────────── (3) /api/scan 전체 — 네트워크 없이 ─────────────────────────

class _FakeTicker:
    def __init__(self, frames, infos, symbol):
        self._frames, self._infos, self._symbol = frames, infos, symbol

    def history(self, period="1y", **kw):
        frame = self._frames.get(self._symbol)
        return frame.copy() if frame is not None else pd.DataFrame()

    @property
    def info(self):
        return dict(self._infos.get(self._symbol, {}))


class _FakeYF:
    def __init__(self, frames, infos):
        self.frames, self.infos = frames, infos

    def download(self, tickers, **kw):
        names = [t for t in str(tickers).split() if t in self.frames]
        if not names:
            return pd.DataFrame()
        return pd.concat({t: self.frames[t] for t in names}, axis=1)

    def Ticker(self, symbol):
        return _FakeTicker(self.frames, self.infos, symbol)


def _frame(seed, end: date, n=260, **kw):
    d = _ohlcv(seed, n, **kw)
    index = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    return pd.DataFrame({"Open": d["opens"], "High": d["highs"], "Low": d["lows"], "Close": d["closes"],
                         "Volume": d["volumes"]}, index=index)


@pytest.fixture
def offline_scan(monkeypatch):
    from api import index
    end = date(2026, 6, 30)                                      # 과거 날짜라 '장중 진행 중 막대'가 되지 않는다
    names = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
    frames = {t: _frame(20 + k, end, start=10 + k, drift=0.0004 * (k % 3)) for k, t in enumerate(names)}
    frames["STALE"] = _frame(40, end - timedelta(days=12), start=9)            # 마지막 일봉이 12일 전 — 거래정지 가정
    frames["SPY"] = _frame(41, end, start=400, vol=0.008)
    frames["^VIX"] = _frame(42, end, n=10, start=16)
    infos = {t: {"sector": "Technology", "marketCap": 5e9, "returnOnEquity": 0.2, "debtToEquity": 50,
                 "revenueGrowth": 0.1, "recommendationKey": "buy"} for t in names + ["STALE"]}
    infos["CCC"] = {"sector": "Energy", "marketCap": 3e9}                       # 재무 정보 없음 → unknown 등급
    monkeypatch.setattr(index, "yf", _FakeYF(frames, infos))
    monkeypatch.setattr(index, "_SCAN_RESULT_CACHE", {})

    def all_complete(cands, snap_map, **kwargs):                                 # 박스 플랜 엔진은 따로 검증되므로 완성으로 고정
        for cd in cands:
            snap = snap_map[cd["ticker"]]
            cd["box_plan"] = {"complete": True, "missing": [], "quote": snap.closes[-1], "target2": {"price": 0.01}}
        return {"version": "stub", "evaluated": len(cands), "complete": len(cands), "incomplete": 0, "excluded": {}}

    monkeypatch.setattr("market_briefing.scan_box_plan.attach_box_plans", all_complete)
    return index, names


def _scan(index, **extra):
    params = {"market": "US", "tickers": "AAA,BBB,CCC,DDD,EEE,FFF,GGG,HHH,STALE", "refresh": "1"}
    params.update(extra)
    return index.route("/api/scan", params)


def test_scan_route_keeps_the_fields_the_page_reads_and_adds_transparency_fields(offline_scan):
    index, names = offline_scan
    response = _scan(index)

    assert "error" not in response, response.get("error")
    for key in ("regime", "vol_regime", "total_scanned", "passed_filters", "ready_count", "watch_count", "far_count",
                "good_count", "premium_count", "relaxed", "display_cap", "tier_distribution", "plan_stats",
                "coverage", "candidates", "generated_at"):
        assert key in response, key                                              # 화면(renderScanResult)이 읽는 필드
    for key in ("universe", "collected", "full_review", "unresolved"):
        assert key in response["coverage"], key
    assert response["candidates"], "합성 종목이 하나도 선정되지 않았다"
    row = response["candidates"][0]
    for key in ("ticker", "name", "price", "bqs", "cap_tier", "cap_tier_ko", "category", "status", "box_plan"):
        assert key in row, key
    assert isinstance(row["stage2_passed"], bool)
    assert set(row["entry_signal"]) == {"allowed", "reason"}
    assert row["filter_detail"]["technique_validation"]["market"] == "US"        # 통화를 넣어 만든 미국 종목은 미국 규칙으로 평가


def test_scan_route_excludes_stale_quotes_and_reports_them_without_a_false_timeout_note(offline_scan):
    index, names = offline_scan
    response = _scan(index)

    assert "STALE" not in {c["ticker"] for c in response["candidates"]}
    quality = response["data_quality"]
    assert [row["ticker"] for row in quality["stale_excluded"]] == ["STALE"]
    assert quality["stale_excluded"][0]["lag_days"] >= 10
    funnel = response["stage_funnel"]
    assert funnel["universe"] == 9 and funnel["stale_excluded"] == 1 and funnel["scanned"] == 8
    assert funnel["collected"] == 8
    # 낡은 시세 제외는 '수집 시간 초과로 미수집'이 아니다 — 전수 검토 표기가 깨지면 안 된다
    assert response["coverage"]["unresolved"] == 0 and response["coverage"]["full_review"] is True
    assert response["coverage"]["stale_excluded"] == 1
    assert "미수집" not in response["coverage"]["note"]


def test_scan_route_reports_stage_counts_and_unknown_quality(offline_scan):
    index, names = offline_scan
    response = _scan(index)
    funnel, quality = response["stage_funnel"], response["data_quality"]

    assert funnel["calc_skipped"] == 0 and funnel["quality_excluded"] == 0
    assert funnel["stage2_passed"] >= response["passed_filters"]                  # 게이트 전 통과 수 ≥ 게이트 후 통과 수
    assert funnel["selected"] == response["good_count"]
    assert quality["quality_unknown"] >= 1                                         # CCC: 재무 정보 없음
    assert quality["calc_skipped"] == []
    assert quality["quality_stage"] == "applied" and quality["quality_coverage"] == pytest.approx(7 / 8, abs=0.001)
    assert quality["yf_cookie_reuse"] is index.YF_COOKIE_REUSE                       # 수집 속도 수정(쿠키 재사용)이 켜져 있는지 알린다
    assert set(quality["timings_s"]) >= {"benchmark_s", "bulk_s", "history_s", "info_s", "engine_s", "total_s"}


def test_scan_route_orders_by_composite_without_multiplying_quality(offline_scan, monkeypatch):
    index, names = offline_scan
    monkeypatch.delenv("STOCKORACLE_SCAN_NCS_QMJ_ADJUST", raising=False)
    response = _scan(index)
    by_ticker = {c["ticker"]: c for c in response["candidates"]}
    unknown = by_ticker["CCC"]
    assert unknown["quality_tier"] == "unknown" and unknown["quality_multiplier"] == 0.5
    expected = index.scan_composite_score({**unknown})
    assert unknown["composite_score"] == expected
    # NCS 축은 배율이 곱해지지 않은 NCS 그대로 들어간다(이전에는 adjusted_ncs = NCS × 0.5)
    assert index.scan_ncs_score(unknown) == unknown["ncs"] != unknown["adjusted_ncs"] or unknown["ncs"] == 0


def test_scan_route_survives_a_ticker_whose_prices_turn_invalid(offline_scan, monkeypatch):
    index, names = offline_scan
    real = scan_engine.build_snapshot_from_ohlcv

    def poisoned(ticker, *args, **kwargs):
        snap = real(ticker, *args, **kwargs)
        if ticker == "DDD":
            snap.adx = float("nan")
        return snap

    monkeypatch.setattr(scan_engine, "build_snapshot_from_ohlcv", poisoned)
    monkeypatch.setattr(index, "_SCAN_RESULT_CACHE", {})
    response = _scan(index)
    assert "error" not in response, response.get("error")
    assert "DDD" not in {c["ticker"] for c in response["candidates"]}
    assert response["data_quality"]["calc_skipped"] == ["DDD"]
    assert response["stage_funnel"]["calc_skipped"] == 1
    assert response["good_count"] >= 5                                            # 나머지 종목은 정상 선정


_JUNK_INFO = {"sector": "Technology", "marketCap": 5e9, "returnOnEquity": -0.5, "debtToEquity": 400,
              "revenueGrowth": -0.2, "recommendationKey": "sell"}


def test_scan_route_skips_the_quality_stage_when_most_financial_data_is_missing(offline_scan):
    index, names = offline_scan
    index.yf.infos = {t: {} for t in names + ["STALE"]}        # 정보 조회가 거의 실패한 상황(시간 예산·401 등)
    index.yf.infos["BBB"] = dict(_JUNK_INFO)                    # 한 종목만 정보가 와 있고 '불량'으로 판정된다
    response = _scan(index)

    quality, funnel = response["data_quality"], response["stage_funnel"]
    assert quality["quality_stage"] == "skipped_low_coverage"
    assert quality["quality_coverage"] < index.SCAN_QUALITY_MIN_COVERAGE
    assert funnel["quality_excluded"] == 0                       # 우연히 정보가 도착한 한 종목만 탈락시키지 않는다
    assert "BBB" in {c["ticker"] for c in response["candidates"]}
    assert {c["quality_tier"] for c in response["candidates"]} == {"unknown"}


def test_scan_route_applies_the_quality_filter_when_coverage_is_enough_or_forced(offline_scan, monkeypatch):
    index, names = offline_scan
    index.yf.infos["BBB"] = dict(_JUNK_INFO)                    # 나머지 종목은 정상 재무 정보(커버리지 7/8)
    response = _scan(index)
    assert response["data_quality"]["quality_stage"] == "applied"
    assert response["stage_funnel"]["quality_excluded"] == 1     # 불량 등급 종목은 1단계에서 탈락
    assert "BBB" not in {c["ticker"] for c in response["candidates"]}

    index.yf.infos = {t: {} for t in names + ["STALE"]}
    index.yf.infos["BBB"] = dict(_JUNK_INFO)
    monkeypatch.setattr(index, "SCAN_QUALITY_MIN_COVERAGE", 0.0)   # 이전 동작 복원: 커버리지와 무관하게 항상 적용
    monkeypatch.setattr(index, "_SCAN_RESULT_CACHE", {})
    forced = _scan(index)
    assert forced["data_quality"]["quality_stage"] == "applied"
    assert forced["stage_funnel"]["quality_excluded"] == 1
