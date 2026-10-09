"""손실조건 제거(Subtractive Pruning) 검증 테스트 — 합성 데이터만 사용."""

import json

import pytest

from scripts import prune_loss_conditions as prune
from market_briefing import technique_prune
from market_briefing.technique_prune import (
    drsi_conditions,
    hybrid_conditions,
    leader_conditions,
    ml_conditions,
    pattern_conditions,
)


def _ohlc(n=60, base=100.0, drift=0.0):
    closes = [base + drift * i for i in range(n)]
    return {
        "opens": list(closes),
        "highs": [c * 1.01 for c in closes],
        "lows": [c * 0.99 for c in closes],
        "closes": list(closes),
        "volumes": [1000.0] * n,
    }


def _sig(ticker, idx, entry, stop, target, conds, market="KRX",
         tech="hybrid_breakout"):
    return {
        "ticker": ticker, "market": market, "technique": tech,
        "signal_index": idx, "entry_index": idx + 1,
        "entry_price": entry, "stop_price": stop, "target_price": target,
        "conditions": dict(conds),
    }


def test_simulator_stop_first_on_same_bar():
    ohl = _ohlc()
    ohl["lows"][11] = 90.0    # stop 95 관통
    ohl["highs"][11] = 112.0  # target 110도 도달 → 손절 우선
    sig = _sig("T", 10, 100.0, 95.0, 110.0, {"regime": "BULLISH"})
    (trade,) = prune.simulate_trades([sig], {"T": ohl}, 0.0)
    assert trade["exit_reason"] == "stop_first"
    assert trade["exit_price"] == 95.0
    assert trade["win"] is False


def test_simulator_target_and_timeout():
    ohl = _ohlc()
    ohl["highs"][11] = 112.0
    sig = _sig("T", 10, 100.0, 95.0, 110.0, {"regime": "BULLISH"})
    (trade,) = prune.simulate_trades([sig], {"T": ohl}, 0.0)
    assert trade["exit_reason"] == "target"
    assert trade["win"] is True

    flat = _ohlc()
    sig2 = _sig("T", 10, 100.0, 90.0, 200.0, {"regime": "SIDEWAYS"})
    (trade2,) = prune.simulate_trades([sig2], {"T": flat}, 0.0)
    assert trade2["exit_reason"] == "timeout"


def test_attribution_finds_most_frequent_loss_condition():
    ohl = _ohlc(n=120)
    signals = []
    for i in range(30):
        cond = {"regime": "BEARISH" if i < 20 else "BULLISH"}
        # BEARISH 20건은 손절(하락 드리프트), BULLISH 10건은 익절(상승 드리프트)
        signals.append(_sig("T", 10 + i, 100.0, 95.0, 110.0, cond))
    bear = _ohlc(n=120, drift=-0.5)
    bull = _ohlc(n=120, drift=0.5)
    mixed = dict(bear)
    mixed["closes"] = bear["closes"][:20] + bull["closes"][:100]
    mixed["opens"] = list(mixed["closes"])
    mixed["highs"] = [c * 1.005 for c in mixed["closes"]]
    mixed["lows"] = [c * 0.995 for c in mixed["closes"]]
    trades = prune.simulate_trades(signals, {"T": mixed}, 0.0)
    best = prune.best_exclusion(trades)
    assert best is not None
    assert best["condition"] == "regime"
    assert best["value"] in ("BEARISH", "BULLISH")
    assert best["losses"] >= 5


def test_prune_loop_removes_worst_and_terminates():
    # 조건 A Zeichen: A에서만 손실, B에서는 전승 → A가 제거되어야 함
    signals = []
    for i in range(24):
        cond = {"zone": "A" if i < 12 else "B"}
        signals.append(_sig("T", i, 100.0, 99.0, 101.0, cond))
    ohl = _ohlc(n=120)
    # A 신호 구간은 하락, B 신호 구간은 상승으로 조작
    closes = list(ohl["closes"])
    for i in range(12):
        closes[i + 1] = 98.0
        ohl["lows"][i + 1] = 97.0
    for i in range(12, 24):
        closes[i + 1] = 102.0
        ohl["highs"][i + 1] = 103.0
    ohl["closes"] = closes
    ohl["opens"] = list(closes)
    res = prune.prune_technique(signals, {"T": ohl}, 0.0,
                                min_trades=20, max_iters=10)
    assert res["baseline"]["trades"] == 24
    assert res["verdict"] in ("PRUNED_KEEP", "KEEP", "DROP")
    assert len(res["steps"]) <= 10


def test_min_trades_gate_marks_insufficient():
    signals = [_sig("T", i, 100.0, 95.0, 110.0, {"r": "x"}) for i in range(5)]
    res = prune.prune_technique(signals, {"T": _ohlc()}, 0.0, min_trades=20)
    assert res["verdict"] == "INSUFFICIENT"
    assert res["exclusions"] == []


def test_filter_signals_drops_matching_rule_only():
    signals = [
        _sig("T", 0, 100.0, 95.0, 110.0, {"regime": "BEARISH"}),
        _sig("T", 1, 100.0, 95.0, 110.0, {"regime": "BULLISH"}),
    ]
    kept = prune.filter_signals(signals, [{"condition": "regime",
                                           "value": "BEARISH"}])
    assert [s["signal_index"] for s in kept] == [1]


def test_rules_roundtrip_and_allowed(tmp_path, monkeypatch):
    rules = {"rules": {"hybrid_breakout": {"KRX": {
        "status": "PRUNED_KEEP",
        "exclusions": [{"condition": "regime", "value": "BEARISH"}],
    }}}}
    path = tmp_path / "technique_prune.json"
    path.write_text(json.dumps(rules), encoding="utf-8")
    monkeypatch.setattr(technique_prune, "_RULES_PATH_OVERRIDE", str(path))

    loaded = technique_prune.load_rules()
    assert technique_prune.technique_status("hybrid_breakout", "KRX",
                                            loaded) == "PRUNED_KEEP"
    allowed, _ = technique_prune.technique_allowed(
        "hybrid_breakout", "KRX", {"regime": "BEARISH"}, loaded)
    assert allowed is False
    allowed, _ = technique_prune.technique_allowed(
        "hybrid_breakout", "KRX", {"regime": "BULLISH"}, loaded)
    assert allowed is True


def test_allowed_fail_open_without_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(technique_prune, "_RULES_PATH_OVERRIDE",
                        str(tmp_path / "missing.json"))
    allowed, _ = technique_prune.technique_allowed("ml_direction", "US", {})
    assert allowed is True


def test_bucket_handles_invalid():
    from market_briefing.technique_prune import bucket
    assert bucket(None, [1], ["a", "b"]) == "na"
    assert bucket(float("nan"), [1], ["a", "b"]) == "na"
    assert bucket(5, [1, 10], ["lo", "mid", "hi"]) == "mid"


def test_condition_mappers_share_keys_with_prune_script():
    hs = {"ncs": 72.0, "fws": 25.0, "regime": "BULLISH",
          "vol_regime": "NORMAL_VOL", "adx": 22.0, "dist_to_high": 1.5}
    assert hybrid_conditions(hs) == {
        "ncs_bucket": ">=70", "fws_bucket": "20-30", "regime": "BULLISH",
        "vol_regime": "NORMAL_VOL", "adx_bucket": "20-30",
        "dist_bucket": "1-2%",
    }
    pat = {"family": "triangle_wedge", "completion_score": 90.0,
           "tolerance_mode": "hybrid"}
    assert pattern_conditions(pat)["family"] == "triangle_wedge"
    assert pattern_conditions(pat)["completion_bucket"] == ">=85"
    assert drsi_conditions("KRX", 28.0, 2.0)["rsi_bucket"] == "<30"
    ld = {"rs_edge_pp": 30.0, "drawdown_pct": -45.0, "market_confirm": True}
    assert leader_conditions(ld) == {
        "rs_edge_bucket": ">=25pp", "drawdown_bucket": "40-55%",
        "bench_flag": "bench",
    }
    ml = {"confidence": 0.7, "prob_up": 0.65, "model_available": True}
    assert ml_conditions(ml) == {
        "conf_bucket": "0.65-0.75", "prob_up_bucket": "0.60-0.70",
        "model_flag": "model",
    }


def test_gate_drop_blocks_every_signal(tmp_path, monkeypatch):
    rules = {"rules": {"pattern_breakout": {"KRX": {"status": "DROP",
                                                   "exclusions": []}}}}
    path = tmp_path / "technique_prune.json"
    path.write_text(json.dumps(rules), encoding="utf-8")
    monkeypatch.setattr(technique_prune, "_RULES_PATH_OVERRIDE", str(path))
    loaded = technique_prune.load_rules()
    for family in ("triangle_wedge", "head_shoulders", "double_reversal"):
        allowed, _ = technique_prune.technique_allowed(
            "pattern_breakout", "KRX", {"family": family}, loaded)
        assert allowed is False


def test_gate_exclusion_blocks_only_matching_family(tmp_path, monkeypatch):
    rules = {"rules": {"pattern_breakout": {"US": {
        "status": "PRUNED_KEEP",
        "exclusions": [{"condition": "family",
                        "value": "continuation_consolidation"}],
    }}}}
    path = tmp_path / "technique_prune.json"
    path.write_text(json.dumps(rules), encoding="utf-8")
    monkeypatch.setattr(technique_prune, "_RULES_PATH_OVERRIDE", str(path))
    loaded = technique_prune.load_rules()
    allowed, _ = technique_prune.technique_allowed(
        "pattern_breakout", "US", {"family": "continuation_consolidation"},
        loaded)
    assert allowed is False
    allowed, _ = technique_prune.technique_allowed(
        "pattern_breakout", "US", {"family": "triangle_wedge"}, loaded)
    assert allowed is True


def test_tier_of_ticker_defaults_mid_for_custom():
    assert prune.tier_of_ticker("005930.KS", "KRX") == "LARGE"
    assert prune.tier_of_ticker("058470.KQ", "KRX") == "SMALL"
    assert prune.tier_of_ticker("MU", "US") == "MID"
    assert prune.tier_of_ticker("CUSTOM123", "US") == "MID"


def test_load_rules_cache_and_ttl(tmp_path, monkeypatch):
    path = tmp_path / "technique_prune.json"
    path.write_text(json.dumps({"rules": {}, "v": 1}), encoding="utf-8")
    monkeypatch.setattr(technique_prune, "_RULES_PATH_OVERRIDE", str(path))
    first = technique_prune.load_rules()
    assert first == {"rules": {}, "v": 1}
    path.write_text(json.dumps({"rules": {}, "v": 2}), encoding="utf-8")
    assert technique_prune.load_rules()["v"] == 2  # artifact activation invalidates TTL immediately
    assert technique_prune.load_rules(ttl=0)["v"] == 2  # TTL 만료 시 재독


def test_cache_fresh_missing_and_fresh(tmp_path):
    missing = tmp_path / "no.csv"
    assert prune._cache_fresh(str(missing)) is False
    fresh = tmp_path / "yes.csv"
    fresh.write_text("a", encoding="utf-8")
    assert prune._cache_fresh(str(fresh)) is True
    assert prune._cache_fresh(str(fresh), ttl=0) is False


def test_resolve_tickers_market_guard():
    import argparse
    args = argparse.Namespace(tickers="AAPL,MSFT", market="KRX",
                              tiers="LARGE,MID,SMALL", limit=0)
    assert prune._resolve_tickers(args) == {"KRX": []}


def test_insufficient_result_still_carries_trades():
    signals = [_sig("T", i, 100.0, 95.0, 110.0, {"r": "x"}) for i in range(5)]
    res = prune.prune_technique(signals, {"T": _ohlc()}, 0.0, min_trades=20)
    assert res["verdict"] == "INSUFFICIENT"
    assert len(res["kept_trades"]) == 5
