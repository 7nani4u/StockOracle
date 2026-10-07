"""예측 체인의 중복·충돌·종속 결함 회귀 테스트.

각 테스트는 실제 워크포워드 감사(2022-10~2026-09, 52종목)에서 확인된 결함 하나를 고정한다.
네트워크 없이 합성 데이터와 monkeypatch 만 사용한다.
"""

import numpy as np
import pandas as pd
import pytest

from api import index as ix
from market_briefing import hybrid_signals as hs
from market_briefing.correlation_engine import correlate_and_narrow


def _random_walk_ohlc(n=300, seed=7, drift=0.0004, vol=0.015):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(800_000, 1_400_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})


# ── 하이브리드 레짐: 벤치마크 없음 = 알 수 없음 (상수 페널티 금지) ─────────────────────────

def test_hybrid_without_benchmark_does_not_fabricate_regime_penalty():
    df = _random_walk_ohlc()
    result = hs.compute_hybrid_score(
        closes=df["Close"].tolist(), highs=df["High"].tolist(), lows=df["Low"].tolist(),
        volumes=df["Volume"].tolist(), open_prices=df["Open"].tolist(),
    )
    detail = result["regime_detail"]
    assert detail["ma200_available"] is False
    assert detail["chop_band"] is False          # price == ma200 대체값 때문에 항상 True 였던 값
    assert result["regime_stable"] is True        # FWS 의 '레짐 불안정' +10점 상수 페널티 제거
    assert result["regime"] == hs.REGIME_SIDEWAYS  # 시장 방향을 알 수 없으면 중립


def test_hybrid_fws_has_no_constant_regime_instability_points_without_benchmark():
    df = _random_walk_ohlc(seed=11)
    closes, highs, lows = df["Close"].tolist(), df["High"].tolist(), df["Low"].tolist()
    result = hs.compute_hybrid_score(closes=closes, highs=highs, lows=lows, volumes=df["Volume"].tolist())
    unstable = hs.compute_fws(
        vol_ratio=result["vol_ratio"] or 1.0, ext_atr=result["anti_chase"]["ext_atr"], adx=result["adx"],
        atr_spiking=result["atr_spiking"], atr_collapsing=result["atr_collapsing"], regime_stable=False,
    )
    assert result["fws"] == pytest.approx(unstable - 10.0, abs=0.01)


def test_hybrid_with_real_benchmark_still_detects_regime_and_chop_band():
    df = _random_walk_ohlc(seed=3, drift=0.001)
    closes = df["Close"].tolist()
    uptrend_bench = [100.0 + i * 0.5 for i in range(260)]
    bull = hs.compute_hybrid_score(
        closes=closes, highs=df["High"].tolist(), lows=df["Low"].tolist(), volumes=df["Volume"].tolist(),
        bench_closes=uptrend_bench,
    )
    assert bull["regime_detail"]["ma200_available"] is True
    assert bull["regime_detail"]["bull_pts"] >= 3        # 가격 > MA200 3점은 정상 부여
    flat_bench = [100.0 + (0.1 if i % 2 else -0.1) for i in range(260)]
    chop = hs.compute_hybrid_score(
        closes=closes, highs=df["High"].tolist(), lows=df["Low"].tolist(), volumes=df["Volume"].tolist(),
        bench_closes=flat_bench,
    )
    assert chop["regime_detail"]["chop_band"] is True     # 실제 MA200 ±2% 안이면 CHOP 은 그대로 동작
    assert chop["regime_stable"] is False


def test_compute_regime_skips_ma200_checks_when_unknown():
    unknown = hs.compute_regime(price=100.0, ma200=None, adx_data=None)
    assert unknown["bear_pts"] == 0 and unknown["bull_pts"] == 0
    assert unknown["chop_band"] is False and unknown["ma200_available"] is False
    below = hs.compute_regime(price=90.0, ma200=100.0, adx_data=None)
    assert below["bear_pts"] == 3 and below["ma200_available"] is True


# ── ADX: 하이브리드 구현이 add_indicators(Wilder)와 같은 값을 낸다 ───────────────────────

@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8])
def test_hybrid_adx_matches_wilder_adx_from_add_indicators(seed):
    df = _random_walk_ohlc(n=320, seed=seed, drift=0.0006 if seed % 2 else -0.0003)
    indicators = ix.add_indicators(df.copy(), market="US")
    window = slice(len(df) - 252, len(df))
    hybrid = hs._calc_adx(
        df["High"].tolist()[window], df["Low"].tolist()[window], df["Close"].tolist()[window],
    )
    assert hybrid is not None
    assert hybrid["adx"] == pytest.approx(float(indicators["ADX"].iloc[-1]), abs=0.05)
    assert hybrid["plus_di"] == pytest.approx(float(indicators["DI_Plus"].iloc[-1]), abs=0.05)


# ── 상관 엔진: 신뢰도 표는 신호 방향을 따른다 ──────────────────────────────────────────

def _bearish_dd(n=80):
    closes = [100 - i * 0.3 for i in range(n)]
    return {
        "Close": closes, "High": [c + 1 for c in closes], "Low": [c - 1 for c in closes],
        "Volume": [1e6] * n, "MA20": [None] * 19 + closes[19:], "MA60": [None] * 59 + closes[59:],
        "RSI": [30.0] * n,
    }


def _correlate(score, signal, confidence=78):
    dd = _bearish_dd()
    last = dd["Close"][-1]
    return correlate_and_narrow(
        symbol="X", market="US", dd=dd, last_price=last, atr=2.0, score=score,
        prob_up_base=40.0 if score < 50 else 60.0, prob_down_base=60.0 if score < 50 else 40.0,
        target_price={"min_price": last + 3, "max_price": last + 8},
        signal_confidence={"confidence": confidence, "signal": signal, "confidence_interval": {"spread": 16}},
        indicator_signals={}, candlestick_patterns=[], pullback_analysis=None, investor_flow=None,
        ml_prediction=None, regime="BEAR" if score < 50 else "BULL", pct_change=-1.0, volume_ratio=1.0,
        candle_up=False, rsi=30.0, macd_gap=-0.5, event_risk=None,
    )


def test_confidence_vote_is_negative_for_sell_signal_and_positive_for_buy():
    sell = _correlate(25, "SELL")["correlation"]["dimensions"]["confidence"]
    buy = _correlate(75, "BUY")["correlation"]["dimensions"]["confidence"]
    neutral = _correlate(50, "NEUTRAL")["correlation"]["dimensions"]["confidence"]
    assert sell < 0 < buy
    assert sell == pytest.approx(-buy)
    assert neutral == 0


def test_bearish_setup_agreement_is_not_diluted_by_a_phantom_bullish_vote():
    sell = _correlate(25, "SELL")["correlation"]
    dims = sell["dimensions"]
    # technical·regime·depth 가 모두 하락이면 신뢰도 표도 하락이어야 한다.
    assert dims["technical"] < 0 and dims["regime"] < 0 and dims["confidence"] < 0
    assert sell["dominant"] != "상승"
    # 같은 입력에서 신뢰도 표만 상승으로 잘못 집계하면(예전 동작) 일치도가 낮아졌다.
    mislabeled = _correlate(25, "BUY")["correlation"]
    assert mislabeled["dimensions"]["confidence"] > 0
    assert sell["agreement"] > mislabeled["agreement"]


def test_single_weak_vote_does_not_create_full_agreement():
    # 점수 50(중립)·패턴/수급/ML 없음, 레짐 표 하나만 켜진 상태 — 예전에는 일치도 100%·범위 45% 축소.
    dd = _bearish_dd()
    last = dd["Close"][-1]
    result = correlate_and_narrow(
        symbol="X", market="US", dd=dd, last_price=last, atr=2.0, score=50,
        prob_up_base=50.0, prob_down_base=50.0, target_price={"min_price": last + 3, "max_price": last + 8},
        signal_confidence={"confidence": 50, "signal": "NEUTRAL", "confidence_interval": {"spread": 16}},
        indicator_signals={}, candlestick_patterns=[], pullback_analysis=None, investor_flow=None,
        ml_prediction=None, regime="BULL", pct_change=0.0, volume_ratio=1.0, candle_up=True,
        rsi=50.0, macd_gap=0.0, event_risk=None,
    )
    votes = {k: v for k, v in result["correlation"]["dimensions"].items() if abs(v) > 0.7}
    assert list(votes) == ["regime"]                       # 켜진 표는 하나뿐
    assert result["correlation"]["agreement"] < 0.6         # 기권표가 분모에 들어가므로 만장일치로 보지 않는다
    assert result["target_narrowed"]["factor"] > 0.75       # 하한 0.55 에 붙지 않는다
    assert result["prob_up_corr"] == pytest.approx(50.0, abs=2.0)  # 약한 단일 표가 확률을 ±6p 당기지 않는다


def test_agreement_stays_between_half_and_one():
    for score, regime, signal in ((25, "BEAR", "SELL"), (75, "BULL", "BUY"), (50, "NEUTRAL", "NEUTRAL")):
        result = correlate_and_narrow(
            symbol="X", market="US", dd=_bearish_dd(), last_price=70.0, atr=2.0, score=score,
            prob_up_base=50.0, prob_down_base=50.0, target_price={"min_price": 73.0, "max_price": 78.0},
            signal_confidence={"confidence": 70, "signal": signal, "confidence_interval": {"spread": 16}},
            indicator_signals={}, candlestick_patterns=[], pullback_analysis=None, investor_flow=None,
            ml_prediction=None, regime=regime, pct_change=0.0, volume_ratio=1.0, candle_up=True,
            rsi=50.0, macd_gap=0.0, event_risk=None,
        )
        assert 0.5 <= result["correlation"]["agreement"] <= 1.0
        assert 0.55 <= result["target_narrowed"]["factor"] <= 1.15


def test_complementary_probabilities_stay_complementary_after_correlation():
    # route 는 prob_up+prob_down=100 을 넘긴다. 예전에는 하한 10 의 상수 '횡보'가 down 에서 10p 를 빼
    # 합이 90 이 되고, 이후 up/(up+down) 이 약 11% 부풀었다.
    for regime in ("BULL", "BEAR", "NEUTRAL"):
        result = correlate_and_narrow(
            symbol="X", market="US", dd=_bearish_dd(), last_price=70.0, atr=2.0, score=60,
            prob_up_base=58.0, prob_down_base=42.0, target_price={"min_price": 73.0, "max_price": 78.0},
            signal_confidence={"confidence": 60, "signal": "BUY", "confidence_interval": {"spread": 16}},
            indicator_signals={}, candlestick_patterns=[], pullback_analysis=None, investor_flow=None,
            ml_prediction=None, regime=regime, pct_change=0.0, volume_ratio=1.0, candle_up=True,
            rsi=50.0, macd_gap=0.0, event_risk=None,
        )
        assert result["side_prob_corr"] == 0.0
        assert result["prob_up_corr"] + result["prob_down_corr"] == pytest.approx(100.0, abs=0.11)


def test_explicit_side_probability_is_preserved_when_the_caller_supplies_one():
    result = correlate_and_narrow(
        symbol="X", market="US", dd=_bearish_dd(), last_price=70.0, atr=2.0, score=60,
        prob_up_base=50.0, prob_down_base=30.0, target_price={"min_price": 73.0, "max_price": 78.0},
        signal_confidence={"confidence": 60, "signal": "BUY", "confidence_interval": {"spread": 16}},
        indicator_signals={}, candlestick_patterns=[], pullback_analysis=None, investor_flow=None,
        ml_prediction=None, regime="NEUTRAL", pct_change=0.0, volume_ratio=1.0, candle_up=True,
        rsi=50.0, macd_gap=0.0, event_risk=None,
    )
    assert 10.0 <= result["side_prob_corr"] <= 45.0
    assert result["prob_up_corr"] + result["prob_down_corr"] + result["side_prob_corr"] == pytest.approx(100.0, abs=0.2)


def test_causal_map_claims_volume_confirmation_only_when_volume_confirms():
    dd = _bearish_dd()
    last = dd["Close"][-1]
    pattern = {"direction": "상승", "pattern_status": "confirmed", "completion_score": 90,
               "completion_components": {"volume": 80}}
    flow = {"ok": True, "외국인": 100_000, "기관": 80_000}

    def run(volume_ratio):
        return correlate_and_narrow(
            symbol="005930.KS", market="KRX", dd=dd, last_price=last, atr=2.0, score=60,
            prob_up_base=55.0, prob_down_base=45.0, target_price={"min_price": last + 3, "max_price": last + 8},
            signal_confidence={"confidence": 60, "signal": "BUY", "confidence_interval": {"spread": 16}},
            indicator_signals={}, candlestick_patterns=[pattern], pullback_analysis=None, investor_flow=flow,
            ml_prediction=None, regime="NEUTRAL", pct_change=0.5, volume_ratio=volume_ratio, candle_up=True,
            rsi=55.0, macd_gap=0.1, event_risk=None,
        )["correlation"]["causal_map"]

    assert any("거래량 동반" in line for line in run(1.5))
    assert not any("거래량 동반" in line for line in run(1.15))   # 수급 1.1배↑이지만 1.2배 미만


# ── 점수 확정: 가산 보정 후 상한을 마지막에 적용 ──────────────────────────────────────────

STRONG_HYBRID = {"ncs": 80.0, "fws": 10.0, "regime": "SIDEWAYS"}
WEAK_HYBRID = {"ncs": 30.0, "fws": 70.0, "regime": "SIDEWAYS"}


def test_bear_cap_is_not_pierced_by_flow_and_ncs_bonuses():
    result = ix.finalize_rule_score(60, regime="BEAR", flow_adjust=5, hybrid=STRONG_HYBRID)
    assert result["score"] == 40                       # 예전: 40 → 수급+5 → NCS+5 = 50
    assert result["uncapped_score"] == 70
    assert result["binding_cap"] == "regime_bear"


def test_debt_cap_applies_after_additive_adjustments():
    result = ix.finalize_rule_score(60, debt_healthy=False, flow_adjust=3, hybrid=STRONG_HYBRID)
    assert result["score"] == 45 and result["binding_cap"] == "debt"


def test_lowest_applicable_cap_wins():
    result = ix.finalize_rule_score(
        90, regime="BEAR", debt_healthy=False, hybrid={"ncs": 50, "fws": 40, "regime": "BEARISH"})
    assert result["score"] == 40
    assert any("레짐 BEARISH" in note for note in result["notes"])


def test_additive_adjustments_still_apply_without_caps():
    assert ix.finalize_rule_score(55, flow_adjust=4, hybrid=STRONG_HYBRID)["score"] == 64
    assert ix.finalize_rule_score(55, flow_adjust=-9, hybrid=WEAK_HYBRID)["score"] == 40  # 수급 -5 한도, NCS -10
    assert ix.finalize_rule_score(55, flow_adjust=0, hybrid=None)["score"] == 55


def test_non_breakout_stock_is_not_penalized_as_weak():
    # NCS<40 은 '돌파권이 아님'일 뿐 약점이 아니다 — 약점(AUTO_NO)은 FWS>65 뿐이다(hybrid_signals.ncs_action).
    plain = {"ncs": 30.0, "fws": 40.0, "regime": "SIDEWAYS", "action": "CONDITIONAL"}
    result = ix.finalize_rule_score(55, hybrid=plain)
    assert result["score"] == 55 and result["notes"] == []
    fatal = {"ncs": 30.0, "fws": 70.0, "regime": "SIDEWAYS", "action": "AUTO_NO"}
    assert ix.finalize_rule_score(55, hybrid=fatal)["score"] == 45


def test_finalize_uses_the_library_action_classification():
    for ncs, fws in ((75.0, 20.0), (75.0, 40.0), (20.0, 70.0), (50.0, 66.0)):
        hybrid = {"ncs": ncs, "fws": fws, "regime": "SIDEWAYS"}
        expected = hs.ncs_action(ncs, fws)
        delta = ix.finalize_rule_score(55, hybrid=hybrid)["score"] - 55
        assert delta == {"AUTO_YES": 5, "AUTO_NO": -10, "CONDITIONAL": 0}[expected]


def test_hybrid_error_payload_is_ignored():
    result = ix.finalize_rule_score(55, hybrid={"error": "데이터 부족"})
    assert result["score"] == 55 and result["notes"] == []


# ── 학습 로그: 같은 종목·날짜의 중복 예측/결과를 한 번만 센다 ─────────────────────────────────

def _outcome(symbol, signal_date, zone="primary", **extra):
    row = {"type": "outcome", "market": "US", "execution_model": "band_overlap_stop_first_v2",
           "symbol": symbol, "signal_date": signal_date, "zone": zone, "prediction_id": f"{symbol}|{signal_date}|{extra.get('n', 0)}",
           "extra_drop": True, "stop_hit": False, "bounce_success": True}
    row.update(extra)
    return row


def test_dedupe_learning_outcomes_keeps_first_per_symbol_day_zone():
    rows = [_outcome("AAPL", "2026-09-08", n=i) for i in range(10)]
    rows += [_outcome("AAPL", "2026-09-08", zone="secondary"), _outcome("TSLA", "2026-09-08")]
    unique = ix._dedupe_learning_outcomes(rows)
    assert len(unique) == 3
    assert unique[0]["prediction_id"].endswith("|0")


def test_dedupe_does_not_merge_rows_without_symbol_or_date():
    legacy = [{"type": "outcome", "zone": "primary"}, {"type": "outcome", "zone": "primary"}]
    assert len(ix._dedupe_learning_outcomes(legacy)) == 2


def test_learning_adjustment_counts_unique_outcomes_not_refreshes(monkeypatch):
    # 20행이지만 AAPL 같은 날 10건이 새로고침 중복 — 고유 표본은 11건이라 보정을 적용하면 안 된다.
    rows = [_outcome("AAPL", "2026-09-08", n=i) for i in range(10)]
    rows += [_outcome(f"T{i}", "2026-09-09") for i in range(10)]
    monkeypatch.setattr(ix, "_read_prediction_learning_events", lambda: rows)
    result = ix.calc_learning_adjustment("US")
    assert result["applied"] is False and result["sample_n"] == 0
    unique_rows = rows + [_outcome(f"U{i}", "2026-09-10") for i in range(10)]
    monkeypatch.setattr(ix, "_read_prediction_learning_events", lambda: unique_rows)
    assert ix.calc_learning_adjustment("US")["sample_n"] == 21


def _history(n=40, start="2026-08-01"):
    dates = pd.bdate_range(start=start, periods=n).strftime("%Y-%m-%d").tolist()
    closes = [100 + i * 0.1 for i in range(n)]
    return {"Date": dates, "Close": closes, "High": [c + 1 for c in closes], "Low": [c - 1 for c in closes],
            "Open": closes, "Volume": [1_000_000.0] * n}


def test_prediction_is_recorded_once_per_symbol_day_even_if_price_moves(monkeypatch):
    appended = []
    stored = []
    monkeypatch.setattr(ix, "_read_prediction_learning_events", lambda: list(stored))
    monkeypatch.setattr(ix, "_append_prediction_learning_event", lambda row: (appended.append(row), stored.append(row)))
    dd = _history()
    buy = lambda price: {"current": price, "atr_raw": 2.0, "aggressive_bands": [{"band": "A", "range": [98.0, 99.0]}],
                         "recommended_bands": [{"band": "B", "range": [96.0, 97.5]}], "downside_risk": {"score": 40}}
    risk = {"downside_risk_level": "medium", "downside_risk_score": 40}
    for price in (103.9, 104.2, 104.6):   # 장중 새로고침: 현재가만 달라지고 신호일은 같다
        ix._record_prediction_and_update_outcomes("AAPL", "US", "1y", dd, buy(price), risk, {"score": 0}, {})
    predictions = [row for row in appended if row["type"] == "prediction"]
    assert len(predictions) == 1
    assert predictions[0]["signal_date"] == dd["Date"][-1]


# ── 벤치마크: 1행만 주는 심볼(^KS200)에 의존하지 않는다 ──────────────────────────────────────

class _FakeTicker:
    frames = {}

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, period="1y", **_):
        return self.frames.get(self.symbol, pd.DataFrame())


def _frame(rows):
    idx = pd.bdate_range("2026-01-01", periods=rows)
    return pd.DataFrame({"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": np.linspace(100, 110, rows), "Volume": 1.0}, index=idx)


def test_benchmark_history_skips_symbols_with_too_few_rows(monkeypatch):
    monkeypatch.setattr(ix.yf, "Ticker", _FakeTicker)
    _FakeTicker.frames = {"^KS11": _frame(1), "069500.KS": _frame(120)}   # 1행(^KS200 증상) 후 ETF 대체
    df, symbol, label = ix._benchmark_history("KRX")
    assert symbol == "069500.KS" and len(df) == 120 and "KODEX" in label
    _FakeTicker.frames = {"^KS11": _frame(200), "069500.KS": _frame(120)}
    assert ix._benchmark_history("KRX")[1] == "^KS11"
    _FakeTicker.frames = {}
    empty, symbol, _label = ix._benchmark_history("KRX")
    assert empty.empty and symbol == ""


def test_krx_sentiment_uses_working_benchmark_and_ignores_nan_last_row(monkeypatch):
    monkeypatch.setattr(ix.yf, "Ticker", _FakeTicker)
    frame = _frame(25)
    frame.iloc[-1, frame.columns.get_loc("Close")] = np.nan      # 장 시작 전 NaN 행
    _FakeTicker.frames = {"^KS11": frame}
    result = ix.fetch_sentiment.__wrapped__("KRX")
    assert result is not None and result["name"] == "KOSPI"
    assert np.isfinite(result["value"]) and np.isfinite(result["change"])


# ── 시장 레짐: 마지막 행 NaN·조회 실패가 NEUTRAL 로 굳지 않는다 ─────────────────────────────────

def _index_frame(closes):
    idx = pd.bdate_range("2025-10-01", periods=len(closes))
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1.0}, index=idx)


def _downtrend_closes():
    # 앞 130봉 상승 후 120봉 하락: 종가 < MA120 이고 MA60 < MA120 인 약세 구조
    return list(np.linspace(100, 130, 130)) + list(np.linspace(130, 95, 120))


def test_index_regime_detects_bear_even_when_last_row_close_is_nan(monkeypatch):
    closes = _downtrend_closes()
    frame = _index_frame(closes + [np.nan])      # 장 시작 전 Yahoo 가 붙이는 NaN 행
    monkeypatch.setattr(ix.yf, "Ticker", lambda symbol: type("T", (), {"history": lambda self, **_: frame})())
    assert ix._index_regime_cached.__wrapped__("^KS11") == "BEAR"   # 예전: 비교가 전부 False → NEUTRAL
    assert ix._index_regime("^KS11-nan-row") in {"BEAR", "NEUTRAL"}


def test_index_regime_failure_is_not_cached_as_neutral(monkeypatch):
    ix._CACHE.clear()
    calls = {"n": 0}

    def flaky(self, **_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("temporary yahoo failure")
        return _index_frame(_downtrend_closes())

    monkeypatch.setattr(ix.yf, "Ticker", lambda symbol: type("T", (), {"history": flaky})())
    assert ix._index_regime("^TESTFAIL") == "NEUTRAL"      # 실패는 중립으로 보되
    assert ix._index_regime("^TESTFAIL") == "BEAR"         # 다음 요청이 15분 동안 NEUTRAL 에 고착되지 않는다
    assert calls["n"] == 2


def test_index_regime_with_insufficient_history_is_not_cached(monkeypatch):
    ix._CACHE.clear()
    monkeypatch.setattr(ix.yf, "Ticker", lambda symbol: type("T", (), {"history": lambda self, **_: _index_frame([100.0] * 50)})())
    assert ix._index_regime_cached.__wrapped__("^TESTSHORT") is None
    assert ix._index_regime("^TESTSHORT") == "NEUTRAL"


# ── 목표가 도달 확률: 가점식이 아니라 변동성 터치 확률 ───────────────────────────────────────────

def _indicator_dd(seed=3, n=252, drift=0.0004):
    frame = ix.add_indicators(_random_walk_ohlc(n=n + 80, seed=seed, drift=drift), market="US").iloc[-n:]
    dd = {col: [float(v) if np.isfinite(v) else None for v in frame[col].tolist()] for col in frame.columns}
    dd["Date"] = pd.bdate_range(end="2026-09-30", periods=n).strftime("%Y-%m-%d").tolist()
    return dd


def test_target_reach_probability_is_the_volatility_touch_probability():
    from market_briefing import forecast_model as fm

    dd = _indicator_dd()
    last, atr = dd["Close"][-1], dd["ATR"][-1]
    target = ix.calc_target_price(dd, last, atr, "1mo", "US")
    sigma = fm.blended_daily_sigma(dd["Close"], last, atr, True)["sigma"]
    expected = fm.touch_probability(last, target["min_price"], sigma, 22) * 100.0
    assert target["reach_probability_basis"] == "touch_probability"
    assert target["reach_probability"] == pytest.approx(expected, abs=0.11)
    assert target["reach_probability_inputs"]["horizon_days"] == 22


def test_target_reach_probability_falls_as_the_level_gets_farther():
    from market_briefing import forecast_model as fm

    points = []
    for seed in range(1, 25):
        dd = _indicator_dd(seed=seed, drift=0.0008 if seed % 3 else -0.0004)
        last, atr = dd["Close"][-1], dd["ATR"][-1]
        target = ix.calc_target_price(dd, last, atr, "1mo", "US")
        sigma = fm.blended_daily_sigma(dd["Close"], last, atr, True)["sigma"]
        z = np.log(target["min_price"] / last) / (sigma * np.sqrt(22))
        points.append((z, target["reach_probability"]))
    points.sort()
    reach = [p for _, p in points]
    # 예전 가점식은 추세가 강할수록(= 목표가 멀수록) 확률이 올라 이 순서가 거꾸로였다.
    assert all(later <= earlier + 0.11 for earlier, later in zip(reach, reach[1:]))
    assert reach[0] > reach[-1]


def test_normalize_target_output_recomputes_reach_for_the_final_range():
    dd = _indicator_dd(seed=5)
    last, atr = dd["Close"][-1], dd["ATR"][-1]
    target = ix._normalize_target_output(ix.calc_target_price(dd, last, atr, "1mo", "US"), last, "US")
    before = target["reach_probability"]
    moved = dict(target)
    moved["min_price"] = round(target["min_price"] * 1.04, 2)     # 보정으로 하단이 4% 올라간 경우
    moved["max_price"] = round(target["max_price"] * 1.04, 2)
    after = ix._normalize_target_output(moved, last, "US")["reach_probability"]
    assert after < before


def test_confidence_and_learning_do_not_perturb_a_calibrated_reach_probability():
    dd = _indicator_dd(seed=9)
    last, atr = dd["Close"][-1], dd["ATR"][-1]
    base = ix.calc_target_price(dd, last, atr, "1mo", "US")
    boosted = ix._apply_signal_confidence_to_target(
        dict(base), {"confidence": 85, "signal": "SELL", "confidence_interval": {"spread": 10}}, last, atr, "US")
    assert boosted["confidence_adjustment"]["probability_adj_pp"] == 0.0
    learned = ix._apply_learning_adjustment_to_target(
        dict(base), {"sample_n": 40, "extra_drop_rate": 80.0, "stop_hit_rate": 70.0, "bounce_success_rate": 20.0})
    assert learned["reach_probability"] == base["reach_probability"]
    legacy = dict(base, reach_probability_basis="heuristic_fallback")
    adjusted = ix._apply_signal_confidence_to_target(
        legacy, {"confidence": 85, "signal": "BUY", "confidence_interval": {"spread": 10}}, last, atr, "US")
    assert adjusted["confidence_adjustment"]["probability_adj_pp"] != 0.0   # 폴백 경로는 기존 동작 유지


def test_range_narrowing_is_off_by_default_and_switchable(monkeypatch):
    import market_briefing.correlation_engine as engine

    assert engine.RANGE_NARROWING is False          # 감사 결과: 일치도는 범위 포함률에 정보가 없다
    assert engine.PROBABILITY_PULL is True

    def narrowed_factor():
        dd = _bearish_dd()
        last = dd["Close"][-1]
        return correlate_and_narrow(
            symbol="X", market="US", dd=dd, last_price=last, atr=2.0, score=25,
            prob_up_base=40.0, prob_down_base=60.0, target_price={"min_price": last + 3, "max_price": last + 8},
            signal_confidence={"confidence": 78, "signal": "SELL", "confidence_interval": {"spread": 16}},
            indicator_signals={}, candlestick_patterns=[], pullback_analysis=None, investor_flow=None,
            ml_prediction=None, regime="BEAR", pct_change=-1.0, volume_ratio=1.0, candle_up=False,
            rsi=30.0, macd_gap=-0.5, event_risk=None,
        )["target_narrowed"]

    default = narrowed_factor()
    assert default["factor"] == 1.0 and default["narrow_width"] == pytest.approx(default["orig_width"], abs=0.02)
    monkeypatch.setattr(engine, "RANGE_NARROWING", True)
    enabled = narrowed_factor()
    assert enabled["factor"] < 1.0 and enabled["narrow_width"] < enabled["orig_width"]


def test_probability_pull_switch_leaves_base_probability_untouched(monkeypatch):
    import market_briefing.correlation_engine as engine

    monkeypatch.setattr(engine, "PROBABILITY_PULL", False)
    result = _correlate(25, "SELL")
    assert result["prob_up_corr"] == 40.0 and result["prob_down_corr"] == 60.0


def test_route_uses_the_single_score_finalizer_and_no_inline_caps():
    import inspect

    source = inspect.getsource(ix.route)
    assert "finalize_rule_score(" in source
    # 상한을 가산 앞에서 적용하던 인라인 코드와 route 고유 'NCS<40' 감점이 다시 생기지 않게 한다.
    assert "score = min(score, 40)" not in source
    assert "score = min(score, 45)" not in source
    assert "ncs_v < 40" not in source
    hybrid_pos = source.index("enrich_with_hybrid(")
    finalize_pos = source.index("finalize_rule_score(")
    sync_pos = source.index("_sync_ai_strategy_summary(ai_strategy, score")
    probability_pos = source.index("prob_up, prob_down = calc_probability(score")
    assert hybrid_pos < finalize_pos < sync_pos < probability_pos
