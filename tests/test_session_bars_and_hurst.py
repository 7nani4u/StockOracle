"""허스트 지수와 장중 진행 중 막대 거래량 처리의 회귀 테스트.

2026-10-09 감사에서 52종목·캐시 일봉으로 확인된 결함 두 가지를 고정한다.
  1) `hybrid_signals.calc_hurst` 가 가격 '수준'에 R/S 를 적용해 무엇을 넣어도 0.88~0.93 이 나왔다
     (BQS 허스트 보너스 +8 상수, 화면의 '추세 지속 가능성 높음' 상시 표시).
  2) 장중 일봉 마지막 막대의 거래량(누적 중, 최종값 이하)을 20일 평균과 바로 비교해,
     같은 종목의 NCS 가 조회 시각만으로 평균 -7.5~-11점 흔들렸다.
네트워크 없이 합성 데이터만 쓴다.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from api import index as ix
from market_briefing import hybrid_signals as hs
from market_briefing import scan_engine
from market_briefing.dual_score_v2 import calc_hurst_v2
from market_briefing.session_bars import (
    completed_volume_ratio,
    last_bar_in_progress,
    market_session_in_progress,
    regular_session_open,
)

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")


def _autocorrelated_prices(n=252, phi=0.0, seed=0, vol=0.02):
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    shocks = rng.normal(0, vol, n)
    for i in range(1, n):
        returns[i] = phi * returns[i - 1] + shocks[i]
    return list(100 * np.exp(np.cumsum(returns)))


def _ohlcv(n=300, seed=5, volume=1_000_000.0):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vols = volume * (1 + rng.normal(0, 0.15, n)).clip(0.4, None)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vols})


# ── 허스트 지수 ────────────────────────────────────────────────────────────────

def test_hybrid_hurst_separates_mean_reverting_random_and_trending_returns():
    def mean_hurst(phi):
        values = [hs.calc_hurst(_autocorrelated_prices(phi=phi, seed=seed)) for seed in range(40)]
        return float(np.mean([v for v in values if v is not None]))

    mean_reverting, random_walk, trending = mean_hurst(-0.3), mean_hurst(0.0), mean_hurst(0.3)
    assert mean_reverting < random_walk < trending
    # 가격 수준에 R/S 를 적용하던 구현은 세 경우 모두 0.88~0.93 이었다.
    assert random_walk < 0.7
    assert trending - mean_reverting > 0.08


def test_hybrid_hurst_is_the_same_implementation_the_scan_engine_uses():
    prices = _autocorrelated_prices(phi=0.1, seed=3)
    assert hs.calc_hurst(prices) == pytest.approx(calc_hurst_v2(prices), abs=1e-4)
    assert hs.calc_hurst(prices[:45]) is None          # 50봉 미만은 산정하지 않는다


def test_hybrid_score_hurst_bonus_is_not_a_constant_for_random_walks():
    bonuses = set()
    for seed in range(12):
        df = _ohlcv(seed=seed)
        result = hs.compute_hybrid_score(
            closes=df["Close"].tolist(), highs=df["High"].tolist(), lows=df["Low"].tolist(),
            volumes=df["Volume"].tolist(), open_prices=df["Open"].tolist(),
        )
        assert result["hurst"] is not None and result["hurst"] < 0.75
        bonuses.add(8.0 if result["hurst"] >= 0.7 else 5.0 if result["hurst"] >= 0.6 else 2.0 if result["hurst"] >= 0.5 else 0.0)
    assert bonuses != {8.0}


# ── 진행 중 막대 판정 ──────────────────────────────────────────────────────────

def test_last_bar_in_progress_follows_exchange_clock_and_bar_date():
    wed_open_kst = datetime(2026, 10, 7, 10, 30, tzinfo=KST)
    assert last_bar_in_progress("KRX", "2026-10-07", wed_open_kst) is True
    # 같은 시각이라도 마지막 막대가 어제 것이면(공휴일·미수신) 이미 확정된 막대다.
    assert last_bar_in_progress("KRX", "2026-10-06", wed_open_kst) is False
    # 마감 후(공급자 지연 20분 이후)와 장 시작 전, 주말은 모두 확정 막대.
    assert last_bar_in_progress("KRX", "2026-10-07", datetime(2026, 10, 7, 16, 0, tzinfo=KST)) is False
    assert last_bar_in_progress("KRX", "2026-10-07", datetime(2026, 10, 7, 8, 30, tzinfo=KST)) is False
    assert last_bar_in_progress("KRX", "2026-10-10", datetime(2026, 10, 10, 10, 30, tzinfo=KST)) is False
    # 미국은 동부시간 기준이며 한국 시계의 밤 시간이 정규장이다.
    assert last_bar_in_progress("US", "2026-10-07", datetime(2026, 10, 7, 11, 0, tzinfo=ET)) is True
    assert last_bar_in_progress("US", "2026-10-07", datetime(2026, 10, 7, 11, 0, tzinfo=ET).astimezone(KST)) is True
    assert last_bar_in_progress("US", "2026-10-07", datetime(2026, 10, 7, 17, 0, tzinfo=ET)) is False


def test_market_level_session_check_skips_holidays_via_callback():
    open_wed = datetime(2026, 10, 7, 10, 30, tzinfo=KST)
    assert market_session_in_progress("KRX", open_wed) is True
    assert market_session_in_progress("KRX", open_wed, is_trading_day=lambda day: True) is True
    assert market_session_in_progress("KRX", open_wed, is_trading_day=lambda day: False) is False
    assert regular_session_open("KRX", datetime(2026, 10, 7, 15, 45, tzinfo=KST)) is True   # 마감 직후 지연 허용
    assert regular_session_open("KRX", datetime(2026, 10, 7, 16, 0, tzinfo=KST)) is False


# ── 거래량 비율 ────────────────────────────────────────────────────────────────

def test_completed_volume_ratio_keeps_the_legacy_formula_for_confirmed_bars():
    volumes = [1000.0 + 10 * i for i in range(40)]
    legacy = volumes[-1] / (sum(volumes[-21:-1]) / 20)
    ratio, basis = completed_volume_ratio(volumes, in_progress=False)
    assert basis == "live_bar" and ratio == pytest.approx(legacy)


def test_in_progress_volume_ratio_uses_previous_completed_bar_and_ignores_partial_volume():
    base = [1000.0 + 10 * i for i in range(40)]
    partial_low, partial_high = base[:-1] + [50.0], base[:-1] + [900.0]
    low, basis = completed_volume_ratio(partial_low, in_progress=True)
    high, _ = completed_volume_ratio(partial_high, in_progress=True)
    assert basis == "completed_bar"
    assert low == pytest.approx(high)                                   # 누적 중인 마지막 값에 의존하지 않는다
    assert low == pytest.approx(base[-2] / (sum(base[-22:-2]) / 20))
    assert completed_volume_ratio([1.0] * 10, in_progress=True)[0] is None  # 표본 부족


def test_hybrid_fws_and_ncs_do_not_depend_on_time_of_day_when_bar_is_in_progress():
    df = _ohlcv(seed=11)
    args = dict(closes=df["Close"].tolist(), highs=df["High"].tolist(), lows=df["Low"].tolist(),
                open_prices=df["Open"].tolist())
    full_day = df["Volume"].tolist()
    early = full_day[:-1] + [full_day[-1] * 0.25]                       # 장 초반의 누적 거래량
    late = full_day[:-1] + [full_day[-1] * 0.80]

    old_early = hs.compute_hybrid_score(volumes=early, **args)
    new_early = hs.compute_hybrid_score(volumes=early, volume_in_progress=True, **args)
    new_late = hs.compute_hybrid_score(volumes=late, volume_in_progress=True, **args)

    assert new_early["volume_basis"] == "completed_bar" and new_early["volume_in_progress"] is True
    assert new_early["fws"] == pytest.approx(new_late["fws"])           # 시각이 점수를 바꾸지 않는다
    assert new_early["ncs"] == pytest.approx(new_late["ncs"], abs=1.0)  # BIS 캔들 항목만 미세 차이
    assert old_early["fws"] > new_early["fws"]                          # 예전 계산은 '거래량 위축'으로 FWS 를 키웠다
    # 확정 막대에서는 예전과 같은 결과.
    confirmed = hs.compute_hybrid_score(volumes=full_day, **args)
    assert confirmed["volume_basis"] == "live_bar"


def test_scan_engine_volume_ratio_uses_the_same_basis():
    volumes = [1000.0 + 5 * i for i in range(40)]
    assert scan_engine._calc_vol_ratio(volumes) == pytest.approx(round(volumes[-1] / (sum(volumes[-21:-1]) / 20), 3))
    partial = volumes[:-1] + [10.0]
    assert scan_engine._calc_vol_ratio(partial, in_progress=True) == pytest.approx(
        scan_engine._calc_vol_ratio(volumes[:-1] + [99999.0], in_progress=True))
    assert scan_engine._calc_vol_ratio([1.0] * 5) == 1.0                # 표본 부족은 중립


# ── analyze_score 4단계 ────────────────────────────────────────────────────────

def _dd_from(df, market="US"):
    frame = ix.add_indicators(df.copy(), market=market)
    dd = {"Date": [f"2026-01-{(i % 28) + 1:02d}" for i in range(len(frame))]}
    for column in frame.columns:
        dd[column] = [float(v) if np.isfinite(v) else None for v in frame[column].to_numpy(float)]
    return dd


def _volume_step(steps):
    return next(step for step in steps if step["step"].startswith("4."))


def test_analyze_score_volume_step_uses_previous_confirmed_bar_while_session_is_open():
    df = _ohlcv(seed=2)
    avg = float(df["Volume"].iloc[-21:-1].mean())
    df.loc[df.index[-2], "Volume"] = avg * 3.0                          # 직전 확정 막대: 거래량 3배
    df.loc[df.index[-2], "Open"] = df["Close"].iloc[-2] * 0.97           # 양봉
    df.loc[df.index[-1], "Volume"] = avg * 0.2                          # 오늘 장 초반 누적 거래량
    dd = _dd_from(df)

    live = ix.analyze_score(dd, "US", "1y")
    progress = ix.analyze_score(dd, "US", "1y", volume_in_progress=True)

    live_step, progress_step = _volume_step(live[1]), _volume_step(progress[1])
    assert "급감" in live_step["result"] and live_step["score"] == 0     # 누적 중 값으로 '거래량 급감'이라 오판
    assert "직전 확정 봉 기준" in progress_step["result"]
    assert "양봉" in progress_step["result"] and progress_step["score"] > 0


def test_analyze_score_in_progress_branch_does_not_replace_the_live_close(monkeypatch):
    df = _ohlcv(seed=4)
    dd = _dd_from(df)
    seen = []
    original = ix._ai_strategy_summary_lines

    def spy(score, close, *args, **kwargs):
        seen.append(close)
        return original(score, close, *args, **kwargs)

    monkeypatch.setattr(ix, "_ai_strategy_summary_lines", spy)
    ix.analyze_score(dd, "US", "1y", volume_in_progress=True)
    assert seen and seen[-1] == pytest.approx(dd["Close"][-1])           # 4단계 이후 단계는 현재가를 그대로 쓴다


# ── ML 블렌드 가중: 선택 편향이 없는 검증 추정 중 최솟값 ───────────────────────────────

_ML_META = {
    "metrics": {"test_auc": 0.569},
    "selected_parameters": {"name": "directional_8feat"},
    "walk_forward_backtest": [
        {"candidate": "legacy", "mean_auc": 0.5457},
        {"candidate": "directional_8feat", "mean_auc": 0.5464},
    ],
    "calibration": {"params": {"identity": True, "oof_auc": 0.5146}},
}


def test_ml_blend_weight_uses_the_lowest_recorded_validation_auc():
    auc, basis, estimates = ix._ml_honest_auc(_ML_META)
    assert basis == "oof" and auc == pytest.approx(0.5146)
    assert estimates == {"holdout": 0.569, "walk_forward": 0.5464, "oof": 0.5146}
    assert ix._ml_blend_weight(auc) == 0.0            # 홀드아웃 0.569 로 정하면 15% 였다
    assert ix._ml_blend_weight(0.569) == 0.15


def test_ml_blend_weight_table_and_validation_gate():
    assert [ix._ml_blend_weight(a) for a in (0.50, 0.519, 0.52, 0.559, 0.56, 0.599, 0.60, 0.649, 0.65)] == [
        0.0, 0.0, 0.10, 0.10, 0.15, 0.15, 0.20, 0.20, 0.25]
    assert ix._ml_blend_weight(0.62, approved=False) == 0.0


def test_ml_honest_auc_ignores_other_candidates_and_handles_missing_records(monkeypatch):
    assert ix._ml_honest_auc({}) == (0.55, "default", {})
    only_holdout = ix._ml_honest_auc({"metrics": {"test_auc": 0.58}})
    assert only_holdout[:2] == (0.58, "holdout")
    # 선택되지 않은 후보의 워크포워드 값은 쓰지 않는다.
    meta = dict(_ML_META, selected_parameters={"name": "compact"})
    assert "walk_forward" not in ix._ml_honest_auc(meta)[2]
    monkeypatch.setattr(ix, "_ML_AUC_BASIS", "holdout")
    assert ix._ml_honest_auc(_ML_META)[:2] == (0.569, "holdout")


def test_shipped_model_metadata_would_not_blend_on_the_honest_estimate():
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "training_metadata.json")
    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle)
    auc, basis, estimates = ix._ml_honest_auc(meta)
    assert {"holdout", "walk_forward", "oof"} <= set(estimates)
    assert auc == min(estimates.values()) and basis == min(estimates, key=estimates.get)


def test_correlation_ml_vote_follows_the_validation_trust_ratio():
    from market_briefing.correlation_engine import correlate_and_narrow

    def ml_dimension(ml):
        closes = [100 + i * 0.1 for i in range(80)]
        dd = {"Close": closes, "High": [c + 1 for c in closes], "Low": [c - 1 for c in closes], "Volume": [1e6] * 80,
              "MA20": [None] * 19 + closes[19:], "MA60": [None] * 59 + closes[59:], "RSI": [55.0] * 80}
        report = correlate_and_narrow(
            symbol="X", market="US", dd=dd, last_price=closes[-1], atr=2.0, score=55.0, prob_up_base=55.0,
            prob_down_base=45.0, target_price={"min_price": closes[-1] + 3, "max_price": closes[-1] + 8},
            signal_confidence={"confidence": 60, "signal": "BUY", "confidence_interval": {"spread": 16}},
            indicator_signals={}, candlestick_patterns=[], pullback_analysis=None, investor_flow=None,
            ml_prediction=ml, regime="NEUTRAL", pct_change=0.5, volume_ratio=1.0, candle_up=True, rsi=55.0,
            macd_gap=0.1, event_risk=None)
        return report["correlation"]["dimensions"]["ml"]

    untouched = ml_dimension({"prob_up": 0.80})                       # 기록이 없으면 예전과 같다
    full = ml_dimension({"prob_up": 0.80, "trust": 1.0})
    half = ml_dimension({"prob_up": 0.80, "trust": 0.5})
    none = ml_dimension({"prob_up": 0.80, "trust": 0.0})
    assert untouched > 0 and full == pytest.approx(untouched)
    assert half == pytest.approx(untouched / 2)
    assert none == 0


# ── AI 진단 탭 최종 배지: 서버 점수에 이미 있는 지표를 브라우저가 다시 더하지 않는다 ───────────────

def _run_render_flow_tab(payload):
    import json
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for the browser renderer regression check")
    source = "function renderFlowTab" + ix.HTML.split("function renderFlowTab", 1)[1].split("// ═══", 1)[0]
    script = """
const vm = require('vm');
const elements = {};
const mk = (id) => (elements[id] ||= {innerHTML:'', textContent:'', className:'', style:{},
                                      dataset:{gradeColor:'#3fb950', gradeBg:'#0d2d1a'}});
const sandbox = { document: { getElementById(id) { return mk(id); } } };
vm.createContext(sandbox);
vm.runInContext(SOURCE, sandbox);
sandbox.renderFlowTab(DATA);
console.log(JSON.stringify({badge: elements['flow-rec-badge'].textContent}));
""".replace("SOURCE", json.dumps(source)).replace("DATA", json.dumps(payload))
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])["badge"]


def _flow_payload(score, weighted_score, rsi=25.0):
    return {
        "market": "US", "score": score, "rsi": rsi,
        "chart_data": {"close": [100 + i for i in range(30)], "ma20": [100 + i * 0.5 for i in range(30)]},
        "indicator_signals": {"summary": {"weighted_score": weighted_score, "buy": 8, "sell": 0, "watch": 1, "total": 9}},
        "us_enriched": {"sentiment": {"bullish_pct": 0.5}},
        "pullback_analysis": None,
        "prediction_outlook": {"decision": {"key": "watch"}},
    }


def test_flow_badge_does_not_restack_indicator_signals_that_the_server_score_already_contains():
    # 서버 점수 60(단기 반등 가능 구간). 지표 가중 점수 +80·MA20 상승·RSI 과매도는 점수에 이미 들어 있다.
    badge = _run_render_flow_tab(_flow_payload(score=60, weighted_score=80))
    assert "단기 반등 가능" in badge and "적극 매수" not in badge
    # 같은 점수에서 지표 가중 점수가 바뀌어도 최종 문구는 같다(신뢰도 문구만 지표 일치도를 따른다).
    weak = _run_render_flow_tab(_flow_payload(score=60, weighted_score=-80))
    assert "단기 반등 가능" in weak
    # 점수가 실제로 높으면 여전히 적극 매수로 올라간다.
    assert "적극 매수" in _run_render_flow_tab(_flow_payload(score=76, weighted_score=0, rsi=50.0))


def test_flow_badge_still_adds_news_sentiment_which_the_server_score_lacks():
    payload = _flow_payload(score=60, weighted_score=0, rsi=50.0)
    payload["us_enriched"] = {"sentiment": {"bullish_pct": 0.9}}        # 뉴스 긍정 → +4 → 64(매수 우위)
    assert "매수 우위" in _run_render_flow_tab(payload)


# ── 미국 장기 추천: 정의되지 않은 유니버스 참조 + 실패 응답 고착 + 새로고침 미연결 ───────────────────

def test_us_longterm_recommendation_no_longer_dies_on_an_undefined_universe(monkeypatch):
    assert isinstance(ix._US_GARP_UNIVERSE, list)           # 예전에는 이름 자체가 없어 호출마다 NameError 였다
    ix._CACHE.pop("fetch_us_longterm_reco|()|[]", None)
    monkeypatch.setattr(ix.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    result = ix.fetch_us_longterm_reco()
    assert "not defined" not in str(result.get("error"))
    assert result == {"error": "데이터 없음", "items": []}
    ix._CACHE.pop("fetch_us_longterm_reco|()|[]", None)


def test_ttl_cache_keeps_failure_payloads_only_briefly():
    calls = []

    @ix.ttl_cache(14400)
    def flaky_longterm_probe():
        calls.append(1)
        return {"error": "일시 장애", "items": []} if len(calls) == 1 else {"items": [1, 2, 3]}

    flaky_longterm_probe()
    key = next(k for k in ix._CACHE if k.startswith("flaky_longterm_probe|"))
    value, stored_at = ix._CACHE[key]
    # 4시간이 아니라 약 15초 뒤에 만료되도록 저장된다.
    remaining = 14400 - (ix.time.time() - stored_at)
    assert 0 < remaining <= 15
    ix._CACHE[key] = (value, stored_at - 20)                   # 15초가 지난 것으로 간주
    assert flaky_longterm_probe() == {"items": [1, 2, 3]}      # 회복된 결과가 다시 조회된다
    ok_key = next(k for k in ix._CACHE if k.startswith("flaky_longterm_probe|"))
    assert 14400 - (ix.time.time() - ix._CACHE[ok_key][1]) > 14000   # 정상 결과는 전체 TTL 로 캐시
    for stale in [k for k in ix._CACHE if k.startswith("flaky_longterm_probe|")]:
        ix._CACHE.pop(stale, None)


def test_longterm_and_us_surge_refresh_parameter_clears_the_cache(monkeypatch):
    seen = {}
    for name in ("fetch_kr_longterm_reco", "fetch_us_longterm_reco", "fetch_us_opening_surge"):
        monkeypatch.setitem(ix._CACHE, f"{name}|()|[]", ({"items": ["cached"]}, ix.time.time()))
    monkeypatch.setattr(ix, "fetch_kr_longterm_reco", lambda: seen.setdefault("kr", "fresh"))
    monkeypatch.setattr(ix, "fetch_us_longterm_reco", lambda: seen.setdefault("us", "fresh"))
    monkeypatch.setattr(ix, "fetch_us_opening_surge", lambda: seen.setdefault("surge", "fresh"))
    for path in ("/api/kr/longterm", "/api/us/longterm", "/api/us/opening-surge"):
        ix.route(path, {"refresh": "1"})
    assert not any(key in ix._CACHE for key in (
        "fetch_kr_longterm_reco|()|[]", "fetch_us_longterm_reco|()|[]", "fetch_us_opening_surge|()|[]"))
    assert seen == {"kr": "fresh", "us": "fresh", "surge": "fresh"}


def test_new_force_refresh_is_sent_for_us_surge_from_the_browser():
    assert "fetch('/api/us/opening-surge' + (force ? '?refresh=1' : ''))" in ix.HTML
