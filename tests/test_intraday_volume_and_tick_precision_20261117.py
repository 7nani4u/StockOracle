"""2026-11-17 감사 회귀 테스트 — 장중 진행 중 봉의 거래량 누출과 float32 호가 반올림.

1. 장중(마지막 일봉 진행 중)에는 같은 가격이라도 누적 거래량만으로 예측 탭 값이 바뀌면 안 된다.
   감사 재현(52종목, 누적 30% 대 100%)에서 시각 필드를 뺀 338개 출력 경로가 달라졌다: 돌파 확률 최대 29.8%p,
   1차 단계 가격 최대 9.7%, 하락 위험 점수 ±8점, 횡보 시나리오 +3%p.
2. 야후 일봉은 float32 에서 온 값이라(17.07 → 17.06999969482422) floor 반올림에서 한 호가가 빠졌다
   (미국 OHLC 25.6%). SOFI 보수적 TP1 범위가 현재가에서 시작하던 원인.

네트워크는 전송 계층에서 막고, 가격 이력은 가짜 yf.Ticker 로 공급해 실제 fetch_stock_data → add_indicators
→ route 경로를 그대로 탄다.
"""

import math

import numpy as np
import pandas as pd
import pytest
import requests

from api import index as ix
from market_briefing import box_plan
from market_briefing.session_bars import settle_in_progress_volume


def _ohlcv(n=520, seed=11, last_volume_scale=1.0):
    rng = np.random.default_rng(seed)
    close = 40.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.007, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.007, n)))
    volume = rng.integers(2_000_000, 6_000_000, n).astype(float)
    volume[-1] *= last_volume_scale
    idx = pd.date_range("2024-06-03", periods=n, freq="B", tz="America/New_York")
    frame = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
    # 야후처럼 센트 단위 가격을 float32 정밀도로 저장한 값(17.07 → 17.06999969482422)
    for col in ("Open", "High", "Low", "Close"):
        frame[col] = frame[col].round(2).astype(np.float32).astype(float)
    return frame


def _dd_from_frame(frame):
    df = ix.add_indicators(frame.copy(), market="US")
    out = df.reset_index().rename(columns={"index": "Date"})
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    d = out.where(pd.notna(out), other=None).to_dict(orient="list")
    for key, values in d.items():
        d[key] = [None if isinstance(v, float) and not math.isfinite(v) else v for v in values]
    return d


# ── 1. settle_in_progress_volume: 원본 불변·평균 대입·OBV/매수 압력 공식 일치 ────────────────

def test_settle_replaces_only_analysis_copy_and_matches_add_indicators():
    frame = _ohlcv(last_volume_scale=0.3)
    dd = _dd_from_frame(frame)
    original_volume = list(dd["Volume"])
    settled, meta = settle_in_progress_volume(dd)

    assert meta["applied"] is True and meta["basis"] == "previous_completed_bar"
    assert dd["Volume"] == original_volume                    # 캐시 공유 원본은 그대로
    assert settled["Volume"][-1] == original_volume[-2]       # 직전 확정 봉 거래량(다른 네 소비자와 같은 기준)
    assert settled["Volume"][:-1] == original_volume[:-1]
    mean_completed = float(np.mean(original_volume[-21:-1]))
    assert meta["live_volume"] == pytest.approx(original_volume[-1])
    assert meta["live_ratio"] == pytest.approx(original_volume[-1] / mean_completed)
    assert meta["completed_ratio"] == pytest.approx(original_volume[-2] / float(np.mean(original_volume[-22:-2])))

    # 같은 정착 거래량으로 add_indicators 를 처음부터 다시 돌린 값과 마지막 OBV·매수 압력이 같아야 한다.
    rebuilt = frame.copy()
    rebuilt.iloc[-1, rebuilt.columns.get_loc("Volume")] = settled["Volume"][-1]
    ref = _dd_from_frame(rebuilt)
    assert settled["OBV"][-1] == pytest.approx(ref["OBV"][-1], rel=1e-12)
    assert settled["BUY_PRESSURE"][-1] == pytest.approx(ref["BUY_PRESSURE"][-1], rel=1e-12)
    assert settled["OBV"][:-1] == dd["OBV"][:-1]


def test_settle_falls_back_to_completed_mean_when_previous_bar_has_no_volume():
    """직전 봉이 거래정지 등으로 0이면 0을 이어 쓰지 않고 직전 20개 확정 봉 평균을 쓴다."""
    dd = _dd_from_frame(_ohlcv())
    dd["Volume"] = list(dd["Volume"])
    dd["Volume"][-2] = 0.0
    settled, meta = settle_in_progress_volume(dd)
    assert meta["applied"] is True and meta["basis"] == "completed_bar_mean"
    assert settled["Volume"][-1] == pytest.approx(round(float(np.mean(dd["Volume"][-21:-1]))))


def test_settle_skips_short_or_misaligned_history():
    short = {"Close": [1.0] * 10, "Volume": [100] * 10}
    assert settle_in_progress_volume(short)[1]["applied"] is False
    misaligned = {"Close": [1.0] * 30, "Volume": [100] * 29}
    out, meta = settle_in_progress_volume(misaligned)
    assert meta["applied"] is False and out is misaligned


# ── 2. route 수준: 진행 중 봉이면 누적 거래량과 무관한 예측 탭 ────────────────────────────

class _FakeTicker:
    frame = None

    def __init__(self, sym, *a, **k):
        self.ticker = sym
        self.info = {}
        self.calendar = None
        self.news = []
        self.fast_info = type("FI", (), {})()

    def history(self, period="1y", interval="1d", auto_adjust=True, **_):
        if interval != "1d" or self.ticker != "ZZTEST":
            return pd.DataFrame()
        return _FakeTicker.frame.copy()


def _offline(monkeypatch):
    def _no_net(*_a, **_k):
        raise requests.ConnectionError("offline test")

    for name in ("get", "post", "put"):
        monkeypatch.setattr(requests, name, _no_net)
    monkeypatch.setattr(requests.Session, "request", _no_net)
    monkeypatch.setattr(ix.yf, "Ticker", _FakeTicker)
    monkeypatch.setattr(ix.yf, "download", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(ix, "resolve_ticker", lambda raw: ("ZZTEST", "US", "ZZ Test Corp"))
    monkeypatch.setattr(ix, "_get_us_price_fetcher", lambda: None)
    monkeypatch.setattr(ix, "_bar_in_progress", lambda *a, **k: True)
    try:
        import feedparser
        monkeypatch.setattr(feedparser, "parse", lambda *a, **k: type("F", (), {"entries": []})())
    except ImportError:
        pass


def _route_with_last_volume(monkeypatch, scale):
    _FakeTicker.frame = _ohlcv(last_volume_scale=scale)
    ix._CACHE.clear()
    return ix.route("/api/stock", {"ticker": "ZZTEST"})


def _key_outputs(resp):
    risk = resp["risk_scenarios"]
    buy = resp["buy_price"]
    fc = resp["prediction_outlook"]["forecast"]
    return {
        "breakout": risk.get("breakout_probability_pct"),
        "downside_risk": (buy.get("downside_risk") or {}).get("score"),
        "steps": [[s.get("price") for s in band.get("steps") or []]
                  for key in ("aggressive_bands", "recommended_bands") for band in buy.get(key) or []],
        "tp_probs": [[tp.get("prob_pct") for tp in (risk.get(k) or {}).get("tp_levels") or []]
                     for k in ("conservative", "balanced", "aggressive")],
        "scenario": fc.get("scenario_probabilities"),
        "range": fc.get("range_p10_p90"),
        "buy_pressure": ((resp.get("indicator_signals") or {}).get("signals") or {}).get("buy_pressure", {}).get("value"),
    }


def test_in_progress_bar_volume_does_not_change_prediction_tab(monkeypatch, tmp_path):
    monkeypatch.setenv("STOCKORACLE_PREDICTION_LOG", str(tmp_path / "learning.jsonl"))
    monkeypatch.delenv("STOCKORACLE_INPROGRESS_VOLUME_SETTLE", raising=False)
    _offline(monkeypatch)

    morning = _route_with_last_volume(monkeypatch, 0.3)
    close = _route_with_last_volume(monkeypatch, 1.0)
    assert "error" not in morning and "error" not in close
    assert _key_outputs(morning) == _key_outputs(close)

    # 화면의 거래량·차트는 실제 누적값을 그대로 보여 준다.
    assert morning["volume"] < close["volume"]
    assert morning["chart_data"]["volume"][-1] == pytest.approx(morning["volume"], rel=1e-9)
    assert morning["data_quality"]["volume_settle"]["applied"] is True
    status = {s["key"]: s for s in morning["prediction_outlook"]["status"]}
    assert status["volume"]["value"] == "장중 집계 중"
    assert "판단에 쓰지 않음" in status["volume"]["detail"]


def test_settle_switch_off_restores_previous_time_of_day_dependence(monkeypatch, tmp_path):
    """스위치를 끄면 예전 동작(누적값을 그대로 읽음)으로 돌아가며, 그때는 값이 달라진다 — 위 테스트의 민감도 확인."""
    monkeypatch.setenv("STOCKORACLE_PREDICTION_LOG", str(tmp_path / "learning.jsonl"))
    monkeypatch.setenv("STOCKORACLE_INPROGRESS_VOLUME_SETTLE", "0")
    _offline(monkeypatch)

    morning = _route_with_last_volume(monkeypatch, 0.3)
    close = _route_with_last_volume(monkeypatch, 1.0)
    assert _key_outputs(morning) != _key_outputs(close)
    assert morning["data_quality"]["volume_settle"]["applied"] is False


# ── 3. float32 가격의 호가 반올림 ─────────────────────────────────────────────────────

def test_float32_prices_do_not_lose_a_tick_on_floor_or_ceil():
    p32 = float(np.float32(17.07))                     # 17.06999969482422
    assert ix._round_market_price(p32, "US", "floor") == 17.07
    assert ix._round_market_price(p32 + 0.01, "US", "floor") == 17.08
    above = float(np.float32(17.33))                    # 17.329999923706055 (아래쪽 오차)
    assert ix._round_market_price(above, "US", "ceil") == 17.33
    assert box_plan.round_tick(p32, "US", "floor") == pytest.approx(17.07)
    assert box_plan.round_tick(above, "US", "ceil") == pytest.approx(17.33)
    # 실제로 호가 사이에 있는 값은 그대로 내림/올림
    assert ix._round_market_price(17.074, "US", "floor") == 17.07
    assert ix._round_market_price(17.074, "US", "ceil") == 17.08
    assert ix._round_market_price(4.26, "US", "floor") == 4.26          # 기존 float64 오차 사례 유지


def _volatile_cent_ohlcv(seed, vol=0.035, n=520):
    """ATR 이 가격의 4~6%인 저가 성장주 모양(SOFI 사례). 보수적 TP1 하한이 '현재가+1호가'에 묶이는 조건."""
    rng = np.random.default_rng(seed)
    close = 17.0 * np.exp(np.cumsum(rng.normal(0.0, vol, n)))
    open_ = close * (1 + rng.normal(0, vol / 4, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.integers(2_000_000, 6_000_000, n).astype(float)
    idx = pd.date_range("2024-06-03", periods=n, freq="B", tz="America/New_York")
    frame = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
    for col in ("Open", "High", "Low", "Close"):
        frame[col] = frame[col].round(2).astype(np.float32).astype(float)
    return frame


@pytest.mark.parametrize("seed", [5, 17, 38])   # 1e-9 허용 오차에서는 세 경우 모두 TP1 하한 = 현재가
def test_tp_ranges_start_above_current_price_with_float32_history(seed):
    dd = _dd_from_frame(_volatile_cent_ohlcv(seed))
    price = float(dd["Close"][-1])
    atr = float(dd["ATR"][-1])
    risk = ix.calc_risk(price, atr, "US", dd, {"score": 0, "level": "low", "reasons": [], "days_to_earnings": None}, None)
    for key in ("conservative", "balanced", "aggressive"):
        for level in risk[key]["tp_levels"]:
            low, high = level["price_range"]
            assert low > round(price, 2), (key, level["price_range"], price)
            assert low <= level["price"] <= high
