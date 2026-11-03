"""코스피·코스닥 주간 예측 — 달력·시점(누출)·원장·결측·화면 계약 테스트 (합성 데이터, 네트워크 미사용).

합성 시장: 한국 지수 일간 수익률 = 0.6 × (직전 국내 종가 이후 마감한 미국 세션 누적 수익률) + 잡음.
그래서 장 시작 전 모델은 미국 입력에 양의 기여를 학습해야 하고, 미국 세션이 미확정이면 쓰면 안 된다.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_briefing import index_series as ix  # noqa: E402
from market_briefing import kr_direction_model as km  # noqa: E402
from market_briefing import krx_calendar as kc  # noqa: E402
from market_briefing import weekly_forecast as wf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KST = kc.KST
try:
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    NY = dt.timezone(dt.timedelta(hours=-5))


def _at(y, m, d, hh, mm=0):
    return dt.datetime(y, m, d, hh, mm, tzinfo=KST)


# ── 합성 데이터 ─────────────────────────────────────────────────────────

def _make_bundle(seed: int = 3, end: dt.date = dt.date(2026, 11, 13)) -> dict:
    rng = np.random.default_rng(seed)
    start = dt.date(2021, 6, 1)
    kr_dates = kc.trading_days(start, end)
    us_dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    us_dates = [d for d in us_dates if d.weekday() < 5]
    us_ret = rng.normal(0.0004, 0.011, len(us_dates))
    us_close = 5000.0 * np.exp(np.cumsum(us_ret))
    us_lvl = {d: c for d, c in zip(us_dates, us_close)}
    us_sorted = np.asarray([np.datetime64(d.isoformat()) for d in us_dates])

    def us_level(x):
        p = int(np.searchsorted(us_sorted, np.datetime64(x.isoformat()), side="right")) - 1
        return us_close[p] if p >= 0 else us_close[0]

    def kr_series(beta, noise, base):
        closes = [base]
        for i in range(1, len(kr_dates)):
            a, t = kr_dates[i - 1], kr_dates[i]
            window = math.log(us_level(t - dt.timedelta(days=1)) / us_level(a - dt.timedelta(days=1)))
            closes.append(closes[-1] * math.exp(beta * window + rng.normal(0.0002, noise)))
        closes = np.asarray(closes)
        opens = closes * np.exp(rng.normal(0, 0.003, len(closes)))
        highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.004, len(closes))))
        lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.004, len(closes))))
        return {"dates": list(kr_dates), "open": opens.tolist(), "high": highs.tolist(),
                "low": lows.tolist(), "close": closes.tolist(),
                "volume": (300000 + rng.integers(0, 50000, len(closes))).astype(float).tolist(),
                "source": "합성"}

    walk = lambda scale, base: (base * np.exp(np.cumsum(rng.normal(0, scale, len(us_dates))))).tolist()
    globals_ = {
        "SPX": {"dates": list(us_dates), "close": us_close.tolist()},
        "NDX": {"dates": list(us_dates), "close": (us_close * 3.1).tolist()},
        "SOX": {"dates": list(us_dates), "close": walk(0.02, 4000.0)},
        "VIX": {"dates": list(us_dates), "close": walk(0.05, 18.0)},
        "TNX": {"dates": list(us_dates), "close": (4.0 + np.cumsum(rng.normal(0, 0.03, len(us_dates)))).tolist()},
    }
    return {"kr": {"KOSPI": kr_series(0.6, 0.008, 2500.0), "KOSDAQ": kr_series(0.5, 0.011, 800.0)},
            "globals": globals_, "quotes": {}, "errors": [], "stale": []}


BUNDLE = _make_bundle()


def _truncate(bundle: dict, now: dt.datetime, quotes: dict | None = None) -> dict:
    """now 시점에 실제로 볼 수 있는 모양: 국내 막대는 오늘까지(진행 중 포함), 미국은 뉴욕 날짜까지."""
    out = copy.deepcopy(bundle)
    kr_cut = now.date()
    us_cut = now.astimezone(NY).date()
    for series in out["kr"].values():
        n = sum(1 for d in series["dates"] if d <= kr_cut)
        for k, v in list(series.items()):
            if isinstance(v, list):
                series[k] = v[:n]
    for series in out["globals"].values():
        if not series:
            continue
        n = sum(1 for d in series["dates"] if d <= us_cut)
        for k, v in list(series.items()):
            if isinstance(v, list):
                series[k] = v[:n]
    out["quotes"] = quotes or {}
    return out


def _fetcher(bundle=BUNDLE, quotes=None):
    def fetch(now, force=False):
        return _truncate(bundle, now, quotes)
    return fetch


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKORACLE_WEEKLY_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("STOCKORACLE_WEEKLY_SCHEDULER", "0")
    wf._MODEL_CACHE.clear()
    wf._PAYLOAD_CACHE.update({"at": 0.0, "payload": None, "key": None})
    wf._LAST_GOOD_PAYLOAD["payload"] = None
    yield


def _build(now, bundle=BUNDLE, quotes=None):
    return wf.build_weekly_payload(fetcher=_fetcher(bundle, quotes), now=now)


def _day(payload, iso):
    return next(d for d in payload["days"] if d["date"] == iso)


def _ctx(now, code="KOSPI", bundle=BUNDLE):
    b = _truncate(bundle, now)
    gl = {k: ix.drop_incomplete_sessions(v, now)[0] for k, v in b["globals"].items() if v}
    kr, _ = wf._strip_today_if_unsettled(b["kr"][code], now)
    return wf._IndexContext(code, kr, gl)


# ── 거래일 달력 ─────────────────────────────────────────────────────────

class TestKrxCalendar:
    def test_weekday_closures_missing_from_public_holiday_table(self):
        assert kc.holiday_name(dt.date(2026, 5, 1)) == "근로자의 날"
        assert kc.holiday_name(dt.date(2026, 12, 31)) == "연말 휴장일"
        assert kc.holiday_name(dt.date(2023, 12, 29)) == "연말 휴장일"      # 12/31 이 일요일인 해
        assert kc.holiday_name(dt.date(2026, 7, 17)) == "제헌절"
        assert kc.holiday_name(dt.date(2025, 7, 17)) is None               # 2025 년에는 거래일이었다
        assert kc.holiday_name(dt.date(2026, 9, 24)) is not None           # 추석
        assert kc.is_trading_day(dt.date(2026, 11, 4))

    def test_observed_sessions_override_rules_and_env_extra_holiday(self, monkeypatch):
        observed = {dt.date(2026, 11, 2), dt.date(2026, 11, 4)}
        assert not kc.is_trading_day(dt.date(2026, 11, 3), observed, dt.date(2026, 11, 4))
        assert kc.closure_reason(dt.date(2026, 11, 3), observed, dt.date(2026, 11, 4)) == "휴장(실제 거래 기록 없음)"
        assert kc.is_trading_day(dt.date(2026, 11, 5), observed, dt.date(2026, 11, 4))   # 기록 이후는 규칙
        monkeypatch.setenv("STOCKORACLE_KRX_EXTRA_HOLIDAYS", "2026-11-05")
        assert kc.holiday_name(dt.date(2026, 11, 5)) == "임시 휴장일"
        assert dt.date(2026, 11, 5) not in kc.trading_days(dt.date(2026, 11, 2), dt.date(2026, 11, 6))

    def test_session_hours_late_open_days(self):
        o, c = kc.session_hours(dt.date(2026, 11, 19))        # 수능일
        assert (o.hour, c.hour, c.minute) == (10, 16, 30)
        o, c = kc.session_hours(dt.date(2027, 1, 4))          # 새해 첫 거래일
        assert (o.hour, c.hour, c.minute) == (10, 15, 30)
        o, c = kc.session_hours(dt.date(2026, 11, 4))
        assert (o.hour, o.minute, c.hour, c.minute) == (9, 0, 15, 30)

    def test_calendar_mismatches_reports_disagreement(self):
        observed = set(kc.trading_days(dt.date(2026, 10, 1), dt.date(2026, 10, 30))) - {dt.date(2026, 10, 15)}
        mm = kc.calendar_mismatches(observed)
        assert [m["date"] for m in mm] == ["2026-10-15"]


# ── 데이터 계층 ─────────────────────────────────────────────────────────

class TestIndexSeries:
    def test_parse_fchart_sorts_dedupes_and_drops_bad_rows(self):
        text = ('<item data="20261103|5715.19|5735.92|5615.8|5648.79|207371" />'
                '<item data="20261102|5637.77|5910.83|5616.5|5859.08|198130" />'
                '<item data="20261031|0|0|0|0|0" />'
                '<item data="20261103|5715.19|5735.92|5615.8|5648.79|207371" />')
        out = ix.parse_fchart(text)
        assert out["dates"] == [dt.date(2026, 11, 2), dt.date(2026, 11, 3)]
        assert out["close"] == [5859.08, 5648.79]

    def test_parse_index_basic_uses_signed_diff_once(self):
        q = ix.parse_index_basic({"closePrice": "5,648.79", "compareToPreviousClosePrice": "-210.29",
                                  "fluctuationsRatio": "-3.59", "marketStatus": "CLOSE",
                                  "localTradedAt": "2026-11-03T20:15:00+09:00"})
        assert q["prev_close"] == pytest.approx(5859.08)
        assert q["change_pct"] == -3.59 and q["market_status"] == "CLOSE"

    def test_incomplete_us_session_dropped_until_new_york_1630(self):
        s = {"dates": [dt.date(2026, 11, 2), dt.date(2026, 11, 3)], "close": [1.0, 2.0]}
        before = dt.datetime(2026, 11, 3, 16, 29, tzinfo=NY)
        after = dt.datetime(2026, 11, 3, 16, 31, tzinfo=NY)
        kept, dropped = ix.drop_incomplete_sessions(s, before)
        assert kept["dates"] == [dt.date(2026, 11, 2)] and dropped == ["2026-11-03"]
        assert ix.drop_incomplete_sessions(s, after)[1] == []
        # 서머타임(EDT) 기간에는 한국 05:30 에 확정
        assert ix.us_session_complete_at(dt.date(2026, 7, 6)).astimezone(KST).hour == 5

    def test_cached_fetch_falls_back_to_last_good(self):
        ix.clear_cache()
        calls = {"n": 0}

        def ok():
            calls["n"] += 1
            return {"v": 1}

        assert ix.cached_fetch("t:key", 0.0, ok)[0] == {"v": 1}

        def fail():
            raise RuntimeError("down")

        data, note, stale = ix.cached_fetch("t:key", 0.0, fail)
        assert data == {"v": 1} and stale and "마지막 정상" in note
        data, note, stale = ix.cached_fetch("t:none", 0.0, fail)
        assert data is None and not stale


# ── 시점(미래 정보 누출) ─────────────────────────────────────────────────

class TestPointInTime:
    def test_overnight_window_uses_only_sessions_between_anchor_and_target(self):
        gl = copy.deepcopy(BUNDLE["globals"])
        a, t = dt.date(2026, 10, 30), dt.date(2026, 11, 2)       # 금 → 월: 미국 금요일 세션만
        base = km.overnight_changes(gl, [a], [t]).iloc[0]
        i_fri = gl["SPX"]["dates"].index(dt.date(2026, 10, 30))
        i_mon = gl["SPX"]["dates"].index(dt.date(2026, 11, 2))
        gl["SPX"]["close"][i_mon] *= 1.5                           # 대상일 당일 미국 세션 → 영향 없어야
        assert km.overnight_changes(gl, [a], [t]).iloc[0]["on_spx"] == pytest.approx(base["on_spx"])
        gl["SPX"]["close"][i_fri] *= 1.1                           # 창 안 세션 → 반영
        assert km.overnight_changes(gl, [a], [t]).iloc[0]["on_spx"] == pytest.approx(base["on_spx"] + math.log(1.1))

    def test_forecast_identical_when_future_data_removed_or_altered(self):
        target = dt.date(2026, 10, 28)
        open_t, _ = kc.session_hours(target)
        later = _at(2026, 11, 6, 20)
        full = wf.make_forecast(_ctx(later), target, cutoff=open_t, generated_at=later)
        # 대상일 이후 국내·미국 데이터를 크게 바꿔도 같은 예측이어야 한다
        altered = copy.deepcopy(BUNDLE)
        for s in list(altered["kr"].values()) + list(altered["globals"].values()):
            for i, d in enumerate(s["dates"]):
                if d >= target:
                    s["close"][i] *= 3.0
        alt = wf.make_forecast(_ctx(later, bundle=altered), target, cutoff=open_t, generated_at=later)
        # 대상일 이전까지만 있는 데이터로 만든 예측과도 같아야 한다
        before = _at(2026, 10, 28, 8, 30)
        pre = wf.make_forecast(_ctx(before), target, cutoff=open_t, generated_at=before)
        for rec in (alt, pre):
            for key in ("kind", "prob_up", "expected_pct", "low_pct", "high_pct", "anchor", "us_session", "n_train"):
                assert rec[key] == full[key], key
        assert full["kind"] == "preopen" and full["us_session"] == "2026-10-27"
        assert [d["feature"] for d in full["drivers"]] == [d["feature"] for d in pre["drivers"]]

    def test_training_rows_have_targets_settled_before_anchor(self):
        ctx = _ctx(_at(2026, 11, 6, 20))
        ds = ctx.ds_pre
        k = len(ds) - 30
        mask = km.train_mask(ds, k)
        assert ds.target_pos[mask].max() <= ds.target_pos[k] - ds.horizon
        ds5 = km.build_dataset(ctx.kr, None, horizon=5, feature_set="kr_only", frame=ctx.frame)
        m5 = km.train_mask(ds5, len(ds5) - 1)
        assert ds5.target_pos[m5].max() <= ds5.anchor_pos[-1]

    def test_walk_forward_prediction_ignores_future_labels(self):
        ctx = _ctx(_at(2026, 11, 6, 20))
        ds = ctx.ds_pre
        start = len(ds) - 80
        make = lambda: km.RidgeResidualModel(alpha=300.0)
        a = km.walk_forward(ds, make, start, refit_every=21, min_train=500)
        scrambled = copy.deepcopy(ds)
        scrambled.z[start + 21:] = -scrambled.z[start + 21:]
        b = km.walk_forward(scrambled, make, start, refit_every=21, min_train=500)
        assert np.allclose(a["p"][start:start + 21], b["p"][start:start + 21])

    def test_preopen_model_learns_positive_us_effect(self):
        ctx = _ctx(_at(2026, 11, 7, 8))
        f = wf.make_forecast(ctx, dt.date(2026, 11, 9), cutoff=_at(2026, 11, 7, 8), generated_at=_at(2026, 11, 7, 8))
        coef = dict(zip(ctx.ds_pre.feature_names, wf._fit("KOSPI", "preopen", ctx.ds_pre,
                                                         ctx.pos[dt.date(2026, 11, 6)], dt.date(2026, 11, 6)).coef_))
        assert coef["on_spx"] > 0.2
        # 토요일 아침: 금요일 미국 세션이 끝났고 주말에는 세션이 없으므로 월요일 예측은 장 시작 전 예측
        assert f["kind"] == "preopen" and f["horizon"] == 1 and f["us_session"] == "2026-11-06"
        assert not wf._us_window_complete(dt.date(2026, 11, 9), _at(2026, 11, 7, 6, 0), dt.date(2026, 11, 6))
        assert wf._us_window_complete(dt.date(2026, 11, 9), _at(2026, 11, 7, 6, 31), dt.date(2026, 11, 6))


# ── 원장(불변 기록)·갱신 정책 ────────────────────────────────────────────

class TestLedger:
    def test_week_flow_provisional_preopen_final_intraday_actual(self):
        # 수 00:40 — 미국 화요일 세션 진행 중 → 잠정(기저율)
        p1 = _build(_at(2026, 11, 4, 0, 40))
        wed = _day(p1, "2026-11-04")["per_index"]["KOSPI"]
        assert wed["status"] == "forecast" and wed["forecast"]["kind"] == "provisional"
        assert wed["forecast"]["drivers"] == [] and wed["forecast"]["confidence"]["level"] == "none"
        assert p1["data_status"]["US"]["pending_sessions"] == ["2026-11-03"]
        # 수 07:00 — 미국 마감 반영 장 시작 전 예측(개정 1회 추가, 최초 예측은 그대로)
        p2 = _build(_at(2026, 11, 4, 7, 0))
        wed2 = _day(p2, "2026-11-04")["per_index"]["KOSPI"]
        assert wed2["forecast"]["kind"] == "preopen" and wed2["forecast"]["us_session"] == "2026-11-03"
        assert wed2["first"]["kind"] == "provisional" and wed2["revisions"] == 2
        # 같은 정보로 다시 요청 → 개정 이력 늘지 않음
        p3 = _build(_at(2026, 11, 4, 7, 20))
        assert _day(p3, "2026-11-04")["per_index"]["KOSPI"]["revisions"] == 2
        # 목·금은 선행 예측(기저율) — 방향 판단 유보
        thu = _day(p3, "2026-11-05")["per_index"]["KOSPI"]["forecast"]
        assert thu["kind"] == "advance" and thu["direction"] == "hold" and thu["horizon"] == 2
        # 수 10:00 장중 — final 은 07:00 실시간 장 시작 전 예측으로 고정, 장중 수정은 별도
        prev_close = BUNDLE["kr"]["KOSPI"]["close"][BUNDLE["kr"]["KOSPI"]["dates"].index(dt.date(2026, 11, 3))]
        quote = {"KOSPI": {"price": prev_close * 1.01, "prev_close": prev_close, "change_pct": 1.0,
                           "market_status": "OPEN", "traded_at": _at(2026, 11, 4, 10).isoformat(), "source": "test"}}
        p4 = _build(_at(2026, 11, 4, 10, 0), quotes=quote)
        live = _day(p4, "2026-11-04")["per_index"]["KOSPI"]
        assert live["status"] == "intraday"
        assert live["final"]["generated_at"] == wed2["forecast"]["generated_at"]
        assert not live["final"]["reconstructed"]
        assert live["intraday"]["current_pct"] == pytest.approx(1.0, abs=0.01)
        assert live["intraday"]["prob_up"] > 50
        # 코스닥은 시세가 없으면 장중 수정 없이 사전 예측만
        kq = _day(p4, "2026-11-04")["per_index"]["KOSDAQ"]
        assert kq["status"] == "intraday" and kq["intraday"] is None and "message" in kq
        # 수 16:00 — 확정 일봉 → 실제 결과·적중 판정(사전 예측 기준)
        p5 = _build(_at(2026, 11, 4, 16, 0))
        done = _day(p5, "2026-11-04")["per_index"]["KOSPI"]
        i = BUNDLE["kr"]["KOSPI"]["dates"].index(dt.date(2026, 11, 4))
        chg = BUNDLE["kr"]["KOSPI"]["close"][i] / BUNDLE["kr"]["KOSPI"]["close"][i - 1] - 1
        assert done["status"] == "actual" and done["actual"]["change_pct"] == pytest.approx(chg * 100, abs=0.01)
        ev = done["evaluation"]
        expected_hit = None if done["final"]["direction"] == "hold" else (
            done["final"]["direction"] == ("up" if chg > 0 else "down"))
        assert ev["hit"] == expected_hit and ev["scored_kind"] == "preopen" and not ev["reconstructed"]
        # 지난 거래일(월·화)은 재구성 예측으로 채점하되 실시간 기록과 구분
        mon = _day(p5, "2026-11-02")["per_index"]["KOSPI"]
        assert mon["final"]["reconstructed"] and mon["evaluation"]["reconstructed"]
        rec = p5["live_record"]["KOSPI"]
        assert rec["reconstructed_scored"] + rec["live_scored"] >= 1
        assert rec["live_scored"] == (1 if expected_hit is not None else 0)

    def test_first_and_final_are_never_overwritten(self):
        _build(_at(2026, 11, 4, 7, 0))
        state, _ = wf.load_state()
        first = copy.deepcopy(state["days"]["2026-11-04"]["KOSPI"]["first"])
        _build(_at(2026, 11, 4, 10, 0))
        _build(_at(2026, 11, 4, 16, 0))
        _build(_at(2026, 11, 5, 8, 0))
        state, _ = wf.load_state()
        rec = state["days"]["2026-11-04"]["KOSPI"]
        assert rec["first"] == first and rec["final"] == first
        actual = copy.deepcopy(rec["actual"])
        _build(_at(2026, 11, 6, 8, 0))
        state, _ = wf.load_state()
        assert state["days"]["2026-11-04"]["KOSPI"]["actual"] == actual

    def test_provisional_quote_result_replaced_by_daily_bar(self):
        prev_close = BUNDLE["kr"]["KOSPI"]["close"][BUNDLE["kr"]["KOSPI"]["dates"].index(dt.date(2026, 11, 3))]
        quote = {"KOSPI": {"price": prev_close * 0.98, "prev_close": prev_close, "change_pct": -2.0,
                           "market_status": "CLOSE", "traded_at": _at(2026, 11, 4, 15, 31).isoformat(), "source": "test"}}
        p = _build(_at(2026, 11, 4, 15, 35), quotes=quote)
        e = _day(p, "2026-11-04")["per_index"]["KOSPI"]
        assert e["status"] == "actual" and e["actual"]["provisional"] and e["actual"]["change_pct"] == -2.0
        p2 = _build(_at(2026, 11, 4, 16, 0))
        e2 = _day(p2, "2026-11-04")["per_index"]["KOSPI"]
        assert not e2["actual"]["provisional"]
        assert e2["evaluation"]["provisional"] is False

    def test_old_schema_state_is_backed_up(self, tmp_path):
        path = wf.state_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"weeks": {}, "oof_cache": {}}, f)
        p = _build(_at(2026, 11, 4, 7, 0))
        assert any("이전 형식" in w for w in p["warnings"])
        assert os.path.exists(path + ".schema1.bak")

    def test_corrupt_state_is_backed_up_and_rebuilt(self):
        with open(wf.state_path(), "w", encoding="utf-8") as f:
            f.write("{not json")
        p = _build(_at(2026, 11, 4, 7, 0))
        assert p["ok"] and any("새로 시작" in w for w in p["warnings"])


# ── 주말·휴장·토요일 ────────────────────────────────────────────────────

class TestWeekShape:
    def test_holiday_week_and_no_saturday_prediction(self):
        p = _build(_at(2026, 9, 22, 8, 0))          # 추석 주간(9/24·25 휴장)
        days = {d["date"]: d for d in p["days"]}
        assert [d["date"] for d in p["days"]] == ["2026-09-21", "2026-09-22", "2026-09-23",
                                                  "2026-09-24", "2026-09-25", "2026-09-26"]
        assert not days["2026-09-24"]["trading"] and days["2026-09-24"]["per_index"] == {}
        assert days["2026-09-24"]["closure_reason"]
        sat = days["2026-09-26"]
        assert sat["scope"] == "saturday" and "per_index" not in sat
        assert days["2026-09-22"]["is_today"]

    def test_saturday_summary_and_next_week_preopen_outlook(self):
        for t in (_at(2026, 11, 2, 8), _at(2026, 11, 3, 8), _at(2026, 11, 4, 8), _at(2026, 11, 5, 8), _at(2026, 11, 6, 8)):
            _build(t)
        p = _build(_at(2026, 11, 7, 8, 0))
        sat = _day(p, "2026-11-07")
        assert sat["is_today"] and sat["summary"]["final"]
        s = sat["summary"]["per_index"]["KOSPI"]
        assert s["settled"] == 5 and s["live_scored"] == s["scored"] - 0  # 5일 모두 실시간 장 시작 전 기록
        assert s["hits"] <= s["scored"] <= 5
        nxt = sat["next_week"]["per_index"]["KOSPI"]
        assert nxt["first_day"] == "2026-11-09" and nxt["first"]["kind"] == "preopen"
        assert nxt["first"]["us_session"] == "2026-11-06" and nxt["rest_kind"] == "advance"

    def test_sunday_shows_next_week_with_previous_week_summary(self):
        p = _build(_at(2026, 11, 8, 12, 0))
        assert p["week"]["monday"] == "2026-11-09"
        sat = _day(p, "2026-11-14")
        assert sat["previous_week"]["monday"] == "2026-11-02"


# ── 결측·장애 ────────────────────────────────────────────────────────────

class TestFailures:
    def test_global_inputs_missing_falls_back_to_base_rate(self):
        b = copy.deepcopy(BUNDLE)
        b["globals"] = {k: None for k in b["globals"]}
        p = _build(_at(2026, 11, 4, 7, 0), bundle=b)
        f = _day(p, "2026-11-04")["per_index"]["KOSPI"]["forecast"]
        assert f["kind"] == "provisional" and f["us_session"] is None
        assert any("해외 지표 일부 수집 실패" in w for w in p["warnings"])

    def test_partial_global_missing_keeps_model_but_lowers_confidence(self):
        b = copy.deepcopy(BUNDLE)
        b["globals"]["SOX"] = None
        p = _build(_at(2026, 11, 4, 7, 0), bundle=b)
        f = _day(p, "2026-11-04")["per_index"]["KOSPI"]["forecast"]
        assert f["kind"] == "preopen" and f["confidence"].get("missing_inputs") == ["on_sox"]
        assert "on_sox" not in [d["feature"] for d in f["drivers"]]

    def test_lagging_global_series_is_flagged_not_silently_zero(self):
        b = copy.deepcopy(BUNDLE)
        sox = b["globals"]["SOX"]
        cut = sox["dates"].index(dt.date(2026, 11, 3))          # 최신 세션(11-03)만 빠진 상태
        sox["dates"], sox["close"] = sox["dates"][:cut], sox["close"][:cut]
        p = _build(_at(2026, 11, 4, 7, 0), bundle=b)
        f = _day(p, "2026-11-04")["per_index"]["KOSPI"]["forecast"]
        assert f["kind"] == "preopen" and f["confidence"]["missing_inputs"] == ["on_sox"]
        full = _day(_build(_at(2026, 11, 4, 7, 5)), "2026-11-04")["per_index"]["KOSPI"]["forecast"]
        assert full["confidence"].get("missing_inputs") == []

    def test_index_fetch_failure_returns_last_good_marked_stale(self):
        good = _build(_at(2026, 11, 4, 7, 0))
        failing = lambda now, force=False: {"kr": {"KOSPI": None, "KOSDAQ": None}, "globals": {},
                                            "quotes": {}, "errors": ["네트워크 오류"], "stale": []}
        p = wf.build_weekly_payload(fetcher=failing, now=_at(2026, 11, 4, 7, 30))
        assert p["ok"] and p["data_stale"] and p["stale_since"] == good["generated_at"]
        assert p["days"] == good["days"]

    def test_no_data_and_no_history_means_no_fake_forecast(self):
        failing = lambda now, force=False: {"kr": {}, "globals": {}, "quotes": {}, "errors": [], "stale": []}
        p = wf.build_weekly_payload(fetcher=failing, now=_at(2026, 11, 4, 7, 30))
        assert p["ok"] is False and "days" not in p

    def test_delayed_daily_bars_are_flagged(self):
        b = copy.deepcopy(BUNDLE)
        for s in b["kr"].values():
            n = s["dates"].index(dt.date(2026, 11, 3))
            for k, v in list(s.items()):
                if isinstance(v, list):
                    s[k] = v[:n]
        p = _build(_at(2026, 11, 4, 8, 0), bundle=b)
        assert p["data_status"]["KOSPI"]["delayed"]
        tue = _day(p, "2026-11-03")["per_index"]["KOSPI"]
        assert tue["status"] == "pending"
        wed = _day(p, "2026-11-04")["per_index"]["KOSPI"]["forecast"]
        assert wed["horizon"] == 2 and wed["kind"] == "advance"


# ── 값 정합성 ────────────────────────────────────────────────────────────

def _all_forecasts(payload):
    for day in payload["days"]:
        for e in (day.get("per_index") or {}).values():
            for key in ("forecast", "final", "first"):
                if e.get(key):
                    yield e[key]


def test_probabilities_ranges_and_levels_are_consistent():
    p = _build(_at(2026, 11, 4, 7, 0))
    seen = 0
    for f in _all_forecasts(p):
        seen += 1
        assert 0 <= f["prob_up"] <= 100 and f["prob_up"] + f["prob_down"] == pytest.approx(100.0)
        assert f["low_pct"] <= f["expected_pct"] <= f["high_pct"]
        want = "up" if f["prob_up"] / 100 > 0.5 + wf.HOLD_BAND else "down" if f["prob_up"] / 100 < 0.5 - wf.HOLD_BAND else "hold"
        if abs(abs(f["prob_up"] / 100 - 0.5) - wf.HOLD_BAND) > 0.002:
            assert f["direction"] == want
        if f["horizon"] == 1:
            assert f["expected_close"] == pytest.approx(f["anchor_close"] * (1 + f["expected_pct"] / 100), rel=2e-4)
    assert seen >= 8


def test_validation_constants_match_backtest_artifacts():
    path = os.path.join(ROOT, "docs", "backtests", "weekly_forecast_backtest_20261104.json")
    data = json.load(open(path, encoding="utf-8"))
    for code in wf.INDEXES:
        h1 = data["indexes"][code]["horizons"]["1"]
        cfg = wf.PREOPEN_CONFIG[code]
        assert h1["selected"] == f"ridge[{cfg['feature_set']},a={cfg['alpha']:g},{cfg['prob']}]"
        hold = h1["candidates"][h1["selected"]]["holdout"]
        v = wf.VALIDATION[code]
        assert v["accuracy"] == hold["accuracy"] and v["brier"] == hold["brier"] and v["n"] == hold["n"]
        assert v["majority_accuracy"] == hold["majority_accuracy"]
        assert v["us_direction_accuracy"] == h1["candidates"]["baseline:us_direction"]["holdout"]["accuracy"]
        assert v["confidence_buckets"] == hold["confidence_buckets"]
        for h in ("2", "3", "4", "5"):
            assert data["indexes"][code]["horizons"][h]["selected"] == "baseline:base_rate"
    intraday = json.load(open(os.path.join(ROOT, "docs", "backtests", "weekly_forecast_intraday_20261104.json"),
                              encoding="utf-8"))
    for code in wf.INDEXES:
        assert wf.INTRADAY_SCALE[code] == intraday["indexes"][code]["selected_scale"]


def test_intraday_nowcast_monotone_and_deterministic_at_close():
    lo = km.intraday_prob_up(-0.01, 0.01, 200)["p"]
    mid = km.intraday_prob_up(0.0, 0.01, 200)["p"]
    hi = km.intraday_prob_up(0.01, 0.01, 200)["p"]
    assert lo < mid < hi and mid == pytest.approx(0.5)
    assert km.intraday_prob_up(0.002, 0.01, 0)["p"] == 1.0
    assert km.intraday_prob_up(0.002, 0.01, 30)["p"] > km.intraday_prob_up(0.002, 0.01, 300)["p"]


def test_intraday_ignores_quote_from_another_day():
    ctx = _ctx(_at(2026, 11, 4, 10))
    prev = ctx.kr["close"][-1]
    stale_quote = {"price": prev * 1.02, "prev_close": prev * 0.99, "market_status": "OPEN",
                   "traded_at": _at(2026, 11, 3, 15, 0).isoformat()}
    assert wf._intraday(ctx, stale_quote, dt.date(2026, 11, 4), _at(2026, 11, 4, 10)) is None


# ── 화면 계약 ────────────────────────────────────────────────────────────

SOURCE = open(os.path.join(ROOT, "api", "index.py"), encoding="utf-8").read()


def test_section_sits_between_market_core_and_sector_flow():
    i_core = SOURCE.index('<div id="market-core" class="home-section">')
    i_week = SOURCE.index('<section id="weekly-forecast" class="home-section"')
    i_sector = SOURCE.index('<div id="sector-flow" class="home-section">')
    assert i_core < i_week < i_sector
    assert "saturdayDetail(d, day)" not in SOURCE           # 초안의 ReferenceError(d is not defined) 제거
    assert "loadWeeklyForecast(true)]" in SOURCE or "loadWeeklyForecast(true)])" in SOURCE   # 당겨서 새로고침 연동


def _weekly_js():
    start = SOURCE.index("// 📅 코스피·코스닥 주간 예측 — 월~금 거래일 예측")
    end = SOURCE.index("const _SECTOR_FLOW_CACHE_KEY")
    return SOURCE[start:end]


@pytest.mark.skipif(not shutil.which("node"), reason="node 미설치")
def test_main_script_parses_in_node():
    html_start = SOURCE.index('HTML = r"""')
    body = SOURCE[html_start:]
    s = body.index("<script>\n") if "<script>\n" in body else body.index("<script>\r\n")
    e = body.index("</script>", s)
    script = body[s + len("<script>"):e]
    done = subprocess.run(["node", "--check", "-"], input=script, text=True, encoding="utf-8",
                          capture_output=True, timeout=60)
    assert done.returncode == 0, done.stderr[-2000:]


@pytest.mark.skipif(not shutil.which("node"), reason="node 미설치")
def test_calendar_renders_server_values_only():
    for t in (_at(2026, 11, 4, 0, 40), _at(2026, 11, 4, 7, 0)):
        payload = _build(t)
    harness = r"""
const vm = require('vm');
const elements = {};
function el(id) {
  return elements[id] ||= {id, innerHTML: '', textContent: '', hidden: false, title: '', dataset: {},
    scrollWidth: 0, clientWidth: 0, addEventListener() {}, querySelector() { return null; },
    setAttribute() {}, classList: {toggle() {}}};
}
const sandbox = {
  document: {getElementById: el, querySelectorAll() { return []; }, addEventListener() {}, hidden: false},
  localStorage: {getItem() { return null; }, setItem() {}},
  console, Date, Math, Number, String, JSON, Object, Array, Promise, setTimeout() { return 0; }, clearTimeout() {},
  _escPrediction(v) { return String(v == null ? '' : v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); },
};
el('weekly-forecast');
vm.createContext(sandbox);
vm.runInContext(SOURCE + '\n;globalThis.__wf = {render: _wfRender, select: _wfSelect};', sandbox);
const data = DATA;
sandbox.__wf.render(data, '');
const days = elements['wf-days'].innerHTML;
const detail = elements['wf-detail'].innerHTML;
sandbox.__wf.select('2026-11-07');
console.log(JSON.stringify({days, detail, sat: elements['wf-detail'].innerHTML,
  foot: elements['wf-footnote'].innerHTML, meta: elements['wf-head-meta'].textContent}));
""".replace("SOURCE", json.dumps(_weekly_js(), ensure_ascii=False)).replace("DATA", json.dumps(payload, ensure_ascii=False))
    done = subprocess.run(["node", "-"], input=harness, text=True, encoding="utf-8", capture_output=True, timeout=60)
    assert done.returncode == 0, done.stderr[-3000:]
    out = json.loads(done.stdout)
    wed = _day(payload, "2026-11-04")["per_index"]
    for code in ("KOSPI", "KOSDAQ"):
        f = wed[code]["forecast"]
        assert f"{f['prob_up']:.1f}%" in out["days"]
        assert f"{f['prob_up']:.1f}%" in out["detail"] and f"{f['prob_down']:.1f}%" in out["detail"]
        sign = "+" if f["expected_pct"] > 0 else ""
        assert f"{sign}{f['expected_pct']:.2f}%" in out["detail"]
        assert f"{sign}{f['low_pct']:.2f}%".replace("+-", "-") in out["detail"] or f"{f['low_pct']:.2f}%" in out["detail"]
    assert out["days"].count('data-wf-date="') == 6
    assert "토 아침" in out["days"] and "다음 거래주 전망" in out["days"]
    assert "🔵 실제" in out["days"]                       # 월·화 실제 결과
    assert "장 시작 전 예측" in out["days"]
    assert "오늘" in out["days"]
    assert "다음 거래주 첫 거래일" in out["sat"]
    assert "66.2%" in out["foot"] and "54.5%" in out["foot"]
    assert "미국 11/03 마감" in out["meta"]
    assert "undefined" not in out["days"] + out["detail"] + out["sat"]
    assert "NaN" not in out["days"] + out["detail"] + out["sat"]
