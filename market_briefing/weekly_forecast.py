"""코스피·코스닥 주간 예측 캘린더 서비스 (월~금 거래일 방향 + 토요일 주간 결과).

역할 분리
- 거래일 달력: `krx_calendar`  · 데이터 수집: `index_series`  · 특징·모델·평가: `kr_direction_model`
- 이 모듈: 예측 생성 시점 판단, 기록(불변 원장) 관리, 실제 결과 대조, 화면 페이로드 조립.

예측 종류(기록할 때 구분한다)
- preopen   장 시작 전 예측 — 직전 국내 종가 + 그 뒤 마감한 미국 세션까지 반영(검증된 유일한 모델).
- provisional 잠정 예측 — 다음 거래일이지만 미국 장이 아직 끝나지 않음 → 기저율.
- advance   선행 예측 — 2거래일 이상 앞 → 기저율(국내 지표만으로는 기저율보다 낫지 않았다).
- intraday  장중 수정 예측 — 현재 지수 등락 + 남은 시간 변동성(나우캐스트). 장 시작 전 예측과 별도 기록.
- 실제 결과 — 확정 일봉. 적중 여부는 '장 시작 시각에 고정된 마지막 사전 예측(final)' 으로만 매긴다.

불변 원장 규칙
- `first` 는 그 거래일의 첫 예측으로 절대 바꾸지 않는다.
- `final` 은 장이 열리는 순간 마지막 사전 예측으로 고정한다. 장 전에 기록이 없었으면 장 시작 시점
  정보만으로 다시 계산하고 `reconstructed=True` 로 표시한다(실시간 기록과 따로 집계).
- `revisions` 는 정보가 바뀔 때만 추가(같은 입력으로 재요청하면 추가하지 않음).
- 실제 결과는 일봉 확정값으로만 확정하고, 장 마감 직후 시세 기준 값은 `provisional` 로 표시한다.

검증 수치(VALIDATION)는 scripts/backtest_weekly_forecast.py 의 사전 고정 프로토콜 결과
(docs/backtests/weekly_forecast_backtest_20261104.json)이며 테스트가 두 값의 일치를 지킨다.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import math
import os
import tempfile
import threading
import time
import traceback
from typing import Any, Callable

import numpy as np

from market_briefing import index_series as ix
from market_briefing import kr_direction_model as km
from market_briefing import krx_calendar as kc

KST = kc.KST
MODEL_VERSION = "wkf-2.0"
STATE_SCHEMA = 2
INDEXES = ("KOSPI", "KOSDAQ")
INDEX_NAMES = {"KOSPI": "코스피", "KOSDAQ": "코스닥"}
WEEKDAY_LABELS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
HOLD_BAND = 0.05
SETTLE_MINUTES = 15            # 정규장 종료 후 일봉을 '확정'으로 볼 때까지 기다리는 시간

# 개발 구간(≤2021) Brier 최소로 고른 설정 — 백테스트 JSON 의 selected 와 같아야 한다.
PREOPEN_CONFIG = {
    "KOSPI": {"feature_set": "overnight_compact", "alpha": 300.0, "prob": "empirical"},
    "KOSDAQ": {"feature_set": "overnight_compact", "alpha": 300.0, "prob": "empirical"},
}
INTRADAY_SCALE = {"KOSPI": 0.8, "KOSDAQ": 0.8}   # scripts/backtest_weekly_intraday.py 전반부 선택값

# 보류 구간(2022-01-03~2026-11-03) 성적 — 화면 각주와 신뢰도 라벨의 근거
VALIDATION = {
    "KOSPI": {
        "holdout_range": ["2022-01-03", "2026-11-03"], "n": 1181,
        "accuracy": 0.6622, "accuracy_ci95": [0.6352, 0.6891], "majority_accuracy": 0.5453,
        "us_direction_accuracy": 0.6528, "brier": 0.2139, "brier_base": 0.2481,
        "brier_us_direction": 0.2257, "auc": 0.7149, "coverage80": 0.8103,
        "f1_up": 0.7047, "f1_down": 0.6053,
        "confidence_buckets": [
            {"lo": 0.0, "hi": 0.05, "n": 273, "accuracy": 0.5238},
            {"lo": 0.05, "hi": 0.1, "n": 252, "accuracy": 0.627},
            {"lo": 0.1, "hi": 0.2, "n": 380, "accuracy": 0.6737},
            {"lo": 0.2, "hi": 0.51, "n": 276, "accuracy": 0.8152},
        ],
    },
    "KOSDAQ": {
        "holdout_range": ["2022-01-03", "2026-11-03"], "n": 1180,
        "accuracy": 0.5881, "accuracy_ci95": [0.5601, 0.6162], "majority_accuracy": 0.522,
        "us_direction_accuracy": 0.5992, "brier": 0.2349, "brier_base": 0.2504,
        "brier_us_direction": 0.2406, "auc": 0.6426, "coverage80": 0.7924,
        "f1_up": 0.6563, "f1_down": 0.4863,
        "confidence_buckets": [
            {"lo": 0.0, "hi": 0.05, "n": 347, "accuracy": 0.4611},
            {"lo": 0.05, "hi": 0.1, "n": 322, "accuracy": 0.5683},
            {"lo": 0.1, "hi": 0.2, "n": 362, "accuracy": 0.6685},
            {"lo": 0.2, "hi": 0.51, "n": 149, "accuracy": 0.7315},
        ],
    },
}
CONFIDENCE_LABELS = ["매우 낮음", "낮음", "보통", "높음"]

KIND_LABELS = {
    "preopen": "장 시작 전 예측",
    "provisional": "잠정 예측",
    "advance": "선행 예측",
    "intraday": "장중 수정",
}
DIRECTION_LABELS = {"up": "상승 예상", "down": "하락 예상", "hold": "판단 유보"}

_BUILD_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple, Any] = {}
_PAYLOAD_CACHE: dict[str, Any] = {"at": 0.0, "payload": None, "key": None}
_LAST_GOOD_PAYLOAD: dict[str, Any] = {"payload": None}
_FORCE_MIN_INTERVAL = 20.0
_LAST_FORCE = {"at": 0.0}


# ── 시각·국면 ─────────────────────────────────────────────────────────────

def now_kst() -> _dt.datetime:
    return _dt.datetime.now(KST)


def week_monday(d: _dt.date) -> _dt.date:
    return d - _dt.timedelta(days=d.weekday())


def market_phase(now: _dt.datetime) -> str:
    today = now.date()
    if not kc.is_trading_day(today):
        return "weekend" if today.weekday() >= 5 else "holiday"
    open_t, close_t = kc.session_hours(today)
    if now < open_t:
        return "pre_open"
    if now < close_t:
        return "open"
    return "closed"


def _client_refresh_seconds(now: _dt.datetime) -> int:
    phase = market_phase(now)
    if phase == "open":
        return 300
    minutes = now.hour * 60 + now.minute
    if phase == "pre_open" and minutes >= 5 * 60:
        return 600          # 미국 장 마감(05:00~06:30) 이후 장 시작 전 확정 구간
    if phase == "closed" and minutes < 18 * 60:
        return 600          # 장 마감 후 실제 결과 반영 구간
    return 1800


def _payload_ttl(now: _dt.datetime) -> float:
    phase = market_phase(now)
    if phase == "open":
        return 60.0
    if phase in ("pre_open", "closed"):
        return 120.0
    return 300.0


# ── 상태 파일(불변 원장) ─────────────────────────────────────────────────

def state_path() -> str:
    env = os.getenv("STOCKORACLE_WEEKLY_STATE", "").strip()
    if env:
        return env
    if os.getenv("VERCEL"):
        return os.path.join(tempfile.gettempdir(), "stockoracle_weekly_forecast.json")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "datasets", "weekly_forecast_state.json")


def load_state() -> tuple[dict, str | None]:
    path = state_path()
    if not os.path.exists(path):
        return {"schema": STATE_SCHEMA, "days": {}}, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        backup = f"{path}.corrupt-{int(time.time())}"
        try:
            os.replace(path, backup)
        except OSError:
            backup = None
        return ({"schema": STATE_SCHEMA, "days": {}},
                f"예측 기록 파일을 읽지 못해 새로 시작({type(exc).__name__}); 백업 {backup or '실패'}")
    if not isinstance(data, dict) or data.get("schema") != STATE_SCHEMA:
        backup = f"{path}.schema{(data or {}).get('schema', 1) if isinstance(data, dict) else 'x'}.bak"
        try:
            os.replace(path, backup)
        except OSError:
            pass
        return ({"schema": STATE_SCHEMA, "days": {}},
                "이전 형식(검증 전 초안 모델) 기록을 백업하고 새 원장으로 시작")
    data.setdefault("days", {})
    return data, None


def save_state(state: dict) -> str | None:
    path = state_path()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)
        return None
    except Exception as exc:
        return f"예측 기록 저장 실패({type(exc).__name__}: {exc}) — 메모리에만 유지"


def _compact_old_days(state: dict, today: _dt.date, keep_detail_days: int = 35) -> None:
    """오래된 거래일은 first/final/actual/evaluation 만 남기고 수정 이력을 줄인다."""
    cutoff = today - _dt.timedelta(days=keep_detail_days)
    for d_iso, by_index in (state.get("days") or {}).items():
        try:
            d = _dt.date.fromisoformat(d_iso)
        except ValueError:
            continue
        if d >= cutoff:
            continue
        for rec in (by_index or {}).values():
            if isinstance(rec, dict):
                rec.pop("revisions", None)
                if isinstance(rec.get("intraday"), list):
                    rec["intraday"] = rec["intraday"][-1:]


# ── 데이터 묶음 ──────────────────────────────────────────────────────────

def _strip_today_if_unsettled(kr: dict, now: _dt.datetime) -> tuple[dict, bool]:
    """오늘 일봉이 아직 확정 전(정규장 종료+15분 전)이면 제외한 사본."""
    dates = kr.get("dates") or []
    today = now.date()
    cut = len(dates)
    while cut > 0 and dates[cut - 1] > today:      # 미래 날짜 막대(비정상)는 버린다
        cut -= 1
    if cut > 0 and dates[cut - 1] == today:
        _, close_t = kc.session_hours(today)
        if now < close_t + _dt.timedelta(minutes=SETTLE_MINUTES):
            cut -= 1
    if cut == len(dates):
        return kr, False
    out = {k: (v[:cut] if isinstance(v, list) and len(v) == len(dates) else v) for k, v in kr.items()}
    return out, True


def expected_last_session(now: _dt.datetime) -> _dt.date:
    """지금 확정돼 있어야 할 마지막 국내 세션(일봉 지연 판정용)."""
    today = now.date()
    if kc.is_trading_day(today):
        _, close_t = kc.session_hours(today)
        if now >= close_t + _dt.timedelta(minutes=SETTLE_MINUTES):
            return today
    return kc.previous_trading_day(today)


def default_fetcher(now: _dt.datetime, force: bool = False) -> dict:
    """실제 네트워크 수집. 테스트는 같은 형태의 묶음을 주입한다."""
    phase = market_phase(now)
    kr_ttl = 120.0 if phase in ("open", "closed") else 600.0
    minutes = now.hour * 60 + now.minute
    gl_ttl = 300.0 if 4 * 60 + 30 <= minutes <= 9 * 60 else 900.0
    bundle: dict[str, Any] = {"kr": {}, "globals": {}, "quotes": {}, "errors": [], "stale": []}
    for code in INDEXES:
        data, note, stale = ix.fetch_kr_index_daily(code, count=3000, ttl=kr_ttl, force=force)
        bundle["kr"][code] = data
        if note:
            bundle["errors"].append(note)
        if stale:
            bundle["stale"].append(code)
    for key in ix.GLOBAL_SYMBOLS:
        data, note, stale = ix.fetch_global_daily(key, period="15y", ttl=gl_ttl, force=force)
        bundle["globals"][key] = data
        if note:
            bundle["errors"].append(note)
        if stale:
            bundle["stale"].append(key)
    if phase in ("open", "closed"):
        for code in INDEXES:
            data, note, _ = ix.fetch_kr_index_quote(code, ttl=60.0, force=force)
            bundle["quotes"][code] = data
            if note:
                bundle["errors"].append(note)
    return bundle


# ── 모델 적합(앵커별 캐시) ───────────────────────────────────────────────

def _fit(code: str, kind: str, ds: km.Dataset, anchor_pos: int, anchor_date: _dt.date):
    """앵커 위치 이전에 정답이 확정된 표본으로만 학습."""
    key = (code, kind, tuple(ds.feature_names), anchor_date.isoformat(), len(ds), MODEL_VERSION)
    hit = _MODEL_CACHE.get(key)
    if hit is not None:
        return hit
    mask = ds.target_pos <= anchor_pos
    if mask.sum() < 500:
        return None
    if kind == "preopen":
        cfg = PREOPEN_CONFIG[code]
        model = km.RidgeResidualModel(alpha=cfg["alpha"], prob=cfg["prob"]).fit(ds.X[mask], ds.z[mask])
    else:
        model = _BaseRateModel().fit(ds.z[mask])
    if len(_MODEL_CACHE) > 64:
        _MODEL_CACHE.clear()
    _MODEL_CACHE[key] = model
    return model


class _BaseRateModel:
    """기저율 모델: 학습 구간 상승 비율·z 분포만 사용(특징 없음)."""

    def fit(self, z: np.ndarray) -> "_BaseRateModel":
        self.z_sorted_ = np.sort(z)
        self.z_mean_ = float(z.mean())
        self.p_up_ = float((z > 0).mean())
        self.n_train_ = int(len(z))
        return self

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.z_sorted_, q))


class _IndexContext:
    """지수 1개의 확정 일봉·특징 프레임·학습 표(요청마다 1회 생성)."""

    def __init__(self, code: str, kr_complete: dict, globals_complete: dict):
        self.code = code
        self.kr = kr_complete
        self.dates: list[_dt.date] = list(kr_complete["dates"])
        self.pos = {d: i for i, d in enumerate(self.dates)}
        self.frame = km.kr_anchor_frame(kr_complete)
        self.globals = globals_complete
        fs = PREOPEN_CONFIG[code]["feature_set"]
        available = {k for k, v in (globals_complete or {}).items() if v and v.get("dates")}
        # 통째로 수집 실패한 해외 지표는 학습·예측 모두에서 뺀다(전 행이 결측이 되는 것 방지)
        self.pre_names = [n for n in km.FEATURE_SETS[fs] if not n.startswith("on_") or n[3:].upper() in available]
        self.pre_dropped = [n for n in km.FEATURE_SETS[fs] if n not in self.pre_names]
        self.ds_pre = km.build_dataset(kr_complete, globals_complete, 1, fs, frame=self.frame,
                                       names=self.pre_names)
        self.ds_base = km.build_dataset(kr_complete, None, 1, "kr_only", frame=self.frame)
        self.observed = set(self.dates)
        self.session_sigma = km.session_sigma(kr_complete)

    @property
    def last_date(self) -> _dt.date:
        return self.dates[-1]

    def trading_days_between(self, after: _dt.date, upto: _dt.date) -> list[_dt.date]:
        return kc.trading_days(after + _dt.timedelta(days=1), upto, self.observed, self.last_date)

    def anchor_for(self, target: _dt.date) -> int | None:
        """target 직전 확정 세션 위치."""
        i = int(np.searchsorted(np.asarray(self.dates, dtype="datetime64[D]"),
                                np.datetime64(target.isoformat()), side="left")) - 1
        return i if i >= 60 else None


# ── 예측 레코드 ──────────────────────────────────────────────────────────

def _direction(p: float, band: float = HOLD_BAND) -> str:
    if p > 0.5 + band:
        return "up"
    if p < 0.5 - band:
        return "down"
    return "hold"


def _confidence(code: str, p: float, kind: str, missing: list[str]) -> dict:
    if kind != "preopen":
        return {"level": "none", "label": "정보 없음(기저율)",
                "note": "검증상 이 시점에는 기저율보다 나은 정보가 없어 방향을 판단하지 않습니다."}
    dist = abs(p - 0.5)
    buckets = VALIDATION[code]["confidence_buckets"]
    idx = next((i for i, b in enumerate(buckets) if b["lo"] <= dist < b["hi"]), len(buckets) - 1)
    b = buckets[idx]
    level = ["very_low", "low", "medium", "high"][idx]
    if missing and idx > 0:
        idx -= 1
        level = ["very_low", "low", "medium", "high"][idx]
    return {
        "level": level, "label": CONFIDENCE_LABELS[idx],
        "bucket_accuracy": b["accuracy"], "bucket_n": b["n"],
        "note": (f"과거 검증(2022~2026 보류 구간)에서 확신 거리 {b['lo']*100:.0f}~{min(b['hi'], .5)*100:.0f}%p "
                 f"구간 적중률 {b['accuracy']*100:.1f}% (n={b['n']})"),
        "missing_inputs": missing,
    }


def _value_text(name: str, value: float) -> str:
    unit = km.FEATURE_UNITS.get(name)
    if unit == "pct":
        return f"{value * 100:+.2f}%"
    if unit == "bp":
        return f"{value * 100:+.0f}bp"
    if unit == "rsi":
        return f"{value * 50 + 50:.0f}"
    if unit == "ratio":
        return f"×{math.exp(value):.2f}"
    if unit == "clv":
        return f"{value:+.2f}"
    return f"{value:.3f}"


def _drivers(model: km.RidgeResidualModel, names: list[str], x: np.ndarray) -> list[dict]:
    contrib = model.contributions(x)
    rows = []
    for name, value, c in zip(names, x, contrib):
        if abs(c) < 1e-4:
            continue
        rows.append({
            "feature": name, "label": km.FEATURE_LABELS.get(name, name),
            "value_text": "해당 요일" if name.startswith("dow_") else _value_text(name, float(value)),
            "contribution": round(float(c), 4), "direction": "up" if c > 0 else "down",
        })
    rows.sort(key=lambda r: -abs(r["contribution"]))
    return rows[:5]


def _last_us_session(globals_: dict, upto: _dt.date) -> _dt.date | None:
    spx = (globals_ or {}).get("SPX") or {}
    ds = [d for d in (spx.get("dates") or []) if d <= upto]
    return ds[-1] if ds else None


def _us_window_complete(target: _dt.date, cutoff: _dt.datetime, anchor: _dt.date | None = None) -> bool:
    """[anchor, target-1] 안의 마지막 평일(미국 세션 가능일)이 마감됐는지. 주말에는 세션이 없으므로
    토요일 아침에 다음 주 월요일 창은 이미 완결이다. 미국 평일 휴장일은 보수적으로 16:30 까지 기다린다."""
    d = target - _dt.timedelta(days=1)
    while d.weekday() >= 5 and (anchor is None or d > anchor):
        d -= _dt.timedelta(days=1)
    if d.weekday() >= 5:
        return True
    return ix.us_session_complete_at(d) <= cutoff


def make_forecast(ctx: _IndexContext, target: _dt.date, cutoff: _dt.datetime,
                  generated_at: _dt.datetime) -> dict | None:
    """정보 마감 시각 cutoff 에서 target 의 사전 예측. 데이터 부족이면 None."""
    anchor_pos = ctx.anchor_for(target)
    if anchor_pos is None:
        return None
    anchor = ctx.dates[anchor_pos]
    between = ctx.trading_days_between(anchor, target)
    horizon = max(1, len(between))
    us_ok = horizon == 1 and _us_window_complete(target, cutoff, anchor) and bool((ctx.globals or {}).get("SPX"))
    kind = "preopen" if us_ok else ("provisional" if horizon == 1 else "advance")
    sigma = float(ctx.frame["sigma"].iloc[anchor_pos])
    anchor_close = float(ctx.kr["close"][anchor_pos])
    missing: list[str] = []
    drivers: list[dict] = []
    if kind == "preopen":
        ds = ctx.ds_pre
        model = _fit(ctx.code, "preopen", ds, anchor_pos, anchor)
        if model is None:
            return None
        x, values = km.live_feature_row(ctx.frame, ctx.dates, anchor_pos, target,
                                        PREOPEN_CONFIG[ctx.code]["feature_set"], ctx.globals,
                                        names=ctx.pre_names)
        # 최신 세션이 아직 들어오지 않은 해외 지표는 창 변화가 0 으로 계산되므로 결측으로 처리한다
        spx_last = _last_us_session(ctx.globals, target - _dt.timedelta(days=1))
        for name in ctx.pre_names:
            if name.startswith("on_") and name != "on_spx":
                series = (ctx.globals or {}).get(name[3:].upper()) or {}
                last = [d for d in (series.get("dates") or []) if d <= target - _dt.timedelta(days=1)]
                if spx_last and (not last or last[-1] < spx_last):
                    values[name] = float("nan")
                    x[ctx.pre_names.index(name)] = float("nan")
        missing = list(ctx.pre_dropped) + [n for n, v in values.items() if not math.isfinite(v)]
        if missing:
            # 해외 일부 결측: 학습 평균값(표준화 0)으로 두고 신뢰도를 한 단계 낮춘다.
            x = np.where(np.isfinite(x), x, model.mean_)
        zhat = float(model.predict_z(x)[0])
        p = float(model.prob_up(np.asarray([zhat]))[0])
        q10 = float(model.quantile(np.asarray([zhat]), 0.10)[0])
        q90 = float(model.quantile(np.asarray([zhat]), 0.90)[0])
        drivers = _drivers(model, ds.feature_names, x)
        model_name = (f"릿지 회귀[{PREOPEN_CONFIG[ctx.code]['feature_set']}, α={PREOPEN_CONFIG[ctx.code]['alpha']:g}]"
                      " + 경험적 잔차 분포")
        n_train = model.n_train_
    else:
        model = _fit(ctx.code, "base", ctx.ds_base, anchor_pos, anchor)
        if model is None:
            return None
        p = model.p_up_
        zhat = model.z_mean_
        q10, q90 = model.quantile(0.10), model.quantile(0.90)
        model_name = "기저율(과거 상승 비율·등락 분포)"
        n_train = model.n_train_
    us_session = _last_us_session(ctx.globals, target - _dt.timedelta(days=1)) if kind == "preopen" else None
    exp_r, lo_r, hi_r = zhat * sigma, q10 * sigma, q90 * sigma
    direction = _direction(p)
    record = {
        "kind": kind,
        "kind_label": KIND_LABELS[kind],
        "direction": direction,
        "direction_label": DIRECTION_LABELS[direction],
        "prob_up": round(p * 100.0, 1),
        "prob_down": round(100.0 - round(p * 100.0, 1), 1),
        "expected_pct": round((math.exp(exp_r) - 1) * 100.0, 2),
        "low_pct": round((math.exp(lo_r) - 1) * 100.0, 2),
        "high_pct": round((math.exp(hi_r) - 1) * 100.0, 2),
        "range_level": 80,
        "sigma_pct": round(sigma * 100.0, 2),
        "confidence": _confidence(ctx.code, p, kind, missing),
        "drivers": drivers,
        "model": model_name,
        "n_train": n_train,
        "horizon": horizon,
        "anchor": anchor.isoformat(),
        "anchor_close": round(anchor_close, 2),
        "us_session": us_session.isoformat() if us_session else None,
        "info_cutoff": cutoff.isoformat(),
        "generated_at": generated_at.isoformat(),
        "reconstructed": generated_at > cutoff + _dt.timedelta(minutes=1),
        "model_version": MODEL_VERSION,
    }
    if horizon == 1:
        record["expected_close"] = round(anchor_close * math.exp(exp_r), 2)
        record["low_close"] = round(anchor_close * math.exp(lo_r), 2)
        record["high_close"] = round(anchor_close * math.exp(hi_r), 2)
    if kind == "provisional":
        record["note"] = "미국 장이 아직 끝나지 않아 해외 마감 정보를 쓸 수 없습니다 — 미국 마감 후 장 시작 전 예측으로 갱신됩니다."
    elif kind == "advance":
        record["note"] = (f"{horizon}거래일 앞 예측 — 검증에서 국내 지표·요일만으로는 기저율보다 나은 예측을 하지 "
                          "못해 기저율을 그대로 표시합니다. 전날 미국 장 마감 후 장 시작 전 예측으로 갱신됩니다.")
    return record


def _info_key(rec: dict) -> str:
    """같은 정보로 만든 예측인지 판별 — 결측 입력이 채워져도 새 개정으로 기록한다."""
    missing = ",".join(sorted((rec.get("confidence") or {}).get("missing_inputs") or []))
    return "|".join(str(rec.get(k)) for k in ("kind", "anchor", "us_session", "model_version")) + "|" + missing


def _actual_from_data(ctx: _IndexContext, target: _dt.date) -> dict | None:
    i = ctx.pos.get(target)
    if i is None or i < 1:
        return None
    c, prev = float(ctx.kr["close"][i]), float(ctx.kr["close"][i - 1])
    if c <= 0 or prev <= 0:
        return None
    chg = c / prev - 1.0
    return {
        "close": round(c, 2), "prev_close": round(prev, 2), "change_pct": round(chg * 100.0, 2),
        "direction": "up" if chg > 0 else "down" if chg < 0 else "flat",
        "source": ctx.kr.get("source") or "일봉", "provisional": False,
    }


def _actual_from_quote(quote: dict | None, ctx: _IndexContext, target: _dt.date) -> dict | None:
    """장 마감 직후 일봉이 아직 없을 때 시세 기준 잠정 결과(전일 종가는 확정 일봉 기준)."""
    if not quote or str(quote.get("market_status")).upper() != "CLOSE":
        return None
    try:
        traded = _dt.datetime.fromisoformat(quote.get("traded_at") or "")
    except ValueError:
        return None
    if traded.astimezone(KST).date() != target or ctx.last_date >= target:
        return None
    prev = float(ctx.kr["close"][-1])
    price = float(quote["price"])
    if abs(float(quote.get("prev_close") or prev) / prev - 1.0) > 0.002:
        return None     # 전일 종가가 일봉과 다르면 다른 날 시세 — 쓰지 않는다
    chg = price / prev - 1.0
    return {"close": round(price, 2), "prev_close": round(prev, 2), "change_pct": round(chg * 100.0, 2),
            "direction": "up" if chg > 0 else "down" if chg < 0 else "flat",
            "source": quote.get("source"), "provisional": True}


def _evaluate(final: dict | None, actual: dict | None) -> dict | None:
    if not final or not actual:
        return None
    up = 1.0 if actual["direction"] == "up" else 0.0 if actual["direction"] == "down" else None
    p = float(final["prob_up"]) / 100.0
    out = {
        "scored_kind": final.get("kind"),
        "reconstructed": bool(final.get("reconstructed")),
        "predicted": final.get("direction"),
        "actual": actual["direction"],
        "abs_error_pct": round(abs(float(final.get("expected_pct") or 0.0) - actual["change_pct"]), 2),
        "in_range": (final.get("low_pct") is not None
                     and final["low_pct"] <= actual["change_pct"] <= final["high_pct"]),
        "provisional": bool(actual.get("provisional")),
    }
    if up is None:
        out.update({"hit": None, "brier": None, "note": "보합 마감 — 방향 적중 계산 제외"})
    else:
        out["brier"] = round((p - up) ** 2, 4)
        out["hit"] = None if final.get("direction") == "hold" else (final["direction"] == actual["direction"])
        if out["hit"] is None:
            out["note"] = "판단 유보 예측 — 방향 적중 계산 제외"
    return out


def _intraday(ctx: _IndexContext, quote: dict | None, target: _dt.date, now: _dt.datetime) -> dict | None:
    if not quote or ctx.session_sigma is None:
        return None
    try:
        traded = _dt.datetime.fromisoformat(quote.get("traded_at") or "").astimezone(KST)
    except ValueError:
        return None
    open_t, close_t = kc.session_hours(target)
    if traded.date() != target or not (open_t <= now < close_t + _dt.timedelta(minutes=SETTLE_MINUTES)):
        return None
    prev = float(ctx.kr["close"][-1])
    if ctx.last_date >= target or abs(float(quote.get("prev_close") or prev) / prev - 1.0) > 0.002:
        return None
    price = float(quote["price"])
    r_now = math.log(price / prev)
    total = (close_t - open_t).total_seconds() / 60.0
    left = max(0.0, (close_t - now).total_seconds() / 60.0)
    res = km.intraday_prob_up(r_now, ctx.session_sigma, left, total, INTRADAY_SCALE[ctx.code])
    p = res["p"]
    direction = _direction(p)
    return {
        "kind": "intraday", "kind_label": KIND_LABELS["intraday"],
        "direction": direction, "direction_label": DIRECTION_LABELS[direction],
        "prob_up": round(p * 100.0, 1), "prob_down": round(100.0 - round(p * 100.0, 1), 1),
        "current": round(price, 2), "current_pct": round((price / prev - 1) * 100.0, 2),
        "low_pct": round((math.exp(res["lo"]) - 1) * 100.0, 2),
        "high_pct": round((math.exp(res["hi"]) - 1) * 100.0, 2),
        "minutes_left": round(left), "quote_time": traded.isoformat(),
        "generated_at": now.isoformat(), "model_version": MODEL_VERSION,
        "model": "현재 등락 + 남은 시간 무추세 확산(장중 변동성 EWMA×{:.1f})".format(INTRADAY_SCALE[ctx.code]),
    }


# ── 하루·지수 처리 ───────────────────────────────────────────────────────

def _resolve_day(ctx: _IndexContext, rec: dict, target: _dt.date, now: _dt.datetime,
                 quote: dict | None) -> tuple[dict, bool]:
    """반환 (화면 항목, 상태 변경 여부). rec 은 원장의 해당 (날짜·지수) 사전."""
    changed = False
    open_t, close_t = kc.session_hours(target)

    def ensure_final() -> dict | None:
        """장 시작 시각에 고정되는 사전 예측. 실시간 장 시작 전 예측이 있으면 그것, 없으면 재구성."""
        nonlocal changed
        if rec.get("final"):
            return rec["final"]
        before_open = [r for r in rec.get("revisions") or []
                       if _dt.datetime.fromisoformat(r["generated_at"]) < open_t]
        if before_open and before_open[-1].get("kind") == "preopen":
            rec["final"] = before_open[-1]
        else:
            rebuilt = make_forecast(ctx, target, cutoff=open_t, generated_at=max(now, open_t))
            if rebuilt is None:
                return None
            rebuilt["reconstructed"] = True
            if before_open:
                rebuilt["last_live_kind"] = before_open[-1].get("kind")
            rec["final"] = rebuilt
        if not rec.get("first"):
            rec["first"] = rec["final"]
        changed = True
        return rec["final"]

    if now >= open_t:
        final = ensure_final()
        actual = _actual_from_data(ctx, target)
        if actual is None and now >= close_t:
            actual = _actual_from_quote(quote, ctx, target)
        if actual is not None:
            prev_actual = rec.get("actual")
            # 확정 결과는 불변. 잠정(시세 기준) 결과만 일봉 확정값으로 교체한다.
            if prev_actual is None or (prev_actual.get("provisional") and prev_actual != actual):
                rec["actual"] = actual
                rec["evaluation"] = _evaluate(final, actual)
                changed = True
            entry = {"status": "actual", "actual": rec["actual"], "final": final,
                     "evaluation": rec.get("evaluation"), "intraday": (rec.get("intraday") or [None])[-1]}
            return entry, changed
        if now < close_t:
            nowcast = _intraday(ctx, quote, target, now)
            if nowcast is not None:
                hist = rec.setdefault("intraday", [])
                last = hist[-1] if hist else None
                if (last is None or (now - _dt.datetime.fromisoformat(last["generated_at"])).total_seconds() >= 600
                        or abs(last["prob_up"] - nowcast["prob_up"]) >= 5.0):
                    hist.append(nowcast)
                    del hist[:-24]
                    changed = True
            entry = {"status": "intraday", "final": final, "intraday": nowcast or (rec.get("intraday") or [None])[-1]}
            if nowcast is None:
                entry["message"] = "장중 시세를 받지 못해 장 시작 전 예측만 표시"
            return entry, changed
        if target < now.date() and target not in ctx.pos and ctx.last_date >= target:
            return {"status": "no_data", "final": final,
                    "message": "거래일인데 일봉 기록이 없습니다(임시 휴장 또는 데이터 누락)"}, changed
        return {"status": "pending", "final": final,
                "message": "장 마감 — 확정 종가 수집 대기"}, changed

    # 장 시작 전 / 미래 거래일: 사전 예측(정보가 바뀔 때만 개정 이력 추가)
    rec_new = make_forecast(ctx, target, cutoff=now, generated_at=now)
    if rec_new is None:
        return {"status": "unavailable", "message": "확정 일봉이 부족해 예측을 만들지 않았습니다"}, changed
    revisions = rec.setdefault("revisions", [])
    if not revisions or _info_key(revisions[-1]) != _info_key(rec_new):
        revisions.append(rec_new)
        del revisions[:-12]
        changed = True
        current = rec_new
    else:
        current = revisions[-1]
    if not rec.get("first"):
        rec["first"] = current
        changed = True
    return {"status": "forecast", "forecast": current, "first": rec.get("first"),
            "revisions": len(revisions)}, changed


# ── 주간 요약 ────────────────────────────────────────────────────────────

def _week_summary(state: dict, monday: _dt.date, sessions: list[_dt.date], now: _dt.datetime) -> dict:
    per: dict[str, dict] = {}
    for code in INDEXES:
        days, hits, scored, holds, briers, errs, in_range = [], 0, 0, 0, [], [], 0
        live_scored = live_hits = 0
        for d in sessions:
            rec = ((state.get("days") or {}).get(d.isoformat()) or {}).get(code) or {}
            ev, final, actual = rec.get("evaluation"), rec.get("final"), rec.get("actual")
            days.append({"date": d.isoformat(),
                         "predicted": (final or {}).get("direction"), "prob_up": (final or {}).get("prob_up"),
                         "kind": (final or {}).get("kind"), "reconstructed": (final or {}).get("reconstructed"),
                         "actual": (actual or {}).get("direction"), "change_pct": (actual or {}).get("change_pct"),
                         "hit": (ev or {}).get("hit")})
            if not ev:
                continue
            if ev.get("brier") is not None:
                briers.append(ev["brier"])
            errs.append(ev["abs_error_pct"])
            in_range += 1 if ev.get("in_range") else 0
            if ev.get("hit") is None:
                holds += 1 if ev.get("predicted") == "hold" else 0
                continue
            scored += 1
            hits += 1 if ev["hit"] else 0
            if not ev.get("reconstructed"):
                live_scored += 1
                live_hits += 1 if ev["hit"] else 0
        settled = len(errs)
        per[code] = {
            "days": days, "settled": settled, "scored": scored, "hits": hits, "holds": holds,
            "accuracy": round(hits / scored, 4) if scored else None,
            "live_scored": live_scored, "live_hits": live_hits,
            "mean_brier": round(float(np.mean(briers)), 4) if briers else None,
            "mean_abs_error_pct": round(float(np.mean(errs)), 2) if errs else None,
            "range_hits": in_range,
        }
    week_done = now.date() >= monday + _dt.timedelta(days=5)
    return {"final": week_done, "per_index": per,
            "note": ("방향 적중은 장 시작 시각에 고정된 사전 예측으로만 계산합니다. '판단 유보'와 보합 마감은 "
                     "적중 계산에서 빠지며, 장 시작 후 다시 계산한 예측(재구성)은 실시간 기록과 따로 셉니다.")}


def _ledger_summary(state: dict) -> dict:
    out = {}
    for code in INDEXES:
        scored = hits = rscored = rhits = 0
        for by_index in (state.get("days") or {}).values():
            ev = ((by_index or {}).get(code) or {}).get("evaluation")
            if not ev or ev.get("hit") is None or ev.get("provisional"):
                continue
            if ev.get("reconstructed"):
                rscored += 1
                rhits += 1 if ev["hit"] else 0
            else:
                scored += 1
                hits += 1 if ev["hit"] else 0
        out[code] = {"live_scored": scored, "live_hits": hits,
                     "live_accuracy": round(hits / scored, 4) if scored else None,
                     "reconstructed_scored": rscored, "reconstructed_hits": rhits}
    return out


# ── 페이로드 ─────────────────────────────────────────────────────────────

def _display_week(today: _dt.date) -> _dt.date:
    """일요일에는 다음 주(월요일 시작)를 보여준다."""
    return week_monday(today) + (_dt.timedelta(days=7) if today.weekday() == 6 else _dt.timedelta(0))


def build_weekly_payload(force: bool = False, fetcher: Callable[..., dict] | None = None,
                         now: _dt.datetime | None = None) -> dict:
    now = (now or now_kst()).astimezone(KST)
    cache_key = now.date().isoformat()
    if not force and _PAYLOAD_CACHE["payload"] is not None and _PAYLOAD_CACHE["key"] == cache_key \
            and time.monotonic() - _PAYLOAD_CACHE["at"] < _payload_ttl(now) and fetcher is None:
        return copy.deepcopy(_PAYLOAD_CACHE["payload"])
    if force and fetcher is None and time.monotonic() - _LAST_FORCE["at"] < _FORCE_MIN_INTERVAL:
        force = False      # 짧은 간격 강제 새로고침은 캐시된 수집 결과를 재사용(외부 API 보호)
    if force:
        _LAST_FORCE["at"] = time.monotonic()
    with _BUILD_LOCK:
        try:
            payload = _build(now, force, fetcher or default_fetcher)
        except Exception as exc:
            print(f"[weekly-forecast] 생성 실패: {exc}\n{traceback.format_exc()}")
            payload = _stale_payload(f"주간 예측 생성 실패: {type(exc).__name__}: {exc}", now)
    if payload.get("ok") and fetcher is None:
        _PAYLOAD_CACHE.update({"at": time.monotonic(), "payload": payload, "key": cache_key})
        _LAST_GOOD_PAYLOAD["payload"] = payload
    return copy.deepcopy(payload)


def _stale_payload(reason: str, now: _dt.datetime) -> dict:
    good = _LAST_GOOD_PAYLOAD.get("payload")
    if good is None:
        state, _ = load_state()
        good = state.get("last_payload")
    if good:
        out = copy.deepcopy(good)
        out["data_stale"] = True
        out["stale_reason"] = reason
        out["stale_since"] = good.get("generated_at")
        out.setdefault("warnings", []).insert(0, reason)
        out["refresh"] = {"next_refresh_seconds": 120, "policy": "오류 후 재시도"}
        return out
    return {"ok": False, "error": reason, "data_stale": True, "generated_at": now.isoformat(),
            "refresh": {"next_refresh_seconds": 120, "policy": "오류 후 재시도"}}


def _build(now: _dt.datetime, force: bool, fetcher: Callable[..., dict]) -> dict:
    bundle = fetcher(now, force=force)
    warnings: list[str] = list(bundle.get("errors") or [])
    globals_raw = bundle.get("globals") or {}
    globals_complete: dict[str, dict | None] = {}
    pending_us: list[str] = []
    for key, series in globals_raw.items():
        cleaned, dropped = ix.drop_incomplete_sessions(series, now)
        globals_complete[key] = cleaned
        if key == "SPX" and dropped:
            pending_us = dropped
    contexts: dict[str, _IndexContext] = {}
    data_status: dict[str, Any] = {}
    for code in INDEXES:
        raw = (bundle.get("kr") or {}).get(code)
        if not raw or len(raw.get("dates") or []) < 800:
            data_status[code] = {"available": False, "message": "지수 일봉 수집 실패 또는 이력 부족"}
            continue
        complete, dropped_today = _strip_today_if_unsettled(raw, now)
        try:
            contexts[code] = _IndexContext(code, complete, globals_complete)
        except Exception as exc:
            warnings.append(f"{code} 모델 준비 실패: {type(exc).__name__}: {exc}")
            data_status[code] = {"available": False, "message": "모델 준비 실패"}
            continue
        expected_last = expected_last_session(now)
        data_status[code] = {
            "available": True, "last_session": complete["dates"][-1].isoformat(),
            "source": raw.get("source"), "stale": code in (bundle.get("stale") or []),
            "in_progress_excluded": dropped_today,
            "delayed": complete["dates"][-1] < expected_last,
        }
        if data_status[code]["delayed"]:
            warnings.append(f"{INDEX_NAMES[code]} 일봉이 {complete['dates'][-1].isoformat()} 까지만 수집됨 — 이후 거래일 예측은 기저율로 표시")
    if not contexts:
        return _stale_payload("코스피·코스닥 일봉을 모두 수집하지 못했습니다", now)
    spx = globals_complete.get("SPX") or {}
    data_status["US"] = {
        "available": bool(spx.get("dates")),
        "last_complete_session": spx["dates"][-1].isoformat() if spx.get("dates") else None,
        "pending_sessions": pending_us,
        "missing": [k for k, v in globals_complete.items() if not v],
        "stale": [k for k in (bundle.get("stale") or []) if k in ix.GLOBAL_SYMBOLS],
    }
    if data_status["US"]["missing"]:
        warnings.append("해외 지표 일부 수집 실패: " + ", ".join(data_status["US"]["missing"]))

    state, state_note = load_state()
    if state_note:
        warnings.append(state_note)
    changed = False
    today = now.date()
    monday = _display_week(today)
    week_days = [monday + _dt.timedelta(days=i) for i in range(5)]
    any_ctx = next(iter(contexts.values()))
    sessions: list[_dt.date] = []
    days_out: list[dict] = []
    for d in week_days:
        trading = kc.is_trading_day(d, any_ctx.observed, any_ctx.last_date)
        day = {"date": d.isoformat(), "weekday": WEEKDAY_LABELS[d.weekday()], "is_today": d == today,
               "trading": trading, "closure_reason": None if trading else
               kc.closure_reason(d, any_ctx.observed, any_ctx.last_date), "per_index": {}}
        if trading:
            sessions.append(d)
            for code in INDEXES:
                ctx = contexts.get(code)
                if ctx is None:
                    day["per_index"][code] = {"status": "unavailable", "message": data_status[code]["message"]}
                    continue
                rec = state["days"].setdefault(d.isoformat(), {}).setdefault(code, {})
                try:
                    entry, ch = _resolve_day(ctx, rec, d, now, (bundle.get("quotes") or {}).get(code))
                except Exception as exc:
                    print(f"[weekly-forecast] {code} {d} 처리 실패: {exc}\n{traceback.format_exc()}")
                    entry, ch = {"status": "error", "message": f"처리 실패: {type(exc).__name__}"}, False
                changed = changed or ch
                day["per_index"][code] = entry
        days_out.append(day)

    saturday = monday + _dt.timedelta(days=5)
    summary = _week_summary(state, monday, sessions, now)
    next_week = _next_week_outlook(contexts, state, saturday, now)
    prev_week = None
    if today.weekday() == 6:
        prev_monday = monday - _dt.timedelta(days=7)
        prev_sessions = [d for d in (prev_monday + _dt.timedelta(days=i) for i in range(5))
                         if kc.is_trading_day(d, any_ctx.observed, any_ctx.last_date)]
        prev_week = _week_summary(state, prev_monday, prev_sessions, now)
        prev_week["monday"] = prev_monday.isoformat()
    days_out.append({"date": saturday.isoformat(), "weekday": WEEKDAY_LABELS[5], "is_today": saturday == today,
                     "trading": False, "scope": "saturday", "summary": summary, "next_week": next_week,
                     "previous_week": prev_week})

    _compact_old_days(state, today)
    payload = {
        "ok": True,
        "model_version": MODEL_VERSION,
        "generated_at": now.isoformat(),
        "today": today.isoformat(),
        "market_phase": market_phase(now),
        "week": {"monday": monday.isoformat(), "saturday": saturday.isoformat()},
        "days": days_out,
        "data_status": data_status,
        "validation": {**{c: VALIDATION[c] for c in INDEXES}, "hold_band_pct": HOLD_BAND * 100,
                       "protocol": "docs/weekly_forecast_protocol_20261104.md"},
        "live_record": _ledger_summary(state),
        "refresh": {"next_refresh_seconds": _client_refresh_seconds(now),
                    "policy": "장중 5분 · 미국 마감~개장 전 10분 · 장 마감 후 10분 · 그 외 30분"},
        "warnings": warnings,
        "data_stale": bool(bundle.get("stale")),
    }
    state["last_generated_at"] = now.isoformat()
    state["model_version"] = MODEL_VERSION
    state["last_payload"] = payload
    if changed or not os.path.exists(state_path()) or force:
        note = save_state(state)
        if note:
            payload["warnings"].append(note)
    return payload


def _next_week_outlook(contexts: dict[str, _IndexContext], state: dict, saturday: _dt.date,
                       now: _dt.datetime) -> dict:
    """다음 거래주 전망 — 첫 거래일(장 시작 전 예측 가능 여부 포함)과 나머지 날의 기저율."""
    any_ctx = next(iter(contexts.values()))
    nxt_monday = saturday + _dt.timedelta(days=2)
    days = [d for d in (nxt_monday + _dt.timedelta(days=i) for i in range(5))
            if kc.is_trading_day(d, any_ctx.observed, any_ctx.last_date)]
    out: dict[str, Any] = {"monday": nxt_monday.isoformat(), "sessions": [d.isoformat() for d in days],
                           "per_index": {}}
    if not days:
        return out
    for code, ctx in contexts.items():
        first = make_forecast(ctx, days[0], cutoff=now, generated_at=now)
        rest = make_forecast(ctx, days[1], cutoff=now, generated_at=now) if len(days) > 1 else None
        out["per_index"][code] = {
            "first_day": days[0].isoformat(), "first": first,
            "rest_prob_up": (rest or {}).get("prob_up"), "rest_kind": (rest or {}).get("kind"),
        }
    return out


# ── 백그라운드 갱신(장기 실행 서버 전용) ──────────────────────────────────

_SCHED = {"started": False, "lock": threading.Lock()}


def start_scheduler_once(interval_seconds: int | None = None) -> bool:
    """로컬 서버(dev_server)에서 원장을 주기 갱신 — 장 시작 전 예측·장중 수정·마감 결과 기록.

    간격은 화면 갱신 주기와 같다(장중 5분, 미국 마감~개장 전·장 마감 후 10분, 그 외 30분).

    Vercel(서버리스)에서는 스레드가 유지되지 않으므로 `/api/cron` 이 같은 함수를 호출한다.
    끄기: STOCKORACLE_WEEKLY_SCHEDULER=0
    """
    if os.getenv("VERCEL") or str(os.getenv("STOCKORACLE_WEEKLY_SCHEDULER", "1")).lower() in ("0", "false", "no"):
        return False
    with _SCHED["lock"]:
        if _SCHED["started"]:
            return False
        _SCHED["started"] = True

    def loop():
        time.sleep(15)
        while True:
            try:
                build_weekly_payload()
            except Exception as exc:  # pragma: no cover - 방어
                print(f"[weekly-forecast] 예약 갱신 실패: {exc}")
            time.sleep(interval_seconds or _client_refresh_seconds(now_kst()))

    threading.Thread(target=loop, name="weekly_forecast_scheduler", daemon=True).start()
    return True
