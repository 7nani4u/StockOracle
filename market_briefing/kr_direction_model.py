"""코스피·코스닥 거래일 방향 모델 — 특징(시점 고정)·학습·walk-forward 평가 공용 모듈.

서비스(`weekly_forecast.py`)와 오프라인 백테스트(`scripts/backtest_weekly_forecast.py`)가 같은
함수를 쓴다. 그래서 백테스트 수치가 화면 모델의 수치다.

[시점 규칙 — 미래 정보 누출 방지]
대상 거래일 t, 직전 확정 국내 세션(앵커) a 일 때
- 국내 특징: 앵커 a 까지의 일봉만(rolling·EWMA 모두 과거 방향).
- 해외(미국) 특징: 세션 날짜가 [a, t-1] 인 미국 세션의 누적 변화. 미국 세션 d 는 뉴욕 16:00
  (한국 05:00~06:00) 에 끝나므로 t-1 세션까지는 한국 개장(09:00) 전에 확정된다. 서비스는 아직
  확정되지 않은 막대를 지우고(`index_series.drop_incomplete_sessions`), 창이 덜 찼으면 해외 특징이
  없는 국내 전용 모델을 쓴다.
- h 거래일 앞(h>=2) 예측은 앵커 a 의 국내 특징 + 대상일 요일만 쓴다(그 사이 해외 세션은 미지).
- 학습 표본은 정답(대상일 종가)이 예측 앵커 이전에 확정된 행만 쓴다(`train_mask`).

[모델]
목표값 z = 대상일 로그수익률 / 앵커 시점 EWMA 변동성(λ=0.94). 표준화한 특징에 릿지 회귀.
- 기대 등락률 = ẑ·σ, 상승 확률 = 학습 잔차 분포에서 ẑ+e>0 인 비율(경험적), 80% 범위 = ẑ+잔차
  10·90 분위. 정규 근사(Φ(ẑ/s))·로지스틱은 비교용 후보로만 둔다.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

EWMA_LAMBDA = 0.94
SIGMA_FLOOR = 0.002          # 0.2% — 극저변동 구간에서 z 폭주 방지
Z_CLIP = 6.0

KR_FEATURES = [
    "kr_r1", "kr_r5", "kr_r20", "kr_ma20_gap", "kr_ma20_ma60",
    "kr_rsi14", "kr_vol", "kr_volume_ratio", "kr_clv",
]
ON_FEATURES = ["on_spx", "on_ndx", "on_sox", "on_vix", "on_tnx"]
DOW_FEATURES = ["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri"]

FEATURE_SETS = {
    # 해외 장 마감 정보까지 쓰는 장 시작 전(h=1) 모델 후보
    "overnight_full": KR_FEATURES + ON_FEATURES + DOW_FEATURES,
    "overnight_compact": ["kr_r1", "kr_r5", "kr_vol", "on_spx", "on_sox", "on_vix"] + DOW_FEATURES,
    # 해외 정보가 아직 없거나(h=1 잠정) 며칠 앞(h>=2) 예측
    "kr_only": KR_FEATURES + DOW_FEATURES,
}

FEATURE_LABELS = {
    "kr_r1": "직전 거래일 등락", "kr_r5": "5거래일 모멘텀", "kr_r20": "20거래일 모멘텀",
    "kr_ma20_gap": "종가-20일선 괴리", "kr_ma20_ma60": "20일선-60일선 배열",
    "kr_rsi14": "RSI(14)", "kr_vol": "최근 변동성(EWMA)", "kr_volume_ratio": "거래량(5일/20일)",
    "kr_clv": "직전일 종가 위치(고저 범위 내)",
    "on_spx": "미국 S&P500 마감", "on_ndx": "나스닥 마감", "on_sox": "필라델피아 반도체 마감",
    "on_vix": "VIX 변화", "on_tnx": "미국 10년물 금리 변화",
    "dow_mon": "요일(월)", "dow_tue": "요일(화)", "dow_wed": "요일(수)", "dow_thu": "요일(목)",
    "dow_fri": "요일(금)",
}
# 화면 표시용 원값 단위
FEATURE_UNITS = {
    "kr_r1": "pct", "kr_r5": "pct", "kr_r20": "pct", "kr_ma20_gap": "pct", "kr_ma20_ma60": "pct",
    "kr_rsi14": "rsi", "kr_vol": "pct", "kr_volume_ratio": "ratio", "kr_clv": "clv",
    "on_spx": "pct", "on_ndx": "pct", "on_sox": "pct", "on_vix": "pct", "on_tnx": "bp",
}

GLOBAL_KIND = {"SPX": "log", "NDX": "log", "SOX": "log", "VIX": "log", "TNX": "diff"}


# ── 특징 ──────────────────────────────────────────────────────────────────

def kr_anchor_frame(kr: dict) -> pd.DataFrame:
    """국내 지수 일봉 → 날짜별 앵커 특징(그 날 종가까지만 사용)."""
    idx = pd.DatetimeIndex(pd.to_datetime([d.isoformat() for d in kr["dates"]]))
    close = pd.Series(np.asarray(kr["close"], dtype=float), index=idx)
    high = pd.Series(np.asarray(kr.get("high") or kr["close"], dtype=float), index=idx)
    low = pd.Series(np.asarray(kr.get("low") or kr["close"], dtype=float), index=idx)
    vol = pd.Series(np.asarray(kr.get("volume") or [0.0] * len(idx), dtype=float), index=idx)
    logret = np.log(close).diff()

    out = pd.DataFrame(index=idx)
    out["close"] = close
    out["ret"] = logret
    out["kr_r1"] = logret
    out["kr_r5"] = np.log(close / close.shift(5))
    out["kr_r20"] = np.log(close / close.shift(20))
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    out["kr_ma20_gap"] = close / ma20 - 1.0
    out["kr_ma20_ma60"] = ma20 / ma60 - 1.0
    diff = close.diff()
    gain = diff.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-diff).clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))
    rsi = rsi.where(loss > 0, 100.0)
    out["kr_rsi14"] = (rsi - 50.0) / 50.0
    var = (logret ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False, min_periods=20).mean()
    sigma = np.sqrt(var).clip(lower=SIGMA_FLOOR)
    out["sigma"] = sigma
    out["kr_vol"] = sigma
    v5 = vol.rolling(5).mean()
    v20 = vol.rolling(20).mean()
    ratio = (v5 / v20).where((v5 > 0) & (v20 > 0))
    out["kr_volume_ratio"] = np.log(ratio)
    rng = (high - low)
    out["kr_clv"] = (((close - low) - (high - close)) / rng).where(rng > 0, 0.0)
    out["ma20"] = ma20
    out["ma60"] = ma60
    # 평가용 국면(앵커 시점 정보만): 추세 3분류 + 변동성 확대 여부
    phase = np.where((ma20 > ma60) & (close > ma60), "상승장",
                     np.where((ma20 < ma60) & (close < ma60), "하락장", "횡보·전환"))
    out["phase"] = pd.Series(phase, index=idx).where(ma60.notna(), "판정 불가")
    q80 = sigma.rolling(252, min_periods=120).quantile(0.8)
    out["high_vol"] = (sigma > q80).where(q80.notna(), False)
    return out


def _level_lookup(series: dict, query_dates: Sequence[_dt.date]) -> np.ndarray:
    """날짜 x 이하 마지막 세션의 종가(세션이 없으면 NaN)."""
    s_dates = np.asarray([np.datetime64(d.isoformat()) for d in series["dates"]])
    s_close = np.asarray(series["close"], dtype=float)
    q = np.asarray([np.datetime64(d.isoformat()) for d in query_dates])
    pos = np.searchsorted(s_dates, q, side="right") - 1
    out = np.full(len(q), np.nan)
    ok = pos >= 0
    out[ok] = s_close[pos[ok]]
    return out


def overnight_changes(globals_: dict[str, dict | None], anchors: Sequence[_dt.date],
                      targets: Sequence[_dt.date]) -> pd.DataFrame:
    """세션 날짜 ∈ [anchor, target-1] 인 해외 세션의 누적 변화(로그 또는 차분)."""
    one = _dt.timedelta(days=1)
    starts = [a - one for a in anchors]
    ends = [t - one for t in targets]
    cols: dict[str, np.ndarray] = {}
    for key, kind in GLOBAL_KIND.items():
        series = (globals_ or {}).get(key)
        name = f"on_{key.lower()}"
        if not series or not series.get("dates"):
            cols[name] = np.full(len(targets), np.nan)
            continue
        lv_end = _level_lookup(series, ends)
        lv_start = _level_lookup(series, starts)
        with np.errstate(divide="ignore", invalid="ignore"):
            if kind == "log":
                val = np.log(lv_end / lv_start)
            else:
                val = lv_end - lv_start
        cols[name] = val
    return pd.DataFrame(cols)


def dow_columns(targets: Sequence[_dt.date]) -> pd.DataFrame:
    wd = np.asarray([t.weekday() for t in targets])
    return pd.DataFrame({name: (wd == i).astype(float) for i, name in enumerate(DOW_FEATURES)})


@dataclass
class Dataset:
    feature_names: list[str]
    X: np.ndarray
    z: np.ndarray
    y: np.ndarray
    ret: np.ndarray
    sigma: np.ndarray
    target_pos: np.ndarray       # 대상일의 국내 일봉 위치
    anchor_pos: np.ndarray
    target_dates: list[_dt.date]
    anchor_dates: list[_dt.date]
    phase: list[str]
    high_vol: np.ndarray
    horizon: int
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)


def build_dataset(kr: dict, globals_: dict | None, horizon: int = 1,
                  feature_set: str = "overnight_full", frame: pd.DataFrame | None = None,
                  names: Sequence[str] | None = None) -> Dataset:
    """대상일 i(위치) · 앵커 i-h 의 학습 표. 특징 결측 행·수익률 0 행은 제외한다.

    names 로 특징 일부를 뺄 수 있다(예: 해외 지표 하나가 통째로 수집되지 않은 경우)."""
    names = list(names) if names is not None else list(FEATURE_SETS[feature_set])
    uses_on = any(n.startswith("on_") for n in names)
    if uses_on and horizon != 1:
        raise ValueError("해외 장 마감 특징은 h=1(다음 거래일) 예측에서만 시점이 맞는다")
    frame = frame if frame is not None else kr_anchor_frame(kr)
    dates: list[_dt.date] = list(kr["dates"])
    n = len(dates)
    tpos = np.arange(horizon, n)
    apos = tpos - horizon
    targets = [dates[i] for i in tpos]
    anchors = [dates[i] for i in apos]
    parts = [frame.iloc[apos][[c for c in names if c.startswith("kr_")]].reset_index(drop=True)]
    if uses_on:
        parts.append(overnight_changes(globals_ or {}, anchors, targets)[[c for c in names if c.startswith("on_")]])
    parts.append(dow_columns(targets))
    table = pd.concat(parts, axis=1)[names]
    ret = frame["ret"].to_numpy()[tpos]
    sigma = frame["sigma"].to_numpy()[apos]
    ok = table.notna().all(axis=1).to_numpy() & np.isfinite(ret) & np.isfinite(sigma) & (ret != 0.0)
    z = np.clip(ret / sigma, -Z_CLIP, Z_CLIP)
    phase = frame["phase"].to_numpy()[apos]
    high_vol = frame["high_vol"].to_numpy()[apos].astype(bool)
    sel = np.flatnonzero(ok)
    return Dataset(
        feature_names=names,
        X=table.to_numpy(dtype=float)[sel],
        z=z[sel], y=(ret[sel] > 0).astype(float), ret=ret[sel], sigma=sigma[sel],
        target_pos=tpos[sel], anchor_pos=apos[sel],
        target_dates=[targets[i] for i in sel], anchor_dates=[anchors[i] for i in sel],
        phase=[str(phase[i]) for i in sel], high_vol=high_vol[sel], horizon=horizon,
        meta={"feature_set": feature_set},
    )


def live_feature_row(frame: pd.DataFrame, kr_dates: Sequence[_dt.date], anchor_pos: int,
                     target: _dt.date, feature_set: str, globals_: dict | None = None,
                     names: Sequence[str] | None = None) -> tuple[np.ndarray, dict]:
    """서비스용 1행 특징. frame 은 확정 일봉으로만 만든 것이어야 한다(호출자 책임)."""
    names = list(names) if names is not None else FEATURE_SETS[feature_set]
    row = frame.iloc[anchor_pos]
    values: dict[str, float] = {c: float(row[c]) for c in names if c.startswith("kr_")}
    if any(c.startswith("on_") for c in names):
        on = overnight_changes(globals_ or {}, [kr_dates[anchor_pos]], [target]).iloc[0]
        values.update({c: float(on[c]) for c in names if c.startswith("on_")})
    values.update(dow_columns([target]).iloc[0].to_dict())
    x = np.asarray([values[c] for c in names], dtype=float)
    return x, values


# ── 모델 ──────────────────────────────────────────────────────────────────

def _norm_cdf(x: np.ndarray | float) -> np.ndarray:
    from math import erf, sqrt
    arr = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(arr / sqrt(2.0)))


class RidgeResidualModel:
    """z ~ 릿지(표준화 특징). 확률·분위는 학습 잔차의 경험적 분포로 계산."""

    kind = "ridge"

    def __init__(self, alpha: float = 30.0, prob: str = "empirical"):
        self.alpha = float(alpha)
        self.prob = prob

    def fit(self, X: np.ndarray, z: np.ndarray) -> "RidgeResidualModel":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)
        Xs = np.clip((X - self.mean_) / self.std_, -6.0, 6.0)
        self.z_mean_ = float(z.mean())
        zc = z - self.z_mean_
        p = Xs.shape[1]
        A = Xs.T @ Xs + self.alpha * np.eye(p)
        self.coef_ = np.linalg.solve(A, Xs.T @ zc)
        resid = zc - Xs @ self.coef_
        self.resid_ = np.sort(resid)
        self.s_ = float(np.std(resid, ddof=min(p, max(len(resid) - 1, 0))) or 1.0)
        self.n_train_ = int(len(z))
        return self

    def standardize(self, X: np.ndarray) -> np.ndarray:
        return np.clip((np.atleast_2d(X) - self.mean_) / self.std_, -6.0, 6.0)

    def predict_z(self, X: np.ndarray) -> np.ndarray:
        return self.z_mean_ + self.standardize(X) @ self.coef_

    def prob_up(self, zhat: np.ndarray) -> np.ndarray:
        zhat = np.atleast_1d(zhat)
        if self.prob == "gauss":
            return _norm_cdf(zhat / self.s_)
        # P(zhat + e > 0) = 1 - F_e(-zhat)
        k = np.searchsorted(self.resid_, -zhat, side="right")
        return 1.0 - k / len(self.resid_)

    def quantile(self, zhat: np.ndarray, q: float) -> np.ndarray:
        if self.prob == "gauss":
            from statistics import NormalDist
            return np.atleast_1d(zhat) + NormalDist().inv_cdf(q) * self.s_
        return np.atleast_1d(zhat) + float(np.quantile(self.resid_, q))

    def contributions(self, x: np.ndarray) -> np.ndarray:
        return (self.standardize(x) * self.coef_).ravel()


class LogisticModel:
    """비교 후보: 표준화 특징 로지스틱(L2). 기대 등락률은 같은 특징의 릿지로 따로 계산."""

    kind = "logit"

    def __init__(self, alpha: float = 30.0):
        self.alpha = float(alpha)
        self._ridge = RidgeResidualModel(alpha=alpha)

    def fit(self, X: np.ndarray, z: np.ndarray, y: np.ndarray | None = None) -> "LogisticModel":
        from sklearn.linear_model import LogisticRegression
        y = (z > 0).astype(int) if y is None else y.astype(int)
        self._ridge.fit(X, z)
        Xs = self._ridge.standardize(X)
        self._clf = LogisticRegression(C=1.0 / max(self.alpha, 1e-6) * len(y) / 100.0, max_iter=500)
        self._clf.fit(Xs, y)
        return self

    def predict_z(self, X: np.ndarray) -> np.ndarray:
        return self._ridge.predict_z(X)

    def prob_up_from_X(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(self._ridge.standardize(X))[:, 1]

    def quantile(self, zhat, q):
        return self._ridge.quantile(zhat, q)


# ── 장중 수정 예측(나우캐스트) ───────────────────────────────────────────

def session_sigma(kr: dict) -> float | None:
    """시가→종가 로그수익률의 EWMA 표준편차(장중 변동성, 최근 확정 세션까지)."""
    o = np.asarray(kr.get("open") or [], dtype=float)
    c = np.asarray(kr.get("close") or [], dtype=float)
    if len(o) < 30 or len(o) != len(c):
        return None
    ok = (o > 0) & (c > 0)
    r = np.log(c[ok] / o[ok])
    if len(r) < 30:
        return None
    var = pd.Series(r ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean().iloc[-1]
    return float(max(math.sqrt(var), SIGMA_FLOOR / 2))


def intraday_prob_up(r_now: float, sigma_session: float, minutes_left: float,
                     session_minutes: float = 390.0, scale: float = 1.0) -> dict:
    """현재까지 등락 r_now(전일 종가 대비 로그) + 남은 시간의 무추세 확산 → 종가 상승 확률.

    남은 분산 = (scale·σ_장중)² × 남은 시간 비율. scale 은 scripts/backtest_weekly_intraday.py 의
    전반부 표본으로 고정한 값(후반부 검증)을 서비스가 쓴다.
    """
    frac = min(max(minutes_left / session_minutes, 0.0), 1.0)
    sd = scale * sigma_session * math.sqrt(frac)
    if sd <= 1e-9:
        p = 1.0 if r_now > 0 else 0.0 if r_now < 0 else 0.5
        return {"p": p, "sd_rest": 0.0, "lo": r_now, "hi": r_now}
    p = float(_norm_cdf(r_now / sd))
    return {"p": p, "sd_rest": sd, "lo": r_now - 1.2816 * sd, "hi": r_now + 1.2816 * sd}


# ── walk-forward ─────────────────────────────────────────────────────────

def train_mask(ds: Dataset, first_test_row: int) -> np.ndarray:
    """정답이 확정된 행만: 대상일 위치 <= (첫 시험행 대상일 위치 - h)."""
    limit = ds.target_pos[first_test_row] - ds.horizon
    return ds.target_pos <= limit


def walk_forward(ds: Dataset, make_model: Callable[[], Any], start_row: int,
                 refit_every: int = 21, min_train: int = 750) -> dict[str, np.ndarray]:
    """확장 창 walk-forward. 각 블록의 모델은 블록 첫 행 앵커 이전에 확정된 표본만 본다."""
    n = len(ds)
    p = np.full(n, np.nan)
    zhat = np.full(n, np.nan)
    q10 = np.full(n, np.nan)
    q90 = np.full(n, np.nan)
    for k in range(max(start_row, 0), n, refit_every):
        mask = train_mask(ds, k)
        if mask.sum() < min_train:
            continue
        model = make_model()
        if isinstance(model, LogisticModel):
            model.fit(ds.X[mask], ds.z[mask], ds.y[mask])
        else:
            model.fit(ds.X[mask], ds.z[mask])
        block = slice(k, min(k + refit_every, n))
        zh = model.predict_z(ds.X[block])
        zhat[block] = zh
        if isinstance(model, LogisticModel):
            p[block] = model.prob_up_from_X(ds.X[block])
        else:
            p[block] = model.prob_up(zh)
        q10[block] = model.quantile(zh, 0.10)
        q90[block] = model.quantile(zh, 0.90)
    return {"p": p, "zhat": zhat, "q10": q10, "q90": q90}


def baseline_walk_forward(ds: Dataset, kind: str, start_row: int, refit_every: int = 21,
                          min_train: int = 750, condition: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """기준 모델: 'base'(학습 기저 상승률) · 'conditional'(조건 부호별 기저율, 예: 전일 방향·미국 방향)."""
    n = len(ds)
    p = np.full(n, np.nan)
    mu = np.full(n, np.nan)
    for k in range(max(start_row, 0), n, refit_every):
        mask = train_mask(ds, k)
        if mask.sum() < min_train:
            continue
        block = np.arange(k, min(k + refit_every, n))
        if kind == "base" or condition is None:
            p[block] = ds.y[mask].mean()
            mu[block] = ds.z[mask].mean()
            continue
        cond_train = condition[mask] > 0
        for flag in (True, False):
            sub = mask.copy()
            sub[mask] = cond_train == flag
            rate = ds.y[sub].mean() if sub.sum() >= 30 else ds.y[mask].mean()
            zm = ds.z[sub].mean() if sub.sum() >= 30 else ds.z[mask].mean()
            sel = block[(condition[block] > 0) == flag]
            p[sel] = rate
            mu[sel] = zm
    return {"p": p, "zhat": mu}


# ── 평가 지표 ─────────────────────────────────────────────────────────────

def _auc(p: np.ndarray, y: np.ndarray) -> float | None:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _prf(pred: np.ndarray, y: np.ndarray, cls: int) -> dict:
    tp = float(((pred == cls) & (y == cls)).sum())
    fp = float(((pred == cls) & (y != cls)).sum())
    fn = float(((pred != cls) & (y == cls)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"precision": _r(precision), "recall": _r(recall), "f1": _r(f1), "support": int((y == cls).sum())}


def _r(v, nd: int = 4):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), nd)


def direction_metrics(p: np.ndarray, y: np.ndarray, base_p: np.ndarray | None = None,
                      hold_band: float = 0.05) -> dict:
    ok = np.isfinite(p)
    p, y = p[ok], y[ok].astype(int)
    if base_p is not None:
        base_p = base_p[ok]
    n = len(y)
    if n == 0:
        return {"n": 0}
    pred = (p >= 0.5).astype(int)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    out = {
        "n": int(n),
        "accuracy": _r((pred == y).mean()),
        "up_rate": _r(y.mean()),
        "always_up_accuracy": _r(y.mean()),
        "majority_accuracy": _r(max(y.mean(), 1 - y.mean())),
        "brier": _r(brier),
        "log_loss": _r(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))),
        "auc": _r(_auc(p, y)),
        "up": _prf(pred, y, 1),
        "down": _prf(pred, y, 0),
        "mean_prob_up": _r(p.mean()),
    }
    if base_p is not None:
        bb = float(np.mean((base_p - y) ** 2))
        out["brier_base"] = _r(bb)
        out["brier_skill"] = _r(1.0 - brier / bb) if bb > 0 else None
    decisive = np.abs(p - 0.5) > hold_band
    out["hold_band"] = hold_band
    out["decisive_share"] = _r(decisive.mean())
    out["decisive_accuracy"] = _r((pred[decisive] == y[decisive]).mean()) if decisive.any() else None
    out["hold_share"] = _r(1.0 - decisive.mean())
    # 정확도 95% 신뢰구간(정규 근사)과 다수 클래스 대비 이항 검정 p값(단측)
    acc = (pred == y).mean()
    se = math.sqrt(max(acc * (1 - acc), 1e-12) / n)
    out["accuracy_ci95"] = [_r(acc - 1.96 * se), _r(acc + 1.96 * se)]
    out["accuracy_vs_majority_p"] = _r(_binom_sf(int((pred == y).sum()), n, max(y.mean(), 1 - y.mean())))
    out["calibration"] = calibration_bins(p, y)
    out["confidence_buckets"] = confidence_buckets(p, y)
    return out


def _binom_sf(k: int, n: int, p0: float) -> float:
    """P(X >= k), X~Bin(n, p0). scipy 없이 정규 근사(연속성 보정)."""
    mu = n * p0
    sd = math.sqrt(n * p0 * (1 - p0)) or 1.0
    zval = (k - 0.5 - mu) / sd
    return 0.5 * math.erfc(zval / math.sqrt(2))


def calibration_bins(p: np.ndarray, y: np.ndarray, edges=(0, .35, .45, .5, .55, .65, 1.0001)) -> list[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        out.append({"bin": f"{lo:.2f}-{min(hi, 1):.2f}", "n": int(m.sum()),
                    "mean_p": _r(p[m].mean()), "freq_up": _r(y[m].mean())})
    return out


CONF_EDGES = (0.0, 0.05, 0.10, 0.20, 0.51)


def confidence_buckets(p: np.ndarray, y: np.ndarray) -> list[dict]:
    """|p-0.5| 구간별 방향 적중률 — 화면의 '이 확신 구간 과거 적중률' 근거."""
    dist = np.abs(p - 0.5)
    pred = (p >= 0.5).astype(int)
    out = []
    for lo, hi in zip(CONF_EDGES[:-1], CONF_EDGES[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() == 0:
            out.append({"lo": lo, "hi": hi, "n": 0, "accuracy": None})
            continue
        out.append({"lo": lo, "hi": hi, "n": int(m.sum()), "accuracy": _r((pred[m] == y[m]).mean())})
    return out


def return_metrics(mu: np.ndarray, ret: np.ndarray, lo: np.ndarray | None = None,
                   hi: np.ndarray | None = None) -> dict:
    ok = np.isfinite(mu) & np.isfinite(ret)
    if not ok.any():
        return {"n": 0}
    err = mu[ok] - ret[ok]
    out = {
        "n": int(ok.sum()),
        "mae_pct": _r(np.mean(np.abs(err)) * 100, 4),
        "rmse_pct": _r(math.sqrt(np.mean(err ** 2)) * 100, 4),
        "mae_zero_pct": _r(np.mean(np.abs(ret[ok])) * 100, 4),
        "corr": _r(np.corrcoef(mu[ok], ret[ok])[0, 1]) if ok.sum() > 2 and np.std(mu[ok]) > 0 else None,
        "sign_agreement": _r((np.sign(mu[ok]) == np.sign(ret[ok])).mean()),
    }
    if lo is not None and hi is not None:
        m = ok & np.isfinite(lo) & np.isfinite(hi)
        out["coverage80"] = _r(((ret[m] >= lo[m]) & (ret[m] <= hi[m])).mean()) if m.any() else None
        out["mean_width_pct"] = _r(np.mean(hi[m] - lo[m]) * 100, 4) if m.any() else None
    return out


def sliced_metrics(p: np.ndarray, y: np.ndarray, labels: Sequence[Any], base_p: np.ndarray | None = None,
                   min_n: int = 30) -> dict:
    out = {}
    labels = np.asarray(labels, dtype=object)
    for lab in sorted(set(labels.tolist()), key=str):
        m = (labels == lab) & np.isfinite(p)
        if m.sum() < min_n:
            continue
        d = direction_metrics(p[m], y[m], base_p[m] if base_p is not None else None)
        out[str(lab)] = {k: d[k] for k in ("n", "accuracy", "majority_accuracy", "up_rate", "brier",
                                           "brier_base", "brier_skill", "auc") if k in d}
    return out
