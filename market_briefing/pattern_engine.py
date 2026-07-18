"""StockOracle 공통 가격 패턴 탐지 엔진.

이 모듈은 기존 ``api.index.ChartPatternAnalyzer``의 공개 호출 형식을 깨지
않으면서 다음 책임을 한 곳에서 처리하기 위한 순수 Python/Numpy 구현이다.

* 시간 프레임별 피벗/확인/만료 설정
* 고정 1.5%, ATR 적응형, 하이브리드 허용 오차
* E1~E5 공통 좌표와 넥라인/확인선
* forming -> awaiting_breakout -> confirmed 상태 전이
* 구조 기반 목표가와 무효화 가격
* 동일 구간·동일 계열 패턴 중복 제거
* 기존 TP1~TP5에 패턴 목표가 근거 병합
* 차트가 그대로 소비할 수 있는 오버레이 데이터

탐지 함수는 차트를 직접 그리지 않으며 네트워크 호출도 하지 않는다. 입력
배열 마지막 시점까지만 사용하므로 백테스트에서도 같은 호출 계약을 쓸 수
있다. 확정 피벗은 오른쪽 확인 봉이 지난 뒤에만 반환한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import math

import numpy as np


FIXED_TOLERANCE = 0.015
MIN_TOLERANCE = 0.008
MAX_TOLERANCE = 0.045
TOLERANCE_MODES = {"fixed", "atr", "hybrid"}

SHOW_PATTERN_OVERLAY = True
SHOW_FORMING_PATTERNS = True
SHOW_PIVOT_LABELS = True
SHOW_NECKLINE = True
SHOW_PATTERN_TARGETS = True


# UI 조회 기간(period)은 실제 봉 간격(timeframe)과 다르다. 이 매핑은 현재
# fetch_stock_data()의 interval 선택과 동일하게 유지한다.
PERIOD_TO_TIMEFRAME = {
    "1d": "5M",
    "3d": "15M",
    "1wk": "30M",
    "2wk": "1H",
    "1mo": "1H",
    "6mo": "1D",
    "1y": "1D",
    "2y": "1D",
    "5y": "1D",
}


TIMEFRAME_CONFIG: dict[str, dict[str, int | float]] = {
    "5M": {
        "min_pattern_bars": 12, "max_pattern_bars": 180,
        "pivot_window": 3, "min_pivot_gap": 2,
        "breakout_confirm_bars": 3, "volume_period": 36,
        "atr_period": 21, "forming_valid_bars": 36,
        "dedupe_window": 8, "target_horizon": 72,
        "atr_tolerance_multiplier": 0.65,
        "min_breakout_atr": 0.25,
    },
    "15M": {
        "min_pattern_bars": 10, "max_pattern_bars": 160,
        "pivot_window": 3, "min_pivot_gap": 2,
        "breakout_confirm_bars": 2, "volume_period": 32,
        "atr_period": 18, "forming_valid_bars": 30,
        "dedupe_window": 7, "target_horizon": 60,
        "atr_tolerance_multiplier": 0.70,
        "min_breakout_atr": 0.22,
    },
    "30M": {
        "min_pattern_bars": 10, "max_pattern_bars": 140,
        "pivot_window": 3, "min_pivot_gap": 2,
        "breakout_confirm_bars": 2, "volume_period": 26,
        "atr_period": 16, "forming_valid_bars": 26,
        "dedupe_window": 6, "target_horizon": 48,
        "atr_tolerance_multiplier": 0.75,
        "min_breakout_atr": 0.20,
    },
    "1H": {
        "min_pattern_bars": 10, "max_pattern_bars": 120,
        "pivot_window": 3, "min_pivot_gap": 2,
        "breakout_confirm_bars": 2, "volume_period": 20,
        "atr_period": 14, "forming_valid_bars": 24,
        "dedupe_window": 6, "target_horizon": 40,
        "atr_tolerance_multiplier": 0.80,
        "min_breakout_atr": 0.18,
    },
    "1D": {
        "min_pattern_bars": 12, "max_pattern_bars": 150,
        "pivot_window": 4, "min_pivot_gap": 3,
        "breakout_confirm_bars": 1, "volume_period": 20,
        "atr_period": 14, "forming_valid_bars": 30,
        "dedupe_window": 8, "target_horizon": 60,
        "atr_tolerance_multiplier": 0.85,
        "min_breakout_atr": 0.15,
    },
    "1W": {
        "min_pattern_bars": 8, "max_pattern_bars": 104,
        "pivot_window": 2, "min_pivot_gap": 2,
        "breakout_confirm_bars": 1, "volume_period": 13,
        "atr_period": 14, "forming_valid_bars": 12,
        "dedupe_window": 4, "target_horizon": 26,
        "atr_tolerance_multiplier": 1.00,
        "min_breakout_atr": 0.12,
    },
}


# 174개 후보를 독립 함수 174개로 유지하지 않는다. 아래 감사표는 모든 번호를
# 빠짐없이 한 번씩 포함하며, 실제 엔진 레지스트리/중복 제거 정책의 근거로 쓴다.
PATTERN_REFACTOR_AUDIT = (
    {"range": (1, 42), "count": 42, "action": "merge_feature", "canonical": "candlestick_context",
     "reason": "캔들 이름보다 추세·꼬리·몸통·갭 특성을 기존 캔들 엔진의 보조 근거로 유지"},
    {"range": (43, 44), "count": 2, "action": "retain", "canonical": "head_shoulders",
     "reason": "E1~E5·넥라인·높이 투영 목표가·무효화가 명확"},
    {"range": (45, 46), "count": 2, "action": "merge", "canonical": "head_shoulders",
     "reason": "복합형은 같은 핵심 구조의 추가 어깨이므로 공통 파라미터로 흡수"},
    {"range": (47, 48), "count": 2, "action": "retain", "canonical": "triple_reversal",
     "reason": "세 극점과 두 넥라인 전환점으로 독립 검증 가능"},
    {"range": (49, 56), "count": 8, "action": "merge", "canonical": "double_reversal",
     "reason": "Adam/Eve는 피벗 폭·곡률 파라미터이며 목표가와 무효화는 동일"},
    {"range": (57, 58), "count": 2, "action": "merge", "canonical": "double_reversal",
     "reason": "Big W/M은 이중·삼중 반전의 기간 파라미터로 표현"},
    {"range": (59, 65), "count": 7, "action": "merge", "canonical": "triangle_wedge",
     "reason": "상·하단 회귀선 기울기 조합으로 통합"},
    {"range": (66, 76), "count": 11, "action": "merge", "canonical": "continuation_consolidation",
     "reason": "깃발·페넌트·직사각형·채널은 충격파 뒤 압축/돌파 공통 구조"},
    {"range": (77, 80), "count": 4, "action": "remove", "canonical": None,
     "reason": "곡률·핸들 경계가 데이터 길이와 평활화에 과민해 초기 실전 점수에서 제외"},
    {"range": (81, 95), "count": 15, "action": "remove", "canonical": None,
     "reason": "V·다이아몬드·섬형·범프앤런·파이프·혼 세부형은 재현 가능한 공통 목표/무효화 부족"},
    {"range": (96, 115), "count": 20, "action": "remove", "canonical": None,
     "reason": "하모닉 비율 허용오차와 스윙 선택의 과적합 위험이 큼"},
    {"range": (116, 138), "count": 23, "action": "separate_engine", "canonical": "point_and_figure",
     "reason": "시간축 OHLC가 아닌 box/reversal 변환이 필요해 본 엔진에서 제거"},
    {"range": (139, 150), "count": 12, "action": "merge_feature", "canonical": "gap_context",
     "reason": "시장 세션·기업행사 맥락 특성으로 사용하고 독립 구조 점수는 부여하지 않음"},
    {"range": (151, 158), "count": 8, "action": "merge_feature", "canonical": "breakout_trigger",
     "reason": "ORB·Inside/Outside·NR4/NR7은 독립 패턴보다 돌파 확인 특성"},
    {"range": (159, 162), "count": 4, "action": "remove", "canonical": None,
     "reason": "와이코프 전체 단계 자동 분류는 상태 경계가 주관적"},
    {"range": (163, 170), "count": 8, "action": "merge_feature", "canonical": "wyckoff_event",
     "reason": "Spring/UTAD/SOS/SOW/LPS 사건을 지지·저항 실패/회복 특성으로 축약"},
    {"range": (171, 174), "count": 4, "action": "merge_feature", "canonical": "divergence_context",
     "reason": "방향·정규/히든은 공통 피벗 정렬과 선택 지표 파라미터로 표현"},
)

if sum(int(row["count"]) for row in PATTERN_REFACTOR_AUDIT) != 174:
    raise RuntimeError("174개 패턴 감사표의 항목 수가 일치하지 않습니다.")


RETAINED_PATTERN_REGISTRY: dict[str, dict[str, Any]] = {
    "inverse_head_shoulders": {"family": "head_shoulders", "direction": "bullish", "pivots": "LHLHL"},
    "head_shoulders_top": {"family": "head_shoulders", "direction": "bearish", "pivots": "HLHLH"},
    "double_bottom": {"family": "double_reversal", "direction": "bullish", "pivots": "LHL"},
    "double_top": {"family": "double_reversal", "direction": "bearish", "pivots": "HLH"},
    "triple_bottom": {"family": "triple_reversal", "direction": "bullish", "pivots": "LHLHL"},
    "triple_top": {"family": "triple_reversal", "direction": "bearish", "pivots": "HLHLH"},
    "symmetrical_triangle": {"family": "triangle_wedge", "direction": "neutral", "pivots": "alternating"},
    "ascending_triangle": {"family": "triangle_wedge", "direction": "bullish", "pivots": "alternating"},
    "descending_triangle": {"family": "triangle_wedge", "direction": "bearish", "pivots": "alternating"},
    "falling_wedge": {"family": "triangle_wedge", "direction": "bullish", "pivots": "alternating"},
    "rising_wedge": {"family": "triangle_wedge", "direction": "bearish", "pivots": "alternating"},
    "bullish_consolidation": {"family": "continuation_consolidation", "direction": "bullish", "pivots": "range"},
    "bearish_consolidation": {"family": "continuation_consolidation", "direction": "bearish", "pivots": "range"},
}


@dataclass(frozen=True)
class PatternEngineOptions:
    timeframe: str = "1D"
    tolerance_mode: str = "hybrid"
    fixed_tolerance: float = FIXED_TOLERANCE
    min_tolerance: float = MIN_TOLERANCE
    max_tolerance: float = MAX_TOLERANCE
    include_forming: bool = True

    def validate(self) -> None:
        if self.timeframe not in TIMEFRAME_CONFIG:
            raise ValueError(f"지원하지 않는 시간 프레임: {self.timeframe}")
        if self.tolerance_mode not in TOLERANCE_MODES:
            raise ValueError(f"지원하지 않는 허용 오차 모드: {self.tolerance_mode}")
        if not (0 < self.min_tolerance <= self.max_tolerance):
            raise ValueError("허용 오차 최소/최대값이 올바르지 않습니다.")


def resolve_timeframe(period_or_timeframe: str | None) -> str:
    key = str(period_or_timeframe or "1D").strip()
    upper = key.upper()
    if upper in TIMEFRAME_CONFIG:
        return upper
    mapped = PERIOD_TO_TIMEFRAME.get(key.lower())
    if mapped:
        return mapped
    raise ValueError(f"지원하지 않는 기간/시간 프레임: {period_or_timeframe}")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp_at(timestamps: Sequence[Any], index: int) -> Any:
    if 0 <= index < len(timestamps):
        value = timestamps[index]
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value
    return index


def calculate_atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> np.ndarray:
    """Wilder 단순 이동 평균 기반 ATR. NaN은 이전 유효값으로 보수 처리한다."""
    n = min(len(highs), len(lows), len(closes))
    if n == 0:
        return np.array([], dtype=float)
    h = np.asarray(highs[:n], dtype=float)
    l = np.asarray(lows[:n], dtype=float)
    c = np.asarray(closes[:n], dtype=float)
    tr = np.zeros(n, dtype=float)
    for i in range(n):
        if not np.isfinite(h[i]) or not np.isfinite(l[i]) or not np.isfinite(c[i]):
            tr[i] = tr[i - 1] if i else 0.0
            continue
        prev = c[i - 1] if i and np.isfinite(c[i - 1]) else c[i]
        tr[i] = max(h[i] - l[i], abs(h[i] - prev), abs(l[i] - prev))
    out = np.zeros(n, dtype=float)
    p = max(1, int(period))
    for i in range(n):
        start = max(0, i - p + 1)
        valid = tr[start:i + 1]
        out[i] = float(np.mean(valid)) if len(valid) else 0.0
    return out


def calculate_tolerance(
    price_a: float,
    price_b: float,
    *,
    atr_value: float,
    reference_price: float | None = None,
    mode: str = "hybrid",
    atr_multiplier: float = 0.85,
    fixed_tolerance: float = FIXED_TOLERANCE,
    min_tolerance: float = MIN_TOLERANCE,
    max_tolerance: float = MAX_TOLERANCE,
) -> dict[str, Any]:
    """두 가격의 유사성과 실제 사용된 허용 오차 근거를 반환한다."""
    if mode not in TOLERANCE_MODES:
        raise ValueError(f"지원하지 않는 허용 오차 모드: {mode}")
    ref = abs(float(reference_price or ((abs(price_a) + abs(price_b)) / 2.0)))
    if ref <= 0 or not math.isfinite(ref):
        return {"passed": False, "difference_ratio": None, "tolerance": None, "basis": mode}
    diff = abs(float(price_a) - float(price_b)) / ref
    atr_ratio = abs(float(atr_value or 0.0)) * float(atr_multiplier) / ref
    if mode == "fixed":
        tolerance = float(fixed_tolerance)
        basis = "fixed_1.5pct"
    elif mode == "atr":
        tolerance = max(0.0, atr_ratio)
        basis = "atr_adaptive"
    else:
        calculated = max(float(fixed_tolerance), atr_ratio)
        tolerance = min(max(calculated, float(min_tolerance)), float(max_tolerance))
        basis = "hybrid_clamped"
    return {
        "passed": bool(diff <= tolerance + 1e-12),
        "difference_ratio": round(diff, 8),
        "tolerance": round(tolerance, 8),
        "fixed_tolerance": float(fixed_tolerance),
        "atr_tolerance_ratio": round(atr_ratio, 8),
        "basis": basis,
    }


def _pivot(label: str, index: int, price: float, pivot_type: str, timestamps: Sequence[Any], confirmed_index: int) -> dict[str, Any]:
    return {
        "label": label,
        "index": int(index),
        "timestamp": _timestamp_at(timestamps, index),
        "price": float(price),
        "pivot_type": pivot_type,
        "confirmed_index": int(confirmed_index),
    }


def find_confirmed_pivots(
    highs: Sequence[float], lows: Sequence[float], timestamps: Sequence[Any],
    *, window: int, min_gap: int,
) -> list[dict[str, Any]]:
    """오른쪽 ``window``개 봉이 지난 뒤 확정되는 교차 피벗을 한 번 계산한다."""
    n = min(len(highs), len(lows))
    if n < window * 2 + 1:
        return []
    raw: list[dict[str, Any]] = []
    h = np.asarray(highs[:n], dtype=float)
    l = np.asarray(lows[:n], dtype=float)
    for i in range(window, n - window):
        hw = h[i - window:i + window + 1]
        lw = l[i - window:i + window + 1]
        if np.isfinite(h[i]) and np.isfinite(hw).any() and h[i] >= np.nanmax(hw):
            raw.append(_pivot("", i, h[i], "high", timestamps, i + window))
        if np.isfinite(l[i]) and np.isfinite(lw).any() and l[i] <= np.nanmin(lw):
            raw.append(_pivot("", i, l[i], "low", timestamps, i + window))
    raw.sort(key=lambda p: (p["index"], 0 if p["pivot_type"] == "low" else 1))

    # 같은 종류가 연속되면 더 극단적인 하나만 남긴다. 너무 가까운 반대 피벗도
    # 가격 범위가 더 큰 쪽을 유지해 미세 노이즈를 줄인다.
    result: list[dict[str, Any]] = []
    for item in raw:
        if not result:
            result.append(item)
            continue
        prev = result[-1]
        if item["pivot_type"] == prev["pivot_type"]:
            more_extreme = item["price"] > prev["price"] if item["pivot_type"] == "high" else item["price"] < prev["price"]
            if more_extreme:
                result[-1] = item
            continue
        if item["index"] - prev["index"] < min_gap:
            continue
        result.append(item)
    return result


def _line_value(start_index: int, start_price: float, end_index: int, end_price: float, at_index: int) -> float:
    if end_index == start_index:
        return float((start_price + end_price) / 2.0)
    slope = (end_price - start_price) / (end_index - start_index)
    return float(start_price + slope * (at_index - start_index))


def _neckline(a: Mapping[str, Any], b: Mapping[str, Any], at_index: int, breakout_index: int | None = None) -> dict[str, Any]:
    start_price = float(a["price"])
    end_price = float(b["price"])
    breakout_price = _line_value(int(a["index"]), start_price, int(b["index"]), end_price, at_index)
    flat_ratio = abs(start_price - end_price) / max((abs(start_price) + abs(end_price)) / 2.0, 1e-12)
    return {
        "neckline_type": "horizontal" if flat_ratio <= FIXED_TOLERANCE else "trendline",
        "start_index": int(a["index"]), "end_index": int(b["index"]),
        "start_price": start_price, "end_price": end_price,
        "breakout_price": float(breakout_price),
        "breakout_index": int(breakout_index) if breakout_index is not None else None,
    }


def _volume_ratio(volumes: np.ndarray, index: int, period: int) -> float | None:
    if len(volumes) <= index or index < 1 or not np.isfinite(volumes[index]):
        return None
    start = max(0, index - period)
    base = volumes[start:index]
    base = base[np.isfinite(base) & (base > 0)]
    if not len(base):
        return None
    return float(volumes[index] / np.mean(base))


def _confirmation(
    closes: np.ndarray, volumes: np.ndarray, neckline: Mapping[str, Any],
    *, direction: str, after_index: int, atr: np.ndarray, cfg: Mapping[str, Any],
) -> dict[str, Any]:
    needed = int(cfg["breakout_confirm_bars"])
    min_atr = float(cfg["min_breakout_atr"])
    for i in range(max(after_index + 1, needed - 1), len(closes)):
        held = True
        for j in range(i - needed + 1, i + 1):
            line = _line_value(
                int(neckline["start_index"]), float(neckline["start_price"]),
                int(neckline["end_index"]), float(neckline["end_price"]), j,
            )
            buffer = (atr[j] if j < len(atr) else 0.0) * min_atr
            if direction == "bullish" and not (closes[j] > line + buffer):
                held = False
            if direction == "bearish" and not (closes[j] < line - buffer):
                held = False
        if held:
            return {"confirmed": True, "index": i, "volume_ratio": _volume_ratio(volumes, i, int(cfg["volume_period"]))}
    return {"confirmed": False, "index": None, "volume_ratio": None}


def _score_completion(
    *, pivot_fraction: float, geometry_fit: float, timing_fit: float,
    volume_ratio: float | None, breakout_distance_ratio: float,
) -> tuple[float, dict[str, float]]:
    pivot_score = max(0.0, min(1.0, pivot_fraction))
    geometry_score = max(0.0, min(1.0, geometry_fit))
    timing_score = max(0.0, min(1.0, timing_fit))
    volume_score = 0.5 if volume_ratio is None else max(0.0, min(1.0, volume_ratio / 1.5))
    readiness = max(0.0, min(1.0, 1.0 - abs(breakout_distance_ratio)))
    components = {
        "pivot": round(pivot_score * 100, 1),
        "geometry": round(geometry_score * 100, 1),
        "timing": round(timing_score * 100, 1),
        "volume": round(volume_score * 100, 1),
        "breakout_readiness": round(readiness * 100, 1),
    }
    total = 100.0 * (0.30 * pivot_score + 0.30 * geometry_score + 0.15 * timing_score + 0.10 * volume_score + 0.15 * readiness)
    return round(total, 1), components


class PatternEngine:
    """공통 피벗을 재사용하는 상태 기반 패턴 탐지기."""

    def __init__(
        self,
        opens: Sequence[Any], highs: Sequence[Any], lows: Sequence[Any],
        closes: Sequence[Any], volumes: Sequence[Any] | None = None,
        timestamps: Sequence[Any] | None = None,
        *, options: PatternEngineOptions | None = None,
    ) -> None:
        self.options = options or PatternEngineOptions()
        self.options.validate()
        self.cfg = TIMEFRAME_CONFIG[self.options.timeframe]
        arrays = [opens, highs, lows, closes]
        n = min(len(a) for a in arrays) if arrays else 0
        self.opens = np.asarray([_finite(v) for v in opens[:n]], dtype=float)
        self.highs = np.asarray([_finite(v) for v in highs[:n]], dtype=float)
        self.lows = np.asarray([_finite(v) for v in lows[:n]], dtype=float)
        self.closes = np.asarray([_finite(v) for v in closes[:n]], dtype=float)
        raw_volumes = list(volumes[:n]) if volumes is not None else [0.0] * n
        self.volumes = np.asarray([_finite(v) or 0.0 for v in raw_volumes], dtype=float)
        self.timestamps = list(timestamps[:n]) if timestamps is not None else list(range(n))
        self.atr = calculate_atr(self.highs, self.lows, self.closes, int(self.cfg["atr_period"]))
        self.pivots = find_confirmed_pivots(
            self.highs, self.lows, self.timestamps,
            window=int(self.cfg["pivot_window"]), min_gap=int(self.cfg["min_pivot_gap"]),
        )

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Sequence[Any]], *, timeframe: str = "1D",
        tolerance_mode: str = "hybrid", include_forming: bool = True,
    ) -> "PatternEngine":
        return cls(
            data.get("Open", []), data.get("High", []), data.get("Low", []), data.get("Close", []),
            data.get("Volume", []), data.get("Date", []),
            options=PatternEngineOptions(
                timeframe=resolve_timeframe(timeframe), tolerance_mode=tolerance_mode,
                include_forming=include_forming,
            ),
        )

    def _tol(self, a: float, b: float, index: int) -> dict[str, Any]:
        atr_value = float(self.atr[min(max(index, 0), len(self.atr) - 1)]) if len(self.atr) else 0.0
        return calculate_tolerance(
            a, b, atr_value=atr_value,
            mode=self.options.tolerance_mode,
            atr_multiplier=float(self.cfg["atr_tolerance_multiplier"]),
            fixed_tolerance=self.options.fixed_tolerance,
            min_tolerance=self.options.min_tolerance,
            max_tolerance=self.options.max_tolerance,
        )

    def _base_result(
        self, *, pattern_id: str, name: str, family: str, direction: str,
        points: Sequence[Mapping[str, Any]], status: str, neckline: Mapping[str, Any] | None,
        completion: float, components: Mapping[str, float], invalidation_price: float | None,
        target: Mapping[str, Any] | None, next_pivot_type: str | None = None,
        expected_price_range: Sequence[float] | None = None,
        tolerance_basis: str | None = None, related_patterns: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        start = int(points[0]["index"]) if points else max(0, len(self.closes) - 1)
        end = int(points[-1]["index"]) if points else max(0, len(self.closes) - 1)
        signal = "중립"
        if status == "confirmed":
            signal = "매수" if direction == "bullish" else "매도" if direction == "bearish" else "중립"
        return {
            "id": pattern_id, "name": name, "family": family,
            "direction": "상승" if direction == "bullish" else "하락" if direction == "bearish" else "중립",
            "direction_code": direction, "signal": signal,
            "pattern_status": status, "completion_score": round(float(completion), 1),
            "completion_components": dict(components),
            "points": [dict(p) for p in points],
            "completed_pivots": len(points), "next_pivot_type": next_pivot_type,
            "expected_completion_price_range": list(expected_price_range) if expected_price_range else None,
            "neckline": dict(neckline) if neckline else None,
            "invalidation_price": float(invalidation_price) if invalidation_price is not None else None,
            "pattern_target_price": target.get("pattern_target_price") if target else None,
            "pattern_target_low": target.get("pattern_target_low") if target else None,
            "pattern_target_high": target.get("pattern_target_high") if target else None,
            "target_method": target.get("target_method") if target else None,
            "target_confidence": target.get("target_confidence") if target else None,
            "target_reason": target.get("target_reason") if target else "구조 기반 목표가 공식 없음",
            "start_index": start, "end_index": end,
            "formation_start": _timestamp_at(self.timestamps, start),
            "elapsed_bars": max(0, len(self.closes) - 1 - start),
            "timeframe": self.options.timeframe,
            "tolerance_mode": self.options.tolerance_mode,
            "tolerance_basis": tolerance_basis,
            "score_eligible": status == "confirmed",
            "related_patterns": list(related_patterns or []),
            "desc": self._description(name, status, neckline, invalidation_price),
        }

    @staticmethod
    def _description(name: str, status: str, neckline: Mapping[str, Any] | None, invalidation: float | None) -> str:
        state = {"forming": "형성 중", "awaiting_breakout": "돌파 대기", "confirmed": "확정",
                 "invalidated": "무효화", "expired": "만료"}.get(status, status)
        pieces = [f"{name} · {state}"]
        if neckline and neckline.get("breakout_price") is not None:
            pieces.append(f"확인선 {float(neckline['breakout_price']):,.4g}")
        if invalidation is not None:
            pieces.append(f"무효화 {float(invalidation):,.4g}")
        return " | ".join(pieces)

    def _lifecycle_status(
        self, *, base_status: str, direction: str, structure_end: int,
        invalidation_price: float, breakout_index: int | None,
        neckline: Mapping[str, Any] | None,
    ) -> str:
        """구조 완성 이후에 관측된 봉만으로 무효화·만료를 판정한다."""
        start = int(breakout_index + 1) if breakout_index is not None else int(structure_end + 1)
        for i in range(max(0, start), len(self.closes)):
            close = float(self.closes[i])
            invalid = close < invalidation_price if direction == "bullish" else close > invalidation_price
            if invalid:
                return "invalidated"
            if base_status == "confirmed" and neckline:
                line = _line_value(
                    int(neckline["start_index"]), float(neckline["start_price"]),
                    int(neckline["end_index"]), float(neckline["end_price"]), i,
                )
                atr_v = float(self.atr[i]) if i < len(self.atr) else 0.0
                buffer = atr_v * float(self.cfg["min_breakout_atr"])
                failed_retest = close < line - buffer if direction == "bullish" else close > line + buffer
                if failed_retest:
                    return "invalidated"
        if base_status == "awaiting_breakout" and len(self.closes) - 1 - structure_end > int(self.cfg["forming_valid_bars"]):
            return "expired"
        return base_status

    def _detect_head_shoulders(self, direction: str) -> list[dict[str, Any]]:
        expected = ["low", "high", "low", "high", "low"] if direction == "bullish" else ["high", "low", "high", "low", "high"]
        name = "역헤드앤숄더 (Inverse Head-and-Shoulders)" if direction == "bullish" else "헤드앤숄더 천장 (Head-and-Shoulders Top)"
        pid = "inverse_head_shoulders" if direction == "bullish" else "head_shoulders_top"
        results: list[dict[str, Any]] = []
        sequences: list[list[dict[str, Any]]] = []
        for size in (5, 4, 3):
            if size < 5 and not self.options.include_forming:
                continue
            for i in range(max(0, len(self.pivots) - 14), len(self.pivots) - size + 1):
                seq = self.pivots[i:i + size]
                if [p["pivot_type"] for p in seq] == expected[:size]:
                    sequences.append(seq)
        for seq in sequences:
            if len(seq) >= 3:
                head_ok = seq[2]["price"] < seq[0]["price"] if direction == "bullish" else seq[2]["price"] > seq[0]["price"]
                if not head_ok:
                    continue
            span = seq[-1]["index"] - seq[0]["index"]
            if span < int(self.cfg["min_pattern_bars"]) or span > int(self.cfg["max_pattern_bars"]):
                continue
            points = [dict(p, label=f"E{i + 1}") for i, p in enumerate(seq)]
            if len(seq) < 5:
                ref = float(seq[0]["price"])
                atr_v = float(self.atr[seq[-1]["index"]]) if len(self.atr) else ref * 0.02
                expected_range = [ref - atr_v * 0.75, ref + atr_v * 0.75]
                completion, components = _score_completion(
                    pivot_fraction=len(seq) / 5.0, geometry_fit=0.65, timing_fit=0.65,
                    volume_ratio=None, breakout_distance_ratio=1.0,
                )
                results.append(self._base_result(
                    pattern_id=pid, name=name, family="head_shoulders", direction=direction,
                    points=points, status="forming", neckline=None, completion=completion,
                    components=components, invalidation_price=float(seq[2]["price"]), target=None,
                    next_pivot_type=expected[len(seq)], expected_price_range=expected_range,
                    tolerance_basis=self.options.tolerance_mode,
                ))
                continue

            shoulder_tol = self._tol(float(seq[0]["price"]), float(seq[4]["price"]), int(seq[4]["index"]))
            neck_tol = self._tol(float(seq[1]["price"]), float(seq[3]["price"]), int(seq[3]["index"]))
            if not shoulder_tol["passed"]:
                continue
            head_depth = (min(seq[0]["price"], seq[4]["price"]) - seq[2]["price"]) if direction == "bullish" else (seq[2]["price"] - max(seq[0]["price"], seq[4]["price"]))
            atr_v = max(float(self.atr[int(seq[4]["index"])]), 1e-12)
            if head_depth < atr_v * 0.55:
                continue
            left_bars = max(1, seq[2]["index"] - seq[0]["index"])
            right_bars = max(1, seq[4]["index"] - seq[2]["index"])
            timing_fit = min(left_bars, right_bars) / max(left_bars, right_bars)
            nl = _neckline(seq[1], seq[3], len(self.closes) - 1)
            conf = _confirmation(self.closes, self.volumes, nl, direction=direction,
                                 after_index=int(seq[4]["index"]), atr=self.atr, cfg=self.cfg)
            nl = _neckline(seq[1], seq[3], len(self.closes) - 1, conf["index"])
            line_now = float(nl["breakout_price"])
            last = float(self.closes[-1])
            breakout_distance = (line_now - last) / max(atr_v, 1e-12) if direction == "bullish" else (last - line_now) / max(atr_v, 1e-12)
            geometry_fit = max(0.0, 1.0 - float(shoulder_tol["difference_ratio"]) / max(float(shoulder_tol["tolerance"]), 1e-12))
            geometry_fit = 0.7 * geometry_fit + 0.3 * max(0.0, 1.0 - float(neck_tol["difference_ratio"]) / max(float(neck_tol["tolerance"]), 1e-12))
            completion, components = _score_completion(
                pivot_fraction=1.0, geometry_fit=geometry_fit, timing_fit=timing_fit,
                volume_ratio=conf["volume_ratio"], breakout_distance_ratio=breakout_distance,
            )
            status = "confirmed" if conf["confirmed"] else "awaiting_breakout"
            if conf["confirmed"]:
                completion = max(completion, 85.0)
            neckline_at_head = _line_value(seq[1]["index"], seq[1]["price"], seq[3]["index"], seq[3]["price"], seq[2]["index"])
            height = abs(neckline_at_head - seq[2]["price"])
            target_index = int(conf["index"] if conf["index"] is not None else seq[4]["index"])
            target_base = _line_value(seq[1]["index"], seq[1]["price"], seq[3]["index"], seq[3]["price"], target_index)
            target_price = target_base + height if direction == "bullish" else target_base - height
            target = {
                "pattern_target_price": float(target_price),
                "pattern_target_low": float(target_price - atr_v * 0.25),
                "pattern_target_high": float(target_price + atr_v * 0.25),
                "target_method": "head_to_neckline_height_projection",
                "target_confidence": round(min(95.0, completion * (1.0 if status == "confirmed" else 0.65)), 1),
                "target_reason": None,
            }
            invalidation = (seq[4]["price"] - atr_v * 0.25) if direction == "bullish" else (seq[4]["price"] + atr_v * 0.25)
            status = self._lifecycle_status(
                base_status=status, direction=direction, structure_end=int(seq[4]["index"]),
                invalidation_price=float(invalidation), breakout_index=conf["index"], neckline=nl,
            )
            results.append(self._base_result(
                pattern_id=pid, name=name, family="head_shoulders", direction=direction,
                points=points, status=status, neckline=nl, completion=completion,
                components=components, invalidation_price=invalidation, target=target,
                tolerance_basis=f"shoulder:{shoulder_tol['basis']};neckline:{neck_tol['basis']}",
            ))
        return results

    def _detect_double_triple(self, direction: str, count: int) -> list[dict[str, Any]]:
        size = 3 if count == 2 else 5
        expected = (["low", "high", "low", "high", "low"] if direction == "bullish"
                    else ["high", "low", "high", "low", "high"])
        name = ({("bullish", 2): "이중 바닥 (Double Bottom)", ("bearish", 2): "이중 천장 (Double Top)",
                 ("bullish", 3): "삼중 바닥 (Triple Bottom)", ("bearish", 3): "삼중 천장 (Triple Top)"})[(direction, count)]
        pid = ({("bullish", 2): "double_bottom", ("bearish", 2): "double_top",
                ("bullish", 3): "triple_bottom", ("bearish", 3): "triple_top"})[(direction, count)]
        family = "double_reversal" if count == 2 else "triple_reversal"
        results: list[dict[str, Any]] = []
        for i in range(max(0, len(self.pivots) - 14), len(self.pivots) - size + 1):
            seq = self.pivots[i:i + size]
            if [p["pivot_type"] for p in seq] != expected[:size]:
                continue
            span = seq[-1]["index"] - seq[0]["index"]
            if span < int(self.cfg["min_pattern_bars"]) or span > int(self.cfg["max_pattern_bars"]):
                continue
            extremes = seq[::2]
            tol_checks = [self._tol(extremes[0]["price"], p["price"], p["index"]) for p in extremes[1:]]
            if not all(c["passed"] for c in tol_checks):
                continue
            mid_points = seq[1::2]
            # 이중 패턴은 가운데 한 점을 수평 확인선으로, 삼중은 두 점 연결선을 쓴다.
            if len(mid_points) == 1:
                synthetic = dict(mid_points[0])
                synthetic["index"] = int(seq[-1]["index"])
                nl = _neckline(mid_points[0], synthetic, len(self.closes) - 1)
            else:
                nl = _neckline(mid_points[0], mid_points[-1], len(self.closes) - 1)
            conf = _confirmation(self.closes, self.volumes, nl, direction=direction,
                                 after_index=int(seq[-1]["index"]), atr=self.atr, cfg=self.cfg)
            nl["breakout_index"] = conf["index"]
            avg_extreme = float(np.mean([p["price"] for p in extremes]))
            line_at_last = float(nl["breakout_price"])
            height = abs(line_at_last - avg_extreme)
            atr_v = max(float(self.atr[int(seq[-1]["index"])]), 1e-12)
            geometry = 1.0 - float(np.mean([c["difference_ratio"] / max(c["tolerance"], 1e-12) for c in tol_checks]))
            gaps = np.diff([p["index"] for p in extremes])
            timing = float(np.min(gaps) / np.max(gaps)) if len(gaps) > 1 and np.max(gaps) else 1.0
            last = float(self.closes[-1])
            distance = (line_at_last - last) / atr_v if direction == "bullish" else (last - line_at_last) / atr_v
            completion, components = _score_completion(
                pivot_fraction=1.0, geometry_fit=max(0.0, geometry), timing_fit=timing,
                volume_ratio=conf["volume_ratio"], breakout_distance_ratio=distance,
            )
            status = "confirmed" if conf["confirmed"] else "awaiting_breakout"
            if status == "confirmed":
                completion = max(84.0, completion)
            target_index = int(conf["index"] if conf["index"] is not None else seq[-1]["index"])
            target_base = _line_value(
                int(nl["start_index"]), float(nl["start_price"]),
                int(nl["end_index"]), float(nl["end_price"]), target_index,
            )
            target_price = target_base + height if direction == "bullish" else target_base - height
            target = {
                "pattern_target_price": target_price,
                "pattern_target_low": target_price - atr_v * 0.25,
                "pattern_target_high": target_price + atr_v * 0.25,
                "target_method": f"{count}_extreme_neckline_height_projection",
                "target_confidence": round(completion * (1.0 if status == "confirmed" else 0.65), 1),
                "target_reason": None,
            }
            invalidation = avg_extreme - atr_v * 0.25 if direction == "bullish" else avg_extreme + atr_v * 0.25
            status = self._lifecycle_status(
                base_status=status, direction=direction, structure_end=int(seq[-1]["index"]),
                invalidation_price=float(invalidation), breakout_index=conf["index"], neckline=nl,
            )
            points = [dict(p, label=f"E{j + 1}") for j, p in enumerate(seq)]
            results.append(self._base_result(
                pattern_id=pid, name=name, family=family, direction=direction,
                points=points, status=status, neckline=nl, completion=completion,
                components=components, invalidation_price=invalidation, target=target,
                tolerance_basis=",".join(c["basis"] for c in tol_checks),
            ))
        return results

    def _detect_triangle_wedge(self) -> list[dict[str, Any]]:
        recent = self.pivots[-10:]
        highs = [p for p in recent if p["pivot_type"] == "high"][-3:]
        lows = [p for p in recent if p["pivot_type"] == "low"][-3:]
        if len(highs) < 3 or len(lows) < 3:
            return []
        start = min(highs[0]["index"], lows[0]["index"])
        end = max(highs[-1]["index"], lows[-1]["index"])
        span = end - start
        if span < int(self.cfg["min_pattern_bars"]) or span > int(self.cfg["max_pattern_bars"]):
            return []
        hs = float(np.polyfit([p["index"] for p in highs], [p["price"] for p in highs], 1)[0])
        ls = float(np.polyfit([p["index"] for p in lows], [p["price"] for p in lows], 1)[0])
        price = max(abs(float(self.closes[-1])), 1e-12)
        flat = price * 0.0005
        if hs < -flat and ls > flat:
            pid, name, direction = "symmetrical_triangle", "대칭 삼각형 (Symmetrical Triangle)", "neutral"
        elif abs(hs) <= flat and ls > flat:
            pid, name, direction = "ascending_triangle", "상승 삼각형 (Ascending Triangle)", "bullish"
        elif hs < -flat and abs(ls) <= flat:
            pid, name, direction = "descending_triangle", "하락 삼각형 (Descending Triangle)", "bearish"
        elif hs < -flat and ls < -flat and ls > hs:
            pid, name, direction = "falling_wedge", "하락 쐐기형 (Falling Wedge)", "bullish"
        elif hs > flat and ls > flat and hs > ls:
            pid, name, direction = "rising_wedge", "상승 쐐기형 (Rising Wedge)", "bearish"
        else:
            return []
        upper_now = float(np.polyval(np.polyfit([p["index"] for p in highs], [p["price"] for p in highs], 1), len(self.closes) - 1))
        lower_now = float(np.polyval(np.polyfit([p["index"] for p in lows], [p["price"] for p in lows], 1), len(self.closes) - 1))
        if direction == "neutral":
            upper_line = _neckline(highs[0], highs[-1], len(self.closes) - 1)
            lower_line = _neckline(lows[0], lows[-1], len(self.closes) - 1)
            up_conf = _confirmation(self.closes, self.volumes, upper_line, direction="bullish",
                                    after_index=end, atr=self.atr, cfg=self.cfg)
            down_conf = _confirmation(self.closes, self.volumes, lower_line, direction="bearish",
                                      after_index=end, atr=self.atr, cfg=self.cfg)
            if up_conf["confirmed"]:
                direction, name = "bullish", "대칭 삼각형 상향 돌파 (Symmetrical Triangle Up)"
            elif down_conf["confirmed"]:
                direction, name = "bearish", "대칭 삼각형 하향 이탈 (Symmetrical Triangle Down)"
            else:
                atr_v = max(float(self.atr[-1]), 1e-12)
                width = max(upper_now - lower_now, atr_v * 0.5)
                convergence = min(1.0, max(0.0, 1.0 - width / max(abs(highs[0]["price"] - lows[0]["price"]), atr_v)))
                completion, components = _score_completion(
                    pivot_fraction=1.0, geometry_fit=0.7 + convergence * 0.3,
                    timing_fit=0.75, volume_ratio=None,
                    breakout_distance_ratio=min(abs(upper_now - float(self.closes[-1])), abs(float(self.closes[-1]) - lower_now)) / atr_v,
                )
                points_src = sorted(highs + lows, key=lambda p: p["index"])[-5:]
                points = [dict(p, label=f"E{i + 1}") for i, p in enumerate(points_src)]
                result = self._base_result(
                    pattern_id=pid, name=name, family="triangle_wedge", direction="neutral",
                    points=points, status="awaiting_breakout", neckline=upper_line,
                    completion=completion, components=components,
                    invalidation_price=None, target=None,
                    tolerance_basis="regression_slope_dual_breakout",
                )
                result["secondary_confirmation_line"] = lower_line
                result["score_eligible"] = False
                return [result]
        boundary_points = highs if direction == "bullish" else lows
        nl = _neckline(boundary_points[0], boundary_points[-1], len(self.closes) - 1)
        conf = _confirmation(self.closes, self.volumes, nl, direction=direction,
                             after_index=end, atr=self.atr, cfg=self.cfg)
        nl["breakout_index"] = conf["index"]
        atr_v = max(float(self.atr[-1]), 1e-12)
        width = max(upper_now - lower_now, atr_v * 0.5)
        target_price = float(nl["breakout_price"]) + width if direction == "bullish" else float(nl["breakout_price"]) - width
        status = "confirmed" if conf["confirmed"] else "awaiting_breakout"
        convergence = min(1.0, max(0.0, 1.0 - width / max(abs(highs[0]["price"] - lows[0]["price"]), atr_v)))
        completion, components = _score_completion(
            pivot_fraction=1.0, geometry_fit=0.7 + convergence * 0.3, timing_fit=0.75,
            volume_ratio=conf["volume_ratio"], breakout_distance_ratio=abs(float(nl["breakout_price"]) - price) / atr_v,
        )
        if status == "confirmed":
            completion = max(82.0, completion)
        points_src = sorted(highs + lows, key=lambda p: p["index"])[-5:]
        points = [dict(p, label=f"E{i + 1}") for i, p in enumerate(points_src)]
        invalidation = lower_now - atr_v * 0.25 if direction == "bullish" else upper_now + atr_v * 0.25
        status = self._lifecycle_status(
            base_status=status, direction=direction, structure_end=end,
            invalidation_price=float(invalidation), breakout_index=conf["index"], neckline=nl,
        )
        target = {
            "pattern_target_price": target_price,
            "pattern_target_low": target_price - atr_v * 0.25,
            "pattern_target_high": target_price + atr_v * 0.25,
            "target_method": "pattern_width_projection",
            "target_confidence": round(completion * (1.0 if status == "confirmed" else 0.60), 1),
            "target_reason": None,
        }
        return [self._base_result(
            pattern_id=pid, name=name, family="triangle_wedge", direction=direction,
            points=points, status=status, neckline=nl, completion=completion,
            components=components, invalidation_price=invalidation, target=target,
            tolerance_basis="regression_slope_atr_breakout",
        )]

    def _detect_consolidation(self, direction: str) -> list[dict[str, Any]]:
        """깃발·페넌트·직사각형을 충격파 뒤 압축이라는 공통 구조로 탐지한다."""
        n = len(self.closes)
        consolidation_bars = max(8, int(self.cfg["min_pattern_bars"]) // 2)
        impulse_bars = max(8, consolidation_bars)
        if n < consolidation_bars + impulse_bars + 2:
            return []
        range_start = n - consolidation_bars
        impulse_start = range_start - impulse_bars
        impulse_end = range_start - 1
        atr_v = max(float(self.atr[-1]), abs(float(self.closes[-1])) * 0.002, 1e-12)
        impulse_move = float(self.closes[impulse_end] - self.closes[impulse_start])
        required_move = max(atr_v * 2.0, abs(float(self.closes[impulse_start])) * 0.03)
        if direction == "bullish" and impulse_move < required_move:
            return []
        if direction == "bearish" and impulse_move > -required_move:
            return []
        body_highs = self.highs[range_start:n - 1]
        body_lows = self.lows[range_start:n - 1]
        if not len(body_highs) or not np.isfinite(body_highs).all() or not np.isfinite(body_lows).all():
            return []
        range_high = float(np.max(body_highs))
        range_low = float(np.min(body_lows))
        if range_high - range_low > atr_v * 4.0:
            return []
        x = np.arange(len(body_highs), dtype=float)
        high_slope = float(np.polyfit(x, body_highs, 1)[0]) if len(x) >= 2 else 0.0
        low_slope = float(np.polyfit(x, body_lows, 1)[0]) if len(x) >= 2 else 0.0
        # 상승 충격 뒤에는 수평/완만한 하락, 하락 충격 뒤에는 수평/완만한 상승만 허용한다.
        slope_limit = atr_v * 0.30
        if direction == "bullish" and (high_slope > slope_limit or low_slope > slope_limit):
            return []
        if direction == "bearish" and (high_slope < -slope_limit or low_slope < -slope_limit):
            return []
        boundary = range_high if direction == "bullish" else range_low
        a = {"index": range_start, "price": boundary}
        b = {"index": n - 2, "price": boundary}
        nl = _neckline(a, b, n - 1)
        conf = _confirmation(
            self.closes, self.volumes, nl, direction=direction,
            after_index=n - int(self.cfg["breakout_confirm_bars"]) - 1,
            atr=self.atr, cfg=self.cfg,
        )
        nl["breakout_index"] = conf["index"]
        status = "confirmed" if conf["confirmed"] else "awaiting_breakout"
        invalidation = range_low - atr_v * 0.25 if direction == "bullish" else range_high + atr_v * 0.25
        status = self._lifecycle_status(
            base_status=status, direction=direction, structure_end=n - 2,
            invalidation_price=float(invalidation), breakout_index=conf["index"], neckline=nl,
        )
        pole = abs(impulse_move)
        target_price = boundary + pole if direction == "bullish" else boundary - pole
        compactness = max(0.0, min(1.0, 1.0 - (range_high - range_low) / max(atr_v * 4.0, 1e-12)))
        completion, components = _score_completion(
            pivot_fraction=1.0, geometry_fit=compactness, timing_fit=0.8,
            volume_ratio=conf["volume_ratio"],
            breakout_distance_ratio=abs(boundary - float(self.closes[-1])) / atr_v,
        )
        if status == "confirmed":
            completion = max(82.0, completion)
        impulse_pivot_type = "low" if direction == "bullish" else "high"
        impulse_end_type = "high" if direction == "bullish" else "low"
        range_extreme_index = int(range_start + (np.argmin(body_lows) if direction == "bullish" else np.argmax(body_highs)))
        points = [
            _pivot("E1", impulse_start, float(self.lows[impulse_start] if direction == "bullish" else self.highs[impulse_start]), impulse_pivot_type, self.timestamps, impulse_start),
            _pivot("E2", impulse_end, float(self.highs[impulse_end] if direction == "bullish" else self.lows[impulse_end]), impulse_end_type, self.timestamps, impulse_end),
            _pivot("E3", range_extreme_index, float(self.lows[range_extreme_index] if direction == "bullish" else self.highs[range_extreme_index]), impulse_pivot_type, self.timestamps, range_extreme_index),
            _pivot("E4", n - 1, float(self.closes[-1]), impulse_end_type, self.timestamps, n - 1),
        ]
        name = ("상승 연속형 (Flag/Pennant/Rectangle)" if direction == "bullish"
                else "하락 연속형 (Flag/Pennant/Rectangle)")
        pid = "bullish_consolidation" if direction == "bullish" else "bearish_consolidation"
        return [self._base_result(
            pattern_id=pid, name=name, family="continuation_consolidation", direction=direction,
            points=points, status=status, neckline=nl, completion=completion,
            components=components, invalidation_price=invalidation,
            target={
                "pattern_target_price": target_price,
                "pattern_target_low": target_price - atr_v * 0.35,
                "pattern_target_high": target_price + atr_v * 0.35,
                "target_method": "flagpole_projection",
                "target_confidence": round(completion * (1.0 if status == "confirmed" else 0.60), 1),
                "target_reason": None,
            },
            tolerance_basis="atr_compaction_and_close_breakout",
        )]

    def detect(self) -> list[dict[str, Any]]:
        if len(self.closes) < int(self.cfg["min_pattern_bars"]):
            return []
        if not all(np.isfinite(a[-1]) for a in (self.highs, self.lows, self.closes)):
            return []
        patterns: list[dict[str, Any]] = []
        patterns.extend(self._detect_head_shoulders("bullish"))
        patterns.extend(self._detect_head_shoulders("bearish"))
        patterns.extend(self._detect_double_triple("bullish", 2))
        patterns.extend(self._detect_double_triple("bearish", 2))
        patterns.extend(self._detect_double_triple("bullish", 3))
        patterns.extend(self._detect_double_triple("bearish", 3))
        patterns.extend(self._detect_triangle_wedge())
        patterns.extend(self._detect_consolidation("bullish"))
        patterns.extend(self._detect_consolidation("bearish"))
        patterns = deduplicate_patterns(patterns, dedupe_window=int(self.cfg["dedupe_window"]))
        # 최신·확정·완성도 순. UI와 TP 통합이 같은 대표 패턴을 사용한다.
        rank = {"confirmed": 4, "awaiting_breakout": 3, "forming": 2, "invalidated": 1, "expired": 0}
        patterns.sort(key=lambda p: (p["end_index"], rank.get(p["pattern_status"], 0), p["completion_score"]), reverse=True)
        return patterns


def _index_overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    a0, a1 = int(a["start_index"]), int(a["end_index"])
    b0, b1 = int(b["start_index"]), int(b["end_index"])
    intersection = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = max(a1, b1) - min(a0, b0) + 1
    return intersection / union if union > 0 else 0.0


def deduplicate_patterns(patterns: Sequence[Mapping[str, Any]], *, dedupe_window: int = 8) -> list[dict[str, Any]]:
    """동일 계열·방향·구간을 대표 패턴 하나와 related_patterns로 축약한다."""
    status_rank = {"confirmed": 4, "awaiting_breakout": 3, "forming": 2, "invalidated": 1, "expired": 0}
    ordered = sorted(
        (dict(p) for p in patterns),
        key=lambda p: (len(p.get("points") or []), status_rank.get(p.get("pattern_status"), 0), float(p.get("completion_score") or 0)),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        duplicate = None
        for existing in kept:
            same_family = candidate.get("family") == existing.get("family")
            same_direction = candidate.get("direction_code") == existing.get("direction_code")
            close_end = abs(int(candidate.get("end_index", 0)) - int(existing.get("end_index", 0))) <= dedupe_window
            same_neck = False
            cn, en = candidate.get("neckline") or {}, existing.get("neckline") or {}
            if cn.get("breakout_price") and en.get("breakout_price"):
                ref = max(abs(float(en["breakout_price"])), 1e-12)
                same_neck = abs(float(cn["breakout_price"]) - float(en["breakout_price"])) / ref <= 0.015
            overlap = _index_overlap(candidate, existing)
            candidate_indices = {int(p.get("index", -1)) for p in candidate.get("points") or []}
            existing_indices = {int(p.get("index", -1)) for p in existing.get("points") or []}
            is_partial_duplicate = bool(candidate_indices and candidate_indices < existing_indices)
            if same_family and same_direction and (
                is_partial_duplicate or overlap >= 0.75 or
                (close_end and (overlap >= 0.60 or same_neck))
            ):
                duplicate = existing
                break
        if duplicate is None:
            kept.append(candidate)
        else:
            related = duplicate.setdefault("related_patterns", [])
            name = candidate.get("name")
            if name and name not in related:
                related.append(name)
    return kept


def build_pattern_overlays(patterns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """탐지에 쓴 좌표를 재계산하지 않고 차트 오버레이로 변환한다."""
    overlays: list[dict[str, Any]] = []
    for pattern in patterns:
        status = pattern.get("pattern_status")
        if status == "forming" and not SHOW_FORMING_PATTERNS:
            continue
        points = [dict(p) for p in pattern.get("points") or []]
        overlay = {
            "pattern_id": pattern.get("id"), "name": pattern.get("name"),
            "status": status, "direction": pattern.get("direction_code"),
            "completion_score": pattern.get("completion_score"),
            "line_style": "solid" if status == "confirmed" else "dashed",
            "opacity": 1.0 if status == "confirmed" else 0.55,
            "points": points if SHOW_PIVOT_LABELS else [{k: p[k] for k in ("index", "timestamp", "price") if k in p} for p in points],
            "connector": [{"index": p.get("index"), "timestamp": p.get("timestamp"), "price": p.get("price")} for p in points],
            "neckline": dict(pattern.get("neckline") or {}) if SHOW_NECKLINE else None,
            "secondary_confirmation_line": dict(pattern.get("secondary_confirmation_line") or {}) if SHOW_NECKLINE else None,
            "breakout_index": (pattern.get("neckline") or {}).get("breakout_index"),
            "invalidation_price": pattern.get("invalidation_price"),
            "target": ({"price": pattern.get("pattern_target_price"),
                        "low": pattern.get("pattern_target_low"),
                        "high": pattern.get("pattern_target_high"),
                        "method": pattern.get("target_method")}
                       if SHOW_PATTERN_TARGETS and pattern.get("pattern_target_price") is not None else None),
        }
        overlays.append(overlay)
    return overlays


def integrate_pattern_targets(
    scenarios: Mapping[str, Mapping[str, Any]], patterns: Sequence[Mapping[str, Any]],
    *, current_price: float, atr_value: float, direction: str = "bullish",
) -> dict[str, Any]:
    """기존 TP1~TP5 가격은 보존하고 가까운 패턴 목표의 근거만 병합한다.

    패턴 목표가 기존 후보와 0.75 ATR 이내이면 가장 가까운 TP의 출처·확률을
    보강한다. 크게 충돌하는 후보는 강제로 삽입하지 않고 rejected 목록으로
    돌려준다. 형성 중 목표는 provisional로만 기록한다.
    """
    output = {key: dict(value) for key, value in scenarios.items()}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    candidates = []
    for pattern in patterns:
        target = _finite(pattern.get("pattern_target_price"))
        if target is None:
            continue
        code = pattern.get("direction_code")
        if code != direction:
            continue
        if direction == "bullish" and target <= current_price:
            continue
        if direction == "bearish" and target >= current_price:
            continue
        candidates.append({
            "price": target, "source": "pattern_target", "pattern_name": pattern.get("name"),
            "weight": round((float(pattern.get("completion_score") or 0) / 100.0) *
                            (1.0 if pattern.get("pattern_status") == "confirmed" else 0.55), 3),
            "distance_ratio": abs(target - current_price) / max(abs(current_price), 1e-12),
            "provisional": pattern.get("pattern_status") != "confirmed",
            "status": pattern.get("pattern_status"),
        })
    max_distance = max(abs(float(atr_value or 0)) * 0.75, abs(current_price) * 0.005)
    for candidate in candidates:
        nearest: tuple[str, int, float] | None = None
        for key, scenario in output.items():
            levels = [dict(level) for level in scenario.get("tp_levels") or []]
            scenario["tp_levels"] = levels
            for index, level in enumerate(levels):
                price = _finite(level.get("price"))
                if price is None:
                    continue
                distance = abs(price - candidate["price"])
                if nearest is None or distance < nearest[2]:
                    nearest = (key, index, distance)
        if nearest is None or nearest[2] > max_distance:
            rejected.append(dict(candidate, reason="기존 TP 후보와 거리 충돌"))
            continue
        key, index, distance = nearest
        level = output[key]["tp_levels"][index]
        sources = level.setdefault("sources", [])
        if not any(s.get("pattern_name") == candidate["pattern_name"] for s in sources):
            sources.append(candidate)
        level["source_count"] = len(sources)
        level["pattern_confluence"] = True
        level["provisional_pattern"] = bool(candidate["provisional"])
        if not candidate["provisional"] and candidate["weight"] >= 0.70:
            for field in ("prob_pct", "prob_low_pct", "prob_high_pct"):
                if _finite(level.get(field)) is not None:
                    level[field] = round(min(97.0, float(level[field]) + 2.0), 1)
        accepted.append(dict(candidate, scenario=key, tp_index=index + 1, matched_distance=round(distance, 8)))
    return {"scenarios": output, "accepted": accepted, "rejected": rejected}


def compatibility_patterns(patterns: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """기존 UI가 소비하는 name/signal/desc/conf 필드를 보존한다."""
    result = []
    for pattern in patterns:
        result.append({
            **dict(pattern),
            "name": pattern.get("name"),
            "signal": pattern.get("signal", "중립"),
            "desc": pattern.get("desc", ""),
            "conf": int(round(float(pattern.get("completion_score") or 0))),
        })
    return result
