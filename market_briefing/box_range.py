"""Causal, configurable OHLCV range analysis (general technical analysis).

Confidence is an evidence score, never a calibrated prediction probability.
Call with a completed-bar prefix for historical signals. Recent confirmation bars
are withheld from fitting; pivots use neighbours only inside that training set.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median


@dataclass(frozen=True)
class BoxRangeConfig:
    lookback: int = 80
    min_bars: int = 24
    confirmation_bars: int = 3
    min_touches: int = 3
    touch_spacing: int = 3
    min_duration: int = 15
    atr_period: int = 14
    tolerance_atr: float = .45
    min_width_atr: float = 2.5
    min_containment: float = .8
    min_reaction_ratio: float = .5
    confirmation_closes: int = 2
    recent_touch_bars: int = 30
    min_alternations: int = 3


def _plan(low=None, high=None, eligible=False, reason="근거 부족"):
    return {"low": low, "high": high, "eligible": eligible, "reason": reason}


def analyze_box_range(dd, config=None):
    cfg = config if isinstance(config, BoxRangeConfig) else BoxRangeConfig(**(config or {}))
    if (cfg.lookback < cfg.min_bars or cfg.min_bars < 8 or cfg.confirmation_bars < 2
            or cfg.min_touches < 2 or cfg.touch_spacing < 1 or cfg.min_duration < 1
            or cfg.atr_period < 1 or cfg.confirmation_closes < 1
            or cfg.recent_touch_bars < 1 or cfg.min_alternations < 1
            or cfg.confirmation_closes > cfg.confirmation_bars
            or not all(math.isfinite(x) and x > 0 for x in
                       (cfg.tolerance_atr, cfg.min_width_atr))
            or not 0 < cfg.min_containment <= 1 or not 0 <= cfg.min_reaction_ratio <= 1):
        raise ValueError("Invalid box range configuration")
    result = {"state": "insufficient_data", "support": None, "resistance": None,
              "box": None, "atr": None, "confidence": 0.0, "evidence": [],
              "entries": {"exploration": _plan(), "main": _plan()},
              "targets": {k: _plan() for k in ("conservative", "neutral", "aggressive")},
              "config": asdict(cfg)}
    try:
        arrays = [[float(x) for x in dd.get(plural, dd.get(capital, []))]
                  for plural, capital in (("opens", "Open"), ("highs", "High"),
                                          ("lows", "Low"), ("closes", "Close"),
                                          ("volumes", "Volume"))]
        if len({len(a) for a in arrays}) != 1:
            raise ValueError("misaligned arrays")
        op, hi, lo, cl, vol = arrays
        dates = dd.get("dates", dd.get("Date"))
        if dates is not None:
            normalized_dates = [str(d) for d in dates]
            if (len(dates) != len(cl) or len(set(normalized_dates)) != len(dates)
                    or normalized_dates != sorted(normalized_dates)):
                raise ValueError("dates must align, be unique and chronological")
        for o, h, l, c, v in zip(*arrays):
            if (not all(math.isfinite(x) for x in (o, h, l, c, v))
                    or min(o, h, l, c) <= 0 or v < 0 or h < max(o, c, l)
                    or l > min(o, c, h)):
                raise ValueError("invalid OHLCV")
    except (ValueError, TypeError, AttributeError) as exc:
        result["evidence"] = ["유효하지 않은 OHLCV: " + str(exc)]
        return result
    if len(cl) < cfg.min_bars + cfg.confirmation_bars:
        result["evidence"] = ["박스권 학습 및 최근 확인에 필요한 봉 부족"]
        return result
    end = len(cl) - cfg.confirmation_bars
    start = max(0, end - cfg.lookback)
    tr = [max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))
          for i in range(max(1, end-cfg.atr_period), end)]
    atr = sum(tr) / len(tr)
    result["atr"] = atr
    tolerance = max(atr * cfg.tolerance_atr, median(cl[start:end]) * 1e-6)

    def zones(side):
        prices = lo if side == "support" else hi
        pivots = [i for i in range(start+1, end-1)
                  if (prices[i] <= prices[i-1] and prices[i] <= prices[i+1]
                      if side == "support" else
                      prices[i] >= prices[i-1] and prices[i] >= prices[i+1])]
        groups = []
        for i in sorted(pivots, key=lambda j: prices[j]):
            if groups and abs(prices[i] - median(prices[j] for j in groups[-1])) <= tolerance:
                groups[-1].append(i)
            else:
                groups.append([i])
        found = []
        for group in groups:
            spaced = []
            for i in sorted(group):
                if not spaced or i-spaced[-1] >= cfg.touch_spacing:
                    spaced.append(i)
            if len(spaced) < cfg.min_touches or spaced[-1]-spaced[0] < cfg.min_duration:
                continue
            center = median(prices[i] for i in spaced)
            body = [min(op[i], cl[i]) if side == "support" else max(op[i], cl[i]) for i in spaced]
            # Wick extremes anchor the zone; nearby bodies define rejection boundary.
            boundary = median(body)
            zl = center-tolerance*.5 if side == "support" else max(center-tolerance, boundary)
            zh = min(center+tolerance, boundary) if side == "support" else center+tolerance*.5
            zl = max(center*1e-8, zl)
            zh = max(zl, zh)
            reactions = sum((max(cl[i+1:min(end, i+4)]) >= center+atr*.5
                             if side == "support" else
                             min(cl[i+1:min(end, i+4)]) <= center-atr*.5)
                            for i in spaced)
            if reactions/len(spaced) < cfg.min_reaction_ratio:
                continue
            score = min(1., .45 + .05*len(spaced) + .25*reactions/len(spaced))
            volume_baseline = median(vol[start:end])
            if volume_baseline:
                touch_volume = median(vol[i] for i in spaced)
                score = min(1., score + .03*min(1., touch_volume/volume_baseline-1))
            found.append({"low": zl, "high": zh, "center": center,
                          "touches": len(spaced), "reactions": reactions,
                          "confidence": score, "first_index": spaced[0], "last_index": spaced[-1],
                          "touch_indices": spaced})
        return found

    supports, resistances = zones("support"), zones("resistance")
    candidates = []
    for s in supports:
        for r in resistances:
            events = sorted([(i, "s") for i in s["touch_indices"]] +
                            [(i, "r") for i in r["touch_indices"]])
            alternations = sum(a[1] != b[1] for a, b in zip(events, events[1:]))
            recency_limit = min(cfg.recent_touch_bars, max(5, (end-start)//2))
            if (alternations < cfg.min_alternations or
                    end-1-min(s["last_index"], r["last_index"]) > recency_limit):
                continue
            width = r["low"]-s["high"]
            containment = sum(s["low"]-tolerance <= c <= r["high"]+tolerance
                              for c in cl[start:end])/(end-start)
            if width >= max(atr*cfg.min_width_atr, tolerance*3) and containment >= cfg.min_containment:
                candidates.append((containment*(s["confidence"]+r["confidence"])/2, s, r, containment))
    if not candidates:
        result.update(state="no_box", evidence=["반복 접촉·반응·지속 기간·가격 포함률을 충족하는 박스권 없음"])
        return result
    score, s, r, containment = max(candidates, key=lambda x: x[0])
    result.update(support=s, resistance=r, box={"low": s["low"], "high": r["high"]}, confidence=score)
    recent = cl[end:]
    upper, lower = r["high"]+tolerance, s["low"]-tolerance
    above, below = recent[-1] > upper, recent[-1] < lower
    state = "range"
    if above:
        state = "breakout_confirmed" if all(c > upper for c in recent[-cfg.confirmation_closes:]) else "breakout_pending"
    elif below:
        state = "breakdown_confirmed" if all(c < lower for c in recent[-cfg.confirmation_closes:]) else "breakdown_pending"
    elif any(c > upper for c in recent[:-1]):
        state = "false_breakout"
    elif any(lo[i] <= s["high"]+tolerance for i in range(end, len(cl))) and cl[-1] > cl[-2] and cl[-1] > s["high"]:
        state = "support_rebound"
    elif any(hi[i] >= r["low"]-tolerance for i in range(end, len(cl))) and cl[-1] < cl[-2] and cl[-1] < r["low"]:
        state = "resistance_rejection"
    result["state"] = state
    result["evidence"] = [f"지지 {s['touches']}회 / 저항 {r['touches']}회 독립 접촉",
                           f"학습 종가 포함률 {containment:.0%}; 최근 {cfg.confirmation_bars}봉 학습 제외",
                           "신뢰도는 규칙 기반 근거 점수이며 확률 아님",
                           "ATR·몸통/꼬리·종가 확인은 일반 기술적 분석 보완 규칙"]
    broken = state.startswith("breakdown")
    old_targets_valid = not broken and state != "breakout_confirmed"
    near = cl[-1] <= s["high"] + atr*1.5
    result["entries"]["exploration"] = _plan(s["low"], s["high"], not broken and near,
        "지지 영역 소액 탐색; 반등 미확인 위험" if not broken else "지지 이탈: 매수 부적합")
    main_low = s["high"]+tolerance*.25
    main_high = min(main_low+atr*.5, r["low"]-atr)
    result["entries"]["main"] = (_plan(main_low, main_high, state == "support_rebound" and near,
        "지지 접촉 후 상승 종가 확인; 지지 상단 재확인 구간" if state == "support_rebound" else "지지 반등 확인 대기")
        if main_high > main_low else _plan(reason="지지·저항 간격 부족: 본 진입 구간 미산출"))
    conservative_low = max(s["high"], r["low"]-atr)
    if old_targets_valid:
        conservative_high = r["low"]-tolerance*.25
        result["targets"]["conservative"] = (_plan(conservative_low, conservative_high, True,
            "주요 저항 도달 전 ATR 여유를 두고 수익 실현") if conservative_high > conservative_low
            else _plan(reason="저항 전 매도 구간 폭 부족"))
        result["targets"]["neutral"] = _plan(r["low"], r["high"], True,
            "반복 하락 반응이 확인된 저항 가격대")
    else:
        for strategy in ("conservative", "neutral"):
            result["targets"][strategy] = _plan(reason="박스권 돌파·이탈: 기존 저항 목표 재평가 필요")
    # Earlier observed swing highs are objective next-resistance candidates.
    historical = [hi[i] for i in range(1, start-1)
                  if hi[i] >= hi[i-1] and hi[i] >= hi[i+1] and hi[i] > max(upper, cl[-1]+tolerance)]
    if state == "breakout_confirmed" and historical:
        next_level = min(historical)
        result["targets"]["aggressive"] = _plan(next_level-tolerance*.5, next_level+tolerance*.5, True,
            "종가 돌파 확인 후 과거 관측된 다음 스윙 고점")
    else:
        result["targets"]["aggressive"] = _plan(reason="유효 돌파 및 현재가 위의 과거 다음 저항 근거 필요")
    if all(v == 0 for v in vol[max(0, end-20):]):
        result["evidence"].append("거래량 확인 불가; 거래량 확증 없음")
    else:
        baseline = median(vol[max(0, end-20):end])
        result["evidence"].append(f"최근 거래량/학습 중앙값 {vol[-1]/baseline:.2f}배" if baseline else "거래량 기준값 0")
    return result
