"""Causal trend-channel context; numeric rules are general TA supplements.

Input must contain completed bars only. Regression and confirmation thresholds
are configurable engineering choices, not thresholds stated by the video.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TrendBoxConfig:
    short_window: int = 20
    long_window: int = 60
    min_segment_bars: int = 8
    confirmation_bars: int = 2
    atr_period: int = 14
    flat_slope_atr: float = .08
    min_quality: float = .65
    breakout_atr: float = .15
    entry_band_atr: float = .5


def analyze_trend_boxes(dd, config=None):
    cfg = config if isinstance(config, TrendBoxConfig) else TrendBoxConfig(**(config or {}))
    ints = (cfg.short_window, cfg.long_window, cfg.min_segment_bars,
            cfg.confirmation_bars, cfg.atr_period)
    if (any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in ints)
            or cfg.min_segment_bars < 3 or cfg.confirmation_bars < 2
            or cfg.short_window < cfg.min_segment_bars or cfg.long_window < cfg.short_window
            or not 0 < cfg.min_quality <= 1
            or not all(math.isfinite(v) and v > 0 for v in
                       (cfg.flat_slope_atr, cfg.breakout_atr, cfg.entry_band_atr))):
        raise ValueError("Invalid trend box configuration")
    transition = lambda state, reason: {"state": state, "eligible": False, "reason": reason}
    result = {"status": "insufficient_data", "scales": {}, "atr": None,
              "transition": transition("insufficient", "완료 봉 부족"),
              "scale_conflict": False,
              "entry_zone": {"low": None, "high": None, "eligible": False, "reason": "완료 봉 부족"},
              "evidence": [], "policy_rule": "general_TA_supplement", "config": asdict(cfg)}
    try:
        arrays = [[float(v) for v in dd.get(k, dd.get(alias, []))] for k, alias in
                  (("opens", "Open"), ("highs", "High"), ("lows", "Low"),
                   ("closes", "Close"), ("volumes", "Volume"))]
        if len({len(a) for a in arrays}) != 1:
            raise ValueError("misaligned arrays")
        op, hi, lo, cl, vol = arrays
        dates = dd.get("dates", dd.get("Date"))
        if dates is not None:
            ds = [str(v) for v in dates]
            if len(ds) != len(cl) or len(set(ds)) != len(ds) or ds != sorted(ds):
                raise ValueError("dates must align and be unique chronological values")
        for o, h, l, c, v in zip(*arrays):
            if (not all(math.isfinite(x) for x in (o, h, l, c, v)) or min(o, h, l, c) <= 0
                    or v < 0 or h < max(o, c, l) or l > min(o, c, h)):
                raise ValueError("invalid OHLCV")
    except (ValueError, TypeError, AttributeError) as exc:
        result.update(status="invalid_data", evidence=[str(exc)])
        return result
    end = len(cl) - cfg.confirmation_bars
    if end < 2 * cfg.min_segment_bars:
        return result
    tr = [max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
          for i in range(max(1, end-cfg.atr_period), end)]
    atr = max(sum(tr)/len(tr), cl[end-1]*1e-8)
    result.update(status="valid", atr=atr)

    def fit(start, stop):
        n = stop-start
        mean_x = (n-1)/2
        mean_y = sum(cl[start:stop])/n
        slope = sum((j-mean_x)*(cl[start+j]-mean_y) for j in range(n)) / sum((j-mean_x)**2 for j in range(n))
        intercept = mean_y-slope*mean_x
        lows = sorted(lo[start+j]-(intercept+slope*j) for j in range(n))
        highs = sorted(hi[start+j]-(intercept+slope*j) for j in range(n))
        # Robust residual bounds omit at most ten percent of isolated extremes.
        offset_low, offset_high = lows[int((n-1)*.1)], highs[math.ceil((n-1)*.9)]
        containment = sum(offset_low <= cl[start+j]-(intercept+slope*j) <= offset_high for j in range(n))/n
        width = offset_high-offset_low
        quality = containment * min(1., n/cfg.min_segment_bars) * min(1., width/atr)
        direction = "flat" if abs(slope/atr) <= cfg.flat_slope_atr else ("ascending" if slope > 0 else "descending")
        at_latest = intercept+slope*(len(cl)-1-start)
        at_endpoint = intercept+slope*(n-1)
        return {"direction": direction, "lower": at_latest+offset_low, "upper": at_latest+offset_high,
                "endpoint_lower": at_endpoint+offset_low, "endpoint_upper": at_endpoint+offset_high,
                "slope": slope, "slope_atr": slope/atr, "containment": containment,
                "quality": quality, "start_index": start, "end_index": stop-1}

    for name, window in (("short", cfg.short_window), ("long", cfg.long_window)):
        if end < 2*window:
            result["scales"][name] = {"previous": None, "current": None,
                                      "transition": transition("insufficient", "요청한 기간의 이전·현재 박스 봉 부족")}
            continue
        n = window
        previous, current = fit(end-2*n, end-n), fit(end-n, end)
        recent = cl[end:]
        reliable = (min(previous["quality"], current["quality"]) >= cfg.min_quality
                    and all(0 < c["lower"] <= c["upper"] for c in (previous, current)))
        if not all(0 < c["lower"] <= c["upper"] for c in (previous, current)):
            result["scales"][name] = {"previous": None, "current": None,
                                      "transition": transition("unconfirmed", "현재 시점 채널 투영 가격이 유효하지 않음")}
            continue
        above = all(c > previous["endpoint_upper"]+cfg.breakout_atr*atr for c in recent)
        below = all(c < previous["endpoint_lower"]-cfg.breakout_atr*atr for c in recent)
        recovery_failed = (previous["direction"] == "ascending" and
                           (current["direction"] == "descending" or below))
        # Recovery support is fitted before any confirmation bar is observed.
        support_failed = all(c < current["endpoint_lower"]-cfg.breakout_atr*atr for c in recent)
        if not reliable:
            t = transition("unconfirmed", "채널 근거 부족")
        elif recovery_failed or (current["direction"] == "ascending" and support_failed):
            t = transition("failed_upward_reversal", "회복 채널 지지 이탈 또는 상승에서 하락으로 전환")
        elif current["direction"] == "ascending" and previous["direction"] != "ascending" and above:
            t = transition("upward_reversal", "이전 박스 상단을 완료 종가로 확인하고 상승 채널 형성")
            t["eligible"] = True
        elif current["direction"] == "descending" and previous["direction"] != "descending" and below:
            t = transition("downward_reversal", "이전 박스 하단을 완료 종가로 이탈")
        elif current["direction"] == previous["direction"] and not support_failed:
            t = transition("continuation", "동일 방향 채널 지속")
        else:
            t = transition("unconfirmed", "방향 변화 또는 돌파의 완료 종가 확인 부족")
        result["scales"][name] = {"previous": previous, "current": current, "transition": t}
    short, long = result["scales"]["short"], result["scales"]["long"]
    result["transition"] = dict(short["transition"])
    if short["current"] is None:
        result["status"] = "insufficient_data" if short["transition"]["state"] == "insufficient" else "unconfirmed"
        result["evidence"] = [short["transition"]["reason"]]
        return result
    conflict = (short["current"]["direction"] == "ascending" and long["current"] is not None
                and long["current"]["direction"] == "descending")
    result["scale_conflict"] = conflict
    channel = short["current"]
    low = max(channel["lower"], short["previous"]["endpoint_upper"])
    high = min(channel["upper"], low+cfg.entry_band_atr*atr)
    prior_resistance = short["previous"]["endpoint_upper"]
    retest = (abs(channel["lower"]-prior_resistance) <= cfg.entry_band_atr*atr
              and low > 0 and low <= high
              and any(lo[i] <= prior_resistance+cfg.entry_band_atr*atr
                      and hi[i] >= prior_resistance-cfg.breakout_atr*atr for i in range(end, len(cl)))
              and all(c >= low-cfg.breakout_atr*atr for c in cl[end:])
              and cl[-1] >= cl[-2] and cl[-1] <= high+cfg.entry_band_atr*atr)
    eligible = short["transition"]["eligible"] and retest and not conflict
    reason = ("상승 전환 후 이전 저항 재확인 및 지지 반응" if eligible else
              "장기 하락 채널과 충돌" if conflict else "상승 전환·돌파 재확인·현재 가격 거리 조건 미충족")
    result["entry_zone"] = {"low": low if low > 0 and low <= high else None,
                            "high": high if low > 0 and low <= high else None,
                            "eligible": eligible, "reason": reason}
    if conflict:
        result["transition"]["eligible"] = False
    result["evidence"] = [short["transition"]["reason"], reason,
                          "회귀 채널·ATR 허용폭·확인 봉 수는 영상 외 일반 기술적 분석 보완 조건"]
    return result
