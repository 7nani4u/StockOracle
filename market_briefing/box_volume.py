"""Completed-bar volume evidence for the existing box execution plan.

Volume is compared with the sessions *before* the event. A retest's quiet volume
must not erase the volume observed at the original breakout. No price levels,
target probabilities, position sizes or forecasts are manufactured here.
"""
from __future__ import annotations

import math
from statistics import mean

VERSION = "2026-11-11.box-volume-v2"
POLICIES = ("off", "breakout", "balanced", "recovery")


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def build_volume_evidence(opens, highs, lows, closes, volumes, reference=None, *, dates=None, lookback=20, origin=0):
    """Use completed input bars only; the caller already removes a forming bar."""
    n = len(closes)
    out = {"version": VERSION, "available": False, "lookback": lookback,
           "basis": "완료 일봉 · 사건 이전 20거래일 평균 대비",
           "last_date": str(dates[-1])[:10] if dates is not None and len(dates) == n else None,
           "latest_ratio": None, "breakout": None, "pullback_ratio": None,
           "close_location": None, "reason": None}
    if n-origin < lookback+1 or any(len(x) != n for x in (opens, highs, lows, volumes)):
        return dict(out, reason="완료 거래량 21봉 또는 OHLCV 정렬 부족")

    def volume_window(start, end):
        values = [_finite(v) for v in volumes[start:end]]
        return values if len(values) == end-start and all(v is not None and v >= 0 for v in values) else None

    base = volume_window(n-1-lookback, n-1)
    latest = _finite(volumes[-1])
    if base is None or latest is None or latest < 0 or mean(base) <= 0:
        return dict(out, reason="유효한 과거 평균 거래량 없음")
    h, l, c = map(_finite, (highs[-1], lows[-1], closes[-1]))
    if h is None or l is None or c is None or h < l or not l <= c <= h:
        return dict(out, reason="확인 봉 가격 범위 오류")
    out.update(available=True, latest_ratio=latest/mean(base),
               latest_volume=latest, average_volume=mean(base),
               close_location=(c-l)/(h-l) if h > l else .5,
               latest_return_pct=(c/closes[-2]-1)*100 if closes[-2] > 0 else None)
    recent_volume=volume_window(n-5,n)
    flow=[]
    for j in range(n-5,n):
        hh,ll,cc=map(_finite,(highs[j],lows[j],closes[j]))
        if hh is None or ll is None or cc is None or hh < ll or not ll <= cc <= hh:
            flow=[]
            break
        flow.append((2*cc-hh-ll)/(hh-ll) if hh>ll else 0.)
    out["money_flow_5"] = (sum(f*v for f,v in zip(flow,recent_volume))/sum(recent_volume)
        if len(flow)==5 and recent_volume and sum(recent_volume)>0 else None)
    ref = reference or {}
    k = ref.get("exit_index")
    if ref.get("exit") != "up" or not isinstance(k, int) or not 0 <= k < n:
        return out
    if k-origin < lookback:
        out["breakout"]={"available":False,"reason":"새 가격 구간의 돌파 이전 거래량 20봉 부족"}
        return out
    before = volume_window(k-lookback, k)
    # The first two confirmation sessions are fixed event observations. Extending
    # the input after the second bar cannot silently change their denominator.
    count = min(2, n-k)
    confirmation = volume_window(k, k+count)
    if before is None or confirmation is None or mean(before) <= 0:
        out["breakout"] = {"available": False, "reason": "돌파 이전 거래량 기준 부족"}
        return out
    average, event_volume = mean(before), mean(confirmation)
    out["breakout"] = {"available": True, "index": k, "bars_since": n-k,
        "date": str(dates[k])[:10] if dates is not None and len(dates) == n else None,
        "initial_ratio": confirmation[0]/average, "confirmation_ratio": event_volume/average,
        "confirmation_bars": count, "average_before": average, "average_event": event_volume}
    pullback = volume_window(max(k+count, n-3), n)
    if pullback and event_volume > 0:
        out["pullback_ratio"] = mean(pullback)/event_volume
    return out


def assess_box_volume(profile, strategy, policy="breakout"):
    """Fixed, auditable policies; thresholds are not fitted to the current ticker."""
    if policy not in POLICIES:
        raise ValueError("알 수 없는 박스 거래량 정책")
    result = {"policy": policy, "passed": True, "status": "not_applicable", "reason": ""}
    if policy == "off":
        return dict(result, status="disabled")
    if not isinstance(profile, dict):
        # Old serialized sequences can still be read, without claiming evidence.
        return dict(result, passed=None, status="not_supplied", reason="이전 응답: 거래량 근거 없음")
    if not profile.get("available"):
        return dict(result, passed=False, status="unavailable", reason=profile.get("reason") or "거래량 미확인")
    ratio = profile["latest_ratio"]
    if profile.get("latest_volume", 1) <= 0:
        return dict(result, passed=False, status="unavailable", reason="확인 일봉 거래량 0 — 거래 재개 확인 필요")
    if strategy in ("breakout", "retest"):
        event = profile.get("breakout") or {}
        value = event.get("confirmation_ratio")
        if not event.get("available") or value is None:
            return dict(result, passed=None, status="waiting", reason="돌파 발생 후 완료된 확인 봉의 거래량 대기")
        passed = value >= 1.0
        if policy == "recovery":
            passed = passed or (profile.get("pullback_ratio") is not None and
                profile["pullback_ratio"] <= .7 and profile.get("close_location", 0) >= .5)
        return dict(result, passed=passed, status="confirmed" if passed else "weak",
                    reason=f"돌파 확인 {event['confirmation_bars']}봉 평균 거래량 {value:.2f}배 (돌파 직전 20봉 대비)")
    if strategy == "support" and policy in ("balanced", "recovery"):
        passed = ratio >= .8 and profile.get("close_location", 0) >= .5
        return dict(result, passed=passed, status="confirmed" if passed else "weak",
                    reason=f"지지 확인 거래량 {ratio:.2f}배 · 종가 위치 {profile.get('close_location', 0):.0%}")
    return dict(result, reason=f"완료 봉 거래량 {ratio:.2f}배 · 지지 가격 조건과 함께 참고")


def allows_box_entry(assessment):
    """The service and replay share exactly the same treatment of missing evidence."""
    return assessment.get("passed") is True or assessment.get("status") in ("disabled","not_supplied","not_applicable")
