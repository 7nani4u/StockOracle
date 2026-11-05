"""Box execution plan: one decision with prices, wait conditions and a cancel rule.

The plan answers six questions for the 🧭 박스 실행 플랜 card:
1. can I buy now, 2. at what price, 3. what price/signal am I waiting for,
4. how long might that take (or what to check instead), 5. where are the stop and
targets, 6. when is the plan cancelled.

Sources and what they contribute
--------------------------------
* Video ``NOIOQP-7PPM`` (verified 2026-11-05, auto captions + key frames): falling boxes
  are not bought; the inflection is where a box stops falling and a box steps up; a failed
  upward step is the stop; small boxes are read against the big box.  No numbers.
* Darvas box rule (3-bar top/bottom confirmation) — implemented in ``box_sequence``.
* Range trading (buy near support, sell near resistance) and the measured move
  (box height projected from the breakout) — general technical analysis.
* Existing StockOracle constants reused unchanged: support stop = bottom - 0.35 ATR,
  breakout stop = broken top - 0.6 ATR, 2 confirmation closes, minimum box height
  2.5 ATR (``box_tactics``/``box_range``).
* Wait-time numbers come from the project's driftless touch model
  (``forecast_model.touch_probability_range``/``touch_time_window``), the same model the
  entry bands use (validated Brier 0.190 on 52 tickers).  They are volatility estimates,
  not direction forecasts.

The module is long-only, pure Python and never raises: on any failure it returns an
``unavailable`` plan that says why.
"""
from __future__ import annotations

import math
from datetime import date, datetime

from .forecast_model import blended_daily_sigma, touch_probability_range, touch_time_window

VERSION = "2026-11-05.box-plan-v1"

PLAN_PARAMS = {
    "support_zone_atr": 0.5,       # support buy zone: bottom .. bottom + min(0.25 height, 0.5 ATR)
    "support_zone_frac": 0.25,
    "support_stop_atr": 0.35,      # box_tactics range_support stop
    "breakout_stop_atr": 0.6,      # box_tactics breakout stop
    "retest_zone_atr": 0.5,        # trend_boxes entry band width
    "break_tol_atr": 0.10,         # box_sequence break tolerance
    "confirm_closes": 2,           # box_range confirmation_closes
    "min_height_atr": 2.5,         # box_range min_width_atr
    "min_reward_risk": 1.0,        # final target must at least pay for the stop
    "horizon_days": 20,            # touch-probability horizon (about one month)
    "stale_days": 10,              # calendar days after which bars are treated as stale
    "max_quote_gap": 0.35,         # quote vs last completed close (KRX daily limit is 30%)
    # Switches kept for the walk-forward audit (scripts/audit_box_plan.py); defaults are the
    # service behaviour.
    "block_descending_support": True,  # video: a falling box's bottom is not a buy point
    "require_signal": True,            # a completed-bar confirmation is required before "매수"
}

VIDEO_SOURCE = {
    "url": "https://www.youtube.com/watch?v=NOIOQP-7PPM",
    "title": "모르면 평생 호구 되는 박스권 추세 전환 비밀",
    "reviewed_on": "2026-11-05",
    "method": "한국어 자동 자막 전체(79초) + 0:06~1:17 주요 장면 캡처",
    "stated": [
        "박스는 하락·횡보·우상향 세 종류이고, 박스 성격이 바뀌는 자리를 변곡으로 삼는다 (0:00~0:24)",
        "하락은 박스 이탈이 이어지는 계단이고, 더 이상 빠지지 않는 박스 뒤 올라가는 박스가 나오면 추세 전환 (0:30~0:43)",
        "위로 올라가다 실패해 다시 내려오면 손절 (0:45~0:47, 도식의 '손절' 표시)",
        "작은 박스의 전환과 큰 박스(하락 중)의 전환 자리를 함께 보고 그때부터 공략 (0:53~1:03)",
    ],
    "not_stated": [
        "박스 상·하단 산식, 접촉 횟수, 거래량·캔들 조건, 손절 가격 산식, 목표가, 대기 기간",
    ],
}

SUPPLEMENTS = [
    "Darvas 3봉 규칙: 고점(저점)을 이후 3봉이 넘지(깨지) 않으면 박스 상단(하단) 확정",
    "일반 박스 매매: 하단 지지 매수·상단 저항 매도, 상단 돌파 시 박스 높이만큼 측정 목표",
    "기존 StockOracle 규칙 재사용: 지지 손절 −0.35 ATR, 돌파 손절 −0.6 ATR, 확인 종가 2개, 최소 박스 폭 2.5 ATR",
    "대기 기간: 무추세 변동성 터치 모델(방향 예측 아님)",
]

_VERDICTS = {
    "buy": ("🟢", "매수", "진입 조건 충족"),
    "conditional": ("🟡", "조건부 매수", "가격·신호 대기"),
    "watch": ("🟠", "관망", "매수 조건 미충족"),
    "exclude": ("🔴", "매수 제외", "박스 이탈·계획 취소"),
    "unavailable": ("⚪", "산출 불가", "근거 데이터 부족"),
}

_STAGES = {
    "ready": "신호 확인 완료",
    "signal_wait": "가격 도달 · 신호 확인 중",
    "price_wait": "가격 도달 대기",
    "not_met": "조건 미충족",
    "cancelled": "계획 취소",
    "no_data": "산출 불가",
}

_STRATEGIES = {
    "support": "하단 지지 매수",
    "breakout": "상단 돌파 매수",
    "retest": "돌파 후 재확인 매수",
    "none": "진입 전략 없음",
}


# ── price helpers ──────────────────────────────────────────────────────────────

def _finite(value):
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def tick_size(price, market="KRX"):
    """Exchange tick. KRX unified KOSPI/KOSDAQ ticks on 2023-01-25."""
    p = _finite(price) or 0.0
    if str(market).upper() == "US":
        return 0.0001 if 0 < p < 1.0 else 0.01
    if p < 2_000:
        return 1
    if p < 5_000:
        return 5
    if p < 20_000:
        return 10
    if p < 50_000:
        return 50
    if p < 200_000:
        return 100
    if p < 500_000:
        return 500
    return 1_000


def round_tick(price, market="KRX", mode="nearest"):
    value = _finite(price)
    if value is None or value <= 0:
        return None
    tick = tick_size(value, market)
    units = value / tick
    nearest = round(units)
    if abs(units - nearest) <= 1e-9 * max(1.0, abs(units)):
        units = float(nearest)
    if mode == "floor":
        units = math.floor(units)
    elif mode == "ceil":
        units = math.ceil(units)
    else:
        units = round(units)
    out = max(tick, units * tick)
    if str(market).upper() == "US":
        return round(out, 4 if tick < 0.01 else 2)
    return int(round(out))


def _pct(price, base):
    price, base = _finite(price), _finite(base)
    if price is None or base is None or base <= 0:
        return None
    return round((price / base - 1) * 100, 2)


def _zone(low, high, market):
    lo = round_tick(low, market, "ceil")
    hi = round_tick(high, market, "floor")
    if lo is None or hi is None:
        return None
    if hi < lo:
        mid = round_tick((low + high) / 2, market, "nearest")
        lo = hi = mid
    return [lo, hi]


def _parse_day(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


# ── plan skeleton ──────────────────────────────────────────────────────────────

def _verdict(key, headline=None):
    icon, label, default = _VERDICTS[key]
    return {"key": key, "icon": icon, "label": label, "headline": f"{label} — {headline or default}"}


def _stage(key):
    return {"key": key, "label": _STAGES[key]}


def _skeleton():
    return {
        "version": VERSION,
        "available": False,
        "verdict": _verdict("unavailable"),
        "stage": _stage("no_data"),
        "strategy": {"key": "none", "label": _STRATEGIES["none"]},
        "action": "",
        "when": "",
        "cancel": "",
        "quote": None,
        "box": None,
        "staircase": None,
        "big_box": None,
        "prices": None,
        "alternative": None,
        "timing": None,
        "conditions": [],
        "invalidation": [],
        "evidence": [],
        "warnings": [],
        "params": dict(PLAN_PARAMS),
        "sources": {"video": VIDEO_SOURCE, "supplements": SUPPLEMENTS},
    }


def _unavailable(out, reason, need, review="데이터가 바뀌면 다음 조회 때 다시 계산합니다"):
    """Data-side failure only: ``reason`` says what is wrong, ``need`` what to check or wait for.

    Every market-side state (box broken, no box yet, failed breakout) has its own price path and
    never ends here; if it does, that is a bug the six-question audit reports.
    """
    out["verdict"] = _verdict("unavailable", reason)
    out["stage"] = _stage("no_data")
    out["action"] = f"{reason}. 이 상태에서는 가격을 만들면 근거 없는 숫자가 되므로 계산하지 않습니다."
    out["when"] = need
    out["cancel"] = "아직 세운 매수 계획이 없어 취소할 계획도 없습니다. 데이터가 복구되면 새로 계산합니다."
    out["prices"] = {"entry": None, "trigger": None, "add": None, "stop": None, "targets": [], "rr": None,
                     "rr_t1": None, "entry_ref": None, "overhead": None, "holder_stop": None,
                     "entry_unavailable": reason + " — " + need}
    out["conditions"] = _conditions(None, None, None, "데이터 문제로 판단 불가", "데이터 문제로 판단 불가", "데이터 문제로 판단 불가")
    out["timing"] = {"immediate": False, "confirm_bars": None, "confirm_rule": need, "touch": None,
                     "estimate_label": "데이터 문제가 풀리기 전에는 시점을 말할 수 없습니다. 확인: " + need,
                     "review": review, "expiry_bars": None}
    out["invalidation"] = ["세운 계획이 없음 — 데이터 복구 후 새로 계산"]
    return out


# ── geometry builders ──────────────────────────────────────────────────────────

def _targets(entry_ref, stop, levels, market, base):
    risk = entry_ref - stop
    out = []
    for price, label, basis in levels:
        p = round_tick(price, market, "floor")
        if p is None or p <= entry_ref:
            continue
        if out and p <= out[-1]["price"]:
            continue
        out.append({"price": p, "label": label, "basis": basis,
                    "pct_from_entry": _pct(p, entry_ref), "pct_from_quote": _pct(p, base),
                    "rr": round((p - entry_ref) / risk, 2) if risk > 0 else None})
    return out


def _overhead(boxes, reference, above, below):
    """Nearest edge of an older box between ``above`` and ``below`` (overhead supply)."""
    best = None
    for box in boxes:
        if box is reference:
            continue
        for edge in (box.get("bottom"), box.get("top")):
            edge = _finite(edge)
            if edge is not None and above < edge < below and (best is None or edge < best):
                best = edge
    return best


def _support_geometry(box, atr, market, quote, p):
    top, bottom = box["top"], box["bottom"]
    height = top - bottom
    zone_high = bottom + min(p["support_zone_frac"] * height, p["support_zone_atr"] * atr)
    zone = _zone(bottom, zone_high, market)
    if zone is None:
        return None
    entry_ref = (zone[0] + zone[1]) / 2
    stop = round_tick(bottom - p["support_stop_atr"] * atr, market, "floor")
    if stop is None or stop >= zone[0]:
        stop = round_tick(zone[0] - tick_size(zone[0], market), market, "floor")
    targets = _targets(entry_ref, stop, [
        ((top + bottom) / 2, "1차 목표 · 박스 중간", "박스 상·하단의 중간값(되돌림 1차 매물)"),
        (top, "2차 목표 · 박스 상단", "박스 상단 저항(3봉 확인 고점)"),
    ], market, quote)
    risk = entry_ref - stop
    final = targets[-1] if targets else None
    return {
        "key": "support",
        "entry": {"low": zone[0], "high": zone[1], "ref": round_tick(entry_ref, market, "nearest"),
                  "label": "1차 매수 후보", "basis":
                  f"박스 하단 {_fmt(bottom, market)} ~ 하단+min(박스 높이 25%, 0.5 ATR)"},
        "stop": {"price": stop, "pct_from_entry": _pct(stop, entry_ref), "pct_from_quote": _pct(stop, quote),
                 "basis": "박스 하단 − 0.35 ATR (하단 이탈 = 지지 실패)"},
        "targets": targets,
        "risk": risk,
        "rr": final["rr"] if final else None,
        "rr_t1": targets[0]["rr"] if targets else None,
        "entry_ref": entry_ref,
    }


def _breakout_geometry(box, atr, market, quote, p, boxes):
    top, bottom = box["top"], box["bottom"]
    height = top - bottom
    trigger = round_tick(top + p["break_tol_atr"] * atr, market, "ceil")
    zone = _zone(top, top + p["retest_zone_atr"] * atr, market)
    if zone is None or trigger is None:
        return None
    entry_ref = (zone[0] + zone[1]) / 2
    stop = round_tick(top - p["breakout_stop_atr"] * atr, market, "floor")
    measured = top + height
    half = top + height / 2
    # An older box edge between the entry and the measured target is overhead supply. It is
    # reported as a warning, not used as a target: on 2026-11-05 live data it produced a
    # first target 0.36R above the entry, too close to pay for half the position.
    overhead = _overhead(boxes, box, entry_ref + 0.25 * atr, measured)
    levels = [(half, "1차 목표 · 측정 이동 50%", "박스 상단 + 박스 높이의 절반"),
              (measured, "2차 목표 · 측정 이동 100%", "박스 상단 + 박스 높이(측정 이동)")]
    targets = _targets(entry_ref, stop, levels, market, quote)
    final = targets[-1] if targets else None
    return {
        "key": "retest",
        "trigger": {"price": trigger, "pct_from_quote": _pct(trigger, quote), "label": "돌파 확인 가격",
                    "condition": f"일봉 종가 {_fmt(trigger, market)} 이상 {p['confirm_closes']}회 연속 = 돌파 확인"},
        "entry": {"low": zone[0], "high": zone[1], "ref": round_tick(entry_ref, market, "nearest"),
                  "label": "돌파 후 재확인 매수 구간",
                  "basis": f"깨진 상단 {_fmt(top, market)} ~ 상단+0.5 ATR (저항→지지 전환 확인 자리)"},
        "stop": {"price": stop, "pct_from_entry": _pct(stop, entry_ref), "pct_from_quote": _pct(stop, quote),
                 "basis": "깨진 상단 − 0.6 ATR (영상: 올라가다 실패해 되돌아오면 손절)"},
        "targets": targets,
        "risk": entry_ref - stop,
        "rr": final["rr"] if final else None,
        "rr_t1": targets[0]["rr"] if targets else None,
        "entry_ref": entry_ref,
        "overhead": round_tick(overhead, market, "floor") if overhead is not None else None,
    }


def _conditional(geometry, kind, label, condition):
    """Mark a geometry's buy zone as a *candidate*: shown with prices, but not buyable yet."""
    if not geometry:
        return None
    out = dict(geometry)
    # ``below_zone_pct`` (live quote already under a zone built from completed bars = a new low is in
    # progress) is reported once, as a plan warning, by the callers — not repeated in the condition.
    out["entry"] = {**geometry["entry"], "kind": kind, "conditional": True, "label": label, "condition": condition}
    return out


def _reclaim_geometry(box, atr, market, quote, p):
    """Failed-breakdown reclaim: the broken bottom must turn back into support.

    Mirror of the breakout retest: confirm closes above the old bottom, then buy the pull-back to
    ``bottom .. bottom + 0.5 ATR``. Stop = bottom - 0.6 ATR (reclaim fails), targets = old box
    mid and top. Weak evidence (Bulkowski: about 59 % of pull-backs to a broken level fall again),
    so it is only ever shown as a conditional alternative.
    """
    top, bottom = box["top"], box["bottom"]
    trigger = round_tick(bottom + p["break_tol_atr"] * atr, market, "ceil")
    zone = _zone(bottom, bottom + p["retest_zone_atr"] * atr, market)
    stop = round_tick(bottom - p["breakout_stop_atr"] * atr, market, "floor")
    if trigger is None or zone is None or stop is None or stop >= zone[0]:
        return None
    entry_ref = (zone[0] + zone[1]) / 2
    targets = _targets(entry_ref, stop, [
        ((top + bottom) / 2, "1차 목표 · 이전 박스 중간", "이탈 전 박스의 중간값(되돌림 1차 매물)"),
        (top, "2차 목표 · 이전 박스 상단", "이탈 전 박스 상단(전체 복귀)"),
    ], market, quote)
    if not targets:
        return None
    final = targets[-1]
    return {
        "key": "reclaim",
        "trigger": {"price": trigger, "pct_from_quote": _pct(trigger, quote), "label": "이탈 실패 확인가",
                    "condition": f"일봉 종가 {_fmt(trigger, market)} 이상 {p['confirm_closes']}회 = 이탈 실패(되돌림) 확인"},
        "entry": {"low": zone[0], "high": zone[1], "ref": round_tick(entry_ref, market, "nearest"),
                  "basis": f"이탈한 하단 {_fmt(bottom, market)} ~ 하단+0.5 ATR (지지 회복 재확인 자리)"},
        "stop": {"price": stop, "pct_from_entry": _pct(stop, entry_ref), "pct_from_quote": _pct(stop, quote),
                 "basis": "이탈한 하단 − 0.6 ATR (회복 실패)"},
        "targets": targets, "risk": entry_ref - stop, "rr": final["rr"], "rr_t1": targets[0]["rr"],
        "entry_ref": entry_ref,
    }


def _forming_geometry(forming, atr, market, quote, p):
    """Provisional support-style plan on the box that is forming right now (not confirmed yet).

    Same rules as a confirmed box (zone = bottom .. bottom + min(25 % of height, 0.5 ATR), stop =
    bottom - 0.35 ATR, targets = middle and top); the prices move whenever a new low or high prints,
    which the card says explicitly.
    """
    if not forming:
        return None
    top, bottom = _finite(forming.get("top")), _finite(forming.get("bottom"))
    if top is None or bottom is None or bottom <= 0 or top <= bottom:
        return None
    geometry = _support_geometry({"top": top, "bottom": bottom}, atr, market, quote, p)
    if geometry is None:
        return None
    geometry["height_atr"] = round((top - bottom) / atr, 2)
    geometry["bars_to_confirm"] = _bars_to_confirm(forming)
    low = geometry["entry"]["low"]
    geometry["below_zone_pct"] = round((low / quote - 1) * 100, 1) if quote and quote < low * 0.995 else None
    trigger = round_tick(top + p["break_tol_atr"] * atr, market, "ceil")
    geometry["trigger"] = {"price": trigger, "pct_from_quote": _pct(trigger, quote), "label": "돌파 확인가(잠정)",
                           "condition": f"새 박스 상단 {_fmt(top, market)} 위 종가 {p['confirm_closes']}회"}
    return geometry


def _bars_to_confirm(forming):
    """Fewest additional completed bars before the forming range can be a Darvas box."""
    if not forming:
        return None
    top_age = forming.get("top_age")
    bottom_age = forming.get("bottom_age")
    bars = forming.get("bars")
    if None in (top_age, bottom_age, bars):
        return None
    return int(max(0, 3 - top_age, 3 - bottom_age, 5 - bars))


def _fmt(value, market):
    value = _finite(value)
    if value is None:
        return "-"
    if str(market).upper() == "US":
        return f"${value:,.2f}" if value >= 1 else f"${value:,.4f}"
    return f"{value:,.0f}원"


_JOSA = {"은": ("은", "는"), "을": ("을", "를"), "이": ("이", "가"), "으로": ("으로", "로")}


def _batchim(text):
    """(has final consonant, final is ㄹ) for the last character as read in Korean."""
    ch = str(text).rstrip()[-1:]
    if "가" <= ch <= "힣":
        final = (ord(ch) - 0xAC00) % 28
        return final != 0, final == 8
    if ch.isdigit():
        return ch in "013678", ch in "178"
    return False, False


def _dist(target, quote):
    """'3.7% 아래' / '8.1% 위' — how far the target is from the quote."""
    pct = _pct(target, quote)
    if pct is None:
        return "-"
    if abs(pct) < 0.005:
        return "도달"
    return f"{abs(pct):.2f}% {'위' if pct > 0 else '아래'}"


def _pj(value, market, josa):
    """Price text plus the particle that matches its pronunciation (원은 / $12.5는)."""
    text = _fmt(value, market)
    has, rieul = _batchim(text)
    with_final, without = _JOSA[josa]
    if josa == "으로":
        return text + ("으로" if has and not rieul else "로")
    return text + (with_final if has else without)


def _timing_touch(quote, level, closes, atr, horizon):
    if quote is None or level is None or quote <= 0 or level <= 0:
        return None
    sigma = blended_daily_sigma(closes or [], quote, atr).get("sigma")
    if not sigma:
        return None
    p5 = touch_probability_range(quote, level, sigma, 5)
    p20 = touch_probability_range(quote, level, sigma, horizon)
    window = touch_time_window(quote, level, sigma, horizon)

    def pct(r):
        return None if r is None else {k: round(v * 100) for k, v in r.items()}

    return {
        "level": level,
        "distance_pct": _pct(level, quote),
        "p5": pct(p5),
        "p_horizon": pct(p20),
        "horizon_days": horizon,
        "days_if_reached": window.get("days") if window.get("within_horizon") else None,
        "sigma_daily_pct": round(sigma * 100, 2),
        "basis": "무추세 변동성 터치 모델(방향 예측 아님) · 25~75% 변동성 오차 범위",
    }


def _touch_text(touch):
    if not touch or not touch.get("p_horizon"):
        return "가격 도달 시점은 현재 데이터로 추정할 수 없습니다."
    p5, ph = touch.get("p5") or {}, touch["p_horizon"]
    days = touch.get("days_if_reached")

    def share(value):
        return "95% 넘게" if value >= 95 else f"약 {value}%"

    if p5.get("mid", 0) >= 90:
        text = (f"현재가가 그 가격에 매우 가까워 변동성만으로 보면 5거래일 안에 닿을 가능성이 {share(p5.get('mid', 0))}입니다")
    else:
        text = (f"변동성만으로 보면 5거래일 안에 닿을 가능성 {share(p5.get('mid', 0))}, "
                f"{touch['horizon_days']}거래일 안 {share(ph.get('mid', 0))}({ph.get('low', 0)}~{ph.get('high', 0)}%)")
        if days:
            text += (f", 닿는다면 대략 {days[0]}거래일째" if days[0] == days[1] else f", 닿는다면 대략 {days[0]}~{days[1]}거래일째")
    return text + ". 방향을 예측한 값이 아니므로 날짜를 정해 매수하지 말고 가격·신호를 확인하세요."


# ── main entry ─────────────────────────────────────────────────────────────────

def six_questions(plan):
    """Which of the six decision questions a plan answers (used by tests and the audit).

    q1 buy now? · q2 which price? · q3 what price/signal to wait for? · q4 how long, or what to
    check? · q5 stop and profit targets? · q6 when to cancel? A data-side ``unavailable`` plan is
    complete only if it names the reason and what to check; it never carries stop/targets.
    """
    p = plan or {}
    verdict = p.get("verdict") or {}
    prices = p.get("prices") or {}
    entry = prices.get("entry") or {}
    stop = prices.get("stop") or {}
    targets = prices.get("targets") or []
    timing = p.get("timing") or {}
    data_side = verdict.get("key") == "unavailable"

    def num(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    answers = {
        "q1_buy_now": bool(verdict.get("key") and verdict.get("headline")),
        "q2_price": (num(entry.get("low")) and num(entry.get("high"))) or (data_side and bool(prices.get("entry_unavailable"))),
        "q3_wait_for": bool(p.get("when")) and (num((prices.get("trigger") or {}).get("price"))
                                                or num(entry.get("low")) or bool(timing.get("confirm_rule"))),
        "q4_how_long": bool(timing.get("estimate_label")) and (
            timing.get("touch") is not None or timing.get("confirm_bars") is not None
            or timing.get("immediate") is True or data_side),
        "q5_stop_target": (num(stop.get("price")) and any(num(t.get("price")) for t in targets)) or data_side,
        "q6_cancel": bool(p.get("cancel")) and bool(p.get("invalidation")),
    }
    return {"answers": answers, "all": all(answers.values()), "data_side_unavailable": data_side}


def build_execution_plan(seq, *, current=None, market="KRX", trend=None, risk=None,
                         now=None, params=None):
    """Turn ``box_sequence.analyze_box_sequence`` output into the execution plan."""
    out = _skeleton()
    try:
        return _build(out, seq, current, str(market or "KRX").upper(), trend, risk or {},
                      now, {**PLAN_PARAMS, **(params or {})})
    except Exception as exc:  # never break the prediction response
        return _unavailable(out, f"박스 실행 플랜 계산 실패({type(exc).__name__})",
                            "다음 조회에서 다시 계산합니다.")


def _build(out, seq, current, market, trend, risk, now, p):
    market = "US" if market == "US" else "KRX"
    seq = seq if isinstance(seq, dict) else {}
    out["staircase"] = seq.get("staircase")
    status = seq.get("status")
    if status in (None, "insufficient_data", "invalid_data"):
        reason = (seq.get("warnings") or ["완료 일봉 부족"])[0]
        have, need_bars = seq.get("bars_available"), seq.get("bars_required")
        if status == "insufficient_data" and have is not None and need_bars:
            jump = seq.get("restart_after_jump")
            reason = (f"{jump.get('date') or '최근'} 하루 {jump.get('change_pct')}% 급변 이후 일봉이 {have}개뿐" if jump
                      else f"일봉이 {have}개뿐")
            reason += f"(박스 계산에 최소 {need_bars}개 필요)"
            return _unavailable(
                out, reason,
                f"약 {max(1, need_bars - have)}거래일 뒤 일봉이 쌓이면 계산합니다. "
                + ("급변 전 가격과 이어 붙이지 않고 새로 시작하기 때문입니다." if jump else "상장 직후·거래 재개 직후 종목은 박스가 아직 형성되지 않았습니다."),
                review=f"일봉 {max(1, need_bars - have)}개가 더 쌓이면 자동으로 계산")
        return _unavailable(out, reason, "원본 시세(일봉)의 해당 날짜 고가·저가·종가를 확인하고 데이터가 고쳐지면 다시 조회하세요.")
    last_close = _finite(seq.get("last_close"))
    last_date = seq.get("last_date")
    atr = _finite(seq.get("atr"))
    if last_close is None or atr is None or atr <= 0:
        return _unavailable(out, "현재가·ATR 계산 불가", "가격 데이터를 확인한 뒤 다시 계산합니다.")
    quote = _finite(current)
    quote_source = "realtime" if quote and quote > 0 else "last_close"
    if quote_source == "last_close":
        quote = last_close
    out["quote"] = {"price": quote, "source": quote_source, "last_close": last_close,
                    "last_date": last_date, "bar_basis": "완료 일봉(진행 중 봉 제외)", "timeframe": "일봉",
                    "pct_vs_last_close": _pct(quote, last_close)}
    today = _parse_day(now) or date.today()
    bar_day = _parse_day(last_date)
    if bar_day is not None and (today - bar_day).days > p["stale_days"]:
        out["warnings"].append(f"마지막 완료 일봉 {last_date} — {(today - bar_day).days}일 지연")
        return _unavailable(out, f"일봉 데이터가 {(today - bar_day).days}일 지연됨",
                            "최신 일봉이 수집되면 다시 계산합니다. 오래된 박스로 매수 신호를 내지 않습니다.")
    if abs(quote / last_close - 1) > p["max_quote_gap"]:
        return _unavailable(out, "현재가와 일봉 기준 가격이 35% 넘게 어긋남(분할·병합·데이터 오류 가능)",
                            "수정주가가 반영된 일봉과 현재가가 맞을 때 다시 계산합니다.")
    reference = seq.get("reference")
    boxes = seq.get("boxes") or []
    if status == "no_box" or not reference:
        return _no_box_plan(out, seq, quote, market, p, atr)
    out["warnings"].extend(w for w in (seq.get("warnings") or []) if w not in out["warnings"])

    top, bottom = float(reference["top"]), float(reference["bottom"])
    height = top - bottom
    mid = (top + bottom) / 2
    tol = p["break_tol_atr"] * atr
    exit_dir = reference.get("exit")
    stair = (seq.get("staircase") or {}).get("key")
    relation = reference.get("relation")
    position = ((quote - bottom) / height) if height > 0 else 0.5
    if quote > top + tol:
        pos_key, pos_label = "above", "박스 상단 위"
    elif quote < bottom - tol:
        pos_key, pos_label = "below", "박스 하단 아래"
    elif position <= 0.25:
        pos_key, pos_label = "lower", "박스 하단 구간"
    elif position >= 0.75:
        pos_key, pos_label = "upper", "박스 상단 구간"
    else:
        pos_key, pos_label = "middle", "박스 중간"
    out["box"] = {
        "top": round_tick(top, market, "nearest"), "bottom": round_tick(bottom, market, "nearest"),
        "mid": round_tick(mid, market, "nearest"), "height": height,
        "height_pct": round(height / bottom * 100, 2) if bottom > 0 else None,
        "height_atr": round(height / atr, 2), "atr": atr, "atr_pct": round(atr / quote * 100, 2),
        "bars": reference.get("bars"), "start_date": reference.get("start_date"),
        "confirm_date": reference.get("confirm_date"), "end_date": reference.get("end_date"),
        "top_touches": reference.get("top_touches"), "bottom_touches": reference.get("bottom_touches"),
        "relation": relation, "exit": exit_dir, "exit_date": reference.get("exit_date"),
        "position_key": pos_key, "position_label": pos_label,
        "position_pct": round(position * 100, 1),
        "timeframe": "일봉", "data_basis": f"{last_date} 완료 일봉 기준",
    }
    out["big_box"] = _big_box(trend, quote, market)
    out["available"] = True
    support = _support_geometry(reference, atr, market, quote, p)
    breakout = _breakout_geometry(reference, atr, market, quote, p, boxes)
    if support is None or breakout is None:
        return _unavailable(out, "박스 가격을 호가 단위로 만들 수 없음", "가격 데이터를 확인합니다.")
    out["evidence"].extend(_box_evidence(reference, seq, market, atr))

    if exit_dir == "down":
        return _after_breakdown(out, seq, reference, support, quote, market, p, atr)
    if exit_dir == "up":
        out = _after_breakout(out, seq, reference, breakout, support, quote, market, p, atr, risk)
    else:
        out = _inside_box(out, seq, reference, support, breakout, quote, market, p, atr, risk,
                          pos_key, stair, relation)
    overhead = breakout.get("overhead")
    if (overhead and out["strategy"]["key"] in {"retest", "breakout"}
            and out["verdict"]["key"] not in {"exclude", "unavailable"}):
        out["warnings"].append(
            f"재확인 매수 구간 위 {_fmt(overhead, market)}(매수 기준가 대비 +{_pct(overhead, breakout['entry_ref']):.2f}%)에 "
            "이전 박스 경계(매물대) — 1차 목표 전에 막힐 수 있음")
    return out


def _big_box(trend, quote, market):
    scales = (trend or {}).get("scales") or {}
    long_scale = (scales.get("long") or {}).get("current") or {}
    direction = long_scale.get("direction")
    if not direction:
        return {"available": False, "label": "큰 박스(60봉 채널) 판단 자료 부족"}
    upper = _finite(long_scale.get("upper"))
    lower = _finite(long_scale.get("lower"))
    name = {"descending": "하락", "ascending": "상승", "flat": "횡보"}.get(direction, direction)
    out = {"available": True, "direction": direction, "label": f"큰 박스(60봉 채널) {name}",
           "upper": round_tick(upper, market, "nearest") if upper else None,
           "lower": round_tick(lower, market, "nearest") if lower else None}
    if direction == "descending" and upper:
        out["broken"] = quote > upper
        out["note"] = ("큰 하락 박스의 상단선을 현재가가 넘어섬 — 영상의 '큰 박스가 바뀌는 자리' 후보"
                       if quote > upper else
                       f"큰 박스가 아직 하락 중 — 상단선 {_fmt(upper, market)} 위 종가가 '큰 박스 전환' 확인")
    return out


def _box_evidence(reference, seq, market, atr):
    stair = seq.get("staircase") or {}
    rel = {"up": "직전 박스보다 위(상승 계단)", "down": "직전 박스보다 아래(하락 계단)",
           "sideways": "직전 박스와 겹침(횡보)", "first": "비교할 직전 박스 없음"}.get(reference.get("relation"), "-")
    return [
        f"현재 박스 {_fmt(reference['bottom'], market)} ~ {_fmt(reference['top'], market)} · "
        f"{reference.get('start_date') or '-'}부터 {reference.get('bars')}봉 · 상단 접촉 {reference.get('top_touches')}회 / "
        f"하단 접촉 {reference.get('bottom_touches')}회",
        f"박스 높이 {reference['height_atr']:.1f} ATR · ATR {_fmt(atr, market)} · {rel}",
        f"박스 계단: {stair.get('label') or '-'}",
        "상단·하단은 Darvas 3봉 규칙(이후 3봉이 넘지·깨지 않은 고점·저점), 이탈은 종가가 0.1 ATR 넘게 벗어난 경우",
    ]


def _risk_overlay(out, risk):
    """Apply the service-wide risk decision; never makes a plan more bullish."""
    notes = []
    prior = (risk or {}).get("prior_decision")
    if prior == "caution":
        notes.append("종합 판단이 '주의'(구조 이탈·이벤트·붕괴 대기) — 신규 매수는 확인 후로 미룹니다")
    days = _finite(((risk or {}).get("event_risk") or {}).get("days_to_earnings"))
    if days is not None and 0 <= days <= 5:
        out["warnings"].append(f"실적 발표 D-{int(days)}: 발표 전후 갭으로 손절가가 지켜지지 않을 수 있음")
    return notes


def _conditions(price_ok, signal_ok, risk_ok, price_detail, signal_detail, risk_detail):
    return [
        {"key": "price", "label": "가격 조건", "ok": price_ok, "detail": price_detail},
        {"key": "signal", "label": "신호 확인 조건", "ok": signal_ok, "detail": signal_detail},
        {"key": "risk", "label": "위험관리 조건", "ok": risk_ok, "detail": risk_detail},
    ]


def _prices_block(geometry, add=None, holder_stop=None):
    return {
        "entry": geometry.get("entry"),
        "trigger": geometry.get("trigger"),
        "add": add,
        "stop": geometry.get("stop"),
        "holder_stop": holder_stop,
        "targets": geometry.get("targets") or [],
        "rr": geometry.get("rr"),
        "rr_t1": geometry.get("rr_t1"),
        "entry_ref": geometry.get("entry", {}).get("ref"),
        "overhead": geometry.get("overhead"),
        "entry_unavailable": None,
    }


def _alt_from(geometry, label, note):
    """Compact alternative plan block for the details fold."""
    if not geometry:
        return None
    return {"key": (geometry.get("entry") or {}).get("kind") or geometry.get("key"), "label": label,
            "entry": geometry.get("entry"),
            "trigger": geometry.get("trigger"), "stop": geometry.get("stop"),
            "targets": geometry.get("targets") or [], "rr": geometry.get("rr"), "note": note}


def _touch_to(quote, level, seq, atr, p):
    """Touch-model estimate for a price above the quote, or None (level at/below the quote)."""
    if level is None or quote is None or level <= quote:
        return None
    return _timing_touch(quote, level, seq.get("closes_for_sigma"), atr, p["horizon_days"])


def _min_bars_text(n):
    if n is None:
        return "새 박스가 확정되는 데 필요한 봉 수는 계산할 수 없습니다"
    if n <= 0:
        return "새 박스 확정 조건(3봉 유지)에 거의 도달했습니다"
    return f"새 박스는 최소 {n}거래일 뒤에 확정될 수 있습니다(그 사이 새 고점·새 저점이 나오지 않을 때)"


def _box_stub(top, bottom, atr, quote, market, forming=None, label="박스 형성 중(미확정)"):
    """`out['box']` for states that have no confirmed reference box."""
    height = top - bottom
    return {
        "top": round_tick(top, market, "nearest"), "bottom": round_tick(bottom, market, "nearest"),
        "mid": round_tick((top + bottom) / 2, market, "nearest"), "height": height,
        "height_pct": round(height / bottom * 100, 2) if bottom > 0 else None,
        "height_atr": round(height / atr, 2), "atr": atr, "atr_pct": round(atr / quote * 100, 2),
        "bars": (forming or {}).get("bars"), "start_date": (forming or {}).get("start_date"),
        "top_touches": None, "bottom_touches": None, "relation": None, "exit": None,
        "position_key": "forming", "position_label": label, "forming": True,
        "timeframe": "일봉",
    }


def _no_box_plan(out, seq, quote, market, p, atr):
    """No confirmed box yet: show the range that is forming and what must happen to confirm it."""
    forming = seq.get("forming") or {}
    out["warnings"].extend(w for w in (seq.get("warnings") or []) if w not in out["warnings"])
    base = _forming_geometry(forming, atr, market, quote, p)
    if base is None:
        return _unavailable(out, "형성 중인 박스 범위를 계산할 수 없음",
                            "최근 일봉의 고가·저가를 확인하고 다시 조회하세요.")
    n = base["bars_to_confirm"]
    f_top, f_bottom = float(forming["top"]), float(forming["bottom"])
    zone = base["entry"]
    narrow = base["height_atr"] < p["min_height_atr"]
    condition = (f"새 고점·새 저점 없이 {n}거래일 유지 → 박스 확정"
                 + (" · 형성 폭 2.5 ATR 미만이면 매수 대상 아님" if narrow else "") + " · 확정 전 매수 안 함")
    geometry = _conditional(base, "forming_box", "재진입 후보(조건부) · 형성 중 박스 하단", condition)
    out["available"] = True
    out["box"] = _box_stub(f_top, f_bottom, atr, quote, market, forming)
    out["big_box"] = None
    out["strategy"] = {"key": "support", "label": "박스 확정 후 하단 지지 매수"}
    out["verdict"] = _verdict("watch", "박스 미확정 · 형성 중")
    out["stage"] = _stage("not_met")
    touch = _timing_touch(quote, zone["high"], seq.get("closes_for_sigma"), atr, p["horizon_days"]) if quote > zone["high"] else None
    out["action"] = (f"최근 160봉 안에 3봉 규칙(고점·저점이 3봉 이상 유지)을 채운 박스가 없어 지금은 사지 않습니다. "
                     f"형성 중인 범위 {_fmt(f_bottom, market)}~{_fmt(f_top, market)} 기준으로 박스가 확정되면 "
                     f"{_fmt(zone['low'], market)}~{_fmt(zone['high'], market)}이 1차 매수 후보입니다.")
    out["when"] = (_min_bars_text(n) + ". 확정 뒤 가격이 후보 구간까지 내려왔다가 일봉 종가가 하단 위·전일보다 높게 끝나면 매수 조건 충족. "
                   + (_touch_text(touch) if touch else "현재가가 이미 후보 구간 안·아래라 도달 시점 대신 새 저점 여부를 확인합니다."))
    out["cancel"] = (f"형성 중 범위를 종가 기준으로 벗어나면(상단 {_fmt(f_top, market)} 위 또는 하단 {_fmt(f_bottom, market)} 아래) "
                     "이 후보 가격은 폐기하고 박스를 다시 산출합니다. 박스가 확정된 뒤에는 하단 −0.1 ATR 아래 종가에서 계획을 취소하고, "
                     f"보유 중이면 {_fmt(base['stop']['price'], market)} 아래 종가에서 손절합니다.")
    out["prices"] = _prices_block(geometry)
    if geometry.get("below_zone_pct"):
        out["warnings"].append(
            f"현재가 {_fmt(quote, market)}가 형성 중 박스 후보 구간 하단보다 {geometry['below_zone_pct']:.1f}% 아래 — "
            "새 저점이 진행 중이라 후보 가격은 아래로 다시 계산됩니다")
    out["conditions"] = _conditions(
        zone["low"] <= quote <= zone["high"], False, (base.get("rr") or 0) >= p["min_reward_risk"] and not narrow,
        f"후보 구간까지 {_dist(zone['high'], quote)}", f"박스 미확정(최소 {n}거래일 필요)",
        ("형성 중 폭 2.5 ATR 미만 · " if narrow else "") + f"후보 손익비 {base.get('rr')}:1")
    out["timing"] = {"immediate": False, "confirm_bars": n, "confirm_rule": f"새 고점·새 저점 없이 {n}봉 유지",
                     "touch": touch, "note": _min_bars_text(n),
                     "estimate_label": _min_bars_text(n) + ("." + _touch_text(touch) if touch else "."),
                     "review": "매 거래일 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
    out["invalidation"] = ["형성 중 범위를 종가로 벗어나면 후보 가격 폐기",
                           "새 저점·새 고점이 나오면 후보 가격을 새 기준으로 다시 계산",
                           f"{p['horizon_days']}거래일 안에 박스가 확정되지 않으면 재산출"]
    out["evidence"].append("Darvas 3봉 규칙으로 확정된 박스가 아직 없음 — 가격은 형성 중인 범위 기준 잠정값")
    return out


def _failure_candidates(seq, reference, quote, market, p, atr):
    """Re-entry prices after the box failed (broke down, or the breakout failed below the bottom).

    Primary candidate follows the video: wait for a *new* box that stops falling. The failed
    breakdown reclaim is the alternative (Bulkowski: after a downward break about 58 % of patterns
    pull back to the break level within 30 days, and about 59 % of those fall again).
    Returns ``(primary, alternative, base, reclaim)``.
    """
    top, bottom = float(reference["top"]), float(reference["bottom"])
    last_close = float(seq["last_close"])
    forming = seq.get("forming") or {}
    base_raw = _forming_geometry(forming, atr, market, quote, p)
    n = base_raw["bars_to_confirm"] if base_raw else None
    narrow = bool(base_raw) and base_raw["height_atr"] < p["min_height_atr"]
    base = _conditional(
        base_raw, "new_base", "재진입 후보(조건부) · 새 바닥 박스 하단",
        (f"새 저점·새 고점 없이 {n}거래일 유지 → 박스 확정" if n is not None else "새 박스 확정 필요")
        + (" · 형성 폭 2.5 ATR 미만이면 매수 대상 아님" if narrow else "")
        + " · 새 저점이 나오면 가격 갱신")
    reclaim_raw = _reclaim_geometry({"top": top, "bottom": bottom}, atr, market, quote, p)
    reclaim = _conditional(
        reclaim_raw, "reclaim", "이탈 실패 확인 후 재진입 후보(조건부)",
        (f"종가 {_fmt(reclaim_raw['trigger']['price'], market)} 이상 {p['confirm_closes']}회 후 "
         f"{_fmt(reclaim_raw['entry']['low'], market)}~{_fmt(reclaim_raw['entry']['high'], market)}까지 눌렸다가 하단 위 마감")
        if reclaim_raw else "")
    back_inside = last_close >= bottom or quote >= bottom
    if back_inside and reclaim:
        primary, alternative = reclaim, base
    else:
        primary, alternative = (base or reclaim), (reclaim if base else None)
    return primary, alternative, base, reclaim


_KIND_WHEN = {"new_base": "새 바닥 박스가 확정되면", "reclaim": "이탈 실패(되돌림)가 확인되면",
              "forming_box": "박스가 확정되면", "range_support": "하단 지지가 재확인되면",
              "new_box_above": "위쪽 새 박스가 확정되면"}


def _price_summary(geometry, market):
    """'새 바닥 박스가 확정되면 X~Y(손절 S, 목표 T1 → T2, 손익비 R:1)' — shared by action texts."""
    if not geometry:
        return ""
    entry = geometry["entry"]
    targets = " → ".join(_fmt(t["price"], market) for t in geometry.get("targets") or [])
    rr = geometry.get("rr")
    weak = " — 손익비 1:1 미만이라 이 폭 그대로면 매수 대상 아님" if rr is not None and rr < 1 else ""
    return (f"{_KIND_WHEN.get(entry.get('kind'), '조건이 충족되면')} {_fmt(entry['low'], market)}~{_fmt(entry['high'], market)}"
            f"(손절 {_fmt(geometry['stop']['price'], market)}, 목표 {targets}, 손익비 {rr}:1{weak})")


def _apply_failure_plan(out, seq, reference, support, quote, market, p, atr, last_close, bottom):
    """Prices, timing and cancel/holder lines shared by every 'plan failed' state."""
    primary, alternative, base, reclaim = _failure_candidates(seq, reference, quote, market, p, atr)
    forming = seq.get("forming") or {}
    n = _bars_to_confirm(forming)
    if primary is None:
        return None
    trigger_price = (reclaim or {}).get("trigger", {}).get("price")
    touch = _touch_to(quote, trigger_price, seq, atr, p) if reclaim else None
    if primary.get("below_zone_pct"):
        out["warnings"].append(
            f"현재가 {_fmt(quote, market)}가 재진입 후보 구간 하단보다 {primary['below_zone_pct']:.1f}% 아래 — 새 저점이 진행 중이라 "
            "후보 가격은 새 저점이 3봉 유지된 뒤 아래로 다시 계산됩니다")
    out["prices"] = _prices_block(primary, holder_stop=support["stop"])
    out["alternative"] = _alt_from(
        alternative, (alternative or {}).get("entry", {}).get("label", ""),
        (alternative or {}).get("entry", {}).get("condition", "")) if alternative else None
    out["conditions"] = _conditions(
        False, False, False, f"재진입 후보 구간 상단까지 {_dist(primary['entry']['high'], quote)}",
        "새 박스 확정·이탈 실패 종가 확인 전", f"박스 이탈 상태 · 후보 손익비 {primary.get('rr')}:1(조건부)")
    out["timing"] = {
        "immediate": False, "confirm_bars": n,
        "confirm_rule": f"새 저점·새 고점 없이 {n}봉 유지(새 박스 확정)" if n is not None else "새 박스 확정 필요",
        "touch": touch, "note": _min_bars_text(n),
        "estimate_label": (_min_bars_text(n) + "." + (" 이탈 실패 확인가 " + _fmt(trigger_price, market) + " 기준 " + _touch_text(touch) if touch else "")),
        "review": "매 거래일 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
    out["invalidation"] = [
        f"새 박스가 확정되기 전 새 저점이 나오면 후보 가격을 아래로 다시 계산",
        (f"이탈 실패 시나리오는 종가가 {_fmt(reclaim['stop']['price'], market)} 아래로 다시 내려가면 폐기" if reclaim else "이탈 실패 시나리오 계산 불가"),
        f"{p['horizon_days']}거래일 안에 새 박스가 확정되지 않으면 재산출"]
    out["evidence"].append(
        "하단 이탈 뒤 이탈선까지 되돌림은 Bulkowski 표본에서 약 58%가 30일 안에 나오지만 되돌린 뒤 약 59%는 다시 하락 — "
        "그래서 이탈선 회복은 '대안'이고 영상의 '더 이상 빠지지 않는 새 박스'를 1순위 후보로 둠")
    return {"primary": primary, "alternative": alternative, "base": base, "reclaim": reclaim,
            "n": n, "touch": touch, "trigger_price": trigger_price}


def _after_breakdown(out, seq, reference, support, quote, market, p, atr=None):
    atr = atr or float(seq.get("atr") or 0)
    bottom = float(reference["bottom"])
    last_close = float(seq["last_close"])
    forming = seq.get("forming") or {}
    failed_rise = reference.get("relation") == "up" or (seq.get("staircase") or {}).get("key") in {"ascending", "turn_up"}
    recover = round_tick(bottom, market, "ceil")
    out["strategy"] = {"key": "none", "label": _STRATEGIES["none"]}
    out["box"]["position_key"] = "below" if last_close < bottom else out["box"]["position_key"]
    plan = _apply_failure_plan(out, seq, reference, support, quote, market, p, atr, last_close, bottom)
    if plan is None:
        return _unavailable(out, "재진입 후보 가격을 호가 단위로 만들 수 없음", "최근 일봉의 고가·저가를 확인하고 다시 조회하세요.")
    primary, base, reclaim = plan["primary"], plan["base"], plan["reclaim"]
    again = _price_summary(primary, market)
    other = _price_summary(plan["alternative"], market)
    wait = _min_bars_text(plan["n"])
    if last_close < bottom <= quote:
        # The completed bar broke down but the live quote is back above the bottom. Only a
        # daily close can confirm the failed breakdown, so this stays a no-buy state.
        out["verdict"] = _verdict("watch", "하단 이탈 후 장중 회복 시도")
        out["stage"] = _stage("signal_wait")
        out["action"] = (f"직전 완료 일봉 종가는 박스 하단 {_fmt(bottom, market)} 아래(이탈)였지만 현재가 {_pj(quote, market, '은')} "
                         "다시 그 위에 있습니다. 장중 회복은 아직 확인이 아니므로 지금은 사지 않습니다. "
                         f"다시 산다면 {again}.")
        out["when"] = (f"오늘 일봉 종가가 {_fmt(recover, market)} 이상으로 마감하고 {p['confirm_closes']}일 유지되면 이탈 실패(되돌림)로 봅니다. "
                       f"그렇지 않으면 {wait}. " + (f"대안: {other}." if other else ""))
    elif last_close >= bottom:
        out["verdict"] = _verdict("watch", "하단 이탈 후 박스 안으로 복귀")
        out["stage"] = _stage("not_met")
        out["action"] = (f"하단 {_fmt(bottom, market)} 이탈 뒤 종가가 다시 박스 안으로 돌아왔습니다. 이탈 실패(되돌림)인지 새 하락 박스인지 "
                         f"확인될 때까지 사지 않습니다. 다시 산다면 {again}.")
        out["when"] = (f"종가가 {_fmt(recover, market)} 위에서 {p['confirm_closes']}일 유지돼야 이탈 실패로 확인됩니다. "
                       f"그렇지 않으면 {wait}. " + (f"대안: {other}." if other else ""))
    else:
        out["verdict"] = _verdict("exclude", "박스 하단 이탈" + (" · 상승 시도 실패" if failed_rise else ""))
        out["stage"] = _stage("cancelled")
        out["action"] = (f"종가가 박스 하단 {_fmt(bottom, market)} 아래로 이탈했습니다. "
                         + ("영상 기준으로 위로 올라가던 박스가 실패해 내려온 자리라 손절 구간입니다. "
                            if failed_rise else "하락 박스 계단이 이어질 수 있어 ")
                         + f"신규 매수는 하지 않습니다. 다시 산다면 {again}.")
        out["when"] = (f"{wait}. 새 박스는 직전 박스보다 낮아지지 않아야(더 이상 빠지지 않아야) 영상의 전환 자리입니다. "
                       + (f"대안: {other}." if other else ""))
    out["cancel"] = ("이미 보유 중이라면 하단 −0.35 ATR "
                     f"{_fmt(support['stop']['price'], market)} 아래 종가가 손절 기준입니다. "
                     "위 재진입 후보도 새 박스 확정 뒤 그 박스 하단(−0.1 ATR) 아래로 종가가 내려가면 취소합니다."
                     + (f" 오늘 종가가 다시 {_fmt(recover, market)} 아래로 끝나면 매수 제외가 유지됩니다."
                        if last_close < bottom <= quote else ""))
    return out


def _after_breakout(out, seq, reference, breakout, support, quote, market, p, atr, risk):
    top, bottom = float(reference["top"]), float(reference["bottom"])
    height = top - bottom
    post = seq.get("post_exit") or {}
    closes = post.get("closes") or []
    lows = post.get("lows") or []
    last_close = float(seq["last_close"])
    stop = breakout["stop"]["price"]
    zone = breakout["entry"]
    measured = top + height
    stair = (seq.get("staircase") or {}).get("key")
    context = []
    if reference.get("relation") == "down":
        context.append("하락 계단 중인 박스를 위로 이탈 — 영상의 '바뀌는 자리' 후보(아직 새 박스 미확정)")
    elif stair == "base":
        context.append("하락 후 더 이상 빠지지 않던 박스를 위로 이탈 — 영상의 추세 전환 자리")
    out["evidence"].extend(context)
    out["strategy"] = {"key": "retest", "label": _STRATEGIES["retest"]}
    prices = _prices_block(breakout, add={"low": None, "high": None,
                                          "condition": "첫 매수 후 새 박스가 위에서 확정되고 그 상단을 다시 돌파할 때만 추가(가격은 새 박스 확정 후 산출)"})
    out["prices"] = prices
    if last_close < stop:
        out["verdict"] = _verdict("exclude", "돌파 실패 · 손절 기준 아래로 복귀")
        out["stage"] = _stage("cancelled")
        out["strategy"] = {"key": "none", "label": _STRATEGIES["none"]}
        failed = (f"상단 {_fmt(top, market)} 돌파 뒤 종가가 손절 기준 {_fmt(stop, market)} 아래로 되돌아왔습니다. "
                  "영상에서 말한 '올라가다 실패해 내려오는' 경우라 돌파 매수 계획은 취소합니다. ")
        holder = f"이미 보유 중이라면 {_fmt(stop, market)} 아래 종가에서 손절합니다(돌파 실패). "
        if last_close < bottom:
            plan = _apply_failure_plan(out, seq, reference, support, quote, market, p, atr, last_close, bottom)
            if plan is None:
                return _unavailable(out, "재진입 후보 가격을 호가 단위로 만들 수 없음", "최근 일봉의 고가·저가를 확인하고 다시 조회하세요.")
            out["action"] = failed + f"종가가 박스 하단 {_fmt(bottom, market)}까지 깨져 신규 매수는 하지 않습니다. 다시 산다면 {_price_summary(plan['primary'], market)}."
            out["when"] = _min_bars_text(plan["n"]) + ". " + (f"대안: {_price_summary(plan['alternative'], market)}." if plan["alternative"] else "")
            out["cancel"] = holder + "위 재진입 후보도 새 박스 확정 뒤 그 박스 하단(−0.1 ATR) 아래 종가면 취소합니다."
            return out
        # Back inside the old range: the old bottom is the nearest structural support again.
        forming = seq.get("forming") or {}
        n = _bars_to_confirm(forming)
        sup_zone = support["entry"]
        range_support = _conditional(
            support, "range_support", "재진입 후보(조건부) · 박스 하단 지지 재확인",
            f"가격이 {_fmt(sup_zone['low'], market)}~{_fmt(sup_zone['high'], market)}까지 내려온 뒤 일봉 종가가 하단 위·전일보다 높게 마감")
        touch = _timing_touch(quote, sup_zone["high"], seq.get("closes_for_sigma"), atr, p["horizon_days"]) if quote > sup_zone["high"] else None
        out["prices"] = _prices_block(range_support, holder_stop=support["stop"])
        base_alt = _conditional(_forming_geometry(forming, atr, market, quote, p), "new_base",
                                "재진입 후보(조건부) · 새 박스 하단", "새 저점·새 고점 없이 3봉 유지되면 새 박스 확정")
        out["alternative"] = _alt_from(base_alt, "새 박스가 확정될 때", (base_alt or {}).get("entry", {}).get("condition", "")) if base_alt else None
        narrow = (top - bottom) < p["min_height_atr"] * atr
        out["action"] = (failed + f"종가는 아직 이전 박스 안이라 하단 지지를 다시 시험할 수 있습니다. 신규 매수는 하지 않고, 다시 산다면 "
                         f"{_price_summary(range_support, market)}.")
        out["when"] = (f"가격이 {_fmt(sup_zone['low'], market)}~{_fmt(sup_zone['high'], market)}까지 내려온 뒤 일봉 종가가 하단 위·전일보다 높게 끝나면 지지 재확인. "
                       + (_touch_text(touch) + " " if touch else "")
                       + ("하단 아래로 깨지면 " + (_min_bars_text(n) if n is not None else "새 박스를 다시 계산합니다") + "."))
        out["cancel"] = holder + f"재진입 후보는 종가가 하단 {_pj(bottom, market, '을')} −0.1 ATR 넘게 깨면 취소합니다."
        out["conditions"] = _conditions(
            sup_zone["low"] <= quote <= sup_zone["high"], False, (support.get("rr") or 0) >= p["min_reward_risk"] and not narrow,
            f"후보 구간 상단까지 {_dist(sup_zone['high'], quote)}", "지지 재확인 봉 미발생",
            ("박스 폭 2.5 ATR 미만 · " if narrow else "") + f"후보 손익비 {support.get('rr')}:1(조건부)")
        out["timing"] = {"immediate": False, "confirm_bars": 1, "confirm_rule": "가격 도달 후 일봉 종가 하단 위 + 전일 대비 상승",
                         "touch": touch, "estimate_label": (_touch_text(touch) if touch else "후보 구간 안·아래라 종가 확인(1봉)만 남았습니다."),
                         "review": "매 거래일 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
        out["invalidation"] = [f"종가 {_fmt(stop, market)} 아래 = 돌파 실패(이미 발생)",
                               f"종가가 하단 {_fmt(bottom, market)} − 0.1 ATR 아래면 재진입 후보 취소",
                               f"{p['horizon_days']}거래일 안에 지지 재확인이 없으면 재산출"]
        return out
    above = 0
    for c in reversed(closes):  # consecutive completed closes above the broken top
        if c > top:
            above += 1
        else:
            break
    confirmed = above >= p["confirm_closes"] and all(c >= top for c in closes)
    risk_notes = _risk_overlay(out, risk)
    rr_ok = (breakout.get("rr") or 0) >= p["min_reward_risk"]
    invalid = [f"종가 {_fmt(stop, market)} 아래 = 돌파 실패, 계획 취소(손절)",
               f"재확인 없이 2차 목표 {_fmt(measured, market)}에 먼저 도달하면 재확인 매수 기회 소멸 → 새 박스 대기",
               f"{p['horizon_days']}거래일 안에 재확인이 없으면 새 박스 기준으로 다시 산출"]
    out["invalidation"] = invalid
    out["cancel"] = (f"종가가 {_fmt(stop, market)}(깨진 상단 − 0.6 ATR) 아래로 내려오면 돌파 실패로 보고 계획을 취소합니다. "
                     f"재확인 없이 {_fmt(measured, market)}까지 올라가 버려도 추격하지 않고 새 박스를 기다립니다.")
    if quote <= stop:
        out["verdict"] = _verdict("watch", "장중 손절 기준 아래 · 종가 확인 중")
        out["stage"] = _stage("signal_wait")
        out["action"] = (f"현재가 {_pj(quote, market, '이')} 돌파 실패 기준 {_fmt(stop, market)} 아래에 있습니다. "
                         "새로 사지 않습니다. 오늘 종가가 이 가격 아래로 끝나면 돌파 실패로 계획을 취소합니다.")
        out["when"] = f"종가가 {_fmt(stop, market)} 위에서 끝나고 다시 {_fmt(top, market)} 위로 올라서야 재확인 매수를 검토합니다."
        out["conditions"] = _conditions(False, False, False, "손절 기준 아래(장중)", "종가 확인 전", "돌파 실패 위험")
        out["timing"] = {"immediate": False, "confirm_bars": 1, "confirm_rule": "오늘 종가 확인",
                         "touch": None, "estimate_label": "오늘 종가로 판단합니다.",
                         "review": "오늘 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
        return out
    if last_close < top:
        out["verdict"] = _verdict("watch", "돌파 후 상단 아래로 되밀림")
        out["stage"] = _stage("not_met")
        out["action"] = (f"상단 {_pj(top, market, '을')} 넘었다가 종가가 다시 그 아래로 내려왔습니다. "
                         f"손절 기준 {_pj(stop, market, '은')} 아직 지키고 있지만 재확인 실패 위험이 있어 매수하지 않습니다.")
        out["when"] = f"종가가 다시 {_fmt(breakout['trigger']['price'], market)} 이상으로 올라서면 재확인 매수를 검토합니다."
        out["conditions"] = _conditions(None, False, rr_ok, "재확인 구간 근처", "상단 위 종가 회복 필요",
                                        f"손익비 {breakout.get('rr')}")
        out["timing"] = {"immediate": False, "confirm_bars": 1, "confirm_rule": "상단 위 종가 1회 회복",
                         "touch": None, "estimate_label": "회복 시점은 예측하지 않습니다.",
                         "review": "매 거래일 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
        return out
    if not confirmed:
        out["verdict"] = _verdict("conditional", f"돌파 {above}일차 · 확인 대기")
        out["stage"] = _stage("signal_wait")
        need = max(1, p["confirm_closes"] - above)
        out["action"] = (f"상단 {_fmt(top, market)} 위 종가가 {above}일째입니다. 아직 돌파 확인 전이라 추격 매수하지 않습니다. "
                         f"확인 후 {_fmt(zone['low'], market)}~{_pj(zone['high'], market, '으로')} 되돌릴 때 매수 후보입니다.")
        out["when"] = (f"다음 {need}개 일봉 종가가 {_fmt(top, market)} 위에 머물면 돌파 확인. 그 뒤 가격이 재확인 구간에 들어와 "
                       f"종가가 {_fmt(top, market)} 이상으로 끝나면 매수 조건이 충족됩니다.")
        touch = _timing_touch(quote, zone["high"], seq.get("closes_for_sigma"), atr, p["horizon_days"]) if quote > zone["high"] else None
        out["conditions"] = _conditions(zone["low"] <= quote <= zone["high"], False, rr_ok and not risk_notes,
                                        f"재확인 구간 {_fmt(zone['low'], market)}~{_fmt(zone['high'], market)}",
                                        f"상단 위 종가 {above}/{p['confirm_closes']}",
                                        "; ".join(risk_notes) or f"손익비 {breakout.get('rr')}:1")
        out["timing"] = {"immediate": False, "confirm_bars": need, "confirm_rule": f"일봉 종가 {p['confirm_closes']}회 상단 위",
                         "touch": touch, "estimate_label": _touch_text(touch) if touch else
                         "가격은 이미 재확인 구간 근처입니다. 확인 종가가 남았습니다.",
                         "review": "매 거래일 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
        return out
    if quote >= measured:
        out["verdict"] = _verdict("watch", "측정 목표 도달 · 재확인 기회 지남")
        out["stage"] = _stage("not_met")
        forming = seq.get("forming") or {}
        n = _bars_to_confirm(forming)
        upper_box = _conditional(
            _forming_geometry(forming, atr, market, quote, p), "new_box_above", "재진입 후보(조건부) · 위쪽 새 박스 하단",
            (f"새 고점·새 저점 없이 {n}거래일 유지 → 새 박스 확정" if n is not None else "새 박스 확정 필요")
            + " · 새 고점·저점이 나오면 가격 갱신")
        out["action"] = (f"돌파 후 재확인 없이 측정 목표 {_pj(measured, market, '을')} 넘었습니다. 지금 사는 것은 추격 매수입니다. "
                         "새 박스가 위에서 확정될 때까지 기다립니다."
                         + (f" 다시 산다면 {_price_summary(upper_box, market)}." if upper_box else ""))
        out["when"] = (_min_bars_text(n) + ". 확정 뒤 가격이 새 박스 하단 구간까지 내려왔다가 하단 위·전일보다 높게 마감하면 매수 조건 충족."
                       + (f" 위쪽 박스 확정 전에 종가가 {_fmt(top, market)} 아래로 돌아오면 돌파 실패로 계획을 취소합니다."))
        if upper_box:
            out["prices"] = _prices_block(upper_box, holder_stop=breakout["stop"])
        out["conditions"] = _conditions(False, True, False, "재확인 구간보다 너무 높음", "돌파 확인 완료",
                                        "추격 시 손익비 불리")
        out["timing"] = {"immediate": False, "confirm_bars": n if n is not None else 3,
                         "confirm_rule": "새 박스 고점·저점 각각 3봉 유지", "touch": None,
                         "estimate_label": _min_bars_text(n) + ".",
                         "review": "매 거래일 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
        return out
    latest_low = lows[-1] if lows else None
    retest_bar = (latest_low is not None and latest_low <= zone["high"] and last_close >= top
                  and above >= 1 and len(closes) > p["confirm_closes"]) or not p["require_signal"]
    # Display state only (the buy test above still needs quote <= zone top): a quote within 0.05 ATR
    # of the top is "at the zone", not "waiting for price".
    in_zone = stop < quote <= zone["high"] + max(tick_size(quote, market), 0.05 * atr)
    checks_ok = rr_ok and not risk_notes
    if retest_bar and quote <= zone["high"] and quote >= stop:
        key = "buy" if checks_ok else "conditional"
        out["verdict"] = _verdict(key, "돌파 후 재확인 완료" if checks_ok else "재확인 완료 · 위험 조건 확인 필요")
        out["stage"] = _stage("ready" if checks_ok else "signal_wait")
        out["action"] = (f"돌파가 확인됐고 직전 일봉이 {_fmt(zone['high'], market)} 아래까지 눌렸다가 상단 위에서 마감했습니다. "
                         f"{_fmt(zone['low'], market)}~{_fmt(zone['high'], market)}에서 매수할 수 있습니다(현재가 {_fmt(quote, market)})."
                         + (" " + " · ".join(risk_notes) if risk_notes else ""))
        out["when"] = "지금 조건이 충족된 상태입니다. 재확인 구간 위로 벌어지면 추격하지 말고 다시 눌릴 때를 기다립니다."
        out["conditions"] = _conditions(True, True, checks_ok, f"현재가 {_fmt(quote, market)} 재확인 구간 안",
                                        "돌파 확인 + 재확인 봉 종가 상단 위", "; ".join(risk_notes) or f"손익비 {breakout.get('rr')}:1")
        out["timing"] = {"immediate": checks_ok, "confirm_bars": 0, "confirm_rule": "충족",
                         "touch": None, "estimate_label": "대기 없음 — 조건 충족 상태",
                         "review": "매 거래일 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
        return out
    if in_zone:
        out["verdict"] = _verdict("conditional", "재확인 구간 도달 · 종가 확인 중")
        out["stage"] = _stage("signal_wait")
        out["action"] = (f"현재가 {_pj(quote, market, '이')} 재확인 구간 {_fmt(zone['low'], market)}~{_fmt(zone['high'], market)}에 있습니다. "
                         f"오늘 일봉 종가가 {_fmt(top, market)} 이상으로 끝나야 저항이 지지로 바뀐 것이 확인됩니다.")
        out["when"] = f"이번 일봉 마감 후 종가가 {_fmt(top, market)} 이상이면 매수 조건 충족(1봉 확인)."
        out["conditions"] = _conditions(True, False, checks_ok, "재확인 구간 안", "재확인 봉 종가 미확정",
                                        "; ".join(risk_notes) or f"손익비 {breakout.get('rr')}:1")
        out["timing"] = {"immediate": False, "confirm_bars": 1, "confirm_rule": f"재확인 봉 종가 {_fmt(top, market)} 이상",
                         "touch": None, "estimate_label": "가격은 도달했고 일봉 1개 마감 확인이 남았습니다.",
                         "review": "오늘 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
        return out
    touch = _timing_touch(quote, zone["high"], seq.get("closes_for_sigma"), atr, p["horizon_days"])
    out["verdict"] = _verdict("conditional", "돌파 확인 · 재확인 가격 대기")
    out["stage"] = _stage("price_wait")
    out["action"] = (f"상단 {_fmt(top, market)} 돌파는 확인됐지만 현재가 {_pj(quote, market, '은')} 재확인 구간보다 "
                     f"{_pct(quote, zone['high'])}% 위입니다. 추격하지 말고 {_fmt(zone['low'], market)}~{_fmt(zone['high'], market)} 되돌림을 기다립니다.")
    out["when"] = (f"가격이 {_fmt(zone['high'], market)} 이하로 내려와 그날 종가가 {_fmt(top, market)} 이상으로 끝나면 매수 조건 충족. "
                   + _touch_text(touch))
    out["conditions"] = _conditions(False, True, checks_ok, f"재확인 구간 상단까지 {_dist(zone['high'], quote)}",
                                    "돌파 확인 완료(재확인 대기)", "; ".join(risk_notes) or f"손익비 {breakout.get('rr')}:1")
    out["timing"] = {"immediate": False, "confirm_bars": 1, "confirm_rule": f"재확인 봉 종가 {_fmt(top, market)} 이상",
                     "touch": touch, "estimate_label": _touch_text(touch),
                     "review": "매 거래일 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
    return out


def _inside_box(out, seq, reference, support, breakout, quote, market, p, atr, risk, pos_key, stair, relation):
    top, bottom = float(reference["top"]), float(reference["bottom"])
    height = top - bottom
    last_close = float(seq["last_close"])
    prev_close = _finite(seq.get("prev_close"))
    bar = seq.get("last_bar") or {}
    risk_notes = _risk_overlay(out, risk)
    descending = relation == "down" and p["block_descending_support"]
    narrow = height < p["min_height_atr"] * atr
    rr_ok = (support.get("rr") or 0) >= p["min_reward_risk"]
    trigger = breakout["trigger"]["price"]
    zone = support["entry"]
    breakout_alt = {
        "key": "breakout", "label": _STRATEGIES["breakout"],
        "trigger": breakout["trigger"], "entry": breakout["entry"], "stop": breakout["stop"],
        "targets": breakout["targets"], "rr": breakout["rr"],
        "note": f"일봉 종가 {_fmt(trigger, market)} 이상 {p['confirm_closes']}회 → 돌파 확인, 이후 "
                f"{_fmt(breakout['entry']['low'], market)}~{_fmt(breakout['entry']['high'], market)} 재확인 시 매수",
    }
    support_alt = {
        "key": "support", "label": _STRATEGIES["support"], "entry": support["entry"],
        "stop": support["stop"], "targets": support["targets"], "rr": support["rr"],
        "note": f"{_fmt(zone['low'], market)}~{_fmt(zone['high'], market)}에서 지지 확인(종가 하단 위 + 전일보다 상승) 시 매수",
    }
    out["cancel"] = (f"종가가 박스 하단 {_pj(bottom, market, '을')} 0.1 ATR 넘게 깨고 내려가면 박스가 무너진 것으로 보고 계획을 취소합니다"
                     f"(보유 중이면 {_fmt(support['stop']['price'], market)} 아래 종가에서 손절). "
                     f"{p['horizon_days']}거래일 안에 조건이 나오지 않으면 박스를 다시 산출합니다.")
    out["invalidation"] = [f"종가 {_fmt(bottom, market)} − 0.1 ATR 아래 = 박스 하단 이탈, 계획 취소",
                           f"손절: {_fmt(support['stop']['price'], market)} 아래 종가",
                           f"{p['horizon_days']}거래일 안에 진입 조건 미발생 시 재산출"]
    if descending:
        out["verdict"] = _verdict("watch", "하락 박스 진행 중")
        out["stage"] = _stage("not_met")
        out["strategy"] = {"key": "breakout", "label": "상단 돌파 확인 대기(하락 계단 전환)"}
        out["action"] = (f"이 박스는 직전 박스보다 아래에 있는 하락 계단입니다. 영상 기준으로 하락 박스의 하단은 매수 자리가 아닙니다. "
                         f"박스 상단 {_fmt(top, market)} 돌파로 '바뀌는 자리'가 나올 때까지 관망합니다.")
        out["when"] = (f"일봉 종가가 {_fmt(trigger, market)} 이상으로 {p['confirm_closes']}회 마감하면 하락 계단의 전환 후보입니다. "
                       f"그 뒤 {_fmt(breakout['entry']['low'], market)}~{_fmt(breakout['entry']['high'], market)} 재확인에서 매수를 검토합니다. "
                       + _touch_text(_timing_touch(quote, trigger, seq.get("closes_for_sigma"), atr, p["horizon_days"])))
        out["cancel"] = (f"종가가 하단 {_fmt(bottom, market)} 아래로 이탈하면 하락 계단이 이어지는 것이므로 관망을 연장하고 박스를 다시 산출합니다.")
        out["prices"] = {**_prices_block(breakout), "entry_unavailable": None}
        out["prices"]["entry"] = {**breakout["entry"], "label": "돌파 확인 후 재확인 매수 구간(조건부)"}
        out["alternative"] = None
        out["conditions"] = _conditions(False, False, False, "하락 박스 안", "상단 돌파 미발생",
                                        "하락 계단에서 하단 매수 금지(영상)")
        touch = _timing_touch(quote, trigger, seq.get("closes_for_sigma"), atr, p["horizon_days"])
        out["timing"] = {"immediate": False, "confirm_bars": p["confirm_closes"],
                         "confirm_rule": f"일봉 종가 {_fmt(trigger, market)} 이상 {p['confirm_closes']}회",
                         "touch": touch, "estimate_label": _touch_text(touch),
                         "review": "매 거래일 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
        out["invalidation"] = [f"종가 {_fmt(bottom, market)} 아래 = 하락 계단 지속, 관망 연장"]
        return out
    support_usable = not narrow and rr_ok
    support_reason = ("박스 높이가 2.5 ATR 미만이라 손절폭 대비 수익 여유가 부족" if narrow else
                      f"하단 매수의 손익비 {support.get('rr')}:1 < {p['min_reward_risk']}:1" if not rr_ok else None)
    near_support = quote <= zone["high"] or pos_key == "lower"
    signal_bar = (bar.get("low") is not None and bar["low"] <= zone["high"] and last_close >= bottom
                  and prev_close is not None and last_close > prev_close) or not p["require_signal"]
    stair_label = (seq.get("staircase") or {}).get("label") or ""
    if support_usable and (near_support or pos_key in {"middle"} and quote <= (top + bottom) / 2):
        out["strategy"] = {"key": "support", "label": _STRATEGIES["support"]}
        add = {"low": breakout["entry"]["low"], "high": breakout["entry"]["high"],
               "condition": f"추가 매수: 상단 {_fmt(trigger, market)} 돌파가 종가 {p['confirm_closes']}회로 확인된 뒤 재확인 구간에서만"}
        out["prices"] = _prices_block(support, add=add)
        out["alternative"] = breakout_alt
        checks_ok = not risk_notes
        if quote < bottom:
            out["verdict"] = _verdict("conditional", "하단 일시 이탈 · 종가 회복 확인")
            out["stage"] = _stage("signal_wait")
            out["action"] = (f"현재가 {_pj(quote, market, '이')} 박스 하단 {_pj(bottom, market, '을')} 잠시 밑돌고 있습니다. "
                             "종가가 하단 위로 회복해야 지지가 확인됩니다. 그 전에는 매수하지 않습니다.")
            out["when"] = f"이번 일봉 종가가 {_fmt(round_tick(bottom, market, 'ceil'), market)} 이상이고 전일 종가보다 높으면 지지 확인."
            out["conditions"] = _conditions(True, False, checks_ok, "하단 근처", "종가 하단 회복 미확정",
                                            "; ".join(risk_notes) or f"손익비 {support['rr']}:1")
            out["timing"] = {"immediate": False, "confirm_bars": 1, "confirm_rule": "일봉 종가 하단 위 + 전일 대비 상승",
                             "touch": None, "estimate_label": "가격은 도달했고 일봉 1개 마감 확인이 남았습니다.",
                             "review": "오늘 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
            return out
        if quote <= zone["high"]:
            if signal_bar and checks_ok:
                out["verdict"] = _verdict("buy", "하단 지지 확인")
                out["stage"] = _stage("ready")
                out["action"] = (f"직전 일봉이 하단 구간까지 내려왔다가 하단 위에서 전일보다 높게 마감해 지지가 확인됐습니다. "
                                 f"{_fmt(zone['low'], market)}~{_fmt(zone['high'], market)}에서 계획 수량의 절반을 매수할 수 있습니다.")
                out["when"] = "지금 조건이 충족된 상태입니다. 구간 위로 벌어지면 추격하지 않습니다."
                out["conditions"] = _conditions(True, True, True, f"현재가 {_fmt(quote, market)} 매수 구간 안",
                                                "하단 터치 후 상승 마감", f"손익비 {support['rr']}:1")
                out["timing"] = {"immediate": True, "confirm_bars": 0, "confirm_rule": "충족", "touch": None,
                                 "estimate_label": "대기 없음 — 조건 충족 상태",
                                 "review": "매 거래일 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
                return out
            out["verdict"] = _verdict("conditional", "매수 구간 도달 · 지지 확인 중")
            out["stage"] = _stage("signal_wait")
            out["action"] = (f"현재가 {_pj(quote, market, '이')} 1차 매수 후보 {_fmt(zone['low'], market)}~{_fmt(zone['high'], market)}에 있습니다. "
                             "지지 확인 신호(일봉 종가가 하단 위 + 전일 종가보다 상승)가 나온 뒤 매수합니다."
                             + (" " + " · ".join(risk_notes) if risk_notes else ""))
            out["when"] = "이번 일봉 마감 결과로 판단합니다(최소 1봉). 하단 아래 종가가 나오면 계획을 취소합니다."
            out["conditions"] = _conditions(True, False, checks_ok, "매수 구간 안", "지지 확인 봉 미발생",
                                            "; ".join(risk_notes) or f"손익비 {support['rr']}:1")
            out["timing"] = {"immediate": False, "confirm_bars": 1, "confirm_rule": "일봉 종가 하단 위 + 전일 대비 상승",
                             "touch": None, "estimate_label": "가격은 도달했고 일봉 1개 마감 확인이 남았습니다.",
                             "review": "오늘 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
            return out
        touch = _timing_touch(quote, zone["high"], seq.get("closes_for_sigma"), atr, p["horizon_days"])
        out["verdict"] = _verdict("conditional", "하단 매수 구간까지 가격 대기")
        out["stage"] = _stage("price_wait")
        out["action"] = (f"현재가 {_pj(quote, market, '은')} 1차 매수 후보 {_fmt(zone['low'], market)}~{_fmt(zone['high'], market)}보다 "
                         f"{_pct(quote, zone['high'])}% 위({out['box']['position_label']})입니다. 박스 중간에서 추격하지 말고 하단 접근을 기다립니다.")
        out["when"] = ("가격이 매수 구간에 들어온 뒤 일봉 종가가 하단 위에서 전일보다 높게 끝나면 매수 조건 충족. " + _touch_text(touch))
        out["conditions"] = _conditions(False, False, checks_ok, f"매수 구간 상단까지 {_dist(zone['high'], quote)}",
                                        "가격 도달 후 확인", "; ".join(risk_notes) or f"손익비 {support['rr']}:1")
        out["timing"] = {"immediate": False, "confirm_bars": 1, "confirm_rule": "가격 도달 후 일봉 종가 하단 위 + 전일 대비 상승",
                         "touch": touch, "estimate_label": _touch_text(touch),
                         "review": "매 거래일 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
        if stair_label:
            out["evidence"].append(f"하단 매수 허용 근거: {stair_label}(직전 박스보다 낮아지지 않음)")
        return out
    # Upper half (or support not usable): the actionable plan is a confirmed breakout.
    out["strategy"] = {"key": "breakout", "label": _STRATEGIES["breakout"]}
    out["prices"] = _prices_block(breakout)
    out["alternative"] = support_alt if support_usable else None
    if support_reason:
        out["warnings"].append("하단 지지 매수 제외: " + support_reason)
    touch = _timing_touch(quote, trigger, seq.get("closes_for_sigma"), atr, p["horizon_days"])
    if support_usable or pos_key == "upper":
        out["verdict"] = _verdict("watch", f"{out['box']['position_label']} · 돌파 확인 전")
        lead = f"상단 {_fmt(top, market)} 바로 아래에서는 사지 않습니다. "
    else:
        out["verdict"] = _verdict("watch", ("좁은 박스" if narrow else "하단 매수 손익비 부족") + " · 돌파 확인 전")
        lead = f"{support_reason}해 하단 매수는 하지 않습니다. "
    out["stage"] = _stage("not_met")
    out["action"] = (f"현재가 {_pj(quote, market, '은')} {out['box']['position_label']}입니다. " + lead
                     + "상단 돌파가 일봉 종가로 확인되기를 기다립니다"
                     + (f"(또는 하단 {_fmt(zone['low'], market)}~{_fmt(zone['high'], market)} 되돌림)." if support_usable else "."))
    out["when"] = (f"일봉 종가가 {_fmt(trigger, market)} 이상으로 {p['confirm_closes']}회 마감하면 돌파 확인, 이후 "
                   f"{_fmt(breakout['entry']['low'], market)}~{_fmt(breakout['entry']['high'], market)} 재확인에서 매수. " + _touch_text(touch))
    out["conditions"] = _conditions(False, False, (breakout.get("rr") or 0) >= p["min_reward_risk"] and not risk_notes,
                                    f"돌파 기준가까지 {_dist(trigger, quote)}", "돌파 종가 미발생",
                                    "; ".join(risk_notes) or f"돌파 재확인 손익비 {breakout.get('rr')}:1")
    out["timing"] = {"immediate": False, "confirm_bars": p["confirm_closes"],
                     "confirm_rule": f"일봉 종가 {_fmt(trigger, market)} 이상 {p['confirm_closes']}회",
                     "touch": touch, "estimate_label": _touch_text(touch),
                     "review": "매 거래일 종가 마감 후 재계산", "expiry_bars": p["horizon_days"]}
    out["invalidation"] = [f"종가 {_fmt(bottom, market)} − 0.1 ATR 아래 = 박스 하단 이탈, 계획 취소",
                           f"돌파 후 손절: {_fmt(breakout['stop']['price'], market)} 아래 종가",
                           f"{p['horizon_days']}거래일 안에 돌파 미발생 시 재산출"]
    return out
