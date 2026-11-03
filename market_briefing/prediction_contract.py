"""예측 탭의 단일 현재가 기준 계약.

가격 수집과 예측 계산은 서로 다른 모듈에서 수행되므로, 최종 응답 직전에
모든 가격형 섹션이 같은 현재가 스냅샷을 참조하는지 확인한다. 확정 일봉과
시간외 시세는 합치지 않고 각각의 기준 시각을 명시한다.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, MutableMapping


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _same_price(left: Any, right: Any) -> bool:
    lhs = _finite_number(left)
    rhs = _finite_number(right)
    if lhs is None or rhs is None:
        return lhs is rhs
    tolerance = max(abs(rhs) * 5e-7, 1e-8)
    return abs(lhs - rhs) <= tolerance


def build_price_anchor(
    *,
    current_price: float,
    previous_close: float | None,
    market: str,
    source: str,
    session: str,
    quote_date: str | None,
    quote_time: str | None,
    last_bar_date: str | None,
    is_extended: bool,
    applied_to_chart: bool,
) -> dict[str, Any]:
    """현재가와 확정 지표 시점을 하나의 추적 가능한 스냅샷으로 만든다."""
    current = _finite_number(current_price)
    if current is None or current <= 0:
        raise ValueError("current_price must be a positive finite number")
    previous = _finite_number(previous_close)
    market_key = "US" if str(market).upper() == "US" else "KRX"
    digits = 4 if market_key == "US" else 2
    quote_day = str(quote_date or "")[:10] or None
    bar_day = str(last_bar_date or "")[:10] or None
    source_label = str(source or "historical_daily")
    session_label = str(session or "확정 종가")
    is_live_quote = source_label.lower() not in {
        "historical_daily",
        "yfinance_daily",
        "daily_history",
    }
    mixed_time_basis = bool(
        is_extended
        or not applied_to_chart
        or (quote_day and bar_day and quote_day != bar_day)
    )
    identity = "|".join(
        [
            market_key,
            f"{current:.8f}",
            source_label,
            session_label,
            quote_day or "",
            str(quote_time or ""),
            bar_day or "",
        ]
    )
    change_pct = None
    if previous is not None and previous > 0:
        change_pct = round((current / previous - 1.0) * 100.0, 3)
    return {
        "anchor_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
        "current_price": round(current, digits),
        "previous_close": round(previous, digits) if previous is not None else None,
        "change_pct": change_pct,
        "market": market_key,
        "currency": "USD" if market_key == "US" else "KRW",
        "source": source_label,
        "session": session_label,
        "quote_date": quote_day,
        "quote_time": str(quote_time or "") or None,
        "last_bar_date": bar_day,
        "is_extended": bool(is_extended),
        "applied_to_chart": bool(applied_to_chart),
        "mixed_time_basis": mixed_time_basis,
        "price_basis": "현재 시세" if is_live_quote else "최근 확정 종가",
        "indicator_basis": f"{bar_day} 확정 일봉" if bar_day else "최근 확정 일봉",
    }


def _attach_anchor(section: Any, anchor: dict[str, Any]) -> None:
    if isinstance(section, MutableMapping):
        section["price_anchor"] = dict(anchor)


def apply_prediction_price_contract(
    response: MutableMapping[str, Any], anchor: dict[str, Any]
) -> MutableMapping[str, Any]:
    """예측 응답에 가격 앵커를 적용하고 명시적 현재가 불일치를 제한 상태로 낮춘다.

    가격대 자체를 비율 이동시키지는 않는다. 구조적 지지·저항을 임의로 옮기면
    의미가 달라지기 때문이다. 대신 현재가 필드만 단일화하고 불일치를 응답에
    기록하여 사용자가 제한 상태를 확인할 수 있게 한다.
    """
    current = _finite_number(anchor.get("current_price"))
    if current is None or current <= 0:
        raise ValueError("price anchor is missing a valid current_price")

    mismatches: list[str] = []
    response["price_anchor"] = dict(anchor)
    if "last_close" in response and not _same_price(response.get("last_close"), current):
        mismatches.append("last_close")
    response["last_close"] = current

    buy_price = response.get("buy_price")
    if isinstance(buy_price, MutableMapping):
        if "current" in buy_price and not _same_price(buy_price.get("current"), current):
            mismatches.append("buy_price.current")
        buy_price["current"] = current
        _attach_anchor(buy_price, anchor)

    risk = response.get("risk_scenarios")
    if isinstance(risk, MutableMapping):
        risk["current_price"] = current
        _attach_anchor(risk, anchor)

    target = response.get("target_price")
    if isinstance(target, MutableMapping):
        target["current_price"] = current
        _attach_anchor(target, anchor)

    outlook = response.get("prediction_outlook")
    if isinstance(outlook, MutableMapping):
        _attach_anchor(outlook, anchor)
        forecast = outlook.get("forecast")
        if isinstance(forecast, MutableMapping):
            if "current_price" in forecast and not _same_price(forecast.get("current_price"), current):
                mismatches.append("prediction_outlook.forecast.current_price")
            forecast["current_price"] = current
            forecast["price_anchor_id"] = anchor.get("anchor_id")
            if mismatches and forecast.get("status") == "ok":
                forecast["status"] = "limited"
                forecast["status_label"] = "예측 제한(현재가 기준 재정렬)"
        market_context = outlook.setdefault("market_context", {})
        if mismatches and isinstance(market_context, MutableMapping):
            gaps = market_context.setdefault("data_gaps", [])
            warning = "현재가 앵커와 일부 계산 결과의 기준가가 달라 현재가 필드를 재정렬했습니다."
            if isinstance(gaps, list) and warning not in gaps:
                gaps.append(warning)

    unique_mismatches = list(dict.fromkeys(mismatches))
    response["prediction_price_contract"] = {
        "status": "corrected" if unique_mismatches else "ok",
        "anchor_id": anchor.get("anchor_id"),
        "mismatches": unique_mismatches,
        "mixed_time_basis": bool(anchor.get("mixed_time_basis")),
    }
    return response


__all__ = ["build_price_anchor", "apply_prediction_price_contract"]
