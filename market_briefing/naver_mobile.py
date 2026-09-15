"""네이버 증권 모바일 JSON API 기반 국내 종목 펀더멘털·동일업종 비교군 수집.

finance.naver.com/item/main.naver 는 신규 stock.naver.com 으로 리다이렉트되어 기존 HTML
선택자(#_per, #_pbr, #_market_sum, .cop_analysis)가 모두 비어 PER·PBR·시가총액·ROE·
부채비율이 '-' 로 표시됐다. 모바일 JSON API 는 동일 수치를 구조화된 필드로 제공한다.

- /api/stock/{code}/integration : PER·PBR·EPS·BPS·시총·52주 고저·배당·업종코드·동일업종 비교군·컨센서스
- /api/stock/{code}/finance/annual : 연간 매출·영업이익·순이익·ROE·부채비율·영업이익률 (실적/추정 구분)

모든 함수는 실패 시 예외 대신 빈 값을 돌려주며, 추정치(isConsensus=Y)는 실적과 섞지 않는다.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

_BASE = "https://m.stock.naver.com/api/stock"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
                   "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"),
    "Referer": "https://m.stock.naver.com/",
}


def to_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    text = re.sub(r"[,\s배%원+]", "", str(value))
    if text in ("", "-", "N/A", "None", "null"):
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_korean_amount(text: Any) -> float | None:
    """'1,517조 1,093억' / '905억' / '3,336,059'(억원 아님) → 원 단위. 조·억 표기가 없으면 None."""
    raw = str(text or "").replace(",", "").replace(" ", "")
    if not raw or ("조" not in raw and "억" not in raw):
        return None
    total = 0.0
    jo = re.search(r"(\d+(?:\.\d+)?)조", raw)
    eok = re.search(r"(\d+(?:\.\d+)?)억", raw)
    if jo:
        total += float(jo.group(1)) * 1e12
    if eok:
        total += float(eok.group(1)) * 1e8
    return total if total > 0 else None


def _get_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        response = requests.get(url, headers=_HEADERS, timeout=timeout)
        if response.status_code != 200:
            print(f"[naver_mobile][warn] HTTP {response.status_code} {url}")
            return None
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception as exc:  # 네트워크/JSON 오류는 호출측에서 '수집 실패'로 표시
        print(f"[naver_mobile][warn] {url} {type(exc).__name__}: {exc}")
        return None


def parse_integration(payload: dict | None) -> dict:
    payload = payload or {}
    infos = {str(item.get("code") or ""): item.get("value") for item in (payload.get("totalInfos") or [])
             if isinstance(item, dict)}
    market_cap_text = str(infos.get("marketValue") or "").strip()
    peers = []
    for item in payload.get("industryCompareInfo") or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("itemCode") or "").strip()
        name = str(item.get("stockName") or "").strip()
        exchange = str(((item.get("stockExchangeType") or {}).get("code")) or "").upper()
        if not code or not name:
            continue
        peers.append({
            "code": code,
            "name": name,
            "ticker": f"{code}.{'KQ' if exchange == 'KQ' else 'KS'}",
            "exchange": "KOSDAQ" if exchange == "KQ" else "KOSPI",
            "close": to_number(item.get("closePrice")),
            "change_pct": to_number(item.get("fluctuationsRatio")),
            # industryCompareInfo.marketValue 는 백만원 단위 문자열
            "market_cap": (to_number(item.get("marketValue")) or 0) * 1e6 or None,
        })
    consensus = payload.get("consensusInfo") or {}
    return {
        "per": to_number(infos.get("per")),
        "pbr": to_number(infos.get("pbr")),
        "eps": to_number(infos.get("eps")),
        "bps": to_number(infos.get("bps")),
        "forward_per": to_number(infos.get("cnsPer")),
        "dividend_yield": to_number(infos.get("dividendYieldRatio")),
        "foreign_rate": to_number(infos.get("foreignRate")),
        "high_52w": to_number(infos.get("highPriceOf52Weeks")),
        "low_52w": to_number(infos.get("lowPriceOf52Weeks")),
        "prev_close": to_number(infos.get("lastClosePrice")),
        "open": to_number(infos.get("openPrice")),
        "high": to_number(infos.get("highPrice")),
        "low": to_number(infos.get("lowPrice")),
        "volume": to_number(infos.get("accumulatedTradingVolume")),
        "market_cap_text": market_cap_text or None,
        "market_cap_raw": parse_korean_amount(market_cap_text),
        "industry_code": str(payload.get("industryCode") or "") or None,
        "industry_peers": peers,
        "consensus_target_price": to_number(consensus.get("priceTargetMean")),
        "consensus_recommendation": to_number(consensus.get("recommMean")),
        "consensus_date": consensus.get("createDate"),
        "stock_name": payload.get("stockName"),
    }


def parse_annual_finance(payload: dict | None) -> dict:
    info = ((payload or {}).get("financeInfo") or {})
    periods = [p for p in (info.get("trTitleList") or []) if isinstance(p, dict)]
    actual_keys = [str(p.get("key")) for p in periods if str(p.get("isConsensus") or "N").upper() != "Y"]
    rows = {str(row.get("title") or ""): row.get("columns") or {} for row in (info.get("rowList") or [])
            if isinstance(row, dict)}

    def _series(title: str) -> list[tuple[str, float]]:
        columns = rows.get(title) or {}
        out = []
        for key in actual_keys:
            value = to_number((columns.get(key) or {}).get("value"))
            if value is not None:
                out.append((key, value))
        return out

    def _latest(title: str) -> float | None:
        series = _series(title)
        return series[-1][1] if series else None

    def _growth(title: str) -> tuple[float | None, str | None]:
        series = _series(title)
        if len(series) < 2:
            return None, None
        prev_value, last_value = series[-2][1], series[-1][1]
        if prev_value < 0 <= last_value:
            return None, "흑자전환"
        if prev_value >= 0 > last_value:
            return None, "적자전환"
        if prev_value < 0 and last_value < 0:
            return None, "적자 지속"
        if prev_value == 0:
            return None, None
        return round((last_value - prev_value) / abs(prev_value) * 100.0, 1), None

    revenue_growth, _ = _growth("매출액")
    op_growth, op_label = _growth("영업이익")
    np_growth, np_label = _growth("당기순이익")
    return {
        "latest_actual_period": actual_keys[-1] if actual_keys else None,
        "roe": _latest("ROE"),
        "debt": _latest("부채비율"),
        "op_margin": _latest("영업이익률"),
        "net_margin": _latest("순이익률"),
        "revenue": _latest("매출액"),
        "revenue_growth": revenue_growth,
        "operating_profit_growth": op_growth,
        "operating_profit_growth_label": op_label,
        "net_profit_growth": np_growth,
        "net_profit_growth_label": np_label,
    }


_INDUSTRY_NAME_CACHE: dict[str, str] = {}


def fetch_naver_industry_name(industry_code: str | None, timeout: float = 4.0) -> str | None:
    """네이버 업종 코드(예: 278) → 업종명(예: 반도체와반도체장비). 업종명은 거의 바뀌지 않아 프로세스 캐시."""
    code = str(industry_code or "").strip()
    if not code:
        return None
    if code in _INDUSTRY_NAME_CACHE:
        return _INDUSTRY_NAME_CACHE[code]
    payload = _get_json(f"https://m.stock.naver.com/api/stocks/industry/{code}?page=1&pageSize=1", timeout)
    name = str(((payload or {}).get("groupInfo") or {}).get("name") or "").strip() or None
    if name:
        _INDUSTRY_NAME_CACHE[code] = name
    return name


def fetch_naver_mobile_fundamentals(code: str, include_finance: bool = True, timeout: float = 5.0,
                                    include_industry_name: bool = True) -> dict:
    """국내 종목 펀더멘털을 모바일 API 두 곳에서 병렬 수집한다."""
    code = str(code or "").split(".", 1)[0].strip()
    if not code:
        return {"ok": False, "reason": "종목코드 없음"}
    urls = {"integration": f"{_BASE}/{code}/integration"}
    if include_finance:
        urls["annual"] = f"{_BASE}/{code}/finance/annual"
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = {key: pool.submit(_get_json, url, timeout) for key, url in urls.items()}
        payloads = {key: future.result() for key, future in futures.items()}
    result: dict = {"ok": False, "code": code, "source": "naver_mobile_api"}
    if payloads.get("integration"):
        result.update(parse_integration(payloads["integration"]))
        result["ok"] = True
        if include_industry_name and result.get("industry_code"):
            result["industry_name"] = fetch_naver_industry_name(result["industry_code"], timeout=min(timeout, 4.0))
    if payloads.get("annual"):
        result.update(parse_annual_finance(payloads["annual"]))
        result["ok"] = True
    if not result["ok"]:
        result["reason"] = "네이버 모바일 API 응답 없음"
    return result
