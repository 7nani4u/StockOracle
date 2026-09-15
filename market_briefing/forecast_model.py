"""예측 탭·뉴스 탭에서 공통으로 쓰는 순수 계산 함수.

외부 호출 없이 입력 배열만으로 계산해 단위 테스트가 가능하도록 분리했다.

- 변동성 기반 예측 구간: 일간 로그수익률 표준편차(σ)와 ATR 대체 추정치를 결합하고,
  보유 기간 H 거래일의 σ√H 로 가격 범위를 만든다. 시나리오 비중에서 나온 방향 우위는
  최대 ±0.35σ√H 의 약한 기울기로만 반영해, 방향 라벨·기준가·범위가 서로 모순되지 않게 한다.
- 도달 가능성: 무추세 브라운 운동의 최대값 분포 P(max ≥ a) = 2(1-Φ(a/σ√T)) 로
  "기간 내 한 번이라도 닿을 가능성"을 계산한다. 적중 확률 보장이 아니라 변동성 척도다.
- 뉴스 정규화: 게시 시각을 UTC ISO 로 통일하고, 오래된 기사 제외·최신순 정렬·중복 제거·
  종목 연관성 판정을 한 곳에서 처리한다.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable

KST = timezone(timedelta(hours=9))
Z_P90 = 1.2815515655446004   # 10%/90% 분위
Z_P95 = 1.6448536269514722   # 5%/95% 분위
_TOUCH_Z_25 = 1.1503493803760079  # P(max≥a)=25% 가 되는 a/(σ√n)
_TOUCH_Z_50 = 0.6744897501960817  # P(max≥a)=50%
# 브라운 운동에서 하루 True Range 기댓값 ≈ √(8/π)·σ ≈ 1.596σ
ATR_TO_SIGMA = 1.596


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def daily_log_volatility(closes: Iterable[Any], window: int = 60, min_obs: int = 15) -> tuple[float | None, int]:
    """최근 window 개 일간 로그수익률의 표본 표준편차와 관측 수를 반환한다."""
    values = [v for v in (_finite(c) for c in closes or []) if v is not None and v > 0]
    returns = [math.log(b / a) for a, b in zip(values[:-1], values[1:])][-window:]
    if len(returns) < min_obs:
        return None, len(returns)
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(max(variance, 0.0)), len(returns)


def blended_daily_sigma(closes: Iterable[Any], last_price: float, atr: float | None,
                        atr_observed: bool = True) -> dict:
    """실현 변동성과 ATR 기반 추정을 RMS 결합한 일간 σ(비율)."""
    realized, n_obs = daily_log_volatility(closes)
    atr_sigma = None
    atr_value = _finite(atr)
    if atr_value and atr_value > 0 and last_price > 0:
        atr_sigma = atr_value / last_price / ATR_TO_SIGMA
    if realized is not None and atr_sigma is not None and atr_observed:
        sigma = math.sqrt(0.5 * realized ** 2 + 0.5 * atr_sigma ** 2)
        basis = f"최근 {n_obs}일 실현 변동성 + ATR 결합"
    elif realized is not None:
        sigma = realized
        basis = f"최근 {n_obs}일 실현 변동성"
    elif atr_sigma is not None:
        sigma = atr_sigma
        basis = "ATR 기반 추정(수익률 표본 부족)" if atr_observed else "대체 ATR 기반 추정(표본 부족)"
    else:
        return {"sigma": None, "basis": "변동성 산출 불가", "observations": n_obs, "realized": None, "atr_based": None}
    # 호가 단위·거래정지 등으로 0 에 가까운 σ 가 나오면 범위가 붕괴하므로 하한을 둔다.
    sigma = max(sigma, 0.002)
    return {
        "sigma": sigma, "basis": basis, "observations": n_obs,
        "realized": realized, "atr_based": atr_sigma,
    }


def touch_probability(price: float, level: float, sigma_daily: float, days: int) -> float | None:
    """무추세 가정에서 days 거래일 안에 level 을 한 번 이상 터치할 가능성(0~1)."""
    if not price or not level or price <= 0 or level <= 0 or not sigma_daily or sigma_daily <= 0 or days <= 0:
        return None
    gap = abs(math.log(level / price))
    if gap == 0:
        return 1.0
    return max(0.0, min(1.0, 2.0 * (1.0 - normal_cdf(gap / (sigma_daily * math.sqrt(days))))))


def touch_day_window(price: float, level: float, sigma_daily: float, horizon_days: int) -> dict:
    """도달 가능성이 25%·50%에 이르는 거래일 수(무추세 가정)와 기간 내 여부."""
    if not price or not level or price <= 0 or level <= 0 or not sigma_daily or sigma_daily <= 0:
        return {"days": [1, max(1, horizon_days)], "within_horizon": False, "basis": "변동성 미확보"}
    z = abs(math.log(level / price)) / sigma_daily
    early = max(1, math.ceil((z / _TOUCH_Z_25) ** 2))
    typical = max(early, math.ceil((z / _TOUCH_Z_50) ** 2))
    horizon = max(1, int(horizon_days))
    return {
        "days": [min(early, horizon), min(typical, horizon)],
        "raw_days": [early, typical],
        "within_horizon": typical <= horizon,
        "basis": "무추세 변동성 기준 도달 가능성 25%~50% 구간",
    }


def build_forecast_summary(*, last_price: float, sigma_daily: float | None, horizon_days: int,
                           up_prob: float, down_prob: float, tilt: float = 0.35) -> dict | None:
    """시나리오 비중과 변동성으로 기준가·P10~P90·P5~P95 범위를 만든다."""
    if not sigma_daily or sigma_daily <= 0 or not last_price or last_price <= 0 or horizon_days <= 0:
        return None
    sigma_h = sigma_daily * math.sqrt(horizon_days)
    edge = max(-1.0, min(1.0, (float(up_prob) - float(down_prob)) / 100.0))
    drift = edge * tilt * sigma_h

    def _price(z: float) -> float:
        return last_price * math.exp(drift + z * sigma_h)

    base = _price(0.0)
    if up_prob >= down_prob + 8:
        direction_key, direction = "up", "상승 우위"
    elif down_prob >= up_prob + 8:
        direction_key, direction = "down", "하락 우위"
    else:
        direction_key, direction = "neutral", "중립"
    if sigma_h < 0.05:
        uncertainty = "낮음"
    elif sigma_h < 0.10:
        uncertainty = "보통"
    elif sigma_h < 0.18:
        uncertainty = "높음"
    else:
        uncertainty = "매우 높음"
    return {
        "base_price": base,
        "expected_return_pct": (base / last_price - 1.0) * 100.0,
        "range_p10_p90": [_price(-Z_P90), _price(Z_P90)],
        "range_p05_p95": [_price(-Z_P95), _price(Z_P95)],
        "sigma_daily_pct": sigma_daily * 100.0,
        "sigma_horizon_pct": sigma_h * 100.0,
        "direction_key": direction_key,
        "direction": direction,
        "edge": edge,
        "uncertainty": uncertainty,
    }


# ── 뉴스 정규화 ───────────────────────────────────────────────────────────

_CORP_SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|n\.?v|s\.?a|ag|se|class [a-c])\b\.?",
    re.IGNORECASE,
)


def parse_news_datetime(value: Any, now: datetime | None = None) -> datetime | None:
    """RSS(RFC822)·ISO·네이버('2026.09.12 20:07', '09.12')·epoch 를 UTC aware datetime 으로."""
    if value is None or value == "":
        return None
    now = now or datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        seconds = float(value) / 1000.0 if abs(float(value)) >= 1e11 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass
    iso = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=KST)
    except ValueError:
        pass
    match = re.match(r"^(\d{4})[./-](\d{1,2})[./-](\d{1,2})(?:\.?\s+(\d{1,2}):(\d{2}))?", text)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        hour, minute = int(match.group(4) or 12), int(match.group(5) or 0)
        try:
            return datetime(year, month, day, hour, minute, tzinfo=KST).astimezone(timezone.utc)
        except ValueError:
            return None
    match = re.match(r"^(\d{1,2})[./-](\d{1,2})(?:\s+(\d{1,2}):(\d{2}))?$", text)
    if match:
        local_now = now.astimezone(KST)
        month, day = int(match.group(1)), int(match.group(2))
        hour, minute = int(match.group(3) or 12), int(match.group(4) or 0)
        try:
            candidate = datetime(local_now.year, month, day, hour, minute, tzinfo=KST)
        except ValueError:
            return None
        if candidate - local_now > timedelta(hours=36):
            candidate = candidate.replace(year=local_now.year - 1)
        return candidate.astimezone(timezone.utc)
    return None


def company_name_terms(*names: Any) -> list[str]:
    """회사명 후보에서 검색·연관성 판정용 핵심 표기를 만든다 (법인 접미사 제거)."""
    terms: list[str] = []
    for name in names:
        raw = str(name or "").strip()
        if not raw:
            continue
        cleaned = _CORP_SUFFIX_RE.sub(" ", raw)
        cleaned = re.sub(r"[(),.&]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        for candidate in (raw, cleaned):
            if candidate and len(candidate) >= 2 and candidate.lower() not in (t.lower() for t in terms):
                terms.append(candidate)
        words = cleaned.split(" ")
        # "Samsung Electronics" → "Samsung" 처럼 첫 단어가 충분히 고유하면 보조어로 사용
        if len(words) >= 2 and len(words[0]) >= 4 and words[0].lower() not in (t.lower() for t in terms):
            terms.append(words[0])
    return terms


def is_relevant_title(title: str, terms: Iterable[str], tickers: Iterable[str] = ()) -> bool:
    """제목에 회사명 핵심어 또는 티커(단어 경계)가 포함되는지 판정한다."""
    text = str(title or "")
    if not text:
        return False
    compact = re.sub(r"\s+", "", text).lower()
    for term in terms or []:
        term = str(term or "").strip()
        if len(term) < 2:
            continue
        if re.search(r"[가-힣]", term):
            if re.sub(r"\s+", "", term).lower() in compact:
                return True
        elif re.search(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", text, re.IGNORECASE):
            return True
    for ticker in tickers or []:
        ticker = str(ticker or "").split(".", 1)[0].strip()
        if len(ticker) >= 2 and re.search(r"(?<![A-Za-z0-9])" + re.escape(ticker) + r"(?![A-Za-z0-9])", text, re.IGNORECASE):
            return True
    return False


def news_dedupe_key(title: str) -> str:
    text = str(title or "").replace(" ", " ").strip()
    if " - " in text:
        text = text.rsplit(" - ", 1)[0]
    return re.sub(r"[^0-9a-z가-힣]", "", text.lower())


def normalize_news_items(items: Iterable[dict] | None, *, now: datetime | None = None,
                         max_age_days: float = 30.0, relevance_terms: Iterable[str] | None = None,
                         tickers: Iterable[str] = (), require_relevance: bool = False,
                         date_keys: tuple[str, ...] = ("published_at", "published", "date", "datetime"),
                         title_keys: tuple[str, ...] = ("title", "title_ko", "original_title"),
                         seen: set | None = None) -> tuple[list[dict], dict]:
    """게시 시각 통일·오래된 기사 제외·연관성 필터·중복 제거 후 최신순으로 정렬한다."""
    now = now or datetime.now(timezone.utc)
    seen = seen if seen is not None else set()
    terms = list(relevance_terms or [])
    stats = {"input": 0, "kept": 0, "stale_dropped": 0, "irrelevant_dropped": 0,
             "duplicate_dropped": 0, "undated": 0}
    kept: list[tuple[datetime | None, dict]] = []
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        stats["input"] += 1
        titles = [str(raw.get(key) or "") for key in title_keys if raw.get(key)]
        title = titles[0] if titles else ""
        if not title:
            continue
        key = news_dedupe_key(raw.get("original_title") or raw.get("title") or title)
        if not key or key in seen:
            stats["duplicate_dropped"] += 1
            continue
        if require_relevance and terms and not any(is_relevant_title(t, terms, tickers) for t in titles):
            stats["irrelevant_dropped"] += 1
            continue
        published = None
        for date_key in date_keys:
            published = parse_news_datetime(raw.get(date_key), now)
            if published:
                break
        item = dict(raw)
        if published:
            age_hours = max(0.0, (now - published).total_seconds() / 3600.0)
            if age_hours > max_age_days * 24.0:
                stats["stale_dropped"] += 1
                continue
            item["published_at"] = published.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            item["age_hours"] = round(age_hours, 1)
            item["stale"] = age_hours > 7 * 24.0
        else:
            stats["undated"] += 1
            item["age_hours"] = None
            item["stale"] = None
        seen.add(key)
        kept.append((published, item))
    kept.sort(key=lambda row: row[0] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    stats["kept"] = len(kept)
    return [item for _, item in kept], stats
