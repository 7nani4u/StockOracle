"""📈 종목별 분석 — 예측 평가 + 사후 검증 모듈.

출처:
  예측 로직   → k-ant-daily /daily-report 스킬의 3-신호 매트릭스 재현
  검증 로직   → k-ant-daily/scripts/compute_review.py
  표시 로직   → k-ant-daily/scripts/render.py (_normalize_stocks, _normalize_stock_*)

핵심 함수:
  analyze_stock()         — 단일 종목 3-신호 평가 → 추천/신뢰도 판정
  classify_prediction()   — (추천, 실제등락%) → (적중/부분/실패, 노트)
  build_stock_report()    — 종목 스냅샷 리스트 → 전체 정규화 보고서
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from .labels import (
    IMPACT_LABEL, RECOMMENDATION_LABEL, OUTCOME_LABEL,
    SENTIMENT_LABEL, OVERNIGHT_LABEL, CONFIDENCE_LABEL,
    DIRECTION_META,
)

KST = timezone(timedelta(hours=9))

# ── 헤드라인 감성 패턴 ───────────────────────────────────────────────────────

_POS_PATTERNS = re.compile(
    r"(상승|급등|강세|반등|신고가|최고치|돌파|호실적|실적\s*개선|영업이익\s*증가|"
    r"어닝\s*서프라이즈|상향|수주|체결|공급\s*계약|라이선스|승인|허가|특허|"
    r"자사주\s*매입|자사주\s*소각|배당\s*(인상|증가|확대)|흑자\s*전환|"
    r"수혜|성장|호재|랠리|폭등|인수|M&A|골든크로스|최대\s*실적|사상\s*최대|"
    r"어닝\s*비트|목표가\s*상향|매수\s*추천|투자의견\s*상향|긍정적|수주잔고|호조|낙관)"
)
_NEG_PATTERNS = re.compile(
    r"(하락|급락|약세|부진|적자|쇼크|저조|감소|하향|실패|철회|취소|"
    r"리스크|부담|제재|과징금|횡령|배임|소송|경고|우려|손실|위기|"
    r"낙폭|추락|불안|악재|하회|둔화|논란|감원|구조조정|"
    r"적자\s*전환|매도\s*추천|투자의견\s*하향|목표가\s*하향|부정적|"
    r"데드크로스|실망감|어닝\s*쇼크|실적\s*쇼크|규제|조사|경영권\s*분쟁)"
)


def _auto_impact(title: str) -> str:
    pos = bool(_POS_PATTERNS.search(title))
    neg = bool(_NEG_PATTERNS.search(title))
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return "neutral"


def _annotate(obj: dict) -> None:
    impact = obj.get("impact")
    if impact and "impact_label" not in obj:
        obj["impact_label"] = IMPACT_LABEL.get(impact, impact)


# ── 타임스탬프 헬퍼 ──────────────────────────────────────────────────────────

def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    s = str(raw).strip()
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _time_ago(raw: str | None, now: datetime) -> str:
    dt = _parse_ts(raw)
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=now.tzinfo or KST)
    secs = int((now - dt).total_seconds())
    if secs < 60:
        return "방금 전"
    if secs < 3600:
        return f"{secs // 60}분 전"
    if secs < 86400:
        return f"{secs // 3600}시간 전"
    days = secs // 86400
    return f"{days}일 전" if days < 7 else dt.strftime("%m/%d")


# ── 3-신호 매트릭스 ──────────────────────────────────────────────────────────
#
# k-ant-daily /daily-report 스킬은 Claude 에이전트가 summary.json 을 작성하지만,
# 이 모듈은 뉴스 헤드라인 감성 + 오버나이트 신호 + 52주 위치 세 가지로
# 규칙 기반 추천을 도출한다. 에이전트 없이 빠른 초기 평가에 활용하거나
# Claude API 호출 전 사전 필터링에 사용한다.

def _news_sentiment(news: list[dict]) -> str:
    """최근 뉴스에서 감성 집계 → positive / negative / neutral."""
    pos = neg = 0
    for n in news:
        impact = n.get("impact") or _auto_impact(n.get("title", ""))
        if impact == "positive":
            pos += 1
        elif impact == "negative":
            neg += 1
    total = pos + neg
    if not total:
        return "neutral"
    ratio = pos / total
    if ratio >= 0.6:
        return "positive"
    if ratio <= 0.3:
        return "negative"
    return "neutral"


def _overnight_direction(signal: dict | str | None) -> str:
    if isinstance(signal, dict):
        return signal.get("direction") or "neutral"
    return str(signal) if signal else "neutral"


def _price_position_signal(history: dict) -> str:
    """52주 포지션 기반 추가 신호.

    pos_52w_pct >= 80 → 고가권 (상승 탄력 약함 / 단기 경계)
    pos_52w_pct <= 20 → 저가권 (반등 탄력 가능성)
    """
    pos = history.get("pos_52w_pct")
    if pos is None:
        return "neutral"
    if pos >= 80:
        return "high_zone"    # 52주 고점 근처 → 추가 상승 제한적
    if pos <= 20:
        return "low_zone"     # 52주 저점 근처 → 반등 가능성
    return "neutral"


def _volume_spike(stock: dict) -> bool:
    """오늘 거래량이 20일 평균의 2배 이상 → 투기성 거래량 급증."""
    vol_today = (stock.get("quote") or {}).get("volume")
    vol_avg   = (stock.get("history") or {}).get("volume_20d_avg")
    if vol_today and vol_avg and vol_avg > 0:
        return (vol_today / vol_avg) >= 2.0
    return False


def _three_signal_recommendation(
    news_sent:    str,
    overnight:    str,
    price_zone:   str,
) -> tuple[str, str, str]:
    """3-신호 매트릭스 → (recommendation, confidence, rationale_hint).

    신호 조합 로직:
      모두 일치 → strong signal
      2/3 일치  → moderate signal
      혼재      → hold

    반환 예:  ("buy", "medium", "뉴스 긍정+간밤 강세, 저가권 진입 가능성")
    """
    up_signals   = sum([news_sent == "positive", overnight == "up",   price_zone == "low_zone"])
    down_signals = sum([news_sent == "negative", overnight == "down", price_zone == "high_zone"])

    if up_signals == 3:
        return "strong_buy",  "high",   "3개 신호 모두 상승 일치"
    if up_signals == 2:
        rec = "buy"
        conf = "medium"
        hints = []
        if news_sent == "positive": hints.append("뉴스 긍정")
        if overnight == "up":       hints.append("간밤 강세")
        if price_zone == "low_zone":hints.append("저가권 위치")
        return rec, conf, " · ".join(hints)
    if down_signals == 3:
        return "strong_sell", "high",   "3개 신호 모두 하락 일치"
    if down_signals == 2:
        rec = "sell"
        conf = "medium"
        hints = []
        if news_sent == "negative":  hints.append("뉴스 부정")
        if overnight == "down":      hints.append("간밤 약세")
        if price_zone == "high_zone":hints.append("고가권 위치")
        return rec, conf, " · ".join(hints)

    # 혼재 — hold
    conf = "low"
    return "hold", conf, "신호 혼재 — 추가 확인 필요"


# ── 공개 API: 단일 종목 분석 ────────────────────────────────────────────────

def analyze_stock(stock: dict) -> dict:
    """📈 단일 종목 3-신호 분석.

    Args:
        stock: fetch_stock_snapshot() 반환값
            필수 키: code, quote, news, history, overnight_signal

    Returns:
        입력 dict 를 인플레이스 수정 후 반환.
        추가 키:
          news_sentiment       — positive/negative/neutral
          overnight_signal_dir — up/down/neutral
          price_zone           — high_zone/low_zone/neutral
          volume_spike         — bool
          recommendation       — strong_buy/.../strong_sell
          recommendation_label — 한국어 레이블
          confidence           — high/medium/low
          confidence_label     — 한국어
          rationale            — 자동 생성 근거 문장
          direction_arrow/label/cls — UI 표시용
          volume_ratio         — float|None
    """
    s = dict(stock)
    news = s.get("news") or []
    news_sent    = _news_sentiment(news)
    overnight    = _overnight_direction(s.get("overnight_signal"))
    price_zone   = _price_position_signal(s.get("history") or {})
    is_spike     = _volume_spike(s)

    rec, conf, hint = _three_signal_recommendation(news_sent, overnight, price_zone)

    s["news_sentiment"]       = news_sent
    s["overnight_signal_dir"] = overnight
    s["price_zone"]           = price_zone
    s["volume_spike"]         = is_spike

    # 거래량 비율
    vol_today = (s.get("quote") or {}).get("volume")
    vol_avg   = (s.get("history") or {}).get("volume_20d_avg")
    if vol_today and vol_avg and vol_avg > 0:
        s["volume_ratio"] = round(vol_today / vol_avg, 2)

    # 거래량 급증 시 신뢰도 보정 (변동성 급증 = 예측 불확실)
    if is_spike and conf == "high":
        conf = "medium"
        hint += " (거래량 급증으로 신뢰도 하향)"

    s["recommendation"]       = rec
    s["recommendation_label"] = RECOMMENDATION_LABEL.get(rec, rec)
    s["confidence"]           = conf
    s["confidence_label"]     = CONFIDENCE_LABEL.get(conf, conf)
    s["rationale"]            = hint

    # 신호 레이블
    s["news_sentiment_label"]  = SENTIMENT_LABEL.get(news_sent, news_sent)
    sig = s.get("overnight_signal")
    if isinstance(sig, dict) and sig.get("direction"):
        sig["label"] = OVERNIGHT_LABEL.get(sig["direction"], sig["direction"])

    # 방향 메타
    meta = DIRECTION_META.get(rec, DIRECTION_META["hold"])
    s["direction_arrow"] = meta["arrow"]
    s["direction_label"] = meta["label"]
    s["direction_cls"]   = meta["cls"]

    return s


# ── 공개 API: 예측 사후 검증 ────────────────────────────────────────────────

def classify_prediction(recommendation: str, actual_pct: float) -> tuple[str, str]:
    """(추천값, 실제 등락%) → (outcome, 설명 노트).

    출처: k-ant-daily/scripts/compute_review.py classify()

    outcome:  "hit" | "partial" | "miss" | "n/a"
    """
    rec = recommendation
    if rec == "strong_buy":
        if actual_pct >= 2.0:
            return "hit",     f"강한 상승 기대 · 실제 +{actual_pct:.2f}% · 강한 상승 확인"
        if actual_pct >= 0.0:
            return "partial", f"강한 상승 기대 · 실제 +{actual_pct:.2f}% · 방향은 맞았으나 강도 약함"
        return "miss",        f"강한 상승 기대 · 실제 {actual_pct:+.2f}% · 방향 틀림"
    if rec == "buy":
        if actual_pct > 0.5:
            return "hit",     f"상승 기대 · 실제 +{actual_pct:.2f}% · 상승 적중"
        if actual_pct >= -0.5:
            return "partial", f"상승 기대 · 실제 {actual_pct:+.2f}% · 사실상 보합"
        return "miss",        f"상승 기대 · 실제 {actual_pct:.2f}% · 하락"
    if rec == "hold":
        if abs(actual_pct) < 1.5:
            return "hit",     f"관망 · 실제 {actual_pct:+.2f}% · 보합 구간 적중"
        if abs(actual_pct) < 3.0:
            return "partial", f"관망 · 실제 {actual_pct:+.2f}% · 움직임은 있었음"
        return "miss",        f"관망 · 실제 {actual_pct:+.2f}% · 큰 변동 놓침"
    if rec == "sell":
        if actual_pct < -0.5:
            return "hit",     f"하락 경계 · 실제 {actual_pct:.2f}% · 하락 적중"
        if actual_pct <= 0.5:
            return "partial", f"하락 경계 · 실제 {actual_pct:+.2f}% · 사실상 보합"
        return "miss",        f"하락 경계 · 실제 +{actual_pct:.2f}% · 상승"
    if rec == "strong_sell":
        if actual_pct <= -2.0:
            return "hit",     f"강한 하락 경계 · 실제 {actual_pct:.2f}% · 강한 하락 확인"
        if actual_pct <= 0.0:
            return "partial", f"강한 하락 경계 · 실제 {actual_pct:.2f}% · 방향은 맞았으나 강도 약함"
        return "miss",        f"강한 하락 경계 · 실제 +{actual_pct:.2f}% · 방향 틀림"
    return "n/a", "투자의견 없음"


def _direction_from_pct(pct: float) -> str:
    if pct > 0.3:
        return "up"
    if pct < -0.3:
        return "down"
    return "flat"


def _matrix_direction(rec: str) -> str:
    if rec in ("strong_buy", "buy"):
        return "up"
    if rec in ("strong_sell", "sell"):
        return "down"
    return "flat"


# ── 공개 API: 종목 리스트 정규화 보고서 ─────────────────────────────────────

def build_stock_report(
    stock_snapshots: list[dict],
    evening_quotes:  dict[str, dict] | None = None,
) -> dict:
    """📈 종목 리스트 전체 보고서.

    Args:
        stock_snapshots:
            fetch_stock_list_snapshot() 반환값.
        evening_quotes:
            저녁 검증용 실제 종가 dict {"code": quote_dict}.
            제공 시 각 종목에 result (hit/partial/miss) 를 추가로 계산.

    Returns:
        {
          "generated_at":   str,
          "stocks":         list[dict],   # analyze_stock() 결과 + optional result
          "groups": {                     # 추천별 그룹 (UI 레이아웃용)
              "strong_buy": list,
              "buy":        list,
              "hold":       list,
              "sell":       list,
              "strong_sell":list,
          },
          "review_accuracy": dict|None,   # evening_quotes 있을 때만
        }
    """
    now = datetime.now(KST)
    analyzed: list[dict] = []

    # 뉴스 24시간 컷오프 + 시간표시 정규화
    cutoff = now - timedelta(hours=24)
    for raw_stock in stock_snapshots:
        s = analyze_stock(raw_stock)

        # 뉴스 24h 필터 + time_ago 주입
        filtered_news = []
        latest_dt: datetime | None = None
        for n in (s.get("news") or []):
            if "published_at" not in n and n.get("date"):
                n["published_at"] = n["date"]
            dt = _parse_ts(n.get("published_at"))
            if dt is None:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=KST)
            if dt < cutoff:
                continue
            if not n.get("impact"):
                n["impact"] = _auto_impact(n.get("title", ""))
            _annotate(n)
            n["time_ago"] = _time_ago(n.get("published_at"), now)
            filtered_news.append(n)
            if latest_dt is None or dt > latest_dt:
                latest_dt = dt
        filtered_news.sort(key=lambda n: n.get("published_at") or "", reverse=True)
        s["news"]       = filtered_news
        s["news_count"] = len(filtered_news)
        if latest_dt:
            s["latest_news_at"]  = latest_dt.isoformat()
            s["latest_news_ago"] = _time_ago(latest_dt.isoformat(), now)

        # 저녁 검증
        if evening_quotes and s.get("code") in evening_quotes:
            eq   = evening_quotes[s["code"]]
            pct  = eq.get("change_pct_num")
            if pct is None and eq.get("change_pct"):
                try:
                    pct = float(str(eq["change_pct"]).replace("%", "").replace("+", ""))
                except ValueError:
                    pass
            rec = s.get("recommendation") or ""
            if pct is not None and rec:
                outcome, note = classify_prediction(rec, float(pct))
                pred_dir   = _matrix_direction(rec)
                actual_dir = _direction_from_pct(float(pct))
                s["result"] = {
                    "outcome":            outcome,
                    "label":              OUTCOME_LABEL.get(outcome, outcome),
                    "actual_change_pct":  pct,
                    "predicted_direction":pred_dir,
                    "actual_direction":   actual_dir,
                    "note":               note,
                }
                s["quote"] = {**s.get("quote", {}), **eq}
            else:
                s["result"] = {"outcome": "n/a", "label": "데이터 없음", "note": "종가 없음"}

        analyzed.append(s)

    # 추천별 그룹핑 (최신 뉴스 우선)
    groups: dict[str, list] = {k: [] for k in ("strong_buy","buy","hold","sell","strong_sell")}
    for s in analyzed:
        rec = s.get("recommendation", "hold")
        groups.get(rec, groups["hold"]).append(s)
    for g in groups.values():
        g.sort(key=lambda x: x.get("latest_news_at") or "", reverse=True)

    # 검증 집계
    review_accuracy = None
    if evening_quotes:
        hits = partial = misses = 0
        dir_correct = dir_total = 0
        for s in analyzed:
            r = s.get("result") or {}
            o = r.get("outcome")
            if o == "hit":       hits += 1
            elif o == "partial": partial += 1
            elif o == "miss":    misses += 1
            else: continue
            pred = r.get("predicted_direction", "")
            actual = r.get("actual_direction", "")
            if pred:
                dir_total += 1
                if pred == actual:
                    dir_correct += 1
        total = hits + partial + misses
        review_accuracy = {
            "total":               total,
            "hits":                hits,
            "partial":             partial,
            "misses":              misses,
            "hit_rate":            round((hits + 0.5 * partial) / total, 3) if total else 0,
            "directional_accuracy":round(dir_correct / dir_total, 3) if dir_total else 0,
        }

    return {
        "generated_at":   now.isoformat(),
        "stocks":         analyzed,
        "groups":         groups,
        "review_accuracy":review_accuracy,
    }
