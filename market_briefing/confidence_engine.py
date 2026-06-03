"""
confidence_engine.py — 신호 신뢰도 종합 엔진 (Market-Analysis 이식 + StockOracle 통합)

Market-Analysis(JS)의 confidence 조정 로직을 Python으로 이식하여, StockOracle의
종목 분석 신호에 다음 6개 보정 요소를 한 곳에서 일관되게 적용한다.

  1. 거시경제 체제 중첩 분석   get_macro_regime()        — VIX·S&P500·DXY → Risk-On/Off/Neutral/Transition
  2. 섹터 상대 점수            get_sector_relative()     — 섹터 ETF 5일 추세 vs 개별 종목 신호
  3. 실적 발표 임박 페널티     get_earnings_proximity()  — 실적일 5거래일 이내 → confidence 상한
  4. 신뢰 구간 반환            confidence_interval()     — 소스 분산·거시·실적 반영 lower/upper/spread
  5. 불일치 페널티             disagreement_penalty()    — 소스 점수 분산 → confidence 상한
  6. 뉴스 감정 분석            analyze_news_sentiment()  — FinBERT(HF) → 키워드 fallback + 최근성 감쇠

핵심 진입점:
  build_signal_confidence(...) — 위 요소를 종합해 단일 confidence + confidence_interval 반환.

설계 원칙:
  · 모든 외부 호출(yfinance/HuggingFace)은 실패해도 안전한 기본값으로 degrade — 전체 분석 비중단.
  · 점수 단위는 0~100 (50 = 중립). confidence도 0~100.
  · 시장별(KRX/US) 공통 적용. 섹터 ETF(XL*)는 US 한정 → KRX는 자연히 무보정.
"""
from __future__ import annotations

import math
import os
import time
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 외부 의존성 (실패 허용) ──────────────────────────────────────────────────
try:
    import yfinance as yf
except Exception:                       # pragma: no cover
    yf = None
try:
    import requests
except Exception:                       # pragma: no cover
    requests = None


# ════════════════════════════════════════════════════════════════════════════
# 공통 유틸 / 캐시
# ════════════════════════════════════════════════════════════════════════════
_CACHE: Dict[str, Any] = {}             # key -> (value, expire_ts)


def _cache_get(key: str):
    v = _CACHE.get(key)
    if v and v[1] > time.time():
        return v[0]
    return None


def _cache_set(key: str, value, ttl: float):
    _CACHE[key] = (value, time.time() + ttl)
    if len(_CACHE) > 300:               # 단순 만료 청소
        now = time.time()
        for k in [k for k, (_, e) in list(_CACHE.items()) if e < now]:
            _CACHE.pop(k, None)
    return value


def _num(v, default: float = 0.0) -> float:
    """방어적 숫자 변환."""
    try:
        if v is None:
            return default
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _closes(symbol: str, period: str = "1mo") -> List[float]:
    """yfinance 종가 리스트 (오래된→최신). 실패 시 빈 리스트."""
    if yf is None:
        return []
    ck = f"closes|{symbol}|{period}"
    cached = _cache_get(ck)
    if cached is not None:
        return cached
    try:
        df = yf.Ticker(symbol).history(period=period)
        closes = [float(x) for x in df["Close"].dropna().tolist()] if not df.empty else []
    except Exception:
        closes = []
    return _cache_set(ck, closes, 600)   # 10분 캐시


def _pct_change(closes: List[float], lookback: int) -> Optional[float]:
    if len(closes) <= lookback:
        return None
    base = closes[-1 - lookback]
    if base == 0:
        return None
    return (closes[-1] - base) / base * 100.0


# ════════════════════════════════════════════════════════════════════════════
# 1. 거시경제 체제 중첩 분석 (VIX · S&P500 · DXY)
# ════════════════════════════════════════════════════════════════════════════
def get_macro_regime() -> Dict[str, Any]:
    """VIX 추이·S&P500 추세·DXY 변화로 시장 체제를 판별.

    Returns:
        {regime, components, confidence_weight, error?}
        regime ∈ {"Risk-On", "Risk-Off", "Neutral", "Transition"}
        confidence_weight: confidence에 더할 보정치 (음수=페널티)
    """
    cached = _cache_get("macro_regime")
    if cached is not None:
        return cached

    out: Dict[str, Any] = {"regime": "Neutral", "components": {}, "confidence_weight": 0}
    try:
        vix = _closes("^VIX", "1mo")
        sp  = _closes("^GSPC", "1mo")
        dxy = _closes("DX-Y.NYB", "1mo")

        vix_level = vix[-1] if vix else None
        vix_chg5  = _pct_change(vix, 5) if len(vix) > 5 else None
        sp5       = _pct_change(sp, 5)
        sp10      = _pct_change(sp, 10)
        dxy5      = _pct_change(dxy, 5) or 0.0

        out["components"] = {
            "vix": {"level": round(vix_level, 2) if vix_level is not None else None,
                    "change5d_pct": round(vix_chg5, 2) if vix_chg5 is not None else None},
            "sp500": {"pct5d": round(sp5, 2) if sp5 is not None else None,
                      "pct10d": round(sp10, 2) if sp10 is not None else None},
            "dxy": {"pct5d": round(dxy5, 2)},
        }

        regime = "Neutral"
        if vix_level is not None and sp5 is not None:
            fear_on    = vix_level > 22
            fear_off   = vix_level < 16
            sp_up      = sp5 > 1.0
            sp_down    = sp5 < -1.0
            dxy_strong = dxy5 > 1.0
            # 최근 방향 급변(VIX 5일 급등/급락 + S&P와 10일 추세 상충) → Transition
            sp_flip = (sp5 is not None and sp10 is not None and (sp5 * sp10 < 0)
                       and abs(sp5) > 1.0)
            vix_spike = vix_chg5 is not None and abs(vix_chg5) > 25

            if fear_off and sp_up and not dxy_strong:
                regime = "Risk-On"
            elif fear_on and sp_down:
                regime = "Risk-Off"
            elif sp_flip or vix_spike:
                regime = "Transition"
            elif abs(sp5) < 0.5 and abs(dxy5) < 0.5 and vix_level < 22:
                regime = "Neutral"
            else:
                regime = "Transition"
        out["regime"] = regime
        out["confidence_weight"] = _regime_confidence_weight(regime)
    except Exception as e:
        out["error"] = str(e)

    return _cache_set("macro_regime", out, 600)


def _regime_confidence_weight(regime: str) -> int:
    """체제별 confidence 보정치 (Market-Analysis regimeBias.pen 대응)."""
    return {
        "Risk-On":    0,
        "Risk-Off":  -6,
        "Transition": -3,
        "Neutral":    0,
    }.get(regime, 0)


# ════════════════════════════════════════════════════════════════════════════
# 2. 섹터 상대 점수 (섹터 ETF 5일 추세 vs 개별 종목 신호)
# ════════════════════════════════════════════════════════════════════════════
# 대표 SPDR 섹터 ETF 매핑 (US 대형주 위주, 미등록 종목은 무보정)
_SECTOR_ETF = {
    # Tech
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "INTC": "XLK",
    "AVGO": "XLK", "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK", "CSCO": "XLK",
    "QCOM": "XLK", "TXN": "XLK", "AMAT": "XLK", "MU": "XLK", "PLTR": "XLK",
    "SMCI": "XLK", "NOW": "XLK", "INTU": "XLK",
    # Communication
    "META": "XLC", "GOOGL": "XLC", "GOOG": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "T": "XLC", "VZ": "XLC", "TMUS": "XLC", "SNAP": "XLC", "PINS": "XLC", "ROKU": "XLC",
    # Consumer Discretionary
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "NKE": "XLY", "MCD": "XLY",
    "SBUX": "XLY", "BKNG": "XLY", "ABNB": "XLY", "RIVN": "XLY", "F": "XLY", "GM": "XLY",
    # Consumer Staples
    "WMT": "XLP", "COST": "XLP", "PG": "XLP", "KO": "XLP", "PEP": "XLP", "MO": "XLP",
    # Health
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "PFE": "XLV", "MRK": "XLV", "ABBV": "XLV",
    "TMO": "XLV", "ABT": "XLV", "DHR": "XLV", "BMY": "XLV", "AMGN": "XLV",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF", "C": "XLF",
    "BLK": "XLF", "V": "XLF", "MA": "XLF", "AXP": "XLF", "SCHW": "XLF", "SOFI": "XLF", "COIN": "XLF",
    # Energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "PSX": "XLE", "MPC": "XLE",
    "VLO": "XLE", "OXY": "XLE", "EOG": "XLE",
    # Industrials
    "CAT": "XLI", "BA": "XLI", "UNP": "XLI", "UPS": "XLI", "HON": "XLI", "GE": "XLI",
    "RTX": "XLI", "DE": "XLI", "LMT": "XLI", "NOC": "XLI", "MMM": "XLI",
    # Utilities
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "AEP": "XLU", "SRE": "XLU", "D": "XLU",
    # Materials
    "LIN": "XLB", "SHW": "XLB", "APD": "XLB", "FCX": "XLB", "NEM": "XLB", "CF": "XLB",
    # Real Estate
    "PLD": "XLRE", "AMT": "XLRE", "CCI": "XLRE", "EQIX": "XLRE", "PSA": "XLRE",
}
_SECTOR_NAME = {
    "XLK": "Technology", "XLC": "Communication Services", "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples", "XLV": "Health Care", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities",
    "XLB": "Materials", "XLRE": "Real Estate",
}


def symbol_sector(symbol: str) -> Optional[Dict[str, str]]:
    if not symbol:
        return None
    etf = _SECTOR_ETF.get(symbol.upper().split(".")[0])
    return {"etf": etf, "name": _SECTOR_NAME.get(etf, etf)} if etf else None


def get_sector_relative(symbol: str, signal: str,
                        stock_pct5d: Optional[float] = None) -> Dict[str, Any]:
    """섹터 ETF 5일 추세 vs 개별 종목 신호/움직임 비교.

    Args:
        symbol: 종목 코드
        signal: "BUY" / "SELL" / "NEUTRAL"
        stock_pct5d: 개별 종목 5일 변화율(%) — 없으면 yfinance 조회

    Returns:
        {sector_relative_score, adjust, sector, aligned, reason}
        · sector_relative_score: -100~+100 (종목·섹터 정렬도)
        · adjust: confidence 보정치 (-6~+3)
    """
    out = {"sector_relative_score": 0, "adjust": 0, "sector": None,
           "aligned": None, "reason": None}
    sec = symbol_sector(symbol)
    if not sec:
        return out
    out["sector"] = sec
    try:
        etf_closes = _closes(sec["etf"], "1mo")
        etf_pct5d = _pct_change(etf_closes, 5)
        if etf_pct5d is None:
            return out
        if stock_pct5d is None:
            stock_pct5d = _pct_change(_closes(symbol, "1mo"), 5)
        stock_pct5d = _num(stock_pct5d)

        sector_rising  = etf_pct5d > 1.0
        sector_falling = etf_pct5d < -1.0

        # 상대 점수: 종목이 섹터 대비 얼마나 강한가 (방향 정렬 포함)
        rel = stock_pct5d - etf_pct5d
        out["sector_relative_score"] = int(_clamp(rel * 10, -100, 100))

        adjust = 0
        aligned = None
        reason = None
        if signal == "BUY":
            if sector_rising:
                adjust, aligned = 3, True
                reason = f"{sec['name']} 섹터 상승({etf_pct5d:+.1f}%)과 매수 신호 정렬 — 신뢰 강화"
            elif sector_falling:
                adjust, aligned = -6, False
                reason = f"{sec['name']} 섹터 하락({etf_pct5d:+.1f}%) 속 매수 신호 — 신뢰 약화"
        elif signal == "SELL":
            if sector_falling:
                adjust, aligned = 3, True
                reason = f"{sec['name']} 섹터 하락({etf_pct5d:+.1f}%)과 매도 신호 정렬 — 신뢰 강화"
            elif sector_rising:
                adjust, aligned = -6, False
                reason = f"{sec['name']} 섹터 상승({etf_pct5d:+.1f}%) 속 매도 신호 — 신뢰 약화"

        sec.update({"pct5d": round(etf_pct5d, 2), "rising": sector_rising,
                    "falling": sector_falling, "stock_pct5d": round(stock_pct5d, 2)})
        out.update({"adjust": adjust, "aligned": aligned, "reason": reason})
    except Exception as e:
        out["error"] = str(e)
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. 실적 발표 임박 페널티
# ════════════════════════════════════════════════════════════════════════════
def get_earnings_proximity(symbol: str) -> Dict[str, Any]:
    """다음 실적 발표일까지 남은 일수 조회 (yfinance calendar). 실패 시 fallback.

    Returns:
        {days_to_earnings, earnings_date?, source, earnings_risk}
    """
    out = {"days_to_earnings": None, "earnings_date": None,
           "source": "none", "earnings_risk": False}
    if not symbol or yf is None:
        return out
    ck = f"earnings|{symbol}"
    cached = _cache_get(ck)
    if cached is not None:
        return cached
    try:
        tk = yf.Ticker(symbol)
        next_date = None
        # 1순위: get_earnings_dates()
        try:
            ed = tk.get_earnings_dates(limit=8)
            if ed is not None and not ed.empty:
                import pandas as _pd
                now = _pd.Timestamp.now(tz=ed.index.tz) if ed.index.tz else _pd.Timestamp.now()
                future = [d for d in ed.index if d >= now]
                if future:
                    next_date = min(future)
                    out["source"] = "earnings_dates"
        except Exception:
            pass
        # 2순위(fallback): calendar
        if next_date is None:
            try:
                cal = tk.calendar
                ed_val = None
                if isinstance(cal, dict):
                    ed_val = cal.get("Earnings Date")
                    if isinstance(ed_val, (list, tuple)) and ed_val:
                        ed_val = ed_val[0]
                if ed_val is not None:
                    import pandas as _pd
                    next_date = _pd.Timestamp(ed_val)
                    out["source"] = "calendar"
            except Exception:
                pass

        if next_date is not None:
            import pandas as _pd
            now = _pd.Timestamp.now(tz=getattr(next_date, "tz", None)) \
                if getattr(next_date, "tz", None) else _pd.Timestamp.now()
            days = int(math.ceil((next_date - now).total_seconds() / 86400.0))
            out["days_to_earnings"] = days
            out["earnings_date"] = str(next_date.date()) if hasattr(next_date, "date") else str(next_date)
            out["earnings_risk"] = (0 <= days <= 5)
    except Exception as e:
        out["error"] = str(e)
    return _cache_set(ck, out, 3600)     # 1시간 캐시


def earnings_cap(days_to_earnings: Optional[int]) -> Dict[str, Any]:
    """실적 임박 → confidence 상한. {cap, reason}"""
    d = days_to_earnings
    if d is None or d < 0:
        return {"cap": 100, "reason": None}
    if d <= 1:
        return {"cap": 60, "reason": "Earnings within 1 trading day — binary event risk"}
    if d <= 5:
        return {"cap": 70, "reason": f"Earnings within 5 trading days ({d}d) — reduced predictiveness"}
    return {"cap": 100, "reason": None}


# ════════════════════════════════════════════════════════════════════════════
# 5. 불일치 페널티 (소스 점수 분산)
# ════════════════════════════════════════════════════════════════════════════
def disagreement_penalty(scores: List[float]) -> Dict[str, Any]:
    """AI·기술·심리·시장 점수 간 최대 차이 → confidence 상한.

    Returns:
        {disagreement_penalty, source_score_spread, confidence_cap, reason}
    """
    vals = [_num(s) for s in scores if s is not None]
    if len(vals) < 2:
        return {"disagreement_penalty": False, "source_score_spread": 0,
                "confidence_cap": 100, "reason": None}
    spread = max(vals) - min(vals)
    cap, triggered = 100, False
    if spread > 50:
        cap, triggered = 55, True
    elif spread > 35:
        cap, triggered = 65, True
    elif spread > 25:
        cap, triggered = 75, True
    reason = (f"High disagreement between AI, technical, sentiment, and market "
              f"scores (spread {spread:.0f})") if triggered else None
    return {"disagreement_penalty": triggered, "source_score_spread": round(spread, 1),
            "confidence_cap": cap, "reason": reason}


# ════════════════════════════════════════════════════════════════════════════
# 6. 뉴스 감정 분석 (FinBERT → 키워드 fallback + 최근성 감쇠)
# ════════════════════════════════════════════════════════════════════════════
_HF_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
_RECENCY_HALF_LIFE_H = 48.0

_BULLISH = {"surge", "surges", "rally", "soar", "jump", "gain", "gains", "rise", "rises",
            "high", "record", "boom", "bull", "bullish", "breakout", "upgrade", "beat",
            "beats", "strong", "growth", "profit", "buy", "outperform", "optimistic",
            "boost", "recover", "recovery", "momentum", "upside", "milestone", "approval",
            "급등", "강세", "상승", "호재", "신고가", "돌파", "최대", "흑자", "성장"}
_BEARISH = {"crash", "plunge", "drop", "drops", "fall", "falls", "decline", "low", "sell",
            "bear", "bearish", "loss", "losses", "miss", "misses", "weak", "warning",
            "fear", "risk", "cut", "cuts", "downgrade", "layoff", "bankruptcy", "debt",
            "recession", "crisis", "lawsuit", "fraud", "hack", "worst", "collapse", "tank",
            "급락", "약세", "하락", "악재", "신저가", "적자", "감소", "경고", "리스크"}


def _hf_token() -> Optional[str]:
    return (os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))


def _finbert(texts: List[str]) -> Optional[List[Dict]]:
    """HuggingFace FinBERT 추론. 실패/키없음/쿨다운 시 None."""
    if requests is None or not texts:
        return None
    if _cache_get("hf_cooldown"):        # 직전 실패 → 10분 쿨다운
        return None
    headers = {"Content-Type": "application/json"}
    tok = _hf_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    try:
        res = requests.post(_HF_URL, headers=headers,
                            json={"inputs": texts}, timeout=15)
        if res.status_code != 200:
            if res.status_code in (401, 403, 429, 503):
                _cache_set("hf_cooldown", True, 600)
            return None
        return res.json()
    except Exception:
        _cache_set("hf_cooldown", True, 600)
        return None


def _parse_finbert(item) -> Dict[str, float]:
    """FinBERT 단일 결과 → {score, label}. score = positive - negative (-1~1)."""
    pos = neg = 0.0
    seq = item if isinstance(item, list) else [item]
    for r in seq:
        if not isinstance(r, dict):
            continue
        lab = (r.get("label") or "").lower()
        sc = _num(r.get("score"))
        if lab == "positive":
            pos = sc
        elif lab == "negative":
            neg = sc
    score = pos - neg
    label = "positive" if score > 0.3 else "negative" if score < -0.3 else "neutral"
    return {"score": score, "label": label}


def _keyword_sentiment(text: str) -> Dict[str, float]:
    low = (text or "").lower()
    words = set(__import__("re").split(r"\W+", low))
    bull = len(words & _BULLISH)
    bear = len(words & _BEARISH)
    score = _clamp((bull - bear) / 3.0, -1.0, 1.0)
    label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
    return {"score": score, "label": label}


def _recency_weight(age_hours: Optional[float]) -> float:
    if age_hours is None:
        return 0.5
    if age_hours < 0:
        return 1.0
    return math.exp(-age_hours / _RECENCY_HALF_LIFE_H)


def analyze_news_sentiment(news_items: List[Dict]) -> Dict[str, Any]:
    """뉴스 헤드라인 감정 분석 — FinBERT 우선, 실패 시 키워드, 최근성 감쇠 가중.

    Args:
        news_items: [{title, age_hours?|published_ts?, source?, url?}, ...]

    Returns:
        {sentiment_score(0~100), overall, sentiment_source, sentiment_decay_applied,
         fallback_used, positive_count, negative_count, neutral_count, reasons, items}
    """
    if not news_items:
        return {"sentiment_score": 50, "overall": "neutral",
                "sentiment_source": "none", "sentiment_decay_applied": False,
                "fallback_used": False, "positive_count": 0, "negative_count": 0,
                "neutral_count": 0, "reasons": ["No recent news available"], "items": []}

    items = news_items[:8]
    titles = [str(n.get("title") or "") for n in items]

    finbert_raw = _finbert(titles)
    source = "finbert" if finbert_raw else "keyword"
    fallback_used = finbert_raw is None

    weighted_sum = 0.0
    total_w = 0.0
    pos = neg = 0
    analyzed = []
    decay_applied = False

    for i, item in enumerate(items):
        if source == "finbert":
            sent = _parse_finbert(finbert_raw[i]) if i < len(finbert_raw) else _keyword_sentiment(titles[i])
        else:
            sent = _keyword_sentiment(titles[i])

        age_h = _news_age_hours(item)
        w = _recency_weight(age_h)
        if age_h is not None and age_h > 1:
            decay_applied = True
        weighted_sum += sent["score"] * w
        total_w += w
        if sent["label"] == "positive":
            pos += 1
        elif sent["label"] == "negative":
            neg += 1
        analyzed.append({"title": titles[i], "label": sent["label"],
                         "score": round(sent["score"], 3),
                         "recency_weight": round(w, 2),
                         "source": item.get("source")})

    avg = weighted_sum / total_w if total_w > 0 else 0.0
    score100 = int(round((avg + 1) * 50))    # -1~1 → 0~100
    overall = "positive" if avg > 0.2 else "negative" if avg < -0.2 else "neutral"
    eng = "FinBERT" if source == "finbert" else "keyword"
    if overall == "positive":
        reasons = [f"{pos}/{len(items)} headlines bullish, recency-weighted ({eng})"]
    elif overall == "negative":
        reasons = [f"{neg}/{len(items)} headlines bearish, recency-weighted ({eng})"]
    else:
        reasons = [f"Mixed sentiment: {pos}+ {neg}- (recency-weighted {eng})"]

    return {"sentiment_score": score100, "overall": overall,
            "sentiment_source": source, "sentiment_decay_applied": decay_applied,
            "fallback_used": fallback_used, "positive_count": pos, "negative_count": neg,
            "neutral_count": len(items) - pos - neg, "reasons": reasons, "items": analyzed}


def _news_age_hours(item: Dict) -> Optional[float]:
    """뉴스 항목에서 경과 시간(시간) 추출. 다양한 키/형식 지원."""
    if "age_hours" in item and item["age_hours"] is not None:
        return _num(item["age_hours"])
    ts = item.get("published_ts") or item.get("timestamp") or item.get("ts")
    if ts:
        try:
            return max(0.0, (time.time() - float(ts)) / 3600.0)
        except (TypeError, ValueError):
            pass
    # RSS/ISO 날짜 문자열 (예: "Mon, 02 Jun 2026 09:00:00 GMT")
    date_str = item.get("published") or item.get("date") or item.get("pubDate")
    if date_str and isinstance(date_str, str):
        try:
            from email.utils import parsedate_to_datetime
            dtv = parsedate_to_datetime(date_str)
            if dtv is not None:
                epoch = dtv.timestamp()
                return max(0.0, (time.time() - epoch) / 3600.0)
        except Exception:
            pass
    return None


# ════════════════════════════════════════════════════════════════════════════
# 4. 신뢰 구간 계산
# ════════════════════════════════════════════════════════════════════════════
def confidence_interval(confidence: float, source_scores: List[float],
                        macro_regime: str = "Neutral",
                        days_to_earnings: Optional[int] = None) -> Dict[str, Any]:
    """confidence 점추정치 → 신뢰 구간(lower/upper/spread/reason).

    spread는 소스 점수 분산 + 거시 불확실성 + 실적 근접성을 합산.
    """
    vals = [_num(s) for s in source_scores if s is not None]
    reasons: List[str] = []

    # (a) 소스 분산 — 표준편차를 spread로 환산
    spread = 0.0
    if len(vals) >= 2:
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        disp_spread = _clamp(std * 1.2, 0, 30)
        spread += disp_spread
        if disp_spread >= 10:
            reasons.append("High source dispersion")

    # (b) 거시 체제 불확실성
    macro_add = {"Transition": 8, "Risk-Off": 6, "Risk-On": 2, "Neutral": 2}.get(macro_regime, 4)
    spread += macro_add
    if macro_regime in ("Transition", "Risk-Off"):
        reasons.append("Macro regime uncertainty")

    # (c) 실적 근접성
    if days_to_earnings is not None and 0 <= days_to_earnings <= 5:
        spread += 6
        reasons.append("Upcoming earnings")

    spread = round(_clamp(spread, 4, 45), 0)
    half = spread / 2.0
    lower = int(round(_clamp(confidence - half, 0, 100)))
    upper = int(round(_clamp(confidence + half, 0, 100)))
    if not reasons:
        reasons.append("Sources broadly aligned")
    return {"lower": lower, "upper": upper, "spread": int(spread), "reason": reasons}


# ════════════════════════════════════════════════════════════════════════════
# 오케스트레이터 — confidence 일원화 진입점
# ════════════════════════════════════════════════════════════════════════════
# 소스 가중치 (합 1.0). AI 점수가 없으면 기술점수로 대체.
_W = {"ai": 0.30, "technical": 0.35, "sentiment": 0.15, "market": 0.20}


def _signal_from_score(weighted: float) -> str:
    if weighted >= 60:
        return "BUY"
    if weighted <= 40:
        return "SELL"
    return "NEUTRAL"


def build_signal_confidence(
    *,
    technical_score: float,
    ai_score: Optional[float] = None,
    sentiment_score: Optional[float] = None,
    market_score: Optional[float] = None,
    symbol: Optional[str] = None,
    market: str = "US",
    signal: Optional[str] = None,
    stock_pct5d: Optional[float] = None,
    news_items: Optional[List[Dict]] = None,
    include_macro: bool = True,
    include_sector: bool = True,
    include_earnings: bool = True,
) -> Dict[str, Any]:
    """신호 신뢰도 종합 — 6개 보정 요소를 한 곳에서 적용해 단일 confidence 반환.

    모든 외부 요소는 실패 시 안전하게 건너뛴다(부분 결과 허용).

    Returns: signal_confidence dict (응답에 그대로 삽입 가능)
    """
    tech = _num(technical_score, 50)
    ai = _num(ai_score, tech)            # AI 없으면 기술점수로 대체
    ai_available = ai_score is not None

    # 뉴스 감정: 외부 제공(sentiment_score) 우선, news_items 있으면 분석
    sentiment_block = None
    if sentiment_score is None and news_items:
        sentiment_block = analyze_news_sentiment(news_items)
        sentiment = _num(sentiment_block.get("sentiment_score"), 50)
    else:
        sentiment = _num(sentiment_score, 50)

    market_sc = _num(market_score, 50)

    # ── 1) 가중 종합 점수 → 기본 confidence ──────────────────────────────
    weighted = (ai * _W["ai"] + tech * _W["technical"]
                + sentiment * _W["sentiment"] + market_sc * _W["market"])
    deviation = abs(weighted - 50) / 50.0
    confidence = 50 + deviation * 38           # 50 ~ 88
    if signal is None:
        signal = _signal_from_score(weighted)

    caps: List[int] = []
    cap_reasons: List[str] = []

    # ── 2) 불일치 페널티 ──────────────────────────────────────────────────
    src_scores = [tech, sentiment, market_sc] + ([ai] if ai_available else [])
    disagreement = disagreement_penalty(src_scores)
    if disagreement["disagreement_penalty"]:
        caps.append(disagreement["confidence_cap"])
        cap_reasons.append(disagreement["reason"])

    # ── 3) 거시 체제 ──────────────────────────────────────────────────────
    macro = {"regime": "Neutral", "confidence_weight": 0}
    if include_macro:
        macro = get_macro_regime()
        confidence += _num(macro.get("confidence_weight"), 0)

    # ── 4) 섹터 상대 점수 ────────────────────────────────────────────────
    sector_rel = {"sector_relative_score": 0, "adjust": 0, "sector": None}
    if include_sector and symbol:
        sector_rel = get_sector_relative(symbol, signal, stock_pct5d)
        confidence += _num(sector_rel.get("adjust"), 0)

    # ── 5) 실적 임박 페널티 ──────────────────────────────────────────────
    earnings = {"days_to_earnings": None, "earnings_risk": False}
    if include_earnings and symbol:
        earnings = get_earnings_proximity(symbol)
        ecap = earnings_cap(earnings.get("days_to_earnings"))
        if ecap["cap"] < 100:
            caps.append(ecap["cap"])
            cap_reasons.append(ecap["reason"])
            earnings["confidence_cap_reason"] = ecap["reason"]

    # ── 캡 적용 + 클램프 ─────────────────────────────────────────────────
    confidence = _clamp(confidence, 5, 95)
    if caps:
        confidence = min(confidence, min(caps))
    confidence = int(round(_clamp(confidence, 5, 95)))

    # ── 6) 신뢰 구간 ─────────────────────────────────────────────────────
    interval = confidence_interval(
        confidence, src_scores,
        macro_regime=macro.get("regime", "Neutral"),
        days_to_earnings=earnings.get("days_to_earnings"),
    )

    return {
        "confidence": confidence,
        "signal": signal,
        "confidence_interval": interval,
        "weighted_score": round(weighted, 1),
        "source_scores": {"ai": round(ai, 1) if ai_available else None,
                          "technical": round(tech, 1),
                          "sentiment": round(sentiment, 1),
                          "market": round(market_sc, 1)},
        "macro_regime": macro,
        "sector_relative": sector_rel,
        "earnings": earnings,
        "earnings_risk": bool(earnings.get("earnings_risk")),
        "days_to_earnings": earnings.get("days_to_earnings"),
        "disagreement": disagreement,
        "sentiment": sentiment_block,        # news_items 분석 시에만 채워짐
        "cap_reasons": [r for r in cap_reasons if r],
        "weights": _W,
    }
