"""
cross_reference.py — 스캔 결과 × 분석 결과 교차 참조 엔진

역할:
  - scan_engine.ScanCandidate (7단계 스캔 결과)와
    stock_analyzer.analyze_stock() 결과를 교차 참조하여
    최종 통합 점수를 생성
  - 뉴스 감성 + 3-신호 매트릭스 + NCS/BQS/FWS를 하나의 final_score로 통합
  - 불일치 신호(스캔 READY인데 뉴스 매도 등)를 플래그

공개 API:
  CrossReferenceResult  — 교차 참조 결과
  CrossReferenceEngine  — 핵심 엔진
    merge()             — ScanCandidate + analyze_stock 결과 통합
    merge_batch()       — 배치 처리
    final_score_formula()— 최종 점수 공식 설명 반환
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── 가중치 ────────────────────────────────────────────────────────────────────

# final_score = model_score * W_MODEL
#             + ncs_score   * W_NCS
#             + bqs_score   * W_BQS
#             + news_score  * W_NEWS
#             - market_risk_penalty
#             - concentration_penalty

W_MODEL = 0.40   # 기존 3-신호 매트릭스 점수 (0~100)
W_NCS   = 0.30   # NCS (0~100)
W_BQS   = 0.15   # BQS (0~100)
W_NEWS  = 0.15   # 뉴스 감성 점수 (0~100)

# 충돌 페널티
SIGNAL_CONFLICT_PENALTY = 10.0   # 스캔 READY인데 뉴스 negative
REGIME_BEARISH_PENALTY  = 15.0   # 레짐 BEARISH 시 신규 매수 억제
LOW_CONFIDENCE_PENALTY  = 8.0    # 신뢰도 low 시


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class CrossReferenceResult:
    """교차 참조 최종 결과."""
    ticker:         str
    name:           str

    # 점수 컴포넌트
    model_score:    float        # analyze_score() 결과 (0~100)
    ncs_score:      float        # NCS (0~100)
    bqs_score:      float        # BQS (0~100)
    fws_score:      float        # FWS (0~100, 높을수록 위험)
    news_score:     float        # 뉴스 감성 → 점수화 (0~100)

    # 최종 통합 점수
    final_score:    float        # 0~100
    final_label:    str          # STRONG_BUY / BUY / HOLD / SELL / AVOID
    confidence:     str          # high / medium / low

    # 신호 상태
    scan_status:    str          # READY / WATCH / FAR / WAIT_PULLBACK / ...
    recommendation: str          # analyze_stock() 추천
    regime:         str          # BULLISH / BEARISH / SIDEWAYS
    action_note:    str          # NCS action note

    # 페널티
    penalties:      Dict[str, float] = field(default_factory=dict)
    conflicts:      List[str]        = field(default_factory=list)
    warnings:       List[str]        = field(default_factory=list)
    reasons:        List[str]        = field(default_factory=list)

    # 포지션 사이징 (스캔 결과에서)
    entry_trigger:  Optional[float] = None
    stop_price:     Optional[float] = None
    shares:         Optional[float] = None
    risk_amount:    Optional[float] = None
    risk_pct:       Optional[float] = None

    generated_at:   str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker":         self.ticker,
            "name":           self.name,
            "final_score":    round(self.final_score, 1),
            "final_label":    self.final_label,
            "confidence":     self.confidence,
            "scan_status":    self.scan_status,
            "recommendation": self.recommendation,
            "regime":         self.regime,
            "action_note":    self.action_note,
            "scores": {
                "model": round(self.model_score, 1),
                "ncs":   round(self.ncs_score, 1),
                "bqs":   round(self.bqs_score, 1),
                "fws":   round(self.fws_score, 1),
                "news":  round(self.news_score, 1),
            },
            "penalties":     self.penalties,
            "conflicts":     self.conflicts,
            "warnings":      self.warnings,
            "reasons":       self.reasons,
            "sizing": {
                "entry_trigger": self.entry_trigger,
                "stop_price":    self.stop_price,
                "shares":        self.shares,
                "risk_amount":   self.risk_amount,
                "risk_pct":      self.risk_pct,
            },
            "generated_at":  self.generated_at,
        }


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _news_sentiment_to_score(sentiment: str) -> float:
    """뉴스 감성 → 0~100 점수 변환."""
    return {"positive": 75.0, "neutral": 50.0, "negative": 25.0}.get(sentiment, 50.0)


def _recommendation_to_score(rec: str) -> float:
    """3-신호 추천 → 0~100 점수 변환."""
    return {
        "strong_buy":  90.0,
        "buy":         70.0,
        "hold":        50.0,
        "sell":        30.0,
        "strong_sell": 10.0,
    }.get(rec, 50.0)


def _label_from_score(score: float) -> str:
    if score >= 75:
        return "STRONG_BUY"
    if score >= 60:
        return "BUY"
    if score >= 40:
        return "HOLD"
    if score >= 25:
        return "SELL"
    return "AVOID"


def _confidence_from_components(
    ncs: float, fws: float, scan_status: str, conflicts: List[str]
) -> str:
    if conflicts:
        return "low"
    if ncs >= 70 and fws <= 30 and scan_status == "READY":
        return "high"
    if ncs >= 50 and fws <= 50:
        return "medium"
    return "low"


# ── 교차 참조 엔진 ────────────────────────────────────────────────────────────

class CrossReferenceEngine:
    """스캔 결과 × 분석 결과 교차 참조."""

    def merge(
        self,
        scan_candidate: Optional[Dict[str, Any]],   # ScanCandidate.to_dict() 또는 None
        analyzed_stock: Optional[Dict[str, Any]],   # analyze_stock() 반환값 또는 None
        model_score:    float = 50.0,                # analyze_score() 결과
    ) -> CrossReferenceResult:
        """단일 종목 교차 참조.

        scan_candidate와 analyzed_stock 중 하나만 있어도 부분 결과 반환.
        """
        sc   = scan_candidate or {}
        ast  = analyzed_stock or {}
        hs   = ast.get("hybrid_score") or {}

        ticker = sc.get("ticker") or ast.get("code", "UNKNOWN")
        name   = sc.get("name")   or ast.get("name", ticker)

        # ── 컴포넌트 점수 ────────────────────────────────────────────────────
        ncs_score  = float(hs.get("ncs",  sc.get("ncs",  50.0)) or 50.0)
        bqs_score  = float(hs.get("bqs",  sc.get("bqs",  50.0)) or 50.0)
        fws_score  = float(hs.get("fws",  sc.get("fws",  50.0)) or 50.0)

        news_sent  = ast.get("news_sentiment", "neutral")
        news_score = _news_sentiment_to_score(news_sent)

        rec        = ast.get("recommendation", "hold")
        rec_score  = _recommendation_to_score(rec)

        # model_score는 외부 주입 (analyze_score() 반환값)
        # 추천이 있으면 rec_score로 보정
        effective_model = model_score if model_score != 50.0 else rec_score

        # ── 기본 가중 합산 ─────────────────────────────────────────────────
        raw_score = (
            effective_model * W_MODEL
            + ncs_score     * W_NCS
            + bqs_score     * W_BQS
            + news_score    * W_NEWS
        )

        # ── 페널티 계산 ───────────────────────────────────────────────────
        penalties: Dict[str, float] = {}
        conflicts: List[str] = []
        warnings:  List[str] = []
        reasons:   List[str] = []

        regime = hs.get("regime") or sc.get("regime", "SIDEWAYS")
        if regime == "BEARISH" and rec in ("buy", "strong_buy"):
            penalties["regime_bearish"] = REGIME_BEARISH_PENALTY
            conflicts.append("레짐 BEARISH — 매수 추천 충돌")

        scan_status = sc.get("status", "UNKNOWN")
        if scan_status == "READY" and news_sent == "negative":
            penalties["signal_conflict_news"] = SIGNAL_CONFLICT_PENALTY
            conflicts.append("스캔 READY인데 뉴스 부정적")

        if scan_status in ("FAR", "EARNINGS_BLOCK"):
            penalties["scan_blocked"] = SIGNAL_CONFLICT_PENALTY
            warnings.append(f"스캔 상태 {scan_status} — 진입 불가")

        conf_label = ast.get("confidence", "medium")
        if conf_label == "low":
            penalties["low_confidence"] = LOW_CONFIDENCE_PENALTY
            warnings.append("신뢰도 낮음")

        action_note_raw = sc.get("action_note") or hs.get("action", "CONDITIONAL")
        if "Auto-No" in str(action_note_raw) or action_note_raw == "AUTO_NO":
            penalties["fws_high"] = 15.0
            conflicts.append(f"FWS 과도 — {action_note_raw}")

        total_penalty = min(sum(penalties.values()), 40.0)
        final_score   = max(0.0, min(100.0, raw_score - total_penalty))

        # ── 이유 수집 ─────────────────────────────────────────────────────
        if ncs_score >= 70 and fws_score <= 30:
            reasons.append(f"NCS {ncs_score:.0f} 우수 (FWS {fws_score:.0f})")
        if bqs_score >= 65:
            reasons.append(f"BQS {bqs_score:.0f} — 브레이크아웃 품질 양호")
        if scan_status == "READY":
            reasons.append("스캔 READY — 진입 트리거 도달")
        if news_sent == "positive":
            reasons.append("뉴스 감성 긍정적")

        action_note_str = str(sc.get("action_note", "") or hs.get("action", "CONDITIONAL"))

        final_label = _label_from_score(final_score)
        confidence  = _confidence_from_components(ncs_score, fws_score, scan_status, conflicts)

        return CrossReferenceResult(
            ticker         = ticker,
            name           = name,
            model_score    = effective_model,
            ncs_score      = ncs_score,
            bqs_score      = bqs_score,
            fws_score      = fws_score,
            news_score     = news_score,
            final_score    = round(final_score, 1),
            final_label    = final_label,
            confidence     = confidence,
            scan_status    = scan_status,
            recommendation = rec,
            regime         = regime,
            action_note    = action_note_str,
            penalties      = penalties,
            conflicts      = conflicts,
            warnings       = warnings,
            reasons        = reasons,
            entry_trigger  = sc.get("entry_trigger"),
            stop_price     = sc.get("stop_price"),
            shares         = sc.get("shares"),
            risk_amount    = sc.get("risk_amount"),
            risk_pct       = sc.get("risk_pct"),
        )

    def merge_batch(
        self,
        scan_candidates:  List[Dict[str, Any]],
        analyzed_map:     Dict[str, Dict[str, Any]],  # ticker → analyze_stock()
        model_score_map:  Dict[str, float] = None,    # ticker → score
    ) -> List[CrossReferenceResult]:
        """배치 교차 참조 + 최종 점수 기준 정렬.

        Args:
            scan_candidates: run_full_scan() 결과 후보 리스트
            analyzed_map:    ticker → analyze_stock() 결과 dict
            model_score_map: ticker → analyze_score() 점수 (없으면 50 사용)

        Returns:
            CrossReferenceResult 리스트 (final_score 내림차순)
        """
        if model_score_map is None:
            model_score_map = {}

        results: List[CrossReferenceResult] = []

        for sc in scan_candidates:
            ticker = sc.get("ticker", "")
            ast    = analyzed_map.get(ticker)
            score  = model_score_map.get(ticker, 50.0)
            result = self.merge(sc, ast, score)
            results.append(result)

        # analyzed_map에만 있는 종목 처리 (스캔 미통과)
        scanned_tickers = {sc.get("ticker","") for sc in scan_candidates}
        for ticker, ast in analyzed_map.items():
            if ticker not in scanned_tickers:
                score = model_score_map.get(ticker, 50.0)
                result = self.merge(None, ast, score)
                result.ticker = ticker
                results.append(result)

        results.sort(key=lambda r: r.final_score, reverse=True)
        return results

    @staticmethod
    def final_score_formula() -> Dict[str, Any]:
        """최종 점수 공식 및 가중치 설명."""
        return {
            "formula": (
                "final_score = model_score × 0.40"
                " + ncs_score × 0.30"
                " + bqs_score × 0.15"
                " + news_score × 0.15"
                " - penalties (최대 40점)"
            ),
            "weights": {
                "model_score": W_MODEL,
                "ncs_score":   W_NCS,
                "bqs_score":   W_BQS,
                "news_score":  W_NEWS,
            },
            "penalties": {
                "regime_bearish":      REGIME_BEARISH_PENALTY,
                "signal_conflict_news":SIGNAL_CONFLICT_PENALTY,
                "low_confidence":      LOW_CONFIDENCE_PENALTY,
                "fws_high":            15.0,
                "scan_blocked":        SIGNAL_CONFLICT_PENALTY,
                "cap":                 40.0,
            },
            "labels": {
                "75~100": "STRONG_BUY",
                "60~75":  "BUY",
                "40~60":  "HOLD",
                "25~40":  "SELL",
                "0~25":   "AVOID",
            },
        }
