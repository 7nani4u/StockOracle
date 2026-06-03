"""
test_confidence_engine.py — confidence_engine 단위 검증 (네트워크 불필요)

외부 호출(yfinance/HuggingFace)은 모두 fallback 경로로 검증되며, 순수 계산
함수(disagreement_penalty / confidence_interval / keyword 감정 / 오케스트레이터)는
결정적으로 테스트한다.

실행:  python tests/test_confidence_engine.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_briefing import confidence_engine as ce

_fail = 0


def check(name, cond):
    global _fail
    print(("  ✓ " if cond else "  ✗ ") + name)
    if not cond:
        _fail += 1


def test_disagreement():
    print("[1] disagreement_penalty")
    check("정렬된 점수 → 페널티 없음", ce.disagreement_penalty([60, 62, 58, 61])["confidence_cap"] == 100)
    check("spread>25 → cap 75", ce.disagreement_penalty([40, 66])["confidence_cap"] == 75)
    check("spread>35 → cap 65", ce.disagreement_penalty([30, 66])["confidence_cap"] == 65)
    check("spread>50 → cap 55", ce.disagreement_penalty([20, 75])["confidence_cap"] == 55)
    r = ce.disagreement_penalty([20, 75])
    check("페널티 사유 명시", r["disagreement_penalty"] and r["reason"])


def test_interval():
    print("[2] confidence_interval")
    i = ce.confidence_interval(68, [60, 80, 55, 76], macro_regime="Transition", days_to_earnings=3)
    check("lower < confidence < upper", i["lower"] < 68 < i["upper"])
    check("spread 양수", i["spread"] > 0)
    check("Transition 사유 포함", "Macro regime uncertainty" in i["reason"])
    check("실적 사유 포함", "Upcoming earnings" in i["reason"])
    tight = ce.confidence_interval(70, [70, 71, 69], macro_regime="Neutral")
    check("정렬 시 spread 더 좁음", tight["spread"] < i["spread"])


def test_earnings_cap():
    print("[3] earnings_cap")
    check("None → 무제한", ce.earnings_cap(None)["cap"] == 100)
    check("과거 → 무제한", ce.earnings_cap(-2)["cap"] == 100)
    check("1일 → cap 60", ce.earnings_cap(1)["cap"] == 60)
    check("3일 → cap 70", ce.earnings_cap(3)["cap"] == 70)
    check("10일 → 무제한", ce.earnings_cap(10)["cap"] == 100)


def _disable_hf():
    """모든 HF 모델을 쿨다운 처리 → 키워드 경로 강제 (네트워크 차단)."""
    for url in (ce._HF_URL_EN, ce._HF_URL_KR):
        ce._CACHE[f"hf_cooldown|{url}"] = (True, 9e18)


def _enable_hf():
    for url in (ce._HF_URL_EN, ce._HF_URL_KR):
        ce._CACHE.pop(f"hf_cooldown|{url}", None)


def test_keyword_sentiment():
    print("[4] analyze_news_sentiment — 영어 키워드 fallback")
    _disable_hf()
    bull = ce.analyze_news_sentiment([
        {"title": "Stock surges to record high on strong profit beat", "age_hours": 1},
        {"title": "Analysts upgrade with bullish growth outlook", "age_hours": 5},
    ])
    check("호재 → positive", bull["overall"] == "positive")
    check("source=keyword", bull["sentiment_source"] == "keyword")
    check("lang=en", bull["sentiment_lang"] == "en")
    check("fallback_used=True", bull["fallback_used"] is True)
    check("점수>50", bull["sentiment_score"] > 50)
    bear = ce.analyze_news_sentiment([
        {"title": "Shares crash on fraud lawsuit and bankruptcy warning", "age_hours": 2},
    ])
    check("악재 → negative", bear["overall"] == "negative")
    check("decay 적용 표기", bear["sentiment_decay_applied"] is True)
    empty = ce.analyze_news_sentiment([])
    check("뉴스 없음 → 중립 50", empty["sentiment_score"] == 50 and empty["overall"] == "neutral")
    _enable_hf()


def test_korean_sentiment():
    print("[4-KR] analyze_news_sentiment — 한국어 substring 키워드")
    _disable_hf()
    # 교착어: '급등세'·'돌파했다'처럼 어미가 붙어도 substring으로 매칭돼야 함
    bull = ce.analyze_news_sentiment([
        {"title": "삼성전자 급등세 지속…신고가 돌파했다", "age_hours": 1},
        {"title": "SK하이닉스 호실적에 강세 전환, 목표가 상향", "age_hours": 4},
    ])
    check("한국어 감지 lang=ko", bull["sentiment_lang"] == "ko")
    check("호재 → positive", bull["overall"] == "positive")
    check("source=keyword(모델 차단 시)", bull["sentiment_source"] == "keyword")
    check("점수>50", bull["sentiment_score"] > 50)
    bear = ce.analyze_news_sentiment([
        {"title": "코스닥 급락…실적 쇼크에 약세 지속, 손실 우려 확대", "age_hours": 2},
        {"title": "OO전자 적자전환 충격, 목표가 하향", "age_hours": 6},
    ])
    check("악재 → negative", bear["overall"] == "negative")
    check("점수<50", bear["sentiment_score"] < 50)
    # 단어 경계 없는 한국어에서도 매칭되는지 직접 검증
    s = ce._keyword_sentiment("급등세 신고가 돌파")
    check("substring 매칭(급등/신고가/돌파)", s["label"] == "positive")
    _enable_hf()


def test_orchestrator():
    print("[5] build_signal_confidence (오케스트레이터, 외부호출 off)")
    # 정렬된 강세 신호
    r = ce.build_signal_confidence(
        technical_score=72, ai_score=70, sentiment_score=68, market_score=66,
        symbol="AAPL", market="US",
        include_macro=False, include_sector=False, include_earnings=False)
    check("confidence 범위 5~95", 5 <= r["confidence"] <= 95)
    check("신호 BUY", r["signal"] == "BUY")
    check("confidence_interval 존재", "confidence_interval" in r and r["confidence_interval"]["spread"] >= 0)
    check("source_scores 4종", set(r["source_scores"]) == {"ai", "technical", "sentiment", "market"})

    # 큰 불일치 → confidence 캡
    r2 = ce.build_signal_confidence(
        technical_score=85, ai_score=30, sentiment_score=80, market_score=35,
        symbol=None, market="US",
        include_macro=False, include_sector=False, include_earnings=False)
    check("불일치 → cap 적용", r2["disagreement"]["disagreement_penalty"] is True)
    check("confidence <= 불일치 cap", r2["confidence"] <= r2["disagreement"]["confidence_cap"])

    # AI 점수 없음 → 기술점수 대체, 비중단
    r3 = ce.build_signal_confidence(
        technical_score=58, ai_score=None, sentiment_score=None, market_score=None,
        symbol=None, market="KRX",
        include_macro=False, include_sector=False, include_earnings=False)
    check("AI 없음에도 정상 반환", isinstance(r3["confidence"], int))
    check("AI source=None", r3["source_scores"]["ai"] is None)


def test_resilience():
    print("[6] 외부호출 fallback (네트워크 없어도 비중단)")
    macro = ce.get_macro_regime()
    check("macro regime 태그 유효", macro["regime"] in ("Risk-On", "Risk-Off", "Neutral", "Transition"))
    sec = ce.get_sector_relative("UNKNOWN_TICKER_XYZ", "BUY")
    check("미등록 종목 → 무보정", sec["adjust"] == 0 and sec["sector"] is None)
    er = ce.get_earnings_proximity("")
    check("빈 심볼 → 안전 기본값", er["days_to_earnings"] is None and er["earnings_risk"] is False)


if __name__ == "__main__":
    print("=" * 60)
    print("  confidence_engine 검증")
    print("=" * 60)
    for t in (test_disagreement, test_interval, test_earnings_cap,
              test_keyword_sentiment, test_korean_sentiment, test_orchestrator, test_resilience):
        t()
    print("=" * 60)
    print("  결과:", "✅ 전체 통과" if _fail == 0 else f"❌ {_fail}건 실패")
    print("=" * 60)
    sys.exit(1 if _fail else 0)
