"""공유 레이블 맵 — enum 값 → 사람이 읽을 수 있는 한국어 텍스트.

출처: k-ant-daily/scripts/labels.py 에서 이식 + StockOracle 용도에 맞게 확장.
"""
from __future__ import annotations

IMPACT_LABEL = {
    "positive": "호재",
    "negative": "악재",
    "neutral":  "중립",
}

# forward-looking: "오늘 이 종목이 오를까?"
RECOMMENDATION_LABEL = {
    "strong_buy":  "강한 상승 기대",
    "buy":         "상승 기대",
    "hold":        "관망",
    "sell":        "하락 경계",
    "strong_sell": "강한 하락 경계",
}

OUTCOME_LABEL = {
    "hit":     "적중",
    "partial": "부분",
    "miss":    "실패",
    "n/a":     "데이터 없음",
}

SENTIMENT_LABEL  = {"positive": "긍정", "neutral": "중립", "negative": "부정"}
OVERNIGHT_LABEL  = {"up": "강세", "neutral": "중립", "down": "약세"}
CONFIDENCE_LABEL = {"high": "높음", "medium": "중간", "low": "낮음"}
MOOD_LABEL       = {"positive": "우호", "neutral": "혼조", "negative": "부담"}

CATEGORY_LABEL = {
    "policy":      "🏛️ 정책",
    "geopolitics": "🌏 국제",
    "macro":       "💱 거시",
    "sector":      "🏭 섹터",
    "market":      "📊 시장",
}

MOOD_AXES = [
    {"key": "policy",      "emoji": "🏛️", "name": "정책·규제"},
    {"key": "geopolitics", "emoji": "🌏", "name": "국제정세"},
    {"key": "overnight",   "emoji": "🌙", "name": "간밤 해외"},
    {"key": "fx_macro",    "emoji": "💱", "name": "환율·원자재"},
]

# 섹터명 → 이모지 폴백
SECTOR_EMOJI = {
    "반도체": "🔧", "바이오": "🧬", "건설": "🏗️", "EPC": "🏗️", "플랜트": "🏗️",
    "조선": "🚢", "방산": "🛡️", "자동차": "🚗", "배터리": "🔋",
    "에너지": "💡", "정유": "⛽", "전력": "⚡", "통신": "📡",
    "엔터": "🎤", "게임": "🎮", "철강": "🏭", "화학": "🧪",
    "금융": "🏦", "은행": "🏦", "증권": "📈", "유통": "🛒",
    "식음료": "🍽️", "제약": "💊", "항공": "✈️",
    "플랫폼": "💻", "인터넷": "🌐", "전기전자": "🔌",
}

DIRECTION_META = {
    "strong_buy":  {"arrow": "↑↑", "label": "강한 상승 기대",  "cls": "up-strong"},
    "buy":         {"arrow": "↑",  "label": "상승 기대",       "cls": "up"},
    "hold":        {"arrow": "—",  "label": "관망",            "cls": "flat"},
    "sell":        {"arrow": "↓",  "label": "하락 경계",       "cls": "down"},
    "strong_sell": {"arrow": "↓↓", "label": "강한 하락 경계",  "cls": "down-strong"},
}
