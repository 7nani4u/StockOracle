"""뉴스 탭 상세 카드·한글화·외부 데이터 안전 처리 회귀 테스트."""

from pathlib import Path

from api.index import HTML, _enrich_naver_news_item
from market_briefing import us_enricher


SOURCE = Path(__file__).parents[1].joinpath("api", "index.py").read_text(encoding="utf-8")


class _FakeResponse:
    def __init__(self, *, text="", json_data=None):
        self.text = text
        self._json_data = json_data

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_data


def test_naver_article_metadata_preserves_full_title_summary_and_thumbnail(monkeypatch):
    page = """
    <html><head>
      <meta property="og:title" content="잘리지 않은 한국 주식 뉴스 전체 제목">
      <meta property="og:description" content="투자자가 이해할 수 있도록 정리된 기사 요약입니다.">
      <meta property="og:image" content="https://img.example.com/news.jpg">
    </head></html>
    """
    monkeypatch.setattr("api.index.requests.get", lambda *args, **kwargs: _FakeResponse(text=page))

    result = _enrich_naver_news_item({
        "title": "잘린 제목...",
        "link": "https://n.news.naver.com/mnews/article/001/1234567",
        "source": "테스트경제",
    })

    assert result["title"] == "잘리지 않은 한국 주식 뉴스 전체 제목"
    assert result["summary"].startswith("투자자가 이해할 수 있도록")
    assert result["image_url"] == "https://img.example.com/news.jpg"
    assert result["source"] == "테스트경제"


def test_finnhub_news_keeps_image_iso_time_and_original_title(monkeypatch):
    monkeypatch.setattr(us_enricher, "_get", lambda *_args, **_kwargs: [{
        "headline": "Apple shares rise after earnings",
        "url": "https://example.com/apple",
        "source": "Example Wire",
        "datetime": 1_700_000_000,
        "summary": "Revenue and guidance exceeded expectations.",
        "image": "https://img.example.com/apple.jpg",
    }])
    monkeypatch.setattr(us_enricher, "_translate_news_batch_ko", lambda items: items)

    item = us_enricher.fetch_finnhub_news("AAPL")[0]

    assert item["original_title"] == "Apple shares rise after earnings"
    assert item["image_url"] == "https://img.example.com/apple.jpg"
    assert item["published_at"].endswith("Z")
    assert item["source_type"] == "finnhub"


def test_us_news_batch_translation_adds_korean_display_fields(monkeypatch):
    translated_payload = [[
        ["애플 주가가 실적 발표 후 상승 ", "", None],
        ["<<<SO_FIELD_break>>> ", "", None],
        ["매출과 가이던스가 시장 예상치를 웃돌았습니다.", "", None],
    ]]
    monkeypatch.setattr(
        us_enricher.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(json_data=translated_payload),
    )
    items = [{
        "title": "Apple shares rise after earnings",
        "summary": "Revenue and guidance exceeded expectations.",
    }]

    result = us_enricher._translate_news_batch_ko(items)

    assert result[0]["title_ko"] == "애플 주가가 실적 발표 후 상승"
    assert "시장 예상치" in result[0]["summary_ko"]
    assert result[0]["translation_source"] == "자동 번역"


def test_news_tab_uses_detailed_single_column_cards_and_korean_us_titles():
    news_html = HTML.split('<!-- 뉴스 탭 -->', 1)[1].split('<!-- 📋 KRX 전용 탭 -->', 1)[0]
    renderer = SOURCE.split("function _safeNewsUrl", 1)[1].split("// ── 차트", 1)[0]

    assert 'class="news-tab-stack"' in news_html
    assert 'role="list"' in news_html
    assert "two-col-grid" not in news_html
    assert "n.title_ko || n.title" in renderer
    assert "_newsHasHangul" in renderer
    assert "summary_ko || n.summary" in renderer
    assert 'rel="noopener noreferrer"' in renderer
    assert 'loading="lazy"' in renderer
    assert "_safeNewsUrl" in renderer
    assert "한국어로 제공 가능한 관련 뉴스를 찾지 못했습니다" in renderer


def test_news_card_css_has_thumbnail_summary_and_mobile_breakpoint():
    assert ".news-row{display:grid;grid-template-columns:minmax(0,1fr) 80px" in HTML
    assert ".news-summary{" in HTML
    assert ".news-thumb-shell{" in HTML
    assert ".news-thumb.is-error{display:none}" in HTML
    assert "@media(max-width:480px)" in HTML
    assert ".news-row{grid-template-columns:minmax(0,1fr) 64px" in HTML
