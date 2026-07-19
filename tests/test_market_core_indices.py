from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from market_briefing import core_summary, data_fetcher


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "api" / "index.py").read_text(encoding="utf-8")


class _FakeKrxResponse:
    def __init__(self, output):
        self._output = output
        self.headers = {"Date": "Sat, 18 Jul 2026 08:00:00 GMT"}

    def raise_for_status(self):
        return None

    def json(self):
        return {"output": self._output}


def _official_output():
    # 공식 응답 배열 순서와 무관하게 idx_nm으로 매핑되어야 한다.
    return [
        {"idx_nm": "KOSDAQ", "clsprc_idx": "791.84", "cmpr_idx": "37.59", "cmpr_rt": "4.532028", "updown": "down"},
        {"idx_nm": "KOSPI 200", "clsprc_idx": "1080.36", "cmpr_idx": "83.54", "cmpr_rt": "7.177593", "updown": "down"},
        {"idx_nm": "KOSPI", "clsprc_idx": "6820.60", "cmpr_idx": "463.81", "cmpr_rt": "6.367159", "updown": "down"},
    ]


def test_krx_indices_are_name_mapped_signed_and_arithmetically_verified(monkeypatch):
    monkeypatch.setattr(data_fetcher.requests, "get", lambda *args, **kwargs: _FakeKrxResponse(_official_output()))

    result = data_fetcher.fetch_market_indices()

    assert result["KOSPI"]["value"] == "6,820.60"
    assert result["KOSPI"]["prev_close"] == 7284.41
    assert result["KOSPI"]["change_abs"] == "-463.81"
    assert result["KOSPI"]["change_pct"] == "-6.37%"
    assert result["KOSPI"]["direction"] == "down"
    assert result["KOSPI"]["source"] == "KRX Global"
    assert result["KOSPI200"]["source_symbol"] == "KOSPI 200"
    assert all(item["validation"] == "verified" for item in result.values())


def test_naver_fallback_uses_direction_marker_not_incorrect_blind_text(monkeypatch):
    html = """
    <span id="KOSPI_now">6,820.60</span><span id="KOSPI_change"><span class="ndown"></span>463.81 -6.37%<span class="blind">상승</span></span>
    <span id="KOSDAQ_now">791.84</span><span id="KOSDAQ_change"><span class="ndown"></span>37.59 -4.53%<span class="blind">상승</span></span>
    <span id="KPI200_now">1,080.36</span><span id="KPI200_change"><span class="ndown"></span>83.54 -7.18%<span class="blind">상승</span></span>
    """
    monkeypatch.setattr(data_fetcher, "_get", lambda *_args, **_kwargs: BeautifulSoup(html, "html.parser"))

    result = data_fetcher._fetch_market_indices_naver()

    assert result["KOSPI"]["direction"] == "down"
    assert result["KOSPI"]["change_pct"] == "-6.37%"
    assert result["KOSPI200"]["prev_close"] == 1163.9


def test_invalid_official_index_is_not_presented_as_current(monkeypatch):
    output = _official_output()
    output[0] = dict(output[0], cmpr_rt="99.0")
    monkeypatch.setattr(data_fetcher.requests, "get", lambda *args, **kwargs: _FakeKrxResponse(output))

    result = data_fetcher.fetch_market_indices()

    assert result["KOSDAQ"]["available"] is False
    assert result["KOSDAQ"]["value"] == ""
    assert result["KOSDAQ"]["validation"] == "unavailable"
    assert result["KOSPI"]["available"] is True


def test_unavailable_indices_do_not_bias_market_mood():
    indices = {
        "KOSPI": {"available": False, "direction": "down"},
        "KOSDAQ": {"available": True, "direction": "up"},
    }

    assert core_summary._derive_kr_mood(indices) == "positive"


def test_market_status_distinguishes_preopen_live_close_and_weekend():
    kst = data_fetcher.KST

    assert data_fetcher._kr_market_status(datetime(2026, 7, 18, 10, 0, tzinfo=kst)) == "휴장 · 최근 영업일 종가"
    assert data_fetcher._kr_market_status(datetime(2026, 7, 20, 8, 30, tzinfo=kst)) == "장 시작 전 · 직전 종가"
    assert data_fetcher._kr_market_status(datetime(2026, 7, 20, 10, 0, tzinfo=kst)) == "장중"
    assert data_fetcher._kr_market_status(datetime(2026, 7, 20, 16, 0, tzinfo=kst)) == "장 마감 · 종가"


def test_weekday_without_current_trading_bar_is_not_labeled_live():
    indices = {"KOSPI": {"market_status": "장중", "available": True}}
    overnight = [{"symbol": "^KS200", "as_of": "2026-07-17"}]

    data_fetcher._reconcile_index_market_status(
        indices,
        overnight,
        datetime(2026, 7, 20, 10, 0, tzinfo=data_fetcher.KST),
    )

    assert indices["KOSPI"]["trade_date"] == "2026-07-17"
    assert indices["KOSPI"]["market_status"] == "휴장 또는 데이터 미갱신 · 최근 영업일 종가"

    no_date = {"KOSPI": {"market_status": "장중", "available": True}}
    data_fetcher._reconcile_index_market_status(
        no_date,
        [],
        datetime(2026, 7, 20, 10, 0, tzinfo=data_fetcher.KST),
    )
    assert no_date["KOSPI"]["market_status"] == "시장 상태 확인 불가 · 최근 수신 지수"


def test_ai_sector_card_removed_and_market_detail_reuses_home_snapshot():
    ai_html = SOURCE.split('<!-- AI 탭 -->', 1)[1].split('<!-- 단계별 분석 리포트 탭 -->', 1)[0]
    drawer = SOURCE.split("function _buildImmuneHtml", 1)[1].split("</script>", 1)[0]
    renderer = SOURCE.split("function renderMarketCore", 1)[1].split("async function loadSectorFlow", 1)[0]

    assert "flow-sector-card" not in SOURCE
    assert "flow-sector-content" not in SOURCE
    assert "섹터 / 업종 정보" not in ai_html
    assert "주요 근거" in drawer
    assert "거시·자금 흐름" in drawer
    assert "긍정 요인" in drawer
    assert "부담 요인" in drawer
    assert "주의사항" in drawer
    assert "_marketCoreSnapshot || {}" in SOURCE
    assert "idx.change_abs" in renderer
    assert "데이터 미수신" in renderer


def test_fundamental_cards_use_one_desktop_row_and_keep_mobile_grid():
    assert ".fund-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}" in SOURCE
    assert ".fund-grid{grid-template-columns:repeat(2,1fr)}" in SOURCE


def test_mobile_volume_card_stretches_to_match_atr_card_height():
    mobile_css = SOURCE.split("@media(max-width:480px)", 1)[1].split(
        "@media(max-width:360px)", 1
    )[0]

    assert ".metrics-grid{grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;align-items:stretch}" in mobile_css
    assert ".metric-volume-card,.metric-atr-card{grid-column:span 2}" in mobile_css
