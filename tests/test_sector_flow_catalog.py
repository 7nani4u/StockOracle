"""메인 업종별 흐름의 24개 업종 구성과 8×3 레이아웃 회귀 테스트."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from api import index
from market_briefing.labels import SECTOR_EMOJI
from market_briefing.sector_flow import build_sector_flow


EXPECTED_SECTORS = [
    "제조", "금융", "화학", "제약", "유통", "운송장비·부품", "음식료·담배", "금속",
    "섬유·의류", "일반서비스", "종이·목재", "비금속", "부동산", "운송·창고",
    "전기·전자", "보험", "IT 서비스", "건설", "오락·문화", "기계·장비",
    "전기·가스", "통신", "증권", "의료·정밀기기",
]


def test_sector_catalog_matches_requested_24_industries_without_duplicates():
    configured = [item["sector"] for item in index._SECTOR_DEFAULT_STOCKS]
    ordered_unique = list(dict.fromkeys(configured))

    assert ordered_unique == EXPECTED_SECTORS
    assert len(ordered_unique) == 24
    assert Counter(configured) == Counter({sector: 2 for sector in EXPECTED_SECTORS})
    assert len({item["code"] for item in index._SECTOR_DEFAULT_STOCKS}) == 48


def test_all_requested_industries_have_an_explicit_emoji_mapping():
    assert all(sector in SECTOR_EMOJI for sector in EXPECTED_SECTORS)
    result = build_sector_flow([
        {"code": "TEST", "name": "테스트", "sector": "비금속", "quote": {}}
    ])
    assert result["sectors"][0]["emoji"] == "🧱"


def test_desktop_sector_cards_use_eight_columns_for_three_rows():
    css = index.HTML.split("/* 8열 × 3행", 1)[1].split("/* 카드 기본 스타일 */", 1)[0]

    assert "grid-template-columns:repeat(8,minmax(0,1fr))" in css
    assert "gap:7px" in css
    assert "repeat(7" not in css
    assert len(EXPECTED_SECTORS) // 8 == 3


def test_sector_cards_keep_existing_tablet_and_mobile_breakpoints():
    responsive = index.HTML.split("/* ── 업종별 흐름 반응형", 1)[1].split(
        "/* ── 단계별 리포트", 1
    )[0]

    assert "@media(max-width:1100px)" in responsive
    assert "repeat(4,minmax(0,1fr))" in responsive
    assert "@media(max-width:768px)" in responsive
    assert "repeat(3,minmax(0,1fr))" in responsive
    assert "@media(max-width:480px)" in responsive
    assert "repeat(2,minmax(0,1fr))" in responsive
