"""네이버 모바일 API 파서 테스트 — 실제 응답 구조를 축약한 고정 입력 사용(네트워크 없음)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from market_briefing import naver_mobile as nm


INTEGRATION = {
    "stockName": "삼성전자",
    "industryCode": "278",
    "totalInfos": [
        {"code": "lastClosePrice", "value": "269,000"},
        {"code": "marketValue", "value": "1,517조 1,093억"},
        {"code": "per", "value": "11.64배"},
        {"code": "pbr", "value": "3.02배"},
        {"code": "eps", "value": "22,292원"},
        {"code": "cnsPer", "value": "5.38배"},
        {"code": "dividendYieldRatio", "value": "0.64%"},
        {"code": "highPriceOf52Weeks", "value": "380,000"},
    ],
    "industryCompareInfo": [
        {"itemCode": "000660", "stockName": "SK하이닉스", "closePrice": "1,812,000",
         "fluctuationsRatio": "-2.21", "marketValue": "1,323,652,165",
         "stockExchangeType": {"code": "KS"}},
        {"itemCode": "036930", "stockName": "주성엔지니어링", "closePrice": "30,000",
         "fluctuationsRatio": "1.5", "marketValue": "9,714,554",
         "stockExchangeType": {"code": "KQ"}},
    ],
    "consensusInfo": {"priceTargetMean": "488,409", "recommMean": "4.05", "createDate": "2026-09-10"},
}

ANNUAL = {"financeInfo": {
    "trTitleList": [
        {"isConsensus": "N", "key": "202412"},
        {"isConsensus": "N", "key": "202512"},
        {"isConsensus": "Y", "key": "202612"},
    ],
    "rowList": [
        {"title": "매출액", "columns": {"202412": {"value": "3,008,709"}, "202512": {"value": "3,336,059"},
                                      "202612": {"value": "7,396,375"}}},
        {"title": "당기순이익", "columns": {"202412": {"value": "-1,000"}, "202512": {"value": "452,068"},
                                        "202612": {"value": "3,258,598"}}},
        {"title": "ROE", "columns": {"202412": {"value": "9.03"}, "202512": {"value": "10.85"},
                                   "202612": {"value": "40.1"}}},
        {"title": "부채비율", "columns": {"202512": {"value": "29.94"}, "202612": {"value": "-"}}},
    ],
}}


def test_integration_parses_valuation_market_cap_and_industry_peers():
    result = nm.parse_integration(INTEGRATION)
    assert result["per"] == 11.64 and result["pbr"] == 3.02
    assert result["market_cap_raw"] == 1_517_109_300_000_000
    assert result["industry_code"] == "278"
    assert [p["ticker"] for p in result["industry_peers"]] == ["000660.KS", "036930.KQ"]
    assert result["industry_peers"][0]["market_cap"] == 1_323_652_165 * 1e6
    assert result["consensus_target_price"] == 488409


def test_annual_finance_uses_actuals_only_and_labels_turnaround():
    result = nm.parse_annual_finance(ANNUAL)
    assert result["latest_actual_period"] == "202512"
    assert result["roe"] == 10.85            # 2026(E) 추정치 40.1 을 쓰지 않는다
    assert result["debt"] == 29.94
    assert result["revenue_growth"] == round((3336059 - 3008709) / 3008709 * 100, 1)
    assert result["net_profit_growth"] is None
    assert result["net_profit_growth_label"] == "흑자전환"


def test_amount_and_number_parsers_reject_placeholders():
    assert nm.parse_korean_amount("905억") == 905e8
    assert nm.parse_korean_amount("3,336,059") is None
    assert nm.to_number("-") is None
    assert nm.to_number("+3,130") == 3130
