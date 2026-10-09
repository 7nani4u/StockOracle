"""hybrid_signals(상세 화면)와 dual_score_v2(스캔) 이중 구현의 공유 부분 일치 테스트.

같은 BQS/FWS/NCS 개념이 두 모듈에 따로 구현돼 있다. 하나만 고치면 같은 종목의 상세 화면과 스캔 표가 어긋난다
(허스트가 한쪽만 가격 '수준'에 계산되던 결함이 그 예). 의도된 차이(추격 FWS 판정 방식, 주간 ADX, dual_aligned 기본값)를
뺀 나머지 하위 점수와 합성식이 같은 입력에서 같은 값을 내는지 고정한다.
"""

import random

import pytest

from market_briefing import dual_score_v2 as v2
from market_briefing import hybrid_signals as v1


def _regime_vol_cases():
    return [(r, v) for r in (v1.REGIME_BULLISH, v1.REGIME_BEARISH, v1.REGIME_SIDEWAYS)
            for v in ("LOW_VOL", "NORMAL_VOL", "HIGH_VOL")]


def test_bqs_matches_between_detail_and_scan_implementations():
    rnd = random.Random(2026)
    for _ in range(400):
        regime, vol_regime = rnd.choice(_regime_vol_cases())
        aligned = rnd.choice([True, False])
        hurst = rnd.choice([None, rnd.uniform(0.2, 0.95)])
        args = dict(
            adx=rnd.uniform(5, 55), plus_di=rnd.uniform(5, 45), minus_di=rnd.uniform(5, 45),
            atr_percent=rnd.uniform(0.3, 9), dist_to_high=rnd.uniform(0, 8), rs_pct=rnd.uniform(-15, 30),
            vol_ratio=rnd.uniform(0.3, 3.0), bis_score=rnd.randint(0, 15),
        )
        detail = v1.compute_bqs(regime=regime, vol_regime=vol_regime, hurst=hurst, dual_aligned=aligned, **args)
        row = v2.SnapshotRow(
            adx_14=args["adx"], plus_di=args["plus_di"], minus_di=args["minus_di"], atr_pct=args["atr_percent"],
            distance_to_20d_high_pct=args["dist_to_high"], rs_vs_benchmark_pct=args["rs_pct"], vol_ratio=args["vol_ratio"],
            bis_score=float(args["bis_score"]), market_regime=regime, vol_regime=vol_regime,
            dual_regime_aligned=aligned, hurst_exponent=hurst or 0.0, weekly_adx=0.0,
        )
        assert v2.compute_bqs(row)["BQS"] == pytest.approx(detail, abs=0.011)


def test_fws_matches_when_extension_flags_are_mapped_to_ext_atr_bands():
    rnd = random.Random(7)
    for _ in range(300):
        chasing = rnd.choice([(False, False), (True, False), (False, True), (True, True)])
        ext_atr = {0: -0.3, 1: 0.5, 2: 1.0}[sum(chasing)]            # 0개 / 1개 / 2개 → 0 / 15 / 25점
        stable = rnd.choice([True, False])
        spiking, collapsing = rnd.choice([(False, False), (True, False), (False, True)])
        vol_ratio, adx = rnd.uniform(0.3, 2.5), rnd.uniform(5, 55)
        detail = v1.compute_fws(vol_ratio=vol_ratio, ext_atr=ext_atr, adx=adx, atr_spiking=spiking,
                                atr_collapsing=collapsing, regime_stable=stable)
        row = v2.SnapshotRow(vol_ratio=vol_ratio, adx_14=adx, atr_spiking=spiking, atr_collapsing=collapsing,
                             market_regime_stable=stable, chasing_20_last5=chasing[0], chasing_55_last5=chasing[1])
        assert v2.compute_fws(row)["FWS"] == pytest.approx(detail, abs=0.011)


def test_ncs_and_action_rules_match():
    rnd = random.Random(11)
    for _ in range(300):
        bqs, fws = rnd.uniform(0, 100), rnd.uniform(0, 100)
        earnings = rnd.choice([0.0, 10.0, 15.0, 20.0])
        detail = v1.compute_ncs(bqs, fws, earnings_penalty=earnings)
        scan = v2.compute_ncs(bqs, fws, {"EarningsPenalty": earnings, "ClusterPenalty": 0.0, "SuperClusterPenalty": 0.0})["NCS"]
        assert scan == pytest.approx(detail, abs=0.011)
        label = v1.ncs_action(detail, fws)
        note = v2.action_note(fws, scan, earnings)
        assert {"AUTO_NO": "Auto-No", "AUTO_YES": "Auto-Yes", "CONDITIONAL": "Conditional"}[label] in note


def test_bis_and_hurst_implementations_agree():
    rnd = random.Random(3)
    for _ in range(200):
        o = rnd.uniform(90, 110)
        c = o + rnd.uniform(-5, 5)
        high, low = max(o, c) + rnd.uniform(0, 3), min(o, c) - rnd.uniform(0, 3)
        volume, avg = rnd.uniform(1e5, 3e6), rnd.uniform(1e5, 2e6)
        assert v1.compute_bis(o, high, low, c, volume, avg) == v2.compute_bis_from_candle(o, high, low, c, volume, avg)
    prices = [100 * (1 + 0.01 * ((i * 7919) % 13 - 6) / 6) ** 1 for i in range(120)]
    assert v1.calc_hurst(prices) == pytest.approx(v2.calc_hurst_v2(prices), abs=1e-4)
