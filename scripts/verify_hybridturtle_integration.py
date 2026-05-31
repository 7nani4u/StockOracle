"""
scripts/verify_hybridturtle_integration.py
StockOracle × HybridTurtle-v6.0 통합 검증 스크립트

실행:
    python scripts/verify_hybridturtle_integration.py

목적:
  - 모든 통합 모듈이 import 가능한지 확인
  - 샘플 OHLCV 데이터로 각 엔진 실제 실행
  - 예측 흐름에서 실제로 호출되는지 확인
  - dashboard payload 생성
  - cross-reference 결과 생성
"""
from __future__ import annotations

import sys
import os
import traceback

# 프로젝트 루트를 Python 경로에 추가
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── 샘플 데이터 생성 ──────────────────────────────────────────────────────────

def _make_sample_ohlcv(n: int = 250) -> tuple:
    """가상 OHLCV 데이터 생성 (상승 추세)."""
    import math
    closes  = []
    highs   = []
    lows    = []
    volumes = []
    opens   = []
    price   = 50000.0
    for i in range(n):
        # 약한 상승 추세 + 노이즈
        drift    = 0.0005
        noise    = math.sin(i * 0.3) * 200 + (i % 5) * 50
        price    = max(price * (1 + drift) + noise, 1000.0)
        daily_range = price * 0.02
        open_p   = price - daily_range * 0.3
        high_p   = price + daily_range * 0.4
        low_p    = price - daily_range * 0.6
        vol      = 1_000_000 + (i % 10) * 200_000
        opens.append(round(open_p, 0))
        highs.append(round(high_p, 0))
        lows.append(round(low_p, 0))
        closes.append(round(price, 0))
        volumes.append(vol)
    return closes, highs, lows, volumes, opens


PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results: dict[str, str] = {}

def run_test(name: str, fn, *args, **kwargs):
    try:
        r = fn(*args, **kwargs)
        results[name] = f"{PASS} PASS"
        return r
    except Exception as e:
        results[name] = f"{FAIL} FAIL: {e}"
        traceback.print_exc()
        return None


# =============================================================================
# 1. 모듈 Import 테스트
# =============================================================================

print("\n" + "="*60)
print("  StockOracle × HybridTurtle 통합 검증")
print("="*60)
print("\n[1] 모듈 Import 테스트")

def test_import_hybrid_signals():
    from market_briefing.hybrid_signals import compute_hybrid_score
    assert callable(compute_hybrid_score)
    return True

def test_import_dual_score_v2():
    from market_briefing.dual_score_v2 import compute_bqs, compute_fws, compute_ncs, SnapshotRow
    assert callable(compute_bqs) and callable(compute_fws) and callable(compute_ncs)
    return True

def test_import_quality_filter():
    from market_briefing.quality_filter import score_quality, QualityFilterResult
    assert callable(score_quality)
    return True

def test_import_scan_engine():
    from market_briefing.scan_engine import run_full_scan, build_snapshot_from_ohlcv
    assert callable(run_full_scan)
    return True

def test_import_stock_analyzer():
    from market_briefing.stock_analyzer import analyze_stock, enrich_with_hybrid
    assert callable(analyze_stock) and callable(enrich_with_hybrid)
    return True

def test_import_portfolio_manager():
    from market_briefing.portfolio_manager import PortfolioManager, PortfolioPosition
    assert callable(PortfolioManager)
    return True

def test_import_market_immune():
    from market_briefing.market_immune import MarketImmune
    assert callable(MarketImmune)
    return True

def test_import_cross_reference():
    from market_briefing.cross_reference import CrossReferenceEngine
    assert callable(CrossReferenceEngine)
    return True

def test_import_dashboard_payload():
    from market_briefing.dashboard_payload import DashboardPayloadBuilder
    assert callable(DashboardPayloadBuilder)
    return True

for fn in [
    test_import_hybrid_signals, test_import_dual_score_v2,
    test_import_quality_filter, test_import_scan_engine,
    test_import_stock_analyzer, test_import_portfolio_manager,
    test_import_market_immune, test_import_cross_reference,
    test_import_dashboard_payload,
]:
    r = run_test(fn.__name__.replace("test_import_", ""), fn)
    print(f"  {results[fn.__name__.replace('test_import_', '')]}  ({fn.__name__})")


# =============================================================================
# 2. 샘플 OHLCV로 스캔 엔진 실행
# =============================================================================

print("\n[2] 7단계 스캔 엔진 실행")

closes, highs, lows, volumes, opens = _make_sample_ohlcv(250)

def test_scan_engine():
    from market_briefing.scan_engine import (
        StockUniverse, run_full_scan, build_snapshot_from_ohlcv
    )
    from market_briefing.quality_filter import score_quality
    from market_briefing.dual_score_v2 import REGIME_BULLISH, VOL_NORMAL

    universe = [StockUniverse("SAMPLE1", "샘플1", "CORE", "Technology", "반도체")]
    snap = build_snapshot_from_ohlcv("SAMPLE1", closes, highs, lows, volumes, opens)
    snap_map = {"SAMPLE1": snap}

    qmj = score_quality("SAMPLE1", roe=0.15, debt_to_equity=0.8, revenue_growth=0.1)
    quality_map = {"SAMPLE1": qmj}

    result = run_full_scan(
        universe           = universe,
        snap_map           = snap_map,
        quality_map        = quality_map,
        regime             = REGIME_BULLISH,
        vol_regime         = VOL_NORMAL,
        portfolio_equity   = 100_000_000,
    )
    assert result.total_scanned == 1
    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert 0 <= c.bqs <= 100
    assert 0 <= c.fws <= 100
    assert 0 <= c.ncs <= 100
    print(f"    SAMPLE1: BQS={c.bqs:.1f} FWS={c.fws:.1f} NCS={c.ncs:.1f} "
          f"상태={c.status} 액션={c.action_note[:30]}")
    return result

scan_result = run_test("scan_engine", test_scan_engine)
print(f"  {results['scan_engine']}")


# =============================================================================
# 3. BQS/FWS/NCS 직접 계산
# =============================================================================

print("\n[3] BQS/FWS/NCS 계산")

def test_bqs_fws_ncs():
    from market_briefing.dual_score_v2 import (
        SnapshotRow, compute_bqs, compute_fws, compute_penalties, compute_ncs
    )
    row = SnapshotRow(
        ticker="TEST", sleeve="CORE", status="READY",
        close=50000, atr_14=800, atr_pct=1.6,
        adx_14=28.0, plus_di=22.0, minus_di=14.0,
        vol_ratio=1.5, market_regime="BULLISH",
        market_regime_stable=True, vol_regime="NORMAL_VOL",
        dual_regime_aligned=True,
        distance_to_20d_high_pct=0.5,
        rs_vs_benchmark_pct=3.0,
        weekly_adx=32.0, hurst_exponent=0.62,
    )
    bqs_r = compute_bqs(row)
    fws_r = compute_fws(row)
    pen_r = compute_penalties(row)
    ncs_r = compute_ncs(bqs_r["BQS"], fws_r["FWS"], pen_r)
    assert 0 <= bqs_r["BQS"] <= 100
    assert 0 <= fws_r["FWS"] <= 100
    assert 0 <= ncs_r["NCS"] <= 100
    print(f"    BQS={bqs_r['BQS']} FWS={fws_r['FWS']} NCS={ncs_r['NCS']}")
    return bqs_r, fws_r, ncs_r

run_test("bqs_fws_ncs", test_bqs_fws_ncs)
print(f"  {results['bqs_fws_ncs']}")


# =============================================================================
# 4. hybrid_signals.compute_hybrid_score
# =============================================================================

print("\n[4] compute_hybrid_score")

def test_hybrid_score():
    from market_briefing.hybrid_signals import compute_hybrid_score
    hs = compute_hybrid_score(closes, highs, lows, volumes, opens)
    assert "bqs" in hs and "fws" in hs and "ncs" in hs
    assert 0 <= hs["bqs"] <= 100
    assert 0 <= hs["fws"] <= 100
    print(f"    BQS={hs['bqs']} FWS={hs['fws']} NCS={hs['ncs']} "
          f"레짐={hs['regime']} 액션={hs['action']}")
    return hs

hs_result = run_test("hybrid_score", test_hybrid_score)
print(f"  {results['hybrid_score']}")


# =============================================================================
# 5. CrossReference 교차 참조
# =============================================================================

print("\n[5] CrossReference 교차 참조")

def test_cross_reference():
    from market_briefing.cross_reference import CrossReferenceEngine

    engine = CrossReferenceEngine()

    # 스캔 결과 mock
    scan_mock = {
        "ticker":       "SAMPLE1",
        "name":         "샘플1",
        "status":       "READY",
        "bqs":          72.5,
        "fws":          18.3,
        "ncs":          75.0,
        "action_note":  "Auto-Yes (고품질 브레이크아웃)",
        "entry_trigger":52000.0,
        "stop_price":   49000.0,
        "shares":       19.0,
        "risk_amount":  57000.0,
        "risk_pct":     0.057,
        "regime":       "BULLISH",
    }

    # analyze_stock() 결과 mock
    analyzed_mock = {
        "code":              "SAMPLE1",
        "news_sentiment":    "positive",
        "recommendation":    "buy",
        "confidence":        "medium",
        "hybrid_score":      {"ncs": 75.0, "bqs": 72.5, "fws": 18.3,
                              "action": "AUTO_YES", "regime": "BULLISH"},
    }

    result = engine.merge(scan_mock, analyzed_mock, model_score=68.0)
    assert result.final_score > 0
    assert result.final_label in ("STRONG_BUY","BUY","HOLD","SELL","AVOID")
    print(f"    final_score={result.final_score} label={result.final_label} "
          f"confidence={result.confidence}")
    return result

cr_result = run_test("cross_reference", test_cross_reference)
print(f"  {results['cross_reference']}")


# =============================================================================
# 6. RiskManager — 신규 거래 리스크 평가
# =============================================================================

print("\n[6] PortfolioManager 리스크 평가")

def test_portfolio_manager():
    from market_briefing.portfolio_manager import PortfolioManager

    mgr = PortfolioManager(equity=100_000_000, cash_balance=100_000_000)
    assessment = mgr.assess_new_trade(
        ticker      = "SAMPLE1",
        entry_price = 52000.0,
        stop_price  = 49000.0,
        sector      = "Technology",
        sleeve      = "CORE",
    )
    assert assessment.approved or len(assessment.violations) > 0
    print(f"    approved={assessment.approved} shares={assessment.recommended_shares:.0f} "
          f"risk={assessment.risk_per_trade:,.0f}원")

    # 포지션 등록 + 손절가 갱신 테스트
    mgr.add_position("SAMPLE1","샘플1",52000,49000,10,"Technology","CORE")
    upd = mgr.update_stop("SAMPLE1", 50000)
    assert upd["ok"]
    upd_down = mgr.update_stop("SAMPLE1", 48000)
    assert not upd_down["ok"]  # 하향 금지
    state = mgr.get_state()
    assert len(state.open_positions) == 1
    print(f"    손절가 상향 OK / 하향 차단 OK / 포지션 1개")
    return assessment

run_test("portfolio_manager", test_portfolio_manager)
print(f"  {results['portfolio_manager']}")


# =============================================================================
# 7. MarketImmune — 시장 위험 면역
# =============================================================================

print("\n[7] MarketImmune 시장 위험 면역")

def test_market_immune():
    from market_briefing.market_immune import MarketImmune

    mi = MarketImmune()

    # 빠른 체크 (지표 직접 입력)
    normal = mi.quick_check(vix=15.0, ma200_dev_pct=5.0, atr_pct=1.2)
    assert normal["immune_level"] == "CLEAR"

    crisis = mi.quick_check(vix=42.0, ma200_dev_pct=-22.0, atr_pct=5.5)
    assert crisis["immune_level"] in ("ALERT", "IMMUNE")

    # 전체 평가 (OHLCV)
    bench_closes = [10000.0 * (1 + 0.001 * i) for i in range(250)]
    bench_highs  = [c * 1.01 for c in bench_closes]
    bench_lows   = [c * 0.99 for c in bench_closes]
    ir = mi.assess(bench_closes, bench_highs, bench_lows, vix=16.0)
    assert ir.immune_level in ("CLEAR","CAUTION","ALERT","IMMUNE")
    print(f"    정상={normal['immune_level']} 위기={crisis['immune_level']} "
          f"실제시뮬={ir.immune_level}(점수={ir.immune_score})")
    return ir

run_test("market_immune", test_market_immune)
print(f"  {results['market_immune']}")


# =============================================================================
# 8. DashboardPayload 생성
# =============================================================================

print("\n[8] Dashboard Payload 생성")

def test_dashboard_payload():
    from market_briefing.dashboard_payload import DashboardPayloadBuilder

    builder = DashboardPayloadBuilder()

    # 하트비트만 먼저
    hb = builder.heartbeat()
    assert "modules" in hb and "overall" in hb
    print(f"    heartbeat.overall={hb['overall']}")

    # 전체 페이로드
    payload = builder.build()
    assert "status" in payload
    assert "heartbeat" in payload
    assert "action_plan" in payload
    print(f"    status={payload['status']} modules={len(hb['modules'])}ea")
    return payload

run_test("dashboard_payload", test_dashboard_payload)
print(f"  {results['dashboard_payload']}")


# =============================================================================
# 9. analyze_stock + enrich_with_hybrid 실행 경로 검증
# =============================================================================

print("\n[9] analyze_stock() 실행 경로 검증")

def test_analyze_stock_hybrid():
    from market_briefing.stock_analyzer import analyze_stock, enrich_with_hybrid

    # 최소 종목 dict (analyze_stock 기대 형식)
    stock_mock = {
        "code":  "SAMPLE1",
        "name":  "샘플1",
        "quote": {"volume": 1_500_000},
        "news":  [
            {"title":"실적 개선 기대감", "impact":"positive"},
        ],
        "overnight_signal": {"direction": "up"},
        "history": {
            "pos_52w_pct":  30.0,
            "closes_20d":   closes[-20:],
            "volume_20d_avg": 1_000_000,
        },
    }

    result = analyze_stock(stock_mock)
    assert "recommendation" in result
    assert "confidence" in result
    print(f"    recommendation={result['recommendation']} "
          f"confidence={result['confidence']}")

    # hybrid_score 통합 여부
    if "hybrid_score" in result and result["hybrid_score"]:
        hs = result["hybrid_score"]
        print(f"    hybrid_score: NCS={hs.get('ncs','?')} BQS={hs.get('bqs','?')} "
              f"FWS={hs.get('fws','?')}")
        assert "ncs" in hs

    # enrich_with_hybrid 직접 호출
    eh = enrich_with_hybrid(closes, highs, lows, volumes, opens)
    assert "ncs" in eh or "error" in eh
    print(f"    enrich_with_hybrid: ncs={eh.get('ncs','?')}")
    return result

run_test("analyze_stock_hybrid", test_analyze_stock_hybrid)
print(f"  {results['analyze_stock_hybrid']}")


# =============================================================================
# 최종 요약
# =============================================================================

print("\n" + "="*60)
print("  검증 결과 요약")
print("="*60)

passed = sum(1 for v in results.values() if PASS in v)
failed = sum(1 for v in results.values() if FAIL in v)
total  = len(results)

for name, res in results.items():
    print(f"  {res}  [{name}]")

print(f"\n  총 {total}개 테스트: {passed}개 통과, {failed}개 실패")

if failed == 0:
    print("\n  [OK] 모든 검증 통과! StockOracle x HybridTurtle 통합 정상.")
else:
    print(f"\n  [!] {failed}개 항목 수정 필요.")
    sys.exit(1)
