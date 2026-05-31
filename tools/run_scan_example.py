"""
run_scan_example.py — 7단계 스캔 엔진 사용 예시

실행:
    python -m tools.run_scan_example

내부적으로 yfinance를 사용해 OHLCV 데이터를 가져오고,
7단계 스캔 파이프라인을 실행해 결과를 출력합니다.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yfinance as yf
import numpy as np
from market_briefing.scan_engine import (
    StockUniverse, TechnicalSnapshot, run_full_scan, build_snapshot_from_ohlcv,
)
from market_briefing.quality_filter import get_quality_score_from_info
from market_briefing.dual_score_v2 import REGIME_BULLISH, VOL_NORMAL


# ── 스캔 대상 종목 정의 ─────────────────────────────────────────────────────
UNIVERSE = [
    StockUniverse("005930.KS", "삼성전자",      "CORE",      "Technology", "반도체"),
    StockUniverse("000660.KS", "SK하이닉스",    "CORE",      "Technology", "반도체"),
    StockUniverse("035420.KS", "NAVER",         "CORE",      "Technology", "인터넷"),
    StockUniverse("051910.KS", "LG화학",        "CORE",      "Materials",  "화학"),
    StockUniverse("035720.KS", "카카오",        "HIGH_RISK", "Technology", "인터넷"),
    StockUniverse("207940.KS", "삼성바이오로직스","CORE",     "Healthcare", "바이오"),
    StockUniverse("006400.KS", "삼성SDI",       "CORE",      "Technology", "배터리"),
    StockUniverse("028260.KS", "삼성물산",      "ETF",       "Industrials","대기업"),
]

def fetch_ohlcv(ticker: str, period: str = "1y") -> dict:
    """yfinance에서 OHLCV 데이터 다운로드 (Ticker.history 사용 — MultiIndex 안전)."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return {}
        cl = hist["Close"].dropna().tolist()
        hi = hist["High"].dropna().tolist()
        lo = hist["Low"].dropna().tolist()
        vo = hist["Volume"].dropna().tolist()
        op = hist["Open"].dropna().tolist()
        n  = min(len(cl), len(hi), len(lo), len(vo), len(op))
        return {
            "closes":  cl[:n],
            "highs":   hi[:n],
            "lows":    lo[:n],
            "volumes": vo[:n],
            "opens":   op[:n],
        }
    except Exception as e:
        print(f"[ERROR] {ticker} 데이터 로드 실패: {e}")
        return {}


def main():
    print("=" * 60)
    print("  StockOracle — 7단계 종목 스캔 엔진")
    print("=" * 60)

    # KOSPI200 벤치마크 데이터
    bench_data = fetch_ohlcv("^KS200", "1y")
    bench_closes = bench_data.get("closes", [])

    # 스냅샷 & 품질 필터 빌드
    snap_map    = {}
    quality_map = {}

    for stock in UNIVERSE:
        print(f"\n[{stock.ticker}] {stock.name} 데이터 로드 중...")
        ohlcv = fetch_ohlcv(stock.ticker, "1y")
        if not ohlcv or len(ohlcv.get("closes", [])) < 60:
            print(f"  → 데이터 부족, 건너뜀")
            continue

        # TechnicalSnapshot 구성
        snap = build_snapshot_from_ohlcv(
            ticker       = stock.ticker,
            closes       = ohlcv["closes"],
            highs        = ohlcv["highs"],
            lows         = ohlcv["lows"],
            volumes      = ohlcv["volumes"],
            opens        = ohlcv["opens"],
            bench_closes = bench_closes,
        )
        snap_map[stock.ticker] = snap

        # QMJ 품질 필터 (yfinance info)
        try:
            t    = yf.Ticker(stock.ticker)
            info = t.info
            qmj  = get_quality_score_from_info(stock.ticker, info)
            quality_map[stock.ticker] = qmj
            print(f"  → 현재가: {snap.current_price:,.0f} | ADX: {snap.adx:.1f} | ATR%: {snap.atr_pct:.1f}% | QMJ: {qmj.quality_tier}")
        except Exception as e:
            print(f"  → QMJ 로드 실패: {e}")

    if not snap_map:
        print("\n스냅샷 데이터가 없습니다. 네트워크 연결을 확인하세요.")
        return

    # ── 7단계 스캔 실행 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  7단계 스캔 파이프라인 실행")
    print("=" * 60)

    result = run_full_scan(
        universe           = [s for s in UNIVERSE if s.ticker in snap_map],
        snap_map           = snap_map,
        quality_map        = quality_map,
        regime             = REGIME_BULLISH,   # 실제로는 시장 레짐 자동 감지 사용
        vol_regime         = VOL_NORMAL,
        regime_stable      = True,
        dual_aligned       = False,
        portfolio_equity   = 100_000_000,      # 1억원 포트폴리오 예시
        existing_positions = [],
        risk_pct_per_trade = 1.0,
        scan_mode          = "FULL",
    )

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print(f"\n📊 스캔 요약")
    print(f"  전체 스캔: {result.total_scanned}개 | 필터 통과: {result.passed_filters}개")
    print(f"  READY: {result.ready_count} | WATCH: {result.watch_count} | FAR: {result.far_count}")
    print(f"  레짐: {result.regime} | 변동성: {result.vol_regime}")

    print(f"\n{'순위':>4} {'종목':>12} {'상태':>12} {'BQS':>6} {'FWS':>6} {'NCS':>6} {'adj.NCS':>8} {'QMJ':>8} {'액션'}")
    print("-" * 90)

    for i, c in enumerate(result.candidates[:10], 1):
        status_emoji = {"READY":"🟢","WATCH":"🟡","FAR":"⚪","WAIT_PULLBACK":"🔵","COOLDOWN":"🔴","EARNINGS_BLOCK":"🚫"}.get(c.status,"")
        print(
            f"{i:>4} {c.ticker:>12} {status_emoji}{c.status:>10} "
            f"{c.bqs:>6.1f} {c.fws:>6.1f} {c.ncs:>6.1f} {c.adjusted_ncs:>8.1f} "
            f"{c.quality_tier:>8} {c.action_note[:25]}"
        )

    # READY 후보 상세 출력
    ready = [c for c in result.candidates if c.status == "READY" and c.passes_tech_filters]
    if ready:
        print(f"\n🟢 READY 후보 상세 ({len(ready)}개)")
        for c in ready:
            print(f"\n  [{c.ticker}] {c.name}")
            print(f"    현재가: {c.price:,.0f} | 진입: {c.entry_trigger:,.0f} | 손절: {c.stop_price:,.0f}")
            print(f"    BQS 서브: 추세={c.bqs_components.get('bqs_trend',0):.1f} | "
                  f"방향={c.bqs_components.get('bqs_direction',0):.1f} | "
                  f"테일윈드={c.bqs_components.get('bqs_tailwind',0):.1f} | "
                  f"Hurst={c.bqs_components.get('bqs_hurst',0):.1f}")
            print(f"    액션: {c.action_note}")
            if c.shares:
                print(f"    포지션: {c.shares:.2f}주 | 리스크: {c.risk_amount:,.0f}원 ({c.risk_pct:.2f}%)")
    else:
        print("\n  READY 후보 없음 (시장 상황 또는 조건 미충족)")

    print("\n✅ 스캔 완료")


if __name__ == "__main__":
    main()
