# market_briefing — k-ant-daily 로직을 StockOracle에 통합한 모듈
#
# 핵심 기능 모듈:
#   core_summary   → ⭐ 오늘의 핵심 (거시 시장 요약)
#   sector_flow    → 🏭 섹터 흐름 (산업별 방향성)
#   stock_analyzer → 📈 종목별 분석 (예측 + 사후 검증)
#
# HybridTurtle v6.0 통합 모듈 (2026-05):
#   hybrid_signals  → BQS/FWS/NCS 복합 점수 (v1)
#   dual_score_v2   → BQS/FWS/NCS v2 (주간ADX, Hurst, DualRegime, 클러스터 패널티)
#   quality_filter  → QMJ(AQR Quality Minus Junk) 재무 사전 필터
#   scan_engine     → 7단계 종목 후보 스캔 파이프라인

from .core_summary import build_core_summary
from .sector_flow import build_sector_flow
from .stock_analyzer import analyze_stock, classify_prediction, build_stock_report
from .dual_score_v2 import (
    SnapshotRow, score_row, score_all,
    compute_bqs, compute_fws, compute_penalties, compute_ncs, action_note,
    calc_hurst_v2, compute_bis_from_candle, compute_chasing_flags,
    calc_dual_regime_score,
)
from .quality_filter import (
    QualityFilterResult, score_quality, get_quality_score_from_info,
    get_conviction_bonus,
)
from .scan_engine import (
    StockUniverse, TechnicalSnapshot, ScanCandidate, ScanResult,
    run_full_scan, build_snapshot_from_ohlcv,
    run_technical_filters, classify_candidate, rank_candidate,
    check_anti_chase, check_pullback_continuation,
    calculate_position_size,
)

__all__ = [
    # 기존
    "build_core_summary",
    "build_sector_flow",
    "analyze_stock",
    "classify_prediction",
    "build_stock_report",
    # dual_score_v2
    "SnapshotRow", "score_row", "score_all",
    "compute_bqs", "compute_fws", "compute_penalties", "compute_ncs", "action_note",
    "calc_hurst_v2", "compute_bis_from_candle", "compute_chasing_flags",
    "calc_dual_regime_score",
    # quality_filter
    "QualityFilterResult", "score_quality", "get_quality_score_from_info",
    "get_conviction_bonus",
    # scan_engine
    "StockUniverse", "TechnicalSnapshot", "ScanCandidate", "ScanResult",
    "run_full_scan", "build_snapshot_from_ohlcv",
    "run_technical_filters", "classify_candidate", "rank_candidate",
    "check_anti_chase", "check_pullback_continuation",
    "calculate_position_size",
]
