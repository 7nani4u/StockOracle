"""
dashboard_payload.py — 대시보드 커맨드 센터 상태 페이로드 생성기

HybridTurtle-v6.0 참조:
  packages/workflow/src/dashboard.ts → Tonight's Workflow 카드
  packages/stops/src/dashboard.ts    → 손절가 대시보드

역할:
  - 각 모듈의 상태 (OK / WARN / ERROR)를 수집
  - 스캔 결과 요약 생성
  - 시스템 하트비트 정보 생성
  - 활동 계획 (액션 아이템) 생성
  - 전체 대시보드 페이로드 dict 반환

공개 API:
  DashboardPayloadBuilder.build()     — 전체 대시보드 페이로드 생성
  DashboardPayloadBuilder.heartbeat() — 모듈 하트비트만 반환
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── 모듈 상태 확인 ────────────────────────────────────────────────────────────

def _check_module_status(module_name: str) -> Dict[str, Any]:
    """Python 모듈 import 가능 여부 확인."""
    try:
        __import__(module_name)
        return {"status": "OK", "message": "로드됨"}
    except ImportError as e:
        return {"status": "ERROR", "message": str(e)}
    except Exception as e:
        return {"status": "WARN", "message": str(e)}


def _probe_market_briefing_modules() -> Dict[str, Dict[str, Any]]:
    """market_briefing 하위 모듈 상태 프로브."""
    modules = {
        "hybrid_signals":   "market_briefing.hybrid_signals",
        "dual_score_v2":    "market_briefing.dual_score_v2",
        "quality_filter":   "market_briefing.quality_filter",
        "scan_engine":      "market_briefing.scan_engine",
        "stock_analyzer":   "market_briefing.stock_analyzer",
        "portfolio_manager":"market_briefing.portfolio_manager",
        "market_immune":    "market_briefing.market_immune",
        "cross_reference":  "market_briefing.cross_reference",
    }
    results = {}
    for key, mod_path in modules.items():
        results[key] = _check_module_status(mod_path)
    return results


def _probe_optional_deps() -> Dict[str, Dict[str, Any]]:
    """선택적 의존성 상태 확인."""
    deps = {
        "numpy":      "numpy",
        "pandas":     "pandas",
        "yfinance":   "yfinance",
        "requests":   "requests",
        "feedparser": "feedparser",
    }
    results = {}
    for key, mod in deps.items():
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            results[key] = {"status": "OK", "version": ver}
        except ImportError:
            results[key] = {"status": "MISSING", "version": None}
    return results


# ── 액션 아이템 ────────────────────────────────────────────────────────────────

_DEFAULT_ACTIONS = [
    {
        "key":         "refresh-data",
        "label":       "시장 데이터 새로고침",
        "description": "장 마감 후 유니버스 OHLCV 데이터 갱신",
        "priority":    1,
    },
    {
        "key":         "run-scan",
        "label":       "7단계 스캔 실행",
        "description": "오늘의 후보 종목 스캔 — tools/run_scan_example.py 또는 /api/scan",
        "priority":    2,
    },
    {
        "key":         "review-candidates",
        "label":       "후보 검토",
        "description": "READY/WATCH 후보 BQS/FWS/NCS 확인",
        "priority":    3,
    },
    {
        "key":         "review-risk",
        "label":       "리스크 검토",
        "description": "포트폴리오 오픈 리스크 및 손절가 확인",
        "priority":    4,
    },
    {
        "key":         "cross-reference",
        "label":       "교차 참조",
        "description": "스캔 결과 × 뉴스 감성 × NCS 최종 점수 통합",
        "priority":    5,
    },
    {
        "key":         "check-immune",
        "label":       "시장 위험 면역 확인",
        "description": "VIX / MA200 이격도 기반 시장 위험 레벨 평가",
        "priority":    6,
    },
    {
        "key":         "update-stops",
        "label":       "손절가 갱신",
        "description": "기존 포지션 손절가 상향 가능 여부 확인",
        "priority":    7,
    },
]


# ── 빌더 ──────────────────────────────────────────────────────────────────────

class DashboardPayloadBuilder:
    """대시보드 페이로드 생성기."""

    def heartbeat(self) -> Dict[str, Any]:
        """모듈 하트비트 — 각 모듈 상태만 빠르게 수집."""
        modules = _probe_market_briefing_modules()
        deps    = _probe_optional_deps()

        all_ok      = all(v["status"] == "OK" for v in modules.values())
        error_count = sum(1 for v in modules.values() if v["status"] == "ERROR")
        warn_count  = sum(1 for v in modules.values() if v["status"] == "WARN")

        overall = "OK" if all_ok else ("ERROR" if error_count > 0 else "WARN")

        return {
            "overall":    overall,
            "modules":    modules,
            "deps":       deps,
            "python_ver": sys.version,
            "checked_at": datetime.now().isoformat(),
        }

    def build(
        self,
        scan_result:          Optional[Dict[str, Any]] = None,
        portfolio_payload:    Optional[Dict[str, Any]] = None,
        immune_result:        Optional[Dict[str, Any]] = None,
        cross_ref_summary:    Optional[List[Dict]]    = None,
        last_scan_at:         Optional[str]            = None,
        last_refresh_at:      Optional[str]            = None,
    ) -> Dict[str, Any]:
        """전체 대시보드 페이로드 생성.

        Args:
            scan_result:       run_full_scan() 반환값의 to_dict() (없으면 None)
            portfolio_payload: PortfolioManager.get_dashboard_payload() (없으면 None)
            immune_result:     MarketImmuneResult.to_dict() (없으면 None)
            cross_ref_summary: CrossReferenceResult 리스트의 to_dict() 목록
            last_scan_at:      마지막 스캔 시각 ISO 문자열
            last_refresh_at:   마지막 데이터 새로고침 시각

        Returns:
            전체 대시보드 페이로드 dict
        """
        now = datetime.now().isoformat()
        heartbeat = self.heartbeat()

        # ── 스캔 요약 ──────────────────────────────────────────────────────
        scan_summary = None
        if scan_result:
            cands  = scan_result.get("candidates", [])
            ready  = [c for c in cands if c.get("status") == "READY" and c.get("passes_tech_filters", True)]
            watch  = [c for c in cands if c.get("status") in ("WATCH", "WAIT_PULLBACK")]
            scan_summary = {
                "total_scanned":   scan_result.get("total_scanned", 0),
                "passed_filters":  scan_result.get("passed_filters", 0),
                "ready_count":     scan_result.get("ready_count", 0),
                "watch_count":     scan_result.get("watch_count", 0),
                "far_count":       scan_result.get("far_count", 0),
                "regime":          scan_result.get("regime", "UNKNOWN"),
                "vol_regime":      scan_result.get("vol_regime", "UNKNOWN"),
                "top_ready":       [
                    {
                        "ticker":    c.get("ticker", ""),
                        "name":      c.get("name", ""),
                        "bqs":       c.get("bqs", 0),
                        "ncs":       c.get("ncs", 0),
                        "fws":       c.get("fws", 0),
                        "action_note": c.get("action_note", ""),
                    }
                    for c in ready[:5]
                ],
                "last_scan_at":    last_scan_at or scan_result.get("generated_at"),
            }

        # ── 포트폴리오 요약 ────────────────────────────────────────────────
        portfolio_summary = None
        if portfolio_payload:
            s = portfolio_payload.get("summary", {})
            portfolio_summary = {
                "open_positions": s.get("open_positions", 0),
                "risk_level":     s.get("risk_level", "UNKNOWN"),
                "open_risk_pct":  s.get("open_risk_pct", 0),
                "missing_stops":  s.get("missing_stops", 0),
                "expectancy":     portfolio_payload.get("expectancy"),
            }

        # ── 면역 레벨 요약 ─────────────────────────────────────────────────
        immune_summary = None
        if immune_result:
            immune_summary = {
                "immune_level": immune_result.get("immune_level", "CLEAR"),
                "immune_score": immune_result.get("immune_score", 0),
                "vix_level":    immune_result.get("vix_level"),
                "warnings":     immune_result.get("warnings", [])[:3],
                "kill_switch":  immune_result.get("kill_switch", {}),
                "top_crisis":   immune_result.get("top_crisis"),
            }

        # ── 교차 참조 Top 5 ───────────────────────────────────────────────
        top_cross_ref = None
        if cross_ref_summary:
            top_cross_ref = [
                {
                    "ticker":      r.get("ticker", ""),
                    "name":        r.get("name", ""),
                    "final_score": r.get("final_score", 0),
                    "final_label": r.get("final_label", "HOLD"),
                    "confidence":  r.get("confidence", "low"),
                    "scan_status": r.get("scan_status", ""),
                    "action_note": r.get("action_note", ""),
                }
                for r in cross_ref_summary[:5]
            ]

        # ── 전체 시스템 상태 ──────────────────────────────────────────────
        system_ok = heartbeat["overall"] == "OK"
        immune_lv = (immune_result or {}).get("immune_level", "CLEAR")
        risk_lv   = (portfolio_summary or {}).get("risk_level", "LOW") if portfolio_summary else "LOW"

        if immune_lv in ("IMMUNE", "ALERT") or risk_lv == "HIGH":
            overall_status = "ALERT"
        elif immune_lv == "CAUTION" or risk_lv == "MEDIUM" or not system_ok:
            overall_status = "CAUTION"
        else:
            overall_status = "OK"

        return {
            "status":           overall_status,
            "title":            "StockOracle 대시보드",
            "generated_at":     now,
            "last_refresh_at":  last_refresh_at,

            # 모듈 하트비트
            "heartbeat":        heartbeat,

            # 핵심 섹션
            "scan":             scan_summary,
            "portfolio":        portfolio_summary,
            "immune":           immune_summary,
            "top_candidates":   top_cross_ref,

            # 활동 계획
            "action_plan":      _build_action_plan(
                scan_summary, portfolio_summary, immune_summary, heartbeat
            ),
        }


def _build_action_plan(
    scan_summary:      Optional[Dict],
    portfolio_summary: Optional[Dict],
    immune_summary:    Optional[Dict],
    heartbeat:         Dict,
) -> List[Dict[str, Any]]:
    """현재 상태 기반 우선순위 액션 플랜 생성."""
    actions = list(_DEFAULT_ACTIONS)

    # 모듈 오류 시 수정 액션 삽입
    error_mods = [
        k for k, v in heartbeat.get("modules", {}).items()
        if v.get("status") == "ERROR"
    ]
    if error_mods:
        actions.insert(0, {
            "key":         "fix-modules",
            "label":       "⚠️ 모듈 오류 수정 필요",
            "description": f"오류 모듈: {', '.join(error_mods)}",
            "priority":    0,
            "urgent":      True,
        })

    # 면역 경보 시 우선 액션
    if immune_summary and immune_summary.get("immune_level") in ("IMMUNE", "ALERT"):
        actions.insert(0, {
            "key":         "immune-alert",
            "label":       f"🚨 시장 위험 {immune_summary['immune_level']}",
            "description": "신규 매수 중단, 기존 포지션 손절가 확인",
            "priority":    0,
            "urgent":      True,
        })

    # 손절가 없는 포지션
    if portfolio_summary and portfolio_summary.get("missing_stops", 0) > 0:
        n = portfolio_summary["missing_stops"]
        actions.insert(1, {
            "key":         "missing-stops-urgent",
            "label":       f"⚠️ 손절가 없는 포지션 {n}개",
            "description": "즉시 손절가 설정 필요",
            "priority":    0,
            "urgent":      True,
        })

    # 스캔 미실행
    if scan_summary is None:
        for a in actions:
            if a["key"] == "run-scan":
                a["urgent"] = True
                a["label"]  = "🔍 스캔 미실행 — 즉시 실행 필요"

    return actions
