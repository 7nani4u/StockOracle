"""
portfolio_manager.py — 포트폴리오·손절가·R-멀티플 관리

HybridTurtle-v6.0 이식 대상:
  packages/stops/src/service.ts    → 손절가 관리 (상향 조정만 허용)
  packages/portfolio/src/view.ts   → 포트폴리오 뷰
  packages/risk/src/validation.ts  → 리스크 게이트

공개 API:
  PortfolioPosition  — 포지션 데이터 클래스
  PortfolioState     — 전체 포트폴리오 상태
  PortfolioManager   — 핵심 관리 클래스
    add_position()      — 포지션 등록
    update_stop()       — 손절가 갱신 (상향만 허용)
    close_position()    — 포지션 청산 + R-멀티플 기록
    get_state()         — 현재 포트폴리오 상태 반환
    assess_new_trade()  — 신규 거래 리스크 평가
    get_dashboard_payload() — 대시보드용 상태 dict
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── 상수 ──────────────────────────────────────────────────────────────────────

MAX_OPEN_RISK_PCT    = 10.0   # 총 오픈 리스크 상한 (자본 대비 %)
MAX_POSITIONS        = 6      # 최대 동시 포지션 수
MAX_POSITION_PCT     = 25.0   # 단일 포지션 최대 비중 (%)
MAX_STOP_DISTANCE_PCT= 10.0   # 최대 손절 거리 (%)
MAX_SECTOR_PCT       = 40.0   # 섹터 집중도 상한 (%)


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────

@dataclass
class PortfolioPosition:
    """단일 포지션."""
    ticker:        str
    name:          str
    entry_price:   float
    stop_price:    float
    shares:        float
    entry_date:    str = ""
    sector:        str = ""
    sleeve:        str = "CORE"
    current_price: float = 0.0
    status:        str = "OPEN"   # OPEN / CLOSED
    close_price:   float = 0.0
    close_date:    str = ""
    notes:         str = ""

    @property
    def risk_per_share(self) -> float:
        return max(self.entry_price - self.stop_price, 0.01)

    @property
    def position_value(self) -> float:
        price = self.current_price if self.current_price > 0 else self.entry_price
        return self.shares * price

    @property
    def open_risk(self) -> float:
        price = self.current_price if self.current_price > 0 else self.entry_price
        return max(price - self.stop_price, 0) * self.shares

    @property
    def r_multiple(self) -> Optional[float]:
        """청산 포지션의 R-멀티플. CLOSED 상태일 때만 유효."""
        if self.status != "CLOSED" or self.close_price <= 0:
            return None
        profit = (self.close_price - self.entry_price) * self.shares
        initial_risk = self.risk_per_share * self.shares
        if initial_risk <= 0:
            return None
        return round(profit / initial_risk, 2)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price <= 0 or self.current_price <= 0:
            return 0.0
        return round((self.current_price / self.entry_price - 1) * 100, 2)

    @property
    def stop_distance_pct(self) -> float:
        price = self.current_price if self.current_price > 0 else self.entry_price
        if price <= 0:
            return 0.0
        return round((price - self.stop_price) / price * 100, 2)

    @property
    def has_stop(self) -> bool:
        return self.stop_price > 0 and self.stop_price < self.entry_price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker":            self.ticker,
            "name":              self.name,
            "entry_price":       self.entry_price,
            "stop_price":        self.stop_price,
            "shares":            self.shares,
            "entry_date":        self.entry_date,
            "sector":            self.sector,
            "sleeve":            self.sleeve,
            "current_price":     self.current_price,
            "status":            self.status,
            "close_price":       self.close_price,
            "close_date":        self.close_date,
            "notes":             self.notes,
            "risk_per_share":    round(self.risk_per_share, 4),
            "position_value":    round(self.position_value, 2),
            "open_risk":         round(self.open_risk, 2),
            "r_multiple":        self.r_multiple,
            "unrealized_pnl_pct":self.unrealized_pnl_pct,
            "stop_distance_pct": self.stop_distance_pct,
            "has_stop":          self.has_stop,
        }


@dataclass
class PortfolioState:
    """전체 포트폴리오 상태 스냅샷."""
    equity:              float
    cash_balance:        float
    positions:           List[PortfolioPosition] = field(default_factory=list)
    closed_positions:    List[PortfolioPosition] = field(default_factory=list)
    generated_at:        str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def open_positions(self) -> List[PortfolioPosition]:
        return [p for p in self.positions if p.status == "OPEN"]

    @property
    def total_market_value(self) -> float:
        return sum(p.position_value for p in self.open_positions)

    @property
    def total_open_risk(self) -> float:
        return sum(p.open_risk for p in self.open_positions)

    @property
    def open_risk_pct(self) -> float:
        if self.equity <= 0:
            return 0.0
        return round(self.total_open_risk / self.equity * 100, 2)

    @property
    def missing_stops_count(self) -> int:
        return sum(1 for p in self.open_positions if not p.has_stop)

    @property
    def risk_level(self) -> str:
        if self.missing_stops_count > 0 or self.open_risk_pct > 8.0:
            return "HIGH"
        if self.open_risk_pct > 5.0:
            return "MEDIUM"
        return "LOW"

    @property
    def concentration(self) -> List[Dict]:
        mv = self.total_market_value
        result = []
        for p in self.open_positions:
            result.append({
                "ticker":    p.ticker,
                "value":     round(p.position_value, 2),
                "weight_pct":round(p.position_value / mv * 100, 1) if mv > 0 else 0.0,
                "sector":    p.sector,
            })
        result.sort(key=lambda x: x["weight_pct"], reverse=True)
        return result

    @property
    def r_multiple_history(self) -> List[Dict]:
        result = []
        for p in self.closed_positions:
            rm = p.r_multiple
            if rm is not None:
                result.append({
                    "ticker":     p.ticker,
                    "r_multiple": rm,
                    "close_date": p.close_date,
                    "status":     "WIN" if rm > 0 else "LOSS",
                })
        return result

    @property
    def expectancy(self) -> Optional[float]:
        """기대값 = 평균 R-멀티플."""
        rms = [p.r_multiple for p in self.closed_positions if p.r_multiple is not None]
        if not rms:
            return None
        return round(sum(rms) / len(rms), 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity":            self.equity,
            "cash_balance":      self.cash_balance,
            "total_market_value":round(self.total_market_value, 2),
            "total_open_risk":   round(self.total_open_risk, 2),
            "open_risk_pct":     self.open_risk_pct,
            "open_positions_count": len(self.open_positions),
            "missing_stops_count":  self.missing_stops_count,
            "risk_level":        self.risk_level,
            "concentration":     self.concentration,
            "r_multiple_history":self.r_multiple_history[:20],
            "expectancy":        self.expectancy,
            "generated_at":      self.generated_at,
            "positions":         [p.to_dict() for p in self.open_positions],
        }


@dataclass
class TradeRiskAssessment:
    """신규 거래 리스크 평가 결과."""
    ticker:             str
    approved:           bool
    recommended_shares: float
    risk_per_trade:     float
    risk_per_share:     float
    entry_price:        float
    stop_price:         float
    stop_distance_pct:  float
    position_value:     float
    open_risk_after:    float
    open_risk_pct_after:float
    violations:         List[Dict[str, str]] = field(default_factory=list)
    rationale:          str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker":              self.ticker,
            "approved":            self.approved,
            "recommended_shares":  round(self.recommended_shares, 4),
            "risk_per_trade":      round(self.risk_per_trade, 2),
            "risk_per_share":      round(self.risk_per_share, 4),
            "entry_price":         self.entry_price,
            "stop_price":          self.stop_price,
            "stop_distance_pct":   round(self.stop_distance_pct, 2),
            "position_value":      round(self.position_value, 2),
            "open_risk_after":     round(self.open_risk_after, 2),
            "open_risk_pct_after": round(self.open_risk_pct_after, 2),
            "violations":          self.violations,
            "rationale":           self.rationale,
        }


# ── 포트폴리오 매니저 ──────────────────────────────────────────────────────────

class PortfolioManager:
    """포트폴리오 관리 클래스.

    인스턴스를 메모리에 유지하거나, 외부 JSON 파일에서 상태를 직렬화/역직렬화.
    """

    def __init__(
        self,
        equity:            float = 10_000_000.0,
        cash_balance:      float = 10_000_000.0,
        risk_pct_per_trade:float = 1.0,
    ):
        self._equity             = equity
        self._cash_balance       = cash_balance
        self._risk_pct_per_trade = risk_pct_per_trade
        self._positions:         List[PortfolioPosition] = []
        self._closed_positions:  List[PortfolioPosition] = []

    # ── 상태 조회 ────────────────────────────────────────────────────────────

    def get_state(self) -> PortfolioState:
        return PortfolioState(
            equity           = self._equity,
            cash_balance     = self._cash_balance,
            positions        = list(self._positions),
            closed_positions = list(self._closed_positions),
        )

    def get_open_position(self, ticker: str) -> Optional[PortfolioPosition]:
        for p in self._positions:
            if p.ticker == ticker and p.status == "OPEN":
                return p
        return None

    # ── 포지션 등록 ───────────────────────────────────────────────────────────

    def add_position(
        self,
        ticker:       str,
        name:         str,
        entry_price:  float,
        stop_price:   float,
        shares:       float,
        sector:       str = "",
        sleeve:       str = "CORE",
        notes:        str = "",
    ) -> PortfolioPosition:
        """신규 포지션 등록.

        stop_price < entry_price 조건 미충족 시 ValueError 발생.
        """
        if stop_price >= entry_price:
            raise ValueError(f"손절가({stop_price})는 진입가({entry_price})보다 낮아야 합니다.")
        if shares <= 0:
            raise ValueError("수량(shares)은 0보다 커야 합니다.")

        pos = PortfolioPosition(
            ticker       = ticker,
            name         = name,
            entry_price  = entry_price,
            stop_price   = stop_price,
            shares       = shares,
            entry_date   = datetime.now().isoformat(),
            sector       = sector,
            sleeve       = sleeve,
            current_price= entry_price,
            status       = "OPEN",
            notes        = notes,
        )
        self._positions.append(pos)
        cost = entry_price * shares
        self._cash_balance = max(0.0, self._cash_balance - cost)
        return pos

    # ── 손절가 갱신 (상향만 허용) ─────────────────────────────────────────────

    def update_stop(
        self,
        ticker:        str,
        new_stop:      float,
        force_lower:   bool = False,
    ) -> Dict[str, Any]:
        """손절가 갱신. 기본적으로 상향 조정만 허용 (HybridTurtle 원칙).

        Args:
            ticker:     종목 코드
            new_stop:   새 손절가
            force_lower: True이면 하향 허용 (긴급 상황)

        Returns:
            {"ok": bool, "old_stop": float, "new_stop": float, "message": str}
        """
        pos = self.get_open_position(ticker)
        if pos is None:
            return {"ok": False, "message": f"{ticker}: 오픈 포지션 없음"}

        old_stop = pos.stop_price

        if not force_lower and new_stop < old_stop:
            return {
                "ok":      False,
                "old_stop":old_stop,
                "new_stop":new_stop,
                "message": f"손절가 하향 불가 ({old_stop} → {new_stop}) — force_lower=True로 강제 가능",
            }

        pos.stop_price = new_stop
        action = "갱신" if new_stop >= old_stop else "강제 하향"
        return {
            "ok":      True,
            "old_stop":old_stop,
            "new_stop":new_stop,
            "message": f"{ticker} 손절가 {action}: {old_stop} → {new_stop}",
        }

    # ── 현재가 갱신 ───────────────────────────────────────────────────────────

    def update_prices(self, price_map: Dict[str, float]) -> None:
        """종목별 현재가 일괄 갱신."""
        for pos in self._positions:
            if pos.ticker in price_map and pos.status == "OPEN":
                pos.current_price = price_map[pos.ticker]

    # ── 포지션 청산 ───────────────────────────────────────────────────────────

    def close_position(
        self,
        ticker:      str,
        close_price: float,
        notes:       str = "",
    ) -> Dict[str, Any]:
        """포지션 청산 및 R-멀티플 계산."""
        pos = self.get_open_position(ticker)
        if pos is None:
            return {"ok": False, "message": f"{ticker}: 오픈 포지션 없음"}

        pos.status      = "CLOSED"
        pos.close_price = close_price
        pos.close_date  = datetime.now().isoformat()
        if notes:
            pos.notes += f" | 청산: {notes}"

        self._positions.remove(pos)
        self._closed_positions.append(pos)

        # 현금 복원
        proceeds = close_price * pos.shares
        self._cash_balance += proceeds

        rm = pos.r_multiple
        return {
            "ok":         True,
            "ticker":     ticker,
            "r_multiple": rm,
            "pnl":        round((close_price - pos.entry_price) * pos.shares, 2),
            "pnl_pct":    round((close_price / pos.entry_price - 1) * 100, 2),
            "message":    f"R={rm:+.2f}" if rm is not None else "청산 완료",
        }

    # ── 신규 거래 리스크 평가 ────────────────────────────────────────────────

    def assess_new_trade(
        self,
        ticker:      str,
        entry_price: float,
        stop_price:  float,
        sector:      str = "",
        sleeve:      str = "CORE",
    ) -> TradeRiskAssessment:
        """신규 거래의 리스크를 사전 평가. HybridTurtle validation.ts 이식.

        6가지 하드 게이트 + 2가지 소프트 경고.
        모두 통과해야 approved=True.
        """
        state         = self.get_state()
        open_positions = state.open_positions
        violations: List[Dict[str, str]] = []

        if stop_price >= entry_price:
            violations.append({"rule": "STOP_BELOW_ENTRY", "severity": "HARD",
                                "message": f"손절가({stop_price}) ≥ 진입가({entry_price})"})

        risk_per_share  = max(entry_price - stop_price, 0.01)
        stop_dist_pct   = risk_per_share / entry_price * 100
        risk_budget     = self._equity * (self._risk_pct_per_trade / 100.0)
        risk_budget     = min(risk_budget, self._cash_balance)
        shares          = math.floor(risk_budget / risk_per_share) if risk_per_share > 0 else 0
        max_cost        = self._equity * ({
            "CORE": 0.15, "ETF": 0.20, "HIGH_RISK": 0.08, "HEDGE": 0.10,
        }.get(sleeve, 0.15))
        if shares * entry_price > max_cost:
            shares = math.floor(max_cost / entry_price)
        shares = min(shares, math.floor(self._cash_balance / entry_price))

        if shares < 1:
            violations.append({"rule": "MIN_SHARES", "severity": "HARD",
                                "message": f"자본 부족: 최소 1주 매입 불가 (진입가={entry_price})"})

        position_value = shares * entry_price
        risk_per_trade = shares * risk_per_share
        new_open_risk  = state.total_open_risk + risk_per_trade
        new_open_pct   = new_open_risk / self._equity * 100 if self._equity > 0 else 0.0

        if new_open_pct > MAX_OPEN_RISK_PCT:
            violations.append({"rule": "MAX_OPEN_RISK", "severity": "HARD",
                                "message": f"오픈 리스크 {new_open_pct:.1f}% > 상한 {MAX_OPEN_RISK_PCT}%"})

        if len(open_positions) >= MAX_POSITIONS:
            violations.append({"rule": "MAX_POSITIONS", "severity": "HARD",
                                "message": f"최대 포지션 수 초과 ({len(open_positions)}/{MAX_POSITIONS})"})

        if stop_dist_pct > MAX_STOP_DISTANCE_PCT:
            violations.append({"rule": "STOP_DISTANCE", "severity": "HARD",
                                "message": f"손절 거리 {stop_dist_pct:.1f}% > 상한 {MAX_STOP_DISTANCE_PCT}%"})

        total_mv       = state.total_market_value + position_value
        new_pos_weight = position_value / total_mv * 100 if total_mv > 0 else 0.0
        if new_pos_weight > MAX_POSITION_PCT:
            violations.append({"rule": "CONCENTRATION", "severity": "HARD",
                                "message": f"포지션 비중 {new_pos_weight:.1f}% > 상한 {MAX_POSITION_PCT}%"})

        if sector:
            sector_val = sum(
                p.position_value for p in open_positions if p.sector == sector
            )
            sector_pct = (sector_val + position_value) / self._equity * 100
            if sector_pct > MAX_SECTOR_PCT:
                violations.append({"rule": "SECTOR_CONCENTRATION", "severity": "HARD",
                                    "message": f"섹터 집중도 {sector_pct:.1f}% > 상한 {MAX_SECTOR_PCT}%"})

        # 소프트 경고
        if state.missing_stops_count > 0:
            violations.append({"rule": "MISSING_STOPS", "severity": "SOFT",
                                "message": f"손절가 없는 기존 포지션 {state.missing_stops_count}개"})

        if position_value > self._cash_balance * 0.9:
            violations.append({"rule": "LOW_CASH", "severity": "SOFT",
                                "message": "거래 후 가용 현금 90% 이상 소진"})

        hard = [v for v in violations if v["severity"] == "HARD"]
        approved = len(hard) == 0 and shares >= 1

        if approved:
            rationale = (f"승인: {shares}주 @ {entry_price}, 리스크 {risk_per_trade:,.0f}원 | "
                         f"거래 후 오픈 리스크 {new_open_pct:.1f}%")
        else:
            rules = ", ".join(v["rule"] for v in hard)
            rationale = f"거부: {rules}"

        soft = [v for v in violations if v["severity"] == "SOFT"]
        if soft:
            rationale += " | 경고: " + "; ".join(v["message"] for v in soft)

        return TradeRiskAssessment(
            ticker              = ticker,
            approved            = approved,
            recommended_shares  = float(shares),
            risk_per_trade      = risk_per_trade,
            risk_per_share      = risk_per_share,
            entry_price         = entry_price,
            stop_price          = stop_price,
            stop_distance_pct   = stop_dist_pct,
            position_value      = position_value,
            open_risk_after     = new_open_risk,
            open_risk_pct_after = new_open_pct,
            violations          = violations,
            rationale           = rationale,
        )

    # ── 대시보드용 페이로드 ───────────────────────────────────────────────────

    def get_dashboard_payload(self) -> Dict[str, Any]:
        """대시보드에 표시할 포트폴리오 요약."""
        state = self.get_state()
        return {
            "summary": {
                "equity":             state.equity,
                "cash_balance":       state.cash_balance,
                "total_market_value": round(state.total_market_value, 2),
                "total_open_risk":    round(state.total_open_risk, 2),
                "open_risk_pct":      state.open_risk_pct,
                "open_positions":     len(state.open_positions),
                "missing_stops":      state.missing_stops_count,
                "risk_level":         state.risk_level,
            },
            "concentration":  state.concentration,
            "r_multiples":    state.r_multiple_history[:10],
            "expectancy":     state.expectancy,
            "positions":      [p.to_dict() for p in state.open_positions],
            "generated_at":   state.generated_at,
        }

    # ── 직렬화 ────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity":            self._equity,
            "cash_balance":      self._cash_balance,
            "risk_pct_per_trade":self._risk_pct_per_trade,
            "positions":         [p.to_dict() for p in self._positions],
            "closed_positions":  [p.to_dict() for p in self._closed_positions[-50:]],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioManager":
        mgr = cls(
            equity            = data.get("equity", 10_000_000),
            cash_balance      = data.get("cash_balance", 10_000_000),
            risk_pct_per_trade= data.get("risk_pct_per_trade", 1.0),
        )
        for pd_ in data.get("positions", []):
            pos = PortfolioPosition(**{k: pd_[k] for k in (
                "ticker","name","entry_price","stop_price","shares",
                "entry_date","sector","sleeve","current_price","status",
                "close_price","close_date","notes",
            ) if k in pd_})
            mgr._positions.append(pos)
        for pd_ in data.get("closed_positions", []):
            pos = PortfolioPosition(**{k: pd_[k] for k in (
                "ticker","name","entry_price","stop_price","shares",
                "entry_date","sector","sleeve","current_price","status",
                "close_price","close_date","notes",
            ) if k in pd_})
            mgr._closed_positions.append(pos)
        return mgr
