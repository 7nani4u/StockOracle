# 백테스트 기반 예측 로직을 Vercel 서버리스 스크립트에 적용하는 방법

이 문서는 `StockOracle-main/api/index.py`에 앞서 설계한 **1년 가정 기반 백테스트 분포 로직**을 실제로 적용하는 절차를 설명합니다. 현재 프로젝트는 단일 서버리스 파일 `api/index.py` 안에 백엔드 계산 함수와 프론트엔드 렌더링 스크립트가 함께 들어 있으므로, 별도 API 라우트를 새로 만들기보다 기존 `calc_buy_price()`, `calc_risk()`, `renderForecast()`를 수정하는 방식이 가장 안전합니다.

> 적용 원칙은 명확합니다. 백엔드에서는 기존 단일 매수 구간을 **A/B/C 확률 밴드**로 바꾸고, 리스크 시나리오는 기존 보수적·중립적·공격적 구조를 유지하되 각 전략에 **TP1/TP2/TP3 확률·평균 도달 기간·실패 손실폭**을 추가합니다. 프론트엔드에서는 `conservative` 매수 카드를 제거하고, `bands` 배열과 `tp_levels` 배열을 반복 렌더링하도록 바꿉니다.

## 1. 수정해야 하는 파일과 위치

| 수정 위치 | 현재 역할 | 적용 내용 |
|---|---|---|
| `api/index.py` 상단 유틸 함수 영역 | 전역 상수·공통 계산 함수 | `BACKTEST_PRIOR`, `RISK_MODEL`, 시장 파라미터, 확률 계산 함수를 추가 |
| `calc_buy_price()` | 공격적·추천·보수적 단일 매수 범위 산출 | `conservative` 제거, `aggressive.bands`, `recommended.bands` 반환 |
| `calc_risk()` | 전략별 단일 목표가·손절가 범위 산출 | 기존 전략명 유지, `tp_levels` 배열 추가 또는 반환 구조 전면 교체 |
| `renderForecast()` | 예측 탭 HTML 렌더링 | `bp.conservative` 참조 제거, `bands`와 `tp_levels` 반복 출력 |
| 로컬 실행·Vercel 배포 | 동작 검증 | `requirements.txt` 추가 변경 없이 순수 Python 로직만 반영 |

## 2. 1단계: 백테스트 사전분포 상수 추가

`api/index.py`에서 `calc_risk()`보다 위쪽, 예를 들어 `calc_risk()` 직전 또는 지표 계산 유틸 함수 아래에 아래 상수를 추가합니다. 이 상수는 시뮬레이션 결과를 실제 예측 로직에 넣는 **사전확률 테이블**입니다.

```python
BACKTEST_PRIOR = {
    "KRX": {
        "avg_atr_pct": 2.37,
        "band_width": 1.18,
        "prob_adj": -0.06,
        "target_adj": 0.95,
        "zones": {
            "A": {"depth": [0.25, 0.65], "win": 0.495, "loss": 0.505, "exp_ret": 1.94, "sharpe": 3.36, "hold": 15.1, "fail_loss": -1.61},
            "B": {"depth": [0.65, 1.10], "win": 0.604, "loss": 0.396, "exp_ret": 2.74, "sharpe": 5.68, "hold": 13.0, "fail_loss": -1.11},
            "C": {"depth": [1.10, 1.70], "win": 0.743, "loss": 0.257, "exp_ret": 3.72, "sharpe": 11.21, "hold": 9.2, "fail_loss": -0.79},
        },
    },
    "US": {
        "avg_atr_pct": 1.83,
        "band_width": 0.92,
        "prob_adj": 0.05,
        "target_adj": 1.05,
        "zones": {
            "A": {"depth": [0.25, 0.65], "win": 0.594, "loss": 0.406, "exp_ret": 1.49, "sharpe": 3.07, "hold": 14.0, "fail_loss": -1.67},
            "B": {"depth": [0.65, 1.10], "win": 0.663, "loss": 0.337, "exp_ret": 2.00, "sharpe": 4.86, "hold": 12.3, "fail_loss": -1.22},
            "C": {"depth": [1.10, 1.70], "win": 0.752, "loss": 0.248, "exp_ret": 2.70, "sharpe": 9.18, "hold": 9.0, "fail_loss": -0.68},
        },
    },
}

RISK_MODEL = {
    "conservative": {"label": "보수적", "icon": "🛡️", "stop": 0.85, "tp": [1.10, 1.65, 2.20]},
    "balanced": {"label": "중립적", "icon": "⚖️", "stop": 1.20, "tp": [1.50, 2.30, 3.10]},
    "aggressive": {"label": "공격적", "icon": "🚀", "stop": 1.55, "tp": [2.00, 3.20, 4.60]},
}
```

## 3. 2단계: 확률 계산 헬퍼 함수 추가

동일한 위치에 아래 헬퍼 함수를 추가합니다. 이 함수들은 새로운 외부 라이브러리를 요구하지 않으므로 Vercel 배포 안정성을 해치지 않습니다. 핵심은 **백테스트 승률을 기본값으로 사용하되, ATR 변동성·추세·유사 패턴·부채비율**로 보정하는 것입니다.

```python
def _market_key(market: str) -> str:
    return "US" if market == "US" else "KRX"


def _round_price(v: float, market: str):
    return round(float(v), 4 if market == "US" else 2)


def _calc_trend_score(dd: Dict) -> float:
    def _last(k, default=None):
        a = dd.get(k, []) if dd else []
        return float(a[-1]) if a and a[-1] is not None else default

    close = _last("Close", 0) or 0
    ma20 = _last("MA20", close) or close
    ma60 = _last("MA60", close) or close
    rsi = _last("RSI", 50) or 50
    macd = _last("MACD", 0) or 0
    sig = _last("Signal_Line", 0) or 0
    adx = _last("ADX", 20) or 20

    score = 0.0
    score += 0.25 if close > ma20 else -0.15
    score += 0.25 if close > ma60 else -0.15
    score += 0.20 if macd > sig else -0.10
    score += 0.15 if 45 <= rsi <= 65 else (-0.20 if rsi > 75 else 0.05 if rsi < 35 else 0.0)
    score += 0.15 if adx >= 25 else 0.0
    return max(-1.0, min(1.0, score))


def _similar_pattern_winrate(dd: Dict, atr_pct: float, horizon: int = 20) -> Tuple[float, int]:
    closes = [float(x) for x in dd.get("Close", []) if x is not None] if dd else []
    atrs = [float(x) for x in dd.get("ATR", []) if x is not None] if dd else []
    rsis = [float(x) for x in dd.get("RSI", []) if x is not None] if dd else []
    if len(closes) < horizon + 40 or len(atrs) < horizon + 40 or len(rsis) < horizon + 40:
        return 0.55, 0

    cur_rsi = rsis[-1]
    wins = []
    for i in range(20, len(closes) - horizon):
        past_atr_pct = atrs[i] / closes[i] * 100 if closes[i] else 0
        rsi_match = abs(rsis[i] - cur_rsi) <= 8
        atr_match = abs(past_atr_pct - atr_pct) <= 0.7
        if rsi_match and atr_match:
            future_ret = max(closes[i+1:i+1+horizon]) / closes[i] - 1
            wins.append(future_ret >= max(0.012, atr_pct / 100 * 0.8))
    if not wins:
        return 0.55, 0
    return float(np.mean(wins)), len(wins)


def _probability(base_prob: float, atr_pct: float, market_avg_atr: float, trend_score: float, pattern_win: float, pattern_n: int, market: str, debt_ratio=None) -> float:
    mk = _market_key(market)
    prior = BACKTEST_PRIOR[mk]
    vol_ratio = atr_pct / max(market_avg_atr, 0.01)
    vol_adj = -0.08 * max(0.0, vol_ratio - 1.0) + 0.03 * max(0.0, 1.0 - vol_ratio)
    trend_adj = 0.10 * trend_score
    sample_weight = min(pattern_n / 12.0, 1.0)
    pattern_adj = (pattern_win - base_prob) * 0.35 * sample_weight

    debt_penalty = 0.0
    if debt_ratio is not None:
        if debt_ratio > 300:
            debt_penalty = -0.12
        elif debt_ratio > 200:
            debt_penalty = -0.07
        elif debt_ratio > 150:
            debt_penalty = -0.03

    p = base_prob + prior["prob_adj"] + vol_adj + trend_adj + pattern_adj + debt_penalty
    return round(max(0.05, min(0.92, p)) * 100, 1)
```

## 4. 3단계: `calc_buy_price()` 반환 구조 교체

현재 `calc_buy_price()`는 `aggressive`, `recommended`, `conservative`를 각각 단일 `range`로 반환합니다. 적용 후에는 **`conservative` 키를 반환하지 않고**, `aggressive.bands`, `recommended.bands` 배열을 반환해야 합니다.

기존 함수 전체를 한 번에 갈아엎기보다, 현재 함수 내부의 지표 추출·앵커 계산 코드는 유지하고, 마지막 반환부 직전부터 아래 방식으로 교체하는 것이 안전합니다.

```python
def _build_band_payload(zone_name, z, low, high, prob, basis, market, cur):
    return {
        "band": zone_name,
        "price_range": [_round_price(low, market), _round_price(high, market)],
        "pct": [round((low - cur) / cur * 100, 2), round((high - cur) / cur * 100, 2)],
        "basis": basis,
        "expected_return_pct": z["exp_ret"],
        "win_probability_pct": prob,
        "loss_probability_pct": round(100 - prob, 1),
        "avg_holding_days": z["hold"],
        "failed_avg_loss_pct": z["fail_loss"],
        "distribution_basis": f"{z['depth'][0]}~{z['depth'][1]}ATR 진입 깊이의 20거래일 목표 도달 분포 기반",
    }
```

그다음 `calc_buy_price()` 내부에서 `return { ... }` 직전에 아래 로직을 구성합니다.

```python
mk = _market_key(market)
prior = BACKTEST_PRIOR[mk]
w = prior["band_width"]
trend_score = _calc_trend_score(dd)
pattern_win, pattern_n = _similar_pattern_winrate(dd, atr_pct)

anchor_base = float(np.mean([x for x in [bb_m * 0.995, ma20 * 0.995, vwap_approx or None, fib_382 if fib_382 < last_price else None] if x]))
rec_anchors = {
    "A": anchor_base,
    "B": float(np.mean([anchor_base, fib_382])) if fib_382 < last_price else anchor_base - atr * 0.35,
    "C": float(np.mean([strong_support, support_zone])) if strong_support else anchor_base - atr * 0.85,
}

aggressive_bands = []
recommended_bands = []
for zone_name, z in prior["zones"].items():
    d1, d2 = z["depth"]
    prob = _probability(z["win"], atr_pct, prior["avg_atr_pct"], trend_score, pattern_win, pattern_n, market)

    agg_low = last_price - d2 * atr * w
    agg_high = last_price - d1 * atr * w
    aggressive_bands.append(_build_band_payload(
        zone_name, z, agg_low, agg_high, prob,
        [f"ATR {d1:.2f}~{d2:.2f}배 눌림", f"추세 점수 {trend_score:.2f}", f"유사 패턴 {pattern_n}회 / 승률 {pattern_win*100:.1f}%"],
        market, last_price
    ))

    anchor = rec_anchors.get(zone_name, anchor_base)
    rec_low = anchor - d2 * atr * w * 0.75
    rec_high = anchor - d1 * atr * w * 0.45
    recommended_bands.append(_build_band_payload(
        zone_name, z, rec_low, rec_high, prob,
        ["볼린저·MA20·VWAP·피보나치 앵커 기반", f"기대 Sharpe {z['sharpe']}", f"평균 보유 {z['hold']}일"],
        market, last_price
    ))
```

최종 반환부는 아래 형태로 맞춥니다. 기존 프론트가 `range`를 참조하므로, 프론트 수정 전의 임시 호환성을 위해 `range`에 첫 번째 밴드 또는 B밴드 값을 넣어 둘 수도 있습니다. 다만 최종적으로는 프론트에서 `bands`를 읽게 해야 합니다.

```python
return {
    "current": r(last_price),
    "aggressive": {
        "bands": aggressive_bands,
        "range": aggressive_bands[1]["price_range"],
        "basis": aggressive_bands[1]["basis"],
        "interpretation": "공격적 매수는 A/B/C 확률 밴드로 분할하며, A는 강한 추세 확인 시에만 사용합니다.",
    },
    "recommended": {
        "bands": recommended_bands,
        "range": recommended_bands[1]["price_range"],
        "basis": recommended_bands[1]["basis"],
        "interpretation": "추천 매수는 백테스트 승률·Sharpe·평균 보유 기간 기준의 분할 진입 구간입니다.",
    },
    "timing": {"buy": buy_timing_str, "sell": sell_timing_str},
    "support_zone": r(support_zone),
    "fib": {"h60": r(h60), "l60": r(l60), "f382": r(fib_382), "f500": r(fib_500), "f618": r(fib_618)},
    "rsi": round(rsi, 1),
    "rsi_context": rsi_ctx,
    "atr": r(atr),
    "atr_pct": round(atr_pct, 2),
    "vol_trend": vol_trend,
    "probability_note": "백테스트 수익률 분포, ATR 변동성, 추세 강도, 과거 유사 패턴 빈도를 합성해 산출",
}
```

## 5. 4단계: `calc_risk()`에 TP1/TP2/TP3 추가

현재 `calc_risk()`는 `target`, `stop`, `return`, `rr_ratio`를 단일 범위로 반환합니다. 화면 호환성을 위해 기존 키를 유지하면서 `tp_levels`만 추가하는 방식이 가장 안전합니다.

`calc_risk()`의 `return { ... }` 직전에서 아래 보조 함수를 정의합니다.

```python
def _base_tp_probability(market, strategy, tp_idx):
    table = {
        "KRX": {
            "conservative": [0.62, 0.48, 0.34],
            "balanced": [0.56, 0.42, 0.29],
            "aggressive": [0.49, 0.35, 0.22],
        },
        "US": {
            "conservative": [0.71, 0.58, 0.45],
            "balanced": [0.66, 0.53, 0.40],
            "aggressive": [0.61, 0.48, 0.35],
        },
    }
    return table[_market_key(market)][strategy][tp_idx - 1]


def _days_to_hit(tp_idx, market, strategy):
    days = {
        "KRX": {"conservative": [7, 13, 20], "balanced": [9, 16, 24], "aggressive": [11, 20, 32]},
        "US": {"conservative": [6, 11, 17], "balanced": [7, 14, 21], "aggressive": [9, 17, 27]},
    }
    return days[_market_key(market)][strategy][tp_idx - 1]


def _failed_loss_pct(stop_mul, atr_pct, market, tp_idx):
    market_extra = 1.08 if market != "US" else 0.96
    return round(-stop_mul * atr_pct * market_extra * (1 + 0.05 * (tp_idx - 1)), 2)


def _tp_levels(strategy_key, entry_price, stop_price, stop_mul, tp_muls):
    prior = BACKTEST_PRIOR[_market_key(market)]
    target_adj = prior["target_adj"]
    trend_score = _calc_trend_score(dd or {})
    pattern_win, pattern_n = _similar_pattern_winrate(dd or {}, atr_pct)
    risk_amt = max(entry_price - stop_price, 1e-9)
    levels = []
    prev_prob = 100.0
    for i, mul in enumerate(tp_muls, start=1):
        tp_price = entry_price + atr * mul * target_adj
        base = _base_tp_probability(market, strategy_key, i)
        prob = _probability(base, atr_pct, prior["avg_atr_pct"], trend_score, pattern_win, pattern_n, market)
        if i > 1:
            prob = min(prob, prev_prob - 3.0)
        prev_prob = prob
        levels.append({
            "level": f"TP{i}",
            "price": _round_price(tp_price, market),
            "return_pct": round((tp_price / entry_price - 1) * 100, 2),
            "hit_probability_pct": round(prob, 1),
            "avg_days_to_hit": _days_to_hit(i, market, strategy_key),
            "failed_avg_loss_pct": _failed_loss_pct(stop_mul, atr_pct, market, i),
            "risk_reward": round((tp_price - entry_price) / risk_amt, 2),
        })
    return levels
```

그 후 기존 반환 딕셔너리의 각 전략에 `tp_levels`를 추가합니다. 예를 들어 보수적 전략은 다음과 같이 바꿉니다.

```python
"conservative": {
    "label": "보수적",
    "icon": "🛡️",
    "desc": f"리스크 최소화 · 손절 {abs(cons_stp_pct):.2f}%",
    "target": [r(cons_tgt_range[0]), r(cons_tgt_range[1])],
    "stop": [r(cons_stp_range[0]), r(cons_stp_range[1])],
    "return": cons_ret,
    "rr_ratio": cons_rr,
    "stop_pct": cons_stp_pct,
    "atr_mul_tgt": cons_tgt_mul,
    "atr_mul_stp": cons_stp_mul,
    "tp_levels": _tp_levels("conservative", price, (cons_stp_range[0] + cons_stp_range[1]) / 2, cons_stp_mul, RISK_MODEL["conservative"]["tp"]),
    "interpretation": f"BB 하단 참조 손절 · TP1/TP2/TP3 확률 기반 분할 익절",
}
```

동일하게 `balanced`에는 `_tp_levels("balanced", ...)`, `aggressive`에는 `_tp_levels("aggressive", ...)`를 추가합니다. 이 방식은 기존 프론트엔드가 `target`, `stop`, `return`, `rr_ratio`를 계속 읽을 수 있으므로 점진 적용에 유리합니다.

## 6. 5단계: `renderForecast()` 매수 카드 수정

현재 `renderForecast()`는 `bp.aggressive.range`, `bp.recommended.range`, `bp.conservative.range`를 직접 참조합니다. 적용 후에는 `bp.conservative`를 제거하고, 각 영역의 `bands` 배열을 반복 출력해야 합니다.

기존 코드에서 아래 세 줄은 삭제합니다.

```javascript
const aggR = bp.aggressive.range;
const recR = bp.recommended.range;
const conR = bp.conservative.range;
```

그리고 기존 `buyZone()` 함수를 아래처럼 바꿉니다.

```javascript
const buyZone = (zone, color, label) => {
  const z = bp[zone];
  if (!z || !z.bands) return '';
  return `
    <div class="buy-card ${zone}" style="display:flex;flex-direction:column;gap:8px">
      <div class="buy-label">${label}</div>
      ${z.bands.map(b => `
        <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:9px;margin-bottom:6px">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
            <b style="color:${color}">밴드 ${b.band}</b>
            <span style="font-size:11px;color:#8b949e">승률 ${b.win_probability_pct}% · 손실 ${b.loss_probability_pct}%</span>
          </div>
          <div style="font-size:15px;font-weight:800;color:${color};margin-top:4px">
            ${fmt(b.price_range[0], isKrx)} ~ ${fmt(b.price_range[1], isKrx)}
          </div>
          <div style="font-size:11px;color:#8b949e;margin-top:3px">
            기대수익 ${b.expected_return_pct}% · 평균보유 ${b.avg_holding_days}일 · 실패손실 ${b.failed_avg_loss_pct}%
          </div>
          <div style="font-size:11px;color:#cdd9e5;line-height:1.45;margin-top:6px">
            ${(b.basis || []).map(x => `• ${x}`).join('<br>')}
          </div>
        </div>
      `).join('')}
      <div style="font-size:11px;color:#8b949e;line-height:1.5;border-top:1px solid #21262d;padding-top:6px">
        ${z.interpretation || ''}
      </div>
    </div>`;
};
```

마지막으로 매수 카드 출력부를 아래처럼 수정합니다.

```javascript
<div class="buy-price-grid">
  ${buyZone('aggressive',  '#f97316', '공격적 매수')}
  ${buyZone('recommended', '#3fb950', '추천 매수 구간')}
</div>
```

## 7. 6단계: `renderForecast()` 리스크 카드 수정

기존 리스크 카드의 `목표가` 한 줄은 유지하되, 그 아래에 `tp_levels`를 반복 표시하면 됩니다. 기존 `목표가`, `손절가`, `R/R` 영역 아래에 다음 블록을 추가합니다.

```javascript
${sc.tp_levels ? `
  <div style="margin-top:8px;border-top:1px solid #21262d;padding-top:8px">
    ${sc.tp_levels.map(tp => `
      <div style="background:#0d1117;border:1px solid #30363d;border-radius:7px;padding:7px;margin-bottom:6px">
        <div style="display:flex;justify-content:space-between;gap:6px">
          <b style="color:#3fb950">${tp.level} ${fmt(tp.price, isKrx)}</b>
          <span style="font-size:11px;color:${rrColor(tp.risk_reward)}">R/R ${tp.risk_reward}:1</span>
        </div>
        <div style="font-size:11px;color:#8b949e;margin-top:3px">
          도달확률 ${tp.hit_probability_pct}% · 평균 ${tp.avg_days_to_hit}일 · 실패손실 ${tp.failed_avg_loss_pct}% · 수익률 +${tp.return_pct}%
        </div>
      </div>
    `).join('')}
  </div>` : ''}
```

이렇게 하면 기존 보수적·중립적·공격적 리스크 카드는 유지하면서도, 각 카드 안에서 TP1/TP2/TP3가 확률형으로 표시됩니다.

## 8. 7단계: API 반환 데이터 확인

수정 후에는 `/api/stock` 응답에 다음 구조가 나와야 합니다. 핵심은 `buy_price.conservative`가 없어지고, `risk_scenarios.*.tp_levels`가 생기는 것입니다.

```json
{
  "buy_price": {
    "current": 100.0,
    "aggressive": {
      "bands": [
        {"band": "A", "price_range": [98.8, 99.5], "win_probability_pct": 52.1, "loss_probability_pct": 47.9},
        {"band": "B", "price_range": [97.9, 98.8], "win_probability_pct": 62.3, "loss_probability_pct": 37.7},
        {"band": "C", "price_range": [96.6, 97.9], "win_probability_pct": 73.8, "loss_probability_pct": 26.2}
      ]
    },
    "recommended": {"bands": []}
  },
  "risk_scenarios": {
    "conservative": {
      "tp_levels": [
        {"level": "TP1", "hit_probability_pct": 62.0, "avg_days_to_hit": 7, "failed_avg_loss_pct": -2.1},
        {"level": "TP2", "hit_probability_pct": 48.0, "avg_days_to_hit": 13, "failed_avg_loss_pct": -2.2},
        {"level": "TP3", "hit_probability_pct": 34.0, "avg_days_to_hit": 20, "failed_avg_loss_pct": -2.3}
      ]
    }
  }
}
```

## 9. 로컬 검증 절차

먼저 `api/index.py`만 수정하고, 새로운 파일을 만들지 않는 방식으로 시작하는 것을 권장합니다. 현재 Vercel 프로젝트는 단일 서버리스 파일 중심 구조이므로, helper 모듈을 따로 만들면 Vercel 번들 포함 여부를 추가로 점검해야 합니다.

| 검증 단계 | 명령 또는 확인 항목 | 합격 기준 |
|---|---|---|
| 문법 검사 | `python3.11 -m py_compile api/index.py` | 오류 없음 |
| 로컬 서버 실행 | `python3.11 dev_server.py` | `http://localhost:3000` 접속 가능 |
| API 응답 확인 | `/api/stock?symbol=005930` 또는 UI 검색 | `buy_price.aggressive.bands` 존재 |
| 한국 종목 확인 | 삼성전자 등 | 밴드 폭 1.18, 확률 -6%p 보수화 반영 |
| 미국 종목 확인 | AAPL, NVDA 등 | 밴드 폭 0.92, 확률 +5%p 및 TP 거리 1.05 반영 |
| 프론트 확인 | 예측 탭 | 보수적 매수 카드 미표시, TP1/TP2/TP3 표시 |

## 10. Vercel 배포 시 주의점

이번 적용은 `numpy`, `pandas` 등 이미 프로젝트가 사용하는 패키지만 활용하므로 `requirements.txt`를 바꾸지 않는 것이 좋습니다. Vercel 서버리스 함수는 배포 크기와 실행 시간에 민감하므로, 실제 과거 데이터 전체 재백테스트를 요청 시마다 수행하지 말고, 위의 `BACKTEST_PRIOR`처럼 **사전 계산된 분포값을 상수로 보관**해야 합니다.

| 주의 항목 | 권장 방식 | 피해야 할 방식 |
|---|---|---|
| 백테스트 분포 | 상수 테이블로 내장 | 요청마다 1년치 전체 재시뮬레이션 실행 |
| 확률 계산 | 현재 종목 지표로 경량 보정 | 머신러닝 패키지 추가 설치 |
| 파일 구조 | 처음에는 `api/index.py` 안에 통합 | 별도 모듈 생성 후 Vercel 번들 누락 |
| UI 호환성 | 기존 키 유지 + 새 키 추가 | 기존 키 즉시 삭제 후 화면 오류 유발 |
| 배포 | 로컬 API 응답 확인 후 Vercel 재배포 | 프론트만 수정하고 API 구조 미확인 |

## 11. 적용 순서 요약

| 순서 | 작업 | 성공 기준 |
|---:|---|---|
| 1 | `BACKTEST_PRIOR`, `RISK_MODEL` 추가 | 시장별 ATR·승률·Sharpe 사전분포가 코드에 존재 |
| 2 | `_probability()`, `_calc_trend_score()`, `_similar_pattern_winrate()` 추가 | 확률이 감이 아니라 데이터·ATR·추세·패턴 기반으로 계산 |
| 3 | `calc_buy_price()` 반환부 변경 | `aggressive.bands`, `recommended.bands`가 A/B/C로 출력 |
| 4 | `conservative` 매수 제거 | 예측 탭에서 보수적 매수 카드 삭제 |
| 5 | `calc_risk()`에 `tp_levels` 추가 | TP별 도달 확률·평균 기간·실패 손실폭 표시 |
| 6 | `renderForecast()` 수정 | 밴드와 TP 레벨이 반복 렌더링 |
| 7 | 로컬 테스트 후 배포 | 한국·미국 분기와 UI가 정상 동작 |

## 12. 실전 적용 판단 기준

적용 후 예측 탭은 가격 하나를 추천하는 화면이 아니라, **구간별 승률과 손실 확률을 비교하는 의사결정 화면**으로 바뀝니다. 한국 종목은 변동성·수급 충격을 반영해 밴드가 넓고 확률이 보수적으로 표시되어야 하며, 미국 종목은 추세 지속성과 모멘텀 가중으로 TP 확률이 상대적으로 높게 표시되어야 합니다.

실전에서는 B밴드를 기본 진입 후보로 삼고, C밴드는 과매도 반등형 분할 매수, A밴드는 ADX 25 이상·MACD 양전환·MA20 지지 확인 시에만 사용하도록 운영하는 것이 가장 현실적입니다. TP는 TP1에서 일부 익절, TP2에서 절반 이상 익절, TP3는 추세가 유지될 때만 잔여 물량으로 대응하는 구조가 적합합니다.

마지막으로, 부채비율이 150%를 넘는 기업은 기술적 승률이 좋더라도 확률을 낮춰 표시하는 것이 바람직합니다. 특히 200% 초과 기업은 최소 -7%p, 300% 초과 기업은 -12%p를 차감해 레버리지 리스크를 예측 탭에 직접 반영해야 합니다.
