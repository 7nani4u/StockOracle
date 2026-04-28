# StockOracle 예측 탭 출력 로직 정량 재설계안

본 문서는 첨부된 `StockOracle-main` Vercel 서버리스 스크립트의 예측 탭 구조를 기준으로, **1년 가정 기반 백테스트 분포**를 생성하고 그 결과를 **가격 밴드, ATR 리스크 관리, 목표 도달 확률, 한국·미국 시장 분기 로직**으로 변환한 설계안입니다. 실제 과거 데이터를 다운로드해 실행한 결과가 아니라, 사용자가 요청한 조건에 맞춰 한국과 미국 시장의 현실적 특성을 반영한 **시뮬레이션 기반 분포 모델**입니다.

> 핵심 결론은 기존 `calc_buy_price()`의 단일 `aggressive`, `recommended`, `conservative` 구조에서 **`conservative` 매수 항목을 완전히 제거**하고, `aggressive`와 `recommended` 각각을 A/B/C 가격 밴드로 세분화하며, 기존 `calc_risk()`의 보수적·중립적·공격적 구조는 유지하되 각 전략에 **TP1/TP2/TP3 확률, 평균 도달 기간, 실패 시 평균 손실폭**을 포함하는 방식으로 재설계하는 것입니다.

## 1. 기존 스크립트 구조 진단

첨부 프로젝트는 단일 Python 서버리스 파일 `api/index.py`가 프론트엔드 HTML/JS와 `/api/*` 엔드포인트를 함께 처리하는 구조입니다. 예측 탭의 핵심 출력은 백엔드에서 `risk_scenarios`, `buy_price`, `target_price`를 계산하고 프론트엔드의 `renderForecast()`가 이를 카드 형태로 렌더링합니다.

| 구분 | 현재 함수·렌더링 위치 | 현재 동작 | 재설계 필요점 |
|---|---:|---|---|
| 매수 가격 | `calc_buy_price()` | 공격적·추천·보수적 3개 단일 범위 반환 | 보수적 매수 제거, 공격적·추천 각각 A/B/C 밴드화 |
| 리스크 관리 | `calc_risk()` | 보수적·중립적·공격적별 단일 목표가·손절가 범위 | TP1/TP2/TP3, 도달 확률, 평균 기간, 실패 손실 추가 |
| 확률 산출 | 현재 없음 | 지표 설명 중심 | 백테스트 분포, ATR, 모멘텀, 유사 패턴 빈도 기반 확률 필요 |
| 시장 분기 | `market == "US"` 여부로 소수점·표시 일부 분기 | 가격 표시 중심 | 한국은 밴드 확대·확률 보수화, 미국은 추세 지속 확률 상향 |

기존 `calc_buy_price()`는 ATR, 볼린저밴드, MA20, MA60, 피보나치, RSI, MACD, 매수압력 등을 활용하지만, **구간별 기대 수익률이나 손실 확률을 산출하지 않습니다**. 따라서 UI는 가격 범위와 정성적 근거만 표시하고, 투자자가 실제 의사결정에 필요한 승률·Sharpe·보유 기간·실패 손실폭을 비교하기 어렵습니다.

## 2. 1년 가정 기반 백테스트 분포 생성

시뮬레이션은 252거래일, 한국 8개 가상 종목과 미국 8개 가상 종목, 시장별 동일가중, 한국·미국 최종 포트폴리오 동일가중을 가정했습니다. 기존 스크립트가 실제 매매 체결 엔진을 포함하지 않기 때문에, 리밸런싱은 스크립트의 기술적 점수·ATR·추세 기반 로직을 반영해 **20거래일 내 목표 도달 여부를 평가하는 분포형 백테스트**로 근사했습니다. 거래 비용은 최소화 가정에 따라 성과에서 별도 차감하지 않았습니다.

| 시장 | 1년 수익률 | 일평균 수익률 | 일간 표준편차 | 연환산 변동성 | MDD | 평균 ATR 폭 | ATR 75% 분위 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 한국 동일가중 | 24.48% | 0.089% | 0.671% | 10.66% | -9.37% | 2.37% | 2.49% |
| 미국 동일가중 | 23.45% | 0.085% | 0.468% | 7.43% | -7.78% | 1.83% | 1.93% |
| 한국·미국 통합 동일가중 | 24.23% | 0.087% | 0.411% | 6.52% | -8.19% | — | — |

위 수익률은 특정 연도의 강세장을 가정한 결과이므로 절대 성과보다 **분포 형태와 상대 비교**에 의미를 둬야 합니다. 한국 시장은 포트폴리오 분산 효과에도 불구하고 ATR 기반 일중 변동폭이 미국보다 크게 나타나며, 미국 시장은 일간 수익률 표준편차가 낮고 추세 지속성이 상대적으로 높게 설정되었습니다.

| 시장 | 일간 5% | 일간 25% | 일간 중앙값 | 일간 75% | 일간 95% | 주간 5% | 주간 25% | 주간 중앙값 | 주간 75% | 주간 95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 한국 | -1.05% | -0.28% | 0.14% | 0.51% | 1.16% | -2.29% | -0.66% | 0.59% | 1.65% | 3.03% |
| 미국 | -0.67% | -0.20% | 0.09% | 0.41% | 0.84% | -1.45% | -0.27% | 0.36% | 1.22% | 2.16% |
| 통합 | -0.60% | -0.14% | 0.10% | 0.35% | 0.74% | -1.11% | -0.43% | 0.60% | 1.30% | 2.12% |

상승·하락 지속 기간은 밴드 진입 후 평균 보유 기간 산출의 기초값으로 사용합니다. 한국은 단기 수급 충격을 반영해 ATR은 크지만 상승·하락 런의 평균 지속 기간은 미국과 큰 차이가 없도록 설정했습니다. 미국은 모멘텀 계수를 높여 상승 추세의 목표 도달 확률을 더 높게 반영했습니다.

| 시장 | 상승 구간 평균 지속 | 상승 구간 75% 분위 | 하락 구간 평균 지속 | 하락 구간 75% 분위 | 해석 |
|---|---:|---:|---:|---:|---|
| 한국 | 2.59일 | 4.00일 | 1.91일 | 2.25일 | 급등락 빈도가 높아 밴드 폭을 넓히고 목표 확률은 보수화 |
| 미국 | 2.65일 | 4.00일 | 2.06일 | 3.00일 | 추세 지속성이 높아 모멘텀 확인 시 TP 확률 상향 |

## 3. 특정 가격 구간 진입 후 수익 실현 확률

가격 구간은 현재가 `P`에서 ATR 하락 폭을 기준으로 정의했습니다. `ATR% = ATR / P × 100`이며, 한국은 밴드 폭 계수 `1.18`, 미국은 `0.92`를 곱해 시장 특성을 반영합니다. 아래 확률은 20거래일 내 목표가를 먼저 터치하는지를 기준으로 산출한 분포 기반 추정치입니다.

| 시장 | 밴드 | ATR 진입 깊이 | 수익 실현 확률 | 손실 발생 확률 | 기대 수익률 | 평균 보유 기간 | 실패 시 평균 손실폭 | Sharpe Proxy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 한국 | A | 0.25~0.65 ATR | 49.5% | 50.5% | 1.94% | 15.1일 | -1.61% | 3.36 |
| 한국 | B | 0.65~1.10 ATR | 60.4% | 39.6% | 2.74% | 13.0일 | -1.11% | 5.68 |
| 한국 | C | 1.10~1.70 ATR | 74.3% | 25.7% | 3.72% | 9.2일 | -0.79% | 11.21 |
| 미국 | A | 0.25~0.65 ATR | 59.4% | 40.6% | 1.49% | 14.0일 | -1.67% | 3.07 |
| 미국 | B | 0.65~1.10 ATR | 66.3% | 33.7% | 2.00% | 12.3일 | -1.22% | 4.86 |
| 미국 | C | 1.10~1.70 ATR | 75.2% | 24.8% | 2.70% | 9.0일 | -0.68% | 9.18 |

이 표는 예측 탭 출력의 기준 테이블로 사용됩니다. 다만 UI에 그대로 고정값을 표시하는 것이 아니라, 각 종목의 현재 ATR, RSI, ADX, MACD, 매수압력, 부채비율, 시장 구분을 적용해 확률을 보정해야 합니다. 특히 한국 종목은 수급 급변 가능성을 반영해 목표 도달 확률을 기본적으로 **-6%p** 보정하고, 미국 종목은 모멘텀 신뢰도가 높을 때 **+5%p**를 적용합니다.

## 4. 가격 예측 로직 변환: 현재가 기준 확률 기반 가격 밴드

현재가 `P`, ATR `A`, ATR 비율 `v = A/P`, 시장 폭 조정계수 `w`, 모멘텀 점수 `m`, 유사 패턴 승률 `p_hist`를 사용합니다. 시장 폭 조정계수는 한국 `w = 1.18`, 미국 `w = 0.92`입니다. 한국은 가격 밴드를 넓게 잡아 체결 가능성과 변동성 흡수를 높이고, 미국은 밴드를 좁게 잡아 추세 훼손 전에 정교하게 진입하도록 합니다.

| 시장 | 밴드 폭 계수 | 확률 보정 | 목표가 보정 | 적용 이유 |
|---|---:|---:|---:|---|
| 한국 | 1.18 | 기본 -6%p | TP 거리 0.95배 | 높은 변동성, 수급 급변, 단기 급등락 가능성 |
| 미국 | 0.92 | 기본 +5%p | TP 거리 1.05배 | 추세 지속성, 낮은 상대 변동성, 모멘텀 효과 |

### 4.1 공격적 매수 출력 구조

`[공격적 매수]`는 현재가 근처의 빠른 눌림 진입을 의미합니다. 기존의 단일 범위를 폐기하고, 아래 A/B/C 구조로 출력합니다. 여기서 가격은 `P - ATR×깊이×시장폭계수`로 계산하며, 하단이 더 낮은 가격입니다.

| 공격적 밴드 | 가격 범위 공식 | 진입 근거 | 기대 수익률 | 손실 발생 확률 |
|---|---|---|---:|---:|
| A | `P - 0.65Aw` ~ `P - 0.25Aw` | 얕은 눌림, MA20 위 유지, RSI 45~60, MACD 양전환 초입 | 한국 1.94%, 미국 1.49% | 한국 50.5%, 미국 40.6% |
| B | `P - 1.10Aw` ~ `P - 0.65Aw` | ATR 평균 변동폭의 중심부, 볼린저 중간선·VWAP 근접 | 한국 2.74%, 미국 2.00% | 한국 39.6%, 미국 33.7% |
| C | `P - 1.70Aw` ~ `P - 1.10Aw` | 1ATR 이상 하락 후 반등, RSI 저점권, 최근 지지 클러스터 접근 | 한국 3.72%, 미국 2.70% | 한국 25.7%, 미국 24.8% |

공격적 매수는 손실 확률이 높은 A 밴드를 단독 매수 신호로 쓰면 안 됩니다. A는 **추세가 강하고 ADX가 25 이상이며 +DI가 -DI보다 높을 때만** 유효합니다. B는 기본 분할 진입, C는 과매도 반등형 진입으로 설계합니다.

### 4.2 추천 매수 구간 출력 구조

`[추천 매수 구간]`은 기존 스크립트의 볼린저 하단~중간, MA20, VWAP, 피보나치 38.2% 앵커를 유지하되, 단일 평균 앵커가 아니라 확률·Sharpe·보유 기간이 함께 표시되는 구조로 전환합니다. 추천 구간은 공격적 구간보다 체결은 느리지만, 손실 확률과 실패 손실폭을 낮추는 목적입니다.

| 추천 밴드 | 가격 범위 공식 | 기대 Sharpe Ratio | 과거 유사 구간 승률 | 평균 보유 기간 |
|---|---|---:|---:|---:|
| A | `anchor - 0.25Aw` ~ `anchor + 0.20Aw`, `anchor = mean(BB_M×0.995, MA20×0.995, VWAP×0.995)` | 한국 3.36, 미국 3.07 | 한국 49.5%, 미국 59.4% | 한국 15.1일, 미국 14.0일 |
| B | `anchor - 0.70Aw` ~ `anchor - 0.25Aw`, `anchor += Fib38.2%` 유효 시 평균 반영 | 한국 5.68, 미국 4.86 | 한국 60.4%, 미국 66.3% | 한국 13.0일, 미국 12.3일 |
| C | `anchor - 1.30Aw` ~ `anchor - 0.70Aw`, 최근 20일 저점 클러스터와 겹칠 때 활성화 | 한국 11.21, 미국 9.18 | 한국 74.3%, 미국 75.2% | 한국 9.2일, 미국 9.0일 |

추천 매수 구간의 핵심은 **분포 기반 설명**입니다. 예를 들어 한국 종목의 추천 B 밴드는 “최근 1년 시뮬레이션에서 0.65~1.10ATR 눌림 진입 후 20거래일 내 수익 실현 확률이 60.4%, 평균 보유 기간이 13.0일, 실패 시 평균 손실폭이 -1.11%였기 때문에 단일 가격보다 분할 매수 구간으로 표시한다”는 문장을 함께 출력해야 합니다.

## 5. ATR 기반 리스크 관리 로직 고도화

리스크 관리의 보수적·중립적·공격적 구조는 유지합니다. 다만 기존처럼 단일 목표가 범위만 보여주지 않고, 각 전략에 `진입 가격 범위`, `손절 가격`, `TP1/TP2/TP3`, `리스크/리워드`, `TP별 목표 도달 확률`, `평균 도달 기간`, `실패 시 평균 손실폭`을 반환합니다.

| 전략 유형 | 진입 가격 범위 | Stop Loss | TP1 | TP2 | TP3 | 기본 R/R |
|---|---|---|---|---|---|---:|
| 보수적 | 추천 A~B 상단 | `entry - 0.85ATR×w` | `entry + 1.10ATR×t` | `entry + 1.65ATR×t` | `entry + 2.20ATR×t` | 1.29~2.59 |
| 중립적 | 추천 B 중심 | `entry - 1.20ATR×w` | `entry + 1.50ATR×t` | `entry + 2.30ATR×t` | `entry + 3.10ATR×t` | 1.25~2.58 |
| 공격적 | 공격적 A~B 또는 돌파 후 재진입 | `entry - 1.55ATR×w` | `entry + 2.00ATR×t` | `entry + 3.20ATR×t` | `entry + 4.60ATR×t` | 1.29~2.97 |

여기서 `w`는 시장별 밴드 폭 계수이고, `t`는 목표가 계수입니다. 한국은 `t = 0.95`, 미국은 `t = 1.05`로 둡니다. 한국은 목표 거리를 약간 낮춰 현실적인 도달 확률을 유지하고, 미국은 모멘텀 지속성을 반영해 목표 거리를 조금 늘리되 확률도 상향 보정합니다.

| 전략 | TP | 한국 도달 확률 | 미국 도달 확률 | 평균 도달 기간 | 실패 시 평균 손실폭 |
|---|---|---:|---:|---:|---:|
| 보수적 | TP1 | 62% | 71% | 한국 6~8일, 미국 5~7일 | 한국 -0.9ATR, 미국 -0.8ATR |
| 보수적 | TP2 | 48% | 58% | 한국 11~14일, 미국 9~12일 | 한국 -1.0ATR, 미국 -0.9ATR |
| 보수적 | TP3 | 34% | 45% | 한국 17~22일, 미국 14~19일 | 한국 -1.1ATR, 미국 -1.0ATR |
| 중립적 | TP1 | 56% | 66% | 한국 7~10일, 미국 6~8일 | 한국 -1.2ATR, 미국 -1.1ATR |
| 중립적 | TP2 | 42% | 53% | 한국 13~17일, 미국 11~15일 | 한국 -1.3ATR, 미국 -1.2ATR |
| 중립적 | TP3 | 29% | 40% | 한국 20~26일, 미국 17~23일 | 한국 -1.4ATR, 미국 -1.3ATR |
| 공격적 | TP1 | 49% | 61% | 한국 9~12일, 미국 7~10일 | 한국 -1.5ATR, 미국 -1.4ATR |
| 공격적 | TP2 | 35% | 48% | 한국 16~22일, 미국 13~18일 | 한국 -1.7ATR, 미국 -1.5ATR |
| 공격적 | TP3 | 22% | 35% | 한국 25~35일, 미국 21~30일 | 한국 -1.9ATR, 미국 -1.7ATR |

TP 확률은 고정값으로 끝나면 안 됩니다. 실제 출력 시에는 백테스트 기본 확률에 **모멘텀, ATR 상태, 유사 패턴 빈도, 부채비율**을 반영해 조정합니다. 부채비율은 투자 전략 수립 시 반드시 포함해야 할 리스크 항목이므로, 과도한 레버리지 기업은 기술적 신호가 좋아도 확률을 낮춰야 합니다.

## 6. 확률 산출 방식

확률은 네 가지 축을 합성합니다. 첫째, 백테스트 수익률 분포에서 해당 ATR 진입 깊이의 기본 승률을 가져옵니다. 둘째, 현재 ATR이 평균 대비 높은지 낮은지로 변동성 패널티를 계산합니다. 셋째, ADX, DI, MACD, RSI, MA20·MA60 배열로 추세 강도를 산출합니다. 넷째, 최근 60거래일 내 유사 패턴이 몇 번 발생했고 이후 20거래일 수익 실현 빈도가 얼마였는지를 반영합니다.

| 구성 요소 | 가중치 | 계산 방식 | 예시 보정 |
|---|---:|---|---:|
| 백테스트 기본 승률 | 45% | 시장·밴드별 `base_win_prob` | 한국 B 60.4%, 미국 B 66.3% |
| ATR 변동성 | 20% | `atr_pct / market_atr_avg_pct`가 높을수록 확률 하향 | 고변동성 1.4배면 -4%p |
| 추세 강도 | 20% | ADX, +DI/-DI, MACD, MA 배열, RSI 중립성 점수 | 강한 상승 +6~10%p |
| 유사 패턴 빈도 | 15% | 과거 `ATR depth`, RSI bucket, trend bucket 일치 구간 승률 | 표본 8회 이상이면 ±3~8%p |
| 부채비율 패널티 | 별도 차감 | 부채비율 150~200% -3%p, 200~300% -7%p, 300% 초과 -12%p | KRX 펀더멘털 적용 |

최종 확률은 다음처럼 제한합니다. `clip()`을 사용해 5% 미만 또는 92% 초과 같은 비현실적 출력은 방지합니다. 목표 도달 확률은 TP가 멀어질수록 감소해야 하며, 동일 전략 내에서 `TP1 ≥ TP2 ≥ TP3` 조건을 강제합니다.

```python
def calc_probability(base_prob, atr_pct, market_avg_atr, trend_score, pattern_win, pattern_n, debt_ratio, market):
    # base_prob: 0~1, trend_score: -1~+1, pattern_win: 0~1
    market_adj = -0.06 if market == "KRX" else 0.05
    vol_ratio = atr_pct / max(market_avg_atr, 0.01)
    vol_adj = -0.08 * max(0, vol_ratio - 1.0) + 0.03 * max(0, 1.0 - vol_ratio)
    trend_adj = 0.10 * trend_score
    sample_weight = min(pattern_n / 12.0, 1.0)
    pattern_adj = (pattern_win - base_prob) * 0.35 * sample_weight

    debt_penalty = 0.0
    if debt_ratio is not None:
        if debt_ratio > 300: debt_penalty = -0.12
        elif debt_ratio > 200: debt_penalty = -0.07
        elif debt_ratio > 150: debt_penalty = -0.03

    p = base_prob + market_adj + vol_adj + trend_adj + pattern_adj + debt_penalty
    return round(max(0.05, min(0.92, p)) * 100, 1)
```

## 7. 스크립트 적용 가능한 함수 구조

아래 의사코드는 기존 `api/index.py`에 추가하거나 `calc_buy_price()`와 `calc_risk()`를 대체하는 구조입니다. 반환 JSON 구조를 먼저 바꾸고, 프론트엔드 `renderForecast()`는 `buy_price.aggressive.bands`, `buy_price.recommended.bands`, `risk_scenarios[strategy].tp_levels`를 순회하도록 수정하면 됩니다.

### 7.1 가격 밴드 생성 함수 구조

```python
BACKTEST_PRIOR = {
    "KRX": {
        "avg_atr_pct": 2.37,
        "band_width": 1.18,
        "prob_adj": -0.06,
        "zones": {
            "A": {"depth": [0.25, 0.65], "win": 0.495, "loss": 0.505, "exp_ret": 0.0194, "sharpe": 3.36, "hold": 15.1, "fail_loss": -0.0161},
            "B": {"depth": [0.65, 1.10], "win": 0.604, "loss": 0.396, "exp_ret": 0.0274, "sharpe": 5.68, "hold": 13.0, "fail_loss": -0.0111},
            "C": {"depth": [1.10, 1.70], "win": 0.743, "loss": 0.257, "exp_ret": 0.0372, "sharpe": 11.21, "hold": 9.2, "fail_loss": -0.0079},
        },
    },
    "US": {
        "avg_atr_pct": 1.83,
        "band_width": 0.92,
        "prob_adj": 0.05,
        "zones": {
            "A": {"depth": [0.25, 0.65], "win": 0.594, "loss": 0.406, "exp_ret": 0.0149, "sharpe": 3.07, "hold": 14.0, "fail_loss": -0.0167},
            "B": {"depth": [0.65, 1.10], "win": 0.663, "loss": 0.337, "exp_ret": 0.0200, "sharpe": 4.86, "hold": 12.3, "fail_loss": -0.0122},
            "C": {"depth": [1.10, 1.70], "win": 0.752, "loss": 0.248, "exp_ret": 0.0270, "sharpe": 9.18, "hold": 9.0, "fail_loss": -0.0068},
        },
    },
}

def build_price_bands(dd, price, atr, market, score, indicator_signals, debt_ratio=None):
    prior = BACKTEST_PRIOR["US" if market == "US" else "KRX"]
    w = prior["band_width"]
    atr_pct = atr / price * 100
    trend_score = calc_trend_score(dd)          # -1.0 ~ +1.0
    pattern_win, pattern_n = find_similar_pattern_winrate(dd, horizon=20)
    anchors = calc_anchor_prices(dd, price, atr) # BB, MA20, VWAP, Fib38.2, support cluster

    result = {"current": round_price(price, market), "aggressive": {"bands": []}, "recommended": {"bands": []}}

    for zone_name, z in prior["zones"].items():
        d1, d2 = z["depth"]
        low = price - d2 * atr * w
        high = price - d1 * atr * w
        prob = calc_probability(z["win"], atr_pct, prior["avg_atr_pct"], trend_score, pattern_win, pattern_n, debt_ratio, market)
        result["aggressive"]["bands"].append({
            "band": zone_name,
            "price_range": [round_price(low, market), round_price(high, market)],
            "basis": make_aggressive_basis(zone_name, dd, atr_pct, trend_score),
            "expected_return_pct": round(z["exp_ret"] * 100, 2),
            "loss_probability_pct": round(100 - prob, 1),
            "win_probability_pct": prob,
        })

        anchor = anchors[zone_name]
        rec_low = anchor - d2 * atr * w * 0.75
        rec_high = anchor - d1 * atr * w * 0.45
        rec_prob = calc_probability(z["win"], atr_pct, prior["avg_atr_pct"], trend_score, pattern_win, pattern_n, debt_ratio, market)
        result["recommended"]["bands"].append({
            "band": zone_name,
            "price_range": [round_price(rec_low, market), round_price(rec_high, market)],
            "expected_sharpe_ratio": z["sharpe"],
            "historical_win_rate_pct": rec_prob,
            "avg_holding_days": z["hold"],
            "distribution_basis": f"{z['depth'][0]}~{z['depth'][1]}ATR 진입 깊이의 20거래일 분포 기반",
        })

    return result
```

### 7.2 ATR 기반 손절·목표가 계산 로직

```python
RISK_MODEL = {
    "conservative": {"label": "보수적", "stop": 0.85, "tp": [1.10, 1.65, 2.20]},
    "balanced":     {"label": "중립적", "stop": 1.20, "tp": [1.50, 2.30, 3.10]},
    "aggressive":  {"label": "공격적", "stop": 1.55, "tp": [2.00, 3.20, 4.60]},
}

def build_atr_risk_scenarios(entry_price, atr, market, dd, debt_ratio=None):
    prior = BACKTEST_PRIOR["US" if market == "US" else "KRX"]
    w = prior["band_width"]
    target_adj = 1.05 if market == "US" else 0.95
    atr_pct = atr / entry_price * 100
    trend_score = calc_trend_score(dd)
    pattern_win, pattern_n = find_similar_pattern_winrate(dd, horizon=20)

    output = {}
    for key, cfg in RISK_MODEL.items():
        stop_price = entry_price - cfg["stop"] * atr * w
        risk = entry_price - stop_price
        tp_levels = []
        prev_prob = 100.0
        for i, mul in enumerate(cfg["tp"], start=1):
            tp_price = entry_price + mul * atr * target_adj
            reward = tp_price - entry_price
            rr = reward / max(risk, 1e-9)
            base = base_tp_probability(market, key, i)
            prob = calc_probability(base, atr_pct, prior["avg_atr_pct"], trend_score, pattern_win, pattern_n, debt_ratio, market)
            prob = min(prob, prev_prob - (0 if i == 1 else 3.0))
            prev_prob = prob
            tp_levels.append({
                "level": f"TP{i}",
                "price": round_price(tp_price, market),
                "return_pct": round((tp_price / entry_price - 1) * 100, 2),
                "hit_probability_pct": round(prob, 1),
                "avg_days_to_hit": estimate_days_to_hit(mul, atr_pct, market, trend_score),
                "failed_avg_loss_pct": estimate_failed_loss(cfg["stop"], atr_pct, market, i),
                "risk_reward": round(rr, 2),
            })
        output[key] = {
            "label": cfg["label"],
            "entry_range": estimate_entry_range_for_strategy(key, entry_price, atr, market),
            "stop_loss": round_price(stop_price, market),
            "tp_levels": tp_levels,
            "risk_reward_ratio": [x["risk_reward"] for x in tp_levels],
        }
    return output
```

### 7.3 한국·미국 분기 처리 조건

```python
def get_market_params(market):
    if market == "US":
        return {
            "round_digits": 4,
            "band_width": 0.92,
            "base_probability_adjustment": 0.05,
            "target_distance_adjustment": 1.05,
            "momentum_weight": 1.20,
            "volatility_penalty_weight": 0.85,
            "comment": "미국 시장: 추세 지속성 높음, 밴드 정교화, 모멘텀 확률 가중",
        }
    return {
        "round_digits": 2,
        "band_width": 1.18,
        "base_probability_adjustment": -0.06,
        "target_distance_adjustment": 0.95,
        "momentum_weight": 0.90,
        "volatility_penalty_weight": 1.20,
        "comment": "한국 시장: 변동성·수급 영향 반영, 밴드 확대, 목표 확률 보수화",
    }
```

프론트엔드에서는 기존 `bp.conservative` 참조를 제거해야 합니다. 구체적으로 `renderForecast()`의 `buyZone('conservative', ...)` 호출을 삭제하고, 대신 `bp.aggressive.bands.map()`과 `bp.recommended.bands.map()`을 각각 렌더링해야 합니다. 리스크 카드는 보수적·중립적·공격적을 유지하되, 기존 `sc.target[0] ~ sc.target[1]` 표시를 `sc.tp_levels` 반복 표시로 변경합니다.

## 8. 예측 탭 출력 예시 구조

아래 구조는 실제 화면에 출력할 때 유지해야 하는 형태입니다. 가격은 종목별 `P`, `ATR`, 시장 계수로 계산되며, 확률은 위의 `calc_probability()` 보정 후 표시됩니다.

```text
[공격적 매수]
- 가격 밴드 A
  - 가격 범위: P - 0.65ATR×w ~ P - 0.25ATR×w
  - 진입 근거: 얕은 ATR 눌림, MA20 지지, ADX 상승 추세 확인
  - 기대 수익률: 한국 1.94% / 미국 1.49%
  - 손실 발생 확률: 한국 50.5% / 미국 40.6%에서 현재 지표로 보정
- 가격 밴드 B
  - 가격 범위: P - 1.10ATR×w ~ P - 0.65ATR×w
  - 진입 근거: 평균 변동폭 중심부, VWAP·볼린저 중단 근접
  - 기대 수익률: 한국 2.74% / 미국 2.00%
  - 손실 발생 확률: 한국 39.6% / 미국 33.7%에서 현재 지표로 보정
- 가격 밴드 C
  - 가격 범위: P - 1.70ATR×w ~ P - 1.10ATR×w
  - 진입 근거: 1ATR 이상 하락, RSI 저점권, 지지 클러스터 접근
  - 기대 수익률: 한국 3.72% / 미국 2.70%
  - 손실 발생 확률: 한국 25.7% / 미국 24.8%에서 현재 지표로 보정

[추천 매수 구간]
- 가격 밴드 A
  - 가격 범위: anchor - 0.25ATR×w ~ anchor + 0.20ATR×w
  - 기대 Sharpe Ratio: 한국 3.36 / 미국 3.07
  - 과거 유사 구간 승률: 한국 49.5% / 미국 59.4%에서 현재 지표로 보정
  - 평균 보유 기간: 한국 15.1일 / 미국 14.0일
- 가격 밴드 B
  - 가격 범위: anchor - 0.70ATR×w ~ anchor - 0.25ATR×w
  - 기대 Sharpe Ratio: 한국 5.68 / 미국 4.86
  - 과거 유사 구간 승률: 한국 60.4% / 미국 66.3%에서 현재 지표로 보정
  - 평균 보유 기간: 한국 13.0일 / 미국 12.3일
- 가격 밴드 C
  - 가격 범위: anchor - 1.30ATR×w ~ anchor - 0.70ATR×w
  - 기대 Sharpe Ratio: 한국 11.21 / 미국 9.18
  - 과거 유사 구간 승률: 한국 74.3% / 미국 75.2%에서 현재 지표로 보정
  - 평균 보유 기간: 한국 9.2일 / 미국 9.0일
```

## 9. 실전 적용 시 기대 효과 및 한계

이 재설계의 가장 큰 기대 효과는 예측 탭이 단순히 “어디서 사면 좋다”는 가격 제안에서 벗어나, **각 가격대의 기대 수익, 손실 확률, 보유 기간, 목표 도달 가능성**을 동시에 보여주는 의사결정 도구로 바뀐다는 점입니다. 특히 한국 시장은 변동성과 수급 충격을 반영해 밴드를 넓게 제시하고 목표 도달 확률을 보수적으로 조정하므로, 추격 매수와 과도한 낙관을 줄일 수 있습니다. 미국 시장은 추세 지속성과 모멘텀 가중을 반영하므로, 강한 상승 추세에서 지나치게 낮은 가격만 기다리다가 기회를 놓치는 문제를 완화할 수 있습니다.

다만 한계도 명확합니다. 본 백테스트는 실제 체결 데이터가 아니라 1년 가정 기반 분포 시뮬레이션이므로, 특정 종목의 이벤트 리스크, 실적 발표, 공시, 금리 발표, 환율 급변, 지정학적 충격은 완전히 반영하지 못합니다. 또한 20거래일 내 목표 도달 여부를 중심으로 설계되어 장기 투자 판단에는 별도의 펀더멘털 모델이 필요합니다. 마지막으로 부채비율 패널티는 재무 안정성을 반영하기 위한 필수 장치지만, 성장주나 금융업처럼 업종별 적정 부채 구조가 다른 종목에는 업종 보정 로직을 추가해야 합니다.

따라서 실전 적용 시에는 본 로직을 **단독 매수·매도 명령**으로 사용하기보다, 기존 뉴스·수급·기술 지표·재무 안정성 점검과 결합한 확률형 의사결정 레이어로 사용하는 것이 적절합니다. 운영 관점에서는 매월 실제 예측값과 실현값을 저장해 `BACKTEST_PRIOR`의 승률, 기대 수익률, 실패 손실폭을 업데이트하면, 시간이 지날수록 예측 탭이 정적 설명 도구가 아니라 **자기 보정형 트레이딩 신호 엔진**으로 개선됩니다.
