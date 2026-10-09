# StockOracle (Vercel Edition)

Vercel 서버리스 환경에 최적화된 AI 기반 주식 분석 시스템입니다.

Python으로 구현되어 26개 캔들스틱 패턴을 인식하고 핵심 기술적 지표를 산출합니다.

---

## 🚀 주요 기능

| 기능 | 설명 |
|---|---|
| **KRX / US 통합 지원** | 국내(KOSPI·KOSDAQ) 및 미국(NYSE·NASDAQ) 주식 통합 조회 |
| **미국장 세션 완벽 대응** | 서머타임(DST) 자동 적용, 데이마켓/프리마켓/정규장/애프터마켓 장 구분별 최적의 실시간 가격(호가/체결가) 우선순위 반영 |
| **분석 기간** | 종목 상세 분석은 1년 일봉으로 고정하며, 초단기 관찰 기간은 별도 추천 기능에서 다룹니다. |
| **26개 캔들스틱 패턴** | 순수 Python 구현 (TA-Lib 불필요, Vercel 완전 호환) |
| **핵심 기술적 지표 최적화** | MA, EMA, MACD, RSI, 볼린저밴드, ATR, ADX, OBV, Aroon 등 실전 압축 지표 제공 |
| **가중치 기반 종합 점수** | 추세·모멘텀·변동성·거래량·보조지표 5개 축 0~100 점수화 |
| **AI 종합 진단 및 전략** | 점수 기반 Buy/Hold/Sell 요약 및 시나리오별(상승/하락/횡보) 대응 전략 제시 |
| **차트 형태 인식** | 삼각형·쐐기형·이중 천장/바닥 등 7가지 기하학적 패턴 |
| **조건부 가격 예측** | 최근 일간 로그수익률·ATR 변동성과 상승/횡보/하락 시나리오 비중으로 기간별 기준가와 P10~P90 범위를 계산합니다. |
| **ATR 기반 리스크 관리** | 보수적·중립·공격적 3가지 시나리오별 목표가·손절가 자동 산출 |
| **실시간 스크리너** | 국내·해외 주요 종목 시세, 등락률 모니터링 |
| **기업 정보 크롤링** | Naver Finance — 시가총액, PER, PBR, ROE, 부채비율, 공시 |
| **피터 린치 스타일 GARP 점검** | PER·부채비율·보고 EPS 3년 CAGR·동일 CAGR 기반 PEG·시가총액의 5개 기준을 결측 통과 없이 검증 |
| **뉴스 피드** | Google News RSS 한글·영문 통합 |
| **시장 심리 지수** | 미국 VIX, 한국 KOSPI200 |

---

## 🇺🇸 미국장 세션별 시세 반영 로직 (카카오페이증권 기준)

한국시간과 미국시간(ET) 간의 변환 및 서머타임(DST)을 자동 계산하여, 각 장의 특성에 맞는 가장 정확하고 빠른 실시간 가격 데이터를 예측 모델에 투입합니다.

| 세션 | 시간 (ET) | 참조 우선순위 | 설명 |
|---|---|---|---|
| **프리마켓** | 04:00 ~ 09:30 | `preMarketPrice` > `fast_info` | 정규장 개장 전의 호가/체결가 우선 |
| **정규장** | 09:30 ~ 16:00 | `fast_info.last_price` > `currentPrice` | 업데이트 주기가 가장 빠른 체결가 우선 |
| **애프터마켓** | 16:00 ~ 20:00 | `postMarketPrice` > `fast_info` | 정규장 종료 직후의 시간외 거래 시세 우선 |
| **데이마켓** | 20:00 ~ 익일 04:00 | `postMarketPrice` > `regularMarketPrice` | 24시간 대체거래소 데이터 한계 보완을 위해 최근 갱신된 애프터마켓 종가 참조 |

---

## 🕯️ 캔들스틱 패턴 (26개)

Python + NumPy로 구현되어 Vercel 서버리스에서 동작합니다.
최근 10봉 평균 몸통 크기를 기준으로 상대적 크기를 동적으로 판단합니다.

### 단일봉 패턴 (4개)

| 패턴 | 신호 방향 | 신뢰도 | 설명 |
|---|---|---|---|
| ✖️ Doji | 중립 | 70% | 몸통 < 레인지 10%, 방향 전환 경고 |
| 🔨 Hammer | 상승 | 80% | 긴 아래 꼬리, 하락 후 반등 신호 |
| ⭐ Shooting Star | 하락 | 80% | 긴 위 꼬리, 상승 후 하락 신호 |
| 📏 Marubozu | 추세 지속 | 80% | 꼬리 없는 강한 방향성 봉 |

### 2봉 패턴 (7개)

| 패턴 | 신호 방향 | 신뢰도 | 설명 |
|---|---|---|---|
| 🫂 Bullish Engulfing | 상승 | 85% | 음봉을 완전히 감싸는 양봉 |
| 🫂 Bearish Engulfing | 하락 | 85% | 양봉을 완전히 감싸는 음봉 |
| 🤰 Bullish Harami | 상승 | 70% | 큰 음봉 내 작은 양봉 (전환 초기) |
| 🤰 Bearish Harami | 하락 | 70% | 큰 양봉 내 작은 음봉 (전환 초기) |
| ➕ Harami Cross | 상승/하락 | 80% | 큰 봉 내의 도지 (강한 전환 경고) |
| 🎯 Piercing Line | 상승 | 80% | 전일 몸통 절반 초과 관통 (하락 반전) |
| ☁️ Dark Cloud Cover | 하락 | 80% | 전일 몸통 절반 아래 하락 (상승 반전) |

### 3봉 패턴 (8개)

| 패턴 | 신호 방향 | 신뢰도 | 설명 |
|---|---|---|---|
| 🌅 Morning Star | 상승 | 90% | 음봉 + 소형봉 + 양봉 (강한 상승 반전) |
| 🌆 Evening Star | 하락 | 90% | 양봉 + 소형봉 + 음봉 (강한 하락 반전) |
| ⚪ Three White Soldiers | 상승 | 90% | 3연속 상승 양봉 (강한 추세 확인) |
| 🐦 Three Black Crows | 하락 | 90% | 3연속 하락 음봉 (강한 추세 확인) |
| 📦 Three Inside Up | 상승 | 85% | Harami 상승 확인형 |
| 📤 Three Inside Down | 하락 | 85% | Harami 하락 확인형 |
| 📤 Three Outside Up | 상승 | 88% | Engulfing 상승 강세 확인형 |
| 📦 Three Outside Down | 하락 | 88% | Engulfing 하락 강세 확인형 |

### 복합 패턴 — 4~5봉, 갭 포함 (7개)

| 패턴 | 신호 방향 | 신뢰도 | 설명 |
|---|---|---|---|
| 📊 Rising Three Methods | 상승 | 90% | 큰 양봉 + 3소형 음봉 + 큰 양봉 (추세 지속) |
| 📊 Falling Three Methods | 하락 | 90% | 큰 음봉 + 3소형 양봉 + 큰 음봉 (추세 지속) |
| 👶 Abandoned Baby Bull | 상승 | 92% | 갭다운 도지 포함 강한 상승 반전 |
| 👶 Abandoned Baby Bear | 하락 | 92% | 갭업 도지 포함 강한 하락 반전 |
| 🎣 Hikkake Bull | 상승 | 82% | 내부바 하향 속임 후 상승 반전 |
| 🎯 Hikkake Bear | 하락 | 82% | 내부바 상향 속임 후 하락 반전 |
| 🤝 Mat Hold | 상승 | 85% | Rising Three Methods 갭 변형 (추세 지속) |

---

## 📊 기술적 지표 (핵심 지표 최적화)

| 분류 | 지표 |
|---|---|
| **추세** | MA5, MA20, MA60, MA120, EMA20, EMA50 |
| **모멘텀** | RSI(14), MACD, Signal Line, Stochastic %K/%D |
| **변동성** | Bollinger Bands(20,2), ATR(14) |
| **추세 강도/거래량** | ADX(14), DI+, DI−, OBV, Aroon(25), 매수 압력(Buy Pressure) |

---

## 🧮 종합 점수 시스템

Base 50점에서 각 축의 가중치만큼 가감하여 0~100점으로 산출하며, 이 점수를 바탕으로 AI 트레이딩 전략(Buy/Hold/Sell)을 제시합니다.

| 축 | 가중치 | 주요 지표 |
|---|---|---|
| 추세 분석 | 35% | EMA20/50 정배열, 현재가 vs EMA20, MACD 크로스, PSAR |
| 모멘텀 | 30% | RSI(14) 과매수/과매도, ADX(14) + DI 방향 |
| 변동성 | 20% | 볼린저밴드 위치, ATR 대비 변동 비율 |
| 거래량 | 15% | 현재 거래량 vs 20일 평균 |
| 보조 지표 및 패턴 | 보조 | 상승/하락 패턴, OBV 및 Aroon 크로스/다이버전스 |

---

## 🏗️ 가격 예측 모델

`market_briefing/forecast_model.py`는 최근 60거래일의 **연속된 유효 일봉 쌍**에서 로그수익률 변동성을 구하고 ATR 추정치와 결합한 뒤, 이력이 120일 이상이면 최근 약 1년(251일) 실현 변동성을 같은 비중으로 섞습니다(변동성은 평균으로 돌아가므로 22거래일을 볼 때 최근 몇 주만으로 정하지 않음, `STOCKORACLE_VOL_LONG_RUN_WEIGHT` 로 조정·0 이면 끔). 예측 탭은 이 변동성에 기간의 제곱근을 적용하고, 상승·하락 시나리오 비중에 따라 기준가를 제한적으로 기울여 P10~P90 가격 범위를 표시합니다. 거래일 종료 목표일은 한국·미국 휴장일을 건너뛰어 계산합니다.

- 예측 응답은 하나의 `price_anchor`를 사용합니다. 현재가·진입가·목표가·손절가·조건부 예측의 기준가는 동일한 앵커 ID와 시세를 공유합니다.
- 정규장 실시간 가격이 마지막 일봉에 반영되면 가격 의존 지표를 다시 계산합니다. 시간외 시세는 확정 일봉에 섞지 않고 `현재 시세`와 `기술지표 기준`을 화면에 따로 표시합니다.
- 실시간 소스를 확보하지 못하면 현재 시세로 오인하지 않도록 `최근 확정 종가`로 표시합니다.
- 유효 일봉 20개 미만이면 가격 예측을 보류합니다. 60개 미만이거나 대체 ATR 등을 사용하면 제한 상태로 표시합니다.
- 결측 일봉 양쪽 종가를 이어서 하루 수익률로 계산하지 않습니다. 최신 봉의 지표가 비어 있으면 과거 지표를 현재값처럼 사용하지 않습니다.
- 가격 범위는 조건부 통계 추정이며 적중률이나 보장 구간이 아닙니다. 예측 탭의 시나리오 비중과 지표 일치도도 검증된 상승 확률로 해석하면 안 됩니다.
- 별도 LightGBM 방향 모델은 검증 상태가 허용하는 경우에만 보조 신호로 사용합니다. 확률 블렌드 가중과 상관 엔진 투표 신뢰 비율은 메타데이터에 기록된 검증 AUC(홀드아웃·선택 후보 워크포워드·보정 OOF) 중 **최솟값**으로 정합니다(홀드아웃은 피처·파라미터 탐색에 쓰여 낙관적). 현재 최솟값은 OOF 0.515 라 가중 0 이며, 예측값은 `ml_prediction` 에 참고용으로 남습니다. 이전 방식은 `STOCKORACLE_ML_AUC_BASIS=holdout`. 가격 범위 계산에 XGBoost나 Holt-Winters를 사용하지 않습니다.
- 정규장 중 받은 일봉의 마지막 막대는 오늘 진행 중인 막대라 거래량이 누적값(최종값 이하)입니다. 이를 20일 평균과 바로 비교하면 조회 시각만으로 `거래량 위축`이 되어 하이브리드 NCS 가 마감 대비 평균 −7.5~−11점 흔들렸으므로, 이 경우 하이브리드·스캔·`analyze_score` 거래량 단계·상관 엔진 입력은 **직전 확정 막대** 기준으로 거래량 비율을 계산합니다(`market_briefing/session_bars.py`, 응답 `data_quality.volume_basis`). 가격·지표는 현재가를 반영한 마지막 막대를 그대로 씁니다.
- 하이브리드 허스트 지수는 스캔과 같은 **로그수익률 R/S** 구현을 씁니다(이전 상세 화면 값은 가격 수준에 적용돼 모든 종목이 0.88~0.93 이었음). 상세 화면과 스캔의 BQS/FWS/NCS 공유 하위 점수는 `tests/test_hybrid_dual_score_parity.py` 가 일치를 지킵니다.
- `target_price.reach_probability`는 규칙 가점식이 아니라 무추세 변동성 모델의 **하단 가격 터치 확률**(`reach_probability_basis=touch_probability`)입니다. 신뢰도·상관 보정으로 범위가 바뀌면 최종 하단 가격으로 다시 계산하며, 변동성을 구할 수 없을 때만 규칙 가점식(`heuristic_fallback`)을 씁니다.
- 규칙 점수 확정은 `finalize_rule_score` 한 곳에서 합니다: KRX 수급(±5)·HybridTurtle NCS 가산을 모두 반영한 뒤 BEAR(≤40)·부채 150%↑(≤45)·하이브리드 BEARISH(≤40) 상한을 마지막에 적용합니다. NCS 행동 분류는 `hybrid_signals.ncs_action`(AUTO_YES: NCS≥70·FWS≤30 / AUTO_NO: FWS>65) 하나만 사용합니다.
- 상관 엔진의 신호 일치도는 기권(방향 없음) 표를 분모에 포함한 `0.5+0.5·|상승−하락|/전체` 이고, 신뢰도 표는 신호 방향(BUY/SELL)을 따릅니다. 일치도로 목표가 범위를 좁히는 기능은 검증에서 이득이 없어 기본으로 꺼져 있습니다(`STOCKORACLE_CORRELATION_NARROWING=1` 로 켬).
- 동적 RSI의 과매도 접촉·확정 상승 다이버전스·RSI 50 회복이 모두 끝나면 신호 봉 종가를 `1차 매수 기준가`로 표시합니다. 다음 정규장 시가가 신호 봉 ATR 기준 `+0.5 ATR` 상한을 넘으면 추격하지 않으며, 신호가 지난 뒤에는 같은 가격을 과거 기준가로만 표시합니다.
- 예측 탭의 1차 탐색·2차 본 진입 가격은 고정 할인율이 아니라 일봉 OHLCV, 아래꼬리 반등, 확정 스윙 저점, MA20/60/120, 볼린저 중앙·하단과 밴드 폭, 20일 거래량 가중 가격, 거래량 집중 가격대, 피보나치·주봉 지지를 ATR 거리로 군집화해 산정합니다. 강세장은 가까운 군집, 약세장은 깊은 군집을 우선합니다. 두 구간은 **어떤 종목이든 항상 A/B/C 3개 밴드·5단계로 표시**하며(2차 범위는 항상 1차 전체보다 아래), 차트 근거가 얇은 종목은 가격을 숨기는 대신 밴드마다 근거 등급(`evidence_level`: 구조 근거=서로 다른 가격 근거 2종 이상 겹침 / 단일 지표 / 변동성 기반 추정)과 가장 가까운 가격 근거(`nearest_reference`), 단계별 도달 확률 범위를 함께 공개합니다. 과거 경로 표본(40건)이 부족하면 도달 확률·기간을 무추세 변동성 터치 모델로 추정하고 `probability_source=volatility_model`("추정" 표기)로 구분합니다.

---

## 🔬 예측 체인 검증 (워크포워드 감사)

`scripts/audit_prediction_layers.py` 는 `/api/stock` 의 점수·확률·목표가·예측구간 계산을 과거 각 시점에서 서비스와 같은 입력으로 다시 실행하고, 이후 22거래일의 실제 결과와 비교합니다(52종목·2022-10~2026-09, 8,368 시점). 결과 요약은 `docs/backtests/prediction_layer_audit_summary.json`, 해석은 `docs/prediction_layer_audit_20261008.md`·`docs/entry_band_and_calibration_audit_20261009.md`·`docs/system_audit_20261009.md`(허스트·장중 거래량·ML 가중·변동성 비중 변경과 그 전/후 측정) 에 있습니다.

| 출력 | 측정 결과 | 해석 |
|---|---|---|
| 예측구간 P10~P90 / P05~P95 | 실제 포함률 81.3% / 89.8% (명목 80% / 90%, 장기 변동성 혼합 전 79.2% / 88.2%) | 변동성 기반 범위는 보정되어 있음 |
| 상승 시나리오 하단 터치 확률 | AUC 0.72, Brier 0.161 (상수 예측 0.179) | 가격 도달 가능성은 변동성에 대해 의미 있음 |
| 방향 확률(`prob_up`)·시나리오 비중 | AUC 0.51 (95% 구간 0.49~0.53), 보정 기울기 0.06 | 방향에는 검증된 정보가 거의 없음 → **실현 빈도로 보정해 표시**(아래), 원래 값은 `prob_up_raw` 신호 점수로 분리 |
| 1차·2차 매수 구간 단계별 도달 확률 | 터치 모델 Brier 0.191·AUC 0.787 (기존 경험 경로+가산 방식 0.213·0.733, 상수 0.247) | 변동성 터치 모델로 통일, 닿는 날짜 창도 같은 분포에서 산출. 아래쪽 가격만 평가하고 표본이 상승 구간이라 하락 방향 터치를 과대 예측하는 한계가 있음 |

```bash
python scripts/audit_prediction_layers.py --offline --stride 6     # 캐시 사용 (처음에는 --offline 없이 실행해 5년 일봉 수집)
python scripts/audit_prediction_layers.py --tickers AAPL,005930.KS --stride 3
```

```bash
python scripts/audit_entry_bands.py --offline --stride 8                  # 1차·2차 매수 구간 불변식 + 단계별 도달 확률·기간 보정
python scripts/audit_entry_bands.py --offline --shard 0/8 --rows-csv /tmp/entry_0.csv --out /tmp/entry_0.json   # 병렬(0~7) 후
python scripts/audit_entry_bands.py --merge "/tmp/entry_?.csv"            # 조각 합쳐 요약
python scripts/audit_prediction_layers.py --offline --stride 6 --rows-csv /tmp/audit_rows.csv
python scripts/fit_probability_calibration.py --rows-csv /tmp/audit_rows.csv   # models/probability_calibration.json 재생성
```

**방향 확률 보정** (`market_briefing/probability_calibration.py`, `models/probability_calibration.json`)

- 규칙 점수가 만든 상승 확률은 10~91%로 퍼지지만 22거래일 뒤 실제 상승 빈도와는 거의 무관했습니다(표본 밖 Brier 0.275 > 기저율 상수 0.244). 표시 `prob_up`/`prob_down` 은 `p = 시장별 절편 + 기울기×(p_raw − 0.5)` 로 보정한 값이며(KRX 약 55%·US 약 58% 부근), 원래 값은 `prob_up_raw`/`prob_down_raw`(신호 점수)와 `probability_calibration`(적용 여부·기울기·기저율·학습 기간)으로 함께 내려갑니다.
- 기울기는 시장을 합쳐 추정한 값(0.037, 부트스트랩 95% 구간 −0.055~0.135)입니다. 시장별 기울기는 학습/검증 구간에서 부호가 뒤집혀 채택하지 않았고, 구간이 0 을 포함하므로 규칙 점수가 방향을 구분한다는 증거가 없다는 뜻입니다. 보정 파일이 없으면 원래 값을 그대로 보여 줍니다(fail-open).
- `build_prediction_outlook` 의 시나리오 비중은 이 보정 확률만 씁니다. 점수·수급·패턴·시장 체제·등락률을 한 번 더 가산하던 부분(워크포워드 AUC 0.500, Brier 0.296)은 제거했고, 그 요인들은 상승 요인/하락 위험 문장에만 남습니다. `주의·매수 보류` 라벨도 검증되지 않은 `하락 비중 ≥ 48%` 대신 구조 이탈 2건 이상·이벤트 위험 60점 이상·붕괴 대기 신호만 봅니다.

**레짐 라벨 분리** — 기준이 다른 세 가지가 같은 단어로 보이지 않도록 이름을 나눴습니다(`regime_layers`).

| 이름 | 기준 | 값 | 어디서 |
|---|---|---|---|
| 지수 중기 구조 (KOSPI/KOSDAQ/S&P 500 60·120일선) | 지수 종가와 60·120일선 배열 | 상승 구조 / 하락 구조 / 혼조 (`market_regime` BULL/BEAR/NEUTRAL) | 매수 카드 시장 문구, 예측 탭 시장 사실, 점수 상한(BEAR ≤ 40) |
| 시장 장기 국면 (200일선·ADX·VIX) | 벤치마크 200일선·ADX·VIX 5점 합산 | 강세장 / 약세장 / 횡보장 | 스캔 머리말(벤치마크 있을 때만) |
| 종목 추세 (ADX·DI, 시장 기준 미적용) | 벤치마크 없이 종목 자신의 ADX·DI | 상승 추세 / 하락 추세 / 방향 불명 | 단일 종목 복합 기술 신호 카드 |

같은 날 KOSPI 가 `하락 구조`이면서 장기 국면이 `강세장`일 수 있고(`regime_layers.note` 가 서로 다른 기준임을 알립니다), 단일 종목 분석에는 벤치마크가 없어 장기 국면을 계산하지 않습니다.

점수·확률·보정 체인을 바꾼 뒤에는 이 감사를 다시 실행해 AUC·Brier·구간 포함률이 나빠지지 않았는지 확인하세요. 투자자 수급·ML·뉴스·실시간 시세는 오프라인 재현이 불가능해 제외되므로, 결과는 기술적 입력만으로 만든 체인의 성능입니다.

---

## 🛡️ 안정성 및 캐시 최적화

Vercel 서버리스 및 로컬 환경에서 발생하는 `yfinance` 고질적 에러들을 자체적으로 우회 및 방어합니다.

- **IPv6 타임아웃(curl: 28) 방어**: `curl_cffi`가 야후 서버와 통신 시 IPv6 블랙홀에 빠지는 현상을 막기 위해 C레벨의 옵션을 몽키패치하여 **IPv4를 강제**하고 응답 속도를 개선했습니다.
- **SQLite DB Lock 회피**: `yfinance` 내부 캐싱 모듈이 Vercel의 Read-only 파일 시스템에서 에러를 내는 것을 막기 위해 `peewee` 메모리 DB(`:memory:`)로 덮어씌웠습니다.
- **실시간성**: 일봉은 2분 TTL 캐시(`fetch_stock_data`), KRX 실시간 현재가는 10초 캐시 API, 미국은 `USStockPriceFetcher` 의 세션별 시세를 씁니다. (이전 문서가 가리키던 `fetch_metrics` 는 현재 호출처가 없는 함수라 이 서술에서 제외했습니다.)
- **실패 응답 단기 보관**: `ttl_cache` 는 `{"error": …, "items": []}` 형태의 실패 응답을 TTL 전체(장기 추천은 4시간)가 아니라 15초만 보관해, 일시 장애나 코드 수정 뒤에도 빈 화면이 남지 않게 합니다. 장기 추천·급등 화면의 🔄 새로고침(`?refresh=1`)도 서버가 처리합니다.
- **JSON 파싱 에러 방어**: API 서버 오류 시 프론트엔드에서 불완전한 텍스트를 파싱하려다 뻗는 오류(`Unexpected token 'A'`)를 처리하는 에러 핸들링 로직이 반영되었습니다.
- **KRX 벤치마크**: Yahoo `^KS200` 은 일봉을 1행만 돌려주므로 스캔 레짐·상대강도·시장 심리·시장 면역에는 `^KS11` → `069500.KS`(KODEX 200) 순서의 `_benchmark_history` 를 씁니다. 지수 일봉의 마지막 행 종가가 NaN 일 수 있어(장 시작 전·장중) 항상 제거한 뒤 계산하며, 지수 조회 실패는 `NEUTRAL` 로 15분 동안 굳지 않도록 캐시하지 않습니다.
- **학습 로그**: 같은 종목·거래일·기간의 예측은 하루 한 건만 기록하고, 결과 집계는 (종목·신호일·구간) 중복을 제거해 새로고침이 표본 수를 늘리지 못하게 합니다. 로컬 테스트는 `STOCKORACLE_PREDICTION_LOG` 를 임시 경로로 지정하세요.

### 조정·되돌리기용 환경변수

| 변수 | 기본 | 효과 |
|---|---|---|
| `STOCKORACLE_PREDICTION_LOG` | 로컬 `docs/backtests/prediction_learning.jsonl`, Vercel `/tmp` | 예측 학습 로그 경로. 로컬 테스트·감사는 임시 경로로 지정 |
| `STOCKORACLE_VOL_LONG_RUN_WEIGHT` | `0.5` | 변동성 σ 에 섞는 1년 실현 변동성 비중(0~1). `0` 이면 최근 60일+ATR 만 사용(오차 분위 배율도 이전 값으로 복귀) |
| `STOCKORACLE_ML_AUC_BASIS` | `conservative` | ML 가중의 근거 AUC. `holdout` 이면 홀드아웃 `test_auc`(이전 방식) |
| `STOCKORACLE_CORRELATION_NARROWING` | 꺼짐 | `1` 이면 상관 엔진이 일치도로 목표가 범위를 좁힘 |
| `STOCKORACLE_CORRELATION_PROB_PULL` | 켜짐 | `0` 이면 상관 엔진의 확률 당김을 끔 |
| `STOCKORACLE_LEARNING_REMOTE_READ` / `_SYNC_ON_REQUEST` | 꺼짐 | 학습 로그를 GitHub 에서 읽기 / 요청마다 커밋(토큰 `STOCKORACLE_LEARNING_GITHUB_TOKEN` 필요) |
| `STOCKORACLE_US_LONGTERM_GARP_TOP` | `15` | 미국 장기추천 GARP 재무조회 상위 N종목(5~30). 60초 제한 안에 응답하기 위해 기술 상위만 조회 |
| `STOCKORACLE_KR_LONGTERM_GARP_TOP` | `15` | 국내 장기추천 GARP 재무조회 상위 N종목(5~30). US와 동일 이유 |
| `STOCKORACLE_CHASE_PENALTY` | 꺼짐 | `1`이면 하이브리드 FWS 추격 15/25점을 반영. 기본은 표시만 고치고 점수는 그대로(추격-수익 무관련 근거). |
| `STOCKORACLE_MOMENTUM_SCAN` | 켜짐 | `0`이면 7단계 스캔 모멘텀 지속 신호·승격을 끔. 기본 켜짐(점수 미반영, PASS만 READY 승격). |
| `STOCKORACLE_VCP_SCAN` | 켜짐 | `0`이면 7단계 스캔 VCP 신호·승격을 끔. 기본 켜짐(점수 미반영, 위험없는 PASS만 READY 승격). |

---

## ⚙️ API 엔드포인트

| 경로 | 메서드 | 설명 |
|---|---|---|
| `/` | GET | HTML 프론트엔드 |
| `/api/stock?ticker=삼성전자&period=1y` | GET | 종목 상세 분석 |
| `/api/screener?sort_by=price&sort_order=desc` | GET | 종목 스크리닝 |
| `/api/kr/opening-surge` | GET | 국내 개장 급등 후보·VI/시장경보·장중 수급 통합 분석 |
| `/api/kr/opening-surge/performance` | GET | 추천 후 30분·1시간·종가 성과 갱신 및 조회 |
| `/api/toss-overseas` | GET | 토스증권 해외 종목 필터 |
| `/api/sentiment?market=KRX` | GET | 시장 심리 지수 |
| `/api/resolve?q=삼성` | GET | 종목명·코드 검색 |
| `/api/cron` | GET | 캐시 워밍 + 국내 급등 성과 체크포인트 백업 갱신 (매시간) |

`/api/stock` 응답의 `peter_lynch`는 제공된 글의 5개 조건을 개별 종목에 엄격 적용한 결과입니다. EPS CAGR은 Yahoo Finance 연간 손익계산서의 연속 4개 사업연도 희석 EPS만 사용합니다. 연간 EPS가 부족하거나 시작/종료 EPS가 0 이하이면 PEG를 추정하지 않고 `unavailable`로 반환합니다. 이는 장기 펀더멘털 필터이며 단기 가격 예측이나 매수 추천이 아닙니다.

---

## 📂 프로젝트 구조

```
StockOracle/
├── api/index.py               # HTTP 라우팅, 분석 조합, 인라인 HTML/CSS/JS
├── market_briefing/           # 데이터 수집, 신호, 예측·품질·ML 보조 모듈
├── us_price_fetcher.py        # 미국장 세션별 시세 소스
├── models/                  # 학습된 보조 모델과 메타데이터
├── scripts/                 # 학습·백테스트·검증 작업 (audit_prediction_layers.py: 예측 체인 워크포워드 감사)
├── tests/                   # 계산·계약·화면 회귀 테스트
├── docs/                    # 설계·백테스트·분석 문서
├── dev_server.py            # 로컬 서버 (포트 3000)
├── requirements.txt         # 의존성 패키지
└── vercel.json              # 배포 설정
```

---

## 📦 의존성 패키지

```
numpy>=1.26.0           # 수치 계산
pandas>=2.0.0           # 데이터 처리
yfinance>=0.2.36        # 주가 데이터 (KRX·US)
requests>=2.31.0        # HTTP 요청
beautifulsoup4>=4.12.0  # Naver Finance 크롤링
feedparser>=6.0.10      # Google News RSS
lxml>=4.9.0             # XML/HTML 파싱
```

---

## 🛠️ 로컬 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 개발 서버 실행
python dev_server.py

# 3. 브라우저 접속
# http://localhost:3000
```

---

## ☁️ Vercel 배포

```bash
# Vercel CLI 설치
npm i -g vercel

# 배포
vercel
```

GitHub 리포지토리 연동 시 `push` → 자동 배포됩니다.

### 국내 개장 급등 연동 환경변수

| 환경변수 | 용도 | 기본 동작 |
|---|---|---|
| `KR_SURGE_SHORT_OVERHEAT_URL` | 현재 단기과열 지정 종목을 반환하는 공식/증권사 JSON 피드 | `items`, `content`, `data`, `stocks` 배열과 주요 종목코드 필드 자동 인식 |
| `KR_SURGE_SHORT_OVERHEAT_TOKEN` | 위 JSON 피드의 Bearer 인증 토큰 | 미설정 시 무인증 GET |
| `KR_SURGE_SHORT_OVERHEAT_CODES` | 공식/증권사 피드에서 받은 단기과열 6자리 코드를 쉼표로 전달 | 미설정 시 공급자 응답 필드만 검사하며 UI에 커버리지 미연동 표시 |
| `KR_SURGE_TRACKING_PATH` | 추천 성과 이벤트 JSON 저널 경로 | 로컬은 OS 임시 폴더, Vercel은 `/tmp`를 사용하므로 인스턴스 교체 시 유실 가능 |

성과를 배포 환경에서 장기간 보존하려면 `KR_SURGE_TRACKING_PATH`를 지속 볼륨에 연결하거나 별도 DB 어댑터로 교체해야 합니다. 30분·1시간 성과는 예정시각 이후 처음 관측된 서버 가격이며, API 폴링이 중단되면 해당 체크포인트는 관측 불가로 표시됩니다.

### Vercel 설정 (`vercel.json`)

| 항목 | 값 | 설명 |
|---|---|---|
| `maxDuration` | 60초 | 함수 최대 실행 시간 |
| `memory` | 1024 MB | 함수 메모리 |
| `cron` | `0 * * * *` | 매시간 캐시 워밍 |

---

## ⚠️ Vercel 배포 주의사항

- **파일 시스템**: `/tmp` 디렉토리 외 쓰기 금지 → 캐시는 `/tmp` 자동 사용
- **실행 시간**: Hobby 플랜 기본 10초, `vercel.json`으로 최대 60초 설정
- **패키지 제한**: TA-Lib·Prophet 등 C 컴파일 패키지 설치 불가 → 전 기능 순수 Python 구현으로 대체
- **캐시 전략**: `/api/stock` 60초, `/api/screener` 1시간, HTML 1시간 캐시 적용
