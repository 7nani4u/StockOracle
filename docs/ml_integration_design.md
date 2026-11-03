# StockOracle x StockFlow 통합 설계 문서 (사전 검증 #6)

작성일: 2026-08-29
목표: StockOracle 기존 구조 보존하며 StockFlow의 유효 데이터 처리·feature·모델링·검증 기법 선별 통합

## 1. StockOracle 현재 구조

- **진입점**: `api/index.py` 단일 파일 (24021 라인) — Vercel BaseHTTPRequestHandler `handler`로 HTML + 모든 /api/* 처리
- **데이터 수집**: `fetch_stock_data` (yfinance), `fetch_naver`/`fetch_naver_realtime`, Toss/KRX API, `us_price_fetcher.py` — 세션별 가격 (PRE/REGULAR/POST/OVERNIGHT) 구분, 환율/뉴스 병렬 수집
- **지표 계산**: `add_indicators(df)` — MA5/20/60/120, EMA20/50, MACD, RSI(Wilder RMA), BB, ATR(Wilder), Stochastic, ADX(Wilder), OBV, Aroon, Buy Pressure, PSAR, TRIX, Dynamic RSI. 100% causal rolling, Vercel 호환 (pandas_ta 미사용)
- **종합 점수**: `analyze_score(dd, market)` — trend(35%) / momentum(30%) / volatility(20%) / volume(15%) + quality, 50± 가중합, steps 반환
- **확률**: `calc_probability(score, dd, market)` — base_prob = 0.5+(score-50)*0.007, ATR 변동성·추세점수·유사패턴으로 보정, quality profile로 clamping
- **매수/목표**: `calc_buy_price`, `calc_target_price`, `calc_pullback_analysis`, `calc_risk` — ATR 밴드 A/B/C, 시장별 폭 계수 KRX 1.18/US 0.92, TP1-5 확률/기간
- **신뢰도**: `market_briefing/confidence_engine.py` — macro_regime(VIX/SPX/DXY), sector_relative(XL* ETF), earnings_proximity, disagreement_penalty, confidence_interval
- **스캔 엔진**: `market_briefing/scan_engine.py` 7-stage HybridTurtle — Universe → TechFilters → Classification → Ranking(BQS) → RiskGates → AntiChase → Sizing
- **Hybrid 신호**: `hybrid_signals.py` / `dual_score_v2.py` — BQS/FWS/NCS, regime, Hurst, BIS, adaptive ATR
- **패턴**: `pattern_engine.py` — 피벗/넥라인/ATR 허용오차로 헤드앤숄더 등 13개 정식 패턴
- **성과 추적**: `calc_learning_adjustment` + `prediction_learning.jsonl` — 20거래일 후 bounce/stop/extra_drop 실현률로 depth_extra 보정

## 2. StockOracle 현재 예측 파이프라인 (순서)

```
resolve_ticker → fetch_stock_data(지표 포함) → analyze_score → calc_probability
→ fetch_naver + earnings + confidence_engine.build_signal_confidence
→ calc_buy_price / calc_target_price / calc_risk (+ learning_adjustment)
→ build_prediction_outlook (3 시나리오 확률 합 100) → response JSON
```

- 전 단계가 과거 봉 종가까지만 사용 (미래 참조 없음, 단 history_confidence는 walk-forward로 검증)
- 데이터 누수 리스크: 없음으로 보이나, train/test 분리 없이 heuristic이므로 out-of-sample 검증 체계는 제한적

## 3. StockFlow 주요 기능

- **수집**: `collect_data.py` — 8011 종목 5y OHLCV, checkpoint/batch parquet, 1s 딜레이
- **피처**: `engineer_features.py` v2 — 34개 (RSI_14/21, MACD, BB normalied/price, SMA/EMA normalized, momentum, volatility_20d/ratio(버그 수정 per-row), range/volume, ATR_14, ADX, stochastic, VWAP_distance, OBV_ratio, NIFTY/BANKNIFTY/India_VIX, relative_strength). 모든 rolling은 과거만, BB/MA/ATR은 price로 정규화. Label은 14일 forward_return ±3% dead-zone, shift(-14), NaN dead-zone 제거
- **학습**: `train_model.py` — 시계열 80/20 분할(날짜 기준, 셔플 금지), balanced class_weight, LightGBM(early_stopping 100), Platt scaling 보정, AUC/accuracy/confusion
- **평가**: `evaluate_model.py` — stored vs live AUC, confusion, feature importance, 분포
- **추론**: `agents/ml_predictor.py` — 6mo OHLCV fetch, 동일 피처 계산, 시장별 index 캐시(India NIFTY vs US SPY/QQQ/VIX), feature_cols 순서 엄수, 항상 UP/DOWN 반환
- **오케스트레이터**: `core/orchestrator.py` — async 병렬 4-agent, fetch 후 sentiment/anomaly/ML 동시 실행 → memo_writer
- **기타**: sentiment(VADER), anomaly(z>2, volatility, vol_ratio>2), memo_writer(LLM) — rule-heuristic, StockOracle 대비 단순

## 4. 두 프로젝트 차이

| 영역 | StockOracle | StockFlow | 비교 |
|------|-------------|-----------|------|
| 지표 구현 | Wilder RMA, 순수 Python, ATR/RSI/ADX 일치 | pandas_ta wrapper | StockOracle이 Vercel 친화적이며 이미 버그 수정된 volatility 유사 |
| 피처 정규화 | 부분 (BB/MA 일부) | 전면 price 정규화 + per-row vol60 수정 | StockFlow 방식이 종목 간 비교에 유리 |
| 라벨 | 없음 (heuristic) | 14d ±3% dead-zone | StockFlow가 명확한 지도학습 타깃 제공 |
| 분할 | 없음 | time-based 80/20, 시계열 준수 | StockFlow가 leakage 방지 |
| 모델 | 없음 (Holt-Winters 등 규칙) | LightGBM + calibration | StockFlow가 확률적 예측 추가 |
| 평가 | 분포 시뮬레이션·워크포워드 일부 | AUC, balanced_acc, walk-forward | StockFlow가 표준화된 평가 |
| 시장 지수 | KOSPI/KRX 처리 강함 | India NIFTY 고정 | StockOracle이 KRX/US 처리에서 우세 |
| 감성/이상치 | KR-FinBERT + 키워드, 다중 가중 | VADER + simple z | StockOracle 우세 |

## 5. 도입할 기능 (가치-호환성 기준)

1. **Leakage-safe feature engineering 모듈** — StockFlow 34개 스키마를 StockOracle 순수 Python으로 재구현 (BB/SMA/EMA/ATR 정규화, VWAP, OBV_ratio, stochastic, vol_ratio per-row fix). 기존 add_indicators와 중복 계산 없이 ml_features 별도 모듈로 분리. 효과: out-of-sample 일반화 향상, 종목 간 스케일 제거
2. **14d 방향 분류 ML predictor** — LightGBM/HistGradientBoosting fallback, KRX/US 시장별 index 슬롯 매핑 유지 (호환성), Platt scaling. 효과: heuristic score에 확률적 보정 추가, 15% 블렌딩으로 드리프트 방지
3. **Time-based split + walk-forward 평가** — 80/20 chronological, expanding walk-forward 5-fold. 효과: train/test contamination 방지, 과적합 조기 탐지
4. **학습 파이프라인 스크립트** — `scripts/train_ml_model.py` synthetic+real hybrid, --synthetic-only로 오프라인 검증 가능. 효과: 재현성, 지속 학습
5. **API 응답 확장** — `ml_prediction` 필드 추가, 기존 prob_up에 15% 가중 블렌딩 (confidence>=0.56일 때만). 효과: 기존 인터페이스 보존하며 신뢰도 상승 시에만 반영

## 6. 도입하지 않을 기능과 이유

- **pandas_ta 전체 도입**: Vercel 빌드 제한, StockOracle 순수 Python으로 이미 커버, 중복.
- **VADER 감성, simple anomaly (z>2, PE>50)**: StockOracle confidence_engine의 KR-FinBERT·출처 가중·최근성 감쇠보다 단순, 효과 미미, 중복.
- **8011 종목 전체 수집/checkpoint**: StockOracle KR_STOCK_MAP(60+) + US_TICKERS(130+) 샘플로 충분, 전체 수집은 2-4시간 소요로 유지보수 부담.
- **Groq/Llama memo_writer**: 투자 메모 LLM은 StockOracle의 AI 진단 탭과 중복, 비용/지연 증가, 예측 정확도와 무관.
- **Orchestrator async 병렬**: StockOracle 이미 ThreadPoolExecutor로 fetch_macro_context 병렬화됨, 구조 변경 불필요.

## 7. 수정 대상 파일

- `market_briefing/ml_features.py` (신규)
- `market_briefing/ml_predictor.py` (신규)
- `market_briefing/ml_evaluate.py` (신규)
- `scripts/train_ml_model.py` (신규)
- `market_briefing/__init__.py` (수정: graceful export)
- `api/index.py` (수정: import + _get_ml_prediction helper + calc_probability 블렌딩 + response ml_prediction)
- `requirements.txt` (수정: sklearn, scipy 명시, lightgbm 선택적)
- `models/feature_columns.json`, `training_metadata.json`, `lgbm_model.pkl` (생성/갱신)

## 8. 신규 생성이 필요한 파일

위 신규 4개 + `datasets/training_features.parquet` (학습 시 생성), `models/` (이미 존재한다면 갱신)

## 9. Dependency 변경

- 추가: `scikit-learn>=1.3.0` (HistGradientBoosting, calibration, metrics), `scipy>=1.10.0` (Platt optimize), `joblib` (이미 sklearn 의존성으로 포함)
- 선택: `lightgbm>=4.0.0` — 주석으로 표기, Vercel 빌드 시 필요에 따라 해제. 없으면 sklearn fallback으로 정상 동작 (테스트 완료).
- 기존 의존성 변경 없음, Vercel 호환성 유지 (cpp 컴파일 불필요한 sklearn만으로 동작).

## 10. 예상 데이터 흐름 변화

Before:
```
yfinance OHLCV → add_indicators → analyze_score → calc_probability(휴리스틱) → response
```

After:
```
yfinance OHLCV → add_indicators → analyze_score → calc_probability
                └→ _get_ml_prediction(compute_feature_vector) → 15% blend → ml_prediction 필드 추가 ─→ response
                                   ↑
                        models/lgbm_model.pkl (time-based trained)
```

- 학습 시: raw OHLCV → engineer_ticker_features (shift(-14) label) → time-based split → HistGradientBoosting → Platt scaling → models/*.pkl
- 추론 시: 현재봉 close까지만 feature, 지수 캐시(KRX=^KS11/US=SPY) 30분 TTL, 모델 없으면 heuristic fallback(0.32-0.68)
- 기존 실행 흐름 파괴 없음: ml 실패 시에도 기존 prob_up/score 그대로 반환, 기존 테스트 206개 통과.
