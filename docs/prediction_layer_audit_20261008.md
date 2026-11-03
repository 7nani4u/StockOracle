# 예측 체인 감사 — 중복·충돌·종속 분석과 수정 (2026-10-08)

> 후속(2026-10-09): 이 문서가 "보류"로 남긴 방향 확률 표시, 시나리오 이중 가산, 레짐 라벨, 그리고 1차·2차 매수 구간의
> 항상 표시·단계별 확률 재설계는 `docs/entry_band_and_calibration_audit_20261009.md` 에서 처리했다.

## 한눈에

- `/api/stock` 의 점수 → 확률 → 보정 → 예측 탭 체인을 52종목(KRX 26·US 26)·2022-10~2026-09·22거래일 선행으로
  **그대로 재현**해 층마다 AUC·Brier·구간 포함률을 쟀다. 재현 도구는 `scripts/audit_prediction_layers.py`,
  결과는 `docs/backtests/prediction_layer_audit_summary.json` 이다.
- 잘 맞는 것은 **변동성 기반 범위**(P10~P90 포함률 78.9%, 명목 80%)와 **변동성 터치 확률**(AUC 0.72)이다.
  **방향 확률은 무정보에 가깝다**(AUC 0.511, 95% 구간 0.494~0.527, 보정 기울기 0.05). 이번 수정으로 방향 판별력이
  좋아지지는 않았고, 같은 근거가 3~4번 겹쳐 세지던 구조와 조용히 상수가 되던 입력 때문에 생기던 편향·과신·범위 왜곡을 없앴다.
- 조건·임계값은 새로 추가하지 않았다. 수정은 (1) 명백한 결함 교정, (2) 근거 없는 보정을 기본 꺼짐으로 전환(되돌릴 수
  있는 스위치), (3) 한 곳에만 있어야 할 규칙의 단일화다.

## 방법과 한계

| 항목 | 내용 |
|---|---|
| 표본 | 8,368 시점(종목당 약 160개, 6거래일 간격), 52종목 대형·중형·소형 혼합 |
| 입력 | 서비스와 같은 252봉 + 인과적 지표(`add_indicators`), 같은 함수 호출(`analyze_score`·`calc_probability`·`calc_target_price`·`correlate_and_narrow`·`build_prediction_outlook` …) |
| 평가 | t 이후 22거래일 종가 방향·최고가·최저가 |
| 신뢰구간 | (종목, 월) 블록 부트스트랩 — 22거래일 구간이 겹치므로 일반 표준오차를 쓰지 않음 |
| 한계 | 투자자 수급·ML·뉴스·거시·섹터·실적·실시간 시세는 오프라인 재현이 불가능해 제외했다. 따라서 결과는 '기술적 입력만의 체인' 성능이며 제외 입력의 추가 정보는 검증되지 않았다. 표본은 대부분 강세 구간(22거래일 평균 +3.9%, 상승률 56.7%)이라 방향 지표의 절대 수준은 구간에 의존한다. 생존 편향이 있다(현재 상장 종목만). |

## 시스템 지도 — 같은 근거가 몇 번 세어지는가

```
analyze_score ─ 추세·모멘텀·변동성·거래량·패턴 ─┐
check_market_regime(BEAR→≤40) ─────────────────┤
KRX 수급(±5) · HybridTurtle NCS(+5/−10) ───────┼→ score ─→ calc_probability(score, 추세, 유사패턴, 변동성)  [L0]
                                                │               └→ ML 블렌드(≤15%)
                                                └→ build_signal_confidence(기술=score, AI=NCS, 시장=레짐)
correlate_and_narrow: 기술(score)·패턴·수급·ML·신뢰도·레짐·깊이 7표 → 확률 당김·범위 좁히기  [L2]
build_prediction_outlook: score·수급·패턴·레짐·RSI·등락률을 다시 가산 → 시나리오 비중  [L3]
```

점수·레짐·수급·NCS 가 점수(L0)와 신뢰도, 상관 표(L2), 시나리오(L3)에 각각 다시 들어간다. 측정상 이 반복은
정보를 더하지 못했다(아래 표).

## 발견한 결함과 수정

| # | 증상(측정) | 원인 | 수정 | 회귀 테스트 |
|---|---|---|---|---|
| 1 | 하이브리드 레짐이 8,368/8,368 시점에서 SIDEWAYS, FWS 최저 10.0 | 벤치마크 없이 `ma200 or cur_price` 로 종목 자신의 가격을 넣어 price==MA200 → 약세 3점+CHOP 고정, `regime_stable=False` → FWS +10 상수 | MA200 미보유는 '알 수 없음'으로 두고 ①·⑤ 건너뜀 (`hybrid_signals.compute_hybrid_score`, `compute_regime`) | `test_hybrid_without_benchmark_*`, `test_compute_regime_skips_ma200_checks_when_unknown` |
| 2 | 점수가 평균 −8.6 이동, 표본 72%에 −10 | route 가 `ncs_action`(AUTO_NO=FWS>65)에 `NCS<40` 을 덧붙임. NCS 는 돌파 품질이라 비돌파 종목 72%가 40 미만 | `finalize_rule_score` 가 `hybrid_signals.ncs_action` 하나만 사용 | `test_non_breakout_stock_is_not_penalized_as_weak` |
| 3 | BEAR 40점 상한이 수급(+5)·NCS(+5) 뒤에 50점까지 | 상한을 가산보다 먼저 적용 | 가산 → 상한 순서를 `finalize_rule_score` 로 고정 | `test_bear_cap_is_not_pierced_*` |
| 4 | 신호 일치도 중앙값 1.0, 표본 51%가 범위 축소 하한 0.55 | 일치도 분모가 '방향이 있는 표'만 포함 → 표 하나만 켜져도 100% (점수 50 의 AAPL 에서 "일치도 100%·범위 45% 축소") | 기권표를 분모에 포함: `0.5+0.5·|pos−neg|/total` (중앙값 0.62, 하한 비율 0%) | `test_single_weak_vote_*` |
| 5 | SELL 신호의 높은 신뢰도가 '상승 표'로 집계 | confidence(50~88)는 중립에서의 거리일 뿐인데 부호 없이 투표 | 신호 방향(BUY +1/SELL −1/그 외 0)을 곱함 | `test_confidence_vote_is_negative_for_sell_*` |
| 6 | 상관 보정 후 `prob_up+prob_down=90`, up/(up+down) 약 11% 과대 | `side_base=0` 인데 하한 10 을 적용해 상수 '횡보 10%' 생성 | 호출측이 횡보를 주지 않으면 0 | `test_complementary_probabilities_*` |
| 7 | `reach_probability` AUC 0.31(역상관), Brier .298 > 상수 .187 | 추세가 강할수록 min_target(+4ATR)이 멀어지는데 확률은 올라가는 가점식 | 예측 탭과 같은 무추세 변동성 터치 확률로 교체, 범위가 바뀌면 최종 하단 기준으로 재계산, 신뢰도·학습 가감 제거 | `test_target_reach_probability_*` |
| 8 | 범위 좁히기 후 실현 최고가 포함률 30.1%→21.0% | 일치도(포화) 기반 범위 축소 | 기본 꺼짐 (`STOCKORACLE_CORRELATION_NARROWING=1` 로 켬). 일치도 수정 후에도 5분위별 기준 범위 포함률이 0.30·0.33·0.30·0.30·0.28 로 평평해 정보 없음 | `test_range_narrowing_is_off_by_default_*` |
| 9 | 하이브리드 ADX 가 `add_indicators` ADX 와 최대 ±14 어긋남(표본 9%가 25 기준 반대편) | 마지막 14개 DX 단순평균 | Wilder 평활 (차이 ≤ 0.005) | `test_hybrid_adx_matches_wilder_adx_*` |
| 10 | 지수 일봉 마지막 행 종가 NaN 이면 BEAR 구조도 NEUTRAL (2026-10-08 실제 ^KS11·^KQ11) | NaN 비교가 모두 False, 실패 결과가 15분 캐시 | `dropna` 후 계산, 실패는 캐시하지 않음 | `test_index_regime_*` |
| 11 | KRX 스캔 레짐·RS 상수, `fetch_sentiment('KRX')` None, 평일 장중 지수 '상태 확인 불가', 시장면역 'KOSPI 200' 옵션 무동작 | Yahoo `^KS200` 은 history 1행 (`ml_predictor` 만 알고 있었음) | `_benchmark_history` (`^KS11`→`069500.KS`), overnight·reconcile·prune·예제 스크립트 동일 처리 | `test_benchmark_history_*`, `test_krx_sentiment_*`, `test_reconcile_uses_kospi_composite_*` |
| 12 | 학습 로그 71개 예측 중 고유 (종목·날짜) 44개, AAPL 하루 10건 | 예측 id 에 장중 현재가 포함 → 새로고침마다 새 표본 | 하루 1건만 기록, 평가·집계는 (종목·신호일·구간) 중복 제거 | `test_prediction_is_recorded_once_*`, `test_learning_adjustment_counts_unique_*` |
| 13 | `correlation_engine` 의 `... and volume_ratio if 'volume_ratio' in locals() else ...` | 연산자 우선순위로 거래량 배수 미검사 | `volume_ratio >= 1.2` | `test_causal_map_claims_volume_confirmation_*` |

## 측정 결과 (전체 8,368 시점, 수정 전 → 수정 후)

| 지표 | 수정 전 | 수정 후 |
|---|---|---|
| 하이브리드 `regime_stable` 비율 | 0% | 100% |
| 점수 − 기술점수 평균 | −8.62 | −1.30 |
| BEAR 구간에서 점수 > 40 | 1.6% | 0.0% |
| `prob_l0` 평균 / 실현 상승률 | 47.3% / 56.7% | 50.8% / 56.7% |
| `prob_l0` Brier(BSS) | .2807 (−.143) | .2732 (−.113) |
| `prob_final` AUC / Brier(BSS) | .5059 / .2839 (−.156) | .5081 / .2713 (−.105) |
| 시나리오 up/(up+down) AUC / Brier(BSS) | .4996 / .3104 (−.264) | .5004 / .2964 (−.207) |
| 신호 일치도 중앙값 / 하한 비율 | 1.00 / 51% | 0.62 / 0% |
| 목표가 범위의 실현 최고가 포함률 | 21.0% | 31.1% (기준 범위 30.1%) |
| 상승 시나리오 하단 터치 확률 Brier (상수 기준) | .1837 (.2056) | .1619 (.1786) |
| P10~P90 / P05~P95 포함률 | 78.9% / 87.9% | 78.9% / 88.1% |
| `target_price.reach_probability` AUC / Brier (상수 .187) | .309 / .298 | .722 / .167 (같은 하단 가격의 터치 확률) |

방향 AUC 의 신뢰구간(0.49~0.53)은 0.5 를 포함한다. 규칙 점수 자체의 AUC 는 0.495, 추세 가점 0.493, 변동성 항 0.517
(저변동일수록 상승 — 유일하게 약한 신호), 유사 패턴 항 0.507 이다. 따라서 방향 확률을 높이려고 조건을 더하는 것은
근거가 없다. 의사결정 라벨 '주의·매수 보류'는 상승률은 같지만 8% 이상 하락 비율이 44.5%로 다른 라벨(40.0~41.1%)보다
높아 하방 위험 정보는 약하게 가진다.

## 하지 않은 것과 남은 충돌

1. `build_prediction_outlook` 의 시나리오 비중(L3)이 점수·수급·패턴·레짐·RSI·당일 등락률을 한 번 더 가산한다. 측정상
   L0 보다 AUC(.500 vs .511)·Brier(.296 vs .273)가 나쁘지만, 기존 테스트
   (`test_krx_outlook_uses_final_score_flow_and_selected_horizon`)가 이 반응을 의도로 고정하고 있어 바꾸지 않았다.
2. 방향 확률의 규모(10~91%)는 보정 기울기 0.05 와 맞지 않는다. 표시를 실현 빈도에 맞추는 보정(isotonic 등)은
   화면의 의미를 바꾸는 결정이라 사용자 판단이 필요하다.
3. 레짐 개념이 3개다: 대표지수 MA60/120 구조(점수 상한·신뢰도), HybridTurtle MA200+ADX+VIX(스캔), 종목 단독(하이브리드, 이제 중립).
   같은 날 KOSPI 가 앞의 것으로는 BEAR, 뒤의 것으로는 BULLISH 일 수 있다 — 시간축이 달라 버그는 아니지만 라벨 분리가 필요하다.
4. `hybrid_signals` 와 `dual_score_v2` 의 BQS/FWS/NCS 이중 구현은 남아 있다(상세 화면 vs 스캔 표). 공유 하위 점수는
   일치하고, 추격 FWS(ext_atr vs 최근5일 플래그)·주간 ADX·`dual_aligned` 기본값(True vs False)이 다르다.
5. `calc_probability` 의 유사 패턴 항은 '목표 터치 빈도'(평균 0.71)를 방향 확률에 섞어 표본 72%에 +6.7p 상수를 더한다.
6. ML 블렌드 가중은 holdout AUC(0.569)로 정하지만 walk-forward 평균은 0.546, 보정 OOF 는 0.515 이다.
7. `scripts/` 의 `debug_*.py`·`check_*.py`·`test_multiclass.py`·`test_setB.py`·`auc_*.py` 는 문서·테스트가 없는 일회성 탐색
   스크립트이고, `datasets/`(17MB parquet 포함)는 `.gitignore` 대상이 아니다.

## 재현과 되돌리기

```bash
python scripts/audit_prediction_layers.py --offline --stride 6      # 캐시 사용(없으면 먼저 --offline 없이 실행해 5년 일봉 수집)
python scripts/audit_prediction_layers.py --tickers AAPL,005930.KS --stride 3
python -m pytest tests/test_prediction_chain_integrity.py tests/test_prediction_layer_audit.py -q
```

| 되돌릴 대상 | 방법 |
|---|---|
| 상관 엔진 범위 좁히기 | `STOCKORACLE_CORRELATION_NARROWING=1` |
| 상관 엔진 확률 당김 끄기 | `STOCKORACLE_CORRELATION_PROB_PULL=0` |
| 그 외 | 결함 수정이므로 되돌릴 스위치를 두지 않았다 (`git diff` 로 위 표의 함수만 확인) |

변경 파일: `api/index.py`, `market_briefing/{hybrid_signals,correlation_engine,data_fetcher}.py`,
`scripts/{audit_prediction_layers,prune_loss_conditions}.py`, `tools/run_scan_example.py`, `.gitignore`,
`docs/prediction_output_contract.json`, `tests/test_prediction_chain_integrity.py`, `tests/test_prediction_layer_audit.py`,
`tests/test_market_core_indices.py`.
