# 현재가 기반 예측·규칙 진단 개선 TDD 증적

- 기준일: 2026-09-23
- 범위: `GET /api/stock`, 예측 탭 가격 계약, 규칙 기반 진단 출력
- 방법: 새 동작을 계약 테스트로 먼저 고정한 뒤 최소 구현과 회귀 검증 수행

## 사용자 여정과 테스트 매핑

| 사용자 여정 | 위험 | 회귀 테스트 |
|---|---|---|
| 사용자가 예측 탭에서 현재가·진입가·목표가·손절가를 비교한다 | 섹션마다 서로 다른 현재가를 사용 | `test_price_anchor_is_propagated_to_every_prediction_section` |
| 시간외 시세로 예측을 확인한다 | 시간외 가격을 확정 일봉 지표에 섞어 미래 정보를 암묵적으로 반영 | `test_price_anchor_distinguishes_live_quote_from_confirmed_indicator_bar` |
| 실시간 소스가 실패한 상태로 결과를 본다 | 최근 일봉 종가를 실시간 현재가로 오인 | `test_daily_history_fallback_is_not_labeled_as_live_quote` |
| 동일 거래일 정규장 가격이 마지막 봉에 반영된다 | 종가는 바뀌지만 EMA·RSI는 과거 값으로 남음 | `test_same_session_current_bar_refreshes_price_dependent_indicators` |
| 규칙 기반 진단을 위에서 아래로 읽는다 | 결론보다 세부 근거가 먼저 나오고 같은 근거가 반복 | `test_rule_diagnosis_uses_decision_first_order_with_named_sections`, `test_rule_diagnosis_semantically_deduplicates_summary_lines` |
| 서버 처리 순서가 리팩터링된다 | 규칙 점수·리스크가 현재가 확정 전에 다시 계산됨 | `test_route_finalizes_current_price_before_rule_analysis` |
| 예측 탭을 연다 | 가격 기준이 상세 예측 뒤에 묻힘 | `test_prediction_tab_renders_price_anchor_before_forecast_details` |

## RED 증적

1. 가격 계약 테스트 작성 직후 `ModuleNotFoundError: market_briefing.prediction_contract`로 실패했다.
2. 처리 순서 테스트는 `analyze_score`가 현재가 앵커 생성보다 앞서 있어 실패했다.
3. 동일 거래일 지표 갱신 테스트는 `_refresh_current_bar_indicators`가 없어 실패했다.

각 실패는 환경·문법 문제가 아니라 아직 구현하지 않은 요구사항을 직접 가리켰다.

## GREEN 증적

| 검증 | 결과 |
|---|---|
| 현재가·진단 신규 계약 테스트 | 8 passed |
| 신규 계약 + 진단 패턴 + 예측 무결성 | 19 passed (경계 사례 추가 전) |
| 예측 관련 5개 테스트 파일 | 89 passed (경계 사례 추가 전) |
| 전체 테스트 스위트 | 295 passed (최종) |
| 신규 계약 모듈 추적 커버리지 | 85.2% |

최종 문서·경계 사례 반영 후 전체 스위트와 `pip check`를 다시 실행했으며 모두 통과했다.

## 리팩터링과 품질 판단

- 가격 수집, 기술지표 계산, 예측 조합 사이의 암묵적 결합을 `price_anchor` 계약으로 명시했다.
- 구조적 지지·저항 가격대는 현재가 차이만으로 비율 이동시키지 않는다. 대신 모든 명시적 `current_price`를 단일화하고 불일치를 `prediction_price_contract`에 기록한다.
- 시간외 가격은 확정 일봉 OHLCV에 주입하지 않는다. 정규장 동일 거래일 가격만 마지막 봉에 반영하고 가격 의존 지표를 재계산한다.
- 진단 중복 제거는 완전 문자열 비교가 아니라 투자자 수급·시장 상태 등 근거 범주와 정규화된 텍스트를 함께 사용한다.

## 잔여 검증 한계

- `api/index.py`는 2만 줄이 넘는 인라인 HTML·라우팅 모놀리스라 신규 경로만의 의미 있는 파일 전체 커버리지 수치를 산출하기 어렵다.
- 로컬 브라우저 초기 화면과 콘솔은 확인했지만, 외부 공급자 호출과 분석 로그 기록을 피하기 위해 실제 종목 분석 버튼을 누르는 E2E는 수행하지 않았다.
- 실시간 가격 공급자의 필드·지연·휴장 처리 품질은 공급자 응답에 의존하므로 운영 관측이 추가로 필요하다.
- 별도 방향 모델의 기존 검증 AUC가 약 0.569인 상태이므로 보조 신호 지위를 유지했다.

## 커밋 정책

ECC 변경 워크플로의 Gate 2 승인 전에 커밋하지 않는 규칙을 우선 적용했다. 따라서 RED/GREEN 증적은 이 문서와 테스트 실행 결과로 보존하며, 사용자 승인 후 하나의 기능 커밋으로 정리한다.
