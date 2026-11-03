# 7단계 스캔 엔진 출력 정리 TDD 증적

- 기준일: 2026-09-23
- 요구사항: 스캔 표의 손절가를 숨기고, `브레이크아웃 품질`을 이해하기 쉬운 명칭으로 바꾸며, 리더 반전 세부 수치와 진입·손절 참고가를 출력하지 않는다.
- 도출 방식: 사용자 요청에서 직접 도출한 화면 계약

## 사용자 여정

| 사용자 여정 | 보장 사항 | 테스트 |
|---|---|---|
| 스캔 결과 표를 읽는다 | 손절가 열은 보이지 않고 BQS는 `돌파 신뢰도`로 표시된다 | `test_scan_table_hides_stop_price_and_uses_plain_breakout_label` |
| 미국 리더 반전 후보를 본다 | 단계 배지는 보이지만 초과수익·조정폭·상단 이격·진입/손절 참고가는 보이지 않는다 | `test_leader_reversal_keeps_signal_but_hides_its_price_plan_and_metrics` |
| 스캔 엔진이 후보를 승격·사이징한다 | 리더 반전의 진입·손절 데이터는 백엔드 계산에 보존된다 | 기존 `tests/test_leader_reversal.py` |

## RED → GREEN

| 단계 | 명령 | 실제 결과 |
|---|---|---|
| RED | `python -m pytest tests/test_scan_engine_rendering.py -q` | 2 failed: 손절가/브레이크아웃 품질 표기와 리더 반전 세부 수치가 기존 화면에 존재함 |
| GREEN | `python -m pytest tests/test_scan_engine_rendering.py tests/test_leader_reversal.py -q` | 15 passed |
| 전체 회귀 | `python -m pytest -q` | 297 passed |
| 의존성 검사 | `python -m pip check` | No broken requirements found |

## 구현 요약

- 스캔 표에서 손절가 열과 셀을 제거하고 빈 결과 행의 `colspan`을 14에서 13으로 맞췄다.
- BQS 표시 이름을 `돌파 신뢰도`로 바꾸고 점수 의미를 툴팁으로 보완했다.
- 리더 반전 표 셀은 단계 배지만 표시한다. 세부 근거·진입·손절 가격은 API 후보 데이터와 Stage 7 계산에 남겨 두었다.
- 미국 스캔의 리더 반전 읽는 법에서도 가격 세부 출력 언급을 제거했다.

## 커버리지와 한계

- 새 Node VM 회귀 테스트는 실제 `renderScanResult` 함수를 샘플 후보 데이터로 실행해 DOM 문자열을 검증한다.
- 인라인 JavaScript가 2만 줄 이상 Python 문자열 안에 존재해 Python 라인 커버리지는 이 UI 변경의 유효한 범위 지표가 아니다. 별도 JavaScript 커버리지 수집기는 현재 저장소에 구성돼 있지 않다.
- 외부 시세 호출이 필요한 실제 스캔 E2E는 실행하지 않았다. 대신 순수 렌더러와 계산 모듈의 단위·회귀 테스트를 실행했다.
