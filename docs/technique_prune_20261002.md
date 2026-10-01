# 기법별 손실조건 제거(Subtractive Pruning) 설계 및 실행 기록

- 기준일: 2026-10-02
- 스크립트: `scripts/prune_loss_conditions.py`
- 규칙 조회: `market_briefing/technique_prune.py` → `models/technique_prune.json`
- 테스트: `tests/test_technique_prune.py`

## 방법 (요청 ①→④ 그대로)

1. 같은 조건의 매매를 기법·시장별로 최소 20번 반복 재현한다 (`MIN_TRADES=20`).
   미달이면 판정 없이 `INSUFFICIENT` 로 보고하고 규칙을 만들지 않는다.
2. 진입 자리·손절 자리·결과를 전부 기록한다.
   진입=신호 다음 봉 시가, 손절=기법 고유 손절가, 목표가=진입+2.0×ATR14
   (ML만 1.5×ATR, 모델 호라이즌 14일에 맞춤). 같은 봉 목표·손절 동시 도달은
   손절 우선(기존 `test_stratified_backtest.py` 규약과 동일). 비용은
   KRX 0.20% / US 0.10% 왕복으로 `dynamic_rsi` 기준과 통일했다.
   전 매매는 `docs/backtests/technique_prune_trades.csv` 에 저장된다.
3. 손실이 가장 자주 발생한 조건을 찾아낸다.
   손실건수 최대(동률이면 손실율 최대), 표본 5건 이상, 전체 손절율 초과 조건만 후보.
4. 그 조건부터 하나씩 제외한다. 손절율이 실제로 낮아지고,
   매매 수가 20건 이상 유지되며, 기대값이 기준 대비 0.10%p 넘게
   나빠지지 않을 때만 유지를 확정한다. 아니면 중단하고 이전 상태로 둔다.

핵심 원칙: 수익 기법을 새로 찾지 않고, 계속 손실나는 조건을 지워서
남는 것만 쓴다. 제거 후에도 기대값이 0 이하면 기법 전체를 `DROP` 한다.

## 대상 기법 5종 (최근 1년, KRX·US 각각 검증)

| 기법 | 신호 원천 | 손절 | 기록 조건 |
|---|---|---|---|
| hybrid_breakout | `compute_hybrid_score` AUTO_YES | 기법 손절가 | NCS/FWS 구간, 레짐, 변동성레짐, ADX, 고점거리 |
| pattern_breakout | `PatternEngine` 확정 상승 패턴 | 패턴 무효화가 | family, 완성도, 허용오차 모드 |
| dynamic_rsi | `DRSI_Signal==1` | DRSI 손절가 | 시장, RSI 구간, 손절거리 |
| leader_reversal | `detect_leader_reversal` BREAKOUT | 조정저점-0.5ATR | RS 초과폭, 조정폭, 벤치 유무 |
| ml_direction | `predict_from_ohlcv` UP·신뢰도≥0.60 | 진입-1.5ATR | 신뢰도·상승확률 구간, 모델/폴백 |

데이터는 yfinance 1년 일봉을 `datasets/prune_cache/` 에 캐시하고
`--offline` 으로 네트워크 없이 재실행한다. 기존 백테스트 스크립트들이
네트워크 필수였던 점이 이번에 발견된 문제라 캐시·오프라인 경로를 new로
넣었다. 유니버스는 시장별 대형 10·중형 8·소형 8종(`--tiers` 로 선택,
`--limit` 으로 티어별 축소)이며 매매마다 `cap_tier` 조건을 함께 기록한다.
벤치는 KRX `^KS200` / US `SPY`, ML 지수 캐시는 중립값 고정으로
네트워크 변수를 제거했다.

## 2026-10-02 실행 결과 (대형·중형·소형 26+26종)

| 기법 | 시장 | 매매 | 손절율(제거 전→후) | 기대값(제거 전→후, %p) | 판정 |
|---|---|---|---|---|---|
| hybrid_breakout | KRX | 98 | 59.18% → 59.18% | +0.51 → +0.51 | KEEP(개선 조건 없음) |
| hybrid_breakout | US | 445 | 68.54% → 60.84% | -0.09 → +0.35 | PRUNED_KEEP(`adx_bucket=>=30` 제외) |
| pattern_breakout | KRX | 156 | 55.77% → 47.56% | -3.24 → -0.59 | DROP(제거 후에도 기대값 ≤ 0) |
| pattern_breakout | US | 159 | 42.77% → 29.03% | +0.50 → +5.18 | PRUNED_KEEP(`completion 70-85`, `cap_tier=LARGE` 제외) |
| dynamic_rsi | KRX/US | 11/15 | — | — | INSUFFICIENT |
| leader_reversal | KRX/US | 4/5 | — | — | INSUFFICIENT |
| ml_direction | KRX/US | 17/24 | — / 54.17% | +2.68 / +0.28 | INSUFFICIENT / KEEP |

해석: 중·소형 확대 후 `cap_tier=LARGE` 제외가 양 시장 패턴 기법에서
도출됐다(대형주 패턴 돌파의 손실 집중). `pattern_breakout` KRX는 제거
후에도 기대값이 음수라 전체 제외, US는 2조건 제거로 손절율이 42.77%에서
29.03%로 낮아져 유지 판정이다. DRSI·리더반전은 여전히 20건을 못 채워
판정을 유보했다(추정하지 않음). 규칙 파일의 `status` 와 `exclusions` 가
최종 판정이며, `technique_allowed()` 로 조회한다.

## 실전 적용 (2026-10-02 연결)

- 패턴 목표가 병합(`api/index.py` TP 계산): `pattern_breakout` DROP·제외
  family 패턴은 병합에서 빠지고 `pruned` 건수가 응답에 남는다.
- 미국 스캔 리더 승격: 제외 판정된 BREAKOUT은 승격하지 않고
  `leader_pruned_count` 로 집계한다.
- ML 상승 예측(`_get_ml_prediction`): 제외 판정이면 NEUTRAL로 강등하고
  `pruned` 표식을 남긴다.
- 규칙 파일이 없으면 전량 허용이라 기존 동작과 동일하다 (fail-open).
- `hybrid_breakout` 은 스캔 본체라 게이트 대상에서 제외했다. DROP 판정이
  나오면 스캔 재설계가 필요하므로 침묵 차단하지 않는다.

## 알려진 한계

- 실전 패턴 게이트는 엔진 고유 키(family·completion·tolerance)만 판정한다.
  분석 경로(`calc_risk`)에 실시간 시총 값이 없어 `cap_tier` 제외 규칙은
  실전에서 집행되지 않는다. `pattern_breakout` KRX 전체 DROP은 완전 집행,
  US `completion 70-85` 제외는 집행, US `LARGE` 제외는 미집행이다.
- 제외 조건은 검증 데이터를 고른 표본(대형·중형·소형 지정 종목, 1년 일봉)에
  종속된다. 표본을 바꾸면 판정이 바뀔 수 있어 규칙 파일 생성일자를 확인하고
  주기적으로 재실행해야 한다.
- 통일 청산(TP 2.0ATR·20봉)은 기법 간 비교용이며 기법 고유 청산과 다르다.
  특히 `dynamic_rsi` 동적 청산의 고유 엣지는 이 검증에서 측정되지 않는다.

## 시스템 감사·최적화 기록 (2026-10-02)

| # | 발견된 문제 | 수정 |
|---|---|---|
| 1 | 실전 게이트가 스캔당 종목·패턴 수만큼 규칙 JSON을 매번 읽음 (I/O 증폭) | `technique_prune.load_rules()` 60초 TTL 캐시. 200회 호출 0.031s → 0.001s |
| 2 | `datasets/prune_cache/` 가 무기한 재사용이라 오래된 봉으로 검증됨 | 24시간 TTL 경과 시 재수집 (`--offline` 은 캐시만) |
| 3 | `--tickers` 지정 시 `--market` 필터가 무시됨 | 지정 경로에도 동일 필터 적용 |
| 4 | 동작하지 않는 `--full` 플래그 존재 | 제거 (기본값이 이미 전 티어 전체) |
| 5 | `INSUFFICIENT` 경로에서 매매 시뮬레이션 2회 실행 | `kept_trades` 항상 반환, 중복 제거 |
| 6 | 컴파일된 `__pycache__/*.pyc` 가 git에 추적됨 | 인덱스에서 제거 (`.gitignore` 적용됨, 작업 파일은 유지) |

진입가=진출 봉 시가 구조라 갭 필 공백 청산 케이스는 발생할 수 없음을
확인했다 (진입가가 항상 손절·목표가 사이에 있어 별도 처리 불필요).

## 재실행

```bash
python scripts/prune_loss_conditions.py                 # 대형·중형·소형 전체
python scripts/prune_loss_conditions.py --offline       # 캐시만 사용
python scripts/prune_loss_conditions.py --tiers LARGE,MID  # 티어 선택
python scripts/prune_loss_conditions.py --limit 3       # 티어별 종목 수 축소
python scripts/prune_loss_conditions.py --technique hybrid_breakout --market US
```
