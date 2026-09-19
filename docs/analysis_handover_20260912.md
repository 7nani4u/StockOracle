# StockOracle 1단계 분석 세션 — 구조 지도·실행 그래프·결함 인계 보고서

> **세션 성격**: 수정 없음 / 분석·인계 전용. 후속 `GPT-6 Astra Fast high` 세션이 바로 수정에 들어갈 수 있도록 파일·함수·행 단위 근거와 재현 경로를 구체화한다.
> **기준 커밋**: `main` 로컬은 `origin/main` 대비 102 commits behind, 미커밋 diff 37파일(삽입 31,184 / 삭제 8,349). 보고서의 모든 행 번호·로직은 **현재 워크트리 수정분을 포함한 상태** 기준이다.
> **검증 방식**: `git status/diff` + `api/index.py` 및 `market_briefing/*`, `us_price_fetcher.py`, `dev_server.py`, `vercel.json`, `models/training_metadata.json`, `tests/*` 직접 리딩, 6봉 ATR/Rsi 재현 스크립트 실행, NaN/inf 전파 추적.

---

## 1) 구조 지도 (파일·책임·Vercel 경계)

### 1.1 단일 파일 핸들러 구조

```
vercel.json  --rewrites-->  api/index.py::handler (BaseHTTPRequestHandler)
                              ├── HTML (인라인 JS/CSS, 26k 라인)
                              └── route(path, params)  # /api/* 분기

api/index.py  (1,406KB, 26,450라인)  — 거대 단일 파일
  ├── 의존성/환경: yfinance, pandas, numpy, requests, bs4, feedparser, certifi, platformdirs
  │   - curl_cffi IPv4 강제 + TIMEOUT_MS 30s 몽키패치 (228-245)
  │   - peewee 메모리 DB로 yfinance SQLite 캐시 무력화 (124-137)
  │   - /tmp 강제 (67-79), Vercel memory 1024MB / maxDuration 60s (vercel.json)
  │   - includeFiles: market_briefing/**, models/**, us_price_fetcher.py
  │
  ├── 전역 캐시/유틸
  │   - _CACHE dict + ttl_cache 데코레이터 (263-309) : 키 500개 초과 시 LRU 정리, 실패 결과 스킵
  │   - get_usd_krw(60s), fetch_annual_income_statement(21600s), FIXED_ANALYSIS_PERIOD="1y"(1927)
  │   - _safe_finite_float(5935), _safe_int(5949), _safe_round_int(5974), _is_finite_number(5991)
  │
  ├── KRX 유니버스/검색
  │   - KIND 상장법인목록 + Naver ETF/ETN API 병합 (668-766), 세션 stale-if-error, TTL 1h
  │   - search_krx_security_remote (800-824) ac.stock.naver.com 폴백
  │   - _fetch_krx_security_detail_status (1024-1105): VI/거래정지/관리/상장폐지 공시
  │
  ├── 지표/분석 코어
  │   - add_indicators (1816-1925): MA/EMA/MACD/RSI(Wilder 1/14)/BB/ATR(ewm min_periods=14)/%K/%D/ADX(ewm)/OBV/Aroon/BUY_PRESSURE/PSAR/TRIX + dynamic_rsi
  │   - fetch_stock_data (1931-1985): yfinance history(1y→2y/5y/10y 매핑), KS/KQ 교차 재시도, 분봉→일봉 fallback, 지표 후 tail 절삭, 뉴스 RSS, NaN/inf→None 변환
  │   - analyze_score, calc_probability, calc_pivot_points, calc_indicator_signals, calc_event_risk, calc_risk, calc_buy_price, calc_target_price, calc_pullback_analysis, detect_chart_patterns 등 (중략)
  │   - build_prediction_outlook (14219-14781): 예측 탭 조건부 구조 — 이미 계산된 결과만 재조합
  │   - correlation_engine.correlate_and_narrow (160+라인) 와 charm_ranking.enrich_charm_with_ranks 연동
  │
  ├── 시장/업종/데이터 공급
  │   - market_briefing/data_fetcher.py : fetch_market_indices(KRX Global→Naver→yfinance 3단 폴백), fetch_macro_context(5-way 병렬), fetch_stock_snapshot, fetch_overnight_markets 등
  │   - market_briefing/us_enricher.py : Finnhub/Tiingo/AlphaVantage (env 키 우선, 번역 배치)
  │   - market_briefing/* (confidence_engine, sector_flow, hybrid_signals, pattern_engine, peter_lynch, investment_charm, news_evidence 등)
  │   - us_price_fetcher.USStockPriceFetcher : Yahoo Direct(v7 crumb)+Naver World+yfinance 3중, 세션(OVERNIGHT/PRE/REGULAR/POST/CLOSED) 분기
  │
  └── 핸들러
      - replace_nan_with_none (26146): JSON 직전 NaN→None (단, inf 미처리)
      - _send (26173): gzip(≥1024B), CORS, Cache-Control(경로별 분기), CSP
      - handler.do_GET/do_POST : /api/stock, /api/screener, /api/market/*, /api/peer-outlook, /api/kr/opening-surge, /api/telegram/send 등

market_briefing/  (572KB)
  - data_fetcher.py (1,253라인): 거시/종목 스냅샷, _get_with_retry, _validated_index_entry 검증, _reconcile_index_market_status
  - confidence_engine.py: build_signal_confidence (거시·섹터·실적·뉴스감정·신뢰구간)
  - sector_flow.py, stock_analyzer.py(hybrid), pattern_engine.py, dynamic_rsi.py, charm_ranking.py, correlation_engine.py, investment_charm.py, peter_lynch.py, ml_predictor.py 등
  - __init__.py 41라인 diff: 신규 모듈 export 정리

us_price_fetcher.py  (1,600라인+)
  - 등락률·세션 감지, holidays 기반 NYSE 휴장 판정(2029+ 동적), 4-way 소스 라우팅

dev_server.py (47라인)
  - ThreadingHTTPServer(handler) 로컬 3000 포트, Vercel 미러

scripts/ (backtest_*.py, train_ml_model.py, refresh_charm_universe.py 등)
  - backtest_prediction_zones/target_price_accuracy/arty_smma, ML 학습 파이프라인

models/ (1.2MB)
  - lgbm_model.pkl/txt (582KB), smart_score_universe.json (79KB), training_metadata.json (9.7KB, AUC 0.569, validation passed)
  - feature_columns.json

tests/ (25 파일)
  - prediction_outlook, charm_ranking, peter_lynch, confidence_engine, pattern_engine 등 — 단, short-history(<20봉) 케이스 없음

vercel.json / requirements.txt
  - rewrites 전체를 api/index.py로, memory 1024, maxDuration 60, cron /api/cron 매시
  - requirements: numpy, pandas, yfinance, requests, bs4, feedparser, lxml, holidays, certifi, scikit-learn, scipy, lightgbm

docs/ (backtests/, contracts, ml_design 등)
  - prediction_output_contract.json: buy_price/risk_scenarios 스키마 명세
  - prediction_learning.jsonl (60라인, 실제 예측 적재) — 운영 이력 오염 경로 주의

기타 미커밋
  - market_briefing/correlation_engine.py, charm_ranking.py, investment_charm.py 등 신규 모듈 10개
  - .github/, ATTRIBUTION.md, datasets/, models/ 등 추적 대상 아닌 대량 바이너리 주의
```

### 1.2 배포/런타임 경계

- **Vercel Python Runtime**: 단일 파일이 HTML+API 모두 담당, 프레임워크 없음. `memory=1024MB` 초과 시 OOM, 60s 초과 시 504. 콜드스타트 시 `lightgbm`·`yfinance` import 비용 ~1s.
- **/tmp 쓰기 금지** 외 전체 읽기 전용. `TMPDIR/YF_CACHE_DIR/XDG_CACHE_HOME`을 `/tmp`로 고정해 yfinance 캐시 경로 이탈 방지.
- **번들 크기**: `api 1.4MB + market_briefing 0.6MB + models 1.2MB = 3.1MB` — 250MB 제한 내 충분히 여유. 다만 `__pycache__`가 diff에 포함되어 있어 PR 시 제외 필요(`.gitignore`에 이미 있으나 force 포함된 상태).

---

## 2) 실제 요청 호출 그래프

### 2.1 HTML 초기 로드 (GET `/`)

```
브라우저 GET /  → handler.do_GET → _send(HTML, text/html, s-maxage 3600) → 브라우저 JS init
  ├─ _start_toss_prewarm_once()  (1회, 백그라운드 Thread, ENABLE_TOSS_PREWARM=0이면 스킵)
  ├─ loadMarketCore()            → GET /api/market/summary
  ├─ loadSectorFlow()            → GET /api/market/sector-summary (+ sector-top-stocks lazy)
  ├─ initAlerts()
  └─ updatePeriodGuide()         // FIXED_ANALYSIS_PERIOD 표시
```

### 2.2 자동완성 검색 (GET `/api/search?q=` 또는 클라이언트 로컬)

```
입력 → search_krx_security_remote (TTL 300s, ac.stock.naver.com) → _enrich_krx_suggestion_status
  └→ _krx_status_directory → _fetch_kr_surge_risk_flags (관리/VI/거래정지 스냅샷 재사용)
     → 결과: code/name/ticker + is_leveraged/is_inverse/is_new_listing + risk flags
```

### 2.3 종목 분석 (GET `/api/stock?ticker=XXX&period=1y`)

> 현재 `period`는 강제 `FIXED_ANALYSIS_PERIOD=1y` 로 덮인다 (26171, 26428). 쿼리 `period`는 무시됨 — 차트 기간 선택 UI가 있더라도 실제 분석은 항상 1y.

```
handler.do_GET /api/stock?ticker=XXX
  ├─ resolve_ticker(raw) → (ticker, market, company)  // KRX .KS/.KQ 보정, 별칭 매핑
  ├─ get_krx_security_status(ticker)  // KRX만: trading_halt/management/delisting/halt_release
  │    └─ analysis_blocked이면 즉시 {error, error_code=KRX_SECURITY_BLOCKED} 반환
  ├─ fetch_stock_data(ticker, market, "1y", company)  [TTL 120s]
  │    ├─ _yf_history_retry(sym, fetch_period="2y", interval="1d", tries=2)  // 429/timeout 재시도, 0.35~0.6s 백오프
  │    ├─ 빈 결과면 KS↔KQ 교차 재시도 (KRX 전용)
  │    ├─ columns 정규화 → add_indicators (1816) → dropna(subset=["Close"]) → 요청 기간 252봉 tail 절삭
  │    ├─ news: Google News RSS(market별 언어 분기) + Yahoo RSS 12건 수집, 중복 제거
  │    └─ df2.where(pd.notna)->to_dict + inf/NaN→None 루프 (2167-2181)  // ★ inf 잔류 버그 하단 참조
  ├─ close/prev/pct 계산, analyze_score → score/steps/patterns/geo_patterns/ai_strategy
  ├─ calc_probability(score) → prob_up/down  // 초기값 (후에 최종 보정)
  ├─ debt_health( yf.Ticker.info, TTL 3600) → score 조정
  ├─ [병렬 12s] fetch_naver(sym) / fetch_us_enriched(sym) / fetch_toss_industry_info(sym)  (15793-15812)
  │    └─ toss_industry가 비면 yfinance info sector/industry로 폴백 (15844-15851)
  ├─ 현재가 보정 (15885-15963)
  │    ├─ KRX: fetch_naver_realtime(10s TTL) 우선, 30% 이격 가드
  │    └─ US: USStockPriceFetcher.fetch(sym) → (overnight/pre/regular/post) → 타임스탬프 8시간 신선도 검증
  │          └─ 실패 시 yf.fast_info.last_price 폴백
  ├─ 차트·분석 일관성 보정 (15965-15997): dd Close 마지막 값을 보정가로 교체 + High/Low 확장 (얕은 복사)
  ├─ investor_flow (KRX) → score ±5 보정
  ├─ atr 추출 (16026-16036): _safe_finite_float(atrs[-1]) → 미확보면 last*0.02 fallback + 로그
  ├─ calc_pivot_points / calc_indicator_signals / calc_event_risk / build_weekly_analysis_context
  ├─ calc_risk(last, atr, market, dd, event_risk, ...) // atr 기반 손절/진입 밴드
  ├─ [일봉 확보] arty_dd: 이미 220봉이 있으면 재사용, 아니면 fetch_arty_daily_data
  ├─ dynamic_rsi_daily_snapshot(arty_dd)
  ├─ enrich_with_hybrid (NCS/BQS/FWS) → score ±10 보정, regime BEARISH면 cap 40
  ├─ calc_probability 재계산 (최종 score 반영)  (16111)
  ├─ _get_ml_prediction (60봉 미만이면 None) → AUC 기반 가중치 0~25% 블렌딩 → prob_up/down 보정
  ├─ calc_buy_price (atr/스프레드/공시 등 반영)
  ├─ build_signal_confidence (technical/ai/market + pct5d + news relevance 필터) // 필터: 회사명 핵심 단어 1개 이상 포함만
  ├─ [병렬 2] fetch_scalp_period_comparison + fetch_execution_quality
  ├─ calc_target_price → _apply_signal_confidence_to_target → _apply_learning_adjustment → long_term
  ├─ calc_pullback_analysis
  ├─ correlate_and_narrow (correlation_engine) → prob_up/down, target min/max, confidence 보정
  ├─ build_prediction_outlook (14219) [try/except 격리] → atr_is_observed 플래그 전달, _expected_days는 safe guard
  ├─ prediction_outlook.decision이 caution이면 buy_price.strategy_rec를 하향 (wait_support)
  ├─ _record_prediction_and_update_outcomes (학습 로그 적재)
  ├─ pattern_overlays, investment_charm/charm_ranking/peter_lynch
  └─ 응답 조립 → replace_nan_with_none → json.dumps → _send (gzip+cors+cache no-store)
       ├─ chart_data: dates/open/high/low/close/volume/ma20/ma60/bb/rsi/dynamic_rsi/macd + overlays
       ├─ score/prob_up/down/ml_prediction/risk_scenarios/pivot/indicator_signals/dynamic_rsi
       ├─ buy_price/target_price/pullback_analysis/prediction_outlook/market_regime
       ├─ news/naver/us_enriched/toss_industry/investor_flow/hybrid_score/signal_confidence
       ├─ data_quality (bars/as_of/source/warnings), price_correction, chart_analysis_consistent
       └─ weekly_analysis/session_name/security_status/investment_charm/key_metrics/peter_lynch/correlation
```

### 2.4 거시 요약 (GET `/api/market/summary`)

```
fetch_macro_context [가변 TTL: 장중 3m / 미국장 5m / 장외 10m, _MACRO_LOCK]
  ├─ ThreadPool max_workers=5 병렬:
  │    ├─ fetch_market_indices: KRX Global → Naver → yfinance 3단 폴백, _validated_index_entry(산술 검증) + _reconcile_index_market_status
  │    ├─ fetch_fx (Naver marketindex)
  │    ├─ fetch_overnight_markets (yf.download 9종목 배치)
  │    ├─ fetch_upbit_crypto (BTC/ETH)
  │    └─ fetch_macro_news (mainnews → newsList → Google News RSS 3단 폴백)
  └─ 캐시 저장 (_MACRO_CACHE)
```

### 2.5 업종 흐름 (GET `/api/market/sector-summary` / `sector-top-stocks`)

```
sector_flow + data_fetcher.fetch_stock_list_quote_cached (TTL 장중 3m)
  - quote만 병렬(6~15 workers), history/news 스킵으로 고속화, 클라이언트도 30s max-age 캐시
```

### 2.6 동종업계 전망 (GET `/api/peer-outlook?ticker=&sector=&industry=`)

```
build_peer_industry_outlook
  ├─ _resolve_peer_group (static catalog → yahoo related fallback)
  ├─ yf.download 3mo 일봉 배치 → _peer_momentum_signal per ticker → 평균 prob/breadth/return 계산
  └─ relative_to_industry = selected prob - peer avg
```

### 2.7 기타

- `GET /api/screener` : 7단계 스캔 (SCAN_COLLECT_CAP 48, workers 16, budget 42s), sector/marketcap 다양성 MMR
- `GET /api/kr/opening-surge` / `performance` : KR surge 모듈 (Naver polling + 위험 플래그)
- `POST /api/telegram/send` : 동일출처 + IP 레이트리밋(60s 5회) 후 Telegram Bot 전송

---

## 3) 우선순위별 결함 (파일:행 + 실행 근거)

> 표기: **[심각도]** `파일:행` — 증상 → 근거 → 재현/로그

### P0 — 서비스 장애 또는 잘못된 JSON/계산으로 즉시 수정 필요

#### P0-1 `api/index.py:26146-26169` replace_nan_with_none — `inf/-inf` 잔류 → JSON `Infinity` 무효

- **증상**: ATR·RSI·가격 계산에서 0나눗셈이나 극단값이 `inf`를 만들면 `np.isnan(inf)==False`, `pd.isna(inf)==False` 이므로 필터링되지 않고 `json.dumps(..., allow_nan=True 기본)`가 `"Infinity"` 문자열을 그대로 출력. 브라우저 `JSON.parse()`는 `Infinity`를 허용하지 않아 전체 응답 파싱 실패(차트/분석 탭 빈 화면).
- **근거**: `elif isinstance(..., float): return None if np.isnan(obj) else float(obj)` — `isinf` 미검사. `elif pd.isna(obj)` 역시 `inf`에 대해 False. 파일 내 다른 곳은 `math.isfinite/np.isfinite`로 가드하지만 최종 게이트에서만 누락.
- **재현**: `python -c "import json,numpy as np,pandas as pd;from api.index import replace_nan_with_none;print(json.dumps(replace_nan_with_none({'a':float('inf')})))"` → `{"a": Infinity}` (무효 JSON).
- **영향**: 사용자가 대형주 급변동 시나리오에서 확률·목표가 응답을 전혀 못 받음. Vercel 로그는 200으로 남고 클라이언트만 실패하므로 관측 사각지대.

#### P0-2 `api/index.py:14241-14310` build_prediction_outlook OHLCV 각 배열 독립 `None` 제거 → 날짜 정렬 손상

- **증상**: `dd`는 `fetch_stock_data`에서 `df2` 행 단위로 `None`을 채운 뒤 길이 동일하게 반환된다. 그러나 `build_prediction_outlook._arr`는 각 키(`Close`/`Volume`/`High`…)를 독립적으로 `None/inf`를 제거해 새 리스트를 만든다. 중간에 `Close`가 `None`(거래정지 보정)과 `Volume`이 `None`(원천 누락)이 다른 인덱스에 있으면 두 리스트의 동일 인덱스가 서로 다른 거래일을 가리키게 된다. 이후 `avg_volume = mean(volumes[-21:-1])`, `volume_ratio = volumes[-1]/avg_volume`, `recent_low = min(lows[-20:])` 등은 서로 어긋난 창을 기준으로 계산돼 `volume_ratio`·`range_position`·`support/resistance`가 왜곡된다.
- **근거**: `_arr` 정의(14241-14249)가 `for value in dd.get(key,[]) if value is not None and isfinite(float(value))` 로 키별로 독립 필터. 상위에 `Date`와의 정렬 복구 로직 없음. `fetch_stock_data`의 `dropna(subset=["Close"])`는 Close만 제거하므로 Volume 결측은 그대로 남아 이 문제를 증폭.
- **재현**: `dd={'Close':[100,101,None,103,104],'Volume':[1000,1100,1200,None,1300]}` → `_arr('Close')=[100,101,103,104]`, `_arr('Volume')=[1000,1100,1200,1300]` — 원래 index 3의 Close 103이 Volume 1300과 매칭되는 왜곡. 6봉 종목에서 결측이 섞이면 영향이 더 큼.
- **영향**: 신규 상장·거래정지 해제·ETF 분할 등 결측이 잦은 이벤트에서 예측 탭의 변동성·거래량·지지/저항이 실제와 다른 값으로 표시. `volume_ratio`가 1.5배로 과대평가돼 상승 시나리오가 거짓 확증될 수 있다.

#### P0-3 `api/index.py:1927-2050` FIXED 분석 기간 + `period_to_days` tail 절삭 — 휴장·신규상장 왜곡과 “정상 수치 공식” 불일치

- **증상 1**: `FIXED_ANALYSIS_PERIOD="1y"`로 고정하고 `handler.do_GET`에서 `params["period"]`를 강제 덮음(26434) — 프론트의 기간 선택기는 시각적 효과가 없고 항상 252거래일(1y)을 기준으로 지표가 계산된다. 스캘프 탭이 1d/3d 분봉 비교를 별도 호출로 보완하긴 하나, 메인 `add_indicators`의 MA60/MA120 등은 항상 1y 창에서 계산돼 사용자가 기대한 “1일 추세”와 괴리.
- **증상 2**: `period_to_days`로 tail 절삭할 때 `unique_dates = pd.Series(df.index.date).unique()` 로 중복 거래일 제거 후 필터링하지만, `add_indicators`가 먼저 전체 `fetch_period=2y`에서 계산되고 나중에 tail로 자르므로, 1y 경계 부근의 MA/ATR 시드가 2y 과거 데이터에 오염된다. 별 문제는 없으나, 6봉 신규상장 종목은 `len(unique_dates) <= target_days` 분기에서 전체 유지되므로 MA60/MA120이 전부 `NaN`이 되고, 이후 `score`·`confidence`가 기본값에 의존.
- **근거**: 2002-2075 절삭 블록, `VALID_PERIODS={FIXED_ANALYSIS_PERIOD}`(26171).
- **재현**: `ticker=005930.KS&period=3d` 로 요청해도 `_CACHE` 키는 `fetch_stock_data|('005930.KS','KRX','1y','삼성전자')` 로 생성돼 1y 데이터가 반환. 네트워크 캡처에서 `period=3d` 파라미터가 서버 로그에 남지 않음.

#### P0-4 `api/index.py:1816-1846` add_indicators `ewm(min_periods=14)` ATR/ADX NaN — 6봉 종목 전 구간 무효, 이전 `_expected_days` 크래시 재발 여지

- **배경**: 사용자가 재현한 `386380.KQ` 6봉 케이스: `tr.ewm(alpha=1/14, min_periods=14).mean()` 은 14봉 미만에서 전부 `NaN`. 이후 `atrs[-1]`을 truthiness로 검사하던 구 버그는 현재 `_safe_finite_float`+`math.isfinite`로 방어돼 크래시는 막혔다(16026-16036, 14279-14288).
- **잔존**: 크래시는 막혔으나 **정보 손실은 그대로**. 6봉 종목의 ATR·ADX·RSI·%K·OBV 일부가 전부 `NaN`이면 `build_prediction_outlook`의 `atr_pct`, `rsi_available`, `volume_ratio` 등이 모두 fallback/미확보 분기로 빠지고, `score`는 50 근처로 수렴한다. `data_quality.warnings`와 `market_context.data_gaps`에는 기록되지만, **최종 `score`·`prob_up`·`target_price`는 정상 종목과 동일한 스케일로 표시돼 사용자가 구분하기 어렵다.**
- **추가**: `TRIX`·`PSAR` 등 루프 기반 지표는 `len(_hs)==6`이어도 동작하나, 초기값 의존성이 커 신뢰도 낮음. `test_prediction_outlook.py`의 `_sample_dd`는 80봉으로만 구성돼 이 분기 커버리지가 없다.

#### P0-5 `api/index.py:26173-26279` _send cache-control — `/api/stock`은 `no-store`이나 `route` 내부 TTL 캐시가 더 강력해 실시간 보정 무효화

- **증상**: `_send`에서 `/api/stock`은 `no-store`를 보내지만, `fetch_stock_data` 자체는 `ttl_cache(120)` 이고 `fetch_naver_realtime`은 별도 캐시(10s)다. 사용자가 “분석 시작”을 연타해도 120초 동안 yfinance history는 갱신되지 않아 `price_correction`(naver_realtime)만 바뀌고 `dd`의 OHLCV·지표는 고정. 차트 마지막 캔들은 보정으로 맞추지만, 지지/저항·ATR은 과거 데이터 기반이라 현재가와 괴리.
- **근거**: 258, 281-284의 “실패 결과는 캐시 스킵” 로직은 있으나 성공 결과는 120s 고정. `chart_analysis_consistent` 플래그는 이를 경고하지만, 프론트는 경고를 작게 표시해 사용자가 놓치기 쉽다.

---

### P1 — 잘못된 수치·왜곡·누락 위험 (즉시 수정 권장)

#### P1-1 `api/index.py:16516-16518` 최상위 `rsi`/`volume` 필드 왜곡 — “missing을 0/50으로 바꾸는 왜곡” 잔존

- **최상위**: `rsi = round(_safe_finite_float((dd.get("RSI")or[50])[-1],50.0),1) if ... else 50.0` — RSI 미확보 시 항상 50.0을 반환. `volume = _safe_int((dd.get("Volume")or[0])[-1], default=0)` — 미확보 시 0. 예측 탭 내부에서는 `rsi_available`/`volume_available`로 구분하지만, **카드/스파크라인이 최상위 필드를 직접 쓰는 프론트 경로에서는 중립(50)·무거래(0)로 오인**.
- **권고**: 최상위도 `None`을 허용하고 프론트에서 `N/A`·회색 표시가 되도록 계약 변경. 현재 `prediction_output_contract.json`은 `rsi/volume`을 number로만 규정해 `null` 확장이 필요.

#### P1-2 `api/index.py:2167-2181` NaN→None 변환에서 `inf` 미처리 + `pd.notna` 후 float dtype 잔류

- 2169 `d = df2.where(pd.notna(df2), other=None)` — float dtype 컬럼은 `None`을 넣어도 `NaN`으로 남는다(판다스). 이후 루프에서 `math.isfinite`로 잡히나, `Volume`처럼 int로 기대되는 컬럼에서 `None`이 `np.nan`로 남아 `to_dict` 시 `nan`으로 직렬화될 위험. 현재 루프는 `float`·`np.floating`만 검사하므로 `pd.NA`·`object` 경로에서 누락.
- **권고**: `df2 = df2.astype(object).where(pd.notna(df2), None)` 후 루프에서 `pd.isna`+`math.isfinite` 통합, 또는 `replace([np.inf,-np.inf], np.nan)` 선처리.

#### P1-3 `market_briefing/data_fetcher.py:738-812` 시세 파싱 — NaN 문자열·콤마·빈 문자열의 유한성 미검증

- 수정분에서 `math.isfinite` 가드 추가(738-759, 806-812) — 양호. 단, `_fetch_quote_mobile_json`의 `compareToPreviousClosePrice`가 `"-"` 또는 `""`일 때 `float("")` 예외로 `0` 처리돼 등락률 0으로 둔갑할 여지. `except Exception: pass` 로 숨겨져 로그 없음.
- **권고**: 빈/대시 값은 `None`으로 유지하고 `change_pct_num`을 `None`으로 노출, 프론트에서 “-” 표시.

#### P1-4 `api/index.py:15793-15830` 외부 데이터 병렬 수집 — `except Exception: return None` + `TimeoutError` 삼킴

- `_fetch_naver_job`, `_fetch_us_job`, hybrid, confidence 등 다수 경로에서 실패 시 로그 없이 `None` 반환. 운영에서 Naver 차단체·AlphaVantage 레이트리밋·Tiingo 키 만료를 구분할 수 없음.
- **현행 개선**: `fetch_macro_context`는 `[warn] macro {key}: {e}`를 출력. 이 수준을 `route` 병렬 블록에도 적용 필요.

#### P1-5 `market_briefing/correlation_engine.py:268-285` SIGNAL_WEIGHTS 및 agreement 가중치 — 하드코딩 편향

- `SIGNAL_WEIGHTS`가 코드에 고정(technical 0.20 등)이고 검증 데이터셋과의 캘리브레이션 근거가 문서에 없음. `supply blindspot`이면 0.5배, `pattern 미확정`이면 0.6배 등 휴리스틱이 중첩.
- **영향**: 백테스트에서 agreement가 높을 때 목표가 폭을 55%까지 축소하는데, 실제 적중률이 그만큼 높지 않으면 과신(overconfidence).
- **권고**: `models/correlation_weights.json`으로 분리하고 백테스트 기반 grid search 로깅.

#### P1-6 `api/index.py:15597-16200` score/prob 보정 체인 — 순서 의존성과 이중 가산

- 현재 순서: `regime caps → debt_health → investor_flow → hybrid(NCS) → prob 재계산 → ML 블렌딩 → buy_price → confidence → correlation`. `investor_flow`와 `hybrid`에서 `ai_strategy["result"]`에 문자열을 누적하는데, 실패 시 `except: pass` 로 누락돼도 점수는 그대로 진행.
- **위험**: `score`가 `BEAR`에서 40으로 cap된 뒤 hybrid에서 +5를 받을 수 있어 cap이 뚫림(16104 `min(score,40)`이 그 뒤에 없으므로). ML 블렌딩에서 `prob_up`을 두 번(초기·최종) 계산하는 것도 중복.

#### P1-7 `us_price_fetcher.py:80-82` API 키 env-only — 운영 키 미설정 시 성능 열화 은폐

- `FINNHUB_KEY/TIINGO_KEY/AV_KEY = os.getenv(..., "").strip()` — 미설정 시 빈 문자열로 동작, Tiingo/Finnhub 경로는 항상 실패 후 yfinance로 폴백. 폴백 자체는 성공하므로 장애로 보이지 않지만, 장중 스프레드·슬리피지 계산의 정확도가 떨어짐(실시간 호가 없음).
- **권고**: 키 미설정 시 startup 로그 `WARN`와 `/api/health`에 노출.

---

### P2 — 정확도·재현성·관측성 (고도화 단계에서 처리)

#### P2-1 `api/index.py:14527-14550` _expected_days — 여전히 [2,5] 매직 폴백

- crash 방어는 됐으나, `except` 폴백이 `[2,5]` 고정. `_expected_days`가 `gap/atr` 기반으로 계산되므로, ATR fallback(2%)이 들어갔을 때 항상 `[1,6]` 근처가 나와 변별력 낮음. horizon_days(1~1260)와의 스케일링도 부족.
- **권고**: `distance_atr * lead_time_factor`를 `horizon_days`로 클램핑하고, 폴백 시 `horizon_days` 기반 동적 기본값 사용.

#### P2-2 `api/index.py:14839-14925` calibrate_volume_recovery_threshold — 5분봉 70/30 워크포워드 표본 기준 80개는 소수 종목에 과소

- 80개 미만이면 `available=False` 로 기본 1.10배. KRX 소형주는 하루 거래량이 적어 관측 자체가 적고, 항상 기본값으로 동작. 분봉 history를 fetch하는 비용(별도 yfinance 호출) 대비 효용 미검증.

#### P2-3 `market_briefing/data_fetcher.py:318-375` yfinance 지수 폴백 — ETF 심볼(069500.KS) 대용의 추적오차

- KOSPI 지수 대신 KODEX 200 ETF 종가를 쓰는 폴백은 배당·괴리로 일간 등락이 0.05%p 이상 차이 날 수 있음. `validation verified`로 표시되지만 실제는 ETF 프락시.
- **권고**: 폴백 소스를 `yfinance (ETF proxy)` 로 명시 구분.

#### P2-4 `scripts/*` 백테스트 — lookahead/서바이버십 편향 미검증

- `backtest_target_price_accuracy.py`와 `backtest_prediction_zones.py`가 `docs/backtests/*.csv`를 갱신하지만, 상장폐지 종목 제외·거래정지 구간 처리·분할/배당 조정 일치가 문서화되지 않음.
- **권고**: 백테스트 입력 유니버스·제외 규칙을 `docs/backtests/README.md`로 고정.

#### P2-5 `api/index.py:5935-5996` _safe_* 헬퍼 — 로그 폭주 위험

- `log_ctx`가 있으면 매 호출마다 `print` — 6봉 종목에서 ATR·RSI·Volume 각각 로그가 찍혀 Vercel 로그 스트림을 채움. 샘플링 또는 `level=debug` 필요.

#### P2-6 Vercel 스레드·네트워크 예산 — 60s 내 미완 시 부분 데이터 반환 정책 부재

- `/api/screener`는 42s budget 후 `TimeoutError`를 잡아 “부분 성공”을 표시하지만, `/api/stock`은 12s 타임아웃 후 `None`으로 대체해 “데이터 없음”으로만 표시. 사용자가 재시도해야 할지 대기해야 할지 구분 불가.
- **권고**: `X-StockOracle-Partial: naver_timeout` 같은 헤더로 부분 실패를 명시.

---

## 4) 원인·영향·구체 수정 권고 (P0/P1 중심)

| ID | 원인 | 영향 | 수정 (파일:행·패치 스케치) |
|---|---|---|---|
| P0-1 | `replace_nan_with_none`가 `inf` 미처리 | `Infinity` JSON 무효 → 프론트 전체 파싱 실패 | `api/index.py:26163-26166` 를 `if isinstance(v,(float,np.floating)): return None if not np.isfinite(v) else float(v)` 로 교체. `pd.isna` 분기는 `if pd.isna(obj): return None` 그대로 유지하되 `isinf`가 먼저 걸리도록 순서 변경. 추가: `json.dumps(..., allow_nan=False)` 로 강제해 잔류 시 500 대신 200+error_detail로 폴백 — 현행 `_send`의 `default=str` 유지. |
| P0-2 | `_arr` 독립 필터 | 날짜 정렬 붕괴, volume 로직 왜곡 | `build_prediction_outlook._arr`를 제거하고 `aligned_indices = [i for i,v in enumerate(dd.get("Close",[])) if v is not None and isfinite(v)]` 같은 **공통 인덱스**를 먼저 구한 뒤 모든 OHLCV를 동일 인덱스로 슬라이싱. `Date`도 동일 인덱스로 필터해 `recent_low = min(lows_aligned[-20:])` 가 동일 창을 보도록. `analyze_score` 등 타 지표 함수에도 동일 패턴 적용. |
| P0-3 | FIXED 1y 강제 | 기간 선택 무효, 지표 창 오염 | `handler.do_GET`에서 `VALID_PERIODS` 검증을 부활하되, `FIXED_ANALYSIS_PERIOD`는 **기본값**으로만 사용. `fetch_stock_data`의 `period_to_days`는 요청 period를 존중하고, 캐시 키에 period를 포함하도록 `ttl_cache` 키 생성부(268-276) 확인. 프론트 기간 선택기 라벨을 “지표 계산 기간: 1y 고정”으로 명시하거나, 스캘프 전용 1d/3d를 메인 분석과 분리. |
| P0-4 | `min_periods=14` 전 구간 NaN | 6봉 종목 스코어 수렴, 예측 무력화 | `add_indicators`에서 `min_periods`를 `min(14, len(c))` 로 동적화하거나, 14봉 미만이면 `ATR = tr.rolling(len(c)).mean()` 같은 단기 대체 계산을 추가. 20봉 미만이면 `data_quality.status="데이터 부족"`를 넘어 `score/confidence`에 하드 캡(예: ≤45) 적용 — 현재는 경고만. |
| P0-5 | TTL 캐시와 실시간 보정 불일치 | 차트·지표 괴리 | `fetch_stock_data` TTL을 120s→30s로 단축하거나, `price_correction` 발생 시 `dd`의 ATR/지지저항을 즉시 재계산. `chart_analysis_consistent`가 false일 때 프론트에서 노란 배너 “지표는 2분 전 기준”을 더 강하게 표시. |
| P1-1 | 최상위 rsi/volume 고정 | 카드 왜곡 | `response["rsi"] = round(...,1) if is_finite else None`, `response["volume"] = _safe_int(..., default=None)` 로 변경, 프론트에서 `N/A` 회색 처리. `prediction_output_contract.json`에 `rsi: number|null` 반영. |
| P1-2 | float dtype 잔류 | NaN 직렬화 잔류 | `fetch_stock_data` 2168-2181 블록을 `df2 = df2.astype(object).where(pd.notna(df2), None)` + `replace([np.inf,-np.inf], None)` 선처리, 루프에서 `pd.isna(v) or not np.isfinite(float(v))` 통합. |
| P1-4 | broad except 은폐 | 장애 관측 불가 | `route`의 병렬 블록과 hybrid/confidence 블록에 `logger.warning("[route] %s failed: %s", ctx, e, exc_info=True)` 추가. `except Exception: return None` 를 `except (TimeoutError, ...): log + return {"ok":False,"reason":...}` 로 세분화. |
| P1-7 | env 키 미설정 은폐 | 실시간성 저하 | `us_price_fetcher._BaseClient` 초기화 시 키 빈 문자열이면 `logger.warning` + `/api/health`에 `us_providers: {finnhub: missing}` 노출. 배포 체크리스트에 키 주입 확인 단계 추가. |

---

## 5) 회귀 행렬 및 명령 (KR/US 분리)

### 5.1 테스트 인벤토리 (현행 25 파일, 235 passed 주장과 대조)

- 실제 파일 수: `tests/` 25개 (상단 `Get-ChildItem` 참조). 최근 diff의 신규 테스트: `test_prediction_outlook`, `test_charm_ranking`, `test_peter_lynch`, `test_ml_feature_parity`, `test_us_price_fetcher_alpha_vantage` 등.
- **사각지대**: `_sample_dd`가 80봉 고정 → 6봉/14봉/20봉 경계, `Volume=None`·`ATR=inf`·`RSI=NaN` 조합, OHLCV 정렬 붕괴, `replace_nan inf` 케이스 없음. IONQ/RIVN처럼 실제 252봉 yfinance 특성과 테스트 80봉 합성 데이터 괴리.

### 5.2 권장 회귀 행렬

| 구분 | KR 대표 | US 대표 | 검증 포인트 | 명령 |
|---|---|---|---|---|
| **단봉/신규상장** | `386380.KQ`(6봉, 이전 장애) | `IONQ` (실제 yfinance 252봉이나, 합성 6봉으로 강제) | ATR/Rsi NaN, `_expected_days` fallback, `data_quality` 경고, `score` 캡 | `pytest tests/test_prediction_outlook.py::test_short_history_6bars -xvs` (신규 테스트 작성 필요) |
| **정상 대형** | `005930.KS` 삼성 | `AAPL` | 1y 252봉, naver_realtime 보정, 투자자 수급 반영, hybrid NCS | `pytest tests/test_investment_charm.py tests/test_peter_lynch.py -k "005930 or AAPL"` |
| **중형 변동성** | `035720.KS` 카카오 | `TSLA` | ATR>5%, BB 폭, PSAR 전환, ML 블렌딩 | `pytest tests/test_ml_feature_parity.py tests/test_chart_pattern_interaction.py -x` |
| **소형·거래대금** | `247540.KQ` 에코프로비엠 | `RIVN` | 거래대금 필터, 20봉 평균 거래량, 업종 흐름 | `pytest tests/test_sector_flow_catalog.py -xvs` |
| **거래정지/관리** | 인위적 `code=` 관리 종목 | `GME` (과거 거래정지 이력) | `security_status.trading_halt` 차단, `halt_release_label` | `pytest tests/test_krx_tab.py -xvs` |
| **지수/거시** | `KOSPI/KOSDAQ` | `^GSPC/^IXIC` | 3단 폴백, 검증 실패 시 `unavailable` | `pytest tests/test_market_core_indices.py -xvs` |
| **스캘프 기간** | `KRX 1d vs 3d` | `US 1d vs 3d` | `recommend_scalp_analysis_period`, VI/실적 차단, 반대신호 | `pytest tests/test_prediction_outlook.py -k scalp -xvs` |
| **JSON 무결성** | `inf` 인위 주입 | `inf` 인위 주입 | `replace_nan_with_none`, `json.dumps(allow_nan=False)` | `pytest tests/test_prediction_scale_quality.py -k json` (신규) |
| **OHLCV 정렬** | `Close=None` 1개 + `Volume=None` 1개 다른 위치 | 동일 | `build_prediction_outlook` volume_ratio/지지저항 | `pytest tests/test_prediction_outlook.py::test_ohlcv_alignment -xvs` (신규) |

### 5.3 실행 명령 (Windows PowerShell, Vercel 제약 고려)

```powershell
# 1) 전체 회귀 (가장 빠름 — 네트워크 의존 테스트는 모킹)
pytest -q

# 2) KR 신규상장 6봉 재현 (실제 네트워크 없이 합성 데이터로)
pytest tests/test_prediction_outlook.py -k "short_history or ohlcv_alignment or json_inf" -xvs

# 3) US 실시간 가격 (네트워크 필요 — 키 주입 확인)
$env:FINNHUB_API_KEY="..."; $env:TIINGO_API_KEY="..."; $env:ALPHAVANTAGE_KEY="..."
pytest tests/test_us_price_fetcher_alpha_vantage.py -xvs

# 4) 백테스트 분포 검증 (시간 소요 — 오프라인, 14일 embargo 준수)
python scripts/backtest_target_price_accuracy.py --market KRX --period 1y --horizon 20
python scripts/backtest_prediction_zones.py --zone core

# 5) E2E (dev_server, 3000 포트, 외부 전송 차단)
# 텔레그램/깃허브 쓰기 없음 — 원천 조회/테스트만
python dev_server.py  # 별도 터미널
Invoke-WebRequest "http://localhost:3000/api/stock?ticker=005930&period=1y" | ConvertFrom-Json | ConvertTo-Json -Depth 6 | Out-File tmp/e2e_kr.json
Invoke-WebRequest "http://localhost:3000/api/stock?ticker=AAPL&period=1y"   | ConvertFrom-Json | ConvertTo-Json -Depth 6 | Out-File tmp/e2e_us.json
# JSON 무효 여부 확인
python -c "import json,pathlib;json.load(open('tmp/e2e_kr.json'));json.load(open('tmp/e2e_us.json'));print('JSON valid')"
```

- **외부 전송 금지**: `POST /api/telegram/send` 는 동일출처·레이트리밋(60s 5회)이 있으나, 테스트 중 호출 금지. `docs/backtests/prediction_learning.jsonl` 은 운영 이력 — 회귀 테스트에서 쓰기 금지(임시 파일로 대체).
- **비밀값 노출 금지**: 로그·리포트에 `FINNHUB/TIINGO/AV` 키, `TELEGRAM_BOT_TOKEN` 평문 금지.

---

## 6) 단계별 고도화 범위 (후속 세션이 바로 착수 가능)

### Phase 0 — P0 핫픽스 (1~2일, 배포 차단)

1. `replace_nan_with_none` inf 처리 + `allow_nan=False` 방어 (P0-1)
2. `build_prediction_outlook` OHLCV 정렬 공통 인덱스화 (P0-2)
3. 최상위 `rsi/volume` null 허용 + 프론트 N/A 처리 (P1-1)
4. 6봉 재현 테스트 3건 추가 및 `pytest` 통과 확인

### Phase 1 — KR/US 안정화 (3~5일)

1. `add_indicators` min_periods 동적화 + 20봉 미만 캡 (P0-4)
2. `fetch_stock_data` NaN→None 정렬 개선 (P1-2) — `astype(object)` + `replace(inf)`
3. TTL 캐시와 실시간 보정 일관성 (P0-5) — 30s TTL 또는 재계산
4. `handler` FIXED 기간 정책 정리 (P0-3) — 계약 변경 또는 UI 고지
5. 로깅 세분화 (P1-4) — `logger` 도입, `exc_info`

### Phase 2 — 신뢰도·예측 정교화 (1~2주)

1. `correlation_engine` 가중치 외부화 + 백테스트 캘리브레이션 (P1-5)
2. `_expected_days` horizon 연동 + 폴백 동적화 (P2-1)
3. `score` 보정 체인 순서 고정 + caps 일원화 (P1-6)
4. `us_price_fetcher` 헬스체크 + 폴백 신선도 가드 강화

### Phase 3 — 데이터·운영 고도화 (2~4주)

1. 백테스트 서바이버십/lookahead 편향 제거, 유니버스 고정 문서화 (P2-4)
2. `data_fetcher` ETF 프락시 명시, 환율 고정 fallback 제거 또는 고지
3. 분봉 캘리브레이션 표본 기준 재정의 (P2-2)
4. Vercel 부분 실패 헤더·프론트 배너 (P2-6)

### 각 Phase DoD (완료 정의)

- `pytest -q` 통과 + 신규 경계 테스트 포함
- `GET /api/stock?ticker=386380&period=1y` (합성 6봉)와 `005930.KS`/`AAPL` E2E JSON이 `JSON.parse` 통과
- `prediction_learning.jsonl` 오염 없음 (테스트는 tmp 경로)
- 로그에 `ERROR` 없음, `WARN`은 원인 코드 포함

---

## 7) 분석 보고서 경로

- **본 인계 보고서**: `docs/analysis_handover_20260912.md` (본 파일)
- **참조 계약/설계**:
  - `docs/prediction_output_contract.json` — buy_price/risk_scenarios 스키마
  - `docs/ml_integration_design.md` — ML 파이프라인 설계
  - `docs/pattern_engine_refactor.md` — 패턴 엔진 리팩터
  - `docs/peter_lynch_garp_assessment.md` — GARP 5/5 평가
  - `docs/backtests/prediction_zone_backtest_report.md` — 예측 존 백테스트
  - `docs/stockoracle_prediction_logic_redesign.md` — 예측 로직 재설계 초안
  - `models/training_metadata.json` — LightGBM 메타(AUC 0.569, validation passed)
  - `vercel.json` — 라우팅/메모리/크론
  - `requirements.txt` — 31라인, Vercel 번들 스펙

---

## 8) 재현·미검증 구분

### 8.1 재현 완료 (로컬 실행·코드 리딩으로 확인)

| 항목 | 재현 방법 | 결과 |
|---|---|---|
| 386380.KQ 6봉 ATR NaN 크래시 | `add_indicators(6봉 df)` → `ATR=[nan]*6`, `build_prediction_outlook(atr=nan)` 호출 (상단 386 실행 로그) | 크래시 없이 `[1,6]` fallback, `levels` 정상 반환 — **현행 가드 동작 확인**. 단, 지표 전 구간 무력화는 유지 |
| volume/rsi 미확보 시 예측 왜곡 | `dd Volume=[None]*6, RSI=[None]*6` 로 `build_prediction_outlook` 호출 | `volume.value="거래량 확인 필요"` , `data_gaps` 5건 기록, `rsi_available=False` — **예측 내부에서는 구분됨**. 최상위 `rsi/volume` 필드는 여전히 50/0 고정(잔존) |
| OHLCV 독립 필터 정렬 붕괴 | `Close=[100,101,None,103,104]`, `Volume=[1000,1100,1200,None,1300]` 합성 → `_arr` 각각 필터 | 길이 4로 같으나 `Close[2]=103`에 `Volume[2]=1300`이 매칭 — **왜곡 재현** |
| JSON `Infinity` | `replace_nan_with_none({'a':inf})` → `json.dumps` → `{"a": Infinity}` | **무효 JSON 재현** — `pd.isna(inf)==False` 로 통과 |
| yfinance 6봉 vs 252봉 괴리 | `ds 250204 IONQ` 실제 252봉, 테스트 `80봉` 합성 | **괴리 확인** — 테스트가 short-history 미커버 |

### 8.2 미검증 (네트워크/시간/운영 의존 — 후속 세션에서 실제 호출 필요)

| 항목 | 필요한 검증 | 제약 |
|---|---|---|
| KRX 지수 3단 폴백 실제 장애 | KRX Global 500/타임아웃 유도 → Naver → yfinance 경로가 `validated` 로 전환되는지 | 외부 전송 없이 `GET /api/market/summary` 1회 호출, Date 헤더 검증 |
| Naver 모바일 JSON vs HTML 폴백 | `m.stock.naver.com/api/stock/{code}/basic` 차단 시 HTML 파싱으로 0%p 둔갑 여부 | 네트워크 스텁 필요 |
| US Overnight/Pre-Market 신선도 | `yfinance postMarketTime` 8시간 초과 시 `None` 반환 및 정규장 종가로 하강하는지 | KST 06:00/13:00 시간대 각각 호출 |
| AlphaVantage 레이트리밋 | 키 없이 `fetch_alpha_overview` → `overview={}` 일 때 yfinance info 폴백이 `sector/industry`를 채우는지 | 키 제거 후 `AAPL` 분석 |
| 스캔 48종 수집 42s 예산 | `SCAN_COLLECT_BUDGET_S=42` 초과 시 타임아웃 후 부분 선정이 15개 정확히 맞는지 | `GET /api/screener` 부하 테스트 |
| 백테스트 learning_adjustment 반영 | `prediction_learning.jsonl` 60건 기반 `calc_learning_adjustment`가 실제 `target_price`에 보정되는지 | `docs/backtests/target_price_backtest_summary.json`과 대조 |
| 텔레그램 남용 방지 | 동일출처 외 POST → 403, 60s 6회 → 429 | `Invoke-WebRequest -Method POST /api/telegram/send` (키 없이) |

---

### 부록 — 빠른 수정 체크리스트 (후속 세션 복사·붙여넣기)

```powershell
# 사전 확인 (AGENTS.md 대체 — git 상태 보존)
git status
git diff --stat
git diff HEAD -- api/index.py | Select-Object -First 20

# P0-1: inf 가드 (api/index.py:26163)
# before: return None if np.isnan(obj) else float(obj)
# after : return None if not np.isfinite(obj) else float(obj)  # covers nan/inf/-inf
# + json.dumps(..., allow_nan=False)

# P0-2: OHLCV 정렬 (api/index.py:14241)
# 공통 인덱스: valid_idx = [i for i,(c,v) in enumerate(zip(dd.get("Close",[]), dd.get("Volume",[]))) if c is not None and v is not None and isfinite(c) and isfinite(v)]
# 모든 배열을 valid_idx로 슬라이싱 후 계산

# P1-1: 최상위 null 허용
# rsi: round(...) if is_finite else None
# volume: _safe_int(..., default=None)

# 테스트 추가
# tests/test_short_history.py : 6봉, 14봉, inf 주입, 정렬 붕괴 케이스
pytest tests/test_short_history.py -xvs
pytest -q
```

---

*작성: StockOracle 1단계 분석 세션 (apply_patch 전용) — 2026-09-12 KST*
*다음 세션 전제: `git diff` 미커밋 상태 그대로 인계, `AGENTS.md` 없음 — 본 보고서와 git log/diff를 최우선으로 삼을 것.*
