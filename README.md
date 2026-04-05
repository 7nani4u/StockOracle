# StockOracle (Vercel Edition)

Vercel 서버리스 환경에 최적화된 AI 기반 주식 분석 시스템입니다.
순수 Python으로 구현되어 TA-Lib 없이 26개 캔들스틱 패턴을 인식하고 핵심 기술적 지표를 산출합니다.

---

## 🚀 주요 기능

| 기능 | 설명 |
|---|---|
| **KRX / US 통합 지원** | 국내(KOSPI·KOSDAQ) 및 미국(NYSE·NASDAQ) 주식 통합 조회 |
| **다양한 분석 기간** | 초단기(1일/3일/1주), 단기(2주/1개월), 중장기(6개월~5년) 동적 분봉/일봉 분석 |
| **26개 캔들스틱 패턴** | 순수 Python 구현 (TA-Lib 불필요, Vercel 완전 호환) |
| **핵심 기술적 지표 최적화** | MA, EMA, MACD, RSI, 볼린저밴드, ATR, ADX, OBV, Aroon 등 실전 압축 지표 제공 |
| **가중치 기반 종합 점수** | 추세·모멘텀·변동성·거래량·보조지표 5개 축 0~100 점수화 |
| **AI 종합 진단 및 전략** | 점수 기반 Buy/Hold/Sell 요약 및 시나리오별(상승/하락/횡보) 대응 전략 제시 |
| **차트 형태 인식** | 삼각형·쐐기형·이중 천장/바닥 등 7가지 기하학적 패턴 |
| **가격 예측 모델** | Holt-Winters(계절성) + 선형회귀 + XGBoost 시뮬레이션 앙상블 |
| **ATR 기반 리스크 관리** | 보수적·중립·공격적 3가지 시나리오별 목표가·손절가 자동 산출 |
| **실시간 스크리너** | 국내·해외 주요 종목 시세, 등락률 모니터링 |
| **기업 정보 크롤링** | Naver Finance — 시가총액, PER, PBR, ROE, 부채비율, 공시 |
| **뉴스 피드** | Google News RSS 한글·영문 통합 |
| **시장 심리 지수** | 미국 VIX, 한국 KOSPI200 |

---

## 🕯️ 캔들스틱 패턴 (26개)

TA-Lib 없이 순수 Python + NumPy로 구현되어 Vercel 서버리스에서 동작합니다.
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

```
Holt-Winters (이중 지수 평활)  ─┐
선형 회귀 (numpy.polyfit)      ─┤─▶ 앙상블 → 향후 30일 예측
XGBoost 시뮬레이션              ─┘
```

- 데이터 부족 시 선형 회귀 폴백
- 95% 신뢰 구간 포함
- Holt-Winters: 단기 추세·계절성 포착
- XGBoost 시뮬레이션: 단기/장기 MA 모멘텀 강도 및 시간 감쇠 적용

---

## ⚙️ API 엔드포인트

| 경로 | 메서드 | 설명 |
|---|---|---|
| `/` | GET | HTML 프론트엔드 |
| `/api/stock?ticker=삼성전자&period=1y` | GET | 종목 상세 분석 |
| `/api/screener?sort_by=price&sort_order=desc` | GET | 종목 스크리닝 |
| `/api/toss-overseas` | GET | 토스증권 해외 종목 필터 |
| `/api/sentiment?market=KRX` | GET | 시장 심리 지수 |
| `/api/resolve?q=삼성` | GET | 종목명·코드 검색 |
| `/api/cron` | GET | 캐시 워밍 (매시간 자동 실행) |

---

## 📂 프로젝트 구조

```
StockOracle/
├── api/
│   └── index.py        # 핵심 로직 (백엔드 API + 프론트엔드 HTML 통합)
├── dev_server.py       # 로컬 개발용 서버 (포트 3000)
├── requirements.txt    # 의존성 패키지 (Vercel 호환)
├── vercel.json         # Vercel 배포 설정
└── README.md
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

> TA-Lib, Prophet 등 C 컴파일 필요 패키지는 포함되지 않습니다.
> 모든 기술적 분석은 순수 Python으로 구현되어 있습니다.

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
