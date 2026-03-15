# StockOracle (Vercel Edition)

Vercel 서버리스 환경에 최적화된 주식 예측 AI 시스템입니다.

## 🚀 주요 기능
- **KRX/US 통합 지원**: 국내 및 해외 주식 데이터 조회 및 분석
- **AI 기술적 분석**: 캔들 패턴, RSI, MACD, 볼린저 밴드 등 종합 분석
- **미래 가격 예측**: Holt-Winters(계절성) + XGBoost(머신러닝) 앙상블 모델
- **실시간 스크리너**: 주요 종목 시세 및 등락률 모니터링
- **서버리스 아키텍처**: Vercel Python Runtime 최적화 (단일 파일 구조)

## 📂 프로젝트 구조
```
StockOracle/
├── api/
│   └── index.py        # 핵심 로직 (백엔드 API + 프론트엔드 HTML 통합)
├── dev_server.py       # 로컬 개발용 서버 (Vercel 환경 시뮬레이션)
├── requirements.txt    # 의존성 패키지 (Vercel 호환 버전)
└── vercel.json         # Vercel 배포 설정
```

## 🛠️ 로컬 실행 방법
Vercel에 배포하기 전에 로컬 환경에서 테스트할 수 있습니다.

1. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

2. **개발 서버 실행**
   ```bash
   python dev_server.py
   ```

3. **브라우저 접속**
   - http://localhost:3000

## ☁️ Vercel 배포 방법
1. [Vercel CLI](https://vercel.com/docs/cli) 설치:
   ```bash
   npm i -g vercel
   ```

2. 배포:
   ```bash
   vercel
   ```
   (또는 GitHub 리포지토리에 푸시하여 자동 배포 연동)

## ⚠️ Vercel 배포 시 주의사항
- **파일 시스템 제한**: Vercel은 `/tmp` 디렉토리 외에는 파일 쓰기가 금지됩니다. 이 프로젝트는 자동으로 `/tmp`를 캐시 경로로 사용하도록 설정되어 있습니다.
- **실행 시간 제한**: Hobby 플랜의 경우 함수 실행 시간이 10초(최대 60초 설정 가능)로 제한됩니다. 복잡한 모델 학습은 시간이 걸릴 수 있으므로, 데이터 조회 및 예측 로직이 최적화되어 있습니다.
- **패키지 호환성**: C/C++ 컴파일이 필요한 일부 패키지(TA-Lib, Prophet 등)는 Vercel 빌드 환경에서 지원되지 않아 순수 Python 대안(pandas-ta, statsmodels)으로 교체되었습니다.
