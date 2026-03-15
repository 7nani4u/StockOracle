# StockOracle

Vercel 서버리스 환경에서 `api/index.py` 단일 엔트리포인트로 동작하는 주식 분석/예측 앱입니다.

## 배포 엔트리포인트

- API + 웹 화면 모두 `api/index.py`에서 처리합니다.
- `/api/*` 요청은 JSON 응답, 그 외 경로는 인라인 HTML을 반환합니다.

## Vercel 설정 요약

- `vercel.json`은 `functions` + `rewrites` 구성만 사용합니다.
- `/(.*)`와 `/api/(.*)` 모두 `api/index.py`로 리라이트되어 404(NOT_FOUND) 발생을 방지합니다.

## 로컬 실행

```bash
python dev_server.py
```

실행 후:

- 웹 UI: `http://localhost:8000/`
- API 예시:
  - `http://localhost:8000/api/stock?ticker=삼성전자&period=1y`
  - `http://localhost:8000/api/sentiment?market=KRX`
  - `http://localhost:8000/api/screener`
