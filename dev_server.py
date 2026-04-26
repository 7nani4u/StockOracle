#!/usr/bin/env python3
"""
로컬 개발 서버 (Vercel 환경 시뮬레이션)
=====================================
이 스크립트는 로컬에서 `api/index.py`를 실행하여 Vercel 배포 전 기능을 테스트할 수 있게 합니다.
`http.server.HTTPServer`를 사용하여 Vercel의 Serverless Function 환경과 유사하게 동작합니다.

사용법:
    python dev_server.py

접속:
    http://localhost:3000
"""
import sys
import os
from http.server import HTTPServer
from socketserver import ThreadingMixIn

# 현재 디렉토리를 Python 경로에 추가하여 api 모듈 import 가능하게 함
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # api/index.py의 handler 클래스를 import
    from api.index import handler
except ImportError as e:
    print(f"오류: api/index.py를 찾을 수 없거나 import 할 수 없습니다.\n{e}")
    sys.exit(1)

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """동시 요청을 스레드로 병렬 처리 — 단일 스레드 블로킹 방지."""
    daemon_threads = True  # 메인 종료 시 워커 스레드도 함께 종료

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"🚀 로컬 개발 서버 실행 중: http://localhost:{port}")
    print(f"   - 메인 페이지 (HTML): http://localhost:{port}/")
    print(f"   - API 예시:")
    print(f"     GET /api/stock?ticker=삼성전자&period=1y")
    print(f"     GET /api/screener")
    print(f"     GET /api/sentiment?market=KRX")
    print(f"---------------------------------------------------")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
        server.server_close()
