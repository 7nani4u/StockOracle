#!/usr/bin/env python3
"""
로컬 개발 서버
Vercel 배포 전 로컬에서 api/stock.py를 테스트하기 위한 래퍼.
실제 Vercel 환경과 동일하게 BaseHTTPRequestHandler 기반으로 동작.
"""
import sys
import os
from http.server import HTTPServer

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# api/stock.py의 handler 클래스를 import
from api.stock import handler

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), handler)
    print(f"🚀 로컬 API 서버 실행 중: http://localhost:{port}")
    print(f"   엔드포인트 목록:")
    print(f"   GET /api/stock?ticker=삼성전자&period=1y")
    print(f"   GET /api/screener")
    print(f"   GET /api/sentiment?market=KRX")
    print(f"   GET /api/resolve?q=삼성전자")
    print(f"")
    print(f"   프론트엔드: cd frontend && npm run dev (포트 3000)")
    server.serve_forever()
