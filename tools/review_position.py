"""Read-only holding review using the same code as /api/position-review.

python -m tools.review_position --ticker GRAB --entry-price 3.41 --shares 5 \
    --entered-at 2026-11-10T01:00:00+09:00 --horizon-days 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from api.index import build_position_review


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="일봉·거래량 기반 보유 종목 점검 (주문 전송 없음)")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--market", choices=("US", "KRX"), default="US")
    parser.add_argument("--entry-price", type=float, required=True)
    parser.add_argument("--shares", type=float, required=True)
    parser.add_argument("--entered-at", help="ISO 시각, 시간대 미지정 시 한국시간")
    parser.add_argument("--horizon-days", type=int, default=3)
    parser.add_argument("--target-price", type=float)
    parser.add_argument("--stop-price", type=float)
    parser.add_argument("--round-trip-bps", type=float, default=20)
    parser.add_argument("--participation-rate", type=float, default=.01)
    parser.add_argument("--exit-days", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = vars(parser.parse_args())
    output = args.pop("output")
    try:
        result = build_position_review({k: v for k, v in args.items() if v is not None})
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result.get("available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
