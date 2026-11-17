"""Keep the restored scan page and its server imports deployable together."""

import ast
import io
import re
from pathlib import Path

import pytest

from api import index


def test_scan_route_imports_exist_in_the_shipped_modules():
    source_path = Path(index.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in {
            "market_briefing.scan_engine", "market_briefing.scan_box_plan",
            "market_briefing.scan_execution",
        }
    ]
    assert imports, "The scan endpoint's imports must be checked"
    # Execute the real import statements, without running a network scan.
    for node in imports:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), {})


def test_scan_controls_keep_the_original_market_and_mode_selectors():
    markup = index.HTML.split('<div id="page-scan"', 1)[1].split(
        '<div id="page-immune"', 1
    )[0]
    assert re.findall(r'<select\b[^>]*\bid="([^"]+)"', markup) == [
        "scan-market", "scan-mode",
    ]
    for removed_id in ("scan-horizon", "scan-equity", "scan-tickers", "position-review"):
        assert removed_id not in markup


@pytest.mark.parametrize("path,data,content_type,status", [
    ("/", "<html></html>", "text/html", 200),
    ("/index.html", "<html></html>", "text/html", 200),
    ("/api/scan?market=KRX&mode=FULL", {"candidates": []}, "application/json", 200),
    ("/api/scan?market=US&refresh=1", {"candidates": []}, "application/json", 200),
    ("/api/scan", {"error": "scan import failed"}, "application/json", 200),
    ("/api/scan", {"error": "scan import failed"}, "application/json", 500),
])
def test_scan_page_and_results_are_not_reused_by_http_caches(path, data, content_type, status):
    class Handler:
        headers = {}

        def __init__(self):
            self.path = path
            self.output_headers = {}
            self.wfile = io.BytesIO()

        def send_response(self, value):
            self.status = value

        def send_header(self, key, value):
            self.output_headers[key] = value

        def end_headers(self):
            pass

    response = Handler()
    index._send(response, data, status=status, content_type=content_type)

    assert response.status == status
    cache_control = response.output_headers["Cache-Control"]
    assert "no-store" in cache_control
    assert "s-maxage" not in cache_control
    assert "stale-while-revalidate" not in cache_control

