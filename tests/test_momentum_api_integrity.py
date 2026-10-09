"""Scan switches and per-symbol completed daily bars regression coverage."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from api import index
from market_briefing.momentum_persistence import detect_momentum_persistence


def test_scan_effective_options_respect_env_and_request(monkeypatch):
    monkeypatch.setenv("STOCKORACLE_MOMENTUM_SCAN", "0")
    monkeypatch.setenv("STOCKORACLE_VCP_SCAN", "false")
    assert index._scan_signal_options({}) == (False, False)
    assert index._scan_signal_options({"momentum": "1", "vcp": "1"}) == (True, True)
    monkeypatch.setenv("STOCKORACLE_MOMENTUM_SCAN", "1")
    monkeypatch.setenv("STOCKORACLE_VCP_SCAN", "1")
    assert index._scan_signal_options({"momentum": "OFF", "vcp": "no"}) == (False, False)


def test_scan_cache_identity_includes_effective_switches():
    # Evaluate the actual cache expression without network collection.
    import ast
    source = Path(index.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    expression = next(node.value for node in ast.walk(tree)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "_cache_key"
                              for target in node.targets)
                      and isinstance(node.value, ast.JoinedStr)
                      and any(isinstance(part, ast.Constant) and "scan|v6|" in str(part.value)
                              for part in node.value.values))
    code = compile(ast.Expression(expression), "scan-cache", "eval")
    values = dict(market_p="US", mode_p="full", equity=10000, risk_pct=1,
                  raw_list=["TEST"], _prune_cache_token="artifact1")
    keys = {eval(code, dict(values, _momentum_on=momentum, _vcp_on=vcp))
            for momentum in (False, True) for vcp in (False, True)}
    assert len(keys) == 4
    first = eval(code, dict(values, _momentum_on=True, _vcp_on=True))
    second = eval(code, dict(values, _momentum_on=True, _vcp_on=True, _prune_cache_token="artifact2"))
    assert first != second


def test_last_bar_date_matches_aligned_ohlcv_rows():
    frame = pd.DataFrame({name: [100., 101., 102.] for name in
                          ("Close", "High", "Low", "Volume", "Open")},
                         index=pd.date_range("2026-10-05", periods=3))
    frame.loc[frame.index[-1], "Low"] = float("nan")
    assert index._scan_last_bar_date(frame) == "2026-10-06"
    assert index._scan_last_bar_date(frame.iloc[:0]) == ""


def test_third_observation_intraday_waits_but_stale_symbol_keeps_completed_day():
    now = datetime(2026, 10, 7, 11, tzinfo=ZoneInfo("Asia/Seoul"))
    closes = [100.] * 15 + [125., 120., 119., 118.]
    partial = index._bar_in_progress("KRX", "2026-10-07", now)
    completed = index._bar_in_progress("KRX", "2026-10-06", now)
    assert partial is True and completed is False
    assert detect_momentum_persistence(closes, in_progress=partial)["stage"] == "WAIT"
    assert detect_momentum_persistence(closes, in_progress=completed)["stage"] == "PASS"
