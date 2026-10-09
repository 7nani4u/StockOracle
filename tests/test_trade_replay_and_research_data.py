import pandas as pd
import pytest
from market_briefing.trade_replay import replay_signals
from market_briefing.research_data import normalize_history, aligned_benchmark, expected_last_session


def history(n=30):
    return {"dates": pd.bdate_range("2026-01-01", periods=n).strftime("%Y-%m-%d").tolist(),
            "opens": [100.] * n, "highs": [101.] * n, "lows": [99.] * n,
            "closes": [100.] * n, "volumes": [1000.] * n}


def signal(index=0):
    return {"technique": "T", "signal_index": index, "stop_price": 95.,
            "target_price": 110., "atr": 5., "conditions": {"state": "A"}}


def replay(signals, ohl):
    return replay_signals(signals, ohl, "X", "US", "2026-01-01", "2027-01-01", .1, 0)


def test_gap_stop_fills_at_open_not_fictional_stop():
    ohl = history()
    ohl["opens"][2] = 85.
    ohl["lows"][2] = 84.
    rows, _ = replay([signal()], ohl)
    assert rows[0]["entry_index"] == 1
    assert rows[0]["exit_price"] == 85.
    assert rows[0]["exit_reason"] == "stop_gap"
    assert rows[0]["return_pct"] == pytest.approx(-15.1)


def test_same_bar_ambiguity_and_twenty_actual_sessions():
    ohl = history()
    rows, _ = replay([signal()], ohl)
    assert rows[0]["hold_sessions"] == 20
    ohl["highs"][1], ohl["lows"][1] = 111., 90.
    rows, _ = replay([signal()], ohl)
    assert rows[0]["exit_reason"] == "stop_first"
    assert rows[0]["ambiguous_daily_bar"]


def test_overlapping_or_right_censored_not_counted_as_repetitions():
    rows, reject = replay([signal(), signal(1)], history())
    assert len(rows) == 1 and reject["overlapping_position"] == 1
    rows, reject = replay([signal(25)], history())
    assert not rows and reject["right_censored"] == 1


def test_gap_invalid_entry_not_silently_filled():
    ohl = history()
    ohl["opens"][1] = 90.
    rows, reject = replay([signal()], ohl)
    assert not rows and reject["gap_or_invalid_levels"] == 1


def raw_frame():
    return pd.DataFrame({"Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
                         "Open": [100.] * 3, "High": [101.] * 3, "Low": [99.] * 3,
                         "Close": [100.] * 3, "Volume": [1000.] * 3})


def test_missing_or_duplicate_rows_are_not_compressed():
    d = raw_frame()
    d.loc[1, "Close"] = float("nan")
    with pytest.raises(ValueError, match="missing"):
        normalize_history(d)
    d = raw_frame()
    d.loc[1, "Date"] = d.loc[0, "Date"]
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_history(d)


def test_only_invalid_warmup_prefix_can_be_trimmed():
    d = raw_frame()
    d.loc[0, "Close"] = 98.
    result = normalize_history(d, "2025-01-02")
    assert len(result) == 2
    assert result.attrs["warmup_trimmed_through"] == "2025-01-01"
    with pytest.raises(ValueError, match="evaluated period"):
        normalize_history(d, "2025-01-01")


def test_benchmark_alignment_cannot_read_future_suffix():
    stock = normalize_history(raw_frame()).iloc[:2]
    benchmark = normalize_history(raw_frame())
    benchmark.loc[benchmark.index[-1], "Close"] = 999.
    assert aligned_benchmark(stock, benchmark) == [100., 100.]


def test_expected_completed_session_uses_market_holidays():
    assert expected_last_session("2026-10-09", "KRX") == "2026-10-08"
    assert expected_last_session("2026-07-04", "US") == "2026-07-02"


def test_unfinished_position_still_blocks_later_entry():
    ohl = history()
    ohl["highs"][28] = 111.
    rows, reject = replay([signal(20), signal(26)], ohl)
    # Both would exit at day28: first is not censored when target hit, so it occupies slot.
    assert len(rows) == 1 and rows[0]["signal_index"] == 20
    ohl["highs"][28] = 101.
    rows, reject = replay([signal(20), signal(26)], ohl)
    assert not rows and reject["right_censored"] == 1 and reject["overlapping_position"] == 1
