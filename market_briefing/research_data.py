"""Date-preserving adjusted OHLCV collection for reproducible one-year research."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED = ["Open", "High", "Low", "Close", "Volume"]


def normalize_history(frame: pd.DataFrame, trim_invalid_before: str | None = None) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    rename = {str(c).lower(): c for c in data.columns}
    if "date" in rename:
        dates = data[rename["date"]].map(lambda v: str(v)[:10])
    else:
        dates = pd.Series([str(v)[:10] for v in data.index], index=data.index)
    parsed = pd.to_datetime(dates, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Invalid daily dates")
    output = pd.DataFrame(index=pd.DatetimeIndex(parsed.to_numpy()))
    for key in REQUIRED:
        if key.lower() not in rename:
            raise ValueError(f"Missing {key}")
        output[key] = pd.to_numeric(data[rename[key.lower()]], errors="coerce").to_numpy()
    if output.index.has_duplicates or not output.index.is_monotonic_increasing:
        raise ValueError("Duplicate or unsorted daily dates")
    values = output.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (output[REQUIRED[:-1]] <= 0).any().any() or (output.Volume < 0).any():
        raise ValueError("Invalid or missing daily OHLCV; no compression allowed")
    invalid = ((output.High + output.Close * 1e-10 < output[["Open", "Close", "Low"]].max(axis=1))
               | (output.Low - output.Close * 1e-10 > output[["Open", "Close", "High"]].min(axis=1)))
    if invalid.any():
        # An invalid warmup prefix may be trimmed, never a row within the tested year.
        last_invalid = output.index[invalid][-1]
        if trim_invalid_before is None or last_invalid >= pd.Timestamp(trim_invalid_before):
            raise ValueError("Invalid OHLC ordering in evaluated period")
        output = output[output.index > last_invalid]
        output.attrs["warmup_trimmed_through"] = last_invalid.date().isoformat()
    output.index.name = "Date"
    return output


def expected_last_session(as_of: str, market: str) -> str:
    # Only completed previous calendar days. US holiday calendar is approximated by NYSE holidays.
    import holidays
    current = pd.Timestamp(as_of) - pd.Timedelta(days=1)
    holiday_dates = (holidays.KR(years=[current.year - 1, current.year]) if market == "KRX"
                     else holidays.NYSE(years=[current.year - 1, current.year]))
    while current.weekday() >= 5 or current.date() in holiday_dates:
        current -= pd.Timedelta(days=1)
    return current.date().isoformat()


def frame_to_ohlcv(frame: pd.DataFrame) -> dict:
    return {"dates": [d.date().isoformat() for d in frame.index],
            **{key: frame[source].tolist() for key, source in
               (("opens", "Open"), ("highs", "High"), ("lows", "Low"),
                ("closes", "Close"), ("volumes", "Volume"))}}


def collect_histories(symbols: list[str], as_of: str, cache_dir: Path, offline: bool = False,
                      fallback_dirs: list[Path] | None = None, workers: int = 4) -> tuple[dict, list[dict]]:
    end = pd.Timestamp(as_of)
    start = end - pd.DateOffset(years=2)
    cache_dir.mkdir(parents=True, exist_ok=True)
    directories = [cache_dir] + list(fallback_dirs or [])
    results, evidence = {}, []

    def collect(symbol):
        history, source, error = None, "", ""
        target = cache_dir / (symbol.replace(".", "_") + ".csv")
        if not offline:
            try:
                import yfinance as yf
                raw = yf.Ticker(symbol).history(start=start.date().isoformat(), end=as_of,
                    auto_adjust=True, actions=False, timeout=15, raise_errors=True)
                if raw is None or raw.empty:
                    raise ValueError("Empty provider history")
                history = normalize_history(raw, (end - pd.DateOffset(years=1)).date().isoformat())
                history.to_csv(target)
                source = "yfinance_auto_adjust_true"
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:150]}"
        if history is None:
            for directory in directories:
                for name in (symbol.replace(".", "_") + ".csv", symbol + ".csv"):
                    path = directory / name
                    if not path.exists():
                        continue
                    try:
                        history = normalize_history(pd.read_csv(path), (end - pd.DateOffset(years=1)).date().isoformat())
                        source = f"cache:{path.resolve()}"
                        break
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {str(exc)[:150]}"
                if history is not None:
                    break
        market = "KRX" if symbol.endswith((".KS", ".KQ")) or symbol in ("^KS11", "^KQ11") else "US"
        expected = expected_last_session(as_of, market)
        if history is not None:
            history = history[(history.index >= start) & (history.index < end)]
            last = history.index[-1].date().isoformat() if len(history) else ""
            record = {"symbol": symbol, "market": market, "source": source,
                      "first_date": history.index[0].date().isoformat() if len(history) else "",
                      "last_date": last, "expected_last_date": expected, "bars": len(history),
                      "fresh": last >= expected, "download_error": error,
                      "warmup_trimmed_through": history.attrs.get("warmup_trimmed_through"),
                      "adjustment_provenance": "explicit_adjusted" if source.startswith("yfinance") else "legacy_cache_not_independently_verified"}
            return symbol, history, record
        return symbol, None, {"symbol": symbol, "market": market, "source": source,
                              "fresh": False, "bars": 0, "download_error": error or "No valid cache"}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(collect, s): s for s in dict.fromkeys(symbols)}
        for task in as_completed(tasks):
            symbol, history, record = task.result()
            evidence.append(record)
            if history is not None and len(history) >= 60:
                results[symbol] = history
    return results, sorted(evidence, key=lambda e: e["symbol"])


def aligned_benchmark(frame: pd.DataFrame, benchmark: pd.DataFrame | None) -> list[float]:
    if benchmark is None:
        return []
    # Reindex by date; never tail-align different trading calendars or backfill future information.
    return benchmark.Close.reindex(frame.index).ffill().tolist()
