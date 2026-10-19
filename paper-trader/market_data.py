"""Intraday market-data completeness checks shared by backtest and paper simulator."""

from __future__ import annotations

from datetime import time as dt_time

import pandas as pd


_INTERVALS = {
    "5m": ("5min", dt_time(15, 25)),
    "15m": ("15min", dt_time(15, 15)),
}


def expected_session_times(interval: str) -> tuple[dt_time, ...]:
    """Return expected NSE regular-session candle timestamps for an interval."""
    if interval not in _INTERVALS:
        raise ValueError(f"Unsupported intraday interval: {interval}")
    frequency, final_bar = _INTERVALS[interval]
    return tuple(
        pd.date_range(
            start="2000-01-01 09:15",
            end=f"2000-01-01 {final_bar:%H:%M}",
            freq=frequency,
        ).time
    )


def is_complete_session(index: pd.Index, interval: str) -> bool:
    """Return True only for one sorted, duplicate-free, fully populated NSE session."""
    expected = expected_session_times(interval)
    try:
        stamps = pd.DatetimeIndex(index)
    except (TypeError, ValueError):
        return False
    if len(stamps) != len(expected) or stamps.has_duplicates or not stamps.is_monotonic_increasing:
        return False
    if not len(stamps) or len({stamp.date() for stamp in stamps}) != 1:
        return False
    actual_times = tuple(stamp.time().replace(tzinfo=None) for stamp in stamps)
    expected_times = tuple(value.replace(tzinfo=None) for value in expected)
    return actual_times == expected_times
