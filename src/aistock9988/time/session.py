"""Exchange-session timestamps used by all PIT decisions."""
from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd


SHANGHAI = ZoneInfo("Asia/Shanghai")


def session_close(value: object) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if day.tzinfo is not None:
        day = day.tz_convert(SHANGHAI)
    local = datetime.combine(day.date(), time(15, 0), tzinfo=SHANGHAI)
    return pd.Timestamp(local).tz_convert("UTC")


def session_open(value: object) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if day.tzinfo is not None:
        day = day.tz_convert(SHANGHAI)
    local = datetime.combine(day.date(), time(9, 30), tzinfo=SHANGHAI)
    return pd.Timestamp(local).tz_convert("UTC")


def normalize_session_day(value: object) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if day.tzinfo is not None:
        day = day.tz_convert("UTC")
    return day.normalize().tz_localize("UTC") if day.tzinfo is None else day.normalize()


def parse_source_time(values: object) -> pd.Series | pd.Timestamp:
    """Parse DB timestamps: naive MySQL DATETIME is exchange-local time."""
    parsed = pd.to_datetime(values, errors="raise")
    if isinstance(parsed, pd.Timestamp):
        return parsed.tz_localize(SHANGHAI).tz_convert("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(SHANGHAI).dt.tz_convert("UTC")
    return parsed.dt.tz_convert("UTC")
