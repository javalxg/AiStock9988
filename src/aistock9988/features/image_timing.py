"""PIT-safe indicators for the screenshot timing-rule diagnostic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import hashlib
import json

from ..data.bundle import DataBundle


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    typical = (high + low + close) / 3.0
    mean = typical.rolling(window, min_periods=window).mean()
    mad = typical.rolling(window, min_periods=window).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))), raw=True
    )
    return (typical - mean) / (0.015 * mad.replace(0.0, np.nan))


def _daily_indicators(group: pd.DataFrame) -> pd.DataFrame:
    out = group.sort_values("trade_date", kind="mergesort").copy()
    close = pd.to_numeric(out["economic_close"], errors="coerce")
    high = pd.to_numeric(out["economic_high"], errors="coerce")
    low = pd.to_numeric(out["economic_low"], errors="coerce")
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
    out["daily_dif"] = dif
    out["daily_dea"] = dea
    out["daily_macd_cross"] = ((dif > dea) & (dif.shift(1) <= dea.shift(1))).astype(float)
    cci = _cci(high, low, close, 14)
    out["daily_cci"] = cci
    out["daily_cci_cross_100"] = ((cci > 100.0) & (cci.shift(1) <= 100.0)).astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["daily_rsi"] = 100.0 - (100.0 / (1.0 + rs))
    out["rsi_top_warning"] = (
        out["daily_rsi"].gt(80.0) & out["daily_rsi"].lt(out["daily_rsi"].shift(1))
    ).astype(float)
    return out


def _weekly_indicators(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("trade_date", kind="mergesort").copy()
    g["week"] = g["trade_date"].dt.tz_localize(None).dt.to_period("W-FRI")
    weekly = g.groupby("week", sort=True).agg(
        week_end=("trade_date", "max"),
        open=("economic_open", "first"),
        high=("economic_high", "max"),
        low=("economic_low", "min"),
        close=("economic_close", "last"),
        amount=("amount", "sum"),
    ).reset_index()
    weekly["ema12"] = weekly["close"].ewm(span=12, adjust=False, min_periods=12).mean()
    weekly["ema50"] = weekly["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    low9 = weekly["low"].rolling(9, min_periods=9).min()
    high9 = weekly["high"].rolling(9, min_periods=9).max()
    denominator = (high9 - low9).replace(0.0, np.nan)
    rsv = (weekly["close"] - low9) / denominator * 100.0
    weekly["k"] = rsv.ewm(alpha=1.0 / 3.0, adjust=False, min_periods=1).mean()
    weekly["d"] = weekly["k"].ewm(alpha=1.0 / 3.0, adjust=False, min_periods=1).mean()
    weekly["j"] = 3.0 * weekly["k"] - 2.0 * weekly["d"]
    weekly["weekly_trend"] = (
        (weekly["close"] > weekly["ema12"]) & (weekly["ema12"] > weekly["ema50"])
    ).astype(float)
    weekly["weekly_j_oversold"] = weekly["j"].lt(20.0).astype(float)
    return weekly[["week_end", "ema12", "ema50", "j", "weekly_trend", "weekly_j_oversold"]]


def build_image_timing_feature_ledger(bundle: DataBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return daily rows plus a compact RSI-warning audit frame.

    All fields are calculated from economic OHLCV rows at or before ``asof``;
    no database ingestion timestamp is used to backfill a historical signal.
    """
    frame = bundle.execution.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], utc=True).dt.normalize()
    frame = frame.sort_values(["ts_code", "trade_date"], kind="mergesort")
    parts = []
    warning_rows = []
    for code, group in frame.groupby("ts_code", sort=True):
        daily = _daily_indicators(group)
        weekly = _weekly_indicators(daily)
        daily["week"] = daily["trade_date"].dt.tz_localize(None).dt.to_period("W-FRI")
        weekly["week"] = weekly["week_end"].dt.tz_localize(None).dt.to_period("W-FRI")
        daily = daily.merge(weekly, on="week", how="left", validate="many_to_one")
        # A completed weekly bar is only observable on its final exchange
        # session. Earlier rows keep the week key for diagnostics but must not
        # inherit the eventual week's close/EMA/KDJ values.
        is_week_end = daily["trade_date"].eq(daily["week_end"])
        for column in ("ema12", "ema50", "j", "weekly_trend", "weekly_j_oversold"):
            daily.loc[~is_week_end, column] = np.nan
        warning_rows.append(daily.loc[daily["rsi_top_warning"].eq(1), ["trade_date", "ts_code", "daily_rsi"]])
        parts.append(daily)
    out = pd.concat(parts, ignore_index=True)
    out["stable_rank"] = out["ts_code"].astype(str)
    feature_payload = {
        "provider": "image_timing_v1",
        "daily": {"macd": [12, 26, 9], "cci": 14, "rsi": 14},
        "weekly": {"ema": [12, 50], "kdj": [9, 3, 3]},
        "window_semantics": "per_security_observed_sessions",
    }
    out["feature_set_hash"] = hashlib.sha256(
        json.dumps(feature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    out["feature_ready"] = (
        out["universe_pass"].astype(bool)
        & out["selection_data_eligible"].astype(bool)
        & np.isfinite(out[["daily_macd_cross", "daily_cci_cross_100", "weekly_trend", "weekly_j_oversold", "j"]].to_numpy(dtype=float)).all(axis=1)
    )
    out["feature_rejection_reason"] = np.select(
        [~out["universe_pass"].astype(bool), ~out["selection_data_eligible"].astype(bool), ~out["feature_ready"]],
        ["UNIVERSE_REJECTED", out.get("selection_data_rejection_reason", "SELECTION_DATA_INELIGIBLE"), "FEATURE_NOT_MATURE"],
        default="",
    )
    out = out.rename(columns={"trade_date": "asof"})
    return out, pd.concat(warning_rows, ignore_index=True) if warning_rows else pd.DataFrame()


__all__ = ["build_image_timing_feature_ledger"]
