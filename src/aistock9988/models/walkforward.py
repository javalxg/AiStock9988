from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ModelWindow:
    model_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    prediction_date: pd.Timestamp


def monthly_windows(*, train_start: str, prediction_start: str, prediction_end: str,
                    window_months: int = 12, rebalance: str = "weekly") -> list[ModelWindow]:
    if window_months <= 0 or rebalance != "weekly":
        raise ValueError("only positive monthly window and weekly rebalance are supported")
    start = pd.Timestamp(train_start)
    pred_start = pd.Timestamp(prediction_start)
    pred_end = pd.Timestamp(prediction_end)
    dates = pd.date_range(pred_start, pred_end, freq="W-FRI")
    # One model per calendar month; weekly predictions reuse the same monthly model.
    model_dates = sorted(set(pd.Timestamp(d.year, d.month, 1) for d in dates))
    windows: list[ModelWindow] = []
    for month in model_dates:
        prediction = next(d for d in dates if d.year == month.year and d.month == month.month)
        train_end = month - pd.Timedelta(days=1)
        train_begin = max(start, train_end - pd.DateOffset(months=window_months))
        model_id = f"q70_{prediction.strftime('%Y%m')}_cutoff_{train_end.strftime('%Y%m%d')}"
        windows.append(ModelWindow(model_id, train_begin, train_end, prediction))
    return windows


def validate_window_labels(labels: pd.DataFrame, window: ModelWindow, *, available_column: str = "available_time") -> None:
    if available_column not in labels.columns:
        raise ValueError(f"missing label PIT column: {available_column}")
    available = pd.to_datetime(labels[available_column], errors="raise", utc=True)
    cutoff = window.train_end
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    if bool((available > cutoff).any()):
        raise AssertionError(f"window {window.model_id} contains labels newer than cutoff")
