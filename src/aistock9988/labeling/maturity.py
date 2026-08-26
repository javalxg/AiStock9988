from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LabelProfile:
    id: str
    entry_delay_sessions: int
    horizon_sessions: int
    maturity_sessions: int
    entry_price: str = "economic_open"
    exit_price: str = "economic_open"


def assert_labels_mature(labels: pd.DataFrame, *, training_cutoff: pd.Timestamp,
                         available_column: str = "available_time") -> None:
    if available_column not in labels.columns:
        raise ValueError(f"missing label PIT column: {available_column}")
    available = pd.to_datetime(labels[available_column], errors="raise", utc=True)
    cutoff = pd.Timestamp(training_cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    future = available > cutoff
    if bool(future.any()):
        raise AssertionError(
            f"label leakage: {int(future.sum())} labels mature after training cutoff "
            f"({cutoff.isoformat()})"
        )


def mature_training_rows(labels: pd.DataFrame, *, training_cutoff: pd.Timestamp,
                         available_column: str = "available_time") -> pd.DataFrame:
    """Return only mature rows, while failing if the caller supplied future labels."""
    assert_labels_mature(labels, training_cutoff=training_cutoff, available_column=available_column)
    return labels.copy().reset_index(drop=True)


def build_endpoint_labels(prices: pd.DataFrame, *, profile: LabelProfile,
                          signal_column: str = "signal_time", entry_column: str = "entry_time",
                          exit_column: str = "exit_time", price_column: str = "economic_open",
                          entry_price_column: str | None = None,
                          exit_price_column: str | None = None) -> pd.DataFrame:
    """Build labels from a pre-joined causal price panel.

    The caller must provide only rows whose exit observation is already available; this function
    records that observation time explicitly instead of inferring maturity from trade_date.
    """
    entry_price_column = entry_price_column or price_column
    exit_price_column = exit_price_column or price_column
    required = {"ts_code", signal_column, entry_column, exit_column, entry_price_column, exit_price_column}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"missing label columns: {missing}")
    out = prices[["ts_code", signal_column, entry_column, exit_column,
                  entry_price_column, exit_price_column]].copy()
    out[signal_column] = pd.to_datetime(out[signal_column], utc=True)
    out[entry_column] = pd.to_datetime(out[entry_column], utc=True)
    out[exit_column] = pd.to_datetime(out[exit_column], utc=True)
    out["available_time"] = out[exit_column]
    entry = pd.to_numeric(out[entry_price_column], errors="raise")
    exit_ = pd.to_numeric(out[exit_price_column], errors="raise")
    if bool((entry <= 0).any()) or bool((exit_ <= 0).any()):
        raise ValueError("label prices must be positive")
    out["label_return"] = exit_ / entry - 1.0
    return out
