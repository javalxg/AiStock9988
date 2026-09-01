"""Frozen daily cross-sectional preprocessing for the F0=123 contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .registry import FeatureSet


@dataclass(frozen=True)
class F0CrossSectionAudit:
    source_rows: int
    eligible_rows: int
    output_rows: int
    source_sessions: int
    output_sessions: int
    minimum_non_null_features: int
    maximum_rows_per_date: int | None


def prepare_f0_cross_sections(
    panel: pd.DataFrame,
    feature_set: FeatureSet,
    *,
    minimum_non_null_features: int,
    maximum_rows_per_date: int | None = None,
    sample_seed: int = 42,
    uncapped_dates: Iterable[object] = (),
) -> tuple[pd.DataFrame, F0CrossSectionAudit]:
    """Normalize each date independently and optionally cap training rows.

    Missing values remain missing. Sampling happens after normalization so a
    training cap cannot change the full-market cross-sectional percentiles.
    """
    metadata = ["ts_code", "event_time", "available_time"]
    missing = (set(metadata) | set(feature_set.columns)) - set(panel.columns)
    if missing:
        raise ValueError(f"F0 panel missing registered columns: {sorted(missing)}")
    if not 1 <= minimum_non_null_features <= len(feature_set.columns):
        raise ValueError("minimum_non_null_features is outside the F0 manifest")
    if maximum_rows_per_date is not None and maximum_rows_per_date < 2:
        raise ValueError("maximum_rows_per_date must be at least two")

    source = panel[[*metadata, *feature_set.columns]].copy()
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True).dt.normalize()
    source["available_time"] = pd.to_datetime(source["available_time"], utc=True)
    source["ts_code"] = source["ts_code"].astype(str)
    if source.duplicated(["event_time", "ts_code"]).any():
        raise ValueError("F0 panel contains duplicate stock-session keys")

    values = source[list(feature_set.columns)].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    eligible = values.notna().sum(axis=1).ge(minimum_non_null_features)
    source = source.loc[eligible, metadata].copy()
    values = values.loc[eligible]
    uncapped = set(pd.to_datetime(list(uncapped_dates), utc=True).normalize())

    parts: list[pd.DataFrame] = []
    for day, index in source.groupby("event_time", sort=True).groups.items():
        day_values = values.loc[index]
        ranked = day_values.rank(method="average", pct=True)
        normalized = (ranked - ranked.mean()) / ranked.std(ddof=1).replace(0.0, np.nan)
        normalized = normalized.astype("float32")
        day_frame = source.loc[index].copy()
        day_frame.loc[:, list(feature_set.columns)] = normalized.to_numpy()
        if (
            maximum_rows_per_date is not None
            and day not in uncapped
            and len(day_frame) > maximum_rows_per_date
        ):
            day_frame = day_frame.sample(n=maximum_rows_per_date, random_state=sample_seed)
        parts.append(day_frame)

    result = pd.concat(parts, ignore_index=True) if parts else source.iloc[0:0].copy()
    result = result.sort_values(["event_time", "ts_code"], kind="mergesort").reset_index(drop=True)
    if result.empty:
        raise ValueError("F0 preprocessing removed every stock-session")
    audit = F0CrossSectionAudit(
        source_rows=int(len(panel)),
        eligible_rows=int(eligible.sum()),
        output_rows=int(len(result)),
        source_sessions=int(pd.to_datetime(panel["event_time"], utc=True).dt.normalize().nunique()),
        output_sessions=int(result["event_time"].nunique()),
        minimum_non_null_features=int(minimum_non_null_features),
        maximum_rows_per_date=maximum_rows_per_date,
    )
    return result, audit


__all__ = ["F0CrossSectionAudit", "prepare_f0_cross_sections"]
