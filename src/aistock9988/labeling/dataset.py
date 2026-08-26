from __future__ import annotations

import pandas as pd

from .maturity import assert_labels_mature
from ..data.pit import assert_no_future
from ..features.registry import FeatureSet, assemble_matrix


def build_training_dataset(features: pd.DataFrame, labels: pd.DataFrame, *, feature_set: FeatureSet,
                           training_cutoff: pd.Timestamp, signal_column: str = "event_time") -> tuple[pd.DataFrame, pd.Series]:
    """Join point-in-time features with labels; future data is a hard error."""
    assert_no_future(features, decision_time=training_cutoff.to_pydatetime())
    assert_labels_mature(labels, training_cutoff=training_cutoff)
    if signal_column not in labels.columns or "ts_code" not in labels.columns:
        raise ValueError("labels require ts_code and signal/event time")
    feature_frame = assemble_matrix(features, feature_set)
    left = feature_frame.rename(columns={"event_time": signal_column})
    right = labels[["ts_code", signal_column, "label_return"]].copy()
    left[signal_column] = pd.to_datetime(left[signal_column], utc=True)
    right[signal_column] = pd.to_datetime(right[signal_column], utc=True)
    if right[["ts_code", signal_column]].duplicated().any():
        raise ValueError("duplicate label key")
    merged = left.merge(right, on=["ts_code", signal_column], how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("no feature/label keys matched")
    merged = merged.sort_values([signal_column, "ts_code"], kind="mergesort").reset_index(drop=True)
    return merged[list(feature_set.columns)], merged["label_return"]
