from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FeatureSet:
    id: str
    columns: tuple[str, ...]
    order_hash: str

    @classmethod
    def create(cls, id: str, columns: list[str] | tuple[str, ...]) -> "FeatureSet":
        cols = tuple(columns)
        if not cols or len(set(cols)) != len(cols):
            raise ValueError("feature columns must be non-empty and unique")
        digest = hashlib.sha256(json.dumps(cols, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        return cls(id, cols, digest)

    @classmethod
    def from_f0_json(cls, path: str | Path) -> "FeatureSet":
        spec = json.loads(Path(path).read_text())
        technical = list(spec["technical"])
        fundamental = list(spec["fundamental"])
        suffix = spec["sector_relative_suffix"]
        columns = technical + fundamental + [f"{name}{suffix}" for name in technical]
        expected = int(spec["expected_counts"]["total"])
        result = cls.create(spec["id"], columns)
        if len(columns) != expected or result.order_hash != spec["column_order_hash"]:
            raise ValueError("F0 feature manifest count or order hash mismatch")
        return result


def assemble_matrix(frame: pd.DataFrame, feature_set: FeatureSet, *, key_columns=("ts_code", "event_time"),
                    allow_missing: bool = False) -> pd.DataFrame:
    missing = [c for c in (*key_columns, *feature_set.columns) if c not in frame.columns]
    if missing:
        raise ValueError(f"feature snapshot missing columns: {missing}")
    out = frame[[*key_columns, *feature_set.columns]].copy()
    if out[list(key_columns)].duplicated().any():
        raise ValueError("duplicate security/time key in feature snapshot")
    if not allow_missing and out[list(feature_set.columns)].isna().any().any():
        bad = out[list(feature_set.columns)].columns[out[list(feature_set.columns)].isna().any()].tolist()
        raise ValueError(f"feature snapshot contains missing values: {bad}")
    for col in feature_set.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            raise TypeError(f"feature {col} must be numeric")
    return out.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)
