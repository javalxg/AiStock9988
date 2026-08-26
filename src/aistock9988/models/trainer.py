from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    feature_set_id: str
    label_profile_id: str
    training_cutoff: str
    feature_names: tuple[str, ...]
    row_count: int
    group_count: int
    model_sha256: str
    metadata_sha256: str


def train_ranker(X: pd.DataFrame, y: pd.Series, *, group_dates: pd.Series,
                 feature_set_id: str, label_profile_id: str, training_cutoff: str,
                 model_id: str, output_dir: Path, params: dict | None = None) -> ModelArtifact:
    if X.empty or len(X) != len(y) or len(X) != len(group_dates):
        raise ValueError("X, y and group_dates must be non-empty and aligned")
    if X.columns.duplicated().any():
        raise ValueError("duplicate feature names")
    if X.isna().any().any() or pd.isna(y).any():
        raise ValueError("training input contains null values")
    order = pd.Series(group_dates.astype(str).to_numpy(), index=X.index).sort_values(kind="mergesort").index
    X = X.loc[order].reset_index(drop=True)
    y = y.loc[order].reset_index(drop=True)
    grouped_dates = group_dates.loc[order].reset_index(drop=True).astype(str)
    groups = grouped_dates.groupby(grouped_dates, sort=False).size().tolist()
    if any(n < 2 for n in groups):
        raise ValueError("each ranking date group must contain at least two rows")
    defaults = dict(objective="rank:pairwise", n_estimators=200, max_depth=6,
                    learning_rate=0.05, min_child_weight=5, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                    random_state=42, n_jobs=1, tree_method="hist")
    defaults.update(params or {})
    model = XGBRanker(**defaults)
    model.fit(X, y.to_numpy(dtype=float), qid=np.repeat(np.arange(len(groups)), groups))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_id}.json"
    metadata_path = output_dir / f"{model_id}.metadata.json"
    if model_path.exists() or metadata_path.exists():
        raise FileExistsError(f"immutable model artifact already exists: {model_id}")
    model.save_model(model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {
        "model_id": model_id, "feature_set_id": feature_set_id,
        "label_profile_id": label_profile_id, "training_cutoff": training_cutoff,
        "feature_names": list(X.columns), "row_count": len(X), "group_count": len(groups),
        "params": defaults, "model_sha256": model_hash,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    return ModelArtifact(model_id, feature_set_id, label_profile_id, training_cutoff,
                         tuple(X.columns), len(X), len(groups), model_hash, metadata_hash)


def audit_model(model_path: Path, metadata_path: Path) -> ModelArtifact:
    metadata = json.loads(metadata_path.read_text())
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual != metadata.get("model_sha256"):
        raise AssertionError("model hash mismatch")
    metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    return ModelArtifact(metadata["model_id"], metadata["feature_set_id"], metadata["label_profile_id"],
                         metadata["training_cutoff"], tuple(metadata["feature_names"]),
                         int(metadata["row_count"]), int(metadata["group_count"]), actual, metadata_hash)
