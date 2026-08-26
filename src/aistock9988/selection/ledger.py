from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PREDICTION_COLUMNS = ("asof", "ts_code", "score", "rank", "feature_set_id", "model_id")
CANDIDATE_COLUMNS = PREDICTION_COLUMNS + ("candidate_rank",)


@dataclass(frozen=True)
class LedgerArtifact:
    path: Path
    row_count: int
    content_hash: str


def build_prediction_ledger(frame: pd.DataFrame, *, asof: str, feature_set_id: str, model_id: str) -> pd.DataFrame:
    missing = {"ts_code", "score"} - set(frame.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    out = frame[["ts_code", "score"]].copy()
    out["score"] = pd.to_numeric(out["score"], errors="raise")
    if out["ts_code"].duplicated().any():
        raise ValueError("prediction ledger has duplicate ts_code")
    out = out.sort_values(["score", "ts_code"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    out.insert(0, "asof", asof)
    out["rank"] = range(1, len(out) + 1)
    out["feature_set_id"] = feature_set_id
    out["model_id"] = model_id
    return out[list(PREDICTION_COLUMNS)]


def freeze_candidates(predictions: pd.DataFrame, *, top_n: int = 20) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"missing ledger columns: {sorted(missing)}")
    out = predictions.sort_values(["rank", "ts_code"], kind="mergesort").head(top_n).copy()
    out["candidate_rank"] = range(1, len(out) + 1)
    return out[list(CANDIDATE_COLUMNS)].reset_index(drop=True)


def write_ledger(frame: pd.DataFrame, path: Path) -> LedgerArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False, lineterminator="\n").encode()
    path.write_bytes(payload)
    return LedgerArtifact(path, len(frame), hashlib.sha256(payload).hexdigest())
