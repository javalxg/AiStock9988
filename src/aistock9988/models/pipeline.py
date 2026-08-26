from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..features.registry import FeatureSet
from ..labeling.dataset import build_training_dataset
from ..data.pit import assert_no_future
from ..selection.ledger import build_prediction_ledger, freeze_candidates, write_ledger
from .trainer import ModelArtifact, train_ranker


@dataclass(frozen=True)
class TrainingRun:
    model: ModelArtifact
    prediction_path: Path
    candidate_path: Path


def train_and_rank(*, features: pd.DataFrame, labels: pd.DataFrame,
                   prediction_features: pd.DataFrame, feature_set: FeatureSet,
                   training_cutoff: pd.Timestamp, asof: str, model_id: str,
                   output_dir: Path, params: dict | None = None, top_n: int = 20) -> TrainingRun:
    X, y = build_training_dataset(features, labels, feature_set=feature_set,
                                  training_cutoff=training_cutoff)
    # Recover the date group from the already validated training feature keys.
    train_keys = features[["ts_code", "event_time"]].copy()
    train_keys["event_time"] = pd.to_datetime(train_keys["event_time"], utc=True)
    label_keys = labels[["ts_code", "event_time"]].copy()
    label_keys["event_time"] = pd.to_datetime(label_keys["event_time"], utc=True)
    merged_keys = train_keys.merge(label_keys, on=["ts_code", "event_time"], how="inner", validate="one_to_one")
    merged_keys = merged_keys.sort_values(["event_time", "ts_code"], kind="mergesort")
    if len(merged_keys) != len(X):
        raise ValueError("training key alignment mismatch")
    model = train_ranker(X, y, group_dates=merged_keys["event_time"],
                         feature_set_id=feature_set.id, label_profile_id="configured",
                         training_cutoff=str(training_cutoff), model_id=model_id,
                         output_dir=output_dir / "models", params=params)
    required_pred = {"ts_code", "event_time", "available_time", *feature_set.columns}
    missing_pred = required_pred - set(prediction_features.columns)
    if missing_pred:
        raise ValueError(f"prediction snapshot missing columns: {sorted(missing_pred)}")
    pred_source = prediction_features.copy()
    pred_source["event_time"] = pd.to_datetime(pred_source["event_time"], utc=True)
    pred_source["available_time"] = pd.to_datetime(pred_source["available_time"], utc=True)
    decision = pd.Timestamp(asof)
    if decision.tzinfo is None:
        decision = decision.tz_localize("UTC")
    # A date-form asof denotes the end of that trading session, not midnight.
    if decision == decision.normalize():
        decision = decision + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    assert_no_future(pred_source, decision_time=decision.to_pydatetime())
    if (pred_source["event_time"].dt.normalize() != decision.normalize()).any():
        raise ValueError("prediction snapshot contains rows outside the requested asof session")
    if pred_source["ts_code"].duplicated().any():
        raise ValueError("prediction snapshot has duplicate ts_code")
    pred = pred_source[["ts_code", *feature_set.columns]].copy()
    scores = pd.Series(model_for_prediction(output_dir / "models" / f"{model_id}.json", pred[list(feature_set.columns)]), index=pred.index)
    predictions = build_prediction_ledger(pd.DataFrame({"ts_code": pred["ts_code"], "score": scores}),
                                          asof=asof, feature_set_id=feature_set.id, model_id=model_id)
    prediction_path = output_dir / "predictions" / f"{asof}_prediction.csv"
    candidate_path = output_dir / "selections" / f"{asof}_top{top_n}.csv"
    write_ledger(predictions, prediction_path)
    write_ledger(freeze_candidates(predictions, top_n=top_n), candidate_path)
    return TrainingRun(model, prediction_path, candidate_path)


def model_for_prediction(model_path: Path, X: pd.DataFrame):
    from xgboost import XGBRanker
    model = XGBRanker()
    model.load_model(model_path)
    return model.predict(X)
