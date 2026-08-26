from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from aistock9988.data.q70_source import load_f0_panel
from aistock9988.features.registry import FeatureSet
from aistock9988.models.pipeline import train_and_rank


def main() -> None:
    panel = load_f0_panel("2025-01-01", "2026-01-23")
    spec = FeatureSet.from_f0_json(Path(__file__).parents[1] / "configs/feature_sets/f0_123_columns.json")
    ordered = panel.sort_values(["ts_code", "event_time"], kind="mergesort").copy()
    # T+1 entry and T+10 exit: ten sessions after signal in this endpoint label.
    ordered["exit_open"] = ordered.groupby("ts_code", sort=False)["open"].shift(-10)
    ordered["exit_time"] = ordered.groupby("ts_code", sort=False)["event_time"].shift(-10)
    labels = ordered[["ts_code", "event_time", "exit_time", "exit_open"]].dropna().copy()
    labels["available_time"] = labels["exit_time"] + pd.Timedelta(hours=15)
    labels["label_return"] = labels["exit_open"] / ordered.loc[labels.index, "open"] - 1.0
    cutoff = pd.Timestamp("2025-12-31T23:59:59Z")
    train_features = panel[panel.event_time <= cutoff].copy()
    train_labels = labels[labels.available_time <= cutoff].copy()
    pred = panel[panel.event_time == pd.Timestamp("2026-01-09", tz="UTC")].copy()
    if pred.empty:
        raise RuntimeError("2026-01-09 prediction session is absent")
    out = Path(__file__).parents[1] / "experiments/.running/first_q70_202601"
    out.mkdir(parents=True, exist_ok=True)
    result = train_and_rank(features=train_features, labels=train_labels, prediction_features=pred,
                            feature_set=spec, training_cutoff=cutoff, asof="2026-01-09",
                            model_id="q70_202601_cutoff_20251231", output_dir=out,
                            params={"n_estimators": 200, "max_depth": 6}, top_n=20)
    print(result.prediction_path)
    print(result.candidate_path)


if __name__ == "__main__":
    main()
