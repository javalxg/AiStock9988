from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from aistock9988.data.q70_source import load_f0_panel
from aistock9988.data.snapshot import build_snapshot_meta
from aistock9988.features.registry import FeatureSet
from aistock9988.labeling.maturity import LabelProfile, mature_training_rows
from aistock9988.labeling.q70 import build_q70_t10_labels
from aistock9988.models.pipeline import train_and_rank


def run_training(*, run_dir: Path, start: str, end: str, cutoff_value: str, asof: str):
    run_dir = run_dir.resolve()
    if not (run_dir / "RUN_STATUS.json").is_file():
        raise RuntimeError("run directory must be created by the project CLI")
    panel, data_audit = load_f0_panel(start, end, return_audit=True)
    spec = FeatureSet.from_f0_json(Path(__file__).parents[1] / "configs/feature_sets/f0_123_columns.json")
    sessions = pd.DatetimeIndex(sorted(panel["event_time"].drop_duplicates()))
    labels = build_q70_t10_labels(
        panel,
        profile=LabelProfile("label.endpoint_open_open_t10.v1", 1, 10, 11),
        session_dates=sessions,
    )
    cutoff = pd.Timestamp(cutoff_value)
    train_features = panel[panel.available_time <= cutoff].copy()
    train_labels = mature_training_rows(labels[labels.available_time <= cutoff].copy(), training_cutoff=cutoff)
    pred = panel[panel.event_time == pd.Timestamp(asof, tz="UTC")].copy()
    if pred.empty:
        raise RuntimeError(f"{asof} prediction session is absent")
    snapshot = build_snapshot_meta(panel, source_id="quant_db.q70_f0", query={"start": start, "end": end})
    manifest_path = run_dir / "data_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"immutable data manifest already exists: {manifest_path}")
    manifest_path.write_text(json.dumps({"snapshot": asdict(snapshot), "industry": data_audit,
                                         "pit_rule": "available_time <= decision_time"},
                                        ensure_ascii=False, indent=2) + "\n")
    model_id = f"q70_{pd.Timestamp(asof).strftime('%Y%m')}_cutoff_{cutoff.strftime('%Y%m%d')}"
    result = train_and_rank(features=train_features, labels=train_labels, prediction_features=pred,
                            feature_set=spec, training_cutoff=cutoff, asof=asof,
                            model_id=model_id, output_dir=run_dir,
                            params={"n_estimators": 200, "max_depth": 6}, top_n=20)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one auditable q70 training window")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-23")
    parser.add_argument("--cutoff", default="2025-12-31T23:59:59Z")
    parser.add_argument("--asof", default="2026-01-09")
    args = parser.parse_args()
    result = run_training(run_dir=args.run_dir, start=args.start, end=args.end,
                          cutoff_value=args.cutoff, asof=args.asof)
    print(result.prediction_path)
    print(result.candidate_path)


if __name__ == "__main__":
    main()
