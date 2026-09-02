#!/usr/bin/env python3
"""Run the preregistered CAP1 same-day top-30% classifier experiment."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

from aistock9988.configuration import StrategyConfig
from aistock9988.features.registry import FeatureSet
from aistock9988.time.session import session_close
from conditional_f0_123_reset_ranker_runner import (
    _build_stage1_period,
    _candidate_ledgers,
    _load_candidate_f0,
    _period_plan,
    _run_portfolios,
    _sha,
    _source_cutoffs,
)


ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "configs/strategy/reset_weak_confirm_v3_cap1_20.yaml"
PROFILE = ROOT / "configs/model_profiles/f0_123_cap1_top30_classifier_v1.yaml"
PREREG = ROOT / "docs/council_20260828/CAP1_F0_123_TOP30_CLASSIFIER_V1_PREREG_20260902.md"
OUTPUT = ROOT / "docs/council_20260828/CAP1_F0_123_TOP30_CLASSIFIER_V1_2026_TO_DB_CUTOFF_20260902"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load_profile() -> dict[str, Any]:
    config = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    if config["identity"] != {"model_profile_id": "f0_123_cap1_top30_classifier_v1", "version": 1}:
        raise ValueError("unexpected top-30 model profile")
    if config["model"]["objective"] != "binary:logistic" or config["evaluation"]["parameter_sweep"] is not False:
        raise ValueError("top-30 profile must remain a fixed binary classifier")
    if config["label"]["class_definition"] != "within_stage1_date_top_30pct_executable_return":
        raise ValueError("unexpected class definition")
    return config


def _top30_labels(labels: pd.DataFrame, minimum_group_size: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = labels.rename(columns={"available_time": "label_available_time"}).copy()
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True).dt.normalize()
    frame = frame.sort_values(["event_time", "label_return", "ts_code"], ascending=[True, False, True], kind="mergesort")
    frame["group_size"] = frame.groupby("event_time", sort=False)["ts_code"].transform("size")
    frame["return_rank"] = frame.groupby("event_time", sort=False).cumcount() + 1
    valid = frame["group_size"].ge(minimum_group_size)
    frame["top30_label"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    cutoff = np.ceil(frame.loc[valid, "group_size"].astype(float) * 0.30).astype(int)
    frame.loc[valid, "top30_label"] = frame.loc[valid, "return_rank"].le(cutoff).astype(int).to_numpy()
    labeled = frame[frame["top30_label"].notna()].copy()
    if labeled.empty or labeled["top30_label"].nunique() != 2:
        raise ValueError("top-30 label has no valid binary training sample")
    return frame, {
        "mature_executable_rows": int(len(frame)),
        "labeled_rows": int(len(labeled)),
        "labeled_dates": int(labeled["event_time"].nunique()),
        "dates_below_minimum_group": int((frame.groupby("event_time").size() < minimum_group_size).sum()),
        "positive_rate": float(labeled["top30_label"].mean()),
        "minimum_group_size": minimum_group_size,
    }


def _fit_twice(train: pd.DataFrame, feature_set: FeatureSet, params: dict[str, Any], model_id: str) -> tuple[XGBClassifier, dict[str, Any]]:
    if train["top30_label"].nunique() != 2:
        raise ValueError(f"{model_id} lacks both top-30 classes")
    models, hashes = [], []
    with tempfile.TemporaryDirectory(prefix=f"aistock-{model_id}-") as temporary:
        for number in (1, 2):
            model = XGBClassifier(**params)
            model.fit(train[list(feature_set.columns)], train["top30_label"].astype(int))
            path = Path(temporary) / f"model-{number}.json"
            model.save_model(path)
            hashes.append(_sha(path))
            models.append(model)
    if hashes[0] != hashes[1]:
        raise AssertionError(f"{model_id} is not deterministic")
    return models[0], {
        "model_id": model_id,
        "training_rows": int(len(train)),
        "training_dates": int(train["event_time"].nunique()),
        "positive_rate": float(train["top30_label"].mean()),
        "training_start": str(train["event_time"].min().date()),
        "training_end": str(train["event_time"].max().date()),
        "maximum_label_available_time": pd.Timestamp(train["label_available_time"].max()).isoformat(),
        "model_sha256_run1": hashes[0], "model_sha256_run2": hashes[1], "deterministic": True,
    }


def _monthly_predictions(f0: pd.DataFrame, classified: pd.DataFrame, feature_set: FeatureSet, config: dict[str, Any], sessions: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    joined = f0.rename(columns={"available_time": "feature_available_time"}).merge(
        classified[["event_time", "ts_code", "top30_label", "label_available_time"]], on=["event_time", "ts_code"], how="left", validate="one_to_one"
    )
    timeline = config["timeline"]
    start, end = pd.Timestamp(timeline["prediction_start"], tz="UTC"), pd.Timestamp(timeline["prediction_end"], tz="UTC")
    expected = pd.period_range(start.tz_localize(None).to_period("M"), end.tz_localize(None).to_period("M"), freq="M")
    params = dict(config["model"]); params.pop("family")
    output, manifest = [], []
    for period in expected:
        month_start, month_end = pd.Timestamp(period.start_time, tz="UTC"), pd.Timestamp(period.end_time.date(), tz="UTC")
        prior = sessions[sessions < month_start]
        if prior.empty:
            raise ValueError(f"no training cutoff before {period}")
        cutoff = prior[-1]
        window_start = cutoff - pd.DateOffset(months=int(timeline["train_window_months"]))
        train = joined.loc[
            joined["event_time"].gt(window_start) & joined["event_time"].le(cutoff)
            & joined["top30_label"].notna()
            & pd.to_datetime(joined["label_available_time"], utc=True).le(session_close(cutoff))
        ].copy()
        model_id = f"{config['identity']['model_profile_id']}_{period.strftime('%Y%m')}"
        model, metadata = _fit_twice(train, feature_set, params, model_id)
        prediction = f0.loc[f0["event_time"].between(max(start, month_start), min(end, month_end))].copy()
        if pd.to_datetime(prediction["available_time"], utc=True).gt(prediction["event_time"].map(session_close)).any():
            raise AssertionError(f"{model_id} uses unavailable features")
        prediction["model_score"] = model.predict_proba(prediction[list(feature_set.columns)])[:, 1]
        prediction["model_id"] = model_id
        output.append(prediction[["event_time", "ts_code", "model_score", "model_id"]])
        metadata.update({"training_cutoff": str(cutoff.date()), "prediction_rows": int(len(prediction)), "prediction_sessions": int(prediction["event_time"].nunique())})
        manifest.append(metadata)
        del model, train, prediction
        gc.collect()
    return pd.concat(output, ignore_index=True), manifest


def _accept(control: dict[str, Any], challenger: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for arm in ("base", "stress"):
        c, x = control[arm], challenger[arm]
        prefix = arm + "_"
        checks[prefix + "paired_return"] = x["total_return"] > c["total_return"]
        checks[prefix + "full_cap1_return"] = x["total_return"] > float(config["evaluation"][f"cap1_{arm}_return"])
        checks[prefix + "win_rate"] = x["trade_win_rate"] is not None and x["trade_win_rate"] >= float(config["evaluation"]["trade_win_rate_min"])
        checks[prefix + "pf"] = (x["portfolio_profit_factor"] or 0) >= float(config["evaluation"]["portfolio_profit_factor_min"])
        checks[prefix + "maxdd"] = abs(x["max_drawdown"]) <= float(config["evaluation"]["max_drawdown_abs_max"])
        checks[prefix + "ex_best"] = x["return_excluding_best_week"] > 0
        checks[prefix + "ex_top3"] = x["return_excluding_top3_profit"] > 0
        checks[prefix + "trades"] = x["trade_count"] >= int(config["evaluation"]["minimum_closed_trades"])
        checks[prefix + "positions"] = x["max_open_positions"] <= int(config["evaluation"]["maximum_positions"])
    return {"passed": all(checks.values()), "checks": checks}


def run(output: Path) -> Path:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    config, strategy = _load_profile(), StrategyConfig.from_yaml(STRATEGY)
    feature_set = FeatureSet.from_f0_json(ROOT / "configs/feature_sets/f0_123_columns.json")
    if len(feature_set.columns) != 123 or feature_set.order_hash != config["feature_set"]["column_order_hash"]:
        raise ValueError("F0-123 feature contract drift")
    cutoffs, timeline = _source_cutoffs(), config["timeline"]
    if min(cutoffs.values()) < timeline["prediction_end"]:
        raise ValueError(f"F0 source cutoff too early: {cutoffs}")
    train_plan, _ = _period_plan(strategy, signal_start=timeline["train_start"], signal_end="2025-12-31", execution_end="2026-01-19", output=output, name="cap1_top30_training")
    train_pool, train_labels, train_audit, _ = _build_stage1_period(strategy, train_plan, output, retain_bundle=False, label_config=config["label"])
    prediction_plan, prediction_calendar = _period_plan(strategy, signal_start=timeline["prediction_start"], signal_end=timeline["prediction_end"], execution_end=timeline["execution_end"], output=output, name="cap1_top30_prediction", require_complete_horizon=False)
    prediction_pool, prediction_labels, prediction_audit, bundle = _build_stage1_period(strategy, prediction_plan, output, retain_bundle=True, label_config=config["label"])
    combined_pool, combined_labels = pd.concat([train_pool, prediction_pool], ignore_index=True), pd.concat([train_labels, prediction_labels], ignore_index=True)
    classified, label_audit = _top30_labels(combined_labels, int(config["label"]["minimum_label_group_size"]))
    f0, f0_audit = _load_candidate_f0(combined_pool, feature_set)
    prediction_f0 = f0[f0["event_time"].between(pd.Timestamp(timeline["prediction_start"], tz="UTC"), pd.Timestamp(timeline["prediction_end"], tz="UTC"))].copy()
    sessions = pd.DatetimeIndex(pd.to_datetime(prediction_calendar["session"], utc=True)).normalize()
    predictions, models = _monthly_predictions(f0, classified, feature_set, config, sessions)
    if len(predictions) != len(prediction_f0):
        raise AssertionError("not every F0-eligible prediction row was scored")
    eligible = prediction_pool.merge(prediction_f0[["event_time", "ts_code"]].rename(columns={"event_time": "asof"}), on=["asof", "ts_code"], validate="one_to_one")
    control_ledgers = _candidate_ledgers(eligible, predictions, prediction_plan.signal_sessions, rank_column="rule_score", policy_id="cap1_top30_transparent_control", strategy=strategy)
    challenger_ledgers = _candidate_ledgers(eligible, predictions, prediction_plan.signal_sessions, rank_column="model_score", policy_id="cap1_top30_classifier", strategy=strategy)
    control, control_results = _run_portfolios(control_ledgers, bundle, strategy, prediction_plan.execution_sessions)
    challenger, challenger_results = _run_portfolios(challenger_ledgers, bundle, strategy, prediction_plan.execution_sessions)
    verification = {"passed": bool(len(models) == 8 and all(row["deterministic"] and row["training_rows"] > 0 and pd.Timestamp(row["maximum_label_available_time"]) <= session_close(row["training_cutoff"]) for row in models) and all(challenger[arm]["max_open_positions"] <= 5 for arm in ("base", "stress"))), "monthly_models": len(models), "all_training_labels_mature": all(pd.Timestamp(row["maximum_label_available_time"]) <= session_close(row["training_cutoff"]) for row in models)}
    if not verification["passed"]:
        raise AssertionError(f"verification failed: {verification}")
    acceptance = _accept(control, challenger, config)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "RUN_STATUS.json", {"status": "COMPLETED_ACCEPT" if acceptance["passed"] else "COMPLETED_REJECT", "completed_at": datetime.now(timezone.utc).isoformat(), "credentials_persisted": False, "raw_business_data_persisted": False, "years_used_for_performance": [2026]})
    _write_json(output / "plan.json", {"training": train_plan.to_dict(), "prediction": prediction_plan.to_dict(), "source_cutoffs": cutoffs})
    _write_json(output / "sample_audit.json", {"train": train_audit, "prediction": prediction_audit, "labels": label_audit, "f0": f0_audit})
    _write_json(output / "model_manifest.json", models)
    _write_json(output / "control_metrics.json", control); _write_json(output / "challenger_metrics.json", challenger)
    _write_json(output / "verification.json", verification); _write_json(output / "acceptance.json", acceptance)
    _write_json(output / "code_manifest.json", {str(path.relative_to(ROOT)): _sha(path) for path in (STRATEGY, PROFILE, PREREG, ROOT / "src/aistock9988/backtest/engine.py", ROOT / "src/aistock9988/labeling/executable_path.py", Path(__file__).resolve())})
    rows = []
    for arm in ("base", "stress"):
        for name, metrics in (("Transparent control", control[arm]), ("Top-30 classifier", challenger[arm])):
            pf = "NA" if metrics["portfolio_profit_factor"] is None else f"{metrics['portfolio_profit_factor']:.3f}"
            rows.append(f"| {arm} | {name} | {metrics['total_return']:+.2%} | {metrics['trade_win_rate']:.1%} | {pf} | {metrics['max_drawdown']:.2%} | {metrics['return_excluding_best_week']:+.2%} | {metrics['weekly_ge_5_count']} | {metrics['trade_count']} |")
    (output / "RESULT.md").write_text("\n".join(["# CAP1 F0-123 Top-30% Classifier V1", "", f"Status: `{'ACCEPT' if acceptance['passed'] else 'REJECT'}`. Paired seen-2026 historical replay, not a forward claim.", "", "## Integrity", "", f"- 2026 signals `{prediction_plan.signal_start}` through `{prediction_plan.signal_end}`; execution and marks `{prediction_plan.execution_end}`.", f"- {label_audit['labeled_rows']} mature labels across {label_audit['labeled_dates']} eligible Stage-1 dates; top-30 positive rate {label_audit['positive_rate']:.1%}.", "- Eight deterministic monthly classifiers; every training label was mature at that model cutoff.", "- No raw business data, candidate, fill, position, model, CSV, or Parquet was persisted.", "", "## Portfolio", "", "| Cost | Strategy | Return | Win rate | PF | MaxDD | Ex-best | Weeks >=5% | Trades |", "|---|---|---:|---:|---:|---:|---:|---:|---:|", *rows, "", "## Decision", "", f"Promotion passed: `{acceptance['passed']}`. Failure closes this exact top-30 label without fraction, group-size, model, TopN, or holding-rule tuning.", ""]) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(run(args.output.resolve()))


if __name__ == "__main__":
    main()
