"""Configured q70 walk-forward runner for the first AiStock9988 experiment."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.data.execution_source import load_execution_panel, load_market_context_panel
from aistock9988.data.q70_source import load_f0_panel
from aistock9988.data.snapshot import build_snapshot_meta
from aistock9988.features.registry import FeatureSet
from aistock9988.labeling.maturity import LabelProfile, mature_training_rows
from aistock9988.labeling.q70 import build_q70_t10_labels
from aistock9988.models.pipeline import model_for_prediction
from aistock9988.models.trainer import train_ranker
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.selection.ledger import build_prediction_ledger, freeze_candidates, write_ledger
from aistock9988.selection.q70_policy import build_q70_selection_ledger
from aistock9988.time.session import session_close


ROOT = Path(__file__).resolve().parents[1]
LABEL_PROFILE = LabelProfile("label.endpoint_open_open_t10.v1", 1, 10, 11)


def _train_one(panel, labels, spec, run_dir, train_start, cutoff, model_id, params):
    from aistock9988.labeling.dataset import build_training_dataset

    cutoff_day = pd.Timestamp(cutoff).normalize()
    cutoff_time = session_close(cutoff_day)
    features = panel[(panel["event_time"] <= cutoff_day) &
                     (panel["available_time"] <= cutoff_time)]
    mature = mature_training_rows(labels[labels["available_time"] <= cutoff_time].copy(), training_cutoff=cutoff_time)
    X, y = build_training_dataset(features, mature, feature_set=spec, training_cutoff=cutoff_time,
                                  allow_feature_missing=True)
    keys = features[["ts_code", "event_time"]].merge(
        mature[["ts_code", "event_time"]], on=["ts_code", "event_time"], validate="one_to_one"
    ).sort_values(["event_time", "ts_code"], kind="mergesort")
    if len(keys) != len(X):
        raise ValueError(f"training key mismatch for {model_id}")
    return train_ranker(X, y, group_dates=keys["event_time"], feature_set_id=spec.id,
                        label_profile_id=LABEL_PROFILE.id, training_cutoff=str(cutoff_time),
                        model_id=model_id, output_dir=run_dir, params=params)


def run(*, run_dir: Path, config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    data = config["data"]
    if data["forbid_old_ledger"] or data["forbid_stage2"] is True:
        # The runner has no imports from either legacy artifact path by design.
        pass
    if config["execution"]["minute_data"] != "5min":
        raise ValueError("first q70 runner requires configured 5min execution data")
    run_dir = run_dir.resolve()
    panel, audit = load_f0_panel(data["train_start"], data["raw_end"], return_audit=True)
    spec = FeatureSet.from_f0_json(ROOT / "configs/feature_sets/f0_123_columns.json")
    sessions = pd.DatetimeIndex(sorted(panel["event_time"].drop_duplicates()))
    labels = build_q70_t10_labels(panel, profile=LABEL_PROFILE, session_dates=sessions)
    model_dates = [pd.Timestamp(x).date() for x in config["model"]["expected_monthly_models"]]
    prediction_dates = [d for d in sessions if d.date() >= pd.Timestamp(data["oos_start"]).date()
                        and d.date() <= pd.Timestamp(data["raw_end"]).date() and d.weekday() == 4]
    context = load_market_context_panel([d.strftime("%Y-%m-%d") for d in prediction_dates])
    all_selected = []
    all_candidates = []
    params = {k: config["model"][k] for k in ("objective", "n_estimators", "max_depth", "learning_rate",
                                                "min_child_weight", "subsample", "colsample_bytree", "reg_alpha",
                                                "reg_lambda", "seed")}
    params["random_state"] = params.pop("seed")
    for model_date in model_dates:
        month_start = pd.Timestamp(model_date).replace(day=1)
        prior = sessions[sessions < month_start.tz_localize("UTC")]
        if len(prior) == 0:
            continue
        cutoff = prior[-1]
        model_id = f"q70_{pd.Timestamp(model_date).strftime('%Y%m%d')}_cutoff_{cutoff.strftime('%Y%m%d')}"
        artifact = _train_one(panel, labels, spec, run_dir, data["train_start"], cutoff, model_id, params)
        month_predictions = [d for d in prediction_dates if d.year == model_date.year and d.month == model_date.month]
        for prediction_date in month_predictions:
            source = panel[panel["event_time"] == prediction_date].copy()
            source = source[source["available_time"] <= session_close(prediction_date)]
            if source.empty:
                continue
            scores = model_for_prediction(run_dir / "models" / f"{model_id}.json", source[list(spec.columns)])
            predictions = build_prediction_ledger(pd.DataFrame({"ts_code": source["ts_code"], "score": scores}),
                                                   asof=str(prediction_date.date()), feature_set_id=spec.id, model_id=model_id)
            gate_columns = [c for c in ("xsii_td3_bfq_sector_rel", "expma_12_bfq_sector_rel", "boll_mid_bfq_sector_rel") if c in source]
            top20 = freeze_candidates(predictions, top_n=20).merge(
                source[["ts_code", *gate_columns]], on="ts_code", how="left"
            )
            selected = build_q70_selection_ledger(top20, context, asof=str(prediction_date.date()),
                                                   max_positions=config["selection"]["max_positions"],
                                                   breadth_min=config["selection"]["breadth_min"],
                                                   factor_floor=config["selection"]["sector_relative_floor"])
            write_ledger(predictions, run_dir / "predictions" / f"{prediction_date.date()}_prediction.csv")
            write_ledger(selected, run_dir / "selections" / f"{prediction_date.date()}_selection.csv")
            all_selected.append(selected[selected["selected"]].assign(asof=str(prediction_date.date())))
            all_candidates.append(selected)
    if not all_selected:
        raise RuntimeError("runner produced no selected signals")
    signals = pd.concat(all_selected, ignore_index=True)
    codes = sorted(signals["ts_code"].astype(str).unique())
    prices = load_execution_panel(data["oos_start"], data["raw_end"], ts_codes=codes)
    actions = load_corporate_actions(data["oos_start"], data["raw_end"], ts_codes=codes)
    from aistock9988.data.minute_source import load_minute_execution_panel
    minutes = load_minute_execution_panel(data["oos_start"], data["raw_end"], freq="5min", ts_codes=codes)
    result = run_backtest(signals, prices, corporate_actions=actions, minute_prices=minutes,
                          config=BacktestConfig(max_positions=config["selection"]["max_positions"],
                                                hold_sessions=config["selection"]["hold_buffer_sessions"],
                                                stop_loss_pct=config["execution"]["stop_loss_pct"],
                                                take_profit_pct=config["execution"]["take_profit_pct"],
                                                stop_loss_mode="intraday_5min"))
    for key, filename in (("orders", "orders.csv"), ("trades", "fills.csv"), ("nav", "nav.csv"),
                          ("positions", "positions.csv"), ("corporate_actions", "corporate_actions.csv")):
        result[key].to_csv(run_dir / "trades" / filename, index=False)
    (run_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    metrics = summarize_backtest(result["nav"], result["trades"], initial_cash=1_000_000.0)
    (run_dir / "diagnostics" / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "models": len(model_dates), "prediction_dates": len(prediction_dates),
                    "selected_rows": len(signals), "execution": "raw accounting + economic trigger + 5min"},
                   ensure_ascii=False, indent=2, default=str) + "\n"
    )
    manifest = {"snapshot": asdict(build_snapshot_meta(panel, source_id="quant_db.q70_f0", query=data)),
                "industry": audit, "config": str(config_path.relative_to(ROOT)),
                "contract": "raw accounting, economic risk triggers, 5min intraday stop"}
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n")
    return {"models": len(model_dates), "prediction_dates": len(prediction_dates), "selected_rows": len(signals)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(run_dir=args.run_dir, config_path=args.config.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
