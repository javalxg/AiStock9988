"""Configured q70 walk-forward runner for the first AiStock9988 experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
LOGGER = logging.getLogger("aistock9988.q70_runner")


def _configure_logging(run_dir: Path) -> None:
    log_path = run_dir / "logs" / "runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S%z")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def _log_frame(name: str, frame: pd.DataFrame) -> None:
    LOGGER.info("%s rows=%d cols=%d", name, len(frame), len(frame.columns))


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    _write_bytes_once(path, (json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n").encode())


def _write_frame_once(path: Path, frame: pd.DataFrame) -> None:
    _write_bytes_once(path, frame.to_csv(index=False, lineterminator="\n").encode())


def _write_bytes_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _weekly_signal_dates(sessions: pd.DatetimeIndex, *, start: object, end: object) -> list[pd.Timestamp]:
    start_day = pd.Timestamp(start).date()
    end_day = pd.Timestamp(end).date()
    by_week: dict[tuple[int, int], pd.Timestamp] = {}
    for session in sorted(pd.Timestamp(value) for value in sessions):
        if start_day <= session.date() <= end_day:
            iso = session.date().isocalendar()
            by_week[(iso.year, iso.week)] = session
    return sorted(by_week.values())


def _mature_signal_dates(sessions: pd.DatetimeIndex, candidates: list[pd.Timestamp], *,
                         entry_delay_sessions: int, horizon_sessions: int,
                         mature_end: object) -> list[pd.Timestamp]:
    ordered = list(sorted(pd.Timestamp(value) for value in sessions))
    positions = {day: index for index, day in enumerate(ordered)}
    terminal = pd.Timestamp(mature_end).date()
    lag = int(entry_delay_sessions) + int(horizon_sessions)
    return [day for day in candidates
            if day in positions and positions[day] + lag < len(ordered)
            and ordered[positions[day] + lag].date() <= terminal]


def _train_one(panel, labels, spec, run_dir, train_window_months, cutoff, model_id, params):
    from aistock9988.labeling.dataset import build_training_dataset

    cutoff_day = pd.Timestamp(cutoff).normalize()
    cutoff_time = session_close(cutoff_day)
    window_start = cutoff_day - pd.DateOffset(months=int(train_window_months))
    features = panel[(panel["event_time"] > window_start) & (panel["event_time"] <= cutoff_day) &
                     (panel["available_time"] <= cutoff_time)]
    mature_source = labels[(labels["event_time"] > window_start) &
                           (labels["event_time"] <= cutoff_day) &
                           (labels["available_time"] <= cutoff_time)].copy()
    mature = mature_training_rows(mature_source, training_cutoff=cutoff_time)
    X, y = build_training_dataset(features, mature, feature_set=spec, training_cutoff=cutoff_time,
                                  allow_feature_missing=True)
    keys = features[["ts_code", "event_time"]].merge(
        mature[["ts_code", "event_time"]], on=["ts_code", "event_time"], validate="one_to_one"
    ).sort_values(["event_time", "ts_code"], kind="mergesort")
    if len(keys) != len(X):
        raise ValueError(f"training key mismatch for {model_id}")
    return train_ranker(X, y, group_dates=keys["event_time"], feature_set_id=spec.id,
                        label_profile_id=LABEL_PROFILE.id, training_cutoff=str(cutoff_time),
                        model_id=model_id, output_dir=run_dir / "models", params=params)


def run(*, run_dir: Path, config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    data = config["data"]
    if data.get("forbid_old_ledger") is not True:
        raise ValueError("formal q70 runs must forbid legacy ledgers")
    if data.get("forbid_stage2") is not True:
        raise ValueError("formal q70 runs must keep Stage2 disabled")
    if data.get("allow_minute_execution_data") is not True or config["execution"]["minute_data"] != "5min":
        raise ValueError("first q70 runner requires configured 5min execution data")
    if config["execution"].get("stop_loss_mode") != "intraday_5min":
        raise ValueError("5min execution must select intraday_5min explicitly")
    run_dir = run_dir.resolve()
    if not (run_dir / "RUN_STATUS.json").is_file():
        raise RuntimeError("run directory must be created by the project CLI")
    _configure_logging(run_dir)
    started = time.monotonic()
    LOGGER.info("run_start run_dir=%s config=%s", run_dir, config_path)
    spec = FeatureSet.from_f0_json(ROOT / "configs/feature_sets/f0_123_columns.json")
    if data["feature_set"] != spec.id or len(spec.columns) != 123:
        raise ValueError("formal runner requires the frozen feature.f0_123.v1 contract")
    if (config["model"]["train_window_months"] != 12 or
            config["label"]["signal_to_entry_sessions"] != LABEL_PROFILE.entry_delay_sessions or
            config["label"]["entry_to_exit_sessions"] != LABEL_PROFILE.horizon_sessions):
        raise ValueError("formal runner requires a 12-month window and T+1/T+10 labels")
    _write_bytes_once(run_dir / "data" / "experiment_config.yaml", config_path.read_bytes())
    LOGGER.info("phase=f0_load start=%s end=%s", data["train_start"], data["raw_end"])
    panel, audit = load_f0_panel(data["train_start"], data["raw_end"], return_audit=True)
    _log_frame("f0_panel", panel)
    LOGGER.info("phase=f0_load elapsed_seconds=%.1f", time.monotonic() - started)
    sessions = pd.DatetimeIndex(sorted(panel["event_time"].drop_duplicates()))
    LOGGER.info("phase=labels start sessions=%d", len(sessions))
    labels = build_q70_t10_labels(panel, profile=LABEL_PROFILE, session_dates=sessions)
    _log_frame("labels", labels)
    formal_end = pd.Timestamp(data["mature_end"])
    model_dates = [pd.Timestamp(x) for x in config["model"]["expected_monthly_models"]
                   if pd.Timestamp(x) <= formal_end]
    weekly_dates = _weekly_signal_dates(sessions, start=data["oos_start"], end=formal_end)
    prediction_dates = _mature_signal_dates(
        sessions, weekly_dates,
        entry_delay_sessions=config["label"]["signal_to_entry_sessions"],
        horizon_sessions=config["label"]["entry_to_exit_sessions"], mature_end=formal_end,
    )
    context_start = pd.Timestamp(data["oos_start"]) - pd.Timedelta(days=45)
    LOGGER.info("phase=market_context_load start=%s end=%s", context_start.date(), formal_end.date())
    context = load_market_context_panel(context_start.strftime("%Y-%m-%d"), formal_end.strftime("%Y-%m-%d"))
    _log_frame("market_context", context)
    _write_frame_once(run_dir / "data" / "f0_panel.csv", panel)
    _write_frame_once(run_dir / "data" / "labels.csv", labels)
    _write_frame_once(run_dir / "data" / "market_context.csv", context)
    all_selected = []
    trained_models = 0
    params = {k: config["model"][k] for k in ("objective", "n_estimators", "max_depth", "learning_rate",
                                                "min_child_weight", "subsample", "colsample_bytree", "reg_alpha",
                                                "reg_lambda", "seed")}
    params["random_state"] = params.pop("seed")
    for model_index, model_date in enumerate(model_dates):
        model_session = model_date.tz_localize("UTC") if model_date.tzinfo is None else model_date.tz_convert("UTC")
        prior = sessions[sessions <= model_session]
        if len(prior) == 0:
            continue
        cutoff = prior[-1]
        model_id = f"q70_{model_date.strftime('%Y%m%d')}_cutoff_{cutoff.strftime('%Y%m%d')}"
        model_started = time.monotonic()
        LOGGER.info("phase=model_train model=%s cutoff=%s window_months=%s", model_id, cutoff.date(), config["model"]["train_window_months"])
        _train_one(panel, labels, spec, run_dir, config["model"]["train_window_months"], cutoff, model_id, params)
        trained_models += 1
        next_model_date = model_dates[model_index + 1] if model_index + 1 < len(model_dates) else formal_end + pd.Timedelta(days=1)
        month_predictions = [d for d in prediction_dates if model_date.date() <= d.date() < next_model_date.date()]
        LOGGER.info("phase=model_train_complete model=%s prediction_dates=%d elapsed_seconds=%.1f", model_id, len(month_predictions), time.monotonic() - model_started)
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
                                                   factor_floor=config["selection"]["sector_relative_floor"],
                                                   weak_breadth_positions=config["selection"]["weak_breadth_positions"],
                                                   volatility_window_sessions=config["selection"]["volatility_window_sessions"],
                                                   volatility_max=config["selection"]["volatility_max"],
                                                   recent_limit_down_window_sessions=config["selection"]["recent_limit_down_window_sessions"],
                                                   recent_limit_down_threshold=config["selection"]["recent_limit_down_threshold"],
                                                   peak_drawdown_window_sessions=config["selection"]["peak_drawdown_window_sessions"],
                                                   peak_drawdown_threshold=config["selection"]["peak_drawdown_threshold"],
                                                   exclude_beijing=config["selection"]["exclude_beijing"],
                                                   alpha_weight=config["selection"]["alpha_weight"],
                                                   alpha_power=config["selection"]["alpha_power"])
            LOGGER.info("phase=selection date=%s candidates=%d selected=%d", prediction_date.date(), len(selected), int(selected["selected"].sum()))
            write_ledger(predictions, run_dir / "predictions" / f"{prediction_date.date()}_prediction.csv")
            write_ledger(selected, run_dir / "selections" / f"{prediction_date.date()}_selection.csv")
            all_selected.append(selected[selected["selected"]].assign(asof=str(prediction_date.date())))
    if not all_selected:
        raise RuntimeError("runner produced no selected signals")
    signals = pd.concat(all_selected, ignore_index=True)
    if signals.empty:
        raise RuntimeError("SelectionPolicy rejected every candidate")
    codes = sorted(signals["ts_code"].astype(str).unique())
    LOGGER.info("phase=execution_data_load start=%s end=%s codes=%d", data["oos_start"], formal_end.date(), len(codes))
    prices = load_execution_panel(data["oos_start"], str(formal_end.date()), ts_codes=codes)
    actions = load_corporate_actions(data["oos_start"], str(formal_end.date()), ts_codes=codes)
    from aistock9988.data.minute_source import load_minute_execution_panel
    minutes = load_minute_execution_panel(data["oos_start"], str(formal_end.date()), freq="5min", ts_codes=codes)
    _log_frame("execution_daily", prices)
    _log_frame("corporate_actions", actions)
    _log_frame("execution_5min", minutes)
    LOGGER.info("phase=backtest start hold_sessions=%d stop_loss_mode=%s", config["label"]["entry_to_exit_sessions"], config["execution"]["stop_loss_mode"])
    _write_frame_once(run_dir / "data" / "execution_daily.csv", prices)
    _write_frame_once(run_dir / "data" / "corporate_actions.csv", actions)
    _write_frame_once(run_dir / "data" / "execution_5min.csv", minutes)
    result = run_backtest(signals, prices, corporate_actions=actions, minute_prices=minutes,
                          config=BacktestConfig(max_positions=config["selection"]["max_positions"],
                                                hold_sessions=config["label"]["entry_to_exit_sessions"],
                                                stop_loss_pct=config["execution"]["stop_loss_pct"],
                                                take_profit_pct=config["execution"]["take_profit_pct"],
                                                stop_loss_mode=config["execution"]["stop_loss_mode"]))
    LOGGER.info("phase=backtest_complete trades=%d orders=%d nav_rows=%d", len(result["trades"]), len(result["orders"]), len(result["nav"]))
    for key, filename in (("orders", "orders.csv"), ("trades", "fills.csv"), ("nav", "nav.csv"),
                          ("positions", "positions.csv"), ("corporate_actions", "corporate_actions.csv")):
        _write_frame_once(run_dir / "trades" / filename, result[key])
    metrics = summarize_backtest(result["nav"], result["trades"], initial_cash=1_000_000.0)
    _write_json_once(run_dir / "diagnostics" / "metrics.json",
                     {"metrics": metrics, "models": trained_models, "prediction_dates": len(prediction_dates),
                      "selected_rows": len(signals), "formal_end": str(formal_end.date()),
                      "execution": "raw accounting + economic trigger + 5min"})
    manifest = {"snapshots": {
                    "f0": asdict(build_snapshot_meta(panel, source_id="quant_db.q70_f0", query=data)),
                    "labels": asdict(build_snapshot_meta(labels, source_id="derived.q70_t10", query=config["label"])),
                    "market_context": asdict(build_snapshot_meta(
                        context, source_id="quant_db.market_context", query={"start": str(context_start.date()), "end": str(formal_end.date())},
                        event_column="trade_date")),
                    "execution_daily": asdict(build_snapshot_meta(
                        prices, source_id="quant_db.execution_daily", query={"start": data["oos_start"], "end": str(formal_end.date())},
                        event_column="trade_date")),
                    "corporate_actions": asdict(build_snapshot_meta(
                        actions, source_id="quant_db.corporate_actions", query={"start": data["oos_start"], "end": str(formal_end.date())},
                        event_column="ex_date")),
                    "execution_5min": asdict(build_snapshot_meta(
                        minutes, source_id="quant_db.execution_5min", query={"start": data["oos_start"], "end": str(formal_end.date())},
                        event_column="trade_time")),
                },
                "industry": audit, "config": str(config_path.relative_to(ROOT)),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "contract": "raw accounting, economic risk triggers, 5min intraday stop"}
    _write_json_once(run_dir / "data_manifest.json", manifest)
    LOGGER.info("run_complete models=%d prediction_dates=%d selected_rows=%d elapsed_seconds=%.1f", trained_models, len(prediction_dates), len(signals), time.monotonic() - started)
    return {"models": trained_models, "prediction_dates": len(prediction_dates), "selected_rows": len(signals)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(run_dir=args.run_dir, config_path=args.config.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
