"""Standalone runner for the explicitly non-production delta comparison."""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from aistock9988.data.q70_source import load_f0_panel
from aistock9988.features.registry import FeatureSet
from aistock9988.labeling.maturity import LabelProfile, mature_training_rows
from aistock9988.labeling.q70 import build_q70_endpoint_labels
from aistock9988.models.pipeline import model_for_prediction
from aistock9988.models.trainer import train_ranker
from aistock9988.selection.delta_compatible import (compute_dynamic_upper_gate, apply_dynamic_upper_gate,
                                                     apply_market_cap_filter, select_rank_holdings,
                                                     weak_breadth_cash_fraction)
from aistock9988.selection.ledger import build_prediction_ledger, freeze_candidates, write_ledger
from aistock9988.selection.q70_policy import build_q70_selection_ledger
from aistock9988.data.execution_source import load_market_context_panel
from aistock9988.data.execution_source import load_execution_panel
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.audit.code_manifest import build_code_manifest
from aistock9988.data.snapshot import build_snapshot_meta
from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.time.session import session_close


ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("aistock9988.q70_delta_runner")


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
    if frame.empty:
        LOGGER.info("data=%s rows=0 cols=%d", name, len(frame.columns))
        return
    ranges = []
    for column in ("event_time", "trade_date", "ex_date"):
        if column in frame:
            values = pd.to_datetime(frame[column], errors="coerce")
            ranges.append(f"{column}={values.min()}..{values.max()}")
            break
    LOGGER.info("data=%s rows=%d cols=%d %s", name, len(frame), len(frame.columns), " ".join(ranges))


def _weekly(sessions, start, end):
    out = {}
    for value in sorted(pd.Timestamp(x) for x in sessions):
        if pd.Timestamp(start).date() <= value.date() <= pd.Timestamp(end).date():
            iso = value.date().isocalendar()
            out[(iso.year, iso.week)] = value
    return sorted(out.values())


def _mature(sessions, days, lag, end):
    sessions = list(sorted(pd.Timestamp(x) for x in sessions))
    positions = {day: i for i, day in enumerate(sessions)}
    terminal = pd.Timestamp(end).date()
    return [day for day in days if day in positions and positions[day] + lag < len(sessions)
            and sessions[positions[day] + lag].date() <= terminal]


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


def _write_json_once(path: Path, payload: object) -> None:
    _write_bytes_once(path, (json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n").encode())


def _write_frame_once(path: Path, frame: pd.DataFrame) -> None:
    _write_bytes_once(path, frame.to_csv(index=False, lineterminator="\n").encode())


def _train(panel, labels, spec, run_dir, cutoff, profile, config):
    from aistock9988.labeling.dataset import build_training_dataset
    cutoff = pd.Timestamp(cutoff)
    cutoff_time = session_close(cutoff)
    start = cutoff - pd.DateOffset(months=int(config["model"]["train_window_months"]))
    features = panel[(panel.event_time > start) & (panel.event_time <= cutoff) &
                     (panel.available_time <= cutoff_time)]
    mature = labels[(labels.event_time > start) & (labels.event_time <= cutoff) &
                    (labels.available_time <= cutoff_time)]
    mature = mature_training_rows(mature, training_cutoff=cutoff_time)
    LOGGER.info("phase=training_dataset cutoff=%s window_start=%s feature_rows=%d mature_rows=%d",
                cutoff.date(), start.date(), len(features), len(mature))
    X, y = build_training_dataset(features, mature, feature_set=spec,
                                   training_cutoff=cutoff_time, allow_feature_missing=True)
    keys = features[["ts_code", "event_time"]].merge(mature[["ts_code", "event_time"]],
                                                        on=["ts_code", "event_time"], validate="one_to_one")
    keys = keys.sort_values(["event_time", "ts_code"], kind="mergesort")
    if len(keys) != len(X):
        raise ValueError("delta-compatible training key mismatch")
    params = {k: config["model"][k] for k in ("objective", "n_estimators", "max_depth", "learning_rate",
                                               "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda")}
    params.update(random_state=config["model"]["seed"], n_jobs=1)
    gate_cfg = config["selection"]["dynamic_upper_gate"]
    gate_input = features[["ts_code", "event_time", gate_cfg["factor"]]].merge(
        mature[["ts_code", "event_time", "label_return"]], on=["ts_code", "event_time"], validate="one_to_one")
    gate = compute_dynamic_upper_gate(gate_input, factor=gate_cfg["factor"],
                                      minimum_samples=gate_cfg["minimum_mature_samples"],
                                      lower_quantile=gate_cfg["lower_tail_quantile"],
                                      upper_quantile=gate_cfg["upper_tail_quantile"])
    LOGGER.info("phase=dynamic_gate model_cutoff=%s factor=%s samples=%d active=%s threshold=%s",
                cutoff.date(), gate.factor, gate.sample_count, gate.active, gate.threshold)
    LOGGER.info("phase=model_fit model_id=q70_delta_%s rows=%d groups=%d features=%d",
                cutoff.strftime("%Y%m%d"), len(X), len(keys.event_time.unique()), len(spec.columns))
    started = time.monotonic()
    artifact = train_ranker(X, y, group_dates=keys.event_time, feature_set_id=spec.id,
                        label_profile_id=profile.id, training_cutoff=str(cutoff_time),
                        model_id=f"q70_delta_{cutoff:%Y%m%d}", output_dir=run_dir / "models", params=params,
                        metadata_extra={"dynamic_gate": asdict(gate)})
    LOGGER.info("phase=model_fit_complete model_id=%s seconds=%.2f model_sha256=%s",
                artifact.model_id, time.monotonic() - started, artifact.model_sha256)
    return artifact, features, mature, gate


def run(*, run_dir: Path, config_path: Path) -> dict:
    run_dir = run_dir.resolve()
    _configure_logging(run_dir)
    started = time.monotonic()
    LOGGER.info("phase=runner_start run_dir=%s config=%s", run_dir, config_path.resolve())
    config = yaml.safe_load(config_path.read_text())
    from scripts.validate_q70_delta_compatible_config import validate
    validate(config_path)
    LOGGER.info("phase=config_validated experiment_id=%s feature_set=%s", config["id"], config["data"]["feature_set"])
    if config.get("reference_only") is not True:
        raise ValueError("delta-compatible runner requires reference_only=true")
    status_path = run_dir / "RUN_STATUS.json"
    if not status_path.is_file():
        raise ValueError("runner requires an initialized run directory with RUN_STATUS.json")
    for directory in ("data", "models", "predictions", "selections", "trades", "diagnostics", "logs"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    _write_bytes_once(run_dir / "data" / "experiment_config.yaml", config_path.read_bytes())
    data, label_cfg = config["data"], config["label"]
    execution, selection = config["execution"], config["selection"]
    spec = FeatureSet.from_f0_json(ROOT / "configs/feature_sets/f0_123_columns.json")
    LOGGER.info("phase=feature_contract feature_count=%d order_hash=%s factors=%s",
                len(spec.columns), spec.order_hash, ",".join(spec.columns))
    profile = LabelProfile(label_cfg["profile"], label_cfg["signal_to_entry_sessions"],
                           label_cfg["entry_to_exit_sessions"], label_cfg["maturity_lag_sessions"])
    LOGGER.info("phase=data_load_start source=q70_f0 start=%s end=%s", data["train_start"], data["raw_end"])
    panel = load_f0_panel(data["train_start"], data["raw_end"])
    _log_frame("f0_panel_raw", panel)
    market_cap = data["market_cap_filter"]
    market_cap_minimum = market_cap["min_value"] if market_cap["enabled"] else None
    panel = apply_market_cap_filter(panel, field=market_cap["field"], minimum=market_cap_minimum)
    LOGGER.info("phase=market_cap_filter enabled=%s field=%s minimum=%s rows_after=%d",
                market_cap["enabled"], market_cap["field"], market_cap_minimum, len(panel))
    sessions = pd.DatetimeIndex(sorted(panel.event_time.drop_duplicates()))
    labels = build_q70_endpoint_labels(panel, profile=profile, session_dates=sessions)
    _log_frame("labels", labels)
    signal_dates = _mature(sessions, _weekly(sessions, data["oos_start"], data["raw_end"]),
                           label_cfg["maturity_lag_sessions"], data["mature_end"])
    model_dates = [pd.Timestamp(x) for x in config["model"]["expected_monthly_models"]
                   if pd.Timestamp(x) <= pd.Timestamp(data["mature_end"])]
    LOGGER.info("phase=timeline sessions=%d signal_dates=%d model_dates=%s mature_end=%s",
                len(sessions), len(signal_dates), ",".join(str(x.date()) for x in model_dates), data["mature_end"])
    LOGGER.info("phase=data_load_start source=market_context start=%s end=%s", data["oos_start"], data["mature_end"])
    context = load_market_context_panel(data["oos_start"], data["mature_end"])
    _log_frame("market_context", context)
    previous_codes: set[str] = set()
    selected_by_date = []
    gate_audit = []
    for index, model_date in enumerate(model_dates):
        prior = sessions[sessions <= model_date]
        if prior.empty:
            continue
        artifact, features, mature, gate = _train(panel, labels, spec, run_dir, prior[-1], profile, config)
        gate_audit.append({"model_id": artifact.model_id, "training_cutoff": str(prior[-1].date()), **asdict(gate)})
        next_date = model_dates[index + 1] if index + 1 < len(model_dates) else pd.Timestamp(data["mature_end"]) + pd.Timedelta(days=1)
        model_signal_dates = [x for x in signal_dates if model_date.date() <= x.date() < next_date.date()]
        LOGGER.info("phase=prediction_start model_id=%s model_date=%s signal_dates=%d", artifact.model_id, model_date.date(), len(model_signal_dates))
        for signal_index, asof in enumerate(model_signal_dates, 1):
            source = panel[panel.event_time == asof].copy()
            if source.empty:
                LOGGER.warning("phase=prediction_skip date=%s reason=empty_feature_panel", asof.date())
                continue
            scores = model_for_prediction(run_dir / "models" / f"{artifact.model_id}.json", source[list(spec.columns)])
            pred = build_prediction_ledger(pd.DataFrame({"ts_code": source.ts_code, "score": scores}),
                                           asof=str(asof.date()), feature_set_id=spec.id, model_id=artifact.model_id)
            top20 = freeze_candidates(pred, top_n=selection["candidate_pool"]).merge(source[["ts_code", "dmi_adx_bfq", "xsii_td3_bfq_sector_rel",
                                                                       "expma_12_bfq_sector_rel", "boll_mid_bfq_sector_rel"]], on="ts_code", how="left")
            top20 = apply_dynamic_upper_gate(top20, factor=gate.factor, threshold=gate.threshold)
            gate_pass_count = int(top20.dynamic_gate_passed.sum())
            top20 = top20[top20.dynamic_gate_passed].copy()
            chosen = build_q70_selection_ledger(top20, context, asof=str(asof.date()),
                                                max_positions=selection["max_positions"],
                                                breadth_min=selection["market_breadth_min"], factor_floor=selection["sector_relative_floor"],
                                                weak_breadth_positions=selection["low_breadth_top_n"], volatility_window_sessions=selection["volatility_window_sessions"],
                                                volatility_max=selection["volatility_max"], recent_limit_down_window_sessions=selection["recent_limit_down_window_sessions"],
                                                recent_limit_down_threshold=selection["recent_limit_down_threshold"], peak_drawdown_window_sessions=selection["peak_drawdown_window_sessions"],
                                                peak_drawdown_threshold=selection["peak_drawdown_threshold"], exclude_beijing=selection["exclude_beijing"],
                                                alpha_weight=selection["alpha_weight"], alpha_power=selection["alpha_power"])
            eligible = chosen[chosen.rejection_reason == ""].copy()
            held = select_rank_holdings(eligible, previous_codes, max_positions=selection["max_positions"], hold_buffer_n=selection["hold_buffer_n"])
            selected = chosen.assign(selected=False, target_weight=0.0)
            selected.loc[selected.ts_code.astype(str).isin(held.ts_code.astype(str)), "selected"] = True
            selected.loc[selected.selected, "target_weight"] = 1.0 / max(1, int(selected.selected.sum()))
            breadth = float(selected.context_breadth_ratio.iloc[0]) if not selected.empty else 0.0
            fraction = weak_breadth_cash_fraction(breadth=breadth, minimum=selection["market_breadth_min"],
                                                   candidate_count=len(held), configured_fraction=selection["weak_breadth_single_candidate_cash_fraction"])
            selected["cash_fraction"] = fraction
            previous_codes = set(held.ts_code.astype(str))
            write_ledger(pred, run_dir / "predictions" / f"{asof.date()}_prediction.csv")
            write_ledger(selected, run_dir / "selections" / f"{asof.date()}_selection.csv")
            selected_by_date.append(selected)
            LOGGER.info("phase=selection date=%s progress=%d/%d source=%d top_pool=%d gate_pass=%d eligible=%d held=%d breadth=%.4f cash_fraction=%.2f",
                        asof.date(), signal_index, len(model_signal_dates), len(source), len(top20),
                        gate_pass_count,
                        len(eligible), len(held), breadth, fraction)
    if not selected_by_date:
        raise RuntimeError("delta-compatible runner produced no selections")
    signals = pd.concat(selected_by_date, ignore_index=True)
    codes = sorted(signals.loc[signals.selected, "ts_code"].astype(str).unique())
    LOGGER.info("phase=execution_data_load_start start=%s end=%s selected_signal_rows=%d codes=%d",
                data["oos_start"], data["mature_end"], len(signals), len(codes))
    prices = load_execution_panel(data["oos_start"], data["mature_end"], ts_codes=codes)
    actions = load_corporate_actions(data["oos_start"], data["mature_end"], ts_codes=codes)
    _log_frame("execution_daily", prices)
    _log_frame("corporate_actions", actions)
    _write_frame_once(run_dir / "data" / "f0_panel.csv", panel)
    _write_frame_once(run_dir / "data" / "labels.csv", labels)
    _write_frame_once(run_dir / "data" / "market_context.csv", context)
    _write_frame_once(run_dir / "data" / "execution_daily.csv", prices)
    _write_frame_once(run_dir / "data" / "corporate_actions.csv", actions)
    LOGGER.info("phase=backtest_start accounting_basis=%s corporate_actions_mode=%s stop_mode=%s hold_sessions=%d",
                execution["accounting_price_basis"], execution["corporate_actions_mode"], execution["stop_loss_mode"], label_cfg["entry_to_exit_sessions"])
    def _backtest_progress(current: int, total: int, day: pd.Timestamp) -> None:
        if current == 1 or current == total or current % max(1, total // 10) == 0:
            LOGGER.info("phase=backtest_progress sessions=%d/%d date=%s", current, total, day.date())

    result = run_backtest(signals, prices, corporate_actions=actions,
                          config=BacktestConfig(max_positions=selection["max_positions"],
                                                hold_sessions=label_cfg["entry_to_exit_sessions"],
                                                stop_loss_pct=execution["stop_loss_pct"],
                                                take_profit_pct=execution["take_profit_pct"],
                                                stop_loss_mode=execution["stop_loss_mode"],
                                                accounting_price_basis=execution["accounting_price_basis"],
                                                corporate_actions_mode=execution["corporate_actions_mode"],
                                                progress_callback=_backtest_progress))
    for key, filename in (("orders", "orders.csv"), ("trades", "fills.csv"), ("nav", "nav.csv"),
                          ("positions", "positions.csv"), ("corporate_actions", "corporate_actions.csv")):
        _write_frame_once(run_dir / "trades" / filename, result[key])
    _write_json_once(run_dir / "diagnostics" / "dynamic_gate_audit.json", gate_audit)
    _write_json_once(run_dir / "diagnostics" / "metrics.json", {
        "metrics": summarize_backtest(result["nav"], result["trades"], initial_cash=1_000_000.0),
        "contract": "delta-compatible economic accounting; economic prices include corporate actions; raw limit-state checks; no minute execution",
        "corporate_actions_applied": False,
    })
    manifest = {
        "snapshots": {
            "f0": asdict(build_snapshot_meta(panel, source_id="quant_db.q70_f0", query=data)),
            "labels": asdict(build_snapshot_meta(labels, source_id="derived.q70_endpoint_labels", query=label_cfg)),
            "market_context": asdict(build_snapshot_meta(context, source_id="quant_db.market_context", query={"start": data["oos_start"], "end": data["mature_end"]}, event_column="trade_date")),
            "execution_daily": asdict(build_snapshot_meta(prices, source_id="quant_db.execution_daily", query={"start": data["oos_start"], "end": data["mature_end"]}, event_column="trade_date")),
            "corporate_actions": asdict(build_snapshot_meta(actions, source_id="quant_db.corporate_actions", query={"start": data["oos_start"], "end": data["mature_end"]}, event_column="ex_date")),
        },
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "contract": "economic execution/NAV prices already include company actions; company-action mutation skipped; raw limit-state checks",
        "corporate_actions_applied": False,
    }
    _write_json_once(run_dir / "data_manifest.json", manifest)
    _write_json_once(run_dir / "code_manifest.json", build_code_manifest(
        repo_root=ROOT, config_path=config_path.resolve(), entrypoint=Path(__file__).resolve()))
    LOGGER.info("phase=runner_complete models=%d prediction_dates=%d trades=%d nav_rows=%d seconds=%.2f",
                len(model_dates), len(selected_by_date), len(result["trades"]), len(result["nav"]), time.monotonic() - started)
    return {"models": len(model_dates), "prediction_dates": len(selected_by_date), "status": "executed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(run_dir=args.run_dir, config_path=args.config.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
