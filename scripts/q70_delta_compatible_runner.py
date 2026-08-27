"""Standalone runner for the explicitly non-production delta comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from aistock9988.data.q70_source import load_f0_panel
from aistock9988.features.registry import FeatureSet
from aistock9988.labeling.maturity import LabelProfile, mature_training_rows
from aistock9988.labeling.q70 import build_q70_endpoint_labels
from aistock9988.models.pipeline import model_for_prediction
from aistock9988.models.trainer import train_ranker
from aistock9988.selection.delta_compatible import compute_dynamic_upper_gate, apply_dynamic_upper_gate, select_rank_holdings, weak_breadth_cash_fraction
from aistock9988.selection.ledger import build_prediction_ledger, freeze_candidates, write_ledger
from aistock9988.selection.q70_policy import build_q70_selection_ledger
from aistock9988.data.execution_source import load_market_context_panel
from aistock9988.data.execution_source import load_execution_panel
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.time.session import session_close


ROOT = Path(__file__).resolve().parents[1]


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
    return train_ranker(X, y, group_dates=keys.event_time, feature_set_id=spec.id,
                        label_profile_id=profile.id, training_cutoff=str(cutoff_time),
                        model_id=f"q70_delta_{cutoff:%Y%m%d}", output_dir=run_dir / "models", params=params), features, mature


def run(*, run_dir: Path, config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    if config.get("reference_only") is not True:
        raise ValueError("delta-compatible runner requires reference_only=true")
    data, label_cfg = config["data"], config["label"]
    execution, selection = config["execution"], config["selection"]
    spec = FeatureSet.from_f0_json(ROOT / "configs/feature_sets/f0_123_columns.json")
    profile = LabelProfile(label_cfg["profile"], label_cfg["signal_to_entry_sessions"],
                           label_cfg["entry_to_exit_sessions"], label_cfg["maturity_lag_sessions"])
    panel = load_f0_panel(data["train_start"], data["raw_end"])
    sessions = pd.DatetimeIndex(sorted(panel.event_time.drop_duplicates()))
    labels = build_q70_endpoint_labels(panel, profile=profile, session_dates=sessions)
    signal_dates = _mature(sessions, _weekly(sessions, data["oos_start"], data["raw_end"]),
                           label_cfg["maturity_lag_sessions"], data["mature_end"])
    model_dates = [pd.Timestamp(x) for x in config["model"]["expected_monthly_models"]
                   if pd.Timestamp(x) <= pd.Timestamp(data["mature_end"])]
    context = load_market_context_panel(data["oos_start"], data["mature_end"])
    previous_codes: set[str] = set()
    selected_by_date = []
    gate_audit = []
    for index, model_date in enumerate(model_dates):
        prior = sessions[sessions <= model_date]
        if prior.empty:
            continue
        artifact, features, mature = _train(panel, labels, spec, run_dir, prior[-1], profile, config)
        next_date = model_dates[index + 1] if index + 1 < len(model_dates) else pd.Timestamp(data["mature_end"]) + pd.Timedelta(days=1)
        for asof in [x for x in signal_dates if model_date.date() <= x.date() < next_date.date()]:
            source = panel[panel.event_time == asof].copy()
            if source.empty:
                continue
            scores = model_for_prediction(run_dir / "models" / f"{artifact.model_id}.json", source[list(spec.columns)])
            pred = build_prediction_ledger(pd.DataFrame({"ts_code": source.ts_code, "score": scores}),
                                           asof=str(asof.date()), feature_set_id=spec.id, model_id=artifact.model_id)
            top20 = freeze_candidates(pred, top_n=20).merge(source[["ts_code", "dmi_adx_bfq", "xsii_td3_bfq_sector_rel",
                                                                       "expma_12_bfq_sector_rel", "boll_mid_bfq_sector_rel"]], on="ts_code", how="left")
            training_gate = features[["ts_code", "event_time", "dmi_adx_bfq"]].merge(
                mature[["ts_code", "event_time", "label_return"]], on=["ts_code", "event_time"], validate="one_to_one")
            gate = compute_dynamic_upper_gate(training_gate, factor="dmi_adx_bfq",
                                               minimum_samples=selection["dynamic_upper_gate"]["minimum_mature_samples"],
                                               lower_quantile=selection["dynamic_upper_gate"]["lower_tail_quantile"],
                                               upper_quantile=selection["dynamic_upper_gate"]["upper_tail_quantile"])
            top20 = apply_dynamic_upper_gate(top20, factor=gate.factor, threshold=gate.threshold)
            top20 = top20[top20.dynamic_gate_passed].copy()
            chosen = build_q70_selection_ledger(top20, context, asof=str(asof.date()), max_positions=2,
                                                breadth_min=selection["market_breadth_min"], factor_floor=0.8,
                                                weak_breadth_positions=2, volatility_window_sessions=20,
                                                volatility_max=0.07, recent_limit_down_window_sessions=20,
                                                recent_limit_down_threshold=-0.098, peak_drawdown_window_sessions=5,
                                                peak_drawdown_threshold=-0.10, exclude_beijing=True,
                                                alpha_weight=True, alpha_power=1.0)
            eligible = chosen[chosen.rejection_reason == ""].copy()
            held = select_rank_holdings(eligible, previous_codes, max_positions=2, hold_buffer_n=5)
            selected = chosen.assign(selected=False, target_weight=0.0)
            selected.loc[selected.ts_code.astype(str).isin(held.ts_code.astype(str)), "selected"] = True
            selected.loc[selected.selected, "target_weight"] = 1.0 / max(1, int(selected.selected.sum()))
            breadth = float(selected.context_breadth_ratio.iloc[0]) if not selected.empty else 0.0
            fraction = weak_breadth_cash_fraction(breadth=breadth, minimum=.40,
                                                   candidate_count=len(held), configured_fraction=.50)
            selected["cash_fraction"] = fraction
            previous_codes = set(held.ts_code.astype(str))
            write_ledger(pred, run_dir / "predictions" / f"{asof.date()}_prediction.csv")
            write_ledger(selected, run_dir / "selections" / f"{asof.date()}_selection.csv")
            selected_by_date.append(selected)
            gate_audit.append({"model_id": artifact.model_id, "asof": str(asof.date()), **gate.__dict__})
    if not selected_by_date:
        raise RuntimeError("delta-compatible runner produced no selections")
    signals = pd.concat(selected_by_date, ignore_index=True)
    codes = sorted(signals.loc[signals.selected, "ts_code"].astype(str).unique())
    prices = load_execution_panel(data["oos_start"], data["mature_end"], ts_codes=codes)
    actions = load_corporate_actions(data["oos_start"], data["mature_end"], ts_codes=codes)
    result = run_backtest(signals, prices, corporate_actions=actions,
                          config=BacktestConfig(max_positions=2, hold_sessions=9,
                                                stop_loss_pct=execution["stop_loss_pct"],
                                                stop_loss_mode="close_next_session_open",
                                                accounting_price_basis="economic"))
    (run_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics" / "dynamic_gate_audit.json").write_text(json.dumps(gate_audit, indent=2, ensure_ascii=False))
    (run_dir / "diagnostics" / "metrics.json").write_text(json.dumps(
        {"metrics": summarize_backtest(result["nav"], result["trades"], initial_cash=1_000_000.0),
         "contract": "delta-compatible economic accounting; raw limit-state checks; no minute execution"},
        indent=2, ensure_ascii=False, default=str) + "\n")
    return {"models": len(model_dates), "prediction_dates": len(selected_by_date), "status": "completed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(run_dir=args.run_dir, config_path=args.config.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
