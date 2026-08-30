"""Resume a completed q70 selection phase without retraining models."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from aistock9988.audit.code_manifest import build_code_manifest
from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.data.execution_source import load_execution_panel
from aistock9988.data.snapshot import build_snapshot_meta
from aistock9988.reporting.metrics import summarize_backtest

LOGGER = logging.getLogger("aistock9988.q70_replay")
ROOT = Path(__file__).resolve().parents[1]


def _configure_logging(run_dir: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    handler = logging.FileHandler(run_dir / "logs" / "runner.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S%z"))
    LOGGER.addHandler(handler)


def _write(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def replay(*, run_dir: Path, config_path: Path) -> dict:
    run_dir = run_dir.resolve()
    _configure_logging(run_dir)
    config = yaml.safe_load(config_path.read_text())
    data, label_cfg = config["data"], config["label"]
    execution, selection = config["execution"], config["selection"]
    selection_files = sorted((run_dir / "selections").glob("*.csv"))
    if not selection_files:
        raise RuntimeError("cannot replay without existing selection ledgers")
    signals = pd.concat([pd.read_csv(path) for path in selection_files], ignore_index=True)
    signals["selected"] = signals["selected"].astype(str).str.lower().map({"true": True, "false": False})
    if signals["selected"].isna().any():
        raise ValueError("selection ledger contains invalid selected values")
    model_ids = sorted(signals["model_id"].dropna().astype(str).unique())
    missing_models = [model_id for model_id in model_ids
                      if not (run_dir / "models" / f"{model_id}.json").is_file()
                      or not (run_dir / "models" / f"{model_id}.metadata.json").is_file()]
    if missing_models:
        raise RuntimeError(f"existing model artifacts missing: {missing_models}")
    codes = sorted(signals.loc[signals["selected"], "ts_code"].astype(str).unique())
    LOGGER.info("phase=replay_start selection_files=%d signal_rows=%d models=%d codes=%d",
                len(selection_files), len(signals), len(model_ids), len(codes))
    prices = load_execution_panel(data["oos_start"], data["mature_end"], ts_codes=codes)
    actions = load_corporate_actions(data["oos_start"], data["mature_end"], ts_codes=codes)
    LOGGER.info("phase=execution_loaded daily_rows=%d action_rows=%d", len(prices), len(actions))

    def progress(current: int, total: int, day: pd.Timestamp) -> None:
        if current == 1 or current == total or current % max(1, total // 10) == 0:
            LOGGER.info("phase=backtest_progress sessions=%d/%d date=%s", current, total, day.date())

    result = run_backtest(
        signals, prices, corporate_actions=actions,
        config=BacktestConfig(max_positions=selection["max_positions"],
                              hold_sessions=label_cfg["entry_to_exit_sessions"],
                              stop_loss_pct=execution["stop_loss_pct"],
                              take_profit_pct=execution["take_profit_pct"],
                              stop_loss_mode=execution["stop_loss_mode"],
                              accounting_price_basis=execution["accounting_price_basis"],
                              corporate_actions_mode=execution["corporate_actions_mode"],
                              progress_callback=progress),
    )
    for key, filename in (("orders", "orders.csv"), ("trades", "fills.csv"), ("nav", "nav.csv"),
                          ("positions", "positions.csv"), ("corporate_actions", "corporate_actions.csv")):
        _write(run_dir / "trades" / filename, result[key].to_csv(index=False, lineterminator="\n"))
    manifest = {"snapshots": {
        "selection_ledgers": asdict(build_snapshot_meta(signals, source_id="derived.q70_selection_ledgers",
                                                         query={"files": len(selection_files)}, event_column="asof")),
        "execution_daily": asdict(build_snapshot_meta(prices, source_id="quant_db.execution_daily",
                                                        query={"start": data["oos_start"], "end": data["mature_end"]}, event_column="trade_date")),
        "corporate_actions": asdict(build_snapshot_meta(actions, source_id="quant_db.corporate_actions",
                                                          query={"start": data["oos_start"], "end": data["mature_end"]}, event_column="ex_date")),
    }, "config": str(config_path.resolve()), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "contract": "replay existing models and selection ledgers; no retraining; economic accounting with automatic company-action exclusion",
        "models_reused": model_ids, "corporate_actions_applied": False}
    _write(run_dir / "data_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n")
    _write(run_dir / "code_manifest.json", json.dumps(build_code_manifest(repo_root=ROOT, config_path=config_path.resolve(),
                                                                           entrypoint=Path(__file__).resolve()),
                                                        ensure_ascii=False, indent=2, default=str) + "\n")
    _write(run_dir / "diagnostics/metrics.json", json.dumps({"metrics": summarize_backtest(
        result["nav"], result["trades"], initial_cash=1_000_000.0), "models_reused": model_ids,
        "retrained": False, "corporate_actions_applied": False}, ensure_ascii=False, indent=2, default=str) + "\n")
    LOGGER.info("phase=replay_complete trades=%d nav_rows=%d", len(result["trades"]), len(result["nav"]))
    return {"status": "executed", "models_reused": len(model_ids), "trades": len(result["trades"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(run_dir=args.run_dir, config_path=args.config.resolve()), ensure_ascii=False))
