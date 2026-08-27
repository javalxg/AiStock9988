"""Replay a frozen q70 selection ledger through the production execution engine.

This is intentionally separate from model training.  It is used when an
execution-data adapter is corrected while the frozen model/prediction/
selection artifacts remain valid and must not be regenerated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.audit.code_manifest import build_code_manifest
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.data.execution_source import load_execution_panel
from aistock9988.data.minute_source import load_minute_execution_panel
from aistock9988.data.snapshot import build_snapshot_meta
from aistock9988.reporting.metrics import summarize_backtest


ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("aistock9988.q70_execution_replay")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_logging(run_dir: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S%z")
    handler = logging.FileHandler(run_dir / "logs" / "runner.log", encoding="utf-8")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)


def _write_once(path: Path, payload: bytes) -> None:
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _write_once(path, (json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n").encode())


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    _write_once(path, frame.to_csv(index=False, lineterminator="\n").encode())


def _copy_frozen_artifacts(source_dir: Path, run_dir: Path) -> dict[str, Any]:
    source_status = json.loads((source_dir / "RUN_STATUS.json").read_text())
    if source_status.get("status") not in {"FAILED", "COMPLETED", "VERIFIED", "CREATED", "RUNNING"}:
        raise ValueError(f"unsupported frozen source status: {source_status.get('status')!r}")
    copied: dict[str, list[dict[str, Any]]] = {}
    for name in ("models", "predictions", "selections"):
        files = sorted(p for p in (source_dir / name).glob("*") if p.is_file())
        if not files:
            raise ValueError(f"frozen source has no {name} artifacts")
        entries = []
        for source in files:
            target = run_dir / name / source.name
            shutil.copy2(source, target)
            entries.append({"path": str(source.relative_to(source_dir)), "sha256": _sha256(source),
                            "bytes": source.stat().st_size})
        copied[name] = entries
    return {"run_id": source_status.get("run_id", source_dir.name),
            "status": source_status.get("status"),
            "git": source_status.get("git"), "artifacts": copied}


def _load_signals(run_dir: Path) -> tuple[pd.DataFrame, int]:
    files = sorted((run_dir / "selections").glob("*.csv"))
    frames = [pd.read_csv(path) for path in files]
    signals = pd.concat(frames, ignore_index=True)
    if "selected" not in signals:
        raise ValueError("selection ledgers must contain selected")
    if signals["selected"].dtype != bool:
        signals["selected"] = signals["selected"].astype(str).str.lower().map({"true": True, "false": False})
    if signals["selected"].isna().any():
        raise ValueError("selection ledgers contain invalid selected values")
    selected = signals[signals["selected"]].copy()
    if selected.empty:
        raise ValueError("frozen selection ledgers contain no selected signals")
    return selected, len(files)


def run(*, run_dir: Path, source_run: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    data = config["data"]
    execution = config["execution"]
    if not data.get("forbid_old_ledger") or not data.get("forbid_stage2"):
        raise ValueError("formal q70 replay must forbid old ledgers and Stage2")
    if execution.get("minute_data") != "5min" or execution.get("stop_loss_mode") != "intraday_5min":
        raise ValueError("formal q70 replay requires 5min intraday execution")
    run_dir = run_dir.resolve()
    source_run = source_run.resolve()
    config_path = config_path.resolve()
    if not (run_dir / "RUN_STATUS.json").is_file():
        raise RuntimeError("run directory must be initialized by the project CLI")
    if not source_run.is_dir() or source_run == run_dir:
        raise ValueError("a distinct frozen selection source run is required")
    _configure_logging(run_dir)
    started = time.monotonic()
    LOGGER.info("run_start run_dir=%s source_run=%s config=%s", run_dir, source_run, config_path)
    source_meta = _copy_frozen_artifacts(source_run, run_dir)
    signals, selection_file_count = _load_signals(run_dir)
    codes = sorted(signals["ts_code"].astype(str).unique())
    formal_end = str(data["mature_end"])
    LOGGER.info("phase=frozen_selection_loaded files=%d selected_rows=%d codes=%d", selection_file_count, len(signals), len(codes))
    LOGGER.info("phase=execution_data_load start=%s end=%s codes=%d", data["oos_start"], formal_end, len(codes))
    prices = load_execution_panel(data["oos_start"], formal_end, ts_codes=codes)
    actions = load_corporate_actions(data["oos_start"], formal_end, ts_codes=codes)
    minutes = load_minute_execution_panel(data["oos_start"], formal_end, freq="5min", ts_codes=codes)
    requested_end = pd.Timestamp(formal_end).date()
    actual_daily_end = pd.to_datetime(prices["trade_date"], utc=True).dt.date.max()
    actual_minute_end = pd.to_datetime(minutes["trade_time"], utc=True).dt.date.max()
    if actual_daily_end < requested_end or actual_minute_end < requested_end:
        raise ValueError(
            f"execution data does not cover formal mature_end={requested_end}: "
            f"daily_end={actual_daily_end}, minute_end={actual_minute_end}"
        )
    LOGGER.info("execution_daily rows=%d cols=%d", len(prices), len(prices.columns))
    LOGGER.info("corporate_actions rows=%d cols=%d", len(actions), len(actions.columns))
    LOGGER.info("execution_5min rows=%d cols=%d", len(minutes), len(minutes.columns))
    _write_frame(run_dir / "data" / "execution_daily.csv", prices)
    _write_frame(run_dir / "data" / "corporate_actions.csv", actions)
    _write_frame(run_dir / "data" / "execution_5min.csv", minutes)
    LOGGER.info("phase=backtest start hold_sessions=%d stop_loss_mode=%s", config["label"]["entry_to_exit_sessions"], execution["stop_loss_mode"])
    result = run_backtest(signals, prices, corporate_actions=actions, minute_prices=minutes,
                          config=BacktestConfig(max_positions=config["selection"]["max_positions"],
                                                hold_sessions=config["label"]["entry_to_exit_sessions"],
                                                stop_loss_pct=execution["stop_loss_pct"],
                                                take_profit_pct=execution["take_profit_pct"],
                                                stop_loss_mode=execution["stop_loss_mode"]))
    LOGGER.info("phase=backtest_complete trades=%d orders=%d nav_rows=%d", len(result["trades"]), len(result["orders"]), len(result["nav"]))
    for key, filename in (("orders", "orders.csv"), ("trades", "fills.csv"), ("nav", "nav.csv"),
                          ("positions", "positions.csv")):
        _write_frame(run_dir / "trades" / filename, result[key])
    metrics = summarize_backtest(result["nav"], result["trades"], initial_cash=1_000_000.0)
    _write_json(run_dir / "diagnostics" / "metrics.json", {
        "metrics": metrics, "models_reused": len(source_meta["artifacts"]["models"]),
        "prediction_files_reused": len(source_meta["artifacts"]["predictions"]),
        "selection_files_reused": selection_file_count, "selected_rows": len(signals),
        "formal_end": formal_end, "execution": "raw accounting + economic trigger + 5min",
        "frozen_source_run": source_meta["run_id"],
    })
    manifest = {
        "snapshots": {
            "execution_daily": asdict(build_snapshot_meta(prices, source_id="quant_db.execution_daily",
                query={"start": data["oos_start"], "end": formal_end}, event_column="trade_date")),
            "corporate_actions": asdict(build_snapshot_meta(actions, source_id="quant_db.corporate_actions",
                query={"start": data["oos_start"], "end": formal_end}, event_column="ex_date")),
            "execution_5min": asdict(build_snapshot_meta(minutes, source_id="quant_db.execution_5min",
                query={"start": data["oos_start"], "end": formal_end, "freq": "5min",
                       "storage": "16 sha256 buckets", "non_trading_placeholders": "excluded by market_daily join"},
                event_column="trade_time")),
        },
        "frozen_selection_source": source_meta,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "contract": "reused frozen q70 selection; raw accounting, economic risk triggers, 5min intraday stop",
    }
    _write_json(run_dir / "data_manifest.json", manifest)
    _write_json(run_dir / "code_manifest.json", build_code_manifest(
        repo_root=ROOT, config_path=config_path, entrypoint=Path(__file__).resolve()))
    LOGGER.info("run_complete models_reused=%d selected_rows=%d elapsed_seconds=%.1f",
                len(source_meta["artifacts"]["models"]), len(signals), time.monotonic() - started)
    return {"models_reused": len(source_meta["artifacts"]["models"]),
            "prediction_files_reused": len(source_meta["artifacts"]["predictions"]),
            "selection_files_reused": selection_file_count, "selected_rows": len(signals)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(run_dir=args.run_dir, source_run=args.source_run,
                         config_path=args.config), ensure_ascii=False))


if __name__ == "__main__":
    main()
