"""Formal RCQT entrypoint.

The runner refuses to claim a historical result when quant_db credentials are
absent.  With a frozen feature/execution snapshot it delegates to the same
audited smoke path, preserving all ledgers and hashes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
import re

from aistock9988.data.quantdb import connection_kwargs
from aistock9988.data.snapshot import bind_file_snapshot
from aistock9988.audit.run import audit_run, write_audit_report
from scripts.rcqt_smoke_runner import main as run_snapshot


ROOT = Path(__file__).resolve().parents[1]
_RUN_EVIDENCE_DIRS = ("data", "models", "predictions", "selections", "trades", "diagnostics", "logs")


def _write_json_atomic(path: Path, payload: dict) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _load_initialized_status(run_dir: Path) -> dict:
    try:
        status = json.loads((run_dir / "RUN_STATUS.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("formal RCQT requires a valid init-run directory") from exc
    if status.get("status") != "CREATED" or status.get("run_id") != run_dir.name:
        raise SystemExit("formal RCQT requires an untouched CREATED init-run directory")
    return status


def _assert_untouched_run_dir(run_dir: Path) -> None:
    """Reject pre-seeded artifacts before the formal runner can write.

    ``init-run`` owns the lifecycle directory.  Accepting an existing
    ``CREATED`` status alone would let a caller inject ledgers or manifests
    that the runner did not produce, then seal them as if they were part of
    this run.  Only the lifecycle marker, optional command log, and the empty
    evidence directories may exist at this point.
    """
    allowed = {"RUN_STATUS.json", "commands.sh", *_RUN_EVIDENCE_DIRS}
    children = list(run_dir.iterdir())
    unexpected = sorted(child.name for child in children if child.name not in allowed)
    if unexpected:
        raise SystemExit("formal RCQT requires an untouched init-run directory; unexpected entries: " + ", ".join(unexpected))
    for child in children:
        if child.is_symlink():
            raise SystemExit("formal RCQT refuses symlinked init-run entries")
    commands = run_dir / "commands.sh"
    if commands.exists() and not commands.is_file():
        raise SystemExit("formal RCQT requires commands.sh to be a regular file")
    for dirname in _RUN_EVIDENCE_DIRS:
        evidence_dir = run_dir / dirname
        if not evidence_dir.is_dir():
            raise SystemExit(f"formal RCQT init-run is missing evidence directory: {dirname}/")
        if any(evidence_dir.iterdir()):
            raise SystemExit(f"formal RCQT requires an untouched init-run directory; {dirname}/ is not empty")


def _resolve_run_dir(raw_run_dir: Path) -> Path:
    """Resolve and constrain a formal run path before reading any artifacts."""
    # Check the user-supplied path itself first: resolve() would otherwise
    # silently follow a symlink and make an external directory look valid.
    if raw_run_dir.is_symlink():
        raise SystemExit("formal RCQT refuses a symlinked --run-dir")
    try:
        run_dir = raw_run_dir.resolve(strict=True)
    except OSError as exc:
        raise SystemExit("formal RCQT run directory does not exist") from exc
    expected_parent = ROOT.resolve() / "experiments" / ".running"
    if run_dir.parent != expected_parent:
        raise SystemExit("formal RCQT run must resolve under ROOT/experiments/.running")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--run-dir", "--output", dest="run_dir", type=Path, required=True)
    parser.add_argument("--st-codes", type=Path)
    parser.add_argument("--weekly", action="store_true")
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--no-right-confirm", action="store_true")
    parser.add_argument("--reset-slots", type=int, default=4)
    parser.add_argument("--quiet-slots", type=int, default=2)
    parser.add_argument("--hold-sessions", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--single-weight-cap", type=float, default=0.15)
    parser.add_argument("--sector-weight-cap", type=float, default=0.30)
    parser.add_argument("--equity-cap", type=float, default=0.72)
    parser.add_argument("--stop-loss-pct", type=float, default=-0.08)
    parser.add_argument("--no-trailing", action="store_true")
    parser.add_argument("--max-order-to-adv20", type=float)
    parser.add_argument("--slippage", type=float, default=0.0)
    args = parser.parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    if not run_dir.is_dir() or run_dir.parent.name != ".running" or run_dir.parent.parent.name != "experiments":
        raise SystemExit("formal RCQT run must be created by init-run under experiments/.running/<run_id>")
    _assert_untouched_run_dir(run_dir)
    status = _load_initialized_status(run_dir)
    if args.experiment_id and not re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", args.experiment_id):
        raise SystemExit("experiment-id must be 3-64 lowercase safe ASCII characters")
    try:
        connection_kwargs()
        db_available = True
    except RuntimeError as exc:
        db_available = False
        db_error = str(exc)
    if not args.features or not args.prices:
        reason = db_error if not db_available else "frozen --features and --prices snapshots are required"
        status.update({
            "status": "BLOCKED",
            "reason": reason,
            "required_env": ["AISTOCK_DB_HOST", "AISTOCK_DB_USER", "AISTOCK_DB_NAME"],
        })
        _write_json_atomic(run_dir / "RUN_STATUS.json", status)
        raise SystemExit(f"formal RCQT blocked: {reason}")

    # This runner always consumes explicit files; database credentials do not
    # turn those files into a direct quant_db extraction.
    source_mode = "frozen_snapshot"
    bound = {
        "features": bind_file_snapshot(
            args.features,
            run_dir,
            logical_name="features",
            source_id="rcqt.features",
            query={"source_path": str(args.features.resolve())},
            event_column="asof",
        ),
        "prices": bind_file_snapshot(
            args.prices,
            run_dir,
            logical_name="prices",
            source_id="rcqt.execution_prices",
            query={"source_path": str(args.prices.resolve())},
            event_column="trade_date",
        ),
    }
    if args.st_codes:
        bound["st_codes"] = bind_file_snapshot(
            args.st_codes,
            run_dir,
            logical_name="st_codes",
            source_id="rcqt.current_st_blacklist",
            query={"source_path": str(args.st_codes.resolve())},
        )
    if args.experiment_config:
        bound["experiment_config"] = bind_file_snapshot(
            args.experiment_config,
            run_dir,
            logical_name="experiment_config",
            source_id="aistock.experiment_config",
            query={"source_path": str(args.experiment_config.resolve())},
        )
    status.update({"status": "RUNNING", "strategy_type": "rules", "source_mode": source_mode})
    if args.experiment_id:
        status["experiment_id"] = args.experiment_id
    _write_json_atomic(run_dir / "RUN_STATUS.json", status)

    # Compute only from the run-scoped copies, never from mutable source paths.
    old_argv = sys.argv
    sys.argv = [
        "rcqt_smoke_runner",
        "--features", str(run_dir / bound["features"].relative_path),
        "--prices", str(run_dir / bound["prices"].relative_path),
        "--output", str(run_dir),
    ]
    if "st_codes" in bound:
        sys.argv.extend(["--st-codes", str(run_dir / bound["st_codes"].relative_path)])
    if args.weekly:
        sys.argv.append("--weekly")
    sys.argv.extend([
        "--reset-slots", str(args.reset_slots),
        "--quiet-slots", str(args.quiet_slots),
        "--hold-sessions", str(args.hold_sessions),
        "--max-positions", str(args.max_positions),
        "--single-weight-cap", str(args.single_weight_cap),
        "--sector-weight-cap", str(args.sector_weight_cap),
        "--equity-cap", str(args.equity_cap),
        "--stop-loss-pct", str(args.stop_loss_pct),
        "--slippage", str(args.slippage),
    ])
    if args.no_right_confirm:
        sys.argv.append("--no-right-confirm")
    if args.no_trailing:
        sys.argv.append("--no-trailing")
    if args.max_order_to_adv20 is not None:
        sys.argv.extend(["--max-order-to-adv20", str(args.max_order_to_adv20)])
    try:
        run_snapshot()
    finally:
        sys.argv = old_argv
    # Normalize the flat snapshot-run artifacts into the production audit
    # layout.  Keep the original files for convenient human inspection.
    for dirname in ("data", "predictions", "selections", "trades", "diagnostics", "logs", "models"):
        (run_dir / dirname).mkdir(parents=True, exist_ok=True)
    for name in ("features.csv", "prices.csv", "corporate_actions.csv", "codes.csv"):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, run_dir / "data" / name)
    for name in ("score_ledger.csv",):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, run_dir / "predictions" / name)
    for filename in ("selection_ledger.csv", "candidate_ledger.csv"):
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, run_dir / "selections" / source.name)
    for name in ("trades.csv", "fills.csv", "nav.csv", "orders.csv"):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, run_dir / "trades" / name)
    for name in ("metrics.json", "RESULT.md", "st_manifest.json", "universe_exclusion_ledger.csv"):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, run_dir / "diagnostics" / name)
    fills = run_dir / "trades" / "fills.csv"
    if not fills.exists() and (run_dir / "trades.csv").exists():
        shutil.copy2(run_dir / "trades.csv", fills)

    manifest_path = run_dir / "data_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload.update({
        "source_mode": source_mode,
        "features": bound["features"].relative_path,
        "prices": bound["prices"].relative_path,
        "bound_inputs": {name: asdict(snapshot) for name, snapshot in bound.items()},
    })
    if args.experiment_id:
        payload["experiment_id"] = args.experiment_id
    if "st_codes" in bound:
        payload["st_blacklist"]["source"] = bound["st_codes"].relative_path
    _write_json_atomic(manifest_path, payload)

    # The audit report is the final artifact. RUN_STATUS is deliberately outside
    # the artifact seal so its lifecycle state can advance without breaking it.
    report = audit_run(run_dir)
    write_audit_report(run_dir, report)
    status.update({"status": "VERIFIED", "audit_artifact_count": report["artifact_count"]})
    _write_json_atomic(run_dir / "RUN_STATUS.json", status)


if __name__ == "__main__":
    main()
