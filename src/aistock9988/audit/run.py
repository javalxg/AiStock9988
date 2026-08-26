"""Run-level production gates and immutable artifact bookkeeping."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


class RunAuditError(RuntimeError):
    pass


REQUIRED_FILES = ("RUN_STATUS.json", "data_manifest.json")
REQUIRED_DIRS = ("data", "models", "predictions", "selections", "trades", "diagnostics", "logs")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_run(run_dir: Path) -> dict:
    """Validate that a run has enough immutable evidence to be completed."""
    run_dir = run_dir.resolve()
    if run_dir.parent.name != ".running" or run_dir.parent.parent.name != "experiments":
        raise RunAuditError("run must be audited from experiments/.running/<run_id>")
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    missing += [name + "/" for name in REQUIRED_DIRS if not (run_dir / name).is_dir()]
    if not (run_dir / "trades").is_dir() or not any((run_dir / "trades").iterdir()):
        missing.append("trades/<fills>")
    if not (run_dir / "data").is_dir() or not any((run_dir / "data").iterdir()):
        missing.append("data/<snapshot>")
    if not (run_dir / "predictions").is_dir() or not any((run_dir / "predictions").iterdir()):
        missing.append("predictions/<ledger>")
    if not (run_dir / "models").is_dir() or not any((run_dir / "models").iterdir()):
        missing.append("models/<artifact>")
    if not (run_dir / "selections").is_dir() or not any((run_dir / "selections").iterdir()):
        missing.append("selections/<ledger>")
    if not (run_dir / "diagnostics").is_dir() or not any((run_dir / "diagnostics").iterdir()):
        missing.append("diagnostics/<checks>")
    if missing:
        raise RunAuditError("missing required run evidence: " + ", ".join(missing))
    _validate_ledgers(run_dir)
    try:
        status = json.loads((run_dir / "RUN_STATUS.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunAuditError("invalid RUN_STATUS.json") from exc
    if status.get("status") not in {"CREATED", "RUNNING", "VERIFIED"}:
        raise RunAuditError(f"run status cannot be completed: {status.get('status')!r}")
    artifacts = {}
    excluded = {run_dir / "RUN_STATUS.json", run_dir / "diagnostics" / "audit.json"}
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file() and p not in excluded):
        artifacts[str(path.relative_to(run_dir))] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    return {"run_id": status.get("run_id", run_dir.name), "artifact_count": len(artifacts), "artifacts": artifacts}


def _validate_ledgers(run_dir: Path) -> None:
    selection_files = sorted((run_dir / "selections").glob("*.csv"))
    for selection_file in selection_files:
        selection = pd.read_csv(selection_file)
        required = {"selected", "selection_decision_id", "policy_id", "candidate_rank",
                    "target_weight", "context_hash"}
        if not required <= set(selection.columns):
            raise RunAuditError(f"selection ledger is not a SelectionDecision ledger: {selection_file.name}")
        selected = selection["selected"].astype(str).str.lower().map({"true": True, "false": False})
        weights = pd.to_numeric(selection["target_weight"], errors="coerce")
        if selected.isna().any() or weights.isna().any() or (weights < 0).any():
            raise RunAuditError(f"selection ledger has invalid selected/weight values: {selection_file.name}")
        if selected.any() and abs(float(weights[selected].sum()) - 1.0) > 1e-8:
            raise RunAuditError(f"selection weights do not sum to one: {selection_file.name}")
        if (weights[~selected] != 0).any():
            raise RunAuditError(f"rejected candidates carry target weight: {selection_file.name}")
    fills_files = sorted((run_dir / "trades").glob("*fills*.csv"))
    for fills_file in fills_files:
        fills = pd.read_csv(fills_file)
        if not {"order_id", "side", "price", "shares"} <= set(fills.columns):
            raise RunAuditError(f"fills ledger is missing accounting columns: {fills_file.name}")
        if not fills.empty and ((pd.to_numeric(fills["price"], errors="coerce") <= 0).any() or
                                (pd.to_numeric(fills["shares"], errors="coerce") <= 0).any()):
            raise RunAuditError(f"fills ledger has invalid price/shares: {fills_file.name}")
    nav_files = sorted((run_dir / "trades").glob("nav*.csv"))
    for nav_file in nav_files:
        nav = pd.read_csv(nav_file)
        if not {"cash", "market_value", "nav"} <= set(nav.columns):
            raise RunAuditError(f"NAV ledger is missing accounting identity columns: {nav_file.name}")
        values = nav[["cash", "market_value", "nav"]].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any() or ((values["cash"] + values["market_value"] - values["nav"]).abs() > 1e-8).any():
            raise RunAuditError(f"NAV accounting identity failed: {nav_file.name}")


def write_audit_report(run_dir: Path, report: dict) -> Path:
    path = run_dir / "diagnostics" / "audit.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RunAuditError(f"invalid existing audit report: {path}") from exc
        if existing != report:
            raise RunAuditError(f"immutable audit report does not match current artifacts: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path
