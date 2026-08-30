"""Run-level production gates and immutable artifact bookkeeping."""
from __future__ import annotations

import hashlib
import json
import numpy as np
import os
import tempfile
from pathlib import Path

import pandas as pd


class RunAuditError(RuntimeError):
    pass


REQUIRED_FILES = ("RUN_STATUS.json", "data_manifest.json")
REQUIRED_DIRS = ("data", "models", "predictions", "selections", "trades", "diagnostics", "logs")
_STATUS_MUTABLE_KEYS = {"status", "created_at", "completed_at", "audit_artifact_count", "completion_seal"}


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
    return _build_report(run_dir, allowed_statuses={"CREATED", "RUNNING", "VERIFIED"})


def reverify_run(run_dir: Path) -> dict:
    """Recompute the seal of a completed run without writing into it."""
    run_dir = run_dir.resolve()
    if run_dir.parent.name != "completed" or run_dir.parent.parent.name != "experiments":
        raise RunAuditError("completed run must live under experiments/completed/<run_id>")
    audit_path = run_dir / "diagnostics" / "audit.json"
    # Keep all completed-bundle tampering failures inside the audit contract.
    # Without this guard a deleted seal report leaked FileNotFoundError from
    # build_completion_seal, making callers handle a filesystem exception
    # instead of the lifecycle's fail-closed RunAuditError.
    if not audit_path.is_file():
        raise RunAuditError("completed audit report is missing")
    try:
        status = json.loads((run_dir / "RUN_STATUS.json").read_text())
        recorded = json.loads(audit_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunAuditError("invalid completed status or audit report") from exc
    if status.get("completion_seal") != build_completion_seal(status, audit_path):
        raise RunAuditError("completed RUN_STATUS.json no longer matches its completion seal")
    report = _build_report(run_dir, allowed_statuses={"COMPLETED"})
    if recorded != report:
        raise RunAuditError("completed run artifacts no longer match the sealed audit report")
    return report


def build_completion_seal(status: dict, audit_path: Path) -> str:
    """Bind the final lifecycle status to the immutable artifact audit."""
    payload = {
        "status": {key: value for key, value in status.items() if key != "completion_seal"},
        "audit_sha256": _sha256(audit_path),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _build_report(run_dir: Path, *, allowed_statuses: set[str]) -> dict:
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    missing += [name + "/" for name in REQUIRED_DIRS if not (run_dir / name).is_dir()]
    if not (run_dir / "trades").is_dir() or not any((run_dir / "trades").iterdir()):
        missing.append("trades/<fills>")
    if not (run_dir / "data").is_dir() or not any((run_dir / "data").iterdir()):
        missing.append("data/<snapshot>")
    if not (run_dir / "predictions").is_dir() or not any((run_dir / "predictions").iterdir()):
        missing.append("predictions/<ledger>")
    # Rule-only experiments intentionally have no model artifact.  They must
    # provide a frozen score ledger instead; model runs retain the old gate.
    status_payload = {}
    try:
        status_payload = json.loads((run_dir / "RUN_STATUS.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    if status_payload.get("strategy_type") != "rules" and (not (run_dir / "models").is_dir() or not any((run_dir / "models").iterdir())):
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
    if status.get("status") not in allowed_statuses:
        raise RunAuditError(f"run status cannot be completed: {status.get('status')!r}")
    artifacts = {}
    excluded = {run_dir / "RUN_STATUS.json", run_dir / "diagnostics" / "audit.json"}
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file() and p not in excluded):
        artifacts[str(path.relative_to(run_dir))] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    run_contract = {key: value for key, value in status.items() if key not in _STATUS_MUTABLE_KEYS}
    return {
        "run_id": status.get("run_id", run_dir.name),
        "run_contract": run_contract,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _validate_ledgers(run_dir: Path) -> None:
    try:
        status_payload = json.loads((run_dir / "RUN_STATUS.json").read_text())
    except (OSError, json.JSONDecodeError):
        status_payload = {}
    selection_files = sorted((run_dir / "selections").glob("*.csv"))
    for selection_file in selection_files:
        selection = pd.read_csv(selection_file)
        if {"asof", "ts_code"} <= set(selection.columns) and selection.duplicated(["asof", "ts_code"]).any():
            raise RunAuditError(f"selection ledger has duplicate asof/ts_code keys: {selection_file.name}")
        required = {"selected", "selection_decision_id", "policy_id", "candidate_rank",
                    "target_weight", "context_hash"}
        if not required <= set(selection.columns):
            raise RunAuditError(f"selection ledger is not a SelectionDecision ledger: {selection_file.name}")
        selected = selection["selected"].astype(str).str.lower().map({"true": True, "false": False})
        ranks = pd.to_numeric(selection["candidate_rank"], errors="coerce")
        weights = pd.to_numeric(selection["target_weight"], errors="coerce")
        if (selected.isna().any() or weights.isna().any() or ranks.isna().any() or
                not np.isfinite(weights.to_numpy(dtype=float)).all() or
                not np.isfinite(ranks.to_numpy(dtype=float)).all() or (weights < 0).any()):
            raise RunAuditError(f"selection ledger has invalid selected/weight values: {selection_file.name}")
        if selected.any() and not (0.0 < float(weights[selected].sum()) <= 1.0 + 1e-8):
            raise RunAuditError(f"selection weights must sum to (0,1]: {selection_file.name}")
        if (weights[~selected] != 0).any():
            raise RunAuditError(f"rejected candidates carry target weight: {selection_file.name}")
        chosen = selection.loc[selected]
        if status_payload.get("strategy_type") == "rules" and not chosen.empty and (weights[selected] > 0.150000001).any():
            raise RunAuditError(f"selection exceeds single-name 15% cap: {selection_file.name}")
        if status_payload.get("strategy_type") == "rules" and not chosen.empty and "industry" in chosen.columns:
            sector_totals = chosen.assign(_w=weights[selected].to_numpy()).groupby("industry")['_w'].sum()
            if (sector_totals > 0.300000001).any():
                raise RunAuditError(f"selection exceeds industry 30% cap: {selection_file.name}")
    for prediction_file in sorted((run_dir / "predictions").glob("*.csv")):
        predictions = pd.read_csv(prediction_file)
        if status_payload.get("strategy_type") == "rules" and "score" not in predictions.columns:
            raise RunAuditError(f"rule prediction ledger must contain frozen score column: {prediction_file.name}")
        if {"asof", "ts_code"} <= set(predictions.columns) and predictions.duplicated(["asof", "ts_code"]).any():
            raise RunAuditError(f"prediction ledger has duplicate asof/ts_code keys: {prediction_file.name}")
        if "score" in predictions:
            scores = pd.to_numeric(predictions["score"], errors="coerce")
            if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
                raise RunAuditError(f"prediction ledger has non-finite scores: {prediction_file.name}")
    fills_files = sorted((run_dir / "trades").glob("*fills*.csv"))
    for fills_file in fills_files:
        fills = pd.read_csv(fills_file)
        if not {"order_id", "side", "price", "shares"} <= set(fills.columns):
            raise RunAuditError(f"fills ledger is missing accounting columns: {fills_file.name}")
        prices = pd.to_numeric(fills["price"], errors="coerce")
        shares = pd.to_numeric(fills["shares"], errors="coerce")
        if not fills.empty and (prices.isna().any() or shares.isna().any() or
                                not np.isfinite(prices.to_numpy(dtype=float)).all() or
                                not np.isfinite(shares.to_numpy(dtype=float)).all() or
                                (prices <= 0).any() or (shares <= 0).any()):
            raise RunAuditError(f"fills ledger has invalid price/shares: {fills_file.name}")
        if "order_id" in fills and fills["order_id"].duplicated().any():
            raise RunAuditError(f"fills ledger has duplicate order_id: {fills_file.name}")
    nav_files = sorted((run_dir / "trades").glob("nav*.csv"))
    for nav_file in nav_files:
        nav = pd.read_csv(nav_file)
        if not {"cash", "market_value", "nav"} <= set(nav.columns):
            raise RunAuditError(f"NAV ledger is missing accounting identity columns: {nav_file.name}")
        values = nav[["cash", "market_value", "nav"]].apply(pd.to_numeric, errors="coerce")
        if (values.isna().any().any() or
                not np.isfinite(values.to_numpy(dtype=float)).all() or
                ((values["cash"] + values["market_value"] - values["nav"]).abs() > 1e-8).any()):
            raise RunAuditError(f"NAV accounting identity failed: {nav_file.name}")
        if "trade_date" in nav and nav["trade_date"].duplicated().any():
            raise RunAuditError(f"NAV ledger has duplicate trade_date: {nav_file.name}")


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
