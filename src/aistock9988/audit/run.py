"""Run-level production gates and immutable artifact bookkeeping."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


class RunAuditError(RuntimeError):
    pass


REQUIRED_FILES = ("RUN_STATUS.json", "data_manifest.json")
REQUIRED_DIRS = ("models", "predictions", "selections", "trades", "diagnostics", "logs")


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
    try:
        status = json.loads((run_dir / "RUN_STATUS.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunAuditError("invalid RUN_STATUS.json") from exc
    if status.get("status") not in {"CREATED", "RUNNING", "VERIFIED"}:
        raise RunAuditError(f"run status cannot be completed: {status.get('status')!r}")
    artifacts = {}
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "RUN_STATUS.json"):
        artifacts[str(path.relative_to(run_dir))] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    return {"run_id": status.get("run_id", run_dir.name), "artifact_count": len(artifacts), "artifacts": artifacts}


def write_audit_report(run_dir: Path, report: dict) -> Path:
    path = run_dir / "diagnostics" / "audit.json"
    if path.exists():
        raise FileExistsError(f"audit report is immutable: {path}")
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
