"""Lifecycle-only CLI for the first experiment; model/data adapters come next."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from aistock9988.audit.run import (
    RunAuditError,
    audit_run,
    build_completion_seal,
    reverify_run as audit_completed_run,
    write_audit_report,
)

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_file():
        h.update(path.read_bytes())
    return h.hexdigest()


def git_guard() -> dict:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError("project is not a git repository")
    if result.stdout.strip():
        raise RuntimeError("正式实验拒绝启动：Git 工作区存在未提交或未跟踪文件")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {"commit": commit, "status": "clean"}


def init_run(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,63}", name):
        raise ValueError("experiment name must be 3-64 safe ASCII characters")
    guard = git_guard()
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_dir = ROOT / "configs"
    config_paths = sorted([*config_dir.rglob("*.yaml"), *config_dir.rglob("*.yml"), *config_dir.rglob("*.json")])
    config_hash = hashlib.sha256(b"".join(p.read_bytes() for p in config_paths)).hexdigest()[:8]
    run_id = f"{utc}_{name}_{config_hash}"
    running = ROOT / "experiments" / ".running" / run_id
    running.mkdir(parents=True, exist_ok=False)
    for d in ("data", "models", "predictions", "selections", "trades", "diagnostics", "logs"):
        (running / d).mkdir()
    manifest = {
        "run_id": run_id,
        "status": "CREATED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": guard,
        "python": sys.version,
        "platform": platform.platform(),
        "thread_count": 1,
        "config_files": {str(p.relative_to(ROOT)): sha256(p) for p in config_paths},
    }
    (running / "RUN_STATUS.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (running / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"# Runner command is selected by the experiment package; run_id={run_id}\n"
    )
    return running


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


def _update_run_status(path: Path, **updates: object) -> dict:
    status_path = path / "RUN_STATUS.json"
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunAuditError("invalid RUN_STATUS.json") from exc
    status.update(updates)
    _write_json_atomic(status_path, status)
    return status


def verify_run(path: Path) -> dict:
    report = audit_run(path)
    write_audit_report(path, report)
    _update_run_status(path, status="VERIFIED", audit_artifact_count=report["artifact_count"])
    return report


def complete_run(path: Path) -> Path:
    path = path.resolve()
    try:
        current_status = json.loads((path / "RUN_STATUS.json").read_text()).get("status")
    except (OSError, json.JSONDecodeError) as exc:
        raise RunAuditError("invalid RUN_STATUS.json") from exc
    if current_status != "VERIFIED":
        raise RunAuditError("run must pass verify-run before complete-run")
    destination = ROOT / "experiments" / "completed" / path.name
    if destination.exists():
        raise FileExistsError(f"completed run already exists: {destination}")
    report = audit_run(path)
    audit_path = write_audit_report(path, report)
    original_status = json.loads((path / "RUN_STATUS.json").read_text())
    final_status = dict(original_status)
    final_status.update({
        "status": "COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "audit_artifact_count": report["artifact_count"],
    })
    final_status["completion_seal"] = build_completion_seal(final_status, audit_path)
    _write_json_atomic(path / "RUN_STATUS.json", final_status)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(path, destination)
    except Exception:
        _write_json_atomic(path / "RUN_STATUS.json", original_status)
        raise
    return destination


def reverify_run(path: Path) -> dict:
    return audit_completed_run(path)


def main() -> int:
    parser = argparse.ArgumentParser(prog="aistock9988")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init-run")
    p.add_argument("name")
    p = sub.add_parser("verify-run")
    p.add_argument("run_dir", type=Path)
    p = sub.add_parser("complete-run")
    p.add_argument("run_dir", type=Path)
    p = sub.add_parser("reverify-run")
    p.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.command == "init-run":
        print(init_run(args.name))
        return 0
    if args.command == "verify-run":
        try:
            report = verify_run(args.run_dir)
        except RunAuditError as exc:
            parser.error(str(exc))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "complete-run":
        try:
            print(complete_run(args.run_dir))
        except (RunAuditError, FileExistsError) as exc:
            parser.error(str(exc))
        return 0
    if args.command == "reverify-run":
        try:
            report = reverify_run(args.run_dir)
        except RunAuditError as exc:
            parser.error(str(exc))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
