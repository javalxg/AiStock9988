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
from datetime import datetime, timezone
from pathlib import Path

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
    config_hash = hashlib.sha256(b"".join(sorted(p.read_bytes() for p in config_dir.rglob("*.yaml")))).hexdigest()[:8]
    run_id = f"{utc}_{name}_{config_hash}"
    running = ROOT / "experiments" / ".running" / run_id
    running.mkdir(parents=True, exist_ok=False)
    for d in ("models", "predictions", "selections", "trades", "diagnostics", "logs"):
        (running / d).mkdir()
    manifest = {
        "run_id": run_id,
        "status": "CREATED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": guard,
        "python": sys.version,
        "platform": platform.platform(),
        "thread_count": 1,
        "config_files": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(config_dir.rglob("*.yaml"))},
    }
    (running / "RUN_STATUS.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (running / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "PYTHONPATH=src python -m aistock9988.cli init-run q70_source_parity_rebuild\n"
    )
    return running


def main() -> int:
    parser = argparse.ArgumentParser(prog="aistock9988")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init-run")
    p.add_argument("name")
    args = parser.parse_args()
    if args.command == "init-run":
        print(init_run(args.name))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
