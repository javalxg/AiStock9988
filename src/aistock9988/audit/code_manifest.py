"""Immutable code provenance for experiment bundles."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_code_manifest(*, repo_root: Path, config_path: Path, entrypoint: Path) -> dict:
    """Capture the commit and exact source hashes used by a run.

    The working-tree state is recorded rather than hidden.  A dirty source
    tree therefore remains visible evidence and cannot be mistaken for a
    clean, reproducible baseline.
    """
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True,
        capture_output=True, check=True,
    ).stdout.splitlines()
    paths = [config_path.resolve(), entrypoint.resolve(),
             *sorted((repo_root / "src" / "aistock9988").rglob("*.py"))]
    files = {}
    for path in paths:
        if path.is_file():
            files[str(path.relative_to(repo_root))] = {
                "sha256": _sha256(path), "bytes": path.stat().st_size,
            }
    config_hash = _sha256(config_path.resolve())
    return {
        "git_commit": commit,
        "git_status": "clean" if not status else "dirty",
        "dirty_paths": status,
        "config": str(config_path.resolve().relative_to(repo_root)),
        "config_sha256": config_hash,
        "entrypoint": str(entrypoint.resolve().relative_to(repo_root)),
        "source_files": files,
    }
