"""Content-addressed, read-only source snapshots shared by backtest runs."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


CACHE_VERSION = "source-snapshot-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SharedSnapshotStore:
    """Store one immutable parquet snapshot per source-query contract.

    The cache is deliberately outside experiment output directories. Experiment
    bundles receive hard links to these files, so their audit paths remain
    intact without allocating another copy of the source bytes.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            configured = os.environ.get("AISTOCK_DATA_CACHE_DIR")
            root = configured or Path(__file__).resolve().parents[3] / "experiments/.cache/shared_snapshots"
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, *, feature_start: str, execution_end: str,
            required_sources: tuple[str, ...] | list[str], codes: list[str]) -> str:
        payload = {
            "cache_version": CACHE_VERSION,
            "feature_start": str(feature_start),
            "execution_end": str(execution_end),
            "required_sources": sorted(str(value) for value in required_sources),
            "codes_sha256": hashlib.sha256("\n".join(sorted(codes)).encode("utf-8")).hexdigest(),
        }
        return _canonical_hash(payload)

    def _directory(self, key: str) -> Path:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("snapshot key must be a lowercase SHA-256 hex digest")
        return self.root / key

    def load(self, key: str, names: list[str]) -> dict[str, pd.DataFrame] | None:
        directory = self._directory(key)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("cache_version") != CACHE_VERSION:
                return None
            files = manifest.get("files", {})
            if sorted(files) != sorted(names):
                return None
            frames: dict[str, pd.DataFrame] = {}
            for name in names:
                path = directory / f"{name}.parquet"
                if not path.is_file():
                    return None
                expected = str(files[name].get("sha256", ""))
                if expected and _sha256_file(path) != expected:
                    return None
                frames[name] = pd.read_parquet(path)
            return frames
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def write(self, key: str, frames: Mapping[str, pd.DataFrame], metadata: Mapping[str, Any]) -> None:
        directory = self._directory(key)
        if (directory / "manifest.json").is_file():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=self.root))
        try:
            files: dict[str, dict[str, Any]] = {}
            for name, frame in sorted(frames.items()):
                if not name or Path(name).name != name:
                    raise ValueError(f"invalid snapshot name: {name!r}")
                target = temp / f"{name}.parquet"
                frame.to_parquet(target, index=False)
                files[name] = {
                    "sha256": _sha256_file(target),
                    "rows": int(len(frame)),
                    "columns": [str(column) for column in frame.columns],
                }
            manifest = {
                "cache_version": CACHE_VERSION,
                "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "metadata": dict(metadata),
                "files": files,
            }
            (temp / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                os.replace(temp, directory)
            except FileExistsError:
                # Another process completed the same immutable key first.
                shutil.rmtree(temp)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise

    def materialize(self, key: str, names: list[str], raw_dir: Path) -> dict[str, str]:
        directory = self._directory(key)
        if not (directory / "manifest.json").is_file():
            raise FileNotFoundError(f"shared snapshot is not complete: {directory}")
        links: dict[str, str] = {}
        for name in names:
            source = directory / f"{name}.parquet"
            target = raw_dir / source.name
            if target.exists():
                raise FileExistsError(f"raw snapshot target already exists: {target}")
            try:
                os.link(source, target)
                method = "hardlink"
            except OSError:
                shutil.copyfile(source, target)
                method = "copy_fallback"
            links[name] = method
        return links

    def manifest(self, key: str) -> dict[str, Any]:
        path = self._directory(key) / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["CACHE_VERSION", "SharedSnapshotStore"]
