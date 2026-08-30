from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SnapshotMeta:
    source_id: str
    query_hash: str
    schema_hash: str
    content_hash: str
    row_count: int
    min_event_time: str | None
    max_event_time: str | None
    extracted_at: str


@dataclass(frozen=True)
class BoundFileSnapshot:
    """Recoverable, run-scoped copy of one raw experiment input."""

    logical_name: str
    source_id: str
    relative_path: str
    raw_sha256: str
    query_hash: str
    schema_hash: str
    content_hash: str
    byte_count: int
    row_count: int
    columns: tuple[str, ...]
    min_event_time: str | None
    max_event_time: str | None


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bound_frame(path: Path) -> pd.DataFrame | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return None


def bind_file_snapshot(
    source: Path,
    run_dir: Path,
    *,
    logical_name: str,
    source_id: str,
    query: dict[str, Any],
    event_column: str | None = None,
) -> BoundFileSnapshot:
    """Copy an input into ``run_dir`` before computation and describe it deterministically."""
    source = source.resolve()
    run_dir = run_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"snapshot source is not a file: {source}")
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", logical_name):
        raise ValueError("logical_name must be 2-64 lowercase ASCII characters")

    suffix = source.suffix.lower() or ".bin"
    target_dir = run_dir / "data" / "inputs"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{logical_name}{suffix}"
    if target.exists():
        raise FileExistsError(f"immutable bound snapshot already exists: {target}")

    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target_dir)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        shutil.copyfile(source, temp_path)
        os.replace(temp_path, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    try:
        raw_sha256 = _hash_file(target)
        if raw_sha256 != _hash_file(source):
            raise IOError(f"bound snapshot hash mismatch: {source}")

        frame = _read_bound_frame(target)
        if frame is None:
            raw = target.read_bytes()
            row_count = sum(1 for line in raw.splitlines() if line.strip())
            columns: tuple[str, ...] = ()
            schema_hash = _hash_bytes(json.dumps({"format": suffix}, sort_keys=True).encode())
            content_hash = raw_sha256
            min_event_time = max_event_time = None
        else:
            columns = tuple(str(column) for column in frame.columns)
            schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
            schema_hash = _hash_bytes(json.dumps(schema, ensure_ascii=False).encode())
            canonical = frame.sort_index(axis=1).to_csv(index=False, lineterminator="\n").encode()
            content_hash = _hash_bytes(canonical)
            row_count = len(frame)
            if event_column is not None:
                if event_column not in frame.columns:
                    raise ValueError(f"missing event column in {logical_name}: {event_column}")
                values = pd.to_datetime(frame[event_column], errors="coerce", utc=True).dropna()
                min_event_time = values.min().isoformat() if not values.empty else None
                max_event_time = values.max().isoformat() if not values.empty else None
            else:
                min_event_time = max_event_time = None
    except Exception:
        target.unlink(missing_ok=True)
        raise

    return BoundFileSnapshot(
        logical_name=logical_name,
        source_id=source_id,
        relative_path=str(target.relative_to(run_dir)),
        raw_sha256=raw_sha256,
        query_hash=_hash_bytes(json.dumps(query, sort_keys=True, default=str).encode()),
        schema_hash=schema_hash,
        content_hash=content_hash,
        byte_count=target.stat().st_size,
        row_count=row_count,
        columns=columns,
        min_event_time=min_event_time,
        max_event_time=max_event_time,
    )


def build_snapshot_meta(frame: pd.DataFrame, *, source_id: str, query: dict[str, Any],
                        event_column: str = "event_time") -> SnapshotMeta:
    if event_column not in frame.columns:
        raise ValueError(f"missing event column: {event_column}")
    ordered = frame.sort_index(axis=1).sort_values(list(frame.columns), kind="mergesort")
    schema = json.dumps([(c, str(frame[c].dtype)) for c in sorted(frame.columns)], ensure_ascii=False).encode()
    content = ordered.to_csv(index=False, lineterminator="\n").encode()
    values = pd.to_datetime(frame[event_column], errors="coerce").dropna()
    return SnapshotMeta(
        source_id=source_id,
        query_hash=_hash_bytes(json.dumps(query, sort_keys=True, default=str).encode()),
        schema_hash=_hash_bytes(schema),
        content_hash=_hash_bytes(content),
        row_count=len(frame),
        min_event_time=values.min().isoformat() if not values.empty else None,
        max_event_time=values.max().isoformat() if not values.empty else None,
        extracted_at=datetime.now().astimezone().isoformat(),
    )


def write_snapshot_manifest(meta: SnapshotMeta, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable snapshot manifest already exists: {path}")
    payload = json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
