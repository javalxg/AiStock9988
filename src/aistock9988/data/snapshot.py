from __future__ import annotations

import hashlib
import json
import os
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


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
