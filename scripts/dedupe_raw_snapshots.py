#!/usr/bin/env python3
"""Deduplicate immutable experiment parquet snapshots with hard links.

Only byte-identical files are coalesced. The original paths remain available
to each audit bundle, while the filesystem stores one copy of each content
hash. Raw snapshots are the default; ``--all-parquet`` also coalesces exact
duplicates in derived ledgers and backtest outputs. Run without ``--apply``
for a dry-run report.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate(root: Path, *, apply: bool, all_parquet: bool = False) -> dict[str, int]:
    groups: dict[str, list[Path]] = defaultdict(list)
    pattern = "**/*.parquet" if all_parquet else "**/data/raw/*.parquet"
    for path in sorted(root.glob(pattern)):
        if path.is_file() and not path.is_symlink():
            groups[_sha256(path)].append(path)

    duplicate_paths = 0
    reclaimed = 0
    linked = 0
    for paths in groups.values():
        if len(paths) < 2:
            continue
        canonical = paths[0]
        size = canonical.stat().st_size
        for duplicate in paths[1:]:
            duplicate_paths += 1
            reclaimed += size
            if not apply:
                continue
            temporary = duplicate.with_name(f".{duplicate.name}.link-tmp")
            if temporary.exists():
                temporary.unlink()
            os.link(canonical, temporary)
            os.replace(temporary, duplicate)
            # Some filesystems retain the temporary directory entry after an
            # atomic replace of an existing hard link; never leave it behind.
            if temporary.exists():
                temporary.unlink()
            linked += 1
    return {
        "files": sum(len(paths) for paths in groups.values()),
        "unique_contents": len(groups),
        "duplicate_paths": duplicate_paths,
        "reclaimable_bytes": reclaimed,
        "hardlinks_created": linked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="replace duplicates with hard links")
    parser.add_argument(
        "--all-parquet", action="store_true",
        help="include derived ledgers and backtest parquet, not only data/raw",
    )
    args = parser.parse_args()
    result = deduplicate(args.root.resolve(), apply=args.apply, all_parquet=args.all_parquet)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
