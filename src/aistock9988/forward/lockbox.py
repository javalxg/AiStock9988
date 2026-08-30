"""Append-only, hash-chained storage for forward shadow ledgers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class ForwardLockbox:
    """Store one immutable partition per observed signal date.

    The lockbox intentionally does not overwrite an existing date. A changed
    source snapshot or strategy config therefore creates a new experiment
    directory instead of silently changing historical forward decisions.
    """

    def __init__(self, root: str | Path, *, experiment_id: str, config_sha256: str):
        self.root = Path(root).resolve()
        self.experiment_id = str(experiment_id)
        self.config_sha256 = str(config_sha256)
        for name in ("score", "candidate", "selection", "manifests"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def append(self, frames: dict[str, pd.DataFrame], *, bundle_id: str, source_end: str,
               metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if set(frames) - {"score", "candidate", "selection"}:
            raise ValueError("lockbox accepts score/candidate/selection ledgers only")
        if set(frames) != {"score", "candidate", "selection"}:
            raise ValueError("lockbox append requires score, candidate, and selection ledgers")
        normalized = {
            name: self._normalize(frame, name)
            for name, frame in frames.items()
        }
        if any(frame.empty for frame in normalized.values()):
            raise ValueError("lockbox ledgers must be non-empty")
        dates = sorted({
            str(pd.Timestamp(value).date())
            for frame in normalized.values()
            for value in frame.get("asof", pd.Series(dtype="datetime64[ns]"))
        })
        previous = self._latest_manifest()
        if previous and (
            previous.get("experiment_id") != self.experiment_id
            or previous.get("config_sha256") != self.config_sha256
        ):
            raise ValueError("forward lockbox config or experiment identity changed; start a new lockbox")
        previous_max = previous.get("max_asof") if previous else None
        if previous_max and any(day <= str(previous_max) for day in dates):
            raise ValueError("forward lockbox is append-only; an existing signal date cannot be rewritten")

        frame_dates = {
            name: set(frame["asof"].dt.strftime("%Y-%m-%d"))
            for name, frame in normalized.items()
        }
        if len({frozenset(value) for value in frame_dates.values()}) != 1:
            raise ValueError("score, candidate, and selection ledgers must cover the same signal dates")
        for day in dates:
            candidate = normalized["candidate"]
            candidate_day = candidate[candidate["asof"].dt.strftime("%Y-%m-%d").eq(day)]
            selection_day = normalized["selection"][normalized["selection"]["asof"].dt.strftime("%Y-%m-%d").eq(day)]
            if not {"candidate_status", "candidate_snapshot_id", "candidate_rank"}.issubset(candidate_day.columns):
                raise ValueError("candidate ledger must contain candidate_status, candidate_rank, and candidate_snapshot_id")
            if len(selection_day) != 1 or "candidate_snapshot_id" not in selection_day:
                raise ValueError("selection ledger must contain exactly one decision and candidate_snapshot_id per day")
            in_view = candidate_day[candidate_day["candidate_status"].astype(str).eq("IN_VIEW")]
            payload = "|".join(
                f"{row.ts_code}:{int(row.candidate_rank)}"
                for row in in_view.sort_values(["candidate_rank", "ts_code"], kind="mergesort").itertuples()
            )
            expected_snapshot = hashlib.sha256(payload.encode()).hexdigest()
            actual_values = selection_day["candidate_snapshot_id"].astype(str).unique()
            if len(actual_values) != 1 or actual_values[0] != expected_snapshot:
                raise ValueError(f"selection snapshot does not match candidate ledger for {day}")

        parts: list[dict[str, Any]] = []
        for name, frame in normalized.items():
            if frame.empty:
                continue
            for day, part in frame.groupby(frame["asof"].dt.strftime("%Y-%m-%d"), sort=True):
                target = self.root / name / f"part-{day}.parquet"
                if target.exists():
                    raise FileExistsError(f"immutable forward partition exists: {target}")
                sort_columns = ["asof"] + (["ts_code"] if "ts_code" in part.columns else [])
                part = part.sort_values(sort_columns, kind="mergesort")
                part.to_parquet(target, index=False)
                parts.append({
                    "kind": name,
                    "asof": day,
                    "path": str(target.relative_to(self.root)),
                    "sha256": _sha256(target),
                    "rows": int(len(part)),
                })
        parts.sort(key=lambda item: (item["asof"], item["kind"]))
        batch_sha = _hash_payload(parts)
        manifest = {
            "schema_version": "forward-lockbox-v1",
            "experiment_id": self.experiment_id,
            "config_sha256": self.config_sha256,
            "bundle_id": str(bundle_id),
            "source_end": str(source_end),
            "previous_manifest_sha256": previous.get("manifest_sha256") if previous else None,
            "batch_sha256": batch_sha,
            "parts": parts,
            "max_asof": max((item["asof"] for item in parts), default=previous_max),
        }
        if metadata:
            manifest.update({str(key): value for key, value in metadata.items()})
        manifest["manifest_sha256"] = _hash_payload(manifest)
        manifest_path = self.root / "manifests" / f"manifest-{batch_sha}.json"
        if manifest_path.exists():
            raise FileExistsError(f"immutable manifest exists: {manifest_path}")
        _write_json(manifest_path, manifest)
        _write_json(self.root / "FORWARD_STATUS.json", {
            "experiment_id": self.experiment_id,
            "config_sha256": self.config_sha256,
            "bundle_id": str(bundle_id),
            "source_end": str(source_end),
            "max_asof": manifest["max_asof"],
            "new_rows": {name: int(len(frame)) for name, frame in normalized.items()},
            "manifest_sha256": manifest["manifest_sha256"],
            "append_only": True,
        }, replace=True)
        return manifest

    def read_day(self, asof: str | pd.Timestamp) -> dict[str, pd.DataFrame]:
        """Verify the hash chain and return the immutable ledgers for one day."""
        day = str(pd.Timestamp(asof).date())
        manifest = self.manifest_for_day(day)
        output: dict[str, pd.DataFrame] = {}
        for part in manifest.get("parts", []):
            if part.get("asof") != day:
                continue
            path = self._safe_part_path(str(part["path"]))
            if not path.exists() or _sha256(path) != part.get("sha256"):
                raise ValueError(f"forward partition hash mismatch: {path}")
            output[str(part["kind"])] = pd.read_parquet(path)
        return output

    def manifest_for_day(self, asof: str | pd.Timestamp) -> dict[str, Any]:
        day = str(pd.Timestamp(asof).date())
        matches = [
            item for item in self._verified_manifests()
            if any(part.get("asof") == day for part in item.get("parts", []))
        ]
        if not matches:
            raise FileNotFoundError(f"no committed forward day: {day}")
        manifest = matches[-1]
        if manifest.get("experiment_id") != self.experiment_id or manifest.get("config_sha256") != self.config_sha256:
            raise ValueError("forward day identity does not match current strategy")
        return manifest

    def _latest_manifest(self) -> dict[str, Any] | None:
        payloads = self._verified_manifests()
        if not payloads:
            return None
        return max(payloads, key=lambda item: str(item.get("max_asof") or ""))

    def _verified_manifests(self) -> list[dict[str, Any]]:
        paths = sorted((self.root / "manifests").glob("manifest-*.json"))
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        by_hash = {str(item.get("manifest_sha256")): item for item in payloads}
        for item in payloads:
            expected = item.get("manifest_sha256")
            body = {key: value for key, value in item.items() if key != "manifest_sha256"}
            if expected != _hash_payload(body):
                raise ValueError("forward manifest self-hash mismatch")
            previous = item.get("previous_manifest_sha256")
            if previous and previous not in by_hash:
                raise ValueError("forward manifest chain is broken")
            if previous and str(item.get("max_asof") or "") <= str(by_hash[previous].get("max_asof") or ""):
                raise ValueError("forward manifest dates are not strictly increasing")
            for part in item.get("parts", []):
                path = self._safe_part_path(str(part["path"]))
                if not path.exists() or _sha256(path) != part.get("sha256"):
                    raise ValueError(f"forward partition hash mismatch: {path}")
        return sorted(payloads, key=lambda item: str(item.get("max_asof") or ""))

    def _safe_part_path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("forward partition path escapes lockbox root") from exc
        return path

    @staticmethod
    def _normalize(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
        out = frame.copy()
        if out.empty:
            return out
        if "asof" not in out:
            raise ValueError("forward ledger missing asof")
        out["asof"] = pd.to_datetime(out["asof"], utc=True, errors="raise").dt.normalize()
        if kind in {"score", "candidate"}:
            if "ts_code" not in out:
                raise ValueError(f"forward {kind} ledger missing ts_code")
            out["ts_code"] = out["ts_code"].astype(str).str.upper()
            if out.duplicated(["asof", "ts_code"]).any():
                raise ValueError("forward ledger contains duplicate signal keys")
        elif "decision_id" in out:
            if out["decision_id"].astype(str).duplicated().any():
                raise ValueError("forward selection ledger contains duplicate decision_id")
        elif out.duplicated(["asof"]).any():
            raise ValueError("forward selection ledger contains duplicate asof")
        return out.reset_index(drop=True)


def _write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"immutable artifact exists: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
