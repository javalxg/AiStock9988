"""Deterministic universe filters used by the formal RCQT strategy."""
from __future__ import annotations

import pandas as pd
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def filter_current_st_history(frame: pd.DataFrame, current_st_codes: set[str]) -> pd.DataFrame:
    """Apply the v1 policy: a currently known ST code is excluded on every date.

    The input frame is not mutated.  The caller is responsible for freezing and
    hashing ``current_st_codes`` in the run data manifest.
    """
    if "ts_code" not in frame.columns:
        raise ValueError("universe frame must contain ts_code")
    codes = {str(code).strip().upper() for code in current_st_codes}
    return frame.loc[~frame["ts_code"].astype(str).str.upper().isin(codes)].copy()


def mark_current_st(frame: pd.DataFrame, current_st_codes: set[str]) -> pd.DataFrame:
    """Return a copy with an auditable current-ST exclusion flag."""
    if "ts_code" not in frame.columns:
        raise ValueError("universe frame must contain ts_code")
    codes = {str(code).strip().upper() for code in current_st_codes}
    out = frame.copy()
    out["excluded_current_st"] = out["ts_code"].astype(str).str.upper().isin(codes)
    return out


@dataclass(frozen=True)
class STBlacklistManifest:
    source: str
    extracted_at: str
    codes: tuple[str, ...]
    content_hash: str

    @classmethod
    def build(cls, codes: set[str], *, source: str, extracted_at: str) -> "STBlacklistManifest":
        normalized = tuple(sorted({str(c).strip().upper() for c in codes}))
        payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode()
        return cls(source, extracted_at, normalized, hashlib.sha256(payload).hexdigest())

    def to_dict(self) -> dict[str, object]:
        return {"source": self.source, "extracted_at": self.extracted_at,
                "code_count": len(self.codes), "codes": list(self.codes),
                "content_hash": self.content_hash}


def build_universe_exclusion_ledger(frame: pd.DataFrame, current_st_codes: set[str], *, asof: str) -> pd.DataFrame:
    """Return only excluded rows, suitable for an auditable per-asof ledger."""
    marked = mark_current_st(frame, current_st_codes)
    out = marked[marked["excluded_current_st"]].copy()
    out.insert(0, "asof", asof)
    out["exclusion_reason"] = "current_st_applied_to_full_history"
    return out[["asof", "ts_code", "exclusion_reason"]].reset_index(drop=True)


def write_st_manifest(manifest: STBlacklistManifest, path: Path) -> None:
    """Write an immutable ST manifest for a run."""
    if path.exists():
        raise FileExistsError(f"manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
