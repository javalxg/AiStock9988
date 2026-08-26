"""Point-in-time resolution of historical industry membership."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..time.session import parse_source_time


@dataclass(frozen=True)
class IndustryResolutionAudit:
    signal_date: str
    universe_count: int
    covered_count: int
    coverage_ratio: float
    conflict_security_count: int
    active_membership_count: int


REQUIRED_COLUMNS = {"index_code", "con_code", "in_date", "out_date"}


def resolve_industry_map(membership: pd.DataFrame, *, signal_date: object,
                         decision_time: object | None = None,
                         universe_codes: list[str] | None = None) -> tuple[dict[str, str], IndustryResolutionAudit]:
    """Resolve one signal-date map using only historical active memberships.

    When a security belongs to several active industry indices, the newest
    ``in_date`` wins, then the lexicographically smallest ``index_code``.
    ``update_time`` is treated as source availability when present.
    """
    missing = REQUIRED_COLUMNS - set(membership.columns)
    if missing:
        raise ValueError(f"industry membership missing columns: {sorted(missing)}")
    day = pd.Timestamp(signal_date)
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    day = day.normalize()
    frame = membership.copy()
    frame["con_code"] = frame["con_code"].astype(str)
    frame["index_code"] = frame["index_code"].astype(str)
    frame["in_date"] = pd.to_datetime(frame["in_date"], errors="coerce").dt.normalize()
    frame["out_date"] = pd.to_datetime(frame["out_date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["con_code", "index_code", "in_date"])
    active = frame[(frame["in_date"] <= day) & (frame["out_date"].isna() | (frame["out_date"] > day))].copy()
    if decision_time is not None and "update_time" in active.columns:
        cutoff = pd.Timestamp(decision_time)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        available = parse_source_time(active["update_time"])
        active = active.loc[available <= cutoff].copy()
    universe = set(str(code) for code in universe_codes) if universe_codes is not None else None
    if universe is not None:
        active = active[active["con_code"].isin(universe)].copy()
    if active.empty:
        total = len(universe) if universe is not None else 0
        return {}, IndustryResolutionAudit(str(day.date()), total, 0, 0.0, 0, 0)
    active = active.sort_values(["con_code", "in_date", "index_code"], ascending=[True, False, True], kind="mergesort")
    counts = active.groupby("con_code", sort=False).size()
    chosen = active.drop_duplicates("con_code", keep="first")
    # The relation table's ``name`` is the security/member name in the source
    # feed, not the industry label.  The stable industry identity is index_code.
    mapping = dict(zip(chosen["con_code"], chosen["index_code"].astype(str)))
    universe_count = len(universe) if universe is not None else int(counts.size)
    covered_count = len(mapping)
    return mapping, IndustryResolutionAudit(
        signal_date=str(day.date()),
        universe_count=universe_count,
        covered_count=covered_count,
        coverage_ratio=covered_count / universe_count if universe_count else 0.0,
        conflict_security_count=int((counts > 1).sum()),
        active_membership_count=len(active),
    )
