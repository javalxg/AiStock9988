"""PIT company-action source for accounting backtests.

``adj_factor`` is intentionally not treated as an accounting ledger.  This
module converts the dividend feed into explicit ex-date actions so the engine
can adjust shares, cost basis, and cash separately from economic prices.
"""
from __future__ import annotations

import pandas as pd
import warnings

from .quantdb import readonly_connection
from ..time.session import parse_source_time, session_open, session_close


def normalize_corporate_actions(frame: pd.DataFrame, *, strict_pit: bool = True) -> pd.DataFrame:
    """Normalize current quant_db dividend columns to the engine contract.

    Values are already per-share in quant_db. Rights issues are not silently
    modeled as free shares; they must be added later with subscription-price
    cash-flow semantics.
    """
    required = {"ts_code", "ex_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"corporate action source missing columns: {sorted(missing)}")
    out = frame.copy()
    source_id_present = "id" in out
    if not source_id_present:
        out["id"] = range(len(out))
    out["ex_date"] = pd.to_datetime(out["ex_date"], errors="coerce", utc=True).dt.normalize()
    if out["ex_date"].isna().any():
        raise ValueError("corporate action source contains invalid ex_date")

    def numeric(name: str) -> pd.Series:
        if name not in out:
            return pd.Series(0.0, index=out.index)
        raw = out[name]
        converted = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"corporate action field {name} contains non-numeric values")
        return converted.fillna(0.0)

    if "cash_div" not in out:
        raise ValueError("corporate action source requires quant_db per-share cash_div")
    cash = numeric("cash_div")
    components = numeric("stk_bo_rate") + numeric("stk_co_rate")
    if "stk_div" in out:
        reported_total = numeric("stk_div")
        total_present = out["stk_div"].notna()
        component_present = pd.Series(False, index=out.index)
        for column in ("stk_bo_rate", "stk_co_rate"):
            if column in out:
                component_present |= out[column].notna()
        mismatch = total_present & component_present & ~reported_total.sub(components).abs().le(1e-8)
        if mismatch.any():
            raise ValueError("stk_div conflicts with stk_bo_rate + stk_co_rate")
        bonus_total = reported_total.where(total_present, components)
    else:
        bonus_total = components
    bonus = bonus_total
    out["cash_dividend"] = cash
    out["split_ratio"] = 1.0 + bonus
    if (out["cash_dividend"] < 0).any() or (out["split_ratio"] <= 0).any():
        raise ValueError("corporate action values are invalid")
    if "div_proc" not in out:
        raise ValueError("corporate action source must include div_proc implementation status")
    status = out["div_proc"].astype(str)
    out = out[status.str.contains("实施", na=False)].copy()
    if "update_time" not in out:
        raise ValueError("corporate action source must include update_time for provenance")
    out["action_type"] = out["div_proc"]
    # update_time is the local ingestion time of the current database
    # snapshot, not the historical time at which the event became known.
    # Prefer Tushare's implementation announcement date, then the initial
    # announcement date. If neither date is available, the snapshot timestamp
    # is used and the PIT check below still rejects a late snapshot.
    announcement = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    for column in ("imp_ann_date", "ann_date"):
        if column in out:
            candidate = pd.to_datetime(out[column], errors="coerce", utc=True)
            announcement = announcement.fillna(candidate.dt.normalize())
    announced = announcement.notna()
    announcement_available = pd.to_datetime(
        announcement.loc[announced].dt.date.map(session_close), errors="coerce", utc=True
    ).reindex(out.index)
    snapshot_available = parse_source_time(out["update_time"])
    out["available_time"] = announcement_available.fillna(snapshot_available)
    if out["available_time"].isna().any():
        raise ValueError("corporate action source has null update_time")
    ex_open = out["ex_date"].map(session_open)
    pit_invalid = out["available_time"] >= ex_open
    if pit_invalid.any():
        if strict_pit:
            raise ValueError("corporate action is not PIT-visible before ex-date market open")
        excluded = int(pit_invalid.sum())
        warnings.warn(
            f"excluding {excluded} non-PIT-visible corporate action(s) from this run",
            RuntimeWarning,
            stacklevel=2,
        )
        out = out.loc[~pit_invalid].copy()
        out.attrs["pit_excluded_rows"] = excluded
    # The provider retains older proposals and marks each row as implemented.
    # Resolve to the latest implementation announcement, then the latest
    # underlying proposal announcement, both of which must precede ex-date.
    def normalized_date(name: str) -> pd.Series:
        if name not in out:
            return pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
        return pd.to_datetime(out[name], errors="coerce", utc=True).dt.normalize()

    out["implementation_ann_date"] = normalized_date("imp_ann_date")
    out["source_ann_date"] = normalized_date("ann_date")
    out["revision_count"] = out.groupby(["ts_code", "ex_date"])["ts_code"].transform("size")
    floor = pd.Timestamp("1900-01-01", tz="UTC")
    out["_implementation_order"] = out["implementation_ann_date"].fillna(out["source_ann_date"]).fillna(floor)
    out["_source_order"] = out["source_ann_date"].fillna(out["_implementation_order"])
    max_implementation = out.groupby(["ts_code", "ex_date"])["_implementation_order"].transform("max")
    latest = out[out["_implementation_order"].eq(max_implementation)].copy()
    max_source = latest.groupby(["ts_code", "ex_date"])["_source_order"].transform("max")
    latest = latest[latest["_source_order"].eq(max_source)].copy()
    # Some provider revisions share the same announcement dates.  Resolve
    # those by the immutable source row/update order before fail-closed.
    if source_id_present:
        latest = latest.sort_values(
            ["ts_code", "ex_date", "update_time", "id"], kind="mergesort", na_position="first"
        ).drop_duplicates(["ts_code", "ex_date"], keep="last")
    economic_terms = latest.groupby(["ts_code", "ex_date"], sort=False)[["split_ratio", "cash_dividend"]].nunique()
    if (economic_terms > 1).any(axis=None):
        raise ValueError("latest PIT corporate action revision still has conflicting economic terms")
    out = latest
    out = out.sort_values(
        ["ts_code", "ex_date", "_implementation_order", "_source_order", "available_time"],
        kind="mergesort", na_position="first",
    )
    out = out.drop_duplicates(["ts_code", "ex_date"], keep="last")
    cols = [
        "ts_code", "ex_date", "split_ratio", "cash_dividend", "available_time", "action_type",
        "source_ann_date", "implementation_ann_date", "revision_count",
    ]
    return out[cols].sort_values(["ex_date", "ts_code"], kind="mergesort").reset_index(drop=True)


def load_corporate_actions(
    start: str,
    end: str,
    *,
    ts_codes: list[str] | None = None,
    strict_pit: bool = True,
) -> pd.DataFrame:
    """Load completed dividend/bonus events from the read-only database.

    The loader discovers the installed dividend table and its available
    columns, because deployments use both ``dividend_ts`` and
    ``stk_dividend_ts`` names.
    """
    candidates = ("dividend_ts", "stk_dividend_ts", "dividend")
    with readonly_connection() as conn:
        table_frame = pd.read_sql_query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema=DATABASE() AND table_name IN (%s,%s,%s)",
            conn, params=candidates,
        )
        table_column = next((column for column in table_frame.columns if column.lower() == "table_name"), None)
        if table_column is None:
            raise RuntimeError("information_schema response has no table_name column")
        tables = table_frame[table_column].tolist()
        if not tables:
            raise RuntimeError("no dividend table found; company actions are required for accounting backtests")
        wanted = ["id", "ts_code", "ex_date", "cash_div", "stk_div", "stk_bo_rate", "stk_co_rate",
                  "div_proc", "ann_date", "imp_ann_date", "update_time"]
        frames = []
        for table in tables:
            column_frame = pd.read_sql_query(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=DATABASE() AND table_name=%s", conn, params=(table,)
            )
            column_name = next((column for column in column_frame.columns if column.lower() == "column_name"), None)
            columns = column_frame[column_name].tolist() if column_name else []
            selected = [c for c in wanted if c in columns]
            if not {"ts_code", "ex_date"}.issubset(selected):
                continue
            params: list[object] = [start, end]
            code_filter = ""
            if ts_codes:
                code_filter = " AND ts_code IN (" + ",".join(["%s"] * len(ts_codes)) + ")"
                params.extend(ts_codes)
            query = ("SELECT " + ", ".join(selected) + " FROM `" + str(table) + "` "
                     "WHERE ex_date >= %s AND ex_date <= %s" + code_filter)
            part = pd.read_sql_query(query, conn, params=tuple(params))
            part["_source_table"] = str(table)
            frames.append(part)
        if not frames:
            raise RuntimeError("dividend tables lack ts_code/ex_date required for PIT company actions")
        frame = pd.concat(frames, ignore_index=True, sort=False)
    return normalize_corporate_actions(frame, strict_pit=strict_pit)
