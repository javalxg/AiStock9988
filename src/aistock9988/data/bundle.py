"""Run-scoped, coverage-preserving data bundle for V3 backtests."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..configuration import StrategyConfig
from ..planning import RunPlan
from ..time.session import session_close, session_open
from .availability import build_data_availability_ledger
from .corporate_actions_source import load_corporate_actions
from .quantdb import readonly_connection
from .shared_cache import SharedSnapshotStore


@dataclass(frozen=True)
class DataBundle:
    bundle_id: str
    calendar: pd.DataFrame
    universe: pd.DataFrame
    availability: pd.DataFrame
    execution: pd.DataFrame
    corporate_actions: pd.DataFrame
    enrichments: dict[str, pd.DataFrame]
    manifest: dict[str, Any]


def load_trading_calendar(start: str, end: str) -> pd.DataFrame:
    with readonly_connection() as connection:
        frame = pd.read_sql_query(
            "SELECT cal_date, pretrade_date, update_time FROM trade_cal_ts "
            "WHERE exchange='SSE' AND is_open=1 AND cal_date BETWEEN %s AND %s ORDER BY cal_date",
            connection,
            params=(start, end),
        )
    if frame.empty:
        raise ValueError("trade_cal_ts returned no sessions")
    frame["session"] = pd.to_datetime(frame.pop("cal_date"), errors="raise", utc=True).dt.normalize()
    if frame["session"].duplicated().any():
        raise ValueError("trade_cal_ts contains duplicate SSE sessions")
    return frame[["session", "pretrade_date", "update_time"]]


def build_data_bundle(plan: RunPlan, strategy: StrategyConfig, output_dir: Path) -> DataBundle:
    output_dir = output_dir.resolve()
    raw_dir = output_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    project_root = Path(__file__).resolve().parents[3]
    codes_path = (project_root / str(strategy.universe["codes_file"])).resolve()
    codes_frame = pd.read_csv(codes_path, dtype={"ts_code": str})
    if "ts_code" not in codes_frame or codes_frame["ts_code"].duplicated().any():
        raise ValueError("universe codes file must contain unique ts_code values")
    excluded_suffixes = tuple(str(value).upper() for value in strategy.universe.get("exclude_suffixes", ()))
    codes = sorted(
        code for code in codes_frame["ts_code"].astype(str).str.upper().unique()
        if not any(code.endswith(suffix) for suffix in excluded_suffixes)
    )
    if not codes:
        raise ValueError("configured universe is empty")

    configured_sources = set(plan.required_sources)
    raw_names = _snapshot_names(configured_sources)
    cache = SharedSnapshotStore()
    cache_key = cache.key(
        feature_start=plan.feature_start,
        execution_end=plan.execution_end,
        required_sources=plan.required_sources,
        codes=codes,
    )
    cached_frames = cache.load(cache_key, raw_names)
    if cached_frames is not None:
        raw_frames = cached_frames
        cache_status = "hit"
    else:
        calendar = load_trading_calendar(plan.feature_start, plan.execution_end)
        security_master = _read_sql(
            "SELECT ts_code, name, market, list_status, list_date, delist_date, update_time "
            "FROM stock_basic_ts ORDER BY ts_code"
        )
        market = _read_chunked(
            "SELECT ts_code, trade_date, open, high, low, close, pre_close, pct_chg, vol, amount, update_time "
            "FROM market_daily_ts WHERE source='daily' AND trade_date BETWEEN %s AND %s ORDER BY trade_date, ts_code",
            plan.feature_start, plan.execution_end,
        )
        adjustment = _read_chunked(
            "SELECT ts_code, trade_date, adj_factor, update_time FROM adj_factor_ts "
            "WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date, ts_code",
            plan.feature_start, plan.execution_end,
        )
        limits = _read_chunked(
            "SELECT ts_code, trade_date, up_limit, down_limit, update_time FROM stk_limit_ts "
            "WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date, ts_code",
            plan.feature_start, plan.execution_end,
        )
        st = _read_chunked(
            "SELECT ts_code, trade_date, name, st_type, update_time FROM stock_st_ts "
            "WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date, ts_code",
            plan.feature_start, plan.execution_end,
        )
        suspensions = _read_sql(
            "SELECT ts_code, suspend_date, resume_date, ann_date, reason_type, update_time FROM suspend_d_ts "
            "WHERE suspend_date <= %s AND (resume_date IS NULL OR resume_date >= %s) ORDER BY suspend_date, ts_code",
            params=(plan.execution_end, plan.feature_start),
        )
        auction_zero = _read_chunked(
            "SELECT ts_code, trade_date, vol, amount, update_time FROM stk_auction_o_ts "
            "WHERE trade_date BETWEEN %s AND %s AND (COALESCE(vol,0)=0 OR COALESCE(amount,0)=0) "
            "ORDER BY trade_date, ts_code",
            plan.feature_start, plan.execution_end,
        )
        corporate_actions = load_corporate_actions(plan.feature_start, plan.execution_end, ts_codes=codes)
        # Strategy-specific daily sources are snapshotted with the core panel so
        # a rerun never reads a second, later database state.
        enrichments = _load_enrichments(plan.feature_start, plan.execution_end, configured_sources)
        raw_frames = {
            "calendar": calendar,
            "security_master": security_master,
            "market_daily": market,
            "adjustment": adjustment,
            "daily_limits": limits,
            "pit_st": st,
            "suspensions": suspensions,
            "auction_zero": auction_zero,
            "corporate_actions": corporate_actions,
            "universe_codes": pd.DataFrame({"ts_code": codes}),
            **enrichments,
        }
        cache.write(
            cache_key,
            raw_frames,
            metadata={
                "feature_start": plan.feature_start,
                "execution_end": plan.execution_end,
                "required_sources": list(plan.required_sources),
                "strategy_id": plan.strategy_id,
                "configured_codes": len(codes),
            },
        )
        cache_status = "miss_written"
    calendar = raw_frames["calendar"]
    security_master = raw_frames["security_master"]
    market = raw_frames["market_daily"]
    adjustment = raw_frames["adjustment"]
    limits = raw_frames["daily_limits"]
    st = raw_frames["pit_st"]
    suspensions = raw_frames["suspensions"]
    auction_zero = raw_frames["auction_zero"]
    corporate_actions = raw_frames["corporate_actions"]
    enrichments = {name: raw_frames[name] for name in raw_names if name not in {
        "calendar", "security_master", "market_daily", "adjustment", "daily_limits",
        "pit_st", "suspensions", "auction_zero", "corporate_actions", "universe_codes",
    }}
    link_methods = cache.materialize(cache_key, raw_names, raw_dir)
    amount_unit_audit = _audit_amount_units(market, strategy)
    hashes: dict[str, str] = {}
    rows: dict[str, int] = {}
    for name, frame in raw_frames.items():
        target = raw_dir / f"{name}.parquet"
        hashes[name] = _sha256(target)
        rows[name] = len(frame)

    universe, availability, execution, coverage = _build_ledgers(
        calendar=calendar,
        codes=codes,
        security_master=security_master,
        market=market,
        adjustment=adjustment,
        limits=limits,
        st=st,
        suspensions=suspensions,
        auction_zero=auction_zero,
        corporate_actions=corporate_actions,
        enrichments=enrichments,
        strategy=strategy,
    )
    bundle_payload = {
        "strategy_hash": plan.strategy_hash,
        "feature_start": plan.feature_start,
        "execution_end": plan.execution_end,
        "sources": hashes,
    }
    bundle_id = hashlib.sha256(
        json.dumps(bundle_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    universe["bundle_id"] = bundle_id
    availability["bundle_id"] = bundle_id
    execution["bundle_id"] = bundle_id
    manifest = {
        "bundle_id": bundle_id,
        "source_id": "quant_db",
        "snapshot_semantics": "source_ingested_time is provenance; event/available time controls PIT",
        "credentials_persisted": False,
        "feature_start": plan.feature_start,
        "execution_end": plan.execution_end,
        "configured_codes": len(codes_frame),
        "eligible_suffix_codes": len(codes),
        "source_rows": rows,
        "source_sha256": hashes,
        "shared_snapshot": {
            "cache_key": cache_key,
            "cache_status": cache_status,
            "cache_root": str(cache.root),
            "materialization": link_methods,
        },
        "data_policy": strategy.to_dict()["data_policy"],
        "source_availability_policy": strategy.data_policy.get("source_availability", {}),
        "source_update_time_summary": {
            name: _update_time_summary(frame)
            for name, frame in enrichments.items()
            if "update_time" in frame.columns
        },
        "amount_unit_audit": amount_unit_audit,
        "coverage": coverage,
        "enrichment_rows": {name: len(frame) for name, frame in enrichments.items()},
    }
    return DataBundle(bundle_id, calendar, universe, availability, execution, corporate_actions, enrichments, manifest)


def _audit_amount_units(market: pd.DataFrame, strategy: StrategyConfig) -> dict[str, Any]:
    """Record an empirical amount/volume/price unit check in every bundle."""
    execution = strategy.execution
    multiplier = float(execution.get("amount_unit_multiplier", 1.0))
    declared_unit = str(execution.get("amount_unit", "unspecified"))
    sample = market.copy()
    sample["trade_date"] = pd.to_datetime(sample["trade_date"], errors="coerce").dt.date
    for column in ("close", "vol", "amount"):
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    valid = sample[
        sample["close"].gt(0) & sample["vol"].gt(0) & sample["amount"].gt(0)
    ].copy()
    if valid.empty:
        raise ValueError("amount unit audit has no positive close/vol/amount rows")
    # Tushare daily volume is in lots of 100 shares; the ratio should be near 1
    # when amount is stored in thousand RMB and converted with multiplier=1000.
    ratio = valid["amount"] * multiplier / (valid["vol"] * 100.0 * valid["close"])
    median_ratio = float(ratio.median())
    audit = {
        "declared_unit": declared_unit,
        "multiplier_to_rmb": multiplier,
        "volume_unit_assumption": "100_shares_per_lot",
        "sample_trade_date_min": str(valid["trade_date"].min()),
        "sample_trade_date_max": str(valid["trade_date"].max()),
        "sample_rows": int(len(valid)),
        "price_volume_amount_ratio_median": median_ratio,
        "price_volume_amount_ratio_p01": float(ratio.quantile(0.01)),
        "price_volume_amount_ratio_p99": float(ratio.quantile(0.99)),
    }
    if str(strategy.identity.get("research_status", "")) == "forward_only":
        if not 0.5 <= median_ratio <= 2.0:
            raise ValueError(
                "forward_only amount unit audit failed: "
                f"median price/volume/amount ratio={median_ratio:.4f}"
            )
    return audit


def _update_time_summary(frame: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_datetime(frame["update_time"], utc=True, errors="coerce").dropna()
    if values.empty:
        return {"rows_with_update_time": 0}
    return {
        "rows_with_update_time": int(len(values)),
        "min": values.min().isoformat(),
        "max": values.max().isoformat(),
    }


def _load_enrichments(start: str, end: str, configured_sources: set[str]) -> dict[str, pd.DataFrame]:
    """Load fixed, auditable daily enrichment sources for rule research."""
    out: dict[str, pd.DataFrame] = {}
    if "daily_basic_ts" in configured_sources:
        out["daily_basic"] = _read_chunked(
            "SELECT ts_code, trade_date, turnover_rate_f, volume_ratio, total_mv, update_time "
            "FROM daily_basic_ts WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date, ts_code",
            start, end,
        )
    if "moneyflow_ts" in configured_sources:
        out["moneyflow"] = _read_chunked(
            "SELECT ts_code, trade_date, buy_md_amount, sell_md_amount, buy_lg_amount, sell_lg_amount, "
            "buy_elg_amount, sell_elg_amount, net_mf_amount, update_time FROM moneyflow_ts "
            "WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date, ts_code",
            start, end,
        )
    if "index_daily_ts" in configured_sources:
        out["index_daily"] = _read_chunked(
            "SELECT ts_code, trade_date, close, update_time FROM index_daily_ts "
            "WHERE ts_code IN ('000001.SH','000905.SH') AND trade_date BETWEEN %s AND %s "
            "ORDER BY trade_date, ts_code",
            start, end,
        )
    if "index_member_all_ts" in configured_sources:
        # Membership is an interval relation rather than a daily fact table.
        # Freeze every interval that can be active during the requested window;
        # the feature provider resolves ``in_date <= T < out_date`` locally.
        # Do not use the date-chunked reader here: interval rows can overlap
        # multiple chunks and would otherwise be duplicated in the snapshot.
        out["index_member_all"] = _read_sql(
            "SELECT index_code, con_code, in_date, out_date, update_time "
            "FROM index_member_all_ts "
            "WHERE in_date <= %s AND (out_date IS NULL OR out_date >= %s) "
            "ORDER BY con_code, in_date, index_code",
            params=(end, start),
        )
    return out


def _snapshot_names(configured_sources: set[str]) -> list[str]:
    """Return the deterministic file set for a shared source snapshot."""
    names = {
        "calendar", "security_master", "market_daily", "adjustment",
        "daily_limits", "pit_st", "suspensions", "auction_zero",
        "corporate_actions", "universe_codes",
    }
    if "daily_basic_ts" in configured_sources:
        names.add("daily_basic")
    if "moneyflow_ts" in configured_sources:
        names.add("moneyflow")
    if "index_daily_ts" in configured_sources:
        names.add("index_daily")
    if "index_member_all_ts" in configured_sources:
        names.add("index_member_all")
    return sorted(names)


def _build_ledgers(
    *,
    calendar: pd.DataFrame,
    codes: list[str],
    security_master: pd.DataFrame,
    market: pd.DataFrame,
    adjustment: pd.DataFrame,
    limits: pd.DataFrame,
    st: pd.DataFrame,
    suspensions: pd.DataFrame,
    auction_zero: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    enrichments: dict[str, pd.DataFrame],
    strategy: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sessions = pd.DatetimeIndex(calendar["session"])
    grid_index = pd.MultiIndex.from_product([sessions, codes], names=["trade_date", "ts_code"])
    grid = grid_index.to_frame(index=False)

    master = security_master.copy()
    master["ts_code"] = master["ts_code"].astype(str).str.upper()
    master = master[master["ts_code"].isin(codes)].copy()
    if master["ts_code"].duplicated().any():
        raise ValueError("stock_basic_ts contains duplicate ts_code values")
    master["list_date"] = pd.to_datetime(master["list_date"], errors="coerce", utc=True).dt.normalize()
    master["delist_date"] = pd.to_datetime(master["delist_date"], errors="coerce", utc=True).dt.normalize()
    grid = grid.merge(master[["ts_code", "name", "market", "list_date", "delist_date"]], on="ts_code", how="left", validate="many_to_one")
    listed = grid["list_date"].notna() & (grid["list_date"] <= grid["trade_date"])
    not_delisted = grid["delist_date"].isna() | (grid["trade_date"] < grid["delist_date"])

    st_keys = _keys(st, "trade_date")
    grid["pit_st"] = pd.MultiIndex.from_frame(grid[["trade_date", "ts_code"]]).isin(st_keys)
    grid["universe_pass"] = listed & not_delisted & ~grid["pit_st"]
    grid["universe_rejection_reason"] = np.select(
        [~listed, ~not_delisted, grid["pit_st"]],
        ["NOT_LISTED", "DELISTED", "PIT_ST"],
        default="",
    )
    universe = grid[[
        "trade_date", "ts_code", "name", "market", "list_date", "delist_date",
        "pit_st", "universe_pass", "universe_rejection_reason",
    ]].rename(columns={"trade_date": "asof"})

    market = _normalize_daily(market, codes, "market_daily_ts")
    adjustment = _normalize_daily(adjustment, codes, "adj_factor_ts")
    limits = _normalize_daily(limits, codes, "stk_limit_ts")
    execution = grid.merge(market, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    execution = execution.merge(
        adjustment.rename(columns={"update_time": "adj_update_time"}),
        on=["trade_date", "ts_code"], how="left", validate="one_to_one",
    )
    execution = execution.merge(
        limits.rename(columns={"update_time": "limit_update_time"}),
        on=["trade_date", "ts_code"], how="left", validate="one_to_one",
    )
    suspension_keys = _expand_suspensions(suspensions, sessions, set(codes))
    auction_keys = _keys(auction_zero, "trade_date")
    keys = pd.MultiIndex.from_frame(execution[["trade_date", "ts_code"]])
    execution["suspension_evidence"] = keys.isin(suspension_keys)
    execution["auction_zero_evidence"] = keys.isin(auction_keys)

    has_market = execution["open"].notna()
    numeric_market = execution[["open", "high", "low", "close", "amount"]].apply(pd.to_numeric, errors="coerce")
    valid_market = (
        has_market
        & numeric_market[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & numeric_market["amount"].notna()
        & numeric_market["amount"].ge(0)
    )
    zero_volume = has_market & (numeric_market["amount"].fillna(0) <= 0)
    valid_adj = pd.to_numeric(execution["adj_factor"], errors="coerce").gt(0)
    up = pd.to_numeric(execution["up_limit"], errors="coerce")
    down = pd.to_numeric(execution["down_limit"], errors="coerce")
    valid_limit = up.gt(0) & down.gt(0) & up.lt(99999) & down.lt(99999)

    action_keys = _keys(corporate_actions, "ex_date")
    source_presence = {
        "market_daily_ts": valid_market,
        "adj_factor_ts": valid_adj,
        "stk_limit_ts": valid_limit,
        "stock_st_ts": pd.Series(keys.isin(st_keys), index=execution.index),
        "suspend_d_ts": execution["suspension_evidence"],
        "stk_auction_o_ts": execution["auction_zero_evidence"],
        "corporate_actions": pd.Series(keys.isin(action_keys), index=execution.index),
    }
    for source_name, enrichment in enrichments.items():
        if source_name == "index_daily":
            dates = pd.to_datetime(enrichment.get("trade_date", pd.Series(dtype="datetime64[ns]")), utc=True, errors="coerce").dt.normalize().dropna().unique()
            source_presence["index_daily_ts"] = pd.Series(execution["trade_date"].isin(dates).to_numpy(), index=execution.index)
        elif source_name == "index_member_all":
            # Membership is sparse.  This flag records security coverage;
            # the feature provider performs the date-accurate in/out resolve.
            # Avoid expanding every interval over the full stock/session grid.
            known_codes = set(enrichment["con_code"].astype(str).str.upper().dropna())
            source_presence["index_member_all_ts"] = execution["ts_code"].isin(known_codes)
        else:
            source_presence[f"{source_name}_ts"] = pd.Series(
                keys.isin(_keys(enrichment, "trade_date")), index=execution.index
            )
    availability = build_data_availability_ledger(execution, source_presence, strategy)
    execution = execution.merge(
        availability,
        on=["trade_date", "ts_code"],
        how="left",
        validate="one_to_one",
    )

    execution["execution_status"] = "TRADABLE"
    execution.loc[valid_market & valid_adj & valid_limit & (numeric_market["open"] >= up), "execution_status"] = "LIMIT_UP"
    execution.loc[valid_market & valid_adj & valid_limit & (numeric_market["open"] <= down), "execution_status"] = "LIMIT_DOWN"
    execution.loc[zero_volume | execution["auction_zero_evidence"], "execution_status"] = "ZERO_VOLUME"
    execution.loc[execution["suspension_evidence"], "execution_status"] = "SUSPENDED"
    proven_nontrade = execution["suspension_evidence"] | execution["auction_zero_evidence"] | zero_volume
    execution.loc[~execution["execution_data_eligible"] & ~proven_nontrade, "execution_status"] = "MISSING_REQUIRED_DATA"
    execution.loc[~execution["universe_pass"], "execution_status"] = "OUT_OF_UNIVERSE"

    for column in ("open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount", "adj_factor", "up_limit", "down_limit"):
        execution[column] = pd.to_numeric(execution[column], errors="coerce")
    execution["raw_open"] = execution["open"]
    execution["raw_high"] = execution["high"]
    execution["raw_low"] = execution["low"]
    execution["raw_close"] = execution["close"]
    for price in ("open", "high", "low", "close"):
        execution[f"economic_{price}"] = execution[price] * execution["adj_factor"]
    execution = execution.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    _fill_proven_nontrade_marks(execution)
    execution["open_available_time"] = execution["trade_date"].map(session_open)
    execution["close_available_time"] = execution["trade_date"].map(session_close)
    execution["available_time"] = execution["close_available_time"]
    execution["adv20_amount"] = execution.groupby("ts_code", sort=False)["amount"].transform(
        lambda values: values.rolling(20, min_periods=20).median()
    )
    availability_columns = [
        column for column in availability.columns
        if column not in {"trade_date", "ts_code"}
    ]
    keep = [
        "trade_date", "ts_code", "universe_pass", "execution_status", "suspension_evidence",
        "auction_zero_evidence", "raw_open", "raw_high", "raw_low", "raw_close",
        "pct_chg",
        "economic_open", "economic_high", "economic_low", "economic_close", "adj_factor",
        "up_limit", "down_limit", "amount", "adv20_amount", "open_available_time",
        "close_available_time", "available_time",
        *availability_columns,
    ]
    execution = execution[keep].sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)
    eligible = execution[execution["universe_pass"]]
    coverage = {
        "expected_eligible_rows": int(len(eligible)),
        "status_counts": {str(key): int(value) for key, value in eligible["execution_status"].value_counts().sort_index().items()},
        "missing_required_data_rows": int((eligible["execution_status"] == "MISSING_REQUIRED_DATA").sum()),
        "data_eligibility": {
            stage: {
                "eligible_rows": int(eligible[f"{stage}_data_eligible"].sum()),
                "excluded_rows": int((~eligible[f"{stage}_data_eligible"]).sum()),
                "missing_source_counts": _missing_source_counts(eligible[f"missing_required_{stage}"]),
            }
            for stage in ("selection", "training", "execution")
        },
        "optional_missing_source_counts": _missing_source_counts(eligible["missing_optional"]),
        "duplicate_execution_keys": int(execution.duplicated(["trade_date", "ts_code"]).sum()),
    }
    return (
        universe.sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True),
        availability,
        execution,
        coverage,
    )


def _normalize_daily(frame: pd.DataFrame, codes: list[str], source: str) -> pd.DataFrame:
    out = frame.copy()
    out["ts_code"] = out["ts_code"].astype(str).str.upper()
    out = out[out["ts_code"].isin(codes)].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="raise", utc=True).dt.normalize()
    if out.duplicated(["trade_date", "ts_code"]).any():
        examples = out[out.duplicated(["trade_date", "ts_code"], keep=False)].head().to_dict("records")
        raise ValueError(f"{source} contains duplicate security/session keys: {examples}")
    return out


def _fill_proven_nontrade_marks(execution: pd.DataFrame) -> None:
    proven = execution["execution_status"].isin(["SUSPENDED", "ZERO_VOLUME"])
    mark_columns = [
        "raw_open", "raw_high", "raw_low", "raw_close",
        "pct_chg",
        "economic_open", "economic_high", "economic_low", "economic_close", "adj_factor",
    ]
    for column in mark_columns:
        carried = execution.groupby("ts_code", sort=False)[column].ffill()
        execution.loc[proven & execution[column].isna(), column] = carried[proven & execution[column].isna()]
    execution.loc[proven & execution["amount"].isna(), "amount"] = 0.0


def _keys(frame: pd.DataFrame, date_column: str) -> pd.MultiIndex:
    if frame.empty:
        return pd.MultiIndex.from_arrays([[], []], names=["trade_date", "ts_code"])
    dates = pd.to_datetime(frame[date_column], errors="raise", utc=True).dt.normalize()
    return pd.MultiIndex.from_arrays([dates, frame["ts_code"].astype(str).str.upper()], names=["trade_date", "ts_code"]).unique()


def _missing_source_counts(values: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values.astype(str):
        for source in filter(None, value.split("|")):
            counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _expand_suspensions(frame: pd.DataFrame, sessions: pd.DatetimeIndex, codes: set[str]) -> pd.MultiIndex:
    rows: list[tuple[pd.Timestamp, str]] = []
    for item in frame.to_dict("records"):
        code = str(item["ts_code"]).upper()
        if code not in codes:
            continue
        start = pd.Timestamp(item["suspend_date"], tz="UTC")
        end = pd.Timestamp(item["resume_date"], tz="UTC") if pd.notna(item.get("resume_date")) else sessions[-1]
        rows.extend((day, code) for day in sessions[(sessions >= start) & (sessions < end)])
    if not rows:
        return pd.MultiIndex.from_arrays([[], []], names=["trade_date", "ts_code"])
    return pd.MultiIndex.from_tuples(rows, names=["trade_date", "ts_code"]).unique()


def _read_chunked(query: str, start: str, end: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    cursor = pd.Timestamp(start)
    terminal = pd.Timestamp(end)
    while cursor <= terminal:
        chunk_end = min(cursor + pd.DateOffset(months=3) - pd.Timedelta(days=1), terminal)
        parts.append(_read_sql(query, params=(str(cursor.date()), str(chunk_end.date()))))
        cursor = chunk_end + pd.Timedelta(days=1)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _read_sql(query: str, params: tuple[object, ...] = ()) -> pd.DataFrame:
    with readonly_connection() as connection:
        return pd.read_sql_query(query, connection, params=params)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["DataBundle", "load_trading_calendar", "build_data_bundle"]
