"""Stock-session data eligibility derived from explicit source policies."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from ..configuration import StrategyConfig


STAGES = ("selection", "training", "execution")


def build_data_availability_ledger(
    grid: pd.DataFrame,
    source_presence: Mapping[str, pd.Series],
    strategy: StrategyConfig,
) -> pd.DataFrame:
    """Build one auditable eligibility row for every configured stock/session."""
    required = {
        stage: tuple(str(source) for source in strategy.data_policy["dense_required"][stage])
        for stage in STAGES
    }
    sparse = tuple(str(source) for source in strategy.data_policy.get("sparse_event", ()))
    optional = tuple(str(source) for source in strategy.data_policy.get("optional_enrichment", ()))
    configured = set().union(*required.values(), sparse, optional)
    unknown = sorted(configured - set(source_presence))
    if unknown:
        raise ValueError(f"data policy references sources without a registered presence rule: {unknown}")

    ledger = grid[["trade_date", "ts_code"]].copy()
    for source in sorted(configured):
        values = source_presence[source]
        if not values.index.equals(grid.index):
            values = values.reindex(grid.index)
        ledger[f"has_{source}"] = values.fillna(False).astype(bool).to_numpy()

    for stage, sources in required.items():
        missing_column = f"missing_required_{stage}"
        ledger[missing_column] = _missing_sources(ledger, sources)
        ledger[f"{stage}_data_eligible"] = ledger[missing_column].eq("")

    ledger["missing_optional"] = _missing_sources(ledger, optional)
    ledger["selection_data_rejection_reason"] = ledger["missing_required_selection"].map(
        lambda value: f"MISSING_REQUIRED_DATA:{value}" if value else ""
    )
    ledger["training_data_rejection_reason"] = ledger["missing_required_training"].map(
        lambda value: f"MISSING_REQUIRED_DATA:{value}" if value else ""
    )
    ledger["execution_data_rejection_reason"] = ledger["missing_required_execution"].map(
        lambda value: f"MISSING_REQUIRED_DATA:{value}" if value else ""
    )
    return ledger.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)


def filter_eligible_stock_sessions(
    frame: pd.DataFrame,
    availability: pd.DataFrame,
    *,
    stage: str,
    date_column: str = "asof",
) -> pd.DataFrame:
    """Filter selection or training rows through the shared eligibility contract."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    flag = f"{stage}_data_eligible"
    keys = availability[["trade_date", "ts_code", flag]].rename(columns={"trade_date": date_column})
    merged = frame.merge(keys, on=[date_column, "ts_code"], how="left", validate="many_to_one")
    if merged[flag].isna().any():
        raise ValueError("input contains stock/session keys absent from the availability ledger")
    return merged.loc[merged[flag].astype(bool)].reset_index(drop=True)


def _missing_sources(ledger: pd.DataFrame, sources: Sequence[str]) -> pd.Series:
    if not sources:
        return pd.Series("", index=ledger.index, dtype="object")
    missing = pd.Series("", index=ledger.index, dtype="object")
    for source in sources:
        absent = ~ledger[f"has_{source}"].astype(bool)
        missing.loc[absent] = missing.loc[absent].map(
            lambda current, name=source: f"{current}|{name}" if current else name
        )
    return missing


__all__ = ["STAGES", "build_data_availability_ledger", "filter_eligible_stock_sessions"]
