"""Sector-relative feature computation for PIT-resolved industry mappings."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def compute_sector_relative(frame: pd.DataFrame, industry_map: Mapping[str, str],
                            factor_columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Compute per-industry median deviations for one signal-date cross-section.

    ``industry_map`` must already be resolved for the signal date.  Missing
    memberships receive neutral zero values and are not used in medians.
    """
    cols = [c for c in factor_columns if c in frame.columns]
    output = pd.DataFrame(0.0, index=frame.index, columns=[f"{c}_sector_rel" for c in cols])
    if frame.empty or not cols:
        return output
    groups = pd.Series({str(code): industry_map.get(str(code)) for code in frame.index})
    groups = groups.dropna()
    common = frame.index.intersection(groups.index)
    if len(common) == 0 or groups.loc[common].nunique() < 2:
        return output
    values = frame.loc[common, cols].copy()
    values["_industry"] = groups.loc[common].to_numpy()
    for col in cols:
        med = values.groupby("_industry", sort=False)[col].transform("median")
        output.loc[common, f"{col}_sector_rel"] = values[col] - med
    return output

