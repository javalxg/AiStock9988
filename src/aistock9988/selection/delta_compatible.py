"""Deterministic, auditable rules used by the delta-compatible experiment.

These rules are deliberately separate from the production q70 policy.  They
make the historical comparison contract executable without changing the
production baseline's selection semantics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DynamicGateResult:
    factor: str
    threshold: float | None
    active: bool
    sample_count: int
    lower_tail_mean: float | None
    upper_tail_mean: float | None
    lower_quantile: float | None
    upper_quantile: float | None
    reason: str


def compute_dynamic_upper_gate(samples: pd.DataFrame, *, factor: str, label: str = "label_return",
                               minimum_samples: int = 1000, lower_quantile: float = 0.30,
                               upper_quantile: float = 0.70) -> DynamicGateResult:
    """Compute the causal delta gate from already-mature training samples.

    The gate activates only when the lower factor tail has a higher mean label
    than the upper tail.  When there are too few finite samples, it is
    disabled rather than silently using a threshold estimated from an
    incomplete window.  The returned object is intended to be written to the
    model audit/manifest for every retrain.
    """
    required = {factor, label} - set(samples.columns)
    if required:
        raise ValueError(f"dynamic gate samples missing columns: {sorted(required)}")
    values = pd.to_numeric(samples[factor], errors="coerce")
    outcomes = pd.to_numeric(samples[label], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float)) & np.isfinite(outcomes.to_numpy(dtype=float))
    values = values[finite]
    outcomes = outcomes[finite]
    n = len(values)
    if n < minimum_samples:
        return DynamicGateResult(factor, None, False, n, None, None, None, None,
                                 "insufficient_mature_samples")
    lo_cut = float(values.quantile(lower_quantile))
    hi_cut = float(values.quantile(upper_quantile))
    lo_mean = float(outcomes[values <= lo_cut].mean())
    hi_mean = float(outcomes[values >= hi_cut].mean())
    active = lo_mean > hi_mean
    return DynamicGateResult(factor, hi_cut if active else None, active, n,
                             lo_mean, hi_mean, lo_cut, hi_cut,
                             "lower_tail_advantage" if active else "upper_tail_not_worse")


def apply_dynamic_upper_gate(candidates: pd.DataFrame, *, factor: str,
                             threshold: float | None) -> pd.DataFrame:
    """Return candidates passing a frozen upper gate; no missing-value bypass."""
    if factor not in candidates.columns:
        raise ValueError(f"candidate ledger missing dynamic gate factor: {factor}")
    out = candidates.copy()
    values = pd.to_numeric(out[factor], errors="coerce")
    out["dynamic_gate_passed"] = True if threshold is None else values.notna() & (values <= threshold)
    return out


def select_rank_holdings(candidates: pd.DataFrame, previous_codes: set[str], *,
                         max_positions: int = 2, hold_buffer_n: int = 5) -> pd.DataFrame:
    """Keep existing names in the Top-N buffer, then fill remaining Top-N slots."""
    required = {"ts_code", "candidate_rank"} - set(candidates.columns)
    if required:
        raise ValueError(f"rank holding candidates missing columns: {sorted(required)}")
    if max_positions <= 0 or hold_buffer_n < max_positions:
        raise ValueError("max_positions must be positive and no greater than hold_buffer_n")
    ordered = candidates.sort_values(["candidate_rank", "ts_code"], kind="mergesort").copy()
    held = ordered[ordered["ts_code"].astype(str).isin(previous_codes) &
                   (pd.to_numeric(ordered["candidate_rank"], errors="raise") <= hold_buffer_n)]
    chosen = held.head(max_positions)
    if len(chosen) < max_positions:
        remaining = ordered[~ordered["ts_code"].astype(str).isin(set(chosen["ts_code"].astype(str)))]
        chosen = pd.concat([chosen, remaining.head(max_positions - len(chosen))], ignore_index=True)
    return chosen.head(max_positions).reset_index(drop=True)


def weak_breadth_cash_fraction(*, breadth: float, minimum: float,
                               candidate_count: int, configured_fraction: float) -> float:
    """Apply the historical 50% cap only to a weak-breadth single candidate."""
    if not 0 < configured_fraction <= 1:
        raise ValueError("configured_fraction must be in (0, 1]")
    if not np.isfinite(breadth) or not 0 <= breadth <= 1:
        raise ValueError("breadth must be finite and in [0, 1]")
    return configured_fraction if breadth < minimum and candidate_count == 1 else 1.0
