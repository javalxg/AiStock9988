"""Validate the explicitly isolated delta-compatible comparison contract."""
from __future__ import annotations

from pathlib import Path

import yaml

from aistock9988.features.registry import FeatureSet


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/q70_delta_compatible_v1.yaml"


def validate(config_path: Path = CONFIG) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text())
    data = config["data"]
    model = config["model"]
    label = config["label"]
    selection = config["selection"]
    execution = config["execution"]
    upper_gate = selection["dynamic_upper_gate"]
    market_cap = data["market_cap_filter"]
    feature = FeatureSet.from_f0_json(ROOT / "configs/feature_sets/f0_123_columns.json")
    errors: list[str] = []
    if config.get("reference_only") is not True:
        errors.append("delta-compatible contract must be reference_only")
    if config["historical_reference"]["experiment_id"] != "cleanroom_history_contract_t10_q70_source_parity_20260822":
        errors.append("historical reference must remain the canonical source-parity experiment")
    if len(feature.columns) != 123 or data["feature_set"] != feature.id:
        errors.append("delta-compatible contract must use frozen F0=123")
    if data["forbid_old_ledger"] is not True or data["forbid_stage2"] is not True:
        errors.append("old ledger and Stage2 must both be forbidden")
    if market_cap["enabled"] is not False or market_cap["min_value"] is not None:
        errors.append("delta-compatible reference must keep market-cap filtering disabled by default")
    if market_cap["field"] != "circ_mv" or market_cap["unit"] != "万元":
        errors.append("market-cap filter must explicitly use circ_mv in 万元")
    if model["objective"] != "rank:pairwise" or model["max_depth"] != 6 or model["seed"] != 42:
        errors.append("model contract does not match the historical delta reference")
    if label["signal_to_entry_sessions"] != 1 or label["entry_to_exit_sessions"] != 9:
        errors.append("delta-compatible label must use T+1 entry and T+9 holding horizon")
    if label["maturity_lag_sessions"] != 10:
        errors.append("delta-compatible label maturity lag must be 10")
    if execution["accounting_price_basis"] != "economic" or execution["nav_price_basis"] != "economic":
        errors.append("delta-compatible accounting and NAV must explicitly use economic prices")
    if execution["limit_state_basis"] != "raw" or execution["minute_data"] != "none":
        errors.append("delta-compatible limit state must remain raw and minute execution must be explicit none")
    if selection["hold_buffer_n"] != 5 or selection["ranked_holdings"] is not True:
        errors.append("delta-compatible selection must use rank holding with a Top5 buffer")
    if selection["market_breadth_min"] != 0.40 or selection["low_breadth_top_n"] != 2:
        errors.append("delta-compatible selection must use the 40% breadth gate and weak-breadth Top2")
    if selection["low_breadth_require_factor_confirmation"] is not True:
        errors.append("weak breadth must require factor confirmation")
    if upper_gate["factor"] != "dmi_adx_bfq" or upper_gate["enabled"] is not True:
        errors.append("delta-compatible selection must enable the dynamic dmi_adx_bfq upper gate")
    if (upper_gate["configured_upper_bound"] != 0.5 or
            upper_gate["configured_upper_bound_behavior"] != "ignored_when_dynamic_enabled" or
            upper_gate["threshold_source"] != "mature_training_70th_percentile"):
        errors.append("dynamic dmi_adx_bfq gate must record that 0.5 is ignored and P70 is used")
    if (upper_gate["lower_tail_quantile"], upper_gate["upper_tail_quantile"], upper_gate["minimum_mature_samples"]) != (0.30, 0.70, 1000):
        errors.append("dynamic dmi_adx_bfq gate quantiles/sample floor are not the registered values")
    if upper_gate["activation"] != "lower_tail_mean_label_gt_upper_tail_mean_label" or upper_gate["insufficient_samples"] != "disable_gate":
        errors.append("dynamic dmi_adx_bfq gate activation and insufficient-sample behavior are not explicit")
    if selection["weak_breadth_single_candidate_cash_fraction"] != 0.50:
        errors.append("weak-breadth single-candidate cash fraction must be 50%")
    if errors:
        raise ValueError("delta-compatible config invalid: " + "; ".join(errors))
    return {"config": str(config_path.relative_to(ROOT)), "feature_count": len(feature.columns),
            "label_horizon": label["entry_to_exit_sessions"], "maturity_lag": label["maturity_lag_sessions"],
            "accounting_price_basis": execution["accounting_price_basis"], "status": "valid"}


if __name__ == "__main__":
    print(validate())
