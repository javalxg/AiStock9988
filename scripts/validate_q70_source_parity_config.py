"""Validate the isolated q70 source-parity experiment contract."""
from __future__ import annotations

from pathlib import Path

import yaml

from aistock9988.features.registry import FeatureSet


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/q70_source_parity_t10_20260822.yaml"


def validate(config_path: Path = CONFIG) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text())
    data = config["data"]
    model = config["model"]
    execution = config["execution"]
    feature = FeatureSet.from_f0_json(ROOT / "configs/feature_sets/f0_123_columns.json")
    errors: list[str] = []
    if len(feature.columns) != 123:
        errors.append(f"F0 feature count is {len(feature.columns)}, expected 123")
    if str(data["mature_end"]) != "2026-07-31":
        errors.append("formal mature boundary must be 2026-07-31")
    if str(data["raw_end"]) != "2026-08-14":
        errors.append("historical raw end must be 2026-08-14")
    if data["forbid_old_ledger"] is not True or data["forbid_stage2"] is not True:
        errors.append("old ledger and Stage2 must both be forbidden")
    if data["forbid_minute_data"] is not False or execution["minute_data"] != "5min":
        errors.append("productionized reference must use 5min data for intraday execution")
    if execution["accounting_price_basis"] != "raw" or execution["trigger_price_basis"] != "economic":
        errors.append("accounting must use raw prices and risk triggers economic prices")
    if model["objective"] != "rank:pairwise" or model["max_depth"] != 6 or model["seed"] != 42:
        errors.append("XGBoost ranker contract does not match the registered experiment")
    if config["selection"]["final_positions"] != 2 or config["selection"]["candidate_pool"] != "top20":
        errors.append("selection must be weekly Top20 -> Top2")
    if config["label"]["maturity_lag_sessions"] != 10:
        errors.append("label maturity lag must be 10 sessions")
    if errors:
        raise ValueError("q70 source-parity config invalid: " + "; ".join(errors))
    return {"config": str(config_path.relative_to(ROOT)), "feature_count": len(feature.columns),
            "formal_end": str(data["mature_end"]), "reference_end": str(data["raw_end"]), "status": "valid"}


if __name__ == "__main__":
    print(validate())
