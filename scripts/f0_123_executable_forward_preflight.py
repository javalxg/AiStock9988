#!/usr/bin/env python3
"""Check whether a post-observation F0 V2 forward signal can be frozen."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from aistock9988.data.bundle import load_source_max_dates
from aistock9988.data.quantdb import readonly_connection
from aistock9988.features.registry import FeatureSet


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "configs/model_profiles/f0_123_executable_forward_v2.yaml"
DEFAULT_OUTPUT = (
    ROOT / "docs/council_20260828" / "F0_123_EXECUTABLE_V2_FORWARD_PREFLIGHT_20260902.json"
)


def _load_profile(path: Path) -> tuple[dict[str, Any], FeatureSet]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("model profile must be a mapping")
    feature_set = FeatureSet.from_f0_json(ROOT / "configs/feature_sets/f0_123_columns.json")
    expected = {
        "id": feature_set.id,
        "expected_columns": len(feature_set.columns),
        "column_order_hash": feature_set.order_hash,
    }
    if any(config["feature_set"].get(key) != value for key, value in expected.items()):
        raise ValueError("F0 V2 feature contract drift")
    if config["evaluation"].get("forward_only") is not True:
        raise ValueError("F0 V2 must remain forward-only")
    if config["evaluation"].get("parameter_sweep") is not False:
        raise ValueError("F0 V2 parameter sweep must remain disabled")
    if config["selection"].get("factor_gate") != "none":
        raise ValueError("F0 V2 factor gates are forbidden")
    return config, feature_set


def _coverage(start: str, end: str) -> pd.DataFrame:
    with readonly_connection() as connection:
        frame = pd.read_sql_query(
            "SELECT d.trade_date, COUNT(DISTINCT d.ts_code) AS market_rows, "
            "COUNT(DISTINCT a.ts_code) AS adj_rows, "
            "COUNT(DISTINCT l.ts_code) AS limit_rows, "
            "COUNT(DISTINCT f.ts_code) AS f0_rows, "
            "COUNT(DISTINCT b.ts_code) AS basic_rows "
            "FROM market_daily_ts d "
            "LEFT JOIN adj_factor_ts a ON a.trade_date=d.trade_date AND a.ts_code=d.ts_code "
            "LEFT JOIN stk_limit_ts l ON l.trade_date=d.trade_date AND l.ts_code=d.ts_code "
            "LEFT JOIN stock_factor_pro_ts f ON f.trade_date=d.trade_date AND f.ts_code=d.ts_code "
            "LEFT JOIN daily_basic_ts b ON b.trade_date=d.trade_date AND b.ts_code=d.ts_code "
            "WHERE d.source='daily' AND d.trade_date BETWEEN %s AND %s "
            "GROUP BY d.trade_date ORDER BY d.trade_date",
            connection,
            params=(start, end),
        )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], utc=True).dt.normalize()
    return frame


def run(profile_path: Path) -> dict[str, Any]:
    config, feature_set = _load_profile(profile_path)
    readiness = config["data_readiness"]
    required_sources = set(readiness["required_sources"])
    cutoffs = load_source_max_dates(required_sources)
    forward_not_before = str(config["timeline"]["forward_not_before"])
    diagnostic_start = str((pd.Timestamp(forward_not_before) - pd.Timedelta(days=14)).date())
    latest_dense = min(cutoffs.values())
    diagnostic_end = max(
        str(pd.Timestamp(forward_not_before).date()),
        max(cutoffs.values()),
    )
    coverage = _coverage(diagnostic_start, diagnostic_end)
    f0_ratio = float(readiness["minimum_f0_to_market_row_ratio"])
    basic_ratio = float(readiness["minimum_daily_basic_to_market_row_ratio"])
    latest_market_session = None if coverage.empty else coverage["trade_date"].max()
    eligible = coverage[
        coverage["trade_date"].ge(pd.Timestamp(forward_not_before, tz="UTC"))
        & coverage["trade_date"].eq(latest_market_session)
        & coverage["market_rows"].gt(0)
        & coverage["adj_rows"].eq(coverage["market_rows"])
        & coverage["limit_rows"].eq(coverage["market_rows"])
        & coverage["f0_rows"].ge(coverage["market_rows"] * f0_ratio)
        & coverage["basic_rows"].ge(coverage["market_rows"] * basic_ratio)
    ]
    first_eligible = None if eligible.empty else str(eligible.iloc[0]["trade_date"].date())
    ready = first_eligible is not None
    recent = coverage.tail(10).copy()
    recent["trade_date"] = recent["trade_date"].dt.strftime("%Y-%m-%d")
    return {
        "status": "READY_TO_FREEZE_FIRST_FORWARD_SIGNAL" if ready else "WAITING_FOR_COMMON_F0",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "model_profile_id": config["identity"]["model_profile_id"],
        "feature_set_id": feature_set.id,
        "feature_order_hash": feature_set.order_hash,
        "forward_not_before": forward_not_before,
        "observed_through": config["timeline"]["observed_through"],
        "latest_dense_required_cutoff": latest_dense,
        "latest_market_session": (
            None if latest_market_session is None else str(latest_market_session.date())
        ),
        "source_cutoffs": cutoffs,
        "first_eligible_forward_signal": first_eligible,
        "recent_session_coverage": recent.to_dict("records"),
        "credentials_persisted": False,
        "business_data_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(args.profile)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable preflight exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
