#!/usr/bin/env python3
"""Audit executable H10 labels against database data and the canonical engine."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aistock9988.backtest.engine import run_backtest
from aistock9988.configuration import StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_source_max_dates, load_trading_calendar
from aistock9988.labeling.executable_path import (
    ExecutablePathLabelProfile,
    build_executable_path_labels,
)
from aistock9988.planning import RunPlan


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRATEGY = ROOT / "configs/strategy/f0_123_full_market_top5_v1.yaml"
DEFAULT_OUTPUT = (
    ROOT / "docs/council_20260828" / "F0_123_EXECUTABLE_LABEL_V2_AUDIT_20260902"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact exists: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _weekly_sessions(sessions: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    start_day = pd.Timestamp(start, tz="UTC")
    end_day = pd.Timestamp(end, tz="UTC")
    selected = sessions[(sessions >= start_day) & (sessions <= end_day)]
    periods = selected.tz_localize(None).to_period("W-FRI")
    grouped = pd.Series(selected, index=selected).groupby(periods, sort=True)
    return [pd.Timestamp(group.iloc[-1]) for _, group in grouped]


def _audit_plan(strategy: StrategyConfig, output: Path) -> tuple[RunPlan, pd.DatetimeIndex]:
    calendar = load_trading_calendar("2025-01-02", "2026-01-16")
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar["session"], utc=True)).normalize()
    required = {
        "trade_cal_ts",
        "stock_basic_ts",
        "market_daily_ts",
        "adj_factor_ts",
        "stk_limit_ts",
        "stock_st_ts",
        "suspend_d_ts",
        "stk_auction_o_ts",
        "corporate_actions",
    }
    plan = RunPlan(
        run_name="f0_123_executable_label_v2_audit",
        output_dir=str(output.resolve()),
        signal_start="2025-01-02",
        signal_end="2025-12-12",
        execution_end="2026-01-16",
        feature_start="2025-01-02",
        signal_sessions=tuple(
            str(day.date()) for day in _weekly_sessions(sessions, "2025-01-02", "2025-12-12")
        ),
        execution_sessions=tuple(str(day.date()) for day in sessions),
        maximum_feature_lookback_sessions=0,
        hold_sessions_from_fill=10,
        strategy_id=strategy.strategy_id,
        strategy_hash=strategy.config_hash,
        mode=strategy.mode,
        required_sources=tuple(sorted(required)),
    )
    return plan, sessions


def _signal_keys(bundle: Any, signal_days: list[pd.Timestamp]) -> pd.DataFrame:
    execution = bundle.execution
    mask = (
        execution["trade_date"].isin(signal_days)
        & execution["universe_pass"]
        & execution["selection_data_eligible"]
    )
    return execution.loc[mask, ["trade_date", "ts_code"]].rename(
        columns={"trade_date": "event_time"}
    ).reset_index(drop=True)


def _sample_ledgers(
    labels: pd.DataFrame,
    execution: pd.DataFrame,
    strategy: StrategyConfig,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    evidence = labels.copy()
    signal_liquidity = execution[["trade_date", "ts_code", "adv20_amount"]].rename(
        columns={"trade_date": "event_time"}
    )
    evidence = evidence.merge(
        signal_liquidity,
        on=["event_time", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    available_days = sorted(evidence["event_time"].drop_duplicates())
    spaced_days = available_days[5::4]
    chosen: list[pd.Series] = []
    desired_type = "STOP_LOSS"
    for day in spaced_days:
        day_rows = evidence[evidence["event_time"].eq(day)].copy()
        day_rows = day_rows[pd.to_numeric(day_rows["adv20_amount"], errors="coerce").gt(0)]
        preferred = day_rows[day_rows["trigger_type"].eq(desired_type)]
        pool = preferred if not preferred.empty else day_rows
        if pool.empty:
            continue
        row = pool.sort_values(["adv20_amount", "ts_code"], ascending=[False, True]).iloc[0]
        chosen.append(row)
        desired_type = "TIME_EXIT" if desired_type == "STOP_LOSS" else "STOP_LOSS"
    sample = pd.DataFrame(chosen).reset_index(drop=True)
    if len(sample) < 8 or sample["trigger_type"].nunique() != 2:
        raise ValueError("parity sample lacks enough stop and time-exit rows")

    candidate = sample.rename(columns={"event_time": "asof"}).copy()
    candidate["candidate_rank"] = 1
    candidate["candidate_status"] = "IN_VIEW"
    candidate["candidate_snapshot_id"] = candidate.apply(
        lambda row: hashlib.sha256(f"{row['ts_code']}:1".encode()).hexdigest(), axis=1
    )
    policy_hash = hashlib.sha256(
        f"executable_label_v2_parity|{strategy.config_hash}".encode()
    ).hexdigest()
    selection = candidate[["asof", "candidate_snapshot_id"]].copy()
    selection["decision_id"] = selection.apply(
        lambda row: hashlib.sha256(
            f"{policy_hash}|{row['asof'].date()}|{row['candidate_snapshot_id']}".encode()
        ).hexdigest(),
        axis=1,
    )
    selection["desired_entries"] = 1
    selection["target_weight_each"] = float(strategy.portfolio["sizing"]["value"])
    selection["primary_rank_end"] = 1
    selection["replacement_rank_end"] = 1
    selection["policy_id"] = "executable_label_v2_parity"
    selection["policy_hash"] = policy_hash
    selection["context_hash"] = selection["asof"].map(
        lambda day: hashlib.sha256(f"{day.date()}|{policy_hash}".encode()).hexdigest()
    )
    sample = sample.merge(selection[["asof", "decision_id"]].rename(columns={"asof": "event_time"}),
                          on="event_time", how="left", validate="one_to_one")
    return {"candidate": candidate, "selection": selection}, sample


def _parity_audit(sample: pd.DataFrame, result: dict[str, pd.DataFrame]) -> dict[str, Any]:
    buys = result["fills"][result["fills"]["side"].eq("BUY")][[
        "decision_id", "ts_code", "trade_date"
    ]].rename(columns={"trade_date": "actual_entry_date"})
    sells = result["fills"][result["fills"]["side"].eq("SELL")][[
        "decision_id", "ts_code", "trade_date", "reason", "economic_return"
    ]].rename(columns={
        "trade_date": "actual_exit_date",
        "reason": "actual_trigger_type",
        "economic_return": "actual_economic_return",
    })
    joined = sample.merge(buys, on=["decision_id", "ts_code"], how="left", validate="one_to_one")
    joined = joined.merge(sells, on=["decision_id", "ts_code"], how="left", validate="one_to_one")
    joined["entry_date_match"] = joined["entry_date"].eq(joined["actual_entry_date"])
    joined["exit_date_match"] = joined["exit_date"].eq(joined["actual_exit_date"])
    joined["trigger_type_match"] = joined["trigger_type"].eq(joined["actual_trigger_type"])
    joined["economic_return_error"] = (
        joined["economic_return"] - joined["actual_economic_return"]
    )
    checks = {
        "all_sample_entries_filled": bool(joined["actual_entry_date"].notna().all()),
        "all_sample_exits_filled": bool(joined["actual_exit_date"].notna().all()),
        "entry_dates_match": bool(joined["entry_date_match"].all()),
        "exit_dates_match": bool(joined["exit_date_match"].all()),
        "trigger_types_match": bool(joined["trigger_type_match"].all()),
        "economic_returns_match": bool(
            joined["economic_return_error"].abs().fillna(np.inf).le(1e-12).all()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sample_rows": int(len(joined)),
        "stop_rows": int(joined["trigger_type"].eq("STOP_LOSS").sum()),
        "time_exit_rows": int(joined["trigger_type"].eq("TIME_EXIT").sum()),
        "maximum_absolute_economic_return_error": float(
            joined["economic_return_error"].abs().max()
        ),
    }


def _distribution(labels: pd.DataFrame) -> dict[str, Any]:
    stop = labels[labels["trigger_type"].eq("STOP_LOSS")]
    time_exit = labels[labels["trigger_type"].eq("TIME_EXIT")]

    def quantiles(values: pd.Series) -> dict[str, float]:
        result = values.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
        return {str(index): float(value) for index, value in result.items()}

    return {
        "all": {
            "rows": int(len(labels)),
            "mean": float(labels["label_return"].mean()),
            "positive_rate": float(labels["label_return"].gt(0).mean()),
            "quantiles": quantiles(labels["label_return"]),
        },
        "stop_loss": {
            "rows": int(len(stop)),
            "rate": float(len(stop) / len(labels)),
            "mean_executable_return": float(stop["label_return"].mean()),
            "mean_crossing_return": float(stop["stop_crossing_return"].mean()),
            "mean_fixed_minus_executable": float((-0.08 - stop["label_return"]).mean()),
            "worse_than_fixed_8pct_rate": float(stop["label_return"].lt(-0.08).mean()),
            "quantiles": quantiles(stop["label_return"]),
        },
        "time_exit": {
            "rows": int(len(time_exit)),
            "mean_executable_return": float(time_exit["label_return"].mean()),
            "positive_rate": float(time_exit["label_return"].gt(0).mean()),
            "quantiles": quantiles(time_exit["label_return"]),
        },
    }


def _write_result(
    output: Path,
    audit: dict[str, Any],
    distribution: dict[str, Any],
    parity: dict[str, Any],
) -> None:
    stop = distribution["stop_loss"]
    time_exit = distribution["time_exit"]
    lines = [
        "# F0-123 Executable Label V2 Audit", "",
        "## Scope", "",
        "- Database-only 2025 weekly full-market signal keys; H10 Base executable label contract.",
        "- Aggregate output only; no labels, prices, fills, predictions, or models persisted.",
        "- This is a label audit, not a strategy-return backtest or parameter search.", "",
        "## Coverage", "",
        f"- Requested rows: `{audit['requested_rows']}`; executable labels: `{audit['label_rows']}`.",
        f"- Entry rejected: `{audit['entry_rejected_rows']}`; unresolved exits: `{audit['unresolved_exit_rows']}`; retried exits: `{audit['retried_exit_rows']}`.", "",
        "## Returns", "",
        f"- STOP_LOSS rows: `{stop['rows']}` ({stop['rate']:.2%}); mean executable return `{stop['mean_executable_return']:+.2%}`.",
        f"- Mean stop crossing return: `{stop['mean_crossing_return']:+.2%}`; executable loss worse than fixed -8% in `{stop['worse_than_fixed_8pct_rate']:.2%}` of stop rows.",
        f"- TIME_EXIT rows: `{time_exit['rows']}`; mean executable return `{time_exit['mean_executable_return']:+.2%}`; positive rate `{time_exit['positive_rate']:.2%}`.", "",
        "## Engine Parity", "",
        f"- Sparse audit trades: `{parity['sample_rows']}`; stop `{parity['stop_rows']}`, time exit `{parity['time_exit_rows']}`.",
        f"- Entry date, exit date, trigger type and economic return parity passed: `{parity['passed']}`.",
        f"- Maximum absolute economic-return error: `{parity['maximum_absolute_economic_return_error']:.3e}`.", "",
        "## Decision", "",
        "The executable label contract is accepted for a separately preregistered forward-only F0 V2 model. No historical 2026 improvement claim is authorized."
        if parity["passed"]
        else "The executable label contract is blocked by canonical-engine parity failure and may not train a model.",
    ]
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    strategy = StrategyConfig.from_yaml(args.strategy)
    plan, sessions = _audit_plan(strategy, output)
    cutoffs = load_source_max_dates({
        "market_daily_ts", "adj_factor_ts", "stk_limit_ts", "stock_st_ts"
    })
    if min(cutoffs["market_daily_ts"], cutoffs["adj_factor_ts"], cutoffs["stk_limit_ts"]) < plan.execution_end:
        raise ValueError(f"execution sources do not cover label audit horizon: {cutoffs}")

    print("phase=execution_bundle_load", flush=True)
    bundle = build_data_bundle(plan, strategy, output)
    signal_days = list(pd.to_datetime(plan.signal_sessions, utc=True))
    keys = _signal_keys(bundle, signal_days)
    print(f"phase=label_build keys={len(keys)}", flush=True)
    profile = ExecutablePathLabelProfile()
    labels, label_audit = build_executable_path_labels(
        keys,
        bundle.execution,
        sessions,
        profile=profile,
    )
    distribution = _distribution(labels)
    ledgers, parity_sample = _sample_ledgers(labels, bundle.execution, strategy)
    print(f"phase=engine_parity sample={len(parity_sample)}", flush=True)
    result = run_backtest(
        candidate_ledger=ledgers["candidate"],
        selection_ledger=ledgers["selection"],
        execution_panel=bundle.execution,
        corporate_actions=bundle.corporate_actions,
        strategy=strategy,
        execution_sessions=plan.execution_sessions,
        scenario_name="base",
    )
    parity = _parity_audit(parity_sample, result)
    if not parity["passed"]:
        raise AssertionError(f"executable label parity failed: {parity['checks']}")

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "RUN_STATUS.json", {
        "status": "LABEL_AUDIT_COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "profile": asdict(profile),
        "parity_passed": parity["passed"],
        "business_data_persisted": False,
        "credentials_persisted": False,
    })
    _write_json(output / "label_audit.json", label_audit)
    _write_json(output / "return_distribution.json", distribution)
    _write_json(output / "engine_parity.json", parity)
    _write_json(output / "plan.json", {
        "run_plan": plan.to_dict(),
        "source_cutoffs": cutoffs,
        "bundle_manifest": bundle.manifest,
    })
    _write_json(output / "code_manifest.json", {
        "src/aistock9988/labeling/executable_path.py": _sha(
            ROOT / "src/aistock9988/labeling/executable_path.py"
        ),
        "src/aistock9988/backtest/engine.py": _sha(ROOT / "src/aistock9988/backtest/engine.py"),
        "scripts/executable_label_contract_audit.py": _sha(Path(__file__).resolve()),
    })
    _write_result(output, label_audit, distribution, parity)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"run_complete={run(args)}")


if __name__ == "__main__":
    main()
