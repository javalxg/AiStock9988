#!/usr/bin/env python3
"""Run the preregistered dragon-tiger pullback/reclaim V1 experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aistock9988.backtest.engine import run_backtest
from aistock9988.configuration import StrategyConfig
from aistock9988.data.bundle import (
    build_data_bundle,
    load_source_max_dates,
    load_trading_calendar,
)
from aistock9988.data.dragon_tiger import (
    load_dragon_tiger_cutoffs,
    load_dragon_tiger_events,
)
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.metrics import summarize
from aistock9988.selection.dragon_tiger import build_pullback_reclaim_ledgers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "strategy" / "dragon_tiger_pullback_reclaim_v1.yaml"
DEFAULT_OUTPUT = (
    ROOT / "docs" / "council_20260828"
    / "DRAGON_TIGER_PULLBACK_RECLAIM_V1_2026_TO_DB_CUTOFF_20260901"
)


def run(args: argparse.Namespace) -> Path:
    strategy = StrategyConfig.from_yaml(args.strategy)
    if strategy.strategy_id != "dragon_tiger_pullback_reclaim_v1":
        raise ValueError("runner requires dragon_tiger_pullback_reclaim_v1 config")

    dense_sources = set(
        str(value)
        for stage in strategy.data_policy["dense_required"].values()
        for value in stage
    )
    dense_cutoffs = load_source_max_dates(dense_sources)
    event_cutoffs = load_dragon_tiger_cutoffs()
    dense_end = min(pd.Timestamp(value) for value in dense_cutoffs.values()).normalize()
    event_end = min(pd.Timestamp(value) for value in event_cutoffs.values()).normalize()
    if args.execution_end != "auto":
        requested = pd.Timestamp(args.execution_end).normalize()
        if requested > dense_end:
            raise ValueError(f"execution_end {requested.date()} exceeds dense cutoff {dense_end.date()}")
        dense_end = requested
    if args.event_end != "auto":
        requested = pd.Timestamp(args.event_end).normalize()
        if requested > event_end:
            raise ValueError(f"event_end {requested.date()} exceeds event cutoff {event_end.date()}")
        event_end = requested
    if event_end > dense_end:
        raise ValueError("event source extends beyond dense execution data")

    planning_calendar = load_trading_calendar(
        str((pd.Timestamp(args.signal_start) - pd.Timedelta(days=500)).date()),
        str(dense_end.date()),
    )
    sessions = pd.DatetimeIndex(planning_calendar["session"])
    event_index = sessions.get_indexer([event_end.tz_localize("UTC")])[0]
    if event_index < 0:
        raise ValueError(f"event cutoff {event_end.date()} is not an exchange session")
    observation_index = min(event_index + 5, len(sessions) - 1)
    observation_end = sessions[observation_index]

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    request = RunRequest(
        signal_start=args.signal_start,
        signal_end=str(observation_end.date()),
        execution_end=str(dense_end.date()),
        output_dir=str(output),
        run_name=args.run_name,
    )
    plan = compile_run_plan(
        strategy, request, planning_calendar["session"], require_complete_horizon=False
    )
    plan_payload = plan.to_dict()
    plan_payload.update({
        "event_source_start": str(pd.Timestamp(args.signal_start).date()),
        "event_source_end": str(event_end.date()),
        "observation_end": str(observation_end.date()),
        "complete_horizon_required": False,
        "end_policy": "mark_open_positions",
        "dense_source_cutoffs": dense_cutoffs,
        "event_source_cutoffs": event_cutoffs,
    })

    output.mkdir(parents=True, exist_ok=True)
    for name in ("configs", "manifests", "backtests", "diagnostics"):
        (output / name).mkdir()
    shutil.copyfile(args.strategy, output / "configs" / "strategy.yaml")
    _write_json(output / "plan.json", plan_payload)
    _write_json(output / "RUN_STATUS.json", {
        "run_name": args.run_name,
        "status": "RUNNING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": strategy.strategy_id,
        "strategy_hash": strategy.config_hash,
        "git_commit": _git_commit(),
        "credentials_persisted": False,
        "business_data_persisted": False,
    })

    print("phase=bundle start", flush=True)
    bundle = build_data_bundle(plan, strategy, output)
    print(f"phase=bundle complete id={bundle.bundle_id}", flush=True)
    source = load_dragon_tiger_events(args.signal_start, str(event_end.date()))
    source_manifest = dict(source.manifest)
    source_manifest["dense_source_cutoffs"] = dense_cutoffs
    source_manifest["event_source_cutoffs"] = event_cutoffs
    source_manifest["execution_bundle_id"] = bundle.bundle_id
    _write_json(output / "data_manifest.json", {
        **bundle.manifest,
        "dragon_tiger": source_manifest,
        "business_data_persisted": False,
    })

    print("phase=event_state start", flush=True)
    ledgers = build_pullback_reclaim_ledgers(
        source.events,
        bundle.execution,
        plan.signal_sessions,
        entries_per_decision=int(strategy.portfolio["entries_per_decision"]),
    )
    state_summary = _state_summary(ledgers)
    state_summary["state_ledger_sha256"] = _frame_hash(ledgers["state"])
    state_summary["candidate_ledger_sha256"] = _frame_hash(ledgers["candidate"])
    state_summary["selection_ledger_sha256"] = _frame_hash(ledgers["selection"])
    _write_json(output / "diagnostics" / "event_state_summary.json", state_summary)
    print(
        f"phase=event_state complete confirmed={state_summary['confirmed_events']} "
        f"active_days={state_summary['active_confirmation_sessions']}",
        flush=True,
    )

    portfolio: dict[str, dict[str, Any]] = {}
    result_hashes: dict[str, Any] = {}
    raw_results: dict[str, dict[str, pd.DataFrame]] = {}
    for scenario in ("base", "stress"):
        print(f"phase=backtest scenario={scenario} start", flush=True)
        result = run_backtest(
            candidate_ledger=ledgers["candidate"],
            selection_ledger=ledgers["selection"],
            execution_panel=bundle.execution,
            corporate_actions=bundle.corporate_actions,
            strategy=strategy,
            execution_sessions=plan.execution_sessions,
            scenario_name=scenario,
        )
        metrics = summarize(
            result["nav"],
            result["fills"],
            initial_cash=float(strategy.execution["initial_cash"]),
            positions=result["positions"],
            corporate_actions=result["corporate_actions"],
        )
        metrics.update(_weekly_metrics(result["nav"], float(strategy.execution["initial_cash"])))
        metrics.update({
            "scenario": scenario,
            "entry_attempts": int(len(result["execution_decisions"])),
            "entry_fills": int(
                result["execution_decisions"]["chosen"].sum()
                if not result["execution_decisions"].empty else 0
            ),
            "open_positions_at_end": int(len(result["open_positions"])),
            "active_signal_days": int(state_summary["active_confirmation_sessions"]),
        })
        metrics["acceptance"] = _acceptance(metrics, strategy)
        _write_json(output / "backtests" / scenario / "metrics.json", metrics)
        portfolio[scenario] = metrics
        result_hashes[scenario] = {
            name: {"rows": int(len(frame)), "sha256": _frame_hash(frame)}
            for name, frame in result.items()
        }
        raw_results[scenario] = result
        print(
            f"phase=backtest scenario={scenario} return={metrics['total_return']:+.6f} "
            f"pf={metrics['portfolio_profit_factor']} win={metrics['trade_win_rate']} "
            f"maxdd={metrics['max_drawdown']:+.6f}",
            flush=True,
        )

    _write_json(output / "PORTFOLIO_SUMMARY.json", portfolio)
    _write_json(output / "manifests" / "in_memory_ledger_manifest.json", {
        "event_and_selection": {
            "state": {"rows": int(len(ledgers["state"])), "sha256": state_summary["state_ledger_sha256"]},
            "candidate": {"rows": int(len(ledgers["candidate"])), "sha256": state_summary["candidate_ledger_sha256"]},
            "selection": {"rows": int(len(ledgers["selection"])), "sha256": state_summary["selection_ledger_sha256"]},
        },
        "backtest": result_hashes,
        "business_data_persisted": False,
    })
    verification = _verify(
        source.events, ledgers, bundle.execution, raw_results, plan_payload, strategy, output
    )
    _write_json(output / "diagnostics" / "verification.json", verification)
    _write_result(output, plan_payload, state_summary, portfolio, verification)

    code_paths = [
        ROOT / "src/aistock9988/data/dragon_tiger.py",
        ROOT / "src/aistock9988/selection/dragon_tiger.py",
        ROOT / "src/aistock9988/backtest/engine.py",
        ROOT / "src/aistock9988/data/bundle.py",
        ROOT / "src/aistock9988/reporting/metrics.py",
        Path(__file__).resolve(),
    ]
    _write_json(output / "manifests" / "code_manifest.json", {
        str(path.relative_to(ROOT)): _file_hash(path) for path in code_paths
    })
    status = json.loads((output / "RUN_STATUS.json").read_text(encoding="utf-8"))
    status.update({
        "status": "DIAGNOSTIC_COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "bundle_id": bundle.bundle_id,
        "verification_passed": bool(verification["passed"]),
        "base_acceptance_passed": bool(portfolio["base"]["acceptance"]["passed"]),
        "stress_acceptance_passed": bool(portfolio["stress"]["acceptance"]["passed"]),
        "overall_acceptance_passed": bool(
            portfolio["base"]["acceptance"]["passed"]
            and portfolio["stress"]["acceptance"]["passed"]
        ),
    })
    _write_json(output / "RUN_STATUS.json", status, replace=True)
    artifacts = {
        str(path.relative_to(output)): {"sha256": _file_hash(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json(output / "manifests" / "artifact_manifest.json", artifacts)
    return output


def _state_summary(ledgers: dict[str, pd.DataFrame]) -> dict[str, Any]:
    state = ledgers["state"]
    candidate = ledgers["candidate"]
    return {
        "events": int(len(state)),
        "status_counts": {
            str(key): int(value) for key, value in state["status"].value_counts().sort_index().items()
        },
        "confirmed_events": int(state["status"].eq("CONFIRMED").sum()),
        "candidate_rows": int(len(candidate)),
        "active_confirmation_sessions": int(candidate["asof"].nunique()) if not candidate.empty else 0,
        "maximum_daily_candidates": int(candidate.groupby("asof").size().max()) if not candidate.empty else 0,
        "overlapping_events_skipped": int(state["status"].eq("OVERLAPPING_EVENT_SKIPPED").sum()),
    }


def _weekly_metrics(nav: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    ordered = nav.sort_values("trade_date", kind="mergesort").copy()
    dates = pd.to_datetime(ordered["trade_date"], errors="raise", utc=True)
    weekly_nav = ordered.assign(
        week=dates.dt.tz_localize(None).dt.to_period("W-SUN")
    ).groupby("week", sort=True)["nav"].last()
    weekly = weekly_nav.pct_change()
    if len(weekly):
        weekly.iloc[0] = float(weekly_nav.iloc[0]) / initial_cash - 1.0
    return {
        "calendar_week_count": int(len(weekly)),
        "mean_weekly_return": float(weekly.mean()) if len(weekly) else None,
        "median_weekly_return": float(weekly.median()) if len(weekly) else None,
    }


def _acceptance(metrics: dict[str, Any], strategy: StrategyConfig) -> dict[str, Any]:
    acceptance = strategy.acceptance
    pf = metrics.get("portfolio_profit_factor")
    win = metrics.get("trade_win_rate")
    mean_weekly = metrics.get("mean_weekly_return")
    tests = {
        "mean_weekly_return": mean_weekly is not None and float(mean_weekly) >= float(acceptance["mean_weekly_return_min"]),
        "trade_win_rate": win is not None and float(win) >= float(acceptance["trade_win_rate_min"]),
        "profit_factor": pf is not None and float(pf) >= float(acceptance["portfolio_profit_factor_min"]),
        "max_drawdown": abs(float(metrics["max_drawdown"])) <= float(acceptance["max_drawdown_abs_max"]),
        "excluding_best_week": float(metrics["return_excluding_best_week"]) > float(acceptance["return_excluding_best_week_min_exclusive"]),
        "excluding_top3_profit": float(metrics["return_excluding_top3_profit"]) > float(acceptance["return_excluding_top3_profit_min_exclusive"]),
        "minimum_closed_trades": int(metrics["trade_count"]) >= int(acceptance["minimum_closed_trades"]),
        "minimum_active_signal_days": int(metrics["active_signal_days"]) >= int(acceptance["minimum_active_signal_days"]),
        "maximum_five_positions": int(metrics["max_open_positions"]) <= 5,
    }
    return {"passed": all(tests.values()), "tests": tests}


def _verify(
    events: pd.DataFrame,
    ledgers: dict[str, pd.DataFrame],
    execution: pd.DataFrame,
    results: dict[str, dict[str, pd.DataFrame]],
    plan: dict[str, Any],
    strategy: StrategyConfig,
    output: Path,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["event_keys_unique"] = not events.duplicated(["event_date", "ts_code"]).any()
    checks["event_cutoff"] = bool(
        (events["event_date"] <= pd.Timestamp(plan["event_source_end"], tz="UTC")).all()
    )
    candidate = ledgers["candidate"]
    checks["candidate_keys_unique"] = not candidate.duplicated(["asof", "ts_code"]).any()
    checks["candidate_dates_bounded"] = bool(
        candidate.empty
        or (
            (candidate["event_date"] < candidate["pullback_date"])
            & (candidate["pullback_date"] < candidate["asof"])
            & (candidate["asof"] <= pd.Timestamp(plan["observation_end"], tz="UTC"))
        ).all()
    )
    checks["no_parameter_sweep"] = strategy.acceptance.get("parameter_sweep") is False
    execution_keys = execution[["trade_date", "ts_code", "execution_data_eligible"]]
    for scenario, result in results.items():
        nav = result["nav"]
        checks[f"{scenario}_nav_identity"] = bool(
            np.allclose(nav["cash"] + nav["market_value"], nav["nav"], rtol=0, atol=1e-8)
        )
        checks[f"{scenario}_cash_nonnegative"] = bool((nav["cash"] >= -1e-8).all())
        checks[f"{scenario}_position_cap"] = bool((nav["open_positions"] <= 5).all())
        checks[f"{scenario}_execution_end"] = str(pd.Timestamp(nav["trade_date"].max()).date()) == plan["execution_end"]
        fills = result["fills"]
        if fills.empty:
            checks[f"{scenario}_fills_execution_eligible"] = True
        else:
            joined = fills[["trade_date", "ts_code"]].merge(
                execution_keys, on=["trade_date", "ts_code"], how="left", validate="many_to_one"
            )
            checks[f"{scenario}_fills_execution_eligible"] = bool(
                joined["execution_data_eligible"].fillna(False).all()
            )
        decisions = result["execution_decisions"]
        checks[f"{scenario}_entries_next_session"] = bool(
            decisions.empty
            or (pd.to_datetime(decisions["execution_session"], utc=True) > pd.to_datetime(decisions["signal_session"], utc=True)).all()
        )
    checks["no_business_data_files"] = not any(
        path.suffix.lower() in {".csv", ".parquet", ".pkl", ".pickle"}
        for path in output.rglob("*") if path.is_file()
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": sorted(key for key, value in checks.items() if not value),
    }


def _write_result(
    output: Path,
    plan: dict[str, Any],
    state: dict[str, Any],
    portfolio: dict[str, dict[str, Any]],
    verification: dict[str, Any],
) -> None:
    lines = [
        "# Dragon-Tiger Pullback Reclaim V1", "",
        "## Contract", "",
        f"- Event source: `{plan['event_source_start']}` through `{plan['event_source_end']}`.",
        f"- Pullback/reclaim observation through `{plan['observation_end']}`; execution and marks through `{plan['execution_end']}`.",
        "- Rules only, next-open entry, maximum five positions, H10, -8% prior-close trailing stop.",
        "- No threshold sweep, XGBoost, frozen 202-factor input, raw-data cache, or persisted business-data ledger.", "",
        "## Event Funnel", "",
        f"- Source stock-days: `{state['events']}`; confirmed: `{state['confirmed_events']}`; active confirmation sessions: `{state['active_confirmation_sessions']}`.",
        f"- Overlapping events skipped during observation: `{state['overlapping_events_skipped']}`.", "",
        "## Portfolio", "",
        "| Cost | Return | Mean week | Win rate | PF | MaxDD | Ex-best-week | Ex-top3 | Trades | Open | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("base", "stress"):
        item = portfolio[name]
        pf = "NA" if item["portfolio_profit_factor"] is None else f"{item['portfolio_profit_factor']:.3f}"
        win = "NA" if item["trade_win_rate"] is None else f"{item['trade_win_rate']:.2%}"
        lines.append(
            f"| {name} | {item['total_return']:+.2%} | {item['mean_weekly_return']:+.2%} | {win} | {pf} | "
            f"{item['max_drawdown']:.2%} | {item['return_excluding_best_week']:+.2%} | "
            f"{item['return_excluding_top3_profit']:+.2%} | {item['trade_count']} | "
            f"{item['open_positions_at_end']} | {item['acceptance']['passed']} |"
        )
    lines.extend(["", "## Decision", ""])
    passed = verification["passed"] and all(
        portfolio[name]["acceptance"]["passed"] for name in ("base", "stress")
    )
    lines.append(
        "V1 advances." if passed else
        "V1 is rejected unchanged. Failure is evidence against this definition and is not repaired with a threshold scan."
    )
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if normalized[column].dtype == "object":
            normalized[column] = normalized[column].map(lambda value: repr(value))
    payload = pd.util.hash_pandas_object(normalized, index=False).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"immutable artifact exists: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="dragon_tiger_pullback_reclaim_v1_2026")
    parser.add_argument("--signal-start", default="2026-01-05")
    parser.add_argument("--event-end", default="auto")
    parser.add_argument("--execution-end", default="auto")
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        status_path = args.output.resolve() / "RUN_STATUS.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status.update({
                "status": "FAILED",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            _write_json(status_path, status, replace=True)
        raise
    print(f"run_complete={result}")


if __name__ == "__main__":
    main()
