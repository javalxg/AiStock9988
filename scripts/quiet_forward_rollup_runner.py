"""Aggregate mature append-only quiet lockbox days into one NAV curve."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aistock9988.backtest.engine import run_backtest
from aistock9988.configuration import StrategyConfig
from aistock9988.data.bundle import (
    build_data_bundle,
    load_source_max_dates,
    load_trading_calendar,
)
from aistock9988.forward.early_path import (
    EarlyPathConfig,
    EarlyPathFailure,
    apply_early_path_overlay,
)
from aistock9988.forward.lockbox import ForwardLockbox
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.metrics import summarize

from quiet_forward_shadow_runner import (
    _code_manifest,
    _overlay_closure_paths,
    _right_tail_recall,
    _verify_code_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _day(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return (stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")).normalize()


def _write_waiting(output: Path, reason: str, details: dict[str, object]) -> str:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "WAITING_FOR_DATA",
        "reason": reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    (output / "WAITING_FOR_DATA.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return "WAITING_FOR_DATA"


def run(args: argparse.Namespace) -> str:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    strategy = StrategyConfig.from_yaml(args.strategy)
    overlay = EarlyPathConfig.from_yaml(args.overlay_config)
    overlay.validate_control(strategy)
    closure_paths = _overlay_closure_paths(args)
    if not all(path.is_file() for path in closure_paths):
        raise FileNotFoundError("early-path overlay config or preregistration is missing")
    start, end, execution_end = _day(args.asof_start), _day(args.asof_end), _day(args.execution_end)
    if end < start or execution_end <= end:
        raise ValueError("require asof_start <= asof_end < execution_end")
    if str(strategy.identity.get("research_status", "historical")) != "forward_only":
        raise ValueError("formal rollup requires a forward_only strategy")
    forward_start = _day(strategy.identity["forward_start"])
    if start < forward_start:
        raise ValueError(
            f"rollup rejects signal dates before forward_start {forward_start.date()}"
        )
    execution_sources = set(
        str(value) for value in strategy.data_policy["dense_required"]["execution"]
    )
    source_cutoffs = load_source_max_dates(execution_sources)
    stale = {
        name: value
        for name, value in source_cutoffs.items()
        if pd.Timestamp(value).date() < execution_end.date()
    }
    if stale:
        return _write_waiting(output, "execution_end_after_database_cutoff", {
            "execution_end": str(execution_end.date()),
            "source_cutoffs": source_cutoffs,
        })

    lockbox_root = args.lockbox.resolve()
    lockbox = ForwardLockbox(lockbox_root, experiment_id=strategy.strategy_id, config_sha256=strategy.config_hash)
    calendar = load_trading_calendar(str((start - pd.Timedelta(days=500)).date()), str(execution_end.date()))
    request = RunRequest(
        signal_start=str(start.date()), signal_end=str(end.date()), execution_end=str(execution_end.date()),
        output_dir=str(output), run_name=f"{strategy.strategy_id}-{start:%Y%m%d}-{end:%Y%m%d}",
    )
    plan = compile_run_plan(
        strategy,
        request,
        calendar["session"],
        require_complete_horizon=False,
    )
    days = pd.DatetimeIndex(pd.to_datetime(plan.signal_sessions, utc=True)).normalize()
    batches: list[tuple[pd.Timestamp, dict[str, pd.DataFrame], dict[str, object]]] = []
    for day in days:
        batch_dir = lockbox_root / "batches" / day.strftime("%Y-%m-%d")
        if not batch_dir.exists():
            raise FileNotFoundError(f"missing frozen batch for {day.date()}: run freeze first")
        manifest = lockbox.manifest_for_day(day)
        if str(manifest.get("freeze_data_cutoff")) != str(day.date()):
            raise ValueError(
                f"frozen batch {day.date()} has invalid freeze_data_cutoff"
            )
        expected_code_hash = manifest.get("code_manifest_sha256")
        code_manifest_path = batch_dir / "code_manifest.json"
        if not expected_code_hash:
            # An unsealed freeze without code-closure metadata is diagnostic-only.
            # Do not let rollup bypass the same formal settlement gate.
            raise ValueError(
                f"frozen batch {day.date()} is unsealed and has no code_manifest_sha256; "
                "formal rollup is refused"
            )
        if not code_manifest_path.exists():
            raise FileNotFoundError(f"frozen batch is missing code manifest: {code_manifest_path}")
        _verify_code_manifest(code_manifest_path, expected_hash=str(expected_code_hash))
        committed = lockbox.read_day(day)
        batches.append((day, committed, manifest))

    freeze_cutoffs = sorted({
        str(manifest["freeze_data_cutoff"]) for _, _, manifest in batches
    })
    candidate = pd.concat([committed["candidate"] for _, committed, _ in batches], ignore_index=True)
    selection = pd.concat([committed["selection"] for _, committed, _ in batches], ignore_index=True)
    bundle = build_data_bundle(plan, strategy, output / "pending_bundle")
    scenario_outputs: dict[str, dict[str, dict[str, pd.DataFrame]]] = {
        "control": {}, "shadow": {},
    }
    open_positions: dict[str, int] = {}
    for scenario in ("base", "stress"):
        result = run_backtest(
            candidate_ledger=candidate, selection_ledger=selection, execution_panel=bundle.execution,
            corporate_actions=bundle.corporate_actions, strategy=strategy,
            execution_sessions=plan.execution_sessions, scenario_name=scenario,
        )
        scenario_outputs["control"][scenario] = result
        try:
            scenario_outputs["shadow"][scenario] = apply_early_path_overlay(
                control_result=result,
                execution_panel=bundle.execution,
                execution_sessions=plan.execution_sessions,
                control_strategy=strategy,
                overlay=overlay,
                scenario_name=scenario,
            )
        except EarlyPathFailure as exc:
            output.mkdir(parents=True, exist_ok=True)
            failure = {
                "status": exc.code,
                "reason": str(exc),
                "scenario": scenario,
                "asof_start": str(start.date()),
                "asof_end": str(end.date()),
                "execution_end": str(execution_end.date()),
                "source_cutoffs": source_cutoffs,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (output / "FAILURE.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            manifest = _code_manifest(Path(args.strategy), closure_paths)
            (output / "rollup_code_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            shutil.copyfile(args.strategy, output / "strategy.yaml")
            shutil.copyfile(args.overlay_config, output / "early_path_overlay.yaml")
            shutil.copyfile(args.overlay_prereg, output / "early_path_preregistration.md")
            return str(output)
        open_positions[scenario] = int(
            result["nav"].sort_values("trade_date").iloc[-1]["open_positions"]
        )
    if any(value > 0 for value in open_positions.values()):
        return _write_waiting(output, "actual_positions_still_open", {
            "asof_start": str(start.date()),
            "asof_end": str(end.date()),
            "execution_end": str(execution_end.date()),
            "open_positions": open_positions,
            "source_cutoffs": source_cutoffs,
        })

    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
    (output / "freeze_data_cutoffs.json").write_text(json.dumps(freeze_cutoffs, indent=2) + "\n", encoding="utf-8")
    (output / "settlement_source_cutoffs.json").write_text(
        json.dumps(source_cutoffs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rollup_manifest = _code_manifest(Path(args.strategy), closure_paths)
    (output / "rollup_code_manifest.json").write_text(
        json.dumps(rollup_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copyfile(args.strategy, output / "strategy.yaml")
    shutil.copyfile(args.overlay_config, output / "early_path_overlay.yaml")
    shutil.copyfile(args.overlay_prereg, output / "early_path_preregistration.md")
    candidate.to_parquet(output / "candidate_ledger.parquet", index=False)
    selection.to_parquet(output / "selection_ledger.parquet", index=False)
    bundle.execution.to_parquet(output / "execution_panel.parquet", index=False)

    summaries: dict[str, dict[str, dict[str, object]]] = {"control": {}, "shadow": {}}
    for arm, arm_outputs in scenario_outputs.items():
        for scenario, result in arm_outputs.items():
            target = output / arm / scenario
            target.mkdir(parents=True)
            for name, frame in result.items():
                frame.to_parquet(target / f"{name}.parquet", index=False)
            metrics = _rollup_metrics(result, strategy, arm=arm, scenario=scenario)
            if arm == "shadow":
                metrics["paired_right_tail"] = _right_tail_recall(
                    arm_outputs=scenario_outputs, scenario=scenario
                )
                path = result["path_events"]
                terminals = path.drop_duplicates("trade_key", keep="last") if not path.empty else path
                metrics["path_state_counts"] = (
                    terminals["resulting_state"].value_counts().sort_index().to_dict()
                    if not terminals.empty else {}
                )
                metrics["break_pending_state_count"] = int(
                    metrics["path_state_counts"].get("BREAK_PENDING", 0)
                )
                metrics["effective_early_exit_count"] = int(len(result["paired_capital"]))
            summaries[arm][scenario] = metrics

    _apply_paired_acceptance(summaries, overlay)
    for arm, arm_summaries in summaries.items():
        for scenario, metrics in arm_summaries.items():
            (output / arm / scenario / "metrics.json").write_text(
                json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
            )
    (output / "ROLLUP_SUMMARY.json").write_text(json.dumps(summaries, indent=2, default=str) + "\n", encoding="utf-8")
    return str(output)


def _rollup_metrics(
    result: dict[str, pd.DataFrame],
    strategy: StrategyConfig,
    *,
    arm: str,
    scenario: str,
) -> dict[str, object]:
    metrics = summarize(
        result["nav"], result["fills"],
        initial_cash=float(strategy.execution["initial_cash"]),
        positions=result["positions"], corporate_actions=result["corporate_actions"],
    )
    nav = result["nav"].sort_values("trade_date")
    dates = pd.to_datetime(nav["trade_date"], utc=True)
    weekly_nav = nav.assign(
        period=dates.dt.tz_localize(None).dt.to_period("W-SUN")
    ).groupby("period")["nav"].last()
    weekly = weekly_nav.pct_change()
    if len(weekly):
        weekly.iloc[0] = weekly_nav.iloc[0] / float(strategy.execution["initial_cash"]) - 1.0
    weeks_ge_5 = int((weekly >= 0.05).sum())
    metrics.update({
        "arm": arm,
        "scenario": scenario,
        "weekly_ge_5_all_weeks": bool(len(weekly) > 0 and (weekly >= 0.05).all()),
        "weeks_ge_5": weeks_ge_5,
        "weeks_ge_5_rate": float(weeks_ge_5 / len(weekly)) if len(weekly) else None,
        "complete_weeks": int(len(weekly)),
        "trade_win_rate_target": bool(
            metrics["trade_win_rate"] is not None and float(metrics["trade_win_rate"]) >= 0.70
        ),
        "acceptance": {"passed": None},
        "status": "INSUFFICIENT_SAMPLE",
    })
    return metrics


def _apply_paired_acceptance(
    summaries: dict[str, dict[str, dict[str, object]]],
    overlay: EarlyPathConfig,
) -> None:
    evaluation = overlay.evaluation
    break_count = min(
        int(summaries["shadow"][scenario].get("effective_early_exit_count", 0))
        for scenario in ("base", "stress")
    )
    control_weeks = min(int(summaries["control"][s]["complete_weeks"]) for s in ("base", "stress"))
    control_trades = min(int(summaries["control"][s]["trade_count"]) for s in ("base", "stress"))
    control_mature = (
        control_weeks >= int(evaluation["minimum_control_weeks"])
        or control_trades >= int(evaluation["minimum_control_trades"])
    )
    sample_ready = control_mature and break_count >= int(evaluation["minimum_break_events"])
    terminal = (
        control_weeks >= int(evaluation["terminal_weeks"])
        or control_trades >= int(evaluation["terminal_control_trades"])
    )
    for scenario in ("base", "stress"):
        control = summaries["control"][scenario]
        shadow = summaries["shadow"][scenario]
        right_tail = shadow["paired_right_tail"]
        checks = {
            "pf_ge_2": shadow["portfolio_profit_factor"] is not None
            and float(shadow["portfolio_profit_factor"]) >= float(evaluation["portfolio_profit_factor_min"]),
            "pf_gt_control": shadow["portfolio_profit_factor"] is not None
            and control["portfolio_profit_factor"] is not None
            and float(shadow["portfolio_profit_factor"]) > float(control["portfolio_profit_factor"]),
            "maxdd_le_15": abs(float(shadow["max_drawdown"])) <= float(evaluation["max_drawdown_abs_max"]),
            "maxdd_no_worse_than_control": abs(float(shadow["max_drawdown"])) <= abs(float(control["max_drawdown"])),
            "return_gt_control": float(shadow["total_return"]) > float(control["total_return"]),
            "ex_best_week_positive_and_gt_control": float(shadow["return_excluding_best_week"]) > 0.0
            and float(shadow["return_excluding_best_week"]) > float(control["return_excluding_best_week"]),
            "ex_top3_positive_and_gt_control": float(shadow["return_excluding_top3_profit"]) > 0.0
            and float(shadow["return_excluding_top3_profit"]) > float(control["return_excluding_top3_profit"]),
            "right_tail_not_decreased": bool(right_tail["not_decreased"]),
            "stress_return_positive": float(summaries["shadow"]["stress"]["total_return"]) > 0.0,
        }
        shadow["sample_ready"] = sample_ready
        shadow["acceptance"] = {**checks, "passed": None if not sample_ready else all(checks.values())}
        if sample_ready:
            shadow["status"] = "READY_FOR_ACCEPTANCE"
        elif terminal and break_count < int(evaluation["minimum_break_events"]):
            shadow["status"] = "FAIL_INSUFFICIENT_EVENT_FREQUENCY"
        else:
            shadow["status"] = "INSUFFICIENT_SAMPLE"
        control["sample_ready"] = control_mature
        control["status"] = "MATURE_CONTROL" if control_mature else "INSUFFICIENT_SAMPLE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof-start", required=True)
    parser.add_argument("--asof-end", required=True)
    parser.add_argument("--execution-end", required=True)
    # Roll up only the clean formal forward root by default.
    parser.add_argument("--lockbox", type=Path, default=ROOT / "docs/council_20260828/CAP1_20_FORWARD_LOCKBOX")
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/reset_weak_confirm_v3_cap1_20_forward.yaml")
    parser.add_argument(
        "--overlay-config", type=Path,
        default=ROOT / "configs/strategy/cap1_early_path_forward_overlay.yaml",
    )
    parser.add_argument(
        "--overlay-prereg", type=Path,
        default=ROOT / "docs/council_20260828/CAP1_EARLY_PATH_FORWARD_PREREG_20260901.md",
    )
    parser.add_argument("--output", type=Path, required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
