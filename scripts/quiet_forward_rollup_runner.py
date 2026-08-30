"""Aggregate mature append-only quiet lockbox days into one NAV curve."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aistock9988.backtest.v3_engine import run_v3_backtest
from aistock9988.configuration import ModelConfig, StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_trading_calendar
from aistock9988.forward.lockbox import ForwardLockbox
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.v3_metrics import summarize_v3

from quiet_forward_shadow_runner import _code_manifest, _verify_code_manifest

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
    model = ModelConfig.from_yaml(args.model)
    start, end, execution_end = _day(args.asof_start), _day(args.asof_end), _day(args.execution_end)
    if end < start or execution_end <= end:
        raise ValueError("require asof_start <= asof_end < execution_end")

    lockbox_root = args.lockbox.resolve()
    lockbox = ForwardLockbox(lockbox_root, experiment_id=strategy.strategy_id, config_sha256=strategy.config_hash)
    calendar = load_trading_calendar(str((start - pd.Timedelta(days=500)).date()), str(execution_end.date()))
    request = RunRequest(
        signal_start=str(start.date()), signal_end=str(end.date()), execution_end=str(execution_end.date()),
        output_dir=str(output), run_name=f"{strategy.strategy_id}-{start:%Y%m%d}-{end:%Y%m%d}",
    )
    plan = compile_run_plan(strategy, model, request, calendar["session"])
    days = pd.DatetimeIndex(pd.to_datetime(plan.signal_sessions, utc=True)).normalize()
    batches: list[tuple[pd.Timestamp, dict[str, pd.DataFrame], dict[str, object]]] = []
    for day in days:
        batch_dir = lockbox_root / "batches" / day.strftime("%Y-%m-%d")
        if not batch_dir.exists():
            raise FileNotFoundError(f"missing frozen batch for {day.date()}: run freeze first")
        manifest = lockbox.manifest_for_day(day)
        expected_code_hash = manifest.get("code_manifest_sha256")
        code_manifest_path = batch_dir / "code_manifest.json"
        if not expected_code_hash:
            # A legacy freeze without code-closure metadata is diagnostic-only.
            # Do not let rollup bypass the same formal settlement gate.
            raise ValueError(
                f"frozen batch {day.date()} is legacy and has no code_manifest_sha256; "
                "formal rollup is refused"
            )
        if not code_manifest_path.exists():
            raise FileNotFoundError(f"frozen batch is missing code manifest: {code_manifest_path}")
        _verify_code_manifest(code_manifest_path, expected_hash=str(expected_code_hash))
        committed = lockbox.read_day(day)
        batches.append((day, committed, manifest))

    source_ends = sorted({str(manifest["source_end"]) for _, _, manifest in batches})
    if _day(max(source_ends)) > execution_end:
        raise ValueError("execution_end does not cover all frozen source_end values")
    candidate = pd.concat([committed["candidate"] for _, committed, _ in batches], ignore_index=True)
    selection = pd.concat([committed["selection"] for _, committed, _ in batches], ignore_index=True)
    # Check every selected symbol through its own scheduled exit before running the engine.
    sessions = pd.DatetimeIndex(pd.to_datetime(plan.execution_sessions, utc=True)).normalize()
    needed_by_day: dict[pd.Timestamp, pd.DatetimeIndex] = {}
    hold = int(strategy.execution["hold_sessions_from_fill"])
    delay = int(strategy.decision["entry_delay_sessions"])
    for day in pd.DatetimeIndex(pd.to_datetime(selection["asof"], utc=True)).normalize().unique():
        index = sessions.get_indexer([day])[0]
        if index < 0:
            raise ValueError(f"signal day {day.date()} is not in execution calendar")
        needed_by_day[day] = sessions[index + delay : index + delay + hold + 1]

    bundle = build_data_bundle(plan, strategy, output / "pending_bundle")
    missing: list[str] = []
    for day, needed in needed_by_day.items():
        codes = set(candidate.loc[
            pd.to_datetime(candidate["asof"], utc=True).eq(day)
            & candidate["candidate_status"].eq("IN_VIEW"), "ts_code"
        ].astype(str))
        panel = bundle.execution[
            bundle.execution["ts_code"].isin(codes) & bundle.execution["trade_date"].isin(needed)
        ]
        expected = pd.MultiIndex.from_product(
            [needed, sorted(codes)], names=["trade_date", "ts_code"]
        )
        actual = (
            panel.drop_duplicates(["trade_date", "ts_code"])
            .set_index(["trade_date", "ts_code"])["execution_data_eligible"]
            if not panel.empty else pd.Series(dtype=bool)
        )
        coverage = actual.reindex(expected)
        if len(needed) == 0 or len(codes) == 0 or coverage.isna().any() or not coverage.astype(bool).all():
            missing.append(str(day.date()))
    if missing:
        eligible_execution = bundle.execution[bundle.execution["execution_data_eligible"].astype(bool)]
        return _write_waiting(output, "execution_horizon_not_complete", {
            "asof_start": str(start.date()), "asof_end": str(end.date()),
            "execution_end": str(execution_end.date()), "missing_signal_days": missing,
            "coverage_scope": "candidate_view_IN_VIEW",
            "database_last_eligible_trade_date": (
                str(pd.to_datetime(eligible_execution["trade_date"], utc=True).max().date())
                if not eligible_execution.empty else None
            ),
        })

    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
    (output / "source_ends.json").write_text(json.dumps(source_ends, indent=2) + "\n", encoding="utf-8")
    rollup_manifest = _code_manifest(Path(args.strategy), Path(args.model))
    (output / "rollup_code_manifest.json").write_text(
        json.dumps(rollup_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copyfile(args.strategy, output / "strategy.yaml")
    shutil.copyfile(args.model, output / "model.yaml")
    (output / "candidate_ledger.parquet").parent.mkdir(parents=True, exist_ok=True)
    candidate.to_parquet(output / "candidate_ledger.parquet", index=False)
    selection.to_parquet(output / "selection_ledger.parquet", index=False)
    bundle.execution.to_parquet(output / "execution_panel.parquet", index=False)

    summaries: dict[str, dict[str, object]] = {}
    for scenario in ("base", "stress"):
        result = run_v3_backtest(
            candidate_ledger=candidate, selection_ledger=selection, execution_panel=bundle.execution,
            corporate_actions=bundle.corporate_actions, strategy=strategy,
            execution_sessions=plan.execution_sessions, scenario_name=scenario,
        )
        target = output / scenario
        target.mkdir()
        for name, frame in result.items():
            frame.to_parquet(target / f"{name}.parquet", index=False)
        metrics = summarize_v3(result["nav"], result["fills"], initial_cash=float(strategy.execution["initial_cash"]))
        nav = result["nav"].sort_values("trade_date")
        dates = pd.to_datetime(nav["trade_date"], utc=True)
        weekly_nav = nav.assign(period=dates.dt.tz_localize(None).dt.to_period("W-SUN")).groupby("period")["nav"].last()
        weekly = weekly_nav.pct_change()
        if len(weekly):
            weekly.iloc[0] = weekly_nav.iloc[0] / float(strategy.execution["initial_cash"]) - 1.0
        metrics["weekly_ge_5_all_weeks"] = bool(len(weekly) > 0 and (weekly >= 0.05).all())
        metrics["complete_weeks"] = int(len(weekly))
        metrics["sample_ready"] = bool(
            int(metrics["trade_count"]) >= 60 or int(metrics["complete_weeks"]) >= 26
        )
        metrics["trade_win_rate_target"] = bool(
            metrics["trade_win_rate"] is not None and float(metrics["trade_win_rate"]) >= 0.70
        )
        metrics["scenario"] = scenario
        metrics["acceptance"] = {
            "pf_ge_2": metrics["portfolio_profit_factor"] is not None and float(metrics["portfolio_profit_factor"]) >= 2.0,
            "maxdd_le_15": abs(float(metrics["max_drawdown"])) <= 0.15,
            "ex_best_week_positive": float(metrics["return_excluding_best_week"]) > 0.0,
            "ex_top3_profit_positive": float(metrics["return_excluding_top3_profit"]) > 0.0,
            # The 70% win-rate objective remains a reported target.  It is not
            # part of the hard promotion gate, which is fixed by the council
            # acceptance contract and must not silently diverge from docs.
            "stress_return_positive": float(metrics["total_return"]) > 0.0 if scenario == "stress" else None,
        }
        metrics["acceptance"]["passed"] = None if not metrics["sample_ready"] else bool(
            all(value is True for value in metrics["acceptance"].values())
        )
        metrics["status"] = "READY_FOR_ACCEPTANCE" if metrics["sample_ready"] else "INSUFFICIENT_SAMPLE"
        (target / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
        summaries[scenario] = metrics
    # The base arm cannot pass the stress requirement until both scenarios are
    # computed; make the cross-arm condition explicit in the rollup artifact.
    if "base" in summaries and "stress" in summaries:
        summaries["base"]["acceptance"]["stress_return_positive"] = (
            float(summaries["stress"]["total_return"]) > 0.0
        )
        for scenario in summaries:
            checks = summaries[scenario]["acceptance"]
            if summaries[scenario]["sample_ready"]:
                checks["passed"] = bool(
                    all(value is True for key, value in checks.items() if key != "passed")
                )
            else:
                checks["passed"] = None
            (output / scenario / "metrics.json").write_text(
                json.dumps(summaries[scenario], indent=2, default=str) + "\n", encoding="utf-8"
            )
    (output / "ROLLUP_SUMMARY.json").write_text(json.dumps(summaries, indent=2, default=str) + "\n", encoding="utf-8")
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof-start", required=True)
    parser.add_argument("--asof-end", required=True)
    parser.add_argument("--execution-end", required=True)
    # Roll up only the clean formal forward root by default.
    parser.add_argument("--lockbox", type=Path, default=ROOT / "docs/council_20260828/S49_QUIET_FORWARD_LOCKBOX_FORMAL")
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/quiet_confirmed_v1.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "configs/model/disabled.yaml")
    parser.add_argument("--output", type=Path, required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
