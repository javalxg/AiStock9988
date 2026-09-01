"""Run the first configuration-driven V3 strategy backtest end to end."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aistock9988.backtest.engine import run_backtest
from aistock9988.configuration import StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_source_max_dates, load_trading_calendar
from aistock9988.features.engine import build_feature_ledger
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.metrics import summarize
from aistock9988.selection.pipeline import build_rule_ledgers, evaluate_expression
from aistock9988.time.session import session_close


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "council_20260828" / "S40_QUIET_CONFIRMED_V3_FULL_UNIVERSE_2026"


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")

    strategy = StrategyConfig.from_yaml(args.strategy)
    if str(strategy.identity.get("research_status", "historical")) == "forward_only" and not getattr(args, "diagnostic_history", False):
        raise ValueError(
            "forward-only strategies require --diagnostic-history for an explicitly labeled historical diagnostic"
        )
    execution_sources = set(str(value) for value in strategy.data_policy["dense_required"]["execution"])
    selection_sources = set(str(value) for value in strategy.data_policy["dense_required"]["selection"])
    # A ranking enrichment may be optional at stock-session granularity while
    # still being required to cover the requested signal period. Missing rows
    # exclude only that stock-day; a stale table must not silently shorten the run.
    selection_sources.update(
        str(value) for value in strategy.data_policy.get("optional_enrichment", ())
    )
    source_cutoffs = load_source_max_dates(execution_sources | selection_sources)
    requested_signal_end = pd.Timestamp(args.signal_end).date()
    requested_execution_end = pd.Timestamp(args.execution_end).date()
    signal_cutoff = min(pd.Timestamp(value).date() for value in source_cutoffs.values())
    execution_cutoffs = {name: source_cutoffs[name] for name in execution_sources}
    execution_cutoff = min(pd.Timestamp(value).date() for value in execution_cutoffs.values())
    if requested_signal_end > signal_cutoff:
        raise ValueError(
            f"signal_end {requested_signal_end} exceeds required-source cutoff {signal_cutoff}: {source_cutoffs}"
        )
    if requested_execution_end > execution_cutoff:
        raise ValueError(
            f"execution_end {requested_execution_end} exceeds execution-source cutoff {execution_cutoff}: {execution_cutoffs}"
        )
    calendar_start = str((pd.Timestamp(args.signal_start) - pd.Timedelta(days=500)).date())
    planning_calendar = load_trading_calendar(calendar_start, args.execution_end)
    request = RunRequest(
        signal_start=args.signal_start,
        signal_end=args.signal_end,
        execution_end=args.execution_end,
        output_dir=str(output),
        run_name=args.run_name,
    )
    plan = compile_run_plan(strategy, request, planning_calendar["session"])
    output.mkdir(parents=True, exist_ok=True)
    for name in ("configs", "manifests", "ledgers", "backtests", "diagnostics", "logs"):
        (output / name).mkdir()
    shutil.copyfile(args.strategy, output / "configs" / "strategy.yaml")
    _write_json(output / "plan.json", plan.to_dict())
    _write_json(output / "RUN_STATUS.json", {
        "run_name": args.run_name,
        "status": "RUNNING",
        "research_status": "DIAGNOSTIC_SEEN_HISTORY",
        "diagnostic_history": bool(getattr(args, "diagnostic_history", False)),
        "strategy_research_status": str(strategy.identity.get("research_status", "historical")),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": strategy.strategy_id,
        "strategy_hash": strategy.config_hash,
        "python": sys.version,
        "credentials_persisted": False,
    })

    print("phase=snapshot start", flush=True)
    bundle = build_data_bundle(plan, strategy, output)
    passthrough_audit = _passthrough_coverage_audit(bundle, strategy, plan.to_dict())
    bundle.manifest["passthrough_enrichment_audit"] = passthrough_audit
    _write_json(output / "data_manifest.json", bundle.manifest)
    _write_json(output / "diagnostics" / "passthrough_enrichment_coverage.json", passthrough_audit)
    if not passthrough_audit["passed"]:
        raise ValueError(
            "passthrough enrichment signal-date coverage failed: "
            f"{passthrough_audit['failed_dates']}"
        )
    _write_json(output / "manifests" / "config_manifest.json", {
        "strategy_hash": strategy.config_hash,
        "plan_hash": _json_hash(plan.to_dict()),
        "diagnostic_history": bool(getattr(args, "diagnostic_history", False)),
        "strategy_research_status": str(strategy.identity.get("research_status", "historical")),
    })
    bundle.universe.to_parquet(output / "ledgers" / "universe_ledger.parquet", index=False)
    bundle.availability.to_parquet(output / "ledgers" / "data_availability_ledger.parquet", index=False)
    bundle.execution.to_parquet(output / "ledgers" / "execution_panel.parquet", index=False)

    control_portfolio: dict[str, dict[str, Any]] | None = None
    control_strategy_path = getattr(args, "control_strategy", None)
    if control_strategy_path is not None:
        control_strategy = StrategyConfig.from_yaml(control_strategy_path)
        _validate_control_contract(strategy, control_strategy)
        control_root = output / "control"
        for name in ("configs", "ledgers", "backtests", "diagnostics"):
            (control_root / name).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(control_strategy_path, control_root / "configs" / "strategy.yaml")
        control_plan = plan.to_dict()
        control_plan.update({
            "run_name": f"{args.run_name}_CONTROL",
            "output_dir": str(control_root.resolve()),
            "strategy_id": control_strategy.strategy_id,
            "strategy_hash": control_strategy.config_hash,
        })
        _write_json(control_root / "plan.json", control_plan)
        print("phase=control_features start", flush=True)
        control_features = build_feature_ledger(bundle, control_strategy)
        control_features.to_parquet(control_root / "ledgers" / "feature_ledger.parquet", index=False)
        control_ledgers = build_rule_ledgers(
            control_features, control_strategy, plan.signal_sessions
        )
        for ledger_name in ("score", "candidate", "selection"):
            control_ledgers[ledger_name].to_parquet(
                control_root / "ledgers" / f"{ledger_name}_ledger.parquet", index=False
            )
        control_selection_summary = _selection_summary(control_ledgers)
        _write_json(
            control_root / "diagnostics" / "selection_summary.json",
            control_selection_summary,
        )
        control_portfolio = _run_scenarios(
            root=control_root,
            ledgers=control_ledgers,
            bundle=bundle,
            strategy=control_strategy,
            plan=control_plan,
            selection_summary=control_selection_summary,
            log_prefix="control_",
        )
        if strategy.strategy_id == "reset_weak_confirm_v3_cap1_20":
            _write_capacity_audit(control_root, bundle, control_strategy)
        _write_json(control_root / "PORTFOLIO_SUMMARY.json", control_portfolio)
        _write_result(
            control_root,
            control_plan,
            bundle.manifest,
            control_selection_summary,
            control_portfolio,
            control_strategy,
        )
        control_verification = _verify_preseal(
            control_root,
            control_plan,
            bundle.availability,
            bundle.execution,
            control_features,
            control_ledgers,
            control_strategy,
        )
        _write_json(
            control_root / "diagnostics" / "verification.json", control_verification
        )
        del control_features, control_ledgers

    print("phase=features start", flush=True)
    features = build_feature_ledger(bundle, strategy)
    features.to_parquet(output / "ledgers" / "feature_ledger.parquet", index=False)
    print(f"phase=features complete rows={len(features)}", flush=True)

    print("phase=selection start", flush=True)
    ledgers = build_rule_ledgers(features, strategy, plan.signal_sessions)
    ledgers["score"].to_parquet(output / "ledgers" / "score_ledger.parquet", index=False)
    ledgers["candidate"].to_parquet(output / "ledgers" / "candidate_ledger.parquet", index=False)
    ledgers["selection"].to_parquet(output / "ledgers" / "selection_ledger.parquet", index=False)
    selection_summary = _selection_summary(ledgers)
    ranking_missing = ledgers["score"].loc[
        ledgers["score"]["stage1_pass"].astype(bool)
        & ~ledgers["score"]["ranking_feature_ready"].astype(bool),
        ["asof", "ts_code", "score_rejection_reason"],
    ].copy()
    ranking_missing.to_parquet(
        output / "diagnostics" / "ranking_enrichment_missing.parquet", index=False
    )
    _write_json(output / "diagnostics" / "ranking_enrichment_missing_summary.json", {
        "rows": int(len(ranking_missing)),
        "dates": int(ranking_missing["asof"].nunique()) if not ranking_missing.empty else 0,
        "symbols": int(ranking_missing["ts_code"].nunique()) if not ranking_missing.empty else 0,
        "rejection_counts": {
            str(key): int(value)
            for key, value in ranking_missing["score_rejection_reason"].value_counts().items()
        },
    })
    _write_json(output / "diagnostics" / "selection_summary.json", selection_summary)
    _write_strategy_diagnostics(
        output=output,
        features=features,
        ledgers=ledgers,
        strategy=strategy,
        signal_sessions=plan.signal_sessions,
    )
    print(
        f"phase=selection complete stage1={selection_summary['stage1_pass_rows']} "
        f"candidate_view={selection_summary['candidate_view_rows']}",
        flush=True,
    )

    portfolio = _run_scenarios(
        root=output,
        ledgers=ledgers,
        bundle=bundle,
        strategy=strategy,
        plan=plan.to_dict(),
        selection_summary=selection_summary,
    )
    if strategy.strategy_id == "reset_weak_confirm_v3_cap1_20":
        _write_capacity_audit(output, bundle, strategy)

    _write_json(output / "PORTFOLIO_SUMMARY.json", portfolio)
    if control_portfolio is not None:
        _write_json(
            output / "diagnostics" / "same_bundle_control_comparison.json",
            _same_bundle_comparison(bundle.bundle_id, control_portfolio, portfolio),
        )
    _write_result(output, plan.to_dict(), bundle.manifest, selection_summary, portfolio, strategy)
    verification = _verify_preseal(
        output, plan.to_dict(), bundle.availability, bundle.execution, features, ledgers, strategy
    )
    _write_json(output / "diagnostics" / "verification.json", verification)
    code_paths = [
        ROOT / "src/aistock9988/configuration.py",
        ROOT / "src/aistock9988/planning.py",
        ROOT / "src/aistock9988/data/availability.py",
        ROOT / "src/aistock9988/data/bundle.py",
        ROOT / "src/aistock9988/features/engine.py",
        ROOT / "src/aistock9988/selection/pipeline.py",
        ROOT / "src/aistock9988/backtest/engine.py",
        ROOT / "src/aistock9988/reporting/metrics.py",
        Path(__file__).resolve(),
    ]
    _write_json(output / "manifests" / "code_manifest.json", {
        str(path.relative_to(ROOT)): _sha(path) for path in code_paths
    })
    status = json.loads((output / "RUN_STATUS.json").read_text(encoding="utf-8"))
    status.update({
        "status": "DIAGNOSTIC_COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "bundle_id": bundle.bundle_id,
        "base_acceptance_passed": bool(portfolio["base"]["acceptance"]["passed"]),
        "stress_acceptance_passed": bool(portfolio["stress"]["acceptance"]["passed"]),
        "overall_acceptance_passed": bool(
            portfolio["base"]["acceptance"]["passed"]
            and portfolio["stress"]["acceptance"]["passed"]
        ),
        "verification_passed": bool(verification["passed"]),
        "same_bundle_control_included": control_portfolio is not None,
    })
    _write_json(output / "RUN_STATUS.json", status, replace=True)
    artifacts = {
        str(path.relative_to(output)): {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json(output / "manifests" / "artifact_manifest.json", artifacts)
    return output


def _validate_control_contract(
    strategy: StrategyConfig,
    control: StrategyConfig,
) -> None:
    strategy_dict = strategy.to_dict()
    control_dict = control.to_dict()
    capacity_comparison = (
        strategy.strategy_id == "reset_weak_confirm_v3_cap1_20"
        and control.strategy_id == "reset_weak_confirm_v3"
    )
    frozen_sections = (
        "universe", "decision", "features", "stage1", "ranking", "execution", "acceptance"
    ) if capacity_comparison else (
        "universe", "decision", "stage1", "portfolio", "execution", "acceptance"
    )
    changed = [
        section for section in frozen_sections
        if strategy_dict[section] != control_dict[section]
    ]
    if changed:
        raise ValueError(
            "same-bundle control changes frozen contract sections: " + ", ".join(changed)
        )
    if capacity_comparison:
        challenger_portfolio = dict(strategy_dict["portfolio"])
        control_portfolio = dict(control_dict["portfolio"])
        challenger_sizing = dict(challenger_portfolio.pop("sizing"))
        control_sizing = dict(control_portfolio.pop("sizing"))
        challenger_cap = challenger_portfolio.pop("target_gross_exposure_cap")
        control_cap = control_portfolio.pop("target_gross_exposure_cap")
        if challenger_portfolio != control_portfolio:
            raise ValueError("capacity challenger changes portfolio fields beyond sizing and gross cap")
        expected = (
            challenger_sizing.get("method") == control_sizing.get("method")
            == "fixed_fraction_of_decision_nav"
            and float(challenger_sizing.get("value", 0.0)) == 0.20
            and float(control_sizing.get("value", 0.0)) == 0.10
            and float(challenger_cap) == 1.0
            and float(control_cap) == 0.50
        )
        if not expected:
            raise ValueError("capacity challenger must be exactly 20%/100% versus 10%/50% control")


def _write_capacity_audit(root: Path, bundle: Any, strategy: StrategyConfig) -> None:
    """Persist actual exposure and ADV use for the one-shot capacity experiment."""
    multiplier = float(strategy.execution["amount_unit_multiplier"])
    scenarios: dict[str, Any] = {}
    for scenario in ("base", "stress"):
        target = root / "backtests" / scenario
        nav = pd.read_parquet(target / "nav.parquet")
        fills = pd.read_parquet(target / "fills.parquet")
        orders = pd.read_parquet(target / "orders.parquet")
        decisions = pd.read_parquet(target / "execution_decisions.parquet")
        buy_fills = fills[fills["side"].eq("BUY")].copy()
        buy_orders = orders[
            orders["side"].eq("BUY") & orders["status"].eq("FILLED")
        ][["decision_id", "ts_code", "decision_session"]]
        buy_fills = buy_fills.merge(
            buy_orders, on=["decision_id", "ts_code"], how="left", validate="one_to_one"
        )
        adv = bundle.execution[["trade_date", "ts_code", "adv20_amount"]].rename(
            columns={"trade_date": "decision_session"}
        )
        buy_fills["decision_session"] = pd.to_datetime(
            buy_fills["decision_session"], errors="raise", utc=True
        ).dt.normalize()
        adv["decision_session"] = pd.to_datetime(
            adv["decision_session"], errors="raise", utc=True
        ).dt.normalize()
        buy_fills["ts_code"] = buy_fills["ts_code"].astype(str).str.upper()
        adv["ts_code"] = adv["ts_code"].astype(str).str.upper()
        buy_fills = buy_fills.merge(
            adv, on=["decision_session", "ts_code"], how="left", validate="many_to_one"
        )
        adv_value = pd.to_numeric(buy_fills["adv20_amount"], errors="coerce") * multiplier
        gross_value = pd.to_numeric(buy_fills["gross_value"], errors="coerce")
        invalid_adv = ~np.isfinite(adv_value) | adv_value.le(0)
        invalid_gross = ~np.isfinite(gross_value) | gross_value.le(0)
        participation = gross_value / adv_value
        finite_participation = participation[np.isfinite(participation)]
        participation_limit = float(strategy.execution["adv20_max_participation"])
        adv_limit_breaches = int((finite_participation > participation_limit + 1e-12).sum())
        cash_min = float(nav["cash"].min())
        scenarios[scenario] = {
            "target_weight_each": float(strategy.portfolio["sizing"]["value"]),
            "target_gross_cap": float(strategy.portfolio["target_gross_exposure_cap"]),
            "actual_gross_exposure_mean": float(nav["gross_exposure"].mean()),
            "actual_gross_exposure_max": float(nav["gross_exposure"].max()),
            "actual_open_positions_max": int(nav["open_positions"].max()),
            "cash_min": cash_min,
            "buy_fill_count": int(len(buy_fills)),
            "buy_fills_missing_valid_adv": int(invalid_adv.sum()),
            "buy_fills_missing_valid_gross_value": int(invalid_gross.sum()),
            "adv_participation_median": (
                float(finite_participation.median()) if not finite_participation.empty else None
            ),
            "adv_participation_p95": (
                float(finite_participation.quantile(0.95)) if not finite_participation.empty else None
            ),
            "adv_participation_max": (
                float(finite_participation.max()) if not finite_participation.empty else None
            ),
            "adv_participation_limit": participation_limit,
            "adv_limit_breaches": adv_limit_breaches,
            "entry_rejection_counts": {
                str(key): int(value)
                for key, value in decisions.loc[
                    ~decisions["chosen"].astype(bool), "reject_reason"
                ].value_counts().sort_index().items()
            },
        }
        failures: list[str] = []
        if invalid_adv.any():
            failures.append(f"{scenario}:buy_fill_missing_valid_adv")
        if invalid_gross.any():
            failures.append(f"{scenario}:buy_fill_missing_valid_gross_value")
        if adv_limit_breaches:
            failures.append(f"{scenario}:adv_limit_breach")
        if cash_min < -1e-6:
            failures.append(f"{scenario}:negative_cash")
        scenarios[scenario]["blocking_failures"] = failures
    all_failures = [
        failure for scenario in scenarios.values() for failure in scenario["blocking_failures"]
    ]
    _write_json(root / "diagnostics" / "capacity_audit.json", {
        "passed": not all_failures,
        "blocking_failures": all_failures,
        "scenarios": scenarios,
    })
    if all_failures:
        raise AssertionError("capacity audit failed: " + ", ".join(all_failures))


def _run_scenarios(
    *,
    root: Path,
    ledgers: dict[str, pd.DataFrame],
    bundle: Any,
    strategy: StrategyConfig,
    plan: dict[str, Any],
    selection_summary: dict[str, Any],
    log_prefix: str = "",
) -> dict[str, dict[str, Any]]:
    portfolio: dict[str, dict[str, Any]] = {}
    for scenario in ("base", "stress"):
        print(f"phase={log_prefix}backtest scenario={scenario} start", flush=True)
        result = run_backtest(
            candidate_ledger=ledgers["candidate"],
            selection_ledger=ledgers["selection"],
            execution_panel=bundle.execution,
            corporate_actions=bundle.corporate_actions,
            strategy=strategy,
            execution_sessions=tuple(plan["execution_sessions"]),
            scenario_name=scenario,
        )
        target = root / "backtests" / scenario
        target.mkdir()
        for name, frame in result.items():
            frame.to_parquet(target / f"{name}.parquet", index=False)
        metrics = summarize(
            result["nav"],
            result["fills"],
            initial_cash=float(strategy.execution["initial_cash"]),
            positions=result["positions"],
            corporate_actions=result["corporate_actions"],
        )
        metrics.update({
            "scenario": scenario,
            "entry_attempts": int(len(result["execution_decisions"])),
            "entry_fills": int(result["execution_decisions"]["chosen"].sum()) if not result["execution_decisions"].empty else 0,
            "entry_rejection_counts": (
                {str(key): int(value) for key, value in result["execution_decisions"].loc[
                    ~result["execution_decisions"]["chosen"], "reject_reason"
                ].value_counts().sort_index().items()}
                if not result["execution_decisions"].empty else {}
            ),
            "open_positions_at_end": int(len(result["open_positions"])),
            "active_signal_days": int(selection_summary["active_signal_days"]),
            "zero_candidate_days": int(selection_summary["zero_candidate_days"]),
            "market_coverage_fail_days": int(selection_summary["market_coverage_fail_days"]),
            "universe_warmup_days": int(selection_summary["universe_warmup_days"]),
            "feature_warmup_days": int(selection_summary["feature_warmup_days"]),
        })
        metrics["acceptance"] = _acceptance(metrics, strategy)
        _write_json(target / "metrics.json", metrics)
        portfolio[scenario] = metrics
        print(
            f"phase={log_prefix}backtest scenario={scenario} "
            f"return={metrics['total_return']:+.6f} "
            f"pf={metrics['portfolio_profit_factor']} "
            f"maxdd={metrics['max_drawdown']:+.6f}",
            flush=True,
        )
    return portfolio


def _same_bundle_comparison(
    bundle_id: str,
    control: dict[str, dict[str, Any]],
    challenger: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fields = (
        "total_return", "portfolio_profit_factor", "max_drawdown",
        "trade_win_rate", "return_excluding_best_week",
        "return_excluding_top3_profit", "trade_count",
    )
    scenarios: dict[str, Any] = {}
    for scenario in ("base", "stress"):
        scenarios[scenario] = {
            field: {
                "control": control[scenario].get(field),
                "challenger": challenger[scenario].get(field),
                "delta": (
                    float(challenger[scenario][field]) - float(control[scenario][field])
                    if control[scenario].get(field) is not None
                    and challenger[scenario].get(field) is not None
                    else None
                ),
            }
            for field in fields
        }
    return {"bundle_id": bundle_id, "scenarios": scenarios}


def _selection_summary(ledgers: dict[str, pd.DataFrame]) -> dict[str, Any]:
    score = ledgers["score"]
    candidate = ledgers["candidate"]
    daily = candidate.groupby("asof", sort=True).agg(
        feature_ready=("feature_ready", "sum"),
        stage1_pass=("stage1_pass", "sum"),
        candidate_view=("candidate_status", lambda values: int(values.eq("IN_VIEW").sum())),
    )
    return {
        "signal_dates": int(score["asof"].nunique()),
        "scored_rows": int(len(score)),
        "feature_ready_rows": int(score["feature_ready"].sum()),
        "selection_data_excluded_rows": int((~score["selection_data_eligible"]).sum()),
        "training_data_excluded_rows": int((~score["training_data_eligible"]).sum()),
        "stage1_pass_rows": int(score["stage1_pass"].sum()),
        "stage1_ranking_data_missing_rows": int(
            (score["stage1_pass"] & ~score["ranking_feature_ready"]).sum()
        ),
        "candidate_view_rows": int(candidate["candidate_status"].eq("IN_VIEW").sum()),
        "daily_min_stage1": int(daily["stage1_pass"].min()),
        "daily_median_stage1": float(daily["stage1_pass"].median()),
        "daily_max_stage1": int(daily["stage1_pass"].max()),
        "active_signal_days": int((daily["stage1_pass"] > 0).sum()),
        "zero_candidate_days": int((daily["candidate_view"] == 0).sum()),
        "market_coverage_fail_days": int(
            (
                score.groupby("asof", sort=True)
                .agg(market_coverage=("market_coverage", "first"), universe_pass_rows=("universe_pass", "sum"))
                .pipe(lambda daily: daily["universe_pass_rows"].gt(0) & (daily["market_coverage"].lt(0.80) | daily["market_coverage"].isna()))
            ).sum()
        ),
        "universe_warmup_days": int(
            score.groupby("asof", sort=True)["universe_pass"].sum().eq(0).sum()
        ),
        "feature_warmup_days": int(
            (
                score.groupby("asof", sort=True)
                .agg(
                    feature_ready_rows=("feature_ready", "sum"),
                    universe_pass_rows=("universe_pass", "sum"),
                    selection_eligible_rows=("selection_data_eligible", "sum"),
                    market_coverage=("market_coverage", "first"),
                )
                .pipe(
                    lambda daily: (
                        daily["universe_pass_rows"].gt(0)
                        & daily["selection_eligible_rows"].gt(0)
                        & daily["feature_ready_rows"].eq(0)
                        & daily["market_coverage"].ge(0.80)
                    )
                )
            ).sum()
        ),
    }


def _write_strategy_diagnostics(
    *,
    output: Path,
    features: pd.DataFrame,
    ledgers: dict[str, pd.DataFrame],
    strategy: StrategyConfig,
    signal_sessions: tuple[str, ...],
) -> None:
    if strategy.strategy_id != "participation_impulse_preconfirmation_v1":
        return
    signal_days = pd.DatetimeIndex(pd.to_datetime(signal_sessions, utc=True)).normalize()
    frame = features[features["asof"].isin(signal_days)].copy()
    ready = frame["feature_ready"].astype(bool)
    sequential = pd.Series(True, index=frame.index)
    condition_rows: list[dict[str, Any]] = []
    conditions = list(strategy.stage1["expression"]["all"])
    for index, condition in enumerate(conditions, start=1):
        passed = evaluate_expression(frame, condition)
        sequential &= passed
        condition_rows.append({
            "order": index,
            "condition": dict(condition),
            "standalone_pass_rows": int((ready & passed).sum()),
            "sequential_pass_rows": int((ready & sequential).sum()),
        })
    _write_json(output / "diagnostics" / "stage1_condition_attrition.json", {
        "signal_rows": int(len(frame)),
        "universe_pass_rows": int(frame["universe_pass"].sum()),
        "selection_eligible_rows": int(frame["selection_data_eligible"].sum()),
        "feature_ready_rows": int(ready.sum()),
        "conditions": condition_rows,
    })

    impulse = pd.to_numeric(frame["turnover_impulse"], errors="coerce")
    selected_impulse = pd.to_numeric(
        frame.loc[ledgers["score"]["stage1_pass"].to_numpy(dtype=bool), "turnover_impulse"],
        errors="coerce",
    )
    quantiles = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    _write_json(output / "diagnostics" / "turnover_impulse_distribution.json", {
        "finite_signal_rows": int(np.isfinite(impulse).sum()),
        "signal_quantiles": {
            str(value): float(impulse.quantile(value)) for value in quantiles
            if impulse.notna().any()
        },
        "stage1_rows": int(selected_impulse.notna().sum()),
        "stage1_quantiles": {
            str(value): float(selected_impulse.quantile(value)) for value in quantiles
            if selected_impulse.notna().any()
        },
    })

    turnover_f = pd.to_numeric(frame["turnover_rate_f"], errors="coerce")
    turnover = pd.to_numeric(frame["turnover_rate"], errors="coerce")
    missing = frame.loc[
        frame["universe_pass"].astype(bool)
        & frame["selection_data_eligible"].astype(bool)
        & turnover_f.isna()
        & turnover.isna(),
        ["asof", "ts_code"],
    ].copy()
    missing["rejection_reason"] = "TURNOVER_RATE_F_AND_TURNOVER_RATE_MISSING"
    missing.to_parquet(
        output / "diagnostics" / "turnover_input_missing.parquet", index=False
    )
    _write_json(output / "diagnostics" / "turnover_input_missing_summary.json", {
        "rows": int(len(missing)),
        "dates": int(missing["asof"].nunique()) if not missing.empty else 0,
        "symbols": int(missing["ts_code"].nunique()) if not missing.empty else 0,
    })


def _passthrough_coverage_audit(
    bundle: Any,
    strategy: StrategyConfig,
    plan: dict[str, Any],
) -> dict[str, Any]:
    registered: dict[str, list[str]] = {}
    for feature_name, spec in strategy.features.items():
        if not hasattr(spec, "get") or str(spec.get("provider", "")) != "passthrough":
            continue
        source = str(spec.get("source", ""))
        if "." not in source:
            continue
        enrichment_name, _ = source.split(".", 1)
        registered.setdefault(f"{enrichment_name}_ts", []).append(str(feature_name))
    if not registered:
        return {
            "passed": True,
            "minimum_daily_coverage": 0.80,
            "sources": {},
            "failed_dates": [],
        }

    signal_days = pd.DatetimeIndex(pd.to_datetime(plan["signal_sessions"], utc=True)).normalize()
    panel = bundle.execution[bundle.execution["trade_date"].isin(signal_days)].copy()
    minimum = 0.80
    failed_dates: list[dict[str, Any]] = []
    source_audits: dict[str, Any] = {}
    for source_name, features in sorted(registered.items()):
        presence_column = f"has_{source_name}"
        if presence_column not in panel:
            raise ValueError(f"passthrough source presence column missing: {presence_column}")
        eligible = panel[
            panel["universe_pass"].astype(bool)
            & panel["selection_data_eligible"].astype(bool)
        ].copy()
        coverage_basis = "source_row_present"
        if (
            strategy.strategy_id == "participation_impulse_preconfirmation_v1"
            and source_name == "daily_basic_ts"
        ):
            daily_basic = bundle.enrichments.get("daily_basic", pd.DataFrame()).copy()
            required = {"trade_date", "ts_code", "turnover_rate_f", "turnover_rate"}
            missing = sorted(required - set(daily_basic.columns))
            if missing:
                raise ValueError(f"PIPC daily_basic coverage columns missing: {missing}")
            daily_basic["trade_date"] = pd.to_datetime(
                daily_basic["trade_date"], errors="raise", utc=True
            ).dt.normalize()
            daily_basic["ts_code"] = daily_basic["ts_code"].astype(str).str.upper()
            if daily_basic.duplicated(["trade_date", "ts_code"]).any():
                raise ValueError("PIPC daily_basic coverage contains duplicate keys")
            turnover_f = pd.to_numeric(daily_basic["turnover_rate_f"], errors="coerce")
            turnover = pd.to_numeric(daily_basic["turnover_rate"], errors="coerce")
            turnover_raw = turnover_f.where(turnover_f.notna(), turnover)
            valid_keys = pd.MultiIndex.from_frame(
                daily_basic.loc[
                    np.isfinite(turnover_raw) & turnover_raw.gt(0),
                    ["trade_date", "ts_code"],
                ]
            )
            eligible[presence_column] = pd.MultiIndex.from_frame(
                eligible[["trade_date", "ts_code"]]
            ).isin(valid_keys)
            coverage_basis = "turnover_rate_f_else_turnover_rate_finite_positive"
        daily = eligible.groupby("trade_date", sort=True).agg(
            eligible_rows=("ts_code", "size"),
            present_rows=(presence_column, "sum"),
        ).reindex(signal_days)
        daily["coverage"] = daily["present_rows"] / daily["eligible_rows"]
        failed = daily[
            daily["eligible_rows"].fillna(0).le(0)
            | daily["coverage"].isna()
            | daily["coverage"].lt(minimum)
        ]
        failed_dates.extend({
            "source": source_name,
            "trade_date": str(day.date()),
            "eligible_rows": int(row["eligible_rows"]) if pd.notna(row["eligible_rows"]) else 0,
            "present_rows": int(row["present_rows"]) if pd.notna(row["present_rows"]) else 0,
            "coverage": float(row["coverage"]) if pd.notna(row["coverage"]) else None,
        } for day, row in failed.iterrows())
        source_audits[source_name] = {
            "features": sorted(features),
            "coverage_basis": coverage_basis,
            "availability_policy": strategy.data_policy.get("source_availability", {}).get(source_name),
            "update_time_provenance": bundle.manifest.get("source_update_time_summary", {}).get(
                source_name.removesuffix("_ts"), {}
            ),
            "minimum_coverage": float(daily["coverage"].min()) if daily["coverage"].notna().any() else None,
            "median_coverage": float(daily["coverage"].median()) if daily["coverage"].notna().any() else None,
            "failed_date_count": int(len(failed)),
        }
    return {
        "passed": not failed_dates,
        "minimum_daily_coverage": minimum,
        "sources": source_audits,
        "failed_dates": failed_dates,
    }


def _acceptance(metrics: dict[str, Any], strategy: StrategyConfig) -> dict[str, Any]:
    pf = metrics["portfolio_profit_factor"]
    minimum_closed_trades = int(strategy.acceptance.get("minimum_closed_trades", 0))
    minimum_active_signal_days = int(strategy.acceptance.get("minimum_active_signal_days", 0))
    tests = {
        "profit_factor": pf is not None and float(pf) >= float(strategy.acceptance["portfolio_profit_factor_min"]),
        "max_drawdown": abs(float(metrics["max_drawdown"])) <= float(strategy.acceptance["max_drawdown_abs_max"]),
        "excluding_best_week": float(metrics["return_excluding_best_week"]) > float(strategy.acceptance["return_excluding_best_week_min_exclusive"]),
        "excluding_top3_profit": float(metrics["return_excluding_top3_profit"]) > 0.0,
        "minimum_closed_trades": int(metrics["trade_count"]) >= minimum_closed_trades,
        "minimum_active_signal_days": int(metrics.get("active_signal_days", 0)) >= minimum_active_signal_days,
    }
    return {"passed": all(tests.values()), "tests": tests}


def _write_result(
    output: Path,
    plan: dict[str, Any],
    data_manifest: dict[str, Any],
    selection: dict[str, Any],
    portfolio: dict[str, dict[str, Any]],
    strategy: StrategyConfig,
) -> None:
    strategy_dict = strategy.to_dict()
    stage1_expression = json.dumps(strategy_dict["stage1"]["expression"], sort_keys=True, default=str)
    entries = int(strategy.portfolio["entries_per_decision"])
    max_positions = int(strategy.portfolio["max_open_positions"])
    weight = float(strategy.portfolio["sizing"]["value"])
    hold = int(strategy.execution["hold_sessions_from_fill"])
    lines = [
        f"# {plan['strategy_id']} V3 Full-Universe 2026", "",
        "FORWARD_ONLY_DIAGNOSTIC: this historical replay is explicitly seen history and is not a locked out-of-sample claim.", "",
        "## Contract", "",
        f"- Signal range: `{plan['signal_start']}` to `{plan['signal_end']}`; execution through `{plan['execution_end']}`.",
        f"- Bundle: `{data_manifest['bundle_id']}`; configured full-universe codes: `{data_manifest['configured_codes']}`.",
        f"- Rules only: full universe -> Stage1 expression `{stage1_expression}` -> frozen daily Top{int(strategy.portfolio['candidate_view_size'])} -> Top{entries}/next-ranked fallback.",
        f"- T+1 raw open entry, {weight:.2%} of prior-close NAV per name, maximum {max_positions} positions, H{hold}, -8% close-trigger/next-open stop.",
        "- No threshold sweep and no XGBoost model.", "",
        "## Selection", "",
        f"- Signal dates: `{selection['signal_dates']}`; feature-ready rows: `{selection['feature_ready_rows']}`.",
        f"- Required-data exclusions on signal dates: selection `{selection['selection_data_excluded_rows']}`, training `{selection['training_data_excluded_rows']}`.",
        f"- Stage1 pass rows: `{selection['stage1_pass_rows']}`; frozen candidate-view rows: `{selection['candidate_view_rows']}`.", "",
        "## Portfolio", "",
        "| Cost | Return | PF | MaxDD | Ex-best-week | Trades | End open | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("base", "stress"):
        metrics = portfolio[name]
        pf = "NA" if metrics["portfolio_profit_factor"] is None else f"{metrics['portfolio_profit_factor']:.3f}"
        lines.append(
            f"| {name} | {metrics['total_return']:+.2%} | {pf} | {metrics['max_drawdown']:.2%} | "
            f"{metrics['return_excluding_best_week']:+.2%} | {metrics['trade_count']} | "
            f"{metrics['open_positions_at_end']} | {metrics['acceptance']['passed']} |"
        )
    lines.extend(["", "## Decision", ""])
    lines.append(
        "The strategy advances only when PF>=2, MaxDD<=15%, and return excluding the best week remains positive. "
        "A negative result is retained unchanged as evidence; it is not repaired with a threshold scan."
    )
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_preseal(
    output: Path,
    plan: dict[str, Any],
    availability: pd.DataFrame,
    execution: pd.DataFrame,
    features: pd.DataFrame,
    ledgers: dict[str, pd.DataFrame],
    strategy: StrategyConfig,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["availability_unique"] = not availability.duplicated(["trade_date", "ts_code"]).any()
    checks["execution_unique"] = not execution.duplicated(["trade_date", "ts_code"]).any()
    available = pd.to_datetime(features["available_time"], errors="raise", utc=True)
    cutoff = pd.to_datetime(features["asof"], errors="raise", utc=True).map(session_close)
    checks["feature_pit"] = bool((available <= cutoff).all())
    market_available = pd.to_datetime(features["market_available_time"], errors="coerce", utc=True)
    checks["market_feature_pit"] = bool(
        market_available.isna().all() or (market_available.fillna(cutoff) <= cutoff).all()
    )
    checks["feature_unique"] = not features.duplicated(["asof", "ts_code"]).any()
    checks["feature_ready_requires_selection_data"] = bool(
        (~features["feature_ready"] | features["selection_data_eligible"]).all()
    )
    checks["score_unique"] = not ledgers["score"].duplicated(["asof", "ts_code"]).any()
    in_view = ledgers["candidate"][ledgers["candidate"]["candidate_status"].eq("IN_VIEW")]
    daily_view = in_view.groupby("asof", sort=True).size()
    checks["candidate_view_cap"] = bool((daily_view <= int(strategy.portfolio["candidate_view_size"])).all())
    if "candidate_min_daily_view_ratio" not in strategy.portfolio:
        raise ValueError("strategy portfolio must declare candidate_min_daily_view_ratio")
    min_ratio = float(strategy.portfolio["candidate_min_daily_view_ratio"])
    if not 0.0 < min_ratio <= 1.0:
        raise ValueError("candidate_min_daily_view_ratio must be in (0, 1]")
    min_rows = int(np.ceil(int(strategy.portfolio["candidate_view_size"]) * min_ratio))
    signal_days = pd.DatetimeIndex(pd.to_datetime(plan["signal_sessions"], utc=True)).normalize()
    counts = daily_view.reindex(signal_days, fill_value=0)
    observed_ratio = float((counts >= min_rows).mean())
    sparse_candidate_view = bool(strategy.portfolio.get("allow_sparse_candidate_view", False))
    candidate_frame = ledgers["candidate"].copy()
    daily_audit = candidate_frame.groupby("asof", sort=True).agg(
        feature_ready_rows=("feature_ready", "sum"),
        selection_eligible_rows=("selection_data_eligible", "sum"),
        universe_pass_rows=("universe_pass", "sum"),
        stage1_pass_rows=("stage1_pass", "sum"),
        candidate_view_rows=("candidate_status", lambda values: int(values.eq("IN_VIEW").sum())),
        market_coverage=("market_coverage", "first"),
    ).reindex(signal_days, fill_value=0)
    zero_candidate_days = [str(day.date()) for day, row in daily_audit.iterrows() if int(row.candidate_view_rows) == 0]
    # A data-gap day has no usable feature/selection rows at all.  It must
    # remain blocking even for a sparse event strategy; individual missing
    # stocks are already excluded at stock-session granularity.
    # A feature warmup (for example, an immature MA60 window) is not a data
    # gap when the underlying selection panel is present and market coverage
    # is healthy. Only a day with no selection-eligible rows is a true gap.
    data_gap_days = [
        str(day.date()) for day, row in daily_audit.iterrows()
        if int(row.selection_eligible_rows) == 0
    ]
    warmup_days = [
        str(day.date()) for day, row in daily_audit.iterrows()
        if int(row.universe_pass_rows) == 0
    ]
    market_coverage_fail_days = [
        str(day.date()) for day, row in daily_audit.iterrows()
        if int(row.universe_pass_rows) > 0
        and (not np.isfinite(float(row.market_coverage)) or float(row.market_coverage) < 0.80)
    ]
    feature_warmup_days = [
        str(day.date()) for day, row in daily_audit.iterrows()
        if int(row.universe_pass_rows) > 0
        and int(row.selection_eligible_rows) > 0
        and int(row.feature_ready_rows) == 0
        and np.isfinite(float(row.market_coverage))
        and float(row.market_coverage) >= 0.80
    ]
    invalid_days = set(data_gap_days) | set(market_coverage_fail_days)
    warmup_all = set(warmup_days) | set(feature_warmup_days)
    abstained_days = [
        day for day in zero_candidate_days if day not in invalid_days and day not in warmup_all
    ]
    active_signal_days = int((daily_audit["stage1_pass_rows"] > 0).sum())
    # Event-driven strategies may legitimately abstain for long periods.  The
    # observed coverage remains in the verification artifact, but only a
    # strategy that explicitly opts into sparse signals may pass this generic
    # structural check; economic acceptance gates remain unchanged.
    checks["candidate_view_minimum_coverage"] = sparse_candidate_view or observed_ratio >= min_ratio
    checks["sparse_data_gap_free"] = not data_gap_days
    checks["sparse_market_coverage_free"] = not market_coverage_fail_days
    max_feature_warmup_days = int(strategy.acceptance.get("max_feature_warmup_days", 0))
    checks["sparse_feature_warmup_bounded"] = (
        max_feature_warmup_days > 0 and len(feature_warmup_days) <= max_feature_warmup_days
    )
    checks["selection_signal_end"] = str(pd.to_datetime(ledgers["selection"]["asof"], utc=True).max().date()) == plan["signal_end"]
    sessions = pd.DatetimeIndex(pd.to_datetime(plan["execution_sessions"], utc=True)).normalize()
    for scenario in ("base", "stress"):
        target = output / "backtests" / scenario
        nav = pd.read_parquet(target / "nav.parquet")
        orders = pd.read_parquet(target / "orders.parquet")
        fills = pd.read_parquet(target / "fills.parquet")
        events = pd.read_parquet(target / "position_events.parquet")
        values = nav[["cash", "market_value", "nav"]].apply(pd.to_numeric, errors="raise")
        checks[f"{scenario}_nav_identity"] = bool(
            ((values["cash"] + values["market_value"] - values["nav"]).abs() <= 1e-8).all()
        )
        checks[f"{scenario}_cash_nonnegative"] = bool((values["cash"] >= -1e-8).all())
        checks[f"{scenario}_position_cap"] = bool(
            (pd.to_numeric(nav["open_positions"], errors="raise") <= int(strategy.portfolio["max_open_positions"])).all()
        )
        checks[f"{scenario}_execution_end"] = str(pd.to_datetime(nav["trade_date"], utc=True).max().date()) == plan["execution_end"]
        filled_orders = (
            set(orders.loc[orders["status"].eq("FILLED"), "order_id"].astype(str))
            if not orders.empty and "status" in orders.columns else set()
        )
        fill_ids = set(fills["order_id"].astype(str))
        checks[f"{scenario}_fill_order_bijection"] = filled_orders == fill_ids and not fills["order_id"].duplicated().any()
        if fills.empty:
            checks[f"{scenario}_fills_require_execution_data"] = True
        else:
            fill_keys = fills[["trade_date", "ts_code"]].merge(
                execution[["trade_date", "ts_code", "execution_data_eligible"]],
                on=["trade_date", "ts_code"],
                how="left",
                validate="many_to_one",
            )
            checks[f"{scenario}_fills_require_execution_data"] = bool(
                fill_keys["execution_data_eligible"].fillna(False).all()
            )
        held_gaps = (
            events[events["event_type"].eq("HELD_DATA_GAP")]
            if "event_type" in events else pd.DataFrame()
        )
        checks[f"{scenario}_held_gap_has_reason"] = bool(
            held_gaps.empty or held_gaps["reason"].astype(str).str.len().gt(0).all()
        )
        buys = (
            orders[orders["side"].eq("BUY") & orders["status"].eq("FILLED")]
            if not orders.empty and {"side", "status"}.issubset(orders.columns)
            else pd.DataFrame()
        )
        causal = True
        for row in buys.itertuples(index=False):
            decision = pd.Timestamp(row.decision_session)
            execution_session = pd.Timestamp(row.execution_session)
            index = sessions.get_indexer([decision])[0]
            causal &= index >= 0 and index + 1 < len(sessions) and sessions[index + 1] == execution_session
        checks[f"{scenario}_buy_t_plus_one"] = bool(causal)
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise AssertionError("pre-seal verification failed: " + ", ".join(failed))
    return {
        "passed": True,
        "checks": checks,
        "failed": failed,
        "candidate_view_coverage": {
            "required_ratio": min_ratio,
            "observed_ratio": observed_ratio,
            "minimum_rows_per_day": min_rows,
            "allow_sparse_candidate_view": sparse_candidate_view,
            "enforced_as_blocking_check": not sparse_candidate_view,
            "active_signal_days": active_signal_days,
            "zero_candidate_days": zero_candidate_days,
            "abstained_days": abstained_days,
            "data_gap_days": data_gap_days,
            "warmup_days": warmup_days,
            "feature_warmup_days": feature_warmup_days,
            "market_coverage_fail_days": market_coverage_fail_days,
            "max_feature_warmup_days": max_feature_warmup_days,
            "sparse_audit_required": bool(signal_days.size and len(zero_candidate_days) / signal_days.size > 0.50),
        },
    }


def _write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/quiet_confirmed_v1.yaml")
    parser.add_argument("--signal-start", default="2026-01-01")
    parser.add_argument("--signal-end", default="2026-08-06")
    parser.add_argument("--execution-end", default="2026-08-21")
    parser.add_argument("--run-name", default="S40_QUIET_CONFIRMED_V3_FULL_UNIVERSE_2026")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--control-strategy",
        type=Path,
        help="run a frozen control against the same in-memory data bundle",
    )
    parser.add_argument(
        "--diagnostic-history",
        action="store_true",
        help="allow a forward-only strategy only as a clearly labeled seen-history diagnostic",
    )
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
