"""Run the one-shot CAP1 entry-clustering experiment without raw artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from aistock9988.backtest.engine import run_backtest
from aistock9988.configuration import StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_source_max_dates, load_trading_calendar
from aistock9988.features.engine import build_feature_ledger
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.metrics import summarize
from aistock9988.selection.pipeline import build_rule_ledgers
from aistock9988.time.session import session_close


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs/strategy/reset_weak_confirm_v3_cap1_20.yaml"
CHALLENGER = ROOT / "configs/strategy/reset_weak_confirm_v3_cap1_staggered_one_entry_v1.yaml"
PREREG = ROOT / "docs/council_20260828/CAP1_STAGGERED_ONE_ENTRY_PER_DAY_V2_PREREG_20260902.md"
DEFAULT_OUTPUT = ROOT / "docs/council_20260828/CAP1_STAGGERED_ONE_ENTRY_PER_DAY_V2_2026_TO_DB_CUTOFF_20260902"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_contract(control: StrategyConfig, challenger: StrategyConfig) -> None:
    left, right = control.to_dict(), challenger.to_dict()
    for section in ("universe", "data_policy", "decision", "features", "stage1", "ranking", "execution", "acceptance"):
        if left[section] != right[section]:
            raise ValueError(f"challenger changes frozen section: {section}")
    control_portfolio, challenger_portfolio = dict(left["portfolio"]), dict(right["portfolio"])
    if control_portfolio.pop("entries_per_decision") != 5 or challenger_portfolio.pop("entries_per_decision") != 1:
        raise ValueError("experiment must be exactly five control entries versus one challenger entry")
    if control_portfolio != challenger_portfolio:
        raise ValueError("challenger changes portfolio fields beyond entries_per_decision")


def _selection_summary(ledgers: dict[str, pd.DataFrame]) -> dict[str, int]:
    score, candidate = ledgers["score"], ledgers["candidate"]
    daily = candidate.groupby("asof", sort=True).agg(
        stage1_pass=("stage1_pass", "sum"),
        candidate_view=("candidate_status", lambda values: int(values.eq("IN_VIEW").sum())),
    )
    return {
        "signal_dates": int(score["asof"].nunique()),
        "active_signal_days": int(daily["stage1_pass"].gt(0).sum()),
        "stage1_pass_rows": int(score["stage1_pass"].sum()),
        "candidate_view_rows": int(candidate["candidate_status"].eq("IN_VIEW").sum()),
        "selection_data_excluded_rows": int((~score["selection_data_eligible"].astype(bool)).sum()),
    }


def _metrics(result: dict[str, pd.DataFrame], strategy: StrategyConfig, selection: dict[str, int]) -> dict[str, Any]:
    metrics = summarize(result["nav"], result["fills"], initial_cash=float(strategy.execution["initial_cash"]), positions=result["positions"], corporate_actions=result["corporate_actions"])
    decisions = result["execution_decisions"]
    entries_by_session = decisions.loc[decisions["chosen"].astype(bool)].groupby("execution_session").size() if not decisions.empty else pd.Series(dtype=int)
    metrics.update({
        "entry_attempts": int(len(decisions)),
        "entry_fills": int(decisions["chosen"].sum()) if not decisions.empty else 0,
        "active_signal_days": int(selection["active_signal_days"]),
        "open_positions_at_end": int(len(result["open_positions"])),
        "maximum_new_entries_on_one_execution_session": int(entries_by_session.max()) if not entries_by_session.empty else 0,
        "sessions_with_multiple_new_entries": int(entries_by_session.gt(1).sum()),
    })
    return metrics


def _promotion(control: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for scenario in ("base", "stress"):
        base, candidate = control[scenario], challenger[scenario]
        prefix = f"{scenario}_"
        checks[prefix + "return_improves"] = candidate["total_return"] > base["total_return"]
        checks[prefix + "pf_min"] = candidate["portfolio_profit_factor"] is not None and candidate["portfolio_profit_factor"] >= 2.0
        checks[prefix + "pf_not_worse"] = candidate["portfolio_profit_factor"] >= base["portfolio_profit_factor"]
        checks[prefix + "drawdown_limit"] = abs(candidate["max_drawdown"]) <= 0.15
        checks[prefix + "drawdown_not_worse"] = abs(candidate["max_drawdown"]) <= abs(base["max_drawdown"])
        checks[prefix + "ex_best_positive"] = candidate["return_excluding_best_week"] > 0.0
        checks[prefix + "ex_best_not_worse"] = candidate["return_excluding_best_week"] >= base["return_excluding_best_week"]
        checks[prefix + "position_cap"] = candidate["max_open_positions"] <= 5
    return {"passed": all(checks.values()), "checks": checks}


def run(output: Path, signal_start: str) -> Path:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    control, challenger = StrategyConfig.from_yaml(CONTROL), StrategyConfig.from_yaml(CHALLENGER)
    _validate_contract(control, challenger)
    sources = set(control.data_policy["dense_required"]["selection"]) | set(control.data_policy["dense_required"]["execution"])
    cutoffs = load_source_max_dates(sources)
    cutoff = min(pd.Timestamp(value).date() for value in cutoffs.values()).isoformat()
    calendar = load_trading_calendar(str((pd.Timestamp(signal_start) - pd.Timedelta(days=500)).date()), cutoff)
    sessions = pd.DatetimeIndex(calendar["session"]).sort_values()
    if len(sessions) < 2:
        raise ValueError("database cutoff does not contain a T+1-executable signal session")
    request = RunRequest(signal_start=signal_start, signal_end=str(sessions[-2].date()), execution_end=cutoff, output_dir=str(output), run_name="CAP1_STAGGERED_ONE_ENTRY_PER_DAY_V1_2026_TO_DB_CUTOFF_20260902")
    plan = compile_run_plan(control, request, calendar["session"], require_complete_horizon=False)
    bundle = build_data_bundle(plan, control, output)
    features = build_feature_ledger(bundle, control)
    feature_cutoff = pd.to_datetime(features["asof"], errors="raise", utc=True).map(session_close)
    if not (pd.to_datetime(features["available_time"], errors="raise", utc=True) <= feature_cutoff).all():
        raise AssertionError("feature PIT audit failed")
    control_ledgers = build_rule_ledgers(features, control, plan.signal_sessions)
    challenger_ledgers = build_rule_ledgers(features, challenger, plan.signal_sessions)
    # The changed portfolio field must alter orders, not the opportunity set.
    if not control_ledgers["score"].equals(challenger_ledgers["score"]):
        raise AssertionError("one-entry challenger changed the shared score ledger")
    if not control_ledgers["candidate"].equals(challenger_ledgers["candidate"]):
        raise AssertionError("one-entry challenger changed the shared candidate ledger")
    selection = _selection_summary(control_ledgers)
    controls: dict[str, Any] = {}
    challengers: dict[str, Any] = {}
    for scenario in ("base", "stress"):
        control_result = run_backtest(candidate_ledger=control_ledgers["candidate"], selection_ledger=control_ledgers["selection"], execution_panel=bundle.execution, corporate_actions=bundle.corporate_actions, strategy=control, execution_sessions=plan.execution_sessions, scenario_name=scenario)
        challenger_result = run_backtest(candidate_ledger=control_ledgers["candidate"], selection_ledger=challenger_ledgers["selection"], execution_panel=bundle.execution, corporate_actions=bundle.corporate_actions, strategy=challenger, execution_sessions=plan.execution_sessions, scenario_name=scenario)
        controls[scenario] = _metrics(control_result, control, selection)
        challengers[scenario] = _metrics(challenger_result, challenger, selection)
    comparison = _promotion(controls, challengers)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "RUN_STATUS.json", {"status": "COMPLETED_ACCEPT" if comparison["passed"] else "COMPLETED_REJECT", "completed_at": datetime.now(timezone.utc).isoformat(), "raw_business_data_persisted": False, "years_used_for_performance": [2026], "credentials_persisted": False})
    _write_json(output / "plan.json", plan.to_dict())
    _write_json(output / "data_manifest.json", bundle.manifest)
    _write_json(output / "selection_summary.json", selection)
    _write_json(output / "control_metrics.json", controls)
    _write_json(output / "challenger_metrics.json", challengers)
    _write_json(output / "promotion.json", comparison)
    _write_json(output / "code_manifest.json", {str(path.relative_to(ROOT)): _sha(path) for path in (CONTROL, CHALLENGER, PREREG, ROOT / "src/aistock9988/backtest/engine.py", ROOT / "src/aistock9988/features/engine.py", Path(__file__).resolve())})
    decision = "ACCEPT" if comparison["passed"] else "REJECT"
    rows = []
    for scenario in ("base", "stress"):
        for label, metrics in (("Control", controls[scenario]), ("One-entry challenger", challengers[scenario])):
            pf = "NA" if metrics["portfolio_profit_factor"] is None else f"{metrics['portfolio_profit_factor']:.3f}"
            rows.append(f"| {scenario} | {label} | {metrics['total_return']:+.2%} | {pf} | {metrics['max_drawdown']:.2%} | {metrics['trade_win_rate']:.1%} | {metrics['return_excluding_best_week']:+.2%} | {metrics['weekly_ge_5_count']} | {metrics['trade_count']} | {metrics['max_open_positions']} | {metrics['maximum_new_entries_on_one_execution_session']} |")
    (output / "RESULT.md").write_text("\n".join([
        "# CAP1 Staggered One Entry per Day V2", "", f"Status: `{decision}`. This is a paired, seen-2026 historical replay through the current database cutoff, not a forward claim.", "", "## Integrity", "", f"- Signal range: `{plan.signal_start}` through `{plan.signal_end}`; execution and marks through `{plan.execution_end}`.", "- Control and challenger used the same in-memory QuantDB bundle, feature ledger, candidate ledger, execution panel, and corporate actions.", "- PIT feature audit passed. No raw market rows, factors, candidates, fills, positions, model, CSV, or Parquet output was retained.", "", "## Portfolio", "", "| Cost | Strategy | Return | PF | MaxDD | Win rate | Ex-best-week | Weeks >=5% | Trades | Max positions | Max same-day entries |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|", *rows, "", "## Decision", "", f"Promotion passed: `{comparison['passed']}`. The challenger changed only `entries_per_decision` from 5 to 1. If rejected, this exact rule is closed without trying 2/3/4-entry variants.", ""]) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--signal-start", default="2026-01-01")
    args = parser.parse_args()
    print(run(args.output.resolve(), args.signal_start))


if __name__ == "__main__":
    main()
