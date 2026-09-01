"""Run one rules strategy from QuantDB while persisting aggregate evidence only."""
from __future__ import annotations

import argparse
import hashlib
import json
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
from aistock9988.features.engine import build_feature_ledger
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.metrics import summarize
from aistock9988.selection.pipeline import build_rule_ledgers
from aistock9988.time.session import session_close


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _selection_summary(
    features: pd.DataFrame,
    ledgers: dict[str, pd.DataFrame],
    signal_sessions: tuple[str, ...],
) -> dict[str, Any]:
    signal_days = pd.DatetimeIndex(pd.to_datetime(signal_sessions, utc=True)).normalize()
    score = ledgers["score"]
    candidate = ledgers["candidate"]
    signal_features = features[features["asof"].isin(signal_days)]
    daily = candidate.groupby("asof", sort=True).agg(
        stage1_pass=("stage1_pass", "sum"),
        candidate_view=("candidate_status", lambda values: int(values.eq("IN_VIEW").sum())),
    ).reindex(signal_days, fill_value=0)
    selection_eligible = signal_features.groupby("asof", sort=True)[
        "selection_data_eligible"
    ].sum().reindex(signal_days, fill_value=0)
    return {
        "signal_dates": int(len(signal_days)),
        "score_signal_dates": int(score["asof"].nunique()),
        "active_signal_days": int(daily["stage1_pass"].gt(0).sum()),
        "zero_candidate_days": int(daily["candidate_view"].eq(0).sum()),
        "stage1_pass_rows": int(score["stage1_pass"].sum()),
        "candidate_view_rows": int(candidate["candidate_status"].eq("IN_VIEW").sum()),
        "selection_data_excluded_rows": int((~score["selection_data_eligible"]).sum()),
        "date_level_data_gap_days": [
            str(day.date()) for day, value in selection_eligible.items() if int(value) == 0
        ],
    }


def _acceptance(metrics: dict[str, Any], strategy: StrategyConfig) -> dict[str, Any]:
    rules = strategy.acceptance
    tests = {
        "profit_factor": metrics["portfolio_profit_factor"] is not None
        and float(metrics["portfolio_profit_factor"])
        >= float(rules["portfolio_profit_factor_min"]),
        "max_drawdown": abs(float(metrics["max_drawdown"]))
        <= float(rules["max_drawdown_abs_max"]),
        "excluding_best_week": float(metrics["return_excluding_best_week"])
        > float(rules["return_excluding_best_week_min_exclusive"]),
        "excluding_top3": float(metrics["return_excluding_top3_profit"]) > 0.0,
        "trade_win_rate": metrics["trade_win_rate"] is not None
        and float(metrics["trade_win_rate"])
        >= float(rules.get("trade_win_rate_min", 0.0)),
        "minimum_closed_trades": int(metrics["trade_count"])
        >= int(rules.get("minimum_closed_trades", 0)),
        "minimum_active_signal_days": int(metrics["active_signal_days"])
        >= int(rules.get("minimum_active_signal_days", 0)),
        "position_cap": int(metrics["max_open_positions"])
        <= int(strategy.portfolio["max_open_positions"]),
    }
    return {"passed": all(tests.values()), "tests": tests}


def _verify(
    *,
    plan: Any,
    features: pd.DataFrame,
    ledgers: dict[str, pd.DataFrame],
    selection: dict[str, Any],
    results: dict[str, dict[str, pd.DataFrame]],
    strategy: StrategyConfig,
) -> dict[str, Any]:
    signal_days = pd.DatetimeIndex(pd.to_datetime(plan.signal_sessions, utc=True)).normalize()
    execution_days = pd.DatetimeIndex(pd.to_datetime(plan.execution_sessions, utc=True)).normalize()
    checks = {
        "feature_unique": not features.duplicated(["asof", "ts_code"]).any(),
        "score_unique": not ledgers["score"].duplicated(["asof", "ts_code"]).any(),
        "candidate_view_cap": bool(
            ledgers["candidate"]
            .loc[ledgers["candidate"]["candidate_status"].eq("IN_VIEW")]
            .groupby("asof")
            .size()
            .le(int(strategy.portfolio["candidate_view_size"]))
            .all()
        ),
        "all_signal_dates_scored": int(selection["score_signal_dates"])
        == int(selection["signal_dates"]),
        "no_date_level_data_gaps": not selection["date_level_data_gap_days"],
        "feature_pit": bool(
            (
                pd.to_datetime(features["available_time"], utc=True)
                <= pd.to_datetime(features["asof"], utc=True).map(session_close)
            ).all()
        ),
    }
    for scenario, result in results.items():
        nav = result["nav"]
        decisions = result["execution_decisions"]
        checks[f"{scenario}_position_cap"] = bool(
            nav["open_positions"].le(int(strategy.portfolio["max_open_positions"])).all()
        )
        checks[f"{scenario}_execution_end"] = (
            str(pd.Timestamp(nav["trade_date"].max()).date()) == plan.execution_end
        )
        causal = True
        for row in decisions[decisions["chosen"].astype(bool)].itertuples(index=False):
            index = signal_days.get_indexer([pd.Timestamp(row.signal_session)])[0]
            execution_index = execution_days.get_indexer([pd.Timestamp(row.execution_session)])[0]
            causal &= index >= 0 and execution_index > 0
            if execution_index > 0:
                causal &= execution_days[execution_index - 1] == pd.Timestamp(
                    row.signal_session
                )
        checks[f"{scenario}_buy_t_plus_one"] = bool(causal)
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"passed": not failed, "checks": checks, "failed": failed}


def run(
    strategy_path: Path,
    output: Path,
    signal_start: str,
    prereg_path: Path | None,
) -> Path:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    strategy = StrategyConfig.from_yaml(strategy_path)
    sources = set(strategy.data_policy["dense_required"]["selection"]) | set(
        strategy.data_policy["dense_required"]["execution"]
    ) | set(strategy.data_policy.get("optional_enrichment", ()))
    source_cutoffs = load_source_max_dates(sources)
    cutoff = min(str(value) for value in source_cutoffs.values())
    calendar = load_trading_calendar(
        str((pd.Timestamp(signal_start) - pd.Timedelta(days=500)).date()), cutoff
    )
    covered_sessions = pd.DatetimeIndex(calendar["session"]).sort_values()
    if len(covered_sessions) < 2:
        raise ValueError("database cutoff does not contain a T+1-executable signal")
    request = RunRequest(
        signal_start=signal_start,
        signal_end=str(covered_sessions[-2].date()),
        execution_end=cutoff,
        output_dir=str(output),
        run_name=output.name,
    )
    plan = compile_run_plan(
        strategy,
        request,
        calendar["session"],
        require_complete_horizon=False,
    )
    bundle = build_data_bundle(plan, strategy, output)
    features = build_feature_ledger(bundle, strategy)
    ledgers = build_rule_ledgers(features, strategy, plan.signal_sessions)
    selection = _selection_summary(features, ledgers, plan.signal_sessions)
    results: dict[str, dict[str, pd.DataFrame]] = {}
    metrics: dict[str, Any] = {}
    for scenario in ("base", "stress"):
        result = run_backtest(
            candidate_ledger=ledgers["candidate"],
            selection_ledger=ledgers["selection"],
            execution_panel=bundle.execution,
            corporate_actions=bundle.corporate_actions,
            strategy=strategy,
            execution_sessions=tuple(plan.execution_sessions),
            scenario_name=scenario,
        )
        results[scenario] = result
        summary = summarize(
            result["nav"],
            result["fills"],
            initial_cash=float(strategy.execution["initial_cash"]),
            positions=result["positions"],
            corporate_actions=result["corporate_actions"],
        )
        summary.update(
            {
                "active_signal_days": selection["active_signal_days"],
                "entry_attempts": int(len(result["execution_decisions"])),
                "entry_fills": int(result["execution_decisions"]["chosen"].sum())
                if not result["execution_decisions"].empty
                else 0,
                "open_positions_at_end": int(len(result["open_positions"])),
            }
        )
        summary["acceptance"] = _acceptance(summary, strategy)
        metrics[scenario] = summary
    verification = _verify(
        plan=plan,
        features=features,
        ledgers=ledgers,
        selection=selection,
        results=results,
        strategy=strategy,
    )
    if not verification["passed"]:
        raise AssertionError("summary-only rules verification failed: " + ", ".join(verification["failed"]))

    passed = all(metrics[name]["acceptance"]["passed"] for name in ("base", "stress"))
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "RUN_STATUS.json",
        {
            "status": "COMPLETED_ACCEPT" if passed else "COMPLETED_REJECT",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "strategy_id": strategy.strategy_id,
            "credentials_persisted": False,
            "raw_business_data_persisted": False,
            "years_used_for_performance": [2026],
        },
    )
    _write_json(output / "plan.json", plan.to_dict())
    _write_json(output / "data_manifest.json", bundle.manifest)
    _write_json(output / "selection_summary.json", selection)
    _write_json(output / "portfolio_metrics.json", metrics)
    _write_json(output / "verification.json", verification)
    code_paths = [
        strategy_path,
        ROOT / "src/aistock9988/configuration.py",
        ROOT / "src/aistock9988/data/bundle.py",
        ROOT / "src/aistock9988/features/engine.py",
        ROOT / "src/aistock9988/selection/pipeline.py",
        ROOT / "src/aistock9988/backtest/engine.py",
        ROOT / "src/aistock9988/reporting/metrics.py",
        Path(__file__).resolve(),
    ]
    if prereg_path is not None:
        code_paths.append(prereg_path)
    _write_json(
        output / "code_manifest.json",
        {str(path.relative_to(ROOT)): _sha256(path) for path in code_paths},
    )
    base = metrics["base"]
    stress = metrics["stress"]
    result = f"""# {strategy.strategy_id} 2026 DB Full-Universe Backtest

Status: `{'ACCEPT' if passed else 'REJECT'}`. This is a seen-2026 diagnostic, not
an out-of-sample claim.

## Scope

- Signals: `{plan.signal_start}` through `{plan.signal_end}`; execution/mark
  cutoff: `{plan.execution_end}`, the common required-source DB cutoff.
- {selection['signal_dates']} signal dates, {selection['active_signal_days']}
  active dates, {selection['stage1_pass_rows']} Stage1 rows, and no date-level
  data gap or sample-size skip.
- Database universe; missing required data exclude only that stock-session.
- No raw business data, CSV, Parquet, factor cache, or model artifact was written.

## Portfolio

| Scenario | Return | PF | MaxDD | Win rate | Ex-best-week | Ex-top3 | Weekly >=5% | Trades | End open | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Base | {base['total_return']:+.2%} | {base['portfolio_profit_factor']:.3f} | {base['max_drawdown']:.2%} | {base['trade_win_rate']:.1%} | {base['return_excluding_best_week']:+.2%} | {base['return_excluding_top3_profit']:+.2%} | {base['weekly_ge_5_count']} ({base['weekly_ge_5_ratio']:.1%}) | {base['trade_count']} | {base['open_positions_at_end']} | {base['acceptance']['passed']} |
| Stress | {stress['total_return']:+.2%} | {stress['portfolio_profit_factor']:.3f} | {stress['max_drawdown']:.2%} | {stress['trade_win_rate']:.1%} | {stress['return_excluding_best_week']:+.2%} | {stress['return_excluding_top3_profit']:+.2%} | {stress['weekly_ge_5_count']} ({stress['weekly_ge_5_ratio']:.1%}) | {stress['trade_count']} | {stress['open_positions_at_end']} | {stress['acceptance']['passed']} |

## Decision

Both Base and Stress acceptance passed: `{passed}`. A failed fixed rule is
retained unchanged as evidence and must not be repaired by a threshold, weight,
TopN, holding-period, gate, or model scan.
"""
    (output / "RESULT.md").write_text(result, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signal-start", default="2026-01-01")
    parser.add_argument("--prereg", type=Path)
    args = parser.parse_args()
    print(
        run(
            args.strategy.resolve(),
            args.output.resolve(),
            args.signal_start,
            args.prereg.resolve() if args.prereg else None,
        )
    )


if __name__ == "__main__":
    main()
