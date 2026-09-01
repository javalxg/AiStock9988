"""Run the preregistered CAP1 H10-profit-to-H20 paired 2026 experiment."""
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
from aistock9988.data.bundle import (
    build_data_bundle,
    load_source_max_dates,
    load_trading_calendar,
)
from aistock9988.features.engine import build_feature_ledger
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.metrics import summarize
from aistock9988.selection.pipeline import build_rule_ledgers


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs/strategy/reset_weak_confirm_v3_cap1_20.yaml"
CHALLENGER = (
    ROOT
    / "configs/strategy/reset_weak_confirm_v3_cap1_h10_profit_extend_h20_v1.yaml"
)
PREREG = (
    ROOT
    / "docs/council_20260828"
    / "CAP1_H10_PROFIT_EXTEND_H20_PREREG_20260902.md"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs/council_20260828"
    / "CAP1_H10_PROFIT_EXTEND_H20_2026_TO_DB_CUTOFF_20260902"
)
SEALED_SUMMARY = (
    ROOT
    / "docs/council_20260828"
    / "RESET_WEAK_CONFIRM_V3_CAP1_20_2026_TO_0828_20260901"
    / "PORTFOLIO_SUMMARY.json"
)


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


def _validate_contract(control: StrategyConfig, challenger: StrategyConfig) -> None:
    control_dict = control.to_dict()
    challenger_dict = challenger.to_dict()
    for section in (
        "universe",
        "data_policy",
        "decision",
        "features",
        "stage1",
        "ranking",
        "portfolio",
        "acceptance",
    ):
        if control_dict[section] != challenger_dict[section]:
            raise ValueError(f"challenger changes frozen section: {section}")
    control_execution = dict(control_dict["execution"])
    challenger_execution = dict(challenger_dict["execution"])
    extension = challenger_execution.pop("time_exit_extension", None)
    if challenger_execution != control_execution:
        raise ValueError("challenger changes execution fields beyond time_exit_extension")
    expected = {
        "enabled": True,
        "condition": "prior_close_unrealized_positive",
        "extended_hold_sessions_from_fill": 20,
    }
    if extension != expected:
        raise ValueError("challenger extension differs from preregistration")


def _selection_summary(ledgers: dict[str, pd.DataFrame]) -> dict[str, int]:
    score = ledgers["score"]
    candidate = ledgers["candidate"]
    return {
        "signal_dates": int(score["asof"].nunique()),
        "active_signal_days": int(
            candidate.groupby("asof", sort=True)["stage1_pass"].sum().gt(0).sum()
        ),
        "stage1_pass_rows": int(score["stage1_pass"].sum()),
        "candidate_view_rows": int(candidate["candidate_status"].eq("IN_VIEW").sum()),
        "selection_data_excluded_rows": int((~score["selection_data_eligible"]).sum()),
    }


def _metrics(
    result: dict[str, pd.DataFrame],
    strategy: StrategyConfig,
    selection: dict[str, int],
) -> dict[str, Any]:
    metrics = summarize(
        result["nav"],
        result["fills"],
        initial_cash=float(strategy.execution["initial_cash"]),
        positions=result["positions"],
        corporate_actions=result["corporate_actions"],
    )
    events = result["position_events"]
    extensions = events[events["event_type"].eq("TIME_EXIT_EXTENSION")]
    metrics.update(
        {
            "entry_attempts": int(len(result["execution_decisions"])),
            "entry_fills": int(result["execution_decisions"]["chosen"].sum())
            if not result["execution_decisions"].empty
            else 0,
            "open_positions_at_end": int(len(result["open_positions"])),
            "active_signal_days": int(selection["active_signal_days"]),
            "time_exit_extension_events": int(len(extensions)),
        }
    )
    return metrics


def _run_pair(
    *,
    ledgers: dict[str, pd.DataFrame],
    execution: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    sessions: tuple[str, ...],
    control: StrategyConfig,
    challenger: StrategyConfig,
    selection: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    controls: dict[str, Any] = {}
    challengers: dict[str, Any] = {}
    mechanisms: dict[str, Any] = {}
    for scenario in ("base", "stress"):
        control_result = run_backtest(
            candidate_ledger=ledgers["candidate"],
            selection_ledger=ledgers["selection"],
            execution_panel=execution,
            corporate_actions=corporate_actions,
            strategy=control,
            execution_sessions=sessions,
            scenario_name=scenario,
        )
        challenger_result = run_backtest(
            candidate_ledger=ledgers["candidate"],
            selection_ledger=ledgers["selection"],
            execution_panel=execution,
            corporate_actions=corporate_actions,
            strategy=challenger,
            execution_sessions=sessions,
            scenario_name=scenario,
        )
        controls[scenario] = _metrics(control_result, control, selection)
        challengers[scenario] = _metrics(challenger_result, challenger, selection)
        mechanisms[scenario] = _mechanism(control_result, challenger_result)
    return controls, challengers, mechanisms


def _mechanism(
    control: dict[str, pd.DataFrame],
    challenger: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    def trades(result: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        fills = result["fills"].copy()
        fills["trade_key"] = (
            fills["decision_id"].astype(str) + "|" + fills["ts_code"].astype(str)
        )
        buys = fills[fills["side"].eq("BUY")].set_index("trade_key", verify_integrity=True)
        sells = fills[fills["side"].eq("SELL")].set_index("trade_key", verify_integrity=True)
        return buys, sells

    control_buys, control_sells = trades(control)
    challenger_buys, challenger_sells = trades(challenger)
    shared_buys = sorted(set(control_buys.index) & set(challenger_buys.index))
    control_only = sorted(set(control_buys.index) - set(challenger_buys.index))
    challenger_only = sorted(set(challenger_buys.index) - set(control_buys.index))
    shared_closed = sorted(
        set(shared_buys) & set(control_sells.index) & set(challenger_sells.index)
    )
    changed_exit = [
        key
        for key in shared_closed
        if pd.Timestamp(control_sells.loc[key, "trade_date"])
        != pd.Timestamp(challenger_sells.loc[key, "trade_date"])
    ]
    changed = pd.DataFrame(
        [
            {
                "control_return": float(control_sells.loc[key, "economic_return"]),
                "challenger_return": float(challenger_sells.loc[key, "economic_return"]),
            }
            for key in changed_exit
        ]
    )
    control_only_closed = control_sells.loc[
        [key for key in control_only if key in control_sells.index]
    ]
    return {
        "control_buy_count": int(len(control_buys)),
        "challenger_buy_count": int(len(challenger_buys)),
        "shared_buy_count": int(len(shared_buys)),
        "control_only_buy_count": int(len(control_only)),
        "challenger_only_buy_count": int(len(challenger_only)),
        "changed_exit_closed_trade_count": int(len(changed)),
        "changed_exit_control_mean_return": float(changed["control_return"].mean())
        if not changed.empty
        else None,
        "changed_exit_challenger_mean_return": float(changed["challenger_return"].mean())
        if not changed.empty
        else None,
        "changed_exit_improved_rate": float(
            changed["challenger_return"].gt(changed["control_return"]).mean()
        )
        if not changed.empty
        else None,
        "control_only_closed_trade_count": int(len(control_only_closed)),
        "control_only_realized_pnl": float(control_only_closed["realized_pnl"].sum())
        if not control_only_closed.empty
        else 0.0,
        "control_only_win_rate": float(control_only_closed["realized_pnl"].gt(0.0).mean())
        if not control_only_closed.empty
        else None,
    }


def _sealed_regression(
    *,
    ledgers: dict[str, pd.DataFrame],
    bundle: Any,
    control: StrategyConfig,
) -> dict[str, Any]:
    sealed = json.loads(SEALED_SUMMARY.read_text(encoding="utf-8"))
    signal_end = pd.Timestamp("2026-08-13", tz="UTC")
    execution_end = pd.Timestamp("2026-08-28", tz="UTC")
    sliced = {
        "candidate": ledgers["candidate"][ledgers["candidate"]["asof"].le(signal_end)],
        "selection": ledgers["selection"][ledgers["selection"]["asof"].le(signal_end)],
    }
    selection = _selection_summary(
        {"score": ledgers["score"][ledgers["score"]["asof"].le(signal_end)], **sliced}
    )
    sessions = tuple(
        str(day.date())
        for day in pd.DatetimeIndex(bundle.execution["trade_date"].drop_duplicates().sort_values())
        if day <= execution_end
    )
    actual: dict[str, Any] = {}
    fields = (
        "total_return",
        "portfolio_profit_factor",
        "max_drawdown",
        "trade_win_rate",
        "return_excluding_best_week",
        "return_excluding_top3_profit",
        "trade_count",
    )
    checks: dict[str, bool] = {}
    for scenario in ("base", "stress"):
        result = run_backtest(
            candidate_ledger=sliced["candidate"],
            selection_ledger=sliced["selection"],
            execution_panel=bundle.execution,
            corporate_actions=bundle.corporate_actions,
            strategy=control,
            execution_sessions=sessions,
            scenario_name=scenario,
        )
        actual[scenario] = _metrics(result, control, selection)
        for field in fields:
            expected = sealed[scenario][field]
            observed = actual[scenario][field]
            checks[f"{scenario}_{field}"] = (
                int(observed) == int(expected)
                if field == "trade_count"
                else abs(float(observed) - float(expected)) <= 1e-10
            )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "actual": actual,
    }


def _comparison(control: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "total_return",
        "portfolio_profit_factor",
        "max_drawdown",
        "trade_win_rate",
        "return_excluding_best_week",
        "return_excluding_top3_profit",
        "weekly_ge_5_count",
        "weekly_ge_5_ratio",
        "trade_count",
    )
    scenarios: dict[str, Any] = {}
    promotion: dict[str, bool] = {}
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
        promotion[f"{scenario}_return"] = (
            float(challenger[scenario]["total_return"])
            > float(control[scenario]["total_return"])
        )
        promotion[f"{scenario}_pf"] = (
            challenger[scenario]["portfolio_profit_factor"] is not None
            and float(challenger[scenario]["portfolio_profit_factor"]) >= 2.0
            and float(challenger[scenario]["portfolio_profit_factor"])
            >= float(control[scenario]["portfolio_profit_factor"])
        )
        promotion[f"{scenario}_maxdd"] = (
            abs(float(challenger[scenario]["max_drawdown"])) <= 0.15
            and abs(float(challenger[scenario]["max_drawdown"]))
            <= abs(float(control[scenario]["max_drawdown"]))
        )
        promotion[f"{scenario}_ex_best_week"] = (
            float(challenger[scenario]["return_excluding_best_week"]) > 0.0
            and float(challenger[scenario]["return_excluding_best_week"])
            >= float(control[scenario]["return_excluding_best_week"])
        )
        promotion[f"{scenario}_ex_top3"] = (
            float(challenger[scenario]["return_excluding_top3_profit"]) > 0.0
            and float(challenger[scenario]["return_excluding_top3_profit"])
            >= float(control[scenario]["return_excluding_top3_profit"])
        )
        promotion[f"{scenario}_win_rate_70"] = (
            challenger[scenario]["trade_win_rate"] is not None
            and float(challenger[scenario]["trade_win_rate"]) >= 0.70
        )
        promotion[f"{scenario}_position_cap"] = (
            int(challenger[scenario]["max_open_positions"]) <= 5
        )
    return {
        "scenarios": scenarios,
        "promotion_tests": promotion,
        "passed": all(promotion.values()),
    }


def run(output: Path, signal_start: str) -> Path:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    control = StrategyConfig.from_yaml(CONTROL)
    challenger = StrategyConfig.from_yaml(CHALLENGER)
    _validate_contract(control, challenger)
    sources = set(control.data_policy["dense_required"]["selection"]) | set(
        control.data_policy["dense_required"]["execution"]
    )
    cutoffs = load_source_max_dates(sources)
    cutoff = min(str(value) for value in cutoffs.values())
    calendar = load_trading_calendar(
        str((pd.Timestamp(signal_start) - pd.Timedelta(days=500)).date()), cutoff
    )
    covered_sessions = pd.DatetimeIndex(calendar["session"]).sort_values()
    if len(covered_sessions) < 2:
        raise ValueError("database cutoff does not contain a T+1-executable signal session")
    signal_end = str(covered_sessions[-2].date())
    request = RunRequest(
        signal_start=signal_start,
        signal_end=signal_end,
        execution_end=cutoff,
        output_dir=str(output),
        run_name="CAP1_H10_PROFIT_EXTEND_H20_2026_TO_DB_CUTOFF_20260902",
    )
    plan = compile_run_plan(
        control,
        request,
        calendar["session"],
        require_complete_horizon=False,
    )
    bundle = build_data_bundle(plan, control, output)
    features = build_feature_ledger(bundle, control)
    ledgers = build_rule_ledgers(features, control, plan.signal_sessions)
    selection = _selection_summary(ledgers)
    regression = _sealed_regression(ledgers=ledgers, bundle=bundle, control=control)
    if not regression["passed"]:
        raise AssertionError("sealed CAP1 control regression failed")
    controls, challengers, mechanisms = _run_pair(
        ledgers=ledgers,
        execution=bundle.execution,
        corporate_actions=bundle.corporate_actions,
        sessions=tuple(plan.execution_sessions),
        control=control,
        challenger=challenger,
        selection=selection,
    )
    comparison = _comparison(controls, challengers)

    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "RUN_STATUS.json",
        {
            "status": "COMPLETED_ACCEPT" if comparison["passed"] else "COMPLETED_REJECT",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "credentials_persisted": False,
            "raw_business_data_persisted": False,
            "years_used_for_performance": [2026],
        },
    )
    _write_json(output / "plan.json", plan.to_dict())
    _write_json(output / "data_manifest.json", bundle.manifest)
    _write_json(output / "selection_summary.json", selection)
    _write_json(output / "sealed_control_regression.json", regression)
    _write_json(output / "control_metrics.json", controls)
    _write_json(output / "challenger_metrics.json", challengers)
    _write_json(output / "mechanism.json", mechanisms)
    _write_json(output / "comparison.json", comparison)
    _write_json(
        output / "code_manifest.json",
        {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                CONTROL,
                CHALLENGER,
                PREREG,
                ROOT / "src/aistock9988/configuration.py",
                ROOT / "src/aistock9988/backtest/engine.py",
                Path(__file__).resolve(),
            )
        },
    )

    base_c = controls["base"]
    base_x = challengers["base"]
    stress_c = controls["stress"]
    stress_x = challengers["stress"]
    mechanism = mechanisms["base"]
    decision = "ACCEPT" if comparison["passed"] else "REJECT"
    result = f"""# CAP1 H10 Profit Extension to H20

Status: `{decision}`. This is a preregistered seen-2026 paired backtest, not an
out-of-sample claim.

## Scope and integrity

- Signal range: `{plan.signal_start}` through `{plan.signal_end}`; execution and
  mark cutoff: `{plan.execution_end}`, the common required-source DB cutoff.
- Signal dates: {selection['signal_dates']}; active signal dates:
  {selection['active_signal_days']}; no date-level sample gate or fallback.
- Sealed 2026 CAP1 regression passed: {regression['passed']}.
- Same in-memory candidates, decisions, prices and corporate actions; only H10
  prior-close-profitable positions may extend to H20.
- No raw business data, model, CSV, or Parquet artifact was written.

## Paired portfolio result

| Scenario | Strategy | Return | PF | MaxDD | Win rate | Ex-best-week | Weekly >=5% | Trades | Extensions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | Control | {base_c['total_return']:+.2%} | {base_c['portfolio_profit_factor']:.3f} | {base_c['max_drawdown']:.2%} | {base_c['trade_win_rate']:.1%} | {base_c['return_excluding_best_week']:+.2%} | {base_c['weekly_ge_5_count']} ({base_c['weekly_ge_5_ratio']:.1%}) | {base_c['trade_count']} | 0 |
| Base | Challenger | {base_x['total_return']:+.2%} | {base_x['portfolio_profit_factor']:.3f} | {base_x['max_drawdown']:.2%} | {base_x['trade_win_rate']:.1%} | {base_x['return_excluding_best_week']:+.2%} | {base_x['weekly_ge_5_count']} ({base_x['weekly_ge_5_ratio']:.1%}) | {base_x['trade_count']} | {base_x['time_exit_extension_events']} |
| Stress | Control | {stress_c['total_return']:+.2%} | {stress_c['portfolio_profit_factor']:.3f} | {stress_c['max_drawdown']:.2%} | {stress_c['trade_win_rate']:.1%} | {stress_c['return_excluding_best_week']:+.2%} | {stress_c['weekly_ge_5_count']} ({stress_c['weekly_ge_5_ratio']:.1%}) | {stress_c['trade_count']} | 0 |
| Stress | Challenger | {stress_x['total_return']:+.2%} | {stress_x['portfolio_profit_factor']:.3f} | {stress_x['max_drawdown']:.2%} | {stress_x['trade_win_rate']:.1%} | {stress_x['return_excluding_best_week']:+.2%} | {stress_x['weekly_ge_5_count']} ({stress_x['weekly_ge_5_ratio']:.1%}) | {stress_x['trade_count']} | {stress_x['time_exit_extension_events']} |

## Failure mechanism

- Among {mechanism['changed_exit_closed_trade_count']} shared closed trades whose
  exit actually changed, H10 control returns averaged
  {mechanism['changed_exit_control_mean_return']:+.2%}; extended exits averaged
  {mechanism['changed_exit_challenger_mean_return']:+.2%}. Only
  {mechanism['changed_exit_improved_rate']:.1%} improved.
- Longer occupancy also removed {mechanism['control_only_buy_count']} control
  buys. Their {mechanism['control_only_closed_trade_count']} closed Base trades
  contributed RMB {mechanism['control_only_realized_pnl']:,.0f} with
  {mechanism['control_only_win_rate']:.1%} winners.

## Decision

Promotion tests passed: `{comparison['passed']}`. On failure, close this exact
H10-profit-to-H20 rule without trying another hold period, threshold, TopN, gate,
or model. CAP1 remains unchanged unless every preregistered paired condition
passes.
"""
    (output / "RESULT.md").write_text(result, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--signal-start", default="2026-01-01")
    args = parser.parse_args()
    print(run(args.output.resolve(), args.signal_start))


if __name__ == "__main__":
    main()
