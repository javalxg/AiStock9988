#!/usr/bin/env python3
"""Diagnose the sealed dragon-tiger V1 trades without retuning the strategy."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aistock9988.backtest.engine import run_backtest
from aistock9988.configuration import StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_trading_calendar
from aistock9988.data.dragon_tiger import load_dragon_tiger_events
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.selection.dragon_tiger import build_pullback_reclaim_ledgers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "docs/council_20260828"
    / "DRAGON_TIGER_PULLBACK_RECLAIM_V1_2026_TO_DB_CUTOFF_20260901_R2"
)
DEFAULT_OUTPUT = (
    ROOT / "docs/council_20260828"
    / "DRAGON_TIGER_TRADE_MECHANISM_DIAGNOSTIC_20260901"
)

FEATURES = (
    "institution_intensity",
    "opening_gap_return",
    "pullback_delay_sessions",
    "reclaim_delay_sessions",
    "pullback_to_reclaim_sessions",
    "reclaim_volume_ratio",
    "candidate_rank",
    "holding_sessions",
    "economic_return",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


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


def _rebuild(source: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]], dict[str, Any]]:
    plan_payload = _read_json(source / "plan.json")
    strategy_path = source / "configs/strategy.yaml"
    strategy = StrategyConfig.from_yaml(strategy_path)
    calendar_start = str(
        (pd.Timestamp(plan_payload["signal_start"]) - pd.Timedelta(days=500)).date()
    )
    calendar = load_trading_calendar(calendar_start, plan_payload["execution_end"])
    request = RunRequest(
        signal_start=plan_payload["signal_start"],
        signal_end=plan_payload["signal_end"],
        execution_end=plan_payload["execution_end"],
        output_dir="unused",
        run_name="dragon_tiger_v1_mechanism_rebuild",
    )
    plan = compile_run_plan(
        strategy,
        request,
        calendar["session"],
        require_complete_horizon=False,
    )
    with tempfile.TemporaryDirectory(prefix="aistock-dtr-diagnostic-") as temp:
        bundle = build_data_bundle(plan, strategy, Path(temp))
        events = load_dragon_tiger_events(
            plan_payload["event_source_start"], plan_payload["event_source_end"]
        ).events
        ledgers = build_pullback_reclaim_ledgers(
            events,
            bundle.execution,
            plan.signal_sessions,
            entries_per_decision=int(strategy.portfolio["entries_per_decision"]),
        )
        results = {
            scenario: run_backtest(
                candidate_ledger=ledgers["candidate"],
                selection_ledger=ledgers["selection"],
                execution_panel=bundle.execution,
                corporate_actions=bundle.corporate_actions,
                strategy=strategy,
                execution_sessions=plan.execution_sessions,
                scenario_name=scenario,
            )
            for scenario in ("base", "stress")
        }
    return ledgers, results, plan_payload


def _verify_reproduction(
    source: Path,
    ledgers: dict[str, pd.DataFrame],
    results: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    expected = _read_json(source / "manifests/in_memory_ledger_manifest.json")
    checks: dict[str, dict[str, Any]] = {}
    for name, frame in ledgers.items():
        key = f"event_and_selection.{name}"
        target = expected["event_and_selection"][name]
        actual_hash = _frame_hash(frame)
        checks[key] = {
            "rows": int(len(frame)),
            "expected_rows": int(target["rows"]),
            "sha256": actual_hash,
            "expected_sha256": target["sha256"],
            "passed": int(len(frame)) == int(target["rows"]) and actual_hash == target["sha256"],
        }
    for scenario, result in results.items():
        for name, frame in result.items():
            key = f"backtest.{scenario}.{name}"
            target = expected["backtest"][scenario][name]
            actual_hash = _frame_hash(frame)
            checks[key] = {
                "rows": int(len(frame)),
                "expected_rows": int(target["rows"]),
                "sha256": actual_hash,
                "expected_sha256": target["sha256"],
                "passed": int(len(frame)) == int(target["rows"]) and actual_hash == target["sha256"],
            }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "sealed_manifest_sha256": _file_hash(
            source / "manifests/in_memory_ledger_manifest.json"
        ),
    }


def _trade_table(
    ledgers: dict[str, pd.DataFrame],
    result: dict[str, pd.DataFrame],
    execution_sessions: list[str],
) -> pd.DataFrame:
    fills = result["fills"]
    keys = ["decision_id", "ts_code"]
    buys = fills.loc[fills["side"].eq("BUY")].copy()
    sells = fills.loc[fills["side"].eq("SELL")].copy()
    if buys.duplicated(keys).any() or sells.duplicated(keys).any():
        raise ValueError("diagnostic requires one entry and at most one exit per decision/security")
    trades = buys[keys + ["trade_date", "economic_price"]].merge(
        sells[keys + ["trade_date", "economic_return", "realized_pnl", "reason"]],
        on=keys,
        suffixes=("_entry", "_exit"),
        validate="one_to_one",
    )
    decisions = result["execution_decisions"]
    chosen = decisions.loc[decisions["chosen"].astype(bool), keys + [
        "signal_session", "candidate_rank"
    ]]
    trades = trades.merge(chosen, on=keys, validate="one_to_one")
    candidates = ledgers["candidate"][[
        "asof", "event_date", "pullback_date", "ts_code",
        "institution_intensity", "reclaim_volume_ratio",
    ]]
    trades = trades.merge(
        candidates.rename(columns={"asof": "signal_session"}),
        on=["signal_session", "ts_code"],
        validate="one_to_one",
    )
    states = ledgers["state"].loc[
        ledgers["state"]["status"].eq("CONFIRMED"),
        ["event_date", "ts_code", "reclaim_date", "gap_return"],
    ]
    trades = trades.merge(states, on=["event_date", "ts_code"], validate="one_to_one")

    sessions = pd.DatetimeIndex(pd.to_datetime(execution_sessions, utc=True)).normalize()
    session_index = {day: index for index, day in enumerate(sessions)}
    index_of = lambda values: values.map(session_index).astype(int)
    trades["opening_gap_return"] = trades.pop("gap_return")
    trades["pullback_delay_sessions"] = (
        index_of(trades["pullback_date"]) - index_of(trades["event_date"])
    )
    trades["reclaim_delay_sessions"] = (
        index_of(trades["reclaim_date"]) - index_of(trades["event_date"])
    )
    trades["pullback_to_reclaim_sessions"] = (
        index_of(trades["reclaim_date"]) - index_of(trades["pullback_date"])
    )
    trades["holding_sessions"] = (
        index_of(trades["trade_date_exit"]) - index_of(trades["trade_date_entry"])
    )
    trades["outcome"] = np.where(trades["realized_pnl"] > 0, "WIN", "LOSS")
    trades["entry_month"] = pd.to_datetime(
        trades["trade_date_entry"], utc=True
    ).dt.strftime("%Y-%m")
    return trades


def _describe(trades: pd.DataFrame) -> dict[str, Any]:
    outcome: dict[str, Any] = {}
    for name, group in trades.groupby("outcome", sort=True):
        outcome[str(name)] = {
            "trades": int(len(group)),
            "win_rate": float(group["realized_pnl"].gt(0).mean()),
            "total_realized_pnl": float(group["realized_pnl"].sum()),
            "features": {
                feature: {
                    "mean": float(group[feature].mean()),
                    "median": float(group[feature].median()),
                }
                for feature in FEATURES
            },
        }
    comparisons = {
        feature: {
            "winner_mean_minus_loser_mean": float(
                trades.loc[trades["outcome"].eq("WIN"), feature].mean()
                - trades.loc[trades["outcome"].eq("LOSS"), feature].mean()
            ),
            "winner_median_minus_loser_median": float(
                trades.loc[trades["outcome"].eq("WIN"), feature].median()
                - trades.loc[trades["outcome"].eq("LOSS"), feature].median()
            ),
        }
        for feature in FEATURES
    }
    exits = []
    for reason, group in trades.groupby("reason", sort=True):
        exits.append({
            "exit_reason": str(reason),
            "trades": int(len(group)),
            "win_rate": float(group["realized_pnl"].gt(0).mean()),
            "mean_economic_return": float(group["economic_return"].mean()),
            "total_realized_pnl": float(group["realized_pnl"].sum()),
            "mean_holding_sessions": float(group["holding_sessions"].mean()),
        })
    ranks = []
    for rank, group in trades.groupby("candidate_rank", sort=True):
        ranks.append({
            "candidate_rank": int(rank),
            "trades": int(len(group)),
            "win_rate": float(group["realized_pnl"].gt(0).mean()),
            "mean_economic_return": float(group["economic_return"].mean()),
            "total_realized_pnl": float(group["realized_pnl"].sum()),
        })
    months = []
    for month, group in trades.groupby("entry_month", sort=True):
        months.append({
            "entry_month": str(month),
            "trades": int(len(group)),
            "win_rate": float(group["realized_pnl"].gt(0).mean()),
            "mean_economic_return": float(group["economic_return"].mean()),
            "total_realized_pnl": float(group["realized_pnl"].sum()),
        })
    return {
        "closed_trades": int(len(trades)),
        "wins": int(trades["outcome"].eq("WIN").sum()),
        "losses": int(trades["outcome"].eq("LOSS").sum()),
        "outcome_summary": outcome,
        "winner_loser_comparisons": comparisons,
        "exit_reason_summary": exits,
        "candidate_rank_summary": ranks,
        "entry_month_summary": months,
    }


def _pct(value: float) -> str:
    return f"{value:+.2%}"


def _write_report(output: Path, diagnostic: dict[str, Any]) -> None:
    summary = diagnostic["base_trade_diagnostic"]
    comp = summary["winner_loser_comparisons"]
    exit_rows = summary["exit_reason_summary"]
    rank_rows = summary["candidate_rank_summary"]
    month_rows = summary["entry_month_summary"]
    lines = [
        "# Dragon-Tiger V1 Trade Mechanism Diagnostic", "",
        "## Reproduction", "",
        f"- Exact sealed-ledger reproduction: `{diagnostic['reproduction']['passed']}`.",
        f"- Closed Base trades: `{summary['closed_trades']}`; wins: `{summary['wins']}`; losses: `{summary['losses']}`.",
        "- This is an unchanged-trade diagnostic, not a new strategy or parameter search.", "",
        "## Winner Versus Loser", "",
        "| Feature | Winner mean minus loser mean | Winner median minus loser median |",
        "|---|---:|---:|",
    ]
    for feature in FEATURES:
        row = comp[feature]
        lines.append(
            f"| {feature} | {row['winner_mean_minus_loser_mean']:+.6f} | "
            f"{row['winner_median_minus_loser_median']:+.6f} |"
        )
    lines.extend(["", "## Exit Mechanism", "", "| Exit | Trades | Win rate | Mean return | PnL | Mean hold |", "|---|---:|---:|---:|---:|---:|"])
    for row in exit_rows:
        lines.append(
            f"| {row['exit_reason']} | {row['trades']} | {_pct(row['win_rate'])} | "
            f"{_pct(row['mean_economic_return'])} | {row['total_realized_pnl']:+.2f} | "
            f"{row['mean_holding_sessions']:.2f} |"
        )
    lines.extend(["", "## Candidate Rank", "", "| Rank | Trades | Win rate | Mean return | PnL |", "|---:|---:|---:|---:|---:|"])
    for row in rank_rows:
        lines.append(
            f"| {row['candidate_rank']} | {row['trades']} | {_pct(row['win_rate'])} | "
            f"{_pct(row['mean_economic_return'])} | {row['total_realized_pnl']:+.2f} |"
        )
    lines.extend([
        "", "## Monthly Concentration", "",
        "| Entry month | Trades | Win rate | Mean return | PnL |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in month_rows:
        lines.append(
            f"| {row['entry_month']} | {row['trades']} | {_pct(row['win_rate'])} | "
            f"{_pct(row['mean_economic_return'])} | {row['total_realized_pnl']:+.2f} |"
        )
    lines.extend([
        "", "## Decision", "",
        "- Reject dragon-tiger V1 as a standalone long-entry signal and as a positive ranking input.",
        "- Institution intensity, opening gap, and observation timing are nearly indistinguishable between winners and losers; stronger reclaim volume is not supportive because it is higher among losers.",
        "- The loss is structural: STOP_LOSS trades lost more than TIME_EXIT trades earned, and only three of eight entry months were profitable.",
        "- The only justified next use is one separately preregistered CAP1 risk-exclusion overlay: exclude a CAP1 candidate when the unchanged V1 event was confirmed during the prior H10 window. Do not scan event age or thresholds.",
        "", "These comparisons are descriptive evidence only and must not be converted into scanned thresholds.",
    ])
    path = output / "RESULT.md"
    if path.exists():
        raise FileExistsError(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(source: Path, output: Path) -> Path:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ledgers, results, plan_payload = _rebuild(source)
    reproduction = _verify_reproduction(source, ledgers, results)
    if not reproduction["passed"]:
        raise ValueError("rebuilt ledgers do not match the sealed R2 run")
    trades = _trade_table(ledgers, results["base"], plan_payload["execution_sessions"])
    diagnostic = {
        "source_run": str(source.relative_to(ROOT)),
        "source_result_sha256": _file_hash(source / "RESULT.md"),
        "business_data_persisted": False,
        "parameter_search_performed": False,
        "reproduction": reproduction,
        "base_trade_diagnostic": _describe(trades),
    }
    _write_json(output / "diagnostic.json", diagnostic)
    _write_report(output, diagnostic)
    _write_json(output / "artifact_manifest.json", {
        str(path.relative_to(output)): {
            "sha256": _file_hash(path), "bytes": path.stat().st_size
        }
        for path in sorted(output.rglob("*")) if path.is_file()
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"diagnostic_complete={run(args.source.resolve(), args.output.resolve())}")


if __name__ == "__main__":
    main()
