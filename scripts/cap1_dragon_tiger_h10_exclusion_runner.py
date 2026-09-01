#!/usr/bin/env python3
"""Run the preregistered CAP1 dragon-tiger H10 exclusion diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aistock9988.backtest.engine import run_backtest
from aistock9988.configuration import StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_trading_calendar
from aistock9988.data.dragon_tiger import load_dragon_tiger_events
from aistock9988.features.engine import build_feature_ledger
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.metrics import summarize
from aistock9988.selection.dragon_tiger import build_pullback_reclaim_ledgers
from aistock9988.selection.pipeline import build_rule_ledgers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRATEGY = ROOT / "configs/strategy/reset_weak_confirm_v3_cap1_20.yaml"
DEFAULT_OVERLAY = ROOT / "configs/strategy/cap1_dragon_tiger_h10_exclusion_v1.yaml"
DEFAULT_CONTROL = (
    ROOT / "docs/council_20260828"
    / "RESET_WEAK_CONFIRM_V3_CAP1_20_2026_TO_0828_20260901"
)
DEFAULT_OUTPUT = (
    ROOT / "docs/council_20260828"
    / "CAP1_DRAGON_TIGER_H10_EXCLUSION_20260901"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _load_overlay(path: Path, strategy: StrategyConfig) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("overlay config must be a mapping")
    identity = raw.get("identity", {})
    control = raw.get("control", {})
    overlay = raw.get("overlay", {})
    evaluation = raw.get("evaluation", {})
    if identity.get("overlay_id") != "cap1_dragon_tiger_h10_exclusion_v1":
        raise ValueError("unexpected overlay_id")
    if int(identity.get("version", 0)) != 1:
        raise ValueError("overlay version must be 1")
    if control.get("strategy_id") != strategy.strategy_id:
        raise ValueError("overlay control strategy does not match")
    if overlay != {
        "action": "exclude_candidate_preserve_cap1_rank",
        "state_window_sessions": 10,
        "include_signal_session": True,
        "ranked_fallback": True,
    }:
        raise ValueError("overlay rule differs from preregistration")
    if evaluation.get("parameter_sweep") is not False:
        raise ValueError("parameter_sweep must be false")
    return raw


def _selection_summary(ledgers: dict[str, pd.DataFrame]) -> dict[str, int]:
    score = ledgers["score"]
    candidate = ledgers["candidate"]
    daily_stage1 = score.groupby("asof", sort=True)["stage1_pass"].sum()
    daily_view = candidate.groupby("asof", sort=True)["candidate_status"].apply(
        lambda values: int(values.eq("IN_VIEW").sum())
    )
    return {
        "signal_dates": int(score["asof"].nunique()),
        "stage1_pass_rows": int(score["stage1_pass"].sum()),
        "candidate_view_rows": int(candidate["candidate_status"].eq("IN_VIEW").sum()),
        "active_signal_days": int(daily_stage1.gt(0).sum()),
        "zero_candidate_days": int(daily_view.eq(0).sum()),
    }


def _portfolio_metrics(
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
    decisions = result["execution_decisions"]
    metrics.update({
        "entry_attempts": int(len(decisions)),
        "entry_fills": int(decisions["chosen"].sum()) if not decisions.empty else 0,
        "open_positions_at_end": int(len(result["open_positions"])),
        "active_signal_days": int(selection["active_signal_days"]),
        "zero_candidate_days": int(selection["zero_candidate_days"]),
    })
    return metrics


def _run_scenarios(
    ledgers: dict[str, pd.DataFrame],
    bundle: Any,
    strategy: StrategyConfig,
    execution_sessions: tuple[str, ...],
    selection: dict[str, int],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, pd.DataFrame]]]:
    portfolio: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, pd.DataFrame]] = {}
    for scenario in ("base", "stress"):
        result = run_backtest(
            candidate_ledger=ledgers["candidate"],
            selection_ledger=ledgers["selection"],
            execution_panel=bundle.execution,
            corporate_actions=bundle.corporate_actions,
            strategy=strategy,
            execution_sessions=execution_sessions,
            scenario_name=scenario,
        )
        results[scenario] = result
        portfolio[scenario] = _portfolio_metrics(result, strategy, selection)
    return portfolio, results


def _verify_control(
    expected_root: Path,
    actual: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = _read_json(expected_root / "PORTFOLIO_SUMMARY.json")
    fields = (
        "final_nav", "total_return", "max_drawdown", "portfolio_profit_factor",
        "trade_count", "trade_win_rate", "return_excluding_best_week",
        "return_excluding_top3_profit", "entry_attempts", "entry_fills",
        "open_positions_at_end", "active_signal_days", "zero_candidate_days",
    )
    checks: dict[str, Any] = {}
    for scenario in ("base", "stress"):
        for field in fields:
            target = expected[scenario][field]
            value = actual[scenario][field]
            passed = (
                value == target
                if isinstance(target, int) and not isinstance(target, bool)
                else np.isclose(float(value), float(target), rtol=0.0, atol=1e-10)
            )
            checks[f"{scenario}.{field}"] = {
                "actual": value, "expected": target, "passed": bool(passed)
            }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "sealed_summary_sha256": _sha256(expected_root / "PORTFOLIO_SUMMARY.json"),
    }


def _event_ledgers(
    bundle: Any,
    plan: Any,
    overlay: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    sessions = pd.DatetimeIndex(
        bundle.execution["trade_date"].drop_duplicates().sort_values()
    )
    signal_start = pd.Timestamp(overlay["control"]["signal_start"], tz="UTC")
    start_index = sessions.get_indexer([signal_start])[0]
    if start_index < 14:
        raise ValueError("execution bundle lacks event history for H10 overlay")
    # Nine prior state sessions plus the frozen V1 five-session observation path.
    event_start = sessions[start_index - 14]
    event_end = pd.Timestamp(overlay["control"]["signal_end"], tz="UTC")
    source = load_dragon_tiger_events(
        str(event_start.date()), str(event_end.date())
    )
    ledgers = build_pullback_reclaim_ledgers(
        source.events,
        bundle.execution,
        plan.signal_sessions,
        entries_per_decision=5,
    )
    return ledgers, {
        **source.manifest,
        "event_source_start": str(event_start.date()),
        "event_source_end": str(event_end.date()),
        "confirmed_events": int(ledgers["state"]["status"].eq("CONFIRMED").sum()),
    }


def _apply_overlay(
    control_ledgers: dict[str, pd.DataFrame],
    event_candidates: pd.DataFrame,
    overlay: dict[str, Any],
    strategy: StrategyConfig,
    sessions: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    candidate = control_ledgers["candidate"].copy()
    selection = control_ledgers["selection"].copy()
    session_index = {day: index for index, day in enumerate(sessions)}
    window = int(overlay["overlay"]["state_window_sessions"])
    confirmations: dict[str, list[tuple[int, pd.Timestamp]]] = {}
    for row in event_candidates.itertuples(index=False):
        day = pd.Timestamp(row.asof)
        if day in session_index:
            confirmations.setdefault(str(row.ts_code), []).append((session_index[day], day))

    excluded_rows: list[dict[str, Any]] = []
    in_view = candidate["candidate_status"].eq("IN_VIEW")
    for index, row in candidate.loc[in_view].iterrows():
        signal = pd.Timestamp(row["asof"])
        signal_index = session_index[signal]
        eligible = [
            (event_index, event_day)
            for event_index, event_day in confirmations.get(str(row["ts_code"]), [])
            if 0 <= signal_index - event_index < window
        ]
        if not eligible:
            continue
        event_index, event_day = max(eligible)
        candidate.at[index, "candidate_status"] = "DTR_H10_EXCLUDED"
        excluded_rows.append({
            "asof": signal,
            "ts_code": str(row["ts_code"]),
            "candidate_rank": int(row["candidate_rank"]),
            "confirmation_session": event_day,
            "event_age_sessions": int(signal_index - event_index),
        })

    empty_snapshot = hashlib.sha256(b"").hexdigest()
    snapshots: dict[pd.Timestamp, str] = {}
    for day in pd.DatetimeIndex(selection["asof"]):
        active = candidate[
            candidate["asof"].eq(day) & candidate["candidate_status"].eq("IN_VIEW")
        ].sort_values(["candidate_rank", "ts_code"], kind="mergesort")
        payload = "|".join(
            f"{row.ts_code}:{int(row.candidate_rank)}" for row in active.itertuples()
        )
        snapshots[day] = hashlib.sha256(payload.encode()).hexdigest() if payload else empty_snapshot
    candidate["candidate_snapshot_id"] = candidate["asof"].map(snapshots)

    policy_hash = _json_hash({
        "overlay": overlay,
        "control_strategy_hash": strategy.config_hash,
    })
    selection["candidate_snapshot_id"] = selection["asof"].map(snapshots)
    selection["policy_id"] = overlay["identity"]["overlay_id"]
    selection["policy_hash"] = policy_hash
    selection["decision_id"] = selection.apply(
        lambda row: hashlib.sha256(
            f"{policy_hash}|{pd.Timestamp(row['asof']).date()}|{row['candidate_snapshot_id']}".encode()
        ).hexdigest(),
        axis=1,
    )
    selection["context_hash"] = selection["asof"].map(
        lambda day: hashlib.sha256(
            f"{pd.Timestamp(day).date()}|{strategy.config_hash}|{policy_hash}".encode()
        ).hexdigest()
    )
    excluded = pd.DataFrame(excluded_rows, columns=[
        "asof", "ts_code", "candidate_rank", "confirmation_session", "event_age_sessions"
    ])
    return {
        "score": control_ledgers["score"],
        "candidate": candidate,
        "selection": selection,
    }, excluded


def _overlap_summary(
    excluded: pd.DataFrame,
    control_results: dict[str, dict[str, pd.DataFrame]],
    overlay_results: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "excluded_candidate_rows": int(len(excluded)),
        "excluded_signal_days": int(excluded["asof"].nunique()) if not excluded.empty else 0,
        "excluded_symbols": int(excluded["ts_code"].nunique()) if not excluded.empty else 0,
        "event_age_counts": (
            {str(key): int(value) for key, value in excluded["event_age_sessions"].value_counts().sort_index().items()}
            if not excluded.empty else {}
        ),
        "scenarios": {},
    }
    excluded_keys = set(zip(excluded["asof"], excluded["ts_code"]))
    for scenario in ("base", "stress"):
        control = control_results[scenario]["execution_decisions"]
        overlay = overlay_results[scenario]["execution_decisions"]
        control_chosen = control[control["chosen"].astype(bool)]
        overlay_chosen = overlay[overlay["chosen"].astype(bool)]
        affected = [
            (row.signal_session, str(row.ts_code))
            for row in control_chosen.itertuples(index=False)
            if (row.signal_session, str(row.ts_code)) in excluded_keys
        ]
        control_keys = set(zip(control_chosen["signal_session"], control_chosen["ts_code"]))
        overlay_keys = set(zip(overlay_chosen["signal_session"], overlay_chosen["ts_code"]))
        summary["scenarios"][scenario] = {
            "executed_control_entries_excluded": int(len(affected)),
            "control_entry_count": int(len(control_chosen)),
            "overlay_entry_count": int(len(overlay_chosen)),
            "removed_entry_keys": int(len(control_keys - overlay_keys)),
            "replacement_entry_keys": int(len(overlay_keys - control_keys)),
        }
    return summary


def _acceptance(
    control: dict[str, dict[str, Any]],
    challenger: dict[str, dict[str, Any]],
    overlap: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    rules = config["evaluation"]
    scenarios: dict[str, Any] = {}
    for scenario in ("base", "stress"):
        base = control[scenario]
        item = challenger[scenario]
        tests = {
            "affected_executed_entry": overlap["scenarios"][scenario]["executed_control_entries_excluded"] > 0,
            "higher_total_return": float(item["total_return"]) > float(base["total_return"]),
            "pf_not_lower": float(item["portfolio_profit_factor"] or 0.0) >= float(base["portfolio_profit_factor"]),
            "pf_minimum": float(item["portfolio_profit_factor"] or 0.0) >= float(rules["portfolio_profit_factor_min"]),
            "maxdd_not_worse": abs(float(item["max_drawdown"])) <= abs(float(base["max_drawdown"])),
            "maxdd_limit": abs(float(item["max_drawdown"])) <= float(rules["max_drawdown_abs_max"]),
            "excluding_best_week_positive": float(item["return_excluding_best_week"]) > 0.0,
            "minimum_closed_trades": int(item["trade_count"]) >= int(rules["minimum_closed_trades"]),
            "position_cap": int(item["max_open_positions"]) <= int(rules["maximum_positions"]),
        }
        scenarios[scenario] = {"passed": all(tests.values()), "tests": tests}
    return {
        "passed": all(item["passed"] for item in scenarios.values()),
        "scenarios": scenarios,
    }


def _comparison(
    control: dict[str, dict[str, Any]],
    challenger: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fields = (
        "total_return", "portfolio_profit_factor", "max_drawdown", "trade_win_rate",
        "return_excluding_best_week", "return_excluding_top3_profit", "trade_count",
    )
    return {
        scenario: {
            field: {
                "control": control[scenario][field],
                "overlay": challenger[scenario][field],
                "delta": (
                    float(challenger[scenario][field]) - float(control[scenario][field])
                    if control[scenario][field] is not None and challenger[scenario][field] is not None
                    else None
                ),
            }
            for field in fields
        }
        for scenario in ("base", "stress")
    }


def _write_result(
    output: Path,
    control: dict[str, dict[str, Any]],
    overlay: dict[str, dict[str, Any]],
    overlap: dict[str, Any],
    acceptance: dict[str, Any],
) -> None:
    lines = [
        "# CAP1 Dragon-Tiger H10 Exclusion", "",
        "## Contract", "",
        "- Historical 2026 diagnostic; same CAP1 rules, engine, costs, sizing, H10 hold, and stop.",
        "- One change: exclude a CAP1 candidate with an unchanged V1 confirmation in the current/prior nine sessions; preserve CAP1 rank and use normal fallback.",
        "- No XGBoost, frozen 202-factor input, parameter/window scan, or persisted business data.", "",
        "## Overlap", "",
        f"- Excluded candidate rows: `{overlap['excluded_candidate_rows']}` across `{overlap['excluded_signal_days']}` signal days and `{overlap['excluded_symbols']}` symbols.",
        f"- Executed Base control entries affected: `{overlap['scenarios']['base']['executed_control_entries_excluded']}`; replacement entries: `{overlap['scenarios']['base']['replacement_entry_keys']}`.", "",
        "## Portfolio", "",
        "| Cost | Control return | Overlay return | Delta | Control PF | Overlay PF | Control MaxDD | Overlay MaxDD | Trades | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for scenario in ("base", "stress"):
        base = control[scenario]
        item = overlay[scenario]
        lines.append(
            f"| {scenario} | {base['total_return']:+.2%} | {item['total_return']:+.2%} | "
            f"{item['total_return'] - base['total_return']:+.2%} | "
            f"{base['portfolio_profit_factor']:.3f} | {item['portfolio_profit_factor']:.3f} | "
            f"{base['max_drawdown']:.2%} | {item['max_drawdown']:.2%} | "
            f"{item['trade_count']} | {acceptance['scenarios'][scenario]['passed']} |"
        )
    lines.extend(["", "## Decision", ""])
    lines.append(
        "The H10 exclusion overlay advances unchanged."
        if acceptance["passed"]
        else "The H10 exclusion overlay is rejected unchanged and will not be repaired with a window or threshold scan."
    )
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    strategy = StrategyConfig.from_yaml(args.strategy)
    config = _load_overlay(args.overlay, strategy)
    control_cfg = config["control"]
    calendar_start = str(
        (pd.Timestamp(control_cfg["signal_start"]) - pd.Timedelta(days=500)).date()
    )
    calendar = load_trading_calendar(calendar_start, control_cfg["execution_end"])
    request = RunRequest(
        signal_start=control_cfg["signal_start"],
        signal_end=control_cfg["signal_end"],
        execution_end=control_cfg["execution_end"],
        output_dir=str(output),
        run_name="cap1_dragon_tiger_h10_exclusion_v1_2026",
    )
    plan = compile_run_plan(strategy, request, calendar["session"])
    bundle = build_data_bundle(plan, strategy, output)
    features = build_feature_ledger(bundle, strategy)
    control_ledgers = build_rule_ledgers(features, strategy, plan.signal_sessions)
    control_selection = _selection_summary(control_ledgers)
    control_portfolio, control_results = _run_scenarios(
        control_ledgers, bundle, strategy, plan.execution_sessions, control_selection
    )
    reproduction = _verify_control(args.control_result.resolve(), control_portfolio)
    if not reproduction["passed"]:
        raise ValueError("same-bundle CAP1 control does not reproduce sealed metrics")

    event_ledgers, event_manifest = _event_ledgers(bundle, plan, config)
    all_sessions = pd.DatetimeIndex(
        bundle.execution["trade_date"].drop_duplicates().sort_values()
    )
    overlay_ledgers, excluded = _apply_overlay(
        control_ledgers, event_ledgers["candidate"], config, strategy, all_sessions
    )
    overlay_selection = _selection_summary(overlay_ledgers)
    overlay_portfolio, overlay_results = _run_scenarios(
        overlay_ledgers, bundle, strategy, plan.execution_sessions, overlay_selection
    )
    overlap = _overlap_summary(excluded, control_results, overlay_results)
    acceptance = _acceptance(control_portfolio, overlay_portfolio, overlap, config)
    comparison = _comparison(control_portfolio, overlay_portfolio)

    pit_passed = bool(
        excluded.empty
        or (
            excluded["confirmation_session"].le(excluded["asof"]).all()
            and excluded["event_age_sessions"].between(0, 9).all()
        )
    )
    verification = {
        "passed": bool(reproduction["passed"] and pit_passed),
        "control_reproduction": reproduction,
        "pit_confirmation_not_after_signal": pit_passed,
        "parameter_sweep": False,
        "business_data_persisted": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "configs").mkdir()
    (output / "manifests").mkdir()
    shutil.copyfile(args.strategy, output / "configs/control_strategy.yaml")
    shutil.copyfile(args.overlay, output / "configs/overlay.yaml")
    _write_json(output / "RUN_STATUS.json", {
        "status": "DIAGNOSTIC_COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": strategy.strategy_id,
        "overlay_id": config["identity"]["overlay_id"],
        "control_reproduced": reproduction["passed"],
        "verification_passed": verification["passed"],
        "acceptance_passed": acceptance["passed"],
        "credentials_persisted": False,
        "business_data_persisted": False,
    })
    _write_json(output / "plan.json", {
        **plan.to_dict(),
        "event_source_start": event_manifest["event_source_start"],
        "event_source_end": event_manifest["event_source_end"],
        "overlay_hash": _json_hash(config),
    })
    _write_json(output / "event_manifest.json", event_manifest)
    _write_json(output / "control_metrics.json", control_portfolio)
    _write_json(output / "overlay_metrics.json", overlay_portfolio)
    _write_json(output / "comparison.json", comparison)
    _write_json(output / "overlap_summary.json", overlap)
    _write_json(output / "acceptance.json", acceptance)
    _write_json(output / "verification.json", verification)
    _write_result(output, control_portfolio, overlay_portfolio, overlap, acceptance)
    code_paths = [
        ROOT / "src/aistock9988/features/engine.py",
        ROOT / "src/aistock9988/selection/pipeline.py",
        ROOT / "src/aistock9988/selection/dragon_tiger.py",
        ROOT / "src/aistock9988/backtest/engine.py",
        Path(__file__).resolve(),
    ]
    _write_json(output / "manifests/code_manifest.json", {
        str(path.relative_to(ROOT)): _sha256(path) for path in code_paths
    })
    artifacts = {
        str(path.relative_to(output)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json(output / "manifests/artifact_manifest.json", artifacts)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--control-result", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"run_complete={run(args)}")


if __name__ == "__main__":
    main()
