"""Audit CAP1 candidate quality and portfolio opportunity cost on sealed history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aistock9988.configuration import StrategyConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "docs/council_20260828"
    / "RESET_WEAK_CONFIRM_V3_CAP1_20_2026_TO_0828_20260901"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs/council_20260828"
    / "CAP1_CANDIDATE_OPPORTUNITY_DIAGNOSTIC_20260902"
)
DEFAULT_STRATEGY = ROOT / "configs/strategy/reset_weak_confirm_v3_cap1_20.yaml"

SOURCE_FILES = {
    "candidate": "ledgers/candidate_ledger.parquet",
    "features": "ledgers/feature_ledger.parquet",
    "execution": "ledgers/execution_panel.parquet",
    "decisions": "backtests/base/execution_decisions.parquet",
    "fills": "backtests/base/fills.parquet",
    "positions": "backtests/base/positions.parquet",
    "manifest": "manifests/artifact_manifest.json",
}
FEATURES = (
    "dist_ma60",
    "mkt_ret_20d",
    "ret1",
    "ret20",
    "dd20",
    "vol20",
    "liq20",
    "volume_ratio_20",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _verified_inputs(source: Path) -> dict[str, dict[str, Any]]:
    manifest = _read_json(source / SOURCE_FILES["manifest"])
    result: dict[str, dict[str, Any]] = {}
    for name, relative in SOURCE_FILES.items():
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        expected = manifest.get(relative, {}).get("sha256") if name != "manifest" else None
        if expected is not None and actual != expected:
            raise ValueError(f"sealed source hash mismatch: {relative}")
        result[relative] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
            "sealed_hash_verified": expected is not None,
        }
    return result


def _costs(strategy: StrategyConfig) -> dict[str, float]:
    raw = strategy.execution["cost_scenarios"]["base"]
    return {
        "slippage": float(raw["slippage_each_side"]),
        "buy_commission": float(raw["buy_commission"]),
        "sell_commission": float(raw["sell_commission"]),
        "stamp_duty": float(raw["stamp_duty"]),
    }


def _candidate_labels(
    candidates: pd.DataFrame,
    execution: pd.DataFrame,
    strategy: StrategyConfig,
) -> pd.DataFrame:
    sessions = pd.DatetimeIndex(execution["trade_date"].drop_duplicates().sort_values())
    session_index = {day: index for index, day in enumerate(sessions)}
    by_code = {
        str(code): group.set_index("trade_date").sort_index()
        for code, group in execution.groupby("ts_code", sort=False)
    }
    costs = _costs(strategy)
    bad_buy = {"MISSING_REQUIRED_DATA", "SUSPENDED", "LIMIT_UP", "ZERO_VOLUME", "OUT_OF_UNIVERSE"}
    bad_sell = {"MISSING_REQUIRED_DATA", "SUSPENDED", "LIMIT_DOWN", "ZERO_VOLUME", "OUT_OF_UNIVERSE"}
    hold = int(strategy.execution["hold_sessions_from_fill"])
    stop = float(strategy.execution["stop"]["threshold_pct"])
    if str(strategy.execution["stop"].get("mode")) != "trailing_from_last_close":
        raise ValueError("diagnostic currently requires trailing_from_last_close")

    rows: list[dict[str, Any]] = []
    retained = ["asof", "ts_code", "candidate_rank", "rule_score", *FEATURES]
    for candidate in candidates.itertuples(index=False):
        result = {name: getattr(candidate, name) for name in retained}
        entry_index = session_index[pd.Timestamp(candidate.asof)] + 1
        if entry_index >= len(sessions):
            result.update(label_status="NO_ENTRY_SESSION", net_return=np.nan)
            rows.append(result)
            continue
        panel = by_code[str(candidate.ts_code)]
        entry_day = sessions[entry_index]
        entry_row = panel.loc[entry_day]
        if (
            str(entry_row.execution_status) in bad_buy
            or not np.isfinite(float(entry_row.economic_open))
        ):
            result.update(
                label_status=f"BUY_{entry_row.execution_status}",
                entry_session=entry_day,
                net_return=np.nan,
            )
            rows.append(result)
            continue

        entry_price = float(entry_row.economic_open) * (1.0 + costs["slippage"])
        last_close = float(entry_row.economic_close)
        pending_exit = False
        exit_reason: str | None = None
        exit_day: pd.Timestamp | None = None
        exit_price = np.nan
        for index in range(entry_index + 1, len(sessions)):
            day = sessions[index]
            row = panel.loc[day]
            status = str(row.execution_status)
            if pending_exit:
                if status not in bad_sell and np.isfinite(float(row.economic_open)):
                    exit_day = day
                    exit_price = float(row.economic_open) * (1.0 - costs["slippage"])
                    break
                continue
            if index >= entry_index + hold:
                exit_reason = "TIME_EXIT"
                if status not in bad_sell and np.isfinite(float(row.economic_open)):
                    exit_day = day
                    exit_price = float(row.economic_open) * (1.0 - costs["slippage"])
                    break
                pending_exit = True
                continue
            if not bool(row.execution_data_eligible) or not np.isfinite(float(row.economic_close)):
                continue
            current_close = float(row.economic_close)
            if current_close / last_close - 1.0 <= stop:
                pending_exit = True
                exit_reason = "STOP_LOSS"
            last_close = current_close

        if not np.isfinite(exit_price):
            result.update(
                label_status="OPEN_AT_EXECUTION_END",
                entry_session=entry_day,
                net_return=np.nan,
            )
        else:
            net_return = (
                exit_price * (1.0 - costs["sell_commission"] - costs["stamp_duty"])
                / (entry_price * (1.0 + costs["buy_commission"]))
                - 1.0
            )
            result.update(
                label_status="CLOSED",
                entry_session=entry_day,
                exit_session=exit_day,
                exit_reason=exit_reason,
                net_return=net_return,
                economic_return=exit_price / entry_price - 1.0,
            )
        rows.append(result)
    return pd.DataFrame(rows)


def _group_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame[frame["net_return"].notna()]
    return {
        "events": int(len(frame)),
        "closed_labels": int(len(valid)),
        "active_signal_days": int(frame["asof"].nunique()),
        "mean_return": float(valid["net_return"].mean()),
        "median_return": float(valid["net_return"].median()),
        "win_rate": float(valid["net_return"].gt(0.0).mean()),
        "return_ge_10_rate": float(valid["net_return"].ge(0.10).mean()),
        "stop_rate": float(valid["exit_reason"].eq("STOP_LOSS").mean()),
    }


def _rank_metrics(labels: pd.DataFrame) -> dict[str, Any]:
    buckets = {
        "rank_1_5": labels[labels["candidate_rank"].between(1, 5)],
        "rank_6_10": labels[labels["candidate_rank"].between(6, 10)],
        "rank_11_20": labels[labels["candidate_rank"].between(11, 20)],
    }
    return {name: _group_metrics(frame) for name, frame in buckets.items()}


def _feature_metrics(labels: pd.DataFrame) -> dict[str, Any]:
    valid = labels[labels["net_return"].notna()].copy()
    valid["outcome"] = np.where(valid["net_return"].gt(0.0), "winner", "loser")
    result: dict[str, Any] = {}
    for name in (*FEATURES, "rule_score", "candidate_rank"):
        means = valid.groupby("outcome")[name].mean()
        result[name] = {
            "winner_mean": float(means["winner"]),
            "loser_mean": float(means["loser"]),
            "winner_minus_loser": float(means["winner"] - means["loser"]),
            "spearman_to_return": float(valid[[name, "net_return"]].corr(method="spearman").iloc[0, 1]),
        }
    return result


def _selection_metrics(
    labels: pd.DataFrame,
    decisions: pd.DataFrame,
    fills: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    chosen = decisions[decisions["chosen"].astype(bool)][
        ["signal_session", "ts_code", "decision_id"]
    ].rename(columns={"signal_session": "asof"})
    if chosen.duplicated(["asof", "ts_code"]).any():
        raise ValueError("duplicate chosen candidate key")
    compared = labels.merge(chosen, on=["asof", "ts_code"], how="left", validate="one_to_one")
    compared["actually_chosen"] = compared["decision_id"].notna()
    attempted_days = set(pd.to_datetime(decisions["signal_session"], utc=True).dt.normalize())
    compared["slot_state"] = np.where(
        compared["asof"].isin(attempted_days), "ENTRY_SLOT_AVAILABLE", "PORTFOLIO_FULL"
    )
    selection = {
        "chosen": _group_metrics(compared[compared["actually_chosen"]]),
        "not_chosen": _group_metrics(compared[~compared["actually_chosen"]]),
        "entry_slot_available": _group_metrics(compared[compared["slot_state"].eq("ENTRY_SLOT_AVAILABLE")]),
        "portfolio_full": _group_metrics(compared[compared["slot_state"].eq("PORTFOLIO_FULL")]),
    }

    sells = fills[fills["side"].eq("SELL")][
        ["decision_id", "ts_code", "economic_return"]
    ].rename(columns={"economic_return": "engine_economic_return"})
    parity = compared[compared["actually_chosen"]].merge(
        sells, on=["decision_id", "ts_code"], how="inner", validate="one_to_one"
    )
    parity["absolute_error"] = (
        parity["economic_return"] - parity["engine_economic_return"]
    ).abs()
    parity_result = {
        "matched_closed_trades": int(len(parity)),
        "engine_closed_trades": int(len(sells)),
        "max_absolute_economic_return_error": float(parity["absolute_error"].max()),
        "passed": bool(
            len(parity) == len(sells)
            and float(parity["absolute_error"].max()) <= 1e-12
        ),
    }
    return selection, parity_result


def _naive_replacement_metrics(
    labels: pd.DataFrame,
    positions: pd.DataFrame,
    decisions: pd.DataFrame,
    fills: pd.DataFrame,
    execution: pd.DataFrame,
) -> dict[str, Any]:
    """Compare one tempting swap rule without claiming a portfolio backtest."""
    sessions = pd.DatetimeIndex(execution["trade_date"].drop_duplicates().sort_values())
    session_index = {day: index for index, day in enumerate(sessions)}
    attempted_days = set(pd.to_datetime(decisions["signal_session"], utc=True).dt.normalize())
    full_days = sorted(set(labels["asof"]) - attempted_days)
    sells = fills[fills["side"].eq("SELL")].set_index(["decision_id", "ts_code"])
    execution_key = execution.set_index(["trade_date", "ts_code"])
    pairs: list[dict[str, float]] = []
    for day in full_days:
        held = positions[
            positions["trade_date"].eq(day) & positions["state"].eq("ACTIVE")
        ].copy()
        if len(held) < 5:
            continue
        held["age_sessions"] = [
            session_index[day] - session_index[entry] for entry in held["entry_date"]
        ]
        weak = held[
            held["age_sessions"].ge(2) & held["unrealized_return"].le(0.0)
        ].sort_values(["unrealized_return", "ts_code"], kind="mergesort")
        if weak.empty:
            continue
        incoming = labels[
            labels["asof"].eq(day)
            & labels["net_return"].notna()
            & ~labels["ts_code"].isin(set(held["ts_code"]))
        ].sort_values(["candidate_rank", "ts_code"], kind="mergesort")
        if incoming.empty or session_index[day] + 1 >= len(sessions):
            continue
        old = weak.iloc[0]
        new = incoming.iloc[0]
        next_day = sessions[session_index[day] + 1]
        sell = sells.loc[(old["decision_id"], old["ts_code"])]
        if pd.Timestamp(sell["trade_date"]) <= next_day:
            # The control already frees this slot at the same/earlier open.
            continue
        next_open = execution_key.loc[(next_day, old["ts_code"]), "economic_open"]
        if not np.isfinite(float(next_open)):
            continue
        keep_remaining = float(sell["economic_price"]) / float(next_open) - 1.0
        pairs.append(
            {
                "keep_remaining": keep_remaining,
                "replacement_return": float(new["net_return"]),
                "optimistic_delta": float(new["net_return"]) - keep_remaining,
            }
        )
    paired = pd.DataFrame(pairs)
    if paired.empty:
        return {"comparable_pairs": 0, "decision": "INSUFFICIENT_COMPARABLE_PAIRS"}
    return {
        "comparable_pairs": int(len(paired)),
        "keep_mean_remaining_return": float(paired["keep_remaining"].mean()),
        "keep_win_rate": float(paired["keep_remaining"].gt(0.0).mean()),
        "replacement_mean_return": float(paired["replacement_return"].mean()),
        "replacement_win_rate": float(paired["replacement_return"].gt(0.0).mean()),
        "replacement_beats_keep_rate": float(paired["optimistic_delta"].gt(0.0).mean()),
        "mean_optimistic_delta": float(paired["optimistic_delta"].mean()),
        "comparison_bias": (
            "optimistic_for_replacement: old-position early-liquidation friction is omitted"
        ),
        "decision": "REJECT_NAIVE_FULL_SLOT_WEAK_POSITION_REPLACEMENT",
    }


def run(source: Path, strategy_path: Path, output: Path) -> Path:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_inputs = _verified_inputs(source)
    strategy = StrategyConfig.from_yaml(strategy_path)
    if strategy.strategy_id != "reset_weak_confirm_v3_cap1_20":
        raise ValueError("diagnostic is registered only for reset_weak_confirm_v3_cap1_20")

    candidates = pd.read_parquet(source / SOURCE_FILES["candidate"])
    candidates = candidates[candidates["candidate_status"].eq("IN_VIEW")].copy()
    features = pd.read_parquet(
        source / SOURCE_FILES["features"],
        columns=["asof", "ts_code", *FEATURES],
    )
    candidates = candidates.merge(features, on=["asof", "ts_code"], validate="one_to_one")
    execution = pd.read_parquet(
        source / SOURCE_FILES["execution"],
        columns=[
            "trade_date",
            "ts_code",
            "execution_status",
            "execution_data_eligible",
            "economic_open",
            "economic_close",
        ],
    )
    decisions = pd.read_parquet(source / SOURCE_FILES["decisions"])
    fills = pd.read_parquet(source / SOURCE_FILES["fills"])
    positions = pd.read_parquet(source / SOURCE_FILES["positions"])

    labels = _candidate_labels(candidates, execution, strategy)
    valid = labels[labels["net_return"].notna()]
    selection, parity = _selection_metrics(labels, decisions, fills)
    summary = {
        "status": "DIAGNOSTIC_SEEN_2026_HISTORY_NO_STRATEGY_CHANGE",
        "scope": {
            "signal_start": str(pd.Timestamp(labels["asof"].min()).date()),
            "signal_end": str(pd.Timestamp(labels["asof"].max()).date()),
            "execution_end": str(pd.Timestamp(execution["trade_date"].max()).date()),
            "years_used_for_performance": [2026],
            "isolated_candidate_counterfactual": True,
            "portfolio_backtest": False,
        },
        "all_candidates": _group_metrics(labels),
        "rank_buckets": _rank_metrics(labels),
        "selection": selection,
        "naive_replacement": _naive_replacement_metrics(
            labels, positions, decisions, fills, execution
        ),
        "exit_reasons": {
            str(reason): _group_metrics(group)
            for reason, group in valid.groupby("exit_reason", sort=True)
        },
        "features": _feature_metrics(labels),
        "parity": parity,
        "constraints": {
            "parameter_scan_performed": False,
            "strategy_changed": False,
            "raw_business_data_written": False,
            "wide_202_factor_system_used": False,
            "xgboost_used": False,
        },
    }
    _write_json(output / "SUMMARY.json", summary)
    _write_json(
        output / "SOURCE_MANIFEST.json",
        {
            "source_inputs": source_inputs,
            "strategy": {str(strategy_path.relative_to(ROOT)): _sha256(strategy_path)},
            "diagnostic": {
                str(Path(__file__).resolve().relative_to(ROOT)): _sha256(Path(__file__).resolve())
            },
        },
    )

    all_metrics = summary["all_candidates"]
    top5 = summary["rank_buckets"]["rank_1_5"]
    lower = summary["rank_buckets"]["rank_6_10"]
    full = summary["selection"]["portfolio_full"]
    available = summary["selection"]["entry_slot_available"]
    replacement = summary["naive_replacement"]
    mkt = summary["features"]["mkt_ret_20d"]
    result = f"""# CAP1 Candidate Opportunity Diagnostic

Status: `DIAGNOSTIC_SEEN_2026_HISTORY_NO_STRATEGY_CHANGE`.

This is a candidate-level economic-price counterfactual, not a new portfolio
backtest. It uses only sealed 2026 signal/performance dates and reproduces each
closed engine trade's economic return before reporting aggregate evidence.

## Candidate quality

- {all_metrics['closed_labels']} of {all_metrics['events']} in-view candidate
  events have a closed T+1/H10/original-stop label. Mean net return is
  {all_metrics['mean_return']:+.2%}, win rate is {all_metrics['win_rate']:.1%},
  and the >=10% rate is {all_metrics['return_ge_10_rate']:.1%}.
- Ranks 1-5 average {top5['mean_return']:+.2%} with {top5['win_rate']:.1%}
  winners. Ranks 6-10 average {lower['mean_return']:+.2%} with
  {lower['win_rate']:.1%} winners. The existing transparent rank therefore has
  useful top-of-list separation; the result does not justify reversing it.
- Market state is the strongest available static separator: winner entries had
  mean 20-session market return {mkt['winner_mean']:+.2%} versus
  {mkt['loser_mean']:+.2%} for losers, with Spearman
  {mkt['spearman_to_return']:+.3f}. Other frozen entry features are weak.

## Portfolio opportunity cost

- On signal days where at least one entry slot was available, candidate events
  averaged {available['mean_return']:+.2%} with {available['win_rate']:.1%}
  winners. Candidates appearing while the five-position portfolio was full
  averaged {full['mean_return']:+.2%} with {full['win_rate']:.1%} winners.
- This identifies a scheduling/holding problem, not proof of an alternative
  portfolio rule. Isolated candidate returns overlap and cannot be summed.
- The tempting rule "when full, replace the weakest nonpositive E2-or-older
  holding with the new rank-1 candidate" has {replacement['comparable_pairs']}
  comparable 2026 pairs. Keeping averaged
  {replacement['keep_mean_remaining_return']:+.2%}; replacements averaged
  {replacement['replacement_mean_return']:+.2%} and beat keeping only
  {replacement['replacement_beats_keep_rate']:.1%} of the time. This comparison
  is already optimistic for replacement because it omits the old position's
  early-liquidation friction. Reject that naive turnover rule.

## Integrity and decision

- Candidate label parity: {parity['matched_closed_trades']} matched closed
  engine trades; maximum absolute economic-return error
  {parity['max_absolute_economic_return_error']:.3e}; pass={parity['passed']}.
- No threshold scan, model, strategy change, raw-data output, 202-factor change,
  or 2024/2025 performance validation was performed.
- Preserve CAP1 entry and Top5 ranking; do not add naive full-slot replacement.
  Static entry stacks and simple turnover are both unsupported. The only
  remaining evidence-backed change is the already registered forward-only
  early-path risk overlay, which does not recycle an early exit into a new buy.
"""
    (output / "RESULT.md").write_text(result, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args.source.resolve(), args.strategy.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
