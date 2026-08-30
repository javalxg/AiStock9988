"""Compare frozen Stage-1 mechanisms without changing thresholds or training a model."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.labeling.q70 import build_q70_t10_labels
from aistock9988.selection.rcqt import score_rcqt, select_rcqt
from aistock9988.time.session import session_close

from rcqt_quantdb_sample_runner import _features
from rcqt_stage1_quality_runner import (
    LABEL_PROFILE,
    _backtest,
    _load_pit_st_keys,
    _load_sources,
    _metrics,
    _rank_skill,
    _sha,
    _uplift,
    _write_json,
)


def _top(frame: pd.DataFrame, score: str, policy: str, count: int = 4) -> pd.DataFrame:
    out = frame.sort_values(
        ["asof", score, "ts_code"], ascending=[True, False, True], kind="mergesort",
    ).groupby("asof", sort=True).head(count).copy()
    out["candidate_rank"] = out.groupby("asof").cumcount() + 1
    out["selected"] = True
    out["selection_decision_id"] = policy + "-" + out["asof"].dt.strftime("%Y%m%d")
    out["policy_id"] = policy
    out["target_weight"] = 0.12
    return out


def _formal_rcqt(scored: pd.DataFrame) -> pd.DataFrame:
    selected = select_rcqt(scored, reset_slots=4, quiet_slots=2, equity_cap=0.72)
    selected["candidate_rank"] = selected["selection_rank"]
    return selected


def _event_summary(control: pd.DataFrame, cohorts: dict[str, pd.DataFrame],
                   selections: dict[str, pd.DataFrame]) -> dict[str, object]:
    result: dict[str, object] = {"universe_control": _metrics(control), "cohorts": {}, "selections": {}}
    for name, frame in cohorts.items():
        result["cohorts"][name] = {
            "overall": _metrics(frame),
            "vs_universe": _uplift(control, frame),
            "monthly": {},
        }
        for month, group in frame.groupby(frame["asof"].dt.strftime("%Y-%m")):
            month_control = control[control["asof"].dt.strftime("%Y-%m") == month]
            result["cohorts"][name]["monthly"][month] = {
                "metrics": _metrics(group),
                "vs_universe": _uplift(month_control, group),
            }
    for name, frame in selections.items():
        result["selections"][name] = {"overall": _metrics(frame), "monthly": {}}
        score = "reset_score" if name == "reset_top4" else "quiet_score"
        parent_name = "reset_confirmed" if name == "reset_top4" else "quiet_confirmed"
        if name != "formal_rcqt_4plus2":
            parent = cohorts[parent_name]
            result["selections"][name]["vs_parent_pool"] = _uplift(parent, frame)
            result["selections"][name]["rank_skill"] = _rank_skill(parent, score)
        for month, group in frame.groupby(frame["asof"].dt.strftime("%Y-%m")):
            result["selections"][name]["monthly"][month] = _metrics(group)
    return result


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    codes = sorted(pd.read_parquet(args.codes_source, columns=["ts_code"])["ts_code"].astype(str).unique())
    raw_start = (pd.Timestamp(args.start) - pd.Timedelta(days=140)).strftime("%Y-%m-%d")
    st_keys, st_audit = _load_pit_st_keys(codes, args.start, args.end)
    panel, prices, source_audit = _load_sources(raw_start, args.end, codes)
    features = _features(panel, prices, args.start)
    features["asof"] = pd.to_datetime(features["asof"], utc=True).dt.normalize()
    features["available_time"] = pd.to_datetime(features["available_time"], utc=True)
    features = features[features["asof"].between(args.start, args.end)].copy()
    if features.empty or features["asof"].max() < pd.Timestamp(args.end, tz="UTC"):
        raise RuntimeError(f"feature coverage ends at {features['asof'].max()}, before {args.end}")
    if (features["available_time"] > features["asof"].map(session_close)).any():
        raise AssertionError("feature PIT violation")

    scored = score_rcqt(features)
    scored["pit_st"] = [
        (day, code) in st_keys for day, code in zip(scored["asof"], scored["ts_code"].astype(str))
    ]
    scored = scored[~scored["pit_st"]].copy()
    candidates = {
        "quiet_confirmed": scored[scored["quiet_eligible"] & scored["right_confirmed"]].copy(),
        "reset_confirmed": scored[scored["reset_eligible"]].copy(),
        "formal_rcqt_union": scored[scored["reset_eligible"] | scored["quiet_eligible"]].copy(),
    }
    selections = {
        "quiet_confirmed_top4": _top(
            candidates["quiet_confirmed"], "quiet_score", "rcqt.stage1.quiet_confirmed.top4.v1",
        ),
        "reset_top4": _top(
            candidates["reset_confirmed"], "reset_score", "rcqt.stage1.reset_confirmed.top4.v1",
        ),
        "formal_rcqt_4plus2": _formal_rcqt(scored),
    }

    sessions = pd.DatetimeIndex(sorted(panel["event_time"].drop_duplicates()))
    labels = build_q70_t10_labels(panel, profile=LABEL_PROFILE, session_dates=sessions)
    labels = labels.rename(columns={"event_time": "asof", "available_time": "label_available_time"})
    labels["asof"] = pd.to_datetime(labels["asof"], utc=True).dt.normalize()
    labels["label_available_time"] = pd.to_datetime(labels["label_available_time"], utc=True)
    mature_labels = labels[
        labels["label_available_time"] <= session_close(pd.Timestamp(args.end))
    ][["asof", "ts_code", "label_return", "label_available_time", "exit_time"]]
    mature_scored = scored.merge(mature_labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
    mature_candidates = {
        name: frame.merge(mature_labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
        for name, frame in candidates.items()
    }
    mature_selections = {
        name: frame.merge(mature_labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
        for name, frame in selections.items()
    }
    mature_signal_end = min(frame["asof"].max() for frame in mature_selections.values())
    event_summary = _event_summary(mature_scored, mature_candidates, mature_selections)
    event_summary["mature_through_signal_date"] = str(mature_signal_end.date())
    event_summary["contracts"] = {
        "quiet_confirmed": "quiet_eligible AND right_confirmed AND NOT PIT-ST",
        "reset_confirmed": "frozen reset_eligible (includes right confirmation) AND NOT PIT-ST",
        "formal_rcqt_union": "reset_eligible OR quiet_eligible AND NOT PIT-ST",
        "parameter_sweep": False,
        "model_training": False,
    }

    px = prices.copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"], utc=True).dt.normalize()
    px["amount"] = pd.to_numeric(px["amount"], errors="raise")
    px = px.sort_values(["ts_code", "trade_date"], kind="mergesort")
    px["adv20"] = px.groupby("ts_code")["amount"].transform(
        lambda series: series.rolling(20, min_periods=20).median()
    )
    px = px[px["trade_date"].between(args.start, args.end)].copy()
    actions = load_corporate_actions(args.start, args.end, ts_codes=codes)
    portfolio_summary: dict[str, object] = {}
    for policy, ledger in selections.items():
        signals = ledger[ledger["asof"] <= mature_signal_end].copy()
        for label, slippage in (("base", 0.001), ("stress", 0.003)):
            # The formal 4+2 policy retains its six-slot target; the two mechanism arms use four slots.
            if policy == "formal_rcqt_4plus2":
                from aistock9988.backtest.engine import BacktestConfig, run_backtest
                from aistock9988.reporting.metrics import summarize_backtest

                result = run_backtest(
                    signals,
                    px,
                    corporate_actions=actions,
                    config=BacktestConfig(
                        max_positions=6, hold_sessions=10, stop_loss_pct=-0.08,
                        stop_loss_mode="close_next_session_open", accounting_price_basis="raw",
                        lot_size=100, max_order_to_adv20=0.02,
                        buy_slippage=slippage, sell_slippage=slippage,
                    ),
                )
                metrics = summarize_backtest(result["nav"], result["trades"], initial_cash=1_000_000)
                metrics["slippage_each_side"] = slippage
                metrics["forced_final_liquidation_count"] = int(
                    (result["trades"].get("reason", pd.Series(dtype=str)) == "end_of_test_liquidation").sum()
                )
            else:
                result, metrics = _backtest(signals, px, actions, slippage=slippage)
            target = output / "backtests" / policy / label
            target.mkdir(parents=True, exist_ok=True)
            for name in ("orders", "trades", "nav", "positions", "corporate_actions"):
                result[name].to_csv(target / f"{name}.csv", index=False)
            _write_json(target / "metrics.json", metrics)
            portfolio_summary[f"{policy}_{label}"] = metrics

    for name, frame in candidates.items():
        frame.to_parquet(output / f"candidate_{name}.parquet", index=False)
    for name, frame in selections.items():
        frame.to_csv(output / f"selection_{name}.csv", index=False)
    _write_json(output / "EVENT_SUMMARY.json", event_summary)
    _write_json(output / "PORTFOLIO_SUMMARY.json", portfolio_summary)
    _write_json(output / "DATA_MANIFEST.json", {
        "kind": "stage1_frozen_mechanism_comparison",
        "requested_start": args.start,
        "requested_end": args.end,
        "raw_lookback_start": raw_start,
        "codes_source": str(args.codes_source.resolve()),
        "codes_source_sha256": _sha(args.codes_source),
        "fixed_universe_codes": len(codes),
        "mature_signal_end": str(mature_signal_end),
        "model_training": False,
        "parameter_sweep": False,
        "pit_st_audit": st_audit,
        "source_audit": source_audit,
        "credentials_persisted": False,
    })

    lines = [
        "# S17 Stage-1 frozen mechanism comparison", "",
        "No thresholds were scanned and no model was trained. This compares only pre-existing frozen RCQT mechanisms.", "",
        "## Mature T+10 candidate quality", "",
        "| Mechanism | Mean | PF | Win rate | <=-8% | >=+10% | Rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("quiet_confirmed", "reset_confirmed", "formal_rcqt_union"):
        item = event_summary["cohorts"][name]["overall"]
        lines.append(
            f"| {name} | {item['mean_return']:+.2%} | {item['profit_factor']:.3f} | "
            f"{item['win_rate']:.1%} | {item['down_8pct_rate']:.1%} | {item['up_10pct_rate']:.1%} | {item['rows']} |"
        )
    lines.extend(["", "## Executable portfolios", "",
                  "| Policy | Return | PF | MaxDD | Ex best week | Trades |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name in ("quiet_confirmed_top4", "reset_top4", "formal_rcqt_4plus2"):
        item = portfolio_summary[f"{name}_base"]
        lines.append(
            f"| {name} | {item['total_return']:+.2%} | {item['portfolio_profit_factor']:.3f} | "
            f"{item['max_drawdown']:.2%} | {item['return_excluding_best_week']:+.2%} | {item['trade_count']} |"
        )
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = {
        str(path.relative_to(output)): _sha(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    _write_json(output / "ARTIFACT_HASHES.json", artifacts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes-source", type=Path, required=True)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
