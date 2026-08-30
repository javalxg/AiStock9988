"""Run a deterministic RCQT end-to-end smoke backtest on a supplied CSV panel.

This is intentionally data-source agnostic: production runners bind frozen
quant_db snapshots, while this command proves the selection/execution contract.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import json
import hashlib
from aistock9988.data.universe import STBlacklistManifest, filter_current_st_history, build_universe_exclusion_ledger

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.selection.rcqt import score_rcqt, select_rcqt
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.time.session import session_close


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--st-codes", type=Path, help="newline-delimited current ST codes")
    parser.add_argument("--weekly", action="store_true", help="keep only last as-of per ISO week")
    parser.add_argument("--no-right-confirm", action="store_true")
    parser.add_argument("--reset-slots", type=int, default=4)
    parser.add_argument("--quiet-slots", type=int, default=2)
    parser.add_argument("--hold-sessions", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--single-weight-cap", type=float, default=0.15)
    parser.add_argument("--sector-weight-cap", type=float, default=0.30)
    parser.add_argument("--equity-cap", type=float, default=0.72)
    parser.add_argument("--stop-loss-pct", type=float, default=-0.08)
    parser.add_argument("--no-trailing", action="store_true")
    parser.add_argument("--max-order-to-adv20", type=float)
    parser.add_argument("--slippage", type=float, default=0.0)
    args = parser.parse_args()
    raw_features = pd.read_csv(args.features)
    st_codes = set()
    if args.st_codes:
        st_codes = {x.strip().upper() for x in args.st_codes.read_text().splitlines() if x.strip()}
        raw_features = filter_current_st_history(raw_features, st_codes)
    if "available_time" not in raw_features.columns:
        raise ValueError("features must contain available_time for strict PIT validation")
    if "available_time" in raw_features.columns:
        raw_features["asof"] = pd.to_datetime(raw_features["asof"], utc=True)
        raw_features["available_time"] = pd.to_datetime(raw_features["available_time"], utc=True)
        decision_close = raw_features["asof"].map(session_close)
        if (raw_features["available_time"] > decision_close).any():
            raise ValueError("features contain values unavailable by asof decision time")
    if args.weekly:
        # A weekly signal is one common last trading/as-of date for the whole
        # cross-section; taking a different last row per code creates a
        # look-ahead/misaligned portfolio.
        asof_dt = pd.to_datetime(raw_features["asof"], utc=True)
        week = asof_dt.dt.to_period("W-SUN")
        last_dates = asof_dt.groupby(week).transform("max")
        raw_features = raw_features.loc[asof_dt.eq(last_dates)].copy()
    features = score_rcqt(raw_features, require_right_confirmation=not args.no_right_confirm)
    # The audit contract requires one canonical frozen score while preserving
    # the two transparent sleeve scores used by the rule selector.
    features["score"] = features[["reset_score", "quiet_score"]].max(axis=1)
    selected = select_rcqt(
        features,
        reset_slots=args.reset_slots,
        quiet_slots=args.quiet_slots,
        single_weight_cap=args.single_weight_cap,
        sector_weight_cap=args.sector_weight_cap,
        equity_cap=args.equity_cap,
    )
    if selected.empty:
        raise ValueError("RCQT produced no selections; refusing to emit a green run")
    signals = selected.assign(
        asof=selected["asof"] if "asof" in selected else pd.Timestamp(features["asof"].iloc[0]),
        selected=True,
    )[["asof", "ts_code", "candidate_rank", "selected", "selection_decision_id", "policy_id", "target_weight", "context_hash", "sleeve"]]
    prices = pd.read_csv(args.prices)
    if args.max_order_to_adv20 is not None and "adv20" not in prices.columns:
        raise ValueError("capacity-controlled formal runs require an explicit PIT adv20 column")
    result = run_backtest(
        signals,
        prices,
        config=BacktestConfig(
            max_positions=args.max_positions,
            hold_sessions=args.hold_sessions,
            lot_size=100,
            stop_loss_pct=args.stop_loss_pct,
            trailing_arm_pct=None if args.no_trailing else 0.10,
            trailing_drawdown_pct=None if args.no_trailing else 0.08,
            max_order_to_adv20=args.max_order_to_adv20,
            buy_slippage=args.slippage,
            sell_slippage=args.slippage,
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
    metrics = summarize_backtest(result["nav"], result["trades"], initial_cash=BacktestConfig().initial_cash)
    metrics["forced_final_liquidation_count"] = int((result["trades"].get("reason", pd.Series(dtype=str)) == "end_of_test_liquidation").sum())
    weeks = result["nav"].assign(_week=pd.to_datetime(result["nav"]["trade_date"], utc=True).dt.to_period("W-SUN")).groupby("_week")["nav"].last()
    wr = weeks.pct_change(); wr.iloc[0] = weeks.iloc[0] / BacktestConfig().initial_cash - 1.0
    metrics["weekly_target"] = 0.05
    metrics["weekly_target_hit_ratio"] = float((wr >= 0.05).mean()) if len(wr) else None
    metrics["max_weekly_return"] = float(wr.max()) if len(wr) else None
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    features.to_csv(args.output / "score_ledger.csv", index=False)
    selected.to_csv(args.output / "selection_ledger.csv", index=False)
    selected.assign(candidate_status=lambda d: d["selected"].map({True: "SELECTED", False: "REJECTED_BY_POLICY"})).to_csv(args.output / "candidate_ledger.csv", index=False)
    if st_codes:
        ledger = build_universe_exclusion_ledger(pd.read_csv(args.features), st_codes, asof=str(features["asof"].min()))
        ledger.to_csv(args.output / "universe_exclusion_ledger.csv", index=False)
        st_manifest = STBlacklistManifest.build(st_codes, source=str(args.st_codes.resolve()), extracted_at=pd.Timestamp.utcnow().isoformat())
    else:
        st_manifest = STBlacklistManifest.build(set(), source="none", extracted_at=pd.Timestamp.utcnow().isoformat())
    (args.output / "data_manifest.json").write_text(json.dumps({
        "strategy_type": "rules", "policy_id": "selection.rcqt.v1",
        "features": str(args.features.resolve()), "prices": str(args.prices.resolve()),
        "feature_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "price_sha256": hashlib.sha256(args.prices.read_bytes()).hexdigest(),
        "st_policy": "current_st_codes_filtered_on_full_history",
        "st_blacklist": st_manifest.to_dict(),
        "weekly_signal": bool(args.weekly),
        "selection_contract": {
            "require_right_confirmation": not args.no_right_confirm,
            "reset_slots": args.reset_slots,
            "quiet_slots": args.quiet_slots,
            "single_weight_cap": args.single_weight_cap,
            "sector_weight_cap": args.sector_weight_cap,
            "equity_cap": args.equity_cap,
        },
        "execution_contract": {
            "hold_sessions": args.hold_sessions,
            "max_positions": args.max_positions,
            "stop_loss_pct": args.stop_loss_pct,
            "trailing": not args.no_trailing,
            "max_order_to_adv20": args.max_order_to_adv20,
            "slippage_each_side": args.slippage,
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "RESULT.md").write_text(
        f"# RCQT smoke result\n\ntrades: {len(result['trades'])}\nnav_rows: {len(result['nav'])}\n"
        f"total_return: {metrics.get('total_return')}\nmax_drawdown: {metrics.get('max_drawdown')}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
