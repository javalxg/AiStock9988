"""One complete, auditable q70 training + execution experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.data.execution_source import load_execution_panel
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.data.minute_source import load_minute_execution_panel
from aistock9988.reporting.metrics import summarize_backtest

from first_q70_train import run_training


def _write_once(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete q70 experiment")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-23")
    parser.add_argument("--cutoff", default="2025-12-31T23:59:59Z")
    parser.add_argument("--asof", default="2026-01-09")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    result = run_training(run_dir=run_dir, start=args.start, end=args.end,
                          cutoff_value=args.cutoff, asof=args.asof)
    signals = pd.read_csv(result.candidate_path)
    codes = sorted(signals["ts_code"].astype(str).unique().tolist())
    prices = load_execution_panel(args.start, args.end, ts_codes=codes)
    actions = load_corporate_actions(args.start, args.end, ts_codes=codes)
    profile = yaml.safe_load((Path(__file__).parents[1] / "configs/execution_profiles/q70_v1.yaml").read_text())
    minute_prices = load_minute_execution_panel(args.asof, args.end,
                                                freq=str(profile["execution_bar_frequency"]), ts_codes=codes)
    backtest = run_backtest(
        signals, prices,
        config=BacktestConfig(
            max_positions=int(profile["max_positions"]),
            hold_sessions=int(profile["hold_buffer_n"]),
            stop_loss_pct=float(profile["stop_loss_pct"]),
            take_profit_pct=float(profile["take_profit_pct"]),
            stop_loss_mode=str(profile["stop_loss_mode"]),
        ), corporate_actions=actions, minute_prices=minute_prices,
    )
    _write_once(backtest["orders"], run_dir / "trades/orders.csv")
    _write_once(backtest["trades"], run_dir / "trades/fills.csv")
    _write_once(backtest["nav"], run_dir / "trades/nav.csv")
    _write_once(backtest["positions"], run_dir / "trades/positions.csv")
    _write_once(backtest["corporate_actions"], run_dir / "trades/corporate_actions.csv")
    metrics = summarize_backtest(backtest["nav"], backtest["trades"], initial_cash=1_000_000.0)
    diagnostics = {"metrics": metrics, "execution_panel": {"start": args.asof, "end": args.end,
                                                               "rows": len(prices),
                                                               "pit_rule": "available_time <= decision_time"},
                  "corporate_actions": {"rows": len(actions), "source": "read_only_dividend_table",
                                        "accounting_rule": "shares/cash ledger; never infer cash from adj_factor"}}
    (run_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics/metrics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n")
    print(run_dir)


if __name__ == "__main__":
    main()
