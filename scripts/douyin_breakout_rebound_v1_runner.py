"""Run a preregistered douyin-style breakout rebound event strategy.

This runner keeps the strategy fully transparent:
- T0: real close-limit event with recent quiet liquidity
- T1: next-session washout with volume expansion
- S: support is fixed from T0 open and T1 MA20 on the economic price series
- T2: first close that reclaims T0 close inside a 3-session window

The runner writes both the full event ledger and the executable signal ledger,
then sends the executable signals through the shared backtest engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.data.execution_source import load_execution_panel
from aistock9988.data.quantdb import readonly_connection
from aistock9988.data.universe import (
    STBlacklistManifest,
    build_universe_exclusion_ledger,
    filter_current_st_history,
)
from aistock9988.execution.prices import validate_execution_panel
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.time.session import session_close

FAMILY_ID = "douyin.breakout_rebound.v1"
POLICY_ID = "selection.douyin.breakout_rebound.v1"
DECISION_PREFIX = "douyin.breakout_rebound.v1"
MAX_POSITIONS = 4
HOLD_SESSIONS = 10
TARGET_WEIGHT = 0.12
BUY_SLIPPAGE = 0.001
STRESS_SLIPPAGE = 0.003
MAX_ORDER_TO_ADV20 = 0.02
LOOKBACK_DAYS = 140
WINDOW_AFTER_T1 = 3


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _query_universe_codes(start: str, end: str) -> list[str]:
    with readonly_connection() as conn:
        frame = pd.read_sql_query(
            "SELECT DISTINCT m.ts_code "
            "FROM market_daily_ts m "
            "WHERE m.source = 'daily' AND m.trade_date >= %s AND m.trade_date <= %s "
            "AND m.amount > 0 AND m.ts_code NOT LIKE %s "
            "ORDER BY m.ts_code",
            conn,
            params=(start, end, "%.BJ"),
        )
    codes = frame["ts_code"].astype(str).tolist()
    if not codes:
        raise RuntimeError("no universe codes found for requested range")
    return codes


def _query_current_st_codes() -> set[str]:
    with readonly_connection() as conn:
        frame = pd.read_sql_query(
            "SELECT ts_code, name FROM stock_basic_ts",
            conn,
        )
    mask = frame["name"].fillna("").str.upper().str.contains("ST|退市风险", regex=True)
    return set(frame.loc[mask, "ts_code"].astype(str).str.upper())


def _load_prices(start: str, end: str, codes: list[str]) -> pd.DataFrame:
    prices = load_execution_panel(start, end, ts_codes=codes)
    prices = validate_execution_panel(prices)
    prices = prices.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], utc=True).dt.normalize()
    prices["amount"] = pd.to_numeric(prices["amount"], errors="raise")
    prices = prices.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    return prices


def _close_limit_up(row: pd.Series) -> bool:
    return float(row["raw_close"]) >= float(row["up_limit"]) - 1e-8


def _close_limit_down(row: pd.Series) -> bool:
    return float(row["raw_close"]) <= float(row["down_limit"]) + 1e-8


def _build_event_ledger(panel: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    end_day = pd.Timestamp(end, tz="UTC").normalize()

    for ts_code, group in panel.groupby("ts_code", sort=True):
        g = group.sort_values("trade_date", kind="mergesort").reset_index(drop=True).copy()
        g["close_limit_up"] = g.apply(_close_limit_up, axis=1)
        g["close_limit_down"] = g.apply(_close_limit_down, axis=1)
        g["prev10_close_limit_up_count"] = g["close_limit_up"].shift(1).rolling(10, min_periods=10).sum()
        g["prev20_amount_median"] = g["amount"].shift(1).rolling(20, min_periods=20).median()
        g["ma20_economic"] = g["economic_close"].rolling(20, min_periods=20).mean()
        g["ma60_economic"] = g["economic_close"].rolling(60, min_periods=60).mean()
        g["dist_ma60_economic"] = g["economic_close"] / g["ma60_economic"] - 1.0

        for i, row in g.iterrows():
            trade_date = pd.Timestamp(row["trade_date"])
            if trade_date < pd.Timestamp(start, tz="UTC").normalize():
                continue
            if not bool(row["close_limit_up"]):
                continue

            record = {
                "family_id": FAMILY_ID,
                "ts_code": ts_code,
                "t0_trade_date": trade_date,
                "t1_trade_date": None,
                "t2_trade_date": None,
                "entry_trade_date": None,
                "status": "failed",
                "failure_reason": None,
                "available_time": session_close(trade_date),
                "t0_raw_open": float(row["raw_open"]),
                "t0_raw_close": float(row["raw_close"]),
                "t0_economic_open": float(row["economic_open"]),
                "t0_economic_close": float(row["economic_close"]),
                "t0_amount": float(row["amount"]),
                "t0_prev20_amount_median": float(row["prev20_amount_median"]) if pd.notna(row["prev20_amount_median"]) else None,
                "t0_amount_ratio": float(row["amount"] / row["prev20_amount_median"]) if pd.notna(row["prev20_amount_median"]) and float(row["prev20_amount_median"]) > 0 else None,
                "t0_prev10_close_limit_up_count": int(row["prev10_close_limit_up_count"]) if pd.notna(row["prev10_close_limit_up_count"]) else None,
                "t0_close_limit_up": True,
                "t0_close_to_ma60": float(row["dist_ma60_economic"]) if pd.notna(row["dist_ma60_economic"]) else None,
                "t1_raw_open": None,
                "t1_raw_close": None,
                "t1_raw_return": None,
                "t1_amount": None,
                "t1_amount_ratio": None,
                "t1_ma20_economic": None,
                "support_economic": None,
                "t2_raw_close": None,
                "t2_economic_close": None,
                "t2_amount": None,
                "t2_amount_ratio": None,
                "breakout_margin": None,
                "support_buffer": None,
                "event_score": None,
                "stop_loss_price": None,
            }

            if pd.isna(row["prev20_amount_median"]) or pd.isna(row["ma60_economic"]) or pd.isna(row["dist_ma60_economic"]):
                record["failure_reason"] = "insufficient_history"
                rows.append(record)
                continue
            if int(row["prev10_close_limit_up_count"]) != 0:
                record["failure_reason"] = "t0_preceded_by_limit_up"
                rows.append(record)
                continue
            if float(row["amount"]) > float(row["prev20_amount_median"]):
                record["failure_reason"] = "t0_amount_above_prev20_median"
                rows.append(record)
                continue
            if abs(float(row["dist_ma60_economic"])) > 0.15:
                record["failure_reason"] = "t0_too_far_from_ma60"
                rows.append(record)
                continue

            t1_idx = i + 1
            if t1_idx >= len(g):
                record["failure_reason"] = "missing_t1_session"
                rows.append(record)
                continue
            t1 = g.iloc[t1_idx]
            record["t1_trade_date"] = pd.Timestamp(t1["trade_date"])
            record["t1_raw_open"] = float(t1["raw_open"])
            record["t1_raw_close"] = float(t1["raw_close"])
            record["t1_raw_return"] = float(t1["raw_close"] / t1["raw_open"] - 1.0)
            record["t1_amount"] = float(t1["amount"])
            record["t1_amount_ratio"] = float(t1["amount"] / row["amount"]) if float(row["amount"]) > 0 else None
            record["t1_ma20_economic"] = float(t1["ma20_economic"]) if pd.notna(t1["ma20_economic"]) else None

            if pd.isna(t1["ma20_economic"]):
                record["failure_reason"] = "t1_missing_ma20"
                rows.append(record)
                continue
            if not (float(t1["raw_close"]) < float(t1["raw_open"]) and float(t1["raw_close"]) / float(t1["raw_open"]) - 1.0 <= -0.05):
                record["failure_reason"] = "t1_not_washout"
                rows.append(record)
                continue
            if float(t1["amount"]) < 1.5 * float(row["amount"]):
                record["failure_reason"] = "t1_volume_not_expanded"
                rows.append(record)
                continue
            if bool(t1["close_limit_down"]):
                record["failure_reason"] = "t1_locked_limit_down"
                rows.append(record)
                continue

            support = max(float(row["economic_open"]), float(t1["ma20_economic"]))
            if not np.isfinite(support) or support <= 0:
                record["failure_reason"] = "invalid_support"
                rows.append(record)
                continue
            record["support_economic"] = support

            below_streak = 0
            t2_idx = None
            failure_reason = "no_rebound_within_window"
            window_end = min(t1_idx + WINDOW_AFTER_T1 + 1, len(g))
            for j in range(t1_idx + 1, window_end):
                day = g.iloc[j]
                if float(day["economic_close"]) < support:
                    below_streak += 1
                else:
                    below_streak = 0
                if below_streak >= 2:
                    failure_reason = "support_broken"
                    break
                prev20 = day["prev20_amount_median"]
                if pd.isna(prev20) or float(prev20) <= 0:
                    continue
                breakout_margin = float(day["economic_close"] / row["economic_close"] - 1.0)
                if (
                    float(day["economic_close"]) >= float(row["economic_close"])
                    and float(day["amount"]) >= 1.2 * float(prev20)
                    and not bool(day["close_limit_down"])
                ):
                    t2_idx = j
                    record["t2_trade_date"] = pd.Timestamp(day["trade_date"])
                    record["t2_raw_close"] = float(day["raw_close"])
                    record["t2_economic_close"] = float(day["economic_close"])
                    record["t2_amount"] = float(day["amount"])
                    record["t2_amount_ratio"] = float(day["amount"] / prev20)
                    record["breakout_margin"] = breakout_margin
                    record["support_buffer"] = float(day["economic_close"] / support - 1.0)
                    break
            if t2_idx is None:
                record["failure_reason"] = failure_reason
                rows.append(record)
                continue

            entry_idx = t2_idx + 1
            if entry_idx >= len(g):
                record["status"] = "complete_non_executable"
                record["failure_reason"] = "missing_entry_session"
                record["stop_loss_price"] = support
                rows.append(record)
                continue

            entry_day = g.iloc[entry_idx]
            entry_trade_date = pd.Timestamp(entry_day["trade_date"])
            if entry_trade_date > end_day:
                record["status"] = "complete_non_executable"
                record["failure_reason"] = "entry_beyond_requested_end"
                record["stop_loss_price"] = support
                rows.append(record)
                continue

            record["status"] = "complete"
            record["failure_reason"] = None
            record["entry_trade_date"] = entry_trade_date
            record["stop_loss_price"] = support
            record["event_score"] = (
                float(record["breakout_margin"] or 0.0)
                + 0.20 * float(record["t2_amount_ratio"] or 0.0)
                + 0.15 * float(record["t1_amount_ratio"] or 0.0)
                - 0.10 * abs(float(record["support_buffer"] or 0.0))
                - 0.10 * abs(float(record["t0_close_to_ma60"] or 0.0))
                - 0.05 * float(record["t0_amount_ratio"] or 0.0)
            )
            rows.append(record)

    ledger = pd.DataFrame(rows)
    if ledger.empty:
        raise RuntimeError("no breakboard-rebound events were detected")
    ledger["available_time"] = pd.to_datetime(ledger["available_time"], utc=True)
    for column in ("t0_trade_date", "t1_trade_date", "t2_trade_date", "entry_trade_date"):
        ledger[column] = pd.to_datetime(ledger[column], utc=True)
    return ledger.sort_values(["t0_trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)


def _build_signal_ledger(event_ledger: pd.DataFrame) -> pd.DataFrame:
    signals = event_ledger[event_ledger["status"] == "complete"].copy()
    signals = signals[signals["entry_trade_date"].notna()].copy()
    if signals.empty:
        return signals
    signals = signals.sort_values(
        [
            "t2_trade_date",
            "event_score",
            "breakout_margin",
            "t2_amount_ratio",
            "t1_amount_ratio",
            "support_buffer",
            "t0_close_to_ma60",
            "ts_code",
        ],
        ascending=[True, False, False, False, False, False, True, True],
        kind="mergesort",
    ).copy()
    signals["candidate_rank"] = signals.groupby("t2_trade_date", sort=True).cumcount() + 1
    signals["asof"] = signals["t2_trade_date"]
    signals["selected"] = True
    signals["selection_decision_id"] = signals["asof"].dt.strftime(DECISION_PREFIX + "-%Y%m%d")
    signals["policy_id"] = POLICY_ID
    signals["target_weight"] = TARGET_WEIGHT
    signals["cash_fraction"] = 1.0
    signals["sleeve"] = "breakout_rebound"
    signals["context_hash"] = signals["asof"].map(
        lambda day: hashlib.sha256(
            json.dumps(
                {
                    "family_id": FAMILY_ID,
                    "asof": str(day.date()),
                    "t0_limit_up": "close_at_up_limit",
                    "t1_washout_drop": -0.05,
                    "t1_volume_multiple": 1.5,
                    "support_window_sessions": WINDOW_AFTER_T1,
                    "t2_volume_multiple": 1.2,
                    "ma60_distance_cap": 0.15,
                    "max_positions": MAX_POSITIONS,
                    "hold_sessions": HOLD_SESSIONS,
                    "target_weight": TARGET_WEIGHT,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    return signals.reset_index(drop=True)


def _summarize_events(event_ledger: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "family_id": FAMILY_ID,
        "event_count": int(len(event_ledger)),
        "complete_count": int((event_ledger["status"] == "complete").sum()),
        "complete_non_executable_count": int((event_ledger["status"] == "complete_non_executable").sum()),
        "failed_count": int((event_ledger["status"] == "failed").sum()),
        "selected_count": int((event_ledger["status"] == "complete").sum()),
        "failure_reasons": dict(Counter(event_ledger["failure_reason"].fillna("none"))),
    }
    complete = event_ledger[event_ledger["status"] == "complete"].copy()
    if not complete.empty:
        summary["median_breakout_margin"] = float(pd.to_numeric(complete["breakout_margin"], errors="coerce").median())
        summary["median_support_buffer"] = float(pd.to_numeric(complete["support_buffer"], errors="coerce").median())
        summary["median_t2_amount_ratio"] = float(pd.to_numeric(complete["t2_amount_ratio"], errors="coerce").median())
    return summary


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    codes = _query_universe_codes(args.start, args.end)
    current_st_codes = _query_current_st_codes()
    st_manifest = STBlacklistManifest.build(
        current_st_codes,
        source="stock_basic_ts.name",
        extracted_at=pd.Timestamp.utcnow().isoformat(),
    )

    raw_start = (pd.Timestamp(args.start) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    prices = _load_prices(raw_start, args.end, codes)
    prices = filter_current_st_history(prices, current_st_codes)
    prices = prices.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    if prices.empty:
        raise RuntimeError("execution panel is empty after current-ST filtering")
    prices["amount"] = pd.to_numeric(prices["amount"], errors="raise")
    prices["adv20"] = prices.groupby("ts_code")["amount"].transform(lambda s: s.rolling(20, min_periods=20).median())
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], utc=True).dt.normalize()

    event_ledger = _build_event_ledger(prices, start=args.start, end=args.end)
    signal_ledger = _build_signal_ledger(event_ledger)
    if signal_ledger.empty:
        raise RuntimeError("no executable breakout-rebound signals were found")

    # Keep the executable portfolio transparent and deterministic.
    backtest_signals = signal_ledger[
        [
            "asof",
            "ts_code",
            "candidate_rank",
            "selected",
            "selection_decision_id",
            "policy_id",
            "target_weight",
            "cash_fraction",
            "sleeve",
            "context_hash",
            "stop_loss_price",
            "t0_trade_date",
            "t1_trade_date",
            "t2_trade_date",
            "entry_trade_date",
            "t0_raw_open",
            "t0_raw_close",
            "t0_economic_open",
            "t0_economic_close",
            "t1_raw_open",
            "t1_raw_close",
            "t1_raw_return",
            "t1_amount_ratio",
            "support_economic",
            "t2_raw_close",
            "t2_economic_close",
            "t2_amount_ratio",
            "breakout_margin",
            "support_buffer",
            "event_score",
            "available_time",
        ]
    ].copy()
    backtest_signals["available_time"] = pd.to_datetime(backtest_signals["available_time"], utc=True)

    actions = load_corporate_actions(args.start, args.end, ts_codes=codes)
    actions.to_csv(output / "corporate_actions.csv", index=False)

    portfolios: dict[str, object] = {}
    for label, slippage in (("base", BUY_SLIPPAGE), ("stress", STRESS_SLIPPAGE)):
        result = run_backtest(
            backtest_signals,
            prices,
            corporate_actions=actions,
            config=BacktestConfig(
                initial_cash=1_000_000.0,
                max_positions=MAX_POSITIONS,
                hold_sessions=HOLD_SESSIONS,
                stop_loss_pct=None,
                stop_loss_mode="close_next_session_open",
                accounting_price_basis="raw",
                lot_size=100,
                max_order_to_adv20=MAX_ORDER_TO_ADV20,
                buy_slippage=slippage,
                sell_slippage=slippage,
            ),
        )
        metrics = summarize_backtest(result["nav"], result["trades"], initial_cash=1_000_000.0)
        metrics["slippage_each_side"] = slippage
        metrics["signal_count"] = int(len(backtest_signals))
        metrics["event_count"] = int(len(event_ledger))
        metrics["complete_event_count"] = int((event_ledger["status"] == "complete").sum())
        metrics["complete_non_executable_count"] = int((event_ledger["status"] == "complete_non_executable").sum())
        metrics["failure_reason_counts"] = dict(Counter(event_ledger["failure_reason"].fillna("none")))
        metrics["database_cutoff"] = str(prices["trade_date"].max().date())
        metrics["rule_version"] = FAMILY_ID
        metrics["max_positions"] = MAX_POSITIONS
        metrics["hold_sessions"] = HOLD_SESSIONS
        metrics["target_weight"] = TARGET_WEIGHT
        metrics["stop_loss_contract"] = "absolute_support_price"
        target = output / "backtests" / label
        target.mkdir(parents=True, exist_ok=True)
        for name in ("orders", "trades", "nav", "positions", "corporate_actions"):
            result[name].to_csv(target / f"{name}.csv", index=False)
        _write_json(target / "metrics.json", metrics)
        portfolios[label] = metrics

    event_ledger.to_parquet(output / "event_ledger.parquet", index=False)
    signal_ledger.to_csv(output / "selection_ledger.csv", index=False)
    signal_ledger.to_parquet(output / "signal_ledger.parquet", index=False)
    build_universe_exclusion_ledger(
        pd.DataFrame({"ts_code": sorted(current_st_codes)}),
        current_st_codes,
        asof=args.start,
    ).to_csv(output / "universe_exclusion_ledger.csv", index=False)

    event_summary = _summarize_events(event_ledger)
    summary = {
        "experiment_id": FAMILY_ID,
        "status": "completed_or_non_executable_signals_only",
        "requested_start": args.start,
        "requested_end": args.end,
        "raw_lookback_start": raw_start,
        "codes_count": len(codes),
        "current_st_manifest": st_manifest.to_dict(),
        "event_summary": event_summary,
        "portfolios": portfolios,
    }
    _write_json(output / "SUMMARY.json", summary)

    lines = [
        "# Douyin Breakout Rebound V1",
        "",
        f"- Requested range: {args.start} to {args.end}",
        f"- Database cutoff observed in panel: {str(prices['trade_date'].max().date())}",
        f"- Universe codes: {len(codes)}",
        f"- Current-ST exclusions: {len(current_st_codes)}",
        f"- Total T0 candidates: {event_summary['event_count']}",
        f"- Complete signals: {event_summary['complete_count']}",
        f"- Non-executable complete signals: {event_summary['complete_non_executable_count']}",
        f"- Failed candidates: {event_summary['failed_count']}",
        "",
        "## Portfolio",
        "",
        "| Scenario | Total Return | PF | MaxDD | Excluding Best Week | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("base", "stress"):
        item = portfolios[label]
        lines.append(
            f"| {label} | {item['total_return']:+.2%} | {item['portfolio_profit_factor']:.3f} | "
            f"{item['max_drawdown']:.2%} | {item['return_excluding_best_week']:+.2%} | {item['trade_count']} |"
        )
    lines.extend(
        [
            "",
            "## Failure Reasons",
            "",
        ]
    )
    for reason, count in sorted(event_summary["failure_reasons"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {reason}: {count}")
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- T0 = real close-limit event with quiet liquidity and recent no-limit-up history.",
            "- T1 = next-session washout with volume expansion and no locked limit-down.",
            "- Support = max(T0 economic open, T1 economic MA20).",
            "- T2 = first close in the next 3 sessions that reclaims T0 economic close with expanding turnover.",
            "- Stop = absolute support price on the economic series, executed at the next tradable session open.",
        ]
    )
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_json(
        output / "DATA_MANIFEST.json",
        {
            "experiment_id": FAMILY_ID,
            "start": args.start,
            "end": args.end,
            "raw_lookback_start": raw_start,
            "codes_count": len(codes),
            "codes_source": "quant_db.market_daily_ts distinct ts_code",
            "current_st_manifest": st_manifest.to_dict(),
            "data_cutoff": str(prices["trade_date"].max().date()),
            "event_ledger_sha256": _sha(output / "event_ledger.parquet"),
            "signal_ledger_sha256": _sha(output / "signal_ledger.parquet"),
            "selection_ledger_sha256": _sha(output / "selection_ledger.csv"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-08-21")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
