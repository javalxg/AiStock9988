"""Replay Stage-1 with a pre-registered same-day execution fallback.

The candidate ranking is frozen at signal day T. At T+1 open, a candidate
without an executable row (or with a non-tradable open) is skipped and the
next-ranked candidate from the same T-day pool is tried. No post-T+1 return
or future feature is used to choose a replacement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.data.execution_source import load_execution_panel
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.time.session import session_open


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _date_chunks(start: str, end: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    cursor = pd.Timestamp(start).normalize()
    terminal = pd.Timestamp(end).normalize()
    while cursor <= terminal:
        chunk_end = min(cursor + pd.DateOffset(months=3) - pd.Timedelta(days=1), terminal)
        chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def _load_prices(start: str, end: str, codes: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for index, (chunk_start, chunk_end) in enumerate(_date_chunks(start, end), start=1):
        print(f"execution_chunk {index} start={chunk_start} end={chunk_end}", flush=True)
        parts.append(load_execution_panel(chunk_start, chunk_end, ts_codes=codes))
    prices = pd.concat(parts, ignore_index=True).sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)
    if prices.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("execution source contains duplicate trade_date/ts_code")
    return prices


def _tradable_at_open(row: pd.Series, day: pd.Timestamp) -> bool:
    if row is None:
        return False
    if not bool(row.get("open_available_time", session_open(day)) <= session_open(day)):
        return False
    if bool(row.get("is_suspended", False)) or bool(row.get("is_limit_up", False)):
        return False
    return float(row.get("raw_open", 0.0)) > 0.0


def _build_fallback_signals(candidates: pd.DataFrame, original: pd.DataFrame,
                            prices: pd.DataFrame, mature_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = prices.copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"], utc=True).dt.normalize()
    px_index = px.set_index(["trade_date", "ts_code"])
    sessions = pd.DatetimeIndex(sorted(px["trade_date"].drop_duplicates()))
    rows: list[pd.Series] = []
    audit: list[dict[str, object]] = []
    pool = candidates.copy()
    pool["asof"] = pd.to_datetime(pool["asof"], utc=True).dt.normalize()
    pool = pool[pool["asof"] <= mature_end].copy()
    pool["ts_code"] = pool["ts_code"].astype(str)
    for day, group in pool.groupby("asof", sort=True):
        future_sessions = sessions[sessions > day]
        next_day = future_sessions[0] if len(future_sessions) else pd.NaT
        ranked = group.sort_values(["quiet_score", "ts_code"], ascending=[False, True], kind="mergesort").copy()
        ranked["source_rank"] = range(1, len(ranked) + 1)
        selected = 0
        original_codes = set(original.loc[original["asof"] == day, "ts_code"].astype(str))
        if pd.notna(next_day):
            for _, candidate in ranked.iterrows():
                code = str(candidate["ts_code"])
                key = (next_day, code)
                tradable = key in px_index.index and _tradable_at_open(px_index.loc[key], next_day)
                audit.append({
                    "asof": day,
                    "ts_code": code,
                    "source_rank": int(candidate["source_rank"]),
                    "next_session": next_day,
                    "tradable_at_next_open": bool(tradable),
                    "selected": bool(tradable and selected < 4),
                    "replacement": bool(tradable and selected < 4 and code not in original_codes),
                    "reason": "selected" if tradable and selected < 4 else "no_execution_row" if key not in px_index.index else "not_tradable_at_open" if not tradable else "four_slots_filled",
                })
                if tradable and selected < 4:
                    row = candidate.copy()
                    row["candidate_rank"] = selected + 1
                    row["selected"] = True
                    row["selection_decision_id"] = "rcqt-stage1-fallback-" + day.strftime("%Y%m%d")
                    row["policy_id"] = "rcqt.stage1.quiet_confirmed.fallback_top4.v1"
                    row["target_weight"] = 0.12
                    row["fallback_source_rank"] = int(candidate["source_rank"])
                    row["fallback_reason"] = "original_top4" if code in original_codes else "untradable_original_replaced"
                    rows.append(row)
                    selected += 1
        if selected < 4:
            audit.append({
                "asof": day,
                "ts_code": None,
                "source_rank": None,
                "next_session": next_day,
                "tradable_at_next_open": None,
                "selected": False,
                "replacement": False,
                "reason": "unused_slot_no_tradable_candidate",
            })
    signals = pd.DataFrame(rows)
    if signals.empty:
        raise RuntimeError("fallback signal ledger is empty")
    audit_frame = pd.DataFrame(audit)
    return signals, audit_frame


def _backtest(signals: pd.DataFrame, prices: pd.DataFrame, actions: pd.DataFrame, slippage: float) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    result = run_backtest(
        signals,
        prices,
        corporate_actions=actions,
        config=BacktestConfig(
            max_positions=4,
            hold_sessions=10,
            stop_loss_pct=-0.08,
            stop_loss_mode="close_next_session_open",
            accounting_price_basis="raw",
            lot_size=100,
            max_order_to_adv20=0.02,
            buy_slippage=slippage,
            sell_slippage=slippage,
        ),
    )
    metrics = summarize_backtest(result["nav"], result["trades"], initial_cash=1_000_000)
    metrics["slippage_each_side"] = slippage
    metrics["forced_final_liquidation_count"] = int((result["trades"].get("reason", pd.Series(dtype=str)) == "end_of_test_liquidation").sum())
    return result, metrics


def run(args: argparse.Namespace) -> None:
    source = args.source_run.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(source / "candidate_ledger.parquet")
    original = pd.read_csv(source / "rule_top4_selection_ledger.csv")
    candidates["asof"] = pd.to_datetime(candidates["asof"], utc=True).dt.normalize()
    original["asof"] = pd.to_datetime(original["asof"], utc=True).dt.normalize()
    codes = sorted(candidates["ts_code"].astype(str).unique())
    source_manifest = json.loads((source / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    fixed_universe_codes = int(source_manifest["fixed_universe_codes"])
    mature_end = pd.Timestamp(json.loads((source / "EVENT_SUMMARY.json").read_text(encoding="utf-8"))["mature_through_signal_date"], tz="UTC")
    raw_start = (pd.Timestamp(args.start) - pd.Timedelta(days=140)).strftime("%Y-%m-%d")
    prices = _load_prices(raw_start, args.end, codes)
    signals, audit = _build_fallback_signals(candidates, original, prices, mature_end)
    px = prices.copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"], utc=True).dt.normalize()
    px["amount"] = pd.to_numeric(px["amount"], errors="raise")
    px = px.sort_values(["ts_code", "trade_date"], kind="mergesort")
    px["adv20"] = px.groupby("ts_code")["amount"].transform(lambda series: series.rolling(20, min_periods=20).median())
    px = px[px["trade_date"].between(pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC"))].copy()
    actions = load_corporate_actions(args.start, args.end, ts_codes=codes)
    portfolio: dict[str, object] = {}
    for label, slippage in (("base", 0.001), ("stress", 0.003)):
        result, metrics = _backtest(signals, px, actions, slippage)
        target = output / "backtests" / label
        target.mkdir(parents=True, exist_ok=True)
        for name in ("orders", "trades", "nav", "positions", "corporate_actions"):
            result[name].to_csv(target / f"{name}.csv", index=False)
        _write_json(target / "metrics.json", metrics)
        portfolio[label] = metrics
    signals.to_csv(output / "fallback_selection_ledger.csv", index=False)
    audit.to_csv(output / "fallback_decision_ledger.csv", index=False)
    _write_json(output / "PORTFOLIO_SUMMARY.json", portfolio)
    replacement_count = int(audit["replacement"].fillna(False).sum())
    selected_count = int(audit["selected"].fillna(False).sum())
    unavailable_original = int(((audit["source_rank"] <= 4) & (~audit["tradable_at_next_open"].fillna(False))).sum())
    manifest = {
        "kind": "stage1_fixed_rule_same_day_fallback_replay",
        "requested_start": args.start,
        "requested_end": args.end,
        "mature_signal_end": str(mature_end),
        "source_run": str(source),
        "source_event_summary_sha256": _sha(source / "EVENT_SUMMARY.json"),
        "fixed_universe_codes": fixed_universe_codes,
        "candidate_pool_codes": len(codes),
        "candidate_contract": "source S36 quiet_eligible AND right_confirmed AND NOT PIT-ST",
        "fallback_contract": "rank candidates by T-day quiet_score; at T+1 open skip missing/non-tradable rows and take next-ranked same-day candidates; max 4",
        "execution_contract": "T close signal; T+1 open; H10; -8% close-trigger/next-open stop",
        "selection_uses_future_returns": False,
        "selected_rows": selected_count,
        "replacement_rows": replacement_count,
        "untradable_original_rows": unavailable_original,
        "parameter_sweep": False,
        "model_training": False,
        "codes_source": str(args.codes_source.resolve()) if args.codes_source else None,
    }
    _write_json(output / "DATA_MANIFEST.json", manifest)
    base = portfolio["base"]
    (output / "RESULT.md").write_text(
        "# Stage-1 same-day fallback replay\n\n"
        "This is a separate execution-policy variant of S36. Candidate scores are fixed at T; only T+1 open tradability determines whether the next-ranked candidate fills a vacant slot.\n\n"
        f"- Mature signals through `{mature_end.date()}`; selected rows `{selected_count}`; replacements `{replacement_count}`.\n"
        f"- Base return `{base['total_return']:+.2%}`, PF `{base['portfolio_profit_factor']:.3f}`, MaxDD `{base['max_drawdown']:.2%}`, excluding best week `{base['return_excluding_best_week']:+.2%}`.\n"
        "- This result must be compared with S36 strict no-replacement baseline; it is not a threshold sweep.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--codes-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-08-21")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
