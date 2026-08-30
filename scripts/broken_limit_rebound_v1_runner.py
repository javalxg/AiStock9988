"""Preregistered broken-limit rebound Stage-1 diagnostic.

This runner encodes the team's joint quantized v1.2 contract:
T0 limit-up probe -> T1 washout -> support survives -> T2 reclaim.
Only auditable daily raw/economic prices and amount ratios are used.
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
from aistock9988.labeling.q70 import build_q70_t10_labels
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.time.session import session_close

from rcqt_stage1_quality_runner import (
    LABEL_PROFILE,
    _date_chunks,
    _load_pit_st_keys,
    _sha,
    _write_json,
)


def _load_codes(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=["ts_code"])
    elif path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path, usecols=["ts_code"])
    else:
        raise ValueError(f"unsupported codes source: {path.suffix}")
    codes = sorted(frame["ts_code"].astype(str).str.upper().unique().tolist())
    if not codes:
        raise ValueError("codes source is empty")
    return codes


def _load_sources(start: str, end: str, codes: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    parts: list[pd.DataFrame] = []
    chunks = _date_chunks(start, end)
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"source_chunk {index}/{len(chunks)} start={chunk_start} end={chunk_end}", flush=True)
        parts.append(load_execution_panel(chunk_start, chunk_end, ts_codes=codes))
    prices = pd.concat(parts, ignore_index=True).sort_values(
        ["trade_date", "ts_code"], kind="mergesort",
    ).reset_index(drop=True)
    if prices.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("execution source contains duplicate trade_date/ts_code keys")
    return prices, {
        "source_id": "quant_db.execution_daily",
        "chunks": chunks,
        "rows": int(len(prices)),
        "coverage_start": str(pd.to_datetime(prices["trade_date"], utc=True).min().date()),
        "coverage_end": str(pd.to_datetime(prices["trade_date"], utc=True).max().date()),
    }


def _build_daily_features(prices: pd.DataFrame) -> pd.DataFrame:
    ordered = prices.copy().sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    grouped = ordered.groupby("ts_code", group_keys=False, sort=False)
    ordered["close_limit_up"] = ordered["raw_close"] >= ordered["up_limit"]
    ordered["prior_10_limit_up_count"] = grouped["close_limit_up"].transform(
        lambda series: series.shift(1).rolling(10, min_periods=10).sum()
    )
    ordered["amount_median_20_prev"] = grouped["amount"].transform(
        lambda series: series.shift(1).rolling(20, min_periods=20).median()
    )
    ordered["ma60"] = grouped["economic_close"].transform(lambda series: series.rolling(60, min_periods=60).mean())
    ordered["dist_ma60"] = ordered["economic_close"] / ordered["ma60"] - 1.0
    ordered["t1_body_return"] = ordered["raw_close"] / ordered["raw_open"] - 1.0
    return ordered


def _event_score(reclaim_ratio: float, support_buffer: float, washout_ratio: float) -> float:
    return 0.50 * float(reclaim_ratio) + 0.30 * float(support_buffer) + 0.20 * float(washout_ratio)


def _utc_day(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True).normalize()


def _scan_events(prices: pd.DataFrame, *, start: str, end: str,
                 st_keys: set[tuple[pd.Timestamp, str]]) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    rows: list[dict[str, object]] = []
    for code, frame in prices.groupby("ts_code", sort=True):
        group = frame.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
        if len(group) < 5:
            continue
        for idx in range(len(group) - 2):
            t0 = group.iloc[idx]
            t1 = group.iloc[idx + 1]
            t0_limit_up = bool(t0["close_limit_up"])
            t0_fresh = pd.notna(t0["prior_10_limit_up_count"]) and float(t0["prior_10_limit_up_count"]) == 0.0
            t1_body_ok = float(t1["raw_close"]) < float(t1["raw_open"])
            t1_pullback = float(t1["raw_close"]) / float(t0["raw_close"]) - 1.0
            t1_pullback_ok = t1_pullback <= -0.05
            t1_amount_ratio = float(t1["amount"]) / float(t0["amount"]) if float(t0["amount"]) > 0 else None
            t1_amount_ok = t1_amount_ratio is not None and t1_amount_ratio >= 1.5
            t1_not_down_limit = float(t1["raw_close"]) > float(t1["down_limit"])
            amount_ratio_t0 = (
                float(t0["amount"]) / float(t0["amount_median_20_prev"])
                if pd.notna(t0["amount_median_20_prev"]) and float(t0["amount_median_20_prev"]) > 0
                else None
            )
            dist_ma60 = float(t0["dist_ma60"]) if pd.notna(t0["dist_ma60"]) else None
            support = max(float(t0["raw_open"]), float(t1["raw_low"]))
            candidate_status = "REJECTED"
            rejection_reason = ""
            t2 = None
            below_streak = 0
            if t0_limit_up and t0_fresh and t1_body_ok and t1_pullback_ok and t1_amount_ok and t1_not_down_limit:
                for probe in range(idx + 2, min(idx + 5, len(group))):
                    current = group.iloc[probe]
                    below_support = float(current["raw_close"]) < support
                    below_streak = below_streak + 1 if below_support else 0
                    if below_streak >= 2:
                        rejection_reason = "two_consecutive_closes_below_support"
                        break
                    amount_ratio_t2 = (
                        float(current["amount"]) / float(current["amount_median_20_prev"])
                        if pd.notna(current["amount_median_20_prev"]) and float(current["amount_median_20_prev"]) > 0
                        else None
                    )
                    if (
                        float(current["raw_close"]) >= float(t0["raw_close"])
                        and amount_ratio_t2 is not None
                        and amount_ratio_t2 >= 1.2
                    ):
                        t2 = current
                        break
                if t2 is None and not rejection_reason:
                    rejection_reason = "no_reclaim_within_3_sessions"
            else:
                failed = []
                if not t0_limit_up:
                    failed.append("t0_not_limit_up")
                if not t0_fresh:
                    failed.append("t0_recent_limit_up")
                if not t1_body_ok:
                    failed.append("t1_not_bearish")
                if not t1_pullback_ok:
                    failed.append("t1_pullback_lt_5pct")
                if not t1_amount_ok:
                    failed.append("t1_amount_lt_1p5x")
                if not t1_not_down_limit:
                    failed.append("t1_closed_at_down_limit")
                rejection_reason = ",".join(failed)
            asof = pd.NaT
            entry_date = pd.NaT
            reclaim_ratio = None
            support_buffer = None
            t2_amount_ratio = None
            t2_sessions_after_t1 = None
            selection_score = None
            pit_st = False
            t2_idx = None
            if t2 is not None:
                t2_idx = int(t2.name)
                asof = _utc_day(t2["trade_date"])
                next_idx = t2_idx + 1
                if next_idx < len(group):
                    entry_date = _utc_day(group.iloc[next_idx]["trade_date"])
                reclaim_ratio = float(t2["raw_close"]) / float(t0["raw_close"]) - 1.0
                support_buffer = float(t2["raw_close"]) / support - 1.0
                t2_amount_ratio = (
                    float(t2["amount"]) / float(t2["amount_median_20_prev"])
                    if pd.notna(t2["amount_median_20_prev"]) and float(t2["amount_median_20_prev"]) > 0
                    else None
                )
                t2_sessions_after_t1 = int(t2_idx - (idx + 1))
                selection_score = _event_score(
                    reclaim_ratio=reclaim_ratio,
                    support_buffer=support_buffer,
                    washout_ratio=abs(t1_pullback),
                )
                pit_st = (asof, str(code)) in st_keys
                candidate_status = "REJECTED" if pit_st else "CANDIDATE"
                rejection_reason = "pit_st" if pit_st else ""
            seed_date = _utc_day(t0["trade_date"])
            if seed_date > end_ts:
                continue
            if pd.isna(asof) and seed_date < start_ts:
                continue
            if pd.notna(asof) and (asof < start_ts or asof > end_ts):
                continue
            rows.append({
                "event_id": f"broken_limit_rebound_v1-{code}-{seed_date.strftime('%Y%m%d')}",
                "ts_code": str(code),
                "seed_date": seed_date,
                "t0_date": seed_date,
                "t1_date": _utc_day(t1["trade_date"]),
                "asof": asof,
                "entry_date": entry_date,
                "candidate_status": candidate_status,
                "rejection_reason": rejection_reason,
                "pit_st": bool(pit_st),
                "t0_amount_vs_prior20_median": amount_ratio_t0,
                "t0_dist_ma60": dist_ma60,
                "t1_body_return": float(t1["t1_body_return"]),
                "t1_pullback_vs_t0_close": t1_pullback,
                "t1_amount_vs_t0": t1_amount_ratio,
                "support_level": support,
                "t2_sessions_after_t1": t2_sessions_after_t1,
                "reclaim_vs_t0_close": reclaim_ratio,
                "support_buffer": support_buffer,
                "t2_amount_vs_prior20_median": t2_amount_ratio,
                "selection_score": selection_score,
            })
    columns = [
        "event_id", "ts_code", "seed_date", "t0_date", "t1_date", "asof", "entry_date",
        "candidate_status", "rejection_reason", "pit_st", "t0_amount_vs_prior20_median",
        "t0_dist_ma60", "t1_body_return", "t1_pullback_vs_t0_close", "t1_amount_vs_t0",
        "support_level", "t2_sessions_after_t1", "reclaim_vs_t0_close", "support_buffer",
        "t2_amount_vs_prior20_median", "selection_score",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    date_columns = ["seed_date", "t0_date", "t1_date", "asof", "entry_date"]
    for column in date_columns:
        out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    return out.sort_values(["seed_date", "ts_code"], kind="mergesort").reset_index(drop=True)


def _select_top4(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        columns = list(candidates.columns) + [
            "candidate_rank", "selected", "selection_decision_id", "policy_id", "target_weight", "context_hash",
        ]
        return pd.DataFrame(columns=columns)
    out = candidates.sort_values(
        ["asof", "selection_score", "ts_code"], ascending=[True, False, True], kind="mergesort",
    ).groupby("asof", sort=True).head(4).copy()
    out["candidate_rank"] = out.groupby("asof").cumcount() + 1
    out["selected"] = True
    out["selection_decision_id"] = "broken_limit_rebound_v1-" + out["asof"].dt.strftime("%Y%m%d")
    out["policy_id"] = "stage1.broken_limit_rebound.top4.v1"
    out["target_weight"] = 0.12
    out["context_hash"] = out["asof"].map(
        lambda day: hashlib.sha256(f"broken_limit_rebound_v1|{day}".encode()).hexdigest()
    )
    return out


def _label_panel(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    panel = prices[["ts_code", "trade_date", "economic_open"]].rename(columns={"trade_date": "event_time"}).copy()
    panel["event_time"] = pd.to_datetime(panel["event_time"], utc=True).dt.normalize()
    sessions = pd.DatetimeIndex(sorted(panel["event_time"].drop_duplicates()))
    return panel, sessions


def _mature_event_labels(candidates: pd.DataFrame, prices: pd.DataFrame, end: str) -> pd.DataFrame:
    if candidates.empty:
        out = candidates.copy()
        for column in ("label_return", "label_available_time", "exit_time"):
            out[column] = pd.Series(dtype="float64" if column == "label_return" else "datetime64[ns, UTC]")
        return out
    panel, sessions = _label_panel(prices)
    labels = build_q70_t10_labels(panel, profile=LABEL_PROFILE, session_dates=sessions)
    labels = labels.rename(columns={"event_time": "asof", "available_time": "label_available_time"})
    labels["asof"] = pd.to_datetime(labels["asof"], utc=True).dt.normalize()
    labels["label_available_time"] = pd.to_datetime(labels["label_available_time"], utc=True)
    mature = labels[labels["label_available_time"] <= session_close(pd.Timestamp(end, tz="UTC"))][
        ["asof", "ts_code", "label_return", "label_available_time", "exit_time"]
    ]
    return candidates.merge(mature, on=["asof", "ts_code"], how="left", validate="one_to_one")


def _win_loss_covariates(mature_events: pd.DataFrame) -> pd.DataFrame:
    features = [
        "t0_amount_vs_prior20_median",
        "t0_dist_ma60",
        "t1_body_return",
        "t1_pullback_vs_t0_close",
        "t1_amount_vs_t0",
        "t2_sessions_after_t1",
        "reclaim_vs_t0_close",
        "support_buffer",
        "t2_amount_vs_prior20_median",
        "selection_score",
    ]
    usable = mature_events.dropna(subset=["label_return"]).copy()
    if usable.empty:
        return pd.DataFrame(columns=["feature", "winner_n", "loser_n", "winner_median", "loser_median", "winner_minus_loser"])
    usable["win_flag"] = usable["label_return"] > 0
    winners = usable[usable["win_flag"]]
    losers = usable[~usable["win_flag"]]
    rows: list[dict[str, object]] = []
    for feature in features:
        rows.append({
            "feature": feature,
            "winner_n": int(len(winners)),
            "loser_n": int(len(losers)),
            "winner_median": float(winners[feature].median()) if len(winners) else None,
            "loser_median": float(losers[feature].median()) if len(losers) else None,
            "winner_minus_loser": (
                float(winners[feature].median()) - float(losers[feature].median())
                if len(winners) and len(losers) else None
            ),
        })
    return pd.DataFrame(rows)


def _summarize_events(events: pd.DataFrame, candidates: pd.DataFrame, selected: pd.DataFrame) -> dict[str, object]:
    status_counts = events["candidate_status"].value_counts(dropna=False).to_dict() if not events.empty else {}
    rejection_counts = events["rejection_reason"].fillna("").value_counts().to_dict() if not events.empty else {}
    by_year: dict[str, object] = {}
    if not candidates.empty:
        for year, frame in candidates.groupby(candidates["asof"].dt.year):
            by_year[str(year)] = {
                "candidate_rows": int(len(frame)),
                "selected_rows": int(len(selected[selected["asof"].dt.year == year])),
                "median_selection_score": float(frame["selection_score"].median()),
                "median_t1_pullback": float(frame["t1_pullback_vs_t0_close"].median()),
            }
    return {
        "event_rows": int(len(events)),
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(selected)),
        "status_counts": status_counts,
        "rejection_counts": rejection_counts,
        "by_year": by_year,
    }


def _backtest(signals: pd.DataFrame, prices: pd.DataFrame, actions: pd.DataFrame,
              *, slippage: float) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
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
            max_order_to_adv20=None,
            buy_slippage=slippage,
            sell_slippage=slippage,
        ),
    )
    metrics = summarize_backtest(result["nav"], result["trades"], initial_cash=1_000_000)
    orders = result["orders"] if "orders" in result else pd.DataFrame()
    trades = result["trades"] if "trades" in result else pd.DataFrame()
    metrics["slippage_each_side"] = slippage
    metrics["forced_final_liquidation_count"] = int(
        (trades.get("reason", pd.Series(dtype=str)) == "end_of_test_liquidation").sum()
    )
    metrics["entry_unfilled_count"] = int(
        ((orders.get("side", pd.Series(dtype=str)) == "BUY") & (orders.get("status", pd.Series(dtype=str)) != "FILLED")).sum()
    ) if not orders.empty else 0
    metrics["stop_loss_count"] = int((trades.get("trigger_type", pd.Series(dtype=str)) == "STOP_LOSS").sum())
    return result, metrics


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    codes = _load_codes(args.codes_source)
    raw_start = (pd.Timestamp(args.start) - pd.Timedelta(days=140)).strftime("%Y-%m-%d")
    st_keys, st_audit = _load_pit_st_keys(codes, args.start, args.end)
    prices, source_audit = _load_sources(raw_start, args.end, codes)
    coverage_end = pd.to_datetime(prices["trade_date"], utc=True).max()
    if coverage_end < pd.Timestamp(args.end, tz="UTC"):
        raise RuntimeError(f"price coverage stops at {coverage_end}, before requested end {args.end}")
    prices = _build_daily_features(prices)
    events = _scan_events(prices, start=args.start, end=args.end, st_keys=st_keys)
    candidates = events[events["candidate_status"] == "CANDIDATE"].copy()
    selected = _select_top4(candidates)
    mature_candidates = _mature_event_labels(candidates, prices, args.end)
    win_loss = _win_loss_covariates(mature_candidates)

    bt_prices = prices.copy().sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)
    actions = load_corporate_actions(args.start, args.end, ts_codes=codes)
    portfolios: dict[str, object] = {}
    for cost, slippage in (("base", 0.001), ("stress", 0.003)):
        result, metrics = _backtest(selected, bt_prices, actions, slippage=slippage)
        target = output / "backtests" / cost
        target.mkdir(parents=True, exist_ok=True)
        for artifact in ("orders", "trades", "nav", "positions", "corporate_actions"):
            result[artifact].to_csv(target / f"{artifact}.csv", index=False)
        _write_json(target / "metrics.json", metrics)
        portfolios[cost] = metrics

    _write_json(output / "SUMMARY.json", {
        "experiment_id": "broken_limit_rebound_v1",
        "status": "historical_diagnostic_not_lockbox",
        "events": _summarize_events(events, candidates, selected),
        "portfolios": portfolios,
    })
    events.to_parquet(output / "event_ledger.parquet", index=False)
    candidates.to_parquet(output / "candidate_ledger.parquet", index=False)
    selected.to_csv(output / "selection_ledger.csv", index=False)
    mature_candidates.to_parquet(output / "mature_event_ledger.parquet", index=False)
    win_loss.to_csv(output / "win_loss_covariates.csv", index=False)
    _write_json(output / "DATA_MANIFEST.json", {
        "experiment_id": "broken_limit_rebound_v1",
        "config": str(args.config.resolve()),
        "config_sha256": _sha(args.config),
        "codes_source": str(args.codes_source.resolve()),
        "codes_source_sha256": _sha(args.codes_source),
        "raw_start": raw_start,
        "source_end": args.end,
        "pit_st_audit": st_audit,
        "source_audit": source_audit,
        "model_training": False,
        "parameter_sweep": False,
        "historical_results_are_diagnostic": True,
        "selection_policy": "top4_by_joint_reclaim_score",
        "failed_entry_policy": "record_unfilled_do_not_retry",
    })
    summary = _summarize_events(events, candidates, selected)
    lines = [
        "# Broken Limit Rebound V1",
        "",
        "Historical diagnostic only.",
        "",
        f"- Event rows: {summary['event_rows']}",
        f"- Candidate rows: {summary['candidate_rows']}",
        f"- Selected rows: {summary['selected_rows']}",
    ]
    for label, metrics in portfolios.items():
        lines.append(
            f"- {label}: return={metrics['total_return']:+.2%}, PF={metrics['portfolio_profit_factor']}, "
            f"MaxDD={metrics['max_drawdown']:+.2%}, trades={metrics['trade_count']}, unfilled={metrics['entry_unfilled_count']}"
        )
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--codes-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-08-21")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
