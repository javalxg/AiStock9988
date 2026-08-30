"""One-shot causal recent-follow-through Stage-1 diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.selection.rcqt import score_rcqt
from aistock9988.time.session import session_close

from rcqt_quantdb_sample_runner import _features
from rcqt_stage1_quality_runner import _backtest, _load_pit_st_keys, _sha, _write_json
from relative_acceleration_confirmation_v2_runner import _add_relative_acceleration, _diagnostic_summary
from relative_orderly_continuation_runner import _load_sources


def _pool(scored: pd.DataFrame) -> pd.DataFrame:
    return scored[scored["quiet_eligible"] & scored["right_confirmed"] & scored["ret5"].gt(0)].copy()


def _top4(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.sort_values(["asof", "quiet_score", "ts_code"], ascending=[True, False, True], kind="mergesort").groupby("asof", sort=True).head(4).copy()
    out["candidate_rank"] = out.groupby("asof").cumcount() + 1
    out["selected"] = True
    out["selection_decision_id"] = "rcqt.stage1.recent_followthrough.top4.v1-" + out["asof"].dt.strftime("%Y%m%d")
    out["policy_id"] = "rcqt.stage1.recent_followthrough.top4.v1"
    out["target_weight"] = 0.12
    out["context_hash"] = out["asof"].map(lambda day: hashlib.sha256(f"recent_followthrough_v1|{day}".encode()).hexdigest())
    return out


def _labels(panel: pd.DataFrame, end: str) -> pd.DataFrame:
    from aistock9988.labeling.q70 import build_q70_t10_labels
    from rcqt_stage1_quality_runner import LABEL_PROFILE
    sessions = pd.DatetimeIndex(sorted(panel["event_time"].drop_duplicates()))
    labels = build_q70_t10_labels(panel, profile=LABEL_PROFILE, session_dates=sessions).rename(columns={"event_time": "asof", "available_time": "label_available_time"})
    labels["asof"] = pd.to_datetime(labels["asof"], utc=True).dt.normalize(); labels["label_available_time"] = pd.to_datetime(labels["label_available_time"], utc=True)
    return labels[labels["label_available_time"] <= session_close(pd.Timestamp(end, tz="UTC"))][["asof", "ts_code", "label_return", "label_available_time", "exit_time"]]


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    codes = sorted(pd.read_parquet(args.codes_source, columns=["ts_code"])["ts_code"].astype(str).unique())
    raw_start = (pd.Timestamp(args.start) - pd.Timedelta(days=140)).strftime("%Y-%m-%d")
    st_keys, st_audit = _load_pit_st_keys(codes, args.start, args.end)
    panel, prices, source_audit = _load_sources(raw_start, args.end, codes)
    features = _features(panel, prices, raw_start)
    features["asof"] = pd.to_datetime(features["asof"], utc=True).dt.normalize(); features["available_time"] = pd.to_datetime(features["available_time"], utc=True)
    features = _add_relative_acceleration(features, panel); features = features[features["asof"] >= pd.Timestamp(args.start, tz="UTC")].copy()
    if features.empty or features["asof"].max() < pd.Timestamp(args.end, tz="UTC"): raise RuntimeError("feature coverage does not reach requested end")
    if (features["available_time"] > features["asof"].map(session_close)).any(): raise AssertionError("feature PIT violation")
    scored = score_rcqt(features); scored["pit_st"] = [(d, c) in st_keys for d, c in zip(scored["asof"], scored["ts_code"].astype(str))]
    eligible = scored[~scored["pit_st"]].copy(); pool = _pool(eligible); top4 = _top4(pool)
    labels = _labels(panel, args.end); summary = _diagnostic_summary(eligible, pool, top4, labels)
    scored["score"] = scored["quiet_score"]; scored["policy_id"] = "rcqt.stage1.recent_followthrough.top4.v1"; keys = set(zip(pool["asof"], pool["ts_code"]))
    scored["candidate_status"] = ["CANDIDATE" if (d, c) in keys else "REJECTED" for d, c in zip(scored["asof"], scored["ts_code"])]
    scored["rejection_reason"] = ""; scored.loc[scored["pit_st"], "rejection_reason"] = "pit_st"; scored.loc[~scored["pit_st"] & (scored["candidate_status"] == "REJECTED"), "rejection_reason"] = "quiet_or_right_or_ret5_failed"
    px = prices.copy(); px["trade_date"] = pd.to_datetime(px["trade_date"], utc=True).dt.normalize(); px["amount"] = pd.to_numeric(px["amount"], errors="raise"); px = px.sort_values(["ts_code", "trade_date"], kind="mergesort"); px["adv20"] = px.groupby("ts_code")["amount"].transform(lambda s: s.rolling(20, min_periods=20).median())
    actions = load_corporate_actions(args.start, args.end, ts_codes=codes); sessions = pd.DatetimeIndex(sorted(pd.to_datetime(panel["event_time"], utc=True).dt.normalize().drop_duplicates())); portfolios = {}
    for year in (2024, 2025):
        signals = top4[top4["asof"].dt.year == year]
        if signals.empty: continue
        pos = sessions.get_loc(signals["asof"].max()); execution_end = sessions[min(pos + 11, len(sessions) - 1)]; start = pd.Timestamp(f"{year}-01-01", tz="UTC"); split_px = px[px["trade_date"].between(start, execution_end)]; split_actions = actions[pd.to_datetime(actions["ex_date"], utc=True).between(start, execution_end)] if len(actions) else actions
        for cost, slippage in (("base", 0.001), ("stress", 0.003)):
            result, metrics = _backtest(signals, split_px, split_actions, slippage=slippage); target = output / "backtests" / str(year) / cost; target.mkdir(parents=True, exist_ok=True)
            for name in ("orders", "trades", "nav", "positions", "corporate_actions"): result[name].to_csv(target / f"{name}.csv", index=False)
            metrics.update({"signal_start": str(signals["asof"].min().date()), "signal_end": str(signals["asof"].max().date()), "execution_end": str(execution_end.date())}); _write_json(target / "metrics.json", metrics); portfolios[f"{year}_{cost}"] = metrics
    _write_json(output / "SUMMARY.json", {"experiment_id": "recent_followthrough_v1", "status": "historical_diagnostic_not_lockbox", "events": summary, "portfolios": portfolios})
    scored.to_parquet(output / "score_ledger.parquet", index=False); pool.to_parquet(output / "candidate_ledger.parquet", index=False); top4.to_csv(output / "selection_ledger.csv", index=False)
    _write_json(output / "DATA_MANIFEST.json", {"experiment_id": "recent_followthrough_v1", "config": str(args.config.resolve()), "config_sha256": _sha(args.config), "codes_source": str(args.codes_source.resolve()), "codes_source_sha256": _sha(args.codes_source), "raw_start": raw_start, "source_end": args.end, "pit_st_audit": st_audit, "source_audit": source_audit, "model_training": False, "parameter_sweep": False, "historical_results_are_diagnostic": True})
    lines = ["# Recent Follow-through V1", "", "Historical results are diagnostic only.", "", f"- Candidate rows: {summary['candidate_rows']}", f"- Selected rows: {summary['selected_rows']}"]
    for year, item in summary.get("by_year", {}).items(): lines.append(f"- {year}: candidate mean={item['candidate_pool']['mean_return']:+.2%}, PF={item['candidate_pool']['profit_factor']:.3f}, rows={item['candidate_pool']['rows']}")
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--codes-source", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--start", default="2024-01-01"); parser.add_argument("--end", default="2026-01-20"); run(parser.parse_args())


if __name__ == "__main__": main()
