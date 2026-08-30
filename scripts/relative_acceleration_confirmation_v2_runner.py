"""Run the preregistered transparent Stage-1 Relative Acceleration V2 experiment.

The historical path is diagnostic only.  The forward path is append-only and
never invents labels for signals whose T+10 outcome is not mature yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.data.execution_source import load_execution_panel
from aistock9988.data.q70_source import load_f0_panel
from aistock9988.labeling.q70 import build_q70_t10_labels
from aistock9988.selection.rcqt import score_rcqt
from aistock9988.time.session import session_close

from rcqt_quantdb_sample_runner import _features
from rcqt_stage1_quality_runner import (
    LABEL_PROFILE,
    _backtest,
    _date_chunks,
    _load_pit_st_keys,
    _metrics,
    _rank_skill,
    _sha,
    _uplift,
    _write_json,
)
from relative_orderly_continuation_runner import _load_sources


def _add_relative_acceleration(features: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Add PIT market/industry relative returns and their five-session lags."""
    # Rebuild ret5 with the exchange calendar.  ``pct_change(5)`` over a
    # security's surviving rows can otherwise turn five observations into five
    # calendar sessions when the security was suspended or missing upstream.
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(panel["event_time"], utc=True).dt.normalize().drop_duplicates()))
    if len(sessions) <= 5:
        raise RuntimeError("at least six sessions are required for ret5 acceleration")
    lag_map = pd.Series(sessions[:-5], index=sessions[5:])
    features = features.copy()
    features["acceleration_lag_asof"] = features["asof"].map(lag_map)
    lag_close = features[["ts_code", "asof", "close"]].rename(columns={
        "asof": "acceleration_lag_asof", "close": "close_lag5",
    })
    features = features.merge(lag_close, on=["ts_code", "acceleration_lag_asof"], how="left", validate="many_to_one")
    features["ret5"] = features["close"] / features["close_lag5"] - 1.0
    industry = panel[["event_time", "ts_code", "industry"]].copy()
    industry["asof"] = pd.to_datetime(industry.pop("event_time"), utc=True).dt.normalize()
    industry = industry.drop_duplicates(["asof", "ts_code"], keep="last")
    out = features.merge(industry, on=["asof", "ts_code"], how="left", validate="one_to_one")
    if out["industry"].isna().any():
        raise RuntimeError("PIT industry is missing after F0 feature construction")
    for horizon in (5, 20, 60):
        source = f"ret{horizon}"
        out[f"market_excess_ret{horizon}"] = out[source] - out.groupby("asof", sort=False)[source].transform("median")
        out[f"industry_excess_ret{horizon}"] = out[source] - out.groupby(["asof", "industry"], sort=False)[source].transform("median")

    out = out.sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True)
    # Use the complete exchange calendar, not the feature subset.  Otherwise
    # a date with no surviving feature rows could silently redefine T-5.
    lag_cols = [
        "ts_code", "asof", "market_excess_ret5", "industry_excess_ret5",
    ]
    lag = out[lag_cols].rename(columns={
        "asof": "acceleration_lag_asof",
        "market_excess_ret5": "market_excess_ret5_lag5",
        "industry_excess_ret5": "industry_excess_ret5_lag5",
    })
    out = out.merge(lag, on=["ts_code", "acceleration_lag_asof"], how="left", validate="many_to_one")
    out["market_excess_ret5_acceleration"] = out["market_excess_ret5"] - out["market_excess_ret5_lag5"]
    out["industry_excess_ret5_acceleration"] = out["industry_excess_ret5"] - out["industry_excess_ret5_lag5"]
    out["acceleration_lag_missing"] = out["market_excess_ret5_lag5"].isna() | out["industry_excess_ret5_lag5"].isna()
    return out


def _candidates(scored: pd.DataFrame) -> pd.DataFrame:
    return scored[
        scored["right_confirmed"]
        & scored["market_excess_ret5_acceleration"].gt(0)
        & scored["industry_excess_ret5_acceleration"].gt(0)
        & scored["market_excess_ret20"].gt(0)
        & scored["industry_excess_ret20"].gt(0)
        & scored["close"].ge(scored["ma5"])
        & scored["close"].ge(scored["prev3_high"])
        & scored["dist_ma60"].gt(0)
        & scored["dist_ma60"].le(0.15)
        & scored["dd20"].ge(-0.10)
        & scored["ret20"].le(0.25)
        & scored["ret60"].le(0.35)
        & scored["volume_ratio_20"].between(0.70, 2.50, inclusive="both")
    ].copy()


def _top4(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["asof", "quiet_score", "ts_code"], ascending=[True, False, True], kind="mergesort").groupby("asof", sort=True).head(4).copy()
    out["candidate_rank"] = out.groupby("asof").cumcount() + 1
    out["selected"] = True
    out["selection_decision_id"] = "rcqt.stage1.relative_acceleration.top4.v2-" + out["asof"].dt.strftime("%Y%m%d")
    out["policy_id"] = "rcqt.stage1.relative_acceleration.top4.v2"
    out["target_weight"] = 0.12
    out["context_hash"] = out.apply(
        lambda row: hashlib.sha256(json.dumps({
            "policy_id": "rcqt.stage1.relative_acceleration.top4.v2",
            "asof": str(row["asof"]),
            "candidate_contract": "relative_ret5_acceleration+relative_ret20+right_confirmed+orderly_position",
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), axis=1,
    )
    return out


def _mature_labels(panel: pd.DataFrame, source_end: str) -> pd.DataFrame:
    sessions = pd.DatetimeIndex(sorted(panel["event_time"].drop_duplicates()))
    labels = build_q70_t10_labels(panel, profile=LABEL_PROFILE, session_dates=sessions)
    labels = labels.rename(columns={"event_time": "asof", "available_time": "label_available_time"})
    labels["asof"] = pd.to_datetime(labels["asof"], utc=True).dt.normalize()
    labels["label_available_time"] = pd.to_datetime(labels["label_available_time"], utc=True)
    return labels[
        labels["label_available_time"] <= session_close(pd.Timestamp(source_end, tz="UTC"))
    ][["asof", "ts_code", "label_return", "label_available_time", "exit_time"]]


def _write_partitions(root: Path, frame: pd.DataFrame) -> list[dict[str, object]]:
    """Write immutable one-signal-date files; never rewrite prior evidence."""
    written: list[dict[str, object]] = []
    if frame.empty:
        return written
    if frame.duplicated(["asof", "ts_code"]).any():
        raise RuntimeError("forward batch contains duplicate asof/ts_code keys")
    for day, group in frame.groupby("asof", sort=True):
        day_text = pd.Timestamp(day).strftime("%Y-%m-%d")
        target = root / f"date={day_text}.parquet"
        if target.exists():
            raise RuntimeError(f"append-only partition already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        group.sort_values(["asof", "ts_code"], kind="mergesort").to_parquet(target, index=False)
        written.append({"path": str(target), "sha256": _sha(target), "rows": len(group), "asof": day_text})
    return written


def _diagnostic_summary(scored: pd.DataFrame, candidates: pd.DataFrame, top4: pd.DataFrame, labels: pd.DataFrame) -> dict[str, object]:
    mature_scored = scored.merge(labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
    mature_candidates = candidates.merge(labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
    mature_top4 = top4.merge(labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
    result: dict[str, object] = {
        "universe": _metrics(mature_scored),
        "candidate_pool": _metrics(mature_candidates),
        "top4": _metrics(mature_top4),
        "candidate_vs_universe": _uplift(mature_scored, mature_candidates),
        "top4_vs_candidate": _uplift(mature_candidates, mature_top4),
        "rank_skill": _rank_skill(mature_candidates, "quiet_score") if len(mature_candidates) else {},
        "mature_through": str(mature_candidates["asof"].max().date()) if len(mature_candidates) else None,
        "candidate_rows": len(candidates),
        "selected_rows": len(top4),
    }
    by_year = {}
    for year, group in mature_candidates.groupby(mature_candidates["asof"].dt.year):
        control = mature_scored[mature_scored["asof"].dt.year == year]
        selected = mature_top4[mature_top4["asof"].dt.year == year]
        by_year[str(year)] = {
            "universe": _metrics(control),
            "candidate_pool": _metrics(group),
            "top4": _metrics(selected),
            "candidate_vs_universe": _uplift(control, group),
        }
    result["by_year"] = by_year
    return result


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if args.mode == "diagnostic" and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    codes = sorted(pd.read_parquet(args.codes_source, columns=["ts_code"])["ts_code"].astype(str).unique())
    raw_start = (pd.Timestamp(args.start) - pd.Timedelta(days=140)).strftime("%Y-%m-%d")
    st_keys, st_audit = _load_pit_st_keys(codes, args.start, args.end)
    panel, prices, source_audit = _load_sources(raw_start, args.end, codes)
    # Keep the lookback rows through relative-feature construction so T-5 is
    # available on the first requested signal dates; filter to the signal
    # window only after acceleration has been built.
    features = _features(panel, prices, raw_start)
    features["asof"] = pd.to_datetime(features["asof"], utc=True).dt.normalize()
    features["available_time"] = pd.to_datetime(features["available_time"], utc=True)
    features = _add_relative_acceleration(features, panel)
    features = features[features["asof"] >= pd.Timestamp(args.start, tz="UTC")].copy()
    if features.empty or features["asof"].max() < pd.Timestamp(args.end, tz="UTC"):
        raise RuntimeError(f"feature coverage stops at {features['asof'].max() if len(features) else None}, before {args.end}")
    if (features["available_time"] > features["asof"].map(session_close)).any():
        raise AssertionError("feature PIT violation")
    scored = score_rcqt(features)
    scored["pit_st"] = [(day, code) in st_keys for day, code in zip(scored["asof"], scored["ts_code"].astype(str))]
    scored_all = scored.copy()
    eligible_scored = scored[~scored["pit_st"]].copy()
    candidates = _candidates(eligible_scored)
    top4 = _top4(candidates)
    context_seed = {
        "policy_id": "rcqt.stage1.relative_acceleration.top4.v2",
        "config_sha256": _sha(args.config),
        "codes_source_sha256": _sha(args.codes_source),
        "source_end": args.end,
    }
    scored_all["policy_id"] = context_seed["policy_id"]
    scored_all["score"] = scored_all["quiet_score"]
    scored_all["context_hash"] = scored_all["asof"].map(
        lambda day: hashlib.sha256(json.dumps({**context_seed, "asof": str(day)}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    candidates = scored_all.loc[candidates.index].copy()
    top4 = top4.drop(columns=["context_hash", "policy_id"], errors="ignore").merge(
        scored_all[["asof", "ts_code", "policy_id", "context_hash"]], on=["asof", "ts_code"], how="left", validate="one_to_one"
    )
    labels = _mature_labels(panel, args.end)

    # Preserve a complete decision ledger, including rejected rows and the
    # exact PIT inputs needed to explain every candidate decision.
    requirement_columns = [
        "right_confirmed", "market_excess_ret5_acceleration", "industry_excess_ret5_acceleration",
        "market_excess_ret20", "industry_excess_ret20", "close", "ma5", "prev3_high",
        "dist_ma60", "dd20", "volume_ratio_20",
    ]
    candidate_indices = set(candidates.index)
    scored_all["candidate_status"] = ["CANDIDATE" if idx in candidate_indices else "REJECTED" for idx in scored_all.index]
    scored_all["rejection_reason"] = ""
    for col in requirement_columns:
        if col not in scored_all.columns:
            raise RuntimeError(f"V2 candidate requirement column missing: {col}")
        if col in {"right_confirmed"}:
            failed = ~scored_all[col].astype(bool)
        elif col in {"close", "ma5", "prev3_high"}:
            failed = scored_all["close"].lt(scored_all["ma5"]) | scored_all["close"].lt(scored_all["prev3_high"])
        elif col == "dist_ma60":
            failed = ~scored_all[col].gt(0) | ~scored_all[col].le(0.15)
        elif col == "dd20":
            failed = ~scored_all[col].ge(-0.10)
        elif col == "volume_ratio_20":
            failed = ~scored_all[col].between(0.70, 2.50, inclusive="both")
        else:
            failed = ~scored_all[col].gt(0)
        scored_all.loc[failed & scored_all["rejection_reason"].eq(""), "rejection_reason"] = col
    scored_all.loc[scored_all["acceleration_lag_missing"], "rejection_reason"] = "missing_t_minus_5_session"
    scored_all.loc[scored_all["pit_st"], "rejection_reason"] = "pit_st"

    if args.mode == "forward":
        forward_cut = pd.Timestamp(args.forward_start, tz="UTC")
        prior_status = output / "FORWARD_STATUS.json"
        prior_max = None
        prior_manifest_sha = None
        if prior_status.exists():
            prior_payload = json.loads(prior_status.read_text(encoding="utf-8"))
            prior_max = prior_payload.get("max_asof")
            prior_manifest_sha = prior_payload.get("manifest_sha256")
        candidates = candidates[candidates["asof"] >= forward_cut].copy()
        top4 = top4[top4["asof"] >= forward_cut].copy()
        candidate_full = scored_all[scored_all["asof"] >= forward_cut].copy()
        if prior_max:
            prior_cut = pd.Timestamp(prior_max, tz="UTC")
            candidate_full = candidate_full[candidate_full["asof"] > prior_cut]
            candidates = candidates[candidates["asof"] > prior_cut]
            top4 = top4[top4["asof"] > prior_cut]
        score_parts = _write_partitions(output / "forward" / "score", candidate_full)
        candidate_parts = _write_partitions(output / "forward" / "candidate", candidates)
        selection_parts = _write_partitions(output / "forward" / "selection", top4)
        batch_items = score_parts + candidate_parts + selection_parts
        batch_hash = hashlib.sha256(json.dumps(batch_items, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        max_asof = max([item["asof"] for item in batch_items], default=prior_max)
        manifest = {
            "schema_version": "v2-forward-partition-1",
            "experiment_id": "relative_acceleration_confirmation_v2",
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "previous_manifest_sha256": prior_manifest_sha,
            "batch_sha256": batch_hash,
            "parts": batch_items,
            "config_sha256": _sha(args.config),
            "codes_source_sha256": _sha(args.codes_source),
            "source_end": args.end,
            "lag_missing_rows": int(candidate_full["acceleration_lag_missing"].sum()),
        }
        manifest_path = output / f"FORWARD_MANIFEST_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        _write_json(manifest_path, manifest)
        manifest_sha = _sha(manifest_path)
        _write_json(output / "FORWARD_STATUS.json", {
            "experiment_id": "relative_acceleration_confirmation_v2",
            "mode": "forward_append_only",
            "forward_start": args.forward_start,
            "source_end": args.end,
            "new_score_rows": len(candidate_full),
            "new_candidate_rows": len(candidates),
            "new_selection_rows": len(top4),
            "max_asof": max_asof,
            "manifest_sha256": manifest_sha,
            "mature_forward_rows": int(len(candidates.merge(labels, on=["asof", "ts_code"], how="inner"))),
            "lag_missing_rows": int(candidate_full["acceleration_lag_missing"].sum()),
            "note": "No return is reported for immature forward signals.",
        })
        _write_json(output / "DATA_MANIFEST.json", {
            "experiment_id": "relative_acceleration_confirmation_v2",
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
            "append_only": True,
        })
        return

    summary = _diagnostic_summary(eligible_scored, candidates, top4, labels)
    actions = load_corporate_actions(args.start, args.end, ts_codes=codes)
    px = prices.copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"], utc=True).dt.normalize()
    px["amount"] = pd.to_numeric(px["amount"], errors="raise")
    px = px.sort_values(["ts_code", "trade_date"], kind="mergesort")
    px["adv20"] = px.groupby("ts_code")["amount"].transform(lambda s: s.rolling(20, min_periods=20).median())
    portfolios = {}
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(panel["event_time"], utc=True).dt.normalize().drop_duplicates()))
    for year in (2024, 2025):
        signals = top4[top4["asof"].dt.year == year]
        if signals.empty:
            continue
        last_signal = signals["asof"].max()
        signal_pos = sessions.get_loc(last_signal)
        execution_end = sessions[min(signal_pos + LABEL_PROFILE.entry_delay_sessions + LABEL_PROFILE.horizon_sessions, len(sessions) - 1)]
        year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        split_prices = px[px["trade_date"].between(year_start, execution_end)].copy()
        split_actions = actions[pd.to_datetime(actions["ex_date"], utc=True).between(year_start, execution_end)].copy() if len(actions) else actions.copy()
        for cost, slippage in (("base", 0.001), ("stress", 0.003)):
            result, metrics = _backtest(signals, split_prices, split_actions, slippage=slippage)
            target = output / "backtests" / str(year) / cost
            target.mkdir(parents=True, exist_ok=True)
            for name in ("orders", "trades", "nav", "positions", "corporate_actions"):
                result[name].to_csv(target / f"{name}.csv", index=False)
            _write_json(target / "metrics.json", metrics)
            metrics["signal_start"] = str(signals["asof"].min().date())
            metrics["signal_end"] = str(last_signal.date())
            metrics["execution_end"] = str(execution_end.date())
            portfolios[f"{year}_{cost}"] = metrics
    _write_json(output / "SUMMARY.json", {"experiment_id": "relative_acceleration_confirmation_v2", "status": "historical_diagnostic_not_lockbox", "events": summary, "portfolios": portfolios})
    candidates.to_parquet(output / "candidate_relative_acceleration.parquet", index=False)
    top4.to_csv(output / "selection_relative_acceleration_top4.csv", index=False)
    scored_all.to_parquet(output / "score_ledger.parquet", index=False)
    candidates.assign(candidate_status="CANDIDATE").to_parquet(output / "candidate_ledger.parquet", index=False)
    _write_json(output / "DATA_MANIFEST.json", {
        "experiment_id": "relative_acceleration_confirmation_v2",
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
    })
    lines = ["# Relative Acceleration Confirmation V2", "", "Historical results are diagnostic only; they are not the forward lockbox.", "", f"- Candidate rows: {summary['candidate_rows']}", f"- Selected rows: {summary['selected_rows']}", f"- Mature through: {summary['mature_through']}", ""]
    for year, item in summary.get("by_year", {}).items():
        lines.append(f"- {year}: candidate mean={item['candidate_pool']['mean_return']:+.2%}, PF={item['candidate_pool']['profit_factor']:.3f}, rows={item['candidate_pool']['rows']}")
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.rglob("*")) if path.is_file()}
    _write_json(output / "ARTIFACT_HASHES.json", artifacts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--codes-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("diagnostic", "forward"), default="diagnostic")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--forward-start", default="2026-08-22")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
