"""Strict monthly rolling XGBRanker backtest for corrected RCQT candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.data.execution_source import load_execution_panel
from aistock9988.data.q70_source import load_f0_panel
from aistock9988.data.quantdb import readonly_connection
from aistock9988.labeling.maturity import LabelProfile
from aistock9988.labeling.q70 import build_q70_t10_labels
from aistock9988.models.trainer import train_ranker
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.selection.rcqt import score_rcqt
from aistock9988.time.session import session_close

from rcqt_corrected_xgb_ranker_runner import FEATURES, MODEL_PARAMS
from rcqt_quantdb_sample_runner import _features


LABEL_PROFILE = LabelProfile("label.endpoint_open_open_t10.v1", 1, 10, 11)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def _relevance(labels: pd.Series, dates: pd.Series) -> pd.Series:
    percentile = labels.groupby(dates).rank(method="first", pct=True)
    return np.minimum((percentile * 5).astype(int), 4).astype(float)


def _select(frame: pd.DataFrame, score: str, policy: str) -> pd.DataFrame:
    selected = frame.sort_values(
        ["asof", score, "ts_code"], ascending=[True, False, True], kind="mergesort",
    ).groupby("asof", sort=True).head(4).copy()
    selected["candidate_rank"] = selected.groupby("asof").cumcount() + 1
    selected["selected"] = True
    selected["selection_decision_id"] = policy + "-" + selected["asof"].dt.strftime("%Y%m%d")
    selected["policy_id"] = policy
    selected["target_weight"] = 0.12
    return selected


def _load_pit_st_keys(codes: list[str], start: str, requested_end: str) -> tuple[set[tuple[pd.Timestamp, str]], str, dict]:
    placeholders = ",".join(["%s"] * len(codes))
    with readonly_connection() as connection:
        coverage = pd.read_sql_query(
            "SELECT MIN(trade_date) min_date, MAX(trade_date) max_date, COUNT(*) rows_n "
            "FROM stock_st_ts WHERE trade_date >= %s AND trade_date <= %s",
            connection, params=(start, requested_end),
        ).iloc[0]
        if pd.isna(coverage["max_date"]):
            raise RuntimeError("stock_st_ts has no PIT coverage for the backtest")
        effective_end = min(pd.Timestamp(requested_end), pd.Timestamp(coverage["max_date"])).strftime("%Y-%m-%d")
        frame = pd.read_sql_query(
            f"SELECT trade_date, ts_code, name, update_time FROM stock_st_ts "
            f"WHERE trade_date >= %s AND trade_date <= %s AND ts_code IN ({placeholders}) "
            "ORDER BY trade_date, ts_code",
            connection, params=(start, effective_end, *codes),
        )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], utc=True).dt.normalize()
    keys = set(zip(frame["trade_date"], frame["ts_code"].astype(str)))
    audit = {
        "source": "quant_db.stock_st_ts", "rows": len(frame),
        "coverage_min": str(coverage["min_date"]), "coverage_max": str(coverage["max_date"]),
        "requested_end": requested_end, "effective_end": effective_end,
        "policy": "exclude exact ts_code/trade_date keys present in PIT ST table",
    }
    return keys, effective_end, audit


def _date_chunks(start: str, end: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    cursor = pd.Timestamp(start).normalize()
    terminal = pd.Timestamp(end).normalize()
    while cursor <= terminal:
        chunk_end = min(cursor + pd.DateOffset(months=3) - pd.Timedelta(days=1), terminal)
        chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def _load_sources_chunked(start: str, end: str, codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel_parts: list[pd.DataFrame] = []
    price_parts: list[pd.DataFrame] = []
    audit_parts: list[dict] = []
    chunks = _date_chunks(start, end)
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"source_chunk {index}/{len(chunks)} start={chunk_start} end={chunk_end}", flush=True)
        panel, audit = load_f0_panel(chunk_start, chunk_end, ts_codes=codes, return_audit=True)
        prices = load_execution_panel(chunk_start, chunk_end, ts_codes=codes)
        panel_parts.append(panel)
        price_parts.append(prices)
        audit_parts.append(audit)
    panel = pd.concat(panel_parts, ignore_index=True).sort_values(
        ["event_time", "ts_code"], kind="mergesort",
    ).reset_index(drop=True)
    prices = pd.concat(price_parts, ignore_index=True).sort_values(
        ["trade_date", "ts_code"], kind="mergesort",
    ).reset_index(drop=True)
    if panel.duplicated(["event_time", "ts_code"]).any():
        raise ValueError("chunked F0 panel contains duplicate event_time/ts_code")
    if prices.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("chunked execution panel contains duplicate trade_date/ts_code")
    audit = {
        "source_id": "quant_db", "chunks": chunks,
        "membership_rows_loaded_sum": sum(int(part.get("membership_rows_loaded", 0)) for part in audit_parts),
        "industry_resolution_dates": sum(len(part.get("industry_resolution", [])) for part in audit_parts),
        "sector_relative_statistics": sorted({str(part.get("sector_relative_statistic")) for part in audit_parts}),
    }
    return panel, prices, audit


def _monthly_walkforward(candidates: pd.DataFrame, labels: pd.DataFrame,
                         sessions: pd.DatetimeIndex, *, start: str, end: str,
                         output: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    start_day = pd.Timestamp(start, tz="UTC")
    end_day = pd.Timestamp(end, tz="UTC")
    months = pd.period_range(start_day.tz_localize(None).to_period("M"), end_day.tz_localize(None).to_period("M"), freq="M")
    for month in months:
        month_start = pd.Timestamp(month.start_time, tz="UTC")
        month_end = min(pd.Timestamp(month.end_time, tz="UTC").normalize(), end_day)
        test = candidates[candidates["asof"].between(max(month_start, start_day), month_end)].copy()
        if test.empty:
            audits.append({"month": str(month), "status": "NO_CANDIDATES"})
            continue
        prior_sessions = sessions[sessions < month_start]
        if len(prior_sessions) == 0:
            raise RuntimeError(f"no prior session available for {month}")
        cutoff = prior_sessions[-1]
        cutoff_time = session_close(cutoff)
        window_start = cutoff - pd.DateOffset(months=12)
        mature_labels = labels[
            (labels["asof"] > window_start)
            & (labels["asof"] <= cutoff)
            & (labels["label_available_time"] <= cutoff_time)
        ][["asof", "ts_code", "label_return", "label_available_time", "exit_time"]]
        train = candidates[
            (candidates["asof"] > window_start) & (candidates["asof"] <= cutoff)
        ].merge(mature_labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
        group_size = train.groupby("asof")["ts_code"].transform("size")
        train = train[group_size >= 2].sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True)
        if train.empty or train["label_available_time"].max() > cutoff_time:
            raise RuntimeError(f"invalid or immature training data for {month}")
        if train["asof"].max() >= test["asof"].min():
            raise AssertionError(f"train/test date overlap for {month}")
        model_id = f"rcqt_rolling_xgb_{month.strftime('%Y%m')}_cutoff_{cutoff.strftime('%Y%m%d')}"
        y = _relevance(train["label_return"], train["asof"])
        artifact = train_ranker(
            train[list(FEATURES)], y, group_dates=train["asof"],
            feature_set_id="feature.rcqt_corrected_xgb14.v1",
            label_profile_id=LABEL_PROFILE.id, training_cutoff=str(cutoff_time),
            model_id=model_id, output_dir=output / "models", params=MODEL_PARAMS,
            metadata_extra={
                "walkforward_month": str(month), "train_window_months": 12,
                "candidate_contract": "corrected quiet_eligible AND right_confirmed AND not PIT-ST",
                "target_contract": "within-date five-level relevance from mature T+10 return",
            },
        )
        model = XGBRanker()
        model.load_model(output / "models" / f"{model_id}.json")
        test["xgb_score"] = model.predict(test[list(FEATURES)])
        test["model_id"] = model_id
        test["model_cutoff"] = cutoff_time
        predictions.append(test)
        audits.append({
            "month": str(month), "status": "TRAINED", "model_id": model_id,
            "model_sha256": artifact.model_sha256, "cutoff": str(cutoff_time),
            "window_start_exclusive": str(window_start), "train_rows": len(train),
            "train_dates": int(train["asof"].nunique()), "train_start": str(train["asof"].min()),
            "train_end": str(train["asof"].max()),
            "max_label_available_time": str(train["label_available_time"].max()),
            "test_start": str(test["asof"].min()), "test_end": str(test["asof"].max()),
            "test_rows": len(test), "test_dates": int(test["asof"].nunique()),
        })
    if not predictions:
        raise RuntimeError("walk-forward produced no predictions")
    return pd.concat(predictions, ignore_index=True), audits


def _event_metrics(frame: pd.DataFrame) -> dict[str, object]:
    values = pd.to_numeric(frame["label_return"], errors="raise")
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return {
        "rows": len(frame), "dates": int(frame["asof"].nunique()),
        "mean_return": float(values.mean()), "median_return": float(values.median()),
        "win_rate": float((values > 0).mean()), "profit_factor": gains / losses if losses else None,
        "down_8pct_rate": float((values <= -0.08).mean()),
        "up_10pct_rate": float((values >= 0.10).mean()),
    }


def _rank_skill(frame: pd.DataFrame, score: str) -> dict[str, object]:
    daily = frame.groupby("asof")[[score, "label_return"]].apply(
        lambda group: group[score].corr(group["label_return"], method="spearman")
    ).dropna()
    return {
        "dates": len(daily), "mean_daily_rank_ic": float(daily.mean()),
        "median_daily_rank_ic": float(daily.median()),
        "positive_rank_ic_ratio": float((daily > 0).mean()),
    }


def _backtest(signals: pd.DataFrame, prices: pd.DataFrame, actions: pd.DataFrame,
              *, slippage: float) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    result = run_backtest(
        signals, prices, corporate_actions=actions,
        config=BacktestConfig(
            max_positions=4, hold_sessions=10, stop_loss_pct=-0.08,
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
    return result, metrics


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    codes_source_hash = _sha(args.codes_source)
    codes = sorted(pd.read_parquet(args.codes_source, columns=["ts_code"])["ts_code"].astype(str).unique())
    feature_start = (pd.Timestamp(args.start) - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    st_keys, effective_end, st_audit = _load_pit_st_keys(codes, feature_start, args.end)
    raw_start = (pd.Timestamp(feature_start) - pd.Timedelta(days=140)).strftime("%Y-%m-%d")

    panel, prices, source_audit = _load_sources_chunked(raw_start, effective_end, codes)
    features = _features(panel, prices, feature_start)
    features["asof"] = pd.to_datetime(features["asof"], utc=True).dt.normalize()
    features["available_time"] = pd.to_datetime(features["available_time"], utc=True)
    features = features[features["asof"] <= pd.Timestamp(effective_end, tz="UTC")].copy()
    if (features["available_time"] > features["asof"].map(session_close)).any():
        raise AssertionError("feature PIT violation")
    scored = score_rcqt(features)
    scored["pit_st"] = [
        (day, code) in st_keys for day, code in zip(scored["asof"], scored["ts_code"].astype(str))
    ]
    candidates = scored[
        scored["quiet_eligible"] & scored["right_confirmed"] & ~scored["pit_st"]
    ].copy()
    if candidates.empty:
        raise RuntimeError("rolling candidate pool is empty")

    sessions = pd.DatetimeIndex(sorted(panel["event_time"].drop_duplicates()))
    labels = build_q70_t10_labels(panel, profile=LABEL_PROFILE, session_dates=sessions)
    labels = labels.rename(columns={"event_time": "asof", "available_time": "label_available_time"})
    labels["asof"] = pd.to_datetime(labels["asof"], utc=True).dt.normalize()
    labels["label_available_time"] = pd.to_datetime(labels["label_available_time"], utc=True)
    predictions, training_audit = _monthly_walkforward(
        candidates, labels, sessions, start=args.start, end=effective_end, output=output,
    )
    xgb = _select(predictions, "xgb_score", "rcqt.monthly_rolling_xgb.v1")
    rule = _select(predictions, "quiet_score", "rcqt.monthly_rolling_rule_control.v1")

    mature_labels = labels[
        labels["label_available_time"] <= session_close(pd.Timestamp(effective_end))
    ][["asof", "ts_code", "label_return", "label_available_time", "exit_time"]]
    mature_predictions = predictions.merge(
        mature_labels, on=["asof", "ts_code"], how="inner", validate="one_to_one",
    )
    mature_xgb = xgb.merge(mature_labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
    mature_rule = rule.merge(mature_labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
    mature_signal_end = min(mature_xgb["asof"].max(), mature_rule["asof"].max())

    features.to_parquet(output / "feature_ledger.parquet", index=False)
    predictions.to_parquet(output / "prediction_ledger.parquet", index=False)
    xgb.to_csv(output / "xgb_selection_ledger.csv", index=False)
    rule.to_csv(output / "rule_selection_ledger.csv", index=False)
    mature_xgb.to_csv(output / "mature_xgb_event_ledger.csv", index=False)
    mature_rule.to_csv(output / "mature_rule_event_ledger.csv", index=False)
    _write_json(output / "training_audit.json", training_audit)

    px = prices.copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"], utc=True).dt.normalize()
    px["amount"] = pd.to_numeric(px["amount"], errors="raise")
    px = px.sort_values(["ts_code", "trade_date"], kind="mergesort")
    px["adv20"] = px.groupby("ts_code")["amount"].transform(
        lambda series: series.rolling(20, min_periods=20).median()
    )
    px = px[px["trade_date"].between(args.start, effective_end)].copy()
    actions = load_corporate_actions(args.start, effective_end, ts_codes=codes)
    portfolio_summary: dict[str, object] = {}
    for policy, ledger in (("xgb", xgb), ("rule", rule)):
        signals = ledger[ledger["asof"] <= mature_signal_end].copy()
        for label, slippage in (("base", 0.001), ("stress", 0.003)):
            result, metrics = _backtest(signals, px, actions, slippage=slippage)
            target = output / "backtests" / policy / label
            target.mkdir(parents=True, exist_ok=True)
            for name in ("orders", "trades", "nav", "positions", "corporate_actions"):
                result[name].to_csv(target / f"{name}.csv", index=False)
            _write_json(target / "metrics.json", metrics)
            portfolio_summary[f"{policy}_{label}"] = metrics

    monthly_events: dict[str, object] = {}
    for month, frame in mature_predictions.groupby(mature_predictions["asof"].dt.strftime("%Y-%m")):
        month_xgb = mature_xgb[mature_xgb["asof"].dt.strftime("%Y-%m") == month]
        month_rule = mature_rule[mature_rule["asof"].dt.strftime("%Y-%m") == month]
        monthly_events[month] = {
            "candidate_pool": _event_metrics(frame), "xgb_top4": _event_metrics(month_xgb),
            "rule_top4": _event_metrics(month_rule), "xgb_rank_skill": _rank_skill(frame, "xgb_score"),
            "rule_rank_skill": _rank_skill(frame, "quiet_score"),
        }
    event_summary = {
        "mature_through_signal_date": str(mature_signal_end.date()),
        "candidate_pool": _event_metrics(mature_predictions),
        "xgb_top4": _event_metrics(mature_xgb), "rule_top4": _event_metrics(mature_rule),
        "xgb_rank_skill": _rank_skill(mature_predictions, "xgb_score"),
        "rule_rank_skill": _rank_skill(mature_predictions, "quiet_score"),
        "monthly": monthly_events,
    }
    _write_json(output / "EVENT_SUMMARY.json", event_summary)
    _write_json(output / "PORTFOLIO_SUMMARY.json", portfolio_summary)
    manifest = {
        "kind": "monthly_rolling_xgb_backtest", "requested_start": args.start,
        "requested_end": args.end, "effective_end": effective_end,
        "feature_start": feature_start, "raw_lookback_start": raw_start,
        "codes_source": str(args.codes_source.resolve()), "codes_source_sha256": codes_source_hash,
        "fixed_universe_codes": len(codes), "pit_st_audit": st_audit,
        "source_audit": source_audit, "feature_rows": len(features),
        "candidate_rows": len(candidates), "prediction_rows": len(predictions),
        "prediction_dates": int(predictions["asof"].nunique()),
        "mature_signal_end": str(mature_signal_end), "train_window_months": 12,
        "model_params": MODEL_PARAMS, "features": FEATURES, "parameter_sweep": False,
        "credentials_persisted": False,
    }
    _write_json(output / "DATA_MANIFEST.json", manifest)
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
