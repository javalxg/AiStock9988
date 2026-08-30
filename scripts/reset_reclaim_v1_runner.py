"""One-shot preregistered reset-reclaim fixed-rule replay."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.labeling.q70 import build_q70_t10_labels
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.time.session import session_close

from rcqt_quantdb_sample_runner import _features
from rcqt_stage1_quality_runner import (
    LABEL_PROFILE,
    _load_pit_st_keys,
    _metrics,
    _sha,
    _uplift,
    _write_json,
)
from relative_orderly_continuation_runner import _load_sources


def _rank(series: pd.Series) -> pd.Series:
    ordered = sorted(series.index, key=lambda key: (float(series.loc[key]), str(key)))
    values = pd.Series(index=ordered, data=range(1, len(ordered) + 1), dtype=float)
    return values.reindex(series.index) / max(len(series), 1)


def _load_codes(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=["ts_code"])
    else:
        frame = pd.read_csv(path, usecols=["ts_code"])
    codes = sorted(frame["ts_code"].astype(str).str.upper().unique().tolist())
    if not codes:
        raise ValueError("codes source is empty")
    return codes


def _add_reset_reclaim_features(features: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["asof"] = pd.to_datetime(out["asof"], utc=True).dt.normalize()
    out = out.sort_values(["ts_code", "asof"], kind="mergesort").reset_index(drop=True)
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(panel["event_time"], utc=True).dt.normalize().drop_duplicates()))
    session_pos = {day: index for index, day in enumerate(sessions)}
    out["session_pos"] = out["asof"].map(session_pos)
    if out["session_pos"].isna().any():
        raise RuntimeError("missing session_pos while building reset-reclaim features")
    out["shock_flag"] = out["ret1"].le(-0.03) & out["volume_ratio_20"].ge(1.5)
    grouped = out.groupby("ts_code", sort=False)
    out["shock_session_pos"] = grouped["session_pos"].transform(lambda s: s.where(out.loc[s.index, "shock_flag"]).ffill())
    out["shock_asof"] = grouped["asof"].transform(lambda s: s.where(out.loc[s.index, "shock_flag"]).ffill())
    out["shock_ret1"] = grouped["ret1"].transform(lambda s: s.where(out.loc[s.index, "shock_flag"]).ffill())
    out["shock_volume_ratio_20"] = grouped["volume_ratio_20"].transform(
        lambda s: s.where(out.loc[s.index, "shock_flag"]).ffill()
    )
    out["shock_age_sessions"] = out["session_pos"] - out["shock_session_pos"]
    out["recent_shock_present"] = out["shock_age_sessions"].between(1, 5, inclusive="both")
    out["reclaim_ready"] = (
        out["close"].ge(out["ma5"])
        & out["close"].ge(out["prev3_high"])
        & out["ret1"].gt(0)
    )
    out["position_ok"] = (
        out["dist_ma60"].between(-0.05, 0.15, inclusive="both")
        & out["dd20"].ge(-0.15)
    )
    out["low_overheat_score"] = _rank(-out["dist_ma60"])
    out["low_volatility_score"] = _rank(-out["vol20"])
    out["selection_score"] = 0.55 * out["low_overheat_score"] + 0.45 * out["low_volatility_score"]
    return out


def _candidate_pool(scored: pd.DataFrame) -> pd.DataFrame:
    required = {
        "recent_shock_present", "reclaim_ready", "position_ok", "selection_score",
        "shock_asof", "shock_ret1", "shock_volume_ratio_20", "shock_age_sessions",
    }
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"reset_reclaim frame missing columns: {sorted(missing)}")
    return scored[
        scored["recent_shock_present"]
        & scored["reclaim_ready"]
        & scored["position_ok"]
    ].copy()


def _select_top4(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        columns = list(frame.columns) + [
            "candidate_rank", "selected", "selection_decision_id", "policy_id", "target_weight", "context_hash",
        ]
        return pd.DataFrame(columns=columns)
    out = frame.sort_values(
        ["asof", "selection_score", "ts_code"], ascending=[True, False, True], kind="mergesort",
    ).groupby("asof", sort=True).head(4).copy()
    out["candidate_rank"] = out.groupby("asof").cumcount() + 1
    out["selected"] = True
    out["selection_decision_id"] = "reset_reclaim_v1-" + out["asof"].dt.strftime("%Y%m%d")
    out["policy_id"] = "selection.reset_reclaim.v1"
    out["target_weight"] = 0.12
    out["context_hash"] = out["asof"].map(
        lambda day: hashlib.sha256(f"selection.reset_reclaim.v1|{day}".encode()).hexdigest()
    )
    return out


def _mature_labels(panel: pd.DataFrame, end: str) -> pd.DataFrame:
    sessions = pd.DatetimeIndex(sorted(panel["event_time"].drop_duplicates()))
    labels = build_q70_t10_labels(panel, profile=LABEL_PROFILE, session_dates=sessions)
    labels = labels.rename(columns={"event_time": "asof", "available_time": "label_available_time"})
    labels["asof"] = pd.to_datetime(labels["asof"], utc=True).dt.normalize()
    labels["label_available_time"] = pd.to_datetime(labels["label_available_time"], utc=True)
    return labels[
        labels["label_available_time"] <= session_close(pd.Timestamp(end, tz="UTC"))
    ][["asof", "ts_code", "label_return", "label_available_time", "exit_time"]]


def _win_loss_features(candidates: pd.DataFrame) -> pd.DataFrame:
    features = [
        "selection_score", "low_overheat_score", "low_volatility_score", "dist_ma60",
        "vol20", "ret1", "ret20", "ret60", "dd20", "dd60", "volume_ratio_20",
        "shock_ret1", "shock_volume_ratio_20", "shock_age_sessions",
    ]
    rows: list[dict[str, object]] = []
    periods = {"all": candidates}
    periods.update({month: group for month, group in candidates.groupby(candidates["asof"].dt.strftime("%Y-%m"))})
    for period, frame in periods.items():
        winners = frame[frame["label_return"] > 0]
        losers = frame[frame["label_return"] <= 0]
        for feature in features:
            winner_median = float(winners[feature].median()) if len(winners) else None
            loser_median = float(losers[feature].median()) if len(losers) else None
            scale = float(frame[feature].std(ddof=1)) if len(frame) > 1 else None
            rows.append({
                "period": period,
                "feature": feature,
                "winner_n": len(winners),
                "loser_n": len(losers),
                "winner_median": winner_median,
                "loser_median": loser_median,
                "winner_minus_loser": (
                    float(winner_median - loser_median)
                    if winner_median is not None and loser_median is not None else None
                ),
                "standardized_median_difference": (
                    float((winner_median - loser_median) / scale)
                    if winner_median is not None and loser_median is not None and scale and scale > 0 else None
                ),
            })
    return pd.DataFrame(rows)


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
    metrics = summarize_backtest(result["nav"], result["trades"], initial_cash=1_000_000.0)
    metrics["slippage_each_side"] = slippage
    metrics["forced_final_liquidation_count"] = int(
        (result["trades"].get("reason", pd.Series(dtype=str)) == "end_of_test_liquidation").sum()
    )
    return result, metrics


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
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(top4)),
        "mature_through": str(mature_candidates["asof"].max().date()) if len(mature_candidates) else None,
    }
    by_year: dict[str, object] = {}
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
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    codes = _load_codes(args.codes_source)
    raw_start = (pd.Timestamp(args.start) - pd.Timedelta(days=140)).strftime("%Y-%m-%d")
    st_keys, st_audit = _load_pit_st_keys(codes, args.start, args.end)
    panel, prices, source_audit = _load_sources(raw_start, args.end, codes)
    features = _features(panel, prices, raw_start)
    features["asof"] = pd.to_datetime(features["asof"], utc=True).dt.normalize()
    features["available_time"] = pd.to_datetime(features["available_time"], utc=True)
    features = _add_reset_reclaim_features(features, panel)
    features = features[features["asof"].between(args.start, args.end)].copy()
    if features.empty:
        raise RuntimeError("feature ledger is empty for requested range")
    if features["asof"].max() < pd.Timestamp(args.end, tz="UTC"):
        raise RuntimeError(f"feature coverage stops at {features['asof'].max() if len(features) else None}, before requested end {args.end}")
    if (features["available_time"] > features["asof"].map(session_close)).any():
        raise AssertionError("feature PIT violation")

    features["pit_st"] = [(day, code) in st_keys for day, code in zip(features["asof"], features["ts_code"].astype(str))]
    scored_all = features.copy()
    candidate_indices = set(_candidate_pool(features[~features["pit_st"]].copy()).index)
    candidates = scored_all.loc[sorted(candidate_indices)].copy()
    top4 = _select_top4(candidates)
    labels = _mature_labels(panel, args.end)

    scored_all["score"] = scored_all["selection_score"]
    scored_all["policy_id"] = "selection.reset_reclaim.v1"
    scored_all["candidate_status"] = ["CANDIDATE" if idx in candidate_indices else "REJECTED" for idx in scored_all.index]
    scored_all["rejection_reason"] = ""
    scored_all.loc[~scored_all["recent_shock_present"], "rejection_reason"] = "no_recent_high_turnover_drop"
    scored_all.loc[scored_all["recent_shock_present"] & ~scored_all["reclaim_ready"], "rejection_reason"] = "reclaim_not_confirmed"
    scored_all.loc[scored_all["recent_shock_present"] & scored_all["reclaim_ready"] & ~scored_all["position_ok"], "rejection_reason"] = "position_band_failed"
    scored_all.loc[scored_all["pit_st"], "rejection_reason"] = "pit_st"
    scored_all["context_hash"] = scored_all["asof"].map(
        lambda day: hashlib.sha256(f"selection.reset_reclaim.v1|{day}".encode()).hexdigest()
    )

    mature_candidates = candidates.merge(labels, on=["asof", "ts_code"], how="left", validate="one_to_one")
    win_loss = _win_loss_features(mature_candidates.dropna(subset=["label_return"]).copy())

    actions = load_corporate_actions(args.start, args.end, ts_codes=codes)
    bt_prices = prices.copy()
    portfolios: dict[str, object] = {}
    for cost, slippage in (("base", 0.001), ("stress", 0.003)):
        result, metrics = _backtest(top4, bt_prices, actions, slippage=slippage)
        target = output / "backtests" / cost
        target.mkdir(parents=True, exist_ok=True)
        for artifact in ("orders", "trades", "nav", "positions", "corporate_actions"):
            result[artifact].to_csv(target / f"{artifact}.csv", index=False)
        _write_json(target / "metrics.json", metrics)
        portfolios[cost] = metrics

    summary = _diagnostic_summary(scored_all[~scored_all["pit_st"]].copy(), candidates, top4, labels)
    _write_json(output / "SUMMARY.json", {
        "experiment_id": "reset_reclaim_v1",
        "status": "historical_diagnostic_not_lockbox",
        "events": summary,
        "portfolios": portfolios,
    })
    scored_all.to_parquet(output / "score_ledger.parquet", index=False)
    candidates.to_parquet(output / "candidate_ledger.parquet", index=False)
    top4.to_csv(output / "selection_ledger.csv", index=False)
    mature_candidates.to_parquet(output / "mature_candidate_ledger.parquet", index=False)
    win_loss.to_csv(output / "winner_loser_feature_comparison.csv", index=False)
    _write_json(output / "DATA_MANIFEST.json", {
        "experiment_id": "reset_reclaim_v1",
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
        "selection_rule": "past5 high-turnover drop + same-day right reclaim + low-overheat/low-volatility top4",
    })
    lines = [
        "# Reset Reclaim V1",
        "",
        "Historical diagnostic only.",
        "",
        f"- Candidate rows: {summary['candidate_rows']}",
        f"- Selected rows: {summary['selected_rows']}",
    ]
    for label, metrics in portfolios.items():
        lines.append(
            f"- {label}: return={metrics['total_return']:+.2%}, PF={metrics['portfolio_profit_factor']}, MaxDD={metrics['max_drawdown']:+.2%}, trades={metrics['trade_count']}"
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
