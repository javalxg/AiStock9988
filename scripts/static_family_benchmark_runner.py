"""Static family benchmark runner for the trend-continuation B family.

This is a fixed-rule, fully auditable stage-1 style experiment.  It does not
train a model; it selects a daily Top4 candidate basket using only T-close
information, then replays the basket through the shared event-driven backtest
engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.labeling.q70 import build_q70_t10_labels
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.selection.rcqt import score_rcqt
from aistock9988.time.session import session_close

from rcqt_quantdb_sample_runner import _features
from rcqt_stage1_quality_runner import (
    LABEL_PROFILE,
    _load_pit_st_keys,
    _metrics,
    _rank_skill,
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


def _add_relative_features(features: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["asof"] = pd.to_datetime(out["asof"], utc=True).dt.normalize()
    industry = panel[["event_time", "ts_code", "industry"]].copy()
    industry["asof"] = pd.to_datetime(industry.pop("event_time"), utc=True).dt.normalize()
    industry = industry.drop_duplicates(["asof", "ts_code"], keep="last")
    out = out.merge(industry, on=["asof", "ts_code"], how="left", validate="one_to_one")
    if out["industry"].isna().any():
        raise RuntimeError("PIT industry is missing after F0 feature construction")
    for horizon in (20, 60):
        source = f"ret{horizon}"
        market = out.groupby("asof", sort=False)[source].transform("median")
        sector = out.groupby(["asof", "industry"], sort=False)[source].transform("median")
        out[f"market_excess_ret{horizon}"] = out[source] - market
        out[f"industry_excess_ret{horizon}"] = out[source] - sector
    return out


def _add_macd_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out = out.sort_values(["ts_code", "asof"], kind="mergesort").reset_index(drop=True)
    grouped = out.groupby("ts_code", group_keys=False)["close"]
    ema12 = grouped.transform(lambda series: series.ewm(span=12, adjust=False, min_periods=12).mean())
    ema26 = grouped.transform(lambda series: series.ewm(span=26, adjust=False, min_periods=26).mean())
    out["macd_dif"] = ema12 - ema26
    out["macd_dea"] = out.groupby("ts_code", group_keys=False)["macd_dif"].transform(
        lambda series: series.ewm(span=9, adjust=False, min_periods=9).mean()
    )
    out["macd_hist"] = out["macd_dif"] - out["macd_dea"]
    out["macd_bullish"] = out["macd_dif"] > out["macd_dea"]
    return out


def _trend_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    required = {
        "right_confirmed", "market_excess_ret20", "industry_excess_ret20",
        "market_excess_ret60", "industry_excess_ret60", "macd_bullish",
        "macd_hist", "volume_ratio_20", "dist_ma60", "ret20", "ret60", "dd20",
    }
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"trend family frame missing columns: {sorted(missing)}")
    out = scored[
        scored["right_confirmed"]
        & scored["macd_bullish"]
        & scored["market_excess_ret20"].gt(0)
        & scored["industry_excess_ret20"].gt(0)
        & scored["market_excess_ret60"].gt(0)
        & scored["industry_excess_ret60"].gt(0)
        & scored["dist_ma60"].between(0.00, 0.15, inclusive="both")
        & scored["ret20"].between(0.00, 0.25, inclusive="both")
        & scored["ret60"].between(0.00, 0.35, inclusive="both")
        & scored["dd20"].ge(-0.10)
        & scored["volume_ratio_20"].between(0.80, 2.00, inclusive="both")
    ].copy()
    if out.empty:
        return out
    out["trend_score"] = (
        0.25 * _rank(out["market_excess_ret20"])
        + 0.25 * _rank(out["industry_excess_ret20"])
        + 0.20 * _rank(out["confirmation_strength"])
        + 0.15 * _rank(out["macd_hist"])
        + 0.10 * _rank(-abs(out["volume_ratio_20"] - 1.0))
        + 0.05 * _rank(-out["dd20"])
    )
    return out


def _select_top4(frame: pd.DataFrame, *, policy_id: str) -> pd.DataFrame:
    if frame.empty:
        columns = list(frame.columns) + [
            "candidate_rank", "selected", "selection_decision_id", "policy_id",
            "target_weight", "context_hash", "sleeve",
        ]
        return pd.DataFrame(columns=columns)
    out = frame.sort_values(
        ["asof", "trend_score", "ts_code"], ascending=[True, False, True], kind="mergesort",
    ).groupby("asof", sort=True).head(4).copy()
    out["candidate_rank"] = out.groupby("asof").cumcount() + 1
    out["selected"] = True
    out["selection_decision_id"] = policy_id + "-" + out["asof"].dt.strftime("%Y%m%d")
    out["policy_id"] = policy_id
    out["target_weight"] = 0.12
    out["sleeve"] = "trend_continuation"
    out["context_hash"] = out["asof"].map(
        lambda day: hashlib.sha256(f"{policy_id}|{day}".encode()).hexdigest()
    )
    return out


def _coverage(scored: pd.DataFrame, candidates: pd.DataFrame, top4: pd.DataFrame) -> pd.DataFrame:
    universe = scored.groupby("asof").size().rename("universe_rows")
    candidate = candidates.groupby("asof").size().rename("candidate_rows")
    selected = top4.groupby("asof").size().rename("selected_rows")
    out = pd.concat([universe, candidate, selected], axis=1).fillna(0).reset_index()
    out["candidate_rate"] = out["candidate_rows"] / out["universe_rows"]
    return out


def _win_loss_features(candidates: pd.DataFrame) -> pd.DataFrame:
    features = [
        "trend_score", "macd_dif", "macd_dea", "macd_hist", "dist_ma60", "ret20",
        "ret60", "dd20", "dd60", "vol20", "liq20", "volume_ratio_20",
        "market_excess_ret20", "industry_excess_ret20",
        "market_excess_ret60", "industry_excess_ret60", "confirmation_strength",
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
    result = pd.DataFrame(rows)
    monthly = result[result["period"] != "all"].copy()
    direction = monthly.assign(sign=lambda frame: frame["winner_minus_loser"].map(lambda value: 1 if value is not None and value > 0 else -1 if value is not None and value < 0 else 0)).groupby("feature").agg(
        monthly_periods=("period", "nunique"),
        positive_month_ratio=("sign", lambda value: float((value > 0).mean())),
        negative_month_ratio=("sign", lambda value: float((value < 0).mean())),
    ).reset_index()
    return result.merge(direction, on="feature", how="left")


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
            max_order_to_adv20=0.02,
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


def _mature_labels(panel: pd.DataFrame, end: str) -> pd.DataFrame:
    sessions = pd.DatetimeIndex(sorted(panel["event_time"].drop_duplicates()))
    labels = build_q70_t10_labels(panel, profile=LABEL_PROFILE, session_dates=sessions)
    labels = labels.rename(columns={"event_time": "asof", "available_time": "label_available_time"})
    labels["asof"] = pd.to_datetime(labels["asof"], utc=True).dt.normalize()
    labels["label_available_time"] = pd.to_datetime(labels["label_available_time"], utc=True)
    return labels[
        labels["label_available_time"] <= session_close(pd.Timestamp(end, tz="UTC"))
    ][["asof", "ts_code", "label_return", "label_available_time", "exit_time"]]


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    codes = _load_codes(args.codes_source)
    raw_start = (pd.Timestamp(args.start) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    st_keys, st_audit = _load_pit_st_keys(codes, args.start, args.end)
    panel, prices, source_audit = _load_sources(raw_start, args.end, codes)
    features = _features(panel, prices, raw_start)
    features = _add_relative_features(features, panel)
    features = _add_macd_features(features)
    features["asof"] = pd.to_datetime(features["asof"], utc=True).dt.normalize()
    features["available_time"] = pd.to_datetime(features["available_time"], utc=True)
    features = features[features["asof"].between(args.start, args.end)].copy()
    if features.empty:
        raise RuntimeError("feature ledger is empty for requested range")
    if features["asof"].max() < pd.Timestamp(args.end, tz="UTC"):
        raise RuntimeError(
            f"feature coverage stops at {features['asof'].max() if len(features) else None}, before requested end {args.end}"
        )
    if (features["available_time"] > features["asof"].map(session_close)).any():
        raise AssertionError("feature PIT violation")

    scored = score_rcqt(features)
    scored["pit_st"] = [
        (day, code) in st_keys for day, code in zip(scored["asof"], scored["ts_code"].astype(str))
    ]
    scored = scored[~scored["pit_st"]].copy()
    candidates = _trend_candidates(scored)
    top4 = _select_top4(candidates, policy_id="selection.static_family_b.trend_continuation.v1")
    if candidates.empty or top4.empty:
        raise RuntimeError("Static family B candidate or Top4 ledger is empty")

    labels = _mature_labels(panel, args.end)
    mature_scored = scored.merge(labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
    mature_candidates = candidates.merge(labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
    mature_top4 = top4.merge(labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
    mature_signal_end = mature_top4["asof"].max()

    coverage = _coverage(scored, candidates, top4)
    win_loss = _win_loss_features(mature_candidates)
    monthly: dict[str, object] = {}
    for month, pool in mature_candidates.groupby(mature_candidates["asof"].dt.strftime("%Y-%m")):
        control = mature_scored[mature_scored["asof"].dt.strftime("%Y-%m") == month]
        selected = mature_top4[mature_top4["asof"].dt.strftime("%Y-%m") == month]
        monthly[month] = {
            "universe_control": _metrics(control),
            "candidate_pool": _metrics(pool),
            "rule_top4": _metrics(selected),
            "candidate_vs_universe": _uplift(control, pool),
            "top4_vs_candidate": _uplift(pool, selected),
            "rule_rank_skill": _rank_skill(pool, "trend_score"),
        }

    event_summary = {
        "candidate_contract": (
            "right_confirmed AND macd_bullish AND relative_strength_positive "
            "AND not_overextended AND moderate_volume"
        ),
        "mature_through_signal_date": str(mature_signal_end.date()),
        "universe_control": _metrics(mature_scored),
        "candidate_pool": _metrics(mature_candidates),
        "rule_top4": _metrics(mature_top4),
        "candidate_vs_universe": _uplift(mature_scored, mature_candidates),
        "top4_vs_candidate": _uplift(mature_candidates, mature_top4),
        "rule_rank_skill": _rank_skill(mature_candidates, "trend_score"),
        "monthly": monthly,
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
    bt_prices = px.copy()
    portfolios: dict[str, object] = {}
    for label, slippage in (("base", 0.001), ("stress", 0.003)):
        result, metrics = _backtest(top4, bt_prices, actions, slippage=slippage)
        target = output / "backtests" / label
        target.mkdir(parents=True, exist_ok=True)
        for name in ("orders", "trades", "nav", "positions", "corporate_actions"):
            result[name].to_csv(target / f"{name}.csv", index=False)
        _write_json(target / "metrics.json", metrics)
        portfolios[label] = metrics

    features.to_parquet(output / "feature_ledger.parquet", index=False)
    scored.to_parquet(output / "score_ledger.parquet", index=False)
    candidates.to_parquet(output / "candidate_ledger.parquet", index=False)
    top4.to_csv(output / "selection_ledger.csv", index=False)
    mature_candidates.to_parquet(output / "mature_candidate_event_ledger.parquet", index=False)
    mature_top4.to_csv(output / "mature_selection_event_ledger.csv", index=False)
    coverage.to_csv(output / "daily_coverage.csv", index=False)
    win_loss.to_csv(output / "winner_loser_feature_comparison.csv", index=False)
    _write_json(output / "EVENT_SUMMARY.json", event_summary)
    _write_json(output / "PORTFOLIO_SUMMARY.json", portfolios)

    manifest = {
        "kind": "static_family_benchmark_v1",
        "requested_start": args.start,
        "requested_end": args.end,
        "raw_lookback_start": raw_start,
        "codes_source": str(args.codes_source.resolve()),
        "codes_source_sha256": _sha(args.codes_source),
        "fixed_universe_codes": len(codes),
        "candidate_contract": event_summary["candidate_contract"],
        "selection_contract": "daily Top4 by frozen trend_score",
        "execution_contract": "T close signal; T+1 open; H10; -8% close-trigger/next-open stop",
        "parameter_sweep": False,
        "model_training": False,
        "pit_st_audit": st_audit,
        "source_audit": source_audit,
        "feature_rows": len(features),
        "candidate_rows": len(candidates),
        "mature_candidate_rows": len(mature_candidates),
        "mature_signal_end": str(mature_signal_end),
        "credentials_persisted": False,
    }
    _write_json(output / "DATA_MANIFEST.json", manifest)

    base = portfolios["base"]
    control = event_summary["universe_control"]
    candidate = event_summary["candidate_pool"]
    top = event_summary["rule_top4"]
    lines = [
        "# Static Family Benchmark V1",
        "",
        "This is a fixed-rule stage-1 style benchmark for the trend-continuation family.",
        "No XGBoost model was trained or used.",
        "",
        "## 2026 mature event evidence",
        "",
        "| Cohort | Mean T+10 | PF | Win rate | <=-8% | >=+10% | Rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Fixed universe | {control['mean_return']:+.2%} | {control['profit_factor']:.3f} | {control['win_rate']:.1%} | {control['down_8pct_rate']:.1%} | {control['up_10pct_rate']:.1%} | {control['rows']} |",
        f"| B-family candidates | {candidate['mean_return']:+.2%} | {candidate['profit_factor']:.3f} | {candidate['win_rate']:.1%} | {candidate['down_8pct_rate']:.1%} | {candidate['up_10pct_rate']:.1%} | {candidate['rows']} |",
        f"| Rule Top4 | {top['mean_return']:+.2%} | {top['profit_factor']:.3f} | {top['win_rate']:.1%} | {top['down_8pct_rate']:.1%} | {top['up_10pct_rate']:.1%} | {top['rows']} |",
        "",
        "## Executable rolling backtest",
        "",
        f"- Mature signals through `{mature_signal_end.date()}`; 0.1% each-side slippage.",
        f"- Return `{base['total_return']:+.2%}`, PF `{base['portfolio_profit_factor']:.3f}`, MaxDD `{base['max_drawdown']:.2%}`.",
        f"- Excluding best week `{base['return_excluding_best_week']:+.2%}`; excluding top three profitable trades `{base['trade_return_excluding_top3_profit']:+.2%}`.",
        "",
        "## Decision rule",
        "",
        "This benchmark advances only if candidate quality is positive and stable by month, the candidate pool improves both mean return and left-tail rate versus the fixed universe, and the executable portfolio meets PF>=2, MaxDD<=15%, and positive return excluding its best week.",
    ]
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
    parser.add_argument("--end", default="2026-08-21")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
