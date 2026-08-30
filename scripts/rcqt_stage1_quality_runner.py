"""Audit the frozen Stage-1 RCQT candidate rule without training a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.data.execution_source import load_execution_panel
from aistock9988.data.q70_source import load_f0_panel
from aistock9988.data.quantdb import readonly_connection
from aistock9988.labeling.maturity import LabelProfile
from aistock9988.labeling.q70 import build_q70_t10_labels
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.selection.rcqt import score_rcqt
from aistock9988.time.session import session_close

from rcqt_quantdb_sample_runner import _features


LABEL_PROFILE = LabelProfile("label.endpoint_open_open_t10.v1", 1, 10, 11)
FEATURES = (
    "dist_ma60", "ret1", "ret20", "ret60", "dd20", "dd60", "vol20",
    "liq20", "volume_ratio_20", "kdj_k_bfq", "cci_bfq", "wr_bfq",
    "confirmation_strength", "quiet_score",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


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


def _load_sources(start: str, end: str, codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel_parts: list[pd.DataFrame] = []
    price_parts: list[pd.DataFrame] = []
    audits: list[dict] = []
    chunks = _date_chunks(start, end)
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"source_chunk {index}/{len(chunks)} start={chunk_start} end={chunk_end}", flush=True)
        panel, audit = load_f0_panel(chunk_start, chunk_end, ts_codes=codes, return_audit=True)
        prices = load_execution_panel(chunk_start, chunk_end, ts_codes=codes)
        panel_parts.append(panel)
        price_parts.append(prices)
        audits.append(audit)
    panel = pd.concat(panel_parts, ignore_index=True).sort_values(
        ["event_time", "ts_code"], kind="mergesort",
    ).reset_index(drop=True)
    prices = pd.concat(price_parts, ignore_index=True).sort_values(
        ["trade_date", "ts_code"], kind="mergesort",
    ).reset_index(drop=True)
    if panel.duplicated(["event_time", "ts_code"]).any():
        raise ValueError("feature source contains duplicate event_time/ts_code")
    if prices.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("execution source contains duplicate trade_date/ts_code")
    return panel, prices, {
        "source_id": "quant_db",
        "chunks": chunks,
        "membership_rows_loaded_sum": sum(int(a.get("membership_rows_loaded", 0)) for a in audits),
        "industry_resolution_dates": sum(len(a.get("industry_resolution", [])) for a in audits),
    }


def _load_pit_st_keys(codes: list[str], start: str, end: str) -> tuple[set[tuple[pd.Timestamp, str]], dict]:
    placeholders = ",".join(["%s"] * len(codes))
    with readonly_connection() as connection:
        coverage = pd.read_sql_query(
            "SELECT MIN(trade_date) min_date, MAX(trade_date) max_date, COUNT(*) rows_n "
            "FROM stock_st_ts WHERE trade_date >= %s AND trade_date <= %s",
            connection, params=(start, end),
        ).iloc[0]
        if pd.isna(coverage["max_date"]) or pd.Timestamp(coverage["max_date"]) < pd.Timestamp(end):
            raise RuntimeError(f"stock_st_ts does not cover requested end {end}; coverage={coverage.to_dict()}")
        frame = pd.read_sql_query(
            f"SELECT trade_date, ts_code FROM stock_st_ts WHERE trade_date >= %s AND trade_date <= %s "
            f"AND ts_code IN ({placeholders}) ORDER BY trade_date, ts_code",
            connection, params=(start, end, *codes),
        )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], utc=True).dt.normalize()
    keys = set(zip(frame["trade_date"], frame["ts_code"].astype(str)))
    return keys, {
        "source": "quant_db.stock_st_ts",
        "coverage_min": str(coverage["min_date"]),
        "coverage_max": str(coverage["max_date"]),
        "rows": len(frame),
        "policy": "exclude exact PIT trade_date/ts_code keys",
    }


def _metrics(frame: pd.DataFrame) -> dict[str, object]:
    values = pd.to_numeric(frame["label_return"], errors="raise")
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return {
        "rows": len(frame),
        "dates": int(frame["asof"].nunique()),
        "mean_return": float(values.mean()),
        "median_return": float(values.median()),
        "win_rate": float((values > 0).mean()),
        "profit_factor": gains / losses if losses else None,
        "down_8pct_rate": float((values <= -0.08).mean()),
        "up_10pct_rate": float((values >= 0.10).mean()),
    }


def _uplift(control: pd.DataFrame, treatment: pd.DataFrame) -> dict[str, object]:
    base = _metrics(control)
    trial = _metrics(treatment)
    down = float(base["down_8pct_rate"])
    return {
        "mean_return_lift": float(trial["mean_return"]) - float(base["mean_return"]),
        "win_rate_lift": float(trial["win_rate"]) - float(base["win_rate"]),
        "up_10pct_rate_lift": float(trial["up_10pct_rate"]) - float(base["up_10pct_rate"]),
        "down_8pct_rate_change": float(trial["down_8pct_rate"]) - down,
        "down_8pct_compression_ratio": float(trial["down_8pct_rate"]) / down if down else None,
    }


def _rank_skill(frame: pd.DataFrame, score: str) -> dict[str, object]:
    daily = frame.groupby("asof")[[score, "label_return"]].apply(
        lambda group: group[score].corr(group["label_return"], method="spearman")
    ).dropna()
    return {
        "dates": len(daily),
        "mean_daily_rank_ic": float(daily.mean()),
        "median_daily_rank_ic": float(daily.median()),
        "positive_rank_ic_ratio": float((daily > 0).mean()),
    }


def _select_top4(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.sort_values(
        ["asof", "quiet_score", "ts_code"], ascending=[True, False, True], kind="mergesort",
    ).groupby("asof", sort=True).head(4).copy()
    out["candidate_rank"] = out.groupby("asof").cumcount() + 1
    out["selected"] = True
    out["selection_decision_id"] = "rcqt-stage1-rule-" + out["asof"].dt.strftime("%Y%m%d")
    out["policy_id"] = "rcqt.stage1.quiet_confirmed.rule_top4.v1"
    out["target_weight"] = 0.12
    return out


def _win_loss_features(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    periods = {"all": candidates}
    periods.update({month: group for month, group in candidates.groupby(candidates["asof"].dt.strftime("%Y-%m"))})
    for period, frame in periods.items():
        winners = frame[frame["label_return"] > 0]
        losers = frame[frame["label_return"] <= 0]
        for feature in FEATURES:
            winner_median = float(winners[feature].median())
            loser_median = float(losers[feature].median())
            scale = float(frame[feature].std(ddof=1))
            rows.append({
                "period": period,
                "feature": feature,
                "winner_n": len(winners),
                "loser_n": len(losers),
                "winner_median": winner_median,
                "loser_median": loser_median,
                "winner_minus_loser": winner_median - loser_median,
                "standardized_median_difference": (winner_median - loser_median) / scale if scale > 0 else None,
            })
    result = pd.DataFrame(rows)
    monthly = result[result["period"] != "all"].copy()
    direction = monthly.assign(sign=np.sign(monthly["winner_minus_loser"])).groupby("feature").agg(
        monthly_periods=("period", "nunique"),
        positive_month_ratio=("sign", lambda value: float((value > 0).mean())),
        negative_month_ratio=("sign", lambda value: float((value < 0).mean())),
    ).reset_index()
    return result.merge(direction, on="feature", how="left")


def _coverage(scored: pd.DataFrame, candidates: pd.DataFrame, top4: pd.DataFrame) -> pd.DataFrame:
    universe = scored.groupby("asof").size().rename("universe_rows")
    candidate = candidates.groupby("asof").size().rename("candidate_rows")
    selected = top4.groupby("asof").size().rename("selected_rows")
    out = pd.concat([universe, candidate, selected], axis=1).fillna(0).reset_index()
    out["candidate_rate"] = out["candidate_rows"] / out["universe_rows"]
    return out


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

    codes = sorted(pd.read_parquet(args.codes_source, columns=["ts_code"])["ts_code"].astype(str).unique())
    raw_start = (pd.Timestamp(args.start) - pd.Timedelta(days=140)).strftime("%Y-%m-%d")
    st_keys, st_audit = _load_pit_st_keys(codes, args.start, args.end)
    panel, prices, source_audit = _load_sources(raw_start, args.end, codes)
    features = _features(panel, prices, args.start)
    features["asof"] = pd.to_datetime(features["asof"], utc=True).dt.normalize()
    features["available_time"] = pd.to_datetime(features["available_time"], utc=True)
    features = features[features["asof"].between(args.start, args.end)].copy()
    if features.empty or features["asof"].max() < pd.Timestamp(args.end, tz="UTC"):
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
    candidates = scored[scored["quiet_eligible"] & scored["right_confirmed"]].copy()
    top4 = _select_top4(candidates)
    if candidates.empty or top4.empty:
        raise RuntimeError("Stage-1 candidate or Top4 ledger is empty")

    sessions = pd.DatetimeIndex(sorted(panel["event_time"].drop_duplicates()))
    labels = build_q70_t10_labels(panel, profile=LABEL_PROFILE, session_dates=sessions)
    labels = labels.rename(columns={"event_time": "asof", "available_time": "label_available_time"})
    labels["asof"] = pd.to_datetime(labels["asof"], utc=True).dt.normalize()
    labels["label_available_time"] = pd.to_datetime(labels["label_available_time"], utc=True)
    mature_labels = labels[
        labels["label_available_time"] <= session_close(pd.Timestamp(args.end))
    ][["asof", "ts_code", "label_return", "label_available_time", "exit_time"]]
    mature_scored = scored.merge(mature_labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
    mature_candidates = candidates.merge(
        mature_labels, on=["asof", "ts_code"], how="inner", validate="one_to_one",
    )
    mature_top4 = top4.merge(mature_labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
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
            "rule_rank_skill": _rank_skill(pool, "quiet_score"),
        }

    event_summary = {
        "candidate_contract": "quiet_eligible AND right_confirmed AND NOT PIT-ST",
        "mature_through_signal_date": str(mature_signal_end.date()),
        "universe_control": _metrics(mature_scored),
        "candidate_pool": _metrics(mature_candidates),
        "rule_top4": _metrics(mature_top4),
        "candidate_vs_universe": _uplift(mature_scored, mature_candidates),
        "top4_vs_candidate": _uplift(mature_candidates, mature_top4),
        "rule_rank_skill": _rank_skill(mature_candidates, "quiet_score"),
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
    signals = top4[top4["asof"] <= mature_signal_end].copy()
    portfolio: dict[str, object] = {}
    for label, slippage in (("base", 0.001), ("stress", 0.003)):
        result, metrics = _backtest(signals, px, actions, slippage=slippage)
        target = output / "backtests" / label
        target.mkdir(parents=True, exist_ok=True)
        for name in ("orders", "trades", "nav", "positions", "corporate_actions"):
            result[name].to_csv(target / f"{name}.csv", index=False)
        _write_json(target / "metrics.json", metrics)
        portfolio[label] = metrics

    features.to_parquet(output / "feature_ledger.parquet", index=False)
    scored.to_parquet(output / "score_ledger.parquet", index=False)
    candidates.to_parquet(output / "candidate_ledger.parquet", index=False)
    top4.to_csv(output / "rule_top4_selection_ledger.csv", index=False)
    mature_candidates.to_parquet(output / "mature_candidate_event_ledger.parquet", index=False)
    mature_top4.to_csv(output / "mature_rule_top4_event_ledger.csv", index=False)
    coverage.to_csv(output / "daily_coverage.csv", index=False)
    win_loss.to_csv(output / "winner_loser_feature_comparison.csv", index=False)
    _write_json(output / "EVENT_SUMMARY.json", event_summary)
    _write_json(output / "PORTFOLIO_SUMMARY.json", portfolio)

    manifest = {
        "kind": "stage1_fixed_rule_quality_audit",
        "requested_start": args.start,
        "requested_end": args.end,
        "raw_lookback_start": raw_start,
        "codes_source": str(args.codes_source.resolve()),
        "codes_source_sha256": _sha(args.codes_source),
        "fixed_universe_codes": len(codes),
        "candidate_contract": event_summary["candidate_contract"],
        "selection_contract": "daily Top4 by frozen quiet_score",
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

    base = portfolio["base"]
    candidate = event_summary["candidate_pool"]
    top = event_summary["rule_top4"]
    control = event_summary["universe_control"]
    lines = [
        "# S16 Stage-1 fixed-rule quality audit", "",
        "No XGBoost model was trained or used. The frozen candidate contract is "
        "`quiet_eligible AND right_confirmed AND NOT PIT-ST`; Top4 is ordered only by `quiet_score`.", "",
        "## 2026 mature event evidence", "",
        "| Cohort | Mean T+10 | PF | Win rate | <=-8% | >=+10% | Rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Fixed universe | {control['mean_return']:+.2%} | {control['profit_factor']:.3f} | {control['win_rate']:.1%} | {control['down_8pct_rate']:.1%} | {control['up_10pct_rate']:.1%} | {control['rows']} |",
        f"| Stage-1 candidates | {candidate['mean_return']:+.2%} | {candidate['profit_factor']:.3f} | {candidate['win_rate']:.1%} | {candidate['down_8pct_rate']:.1%} | {candidate['up_10pct_rate']:.1%} | {candidate['rows']} |",
        f"| Rule Top4 | {top['mean_return']:+.2%} | {top['profit_factor']:.3f} | {top['win_rate']:.1%} | {top['down_8pct_rate']:.1%} | {top['up_10pct_rate']:.1%} | {top['rows']} |",
        "", "## Executable rolling backtest", "",
        f"- Mature signals through `{mature_signal_end.date()}`; 0.1% each-side slippage.",
        f"- Return `{base['total_return']:+.2%}`, PF `{base['portfolio_profit_factor']:.3f}`, MaxDD `{base['max_drawdown']:.2%}`.",
        f"- Excluding best week `{base['return_excluding_best_week']:+.2%}`; excluding top three profitable trades `{base['trade_return_excluding_top3_profit']:+.2%}`.",
        "", "## Decision rule", "",
        "Stage 1 advances only if candidate quality is positive and stable by month, the candidate pool improves both mean return and left-tail rate versus the fixed universe, and the executable portfolio meets PF>=2, MaxDD<=15%, and positive return excluding its best week. Retrospective feature differences are diagnosis only and are not converted into gates.",
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
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
