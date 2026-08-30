"""Run the preregistered transparent Stage-1 relative-continuation experiment."""
from __future__ import annotations

import argparse
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


def _load_sources(start: str, end: str, codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel_parts: list[pd.DataFrame] = []
    price_parts: list[pd.DataFrame] = []
    audit_parts: list[dict] = []
    chunks = _date_chunks(start, end)
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"source_chunk {index}/{len(chunks)} start={chunk_start} end={chunk_end}", flush=True)
        panel, audit = load_f0_panel(
            chunk_start,
            chunk_end,
            ts_codes=codes,
            include_industry=True,
            return_audit=True,
        )
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
        raise ValueError("feature source contains duplicate event_time/ts_code")
    if prices.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("execution source contains duplicate trade_date/ts_code")
    return panel, prices, {
        "source_id": "quant_db",
        "chunks": chunks,
        "include_industry": True,
        "membership_rows_loaded_sum": sum(
            int(part.get("membership_rows_loaded", 0)) for part in audit_parts
        ),
        "industry_resolution_dates": sum(
            len(part.get("industry_resolution", [])) for part in audit_parts
        ),
    }


def _add_relative_features(features: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    industry = panel[["event_time", "ts_code", "industry"]].copy()
    industry["asof"] = pd.to_datetime(industry.pop("event_time"), utc=True).dt.normalize()
    industry = industry.drop_duplicates(["asof", "ts_code"], keep="last")
    out = features.merge(industry, on=["asof", "ts_code"], how="left", validate="one_to_one")
    if out["industry"].isna().any():
        raise RuntimeError("PIT industry is missing after F0 feature construction")
    for horizon in (20, 60):
        source = f"ret{horizon}"
        market = out.groupby("asof", sort=False)[source].transform("median")
        sector = out.groupby(["asof", "industry"], sort=False)[source].transform("median")
        out[f"market_excess_ret{horizon}"] = out[source] - market
        out[f"industry_excess_ret{horizon}"] = out[source] - sector
    return out


def _relative_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    return scored[
        scored["right_confirmed"]
        & scored["market_excess_ret20"].gt(0)
        & scored["market_excess_ret60"].gt(0)
        & scored["industry_excess_ret20"].gt(0)
        & scored["industry_excess_ret60"].gt(0)
        & scored["dist_ma60"].gt(0)
        & scored["dist_ma60"].le(0.15)
        & scored["dd20"].ge(-0.10)
        & scored["ret20"].le(0.25)
        & scored["ret60"].le(0.35)
    ].copy()


def _select_top4(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    out = frame.sort_values(
        ["asof", "quiet_score", "ts_code"], ascending=[True, False, True], kind="mergesort",
    ).groupby("asof", sort=True).head(4).copy()
    out["candidate_rank"] = out.groupby("asof").cumcount() + 1
    out["selected"] = True
    out["selection_decision_id"] = policy + "-" + out["asof"].dt.strftime("%Y%m%d")
    out["policy_id"] = policy
    out["target_weight"] = 0.12
    return out


def _split_summary(control: pd.DataFrame, current: pd.DataFrame, challenger: pd.DataFrame,
                   current_top4: pd.DataFrame, challenger_top4: pd.DataFrame) -> dict[str, object]:
    monthly: dict[str, object] = {}
    for month in sorted(challenger["asof"].dt.strftime("%Y-%m").unique()):
        month_control = control[control["asof"].dt.strftime("%Y-%m") == month]
        month_current = current[current["asof"].dt.strftime("%Y-%m") == month]
        month_challenger = challenger[challenger["asof"].dt.strftime("%Y-%m") == month]
        monthly[month] = {
            "universe": _metrics(month_control),
            "current_quiet_confirmed": _metrics(month_current),
            "relative_orderly_continuation": _metrics(month_challenger),
            "challenger_vs_universe": _uplift(month_control, month_challenger),
            "challenger_vs_current": _uplift(month_current, month_challenger),
        }
    return {
        "universe": _metrics(control),
        "current_quiet_confirmed": _metrics(current),
        "relative_orderly_continuation": _metrics(challenger),
        "challenger_vs_universe": _uplift(control, challenger),
        "challenger_vs_current": _uplift(current, challenger),
        "current_top4": _metrics(current_top4),
        "challenger_top4": _metrics(challenger_top4),
        "current_rank_skill": _rank_skill(current, "quiet_score"),
        "challenger_rank_skill": _rank_skill(challenger, "quiet_score"),
        "monthly": monthly,
    }


def _validation_pass(event: dict[str, object], portfolio: dict[str, object]) -> dict[str, bool]:
    candidate = event["relative_orderly_continuation"]
    versus = event["challenger_vs_universe"]
    months = event["monthly"]
    complete_months = [value["relative_orderly_continuation"] for value in months.values()]
    monthly_majority = sum(float(value["profit_factor"]) > 1 for value in complete_months) > len(complete_months) / 2
    return {
        "candidate_pf_ge_1_20": float(candidate["profit_factor"]) >= 1.20,
        "candidate_mean_lift_vs_universe_positive": float(versus["mean_return_lift"]) > 0,
        "candidate_down8_change_vs_universe_negative": float(versus["down_8pct_rate_change"]) < 0,
        "majority_months_pf_above_one": monthly_majority,
        "portfolio_pf_ge_2": float(portfolio["portfolio_profit_factor"]) >= 2,
        "portfolio_maxdd_le_15pct": abs(float(portfolio["max_drawdown"])) <= 0.15,
        "portfolio_ex_best_week_positive": float(portfolio["return_excluding_best_week"]) > 0,
    }


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    codes = sorted(pd.read_parquet(args.codes_source, columns=["ts_code"])["ts_code"].astype(str).unique())
    signal_start = "2024-01-01"
    signal_end = "2025-12-31"
    source_end = "2026-01-20"
    raw_start = (pd.Timestamp(signal_start) - pd.Timedelta(days=140)).strftime("%Y-%m-%d")
    st_keys, st_audit = _load_pit_st_keys(codes, signal_start, signal_end)
    panel, prices, source_audit = _load_sources(raw_start, source_end, codes)
    features = _features(panel, prices, signal_start)
    features["asof"] = pd.to_datetime(features["asof"], utc=True).dt.normalize()
    features["available_time"] = pd.to_datetime(features["available_time"], utc=True)
    features = _add_relative_features(features, panel)
    if features.empty or features["asof"].max() < pd.Timestamp(source_end, tz="UTC"):
        raise RuntimeError(f"feature coverage ends at {features['asof'].max()}, before {source_end}")
    if (features["available_time"] > features["asof"].map(session_close)).any():
        raise AssertionError("feature PIT violation")

    signal_features = features[features["asof"].between(signal_start, signal_end)].copy()
    scored = score_rcqt(signal_features)
    scored["pit_st"] = [
        (day, code) in st_keys for day, code in zip(scored["asof"], scored["ts_code"].astype(str))
    ]
    scored = scored[~scored["pit_st"]].copy()
    current = scored[scored["quiet_eligible"] & scored["right_confirmed"]].copy()
    challenger = _relative_candidates(scored)
    current_top4 = _select_top4(current, "rcqt.stage1.quiet_confirmed.top4.control.v1")
    challenger_top4 = _select_top4(challenger, "rcqt.stage1.relative_orderly.top4.v1")

    sessions = pd.DatetimeIndex(sorted(panel["event_time"].drop_duplicates()))
    labels = build_q70_t10_labels(panel, profile=LABEL_PROFILE, session_dates=sessions)
    labels = labels.rename(columns={"event_time": "asof", "available_time": "label_available_time"})
    labels["asof"] = pd.to_datetime(labels["asof"], utc=True).dt.normalize()
    labels["label_available_time"] = pd.to_datetime(labels["label_available_time"], utc=True)
    mature_labels = labels[
        labels["label_available_time"] <= session_close(pd.Timestamp(source_end))
    ][["asof", "ts_code", "label_return", "label_available_time", "exit_time"]]

    ledgers = {}
    for name, frame in {
        "universe": scored,
        "current": current,
        "challenger": challenger,
        "current_top4": current_top4,
        "challenger_top4": challenger_top4,
    }.items():
        ledgers[name] = frame.merge(
            mature_labels, on=["asof", "ts_code"], how="inner", validate="one_to_one",
        )

    splits = {
        "discovery_2024": ("2024-01-01", "2024-12-31", "2025-01-20"),
        "validation_2025": ("2025-01-01", "2025-12-31", "2026-01-20"),
    }
    event_summary: dict[str, object] = {}
    portfolio_summary: dict[str, object] = {}
    px = prices.copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"], utc=True).dt.normalize()
    px["amount"] = pd.to_numeric(px["amount"], errors="raise")
    px = px.sort_values(["ts_code", "trade_date"], kind="mergesort")
    px["adv20"] = px.groupby("ts_code")["amount"].transform(
        lambda series: series.rolling(20, min_periods=20).median()
    )
    actions = load_corporate_actions(signal_start, source_end, ts_codes=codes)

    for split, (start, end, execution_end) in splits.items():
        sliced = {
            name: frame[frame["asof"].between(start, end)].copy()
            for name, frame in ledgers.items()
        }
        event_summary[split] = _split_summary(
            sliced["universe"], sliced["current"], sliced["challenger"],
            sliced["current_top4"], sliced["challenger_top4"],
        )
        split_prices = px[px["trade_date"].between(start, execution_end)].copy()
        split_actions = actions[
            pd.to_datetime(actions["ex_date"], utc=True).between(start, execution_end)
        ].copy() if len(actions) else actions.copy()
        for policy, ledger in (
            ("current", current_top4[current_top4["asof"].between(start, end)]),
            ("challenger", challenger_top4[challenger_top4["asof"].between(start, end)]),
        ):
            for cost, slippage in (("base", 0.001), ("stress", 0.003)):
                result, metrics = _backtest(ledger, split_prices, split_actions, slippage=slippage)
                target = output / "backtests" / split / policy / cost
                target.mkdir(parents=True, exist_ok=True)
                for artifact in ("orders", "trades", "nav", "positions", "corporate_actions"):
                    result[artifact].to_csv(target / f"{artifact}.csv", index=False)
                _write_json(target / "metrics.json", metrics)
                portfolio_summary[f"{split}_{policy}_{cost}"] = metrics

    validation_checks = _validation_pass(
        event_summary["validation_2025"],
        portfolio_summary["validation_2025_challenger_base"],
    )
    decision = "ADVANCE_TO_FORWARD" if all(validation_checks.values()) else "ABANDON_V1"
    summary = {
        "experiment_id": "relative_orderly_continuation_v1",
        "decision": decision,
        "validation_checks": validation_checks,
        "events": event_summary,
        "portfolios": portfolio_summary,
    }

    challenger.to_parquet(output / "candidate_relative_orderly_continuation.parquet", index=False)
    challenger_top4.to_csv(output / "selection_relative_orderly_top4.csv", index=False)
    _write_json(output / "SUMMARY.json", summary)
    _write_json(output / "DATA_MANIFEST.json", {
        "experiment_id": "relative_orderly_continuation_v1",
        "config": str(args.config.resolve()),
        "config_sha256": _sha(args.config),
        "codes_source": str(args.codes_source.resolve()),
        "codes_source_sha256": _sha(args.codes_source),
        "fixed_universe_codes": len(codes),
        "raw_start": raw_start,
        "source_end": source_end,
        "model_training": False,
        "parameter_sweep": False,
        "pit_st_audit": st_audit,
        "source_audit": source_audit,
        "credentials_persisted": False,
    })

    lines = [
        "# Relative Orderly Continuation V1", "",
        f"Decision: **{decision}**", "",
        "Only the candidate contract changed. Both control and challenger retain frozen `quiet_score` Top4 ordering.", "",
        "## Candidate quality", "",
        "| Split | Cohort | Mean | PF | Win rate | <=-8% | >=+10% | Rows |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("discovery_2024", "validation_2025"):
        for cohort in ("current_quiet_confirmed", "relative_orderly_continuation"):
            item = event_summary[split][cohort]
            lines.append(
                f"| {split} | {cohort} | {item['mean_return']:+.2%} | {item['profit_factor']:.3f} | "
                f"{item['win_rate']:.1%} | {item['down_8pct_rate']:.1%} | {item['up_10pct_rate']:.1%} | {item['rows']} |"
            )
    lines.extend(["", "## Executable backtest", "",
                  "| Split | Policy | Return | PF | MaxDD | Ex best week | Trades |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for split in ("discovery_2024", "validation_2025"):
        for policy in ("current", "challenger"):
            item = portfolio_summary[f"{split}_{policy}_base"]
            lines.append(
                f"| {split} | {policy} | {item['total_return']:+.2%} | {item['portfolio_profit_factor']:.3f} | "
                f"{item['max_drawdown']:.2%} | {item['return_excluding_best_week']:+.2%} | {item['trade_count']} |"
            )
    lines.extend(["", "## Validation checks", ""])
    lines.extend(f"- `{name}`: `{passed}`" for name, passed in validation_checks.items())
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = {
        str(path.relative_to(output)): _sha(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    _write_json(output / "ARTIFACT_HASHES.json", artifacts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--codes-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
