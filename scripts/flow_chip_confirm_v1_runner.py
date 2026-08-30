"""Control/challenger replay using trade-date rolling information only.

Daily rows are treated as available after their trade date closes. Database
ingestion timestamps are retained for diagnostics, not used as a simulated
real-time availability gate. Moneyflow is usable when ``trade_date <= T`` and
its required rolling window is fully present.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.data.quantdb import readonly_connection
from aistock9988.labeling.q70 import build_q70_t10_labels
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.selection.rcqt import score_rcqt
from aistock9988.time.session import session_close

from rcqt_quantdb_sample_runner import _features
from rcqt_stage1_quality_runner import (
    LABEL_PROFILE,
    _load_pit_st_keys,
    _sha,
    _write_json,
)
from relative_orderly_continuation_runner import _load_sources


def _load_codes(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=["ts_code"])
    else:
        frame = pd.read_csv(path, usecols=["ts_code"])
    codes = sorted(frame["ts_code"].astype(str).str.upper().unique().tolist())
    if not codes:
        raise ValueError("codes source is empty")
    return codes


def _base_candidates(scored: pd.DataFrame, st_keys: set[tuple[pd.Timestamp, str]]) -> pd.DataFrame:
    out = scored.copy()
    out["pit_st"] = [(day, code) in st_keys for day, code in zip(out["asof"], out["ts_code"].astype(str))]
    out["base_candidate"] = (
        (~out["pit_st"])
        & out["right_confirmed"]
        & (out["quiet_eligible"] | out["reset_eligible"])
    )
    out["base_score"] = out[["quiet_score", "reset_score"]].max(axis=1)
    out["sleeve"] = out.apply(
        lambda row: "quiet" if float(row["quiet_score"]) >= float(row["reset_score"]) else "reset",
        axis=1,
    )
    return out


def _load_holdernumbers(codes: list[str]) -> pd.DataFrame:
    placeholders = ",".join(["%s"] * len(codes))
    with readonly_connection() as connection:
        frame = pd.read_sql_query(
            "SELECT ts_code, ann_date, end_date, holder_num, update_time "
            "FROM stk_holdernumber_ts "
            "WHERE ann_date IS NOT NULL "
            f"AND ts_code IN ({placeholders}) "
            "ORDER BY ts_code, ann_date, end_date",
            connection,
            params=tuple(codes),
        )
    if frame.empty:
        return pd.DataFrame(columns=[
            "ts_code", "ann_date", "end_date", "holder_num", "update_time",
        ])
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["ann_date"] = pd.to_datetime(frame["ann_date"], utc=True).dt.normalize()
    frame["end_date"] = pd.to_datetime(frame["end_date"], utc=True).dt.normalize()
    frame["holder_num"] = pd.to_numeric(frame["holder_num"], errors="coerce")
    frame["update_time"] = pd.to_datetime(frame["update_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["ann_date", "end_date", "holder_num"]).copy()
    frame = frame.sort_values(["ts_code", "ann_date", "end_date"], kind="mergesort").reset_index(drop=True)
    return frame


def _load_moneyflow(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """Load daily moneyflow components and calculate backward-looking 5-session sums."""
    placeholders = ",".join(["%s"] * len(codes))
    with readonly_connection() as connection:
        frame = pd.read_sql_query(
            "SELECT ts_code, trade_date, "
            "buy_sm_amount - sell_sm_amount AS sm_net_amount, "
            "buy_md_amount - sell_md_amount AS md_net_amount, "
            "(buy_lg_amount - sell_lg_amount) + (buy_elg_amount - sell_elg_amount) AS lg_elg_net_amount, "
            "net_mf_amount, update_time "
            "FROM moneyflow_ts WHERE trade_date BETWEEN %s AND %s "
            f"AND ts_code IN ({placeholders}) ORDER BY ts_code, trade_date",
            connection,
            params=(start, end, *codes),
        )
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "ts_code", "asof", "sm_net_amount", "md_net_amount",
                "lg_elg_net_amount", "net_mf_amount", "mf5_sm_net_amount",
                "mf5_md_net_amount", "mf5_lg_elg_net_amount", "mf5_total_net_amount",
                "mf5_window_rows",
            ]
        )
    frame["ts_code"] = frame["ts_code"].astype(str).str.upper()
    frame["asof"] = pd.to_datetime(frame["trade_date"], utc=True).dt.normalize()
    for column in ("sm_net_amount", "md_net_amount", "lg_elg_net_amount", "net_mf_amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["asof", "sm_net_amount", "md_net_amount", "lg_elg_net_amount", "net_mf_amount"])
    frame = frame.sort_values(["ts_code", "asof"], kind="mergesort")
    frame["mf5_window_rows"] = frame.groupby("ts_code")["net_mf_amount"].transform(
        lambda s: s.rolling(5, min_periods=5).count()
    )
    frame["mf5_sm_net_amount"] = frame.groupby("ts_code")["sm_net_amount"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
    frame["mf5_md_net_amount"] = frame.groupby("ts_code")["md_net_amount"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
    frame["mf5_lg_elg_net_amount"] = frame.groupby("ts_code")["lg_elg_net_amount"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
    frame["mf5_total_net_amount"] = frame.groupby("ts_code")["net_mf_amount"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    )
    return frame[[
        "ts_code", "asof", "sm_net_amount", "md_net_amount", "lg_elg_net_amount",
        "net_mf_amount", "mf5_sm_net_amount", "mf5_md_net_amount",
        "mf5_lg_elg_net_amount", "mf5_total_net_amount", "mf5_window_rows",
    ]]


def _add_moneyflow_ratio(features: pd.DataFrame, prices: pd.DataFrame, moneyflow: pd.DataFrame) -> pd.DataFrame:
    """Join flow features using only dates through T and never filling missing days."""
    out = features.copy()
    out["asof"] = pd.to_datetime(out["asof"], utc=True).dt.normalize()
    px = prices[["ts_code", "trade_date", "amount"]].copy()
    px["ts_code"] = px["ts_code"].astype(str).str.upper()
    px["asof"] = pd.to_datetime(px["trade_date"], utc=True).dt.normalize()
    px["amount"] = pd.to_numeric(px["amount"], errors="coerce")
    px = px.sort_values(["ts_code", "asof"], kind="mergesort")
    px["amount_med5"] = px.groupby("ts_code")["amount"].transform(lambda s: s.rolling(5, min_periods=5).median())
    mf = moneyflow.merge(px[["ts_code", "asof", "amount_med5"]], on=["ts_code", "asof"], how="left", validate="one_to_one")
    mf["net_mf_ratio_5"] = mf["mf5_total_net_amount"] / mf["amount_med5"]
    out = out.merge(
        mf[[
            "ts_code", "asof", "sm_net_amount", "md_net_amount", "lg_elg_net_amount", "net_mf_amount",
            "mf5_sm_net_amount", "mf5_md_net_amount", "mf5_lg_elg_net_amount", "mf5_total_net_amount",
            "mf5_window_rows", "amount_med5", "net_mf_ratio_5",
        ]],
        on=["ts_code", "asof"],
        how="left",
        validate="one_to_one",
    )
    out["moneyflow_confirmed"] = (
        out["mf5_window_rows"].ge(5)
        & out["mf5_lg_elg_net_amount"].gt(0)
        & out["mf5_md_net_amount"].gt(0)
        & out["mf5_sm_net_amount"].le(0)
        & out["mf5_total_net_amount"].gt(0)
    )
    out["moneyflow_rejection_reason"] = ""
    out.loc[out["mf5_window_rows"].isna(), "moneyflow_rejection_reason"] = "moneyflow_missing_on_asof"
    out.loc[out["mf5_window_rows"].notna() & out["mf5_window_rows"].lt(5), "moneyflow_rejection_reason"] = "moneyflow_history_lt_5"
    mask = out["mf5_window_rows"].ge(5) & ~out["moneyflow_confirmed"] & out["moneyflow_rejection_reason"].eq("")
    out.loc[mask, "moneyflow_rejection_reason"] = "moneyflow_confirmation_failed"
    return out


def _holder_overlay(candidates: pd.DataFrame, holder: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        out = candidates.copy()
        for column in (
            "holder_confirmed", "holder_latest_ann_date", "holder_prev_ann_date",
            "holder_latest_num", "holder_prev_num", "holder_delta_ratio",
            "holder_ann_lag_days", "holder_rejection_reason",
        ):
            out[column] = pd.Series(dtype="object")
        return out
    records: list[dict[str, object]] = []
    by_code = {code: frame.copy() for code, frame in holder.groupby("ts_code", sort=False)}
    for row in candidates.to_dict("records"):
        code = str(row["ts_code"])
        asof = pd.Timestamp(row["asof"], tz="UTC") if not isinstance(row["asof"], pd.Timestamp) else row["asof"]
        visible = by_code.get(code, pd.DataFrame()).loc[lambda x: x["ann_date"] <= asof]
        payload = dict(row)
        if len(visible) < 2:
            payload.update({
                "holder_confirmed": False,
                "holder_latest_ann_date": pd.NaT,
                "holder_prev_ann_date": pd.NaT,
                "holder_latest_num": None,
                "holder_prev_num": None,
                "holder_delta_ratio": None,
                "holder_ann_lag_days": None,
                "holder_rejection_reason": "holder_history_lt_2",
            })
            records.append(payload)
            continue
        latest = visible.iloc[-1]
        previous = visible.iloc[-2]
        latest_num = float(latest["holder_num"])
        previous_num = float(previous["holder_num"])
        delta_ratio = latest_num / previous_num - 1.0 if previous_num > 0 else None
        payload.update({
            "holder_confirmed": bool(latest_num <= previous_num),
            "holder_latest_ann_date": latest["ann_date"],
            "holder_prev_ann_date": previous["ann_date"],
            "holder_latest_num": latest_num,
            "holder_prev_num": previous_num,
            "holder_delta_ratio": delta_ratio,
            "holder_ann_lag_days": int((asof - pd.Timestamp(latest["ann_date"])).days),
            "holder_rejection_reason": "" if latest_num <= previous_num else "holder_num_increased",
        })
        records.append(payload)
    out = pd.DataFrame(records)
    for column in ("holder_latest_ann_date", "holder_prev_ann_date"):
        out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    return out


def _select_top4(frame: pd.DataFrame, *, policy_id: str) -> pd.DataFrame:
    if frame.empty:
        columns = list(frame.columns) + [
            "candidate_rank", "selected", "selection_decision_id", "policy_id", "target_weight", "context_hash",
        ]
        return pd.DataFrame(columns=columns)
    out = frame.sort_values(
        ["asof", "base_score", "ts_code"], ascending=[True, False, True], kind="mergesort",
    ).groupby("asof", sort=True).head(4).copy()
    out["candidate_rank"] = out.groupby("asof").cumcount() + 1
    out["selected"] = True
    out["selection_decision_id"] = policy_id + "-" + out["asof"].dt.strftime("%Y%m%d")
    out["policy_id"] = policy_id
    out["target_weight"] = 0.12
    out["context_hash"] = out["asof"].map(
        lambda day: hashlib.sha256(f"{policy_id}|{day}".encode()).hexdigest()
    )
    return out


def _mature_labels(panel: pd.DataFrame, end: str) -> pd.DataFrame:
    sessions = pd.DatetimeIndex(sorted(panel["event_time"].drop_duplicates()))
    labels = build_q70_t10_labels(panel, profile=LABEL_PROFILE, session_dates=sessions)
    labels = labels.rename(columns={"event_time": "asof", "available_time": "label_available_time"})
    labels["asof"] = pd.to_datetime(labels["asof"], utc=True).dt.normalize()
    labels["label_available_time"] = pd.to_datetime(labels["label_available_time"], utc=True)
    return labels[labels["label_available_time"] <= session_close(pd.Timestamp(end, tz="UTC"))][
        ["asof", "ts_code", "label_return", "label_available_time", "exit_time"]
    ]


def _win_loss(frame: pd.DataFrame) -> pd.DataFrame:
    features = [
        "base_score", "quiet_score", "reset_score", "dist_ma60", "ret20", "ret60",
        "dd20", "dd60", "volume_ratio_20", "mf5_sm_net_amount", "mf5_md_net_amount",
        "mf5_lg_elg_net_amount", "mf5_total_net_amount", "net_mf_ratio_5",
        "holder_delta_ratio", "holder_ann_lag_days",
    ]
    if frame.empty:
        return pd.DataFrame(columns=[
            "feature", "winner_n", "loser_n", "winner_median", "loser_median", "winner_minus_loser",
        ])
    usable = frame.dropna(subset=["label_return"]).copy()
    if usable.empty:
        return pd.DataFrame(columns=[
            "feature", "winner_n", "loser_n", "winner_median", "loser_median", "winner_minus_loser",
        ])
    winners = usable[usable["label_return"] > 0]
    losers = usable[usable["label_return"] <= 0]
    rows: list[dict[str, object]] = []
    for feature in features:
        if feature not in usable.columns:
            continue
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


def _run_backtest(signals: pd.DataFrame, prices: pd.DataFrame, actions: pd.DataFrame,
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
    metrics["slippage_each_side"] = slippage
    return result, metrics


def _pit_audit(codes: list[str]) -> dict[str, object]:
    placeholders = ",".join(["%s"] * len(codes))
    with readonly_connection() as connection:
        moneyflow_cov = pd.read_sql_query(
            "SELECT MIN(trade_date) min_d, MAX(trade_date) max_d, COUNT(DISTINCT trade_date) date_n, "
            "MIN(update_time) min_u, MAX(update_time) max_u "
            "FROM moneyflow_ts WHERE ts_code IN (" + placeholders + ")",
            connection,
            params=tuple(codes),
        ).iloc[0].to_dict()
        moneyflow_sample = pd.read_sql_query(
            "SELECT trade_date, MIN(update_time) min_u, MAX(update_time) max_u, COUNT(*) n "
            "FROM moneyflow_ts WHERE ts_code IN (" + placeholders + ") "
            "AND trade_date BETWEEN %s AND %s GROUP BY trade_date ORDER BY trade_date DESC LIMIT 10",
            connection,
            params=tuple(codes) + ("2026-08-01", "2026-08-21"),
        )
        chip_cov = pd.read_sql_query(
            "SELECT MIN(trade_date) min_d, MAX(trade_date) max_d, MIN(update_time) min_u, MAX(update_time) max_u, COUNT(*) rows_n "
            "FROM chip_structure_ts WHERE ts_code IN (" + placeholders + ")",
            connection,
            params=tuple(codes),
        ).iloc[0].to_dict()
        chip_sample = pd.read_sql_query(
            "SELECT trade_date, MIN(update_time) min_u, MAX(update_time) max_u, COUNT(*) n "
            "FROM chip_structure_ts WHERE ts_code IN (" + placeholders + ") "
            "AND trade_date BETWEEN %s AND %s GROUP BY trade_date ORDER BY trade_date DESC LIMIT 10",
            connection,
            params=tuple(codes) + ("2026-07-01", "2026-07-09"),
        )
    return {
        "moneyflow_ts": {
            "usable_in_signal": True,
            "reason": "trade_date <= signal date; require exact T row and complete 5-session history; update_time retained for diagnostics only",
            "coverage": {key: str(value) for key, value in moneyflow_cov.items()},
            "late_update_sample": moneyflow_sample.to_dict("records"),
        },
        "chip_structure_ts": {
            "usable_in_signal": False,
            "reason": "observed update_time is batch backfill after trade date; fail-closed",
            "coverage": {key: str(value) for key, value in chip_cov.items()},
            "late_update_sample": chip_sample.to_dict("records"),
        },
        "holder_signal": {
            "usable_in_signal": True,
            "rule": "ann_date <= T; require latest two announcements",
        },
    }


def _summary(control: pd.DataFrame, challenger: pd.DataFrame, selected_control: pd.DataFrame,
             selected_challenger: pd.DataFrame, portfolios: dict[str, object],
             pit_audit: dict[str, object]) -> dict[str, object]:
    return {
        "control_rows": int(len(control)),
        "challenger_rows": int(len(challenger)),
        "selected_control_rows": int(len(selected_control)),
        "selected_challenger_rows": int(len(selected_challenger)),
        "holder_confirmation_rate": float(challenger["holder_confirmed"].mean()) if len(challenger) else None,
        "moneyflow_confirmation_rate": float(challenger["moneyflow_confirmed"].mean()) if len(challenger) else None,
        "holder_rejection_counts": challenger["holder_rejection_reason"].fillna("").value_counts().to_dict() if len(challenger) else {},
        "moneyflow_rejection_counts": challenger["moneyflow_rejection_reason"].fillna("").value_counts().to_dict() if len(challenger) else {},
        "pit_audit": pit_audit,
        "portfolios": portfolios,
    }


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
    moneyflow = _load_moneyflow(codes, raw_start, args.end)
    features = _add_moneyflow_ratio(features, prices, moneyflow)
    features["asof"] = pd.to_datetime(features["asof"], utc=True).dt.normalize()
    features["available_time"] = pd.to_datetime(features["available_time"], utc=True)
    features = features[features["asof"].between(args.start, args.end)].copy()
    if features.empty:
        raise RuntimeError("feature ledger is empty for requested range")
    if (features["available_time"] > features["asof"].map(session_close)).any():
        raise AssertionError("feature PIT violation")

    scored = score_rcqt(features)
    base = _base_candidates(scored, st_keys)
    control = base[base["base_candidate"]].copy()
    holder = _load_holdernumbers(codes)
    challenger_full = _holder_overlay(control, holder)
    challenger_full["moneyflow_confirmed"] = challenger_full["moneyflow_confirmed"].fillna(False)
    challenger = challenger_full[
        challenger_full["moneyflow_confirmed"] & challenger_full["holder_confirmed"]
    ].copy()

    selected_control = _select_top4(control, policy_id="stage1.flow_chip_confirm.control.top4.v1")
    selected_challenger = _select_top4(challenger, policy_id="stage1.flow_chip_confirm.challenger.top4.v1")

    labels = _mature_labels(panel, args.end)
    mature_control = control.merge(labels, on=["asof", "ts_code"], how="left", validate="one_to_one")
    mature_challenger = challenger.merge(labels, on=["asof", "ts_code"], how="left", validate="one_to_one")

    pit_audit = _pit_audit(codes)
    actions = load_corporate_actions(args.start, args.end, ts_codes=codes)
    bt_prices = prices.copy()
    portfolios: dict[str, object] = {}
    for variant, signals in (("control", selected_control), ("challenger", selected_challenger)):
        for cost, slippage in (("base", 0.001), ("stress", 0.003)):
            result, metrics = _run_backtest(signals, bt_prices, actions, slippage=slippage)
            target = output / "backtests" / variant / cost
            target.mkdir(parents=True, exist_ok=True)
            for artifact in ("orders", "trades", "nav", "positions", "corporate_actions"):
                result[artifact].to_csv(target / f"{artifact}.csv", index=False)
            _write_json(target / "metrics.json", metrics)
            portfolios[f"{variant}_{cost}"] = metrics

    control.to_parquet(output / "candidate_control.parquet", index=False)
    challenger_full.to_parquet(output / "candidate_challenger.parquet", index=False)
    selected_control.to_csv(output / "selection_control.csv", index=False)
    selected_challenger.to_csv(output / "selection_challenger.csv", index=False)
    mature_control.to_parquet(output / "mature_control.parquet", index=False)
    mature_challenger.to_parquet(output / "mature_challenger.parquet", index=False)
    _win_loss(mature_control).to_csv(output / "win_loss_control.csv", index=False)
    _win_loss(mature_challenger).to_csv(output / "win_loss_challenger.csv", index=False)
    _write_json(output / "PIT_AUDIT.json", pit_audit)
    _write_json(output / "SUMMARY.json", _summary(
        control, challenger, selected_control, selected_challenger, portfolios, pit_audit,
    ))
    _write_json(output / "DATA_MANIFEST.json", {
        "experiment_id": "flow_chip_confirm_v1",
        "config": str(args.config.resolve()),
        "config_sha256": _sha(args.config),
        "codes_source": str(args.codes_source.resolve()),
        "codes_source_sha256": _sha(args.codes_source),
        "raw_start": raw_start,
        "source_end": args.end,
        "pit_st_audit": st_audit,
        "source_audit": source_audit,
        "holder_overlay_rule": "latest_two_ann_date_le_asof and latest_holder_num <= previous_holder_num",
        "moneyflow_signal_usable": True,
        "moneyflow_rule": "trade_date<=asof with full 5-session history; require lg_elg>0, md>0, sm<=0, total>0",
        "chip_signal_usable": False,
        "model_training": False,
        "parameter_sweep": False,
    })

    lines = [
        "# Flow Chip Confirm V1",
        "",
        "Historical diagnostic only.",
        "",
        f"- Control candidate rows: {len(control)}",
        f"- Challenger candidate rows: {len(challenger)}",
        f"- Moneyflow usable in signal: {pit_audit['moneyflow_ts']['usable_in_signal']}",
        f"- Chip usable in signal: {pit_audit['chip_structure_ts']['usable_in_signal']}",
    ]
    for key in ("control_base", "control_stress", "challenger_base", "challenger_stress"):
        metrics = portfolios[key]
        lines.append(
            f"- {key}: return={metrics['total_return']:+.2%}, PF={metrics['portfolio_profit_factor']}, MaxDD={metrics['max_drawdown']:+.2%}, trades={metrics['trade_count']}"
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
