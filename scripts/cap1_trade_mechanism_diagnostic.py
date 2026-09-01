"""Describe where CAP1-20 profits and losses came from without retuning it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "docs/council_20260828"
    / "RESET_WEAK_CONFIRM_V3_CAP1_20_2026_TO_0828_20260901"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs/council_20260828"
    / "CAP1_TRADE_MECHANISM_DIAGNOSTIC_20260901"
)

INPUTS = {
    "fills": "backtests/base/fills.parquet",
    "positions": "backtests/base/positions.parquet",
    "stress_fills": "backtests/stress/fills.parquet",
    "execution_decisions": "backtests/base/execution_decisions.parquet",
    "features": "ledgers/feature_ledger.parquet",
    "execution": "ledgers/execution_panel.parquet",
    "base_metrics": "backtests/base/metrics.json",
    "stress_metrics": "backtests/stress/metrics.json",
    "artifact_manifest": "manifests/artifact_manifest.json",
}

ENTRY_FEATURES = (
    "dist_ma60",
    "mkt_ret_20d",
    "ret1",
    "ret20",
    "dd20",
    "vol20",
    "liq20",
    "volume_ratio_20",
    "candidate_rank",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _pct(value: float) -> str:
    return f"{value:+.2%}" if np.isfinite(value) else "NA"


def _verify_inputs(source: Path) -> dict[str, dict[str, object]]:
    manifest = _read_json(source / INPUTS["artifact_manifest"])
    verified: dict[str, dict[str, object]] = {}
    for name, relative in INPUTS.items():
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        expected = manifest.get(relative, {}).get("sha256") if name != "artifact_manifest" else None
        if expected is not None and actual != expected:
            raise ValueError(f"sealed input hash mismatch: {relative}")
        verified[relative] = {
            "sha256": actual,
            "bytes": path.stat().st_size,
            "sealed_hash_verified": expected is not None,
        }
    return verified


def _trade_table(source: Path) -> pd.DataFrame:
    fills = pd.read_parquet(source / INPUTS["fills"])
    stress = pd.read_parquet(source / INPUTS["stress_fills"])
    positions = pd.read_parquet(source / INPUTS["positions"])
    decisions = pd.read_parquet(source / INPUTS["execution_decisions"])
    features = pd.read_parquet(source / INPUTS["features"])
    execution = pd.read_parquet(source / INPUTS["execution"])
    keys = ["decision_id", "ts_code"]

    buys = fills.loc[fills["side"].eq("BUY")].copy()
    sells = fills.loc[fills["side"].eq("SELL")].copy()
    stress_sells = stress.loc[stress["side"].eq("SELL")].copy()
    if not (len(buys) == len(sells) == len(stress_sells)):
        raise ValueError("CAP diagnostic requires one closed Base and Stress trade per buy")
    for name, frame in (("buys", buys), ("sells", sells), ("stress_sells", stress_sells)):
        if frame.duplicated(keys).any():
            raise ValueError(f"duplicate trade key in {name}")

    trades = buys[keys + ["trade_date", "price", "economic_price", "shares"]].merge(
        sells[
            keys
            + [
                "trade_date",
                "economic_return",
                "realized_pnl",
                "reason",
                "gap_return",
            ]
        ],
        on=keys,
        suffixes=("_buy", "_sell"),
        validate="one_to_one",
    )
    trades = trades.merge(
        stress_sells[keys + ["economic_return", "realized_pnl"]].rename(
            columns={
                "economic_return": "stress_economic_return",
                "realized_pnl": "stress_realized_pnl",
            }
        ),
        on=keys,
        validate="one_to_one",
    )
    trades["outcome"] = np.where(trades["realized_pnl"] > 0, "WIN", "LOSS")
    trades["trade_key"] = trades["decision_id"].astype(str) + "|" + trades["ts_code"].astype(str)

    chosen = decisions.loc[decisions["chosen"].astype(bool)].copy()
    chosen["signal_session"] = pd.to_datetime(chosen["signal_session"], utc=True).dt.normalize()
    trades = trades.merge(
        chosen[keys + ["signal_session", "candidate_rank"]],
        on=keys,
        validate="one_to_one",
    )
    features["asof"] = pd.to_datetime(features["asof"], utc=True).dt.normalize()
    trades = trades.merge(
        features[
            ["asof", "ts_code"] + [name for name in ENTRY_FEATURES if name != "candidate_rank"]
        ].rename(columns={"asof": "signal_session"}),
        on=["signal_session", "ts_code"],
        validate="many_to_one",
    )

    sessions = pd.DatetimeIndex(
        pd.to_datetime(execution["trade_date"], utc=True).dt.normalize().drop_duplicates().sort_values()
    )
    session_index = {session: index for index, session in enumerate(sessions)}
    positions["e_index"] = [
        session_index[pd.Timestamp(day)] - session_index[pd.Timestamp(entry)]
        for day, entry in zip(positions["trade_date"], positions["entry_date"])
    ]
    positions = positions.merge(trades[keys + ["outcome"]], on=keys, validate="many_to_one")
    early = positions.loc[positions["e_index"].le(2)].copy()
    early_wide = early.pivot(
        index=keys, columns="e_index", values="unrealized_return"
    ).rename(columns={0: "e0_close_return", 1: "e1_close_return", 2: "e2_close_return"})
    if set(early_wide.columns) != {"e0_close_return", "e1_close_return", "e2_close_return"}:
        raise ValueError("every trade must have E0/E1/E2 close observations")
    path = positions.groupby(keys, sort=False)["unrealized_return"].agg(
        held_close_mfe="max", held_close_mae="min"
    )
    early_path = early.groupby(keys, sort=False)["unrealized_return"].agg(
        early_close_mfe="max", early_close_mae="min"
    )
    trades = trades.merge(early_wide.reset_index(), on=keys, validate="one_to_one")
    trades = trades.merge(path.reset_index(), on=keys, validate="one_to_one")
    trades = trades.merge(early_path.reset_index(), on=keys, validate="one_to_one")

    execution = execution.copy()
    execution["trade_date"] = pd.to_datetime(execution["trade_date"], utc=True).dt.normalize()
    execution = execution.set_index(["trade_date", "ts_code"], verify_integrity=True)
    for horizon in (5, 10):
        values: list[float] = []
        for row in trades.itertuples(index=False):
            exit_index = session_index[pd.Timestamp(row.trade_date_sell)]
            target_index = exit_index + horizon
            if target_index >= len(sessions):
                values.append(np.nan)
                continue
            start = execution.loc[(sessions[exit_index], row.ts_code)]
            target = execution.loc[(sessions[target_index], row.ts_code)]
            valid = (
                bool(start["execution_data_eligible"])
                and bool(target["execution_data_eligible"])
                and np.isfinite(float(start["economic_open"]))
                and np.isfinite(float(target["economic_open"]))
            )
            values.append(
                float(target["economic_open"]) / float(start["economic_open"]) - 1.0
                if valid
                else np.nan
            )
        trades[f"post_exit_open_{horizon}_return"] = values
    return trades


def _group_tables(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    entry_rows = []
    for feature in ENTRY_FEATURES:
        grouped = trades.groupby("outcome")[feature].mean()
        entry_rows.append(
            {
                "feature": feature,
                "winner_mean": float(grouped["WIN"]),
                "loser_mean": float(grouped["LOSS"]),
                "winner_minus_loser": float(grouped["WIN"] - grouped["LOSS"]),
            }
        )
    entry = pd.DataFrame(entry_rows)

    path_columns = [
        "e0_close_return",
        "e1_close_return",
        "e2_close_return",
        "early_close_mfe",
        "early_close_mae",
        "held_close_mfe",
        "held_close_mae",
        "economic_return",
        "post_exit_open_5_return",
        "post_exit_open_10_return",
    ]
    path = trades.groupby("outcome")[path_columns].agg(["count", "mean", "median"])
    path.columns = [f"{column}_{stat}" for column, stat in path.columns]
    path = path.reset_index()

    e2 = trades.assign(
        e2_state=np.where(trades["e2_close_return"] > 0, "E2_POSITIVE", "E2_NONPOSITIVE")
    ).groupby("e2_state").agg(
        trades=("trade_key", "size"),
        winners=("realized_pnl", lambda values: int((values > 0).sum())),
        win_rate=("realized_pnl", lambda values: float((values > 0).mean())),
        mean_final_return=("economic_return", "mean"),
        realized_pnl=("realized_pnl", "sum"),
    ).reset_index()

    exits = trades.groupby("reason").agg(
        trades=("trade_key", "size"),
        win_rate=("realized_pnl", lambda values: float((values > 0).mean())),
        mean_return=("economic_return", "mean"),
        realized_pnl=("realized_pnl", "sum"),
        mean_gap_return=("gap_return", "mean"),
    ).reset_index()
    return {"entry": entry, "path": path, "e2": e2, "exits": exits}


def run(source: Path, output: Path) -> Path:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable diagnostic output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    inputs = _verify_inputs(source)
    trades = _trade_table(source)
    tables = _group_tables(trades)
    base_metrics = _read_json(source / INPUTS["base_metrics"])
    stress_metrics = _read_json(source / INPUTS["stress_metrics"])

    trades.sort_values(["trade_date_buy", "ts_code"]).to_csv(
        output / "trade_mechanism_ledger.csv", index=False
    )
    tables["entry"].to_csv(output / "entry_feature_comparison.csv", index=False)
    tables["path"].to_csv(output / "path_by_outcome.csv", index=False)
    tables["e2"].to_csv(output / "e2_sign_diagnostic.csv", index=False)
    tables["exits"].to_csv(output / "exit_reason_summary.csv", index=False)

    winners = trades.loc[trades["outcome"].eq("WIN")]
    losers = trades.loc[trades["outcome"].eq("LOSS")]
    e2_positive = trades.loc[trades["e2_close_return"].gt(0)]
    e2_nonpositive = trades.loc[trades["e2_close_return"].le(0)]
    top3 = trades.nlargest(3, "realized_pnl")
    summary = {
        "research_status": "DIAGNOSTIC_SEEN_HISTORY_NO_STRATEGY_CHANGE",
        "source_run": str(source.relative_to(ROOT)),
        "trade_count": int(len(trades)),
        "winner_count": int(len(winners)),
        "loser_count": int(len(losers)),
        "win_rate": float(len(winners) / len(trades)),
        "base": {
            "total_return": base_metrics["total_return"],
            "profit_factor": base_metrics["portfolio_profit_factor"],
            "max_drawdown": base_metrics["max_drawdown"],
            "return_excluding_best_week": base_metrics["return_excluding_best_week"],
            "return_excluding_top3": base_metrics["return_excluding_top3_profit"],
        },
        "stress": {
            "total_return": stress_metrics["total_return"],
            "profit_factor": stress_metrics["portfolio_profit_factor"],
            "max_drawdown": stress_metrics["max_drawdown"],
            "return_excluding_best_week": stress_metrics["return_excluding_best_week"],
            "return_excluding_top3": stress_metrics["return_excluding_top3_profit"],
        },
        "pnl": {
            "total_realized": float(trades["realized_pnl"].sum()),
            "winner_realized": float(winners["realized_pnl"].sum()),
            "loser_realized": float(losers["realized_pnl"].sum()),
            "top3_realized": float(top3["realized_pnl"].sum()),
            "top3_share_of_net_pnl": float(
                top3["realized_pnl"].sum() / trades["realized_pnl"].sum()
            ),
        },
        "path": {
            "winner_e2_mean": float(winners["e2_close_return"].mean()),
            "loser_e2_mean": float(losers["e2_close_return"].mean()),
            "winner_held_close_mfe_mean": float(winners["held_close_mfe"].mean()),
            "loser_held_close_mfe_mean": float(losers["held_close_mfe"].mean()),
            "winner_held_close_mae_mean": float(winners["held_close_mae"].mean()),
            "loser_held_close_mae_mean": float(losers["held_close_mae"].mean()),
            "e2_positive_trade_count": int(len(e2_positive)),
            "e2_positive_win_rate": float((e2_positive["realized_pnl"] > 0).mean()),
            "e2_nonpositive_trade_count": int(len(e2_nonpositive)),
            "e2_nonpositive_win_rate": float((e2_nonpositive["realized_pnl"] > 0).mean()),
            "e2_spearman_to_final_return": float(
                trades[["e2_close_return", "economic_return"]]
                .corr(method="spearman")
                .iloc[0, 1]
            ),
            "winner_post_exit_open_5_mean": float(winners["post_exit_open_5_return"].mean()),
            "winner_post_exit_open_10_mean": float(winners["post_exit_open_10_return"].mean()),
            "loser_post_exit_open_5_mean": float(losers["post_exit_open_5_return"].mean()),
            "loser_post_exit_open_10_mean": float(losers["post_exit_open_10_return"].mean()),
        },
        "constraints": {
            "parameter_scan_performed": False,
            "historical_overlay_backtest_performed": False,
            "cap_contract_changed": False,
            "xgboost_used": False,
            "wide_202_factor_system_used": False,
        },
    }
    _write_json(output / "SUMMARY.json", summary)
    _write_json(
        output / "SOURCE_MANIFEST.json",
        {
            "inputs": inputs,
            "diagnostic_code": {
                str(Path(__file__).resolve().relative_to(ROOT)): _sha256(Path(__file__).resolve())
            },
        },
    )

    stop = tables["exits"].set_index("reason").loc["STOP_LOSS"]
    result = f"""# CAP1-20 Trade Mechanism Diagnostic

Status: `DIAGNOSTIC_SEEN_HISTORY_NO_STRATEGY_CHANGE`.

This report describes the 41 sealed Base trades. It does not backtest a new
exit, scan a threshold, alter CAP1-20, use XGBoost, or touch the frozen
202-factor system.

## What the current return means

- Frozen Base return is {_pct(float(base_metrics['total_return']))}, PF
  {float(base_metrics['portfolio_profit_factor']):.3f}, MaxDD
  {_pct(float(base_metrics['max_drawdown']))}.
- Frozen Stress return is {_pct(float(stress_metrics['total_return']))}, PF
  {float(stress_metrics['portfolio_profit_factor']):.3f}, MaxDD
  {_pct(float(stress_metrics['max_drawdown']))}.
- There are {len(trades)} closed trades and {len(winners)} winners
  ({len(winners) / len(trades):.1%}). The top three trades contributed
  {summary['pnl']['top3_share_of_net_pnl']:.1%} of net realized PnL, while the
  already sealed ex-top3 portfolio return remains
  {_pct(float(base_metrics['return_excluding_top3_profit']))} Base and
  {_pct(float(stress_metrics['return_excluding_top3_profit']))} Stress.

## What actually separated winners

- Static entry separation is modest. Winners were deeper below MA60
  ({winners['dist_ma60'].mean():.2%} vs {losers['dist_ma60'].mean():.2%}),
  entered in weaker 20-day markets ({winners['mkt_ret_20d'].mean():.2%} vs
  {losers['mkt_ret_20d'].mean():.2%}), and had milder amount expansion
  ({winners['volume_ratio_20'].mean():.2f} vs
  {losers['volume_ratio_20'].mean():.2f}). Candidate rank was nearly identical
  ({winners['candidate_rank'].mean():.2f} vs
  {losers['candidate_rank'].mean():.2f}).
- The path separated much more strongly. By the E2 close, eventual winners
  averaged {_pct(winners['e2_close_return'].mean())}; eventual losers averaged
  {_pct(losers['e2_close_return'].mean())}. E2 return had Spearman
  {summary['path']['e2_spearman_to_final_return']:.3f} with final trade return.
- Of {len(e2_positive)} trades above entry at E2, {int((e2_positive['realized_pnl'] > 0).sum())}
  became net winners ({(e2_positive['realized_pnl'] > 0).mean():.1%}). Of
  {len(e2_nonpositive)} at or below entry, only
  {int((e2_nonpositive['realized_pnl'] > 0).sum())} became winners
  ({(e2_nonpositive['realized_pnl'] > 0).mean():.1%}). This is in-sample
  mechanism evidence, not an authorized E2 exit rule.
- Across the full held close path, winners averaged MFE
  {_pct(winners['held_close_mfe'].mean())} and MAE
  {_pct(winners['held_close_mae'].mean())}; losers averaged MFE
  {_pct(losers['held_close_mfe'].mean())} and MAE
  {_pct(losers['held_close_mae'].mean())}.

## Where the current contract leaks money

- The 39 time exits averaged {_pct(float(tables['exits'].set_index('reason').loc['TIME_EXIT', 'mean_return']))}
  and produced RMB {float(tables['exits'].set_index('reason').loc['TIME_EXIT', 'realized_pnl']):,.0f}.
  The two stop exits averaged {_pct(float(stop['mean_return']))} and lost RMB
  {abs(float(stop['realized_pnl'])):,.0f}.
- The nominal trailing close stop is not a guaranteed -8% fill. It triggers
  after a close and executes at the next tradable open; the worst realized
  trade was {trades.nsmallest(1, 'economic_return').iloc[0]['ts_code']} at
  {_pct(float(trades['economic_return'].min()))}.
- Winners still rose an average {_pct(winners['post_exit_open_5_return'].mean())}
  over the next five session opens and
  {_pct(winners['post_exit_open_10_return'].mean())} over the next ten; losing
  trades continued {_pct(losers['post_exit_open_5_return'].mean())} and
  {_pct(losers['post_exit_open_10_return'].mean())}. CAP therefore has both a
  left-tail timing problem and a right-tail truncation problem.

## Decision

Do not replace CAP with another static indicator stack or another XGBoost
ranker. The evidence says the entry rule identifies a useful reset state, but
the economically strongest missing information arrives after the fill. Keep
CAP1-20 unchanged and evaluate the already hash-registered early-path overlay
only on append-only signals from 2026-09-01 onward. The latest common database
cutoff is still 2026-08-28, so no forward signal or forward return exists yet.
"""
    (output / "RESULT.md").write_text(result, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args.source.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
