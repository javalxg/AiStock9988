"""Corrected RCQT candidate-pool XGBoost walk-forward diagnostic.

This runner intentionally freezes one pairwise-ranking contract. It does not
search model parameters or candidate gates. The transparent RCQT rules define
the opportunity set; XGBoost only chooses the daily ordering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.models.trainer import train_ranker
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.selection.rcqt import score_rcqt
from aistock9988.time.session import session_close, session_open


FEATURES = (
    "dist_ma60", "ret1", "ret20", "ret60", "dd20", "dd60", "vol20",
    "liq20", "volume_ratio_20", "kdj_k_bfq", "cci_bfq", "wr_bfq",
    "confirmation_strength", "quiet_score",
)
FEATURE_SET_ID = "feature.rcqt_corrected_xgb14.v1"
XGB_POLICY_ID = "rcqt.corrected_xgb_ranker.v1"
MODEL_PARAMS = {
    "objective": "rank:pairwise",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 20.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "random_state": 20260828,
    "n_jobs": 1,
    "tree_method": "hist",
}
SPLITS = (
    ("2025_h2", "2025-06-30", "2025-07-01", "2025-12-12"),
    ("2026", "2025-12-31", "2026-01-01", "2026-05-29"),
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_candidates(score_path: Path, label_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(score_path)
    frame["asof"] = pd.to_datetime(frame["asof"], utc=True).dt.normalize()
    frame["available_time"] = pd.to_datetime(frame["available_time"], utc=True)
    decision_close = frame["asof"].map(session_close)
    if (frame["available_time"] > decision_close).any():
        raise ValueError("feature PIT violation: available_time is after decision close")
    scored = score_rcqt(frame)
    candidates = scored[scored["quiet_eligible"] & scored["right_confirmed"]].copy()

    labels = pd.read_csv(label_path, usecols=[
        "ts_code", "event_time", "exit_time", "available_time", "label_return",
    ])
    labels["asof"] = pd.to_datetime(labels.pop("event_time"), utc=True).dt.normalize()
    labels["label_available_time"] = pd.to_datetime(labels.pop("available_time"), utc=True)
    labels["label_exit_time"] = pd.to_datetime(labels.pop("exit_time"), utc=True).dt.normalize()
    labels["label_return"] = pd.to_numeric(labels["label_return"], errors="raise")
    if labels.duplicated(["asof", "ts_code"]).any():
        raise ValueError("labels contain duplicate asof/ts_code keys")
    candidates = candidates.merge(labels, on=["asof", "ts_code"], how="inner", validate="one_to_one")
    required = [*FEATURES, "label_return"]
    if not np.isfinite(candidates[required].to_numpy(dtype=float)).all():
        raise ValueError("candidate dataset contains non-finite model values")
    return candidates.sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True)


def _relevance(labels: pd.Series, dates: pd.Series) -> pd.Series:
    # Five within-date relevance levels preserve the cross-sectional objective
    # without leaking return magnitudes across different market regimes.
    percentile = labels.groupby(dates).rank(method="first", pct=True)
    return np.minimum((percentile * 5).astype(int), 4).astype(float)


def _train_predict(candidates: pd.DataFrame, output: Path) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    training_audit: list[dict[str, object]] = []
    for split_name, cutoff_text, test_start, test_end in SPLITS:
        cutoff = pd.Timestamp(cutoff_text, tz="UTC")
        cutoff_time = session_close(cutoff)
        train = candidates[
            (candidates["asof"] <= cutoff)
            & (candidates["label_available_time"] <= cutoff_time)
        ].copy()
        test = candidates[candidates["asof"].between(test_start, test_end)].copy()
        group_size = train.groupby("asof")["ts_code"].transform("size")
        train = train[group_size >= 2].sort_values(["asof", "ts_code"], kind="mergesort")
        if train.empty or test.empty:
            raise RuntimeError(f"split {split_name} has empty train or test rows")
        if train["label_available_time"].max() > cutoff_time:
            raise AssertionError(f"split {split_name} contains immature labels")
        y = _relevance(train["label_return"], train["asof"])
        artifact = train_ranker(
            train[list(FEATURES)].reset_index(drop=True), y.reset_index(drop=True),
            group_dates=train["asof"].reset_index(drop=True),
            feature_set_id=FEATURE_SET_ID,
            label_profile_id="label.endpoint_open_open_t10.v1",
            training_cutoff=str(cutoff_time), model_id=f"rcqt_xgb_{split_name}",
            output_dir=output / "models", params=MODEL_PARAMS,
            metadata_extra={
                "candidate_contract": "quiet_eligible AND right_confirmed",
                "target_contract": "within-date five-level relevance from T+10 return",
                "test_start": test_start, "test_end": test_end,
            },
        )
        model = XGBRanker()
        model.load_model(output / "models" / f"rcqt_xgb_{split_name}.json")
        test["xgb_score"] = model.predict(test[list(FEATURES)])
        test["split"] = split_name
        predictions.append(test)
        booster = model.get_booster()
        importance = pd.DataFrame({
            "feature": list(FEATURES),
            "gain": [booster.get_score(importance_type="gain").get(name, 0.0) for name in FEATURES],
        }).sort_values(["gain", "feature"], ascending=[False, True], kind="mergesort")
        importance.to_csv(output / f"{split_name}_feature_importance.csv", index=False)
        training_audit.append({
            "split": split_name, "cutoff": str(cutoff_time), "train_rows": len(train),
            "train_dates": int(train["asof"].nunique()), "train_start": str(train["asof"].min()),
            "train_end": str(train["asof"].max()),
            "max_label_available_time": str(train["label_available_time"].max()),
            "test_rows": len(test), "test_dates": int(test["asof"].nunique()),
            "model_sha256": artifact.model_sha256,
        })
    _write_json(output / "training_audit.json", training_audit)
    return pd.concat(predictions, ignore_index=True)


def _select(predictions: pd.DataFrame, score: str, policy: str) -> pd.DataFrame:
    selected = predictions.sort_values(
        ["asof", score, "ts_code"], ascending=[True, False, True], kind="mergesort",
    ).groupby("asof", sort=True).head(4).copy()
    selected["candidate_rank"] = selected.groupby("asof").cumcount() + 1
    selected["selected"] = True
    selected["selection_decision_id"] = policy + "-" + selected["asof"].dt.strftime("%Y%m%d")
    selected["policy_id"] = policy
    selected["target_weight"] = 0.12
    return selected


def _execution_panel(path: Path) -> pd.DataFrame:
    px = pd.read_parquet(path)
    px["trade_date"] = pd.to_datetime(px["trade_date"], utc=True).dt.normalize()
    px["raw_open"] = px["economic_open"]
    px["raw_close"] = px["economic_close"]
    px["economic_high"] = px[["economic_open", "economic_close"]].max(axis=1)
    px["economic_low"] = px[["economic_open", "economic_close"]].min(axis=1)
    px["raw_high"] = px["economic_high"]
    px["raw_low"] = px["economic_low"]
    px["adj_factor"] = 1.0
    px["open_available_time"] = px["trade_date"].map(session_open)
    px["close_available_time"] = px["trade_date"].map(session_close)
    px["available_time"] = px["close_available_time"]
    px["is_suspended"] = False
    px["is_limit_up"] = False
    px["is_limit_down"] = False
    return px


def _run_portfolio(signals: pd.DataFrame, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    signal_start = signals["asof"].min()
    signal_end = signals["asof"].max()
    sessions = pd.DatetimeIndex(sorted(prices["trade_date"].unique()))
    later = sessions[sessions > signal_end]
    price_end = later[min(14, len(later) - 1)] if len(later) else sessions[-1]
    panel = prices[prices["trade_date"].between(signal_start, price_end)].copy()
    return run_backtest(
        signals, panel,
        config=BacktestConfig(
            max_positions=4, hold_sessions=10, stop_loss_pct=-0.08,
            stop_loss_mode="close_next_session_open", accounting_price_basis="economic",
            corporate_actions_mode="skip", lot_size=100,
            buy_commission=0.0003, sell_commission=0.0003, stamp_duty=0.0005,
        ),
    )


def _portfolio_outputs(output: Path, policy: str, period: str, signals: pd.DataFrame,
                       prices: pd.DataFrame) -> dict[str, object]:
    result = _run_portfolio(signals, prices)
    target = output / "backtests" / policy / period
    target.mkdir(parents=True, exist_ok=True)
    for name in ("orders", "trades", "nav", "positions"):
        result[name].to_csv(target / f"{name}.csv", index=False)
    metrics = summarize_backtest(result["nav"], result["trades"], initial_cash=1_000_000)
    metrics.update({
        "policy": policy, "period": period, "diagnostic_only": True,
        "execution_contract": "economic open/close; no raw limit/suspension/ADV20",
    })
    _write_json(target / "metrics.json", metrics)
    return metrics


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates(args.scores, args.labels)
    candidates.to_parquet(output / "candidate_dataset.parquet", index=False)
    predictions = _train_predict(candidates, output)
    prediction_columns = list(dict.fromkeys([
        "split", "asof", "ts_code", "xgb_score", "quiet_score", "label_return",
        "label_available_time", *FEATURES,
    ]))
    predictions[prediction_columns].to_parquet(output / "prediction_ledger.parquet", index=False)
    xgb = _select(predictions, "xgb_score", XGB_POLICY_ID)
    rule = _select(predictions, "quiet_score", "rcqt.corrected_rule_control.v1")
    xgb.to_csv(output / "xgb_selection_ledger.csv", index=False)
    rule.to_csv(output / "rule_selection_ledger.csv", index=False)

    prices = _execution_panel(args.prices)
    summary: dict[str, dict[str, object]] = {"xgb": {}, "rule": {}}
    for policy, ledger in (("xgb", xgb), ("rule", rule)):
        for split_name, _, _, _ in SPLITS:
            subset = ledger[ledger["split"] == split_name].copy()
            summary[policy][split_name] = _portfolio_outputs(output, policy, split_name, subset, prices)
        summary[policy]["combined"] = _portfolio_outputs(output, policy, "combined", ledger, prices)
    _write_json(output / "SUMMARY.json", summary)
    contract = {
        "candidate_pool": "corrected quiet_eligible AND right_confirmed",
        "features": FEATURES, "model_params": MODEL_PARAMS, "splits": SPLITS,
        "selection": "daily Top4", "entry_exit": "T+1 open, H10 open",
        "stop": "-8% close decision, next-session open execution", "parameter_sweep": False,
        "acceptance": {"portfolio_profit_factor": ">=2", "max_drawdown": ">=-15%",
                       "return_excluding_best_week": ">0"},
    }
    _write_json(output / "MODEL_CONTRACT.json", contract)
    artifacts = {str(path.relative_to(output)): _sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    _write_json(output / "MANIFEST.json", {"artifacts": artifacts})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
