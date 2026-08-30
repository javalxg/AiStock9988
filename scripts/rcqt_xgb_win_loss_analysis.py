"""Diagnose corrected XGB ranker winners, losers, rank skill and drift."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = (
    "dist_ma60", "ret1", "ret20", "ret60", "dd20", "dd60", "vol20",
    "liq20", "volume_ratio_20", "kdj_k_bfq", "cci_bfq", "wr_bfq",
    "confirmation_strength", "quiet_score",
)


def _profit_factor(values: pd.Series) -> float | None:
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return gains / losses if losses else None


def _event_metrics(frame: pd.DataFrame) -> dict[str, object]:
    values = pd.to_numeric(frame["label_return"], errors="raise")
    return {
        "rows": len(frame), "dates": int(frame["asof"].nunique()),
        "mean_return": float(values.mean()), "median_return": float(values.median()),
        "win_rate": float((values > 0).mean()), "profit_factor": _profit_factor(values),
        "down_8pct_rate": float((values <= -0.08).mean()),
        "up_10pct_rate": float((values >= 0.10).mean()),
    }


def _rank_skill(frame: pd.DataFrame, score: str) -> dict[str, object]:
    daily = frame.groupby("asof")[[score, "label_return"]].apply(
        lambda group: group[score].corr(group["label_return"], method="spearman")
    ).dropna()
    return {
        "score": score, "dates": len(daily), "mean_daily_rank_ic": float(daily.mean()),
        "median_daily_rank_ic": float(daily.median()), "positive_rank_ic_ratio": float((daily > 0).mean()),
    }


def _feature_comparison(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, group in selected.groupby("split", sort=True):
        winners = group[group["label_return"] > 0]
        losers = group[group["label_return"] <= 0]
        for feature in FEATURES:
            win = float(winners[feature].median()) if len(winners) else np.nan
            loss = float(losers[feature].median()) if len(losers) else np.nan
            pooled_std = float(group[feature].std(ddof=1))
            rows.append({
                "split": split, "feature": feature, "winner_count": len(winners),
                "loser_count": len(losers), "winner_median": win, "loser_median": loss,
                "winner_minus_loser": win - loss,
                "standardized_median_difference": (win - loss) / pooled_std if pooled_std > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def _drift(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, cutoff_text, start, end in (
        ("2025_h2", "2025-06-30", "2025-07-01", "2025-12-12"),
        ("2026", "2025-12-31", "2026-01-01", "2026-05-29"),
    ):
        cutoff = pd.Timestamp(cutoff_text, tz="UTC")
        cutoff_close = cutoff + pd.Timedelta(hours=7)
        train = candidates[(candidates["asof"] <= cutoff) & (candidates["label_available_time"] <= cutoff_close)]
        test = candidates[candidates["asof"].between(start, end)]
        for feature in FEATURES:
            train_mean = float(train[feature].mean())
            test_mean = float(test[feature].mean())
            train_std = float(train[feature].std(ddof=1))
            rows.append({
                "split": split, "feature": feature, "train_mean": train_mean,
                "test_mean": test_mean,
                "standardized_mean_shift": (test_mean - train_mean) / train_std if train_std > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def run(source: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(source / "prediction_ledger.parquet")
    candidates = pd.read_parquet(source / "candidate_dataset.parquet")
    for frame in (predictions, candidates):
        frame["asof"] = pd.to_datetime(frame["asof"], utc=True)
        frame["label_available_time"] = pd.to_datetime(frame["label_available_time"], utc=True)
    xgb = pd.read_csv(source / "xgb_selection_ledger.csv")
    rule = pd.read_csv(source / "rule_selection_ledger.csv")
    for frame in (xgb, rule):
        frame["asof"] = pd.to_datetime(frame["asof"], utc=True)

    feature_comparison = _feature_comparison(xgb)
    feature_comparison.to_csv(output / "xgb_selected_win_loss_features.csv", index=False)
    drift = _drift(candidates)
    drift.to_csv(output / "candidate_feature_drift.csv", index=False)

    summary: dict[str, object] = {"splits": {}}
    for split, test in predictions.groupby("split", sort=True):
        selected_xgb = xgb[xgb["split"] == split]
        selected_rule = rule[rule["split"] == split]
        xgb_keys = set(zip(selected_xgb["asof"], selected_xgb["ts_code"]))
        rule_keys = set(zip(selected_rule["asof"], selected_rule["ts_code"]))
        daily_overlap = selected_xgb[["asof", "ts_code"]].merge(
            selected_rule[["asof", "ts_code"]], on=["asof", "ts_code"], how="inner",
        ).groupby("asof").size()
        summary["splits"][split] = {
            "candidate_pool": _event_metrics(test),
            "xgb_top4": _event_metrics(selected_xgb),
            "rule_top4": _event_metrics(selected_rule),
            "xgb_rank_skill": _rank_skill(test, "xgb_score"),
            "rule_rank_skill": _rank_skill(test, "quiet_score"),
            "selection_overlap_rows": len(xgb_keys & rule_keys),
            "selection_union_rows": len(xgb_keys | rule_keys),
            "mean_daily_top4_overlap": float(daily_overlap.reindex(test["asof"].unique(), fill_value=0).mean()),
        }
    (output / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    lines = ["# S10 corrected XGB win/loss diagnosis", ""]
    for split in ("2025_h2", "2026"):
        item = summary["splits"][split]
        x = item["xgb_top4"]
        r = item["rule_top4"]
        xi = item["xgb_rank_skill"]
        ri = item["rule_rank_skill"]
        lines.extend([
            f"## {split}", "",
            f"- XGB Top4 event mean {x['mean_return']:+.2%}, PF {x['profit_factor']:.3f}, win rate {x['win_rate']:.1%}.",
            f"- Rule Top4 event mean {r['mean_return']:+.2%}, PF {r['profit_factor']:.3f}, win rate {r['win_rate']:.1%}.",
            f"- XGB daily Rank IC mean {xi['mean_daily_rank_ic']:+.4f}; rule score {ri['mean_daily_rank_ic']:+.4f}.",
            f"- Daily Top4 mean overlap {item['mean_daily_top4_overlap']:.2f}/4.", "",
        ])
    lines.extend([
        "## Interpretation", "",
        "The corrected XGB ranker adds ordering skill in 2025H2 but does not preserve it in 2026. "
        "The next XGB experiment must address training-sample dependence/regime stability rather than tune thresholds or gates.",
    ])
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
