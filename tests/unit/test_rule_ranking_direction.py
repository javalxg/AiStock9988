import pandas as pd
import pytest
from dataclasses import replace
from pathlib import Path

from aistock9988.configuration import StrategyConfig
from aistock9988.selection.pipeline import build_rule_ledgers


def _features() -> pd.DataFrame:
    asof = pd.Timestamp("2026-08-31", tz="UTC")
    return pd.DataFrame(
        {
            "asof": [asof] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "bundle_id": ["b"] * 3,
            "feature_set_hash": ["f"] * 3,
            "universe_pass": [True] * 3,
            "selection_data_eligible": [True] * 3,
            "training_data_eligible": [True] * 3,
            "execution_data_eligible": [True] * 3,
            "missing_required_selection": [""] * 3,
            "missing_required_training": [""] * 3,
            "missing_required_execution": [""] * 3,
            "missing_optional": [""] * 3,
            "feature_ready": [True] * 3,
            "feature_rejection_reason": [""] * 3,
            "execution_status": ["TRADABLE"] * 3,
            "liq20": [200000.0] * 3,
            "ret20": [-0.30, 0.00, 0.20],
        }
    )


@pytest.mark.parametrize(
    ("direction", "expected_first"),
    [("asc", "000001.SZ"), ("desc", "000003.SZ")],
)
def test_rule_ranking_direction_matches_config(direction: str, expected_first: str):
    base = StrategyConfig.from_yaml(
        Path(__file__).resolve().parents[2]
        / "configs/strategy/s46_mild_liquid_rank_v1.yaml"
    )
    strategy = replace(
        base,
        ranking={
            "method": "weighted_cross_sectional_rank",
            "terms": [{"feature": "ret20", "direction": direction, "weight": 1.0}],
        },
    )

    candidate = build_rule_ledgers(_features(), strategy, ("2026-08-31",))["candidate"]
    in_view = candidate[candidate["candidate_status"].eq("IN_VIEW")]
    assert in_view.sort_values("candidate_rank").iloc[0].ts_code == expected_first


def test_stage1_rejections_do_not_enter_rank_denominator():
    base = StrategyConfig.from_yaml(
        Path(__file__).resolve().parents[2]
        / "configs/strategy/s46_mild_liquid_rank_v1.yaml"
    )
    features = _features()
    features.loc[features["ts_code"].eq("000003.SZ"), "liq20"] = 50000.0
    strategy = replace(
        base,
        ranking={
            "method": "weighted_cross_sectional_rank",
            "terms": [{"feature": "ret20", "direction": "asc", "weight": 1.0}],
        },
    )

    candidate = build_rule_ledgers(features, strategy, ("2026-08-31",))["candidate"]
    rejected = candidate[candidate["ts_code"].eq("000003.SZ")].iloc[0]
    passed = candidate[candidate["ts_code"].isin(["000001.SZ", "000002.SZ"])]
    assert not bool(rejected.stage1_pass)
    assert pd.isna(rejected.rule_score)
    assert passed.rule_score.tolist() == [1.0, 0.5]
