import pandas as pd

from aistock9988.features.registry import FeatureSet
from aistock9988.models.pipeline import train_and_rank


def test_training_pipeline_writes_model_full_ledger_and_top20(tmp_path):
    dates = ["2026-08-18", "2026-08-18", "2026-08-19", "2026-08-19"]
    features = pd.DataFrame({"ts_code": ["A", "B", "A", "B"], "event_time": dates,
                             "available_time": [f"{d}T15:00:00Z" for d in dates], "f1": [1., 0., 2., 1.]})
    labels = pd.DataFrame({"ts_code": ["A", "B", "A", "B"], "event_time": dates,
                           "available_time": ["2026-08-20T15:00:00Z"] * 4,
                           "label_return": [1., 0., 1., 0.]})
    pred = pd.DataFrame({"ts_code": ["A", "B"], "event_time": ["2026-08-21"] * 2,
                         "available_time": ["2026-08-21T15:00:00Z"] * 2, "f1": [2.5, 1.5]})
    run = train_and_rank(features=features, labels=labels, prediction_features=pred,
                         feature_set=FeatureSet.create("feature.test.v1", ["f1"]),
                         training_cutoff=pd.Timestamp("2026-08-21T16:00:00Z"), asof="2026-08-21",
                         model_id="m1", output_dir=tmp_path, params={"n_estimators": 2, "max_depth": 2}, top_n=2)
    assert run.prediction_path.exists() and run.candidate_path.exists()
    assert len(pd.read_csv(run.prediction_path)) == 2
    assert len(pd.read_csv(run.candidate_path)) == 2
