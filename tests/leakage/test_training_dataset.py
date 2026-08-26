import pandas as pd
import pytest

from aistock9988.features.registry import FeatureSet
from aistock9988.labeling.dataset import build_training_dataset


def _features():
    return pd.DataFrame({"ts_code": ["A", "B"], "event_time": ["2026-08-20", "2026-08-20"],
                         "available_time": ["2026-08-20T15:00:00Z"] * 2, "f1": [1.0, 2.0]})


def test_training_dataset_joins_only_mature_labels():
    labels = pd.DataFrame({"ts_code": ["A", "B"], "event_time": ["2026-08-20"] * 2,
                           "available_time": ["2026-08-21T15:00:00Z"] * 2, "label_return": [0.1, -0.1]})
    X, y = build_training_dataset(_features(), labels, feature_set=FeatureSet.create("f", ["f1"]),
                                  training_cutoff=pd.Timestamp("2026-08-22T15:00:00Z"))
    assert X["f1"].tolist() == [1.0, 2.0] and y.tolist() == [0.1, -0.1]


def test_training_dataset_blocks_future_features():
    features = _features()
    features.loc[0, "available_time"] = "2026-08-23T15:00:00Z"
    labels = pd.DataFrame({"ts_code": ["A", "B"], "event_time": ["2026-08-20"] * 2,
                           "available_time": ["2026-08-21T15:00:00Z"] * 2, "label_return": [0.1, -0.1]})
    with pytest.raises(AssertionError, match="PIT violation"):
        build_training_dataset(features, labels, feature_set=FeatureSet.create("f", ["f1"]),
                               training_cutoff=pd.Timestamp("2026-08-22T15:00:00Z"))
