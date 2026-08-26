from pathlib import Path

from aistock9988.features.registry import FeatureSet


def test_f0_manifest_has_exactly_123_columns():
    root = Path(__file__).parents[2]
    feature_set = FeatureSet.from_f0_json(root / "configs/feature_sets/f0_123_columns.json")
    assert len(feature_set.columns) == 123
