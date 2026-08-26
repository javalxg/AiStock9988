import pandas as pd
import pytest

from aistock9988.features.registry import FeatureSet, assemble_matrix


def test_feature_order_and_matrix_contract():
    spec = FeatureSet.create("feature.test.v1", ["f2", "f1"])
    frame = pd.DataFrame({"ts_code": ["B", "A"], "event_time": ["2026-08-21", "2026-08-21"],
                          "f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    out = assemble_matrix(frame, spec)
    assert out.ts_code.tolist() == ["A", "B"]
    assert out.columns.tolist() == ["ts_code", "event_time", "f2", "f1"]
    assert len(spec.order_hash) == 64


def test_feature_matrix_rejects_null_and_non_numeric():
    spec = FeatureSet.create("feature.test.v1", ["f1"])
    with pytest.raises(ValueError, match="missing values"):
        assemble_matrix(pd.DataFrame({"ts_code": ["A"], "event_time": ["2026-08-21"], "f1": [None]}), spec)
    with pytest.raises(TypeError, match="numeric"):
        assemble_matrix(pd.DataFrame({"ts_code": ["A"], "event_time": ["2026-08-21"], "f1": ["x"]}), spec)
