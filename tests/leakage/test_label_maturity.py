import pandas as pd
import pytest

from aistock9988.labeling.maturity import assert_labels_mature, mature_training_rows


def _labels():
    return pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"],
                         "available_time": ["2026-08-20T15:00:00Z", "2026-08-22T15:00:00Z"],
                         "label_return": [0.1, -0.1]})


def test_future_label_is_hard_failure():
    with pytest.raises(AssertionError, match="label leakage"):
        assert_labels_mature(_labels(), training_cutoff=pd.Timestamp("2026-08-21T16:00:00Z"))


def test_mature_labels_are_preserved_without_silent_drop():
    labels = _labels().iloc[[0]].copy()
    out = mature_training_rows(labels, training_cutoff=pd.Timestamp("2026-08-21T16:00:00Z"))
    assert len(out) == 1 and out.iloc[0]["ts_code"] == "000001.SZ"
