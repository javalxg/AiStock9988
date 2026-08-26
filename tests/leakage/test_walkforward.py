import pandas as pd
import pytest

from aistock9988.models.walkforward import monthly_windows, validate_window_labels


def test_monthly_model_is_reused_for_weekly_predictions():
    windows = monthly_windows(train_start="2025-01-01", prediction_start="2026-01-01",
                              prediction_end="2026-02-28", window_months=12)
    assert len(windows) == 2
    assert windows[0].train_end < windows[0].prediction_date
    assert windows[0].model_id.startswith("q70_202601")


def test_window_rejects_immature_labels():
    window = monthly_windows(train_start="2025-01-01", prediction_start="2026-01-01",
                             prediction_end="2026-01-31")[0]
    labels = pd.DataFrame({"available_time": ["2026-01-05T15:00:00Z"]})
    with pytest.raises(AssertionError, match="newer than cutoff"):
        validate_window_labels(labels, window)
