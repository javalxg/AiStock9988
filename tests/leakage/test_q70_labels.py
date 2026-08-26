import pandas as pd

from aistock9988.labeling.maturity import LabelProfile
from aistock9988.labeling.q70 import build_q70_t10_labels


def test_q70_t10_uses_t_plus_1_entry_and_t_plus_11_exit():
    sessions = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
    panel = pd.DataFrame({
        "ts_code": ["A"] * len(sessions),
        "event_time": sessions,
        "economic_open": list(range(100, 112)),
    })
    labels = build_q70_t10_labels(
        panel,
        profile=LabelProfile("t10", entry_delay_sessions=1, horizon_sessions=10, maturity_sessions=11),
        session_dates=sessions,
    )
    assert len(labels) == 1
    assert labels.iloc[0].entry_time == sessions[1]
    assert labels.iloc[0].exit_time == sessions[11]
    assert labels.iloc[0].label_return == (111 / 101) - 1
