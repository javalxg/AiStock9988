from dataclasses import asdict

import pandas as pd
import pytest

from aistock9988.data.snapshot import bind_file_snapshot


def test_bind_file_snapshot_freezes_recoverable_input_with_deterministic_contract(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({
        "asof": ["2026-01-02", "2026-01-05"],
        "ts_code": ["000001.SZ", "000002.SZ"],
        "score": [0.2, 0.3],
    }).to_csv(source, index=False)
    run_a = tmp_path / "a" / "experiments" / ".running" / "run-a"
    run_b = tmp_path / "b" / "experiments" / ".running" / "run-b"

    first = bind_file_snapshot(
        source,
        run_a,
        logical_name="features",
        source_id="fixture.features",
        query={"start": "2026-01-01", "end": "2026-01-31"},
        event_column="asof",
    )
    second = bind_file_snapshot(
        source,
        run_b,
        logical_name="features",
        source_id="fixture.features",
        query={"start": "2026-01-01", "end": "2026-01-31"},
        event_column="asof",
    )

    assert asdict(first) == asdict(second)
    assert (run_a / first.relative_path).read_bytes() == source.read_bytes()
    assert first.row_count == 2
    assert first.columns == ("asof", "ts_code", "score")
    assert first.min_event_time == "2026-01-02T00:00:00+00:00"
    assert first.max_event_time == "2026-01-05T00:00:00+00:00"


def test_bind_file_snapshot_is_immutable(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("asof,value\n2026-01-02,1\n")
    run_dir = tmp_path / "experiments" / ".running" / "run-a"
    kwargs = dict(
        logical_name="features",
        source_id="fixture.features",
        query={},
        event_column="asof",
    )
    bind_file_snapshot(source, run_dir, **kwargs)
    with pytest.raises(FileExistsError, match="immutable bound snapshot"):
        bind_file_snapshot(source, run_dir, **kwargs)
