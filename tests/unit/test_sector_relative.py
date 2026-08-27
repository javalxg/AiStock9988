import pandas as pd

from aistock9988.features.sector_relative import compute_sector_relative
from aistock9988.data.industry_pit import resolve_industry_map


def test_sector_relative_uses_resolved_mapping():
    frame = pd.DataFrame({"f": [1.0, 3.0, 10.0]}, index=["A", "B", "C"])
    result = compute_sector_relative(frame, {"A": "x", "B": "x", "C": "y"}, ["f"])
    assert result.loc["A", "f_sector_rel"] == -1.0
    assert result.loc["B", "f_sector_rel"] == 1.0
    assert result.loc["C", "f_sector_rel"] == 0.0


def test_industry_map_resolves_overlap_by_latest_in_date_then_index_code():
    membership = pd.DataFrame({
        "index_code": ["801020", "801010", "801030", "801020"],
        "con_code": ["A", "A", "B", "B"],
        "name": ["member", "member", "member", "member"],
        "in_date": ["2020-01-01", "2021-01-01", "2020-01-01", "2020-01-01"],
        "out_date": [None, None, None, None],
    })
    mapping, audit = resolve_industry_map(membership, signal_date="2022-01-01")
    assert mapping == {"A": "801010", "B": "801020"}
    assert audit.covered_count == 2 and audit.conflict_security_count == 2
