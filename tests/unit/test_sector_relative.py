import pandas as pd

from aistock9988.features.sector_relative import compute_sector_relative


def test_sector_relative_uses_resolved_mapping():
    frame = pd.DataFrame({"f": [1.0, 3.0, 10.0]}, index=["A", "B", "C"])
    result = compute_sector_relative(frame, {"A": "x", "B": "x", "C": "y"}, ["f"])
    assert result.loc["A", "f_sector_rel"] == -1.0
    assert result.loc["B", "f_sector_rel"] == 1.0
    assert result.loc["C", "f_sector_rel"] == 0.0
