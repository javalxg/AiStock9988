import os

import pytest

from aistock9988.data.q70_source import load_f0_panel


@pytest.mark.skipif(not os.getenv("AISTOCK_DB_HOST"), reason="requires explicit quant_db environment")
def test_q70_source_has_frozen_123_columns():
    panel = load_f0_panel("2026-08-20", "2026-08-21")
    assert len([c for c in panel.columns if c.endswith("_sector_rel")]) == 57
    assert "available_time" in panel.columns
