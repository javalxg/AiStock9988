from datetime import datetime, timezone

import pandas as pd

from aistock9988.data.pit import assert_no_future, enforce_available_time
from aistock9988.market.context import build_context
import pytest


def test_pit_filters_future_rows():
    frame = pd.DataFrame({"event_time": ["2026-08-20", "2026-08-21"],
                          "available_time": ["2026-08-20T15:00:00Z", "2026-08-22T15:00:00Z"],
                          "pct_chg": [1.0, -2.0]})
    cutoff = datetime(2026, 8, 21, 16, tzinfo=timezone.utc)
    out = enforce_available_time(frame, decision_time=cutoff)
    assert len(out) == 1
    assert_no_future(out, decision_time=cutoff)


def test_market_context_is_deterministic_and_breadth_is_explicit():
    frame = pd.DataFrame({"pct_chg": [1.0, -1.0, 0.0, 2.0], "amount": [1, 2, 3, 4],
                          "is_limit_up": [True, False, False, False],
                          "is_limit_down": [False, True, False, False]})
    ctx = build_context(frame, asof="2026-08-21")
    assert ctx.universe_count == 4
    assert ctx.breadth_ratio == 0.5
    assert ctx.limit_up_count == 1 and ctx.limit_down_count == 1
    assert ctx.captured_only is True


@pytest.mark.parametrize("column,value", [("pct_chg", float("nan")), ("amount", float("inf"))])
def test_market_context_rejects_non_finite_inputs(column, value):
    frame = pd.DataFrame({"pct_chg": [1.0, -1.0], "amount": [1.0, 2.0]})
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match="finite"):
        build_context(frame, asof="2026-08-21")


def test_market_context_rejects_unknown_limit_state():
    frame = pd.DataFrame({"pct_chg": [1.0], "amount": [1.0], "is_limit_up": [None]})
    with pytest.raises(ValueError, match="limit states"):
        build_context(frame, asof="2026-08-21")
