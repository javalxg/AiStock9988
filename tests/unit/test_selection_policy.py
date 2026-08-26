import pandas as pd

from aistock9988.market.context import build_context
from aistock9988.selection.policy import direct_topk


def _candidates():
    return pd.DataFrame({"asof": ["2026-08-21"] * 3, "ts_code": ["B", "A", "C"],
                         "candidate_rank": [2, 1, 3]})


def test_low_breadth_reduces_exposure_and_records_reason():
    ctx = build_context(pd.DataFrame({"pct_chg": [-1., 0., -2., 1.]}), asof="2026-08-21")
    decision = direct_topk(_candidates(), ctx, max_positions=5, low_breadth_top_n=2, breadth_min=0.40)
    assert decision.selected == ("A", "B")
    assert decision.reason == "low_breadth_reduce_exposure"


def test_normal_breadth_selects_topk_deterministically():
    ctx = build_context(pd.DataFrame({"pct_chg": [1., 2., -1., 0.]}), asof="2026-08-21")
    decision = direct_topk(_candidates(), ctx, max_positions=2, breadth_min=0.40)
    assert decision.selected == ("A", "B")
