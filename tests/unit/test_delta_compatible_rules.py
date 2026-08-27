import pandas as pd
import pytest

from aistock9988.selection.delta_compatible import (
    apply_dynamic_upper_gate,
    compute_dynamic_upper_gate,
    select_rank_holdings,
    weak_breadth_cash_fraction,
)


def test_dynamic_gate_is_causal_and_records_threshold():
    samples = pd.DataFrame({"dmi_adx_bfq": list(range(1000)), "label_return": [-x for x in range(1000)]})
    result = compute_dynamic_upper_gate(samples, factor="dmi_adx_bfq")
    assert result.active is True
    assert result.threshold == pytest.approx(699.3, abs=1)
    assert result.sample_count == 1000
    assert result.reason == "lower_tail_advantage"
    passed = apply_dynamic_upper_gate(pd.DataFrame({"dmi_adx_bfq": [1, 900]}), factor="dmi_adx_bfq",
                                      threshold=result.threshold)
    assert passed.dynamic_gate_passed.tolist() == [True, False]


def test_dynamic_gate_disables_when_mature_sample_window_is_short():
    result = compute_dynamic_upper_gate(pd.DataFrame({"x": [1], "label_return": [0.1]}), factor="x")
    assert result.active is False
    assert result.threshold is None
    assert result.reason == "insufficient_mature_samples"


def test_rank_holding_keeps_existing_top5_then_fills_top2():
    candidates = pd.DataFrame({"ts_code": ["A", "B", "C", "D", "E", "F"],
                               "candidate_rank": [1, 2, 3, 4, 5, 6]})
    out = select_rank_holdings(candidates, {"E", "F"}, max_positions=2, hold_buffer_n=5)
    assert out.ts_code.tolist() == ["E", "A"]


def test_weak_breadth_single_candidate_uses_half_cash_only():
    assert weak_breadth_cash_fraction(breadth=.39, minimum=.40, candidate_count=1,
                                       configured_fraction=.5) == .5
    assert weak_breadth_cash_fraction(breadth=.39, minimum=.40, candidate_count=2,
                                       configured_fraction=.5) == 1.0


def test_dynamic_gate_rejects_missing_factor_column():
    with pytest.raises(ValueError, match="missing dynamic gate factor"):
        apply_dynamic_upper_gate(pd.DataFrame({"x": [1]}), factor="dmi_adx_bfq", threshold=.5)
