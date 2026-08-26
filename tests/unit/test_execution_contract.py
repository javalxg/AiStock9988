import pandas as pd
import pytest

from aistock9988.execution.corporate_actions import CorporateAction, apply_action
from aistock9988.execution.prices import economic_return, validate_execution_panel


def test_price_contract_keeps_raw_and_economic_fields_separate():
    panel = validate_execution_panel(pd.DataFrame({
        "trade_date": ["2026-08-20"], "ts_code": ["A"],
        "raw_open": [10], "raw_high": [11], "raw_low": [9], "raw_close": [10],
        "economic_open": [20], "economic_high": [22], "economic_low": [18],
        "economic_close": [20], "adj_factor": [2],
    }))
    assert panel.iloc[0].raw_close == 10
    assert economic_return(20, 18) == pytest.approx(-0.1)


def test_split_and_dividend_update_position_and_cash():
    position = {"shares": 100, "entry_price": 20}
    cash = apply_action(position, CorporateAction("A", "2026-08-21", split_ratio=2, cash_dividend=0.5))
    assert position["shares"] == 200
    assert position["entry_price"] == 10
    assert cash == 50
