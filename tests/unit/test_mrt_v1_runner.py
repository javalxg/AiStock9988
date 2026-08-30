import pandas as pd

from aistock9988.configuration import StrategyConfig
from scripts.mrt_v1_runner import _build_event_ledger


def _row(code: str, **overrides: object) -> dict[str, object]:
    row = {
        "asof": pd.Timestamp("2026-01-05", tz="UTC"),
        "ts_code": code,
        "mrt_feature_ready": True,
        "universe_pass": True,
        "selection_data_eligible": True,
        "missing_required_selection": "",
        "industry_covered": True,
        "market_excess_ret10": 0.05,
        "industry_excess_ret10": 0.03,
        "ret60": 0.10,
        "ret20": 0.10,
        "dist_ma60": 0.05,
        "vol20": 0.02,
        "vol20_pct": 0.50,
        "shock_close_lt_open": 1.0,
        "pct_chg": -6.0,
        "shock_amount_ratio": 2.0,
        "shock_open_ok": 1.0,
        "shock_close_ok": 1.0,
        "pit_st": False,
        "execution_status": "TRADABLE",
        "execution_data_eligible": True,
        "missing_required_execution": "",
        "shock_tradable": 1.0,
        "list_age_days": 365,
        "list_age_sessions": 250,
        "list_age_sessions": 365,
    }
    row.update(overrides)
    return row


def test_mrt_event_requires_both_setup_and_shock():
    strategy = StrategyConfig.from_yaml("configs/strategy/mrt_v1.yaml")
    frame = pd.DataFrame([
        _row("000001.SZ"),
        _row("000002.SZ", shock_amount_ratio=1.2),
    ])
    ledger = _build_event_ledger(frame, strategy)
    assert ledger["event_pass"].tolist() == [True, False]
    assert ledger.loc[1, "candidate_status"] == "NOT_IN_VIEW"
    assert "AMOUNT_BELOW_1P5_ADV20_PRIOR" in ledger.loc[1, "rejection_reasons"]
