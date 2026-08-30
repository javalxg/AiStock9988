import pandas as pd

from aistock9988.configuration import StrategyConfig
from scripts.bec_v1_runner import _augment_audit_ledgers


def test_bec_config_is_independent_and_uses_single_industry_rank():
    strategy = StrategyConfig.from_yaml("configs/strategy/bec_v1.yaml")
    assert strategy.strategy_id == "bec_breadth_expansion_continuation_v1"
    assert strategy.portfolio["candidate_view_size"] == 20
    assert strategy.portfolio["entries_per_decision"] == 5
    assert len(strategy.ranking["terms"]) == 1
    assert strategy.ranking["terms"][0]["feature"] == "industry_excess_ret10"
    assert strategy.ranking["terms"][0]["weight"] == 1.0
    assert "index_member_all_ts" in strategy.data_policy["dense_required"]["selection"]
    assert "stk_limit_ts" in strategy.data_policy["dense_required"]["selection"]
    assert strategy.data_policy["source_availability"]["index_member_all_ts"] == "interval_in_out_trade_date"


def test_bec_stage_expression_has_all_four_mechanism_groups():
    expression = StrategyConfig.from_yaml("configs/strategy/bec_v1.yaml").stage1["expression"]
    leaves = expression["all"]
    fields = {str(item["left"]) for item in leaves}
    assert {"breadth_ma5", "breadth_ma5_delta5"} <= fields
    assert {"industry_breadth_ma5", "industry_median_ret10", "industry_excess_ret10"} <= fields
    assert {"turnover_stability20", "vol20"} <= fields
    assert {"economic_close", "ma5", "prev3_high", "ret1"} <= fields | {
        str(item.get("right")) for item in leaves if "right" in item
    }


def test_bec_audit_ledger_preserves_selection_and_execution_records(tmp_path):
    (tmp_path / "ledgers").mkdir()
    (tmp_path / "manifests").mkdir()
    for scenario in ("base", "stress"):
        (tmp_path / "backtests" / scenario).mkdir(parents=True)
        pd.DataFrame({"signal_session": [pd.Timestamp("2026-01-05", tz="UTC")], "ts_code": ["000001.SZ"]}).to_parquet(
            tmp_path / "backtests" / scenario / "execution_decisions.parquet"
        )
        pd.DataFrame({"trade_date": [pd.Timestamp("2026-01-06", tz="UTC")], "ts_code": ["000001.SZ"], "side": ["BUY"]}).to_parquet(
            tmp_path / "backtests" / scenario / "fills.parquet"
        )
        pd.DataFrame({"trade_date": [pd.Timestamp("2026-01-06", tz="UTC")], "ts_code": ["000001.SZ"], "event_type": ["ENTRY_FILL"]}).to_parquet(
            tmp_path / "backtests" / scenario / "position_events.parquet"
        )
        pd.DataFrame({"execution_session": [pd.Timestamp("2026-01-06", tz="UTC")], "ts_code": ["000001.SZ"], "status": ["FILLED"]}).to_parquet(
            tmp_path / "backtests" / scenario / "orders.parquet"
        )
    feature = pd.DataFrame({
        "asof": [pd.Timestamp("2026-01-05", tz="UTC")], "ts_code": ["000001.SZ"],
        "feature_rejection_reason": [""],
    })
    candidate = pd.DataFrame({
        "asof": [pd.Timestamp("2026-01-05", tz="UTC")], "ts_code": ["000001.SZ"],
        "stage1_pass": [True], "candidate_rank": [1], "candidate_status": ["IN_VIEW"],
        "candidate_snapshot_id": ["snap"], "score_rejection_reason": [""],
    })
    feature.to_parquet(tmp_path / "ledgers" / "feature_ledger.parquet")
    candidate.to_parquet(tmp_path / "ledgers" / "candidate_ledger.parquet")
    (tmp_path / "manifests" / "code_manifest.json").write_text("{}\n")
    (tmp_path / "manifests" / "artifact_manifest.json").write_text("{}\n")
    _augment_audit_ledgers(tmp_path)
    event = pd.read_parquet(tmp_path / "ledgers" / "event_ledger.parquet")
    assert event.loc[0, "record_type"] == "SELECTION"
    assert event.loc[0, "event_status"] == "IN_VIEW"
    assert {"EXECUTION_DECISION", "ORDER", "FILL", "POSITION_EVENT"} <= set(event["record_type"].dropna())
