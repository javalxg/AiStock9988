from scripts.validate_q70_delta_compatible_config import validate


def test_delta_compatible_contract_is_explicitly_reference_only():
    result = validate()
    assert result["feature_count"] == 123
    assert result["label_horizon"] == 9
    assert result["maturity_lag"] == 10
    assert result["accounting_price_basis"] == "economic"
