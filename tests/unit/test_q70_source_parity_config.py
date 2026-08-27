from scripts.validate_q70_source_parity_config import validate


def test_q70_source_parity_isolated_contract():
    result = validate()
    assert result["feature_count"] == 123
    assert result["formal_end"] == "2026-07-31"
    assert result["reference_end"] == "2026-08-14"
