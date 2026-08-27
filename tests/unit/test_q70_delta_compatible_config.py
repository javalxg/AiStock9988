import yaml
import pytest

from scripts.validate_q70_delta_compatible_config import validate


def test_delta_compatible_contract_is_explicitly_reference_only():
    result = validate()
    assert result["feature_count"] == 123
    assert result["label_horizon"] == 9
    assert result["maturity_lag"] == 10
    assert result["accounting_price_basis"] == "economic"


@pytest.mark.parametrize(("path", "key", "value"), [
    ("reference_only", None, False),
    ("data.forbid_stage2", None, False),
    ("selection.dynamic_upper_gate.enabled", None, False),
    ("selection.market_breadth_min", None, 0.35),
    ("label.entry_to_exit_sessions", None, 10),
    ("execution.accounting_price_basis", None, "raw"),
])
def test_delta_compatible_contract_rejects_tampering(tmp_path, path, key, value):
    source = yaml.safe_load(validate.__globals__["CONFIG"].read_text())
    target = source
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    config_path = tmp_path / "tampered.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False))
    with pytest.raises(ValueError):
        validate(config_path)
