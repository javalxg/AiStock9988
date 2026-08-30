from pathlib import Path

import pytest
import yaml

from scripts.r2e_s1a_formal_runner import DEFAULT_CONFIG, build_formal_argv


def test_s1a_frozen_config_builds_single_variable_formal_command(tmp_path):
    argv = build_formal_argv(
        run_dir=tmp_path / "experiments" / ".running" / "run-s1a",
        features=tmp_path / "features.csv",
        prices=tmp_path / "prices.csv",
        config_path=DEFAULT_CONFIG,
    )
    assert "--no-right-confirm" in argv
    assert argv[argv.index("--quiet-slots") + 1] == "0"
    assert argv[argv.index("--hold-sessions") + 1] == "10"
    assert "--no-trailing" in argv
    assert argv[argv.index("--max-order-to-adv20") + 1] == "0.02"
    assert argv[argv.index("--slippage") + 1] == "0.001"


def test_s1a_rejects_config_that_sneaks_in_right_confirmation(tmp_path):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config["selection"]["require_right_confirmation"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="reset-only"):
        build_formal_argv(
            run_dir=Path("run"),
            features=Path("features.csv"),
            prices=Path("prices.csv"),
            config_path=path,
        )
