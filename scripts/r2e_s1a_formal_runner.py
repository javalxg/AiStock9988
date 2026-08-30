"""Frozen formal entrypoint for the R2E S1-A reset-only control."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from scripts import rcqt_formal_runner

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "r2e_s1a_reset_setup_v1.yaml"


def build_formal_argv(*, run_dir: Path, features: Path, prices: Path, config_path: Path) -> list[str]:
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != 1 or config.get("frozen") is not True:
        raise ValueError("S1-A requires a frozen schema_version=1 config")
    selection = config["selection"]
    execution = config["execution"]
    if selection.get("require_right_confirmation") is not False or selection.get("quiet_slots") != 0:
        raise ValueError("S1-A must remain reset-only without right confirmation")
    if execution.get("hold_sessions") != 10 or execution.get("trailing") is not False:
        raise ValueError("S1-A must remain H10 without trailing")

    return [
        "rcqt_formal_runner",
        "--run-dir", str(run_dir),
        "--features", str(features),
        "--prices", str(prices),
        "--experiment-config", str(config_path),
        "--experiment-id", str(config["experiment_id"]),
        "--no-right-confirm",
        "--reset-slots", str(selection["reset_slots"]),
        "--quiet-slots", "0",
        "--single-weight-cap", str(selection["single_weight_cap"]),
        "--sector-weight-cap", str(selection["sector_weight_cap"]),
        "--equity-cap", str(selection["equity_cap"]),
        "--hold-sessions", "10",
        "--max-positions", str(execution["max_positions"]),
        "--stop-loss-pct", str(execution["stop_loss_pct"]),
        "--no-trailing",
        "--max-order-to-adv20", str(execution["max_order_to_adv20"]),
        "--slippage", str(execution["slippage_each_side"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    old_argv = sys.argv
    sys.argv = build_formal_argv(
        run_dir=args.run_dir,
        features=args.features,
        prices=args.prices,
        config_path=args.config,
    )
    try:
        rcqt_formal_runner.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
