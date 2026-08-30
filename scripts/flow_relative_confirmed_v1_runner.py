"""Audited rule-only flow-relative backtest using the shared V3 runner."""
from __future__ import annotations

import argparse
from pathlib import Path

import quiet_confirmed_v3_runner as base_runner
from aistock9988.features.engine import build_flow_relative_feature_ledger


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/flow_relative_confirmed_v1.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "configs/model/disabled.yaml")
    parser.add_argument("--signal-start", default="2026-01-01")
    parser.add_argument("--signal-end", default="2026-08-06")
    parser.add_argument("--execution-end", default="2026-08-21")
    parser.add_argument("--run-name", default="S42_FLOW_RELATIVE_CONFIRMED_V1_2026")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/council_20260828/S42_FLOW_RELATIVE_CONFIRMED_V1_2026",
    )
    # The base runner owns the immutable run lifecycle and V3 execution
    # contract; only the feature provider is replaced for this experiment.
    base_runner.build_feature_ledger = build_flow_relative_feature_ledger
    print(base_runner.run(parser.parse_args()))


if __name__ == "__main__":
    main()
