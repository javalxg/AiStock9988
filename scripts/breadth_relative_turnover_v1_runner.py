"""Run the single pre-registered breadth-relative rule diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path

import quiet_confirmed_v3_runner as base_runner
from aistock9988.features.engine import build_breadth_relative_feature_ledger


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/breadth_relative_turnover_v1.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "configs/model/disabled.yaml")
    parser.add_argument("--signal-start", default="2025-01-01")
    parser.add_argument("--signal-end", default="2026-07-31")
    parser.add_argument("--execution-end", default="2026-08-21")
    parser.add_argument("--run-name", default="BREADTH_RELATIVE_TURNOVER_V1_DIAGNOSTIC")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs/council_20260828/BREADTH_RELATIVE_TURNOVER_V1_DIAGNOSTIC",
    )
    base_runner.build_feature_ledger = build_breadth_relative_feature_ledger
    print(base_runner.run(parser.parse_args()))


if __name__ == "__main__":
    main()
