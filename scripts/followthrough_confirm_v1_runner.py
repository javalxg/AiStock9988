"""Run the preregistered T+1 follow-through confirmation diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path

import quiet_confirmed_v3_runner as base_runner
from aistock9988.features.followthrough import build_followthrough_feature_ledger


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/followthrough_confirm_v1.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "configs/model/disabled.yaml")
    parser.add_argument("--signal-start", default="2025-01-01")
    parser.add_argument("--signal-end", default="2026-07-31")
    parser.add_argument("--execution-end", default="2026-08-28")
    parser.add_argument("--run-name", default="FOLLOWTHROUGH_CONFIRM_V1_DIAGNOSTIC")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs/council_20260828/FOLLOWTHROUGH_CONFIRM_V1_DIAGNOSTIC",
    )
    base_runner.build_feature_ledger = build_followthrough_feature_ledger
    print(base_runner.run(parser.parse_args()))


if __name__ == "__main__":
    main()
