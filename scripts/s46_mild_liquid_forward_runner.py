"""Freeze/settle the forward-only S46 mild-liquid rank paper strategy.

This wrapper intentionally exposes only the append-only forward lockbox.  It
does not provide a historical backtest mode because the ranking hypothesis was
formed after inspecting S46 winner/loser outcomes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import quiet_forward_shadow_runner as lockbox_runner

ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "configs/strategy/s46_mild_liquid_rank_v1.yaml"
MODEL = ROOT / "configs/model/disabled.yaml"
LOCKBOX = ROOT / "docs/council_20260828/S46_MILD_LIQUID_RANK_FORWARD"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "settle"), required=True)
    parser.add_argument("--asof", required=True, help="One completed exchange session")
    parser.add_argument("--execution-end", required=True, help="Source horizon used for the batch")
    parser.add_argument("--output", type=Path, default=LOCKBOX)
    args = parser.parse_args()
    args.strategy = STRATEGY
    args.model = MODEL
    print(lockbox_runner.run(args))


if __name__ == "__main__":
    main()
