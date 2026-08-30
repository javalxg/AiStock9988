"""Run the pre-registered RCQT hold/risk matrix against quant_db.

Credentials are read only by the child process from AISTOCK_DB_* environment
variables. Each variant receives its own immutable output directory.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-05-31")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    variants = (
        ("h10_stop_no_trailing", 10, False, True),
        ("h20_stop_no_trailing", 20, False, True),
        ("h10_no_stop_no_trailing", 10, True, True),
    )
    rows = []
    script = Path(__file__).with_name("rcqt_quantdb_sample_runner.py")
    for name, hold, no_stop, no_trailing in variants:
        out = args.output / name
        command = [sys.executable, str(script), "--output", str(out), "--limit", str(args.limit),
                   "--start", args.start, "--end", args.end, "--hold-sessions", str(hold)]
        if no_stop:
            command.append("--no-stop")
        if no_trailing:
            command.append("--no-trailing")
        completed = subprocess.run(command, env=os.environ.copy(), text=True)
        rows.append({"variant": name, "hold_sessions": hold, "no_stop": no_stop,
                     "no_trailing": no_trailing, "returncode": completed.returncode,
                     "output": str(out)})
        if completed.returncode != 0:
            raise SystemExit(f"RCQT matrix variant failed: {name}")
    (args.output / "MATRIX_MANIFEST.json").write_text(
        json.dumps({"variants": rows, "limit": args.limit, "start": args.start, "end": args.end},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
