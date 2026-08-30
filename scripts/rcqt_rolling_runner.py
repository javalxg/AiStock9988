"""Run fixed RCQT baseline across pre-registered rolling validation windows."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

WINDOWS = (("2025", "2025-01-01", "2025-12-31"), ("2026_ytd", "2026-01-01", "2026-05-31"), ("crossyear", "2025-01-01", "2026-05-31"))

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True); p.add_argument("--limit", type=int, default=1000); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True); rows = []
    for name, start, end in WINDOWS:
        out = a.output / name
        cmd = [sys.executable, "scripts/rcqt_quantdb_sample_runner.py", "--output", str(out), "--limit", str(a.limit), "--start", start, "--end", end, "--hold-sessions", "10", "--no-trailing", "--max-order-to-adv20", "0.02", "--slippage", "0.001"]
        rc = subprocess.run(cmd, check=False).returncode
        metrics = out / "metrics.json"
        row = {"window": name, "start": start, "end": end, "returncode": rc}
        if metrics.exists():
            data = json.loads(metrics.read_text()); row.update({k: data.get(k) for k in ("total_return", "max_drawdown", "portfolio_profit_factor", "sharpe", "weekly_target_hit_ratio")})
        rows.append(row)
    (a.output / "ROLLING_MANIFEST.json").write_text(json.dumps({"limit": a.limit, "windows": rows}, ensure_ascii=False, indent=2) + "\n")

if __name__ == "__main__": main()
