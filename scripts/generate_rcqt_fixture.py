"""Generate a tiny deterministic RCQT execution-contract fixture."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2026-08-20", periods=12, freq="B")
    codes = ["000001.SZ", "000002.SZ", "600001.SH"]
    feat = []
    for d in days:
        for i, c in enumerate(codes):
            feat.append({"asof": d.strftime("%Y-%m-%d"), "ts_code": c,
                "dist_ma60": -.02 + .02*i, "ret20": -.05 + .04*i, "ret60": .10 + .05*i,
                "dd20": -.05, "dd60": -.20 + .03*i, "vol20": .10 + .01*i,
                "liq20": 3-i, "volume_ratio_20": 1., "close": 10., "ma5": 9.,
                "prev3_high": 9.5, "ret1": .01, "available_time": d.strftime("%Y-%m-%dT07:00:00Z")})
    pd.DataFrame(feat).to_csv(a.output / "features.csv", index=False)
    px = []
    for j, d in enumerate(days):
        for c in codes:
            px.append({"trade_date": d.strftime("%Y-%m-%d"), "ts_code": c,
                "raw_open": 10+j*.02, "raw_high": 10+j*.02, "raw_low": 10+j*.02,
                "raw_close": 10+j*.02, "economic_open": 10+j*.02,
                "economic_high": 10+j*.02, "economic_low": 10+j*.02,
                "economic_close": 10+j*.02, "adj_factor": 1.,
                "available_time": d.strftime("%Y-%m-%dT15:00:00Z"),
                "open_available_time": d.strftime("%Y-%m-%dT00:00:00Z"),
                "close_available_time": d.strftime("%Y-%m-%dT15:00:00Z"),
                "is_suspended": False, "is_limit_up": False, "is_limit_down": False})
    pd.DataFrame(px).to_csv(a.output / "prices.csv", index=False)

if __name__ == "__main__": main()
