"""Compare RCQT selected pre-state features with historical +30% event states."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

EVENTS = Path("/Users/lxg/quant/deltafstation/research/experiments/big_move_event_profile_3yr/outputs/events_all.csv")

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--selection", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); a = ap.parse_args()
    selected = pd.read_csv(a.selection)
    selected = selected[selected["selected"].astype(str).str.lower().eq("true")].copy()
    events = pd.read_csv(EVENTS)
    up = events[events["event_group"].astype(str).str.lower().str.contains("up")].copy()
    rows = []
    for name, frame in (("rcqt_selected", selected), ("historical_up30_event_prestate", up)):
        values = {"population": name, "count": int(len(frame))}
        for col in ("dist_ma60", "ret20", "ret60", "dd20", "dd60", "vol20", "confirmation_strength"):
            if col in frame:
                values[col] = float(pd.to_numeric(frame[col], errors="coerce").mean())
        for col in ("dist_ma60_m1", "kdj_k_bfq_m1", "cci_bfq_m1", "wr_bfq_m1", "mkt_ret_20d", "mfe20", "mae20", "first_up_day"):
            if col in frame:
                values[col] = float(pd.to_numeric(frame[col], errors="coerce").mean())
        rows.append(values)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"source_event_file": str(EVENTS), "rows": rows, "note": "Different populations; descriptive comparison, not causal attribution."}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__": main()
