"""Diagnose M5 confirmation quality without changing the preregistered strategy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from m5_confirmed_entry_v1_runner import _confirm_bars


def run(root: Path) -> dict[str, object]:
    candidates = pd.read_parquet(root / "candidate_view.parquet")
    candidates["asof"] = pd.to_datetime(candidates["asof"], utc=True).dt.normalize()
    daily = pd.read_parquet(root / "execution_daily.parquet")
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], utc=True).dt.normalize()
    daily = daily.sort_values(["trade_date", "ts_code"], kind="mergesort")
    minutes = pd.read_parquet(root / "execution_5min.parquet")
    minutes["trade_date"] = pd.to_datetime(minutes["trade_date"], utc=True).dt.normalize()
    minute_groups = {(day, str(code)): group for (day, code), group in minutes.groupby(["trade_date", "ts_code"], sort=False)}
    by_key = daily.set_index(["trade_date", "ts_code"])
    sessions = pd.DatetimeIndex(sorted(daily["trade_date"].unique()))
    rows: list[dict[str, object]] = []

    def lookup(day: pd.Timestamp, code: str):
        try:
            row = by_key.loc[(day, code)]
        except KeyError:
            return None
        return row if isinstance(row, pd.Series) else None

    for candidate in candidates.itertuples(index=False):
        asof = pd.Timestamp(candidate.asof)
        idx = sessions.get_indexer([asof])[0]
        code = str(candidate.ts_code)
        row = {"asof": asof, "ts_code": code, "candidate_rank": int(candidate.candidate_rank)}
        if idx < 0 or idx + 1 >= len(sessions):
            row["status"] = "NO_NEXT_SESSION"
            rows.append(row)
            continue
        entry_day = sessions[idx + 1]
        signal_row = lookup(asof, code)
        entry_row = lookup(entry_day, code)
        if signal_row is None or entry_row is None:
            row["status"] = "MISSING_DAILY"
            rows.append(row)
            continue
        ok, status, fill = _confirm_bars(minute_groups, entry_day, code,
                                         float(signal_row.economic_close), float(entry_row.economic_open))
        row["status"] = status
        if not ok:
            rows.append(row)
            continue
        row.update({"entry_day": entry_day, "entry_raw_open_1005": float(fill["open"]),
                    "entry_economic_open_1005": float(fill["economic_open"])})
        entry_eco = float(fill["economic_open"])
        h5_idx = idx + 1 + 5
        h10_idx = idx + 1 + 10
        if h10_idx >= len(sessions):
            row["status"] = "CONFIRMED_OUTCOME_INCOMPLETE"
            rows.append(row)
            continue
        h5_day, h10_day = sessions[h5_idx], sessions[h10_idx]
        h5_row, h10_row = lookup(h5_day, code), lookup(h10_day, code)
        if h5_row is None or h10_row is None:
            row["status"] = "CONFIRMED_OUTCOME_MISSING_DAILY"
            rows.append(row)
            continue
        h5_return = float(h5_row.economic_close) / entry_eco - 1.0
        if h5_return <= 0 and h5_idx + 1 < len(sessions):
            exit_row = lookup(sessions[h5_idx + 1], code)
            path_return = float(exit_row.economic_open) / entry_eco - 1.0 if exit_row is not None else np.nan
            exit_kind = "H5_NON_POSITIVE_NEXT_OPEN"
        else:
            path_return = float(h10_row.economic_close) / entry_eco - 1.0
            exit_kind = "H10"
        row.update({"h5_return": h5_return, "h10_return": float(h10_row.economic_close) / entry_eco - 1.0,
                    "path_return": path_return, "exit_kind": exit_kind})
        rows.append(row)

    frame = pd.DataFrame(rows)
    out = root / "diagnostics"
    out.mkdir(exist_ok=True)
    frame.to_parquet(out / "confirmation_outcomes.parquet", index=False)
    confirmed = frame[frame["status"].eq("confirmed")].copy()
    summary: dict[str, object] = {
        "candidate_count": int(len(frame)),
        "status_counts": {str(k): int(v) for k, v in frame["status"].value_counts().items()},
        "confirmed_count": int(len(confirmed)),
        "confirmed_rate": float(len(confirmed) / len(frame)) if len(frame) else None,
    }
    if not confirmed.empty:
        for col in ("h5_return", "h10_return", "path_return"):
            values = pd.to_numeric(confirmed[col], errors="coerce").dropna()
            summary[col] = {"mean": float(values.mean()), "median": float(values.median()),
                            "positive_rate": float((values > 0).mean()), "count": int(len(values))}
        gains = float(confirmed.loc[confirmed["path_return"] > 0, "path_return"].sum())
        losses = float(-confirmed.loc[confirmed["path_return"] < 0, "path_return"].sum())
        summary["path_profit_factor"] = gains / losses if losses else None
        summary["exit_kind_counts"] = {str(k): int(v) for k, v in confirmed["exit_kind"].value_counts().items()}
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().root.resolve()), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
