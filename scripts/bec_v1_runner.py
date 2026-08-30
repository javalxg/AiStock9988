"""Run the independent BEC-V1 breadth-expansion continuation diagnostic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

try:  # direct script execution puts ``scripts`` on sys.path
    import quiet_confirmed_v3_runner as base_runner
except ModuleNotFoundError:  # unit tests import the namespace package
    from scripts import quiet_confirmed_v3_runner as base_runner
from aistock9988.features.bec import build_bec_feature_ledger


ROOT = Path(__file__).resolve().parents[1]
_BASE_ACCEPTANCE = base_runner._acceptance


def _bec_acceptance(metrics, strategy):
    """Apply the BEC preregistered portfolio checks to each cost arm."""
    # Keep a stable reference before monkeypatching the shared runner below;
    # calling base_runner._acceptance here would recurse into this wrapper.
    result = _BASE_ACCEPTANCE(metrics, strategy)
    tests = result["tests"]
    tests["excluding_top3_profit"] = float(metrics.get("return_excluding_top3_profit", 0.0)) > float(
        strategy.acceptance.get("return_excluding_top3_profit_min_exclusive", 0.0)
    )
    tests["minimum_closed_trades"] = int(metrics.get("trade_count", 0)) >= int(
        strategy.acceptance.get("minimum_closed_trades", 0)
    )
    if str(metrics.get("scenario", "")) == "stress":
        tests["stress_return_positive"] = float(metrics.get("total_return", 0.0)) > float(
            strategy.acceptance.get("stress_return_min_exclusive", 0.0)
        )
    result["passed"] = all(tests.values())
    return result


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _augment_audit_ledgers(output: Path) -> None:
    features = pd.read_parquet(output / "ledgers" / "feature_ledger.parquet")
    industry_audit = features.attrs.get("industry_audit", [])
    if industry_audit:
        pd.DataFrame(industry_audit).to_parquet(
            output / "ledgers" / "industry_membership_audit.parquet", index=False
        )
    candidate = pd.read_parquet(output / "ledgers" / "candidate_ledger.parquet")
    candidate_columns = [
        "asof", "ts_code", "stage1_pass", "candidate_rank", "candidate_status",
        "candidate_snapshot_id", "score_rejection_reason",
    ]
    event = features.merge(
        candidate[candidate_columns], on=["asof", "ts_code"], how="left", validate="one_to_one"
    )
    event["record_type"] = "SELECTION"
    event["event_status"] = event["candidate_status"].fillna("NOT_SCORED")
    event["event_rejection_reason"] = event["score_rejection_reason"].fillna(
        event["feature_rejection_reason"]
    )
    execution_records: list[pd.DataFrame] = []
    for scenario in ("base", "stress"):
        decisions = pd.read_parquet(output / "backtests" / scenario / "execution_decisions.parquet")
        fills = pd.read_parquet(output / "backtests" / scenario / "fills.parquet")
        position_events = pd.read_parquet(output / "backtests" / scenario / "position_events.parquet")
        orders = pd.read_parquet(output / "backtests" / scenario / "orders.parquet")
        records: list[pd.DataFrame] = []
        for record_type, frame in (
            ("EXECUTION_DECISION", decisions), ("ORDER", orders),
            ("FILL", fills), ("POSITION_EVENT", position_events),
        ):
            if frame.empty:
                continue
            item = frame.copy()
            item.insert(0, "record_type", record_type)
            item.insert(0, "scenario", scenario)
            records.append(item)
        audit = pd.concat(records, ignore_index=True, sort=False) if records else pd.DataFrame()
        audit.to_parquet(output / "backtests" / scenario / "execution_audit.parquet", index=False)
        if not audit.empty:
            execution_records.append(audit)

    # Keep a full-market selection ledger while appending every execution,
    # fill, and exit event in the same immutable artifact for audit joins.
    if execution_records:
        execution_event = pd.concat(execution_records, ignore_index=True, sort=False)
        event = pd.concat([event, execution_event], ignore_index=True, sort=False)
    event.to_parquet(output / "ledgers" / "event_ledger.parquet", index=False)

    code_manifest_path = output / "manifests" / "code_manifest.json"
    manifest = json.loads(code_manifest_path.read_text(encoding="utf-8"))
    for path in (ROOT / "src/aistock9988/features/bec.py", Path(__file__), ROOT / "configs/strategy/bec_v1.yaml"):
        manifest[str(path.relative_to(ROOT))] = _sha(path)
    code_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_path = output / "manifests" / "artifact_manifest.json"
    artifacts = {
        str(path.relative_to(output)): {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    artifact_path.write_text(json.dumps(artifacts, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args):
    base_runner.build_feature_ledger = build_bec_feature_ledger
    base_runner._acceptance = _bec_acceptance
    output = base_runner.run(args)
    _augment_audit_ledgers(output)
    return output


def main() -> None:
    parser = base_runner.argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/bec_v1.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "configs/model/disabled.yaml")
    parser.add_argument("--signal-start", default="2026-01-01")
    parser.add_argument("--signal-end", default="2026-08-06")
    parser.add_argument("--execution-end", default="2026-08-21")
    parser.add_argument("--run-name", default="BEC_BREADTH_EXPANSION_CONTINUATION_V1_DIAGNOSTIC")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs/council_20260828/BEC_BREADTH_EXPANSION_CONTINUATION_V1_DIAGNOSTIC",
    )
    parser.add_argument("--diagnostic-history", action="store_true")
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
