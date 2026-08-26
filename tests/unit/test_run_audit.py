import json

import pytest

from aistock9988.audit.run import RunAuditError, audit_run, write_audit_report


def _run_dir(tmp_path):
    run_dir = tmp_path / "experiments" / ".running" / "run-1"
    for name in ("models", "predictions", "selections", "trades", "diagnostics", "logs"):
        (run_dir / name).mkdir(parents=True)
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({"run_id": "run-1", "status": "CREATED"}))
    (run_dir / "data_manifest.json").write_text("{}\n")
    (run_dir / "models" / "model.json").write_text("{}\n")
    (run_dir / "predictions" / "ledger.csv").write_text("asof,ts_code\n2026-01-01,A\n")
    (run_dir / "selections" / "topk.csv").write_text("asof,ts_code\n2026-01-01,A\n")
    (run_dir / "trades" / "fills.csv").write_text("order_id,status\n1,FILLED\n")
    (run_dir / "diagnostics" / "checks.json").write_text("{}\n")
    return run_dir


def test_run_audit_writes_immutable_report(tmp_path):
    run_dir = _run_dir(tmp_path)
    report = audit_run(run_dir)
    path = write_audit_report(run_dir, report)
    assert path.exists() and report["artifact_count"] >= 3
    with pytest.raises(FileExistsError):
        write_audit_report(run_dir, report)


def test_run_audit_rejects_partial_run(tmp_path):
    run_dir = tmp_path / "experiments" / ".running" / "partial"
    run_dir.mkdir(parents=True)
    with pytest.raises(RunAuditError, match="missing required"):
        audit_run(run_dir)
