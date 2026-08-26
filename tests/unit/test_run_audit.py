import json

import pytest

from aistock9988.audit.run import RunAuditError, audit_run, write_audit_report
from aistock9988 import cli


def _run_dir(tmp_path):
    run_dir = tmp_path / "experiments" / ".running" / "run-1"
    for name in ("data", "models", "predictions", "selections", "trades", "diagnostics", "logs"):
        (run_dir / name).mkdir(parents=True)
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({"run_id": "run-1", "status": "CREATED"}))
    (run_dir / "data_manifest.json").write_text("{}\n")
    (run_dir / "data" / "snapshot.parquet").write_bytes(b"snapshot")
    (run_dir / "models" / "model.json").write_text("{}\n")
    (run_dir / "predictions" / "ledger.csv").write_text("asof,ts_code\n2026-01-01,A\n")
    (run_dir / "selections" / "topk.csv").write_text(
        "asof,ts_code,candidate_rank,selected,selection_decision_id,policy_id,target_weight,context_hash\n"
        "2026-01-01,A,1,True,d1,p1,1.0,h1\n"
    )
    (run_dir / "trades" / "fills.csv").write_text("order_id,status,side,price,shares\n1,FILLED,BUY,10,1\n")
    (run_dir / "diagnostics" / "checks.json").write_text("{}\n")
    return run_dir


def test_run_audit_writes_immutable_report(tmp_path):
    run_dir = _run_dir(tmp_path)
    report = audit_run(run_dir)
    path = write_audit_report(run_dir, report)
    assert path.exists() and report["artifact_count"] >= 3
    assert write_audit_report(run_dir, report) == path
    second_report = audit_run(run_dir)
    assert second_report == report
    assert write_audit_report(run_dir, second_report) == path


def test_run_audit_rejects_partial_run(tmp_path):
    run_dir = tmp_path / "experiments" / ".running" / "partial"
    run_dir.mkdir(parents=True)
    with pytest.raises(RunAuditError, match="missing required"):
        audit_run(run_dir)


def test_verify_then_complete_reuses_same_audit_report(tmp_path, monkeypatch):
    run_dir = _run_dir(tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    verified = cli.verify_run(run_dir)
    completed = cli.complete_run(run_dir)
    assert completed == tmp_path / "experiments" / "completed" / "run-1"
    assert completed.is_dir()
    status = json.loads((completed / "RUN_STATUS.json").read_text())
    assert status["status"] == "COMPLETED"
    assert status["audit_artifact_count"] == verified["artifact_count"]
