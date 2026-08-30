import json

import pytest

from aistock9988.audit.run import RunAuditError, audit_run, write_audit_report
from aistock9988 import cli
from scripts.rcqt_formal_runner import _assert_untouched_run_dir, _resolve_run_dir


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
    assert json.loads((run_dir / "RUN_STATUS.json").read_text())["status"] == "VERIFIED"
    completed = cli.complete_run(run_dir)
    assert completed == tmp_path / "experiments" / "completed" / "run-1"
    assert completed.is_dir()
    status = json.loads((completed / "RUN_STATUS.json").read_text())
    assert status["status"] == "COMPLETED"
    assert status["audit_artifact_count"] == verified["artifact_count"]
    assert cli.reverify_run(completed) == verified


def test_complete_requires_explicit_verify(tmp_path, monkeypatch):
    run_dir = _run_dir(tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    with pytest.raises(RunAuditError, match="verify-run"):
        cli.complete_run(run_dir)


def test_reverify_rejects_completed_artifact_tampering(tmp_path, monkeypatch):
    run_dir = _run_dir(tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    cli.verify_run(run_dir)
    completed = cli.complete_run(run_dir)
    (completed / "data" / "snapshot.parquet").write_bytes(b"tampered")
    with pytest.raises(RunAuditError, match="no longer match"):
        cli.reverify_run(completed)


def test_reverify_rejects_stable_status_contract_tampering(tmp_path, monkeypatch):
    run_dir = _run_dir(tmp_path)
    status_path = run_dir / "RUN_STATUS.json"
    status = json.loads(status_path.read_text())
    status["source_mode"] = "frozen_snapshot"
    status_path.write_text(json.dumps(status))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    cli.verify_run(run_dir)
    completed = cli.complete_run(run_dir)
    status_path = completed / "RUN_STATUS.json"
    status = json.loads(status_path.read_text())
    status["source_mode"] = "mutated"
    status_path.write_text(json.dumps(status))
    with pytest.raises(RunAuditError, match="no longer match"):
        cli.reverify_run(completed)


def test_reverify_missing_audit_report_is_audit_error(tmp_path, monkeypatch):
    run_dir = _run_dir(tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    cli.verify_run(run_dir)
    completed = cli.complete_run(run_dir)
    (completed / "diagnostics" / "audit.json").unlink()
    with pytest.raises(RunAuditError, match="audit report is missing"):
        cli.reverify_run(completed)


def test_formal_runner_rejects_preseeded_evidence(tmp_path):
    run_dir = tmp_path / "experiments" / ".running" / "run-preseeded"
    for name in ("data", "models", "predictions", "selections", "trades", "diagnostics", "logs"):
        (run_dir / name).mkdir(parents=True)
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({"run_id": run_dir.name, "status": "CREATED"}))
    (run_dir / "predictions" / "old.csv").write_text("asof,ts_code,score\n2026-01-01,A,1\n")
    with pytest.raises(SystemExit, match="predictions/ is not empty"):
        _assert_untouched_run_dir(run_dir)


def test_formal_runner_rejects_unexpected_root_entry(tmp_path):
    run_dir = tmp_path / "experiments" / ".running" / "run-extra"
    for name in ("data", "models", "predictions", "selections", "trades", "diagnostics", "logs"):
        (run_dir / name).mkdir(parents=True)
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({"run_id": run_dir.name, "status": "CREATED"}))
    (run_dir / "injected_manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="unexpected entries: injected_manifest.json"):
        _assert_untouched_run_dir(run_dir)


def test_formal_runner_rejects_commands_directory(tmp_path):
    run_dir = tmp_path / "experiments" / ".running" / "run-commands-dir"
    for name in ("data", "models", "predictions", "selections", "trades", "diagnostics", "logs"):
        (run_dir / name).mkdir(parents=True)
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({"run_id": run_dir.name, "status": "CREATED"}))
    (run_dir / "commands.sh").mkdir()
    with pytest.raises(SystemExit, match="commands.sh to be a regular file"):
        _assert_untouched_run_dir(run_dir)


def test_formal_runner_rejects_symlinked_run_path(tmp_path):
    target = tmp_path / "experiments" / ".running" / "real-run"
    target.mkdir(parents=True)
    link = tmp_path / "experiments" / ".running" / "link-run"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(SystemExit, match="symlinked --run-dir"):
        _resolve_run_dir(link)


def test_formal_runner_rejects_run_outside_project_root(tmp_path, monkeypatch):
    import scripts.rcqt_formal_runner as formal

    project = tmp_path / "project"
    outside = tmp_path / "outside" / "experiments" / ".running" / "run"
    outside.mkdir(parents=True)
    monkeypatch.setattr(formal, "ROOT", project)
    with pytest.raises(SystemExit, match="under ROOT/experiments/.running"):
        _resolve_run_dir(outside)


def test_run_audit_rejects_non_finite_nav(tmp_path):
    run_dir = _run_dir(tmp_path)
    (run_dir / "trades" / "nav.csv").write_text(
        "trade_date,cash,market_value,nav\n2026-01-01,NaN,10,NaN\n"
    )
    with pytest.raises(RunAuditError, match="NAV accounting identity"):
        audit_run(run_dir)


def test_run_audit_rejects_duplicate_ledger_keys(tmp_path):
    run_dir = _run_dir(tmp_path)
    (run_dir / "predictions" / "ledger.csv").write_text(
        "asof,ts_code,score\n2026-01-01,A,1\n2026-01-01,A,2\n"
    )
    with pytest.raises(RunAuditError, match="prediction ledger has duplicate"):
        audit_run(run_dir)


def test_run_audit_rejects_infinite_fill_values(tmp_path):
    run_dir = _run_dir(tmp_path)
    (run_dir / "trades" / "fills.csv").write_text(
        "order_id,status,side,price,shares\n1,FILLED,BUY,inf,1\n"
    )
    with pytest.raises(RunAuditError, match="fills ledger has invalid"):
        audit_run(run_dir)
