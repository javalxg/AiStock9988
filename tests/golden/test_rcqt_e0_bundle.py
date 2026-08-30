import json
import sys

import pandas as pd
import pytest

from aistock9988 import cli
from aistock9988.audit.run import RunAuditError
from scripts import rcqt_formal_runner
from scripts import r2e_s1a_formal_runner


def _seed_rule_run(root, run_id):
    run_dir = root / "experiments" / ".running" / run_id
    for name in ("data", "models", "predictions", "selections", "trades", "diagnostics", "logs"):
        (run_dir / name).mkdir(parents=True)
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({
        "run_id": run_id,
        "status": "RUNNING",
        "strategy_type": "rules",
        "source_mode": "frozen_snapshot",
        "git": {"commit": "fixture", "status": "clean"},
    }) + "\n")
    (run_dir / "data_manifest.json").write_text(json.dumps({
        "bound_inputs": {"features": {"raw_sha256": "fixture"}},
    }) + "\n")
    (run_dir / "data" / "snapshot.csv").write_text("asof,value\n2026-01-02,1\n")
    (run_dir / "predictions" / "scores.csv").write_text(
        "asof,ts_code,score\n2026-01-02,A,1.0\n"
    )
    (run_dir / "selections" / "selection.csv").write_text(
        "asof,ts_code,candidate_rank,selected,selection_decision_id,policy_id,target_weight,context_hash\n"
        "2026-01-02,A,1,True,d1,p1,0.1,h1\n"
    )
    (run_dir / "trades" / "fills.csv").write_text(
        "order_id,side,price,shares\n1,BUY,10,100\n"
    )
    (run_dir / "trades" / "nav.csv").write_text(
        "trade_date,cash,market_value,nav\n2026-01-02,9000,1000,10000\n"
    )
    (run_dir / "diagnostics" / "checks.json").write_text("{}\n")
    return run_dir


def test_e0_bundle_seals_completes_and_rejects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    run_dir = _seed_rule_run(tmp_path, "run-a")

    verified = cli.verify_run(run_dir)
    completed = cli.complete_run(run_dir)
    assert cli.reverify_run(completed) == verified

    (completed / "predictions" / "scores.csv").write_text(
        "asof,ts_code,score\n2026-01-02,A,999.0\n"
    )
    with pytest.raises(RunAuditError, match="no longer match"):
        cli.reverify_run(completed)


def test_e0_independent_runs_have_identical_normalized_ledger_hashes(tmp_path):
    reports = []
    for run_id in ("run-a", "run-b"):
        report = cli.verify_run(_seed_rule_run(tmp_path, run_id))
        reports.append({
            path: record["sha256"]
            for path, record in report["artifacts"].items()
            if path.startswith(("predictions/", "selections/", "trades/"))
        })
    assert reports[0] == reports[1]


def test_formal_runner_consumes_initialized_directory_and_audits_last(tmp_path, monkeypatch):
    run_dir = tmp_path / "experiments" / ".running" / "run-formal"
    for name in ("data", "models", "predictions", "selections", "trades", "diagnostics", "logs"):
        (run_dir / name).mkdir(parents=True)
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({
        "run_id": "run-formal", "status": "CREATED", "git": {"commit": "fixture", "status": "clean"},
    }) + "\n")

    features_path = tmp_path / "features.csv"
    base = {
        "asof": "2026-08-20",
        "available_time": "2026-08-20T07:00:00Z",
        "dist_ma60": -0.02,
        "ret20": -0.05,
        "ret60": 0.10,
        "dd20": -0.05,
        "dd60": -0.20,
        "vol20": 0.10,
        "liq20": 3.0,
        "volume_ratio_20": 1.0,
        "close": 10.0,
        "ma5": 9.0,
        "prev3_high": 9.5,
        "ret1": 0.01,
    }
    pd.DataFrame([dict(base, ts_code=code) for code in ("A", "B", "C")]).to_csv(features_path, index=False)

    price_rows = []
    for date, price in (("2026-08-20", 10.0), ("2026-08-21", 11.0), ("2026-08-24", 12.0)):
        for code in ("A", "B", "C"):
            price_rows.append({
                "trade_date": date,
                "ts_code": code,
                "raw_open": price,
                "raw_high": price,
                "raw_low": price,
                "raw_close": price,
                "economic_open": price,
                "economic_high": price,
                "economic_low": price,
                "economic_close": price,
                "adj_factor": 1.0,
                "open_available_time": f"{date}T01:30:00Z",
                "close_available_time": f"{date}T07:00:00Z",
                "available_time": f"{date}T07:00:00Z",
                "is_suspended": False,
                "is_limit_up": False,
                "is_limit_down": False,
                "adv20": 1_000_000.0,
            })
    prices_path = tmp_path / "prices.csv"
    pd.DataFrame(price_rows).to_csv(prices_path, index=False)

    def no_database():
        raise RuntimeError("fixture has no database")

    monkeypatch.setattr(rcqt_formal_runner, "connection_kwargs", no_database)
    # The formal entrypoint constrains runs to ROOT/experiments/.running;
    # point its project root at this isolated fixture repository.
    monkeypatch.setattr(rcqt_formal_runner, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "r2e_s1a_formal_runner",
        "--run-dir", str(run_dir),
        "--features", str(features_path),
        "--prices", str(prices_path),
    ])
    r2e_s1a_formal_runner.main()

    status = json.loads((run_dir / "RUN_STATUS.json").read_text())
    manifest = json.loads((run_dir / "data_manifest.json").read_text())
    audit = json.loads((run_dir / "diagnostics" / "audit.json").read_text())
    assert status["status"] == "VERIFIED"
    assert status["experiment_id"] == "r2e_s1a_reset_setup_v1"
    assert set(manifest["bound_inputs"]) == {"features", "prices", "experiment_config"}
    assert manifest["features"] == "data/inputs/features.csv"
    assert manifest["selection_contract"]["require_right_confirmation"] is False
    assert manifest["selection_contract"]["quiet_slots"] == 0
    assert manifest["execution_contract"]["hold_sessions"] == 10
    assert manifest["execution_contract"]["trailing"] is False
    selections = pd.read_csv(run_dir / "selections" / "selection_ledger.csv")
    assert set(selections["sleeve"]) == {"recovery"}
    assert cli.verify_run(run_dir) == audit
