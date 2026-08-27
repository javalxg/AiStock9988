import json
import importlib.util
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from aistock9988.audit.run import audit_run
from aistock9988.features.registry import FeatureSet
from aistock9988.selection.delta_compatible import DynamicGateResult
SPEC = importlib.util.spec_from_file_location(
    "q70_delta_compatible_runner", Path(__file__).resolve().parents[2] / "scripts/q70_delta_compatible_runner.py"
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_delta_runner_writes_auditable_artifact_bundle(tmp_path, monkeypatch):
    feature_set = FeatureSet.from_f0_json(runner.ROOT / "configs/feature_sets/f0_123_columns.json")
    sessions = pd.date_range("2025-01-02", "2026-08-14", freq="B", tz="UTC")
    panel_rows = []
    for day in sessions:
        for code in ("A.SZ", "B.SZ"):
            row = {"ts_code": code, "event_time": day, "available_time": day + pd.Timedelta(hours=6),
                   "economic_open": 10.0}
            row.update({column: 1.0 for column in feature_set.columns})
            panel_rows.append(row)
    panel = pd.DataFrame(panel_rows)
    context = pd.DataFrame([
        {"ts_code": code, "trade_date": day, "raw_close": 10.0, "pct_chg": 0.0,
         "amount": 1000.0, "available_time": day + pd.Timedelta(hours=6),
         "is_limit_up": False, "is_limit_down": False}
        for day in sessions for code in ("A.SZ", "B.SZ")
    ])
    execution_days = sessions[sessions >= pd.Timestamp("2026-01-01", tz="UTC")]
    prices = pd.DataFrame([
        {"ts_code": code, "trade_date": day, "raw_open": 10.0, "raw_high": 10.0,
         "raw_low": 10.0, "raw_close": 10.0, "economic_open": 10.0,
         "economic_high": 10.0, "economic_low": 10.0, "economic_close": 10.0,
         "adj_factor": 1.0, "available_time": day + pd.Timedelta(hours=7),
         "open_available_time": day + pd.Timedelta(hours=1),
         "close_available_time": day + pd.Timedelta(hours=7),
         "is_suspended": False, "is_limit_up": False, "is_limit_down": False}
        for day in execution_days for code in ("A.SZ", "B.SZ")
    ])
    actions = pd.DataFrame(columns=["ts_code", "ex_date", "split_ratio", "cash_dividend", "available_time"])
    monkeypatch.setattr(runner, "load_f0_panel", lambda *args, **kwargs: panel)
    monkeypatch.setattr(runner, "load_market_context_panel", lambda *args, **kwargs: context)
    monkeypatch.setattr(runner, "load_execution_panel", lambda *args, **kwargs: prices)
    monkeypatch.setattr(runner, "load_corporate_actions", lambda *args, **kwargs: actions)
    monkeypatch.setattr(runner, "model_for_prediction", lambda *args, **kwargs: np.array([2.0, 1.0]))

    def fake_train(*args, **kwargs):
        run_dir = args[3]
        cutoff = pd.Timestamp(args[4])
        model_id = f"q70_delta_{cutoff:%Y%m%d}"
        (run_dir / "models").mkdir(exist_ok=True)
        (run_dir / "models" / f"{model_id}.json").write_text("{}\n")
        (run_dir / "models" / f"{model_id}.metadata.json").write_text(json.dumps({"model_id": model_id}) + "\n")
        gate = DynamicGateResult(
            factor="dmi_adx_bfq", threshold=None, active=False, sample_count=2,
            lower_quantile=None, upper_quantile=None, lower_tail_mean=None,
            upper_tail_mean=None, reason="insufficient_mature_samples",
        )
        return SimpleNamespace(model_id=model_id), panel, pd.DataFrame(), gate

    monkeypatch.setattr(runner, "_train", fake_train)
    run_dir = tmp_path / "experiments" / ".running" / "audit-run"
    for directory in ("data", "models", "predictions", "selections", "trades", "diagnostics", "logs"):
        (run_dir / directory).mkdir(parents=True)
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({"run_id": "audit-run", "status": "CREATED"}) + "\n")

    result = runner.run(run_dir=run_dir, config_path=runner.ROOT / "configs/experiments/q70_delta_compatible_v1.yaml")
    assert result["status"] == "executed"
    for path in (
        "data_manifest.json", "code_manifest.json", "diagnostics/metrics.json",
        "diagnostics/dynamic_gate_audit.json", "trades/orders.csv", "trades/fills.csv",
        "trades/nav.csv", "trades/positions.csv",
    ):
        assert (run_dir / path).is_file(), path
    assert list((run_dir / "predictions").glob("*.csv"))
    assert list((run_dir / "selections").glob("*.csv"))
    assert list((run_dir / "models").glob("*.metadata.json"))
    report = audit_run(run_dir)
    assert report["artifact_count"] > 0
