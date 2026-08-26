import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aistock9988.features.registry import FeatureSet


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("q70_source_parity_runner", ROOT / "scripts/q70_source_parity_runner.py")
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_train_one_uses_rolling_window_and_models_directory(tmp_path):
    dates = pd.to_datetime([
        "2024-01-02", "2024-01-02",
        "2025-06-02", "2025-06-02",
        "2025-12-01", "2025-12-01",
    ], utc=True)
    panel = pd.DataFrame({
        "ts_code": ["A", "B"] * 3, "event_time": dates,
        "available_time": dates + pd.Timedelta(hours=6),
        "f1": [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
    })
    labels = pd.DataFrame({
        "ts_code": ["A", "B"] * 3, "event_time": dates,
        "available_time": dates + pd.Timedelta(days=12, hours=2),
        "label_return": [0.1, -0.1, 0.2, -0.2, 0.3, -0.3],
    })
    artifact = runner._train_one(
        panel, labels, FeatureSet.create("feature.test.v1", ["f1"]), tmp_path,
        12, pd.Timestamp("2025-12-31", tz="UTC"), "model-test",
        {"n_estimators": 1, "max_depth": 1, "n_jobs": 1},
    )
    assert artifact.row_count == 4
    assert (tmp_path / "models/model-test.json").is_file()
    assert not (tmp_path / "model-test.json").exists()


def test_weekly_dates_use_last_session_and_require_t10_maturity():
    sessions = pd.date_range("2026-01-05", "2026-02-20", freq="B", tz="UTC")
    weekly = runner._weekly_signal_dates(sessions, start="2026-01-05", end="2026-02-13")
    assert weekly[0] == pd.Timestamp("2026-01-09", tz="UTC")
    assert all(day == sessions[sessions.to_series().dt.isocalendar().week.eq(day.isocalendar().week)].max()
               for day in weekly)
    mature = runner._mature_signal_dates(
        sessions, weekly, entry_delay_sessions=1, horizon_sessions=10, mature_end="2026-02-13",
    )
    positions = {day: index for index, day in enumerate(sessions)}
    assert mature
    assert all(sessions[positions[day] + 11] <= pd.Timestamp("2026-02-13", tz="UTC") for day in mature)


def test_configured_runner_honors_formal_boundary_and_writes_immutable_bundle(tmp_path, monkeypatch):
    feature_set = FeatureSet.from_f0_json(ROOT / "configs/feature_sets/f0_123_columns.json")
    sessions = pd.date_range("2025-01-02", "2026-08-14", freq="B", tz="UTC")
    rows = []
    for day in sessions:
        for code in ("A.SZ", "B.SZ"):
            row = {"ts_code": code, "event_time": day,
                   "available_time": day + pd.Timedelta(hours=6),
                   "economic_open": 10.0, "economic_close": 10.0}
            row.update({column: 1.0 for column in feature_set.columns})
            rows.append(row)
    panel = pd.DataFrame(rows)

    context_rows = []
    for day in sessions:
        for code in ("A.SZ", "B.SZ"):
            context_rows.append({"ts_code": code, "trade_date": day, "raw_close": 10.0,
                                 "pct_chg": 0.2, "amount": 1000.0,
                                 "available_time": day + pd.Timedelta(hours=6),
                                 "is_limit_up": False, "is_limit_down": False})
    context = pd.DataFrame(context_rows)

    formal_sessions = sessions[(sessions >= pd.Timestamp("2026-01-01", tz="UTC")) &
                               (sessions <= pd.Timestamp("2026-07-31", tz="UTC"))]
    prices = pd.DataFrame([
        {"ts_code": code, "trade_date": day, "raw_open": 10.0, "raw_high": 10.0,
         "raw_low": 10.0, "raw_close": 10.0, "economic_open": 10.0,
         "economic_high": 10.0, "economic_low": 10.0, "economic_close": 10.0,
         "adj_factor": 1.0, "available_time": day + pd.Timedelta(hours=7),
         "open_available_time": day + pd.Timedelta(hours=1, minutes=30),
         "close_available_time": day + pd.Timedelta(hours=7),
         "is_suspended": False, "is_limit_up": False, "is_limit_down": False}
        for day in formal_sessions for code in ("A.SZ", "B.SZ")
    ])
    actions = pd.DataFrame(columns=["ts_code", "ex_date", "split_ratio", "cash_dividend",
                                    "available_time", "action_type"])
    minutes = pd.DataFrame({"ts_code": ["A.SZ"], "trade_time": ["2026-01-12T01:35:00Z"],
                            "available_time": ["2026-01-12T01:35:00Z"], "open": [10.0]})

    monkeypatch.setattr(runner, "load_f0_panel", lambda *args, **kwargs: (panel, {"coverage": 1.0}))
    monkeypatch.setattr(runner, "load_market_context_panel", lambda start, end: context)
    monkeypatch.setattr(runner, "load_execution_panel", lambda start, end, ts_codes: prices)
    monkeypatch.setattr(runner, "load_corporate_actions", lambda start, end, ts_codes: actions)
    monkeypatch.setattr(runner, "model_for_prediction", lambda path, values: np.arange(len(values), dtype=float)[::-1])
    monkeypatch.setattr("aistock9988.data.minute_source.load_minute_execution_panel",
                        lambda start, end, freq, ts_codes: minutes)

    trained = []

    def fake_train(panel_arg, labels_arg, spec_arg, run_dir, window, cutoff, model_id, params):
        assert window == 12
        trained.append((cutoff, model_id))
        path = run_dir / "models" / f"{model_id}.json"
        path.write_text("{}\n")

    monkeypatch.setattr(runner, "_train_one", fake_train)

    observed = {}

    def fake_backtest(signals, price_frame, *, config, corporate_actions, minute_prices):
        observed["hold_sessions"] = config.hold_sessions
        observed["last_price_date"] = price_frame["trade_date"].max()
        observed["last_signal"] = pd.to_datetime(signals["asof"], utc=True).max()
        return {
            "orders": pd.DataFrame({"order_id": ["o1"], "status": ["FILLED"]}),
            "trades": pd.DataFrame({"order_id": ["o1"], "side": ["BUY"], "price": [10.0],
                                    "shares": [1], "commission": [0.0], "stamp_duty": [0.0]}),
            "nav": pd.DataFrame({"trade_date": [formal_sessions[-1]], "cash": [0.0],
                                 "market_value": [10.0], "nav": [10.0]}),
            "positions": pd.DataFrame(columns=["trade_date", "ts_code", "shares"]),
            "corporate_actions": actions,
        }

    monkeypatch.setattr(runner, "run_backtest", fake_backtest)
    monkeypatch.setattr(runner, "summarize_backtest", lambda *args, **kwargs: {"total_return": 0.0})

    run_dir = tmp_path / "experiments" / ".running" / "run"
    for directory in ("data", "models", "predictions", "selections", "trades", "diagnostics", "logs"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    (run_dir / "RUN_STATUS.json").write_text(json.dumps({"run_id": "run", "status": "CREATED"}) + "\n")

    result = runner.run(
        run_dir=run_dir,
        config_path=ROOT / "configs/experiments/q70_source_parity_t10_20260822.yaml",
    )
    assert result["models"] == 7
    assert len(trained) == 7
    assert observed["hold_sessions"] == 10
    assert observed["last_price_date"] == pd.Timestamp("2026-07-31", tz="UTC")
    assert observed["last_signal"] + pd.offsets.BDay(11) <= pd.Timestamp("2026-07-31", tz="UTC")
    assert (run_dir / "data/f0_panel.parquet").is_file()
    assert (run_dir / "data/execution_5min.parquet").is_file()
    manifest = json.loads((run_dir / "data_manifest.json").read_text())
    assert set(manifest["snapshots"]) == {
        "f0", "labels", "market_context", "execution_daily", "corporate_actions", "execution_5min",
    }
