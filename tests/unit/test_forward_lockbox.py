import pandas as pd
import pytest
import hashlib
from types import SimpleNamespace

from aistock9988.forward.lockbox import ForwardLockbox
from scripts.quiet_forward_shadow_runner import _enforce_forward_only_contract


def _frame(day="2026-08-24"):
    snapshot = hashlib.sha256(b"000001.SZ:1").hexdigest()
    return pd.DataFrame([{
        "asof": day, "ts_code": "000001.SZ", "score": 1.0,
        "candidate_rank": 1, "candidate_status": "IN_VIEW",
        "candidate_snapshot_id": snapshot,
    }])


def _selection(day="2026-08-24"):
    return pd.DataFrame([{
        "asof": day,
        "decision_id": f"decision-{day}",
        "desired_entries": 1,
        "candidate_snapshot_id": hashlib.sha256(b"000001.SZ:1").hexdigest(),
    }])


def test_lockbox_appends_and_rejects_rewrite(tmp_path):
    box = ForwardLockbox(tmp_path / "box", experiment_id="s1", config_sha256="cfg")
    box.append({"score": _frame(), "candidate": _frame(), "selection": _selection()}, bundle_id="b1", source_end="2026-08-24")
    with pytest.raises(ValueError, match="append-only"):
        box.append({"score": _frame(), "candidate": _frame(), "selection": _selection()}, bundle_id="b2", source_end="2026-08-25")


def test_lockbox_rejects_config_change(tmp_path):
    root = tmp_path / "box"
    ForwardLockbox(root, experiment_id="s1", config_sha256="cfg1").append(
        {"score": _frame(), "candidate": _frame(), "selection": _selection()},
        bundle_id="b1", source_end="2026-08-24")
    with pytest.raises(ValueError, match="config or experiment"):
        ForwardLockbox(root, experiment_id="s1", config_sha256="cfg2").append(
            {"score": _frame("2026-08-25"), "candidate": _frame("2026-08-25"), "selection": _selection("2026-08-25")},
            bundle_id="b2", source_end="2026-08-25")


def test_read_day_verifies_partition_hash(tmp_path):
    root = tmp_path / "box"
    box = ForwardLockbox(root, experiment_id="s1", config_sha256="cfg")
    box.append({"score": _frame(), "candidate": _frame(), "selection": _selection()}, bundle_id="b1", source_end="2026-08-24")
    assert len(box.read_day("2026-08-24")["candidate"]) == 1
    path = root / "candidate" / "part-2026-08-24.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="partition hash mismatch"):
        box.read_day("2026-08-24")


def test_selection_ledger_is_decision_scoped_not_stock_scoped(tmp_path):
    root = tmp_path / "box"
    box = ForwardLockbox(root, experiment_id="s1", config_sha256="cfg")
    selection = pd.DataFrame([{
        "asof": "2026-08-24",
        "decision_id": "decision-1",
        "desired_entries": 4,
    }])
    selection["candidate_snapshot_id"] = hashlib.sha256(b"000001.SZ:1").hexdigest()
    box.append({"score": _frame(), "candidate": _frame(), "selection": selection},
               bundle_id="b1", source_end="2026-08-24")
    stored = box.read_day("2026-08-24")["selection"]
    assert stored.iloc[0]["decision_id"] == "decision-1"


def test_lockbox_requires_all_ledgers_and_matching_snapshot(tmp_path):
    box = ForwardLockbox(tmp_path / "box", experiment_id="s1", config_sha256="cfg")
    with pytest.raises(ValueError, match="requires score"):
        box.append({"score": _frame(), "candidate": _frame()}, bundle_id="b1", source_end="2026-08-24")
    bad = _selection()
    bad["candidate_snapshot_id"] = "wrong"
    with pytest.raises(ValueError, match="snapshot"):
        box.append({"score": _frame(), "candidate": _frame(), "selection": bad},
                   bundle_id="b1", source_end="2026-08-24")


def test_lockbox_rejects_partition_path_escape(tmp_path):
    root = tmp_path / "box"
    box = ForwardLockbox(root, experiment_id="s1", config_sha256="cfg")
    box.append({"score": _frame(), "candidate": _frame(), "selection": _selection()},
               bundle_id="b1", source_end="2026-08-24")
    manifest = next((root / "manifests").glob("manifest-*.json"))
    payload = manifest.read_text(encoding="utf-8").replace('"candidate/part-2026-08-24.parquet"', '"../outside.parquet"')
    manifest.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash|escapes"):
        box.read_day("2026-08-24")


def test_abandoned_strategy_cannot_operate_forward_lockbox():
    strategy = SimpleNamespace(
        strategy_id="s46_mild_liquid_rank_v1",
        identity={"research_status": "abandoned"},
    )
    with pytest.raises(ValueError, match="abandoned"):
        _enforce_forward_only_contract(strategy, pd.Timestamp("2026-08-31", tz="UTC"), "freeze")
