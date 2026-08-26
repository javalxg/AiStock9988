import pandas as pd
import pytest

from aistock9988.selection.ledger import build_prediction_ledger, freeze_candidates, write_ledger


def test_full_prediction_and_top20_are_deterministic(tmp_path):
    source = pd.DataFrame({"ts_code": ["000002.SZ", "000001.SZ", "000003.SZ"], "score": [0.5, 0.5, 0.1]})
    pred = build_prediction_ledger(source, asof="2026-08-21", feature_set_id="feature.f0_123.v1", model_id="m1")
    assert pred.ts_code.tolist() == ["000001.SZ", "000002.SZ", "000003.SZ"]
    cand = freeze_candidates(pred, top_n=2)
    assert cand.ts_code.tolist() == ["000001.SZ", "000002.SZ"]
    first = write_ledger(pred, tmp_path / "prediction.csv")
    second = write_ledger(pred, tmp_path / "prediction2.csv")
    assert first.content_hash == second.content_hash
    with pytest.raises(FileExistsError):
        write_ledger(pred, tmp_path / "prediction.csv")


def test_duplicate_prediction_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        build_prediction_ledger(pd.DataFrame({"ts_code": ["A", "A"], "score": [1, 2]}),
                                asof="2026-08-21", feature_set_id="f", model_id="m")
