import numpy as np
import pandas as pd

from aistock9988.models.trainer import audit_model, train_ranker


def test_ranker_artifact_and_hash_audit(tmp_path):
    X = pd.DataFrame({"f1": [0., 1., 2., 3.], "f2": [1., 0., 1., 0.]})
    y = pd.Series([0., 1., 1., 0.])
    groups = pd.Series(["2026-08-21", "2026-08-20", "2026-08-21", "2026-08-20"])
    artifact = train_ranker(X, y, group_dates=groups, feature_set_id="feature.f0_123.v1",
                             label_profile_id="label.endpoint_open_open_t10.v1",
                             training_cutoff="2026-08-22T15:00:00Z", model_id="m1",
                             output_dir=tmp_path, params={"n_estimators": 2, "max_depth": 2})
    audited = audit_model(tmp_path / "m1.json", tmp_path / "m1.metadata.json")
    assert artifact.model_sha256 == audited.model_sha256
    assert audited.group_count == 2
