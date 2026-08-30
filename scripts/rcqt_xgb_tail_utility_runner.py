"""Run a fixed XGBoost right-tail-minus-left-tail utility diagnostic."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import rcqt_corrected_xgb_ranker_runner as runner


runner.FEATURE_SET_ID = "feature.rcqt_corrected_xgb14.v1"
runner.XGB_POLICY_ID = "rcqt.xgb_tail_utility.v1"
runner.MODEL_PARAMS = {
    "objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": 3,
    "n_estimators": 300, "max_depth": 3, "learning_rate": 0.03,
    "min_child_weight": 20.0, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 1.0, "reg_lambda": 5.0, "random_state": 20260828,
    "n_jobs": 1, "tree_method": "hist",
}


def _tail_class(values: pd.Series) -> np.ndarray:
    return np.select([values <= -0.08, values >= 0.10], [0, 2], default=1).astype(int)


def _train_predict(candidates: pd.DataFrame, output):
    predictions = []
    training_audit = []
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for split_name, cutoff_text, test_start, test_end in runner.SPLITS:
        cutoff = pd.Timestamp(cutoff_text, tz="UTC")
        cutoff_time = runner.session_close(cutoff)
        train = candidates[
            (candidates["asof"] <= cutoff)
            & (candidates["label_available_time"] <= cutoff_time)
        ].sort_values(["asof", "ts_code"], kind="mergesort").copy()
        test = candidates[candidates["asof"].between(test_start, test_end)].copy()
        if train.empty or test.empty or train["label_available_time"].max() > cutoff_time:
            raise RuntimeError(f"invalid mature train/test split: {split_name}")
        target = _tail_class(train["label_return"])
        model = XGBClassifier(**runner.MODEL_PARAMS)
        model.fit(train[list(runner.FEATURES)], target)
        model_path = model_dir / f"rcqt_xgb_tail_{split_name}.json"
        model.save_model(model_path)
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        metadata = {
            "model_id": f"rcqt_xgb_tail_{split_name}", "model_sha256": model_hash,
            "feature_set_id": runner.FEATURE_SET_ID, "features": runner.FEATURES,
            "target": {"0": "T+10 return <= -8%", "1": "middle", "2": "T+10 return >= +10%"},
            "score": "P(return >= +10%) - P(return <= -8%)",
            "training_cutoff": str(cutoff_time), "params": runner.MODEL_PARAMS,
            "train_rows": len(train), "test_start": test_start, "test_end": test_end,
        }
        (model_dir / f"rcqt_xgb_tail_{split_name}.metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
        )
        probabilities = model.predict_proba(test[list(runner.FEATURES)])
        test["xgb_score"] = probabilities[:, 2] - probabilities[:, 0]
        test["p_down_8"] = probabilities[:, 0]
        test["p_up_10"] = probabilities[:, 2]
        test["split"] = split_name
        predictions.append(test)
        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        pd.DataFrame({
            "feature": runner.FEATURES,
            "gain": [gain.get(feature, 0.0) for feature in runner.FEATURES],
        }).sort_values(["gain", "feature"], ascending=[False, True], kind="mergesort").to_csv(
            output / f"{split_name}_feature_importance.csv", index=False,
        )
        counts = pd.Series(target).value_counts().sort_index()
        training_audit.append({
            "split": split_name, "cutoff": str(cutoff_time), "train_rows": len(train),
            "train_dates": int(train["asof"].nunique()), "train_start": str(train["asof"].min()),
            "train_end": str(train["asof"].max()),
            "max_label_available_time": str(train["label_available_time"].max()),
            "class_counts": {str(key): int(value) for key, value in counts.items()},
            "test_rows": len(test), "test_dates": int(test["asof"].nunique()),
            "model_sha256": model_hash,
        })
    runner._write_json(output / "training_audit.json", training_audit)
    return pd.concat(predictions, ignore_index=True)


runner._train_predict = _train_predict


if __name__ == "__main__":
    runner.main()
