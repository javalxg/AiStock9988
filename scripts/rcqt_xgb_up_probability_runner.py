"""Run an XGBoost up-probability Top4 diagnostic on corrected RCQT candidates."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
from xgboost import XGBClassifier

import rcqt_corrected_xgb_ranker_runner as runner


runner.FEATURE_SET_ID = "feature.rcqt_corrected_xgb14.v1"
runner.XGB_POLICY_ID = "rcqt.xgb_up_probability.v1"
runner.MODEL_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "n_estimators": 300, "max_depth": 3, "learning_rate": 0.03,
    "min_child_weight": 20.0, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 1.0, "reg_lambda": 5.0, "random_state": 20260828,
    "n_jobs": 1, "tree_method": "hist",
}


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
        target = (train["label_return"] > 0).astype(int)
        model = XGBClassifier(**runner.MODEL_PARAMS)
        model.fit(train[list(runner.FEATURES)], target)
        model_path = model_dir / f"rcqt_xgb_up_{split_name}.json"
        model.save_model(model_path)
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        metadata = {
            "model_id": f"rcqt_xgb_up_{split_name}", "model_sha256": model_hash,
            "feature_set_id": runner.FEATURE_SET_ID, "features": runner.FEATURES,
            "target": "T+10 economic open-to-open return > 0",
            "training_cutoff": str(cutoff_time), "params": runner.MODEL_PARAMS,
            "train_rows": len(train), "train_positive_rate": float(target.mean()),
            "test_start": test_start, "test_end": test_end,
        }
        (model_dir / f"rcqt_xgb_up_{split_name}.metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
        )
        test["xgb_score"] = model.predict_proba(test[list(runner.FEATURES)])[:, 1]
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
        training_audit.append({
            "split": split_name, "cutoff": str(cutoff_time), "train_rows": len(train),
            "train_dates": int(train["asof"].nunique()), "train_start": str(train["asof"].min()),
            "train_end": str(train["asof"].max()),
            "max_label_available_time": str(train["label_available_time"].max()),
            "train_positive_rate": float(target.mean()), "test_rows": len(test),
            "test_dates": int(test["asof"].nunique()), "model_sha256": model_hash,
        })
    runner._write_json(output / "training_audit.json", training_audit)
    return pd.concat(predictions, ignore_index=True)


runner._train_predict = _train_predict


if __name__ == "__main__":
    runner.main()
