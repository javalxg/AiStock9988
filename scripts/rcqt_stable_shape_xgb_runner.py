"""Run the evidence-led stable-shape RCQT XGBRanker diagnostic."""
from __future__ import annotations

import rcqt_corrected_xgb_ranker_runner as runner


# Frozen from the 2025H2 win/loss direction: winners were less extended,
# quieter, and more strongly confirmed. Constraints encode shape, not gates.
runner.FEATURES = (
    "dist_ma60", "ret20", "vol20", "confirmation_strength",
)
runner.FEATURE_SET_ID = "feature.rcqt_stable_shape_xgb4.v1"
runner.XGB_POLICY_ID = "rcqt.stable_shape_xgb_ranker.v1"
runner.MODEL_PARAMS = {
    **runner.MODEL_PARAMS,
    "monotone_constraints": "(-1,-1,-1,1)",
}


if __name__ == "__main__":
    runner.main()
