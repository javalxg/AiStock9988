# F0-123 Conditional Reset Ranker R1 Failure

Status: `FAILED_BEFORE_MODEL_AND_PORTFOLIO`.

R1 produced no model, prediction, trade, NAV, or return. It stopped at the
January 2026 model training gate because its preregistration required all 123
F0 values to be finite for every stock-session.

The source audit proved that requirement cannot represent the original q70
F0=123 contract. In January 2025, the PIT F0 loader returned 96,687 rows over
18 sessions, but `ps_ttm` and `dv_ttm` were null on all 96,687 rows. Therefore
the all-123-finite filter retained zero rows and zero dates. Another 99 columns
also had at least one missing value, while only 22 columns were complete.

This is not evidence that Stage-1 had no candidates, labels were immature, or
F0=123 lacked ranking value. It is evidence that R1 imposed a stricter missing
value contract than q70, whose XGBoost path allows feature-level NaN values.

R1 is closed unchanged. Any retry must be separately preregistered and may only
restore XGBoost-native NaN handling; it may not alter Stage-1, dates, features,
model parameters, TopN, execution, or acceptance criteria.

