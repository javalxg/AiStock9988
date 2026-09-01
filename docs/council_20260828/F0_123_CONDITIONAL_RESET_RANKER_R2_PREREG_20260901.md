# F0-123 Conditional Reset Ranker R2 Preregistration

Date: `2026-09-01`

Status: `PREREGISTERED_AFTER_R1_INTERFACE_FAILURE_BEFORE_ANY_MODEL_OR_RETURN`.

R2 inherits every rule in
`F0_123_CONDITIONAL_RESET_RANKER_PREREG_20260901.md` except the R1
all-123-finite row requirement.

## Authorized Correction

R1 proved that `ps_ttm` and `dv_ttm` are null for every F0 source row in
January 2025, so requiring 123 finite values makes the original F0=123 system
impossible to train. The original q70 training path uses XGBoost's native
feature-level missing-value branches and does not drop a row merely because an
individual F0 value is NaN.

R2 therefore freezes this missing-value contract:

1. a stock-session must still have a joined `stock_factor_pro_ts` row, a
   `daily_basic_ts` row, and a PIT industry mapping from the canonical F0
   loader;
2. numeric positive/negative infinity is normalized to NaN;
3. XGBoost receives feature-level NaN values natively, without mean/median,
   zero, cross-sectional, forward, or backward imputation;
4. all 123 registered columns remain in the matrix and in their frozen order,
   including columns that are entirely missing in an early training month;
5. no feature may be removed after inspecting missingness, IC, SHAP, or return.

This correction changes no economic signal, outcome threshold, training date,
label, model parameter, rank, portfolio rule, or acceptance criterion. R1 had
zero trained models and zero portfolio results, so R2 is not a repair chosen
after seeing performance.

## Unchanged Contract

- Stage-1: `dist_ma60 <= -0.10`, `mkt_ret_20d < 0`, `ret20 < 0`, and
  `liq20 >= 500000` under the canonical universe/PIT policy.
- Frozen F0 manifest: exactly 123 columns, order hash
  `7f88c8912cd268a1bedd5034768ab73827bdce8f8d2f3870fe8fac3c26aa9e39`.
- Monthly trailing-12-month conditional XGBRanker, daily 2026 prediction,
  deterministic double training, and no skipped month/fallback.
- Signal range `2026-01-05..2026-08-17`; execution through `2026-09-01`.
- Transparent same-pool control versus F0=123 challenger, Top20 to Top5,
  maximum five positions, 20% sizing, H10, trailing close stop, and canonical
  Base/Stress costs.
- The original acceptance and abandonment conditions remain binding.
- No wide-table 202 factor or data source outside the original F0=123 inputs.
- No raw business-data cache may be persisted.
