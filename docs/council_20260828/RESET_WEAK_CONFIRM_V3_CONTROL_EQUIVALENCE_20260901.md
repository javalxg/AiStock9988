# Reset Weak Confirm V3 Control Equivalence

Verdict: `PASS`.

The current canonical engine replayed the sealed
`RESET_WEAK_CONFIRM_V3_FULL_2026_TO_0828` control over the same signal and
execution windows.

- Bundle ID: `1f22d8af68ee55f2ced5b70262b41563a5eb339899eec88d3469663ea08dd6c0`
- Source frame hashes: identical for every source
- Feature ledger: 1,209,318 rows, all prior columns exactly equal
- Score/candidate ledgers: 817,256 rows, all prior columns exactly equal
- Selection ledger: 148 rows, exactly equal
- Base and stress fills: 82 fill rows each, exactly equal
- Base and stress NAV: 159 session rows each, exactly equal
- Portfolio summary: exactly equal

The only new columns are `ranking_feature_ready` and
`ranking_feature_rejection_reason` in the feature ledger and
`ranking_feature_ready` in the score/candidate ledgers. They are diagnostic
columns and do not change the control's selection or execution.

Control output:
`docs/council_20260828/RESET_WEAK_CONFIRM_V3_CONTROL_EQUIVALENCE_20260901/`

This proves the current feature and selection changes preserve the sealed V3
behavior. V3-R2 remains a single-variable ranking experiment.
