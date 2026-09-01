# F0-123 Conditional Reset Ranker

## Contract

- Fixed reset-state Stage-1, then frozen F0=123 XGBRanker inside that pool.
- All 123 columns are retained; feature-level NaN is passed to XGBoost without imputation.
- Monthly trailing-12-month training, daily 2026 prediction, no skipped month or fallback.
- Same canonical portfolio engine, Base/Stress costs, H10, trailing stop, 20% sizing and five-position cap.
- No wide-table 202 factors, extra data source, feature selection, threshold scan, or business-data cache.

## Sample

- Stage-1 rows: `21841`; F0-eligible rows: `21841` (100.00%).
- Training labels: `21823`; 2026 prediction rows: `17128`.

## Portfolio

| Cost | Transparent control | F0-123 challenger | Delta | Challenger PF | MaxDD | Ex-best | Ex-top3 | Trades | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | -6.56% | -13.19% | -6.63% | 0.779 | -38.08% | -27.36% | -31.34% | 58 | False |
| stress | -10.82% | -16.81% | -5.99% | 0.725 | -38.94% | -30.62% | -34.71% | 58 | False |

## Decision

The conditional F0=123 ranker is rejected unchanged. This result will not be repaired with feature, model, Stage-1, TopN, or execution tuning.
