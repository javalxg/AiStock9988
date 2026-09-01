# F0-123 Conditional Stage1 Ranker

## Contract

- Configured fixed-rule Stage-1, then frozen F0=123 XGBRanker inside that pool.
- All 123 columns are retained; feature-level NaN is passed to XGBoost without imputation.
- Monthly trailing-12-month training, daily 2026 prediction, no skipped month or fallback.
- Same canonical portfolio engine, Base/Stress costs, H10, trailing stop, 20% sizing and five-position cap.
- No wide-table 202 factors, extra data source, feature selection, threshold scan, or business-data cache.

## Sample

- Stage-1 rows: `501`; F0-eligible rows: `501` (100.00%).
- Training labels: `499`; 2026 prediction rows: `366`.

## Portfolio

| Cost | Transparent control | F0-123 challenger | Delta | Win rate | PF | MaxDD | Ex-best | Ex-top3 | Weekly >=5% | Trades | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | +33.65% | +28.75% | -4.91% | 56.1% | 1.906 | -8.27% | +16.14% | +12.11% | 2 (5.9%) | 41 | False |
| stress | +29.61% | +24.66% | -4.94% | 56.1% | 1.752 | -8.82% | +12.60% | +8.76% | 2 (5.9%) | 41 | False |

## Decision

The conditional F0=123 ranker is rejected unchanged. This result will not be repaired with feature, model, Stage-1, TopN, or execution tuning.
