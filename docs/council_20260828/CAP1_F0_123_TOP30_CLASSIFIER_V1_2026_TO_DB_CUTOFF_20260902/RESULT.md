# CAP1 F0-123 Top-30% Classifier V1

Status: `REJECT`. Paired seen-2026 historical replay, not a forward claim.

## Integrity

- 2026 signals `2026-01-05` through `2026-08-28`; execution and marks `2026-09-01`.
- 401 mature labels across 45 eligible Stage-1 dates; top-30 positive rate 36.4%.
- Eight deterministic monthly classifiers; every training label was mature at that model cutoff.
- No raw business data, candidate, fill, position, model, CSV, or Parquet was persisted.

## Portfolio

| Cost | Strategy | Return | Win rate | PF | MaxDD | Ex-best | Weeks >=5% | Trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | Transparent control | +33.65% | 56.1% | 2.254 | -8.27% | +21.22% | 3 | 41 |
| base | Top-30 classifier | +28.34% | 54.8% | 1.952 | -8.27% | +15.24% | 3 | 42 |
| stress | Transparent control | +29.61% | 56.1% | 2.058 | -8.82% | +17.59% | 3 | 41 |
| stress | Top-30 classifier | +24.41% | 54.8% | 1.789 | -8.82% | +11.72% | 3 | 42 |

## Decision

Promotion passed: `False`. Failure closes this exact top-30 label without fraction, group-size, model, TopN, or holding-rule tuning.

