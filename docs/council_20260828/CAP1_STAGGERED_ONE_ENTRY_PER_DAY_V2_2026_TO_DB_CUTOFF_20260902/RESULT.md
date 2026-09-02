# CAP1 Staggered One Entry per Day V2

Status: `REJECT`. This is a paired, seen-2026 historical replay through the current database cutoff, not a forward claim.

## Integrity

- Signal range: `2026-01-05` through `2026-08-31`; execution and marks through `2026-09-01`.
- Control and challenger used the same in-memory QuantDB bundle, feature ledger, candidate ledger, execution panel, and corporate actions.
- PIT feature audit passed. No raw market rows, factors, candidates, fills, positions, model, CSV, or Parquet output was retained.

## Portfolio

| Cost | Strategy | Return | PF | MaxDD | Win rate | Ex-best-week | Weeks >=5% | Trades | Max positions | Max same-day entries |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | Control | +33.65% | 2.254 | -8.27% | 56.1% | +21.22% | 3 | 41 | 5 | 4 |
| base | One-entry challenger | +2.37% | 1.052 | -11.95% | 51.4% | -5.40% | 3 | 37 | 5 | 1 |
| stress | Control | +29.61% | 2.058 | -8.82% | 56.1% | +17.59% | 3 | 41 | 5 | 4 |
| stress | One-entry challenger | -0.62% | 0.948 | -12.91% | 51.4% | -7.88% | 3 | 37 | 5 | 1 |

## Decision

Promotion passed: `False`. The challenger changed only `entries_per_decision` from 5 to 1. If rejected, this exact rule is closed without trying 2/3/4-entry variants.
