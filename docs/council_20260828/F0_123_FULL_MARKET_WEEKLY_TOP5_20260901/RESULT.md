# F0-123 Full-Market Weekly Top5

## Contract

- Frozen F0=123 only; daily cross-sectional percentile/z-score; at least 61 values per row.
- Daily broad-market training (deterministic cap 1500/date), monthly grouped XGBRanker, weekly full-market scoring.
- Path-stop T+10 label, Top20 to Top5, H10, maximum five positions and canonical Base/Stress execution.
- No auxiliary data, factor gate, feature selection, threshold scan, fallback, or business-data cache.

## Sample

- Prepared training rows: `588000` across `392` sessions.
- Weekly full-market prediction rows: `153753` across `31` signal weeks.
- Mature labels: `692533`; path-stop labels: `145679`.

## Portfolio

| Cost | Return | PF | MaxDD | Ex-best | Ex-top3 | Trades | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| base | -33.98% | 0.477 | -39.43% | -37.32% | -43.76% | 63 | False |
| stress | -36.99% | 0.447 | -41.86% | -42.51% | -47.29% | 65 | False |

## Decision

The exact baseline is rejected. Continue only with the preregistered 123-factor winner/loser and rank-bucket diagnosis; auxiliary data remains unauthorized.
