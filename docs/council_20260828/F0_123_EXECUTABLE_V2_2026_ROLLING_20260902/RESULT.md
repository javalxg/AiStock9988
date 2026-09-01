# F0-123 Full-Market Weekly Top5

## Contract

- Frozen F0=123 only; daily cross-sectional percentile/z-score; at least 61 values per row.
- Daily broad-market training (deterministic cap 1500/date), monthly grouped XGBRanker, weekly full-market scoring.
- Training label: `label.executable_path_open_open_t10_base.v1`; Top20 to Top5, H10, maximum five positions and canonical Base/Stress execution.
- No auxiliary data, factor gate, feature selection, threshold scan, fallback, or business-data cache.

## Sample

- 2026 signal sessions: `2026-01-09` through `2026-08-28`; execution through database cutoff `2026-09-01`.
- Prepared training rows: `603000` across `402` sessions.
- Weekly full-market prediction rows: `163885` across `33` signal weeks.
- Mature executable labels: `692682`; stop labels: `149373`.

## Portfolio

| Cost | Return | Win rate | PF | MaxDD | Ex-best | Ex-top3 | Trades | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | -22.78% | 39.19% | 0.705 | -39.00% | -33.93% | -40.46% | 74 | False |
| stress | -26.23% | 35.14% | 0.664 | -39.09% | -36.76% | -43.54% | 74 | False |

## Decision

The exact baseline is rejected. Continue only with the preregistered 123-factor winner/loser and rank-bucket diagnosis; auxiliary data remains unauthorized.
