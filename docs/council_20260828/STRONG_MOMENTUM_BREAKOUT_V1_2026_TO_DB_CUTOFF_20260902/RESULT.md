# strong_momentum_breakout_v1 2026 DB Full-Universe Backtest

Status: `REJECT`. This is a seen-2026 diagnostic, not
an out-of-sample claim.

## Scope

- Signals: `2026-01-05` through `2026-08-31`; execution/mark
  cutoff: `2026-09-01`, the common required-source DB cutoff.
- 160 signal dates, 160
  active dates, 27383 Stage1 rows, and no date-level
  data gap or sample-size skip.
- Database universe; missing required data exclude only that stock-session.
- No raw business data, CSV, Parquet, factor cache, or model artifact was written.

## Portfolio

| Scenario | Return | PF | MaxDD | Win rate | Ex-best-week | Ex-top3 | Weekly >=5% | Trades | End open | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Base | -6.71% | 0.744 | -17.49% | 37.7% | -10.65% | -17.73% | 0 (0.0%) | 69 | 4 | False |
| Stress | -9.26% | 0.715 | -18.27% | 40.0% | -12.45% | -20.14% | 0 (0.0%) | 70 | 4 | False |

## Decision

Both Base and Stress acceptance passed: `False`. A failed fixed rule is
retained unchanged as evidence and must not be repaired by a threshold, weight,
TopN, holding-period, gate, or model scan.
