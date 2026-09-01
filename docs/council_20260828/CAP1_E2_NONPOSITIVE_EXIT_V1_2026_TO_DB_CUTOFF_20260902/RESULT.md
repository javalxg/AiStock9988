# reset_weak_confirm_v3_cap1_e2_nonpositive_exit_v1 2026 DB Full-Universe Backtest

Status: `REJECT`. This is a seen-2026 diagnostic, not
an out-of-sample claim.

## Scope

- Signals: `2026-01-05` through `2026-08-31`; execution/mark
  cutoff: `2026-09-01`, the common required-source DB cutoff.
- 160 signal dates, 57
  active dates, 366 Stage1 rows, and no date-level
  data gap or sample-size skip.
- Database universe; missing required data exclude only that stock-session.
- No raw business data, CSV, Parquet, factor cache, or model artifact was written.

## Portfolio

| Scenario | Return | PF | MaxDD | Win rate | Ex-best-week | Ex-top3 | Weekly >=5% | Trades | End open | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Base | +19.18% | 1.570 | -6.95% | 39.0% | +10.95% | +4.75% | 2 (5.9%) | 59 | 1 | False |
| Stress | +13.88% | 1.381 | -8.03% | 39.0% | +6.30% | +0.01% | 2 (5.9%) | 59 | 1 | False |

## Unchanged CAP1 control

| Scenario | Control return | Challenger return | Control PF | Challenger PF | Control MaxDD | Challenger MaxDD | Control win rate | Challenger win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | +33.65% | +19.18% | 2.254 | 1.570 | -8.27% | -6.95% | 56.1% | 39.0% |
| Stress | +29.61% | +13.88% | 2.058 | 1.381 | -8.82% | -8.03% | 56.1% | 39.0% |


The challenger executed 32 Base and
32 Stress early-path exits.

## Decision

Absolute Base/Stress acceptance passed: `False`. Relative promotion
against unchanged CAP1 passed: `False`. Final decision passed:
`False`. A failed fixed rule is retained unchanged as evidence and must not
be repaired by a threshold, weight, TopN, holding-period, gate, or model scan.
