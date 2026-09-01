# CAP1 H10 Profit Extension to H20

Status: `REJECT`. This is a preregistered seen-2026 paired backtest, not an
out-of-sample claim.

## Scope and integrity

- Signal range: `2026-01-05` through `2026-08-31`; execution and
  mark cutoff: `2026-09-01`, the common required-source DB cutoff.
- Signal dates: 160; active signal dates:
  57; no date-level sample gate or fallback.
- Sealed 2026 CAP1 regression passed: True.
- Same in-memory candidates, decisions, prices and corporate actions; only H10
  prior-close-profitable positions may extend to H20.
- No raw business data, model, CSV, or Parquet artifact was written.
- Exact reproduction code/config is preserved in Git commit `c44b57d`; the
  rejected executable path is removed from the current mainline.

## Paired portfolio result

| Scenario | Strategy | Return | PF | MaxDD | Win rate | Ex-best-week | Weekly >=5% | Trades | Extensions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | Control | +33.65% | 2.254 | -8.27% | 56.1% | +21.22% | 3 (8.8%) | 41 | 0 |
| Base | Challenger | +20.04% | 1.775 | -11.17% | 51.6% | +9.35% | 3 (8.8%) | 31 | 17 |
| Stress | Control | +29.61% | 2.058 | -8.82% | 56.1% | +17.59% | 3 (8.8%) | 41 | 0 |
| Stress | Challenger | +17.52% | 1.658 | -11.69% | 51.6% | +7.15% | 3 (8.8%) | 31 | 17 |

## Failure mechanism

- Among 14 shared closed trades whose
  exit actually changed, H10 control returns averaged
  +12.06%; extended exits averaged
  +14.92%. Only
  57.1% improved.
- Longer occupancy also removed 13 control
  buys. Their 13 closed Base trades
  contributed RMB 204,882 with
  69.2% winners.

## Decision

Promotion tests passed: `False`. On failure, close this exact
H10-profit-to-H20 rule without trying another hold period, threshold, TopN, gate,
or model. CAP1 remains unchanged unless every preregistered paired condition
passes.
