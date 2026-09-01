# Dragon-Tiger Pullback Reclaim V1

## Contract

- Event source: `2026-01-05` through `2026-08-18`.
- Pullback/reclaim observation through `2026-08-25`; execution and marks through `2026-09-01`.
- Rules only, next-open entry, maximum five positions, H10, -8% prior-close trailing stop.
- No threshold sweep, XGBoost, frozen 202-factor input, raw-data cache, or persisted business-data ledger.

## Event Funnel

- Source stock-days: `11714`; confirmed: `127`; active confirmation sessions: `84`.
- Overlapping events skipped during observation: `122`.

## Portfolio

| Cost | Return | Mean week | Win rate | PF | MaxDD | Ex-best-week | Ex-top3 | Trades | Open | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | -29.24% | -0.85% | 37.66% | 0.696 | -39.90% | -36.43% | -46.15% | 77 | 2 | False |
| stress | -33.09% | -1.01% | 37.66% | 0.651 | -41.07% | -39.74% | -49.60% | 77 | 2 | False |

## Decision

V1 is rejected unchanged. Failure is evidence against this definition and is not repaired with a threshold scan.
