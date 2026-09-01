# F0-123 Causal Top20 Event Selector

## Contract

- Stage-1: causal monthly F0=123 ranker; full-market weekly Top20 in 2025 and 2026.
- Stage-2: trailing-12-month classifier trained only on mature causal Top20 rows; fixed H10 >=10% event.
- Paired control/challenger use identical Top20 membership and canonical Base/Stress execution.
- No auxiliary data, probability gate, threshold scan, fallback, cache, or alternate backtest engine.

## Sample

- Stage-1 training rows: `951000` across `634` sessions.
- Full prediction rows: `415260` across `84` weekly sessions.
- Causal Top20 rows: `1680`; mature labels: `1235967`.

## Paired Portfolio

| Policy | Cost | Return | PF | MaxDD | Ex-best | Ex-top3 | Trades |
|---|---|---:|---:|---:|---:|---:|---:|
| control | base | -33.98% | 0.477 | -39.43% | -37.32% | -43.76% | 63 |
| control | stress | -36.99% | 0.447 | -41.86% | -42.51% | -47.29% | 65 |
| challenger | base | -33.75% | 0.528 | -35.45% | -38.03% | -49.71% | 60 |
| challenger | stress | -26.66% | 0.673 | -28.74% | -33.41% | -46.66% | 68 |

## Event Selection

- Control Top5 event rate: `15.44%`; Top20-event recall: `23.71%`.
- Challenger Top5 event rate: `18.06%`; Top20-event recall: `28.87%`.
- Event recall lift: `+5.15%`.

## Decision

The exact causal Top20 event selector is rejected under its preregistered stop rule.
