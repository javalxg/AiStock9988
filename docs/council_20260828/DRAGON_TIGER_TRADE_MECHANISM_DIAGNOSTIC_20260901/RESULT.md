# Dragon-Tiger V1 Trade Mechanism Diagnostic

## Reproduction

- Exact sealed-ledger reproduction: `True`.
- Closed Base trades: `77`; wins: `29`; losses: `48`.
- This is an unchanged-trade diagnostic, not a new strategy or parameter search.

## Winner Versus Loser

| Feature | Winner mean minus loser mean | Winner median minus loser median |
|---|---:|---:|
| institution_intensity | -0.000576 | +0.005516 |
| opening_gap_return | -0.000586 | -0.002399 |
| pullback_delay_sessions | +0.114224 | +0.000000 |
| reclaim_delay_sessions | -0.060345 | +0.000000 |
| pullback_to_reclaim_sessions | -0.174569 | +0.000000 |
| reclaim_volume_ratio | -0.449705 | -0.141818 |
| candidate_rank | -0.091236 | +0.000000 |
| holding_sessions | +0.783046 | +0.500000 |
| economic_return | +0.222256 | +0.196759 |

## Exit Mechanism

| Exit | Trades | Win rate | Mean return | PnL | Mean hold |
|---|---:|---:|---:|---:|---:|
| STOP_LOSS | 35 | +31.43% | -5.22% | -327982.62 | 5.06 |
| TIME_EXIT | 42 | +42.86% | +1.69% | +65852.51 | 10.00 |

## Candidate Rank

| Rank | Trades | Win rate | Mean return | PnL |
|---:|---:|---:|---:|---:|
| 1 | 63 | +39.68% | -1.47% | -224850.28 |
| 2 | 13 | +30.77% | -0.56% | -15960.69 |
| 3 | 1 | +0.00% | -12.11% | -21319.15 |

## Monthly Concentration

| Entry month | Trades | Win rate | Mean return | PnL |
|---|---:|---:|---:|---:|
| 2026-01 | 10 | +20.00% | -3.19% | -66213.25 |
| 2026-02 | 9 | +11.11% | -6.08% | -100894.13 |
| 2026-03 | 9 | +44.44% | +2.80% | +24722.40 |
| 2026-04 | 8 | +75.00% | +4.66% | +59552.21 |
| 2026-05 | 10 | +50.00% | +7.01% | +125029.30 |
| 2026-06 | 10 | +50.00% | -2.25% | -59659.50 |
| 2026-07 | 16 | +31.25% | -8.42% | -241591.77 |
| 2026-08 | 5 | +20.00% | -0.12% | -3075.37 |

## Decision

- Reject dragon-tiger V1 as a standalone long-entry signal and as a positive ranking input.
- Institution intensity, opening gap, and observation timing are nearly indistinguishable between winners and losers; stronger reclaim volume is not supportive because it is higher among losers.
- The loss is structural: STOP_LOSS trades lost more than TIME_EXIT trades earned, and only three of eight entry months were profitable.
- The only justified next use is one separately preregistered CAP1 risk-exclusion overlay: exclude a CAP1 candidate when the unchanged V1 event was confirmed during the prior H10 window. Do not scan event age or thresholds.

These comparisons are descriptive evidence only and must not be converted into scanned thresholds.
