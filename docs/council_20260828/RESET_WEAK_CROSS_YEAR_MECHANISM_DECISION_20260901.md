# Reset Weak Cross-Year Mechanism Decision

Status: `DIAGNOSTIC_SEEN_HISTORY_FREEZE_STAGE1_REJECT_H10_EXTENSION`.

## Scope

This diagnostic uses the already sealed
`RESET_WEAK_CONFIRM_V3_DIAGNOSTIC_2025_2026_TO_0828` ledgers. The selection
rule, weighted rank, Top5 capacity, T+1 entry, H10 exit, stop, and costs are
unchanged. No model or 202-factor data was used, and no threshold, TopN,
holding-period, or stop alternative was scanned.

The source portfolio sizes each name at 10% of decision NAV, so the yearly
returns below are not CAP1-20 returns. They are useful because 2025 and 2026
share the same selection and execution mechanism without the 20% sizing
amplifier.

## Cross-year portfolio evidence

| Period | Base return | Stress return | Base within-period MaxDD | Stress within-period MaxDD |
|---|---:|---:|---:|---:|
| 2025 | +13.61% | +12.03% | -3.10% | -3.65% |
| 2026 YTD | +16.06% | +14.05% | -4.35% | -4.62% |
| Combined | +31.86% | +27.77% | -4.35% | -5.70% |

The same 10% V3 mechanism was profitable in both calendar segments. The
separately sealed 2026 CAP1-20 result remains +32.44% Base and +28.48% Stress;
this document neither recomputes nor changes it.

## What Stage-1 contributes

Execution-aligned candidate labels use T+1 economic open, H10 economic open,
the global exchange calendar, the fixed -8% trailing-stop proxy, and the frozen
label cap. There are 499 mature Stage-1 events: 135 in 2025 and 364 in 2026.

| Entry year | Candidate ranks | Events | Mean label | Positive | At least +10% |
|---|---|---:|---:|---:|---:|
| 2025 | 1-5 | 111 | +1.76% | 61.26% | 10.81% |
| 2025 | 6-20 | 24 | -3.07% | 16.67% | 0.00% |
| 2026 | 1-5 | 199 | +1.41% | 55.78% | 15.58% |
| 2026 | 6-20 | 142 | -0.47% | 42.25% | 10.56% |

The fixed Top5 capacity cut preserves positive average labels in both years;
ranks 6-20 are negative in both years. Exact ordering inside Top5 is not
monotonic, and candidate rank has only weak Spearman association with the
continuous label. Top5 should therefore be interpreted as a capacity bucket,
not as proof that rank 1 is reliably better than rank 2.

Lower ranks occur only on crowded candidate days, so a same-day paired check is
required. On the five comparable 2025 days, the daily Top5-minus-lower mean was
+2.72pp. On 23 comparable 2026 days it was +1.08pp, but the date-cluster
bootstrap 95% interval was -1.43pp to +3.59pp. The ranking advantage is
directionally useful but not independently proven at the day level.

The strongest segment is naturally sparse Stage-1 days, where no lower bucket
exists: approximate Top5 mean labels were +2.39% in 2025 and +1.85% in 2026.
This is a description, not authorization for a candidate-count gate. The
strategy already abstains when the fixed pattern is absent; no new scarcity
threshold may be added.

The realized portfolio confirms why that restriction matters. Sparse days
produced 31 of 37 trades and 84.18% of 2025 PnL, then 34 of 41 trades and
79.26% of 2026 PnL. However, actual crowded-day trades averaged +3.91% in 2025
and +4.78% in 2026, versus +3.76% and +3.81% on sparse days. Sparse days
contributed most PnL because they supplied most trades, not because every
sparse-day fill was better. A candidate-count gate would discard profitable
trades and is rejected.

## Stable and unstable features

Winner-minus-loser direction was the same in both years for deeper distance
below MA60, more negative stock ret20, and higher liquidity. The differences
for market ret20, daily ret1, drawdown, and amount expansion changed sign.
Volume ratio is the clearest warning: winners averaged 1.40 versus losers 1.25
in 2025, but winners averaged 1.30 versus losers 1.47 in 2026.

Therefore the existing broad 1x-2x amount-expansion band may remain as part of
the frozen setup, but its center or direction must not be tuned from these
years. The evidence does not support adding another static volume or market
gate.

Within all 499 mature Stage-1 events, candidate-level Spearman attribution
clarifies the role of the ranking terms. Lower vol20 was associated with better
labels in both years (-0.157 and -0.154), and smaller distance from the fixed
1.15 amount-expansion center was also directionally consistent (-0.047 and
-0.074). Dist-MA60, ret1, ret20, and raw volume ratio changed correlation sign.
Higher liquidity was not a stable continuous alpha despite its usefulness as a
tradability control.

This is not a contradiction in the strategy contract. Deep distance below
MA60, negative ret20, weak market, positive ret1, MA5 reclaim, and three-day
reclaim define a state transition. Once a stock is inside that state, “more
extreme” need not mean “more profitable.” Low volatility and non-explosive
participation are closer to true ranking preferences. The composite rank can
remain because its Top5 bucket works in aggregate, but no individual weight is
independently validated and no post-hoc reweighting is authorized.

## Path evidence

Early price path remains directionally informative across both years:

| Entry year | Eventual winners E2 mean | Eventual losers E2 mean | E2/final Spearman |
|---|---:|---:|---:|
| 2025 | +3.83% | +0.71% | 0.311 |
| 2026 | +3.36% | -2.86% | 0.573 |

The 2025 separation is weaker because many eventual losers still bounced during
E0-E2, but winners had the stronger path in both years. This supports forward
observation of early path; it does not authorize an E2-sign historical exit.

## H10 extension is rejected

The last held close before H10 almost perfectly describes the return already
earned, but it does not consistently predict the next ten sessions:

| Entry year | H9 state | Trades | Control mean | Post-exit +10 mean |
|---|---|---:|---:|---:|
| 2025 | Positive | 25 | +6.95% | +0.38% |
| 2025 | Nonpositive | 12 | -2.82% | +2.85% |
| 2026 | Positive | 23 | +11.87% | +4.35% |
| 2026 | Nonpositive | 16 | -4.43% | -3.95% |

The 2026 continuation pattern reverses in 2025. Exit-date market ret20 has near
zero correlation with subsequent ten-session return in both years, so a
strong-market extension gate does not repair the instability. Do not register
or implement an H9-positive/H20 challenger from this evidence.

## Decision

1. Freeze the current Stage-1 rule and Top5 capacity cut. Do not tune its
   static thresholds or exact ranking weights on seen 2025-2026 data.
2. Keep CAP1-20 as the current portfolio control. Its 2026 result remains the
   only high-return historical configuration selected for forward locking.
3. Continue only the already hash-registered E0-E2 early-break shadow from
   2026-09-01 onward. It tests left-tail control without claiming a historical
   improvement.
4. Reject H9-positive extension, generic momentum, strong-market gates, and
   another XGBoost ranker.
5. The next formal action is to freeze the first eligible 2026-09-01-or-later
   CAP signal after all required database sources share that session cutoff.
   The latest verified common cutoff remains 2026-08-28, so no forward return
   exists yet.
