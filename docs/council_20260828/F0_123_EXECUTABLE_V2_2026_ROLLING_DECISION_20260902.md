# F0-123 Executable V2 2026 Rolling Decision

Date: `2026-09-02`

Status: `REJECT_FULL_MARKET_F0_RANKER_KEEP_CAP1_CONTROL`.

## Scope

This decision uses portfolio performance only from 2026. Pre-2026 rows were
allowed only inside each causal trailing training window; no 2024 or 2025 return
was used to pass or rescue the strategy.

The run covered all 33 exchange weeks with a session from `2026-01-09` through
the common F0 cutoff `2026-08-28`, with canonical execution through the database
cutoff `2026-09-01` (`F0_123_EXECUTABLE_V2_2026_ROLLING_20260902/RESULT.md:12-15`).
Verification proves no signal week was skipped, all model labels were mature,
one model was trained for every covered month, both cost arms ended on the
database cutoff, and the five-position cap held
(`F0_123_EXECUTABLE_V2_2026_ROLLING_20260902/verification.json:2-16`).

## Result

- Base: `-22.78%`, win rate `39.19%`, PF `0.705`, MaxDD `-39.00%`, ex-best
  `-33.93%`, and ex-top3 `-40.46%` across 74 closed trades.
- Stress: `-26.23%`, win rate `35.14%`, PF `0.664`, MaxDD `-39.09%`, ex-best
  `-36.76%`, and ex-top3 `-43.54%` across 74 closed trades.

The authoritative table is
`F0_123_EXECUTABLE_V2_2026_ROLLING_20260902/RESULT.md:17-22`. Every registered
economic acceptance condition failed except minimum trade count and position
capacity.

Executable labels improved the prior fixed-stop-label Base result from
`-33.98%` to `-22.78%`, but did not change the decision. The prior result is in
`F0_123_FULL_MARKET_WEEKLY_TOP5_20260901/RESULT.md:16-21`.

## Ranking Diagnosis

The failure is selection, not merely costs:

- Rank 1-2 mean executable label was `-0.92%`; rank 3-5 was `-1.05%`.
- Rank 6-10 improved to `-0.23%`; rank 11-20 was positive at `+0.29%`.
- Therefore the model's extreme Top5 compression is anti-monotonic in 2026.
  Reversing or choosing a different rank slice after seeing this result would be
  an unauthorized TopN/rank scan, not a valid repair.

The bucket evidence is recorded at
`F0_123_EXECUTABLE_V2_2026_ROLLING_20260902/f0_diagnostics.json:2-37`.

Within the mature Top20 rows, winners showed stronger industry-relative ASI,
DFMA, and MACD state but lower raw BRAR, ATR, and absolute trend-level ranks than
losers (`f0_diagnostics.json:40-135`). This is descriptive 2026 evidence. It
must not be converted into thresholds and replayed on the same period.

## Route Decision

Close the unchanged full-market F0=123 XGBRanker. Do not try another parameter,
TopN, reverse-rank, gate, or threshold variation on the seen 2026 sample.

Keep CAP1-20 as the historical control: Base `+32.44%`, PF `2.254`, MaxDD
`-8.27%`; Stress `+28.48%`, PF `2.058`, MaxDD `-8.82%`
(`RESET_WEAK_CONFIRM_V3_CAP1_20_FINAL_DECISION_20260901.md:10-17`). Its entry
rule already captures a profitable reset/reclaim state.

The remaining evidence-backed improvement is not another entry-time ranker.
CAP1's static entry features separated winners only weakly, while E2 return
separated eventual winners from losers by `+3.36%` versus `-2.86%`; 18 of 22
positions above entry at E2 ultimately won
(`CAP1_TRADE_MECHANISM_DIAGNOSTIC_20260901/RESULT.md:24-48`). The early-path
overlay therefore remains the next forward-only risk-control experiment. It
cannot be claimed from another historical 2026 replay.
