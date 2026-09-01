# F0-123 Executable V2 Forward Preregistration

Date: `2026-09-02`

Status: `PREREGISTERED_BEFORE_FIRST_POST_OBSERVATION_SIGNAL`.

## Purpose

Test whether the original frozen F0=123 factors can rank executable returns on
genuinely new dates after the completed 2026 historical diagnosis. This is not
a repair or replay of January-August 2026 and may not claim historical V2 return.

## Observed Boundary

- Every signal and return through `2026-09-01` is observed.
- The first eligible V2 signal must be on or after `2026-09-02` and must have
  same-session coverage in market, adjustment, limit, F0, and daily-basic data.
- Sparse ST, suspension, and auction tables affect individual stock eligibility
  but do not define the market-wide cutoff by their maximum event date.
- No older F0 row may be carried forward to a newer market session.

## Frozen Training Contract

- Exactly 123 F0 columns in manifest order, using daily independent
  cross-sectional percentile/z-score normalization.
- At least 61 valid factors per row; missing values remain native XGBoost NaN.
- Deterministic maximum 1,500 training rows per date after full cross-sectional
  normalization.
- Monthly trailing-12-month training; only labels whose actual exit-open
  availability is at or before the training cutoff may enter.
- Target profile is `label.executable_path_open_open_t10_base.v1`, accepted by
  `F0_123_EXECUTABLE_LABEL_V2_AUDIT_20260902/RESULT.md`.
- Fixed grouped XGBRanker: pairwise objective, 200 trees, depth 6, learning rate
  0.05, min-child-weight 5, row/column sample 0.8, alpha/lambda 1.0, seed 42,
  `n_jobs=1`, histogram tree method.
- No feature selection, hyperparameter search, class threshold, factor gate,
  probability gate, auxiliary table, Stage-2 model, or old model reuse.

## Frozen Portfolio Contract

- Score the full eligible market and freeze Top20; take the first five with the
  existing tradability replacement contract.
- Maximum five positions, equal 20% decision-NAV sizing, T+1 entry, H10,
  from-fill close stop at `-8%`, next-sellable-open exit, canonical Base/Stress
  costs and the single canonical engine.
- To prevent the already diagnosed weekly/H10 slot-sampling mismatch, emit a
  decision only every ten exchange sessions. The anchor is the first fully
  covered session on or after `2026-09-02`; no signal is emitted between capacity
  dates. This schedule is evaluated forward only and is not phase-scanned.
- CAP1-20 remains an unchanged paired control on the same eligible forward
  sessions.

## Forward Acceptance

Evaluation begins only after at least 60 V2 trades have closed. Acceptance then
requires:

- maximum five positions and nonnegative cash at every session;
- trade win rate at least `70%`;
- PF at least `2.0` under both Base and Stress;
- absolute MaxDD no greater than `15%` under both costs;
- return excluding the best week and the top three winners remains positive;
- V2 beats unchanged CAP1-20 on the same forward sessions.

The previously disproved weekly `>=5%` hit-rate target is not reinstated. Weekly
returns are reported descriptively, while PF, drawdown, breadth of profit, and
paired forward return determine acceptance.

## Data Gate And Current State

`scripts/f0_123_executable_forward_preflight.py` is the only readiness gate.
After the `2026-09-02` refill attempt, market, adjustment, limit, and daily-basic
data cover `2026-09-01`, but the factor provider still materializes F0 only
through `2026-08-28`; F0 row count is zero on `2026-08-31` and `2026-09-01`.
Therefore no first forward signal exists yet. The state is
`WAITING_FOR_COMMON_F0`, not a skipped model or fallback selection.
