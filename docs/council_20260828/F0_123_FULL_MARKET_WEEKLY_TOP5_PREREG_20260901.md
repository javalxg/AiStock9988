# F0-123 Full-Market Weekly Top5 Preregistration

Date: `2026-09-01`

Status: `PREREGISTERED_BEFORE_MODEL_OR_RETURN`.

## Question

Can the frozen original F0=123 factors produce an auditable 2026 portfolio when
used as q70 intended: daily cross-sectional comparison across the broad market,
monthly grouped ranking, weekly decisions, and no auxiliary data source?

The prior conditional R2 result does not answer this question. It trained only
inside a reset-state pool that was active on 77 of 150 daily sessions, used raw
factor magnitudes, and admitted rows with only one non-null factor.

## Frozen Input And Transform

- Exactly 123 registered F0 columns in manifest order; no 202-factor wide table.
- Each stock-session must contain at least 61 non-null F0 values.
- Positive/negative infinity becomes NaN; there is no imputation.
- Every feature is ranked independently within its own trading-day cross
  section, converted to percentile, then centered and scaled by that day's
  sample standard deviation.
- Training retains a deterministic sample of at most 1500 stocks per day using
  seed 42. Prediction scores every eligible stock on each weekly signal date.
- No feature selection, SHAP pruning, threshold scan, gate, extra data source,
  or post-return repair is permitted.

## Frozen Timeline And Label

- Training begins `2025-01-02`.
- Weekly prediction is the last exchange session of each week from
  `2026-01-05` through `2026-08-14`.
- The first weekly signal in each month trains one new trailing-12-month model;
  later weekly signals in the month reuse it.
- Only labels available by that model's signal close may train it.
- Label: signal T, entry T+1 economic open, exit T+11 economic open. If any
  economic close from entry through the holding path reaches -8% from entry,
  the training label is clipped to -8%.

## Frozen Model, Portfolio And Execution

- Deterministic `XGBRanker(rank:pairwise)`, grouped by trading day, depth 6,
  200 trees, learning rate 0.05, seed 42, `n_jobs=1`.
- Full-market model score -> Top20 candidate ledger -> Top5 entries; maximum
  five positions and 20% decision NAV per fill.
- T+1 raw-open entry, H10, close-triggered -8% from-entry stop executed at the
  next tradable open, canonical raw/corporate-action accounting, Base and
  Stress costs.
- There is no q70 sector threshold or market-breadth gate in this experiment;
  it first isolates whether the 123-factor rank itself is useful.

## Acceptance And Abandonment

Both Base and Stress must beat the same-cost CAP1 reference, have PF >= 2,
absolute MaxDD <= 15%, positive return after removing the best week and top
three profitable trades, at least 20 closed trades, and no position-cap breach.

Failure rejects this exact full-market F0 baseline. It does not authorize new
data. The next permitted action is a winner/loser and rank-bucket diagnosis
using these same 123 features; only if that diagnosis finds no stable signal
may auxiliary data be considered.

No business-data CSV, Parquet, pickle, model cache, prediction ledger, or fill
ledger may be persisted. Only small aggregate audit JSON/Markdown artifacts are
allowed.
