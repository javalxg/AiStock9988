# CAP1 F0-123 Top-30% Classifier V1 Preregistration

Status: `PREREGISTERED_BEFORE_RESULT`.

## Question

Inside the frozen CAP1 reset/reclaim opportunity pool, does estimating whether a
stock belongs to the same-day top 30% of *realizable* H10 outcomes improve the
Top20-to-Top5 portfolio over continuous-return XGB ranking?

## Frozen Contract

- Stage 1 is exactly `reset_weak_confirm_v3_cap1_20.yaml`: all eleven rules,
  transparent candidate generation, T+1 raw-open accounting, 20% sizing,
  maximum five positions, H10, trailing previous-close -8% stop, capacity,
  corporate actions, Base/Stress costs, and stock-session missing-data handling.
- Stage 2 loads exactly F0-123. NaN remains native XGBoost missing; no 202
  factors, external feature, imputation, feature filter, probability threshold,
  abstention gate, TopN change, or transparent fallback is allowed.
- For a signal date with at least four matured Stage-1 outcomes, sort each
  outcome by canonical executable H10 Base economic return. The highest
  `ceil(30% * count)` rows receive label 1; the rest receive 0. Dates with
  fewer than four matured rows do not produce a training label. This is a fixed
  label-validity rule, not a prediction-date skip.
- Every prediction-date Stage-1 row with a complete F0 feature row is scored.
  The candidate set remains the same transparent Top20; only its order changes
  to predicted top-30% probability descending. A missing F0 stock-session
  excludes that stock only.
- Train one deterministic XGBoost binary classifier per calendar month using
  the trailing 12 months and only labels whose executable exit is available by
  the previous month-end close. 2025 is training input only; reported portfolio
  performance is 2026 only.
- Score every daily session from `2026-01-05` through the common F0 database
  cutoff `2026-08-28`, then execute/mark through `2026-09-01`. No model, date,
  or month fallback is allowed.

## Acceptance and Abandonment

Both Base and Stress must beat the paired transparent CAP1 candidate-pool
control and the full CAP1 result, have win rate >= 70%, PF >= 2, MaxDD <= 15%,
positive ex-best-week and ex-top3 return, at least 20 closed trades, and at
most five positions. Weekly >=5% count and ratio are reported.

If any criterion fails, reject this exact top-30% classifier. Do not tune its
class fraction, minimum group size, hyperparameters, TopN, or holding rules.
