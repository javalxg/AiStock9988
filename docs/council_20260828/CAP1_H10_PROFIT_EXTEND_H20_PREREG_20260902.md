# CAP1 H10 Profit Extension to H20 Preregistration

Status: `PREREGISTERED_BEFORE_RESULT`.

## Hypothesis

CAP1 identifies a useful reset state, but its profitable trades continued to
rise after the fixed H10 exit. A causal right-side rule may retain that tail
without extending known losing positions: immediately before the scheduled H10
open, use only the position's last known close. If its unslipped economic return
from the entry economic price is strictly positive, defer the time exit to H20.
Otherwise retain the original H10 exit.

## Frozen Contract

- Control: `reset_weak_confirm_v3_cap1_20`.
- Challenger: `reset_weak_confirm_v3_cap1_h10_profit_extend_h20_v1`.
- Full database universe, all 2026 signal sessions through the latest session
  whose T+1 open exists in the common required-source database coverage;
  execution and remaining-position marks end on the common database cutoff.
- Selection, features, Stage1 rules, transparent Top20/Top5 rank, T+1 raw-open
  entry, 20% sizing, maximum five positions, costs, capacity, corporate actions,
  missing-data behavior, and trailing previous-close -8% stop are identical.
- The extension decision is made at the planned H10 open from the H9 close
  already stored in position state. It may happen at most once. H20 exits at
  that session's tradable open under the unchanged execution contract.
- No XGBoost, 202-factor change, threshold scan, alternate hold period, or
  2024/2025 performance validation is allowed.

## Verification

Before challenger metrics are accepted:

1. The unchanged control must reproduce the sealed `2026-01-05..2026-08-28`
   CAP1 metrics when the current DB bundle is sliced to the sealed signal and
   execution boundaries.
2. Control and challenger consume the same in-memory candidate, selection,
   execution, and corporate-action objects.
3. Every buy remains T+1, no portfolio exceeds five positions, and each
   `TIME_EXIT_EXTENSION` event occurs only once per trade at its base H10
   boundary.
4. The experiment writes only configuration, hashes, manifests, aggregate
   metrics, and conclusions. It must not persist raw prices, factors, candidates,
   fills, positions, models, CSV, or Parquet files.

## Acceptance and Abandonment

Both Base and Stress must satisfy all of the following:

- total return strictly exceeds the paired control;
- PF is at least 2.0 and no lower than the paired control;
- absolute MaxDD is at most 15% and no worse than the paired control;
- return excluding the best week and return excluding the top three winners are
  positive and no lower than the paired control;
- trade win rate is at least 70%;
- maximum open positions is at most five.

Weekly returns of at least 5%, their count, ratio, and the result after removing
the best week are reported explicitly. They are not silently redefined as a
different target.

If any paired condition fails, permanently reject this exact rule. Do not rescue
it by trying H15/H25, changing zero to another profit threshold, reversing the
condition, changing TopN, or adding a gate/model.
