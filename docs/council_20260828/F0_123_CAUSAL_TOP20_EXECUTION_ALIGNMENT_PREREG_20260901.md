# F0-123 Causal Top20 Execution Alignment Preregistration

Date: `2026-09-01`

Status: `DIAGNOSTIC_FROZEN_BEFORE_RERUN`.

## Question

Why did the causal Stage-2 primary Top5 have a positive mean frozen path-stop
H10 label while its actual 2026 portfolio lost money?

## Frozen Replay

- Re-run the exact strategy, F0=123 columns, cross-sectional transform, causal
  Stage-1/Stage-2 models, weekly dates, labels, Top20 membership, Top5 ordering,
  T+1 execution, H10, stop, sizing, capacity and Base/Stress costs from
  `F0_123_CAUSAL_TOP20_EVENT_SELECTOR_PREREG_20260901.md`.
- Require the paired control and challenger portfolio metrics, sample hashes and
  model hashes to reproduce the completed experiment.
- Persist aggregate diagnostics only. Do not persist candidates, predictions,
  fills, prices, models, or any other business data.

## Frozen Diagnostics

For every policy and cost scenario, join actual BUY/SELL pairs to their frozen
signal identity and report:

1. all actual fills versus ranks 1-5 and replacement ranks 6-20;
2. frozen label mean and `>=10%` event rate of the actual-filled subset;
3. realized and economic return by TIME_EXIT versus STOP_LOSS;
4. label-minus-economic return and stop-trigger-to-exit-open gap;
5. rejection reasons and Base/Stress fill-set overlap.

No threshold, gate, TopN, factor, feature, model, target, holding-period, stop,
cost, or execution change is permitted. The output diagnoses the completed
experiment and cannot claim a new strategy improvement.

## Decision Rule

- If TIME_EXIT economic returns fail to reconcile directionally with their
  frozen labels beyond the known cost/slippage difference, stop strategy work
  and treat label/engine parity as an implementation blocker.
- If TIME_EXIT reconciles but actual-filled labels are materially worse than
  all primary Top5 labels, the portfolio-state/fill subset is the failure mode.
- If STOP_LOSS next-open returns are materially worse than the fixed `-8%`
  label, the label does not represent executable left-tail severity; close this
  target contract before any further F0 model work.
- If none of those mechanisms explains the loss, close this historical
  full-market F0 path and require genuinely fresh data for any follow-up.
