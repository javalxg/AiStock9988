# F0-123 Causal Top20 Execution Alignment Decision

Date: `2026-09-01`

## Reproduction

The diagnostic exactly reproduced the completed experiment. Training-feature,
prediction-feature and causal-Top20 key hashes are identical; both Stage-1 and
Stage-2 model manifests are byte-identical; paired portfolio metrics are also
byte-identical. All 2026 portfolio conclusions therefore refer to the same
candidates, models and execution paths.

Evidence:

- `F0_123_CAUSAL_TOP20_EXECUTION_ALIGNMENT_20260901/sample_audit.json:1-57`
- `F0_123_CAUSAL_TOP20_EXECUTION_ALIGNMENT_20260901/verification.json:2-31`
- `F0_123_CAUSAL_TOP20_EXECUTION_ALIGNMENT_20260901/paired_portfolio_metrics.json:1-146`

## Root Cause

The primary failure is portfolio-state sampling, not Top20 replacement. The
challenger had 155 mature primary Top5 labels across all signal weeks with mean
`+0.76%`, but Base actually bought only 61 stocks and only 56 had mature labels.
That actual-filled subset had mean frozen label `-0.61%` and event rate `16.07%`.
Every fill was rank 1-5; ranks 6-20 contributed zero trades.

Evidence:

- `F0_123_CAUSAL_TOP20_EVENT_SELECTOR_20260901/event_diagnostics.json:2-7`
- `F0_123_CAUSAL_TOP20_EXECUTION_ALIGNMENT_20260901/execution_alignment.json:18-62`

Weekly batches and H10 holding occupy all five slots for approximately two
weeks. The portfolio therefore does not implement the cross-sectional mean over
every weekly Top5; it realizes a path-dependent subset determined by prior
entries and stop exits. The classifier improved the population statistic but
did not improve the subset that capital could enter.

The second failure is the fixed stop label. For Base challenger STOP_LOSS trades,
the training label was `-8%`, while actual economic return averaged `-11.64%`
and realized return averaged `-11.74%`. The next-open gap averaged `+0.42%`, so
the underestimation was not caused by an adverse overnight exit gap on average;
daily close crossing overshot the `-8%` boundary before execution.

Evidence:

- `F0_123_CAUSAL_TOP20_EXECUTION_ALIGNMENT_20260901/execution_alignment.json:22-29`

There is no general label/engine parity bug for normal time exits. Base
challenger TIME_EXIT labels averaged `+7.33%`, economic returns averaged
`+7.12%`, and realized returns averaged `+7.00%`. The `0.21pp` label/economic
difference is consistent with the frozen Base slippage contract.

Evidence:

- `F0_123_CAUSAL_TOP20_EXECUTION_ALIGNMENT_20260901/execution_alignment.json:31-38`

Stress outperformed Base by changing the path-dependent fill set, not because
higher costs helped the same trades. The challenger shared 57 buys, but Base had
4 unique buys and Stress had 12; fill-set Jaccard was only `0.781`. Stress actual
fills had mean label `+0.54%`, versus Base `-0.61%`.

Evidence:

- `F0_123_CAUSAL_TOP20_EXECUTION_ALIGNMENT_20260901/execution_alignment.json:2-8`
- `F0_123_CAUSAL_TOP20_EXECUTION_ALIGNMENT_20260901/execution_alignment.json:41-46`
- `F0_123_CAUSAL_TOP20_EXECUTION_ALIGNMENT_20260901/execution_alignment.json:92-105`

## Decision

Close this exact historical full-market F0=123 path. Another XGBoost parameter,
rank bucket, TopN, threshold, stop threshold, holding-period scan or calendar
phase would optimize against an already observed path and is not authorized.

CAP1 remains the current portfolio control. The original 123-factor data still
has research value, but the next valid use must satisfy both conditions:

1. train against an executable-return contract that records the actual stop
   crossing and next-open exit rather than clamping every stop to `-8%`;
2. validate portfolio formation on genuinely fresh signal dates, because the
   weekly/H10 slot path for all historical 2026 dates is already observed.

Until fresh dates accumulate, the active work is the already frozen CAP1
forward lockbox and its early-path risk overlay, not another historical F0
optimization.
