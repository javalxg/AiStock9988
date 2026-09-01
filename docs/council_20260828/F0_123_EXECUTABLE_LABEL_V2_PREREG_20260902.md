# F0-123 Executable Label V2 Preregistration

Date: `2026-09-02`

Status: `LABEL_CONTRACT_FROZEN_BEFORE_V2_MODEL_OR_FORWARD_RETURN`.

## Evidence

The rejected Stage-2 classifier did not fail because normal time exits were
misaccounted. Base TIME_EXIT labels averaged `+7.33%` and actual economic returns
averaged `+7.12%`. Its fixed stop label did fail to describe executable loss:
STOP_LOSS labels were `-8%`, while actual economic exits averaged `-11.64%`.

Evidence:

- `F0_123_CAUSAL_TOP20_EXECUTION_ALIGNMENT_20260901/execution_alignment.json:22-38`
- `F0_123_CAUSAL_TOP20_EXECUTION_ALIGNMENT_DECISION_20260901.md:35-58`

## Frozen Label Contract

- Profile id: `label.executable_path_open_open_t10_base.v1`.
- Signal at session close; entry attempt at the next session open.
- A T+1 candidate blocked by missing required data, suspension, limit-up,
  zero-volume, or out-of-universe status receives no entry label.
- Entry reference is the actual economic open with Base buy slippage.
- A stop triggers on an eligible close at or below `-8%` from that fill
  reference. The trigger return is recorded without clamping.
- Stop exit is the next sellable economic open; suspension, limit-down,
  zero-volume, missing required data, and out-of-universe sessions retry.
- Otherwise time exit begins at entry plus ten sessions and uses the same
  sellability retry contract.
- Label return includes Base slippage, buy/sell commissions, and stamp duty.
- Label availability is the actual exit session open. No label can train a model
  before that timestamp.
- Returns are not clipped or replaced with the stop threshold.

The implementation target is
`src/aistock9988/labeling/executable_path.py`. Delta provides only the historical
execution question; no old label or strategy code may be copied.

## Audit Before Use

Before training any V2 model:

1. Run the builder on database-only historical execution rows.
2. Report entry rejection, stop/time-exit, retry, unresolved-exit, minimum,
   maximum, mean-stop-crossing, and mean executable-return aggregates.
3. Reconcile a deterministic trade sample against the canonical engine's
   economic return and exit date.
4. Persist aggregate audit only; do not persist labels, prices, fills, or models.

## Model And Validation Boundary

No V2 model parameters are authorized by this document. After the label audit,
a separate model preregistration may use 2024-2025 mature labels for training.
All dates through `2026-09-01` are considered observed and may not be used to
claim V2 improvement. Strategy return acceptance begins only with the first
fully covered F0 signal after that date.

The database cutoff is binding. As of this registration, market execution data
ends `2026-09-01`, `daily_basic_ts` ends `2026-08-28`, and
`stock_factor_pro_ts` ends `2026-08-21`; no September F0 selection may be emitted
until required sources share the signal date.
