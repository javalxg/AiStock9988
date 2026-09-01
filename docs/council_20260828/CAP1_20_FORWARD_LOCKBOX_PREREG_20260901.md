# CAP1-20 Forward Lockbox Preregistration

Status: `PREREGISTERED_NOT_STARTED`.

## Purpose

This lockbox measures the future performance of the unchanged historical
CAP1-20 contract. It does not recompute or backfill any signal before
`2026-09-01`, and it does not claim that the historical `+32.44%` return will
continue.

The 202-factor wide table and its feature code are outside this experiment.
The only inputs are the existing fixed-rule features computed from market,
adjustment, limit, universe, and sparse event sources declared by the strategy.

## Frozen Contract

- Strategy: `reset_weak_confirm_v3_cap1_20_forward`.
- First permitted signal boundary: `2026-09-01`.
- First actual freeze: the first completed exchange session on or after that
  boundary for which every dense selection source is complete in QuantDB.
- Decision: daily at the completed T close; execution at T+1 raw open.
- Candidate view: up to 20. A sparse view, including zero candidates, is a
  valid frozen decision and is never filled with fallback history.
- Entries: up to five, maximum five open positions.
- Sizing: 20% of decision NAV per fill, 100% gross cap.
- Exit: H10, trailing close-to-close `-8%` stop executed at the next tradable
  open, with untradable exits retained and retried.
- Capacity: ADV20 participation capped at 2%.
- Costs: Base 0.10% and Stress 0.30% slippage per side, with configured
  commissions and stamp duty.
- Missing required stock-session data excludes that stock-session. Missing
  sparse-event rows do not imply an event and do not exclude an otherwise
  complete stock-session.

## Freeze And Settlement Semantics

Freeze uses data only through the signal close and records
`freeze_data_cutoff=asof`. It never requires a future H10 label or execution
horizon. Score, candidate, selection, strategy, data-content, and executable
code hashes are sealed before any future execution row is used.

Settlement uses a later `settlement_data_cutoff`. The canonical engine is run
against the frozen decisions, and a settlement is sealed only when every
position actually opened by both Base and Stress arms has closed. Readiness is
not determined by whether every candidate-view symbol has eleven eligible
future rows.

## Evaluation

Formal acceptance is deferred until at least 26 completed signal weeks or 60
closed trades. The hard evaluation remains:

- PF at least 2.0;
- absolute MaxDD at most 15%;
- return excluding the best week greater than zero;
- return excluding the three largest winning trades greater than zero;
- Stress total return greater than zero.

The requested 5% weekly return and 70% trade win rate remain visible objectives
in every rollup. They are not silently converted into daily settlement gates.

## Stop Conditions

- Any signal before `2026-09-01`, future local date, incomplete session, or
  required database cutoff is rejected.
- Any config, code-closure, partition, or hash-chain drift invalidates the
  operation rather than recomputing history.
- No threshold, ranking weight, TopN, holding period, stop, position size, or
  gross-cap adjustment is allowed inside this lockbox.
- Failure at the mature evaluation closes this exact contract. It may not be
  rescued by changing the frozen parameters.

## Commands

Run preflight without choosing a future date:

```bash
PYTHONPATH=src python scripts/quiet_forward_preflight.py \
  --strategy configs/strategy/reset_weak_confirm_v3_cap1_20_forward.yaml \
  --lockbox docs/council_20260828/CAP1_20_FORWARD_LOCKBOX
```

When preflight reports `READY_TO_FREEZE`, freeze exactly its `target_asof`:

```bash
PYTHONPATH=src python scripts/quiet_forward_shadow_runner.py \
  --mode freeze \
  --asof YYYY-MM-DD \
  --strategy configs/strategy/reset_weak_confirm_v3_cap1_20_forward.yaml \
  --output docs/council_20260828/CAP1_20_FORWARD_LOCKBOX
```

Settlement is attempted only through an execution date already present in all
dense execution sources:

```bash
PYTHONPATH=src python scripts/quiet_forward_shadow_runner.py \
  --mode settle \
  --asof YYYY-MM-DD \
  --execution-end YYYY-MM-DD \
  --strategy configs/strategy/reset_weak_confirm_v3_cap1_20_forward.yaml \
  --output docs/council_20260828/CAP1_20_FORWARD_LOCKBOX
```
