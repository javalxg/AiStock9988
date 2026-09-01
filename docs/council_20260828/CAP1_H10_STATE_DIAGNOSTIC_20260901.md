# CAP1 H10 State Diagnostic

Status: `DIAGNOSTIC_SEEN_HISTORY_HYPOTHESIS_ONLY`.

## Scope and definition

This is a descriptive cut of the 39 Base `TIME_EXIT` trades from
`RESET_WEAK_CONFIRM_V3_CAP1_20_2026_TO_0828_20260901`. It is not a new
portfolio backtest and does not change the sealed +32.44% result.

- `H9_POSITIVE`: the last eligible held-position close before the scheduled H10
  raw-open sale had positive unrealized return after Base entry slippage.
- `H9_NONPOSITIVE`: the same return was zero or negative.
- Post-exit 5/10-session return: economic open on the fifth/tenth exchange
  session after the control sale divided by the unslipped economic open on the
  control sale session, minus one.
- No threshold other than zero, no holding-period alternative, and no model was
  scanned. Missing future endpoints remain missing rather than being replaced.

Source ledgers:

- `backtests/base/fills.parquet`
- `backtests/base/positions.parquet`
- `ledgers/execution_panel.parquet`

All paths above are under
`docs/council_20260828/RESET_WEAK_CONFIRM_V3_CAP1_20_2026_TO_0828_20260901/`.

## Result

| Last known state | Trades | Control net-win rate | Control mean return | Post +5 mean | Post +10 mean | Valid post +10 positive |
|---|---:|---:|---:|---:|---:|---:|
| `H9_POSITIVE` | 23 | 95.65% | +11.87% | +2.96% | +4.35% | 63.64% (14/22) |
| `H9_NONPOSITIVE` | 16 | 6.25% | -4.43% | -1.62% | -3.95% | 42.86% (6/14) |

The state at the last close before the fixed H10 sale separates both the
already-earned trade outcome and what happened after the sale. This is stronger
economic evidence for path-conditioned holding than for another static entry
ranker. It does not prove that extending `H9_POSITIVE` trades improves portfolio
return because an extension also ties up cash and slots that the control can
reuse.

## Decision

Keep this as a separate forward-only right-tail hypothesis. Do not combine it
with the registered early-path break overlay: the early overlay tests left-tail
control, while an H10 extension tests right-tail capture. A future extension
experiment must model its own cash, blocked entries, maximum five positions,
Base/Stress costs, corporate actions, trailing stop, and H20 terminal exit. It
must be hash-registered before its first eligible signal and must never be run
on the seen 2026 CAP trades for promotion.
