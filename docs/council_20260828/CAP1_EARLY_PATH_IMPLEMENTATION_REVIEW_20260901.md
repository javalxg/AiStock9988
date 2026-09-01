# CAP1 Early Path Implementation Review

Date: 2026-09-01

Scope:
`src/aistock9988/forward/early_path.py`,
`configs/strategy/cap1_early_path_forward_overlay.yaml`,
`scripts/quiet_forward_shadow_runner.py`,
`scripts/quiet_forward_rollup_runner.py`,
`scripts/quiet_forward_preflight.py`,
the early-path preregistration, and the preflight JSON.

No historical backtest was run and no 202-factor wide table is used.

## Verdict: BLOCK

The paired overlay architecture is sound, but two failure paths are not sealed
and break-event counting can be inflated. These must be fixed before running.

## P1 Findings

1. `BREAK_PENDING` is counted as a break event even when no earlier shadow sell
   occurred. `_next_sell_session` returns the control exit when no tradable open
   exists before it (`early_path.py:356-375`), and `_replace_sells` only creates
   a replacement when `overlay_exit < control_exit` (`early_path.py:245`). The
   path state remains `BREAK_PENDING`, so both settle/rollup count it via
   `path_state_counts["BREAK_PENDING"]`
   (`quiet_forward_rollup_runner.py:194`). This can satisfy the 15-event gate
   without any actual early exit.

   Required fix: count break events only for trade keys that have an actual
   earlier shadow sell (i.e., a replacement), or add a terminal state such as
   `BREAK_PENDING_NO_EARLY_EXIT` / `FOLLOW_CONTROL` when no earlier tradable
   exit exists, and exclude it from the break-event count.

2. `FAIL_PAIRED_FUNDING` and reconciliation failures are raised as uncaught
   `ValueError` (`early_path.py:291`, `early_path.py:533`). Settle/rollup do not
   catch them, so no sealed `FAIL_PAIRED_FUNDING` or `FAIL_RECONCILIATION`
   artifact is written. The preregistration requires a permanent failure
   outcome, not a crash.

   Required fix: have the runner catch overlay failures and write a terminal
   failure artifact (or have `apply_early_path_overlay` return a failure result)
   before aborting; the failure must be included in the hash chain.

## P2 Findings

1. `_decision_nav` (`early_path.py:486-492`) falls back to the first NAV row
   when no pre-entry NAV exists. For the first-ever decision this is the entry
   day NAV, not the signal-close decision NAV. This field is informational, but
   should use the control decision/signal NAV from the frozen selection ledger
   or fail when unavailable.

2. `position_events` is returned unchanged from the control
   (`early_path.py:310`), so replaced trades do not show an
   `EARLY_PATH_EXIT` event. Add an overlay exit event or mark the paired ledger
   as event-diagnostic-only to avoid audit ambiguity.

## Verified Correct

- First-terminal-state-wins is implemented: the loop stops on the first
  terminal state, including `CONTROL_EXIT_BEFORE_WINDOW_END`
  (`early_path.py:179-181`).
- Control exits before E2 terminate path observation before post-sell OHLC can
  be used.
- Next tradable open respects `_SELL_BLOCKED` and non-finite opens
  (`early_path.py:356-375`).
- Corporate actions/dividends up to the early-exit session are kept; later
  dividends are dropped (`early_path.py:391-396`, `early_path.py:419-430`).
- Slippage, sell commission, and stamp duty use the same cost scenario as the
  control (`early_path.py:145-148`).
- Restricted cash and slot reservation are modeled in NAV
  (`early_path.py:459-480`), and negative unrestricted cash triggers
  `FAIL_PAIRED_FUNDING` (currently as an uncaught error, see P1).
- Control buys are inherited unchanged and reconciliation checks identical
  buys and one sell per trade key (`early_path.py:515-541`).
- NAV is rebuilt from fills/actions/positions rather than copied from control.
- Right-tail recall and paired acceptance are implemented in
  `quiet_forward_rollup_runner.py:246-295`.
- Code/config/prereg hashes are included in the closure; the preflight JSON
  currently matches the overlay config and preregistration hashes.
- Empty control fills return an empty overlay result
  (`early_path.py:322-334`), and zero-trade forward days are supported.

## Decision

After the two P1 fixes, this implementation can be re-reviewed for GO. No
parameter scan, historical backtest, or 202-factor wide table is authorized.

## Second Review (2026-09-01) - Verdict: GO

The two P1 findings are resolved:

1. Effective early-exit count now comes from `paired_capital` length, which
   contains only replacements with an actual earlier shadow sell
   (`quiet_forward_rollup_runner.py:225`). Acceptance uses
   `effective_early_exit_count`, not raw `BREAK_PENDING` state count
   (`quiet_forward_rollup_runner.py:280-283`). `BREAK_PENDING` states without an
   earlier sell remain reported as `break_pending_state_count` but do not count
   toward the 15-event gate.
2. `EarlyPathFailure` is caught in both settlement and rollup. Failure writes
   `FAILURE.json`, copies strategy/overlay config/preregistration, and seals a
   code manifest before moving the batch or returning the rollup output
   (`quiet_forward_shadow_runner.py:369-379`, `quiet_forward_shadow_runner.py:487-509`,
   `quiet_forward_rollup_runner.py:152-174`).

Additional fixes verified:

- `_decision_nav` raises `FAIL_RECONCILIATION` instead of falling back to entry
  NAV when no signal-close NAV exists (`early_path.py:497-508`).
- `SHADOW_OVERLAY` events are added for every actual early exit
  (`early_path.py:511-534`).
- Path decision timestamps use 15:00 Asia/Shanghai on the session date
  (`early_path.py:362-364`).
- Freeze batches copy and hash overlay config/preregistration
  (`quiet_forward_shadow_runner.py:266-267`, `quiet_forward_shadow_runner.py:283-284`).

No blocking issue remains. The overlay stays inside the CAP1-20 forward lockbox,
uses only the canonical control ledger repricing, and does not touch the
202-factor wide table.
