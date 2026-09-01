# CAP1 Early Path Forward Preregistration - DeepSeek Review

Date: 2026-09-01

Reviewed document:
`docs/council_20260828/CAP1_EARLY_PATH_FORWARD_PREREG_20260901.md`

Scope: original fixed rules / CAP1-20 only. The 202-factor wide table is not
read, referenced, or recommended.

## Verdict: BLOCK

The hypothesis, forward-only boundary, paired evaluation, and sample gates are
reasonable. The early-path state machine has a precedence conflict that must be
fixed before implementation: a `BREAK_PENDING` state can later be reversed by
rule 3 if a subsequent session touches the confirm level.

## Findings

1. E0 is correctly included in the observation window
   (`CAP1_EARLY_PATH_FORWARD_PREREG_20260901.md:30-32`). The decision is made
   from E0/E1/E2 close data and the earliest possible overlay exit is E1/E2/E3
   next open, so this is causal and not a future leak.

2. Same-session dual-touch is handled as `AMBIGUOUS_SAME_SESSION`
   (`CAP1_EARLY_PATH_FORWARD_PREREG_20260901.md:44-46`), which is acceptable.
   It must only apply when no earlier terminal state exists.

3. Data gaps are fail-closed: `UNSCORABLE_DATA_GAP` follows the control and no
   fill-forward is allowed (`CAP1_EARLY_PATH_FORWARD_PREREG_20260901.md:56-58`).

4. Original H10 and trailing `-8%` stop remain active, and same-close dual
   triggers produce one next-open exit (`CAP1_EARLY_PATH_FORWARD_PREREG_20260901.md:60-64`).
   This is correct once state precedence is defined.

5. The minimum 15 unambiguous `BREAK_PENDING` events, continuation to 52 signal
   weeks / 120 closed trades, and terminal
   `FAIL_INSUFFICIENT_EVENT_FREQUENCY` are reasonable
   (`CAP1_EARLY_PATH_FORWARD_PREREG_20260901.md:73-78`).

6. No parameter scan or seen-history leakage was found. The `+3%/-5%`
   thresholds, three-session window, and ambiguity rule are frozen.

## Blocking Issue

State precedence is ambiguous between rules 3 and 4:

- Rule 3 sets `CONFIRMED` when the current session high touches confirm and low
  does not touch break (`CAP1_EARLY_PATH_FORWARD_PREREG_20260901.md:47-48`).
- Rule 4 sets `BREAK_PENDING` when break touched earlier and the current session
  high does not touch confirm, then states "a later recovery cannot cancel a
  pending exit" (`CAP1_EARLY_PATH_FORWARD_PREREG_20260901.md:49-52`).

If break is touched at E0 and confirm is touched at E1, rule 3 would override the
earlier `BREAK_PENDING`, contradicting rule 4. The same issue applies to
same-session dual-touch after an earlier break.

Required fix: define a strict precedence order, for example

```text
CONFIRMED > BREAK_PENDING > AMBIGUOUS_SAME_SESSION > NEUTRAL
```

with the following semantics:

- Once `CONFIRMED`, no later break or dual-touch can change it.
- Once `BREAK_PENDING`, no later confirm or dual-touch can cancel it.
- `AMBIGUOUS_SAME_SESSION` applies only if no earlier terminal state exists and
  the current session is the first to touch both levels.
- `NEUTRAL` applies only when no level is touched through E2 and no earlier
  terminal state exists.

## Required Artifacts To Add

- Per-position path-event ledger recording each E0/E1/E2 session state,
  `confirm_level`, `break_level`, raw high/low, and terminal state.
- Counts of `CONFIRMED`, `BREAK_PENDING`, `AMBIGUOUS_SAME_SESSION`,
  `NEUTRAL`, and `UNSCORABLE_DATA_GAP`.
- Explicit paired comparison of overlay exit vs control stop on the same close,
  proving no duplicate order and a single next-open fill.

## Decision

After the state-machine precedence fix and artifact additions are written into
the preregistration, this review upgrades to GO. The overlay must remain
forward-only and may not touch the 202-factor wide table.

## Second Review (2026-09-01)

Reviewed preregistration revision:
`docs/council_20260828/CAP1_EARLY_PATH_FORWARD_PREREG_20260901.md`
(SHA256 `07b2236700264f224fce358656c584aa6ba58decb895361fa833134620190fab`).

## Verdict: GO

The state-machine conflict is resolved. `CONFIRMED`, `BREAK_PENDING`,
`AMBIGUOUS_SAME_SESSION`, and `UNSCORABLE_DATA_GAP` are terminal states; the
first terminal state wins and later sessions cannot rewrite it. `NEUTRAL` is
only assigned after a fully scorable E2 close with no earlier terminal state.
Same-session dual-touch remains `AMBIGUOUS_SAME_SESSION` only when it is the
first terminal session.

The required per-position path-event ledger now specifies prior state, E-index,
session, raw OHLC, threshold-touch flags, resulting state, trigger reason,
decision timestamp, and linked single exit order. Rollup counts for every
terminal state plus `NEUTRAL` must reconcile exactly to eligible actual control
fills.

No parameter scan, seen-history backtest, or 202-factor wide-table usage is
authorized. The overlay stays inside the CAP1-20 forward lockbox and uses the
canonical engine. Formal evaluation requires at least 26 signal weeks or 60
closed control trades and 15 unambiguous `BREAK_PENDING` events, with a 52-week
/ 120-trade terminal boundary. Any failed paired condition permanently closes
this overlay.

## Third Review - Paired Entry And Capital Semantics (2026-09-01)

The preregistration was not yet frozen on how early exit affects slots, cash,
and subsequent entries. This must be resolved before implementation.

## Choice: Forced-Identical-Fills Overlay, Frozen Capital Until Control Exit

The shadow is not an independent portfolio. It is a re-pricing overlay on the
control's actual fills:

- Shadow consumes the exact control fill set, shares, entry prices, and decision
  NAV. It never creates a buy that the control did not make, and it never skips
  a control buy.
- When an early-path `BREAK_PENDING` exits a position earlier than the control,
  the released cash and position slot are frozen and are NOT reusable for new
  entries inside the shadow.
- Frozen cash/slot is released only when the control's corresponding position
  actually closes (its H10, stop, or retried exit fill). Until then, the shadow
  holds the freed proceeds as a separately flagged "frozen cash" line.
- Shadow NAV is computed as control NAV adjusted by the difference between the
  shadow exit fill and the control exit fill for each affected position, with
  no reinvestment of early proceeds.
- Because the shadow cannot create new buys, control cash insufficiency never
  produces a shadow-only fill or a blocked control winner. The paired
  comparison remains per-fill and auditable.

This is the only choice that satisfies "same canonical control decisions",
prevents divergent fills, preserves right-tail recall measurement, and remains
realistic. An independent shadow portfolio would create fills the control did
not make and would make the paired comparison ambiguous.

## Verdict On Current Preregistration: BLOCK

The current preregistration does not define the above paired capital contract.
Before implementation, it must add a "Frozen Paired Capital" section covering:

- exact control-fill inheritance and no-new-buy rule;
- early-exit proceeds and slot freeze until the control's actual exit fill;
- shadow use of control decision NAV and no reinvestment;
- per-position ledger fields for frozen cash, release date, and linked control
  exit fill;
- reconciliation showing shadow fills equal control fills by trade key.

After those clauses are added, this review upgrades to GO. No second engine,
no 202-factor wide table, and no historical backtest is permitted.

## Fourth Review - Frozen Paired Capital Final (2026-09-01)

Reviewed preregistration revision:
`docs/council_20260828/CAP1_EARLY_PATH_FORWARD_PREREG_20260901.md`
(SHA256 `b21406a756b1d20aac1b7164ab14a0f3e0a5e0988f2209b005885b305929eb6d`).

## Verdict: GO

The Frozen Paired Capital contract is now complete and machine-testable:

- Control actual buys are the only executable buys; shadow inherits identical
  trade key, symbol, decision id, shares, entry session, entry price, and
  control decision NAV.
- Early-exit proceeds become restricted cash and the slot stays reserved until
  the linked control position actually closes; restricted cash counts in NAV
  but is unavailable for sizing or new buys.
- The shadow does not receive control exit proceeds or post-early-exit
  dividends, and it never reinvests before the control could have reused the
  capital.
- The shadow is a deterministic paired ledger repricing of the canonical
  control, not a second engine or independent portfolio.
- Unrestricted cash must be non-negative at every session; any violation closes
  the mechanism as `FAIL_PAIRED_FUNDING`.
- Required artifacts include the paired-capital ledger and exact buy/sell
  trade-key reconciliation: identical buy trade keys, shares, sessions, and
  prices; exactly one economic sell per trade key on each side; only an
  eligible `BREAK_PENDING` trade may have an earlier sell.

No parameter scan, seen-history backtest, or 202-factor wide-table usage is
authorized. The overlay remains append-only inside the CAP1-20 forward lockbox.
This review is now final GO.

## Fifth Review - Control Exit Before Path Window End (2026-09-01)

## Verdict: GO

The proposed `CONTROL_EXIT_BEFORE_WINDOW_END` terminal state is correct and
should be added to the preregistration.

Required semantics:

- If the control position is sold at an open before E2 close and no earlier
  path terminal state exists, path observation terminates at that control sell
  open.
- Only information available while the position was still held may be used.
  Post-sell session OHLC must not be evaluated for confirm/break.
- `CONTROL_EXIT_BEFORE_WINDOW_END` is a first-terminal-state-wins state: it
  follows the control, triggers no overlay exit, and is not counted as a
  `BREAK_PENDING` event.
- The state is included in the per-position path-event ledger and in the rollup
  terminal-state counts.
- If a control exit open and a path break resolve on the same open, the control
  exit precedence from the frozen paired capital contract applies; the trade
  still has exactly one economic sell per side.

No more favorable alternative exists that is both causal and single-engine:
continuing to score OHLC after the position is gone would use non-held prices,
and treating the trade as `NEUTRAL` would falsely claim a complete E2 window.

After this clause is added, the preregistration remains final GO.
