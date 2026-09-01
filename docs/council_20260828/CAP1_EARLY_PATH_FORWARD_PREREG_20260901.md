# CAP1 Early Path Forward Preregistration

Status: `PREREGISTERED_NOT_STARTED`.

Independent contract and implementation reviews: `GO` in
`CAP1_EARLY_PATH_FORWARD_PREREG_DEEPSEEK_REVIEW_20260901.md` and
`CAP1_EARLY_PATH_IMPLEMENTATION_REVIEW_20260901.md`.

## Hypothesis

The unchanged CAP1-20 entry may be a true transition or a failed bounce. Price
path observed after an actual fill can distinguish those states more reliably
than another same-day historical ranker. A position that breaches `-5%` before
ever confirming `+3%` is expected to have a worse remaining H10 payoff; exiting
that path at the next tradable open should improve left-tail control without
removing confirmed right-tail winners.

This is an append-only forward shadow. No signal or fill before the first valid
CAP1-20 lockbox session on or after `2026-09-01` may enter the experiment.

## Frozen Control

The control is `reset_weak_confirm_v3_cap1_20_forward` and its existing forward
lockbox. Candidate selection, ranking, sparse abstention days, T+1 raw-open
entry, 20% decision-NAV sizing, 100% gross cap, maximum five positions, H10,
trailing previous-close `-8%` stop, ADV20 2% limit, missing-data actions, and
Base/Stress costs remain unchanged.

The shadow must consume the exact frozen candidate and selection partitions
from the control lockbox. It may not recompute, reorder, or backfill candidates.

## Early Path State Machine

For each actual control fill, call its fill session `E0`. The observation window
is exactly the three exchange sessions `E0`, `E1`, and `E2`; the entry session is
therefore included. Path thresholds use unslipped raw prices relative to the
actual entry session raw open:

```text
confirm_level = entry_raw_open * 1.03
break_level   = entry_raw_open * 0.95
```

At each session close, process only information available through that close.
`CONFIRMED`, `BREAK_PENDING`, `AMBIGUOUS_SAME_SESSION`, and
`UNSCORABLE_DATA_GAP` are terminal path states: the first terminal state wins
and no later session may rewrite it. If the linked control position sells at an
open before a path terminal state or the E2 close, observation stops before
using that session's OHLC and the terminal state is
`CONTROL_EXIT_BEFORE_WINDOW_END`. It follows the control and is not a break.
`NEUTRAL` is assigned only after a fully scorable E2 close with no earlier
terminal state.

1. If an earlier session already produced any terminal state, preserve that
   state. In particular, `BREAK_PENDING` cannot later become `CONFIRMED`, and
   `AMBIGUOUS_SAME_SESSION` cannot later become directional.
2. If the current session is the first terminal session and touches both
   levels, intraday order is
   unknowable from daily OHLC. State becomes `AMBIGUOUS_SAME_SESSION` and the
   position follows the control with no overlay exit.
3. If the current session is the first terminal session, raw high touches the
   confirm level, and raw low does not touch the break
   level, state becomes `CONFIRMED` permanently.
4. If the current session is the first terminal session, raw low touches the
   break level, and raw high does not touch the confirm level that session,
   state becomes `BREAK_PENDING`; the shadow exits at the next tradable raw
   open. A later recovery cannot cancel a pending exit.
5. If neither level is touched by the E2 close, state is `NEUTRAL`; the position
   follows the control.

If any of E0/E1/E2 lacks required eligible raw high/low data before a terminal
state is known, state becomes `UNSCORABLE_DATA_GAP` and follows the control. No
fill-forward or later replacement session is allowed.

The original stop and time exit remain active. If the control stop is already
pending before the overlay break, the existing exit keeps precedence. If both
triggers arise from the same close, both are recorded and the single next-open
exit is priced identically; no duplicate order is created. Untradable exits are
retained and retried under the control contract.

## Paired Evaluation

### Frozen Paired Capital

The control is the sole source of executable buys. For each cost arm, the
shadow inherits every actual control buy with the same trade key, symbol,
decision id, shares, entry session, entry price, and control decision NAV. It
may never create a buy absent from the control, resize a control buy, or use an
early exit to admit a later candidate that the control did not fill.

When the overlay exits a position early, its net proceeds become restricted
cash and its portfolio slot remains reserved until the linked control position
actually closes. The restricted cash is included in shadow NAV but is not
available for sizing or any new buy. On the linked control exit session the
restriction and slot reservation are released; the shadow does not receive the
control exit proceeds or any dividend paid after its own early exit. Thus the
shadow never reinvests an early exit before the control could have reused that
capital.

The shadow is a deterministic paired ledger repricing of the canonical control,
not an independently selected portfolio or a second backtest engine. Its cash,
restricted cash, reserved slots, positions, NAV, and fills must be reconstructed
from the canonical control ledgers plus close-known path events. Unrestricted
cash must remain non-negative at every session; a violation closes the mechanism
as `FAIL_PAIRED_FUNDING` rather than resizing or skipping a control buy.

Control and shadow use the same frozen decisions, execution panel, corporate
actions, sessions, and cost arm.
The preregistration, shadow strategy config, evaluator code, canonical engine,
and control lockbox identity must be hash-sealed before the first valid signal.

Formal evaluation starts only when the control has at least 26 completed signal
weeks or 60 closed trades, and the shadow has at least 15 unambiguous
`BREAK_PENDING` events that produce an actual shadow sell earlier than the
linked control sell. A break whose first tradable exit is the control exit does
not count toward this maturity gate. If the control is mature but effective
early exits are fewer than 15, collection continues until 52 signal weeks or
120 closed trades. Fewer than 15 effective early exits at that terminal boundary closes the mechanism as
`FAIL_INSUFFICIENT_EVENT_FREQUENCY` without changing thresholds.

The shadow passes only if both Base and Stress satisfy every condition:

- PF is at least 2.0 and strictly higher than the paired control;
- absolute MaxDD is at most 15% and no larger than the paired control;
- total return, return excluding the best week, and return excluding the three
  largest winning trades are each strictly higher than the paired control;
- return excluding the best week and top three winners remain positive;
- Stress total return is positive;
- the count of control trades whose economic return is at least +10% and which
  remain at least +10% in the shadow does not decrease.

Weekly 5% attainment and 70% trade win rate are reported as user objectives,
not used as hidden promotion gates.

## Permanent Stop Rules

Any failed paired condition permanently closes this exact overlay. Do not alter
`+3%`, `-5%`, the three-session window, same-session ambiguity handling, exit
timing, H10, trailing stop, TopN, sizing, or gross cap. Do not run it on seen
2026 history, reverse the meaning of confirm/break, or rescue it with a model.

## Required Artifacts

- pre-signal registration manifest and code/config hashes;
- append-only per-position path-event ledger containing the prior state, E-index,
  session, raw OHLC used, threshold-touch flags, resulting state, trigger reason,
  decision timestamp, and any linked single exit order;
- rollup counts for every terminal state plus `NEUTRAL`, reconciled exactly to
  the number of actual control fills eligible for path observation;
- paired-capital ledger containing trade key, inherited control buy and decision
  NAV, restricted cash, slot-reservation state, release session, linked control
  exit fill, unrestricted cash, and NAV adjustment;
- exact reconciliation that shadow and control have identical buy trade keys,
  shares, sessions, and prices; each side has exactly one economic sell per trade
  key, with only an eligible `BREAK_PENDING` trade permitted an earlier sell;
- paired control/shadow fills, positions, NAV, and corporate-action ledgers;
- Base/Stress paired metric report including right-tail recall;
- source cutoff and control lockbox manifest chain;
- final DeepSeek verification and decision.
