# Dragon-Tiger Pullback Reclaim V1 Preregistration

Date: `2026-09-01`

Status: `PROPOSED_FOR_DEEPSEEK_REVIEW; NO_CODE_OR_BACKTEST_YET`.

## Hypothesis

An upward dragon-tiger event with positive true-institution net buying contains
information, but the next-session opening gap already prices most of it. A
trade is permitted only if that gap is subsequently released and price later
reclaims the event close with expanding participation. This tests a second
executable demand leg; it does not chase the first overnight leg.

The prior path evidence is frozen in
[`DRAGON_TIGER_PATH_DIAGNOSTIC_20260901.json`](DRAGON_TIGER_PATH_DIAGNOSTIC_20260901.json).
This V1 definition was written before inspecting any pullback-reclaim outcome.

## Source And PIT Contract

- Event source: `top_list_ts` plus `top_inst_ts`, currently ending
  `2026-08-18`. No event is created after that source cutoff.
- Price and amount source: the canonical V3 execution panel loaded fresh from
  QuantDB through the dense sources' common available `trade_date` cutoff.
- An event is known only after event-session close. A confirmation is known
  only after confirmation-session close. Every order executes no earlier than
  the next tradable raw open.
- Every event, observation, confirmation, entry, held mark, and exit must pass
  the applicable V3 universe/data contract. A missing required stock-day drops
  only that stock-day; it never drops the market session or invokes a fallback
  rule.
- `update_time` and ingestion time are provenance only. Eligibility is based on
  the event `trade_date` and explicit decision/entry sequence.

## Frozen Event Definition

For stock `S` on event session `T`:

1. At least one distinct `top_list_ts.reason` contains `涨幅`.
2. All distinct top-list reasons are retained as a sorted set. The strategy
   never sums `top_list_ts.net_amount` across reasons.
3. Only `top_inst_ts` rows with `exalter='机构专用'` are institution rows. Their
   `net_buy` is summed by `(T, S)` and must be strictly positive.
4. The top-list daily `amount` denominator is taken once per stock-day after an
   equality-tolerance audit across duplicate reasons; it is never summed.
5. `S` must pass the canonical V3 universe on `T` and every later decision or
   execution session. Convertible bonds, Beijing-board symbols, PIT ST stocks,
   immature listings, and non-stock records are excluded by that contract.

An event identity is `(T, S)`. While an event for `S` is observing, confirmed,
ordered, or held, later events for `S` are logged as
`OVERLAPPING_EVENT_SKIPPED`; they do not reset the clock or create another
sample.

## Frozen Pullback And Reclaim State Machine

All comparisons use economic prices; raw prices are used only for accounting
and fills.

1. `GAP_CONFIRMED`: on `T+1`, `economic_open(T+1) > economic_close(T)`. The
   strategy never buys at this open. A non-positive gap expires the event.
2. `PULLBACK`: on one of `T+1` through `T+4`, the session economic close is at
   or below `economic_close(T)`. The first such close is frozen as `P`.
3. `RECLAIM`: on a later session `R`, no later than `T+5`, all three conditions
   hold at the close: `economic_close(R) > economic_close(T)`,
   `economic_close(R) > economic_high(R-1)`, and `amount(R) > amount(R-1)`.
4. `ENTRY`: submit one order after `R` close and attempt to fill at the next
   tradable raw open. An untradable next open skips this event; it does not delay
   entry or replace the rule with another trigger.
5. Events without a pullback followed by a distinct reclaim session by `T+5`
   expire. A pullback and reclaim cannot be credited to the same session.

There is one definition only. `T+4`, `T+5`, the event-close anchor, prior-high
reclaim, and participation expansion are not search dimensions in V1.

## Ranking And Portfolio

- Decision frequency: daily, event-driven abstention allowed.
- Rank active reclaims by institution intensity
  `institution_net_buy / top_list_daily_amount`, descending; then by
  `amount(R) / amount(R-1)`, descending; then `ts_code`, ascending.
- Fill only available slots in rank order, with at most five concurrent
  positions and at most 20% of decision NAV per position.
- Duplicate symbols are skipped. Pending exits occupy a slot. No XGBoost model,
  market-regime gate, frozen 202-factor input, or candidate-threshold scan is
  allowed.

## Exit And Costs

- Time exit: H10 from actual fill, next tradable raw open.
- Risk exit: the canonical close-triggered `-8%` trailing-from-prior-close stop,
  executed at the next tradable raw open. An untradable exit is retained and
  retried without freeing the slot.
- Corporate actions: canonical raw-price accounting plus the existing action
  ledger.
- Base costs: 10 bps slippage each side, 3 bps commission each side, 5 bps
  stamp duty on sells.
- Stress costs: 30 bps slippage each side with the same commissions and duty.
- End policy: mark open positions at the dense execution cutoff. Late events and
  open positions remain in the ledger; they are not removed to manufacture a
  mature sample.

## Required Outputs And Acceptance

The run covers every 2026 event session from the first available session through
the dragon-tiger source cutoff, with execution through the dense QuantDB cutoff.
2025 is a separate mechanism-stability replay, never pooled into the 2026 target
claim.

Required outputs include event-state counts, every exclusion reason, overlapping
event count, untradable-entry count, open positions at cutoff, Base/Stress NAV,
orders/fills/positions, weekly returns, closed-trade outcomes, ex-best-week and
ex-top-three-profit returns, data manifest, code/config hashes, and PIT checks.
Only compact aggregates are committed; business-data ledgers remain ignored.

V1 advances only if both Base and Stress satisfy all gates:

- mean return across every calendar week in the covered 2026 interval is at
  least `+5%`;
- closed-trade win rate is at least `70%`;
- portfolio PF is at least `2.0`;
- absolute MaxDD is at most `15%`;
- return excluding the best week is positive;
- return excluding the three largest profitable trades is positive;
- at least 20 closed trades and at least 20 active confirmation sessions.

Failure of any gate rejects V1. Zero or very few triggers, year-direction
reversal, PF below 1 in either year, or performance dominated by the largest
three trades is evidence against the mechanism, not permission to loosen or
scan the rule.

## Review Request

DeepSeek must specifically review event availability, reason/net-buy
aggregation, overlapping-event treatment, state timing, next-open execution,
untradable handling, end-of-data treatment, ranking normalization, stop
semantics, and all post-selection risks before implementation. Review comments
must be saved verbatim and every required change incorporated or explicitly
arbitrated before code execution.
