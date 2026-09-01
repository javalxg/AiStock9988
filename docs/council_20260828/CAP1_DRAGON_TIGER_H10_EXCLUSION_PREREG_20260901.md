# CAP1 Dragon-Tiger H10 Exclusion Preregistration

Date: `2026-09-01`

Status: `PREREGISTERED_BEFORE_PORTFOLIO_RESULT`.

## Question

The unchanged dragon-tiger pullback/reclaim V1 lost `-29.24%` Base and its
entry-state variables did not distinguish winners from losers. This experiment
tests one narrower hypothesis: a recently confirmed V1 event identifies an
overheated path that CAP1-20 should avoid, rather than a stock that CAP1-20
should buy or rank higher.

This is a historical diagnostic on already seen 2026 data. It is not an
out-of-sample or live-performance claim.

## Frozen Control

- Control strategy: `reset_weak_confirm_v3_cap1_20`.
- Signal range: `2026-01-05` through `2026-08-13`.
- Execution range: through `2026-08-28`.
- Same canonical feature engine, rule selector, backtest engine, Base/Stress
  costs, T+1 open entry, H10 hold, trailing close stop, 20% sizing, and maximum
  five positions.
- Control metrics must reproduce the sealed CAP1-20 result before the overlay
  result is accepted.

## Single Overlay Rule

1. Rebuild the unchanged dragon-tiger V1 confirmed-event ledger using the
   already frozen definition: upward top-list reason, positive true-institution
   net buy, positive T+1 opening gap, pullback by T+4, and later reclaim by T+5.
2. On each CAP1 signal session, inspect only V1 confirmations whose confirmation
   session is known by that close.
3. Exclude a CAP1 candidate when the same stock has a V1 confirmation on the
   signal session or any of the preceding nine exchange sessions. This is one
   fixed ten-session state window, equal to CAP1's existing H10 holding horizon.
4. Preserve CAP1 ranks and allow its existing next-ranked candidate to fill the
   vacancy. Do not rerank using any dragon-tiger value.

No other event-age window, institution threshold, gap threshold, volume ratio,
event reason, CAP1 factor, stop, holding period, or portfolio parameter may be
tested in this experiment.

## PIT And Missing Data

- The exclusion uses `confirmation_session <= CAP1 signal_session`; the
  underlying event, pullback, and reclaim observations must all be no later than
  that confirmation session.
- Entry remains the next tradable open after the CAP1 signal.
- Missing required stock-day data excludes only that stock-day from constructing
  the V1 event. There is no relaxed fallback.
- Event data stop at the actual database cutoff. Dense execution data stop at
  the frozen control's execution end.
- No write timestamp is used as an economic-time restriction; `trade_date` and
  the event-state chronology define PIT eligibility.

## Acceptance And Abandonment

The overlay advances only if all of the following hold in both Base and Stress:

- total return is greater than the same-bundle CAP1 control;
- portfolio PF is not lower than the control and remains at least `2.0`;
- MaxDD is no worse than the control and remains within `15%`;
- return excluding the best week remains positive;
- at least 20 trades close and the five-position cap is respected.

If no executed control entry is affected, the overlay is operationally
irrelevant and is rejected. Any other acceptance failure rejects the overlay
unchanged. The result will not be repaired with a threshold or window scan.

## Artifacts

Only configuration/code hashes, aggregate overlap counts, portfolio metrics,
verification results, and Markdown/JSON conclusions may be persisted. Raw
market, factor, event, candidate, fill, position, CSV, or Parquet business data
must not be written to the repository.

The frozen 202-factor system and XGBoost are out of scope.
