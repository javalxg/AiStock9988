# CAP1 E2 Nonpositive Exit V1 Preregistration

Status: `PREREGISTERED_BEFORE_RESULT`.

## Research question

The unchanged CAP1 entry has useful 2026 candidate quality, while its static
entry features separate winners from losers only weakly. The sealed trade-path
diagnostic found that 18 of 22 positions above entry at the E2 close eventually
won, versus 5 of 19 positions at or below entry. This experiment tests one exact
exit rule suggested by that mechanism; it does not retune CAP1 selection.

## Frozen challenger

- Run every exchange signal session from `2026-01-01` through the latest signal
  date executable by T+1 at the common required-source database cutoff.
- Preserve the CAP1 universe, required-data policy, Stage1 expression, ranking,
  Top20 view, five entries, five-position cap, 20% decision-NAV sizing, H10,
  trailing close stop, execution prices, capacity, Base/Stress costs, and
  corporate-action accounting.
- The actual entry session is E0. Observe the economic close on E0, E1, and E2.
  If the position is still active and its E2 economic close is less than or
  equal to its actual slipped entry economic price, trigger one exit at the next
  tradable raw open. Existing stop exits have precedence.
- If the required E2 stock-session data is unavailable, do not substitute a
  later date: this position follows unchanged CAP1. Missing data for another
  stock cannot suppress the date or the rest of the universe.
- Early exits release cash and slots normally. This is the complete challenger
  strategy, not a paired repricing diagnostic.

## Prohibitions

No threshold, E-index, TopN, hold, stop, sizing, weight, or gate scan; no model;
no 202-factor change; no auxiliary table; no 2024/2025 performance validation;
no raw business-data artifact. If this exact rule fails, delete its executable
configuration and keep only aggregate rejection evidence.

## Acceptance

Both Base and Stress must have PF at least 2, absolute MaxDD at most 15%, trade
win rate at least 70%, positive return after excluding the best week, positive
return after excluding the three largest winners, at least 20 closed trades,
and at most five open positions. Total return and weekly 5% attainment are
reported. Promotion additionally requires higher total return, PF, and
exclusion returns than unchanged CAP1 without worsening MaxDD.
