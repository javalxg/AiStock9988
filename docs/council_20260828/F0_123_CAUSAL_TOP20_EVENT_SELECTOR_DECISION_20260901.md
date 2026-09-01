# F0-123 Causal Top20 Event Selector Decision

Date: `2026-09-01`

## Verdict

Reject this exact Stage-2 classifier. The 2026 challenger returned `-33.75%`
Base and `-26.66%` Stress, with PF `0.528/0.673` and MaxDD
`-35.45%/-28.74%`. It improved both paired controls but did not approach CAP1
or any portfolio acceptance boundary.

Evidence:

- `F0_123_CAUSAL_TOP20_EVENT_SELECTOR_20260901/RESULT.md:18-23`
- `F0_123_CAUSAL_TOP20_EVENT_SELECTOR_20260901/acceptance.json:1-38`

This is a complete 2026 result rather than a partial model replay. Stage-1 used
951,000 causal training rows over 634 sessions, generated every one of the 84
historical/forward weekly cross-sections, and produced exactly 20 candidates per
week. Stage-2 generated all 31 forward selections. Every label-maturity, paired
membership, NAV, cash, position-cap, and execution-end check passed.

Evidence:

- `F0_123_CAUSAL_TOP20_EVENT_SELECTOR_20260901/RESULT.md:10-14`
- `F0_123_CAUSAL_TOP20_EVENT_SELECTOR_20260901/verification.json:2-31`

## What Was Learned

The conditional model learned some right-tail information but not enough
portfolio information. Among mature 2026 primary Top5 rows, event rate improved
from `15.44%` to `18.06%`, Top20-event recall improved by `5.15pp`, and mean
path-stop H10 label improved from `+0.39%` to `+0.76%`.

Evidence:

- `F0_123_CAUSAL_TOP20_EVENT_SELECTOR_20260901/event_diagnostics.json:2-20`

That lift did not survive portfolio formation. The Base challenger closed 60
trades with a `26.67%` win rate and lost `33.75%`; excluding its three largest
winners worsened return to `-49.71%`. This is not a three-winner strategy hidden
behind a weak aggregate result.

Evidence:

- `F0_123_CAUSAL_TOP20_EVENT_SELECTOR_20260901/paired_portfolio_metrics.json:3-36`

The positive all-signal Top5 label mean and negative realized portfolio are not
yet proof of an engine or label bug: the portfolio enters only the subset that
has free slots, may use ranks 6-20 as tradability replacements, and executes a
stop on the next tradable open while the training label records a fixed `-8%`
stop value. The observed gap is therefore an audit question that must be joined
at actual trade identity before another selection hypothesis is allowed.

## Next Action

Run one non-optimizing execution-alignment diagnostic on this frozen experiment:

1. Join every actual BUY/SELL pair back to its signal date, candidate rank,
   Stage-1/Stage-2 score, and already frozen path-stop label.
2. Report actual-filled versus all-primary Top5 label means, primary versus
   replacement ranks, TIME_EXIT versus STOP_LOSS returns, stop-trigger-to-open
   gaps, costs, and Base/Stress fill-set differences.
3. Persist aggregates only. Do not change factors, models, thresholds, TopN,
   holding period, costs, or the canonical engine.
4. If time-exit economic returns do not reconcile with labels after the known
   cost difference, treat that as an implementation blocker. If they reconcile,
   close this full-market F0 path historically and keep CAP1 as the control.

No rank-bucket cherry-pick, small-cap threshold, probability gate,
hyperparameter search, auxiliary data, or 202-factor work is authorized by this
result. Any single smaller-cap/lower-trend mechanism suggested by the descriptive
winner/loser family requires genuinely fresh-time validation.
