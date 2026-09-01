# CAP1 Candidate Opportunity Diagnostic

Status: `DIAGNOSTIC_SEEN_2026_HISTORY_NO_STRATEGY_CHANGE`.

This is a candidate-level economic-price counterfactual, not a new portfolio
backtest. It uses only sealed 2026 signal/performance dates and reproduces each
closed engine trade's economic return before reporting aggregate evidence.

## Candidate quality

- 341 of 342 in-view candidate
  events have a closed T+1/H10/original-stop label. Mean net return is
  +1.46%, win rate is 57.8%,
  and the >=10% rate is 16.1%.
- Ranks 1-5 average +2.10% with 63.3%
  winners. Ranks 6-10 average +0.32% with
  49.4% winners. The existing transparent rank therefore has
  useful top-of-list separation; the result does not justify reversing it.
- Market state is the strongest available static separator: winner entries had
  mean 20-session market return -9.00% versus
  -5.95% for losers, with Spearman
  -0.271. Other frozen entry features are weak.

## Portfolio opportunity cost

- On signal days where at least one entry slot was available, candidate events
  averaged +1.75% with 49.5%
  winners. Candidates appearing while the five-position portfolio was full
  averaged +1.32% with 61.5% winners.
- This identifies a scheduling/holding problem, not proof of an alternative
  portfolio rule. Isolated candidate returns overlap and cannot be summed.
- The tempting rule "when full, replace the weakest nonpositive E2-or-older
  holding with the new rank-1 candidate" has 17
  comparable 2026 pairs. Keeping averaged
  +2.26%; replacements averaged
  -2.17% and beat keeping only
  23.5% of the time. This comparison
  is already optimistic for replacement because it omits the old position's
  early-liquidation friction. Reject that naive turnover rule.

## Integrity and decision

- Candidate label parity: 41 matched closed
  engine trades; maximum absolute economic-return error
  0.000e+00; pass=True.
- No threshold scan, model, strategy change, raw-data output, 202-factor change,
  or 2024/2025 performance validation was performed.
- Preserve CAP1 entry and Top5 ranking; do not add naive full-slot replacement.
  Static entry stacks and simple turnover are both unsupported. The only
  remaining evidence-backed change is the already registered forward-only
  early-path risk overlay, which does not recycle an early exit into a new buy.
