# Dragon-Tiger DeepSeek Review Blocker

Date: `2026-09-01`

Status: `EXTERNAL_REVIEW_UNAVAILABLE; IMPLEMENTATION_NOT_AUTHORIZED`.

## Frozen Input

The exact design awaiting review is
[`DRAGON_TIGER_PULLBACK_RECLAIM_V1_PREREG_20260901.md`](DRAGON_TIGER_PULLBACK_RECLAIM_V1_PREREG_20260901.md),
committed before any pullback-reclaim outcome was inspected. No strategy code or
portfolio backtest has been run for this mechanism.

## Review Attempts

Three existing DeepSeek review tasks were invoked with the frozen design and a
request for an `APPROVE`, `REVISE`, or `REJECT` decision:

1. `deepseek_f0_next_step_review_20260901`: terminated with upstream HTTP 502;
2. `deepseek_first_strategy_review_20260830`: terminated with upstream HTTP 502;
3. `deepseek_steady_climb_review`: terminated with upstream HTTP 502.
4. `deepseek_f0_next_step_review_20260901` was retried after the preregistration
   was pushed and again terminated with upstream HTTP 502 before returning text.

All failures occurred before any DeepSeek response text was returned. Earlier
DeepSeek tasks also reported HTTP 402 insufficient balance. A submitted request
is not a completed review, so no approval is claimed.

## Resume Contract

When DeepSeek service is available, send the frozen preregistration unchanged
and request a rule/PIT review before editing implementation files. Save the
response verbatim, incorporate or explicitly arbitrate every required change,
then request code review for the resulting implementation before executing a
backtest.

The mainline CAP1 forward lock continues independently. The dragon-tiger source
audit and direct-chase rejection remain valid; only the new pullback-reclaim
experiment is blocked at the review gate.
