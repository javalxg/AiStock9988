# Dragon-Tiger Event Path Decision

Date: `2026-09-01`

Status: `DIRECT_CHASE_REFUTED; ONE_PULLBACK_RECLAIM_DIAGNOSTIC_PROPOSED`.

## What Was Tested

This was an aggregate event-path diagnostic, not a portfolio backtest and not a
threshold scan. A signal is a `top_list_ts` stock-day known after the session
close. True institution rows are only `top_inst_ts.exalter='机构专用'`; their
`net_buy` values are aggregated by stock-day. Top-list reasons are preserved as
a set, and `top_list_ts.net_amount` is never summed across reasons.

Entry is the next exchange session's economic open. Every entry and endpoint
must pass the canonical V3 universe and `execution_data_eligible`, and every
price must be finite and strictly positive. H1/H2/H3/H5/H10 mean entry open to
the open one/two/three/five/ten exchange sessions later. The compact evidence is
in [`DRAGON_TIGER_PATH_DIAGNOSTIC_20260901.json`](DRAGON_TIGER_PATH_DIAGNOSTIC_20260901.json).

## Finding

For stock-days with an upward top-list reason and positive true-institution net
buy, the event-close to next-open gap was positive in both years:

- 2025: `+1.32%` mean, 56.42% win rate, PF `2.308`, `n=2,164`;
- 2026: `+1.00%` mean, 52.93% win rate, PF `1.916`, `n=1,861`.

That is real event information, but it is mostly priced before the executable
next-session open. In 2026 the post-entry returns were H1 `-0.14%`, H2 `-0.19%`,
H3 `-0.35%`, H5 `-1.12%`, and H10 `-1.53%`; every PF was below 1. The H10
ex-top-three mean was even worse at `-1.69%`. Direct T+1-open chasing is therefore
rejected rather than promoted as a new strategy.

## Permitted Use

The source is retained as an event-state candidate pool, not a buy command. One
independently specified follow-up is eligible for review: wait for the initial
event gap to be released by a pullback, then require a later price reclaim before
entering at the following open. This tests whether institutional attention has a
second executable leg instead of paying for the already-realized overnight leg.

The proposal must freeze the pullback and reclaim definitions before looking at
their returns. It may not scan thresholds, may not use the frozen 202-factor
system, and must stop at the `top_list_ts`/`top_inst_ts` cutoff of 2026-08-18.
CAP1-20 remains the production mainline and is unchanged.

No pullback-reclaim portfolio backtest is authorized yet. DeepSeek review is
currently unavailable with HTTP 402 (insufficient balance), and GLM review is
currently unavailable with HTTP 403 (expired key); neither review is claimed.
