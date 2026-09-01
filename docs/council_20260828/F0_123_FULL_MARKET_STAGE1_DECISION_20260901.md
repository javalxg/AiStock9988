# F0-123 Full-Market Stage-1 Decision

Date: `2026-09-01`

## Verdict

The exact direct Top5 portfolio is rejected. Base returned `-33.98%`, PF was
`0.477`, and MaxDD was `-39.43%`; Stress returned `-36.99%`. The result is not
explained by skipped models or an incomplete run: all label-maturity, monthly
model, weekly signal, NAV, cash, execution-end, and position-cap checks passed.

Evidence:

- `F0_123_FULL_MARKET_WEEKLY_TOP5_20260901/RESULT.md:12-21`
- `F0_123_FULL_MARKET_WEEKLY_TOP5_20260901/verification.json:2-16`

## What Failed

The Stage-1 score is not monotonic at its extreme right tail. Top1 mean H10
label was `-1.77%` and cumulative Top1-2 was `-1.16%`, while ranks 3-5 averaged
`+1.41%` and ranks 11-20 averaged `+0.67%`. Across the disjoint Top1-20 buckets,
the weighted mean label remained `+0.40%` and the H10 `>=10%` event rate was
`15.95%` over 608 mature candidates.

This means the model still generates a candidate pool containing profitable
right-tail events, but its raw score cannot safely compress Top20 to Top5.
Buying ranks 3-5 is not authorized: that rule would be selected after observing
the 2026 outcome.

Evidence:

- `F0_123_FULL_MARKET_WEEKLY_TOP5_20260901/f0_diagnostics.json:2-39`

## What The Factors Say

Within the realized Top20, winners were descriptively smaller than losers:
the standardized winner-minus-loser differences were `-0.413` for `total_mv`
and `-0.405` for `circ_mv`. Winners also had lower `dmi_adxr`, `mtmma`, TRIX,
BRAR and several sector-relative trend values. These are diagnosis signals, not
validated buy filters; no threshold may be derived from this 2026 sample.

Evidence:

- `F0_123_FULL_MARKET_WEEKLY_TOP5_20260901/f0_diagnostics.json:40-111`

## Next Authorized Experiment

Keep the same F0=123 Stage-1 and generate its historical Top20 out of fold. A
second model may train only inside those causal historical Top20 rows, using the
same 123 values and the fixed binary target `path-stop H10 return >=10%`. It
then ranks the frozen 2026 Top20 and selects at most five positions.

This is the previously identified Top20-to-Top5 conditional-selection problem,
not a new data-source experiment. Auxiliary money flow, chips, attention,
dragon-tiger, Chan, or 202-factor inputs remain unauthorized.

