# F0-123 Executable V2 2026 Rolling Backtest Preregistration

Date: `2026-09-02`

## Question

Can the unchanged original F0=123 monthly ranker produce an acceptable portfolio
when its training target uses the same executable path as the canonical engine?

## Validation boundary

- Performance validation is only `2026-01-01` through the database cutoff at run time.
- 2024 and 2025 may only supply trailing training observations available before each
  2026 decision. Their returns cannot pass, rescue, or overturn the 2026 result.
- The signal endpoint is the latest exchange session jointly covered by F0,
  daily-basic, market, adjustment, and limit data. The execution endpoint is the
  latest session jointly covered by market, adjustment, and limit data.
- Every last session of every covered 2026 week is scored. There is no skipped-week
  fallback and no minimum-sample month skip.

## Frozen design

- Original registered 123 factors only, transformed independently within each date.
- Deterministic maximum of 1,500 training rows per date; signal cross-sections remain
  full market.
- Monthly trailing-12-month grouped `XGBRanker`, using only labels whose executable
  exit was available before that month's first decision close.
- T+1 entry, close-triggered `-8%` stop, next sellable open with retry, H10 time exit,
  Base costs inside the label, and no clipping.
- Weekly full-market score, Top20 candidate ledger, Top5 desired entries, H10,
  maximum five open positions, canonical Base and Stress execution.
- No gates, thresholds, feature search, auxiliary data, fallback rule, 202-factor
  table, or persisted business-data/model cache.

## Acceptance

Both Base and Stress must satisfy all of the following:

- trade win rate at least `70%`;
- portfolio PF at least `2.0`;
- absolute MaxDD no more than `15%`;
- positive return after removing the best week;
- positive return after replaying without the three largest profitable trades;
- at least 20 closed trades and no more than five simultaneous positions.

CAP1 (`+32.44%` Base, `+28.48%` Stress) remains the descriptive 2026 control. This
run is rejected if the acceptance tests fail, regardless of any 2024/2025 behavior.

## Command

```bash
python3 scripts/full_market_f0_123_ranker_runner.py \
  --model-profile configs/model_profiles/f0_123_executable_2026_rolling_v2.yaml \
  --output docs/council_20260828/F0_123_EXECUTABLE_V2_2026_ROLLING_20260902
```
