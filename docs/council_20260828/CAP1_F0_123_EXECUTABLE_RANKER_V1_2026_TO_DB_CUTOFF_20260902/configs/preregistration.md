# CAP1 F0-123 Executable Ranker V1 Preregistration

Status: `PREREGISTERED_BEFORE_MODEL_OR_RESULT`.

## Question

Can the original 123-factor system improve selection only inside the current
profitable CAP1 fixed-rule pool, when trained against the canonical executable
H10/stop return rather than a clamped or fixed-horizon proxy?

This exact combination has not been tested. Earlier F0 experiments either
ranked the full market or used a materially wider reset pool that omitted seven
CAP1 confirmation conditions; they do not answer this question.

## Frozen Stage 1 and portfolio

- Use `reset_weak_confirm_v3_cap1_20.yaml` unchanged: database universe, PIT ST,
  required-data exclusions, all eleven Stage1 conditions, transparent Top20,
  T+1 entry, 20% decision-NAV sizing, at most five positions, H10, trailing
  close stop, next-tradable-open exits, capacity, corporate actions, and
  Base/Stress costs.
- Bind the canonical configuration hash
  `6b723d32a767b57d45651c8319a6ef285872f645372da41766317a696be2e74a`;
  a strategy-ID match without this exact hash must stop before database access.
- Score every exchange session from `2026-01-05` through `2026-08-28`, the
  common current F0/daily-basic cutoff, and execute/mark through the required
  market-source cutoff `2026-09-01`. Empty CAP1 days remain explicit empty
  decisions; no date or month may be skipped and there is no transparent-rank
  fallback.
- A missing required Stage1 or F0 stock-session excludes only that stock. It
  cannot suppress the date or another stock.

## Frozen Stage 2

- Join exactly the registered 123 F0 columns in their frozen order. Normalize
  infinity to NaN and use XGBoost-native missing branches; no imputation,
  feature selection, IC filter, SHAP filter, 202-factor table, or auxiliary
  source is allowed.
- Train one deterministic `XGBRanker` per 2026 calendar month on the trailing
  12 months of CAP1 Stage1 rows. Only labels whose actual executable exit was
  available before the prior month-end cutoff may enter training. Train twice
  and require byte-identical model artifacts.
- The label uses T+1 economic open with Base costs, the real close-known `-8%`
  stop crossing, next sellable economic open with retry, H10 time exit, and no
  return clipping.
- Rank only that day's CAP1 Stage1 rows by model score. Keep at most 20 in the
  candidate ledger and request at most five entries under the unchanged engine.

## Year boundary

2025 is training input only. Portfolio return, PF, drawdown, win rate, weekly
attainment, and every pass/fail decision use 2026 only. No 2024/2025 return may
pass, rescue, or overturn the experiment.

## Acceptance and stop

Both Base and Stress must have win rate at least 70%, PF at least 2, absolute
MaxDD at most 15%, positive return excluding the best week, positive return
excluding the three largest winners, at least 20 closed trades, and no more
than five positions. They must also beat both the same F0-eligible transparent
CAP1 control and the current full CAP1 return (`+33.65%` Base, `+29.61%`
Stress). Weekly 5% attainment is reported.

No model parameter, feature, TopN, month, threshold, holding period, stop,
position size, or gate scan is permitted. On failure, remove this model profile
from the main branch and retain only aggregate evidence plus the reproduction
commit.
