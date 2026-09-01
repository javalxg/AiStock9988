# Strong Momentum Breakout V1 Preregistration

Status: `PREREGISTERED_BEFORE_RESULT`.

## Research question

CAP1 is a weak-market reset strategy. This experiment asks whether one separate,
transparent strong-regime rule can provide auditable positive performance in
2026. It rebuilds only the ordinary momentum/breakout control hypothesis from
the old breadth-relative-turnover experiment. The breadth-relative ranking
itself already returned -11.41% and remains closed.

## Frozen rule

- Database stock master universe, PIT ST exclusion, at least 120 listed sessions,
  and no fixed CSV universe.
- Every 2026 exchange session through the latest T+1-executable signal date is
  evaluated. Execution and open-position marks end at the common required-source
  database cutoff.
- Required selection data are market daily, adjustment factor, and daily price
  limits. Missing required data exclude only that stock-session. No daily_basic,
  index, moneyflow, model, or 202-factor input is loaded.
- Stage1 requires positive but bounded 20/60-session momentum, price 0-15% above
  MA60, 20-session drawdown no worse than -10%, close at/above MA5 and the prior
  three-close high, one-day return at least -2%, and amount ratio 0.70-2.50.
- Rank is fixed at ret60 30%, dd20 25%, low vol20 20%, liq20 15%, and proximity
  to +5% above MA60 10%.
- T+1 raw-open entry; four entries/positions maximum; 12% decision NAV per name;
  48% gross cap; H10; -8% from-entry close trigger and next-tradable-open exit;
  unchanged Base/Stress costs and 2% ADV capacity.

## Audit and acceptance

The run writes only config, code/data hashes, plan, aggregate selection audit,
aggregate metrics, and conclusion. It must not write raw prices, factors,
candidates, fills, positions, models, CSV, or Parquet files.

Both Base and Stress must have PF at least 2, absolute MaxDD at most 15%, positive
return after excluding the best week and top three winners, at least 20 closed
trades, at least 20 active signal days, win rate at least 70%, and no more than
four positions. Weekly >=5% count/ratio is reported without redefining the user
goal. A passing standalone result advances to a separately preregistered paired
CAP1-combination test; it is not combined automatically.

On any failure, close this exact rule. Do not change momentum bounds, MA window,
rank weights, H10, stop, TopN, sizing, or add breadth/gates/models.
