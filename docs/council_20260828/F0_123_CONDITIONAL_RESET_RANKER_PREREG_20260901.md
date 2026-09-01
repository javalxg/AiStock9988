# F0-123 Conditional Reset Ranker Preregistration

Date: `2026-09-01`

Status: `PREREGISTERED_BEFORE_PORTFOLIO_RESULT`.

## Question

Can the frozen original F0=123 factor set rank stocks inside a fixed,
economically coherent reset-state pool better than the existing transparent
composite rank?

This is not another full-market q70 replay and not a model applied after the
very sparse final CAP1 confirmation. The model trains and predicts only among
stocks that first pass the fixed Stage-1 state rule below.

The experiment is a historical out-of-training-sample diagnostic: training
uses 2025 and prior mature rows; portfolio prediction starts in 2026. It is not
an append-only forward claim.

## Frozen Stage-1 Pool

A stock-session enters Stage-1 only when all of the following are known by the
signal close:

1. the canonical universe, PIT ST/BJ, listing-age, and required-data policies
   pass;
2. `dist_ma60 <= -0.10`;
3. equal-weight PIT `mkt_ret_20d < 0`;
4. stock `ret20 < 0`;
5. `liq20 >= 500000` in the existing amount unit.

These are inherited CAP1 reset-state and tradability conditions. The final
CAP1 confirmation conditions (MA5 reclaim, prior-three-close reclaim, positive
daily return, volume band, drawdown and cross-sectional volatility ceiling)
are deliberately not hard gates here. The experiment asks whether F0=123 can
rank those later transition qualities without shrinking the training pool to
the 499 mature final-CAP1 events already judged too sparse.

No Stage-1 threshold or candidate-count gate may be scanned.

## Frozen F0=123 Input

- Manifest: `configs/feature_sets/f0_123_columns.json`.
- Feature-set id: `feature.f0_123.v1`.
- Column-order hash:
  `7f88c8912cd268a1bedd5034768ab73827bdce8f8d2f3870fe8fac3c26aa9e39`.
- Exactly 57 technical, 9 fundamental, and 57 PIT industry-relative columns.
- No wide-table 202 factor, money-flow, chip, margin, attention, dragon-tiger,
  minute-derived feature, or new data source.
- A stock-session missing any required F0=123 input is excluded only for that
  session from model training and prediction. It is never imputed from future
  rows and never replaced by a relaxed feature set.

## Label And Timeline

- Label: signal close `T` -> entry `T+1` economic open -> exit `T+11`
  economic open, a ten-session entry-to-exit horizon.
- Label availability is the `T+11` session open. A row may train a model only
  when `available_time <=` that model's training cutoff.
- Training window: trailing 12 calendar months, starting no earlier than
  `2025-01-02`.
- Retraining: once per calendar month, using the final completed session of the
  preceding month as cutoff.
- Prediction: every configured 2026 signal session, not only Fridays.
- Prediction range: `2026-01-05` through `2026-08-17`.
- Execution range: `2026-01-05` through `2026-09-01`; the final signal has a
  complete T+1/H10 horizon.
- A month with an invalid ranking dataset is a hard experiment failure. It may
  not silently skip, use a later model, train on the full market, or fall back
  to rules.

## Frozen Model

- Family/objective: deterministic `XGBRanker`, `rank:pairwise`, grouped by
  signal session.
- `n_estimators=200`, `max_depth=6`, `learning_rate=0.05`,
  `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`,
  `reg_alpha=1.0`, `reg_lambda=1.0`, `random_state=42`, `n_jobs=1`,
  `tree_method=hist`.
- All 123 columns enter unchanged. There is no IC/SHAP feature selection,
  reweighting, monotonic constraint, hyperparameter search, early stopping, or
  probability/score gate.
- Each monthly model is trained twice independently; serialized model hashes
  must match before its predictions are accepted.

## Control And Challenger

Both arms use the identical Stage-1 rows and F0-complete prediction rows.

- Control: the existing CAP1 six-term transparent composite rank; freeze its
  Top20 candidate view.
- Challenger: XGBRanker score descending; freeze Top20.
- Both request the first five ranked candidates with existing ranked fallback,
  maximum five open positions, 20% decision-NAV sizing, 100% gross cap, T+1
  raw-open entry, H10, close-triggered `-8%` trailing-from-prior-close stop,
  canonical corporate-action accounting, ADV cap, and Base/Stress costs.
- The canonical engine remains `src/aistock9988/backtest/engine.py`.

## Acceptance And Abandonment

The challenger advances only if:

1. every monthly model exists, uses only mature Stage-1 labels, and passes the
   independent model-byte determinism check;
2. PIT, next-session entry, NAV identity, nonnegative cash, source date,
   complete horizon, and maximum-five-position checks pass;
3. Base and Stress each have PF at least `2.0`, MaxDD no worse than `15%`,
   return excluding the best week positive, return excluding the top three
   winning trades positive, and at least 20 closed trades;
4. Base and Stress total return both exceed the same-pool transparent control;
5. Base and Stress total return both exceed the sealed CAP1-20 benchmarks
   (`+32.44%` Base and `+28.48%` Stress) without failing any risk criterion.

Any failure rejects this exact conditional F0=123 ranker. Results may not be
repaired by changing Stage-1, TopN, features, model parameters, retrain cadence,
holding period, stop, sizing, or cost assumptions.

## Storage

The run may persist configuration, model hashes/metadata, aggregate sample and
coverage counts, portfolio metrics, verification, and compact Markdown/JSON.
It must not persist market, factor, label, prediction, candidate, fill,
position, CSV, Parquet, or database business-data caches in the repository.
