# F0-123 Causal Top20 Event Selector Preregistration

Date: `2026-09-01`

Status: `PREREGISTERED_BEFORE_STAGE2_MODEL_OR_RETURN`.

## Hypothesis

The frozen full-market F0=123 ranker has candidate coverage but poor extreme
score monotonicity. A conditional model trained only on causal Stage-1 Top20
candidates can improve Top20-to-Top5 compression by targeting the H10 right
tail rather than another full-market continuous-return ranking objective.

## Frozen Stage-1

- Use the exact data, universe, daily cross-sectional transform, path-stop H10
  label, monthly grouped XGBRanker and weekly Top20 contract from
  `F0_123_FULL_MARKET_WEEKLY_TOP5_PREREG_20260901.md`.
- Generate 2025 Stage-1 Top20 out of fold using only models whose labels were
  mature at each historical signal. No in-sample Stage-1 scores are allowed.
- Freeze 2026 Stage-1 Top20 before fitting or applying Stage-2. Stage-2 cannot
  introduce a stock outside that Top20.

## Frozen Stage-2

- Training population: only causal historical Stage-1 Top20 candidates.
- Input: the same 123 daily standardized F0 values plus Stage-1 score and rank
  as derived provenance fields; no new business-data table.
- Target: `1` when the same path-stop H10 label is at least `+10%`, otherwise
  `0`. The event threshold is frozen from the preregistered right-tail question
  and may not be scanned.
- Model: deterministic `XGBClassifier(binary:logistic)`, depth 4, 200 trees,
  learning rate 0.05, min child weight 5, subsample/column sample 0.8,
  alpha/lambda 1.0, seed 42 and `n_jobs=1`.
- Training: monthly trailing 12 months; labels must be mature by the model's
  weekly signal close. No class-threshold gate is used.
- Selection: sort the frozen Top20 by Stage-2 probability and take Top5, with
  Stage-1 rank then code as deterministic tie breakers.

## Paired Backtest

- Control: the already frozen direct Stage-1 Top5 ledger.
- Challenger: the conditional Stage-2 Top5 ledger.
- Both use identical Top20 membership, weekly dates, T+1 execution, H10,
  from-entry -8% stop, five-position cap, sizing, capacity and Base/Stress costs.
- Persist aggregate audits only; no business-data or model cache.

## Acceptance And Stop

Both cost scenarios must beat the direct Stage-1 control and CAP1, have PF >=2,
absolute MaxDD <=15%, positive ex-best-week and ex-top3 returns, at least 20
closed trades, and no position-cap breach. Stage-2 must also have strictly
positive lift in Top5 H10 `>=10%` event recall relative to the direct Top5.

Failure closes this exact conditional classifier. It does not authorize rank
buckets, feature thresholds, probability gates, hyperparameter searches, or
auxiliary data. The remaining permitted F0 work would be a fresh-time validation
of a single mechanism derived from the already recorded winner/loser families.
