# CAP1 Staggered One Entry per Day V1 Preregistration

Status: `PREREGISTERED_BEFORE_RESULT`.

## Hypothesis

CAP1's candidate selection may be sound while simultaneous entries concentrate
capital in one market state. Entering only the highest-ranked new candidate on
each decision day could reduce clustered path dependence without changing the
signal, ranking, risk budget per name, exit logic, or maximum five holdings.

## Frozen Contract

- Control: `reset_weak_confirm_v3_cap1_20`.
- Challenger: `reset_weak_confirm_v3_cap1_staggered_one_entry_v1`.
- Both run from `2026-01-01` to the common required-source database cutoff.
  The final session is mark-only when no T+1 open exists.
- Both use the same in-memory QuantDB bundle, features, Stage1 candidates,
  T+1 raw-open fills, H10, prior-close trailing -8% stop, costs, capacity,
  corporate actions, and per-name 20% decision-NAV sizing.
- The only changed field is `portfolio.entries_per_decision: 5 -> 1`.
- No queued/stale signal, forced replacement, XGBoost, new factor, gate,
  threshold scan, TopN scan, holding-period scan, or 2024/2025 portfolio
  performance result is allowed.

## Promotion or Abandonment

For both Base and Stress, the challenger must strictly improve total return,
retain PF >= 2, MaxDD <= 15%, positive return excluding the best week, and no
worse PF, MaxDD, or ex-best-week result than the paired control. It must keep
at most five positions. The 5%-weekly count and closed-trade win rate are
reported but are not used to redefine a pass.

Failure rejects this exact one-entry rule. No 2/3/4-entry variants are run.
