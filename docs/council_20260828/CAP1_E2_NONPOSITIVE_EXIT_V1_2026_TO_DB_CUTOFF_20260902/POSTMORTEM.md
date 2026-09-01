# CAP1 E2 Nonpositive Exit V1 Postmortem

Status: `REJECTED_AND_EXECUTABLE_PATH_REMOVED`.

The exact preregistered implementation is preserved by Git commit `1a2aa24`.
Its strategy configuration and canonical-engine branch were removed after the
result so that a rejected rule cannot remain selectable on the main branch.

## What the experiment established

- It evaluated all 160 exchange signal sessions from `2026-01-05` through
  `2026-08-31`; execution and marking ended at the common database cutoff
  `2026-09-01`. There was no date-level data gap or sample-size skip.
- The rule made 32 Base and 32 Stress early exits. Because those exits released
  cash and slots, the challenger closed 59 trades versus 41 for control.
- Base return fell from `+33.65%` to `+19.18%`, PF from `2.254` to `1.570`, and
  win rate from `56.1%` to `39.0%`. Stress return fell from `+29.61%` to
  `+13.88%`, with PF falling from `2.058` to `1.381`.
- Base MaxDD improved from `-8.27%` to `-6.95%`, but every preregistered return,
  PF, exclusion-return, and win-rate condition failed. Lower drawdown alone is
  not enough to promote a strategy that destroys substantially more return.

## Interpretation and boundary

The earlier descriptive split between E2-positive and E2-nonpositive control
trades did not survive as a portfolio action. Acting on it both cut recoveries
and admitted more low-quality replacement trades. This is selection-on-seen-
outcome evidence: a feature that separates eventual winners descriptively is
not automatically a profitable executable rule.

Do not scan the E-index, zero-return boundary, early-exit threshold, hold,
stop, or replacement policy. Do not combine this rule with CAP1 and do not use
2024/2025 to rescue it. CAP1 remains unchanged.
