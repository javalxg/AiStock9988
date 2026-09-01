# Reset Weak Confirm V3 Cap1-20 Final Decision

Status: `PASS_HISTORICAL_DIAGNOSTIC_FORWARD_LOCK_REQUIRED`.

CAP1-20 keeps the verified V3 selection, composite ranking, T+1 entry, H10
holding period, trailing -8% stop, costs, PIT policy, and ADV limit unchanged.
Its only portfolio change is 20% decision-NAV sizing per name with a 100% gross
cap, versus the same-bundle control's 10% sizing and 50% gross cap.

## Result

| Scenario | Strategy | Return | PF | MaxDD | Ex-best week | Ex-top3 | Trades |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | CAP1-20 | +32.44% | 2.254 | -8.27% | +20.12% | +16.23% | 41 |
| Base | V3 control | +14.92% | 2.226 | -4.30% | +9.28% | +7.29% | 41 |
| Stress | CAP1-20 | +28.48% | 2.058 | -8.82% | +16.57% | +12.86% | 41 |
| Stress | V3 control | +12.86% | 2.003 | -4.60% | +7.44% | +5.54% | 41 |

The challenger passes every registered acceptance condition. The control
reproduces the prior baseline exactly from the same in-memory data bundle.
Trade count and win rate are unchanged, so the improvement comes from actual
capital allocation rather than a changed stock selection or data snapshot.

## Capacity Audit

- Average actual gross exposure was 49.35% base and 49.33% stress; maximum was
  99.69% and 99.83%.
- Maximum open positions was five. Minimum cash remained positive at RMB
  3,409.91 base and RMB 1,934.71 stress.
- Maximum ADV20 participation was 0.0433% base and 0.0425% stress versus the
  fixed 2% limit. No fill had missing ADV/gross data and no limit was breached.
- PIT, T+1, NAV identity, cash, position-cap, fill/order, execution-end, code,
  and artifact verification all passed.

## Decision

CAP1-20 is the current best auditable 2026 historical configuration. It is not
a production or OOS claim: the period is seen history and contains only 41
closed trades. Freeze this exact contract for append-only forward comparison
against the original V3 control. Do not try another position size, gross cap,
ADV limit, stop, holding period, or ranking adjustment.

The original F0=123 system remains available only for separately pre-registered
hypothesis work. The V3 final pool is too sparse for a credible 123-factor
ranker, and the prior 12-feature pool ranker is not evidence that all F0=123
factors fail. The wide-table 202-factor system, its code, and its design remain
frozen and out of scope.

Re-evaluate the forward lock only after at least 26 signal weeks or 60 closed
trades. Until then, report forward fills, gross exposure, ADV participation,
cash, PF, MaxDD, return excluding the best week, and return excluding the top
three winners without refitting or recomputing history.
