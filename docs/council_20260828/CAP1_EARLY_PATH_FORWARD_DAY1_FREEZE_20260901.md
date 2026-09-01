# CAP1 Early-Path Forward Day-1 Freeze

Status: `FROZEN_ABSTENTION`.

## Contract

- Signal session: `2026-09-01`.
- Forward start: `2026-09-01`.
- Decision uses only data with `trade_date <= 2026-09-01`.
- Control: `reset_weak_confirm_v3_cap1_20_forward`.
- Shadow: `cap1_early_path_forward`.
- Maximum positions remains five; no threshold, ranking, holding period, stop,
  sizing, or candidate-count rule was changed.

The R2 registration was verified before any freeze work and again immediately
before the immutable batch was committed. Its code-manifest hash is
`c6c8ca04b8ac2947100eaf3bb5cc6e302c16bc700723c160b2ccb49d8d5f8599`.

## Data Completion

The existing delta CLI was used only as an operational writer to QuantDB. No
delta source file was modified or copied into AiStock9988.

```bash
python3 cli.py sync --type daily --start 20260901 --end 20260901
python3 cli.py sync --type adj_factor --start 20260901 --end 20260901
python3 cli.py sync-all --task-name sync_stk_limit --start 20260901 --end 20260901
```

Post-sync read-only verification for `2026-09-01`:

| Source | Rows | Unique codes | Duplicate `(trade_date, ts_code)` keys |
|---|---:|---:|---:|
| `market_daily_ts` | 5,548 | 5,548 | 0 |
| `adj_factor_ts` | 5,567 | 5,567 | 0 |
| `stk_limit_ts` | 7,569 | 7,569 | 0 |

The three required sources have a common cutoff of `2026-09-01`. The formal R5
preflight status is `READY_TO_FREEZE`.

## Frozen Decision

The score and candidate ledgers contain 5,522 stock rows. Rejection accounting
is complete:

| Reason | Stocks |
|---|---:|
| `STAGE1_REJECTED` | 4,409 |
| `FEATURE_NOT_MATURE` | 757 |
| `UNIVERSE_REJECTED` | 336 |
| Missing `market_daily_ts` and `stk_limit_ts` | 13 |
| Missing `market_daily_ts` | 7 |

There were zero `IN_VIEW` candidates. The lockbox therefore sealed an explicit
abstention rather than falling back to a random stock or relaxing the rule.

The abstention has an economic explanation, not a coverage failure. Among the
4,409 fully eligible and feature-mature stocks:

- 544 were at least 10% below MA60;
- the PIT market 20-session return was `+6.0043%` for the full eligible cross
  section;
- zero passed the frozen `mkt_ret_20d < 0` weak-market state;
- consequently zero passed all Stage-1 conditions.

## Immutable Identity

- Batch SHA256:
  `41cd7c28489c2112293420e1ba0f3282fe800f702f334bbf30f19e879f63e2fc`.
- Lockbox manifest SHA256:
  `937d51b0a82b5a859eda95736083fc941849588a2718f7491abbcde1940e76c5`.
- Bundle ID:
  `4c44a26b439ddd72e8c2257078367c690fdc2decfc3932cc7d2afee52654f8fd`.
- Candidate count: `0`.
- Abstention: `true`.

Local authoritative artifacts:

- `CAP1_EARLY_PATH_FORWARD_PREFLIGHT_LATEST_R5.json`;
- `CAP1_20_FORWARD_LOCKBOX/batches/2026-09-01/`;
- `CAP1_20_FORWARD_LOCKBOX/manifests/manifest-41cd7c28489c2112293420e1ba0f3282fe800f702f334bbf30f19e879f63e2fc.json`.

## Decision

This is the first formal out-of-sample observation and contains no trade or
return. Weekly 5% attainment, 70% trade win rate, PF, MaxDD, and tail-robust
return are therefore `NOT_EVALUABLE`, not failed and not passed.

Do not alter the weak-market condition after seeing this abstention. Continue
the append-only freeze on each newly completed database session. Only a frozen
control buy can enter the paired early-path observation and eventual settlement
process.
