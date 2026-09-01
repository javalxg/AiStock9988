# F0-123 Executable Label V2 Audit

## Scope

- Database-only 2025 weekly full-market signal keys; H10 Base executable label contract.
- Aggregate output only; no labels, prices, fills, predictions, or models persisted.
- This is a label audit, not a strategy-return backtest or parameter search.

## Coverage

- Requested rows: `246745`; executable labels: `244671`.
- Entry rejected: `1855`; unresolved exits: `219`; retried exits: `1725`.

## Returns

- STOP_LOSS rows: `35565` (14.54%); mean executable return `-11.02%`.
- Mean stop crossing return: `-10.62%`; executable loss worse than fixed -8% in `91.88%` of stop rows.
- TIME_EXIT rows: `209106`; mean executable return `+3.31%`; positive rate `63.35%`.

## Engine Parity

- Sparse audit trades: `12`; stop `6`, time exit `6`.
- Entry date, exit date, trigger type and economic return parity passed: `True`.
- Maximum absolute economic-return error: `0.000e+00`.

## Decision

The executable label contract is accepted for a separately preregistered forward-only F0 V2 model. No historical 2026 improvement claim is authorized.
