# V1 Result Invalidated

The V1 runner built challenger configuration correctly but mistakenly supplied
the control `selection_ledger` to the challenger backtest. Consequently the
challenger retained `desired_entries=5` and did not test the registered
one-entry behavior. The matching metrics are an implementation symptom, not a
strategy result.

V2 corrects only this ledger wiring, asserts shared score and candidate
ledgers, and is preregistered before rerun in
`CAP1_STAGGERED_ONE_ENTRY_PER_DAY_V2_PREREG_20260902.md`.
