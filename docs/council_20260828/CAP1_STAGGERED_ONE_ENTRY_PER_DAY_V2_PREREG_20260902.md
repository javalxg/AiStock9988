# CAP1 Staggered One Entry per Day V2 Preregistration

Status: `PREREGISTERED_BEFORE_RESULT`.

V1's generated result directory is invalidated because its runner passed the
control selection ledger to both portfolios. That meant the nominal one-entry
configuration still submitted up to five entries. It is not evidence for or
against this hypothesis.

V2 keeps the V1 hypothesis and every frozen strategy setting unchanged. The
only implementation correction is that the challenger receives its own
selection ledger, produced from the same feature ledger and asserted to have
an identical score and candidate ledger to control. Its `desired_entries` is
therefore exactly one while control remains five.

The full contract, no-scan rule, shared in-memory data bundle, 2026-only scope,
promotion criteria, and abandonment condition are exactly those in
`CAP1_STAGGERED_ONE_ENTRY_PER_DAY_V1_PREREG_20260902.md`.
