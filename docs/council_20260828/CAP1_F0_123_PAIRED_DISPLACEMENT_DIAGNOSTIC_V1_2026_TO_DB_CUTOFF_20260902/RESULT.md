# CAP1 F0-123 Paired Displacement Diagnostic

Status: `DIAGNOSTIC_COMPLETED_MODEL_REMAINS_REJECTED`.

## Reproduction

- Source rejected run reproduced: `True`; maximum metric error `0.000e+00`.
- Eight monthly model hashes matched: `True`.
- Performance attribution uses 2026 only; 2025 is training input, not a portfolio validation period.

## Actual Entries

- Shared entries: `34`; displaced transparent entries: `8`; promoted model entries: `8`.
- Displaced closed mean return: `+1.91%`; promoted: `-0.64%`.
- Direct same-state closed pairs: `1`; isolated mean return delta `+13.38%`; model-side pair win rate `100.0%`.
- Downstream-divergence closed pairs: `7`. They are descriptive, not labeled direct replacements.

## NAV Bridge

- Challenger-minus-control final NAV: `-49,063.41`; bridge: `-49,063.41`; error `2.474e-10`.
- Shared sizing/path effect: `+5,679.17`; direct displacement: `+27,153.05`; downstream divergence: `-81,895.64`.

## Rank Quality

- Transparent Top5 isolated mean/win: `+2.21%` / `63.8%`.
- F0 Top5 isolated mean/win: `+2.04%` / `61.3%`.

## Decision

The F0-123 ranker remains rejected. These aggregate diagnostics describe why; they do not authorize a model repair, factor filter, or threshold scan.
