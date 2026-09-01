# CAP1 Dragon-Tiger H10 Exclusion

## Contract

- Historical 2026 diagnostic; same CAP1 rules, engine, costs, sizing, H10 hold, and stop.
- One change: exclude a CAP1 candidate with an unchanged V1 confirmation in the current/prior nine sessions; preserve CAP1 rank and use normal fallback.
- No XGBoost, frozen 202-factor input, parameter/window scan, or persisted business data.

## Overlap

- Excluded candidate rows: `0` across `0` signal days and `0` symbols.
- Executed Base control entries affected: `0`; replacement entries: `0`.

## Portfolio

| Cost | Control return | Overlay return | Delta | Control PF | Overlay PF | Control MaxDD | Overlay MaxDD | Trades | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | +32.44% | +32.44% | +0.00% | 2.254 | 2.254 | -8.27% | -8.27% | 41 | False |
| stress | +28.48% | +28.48% | +0.00% | 2.058 | 2.058 | -8.82% | -8.82% | 41 | False |

## Decision

The H10 exclusion overlay is rejected unchanged and will not be repaired with a window or threshold scan.
