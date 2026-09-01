# reset_weak_confirm_v3_cap1_20 V3 Full-Universe 2026

FORWARD_ONLY_DIAGNOSTIC: this historical replay is explicitly seen history and is not a locked out-of-sample claim.

## Contract

- Signal range: `2026-01-05` to `2026-08-13`; execution through `2026-08-28`.
- Bundle: `748b1daa94e01d1d1f05fdda0b02455ab8fd6a43009d794ae94af2f4177fd2ce`; configured full-universe codes: `5838`.
- Rules only: full universe -> Stage1 expression `{"all": [{"left": "dist_ma60", "op": "le", "value": -0.1}, {"left": "mkt_ret_20d", "op": "lt", "value": 0.0}, {"left": "ret20", "op": "lt", "value": 0.0}, {"left": "economic_close", "op": "ge", "right": "ma5"}, {"left": "economic_close", "op": "ge", "right": "prev3_high"}, {"left": "ret1", "op": "ge", "value": 0.0}, {"left": "liq20", "op": "ge", "value": 500000.0}, {"left": "volume_ratio_20", "op": "gt", "value": 1.0}, {"left": "volume_ratio_20", "op": "le", "value": 2.0}, {"left": "dd20", "op": "ge", "value": -0.15}, {"left": "vol20", "op": "le", "right": "vol20_p85"}]}` -> frozen daily Top20 -> Top5/next-ranked fallback.
- T+1 raw open entry, 20.00% of prior-close NAV per name, maximum 5 positions, H10, -8% close-trigger/next-open stop.
- No threshold sweep and no XGBoost model.

## Selection

- Signal dates: `148`; feature-ready rows: `615929`.
- Required-data exclusions on signal dates: selection `50661`, training `50509`.
- Stage1 pass rows: `365`; frozen candidate-view rows: `342`.

## Portfolio

| Cost | Return | PF | MaxDD | Ex-best-week | Trades | End open | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| base | +32.44% | 2.254 | -8.27% | +20.12% | 41 | 0 | True |
| stress | +28.48% | 2.058 | -8.82% | +16.57% | 41 | 0 | True |

## Decision

The strategy advances only when PF>=2, MaxDD<=15%, and return excluding the best week remains positive. A negative result is retained unchanged as evidence; it is not repaired with a threshold scan.
