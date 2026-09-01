# Strong-Regime Source Feasibility Decision

Date: `2026-09-01`

Status: `DRAGON_TIGER_EVENT_DIAGNOSTIC_AUTHORIZED_ONLY`.

## Purpose

CAP1-20 is a weak-market deep-reset strategy. Its first forward session on
2026-09-01 correctly abstained because the PIT market 20-session return was
`+6.0043%`. This audit asks whether QuantDB currently contains one independent,
PIT-safe source capable of supporting a separate strong-regime mechanism. It
does not change CAP1, use the frozen 202-factor system, train a model, define a
new strategy threshold, or run a portfolio backtest.

## Source Results

### Closing Auction

`stk_auction_c_ts` has good key and value quality where present: 1,973,593 rows,
351 dates, 8,253 codes, no duplicate `(trade_date, ts_code)` keys, and no null
critical OHLCV rows from 2025-01-02 through 2026-06-17. It has no rows after
2026-06-17 and therefore cannot cover the complete 2026 period through the
dense daily cutoff 2026-09-01.

The existing delta CLI was invoked only as an operational QuantDB writer for
2026-06-18 through 2026-09-01. Tushare rejected `stk_auction_c` for missing API
permission; task `ba5c57b93db94ce7` wrote zero rows. Repeating the same command
cannot repair the source. The exact metadata and task result are recorded in
[`STRONG_REGIME_SOURCE_FEASIBILITY_AUDIT_20260901.json`](STRONG_REGIME_SOURCE_FEASIBILITY_AUDIT_20260901.json#L7).

A final five-minute OHLC bar is not an order-imbalance observation and must not
be substituted for the missing closing-auction source.

### Investor Attention

The THS attention snapshot covers only 92 exchange dates in 2026 and ends on
2026-08-18. The DC snapshot is still sparser at 48 dates and ends on
2026-07-17. A later code audit found that delta's historical sync assigns both
`snapshot_time` and `update_time` from `datetime.now()` during ingestion
(`/Users/lxg/quant/deltafstation/backend/datahub/service.py:2816-2819`). The
stored `snapshot_time` therefore does not prove when the rank content was first
observable and must not, by itself, be called future leakage. Attention remains
blocked because its historical coverage and observation-time semantics are
insufficient, not because late ingestion is automatically leakage. Metadata is
recorded in the same audit JSON
([`ths_hot_snapshot_ts`](STRONG_REGIME_SOURCE_FEASIBILITY_AUDIT_20260901.json#L28),
[`dc_hot_snapshot_ts`](STRONG_REGIME_SOURCE_FEASIBILITY_AUDIT_20260901.json#L42)).

Public evidence supports short-horizon attention-driven momentum in China, but
the evidence depends on investor-level or intraday information, not a sparse
late historical rank dump. See the NBER paper
[Daily Momentum and New Investors in an Emerging Stock Market](https://www.nber.org/papers/w31839)
and the intraday study
[Do order imbalances predict Chinese stock returns?](https://doi.org/10.1016/j.pacfin.2015.07.003).

### Already-Tested Sources

- One existing institutional-event implementation returned `-6.47%`, 50.0%
  win rate, `-20.22%` MaxDD, and `-0.20%` average weekly return
  (`/Users/lxg/quant/deltafstation/doc/research/CURRENT_V6_HANDOFF.md:2409`).
- Raw daily money flow reduced the 2025 q70 result and was rejected; THS flow
  reversed direction across years
  (`/Users/lxg/quant/deltafstation/research/experiments/NEXT_DIRECTION_20260819.md:7-18`).
- Existing chip sources either lack full-market history or produced no
  executable incremental edge
  (`/Users/lxg/quant/deltafstation/research/experiments/q70_unused_source_inventory_20260822/CONCLUSION.md:5-15`).

These results prohibit renaming the same daily flow, chip, or institutional
implementation into another strategy. They do not refute every independently
defined dragon-tiger event mechanism. `top_list_ts` and `top_inst_ts` contain a
separate, auditable event source through 2026-08-18; its direct T+1 chase and
post-entry path must be tested before any portfolio use.

## Decision

No source currently authorizes a new strong-regime portfolio backtest. The
dragon-tiger source does authorize a mechanism diagnostic through its own
2026-08-18 cutoff. An event table is not required to contain one row for every
stock on every day: absence means the stock did not have that event. The source
must nevertheless have a documented daily extraction cutoff, and no signal may
be manufactured after the latest covered event date.

Keep CAP1-20 and its early-path shadow unchanged in the append-only forward
lockbox. A new strong-regime proposal becomes eligible only when one source has:

1. a timestamp proving visibility before the proposed T+1 entry;
2. coverage for every 2026 signal session through the dense daily cutoff, with
   missing stock rows excluding only those stocks;
3. a variable that is not daily money flow, chip position, generic momentum,
   or a renamed copy of the rejected institutional-event implementation;
4. a frozen sample/base-rate audit followed by independent rule and code
   review before any portfolio backtest.

The dragon-tiger source has now entered step 4. Its direct-chase path result and
the only eligible follow-up mechanism are recorded in
[`DRAGON_TIGER_PATH_DECISION_20260901.md`](DRAGON_TIGER_PATH_DECISION_20260901.md).

The weekly `+5%`, 70% win-rate, high-PF, and maximum-five-position objectives
remain unmet. This decision avoids manufacturing another low-return result; it
does not redefine or relax those objectives.
