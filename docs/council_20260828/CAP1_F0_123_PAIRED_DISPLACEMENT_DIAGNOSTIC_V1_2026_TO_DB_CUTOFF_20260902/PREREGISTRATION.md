# CAP1 F0-123 Paired Displacement Diagnostic V1

Status: `PREREGISTERED_DIAGNOSTIC_NOT_RUN`.

## Question

The verified CAP1 F0-123 challenger retained the same number of entry attempts,
fills, closed trades, and winners as the transparent CAP1 control, but reduced
Base return from `+33.65%` to `+28.75%`. This diagnostic asks one question only:
which aggregate return changes were caused when the model promoted securities
that the transparent ranking displaced?

This is attribution of a permanently rejected model. It is not a new strategy,
model repair, parameter search, or promotion attempt.

## Frozen Source

- Source run: `CAP1_F0_123_EXECUTABLE_RANKER_V1_2026_TO_DB_CUTOFF_20260902`.
- Source artifact-manifest SHA256:
  `fc076d6565ca32ee1a465804edde6f42b257aa220a0b2b3b2e5c044703305735`.
- Source strategy SHA256:
  `6d0af91e975a4a4d1e28ce4f508f06c94a3bb6b9e5b6f72db2f366ed5e81404d`;
  source model-profile SHA256:
  `1f44fa586254fa15e3206a3c7a9760a4b427713c57b1c7bcb9acb9a6f28cf57f`.
- Source strategy, model profile, and preregistration are read from the immutable
  source-run copies.
- The source artifact manifest, rejected status, verification result, model
  bytes, portfolio metrics, and source cutoffs must reproduce before attribution
  is reported.
- Training and monthly model construction remain unchanged. The only performance
  period is the source run's 2026 signal period through its database cutoff.
  No 2024 or 2025 portfolio performance is computed.

## Fixed Attribution

For Base execution, actual chosen entries are keyed by signal session and
security. The diagnostic reports only aggregate counts and statistics for:

1. entries shared by both policies;
2. entries selected only by transparent CAP1 (`displaced`);
3. entries selected only by F0-123 (`promoted`);
4. same-session displaced/promoted pairs, matched deterministically after shared
   entries are removed and each side is ordered by its own actual candidate rank.

A pair is called a `direct displacement` only when both policies had the same
pre-buy holdings (security, shares, and entry session), the same available slot
count, and cash equal within `1e-8` immediately before that session's first buy.
Other exclusive entries are `downstream path divergence`, not direct causal
replacements. Unequal exclusive counts are never truncated silently: paired and
unpaired counts and coverage are reported separately.

The report includes closed-trade economic return, realized PnL, win rate,
stop/time-exit mix, paired return delta, winner-state transitions, and monthly
aggregate signs. It also reports isolated executable-label quality for each
policy's Top5 and aggregate 123-factor descriptions:

- valid sample count;
- winner/loser mean and median;
- standardized winner-minus-loser difference;
- Spearman correlation to executable economic return;
- promoted/displaced mean and standardized difference.

No factor is selected, removed, weighted, thresholded, or used to create another
portfolio in this run. Rankings by absolute diagnostic effect are descriptive
only.

An exact portfolio NAV bridge decomposes the final challenger-minus-control NAV
into shared-trade sizing/path effect, direct displacement, and downstream path
divergence. Each category is further decomposed into gross price/ending-mark PnL,
fees and taxes, and dividends. Closed realized PnL and the final mark of any open
position are both included. The bridge must reconcile to final NAV within
`1e-8`; paired isolated returns are never presented as an additive portfolio
return explanation.

## Integrity And Persistence

- Database reads use the project's read-only data layer; all joins occur in
  memory.
- The source 2026 control and challenger must reproduce within `1e-12` for total
  return, PF, MaxDD, trade count, win rate, ex-best-week, and ex-top3 return.
- All eight deterministic monthly model hashes must match the source manifest.
- Output is limited to aggregate JSON/Markdown plus source/config/code hashes.
- The diagnostic must not persist a stock code, individual session, row-level
  factor, prediction, label, fill, ledger, model, CSV, Parquet, pickle, or joblib
  artifact.
- Output is staged and atomically renamed only after aggregate-schema, forbidden
  key, stock-code content, and file-suffix guards pass.
- There is no threshold, TopN, holding-period, stop, feature, or model-parameter
  scan.

## Decision Rule

The rejected F0-123 model remains rejected regardless of the diagnostic result.
The diagnostic passes only when every reproduction, NAV-bridge, and privacy
check passes, at least one displaced/promoted pair exists, and at least one pair
has mature closed returns. A later strategy hypothesis requires
its own preregistration; it cannot be declared successful from this seen-2026
diagnostic.
