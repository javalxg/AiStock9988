# CAP1-20 Trade Mechanism Diagnostic

Status: `DIAGNOSTIC_SEEN_HISTORY_NO_STRATEGY_CHANGE`.

This report describes the 41 sealed Base trades. It does not backtest a new
exit, scan a threshold, alter CAP1-20, use XGBoost, or touch the frozen
202-factor system.

## What the current return means

- Frozen Base return is +32.44%, PF
  2.254, MaxDD
  -8.27%.
- Frozen Stress return is +28.48%, PF
  2.058, MaxDD
  -8.82%.
- There are 41 closed trades and 23 winners
  (56.1%). The top three trades contributed
  50.0% of net realized PnL, while the
  already sealed ex-top3 portfolio return remains
  +16.23% Base and
  +12.86% Stress.

## What actually separated winners

- Static entry separation is modest. Winners were deeper below MA60
  (-14.93% vs -13.76%),
  entered in weaker 20-day markets (-6.97% vs
  -4.39%), and had milder amount expansion
  (1.30 vs
  1.47). Candidate rank was nearly identical
  (1.65 vs
  1.56).
- The path separated much more strongly. By the E2 close, eventual winners
  averaged +3.36%; eventual losers averaged
  -2.86%. E2 return had Spearman
  0.573 with final trade return.
- Of 22 trades above entry at E2, 18
  became net winners (81.8%). Of
  19 at or below entry, only
  5 became winners
  (26.3%). This is in-sample
  mechanism evidence, not an authorized E2 exit rule.
- Across the full held close path, winners averaged MFE
  +15.03% and MAE
  -0.40%; losers averaged MFE
  +1.10% and MAE
  -7.81%.

## Where the current contract leaks money

- The 39 time exits averaged +5.18%
  and produced RMB 413,645.
  The two stop exits averaged -19.53% and lost RMB
  89,242.
- The nominal trailing close stop is not a guaranteed -8% fill. It triggers
  after a close and executes at the next tradable open; the worst realized
  trade was 000628.SZ at
  -29.37%.
- Winners still rose an average +2.50%
  over the next five session opens and
  +4.27% over the next ten; losing
  trades continued -0.50% and
  -2.83%. CAP therefore has both a
  left-tail timing problem and a right-tail truncation problem.

## Decision

Do not replace CAP with another static indicator stack or another XGBoost
ranker. The evidence says the entry rule identifies a useful reset state, but
the economically strongest missing information arrives after the fill. Keep
CAP1-20 unchanged and evaluate the already hash-registered early-path overlay
only on append-only signals from 2026-09-01 onward. The latest common database
cutoff is still 2026-08-28, so no forward signal or forward return exists yet.
