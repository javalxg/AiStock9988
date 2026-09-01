# CAP1 F0-123 Executable Ranker V1 Independent Review

Status: `GO_BEFORE_FORMAL_RUN`.

The independent read-only review initially returned `NO-GO` and the formal run
was not started. The following blockers were corrected before this approval:

- prediction planning now permits latest-date open positions to be marked at
  the database cutoff instead of incorrectly requiring a mature H10 horizon;
- the executable label has a distinct CAP1 profile using the prior valid close
  as the trailing-stop reference, matching the canonical engine;
- the model profile binds the exact CAP1 strategy ID and configuration hash;
- strategy, model profile, output, and preregistration are mandatory CLI
  arguments;
- the old runner's transparent `rule_score` ascending-order bug was corrected
  to the canonical descending order and is now a hard verification check;
- both ledgers must cover every signal date, use the correct monthly model ID,
  and contain unique stock-session keys;
- Base control closed trades must match executable labels one-for-one on
  decision identity, signal date, symbol, trigger type, exit date, and economic
  return within `1e-10`; an empty or partial join cannot pass.

Final review found no remaining blocker. It confirmed prior-valid-close state,
data-gap handling, stop next-open execution, sellability retry, H10 ordering,
monthly label maturity, full-date coverage, and all economic acceptance gates.
`py_compile` and `git diff --check` passed before registration.
