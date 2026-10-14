# LB_LOG — every submission, in order

The leaderboard is the ground truth for any transfer-affecting keep/discard decision.
One row per submission. Never delete a row, including the losses — the losses are the
signal that tells us which levers are dead.

**Metric (UNCONFIRMED — see `research_loop/AGENT_BRIEF.md` "THE ONE HARD RULE"):**
starter code computes `final = 0.7·CER + 0.3·WER`, both length-weighted by `L^weight_factor`
with `weight_factor` = 0.5 (VLM, Kraken configs) or 1.0 (Paddle config).
**Lower is better.**

| # | Date | Config / change | Local score | Public LB | Δ | Keep? | Notes |
|---|------|-----------------|-------------|-----------|---|-------|-------|
| — | — | *no submissions yet — project is in the research phase* | — | — | — | — | — |

## Budget

| | |
|---|---|
| Submissions used | 0 |
| Daily cap | 5 *(unverified — dossier claim)* |
| Total cap | 200 *(unverified — dossier claim)* |
| Deadline | 2026-10-04 *(hour/timezone unverified)* |
