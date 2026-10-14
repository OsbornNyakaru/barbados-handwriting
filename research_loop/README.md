# Research loop — R.O.A.D. Barbados Historic Handwriting Challenge

Ported from `geoai-aquaculture/gemini_loop/`. Same machinery, different competition.

```
Claude (+ sub-agents) research & implement
   → share results + open questions with external deep research   (rounds/UPDATE_NN.md)
   → external model does deep research + critique                 (rounds/RESPONSE_NN.md)
   → triage into AGENT_BRIEF.md queue
   → implement the surviving ideas → submit → leaderboard score
   → repeat (UPDATE_NN+1.md)
```

## Files

| Path | What |
|---|---|
| `AGENT_BRIEF.md` | **The `program.md` of this project.** Any fresh agent session reads this FIRST: hard rules, measured facts, dead ends, experiment queue, per-iteration protocol. |
| `SHARED_CONTEXT.md` | Measured facts (F1–F7) from direct inspection of the competition files. Where this contradicts any research doc, **this wins**. Every sub-agent reads it before reviewing anything. |
| `research_md/` | External deep-research outputs pasted in verbatim (Claude/Compass, ChatGPT, Gemini). Untrusted until reviewed. |
| `findings/` | **Our own** adversarial reviews, cross-examinations, and literature notes. Everything here must be cited. |
| `rounds/UPDATE_NN.md` | Claude → external deep research. Reports what the last round scored, corrects wrong advice, states current approach, poses sharpened questions. |
| `rounds/RESPONSE_NN.md` | The external reply, saved so the loop has a record. |

Convention: one cycle per **major submission / result**. In the research-heavy phase
(this competition is being run 70% research / 30% experiments) a cycle may instead be
gated on a *resolved question* rather than a leaderboard score — see the log below.

## Log

| # | Date | Best LB before | Change tested | Result |
|---|------|----------------|---------------|--------|
| 00 | 2026-08-12 | — | Workspace built; `Train.csv`/`Test.csv`/`Starters.zip` inspected directly | 7 measured facts (F1–F7), **4 of which contradict both uploaded dossiers**. See `SHARED_CONTEXT.md`. |
| 01 | 2026-08-12 | — | 4 adversarial sub-agent reviews of the two dossiers (R1 rules/metric · R2 architecture · R3 decoding/ensembling · R4 data/paleography) | *in progress* |
