# PROJECT_STATE — single source of truth (portable across cloud accounts)

> **What this file is.** The one document carried to any cloud account. It holds everything:
> how to resume anywhere, the current champion, every experiment + LB score, what improved,
> what declined, the lessons, and the next action. Supporting files
> (`research_loop/AGENT_BRIEF.md`, `research_loop/SHARED_CONTEXT.md`,
> `experiments/LB_LOG.md`) go deeper; this file summarizes and points to them.
>
> **Structure ported from the sibling project `geoai-aquaculture`.**

- **Competition:** R.O.A.D. Barbados Historic Handwriting Challenge (Zindi)
- **Task:** line-level offline handwritten text recognition — image crop → transcription
- **Window:** 3 Jul 2026 → 4 Oct 2026 · **$25,000** prize pool
- **Goal:** top-5 private leaderboard
- **Strategy split (competitor-mandated):** 70% research, 30% experiments
- **Last updated:** 2026-08-12

---

## 1. Resume anywhere — read in this order

1. This file.
2. `research_loop/AGENT_BRIEF.md` — the standing directive and the queue.
3. `research_loop/SHARED_CONTEXT.md` — the measured facts. Beats any research document.
4. `experiments/LB_LOG.md` — every submission and its score.

## 2. Layout

```
barbados-handwriting/
├── PROJECT_STATE.md            ← you are here
├── config/                     run configs
├── data/
│   ├── raw/                    Train.csv, Test.csv, SampleSubmission.csv, Starters.zip
│   │                           (images.zip NOT yet downloaded)
│   ├── interim/  processed/  submissions/
├── docs/                       competition brief, rules register, decision log
├── experiments/LB_LOG.md       every submission, in order
├── notebooks/  src/  tools/
├── reference/starter_pack/     extracted official starters: VLM · Paddle-OCR · Kraken-OCR
└── research_loop/              the research loop (ported from geoai-aquaculture/gemini_loop)
    ├── README.md               loop convention + round log
    ├── AGENT_BRIEF.md          the program.md — read first
    ├── SHARED_CONTEXT.md       measured facts F1–F7
    ├── research_md/            external deep-research outputs (untrusted)
    ├── findings/               our own cited reviews
    └── rounds/                 UPDATE_NN.md ↔ RESPONSE_NN.md
```

## 3. Status

**Phase:** research round 01. **Submissions made: 0. Best LB: none.**

Deliberate — no model has been trained and none should be until the metric question (P0)
and the ground-truth markup question (P1) are closed. See `AGENT_BRIEF.md`.

## 4. Ledger

| Round | Date | What | Outcome |
|---|---|---|---|
| 00 | 2026-08-12 | Workspace built; two prior research dossiers ingested; `Train.csv`, `Test.csv`, `Starters.zip` inspected directly | 7 measured facts. **4 contradict both dossiers.** |
| 01 | 2026-08-12 | 4 adversarial sub-agent reviews (R1 rules/metric · R2 architecture · R3 decoding/ensembling · R4 data/paleography) | *in progress* |

## 5. Narrative

Two deep-research dossiers were supplied up front — one from Claude (`CLAUDE_COMPASS_DOSSIER.md`,
competition-intel heavy, conservative, flags its own unknowns) and one from ChatGPT
(`CHATGPT_STRATEGY_BLUEPRINT.md`, architecture heavy, confident, thinly sourced in places).
Rather than build from them, the first action was to measure the actual competition files.

That measurement broke both documents on four points:

1. **The metric.** Both assert 50/50 CER/WER. The organizers' own starter code computes
   **`0.7·CER + 0.3·WER`**, with a length-weighting exponent that is inconsistent across
   their own three starter configs (0.5, 0.5, 1.0), and the weighting functions themselves
   are **not shipped** in `Starters.zip`.
2. **The century.** Both say 18th–19th (Zindi marketing copy). The targets contain years
   **1639–1693** and zero instances of "seven hundred" — this is a **17th-century** corpus,
   i.e. English **secretary hand**, a materially harder and rarer script.
3. **The granularity.** Both describe word / short-line crops. The targets average
   **11.3 words / 62.3 characters** per line. This raises the value of a language model and
   makes TrOCR's IAM line-level prior a *good* match rather than a mismatch.
4. **The transcription convention.** The ground truth encodes scribal abbreviations with a
   markup neither dossier mentions: `^` for superscripts on **16.8% of all training lines**,
   `~` for line-fill/suspension strokes, `:` + `^` compounds, and verbatim archaic
   orthography. A model that emits modernized or unmarked text bleeds CER on ~1 line in 6.
   **This is the most concrete under-exploited edge found so far.**

## 6. Lessons

1. **Measure the shipped files before trusting any research document.** Four of the seven
   facts that matter were wrong in both dossiers, and all four were recoverable in minutes
   from `Train.csv` and `Starters.zip`.
2. **Marketing copy is not data provenance.** "18th–19th century" came from the competition
   page; the corpus itself says 1639–1693.
3. **Read the organizers' starter code, not just their prose.** The 0.7/0.3 blend, the
   `weight_factor` inconsistency, and the CATMuS lineage of their Kraken checkpoint are all
   only visible in the code and configs.

## 7. Next action

Land the four round-01 reviews in `research_loop/findings/`, triage them into the
`AGENT_BRIEF.md` queue, then close P0 (metric) and P1 (markup grammar) before any training.
