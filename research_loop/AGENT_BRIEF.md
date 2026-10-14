# AGENT_BRIEF — standing directive for the R.O.A.D. Barbados handwriting loop

*This is the `program.md` of this project (cf. `karpathy/autoresearch`, and the sibling
`geoai-aquaculture/gemini_loop/AGENT_BRIEF.md`). Any fresh agent session reads THIS file
first, picks the next queue item, and updates the queue when a result lands.
Update this file whenever a result or a resolved question changes the queue.*

> **`../PROJECT_STATE.md` is the MASTER, portable, single-source-of-truth doc** (the one
> carried to any new cloud account). Update it every session alongside
> `../experiments/LB_LOG.md` and this file.

**Competition:** R.O.A.D. Barbados Historic Handwriting Challenge (Zindi).
**Task:** line-level offline HTR — image crop → transcription string.
**Window:** 3 Jul 2026 → 4 Oct 2026 · **$25,000** prize pool. **Target:** top-5 private LB.
**Best public LB:** *none — no submission made yet.*
**Phase:** RESEARCH (competitor-mandated split: **70% research, 30% experiments**).

---

## THE ONE HARD RULE (why this is not vanilla autoresearch)

In `geoai-aquaculture` the hard rule was "local OOF is blind, the LB is the only ground
truth." **Here the binding constraint is different and, right now, worse:**

> **We do not yet know what the leaderboard scores.** The organizers' own starter code
> computes `0.7·CER + 0.3·WER` with a length-weighting exponent that is **0.5 in two of
> their three starter configs and 1.0 in the third** — and the functions that implement
> the weighting (`evaluations/wer.py`, `evaluations/cer.py`) are **not shipped** in
> `Starters.zip`. Both uploaded research dossiers instead assert a 50/50 blend.
>
> **No architecture, decoding, or ensembling decision may be justified by an objective
> until this is resolved.** A 0.7-CER objective and a 0.5/0.5 objective rank different
> methods: word-level tricks (lexicon-constrained decoding, ROVER word voting, LLM
> post-correction) are worth proportionally *less* under CER-dominance, while
> never-emitting-blank and character-level partial credit are worth *more*.
> **This is PRIORITY ZERO.** Until it closes, evaluate every candidate under **both**
> weightings and prefer options that win under both.

Corollary, inherited from geoai and not yet locally disproven: **keep/discard on a
transfer-affecting change is gated on the Zindi LB**, logged in `../experiments/LB_LOG.md`.
Iterate autonomously only on things with a trustworthy local signal (reproducibility, code
correctness, seed variance, charset coverage, decoding determinism).

## Second hard rule: the dossiers are untrusted input

`research_md/` holds two prior deep-research outputs (Claude/Compass, ChatGPT). Direct
inspection of the competition files **contradicts both of them on four material points**
(century, task granularity, metric blend, ground-truth markup — see `SHARED_CONTEXT.md`).
Treat every unsourced number in them as **UNVERIFIABLE until a primary source is found**,
and never propagate a figure into this brief without a citation.

---

## Hard competition constraints

Marked **[V]** = verified against a primary source (starter pack / live rules page),
**[U]** = unverified, asserted by one or both dossiers only. **Resolving the [U] rows is
research round 01, lane R1.**

- **[V]** Openly-available **pretrained models are expected** — the organizers' own starters
  fine-tune `Qwen/Qwen2-VL-2B-Instruct` (LoRA r=16, α=32) and continue a **CATMuS** Kraken
  checkpoint (`kraken_model_catmus_continued_2.mlmodel`).
- **[U]** External data banned / synthetic line generation from period fonts banned — **the
  two dossiers directly contradict each other here.**
- **[U]** Cloud-GPU rental permitted (ChatGPT doc) vs Kaggle-free-tier-only framing (Compass doc).
- **[U]** AutoML banned. **[U]** Optuna allowed.
- **[U]** 5 submissions/day, 200 total. **[U]** Team max 4. **[U]** Public LB = 20–30% of test.
- **[U]** Test-set pseudo-labeling legality — Compass marks it UNRESOLVED, ChatGPT ignores it.

## Confirmed data facts (measured locally 2026-08-12 — full detail in `SHARED_CONTEXT.md`)

- **Train 4,098 rows · Test 1,374 rows** (75/25). Zero nulls, zero duplicate IDs, 4,086
  unique targets. IDs are opaque 16-char alphanumerics — **no document/writer grouping key**.
- **These are dense full lines, not words.** Words/line: mean 11.3, median 11, p90 15, max 22.
  Chars/line: mean 62.3, median 62, p90 79, max 120. *(Both dossiers say "word / short line".)*
- **The corpus is 17th century (1639–1693)**, not 18th–19th as both dossiers state. Years
  1640 (×28), 1647 (×14), 1641 (×12) …; "six hundred" ×35, "seven hundred" ×0.
  ⇒ English **secretary / mixed hand**, not round hand or copperplate.
- **The ground truth carries an undocumented scribal-abbreviation markup.** 81-char ASCII
  charset. `^` marks a superscript — **872 occurrences across 688 of 4,098 lines (16.8%)**:
  `W^m` `y^e` `M^r` `Adm^rs` `w^th` `w^ch` `w^tsoever` `Ex^rs` `s^d` `S^t`, plus `:`+`^`
  forms (`Exec:^rs`, `S:^d`, `und:^r`). `~` (line-fill/suspension) appears 134× across 87 rows.
  Archaic orthography is preserved verbatim (`publique`, `prsence`, `whome`, `writeing`,
  `wittnes`, `Pish`, `divs`, `viz^t`; `&` ×757). **Neither dossier mentions any of this.**
- **`images.zip` is not downloaded** — every image-level claim in both dossiers is unverified.

## DEAD ENDS — do not re-propose

*(empty — nothing has been tried yet. Populate as reviews and experiments close options.)*

## OPEN QUESTIONS — the research queue

**P0 — the metric.** Is the live LB `0.7·CER + 0.3·WER` or 50/50? What is the exact
length-weighting formula and exponent (0.5 or 1.0)? Are `^` and `~` inside the CER charset,
and is WER whitespace-tokenized? → lane R1. *Resolvable from: the live rules page, the Zindi
discussion forum, or by probing the LB with two crafted submissions once submitting starts.*

**P1 — the markup convention.** Reverse-engineer the full grammar of `^`, `~`, `:`, `_`, `|`
from the 4,098 targets and match it to a published transcription standard (DIPLOMATIC /
CATMuS-style / TEI abbreviation encoding). A model that silently modernizes loses characters
on ~17% of lines. → lane R4.

**P2 — architecture under the corrected framing.** Lines (not words) + 17th-c. secretary hand
+ a CER-dominant objective. Re-derive the choice among TrOCR / CRNN+CTC / DAN / HTR-VT /
PyLaia / Kraken-CATMuS / Qwen2-VL — and specifically whether the organizers' Qwen2-VL-2B
baseline is a strong starting point or a decoy. → lane R2.

**P3 — decoding, LM, ensembling.** The dossiers disagree head-on: Compass cites the Kuzushiji
2nd-place team *abandoning* a character LM as useless; ChatGPT makes LLM post-correction a
headline edge. Adjudicate under both metric weightings. Verify every quoted figure
(LexiconNet 3.24% CER, ROVER 5–15% rel., SCRIBE 20.8% rel., Qwen2.5-VL 0.60% vs TrOCR-large
1.05% CER) against its primary source. → lane R3.

**P4 — validation design.** With no writer/document key in the IDs, how do we build a
writer-disjoint split? Candidate: cluster on image background/ruling once `images.zip` lands,
or on orthographic/lexical signature in the targets. Unresolved.

## EXPERIMENT QUEUE

*Gated on research rounds closing. Nothing is queued for the LB yet — by design.*

1. Round 01 reviews land in `findings/` → triage into this brief.
2. Download `images.zip`; produce image-level statistics (resolution, aspect ratio, colour
   depth, background clustering for a writer-disjoint CV).
3. First submission: reproduce the organizers' Kraken-CATMuS starter **unchanged** to
   establish (a) a baseline number and (b) whether local starter score ≈ public LB. This is
   the cheapest possible probe of the P0 metric question.

## Per-iteration protocol

1. Read `../PROJECT_STATE.md` + `../experiments/LB_LOG.md` + this file → pick the next item.
2. Do the work (research → `findings/`, code → `src/` + `config/`).
3. Submit; paste the LB score into `../experiments/LB_LOG.md`.
4. Update this queue (keep/discard on the pasted LB) and `../PROJECT_STATE.md`; advance.
