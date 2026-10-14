# SHARED CONTEXT — R.O.A.D. Barbados Historic Handwriting Challenge

*Ground truth established by direct inspection of the local competition files on 12 Aug 2026.
All sub-agents MUST read this before reviewing the two inbox dossiers. Where this file
contradicts the dossiers, THIS FILE WINS — it is measured, not researched.*

Competition: R.O.A.D. Barbados Historic Handwriting Challenge (Zindi), 3 Jul → 4 Oct 2026,
$25,000 prize pool. Competitor: Osborn Nyakaru (Nairobi). Goal: top-5 private LB.
Strategy split mandated by the competitor: **70% research, 30% experiments.** Do not propose
running experiments; propose *what should be true* and *what the evidence says*.

Workspace root:
`C:\Users\ADMIN\OneDrive\Desktop\OSBORN\AGENTIC-AI-DVPT\ZINDI-PROJECTS\barbados-handwriting`

| Path | What |
|---|---|
| `data/raw/` | Train.csv, Test.csv, SampleSubmission.csv, Starters.zip (images.zip NOT yet downloaded) |
| `reference/starter_pack/Starters/` | Extracted official starter pack: VLM, Paddle-OCR, Kraken-OCR |
| `research_loop/research_md/CLAUDE_COMPASS_DOSSIER.md` | Prior research output #1 (Claude) |
| `research_loop/research_md/CHATGPT_STRATEGY_BLUEPRINT.md` | Prior research output #2 (ChatGPT) |
| `research_loop/findings/` | Where you write your review |

---

## MEASURED FACTS (verified locally, 12 Aug 2026)

### F1. Dataset cardinality
- `Train.csv`: **4,098 rows** (ID, Target). Zero nulls, zero duplicate IDs.
- `Test.csv`: **1,374 rows**. `SampleSubmission.csv`: 1,374 rows.
- Total 5,472 crops. Train/test ratio ≈ 75/25.
- IDs are 16-char mixed-case alphanumeric — no visible document/writer grouping signal.

### F2. These are FULL LINES, not words or short phrases
- Words per target: mean **11.3**, median 11, p90 15, max 22.
- Chars per target: mean **62.3**, median 62, p90 79, max 120.
- Word-count mode is 10–12. Only a negligible tail below 6 words.
- **Both dossiers describe the task as "word / short-line" or "one or more handwritten
  words (e.g. 'the')". That framing is wrong.** Every crop is a dense multi-word line.
  This changes: augmentation choice, aspect-ratio handling (very wide crops), the value of
  a language model (much higher with 11-word context), and TrOCR's IAM line-level prior
  (which is now a *good* match, not a mismatch).

### F3. The corpus is 17th CENTURY, not 18th–19th
- 4-digit years present in targets: **1639, 1640, 1641, 1642, 1643, 1644, 1645, 1647,
  1654, 1659, 1667, 1674, 1676, 1684, 1693**. Most common: 1640 (28), 1647 (14), 1641 (12).
- Spelled-out dates: "six hundred" / "Six hundred" appear 35 times; "seven hundred" **0**
  times; "eight hundred" 2.
- **Both dossiers state "18th–19th century" (taken from Zindi marketing copy). The data says
  mid-to-late 1600s.** Implication: English **secretary hand / mixed hand**, not the round
  hand or copperplate of the 18th–19th c. This is a materially harder and rarer script; it
  changes which pretrained checkpoints have relevant priors (e.g. CATMuS-Print/Medieval,
  Bentham, early-modern English models) and invalidates any reasoning that assumed
  18th–19th c. penmanship.

### F4. The ground truth uses a scribal-abbreviation markup convention nobody has noticed
- Charset is **81 characters**. No long-s (ſ), no diacritics, no Unicode beyond ASCII.
- `^` appears **872 times across 688 of 4,098 rows (16.8% of all training lines)**. It marks
  a **superscript** in the manuscript. Top tokens:
  `W^m`(51) `y^e`(39) `M^r`(35) `Adm^rs`(33) `w^th`(26) `w^ch`(24) `w^tsoever`(23)
  `Ex^rs`(19) `s^d`(17) `S^t`(15) `Exec:^rs`(9) `p^rsents`(9) `S:^d`(9) `24^th`(6) `Esq^r`(7)
  — note the `:` + `^` combination, which encodes a period/colon abbreviation mark *plus*
  a superscript.
- `~` appears **134 times across 87 rows** — a line-fill / continuation / abbreviation
  stroke (e.g. `for ye consideration ~`, `~~~`, `~Add~ Anderson`).
- Other rare-but-live chars: `_`(48) `|`(1) `\`(1) `#`(2) `*`(3) `+`(6).
- Archaic orthography is preserved verbatim: `publique`, `prsence`, `whome`, `writeing`,
  `wittnes`, `Pish` (parish), `divs`, `viz^t`, `&` for "and" (757 occurrences).
- **Neither dossier mentions `^`, `~`, or the `:`+`^` convention at all.** Any model that
  emits modernised or unmarked text loses these characters on ~17% of lines. This is the
  single most concrete, verifiable, under-exploited edge found so far.

### F5. The official starter metric is 0.7·CER + 0.3·WER — NOT 50/50
From `reference/starter_pack/Starters/*/eval_metrics.py` (identical in all three starters):

```python
def evaluate_set(gt_csv_path, pred_csv_path, set_name="SET",
                 weight_factor=0.5, cer_weight=0.7, wer_weight=0.3):
    ...
    final_score = cer_weight * cer_score + wer_weight * wer_score
```

- `weight_factor` (the length-weighting exponent) is **0.5** in the VLM and Kraken configs,
  but **1.0** in the Paddle-OCR config — inconsistent across the organizers' own starters.
- The scripts import `compute_weighted_wer` / `compute_weighted_cer` from an `evaluations/`
  directory that is **NOT shipped in Starters.zip** — so the exact weighting formula is
  still unverified; only the 0.7/0.3 blend and the exponent default are visible.
- **Both dossiers assert an equal 50/50 CER/WER blend.** If 0.7/0.3 is correct, the entire
  optimization posture shifts toward character accuracy: lexicon/word-level decoding tricks
  are worth proportionally less, never-emit-blank is worth more, and partial-credit
  character output beats confident whole-word guessing by a wider margin than assumed.
- **RESOLVING THIS IS PRIORITY ZERO.** It is possible the starter script is a local
  convenience that does not match the live leaderboard. Do not assume either way.

### F6. The organizers' own VLM baseline is Qwen2-VL-2B-Instruct + LoRA
From `reference/starter_pack/Starters/VLM/config.yaml`:
`model_id: Qwen/Qwen2-VL-2B-Instruct`, `r: 16`, `lora_alpha: 32`, `lr: 3e-5`,
`per_device_train_batch_size: 8`, `grad_accum: 2`, `epochs: 3`, `max_pixels: 1500000`,
`num_beams: 3`, `max_new_tokens: 128`, `bf16: true`.
Kraken starter references a checkpoint named `kraken_model_catmus_continued_2.mlmodel` —
i.e. the organizers themselves continued training from a **CATMuS** model. That is a strong
signal about which pretrained historical-HTR family is expected to transfer.

### F7. images.zip has not been downloaded
No image-level statistics exist yet (resolution, colour depth, aspect ratio, DPI, source
page grouping). Any claim in either dossier about image properties is **unverified**.

---

## THE TWO DOSSIERS UNDER REVIEW

- `CLAUDE_COMPASS_DOSSIER.md` — competition-intel heavy, conservative, flags its own
  unknowns, cites Kuzushiji 2019 and Latin-HTR literature with numbers.
- `CHATGPT_STRATEGY_BLUEPRINT.md` — architecture/technique heavy, confident, proposes
  length-weighted loss, decoder MLM pretraining, stroke-direction auxiliary supervision,
  dynamic-resolution batching, LLM post-correction. Several of its citations are GitHub
  repos, a Scribd document, and a personal portfolio page rather than peer-reviewed work.

**They disagree with each other on at least these points — adjudicate the ones in your lane:**
1. Metric blend: both say 50/50; the starter code says 70/30.
2. Compute: Compass assumes Kaggle free tier only; ChatGPT asserts renting cloud GPUs is
   "fully permitted".
3. Synthetic data: Compass says synthetic line generation from period fonts is worth doing
   "if legal"; ChatGPT says external fonts are strictly banned.
4. Language models: Compass cites Kuzushiji 2nd place *abandoning* a character LM as useless;
   ChatGPT makes LLM post-correction a headline edge.
5. Task granularity: both describe short/word-level crops (F2 shows this is false).
6. Century / script: both say 18th–19th c. (F3 shows this is false).
7. Test-set pseudo-labeling legality: Compass says UNRESOLVED; ChatGPT does not address it.

---

## HOW TO WRITE YOUR REVIEW

Write **one markdown file** to the exact path given in your task prompt. Structure:

1. **Verdict table** — every material claim you examined, one row each:
   `Claim | Source dossier | Status (CONFIRMED / REFUTED / UNVERIFIABLE / OVERSTATED / MISSING) | Evidence (URL or file:line) | So what`
2. **What both dossiers missed** — the highest-value section. Be specific and cite.
3. **Ranked recommendations** for a top-5 finish in your lane, each with: expected effect
   size, confidence (high/med/low), evidence basis, and cost.
4. **Open questions** you could not resolve, and exactly what would resolve them.
5. **Bibliography** — full titles, authors, venue, year, and a working URL/arXiv ID for
   everything you cite.

Rules of engagement:
- **Cite or don't claim.** Every number needs a source. If you cannot find a primary source
  for a figure quoted in a dossier, mark it UNVERIFIABLE and say so — do not repeat it.
- **Prefer primary sources**: arXiv, ACL/ICDAR/ICFHR/IJDAR proceedings, Google Scholar,
  Papers with Code, official model cards. Distrust blog posts, Scribd, and random GitHub
  repos presented as literature.
- Be adversarial. Your job is to break these documents, not to agree with them.
- Note licences for every model you recommend (competition requires openly available weights).
- Effect sizes must be realistic and sourced. Say "unknown" rather than inventing a range.
- Do NOT write code, train anything, or download images. Research and analysis only.
- You may need `WebSearch` / `WebFetch`; they are deferred tools — load them first with
  `ToolSearch` using `select:WebSearch,WebFetch`.
