# UPDATE_01 — R.O.A.D. Barbados Historic Handwriting Challenge

*Paste this into an external deep-research model (Gemini Deep Research / ChatGPT / Claude).
Save its reply as `RESPONSE_01.md` in this folder.*

**Date:** 2026-08-12 · **Round:** 01 · **Submissions made:** 0 · **Best LB:** none
**Phase:** research (deliberate — 70% research / 30% experiments)

---

## What this is

I am competing in the **R.O.A.D. Barbados Historic Handwriting Challenge** on Zindi
(3 Jul → 4 Oct 2026, $25,000 pool). The task is **line-level offline handwritten text
recognition**: an image crop of a line from a 17th-century Barbadian manuscript → the
transcription string. Target: **top 5 on the private leaderboard**.

I already have two deep-research documents on this competition — one from Claude, one from
ChatGPT. **Before acting on either, I measured the actual competition files.** The
measurement contradicted both documents on four material points. This brief tells you what
is *measured* so you don't repeat their errors, and then asks you the questions that are
still open.

**Do not restate the four corrections below back to me as findings. They are settled.**

---

## MEASURED FACTS (from `Train.csv`, `Test.csv`, and the official `Starters.zip`)

**M1 — Cardinality.** Train 4,098 rows (`ID,Target`), Test 1,374 rows. Zero nulls, zero
duplicate IDs, 4,086 unique targets. IDs are opaque 16-char mixed-case alphanumerics with
**no document or writer grouping key**.

**M2 — These are dense FULL LINES, not words.** Words per target: mean 11.3, median 11,
p90 15, max 22. Characters per target: mean 62.3, median 62, p90 79, max 120.
*(Both prior documents describe "word / short-line" crops. That is wrong.)*

**M3 — The corpus is 17th century (1639–1693), not 18th–19th.** Years present in the
targets: 1639, 1640 (×28), 1641 (×12), 1642, 1643, 1644, 1645, 1647 (×14), 1654, 1659,
1667, 1674, 1676, 1684, 1693. "six hundred" ×35; **"seven hundred" ×0**.
⇒ English **secretary / mixed hand**, not round hand or copperplate.
*(Both prior documents say 18th–19th c., taken from the competition's marketing copy.)*

**M4 — The ground truth uses an undocumented scribal-abbreviation markup.** 81-character
ASCII charset, no long-s, no diacritics, no non-ASCII.
- `^` marks a **superscript**: **872 occurrences across 688 of 4,098 lines — 16.8% of all
  training lines.** Top forms: `W^m`(51) `y^e`(39) `M^r`(35) `Adm^rs`(33) `w^th`(26)
  `w^ch`(24) `w^tsoever`(23) `Ex^rs`(19) `s^d`(17) `S^t`(15) `p^rsents`(9) `Esq^r`(7)
  `24^th`(6) `viz^t`.
- `:`+`^` compounds: `Exec:^rs`(9), `S:^d`(9), `und:^r`.
- `~` — 134 occurrences across 87 rows; a line-fill / continuation / suspension stroke
  (`for ye consideration ~`, `~~~`, `~Add~ Anderson`).
- Rare live characters: `_`(48) `|`(1) `\`(1) `#`(2) `*`(3) `+`(6).
- Archaic orthography preserved verbatim: `publique`, `prsence`, `whome`, `writeing`,
  `wittnes`, `Pish` (parish), `divs`; `&` ×757.
**Neither prior document mentions any of this.** A model that emits modernized or unmarked
text bleeds character errors on roughly one line in six.

**M5 — The metric in the organizers' own starter code is `0.7·CER + 0.3·WER`, not 50/50.**
From `Starters/*/eval_metrics.py`, identical in all three starters:
```python
def evaluate_set(gt_csv_path, pred_csv_path, set_name="SET",
                 weight_factor=0.5, cer_weight=0.7, wer_weight=0.3):
    ...
    final_score = cer_weight * cer_score + wer_weight * wer_score
```
Both prior documents assert an equal 50/50 blend. Additionally: the per-sample length-weight
exponent `weight_factor` is **0.5** in the VLM and Kraken configs but **1.0** in the
Paddle-OCR config — inconsistent across the organizers' own starters — and the functions
that implement the weighting (`evaluations/wer.py`, `evaluations/cer.py`) are **not shipped
in the zip**, so the exact formula is unverified.

**M6 — The organizers' own baselines.** VLM starter fine-tunes `Qwen/Qwen2-VL-2B-Instruct`
with LoRA (r=16, α=32, lr=3e-5, bs=8×2, 3 epochs, max_pixels=1.5M, num_beams=3,
max_new_tokens=128, bf16). The Kraken starter continues from a checkpoint named
`kraken_model_catmus_continued_2.mlmodel` — i.e. from the **CATMuS** historical-HTR family.

**M7 — `images.zip` has not been downloaded**, so every image-level claim (resolution,
aspect ratio, colour depth, page grouping) is currently unverified.

---

## WHERE THE TWO PRIOR DOCUMENTS DISAGREE WITH EACH OTHER

1. **Metric blend** — both say 50/50; the starter code says 70/30.
2. **Compute** — one assumes Kaggle free tier only; the other asserts renting cloud GPUs is
   "fully permitted".
3. **Synthetic data** — one says synthetic line generation from period fonts is worth doing
   "if legal"; the other says external fonts are strictly banned.
4. **Language models** — one cites the Kuzushiji 2019 2nd-place team *abandoning* a character
   LM as useless; the other makes LLM post-correction a headline edge.
5. **Test-set pseudo-labelling legality** — one marks it UNRESOLVED; the other ignores it.

---

## WHAT I WANT FROM YOU

Answer with **primary sources and working URLs**. Where you cannot source a number, say
"unknown" — do not estimate. I would rather have five sourced answers than fifteen guesses.

**Q1 (highest value). The metric.** Find the exact evaluation code or specification the live
Zindi leaderboard uses for this competition. Specifically: is it 0.7/0.3 or 50/50? What is
the exact per-sample length-weighting formula, and is the exponent 0.5 or 1.0? Is `^`/`~`
inside the CER character set? Is WER whitespace-tokenized? Look for the organizers' public
repository — the starter scripts reference `evaluations/wer.py` and `evaluations/cer.py`,
which implies one exists.
*Then, analytically:* with a per-sample weight of `L^0.5` versus `L^1.0`, how does influence
redistribute across a length distribution with mean 62 and max 120 characters, and what does
each imply for whether to optimize long lines or short ones?

**Q2. The transcription standard.** Is the `^` superscript / `~` suspension / `:`+`^` markup
in M4 a *documented* convention? Check CATMuS transcription guidelines, HTR-United,
Transkribus/READ conventions, TEI abbreviation encoding, and the R.O.A.D. project's own
documentation if it has any. If it follows a published standard, that standard's manual is a
complete free specification of what my model must emit — I want it.

**Q3. Architecture for 17th-c. English secretary hand at line level, under a CER-dominant
objective.** Not IAM-modern benchmarks — historical ones. Head-to-head numbers on
early-modern Latin-script manuscripts for: TrOCR, PyLaia, Kraken/CATMuS, DAN, HTR-VT,
CRNN+CTC, and VLMs (Qwen2-VL/2.5-VL, GOT-OCR2.0, Florence-2). Include licences.
**And specifically: is the organizers' Qwen2-VL-2B baseline a genuinely strong start, or a
decoy?** VLM decoders hallucinate fluent modern text, which is exactly the failure mode that
destroys CER on archaic orthography.

**Q4. Language modelling on 4,098 in-domain lines (~46k words).** Is that enough to train a
character or word n-gram LM that helps HTR decoding? Cite work on LM corpus size vs. HTR
gain. Separately: does **MBR decoding with CER as the risk function** have published HTR
results — it is the one technique whose objective directly matches the metric.

**Q5. Post-OCR correction with an LLM — where has it made historical text *worse*?** I need
the failure literature, not the success literature. Silent modernization of `publique` →
`public` and stripping of `y^e` → `the` would be catastrophic here.

**Q6. Validation design with no grouping key.** With opaque IDs and no writer/document
column, how do published HTR papers construct writer-disjoint splits when writer labels are
absent? Is inferring document membership from named entities and formulaic openers a
recognized technique?

**Q7. What actually won comparable competitions.** ICDAR/ICFHR HTR competitions, Kaggle
Kuzushiji 2019, HIP shared tasks — what the winners *did*, not what the writeups recommend.

**Q8. Rules.** Adjudicate items 2, 3 and 5 in the disagreement list above from the live Zindi
rules page for **this** competition, with a URL. Also: exact deadline hour and timezone,
submission caps, public/private split, and whether final standing includes a code review.
