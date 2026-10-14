# R.O.A.D. Barbados Historic Handwriting Challenge — Competition Dossier

*Prepared for Osborn Nyakaru (Nairobi, Kenya) · Data current as of 9 August 2026 · Verified against zindi.world/zindi.africa competition pages, discussion forum, road.bb, and external prior-art sources.*

## TL;DR
- **The competition is REAL and verified.** It is a word / short-line handwritten text recognition (HTR) task on 18th–19th-century Barbadian archival records — ~6,000 cropped line images — scored by an equal 50/50 blend of length-weighted WER and length-weighted CER (lower is better). It runs **3 Jul → 4 Oct 2026**, has a **$25,000 USD prize pool** (Zindi's largest ever), is **open to all**, and is solo-friendly. Osborn is eligible for the three main prizes and points (not the "Top Barbadian" prize).
- **A top-5 finish is realistically achievable on Kaggle free-tier alone.** The task is line-level (not full-page detection), pretrained open models are allowed, and the decisive edges — exact ground-truth normalization matching, a domain lexicon/LM post-corrector, synthetic-data pretraining, and a multi-model ROVER/character-vote ensemble — none require paid GPUs.
- **Two unresolved rulings and one structural risk dominate strategy.** The multi-line labeling convention is unresolved by organizers; test-set pseudo-labeling legality is unresolved; and the private leaderboard is 70% of a small test set, so **shake-up risk is high — build a writer/document-disjoint CV and trust it over the public leaderboard.**

## Key Findings

**Verified competition facts (Zindi pages + forum, 9 Aug 2026):**
- **Task:** transcribe handwritten text from cropped line images of Barbadian colonial records — the page frames it against records where "faded ink, degraded pages, and unfamiliar handwriting make manual transcription slow, expensive, and inaccessible at scale," covering "deeds, wills, estate inventories, census records."
- **Data:** the dataset "consists of 6K cropped images of lines of text, extracted from scanned historical records"; real 18th/19th-century penmanship. Train.csv holds targets; Test.csv holds IDs; a Starter OCR Pack (VLM, Kraken-OCR, Paddle-OCR) and SampleSubmission.csv are provided. Images ship as one zip (images.zip on Google Cloud Storage).
- **Metric:** multi-metric = 0.5 × weighted WER + 0.5 × weighted CER; each sample weighted by √(reference length). Lower is better. Missing/empty predictions are scored as fully wrong.
- **Timeline:** opens 3 Jul 2026, closes and reveals 4 Oct 2026.
- **Rules:** open-source tools only; pretrained models allowed if openly available to all; only the competition dataset (no external data); no AutoML; 5 submissions/day, 200 total; max team size 4; no new team members in the final 5 days; 2 final submissions selectable; top-10 code review with a 48-hour window; public LB ~20–30% of test.

**R.O.A.D. = Reclaiming Our Atlantic Destiny.** It is "an initiative of the culture ministry under the Prime Minister's Office" (The Art Newspaper, 28 Apr 2026), announced by Prime Minister Mia Amor Mottley in December 2021 and executed on behalf of the Government of Barbados by Barbados Tourism Investment Inc. Per the official programme (road.bb), the archives are "the largest body of records documenting the development of Trans-Atlantic slave societies, outside of Great Britain," with the Archives Department custodian of "tens of millions of pages of these records." Technical delivery is by **GovTech Barbados Ltd.**, "a state-owned enterprise which falls under the remit of the Ministry of Innovation, Industry, Science and Technology." Programme Manager is Chereda Grannum. The launch (with a companion Barbados Lands & Surveys challenge) was unveiled at the 2025 AI for Good Summit in Geneva, and Advomag's coverage notes: "At USD $25,000, this is the largest ever prize pool for a Zindi challenge." **Modeling implication:** documents are colonial-era legal/estate records — English secretary-hand and round-hand, heavy on surnames, place-names, dates, monetary sums (£-s-d), and legal boilerplate.

## Details

### Phase 1 — Competition Comprehension

**Plain-language problem:** Convert scanned handwritten lines/words from old Barbadian legal documents into clean digital text, so archivists, historians, and genealogists can search and study millions of pages.

**ML framing:** Line/word-level offline HTR — image-to-text sequence transduction. Input: a variable-size cropped image containing one or more handwritten words (roughly a line). Output: a text string. This is **NOT** full-page detection+recognition (unlike Kuzushiji). That is good news: it removes the hardest engineering stage (page layout / character detection) and fits free-tier compute.

**Evaluation, exactly (from Zindi's weighted WER/CER guide):** Both metrics use Levenshtein edit distance. WER = edits / words-in-reference; CER = edits / chars-in-reference. "Weighted" means each sample's contribution is scaled by W_i = L_i^0.5 (√ of reference length, in words for WER, in chars for CER), so longer references matter more but sublinearly. Global score per metric = Σ(edits × √L) / Σ(√L), then the two are averaged 50/50. Conversational role tags are stripped before scoring. Strategy-critical facts:
1. Empty/missing predictions are scored as if you output an empty string — **always output something.**
2. On the weighted variant, a single long blank hurts disproportionately.
3. Raw leaderboard numbers can exceed 1.0 and differ substantially from a naive local `jiwer` WER/CER because of the weighting — a competitor reported LB WER≈2.89/CER≈6.82 while local jiwer gave WER~0.27/CER~0.11. **You must replicate the exact weighting locally or your model selection will be miscalibrated.**

**Legitimate metric optimization:**
- **(a) Ground-truth normalization matching** — match casing, punctuation, whitespace, and Unicode/diacritic conventions of the training targets exactly; a systematic mismatch inflates both WER and CER for free.
- **(b) Lexicon-constrained / LM decoding** — word-beam-search or shallow-fusion with a lexicon and character n-gram LM built from training transcriptions; the published LexiconNet system reached IAM 3.24% CER / 8.29% WER, reported as ~27% relative CER and ~43% relative WER improvement over greedy decoding (WER benefits more, which matters since WER is half our score).
- **(c) ROVER-style multi-system voting** — align outputs of several models and vote per word/character. Gains are real but should be estimated conservatively: NIST's original ROVER work reports on the order of a few-percent relative WER reduction (Fiscus's modified ROVER cites "an additional 5% relative word error reduction"), and a 2026 ROVER-ensemble variant (RECOVER) reports "8–46% relative reductions" in entity-phrase WER depending on setup. Treat ~5–15% relative as the realistic planning figure, not the higher end.
- **(d)** Because CER is half the score and more forgiving than WER, prefer outputs that get most characters right even when unsure of a whole word — never leave blanks.

**Hard Rules Register (re-read weekly):**
- **External data:** NOT allowed. Only competition data. This bans IAM/Bentham/etc. as *training data* (they remain fine as sources of pretrained weights).
- **Pretrained models:** ALLOWED if openly available to everyone (Hugging Face weights fine). Research/non-commercial licenses (e.g. Qwen research license) are *probably* fine given the "accessible to all" spirit, but **a Zindi organizer has NOT explicitly confirmed the license class — the only reply came from another competitor. Treat as allowed but document every model license.**
- **Commercial APIs** (Google Vision, GPT-4V, etc.): effectively banned — rules require open-source, no paid services, no free trials needing a credit card.
- **AutoML:** banned. **Optuna is explicitly allowed** (confirmed by Zindi staff).
- **Dropping corrupted training rows:** explicitly allowed; keep Train.csv unchanged and document exclusions. Zindi confirmed this and a competitor already published 21 corrupted IDs.
- **Test-set pseudo-labeling / self-training:** **NOT yet ruled on.** Two separate threads asked; no organizer has confirmed. The nearest official statement ("no use of test data or external labels") was specifically about training-data cleaning. **Treat as UNCONFIRMED; do not depend on it for a final winning submission until Zindi answers in writing.**
- **Submissions:** 5/day, 200 total. 2 final submissions chosen (else best 2 public used).
- **Team merger deadline:** no new members within the final 5 days → hard cutoff **~29 Sep 2026.**
- Winners assign copyright to Zindi; top-3 and top-10 must submit reproducible, seeded code within 48 hours.
- Manual labeling / Excel edits banned — all data manipulation in code.

**Timeline / timezone:** Zindi deadlines are typically 23:59 UTC on the close date. Close/reveal 4 Oct 2026. **Confirm the exact hour on-platform before the final day (UNVERIFIED to the minute).**

### Phase 2 — Data Intelligence

- **Files:** Train.csv (326 KB: ID + transcription), Test.csv (24 KB, IDs), SampleSubmission.csv, images.zip (all train+test crops), Starters.zip (VLM, Kraken-OCR, Paddle-OCR baselines).
- Images are variable-size line crops; expect faded ink, bleed-through, staining, skew, variable spacing.
- **Known data issues:** ~21 corrupted/mislabelled training rows identified by competitor Alpcan_Cepik and acknowledged by Zindi — including `79tMUVyfIdy3GzkG` (empty/unreadable), `F8DYDDp2AvW9Dytw` and `JU7lRwk3jKkus24Z` (text does not match label), plus 18 more. **Drop these** (full list is in the forum's "hyperparameter search & training-data cleaning" thread).
- **Critical ambiguity — multi-line labeling:** competitor youssefilo found most multi-line images are labeled with only the centre/main line, but a portion carry the full multi-line transcription — an inconsistent convention. Zindi acknowledged (5 Aug 2026) and asked for example IDs but **has NOT resolved it.** This directly affects target construction and test scoring. **This is the single highest-value clarification to chase.**
- **Train/test split basis is UNVERIFIED** (not documented whether by document, scribe, or random). In HTR, writer-disjoint vs writer-overlapping splits change everything. **Assume writer/document-disjoint (worst case)** and build CV accordingly; run adversarial validation (train-vs-test classifier on image features) to detect distribution shift.
- Leaderboard benchmark shows WER Weighted ≈ 11.5 and CER Weighted ≈ 53.8 for a trivial baseline (weighted scale, not 0–1), confirming large headroom.

### Phase 3 — Prior Art

**Kuzushiji Recognition (Kaggle 2019)** — most-relevant historic-cursive precedent, but it was full-page detection scored by F1, so architecture transfers only partially. Final private F1: 1st 0.950, 2nd 0.950, 3rd 0.944.
- **1st (tascj):** a single **Cascade R-CNN with HRNet-w32 + w48 backbones**, aggressive multi-scale train/test, backbone ensemble. Score ladder from the repo: single w32 modest multi-scale **0.935 → wider multi-scale 0.946 → +w48 ensemble 0.950**. Decisive edge: *simplicity + strong backbone + multi-scale TTA*. No test pseudo-labeling documented.
- **2nd (lopuhin):** Faster R-CNN (resnet152) detector + separate classifier on large 512×768 context crops (resnet152 + resnext101_32x8d_wsl), a `seg_fp` reject class, then a **second-level LightGBM/XGBoost stack.** Test-set pseudo-labeling ("writer adaptation") used top-80%-confident predictions (>99% accuracy). Ladder: best single 0.935 → fold blend 0.941 → **GBM stack 0.949**. **Notably, a character bi-LSTM language model was tried and abandoned as nearly useless** because the image model was far stronger.
- **3rd (knjcode):** Faster R-CNN (ResNet101) + **5-model classifier ensemble** (EfficientNet-B4, ResNet152×2, SE-ResNeXt101×2, L2-softmax) with mixup/RICAP/random-erasing, TTA×7, and a **LightGBM false-positive remover tuned with Optuna.** Test-set pseudo-labeling for book/writer adaptation. Exact per-step deltas UNVERIFIED (the repo's overview.md was inaccessible).

**Transferable Kuzushiji lessons:** (1) writer/book domain shift between train and test is the core difficulty → pseudo-labeling for adaptation was worth ~0.003–0.01 F1 for 2nd/3rd; (2) a second-level GBM combiner was a reliable late gain; (3) a naive character LM can underperform a strong recognizer — **validate an LM before trusting it.**

**Latin-script HTR quantified gains (peer-reviewed, not competition winners):**
- **Lexicon / word-beam-search decoding:** LexiconNet — IAM 3.24% CER / 8.29% WER, ~27% relative CER and ~43% relative WER improvement over greedy.
- **Test-time augmentation (rotation/shear):** an IAM CNN-BiLSTM study cut WER ~1.8 points (13.8→12.03%) and CER ~0.5 point.
- **Test-time adaptation (SCRIBE):** mean ~20.8% relative CER and ~12.8% relative WER reduction across GW/IAM/RIMES/Bentham.
- **Post-OCR correction:** ICDAR 2019 post-OCR competition best correction improvement 44%; a 2026 LLM judge-and-retry pipeline reported ~27% cMER improvement.

**SOTA models (2023–2026):**
- **Fine-tuned TrOCR-large** ≈ 2.89% CER on modern IAM; on historical material 4–10% CER is more typical, improved by synthetic pretraining (HATFormer showed synthetic Stage-1 pretraining worth ~4.2% CER on Arabic; medieval-manuscript work showed 48–85% relative gains from document-specific fine-tuning).
- **Fine-tuned Qwen2.5-VL-7B vs fine-tuned TrOCR-large head-to-head** on 18th-century English (arXiv:2602.14524, Feb 2026): Qwen **~0.60% CER / 2.48% WER** beat TrOCR **~1.05% CER / 3.89% WER** overall, and Qwen degraded far less on black-and-white/degraded lines (TrOCR CER rose from 0.82% on colour to 1.46% on B&W). **But Qwen silently normalized historical spellings** ("Antient"→"Ancient", "imploy"→"employ") — a real risk against verbatim/diplomatic ground truth. 82% of Qwen's and 77.5% of TrOCR's errors were edit-distance 1–2. The authors decline to crown a universal winner, and note the comparison is confounded by scale (7B vs ~558M).
- **GOT-OCR2.0** (580M), **Florence-2** (0.23–0.77B), **PaddleOCR PP-OCRv4/v5** (CRNN+CTC), and **Kraken** are all free-tier-trainable (Kraken and Paddle ship in the Starter Pack). Florence-2 runs OCR in ~1s/image on a T4 and fine-tunes at batch size 1 on a T4.

**Compute reality (Kaggle free tier, confirmed 2025–2026):** ~30 GPU-hours/week, P100 16GB **or** T4×2 (16GB each, 32GB combined), 12-hour session cap, 20GB persistent storage, no card; 2×T4 counts as 1× quota. This **rules out** full fine-tuning of 32B/72B VLMs and multi-GPU-only training, but comfortably supports TrOCR-base/large fine-tuning, CRNN+CTC, Florence-2, and **QLoRA fine-tuning of a 7B VLM** (rank-8 LoRA, 4-bit, batch 1–4 with grad accumulation — a documented working recipe uses rank 8, α=16, dropout 0.05, lr 2e-4, batch 4, grad-accum 8). Osborn's QLoRA experience applies directly.

### Phase 4 — Strategy Synthesis

**Validation strategy:** Build GroupKFold with groups = document/scribe if any grouping signal is recoverable from IDs or image metadata; otherwise stratify by reference length and an unsupervised hardness proxy (ink contrast, image width). **Replicate the EXACT weighted WER/CER locally** (√ weighting, identical normalization, tag-stripping) — non-negotiable given the LB↔jiwer mismatch competitors already hit. Expect a meaningful shake-up (private LB = 70% of a small test set). **Rule: trust local CV; only believe a public-LB move if it exceeds the fold-to-fold standard deviation of your CV.** Keep a fixed hold-out to detect overfitting to the public slice.

**Preprocessing & augmentation playbook (ranked by impact/effort):**
1. Exact ground-truth normalization matching (highest ROI, near-zero compute).
2. Aspect-ratio-preserving resize to fixed height + width padding (organizers confirmed this is standard/allowed).
3. Drop the 21 known corrupted rows.
4. Contrast normalization / CLAHE; grayscale.
5. Binarization (Sauvola/Otsu) as an optional input channel or TTA variant — test on VLMs, don't assume it helps.
6. Deslanting and deskewing.
7. Elastic distortion + random rotation/shear (best-performing HTR augmentations in the literature).
8. Random erasing / grid distortion; mild blur/Gaussian noise to mimic degradation.
9. Bleed-through / stain simulation.
10. Synthetic line generation from period fonts + training vocabulary for a pretraining stage — **confirm legality with organizers** since it toes the "only provided data" line (you generate it in code from provided transcriptions).
11. Text-line height normalization.
12. Lexicon from training transcriptions (surnames, Barbadian parish/plantation names, legal boilerplate, £-s-d money formats).
13. Character n-gram LM (KenLM) from training targets for shallow-fusion/beam rescoring.
14. Inference TTA (multi-scale, slight rotation) then ROVER/character-vote merge.
15. Curriculum: easy (clean, short) → hard (degraded, long).

**Modeling roadmap (three tiers, with compute budget):**
- **Tier A — Week-1 baseline (~3–5 GPU-hrs):** run the Starter Pack (Paddle/Kraken/VLM) and fine-tune TrOCR-base on the line crops with the exact metric locally. Goal: a valid submission + a calibrated CV harness.
- **Tier B — Workhorse, top-20 (~10–15 GPU-hrs/week):** fine-tune TrOCR-large AND QLoRA-fine-tune Qwen2.5-VL-7B (or fine-tune Florence-2/GOT-OCR2.0). Add synthetic pretraining if allowed. Add lexicon+LM decoding. Back the CV winner of TrOCR-large vs Qwen-7B.
- **Tier C — Frontier, top-5 (~20–30 GPU-hrs/week peak):** ensemble 3–5 diverse models (TrOCR-large, Qwen-7B QLoRA, a CRNN+CTC, Florence-2/GOT-OCR2.0) combined via ROVER/character-level voting; add a lightweight character-LM or GBM second-level corrector on aligned outputs; test-time adaptation / pseudo-labeling **only if Zindi confirms legality**; TTA on every model. This configuration is what historically separates winners.

**The 3–5 biggest edges (concrete):**
1. **Exact ground-truth normalization matching** — most competitors will bleed free WER/CER to casing/punctuation/whitespace/Unicode mismatch. Reverse-engineer the target convention from Train.csv and mirror it exactly.
2. **Domain lexicon + character-LM decoding built only from training transcriptions** — Barbadian surnames, parish/plantation names, legal phrasing, £-s-d money formats. Word-beam-search-style decoding gives outsized WER wins.
3. **Multi-model ROVER/character-vote ensemble** — diverse architectures make different errors; voting reliably beats the best single model (plan for ~5–15% relative WER reduction, per the conservative ROVER literature).
4. **Resolving and correctly handling the multi-line convention** — if centre-line-only vs full-line is knowable from crop geometry, exploiting it (or matching whatever Zindi rules) is a differentiator most will get wrong.
5. **Historic-script specifics** — long-s (ſ), superscript abbreviations, struck-through text, ligatures, and the VLM silent-modernization trap. Force verbatim output matching ground-truth orthography rather than modernized spelling.

**Top 10 failure modes and guards:**
1. Local metric ≠ leaderboard → replicate the weighted metric exactly.
2. Public-LB overfit → trust CV, keep a fixed hold-out, choose 2 final subs by CV.
3. VLM silent modernization of spelling → verbatim prompting + CER check against raw targets.
4. Multi-line convention mismatch → track the Zindi thread; design for both cases.
5. Blank/missing predictions penalized as fully wrong → always emit non-empty output.
6. Using external data (IAM/Bentham) as *training* → banned; pretrained weights only.
7. Relying on test pseudo-labeling that gets ruled illegal → keep a clean, clearly-legal submission as one of the 2 finals.
8. Corrupted rows poisoning training → drop the known 21; scan for more in-code.
9. Writer-overlap leakage inflating CV → group folds by document/scribe.
10. Non-reproducible/unseeded code → seed everything; top-10 must reproduce within 48h or be dropped.

### Phase 5 — Execution Plan (9 Aug → 4 Oct 2026)

- **Week 1 (9–15 Aug):** Join, download data, build the exact weighted-metric harness and writer/document-disjoint CV, drop corrupted rows, run Starter Pack, submit a TrOCR-base baseline. Post the multi-line and pseudo-label questions to force organizer rulings.
- **Week 2 (16–22 Aug):** Fine-tune TrOCR-large; QLoRA Qwen2.5-VL-7B. Establish the CV leader. Build the training-vocabulary lexicon.
- **Week 3 (23–29 Aug):** Add synthetic pretraining (if legal) and the augmentation stack; add lexicon+character-LM decoding. Target top-20.
- **Week 4 (30 Aug–5 Sep):** Add a 3rd/4th diverse model (CRNN+CTC, Florence-2/GOT-OCR2.0). Build ROVER/character-vote merge. Begin ensembling.
- **Week 5 (6–12 Sep):** Tune ensemble weights and decoding on CV; error analysis on historic-script specifics (long-s, abbreviations). **Teaming decision** — if outside top-10 on CV, seek a complementary teammate before the 29 Sep merger cutoff.
- **Week 6 (13–19 Sep):** Freeze architecture. Only refinements (normalization, lexicon, TTA) after this. Finalize the pseudo-labeling decision on Zindi's ruling.
- **Week 7 (20–26 Sep):** Endgame. Lock two final submissions: (1) safest highest-CV ensemble using only clearly-legal methods; (2) highest-CV ensemble including any organizer-approved pseudo-labeling. Complete team merger by 29 Sep if teaming.
- **Week 8 (27 Sep–4 Oct):** Reproducibility pass — seed, clean, document excluded rows and model licenses so the top-10 code review passes. Submit finals; **pick the 2 by CV, not public LB.**

**Submission budget:** 5/day, 200 total — spend early submissions on metric calibration; reserve ~40 for the final fortnight of ensemble/decoding tuning.

## Recommendations
1. **This week:** build the exact weighted-WER/CER evaluator and a document/writer-disjoint CV before training anything serious — everything downstream depends on trustworthy local scoring. Submit a TrOCR-base baseline to validate the pipeline end-to-end.
2. **Force two organizer rulings now:** (a) the multi-line labeling convention for the test set, and (b) whether test-image pseudo-labeling/self-training is allowed. Both materially change the design and both are currently unresolved.
3. **Commit to the Tier-B fork early:** fine-tune TrOCR-large and QLoRA Qwen2.5-VL-7B in parallel in week 2, then back the CV winner. Given the Feb-2026 head-to-head, Qwen-7B likely wins raw error — but watch its silent spelling normalization against verbatim ground truth.
4. **Invest disproportionately in the cheap high-leverage edges** — normalization matching, training-vocab lexicon+LM decoding, ROVER ensembling, and script-specific handling — over chasing a bigger single model. These are what free-tier competitors under-invest in and where top-5 is won.
5. **Manage shake-up risk:** pick both final submissions by cross-validation, keep one fully-legal "safe" submission, and never let a public-LB bump override a CV that disagrees.

**Benchmarks that change the plan:** if a public baseline notebook already scores very high (e.g. a fine-tuned VLM near the top of the LB), pivot immediately to ensembling+decoding for marginal edge rather than base-model work. If Zindi confirms pseudo-labeling is legal, prioritize it (worth ~0.003–0.01 in analogous competitions). If CV scores look implausibly high, suspect writer-overlap leakage and re-group folds before trusting anything.

## Caveats
- Train/test split basis, exact image resolutions/color depth, exact train/test row counts, and the exact submission deadline hour are **UNVERIFIED** beyond what the public pages show — confirm on-platform.
- The **multi-line labeling convention** and the **legality of test-set pseudo-labeling** are **UNRESOLVED** by organizers as of the data cut.
- The **pretrained-model license question** (research/non-commercial weights) was answered only by a competitor, not Zindi — treat as **probable-but-unconfirmed**.
- Kuzushiji was full-page detection with an F1 metric; its architectures transfer only partially to this line-level, edit-distance task. Transferable lessons are about domain adaptation, ensembling, and LM caution — not the specific detectors.
- The TrOCR-vs-Qwen numbers are from a single Feb-2026 paper on English print/handwriting, not this dataset; treat as directional. Several cited CER figures are non-competition academic benchmarks. The ROVER improvement range was revised downward to ~5–15% relative after checking primary sources — earlier higher figures (up to ~55%) come from specific favorable setups and should not be assumed here.
- Osborn is not eligible for the "Top Barbadian" prize; the three main prizes and Zindi points are open to him.