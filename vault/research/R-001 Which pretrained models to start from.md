# R-001 Which pretrained models to start from

Written 2026-10-13 from 3 research reports (licence audit, vision-language readers, small letter-by-letter readers)
and a sceptic's review. Reports are saved in `vault/research/reports/`. No handwriting dataset, model weight or
entrant file was opened. Claims marked (inference) are a researcher's reasoning, not a cited result. Every score
and GPU-hour figure below is an estimate unless marked measured. The competition is closed; this is practice.

## What the research changes

1. **The Qwen models we wanted are allowed.** Every size of Qwen3-VL and Qwen3.5 is Apache 2.0, as are Qwen2-VL
   2B/7B and Qwen2.5-VL 7B/32B. Source: model cards read by the auditor; Qwen3-VL-8B, Qwen3-VL-2B and Qwen3.5-4B
   re-checked by the sceptic. Cited.
2. **One Qwen size is a trap.** Qwen2.5-VL-3B carries the Qwen Research License ("for non-commercial purposes
   only"), and so does everything built on it: churro-3B, Nanonets-OCR-s, Nanonets-OCR2-3B. Cited (LICENSE file).
3. **Other models that fail:** Chandra 1 and 2 (modified OpenRAIL, revenue cap), LFM2-VL (revenue cap), timm
   ConvNeXt V2 weights (CC-BY-NC). Avoid, as unruled: Gemma 3 and PaliGemma (Gemma terms pass restrictions down),
   DINOv3 (own gated licence), Idefics3 (Llama 3.1 base), TrOCR checkpoints other than `trocr-base-handwritten`
   (no licence on the card). Cited.
4. **A vision-language reader alone is not enough.** A fine-tuned Qwen2.5-VL-7B changed "Antient" to "Ancient"
   and "imploy" to "employ" on 18th-century printed English (arXiv 2602.14524, cited; printed text, 195k lines,
   so the transfer to us is by analogy). The winner saw the same and gave a small CTC reader the largest vote.
   About 76% of the top entry's lost score is the word term (sceptic's arithmetic from the leaderboard columns).
5. **The score in error-rate terms.** score = 1 - 0.471 x word error rate - 0.564 x character error rate, for lines
   of 11.3 words and 62 characters (derived by a researcher, re-checked by the sceptic). Rank 1 is 10.9% word
   error and 2.8% character error; rank 50 is 12.2% and 3.3%. A lone reader needs about 4.3% character error to
   reach 0.90 if word error is 3.7 times character error, and 3.7% if the ratio is 4.5 (inference).
6. **Settings for fine-tuning a vision-language reader:** train the vision part too (freezing the encoder hurt in
   arXiv 2606.24302 and arXiv 2503.19546, both on TrOCR-style readers); decode greedily, not with the sampling
   defaults on the Qwen cards; turn "thinking" off on Qwen3.5; do not use 4-bit loading when training the vision
   part (trainer README, not re-checked by the sceptic).
7. **A character language model helps hard hands and can hurt clean ones.** Tarride et al. 2024 (arXiv
   2404.18722, cited, re-checked): word error fell on three datasets and rose on the cleanest (below about 2%
   character error). Tune it separately for the tall and short batches.
8. **Readers pretrained on outside handwriting are unruled.** CATMuS Kraken models (CC-BY 4.0) and the
   Kansallisarkisto model (Apache 2.0) pass on licence. Staff said upstream data need not be audited and the
   organisers' starter continued from a CATMuS checkpoint, but a direct question went unanswered (thread 35088).
   Community fine-tunes of Qwen and LightOnOCR on handwriting are in the same position. We keep them out of any
   score we call rules-clean.

## Shortlist

| # | model | licence | size | role | score alone (estimate) | GPU-hours, 5 folds (estimate) |
|--:|---|---|---|---|---|---|
| 1 | `Qwen/Qwen3-VL-8B-Instruct` | Apache 2.0 | 8.8B | main vision-language reader (LoRA incl. vision, one view) | 0.85 to 0.91 | 8 to 15 on 48 GB; may not fit 24 GB |
| 2 | Kraken, trained from scratch, + 6-gram character language model | Apache 2.0 (software) | small | letter-by-letter reader; keeps old spellings | 0.86 to 0.90 | 5 to 10 |
| 3 | `timm/convnext_large.fb_in22k_ft_in1k` + CTC head | Apache 2.0 | 198M | second letter-by-letter reader | 0.86 to 0.90 | 5 (winner measured 1 h per fold on an A6000) |
| 4 | `lightonai/LightOnOCR-2-1B-base` | Apache 2.0 | 1.0B | cheap second vision-language family (full fine-tune) | 0.86 to 0.90 | 3 to 5 |
| 5 | `Qwen/Qwen3-VL-2B-Instruct` | Apache 2.0 | 2.1B | one-fold smoke test of the pipeline | 0.82 to 0.88 | 2 to 4 |
| 6 | `Qwen/Qwen3.5-4B` | Apache 2.0 | 4.7B | parked: same family as #1, little to add to a vote | 0.85 to 0.90 | 4 to 8 |
| 7 | `PaddlePaddle/PP-OCRv6_medium_rec` | Apache 2.0, staff-approved | 16M | parked: lost to Kraken from scratch without extra pretraining (Kiessling 2026) | unknown | 2 to 5 |

A vote across three families is estimated at 0.915 to 0.925 (inference from the winner's gain with 40 models).

## Plan, in order

| # | step | needs | basis | expected gain |
|--:|---|---|---|---|
| 1 | Decide where GPU runs execute | a decision ([[D-003 GPU runs on molab, code from a public GitHub repository, data in private Hugging Face]]) | - | - |
| 2 | Kraken from scratch, 5 folds; then the 6-gram language model per batch | 5 to 10 GPU-hours | items 4, 7 | first real score, 0.86 to 0.90; answers EDA questions 4 to 6 |
| 3 | Qwen3-VL-2B on one fold | about 1 GPU-hour | item 6 | proves the vision-language pipeline cheaply |
| 4 | Qwen3-VL-8B, 5 folds, one view, vision LoRA | 8 to 15 GPU-hours, 48 GB card | items 1, 6 | best single reader, up to about 0.91 |
| 5 | Character-level vote of 2 and 4 | none | item 4 | +0.01 to +0.02 over the best single |
| 6 | ConvNeXt-large CTC, then LightOnOCR-2, each kept only if the vote gains 0.002 or more | 5 and 3 to 5 GPU-hours | shortlist | small |

The list of readers is fixed here before any is scored. Steps 2 and 4 could swap if the user prefers to start
with Qwen; the cost is a later answer to EDA questions 4 to 6 and a first run about twice as expensive.

## Not doing

- Qwen2.5-VL-3B, churro-3B, Nanonets-OCR-s / OCR2-3B, Chandra, LFM2-VL, ConvNeXt V2: licence fails.
- Gemma 3, PaliGemma, DINOv3, Idefics3, TrOCR-large: licence unclear and nothing we need.
- Kraken continued from McCATMuS as our main reader: unruled. It may be run as a paired side arm, reported
  separately and never mixed into a rules-clean score.
- Nine image views per line: about 105 GPU-hours for 5 folds, and no published ablation.
- Models above 8B: no gain for the winner.
- A "spelling arbiter" that snaps words to the CTC reader's spelling: word snapping failed for others; the vote
  already does this.
- Zero-shot reading, DeepSeek-OCR, GOT-OCR2.0, olmOCR: page-oriented or fixed-prompt; no line-level evidence.

## Where we disagree with the sceptic

- The sceptic marks per-batch resizing "unruled" under the image-metadata rule. Resizing an image before reading
  it is preprocessing, not a feature, and the winner used different heights for normal and tall scans. We treat it
  as allowed; whether it helps is a separate, open question ([[H-002 Bring both batches to one letter size]]).
- The sceptic would run ConvNeXt second. We put Qwen3-VL-8B ahead of it: the vote needs one reader of each kind
  before a second of the same kind.

## Weak points in the research

- No source matches our setting (about 4,000 lines of early-modern English handwriting, this metric). Every score
  range is inference.
- Every GPU-hour figure is an estimate except the winner's 1 hour (ConvNeXt) and 21 hours (nine-view Qwen3-VL-8B)
  per fold. One researcher's "Kraken under 1 hour per fold on a T4" is unsupported.
- Most licences were read from the card's tag through the Hugging Face API; the full LICENSE text was opened only
  for the doubtful ones and for Qwen3.5-4B. Qwen3-VL repositories have no LICENSE file, only the tag.
- Whether stock Kraken supports a language model in decoding was not confirmed; the winner's write-up says theirs did.
- Errors the sceptic caught in the reports: the model beaten by a small CTC reader was 9B, not 4B; a TrOCR result
  was for TrOCR-base, not large; the PP-OCRv6 result relied on the author's own pretraining on 6.2M outside lines.
- One researcher opened a search snippet about another entrant's model card. Nothing from it is used here.

## Hypotheses opened

- [[H-001 A small letter-by-letter reader keeps old spellings]] (updated)
- [[H-004 Qwen3-VL-8B with the vision part trained is the best single reader]]
- [[H-005 A vote of one vision-language and one letter-by-letter reader beats both]]
- [[H-006 A character language model helps the tall batch and not the short one]]
