# Context pack: R.O.A.D. Barbados Historic Handwriting Challenge (Zindi)

Written 2026-10-13. Pasted into every research brief.

## The task

One sample is a JPEG of one handwritten line cut from a scanned 1600s Barbados archive record (deeds, wills), in
early-modern English. We predict the text exactly as written: old spellings, capitals, and markup such as `^` for
superscripts ("Ex^rs", "y^t"). Score = 1 - 0.5 * (word edits per line / 12 + character edits per line / 55),
higher is better. One wrong word costs as much as 4.58 wrong characters, so a reader that "corrects" an old
spelling loses a whole word. The competition closed on 5 Oct 2026; this is practice scored on our own held-out
lines, but we keep to its rules so the score is comparable.

## What we have

- 4,098 training lines with transcripts (mean 11.3 words, 62 characters; range 37 to 120 characters; 81 distinct
  characters), and 1,374 unlabelled test lines from the same scans (a random quarter).
- Two image batches: 75% short scans (about 1,100 x 56 px, 17 px per character, a clean later copyist's hand) and
  25% tall scans (about 4,000 x 350 px, 77 px per character, older hands on stained paper; much harder).
- 10% of the word tokens in a held-out fold never occur in the other folds; 62% of distinct words occur once.
- About 1 training label in 70 is reported wrong by entrants (unchecked).

## What is forbidden

- Training, fine-tuning or adapting on any data other than the competition files (staff: "Competition data only
  may be used for training or fine-tuning"). No outside handwriting datasets in any form.
- Pretrained models under research-only, academic-only or non-commercial licences. The licence must allow
  commercial use and deployment by the host. Staff: "Public availability of the model weights is not, by itself,
  enough". Denied: `stanford-oval/churro-3B` (its base carries the Qwen Research License). Approved: PP-OCRv6
  medium (Apache 2.0). Upstream training data need not be audited unless the licence passes restrictions down.
- Paid model services, proprietary APIs, AutoML, hand labelling.
- Never use another entrant's weights, predictions or files. Their ideas are fine.

## Where we stand

| reference point | score |
|---|--:|
| same transcript for every line (our only run so far) | 0.1404 |
| leaderboard benchmark | 0.0425 |
| entrants' single fine-tuned models, as reported on the forum | about 0.90 to 0.91 |
| private leaderboard: 50th / 10th / 1st | 0.9239 / 0.9277 / 0.9331 |

- No reader trained yet. The validation harness is frozen: 5 folds over the 4,098 lines, so a model is trained
  five times.
- The winner (forum write-up, ideas only): 8 reader variants from 5 base models x 5 folds, weighted
  character-level vote. Qwen3-VL 8B and Qwen3.5 (LoRA incl. vision tower, 2 epochs, lr 2e-4), LightOnOCR-2 (full
  fine-tune, lr 2e-5), a ConvNeXt-large CTC reader (30 epochs) and Kraken (12 epochs) with a 6-gram character
  language model built from the training transcripts. Kraken got the largest vote because the vision-language
  models modernise spellings. Per fold on one RTX A6000 (48 GB): 1 hour (ConvNeXt) to 21 hours (Qwen3-VL with
  nine image views). Bigger models (Qwen3-VL 32B, InternVL3.5-38B, Gemma-4 31B) were no better than 8B.
- The organisers' starter pack: Qwen2-VL-2B with LoRA, PaddleOCR, and Kraken continued from a CATMuS checkpoint.

## Tried and failed (by others, as reported)

- Dictionaries and word-list snapping, learned re-rankers, pseudo-labels, per-batch resolution fixes, bigger models.

## Numbers for sizing

- Noise in one score: about 0.002 on 4,098 held-out lines. Smallest gain we treat as real: 0.002.
- Compute: no local GPU. One rented GPU (24 to 48 GB class) or a hosted notebook GPU; budget matters, so a model
  that needs 20 GPU-hours per fold is 100 GPU-hours per experiment.

## Open questions

- Which openly licensed pretrained models are the strongest starting points for this, and which can we afford?
- Which of the popular OCR vision-language models fail the licence rule?
- What is the cheapest first reader that gets near 0.90?
