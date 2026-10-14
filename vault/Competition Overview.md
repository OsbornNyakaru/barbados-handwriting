# Competition Overview

Source: https://zindi.world/competitions/road-barbados-historic-handwriting-challenge (read 2026-10-13). **Closed.**

## The task

Each sample is an image of one handwritten line cut from a scanned record in Barbados' archive (deeds, wills,
estate inventories). Predict the text of the line exactly as written, old spellings and abbreviation marks included.

## Why the organisers want it

- Host: the Reclaiming Our Atlantic Destiny (R.O.A.D.) Programme, run with GovTech Barbados, which is digitising
  "tens of millions of pages" on Atlantic slave societies and the lives of the enslaved and their descendants.
- Stated aim: "faster, more scalable digitisation of archival data"; the winning solution "could serve as a
  blueprint for digitising other dispersed archives across the Commonwealth".
- They intend to deploy it. That is why staff required licences that allow commercial use, banned outside training
  data and asked for all weights and inference code.

## How they want it solved

- A reproducible handwriting reader trained only on the provided lines, starting from an openly licensed
  pretrained model. No hand labelling, no paid services, no AutoML.
- Starter pack: Qwen2-VL-2B with LoRA, Paddle-OCR, and Kraken continued from a CATMuS checkpoint.
- What won (thread 35246): several fine-tuned vision-language readers plus two small CTC readers, 5 folds each,
  combined by a weighted character-level vote.

## What the data looks like to us (inference, not stated on the page)

- Notarial and probate text from the 1600s in an early-modern English hand, not the 18th and 19th century named
  on the data tab ([[F-003 Full lines from the 1600s with scribal markup]]).
- The transcribers followed a fixed convention (`^` for superscripts, `~` for fill strokes) and kept spellings.

## Numbers

| | |
|---|---|
| Train / test lines | 4,098 / 1,374 |
| Target | text, mean 11.3 words and 62 characters per line |
| Metric | `1 - 0.5 * (word edits / 12 + character edits / 55)`, higher is better ([[F-002 The score formula]]) |
| Public rows (estimated) | 275 to 412 (20% or 30% of 1,374) |
| Prize | USD 25,000: 10,000 / 6,000 / 4,000, plus 5,000 for the top Barbadian |
| Submissions | 5 a day, 200 total, 2 chosen |
| Participants | 1,893 joined, 661 active |
| Leaderboard (private): top / 10th / 50th | 0.9331 / 0.9277 / 0.9239 |
| Closed | 5 Oct 2026 02:29 EAT |
