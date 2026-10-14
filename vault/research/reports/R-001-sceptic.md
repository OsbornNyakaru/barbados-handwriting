# Sceptic's review of R-001 (13 Oct 2026)

Start with a CTC reader, not a vision-language model (VLM). The reports' cheapest evidence for VLMs is thinner than it reads, and two of the three reports contradict each other on whether CATMuS starting points are allowed.

## Ranked options

Scores and costs are my guesses unless marked measured. Cost is for 5 folds, 48 GB / 24 GB.

| Option | Report | Verdict | Evidence | Rules | Score alone | Cost, 5 folds | Reason |
|---|---|---|---|---|---|---|---|
| Kraken from scratch + 6-gram character LM (fold-train text only) | small | adopt | moderate | allowed | 0.86–0.90 | 5–10 h / same | Winner's top-weighted voter; cannot modernise spellings |
| Kraken continued from CATMuS (paired arm) | small | test | moderate | unruled | +0 to +0.01 over scratch | +5–10 h | Organisers did it; use McCATMuS, not Medieval |
| ConvNeXt-large (V1, ImageNet) CTC | small | adopt | moderate | allowed | 0.86–0.90 | 5 h measured / 5–8 h | Only option with a measured cost and no rules doubt |
| LightOnOCR-2-1B-base, full fine-tune | VLM | test | weak | allowed | 0.86–0.90 | 3–5 h / same | Cheap third family; no line-level evidence I could open |
| Qwen3-VL-8B LoRA incl. vision, one view | VLM | adopt after the above | moderate (winner only) | allowed | 0.85–0.91 | 8–15 h / 12–20 h, may not fit | Best single VLM; costly |
| Vote across 3 families | VLM | adopt | moderate | allowed | 0.915–0.925 | none extra | Winner got about +0.02 from 40 models; expect less from 15 |
| Qwen3-VL-2B | VLM | test (one-fold smoke test) | weak | allowed | 0.82–0.88 | 2–4 h | Debugging only |
| Qwen3.5-4B | VLM | park | weak | allowed | 0.85–0.90 | 4–8 h | Same family as 8B, little vote diversity |
| PP-OCRv6 medium rec | audit | park | weak | allowed (staff-approved) | unknown | 2–5 h | Its win relied on forbidden-for-us pretraining |
| Per-batch pixel budget / height / centre crop | VLM, small | park | weak | unruled (image-size rule) | — | 1 fold-run each | Forum says it gained nothing |
| StackMix, HTR-VT, Retsinas CNN | small | park | weak | allowed | 0.85–0.89 | 10–20 h each | Same niche as Kraken |
| CTC "spelling arbiter" | VLM | drop | weak | allowed | — | — | Word snapping failed for others; the vote already does it |
| TrOCR, GOT-OCR2, nine views, >8B models | all | drop | — | mixed | — | 105 h for nine views | See below |

## Contradictions

- **CATMuS starting points.** The VLM report calls outside-handwriting fine-tunes "forbidden"; the small-readers report recommends CATMuS Kraken first; the audit says "licence passes". My reading: allowed by the letter, but unruled. The ruling restricts what we train on, upstream data "need not be audited", and the organisers' own starter did it. The unanswered Kansallisarkisto question is the risk. Run Kraken both ways and quote the from-scratch score as the rules-clean one. The same logic applies to any licensed fine-tune, so the VLM report's blanket ban is inconsistent rather than safe.
- **Resolution.** Both reports propose per-batch scaling; the forum says it gained nothing. The VLM report's support (Kiessling, 48 to 96 px) is about a CTC model whose default squeezed lines to 48 x 320; it says nothing about upscaling for Qwen. Kiessling also found no gain above 96 px. Switching on image size may breach the cited image-metadata rule.
- **Kraken cost.** "Under 1 GPU-hour per fold, feasible on a T4" conflicts with the winner's range, whose minimum was 1 hour (ConvNeXt) on an A6000.
- **Qwen3-VL-8B cost.** The estimate of 1 to 2.5 h per fold is consistent with 21 h / 9 views = 2.3 h, but only at the top end.
- **Medusa size.** The VLM report says Qwen3.5-4B; Kiessling says Qwen3.5-9B. So "a 15.9M CTC reader beat a 4B model" is misstated.
- **TrOCR.** The small-readers report says "TrOCR-large, 333.9M parameters"; the paper used TrOCR-base. Licences agree: base-handwritten is `mit`, large-handwritten has none on the card.
- **GOT-OCR2.** The card says apache-2.0. The GitHub README badges say code Apache 2.0, data CC BY NC 4.0; I found no "research only" wording. Passes on paper; park for its awkward API.
- **Beam search.** No real conflict: greedy for VLMs, LM beam for CTC.

## Formula checks

- Coefficients: 0.5 x 11.3/12 = 0.4708 and 0.5 x 62/55 = 0.5636. Correct.
- At WER = 3.7 x CER the slope is 2.306, so 0.90 needs 4.34% CER and 0.933 needs 2.9%. Correct.
- Forecast table: blended CER 4.875%, WER 18.04%, score 0.8876; with LM 0.9000. Correct, but the 12% tall-batch CER is a pure guess.
- Leaderboard, assuming test lines average 11.3 words and 62 characters:

| Rank | Score recomputed | WER | CER | WER/CER |
|---|--:|--:|--:|--:|
| 1 | 0.9331 | 10.86% | 2.80% | 3.88 |
| 50 | 0.9239 | 12.21% | 3.30% | 3.70 |

- The 3.7 ratio holds for top entries. It is unproven for a lone CTC reader: Kraken ratios were 3.95 (2606.24302) and 4.2 to 4.7 (Kiessling). At 4.5, 0.90 needs 3.7% CER.
- About 76% of the top entry's loss is the word term, at only 1.4 character edits per wrong word.
- Tokens at 32 px: short 34 x 2 = 68; 2x upscaled 69 x 4 = 276; tall 125 x 11 = 1,375. All correct. Half-size tall is 310 only at 2,000 x 175; at the 2,000 x 176 the report prescribes it is 372 to 378. I did not check the processor's minimum-pixel default, which may upscale short lines anyway.

## Licence checks I made

All via `huggingface.co/api/models/<id>` and `/raw/main/README.md`.

- **Qwen/Qwen3-VL-8B-Instruct, -2B-Instruct:** exist; `license: apache-2.0`; no base model; no LICENSE file in the repo.
- **Qwen/Qwen3.5-4B:** exists; LICENSE file opens "Apache License Version 2.0"; base Qwen3.5-4B-Base. Thinking is on by default (confirmed).
- **lightonai/LightOnOCR-2-1B and -base:** exist; "Apache License 2.0"; base card says "Base model for fine-tuning". Component lineage is not on the v2 card.
- **PaddlePaddle/PP-OCRv6_medium_rec:** exists; apache-2.0 (card only).
- **timm/convnext_large.fb_in22k_ft_in1k:** exists; apache-2.0.
- **Kraken:** LICENSE is Apache 2.0 (raw.githubusercontent.com/mittagessen/kraken/main/LICENSE).
- **McCATMuS** (zenodo.org/records/13788177): cc-by-4.0, "16th century to the 21st century", file `McCATMuS_nfd_nofix_V1.mlmodel`. This is the period match.
- **CATMuS Medieval** (records 12743230, 15030337, 21488839): could not open (403). Unverified by me.
- **Kansallisarkisto/multicentury-htr-model:** apache-2.0, base trocr-large-handwritten.

## Papers

- **2609.20064 (Kiessling):** read from the PDF. Table 4 numbers are correct. But PP-OCRv6 lost to Kraken from scratch; it won only after the author's own pretraining on 6.2M historical lines. Medieval, not English.
- **2602.14524:** as claimed; printed, 195k lines, vision frozen. No normalisation rate is given.
- **2606.24302:** as claimed, bar the TrOCR-base slip; Kraken was not statistically tested.
- **2404.18722:** table and the "below 2% CER" caveat confirmed.

## Could not verify

- The 2U1 and Unsloth claims (QLoRA incompatible with vision training; vision learning-rate advice).
- The Medusa card, and the entrant snippet of CER 7.9% / WER 22.3% (that would score about 0.85).
- Whether stock Kraken supports n-gram decoding.
- The LightOnOCR CATMuS result (50 test lines).
- Every GPU-hour figure except ConvNeXt's 1 h and the 21 h.
- No source matches our setting; all scores are inference.

## First three, in order

1. Kraken, 5 folds from scratch, with the 6-gram LM; the McCATMuS arm paired. It is cheapest, rules-clean and produces the first real score.
2. ConvNeXt-large CTC, 5 folds, for a two-reader vote at about 15 GPU-hours total.
3. Qwen3-VL-2B on one fold as a smoke test, then Qwen3-VL-8B single view; add LightOnOCR-2 only if the vote gains at least 0.002.

**Not doing:** nine-view Qwen (about 105 GPU-hours, no ablation published).
