# Vision-language models as line readers: research report (13 Oct 2026)

Tags: [C] cited, [I] my inference, [U] unconfirmed. Model cards and licences were checked on 13 Oct 2026.

## Three things to know first

- **The Feb 2026 paper exists but is not about handwriting.** arXiv:2602.14524 (Vesalainen et al.) covers 18th-century *printed* English, about 195,000 training lines, with Qwen2.5-VL-7B fine-tuned with the vision encoder frozen against TrOCR-large. Qwen won on CER (0.60% vs 1.05%) and WER (2.48% vs 3.89%) but made lexical spelling changes ("Antient" to "Ancient", "imploy" to "employ"). TrOCR's normalisation was mostly limited to ligatures. [C]
- **I found no controlled paper that fine-tunes a vision-language model and a CTC reader on the same ~4,000 historical handwritten lines.** The closest evidence is Kiessling 2026 and the winner's forum write-up in the context pack.
- **All GPU-hour figures below are my estimates, not measurements.** The only anchor is the winner's 21 hours per fold for Qwen3-VL 8B with nine views on an A6000.

## Idea cards

**1. Qwen3-VL-8B LoRA with the vision tower included is the reference reader.**
- Evidence: winner's write-up (context pack: LoRA including vision, 2 epochs, lr 2e-4). Unsloth docs default to training vision and language layers, r=16, alpha=16 [C: https://unsloth.ai/docs/basics/vision-fine-tuning].
- Run: r=16 and r=64, lr 2e-4 on the language model, 2 to 3 epochs, greedy decoding, one view.
- Expect: 0.89 to 0.91, matching forum reports [I].
- Cost: about 1 to 2.5 GPU-hours per fold on 48 GB [I].
- Confidence: high.

**2. Training the vision tower beats freezing it, at a lower learning rate than the language model.**
- Evidence: TrOCR ablation on 13th-century Italian (1,666 lines) and READ-16: freezing 6 encoder layers took CER from 8.0 to 9.7, all 12 to 16.5; decoder freezing was tolerated [C: https://arxiv.org/html/2606.24302]. Kohút and Hradiš 2025: encoder-only matched or beat full tuning [C: https://arxiv.org/html/2503.19546v1]. The 2U1 trainer README says vision lr works best 5 to 10 times smaller, and that QLoRA is incompatible with vision training [C: https://github.com/2U1/Qwen-VL-Series-Finetune].
- Run: three arms on one fold: vision frozen, vision LoRA, vision LoRA at lr 2e-5.
- Expect: +0.005 to +0.02, largest on the tall batch [I].
- Cost: three 8B fold runs.
- Confidence: medium-high.

**3. Set the pixel budget per batch.**
- Evidence: Qwen2 and 2.5 use 28x28 px per visual token; Qwen3-VL uses 32x32 [C: 2U1 README]. Kiessling: doubling line height from 48 to 96 px cut CER by 8.3 to 11.5 points for PP-OCRv6 [C: https://arxiv.org/pdf/2609.20064]. Medusa infers at max_pixels 401,408, about 392 tokens [C: https://huggingface.co/ENC-PSL/Medusa0.1Line-4B].
- Token arithmetic for Qwen3-VL [I]: a short line (1,100x56) is about 68 tokens native, and about 276 if upscaled 2x. A tall line (4,000x350) is about 1,375 tokens native, and about 310 at half size.
- Run: upscale short lines 2x, downscale tall lines to about 2,000x176, cap at roughly 400 tokens; compare with native.
- Expect: +0.005 to +0.015 from upscaling short lines; 3 to 4 times faster on tall lines at little loss [I].
- Confidence: medium (the context pack lists "per-batch resolution fixes" as failed for others).

**4. LightOnOCR-2-1B-base, fully fine-tuned, is the cheap second reader.**
- Evidence: winner used it (full fine-tune, lr 2e-5). A community CATMuS line model went from 105% to 5.4% CER on 50 test lines at lr 6e-5 [C: https://huggingface.co/wjbmattingly/LightOnOCR-2-1B-catmus-textualis]. It takes no text prompt, and the card's longest edge is 1,540 px [C: https://huggingface.co/lightonai/LightOnOCR-2-1B].
- Run: start from the base checkpoint, lr 2e-5 to 6e-5, 3 to 5 epochs.
- Expect: slightly below the 8B reader alone, but useful in the vote [I].
- Cost: about 0.5 to 1 GPU-hour per fold; fits 24 GB [I].
- Confidence: medium-high.

**5. Qwen3.5-4B as a mid-size alternative.**
- Evidence: Medusa is Qwen3.5-4B, trained on about 645,000 lines, greedy decoding [C: card]. Kiessling could not reproduce its claimed CERs, and a fine-tuned 15.9M-parameter PP-OCRv6 beat it on medieval French, Latin and Spanish (3.36, 8.70, 7.99 vs 3.74, 13.07, 11.19) while running about 45 times faster [C: 2609.20064]. Thinking mode is on by default and must be disabled [C: https://huggingface.co/Qwen/Qwen3.5-4B]; the 2U1 README says to disable FlashAttention 2 for Qwen3.5 [C].
- Run: same recipe as card 1.
- Expect: within ±0.005 of the 8B reader [I].
- Cost: about 0.7 to 1.5 GPU-hours per fold [I].
- Confidence: medium.

**6. Greedy decoding only; never the model-card sampling defaults.**
- Evidence: Qwen3-VL cards recommend temperature 0.7 with presence_penalty 1.5 [C: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct], which penalises repeated tokens that a transcript legitimately contains [I]. CHURRO and Medusa both use greedy [C].
- Run: greedy vs beam 3 and 5 on one fold's saved adapter.
- Expect: beam within ±0.002 [I]; I found no reported beam gain for vision-language readers.
- Cost: inference only.
- Confidence: high for greedy, low for beam.

**7. Vote across model families.**
- Evidence: winner's weighted character-level vote lifted singles at about 0.90 to 0.91 up to 0.933, with the largest weight on Kraken. TrOCR ensemble on 16th-century Latin (3,603 lines): top-5 sentence-level vote gave CER 1.60 vs 1.86 best single; character-level voting was dropped as inconsistent [C: https://arxiv.org/html/2508.11499v1]. CHURRO's oracle ensemble reached 76.3 vs 70.1 best single [C: https://arxiv.org/html/2509.19768v1].
- Run: ROVER-style character alignment of out-of-fold predictions from the 8B reader, LightOnOCR-2 and a CTC reader, with weights fitted out of fold.
- Expect: +0.01 to +0.02 over the best single reader [I].
- Cost: none beyond the readers.
- Confidence: high.

**8. Use a CTC reader as a spelling arbiter.**
- Evidence: on Greek and Arabic, 27% and 44.5% of vision-language lexical substitutions were attested words; character perturbation raised their CER by 10 to 42 points vs 2.5 to 6 for classic readers [C: https://arxiv.org/html/2605.27750]. Normalisation examples in 2602.14524 [C].
- Run: where the vision-language reader and the CTC reader differ on a word by two characters or fewer, take the CTC spelling; tune on out-of-fold predictions.
- Expect: +0.003 to +0.008 [I].
- Confidence: medium.

**9. Multi-view self-consistency, capped at 3 views.**
- Evidence: winner used nine views at 21 hours per fold; no ablation published [U].
- Run: 3 test-time views (scale and slight shear), character vote.
- Expect: +0.002 to +0.005 [I].
- Cost: 3 times inference only.
- Confidence: low-medium.

**10. Train longer than 2 epochs with light augmentation.**
- Evidence: Kohút and Hradiš: about 20 epochs near-optimal for small readers; overshooting hurt less than undershooting [C]. Rotation and elastic augmentation gave about 3.6% relative CER; dilation and resize hurt [C: 2508.11499].
- Run: 2 vs 4 vs 6 epochs on the 8B reader with ±1.5 degree rotation and mild elastic warp.
- Expect: +0 to +0.005, with a risk of more normalisation, not less [I].
- Confidence: low.

## Ranked starting points

Licences are quoted as written on the cards I opened.

| Rank | Model | Size | Licence on card | GPU-hours per fold [I] | Why |
|---|---|---|---|---|---|
| 1 | Qwen3-VL-8B-Instruct | 9B | "apache-2.0" [C] | 1 to 2.5 (48 GB; tight on 24 GB) | Winner's reader; tooling exists |
| 2 | LightOnOCR-2-1B-base | 1B | "Apache License 2.0" [C] | 0.5 to 1 | Cheap, different family, fits 24 GB |
| 3 | Qwen3.5-4B | 4B to 5B | "apache-2.0" [C] | 0.7 to 1.5 | Proven as a line reader by Medusa |
| 4 | Qwen3-VL-2B-Instruct | 2B | "apache-2.0" [C] | 0.3 to 0.7 | Cheapest way to debug the pipeline |
| 5 | Qwen3-VL-4B | 4B | card not opened [U] | 0.5 to 1.2 | Middle option |

Qwen2-VL-2B (the organisers' starter) is the fallback; I did not open its card [U].

Awkward or unproven for line-level work:
- **DeepSeek-OCR:** fixed size modes and a fixed prompt; an Unsloth demo only reached 60% CER from 149% on Persian [C: https://unsloth.ai/docs/models/tutorials/deepseek-ocr-how-to-run-and-fine-tune].
- **GOT-OCR2.0:** mode-based API. The Hugging Face card says apache-2.0 [C], but a repository-mirror snippet says research use only [U]; the auditor should resolve this.
- **Florence-2, PaliGemma 2, SmolVLM, InternVL, MiniCPM-V:** I found no fine-tuned handwritten-line results [U].
- **olmOCR:** page-oriented with extra markup output [C: METATR, https://arxiv.org/html/2605.26712].

## What does not work

- **Zero-shot reading.** METATR page-level CERs on handwriting: Qwen3-VL-8B 54.9%, LightOnOCR-2 255.7%, DeepSeek-OCR 234.5% [C]. CHURRO found 36% of zero-shot Qwen2.5-VL-3B outputs had major hallucinations, and none after fine-tuning [C].
- **Script-restricted decoding and text-only post-correction.** Masking raised Greek CER by 62 to 75 points; language-model correction worsened 6 of 7 Arabic systems; contrastive decoding was inconsistent [C: 2605.27750].
- **Bigger models.** No gain past 8B (context pack). A 15.9M-parameter CTC reader beat a 4B model on its own domain [C: Kiessling].
- **Freezing the vision encoder** [C: 2606.24302].
- **4-bit QLoRA when training vision** [C: 2U1 README].
- **Community handwriting fine-tunes as starting points** (Medusa, comma-qwen, LightOnOCR-catmus, churro-3B): all trained on outside handwriting, which the rules forbid [I].

## Sources

**Read in full or substantially**
- Kiessling 2026, PDF pages 1 to 12: https://arxiv.org/pdf/2609.20064
- 2602.14524 (HTML): https://arxiv.org/html/2602.14524
- METATR: https://arxiv.org/html/2605.26712
- "Reading or Guessing?" (first 100k characters): https://arxiv.org/html/2605.27750
- TrOCR medieval ablation: https://arxiv.org/html/2606.24302
- Kohút and Hradiš: https://arxiv.org/html/2503.19546v1
- TrOCR ensemble, 16th-century Latin: https://arxiv.org/html/2508.11499v1
- CHURRO (first 200k of 229k characters; whether training was LoRA or full not found): https://arxiv.org/html/2509.19768v1
- Model cards: Qwen3-VL-8B-Instruct, Qwen3-VL-2B-Instruct, Qwen3.5-4B, LightOnOCR-2-1B and -base, Medusa0.1Line-4B, comma-qwen-3.5-2b-full (https://huggingface.co/wjbmattingly/comma-qwen-3.5-2b-full), LightOnOCR-2-1B-catmus-textualis, GOT-OCR2_0 (https://huggingface.co/stepfun-ai/GOT-OCR2_0)
- Unsloth vision and DeepSeek-OCR docs; 2U1 trainer README

**Abstract or snippet only**
- Uruguayan hallucination study (qualitative only): https://arxiv.org/abs/2607.24077
- ICDAR 2026 medieval handwriting competition (dataset record only, no results found): https://zenodo.org/records/18270331
- Search snippets for Qwen3.5 sizes and the GOT-OCR mirror

**Could not open**
- Medusa system report (bot wall): https://enc.hal.science/hal-05600991
- An entrant's Qwen3-VL-8B adapter card (HTTP 401; a search snippet reported validation CER 0.079 and WER 0.223 [U]; weights not to be used): https://huggingface.co/mlai-dante/road-model
- The Zindi forum write-up itself (not indexed; I relied on the context pack)
- LightOn's fine-tuning notebook (not opened)

No handwriting dataset was opened or queried, and nothing was written to the project.
