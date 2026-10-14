# Small letter-by-letter readers: cheapest first model (research report, 13 Oct 2026)

Labels: [C] cited from a source I opened, [S] search snippet only, [I] my inference, [U] unconfirmed.

**Headline:** no paper I found reports a result at exactly ~4,000 lines of early-modern English. The evidence is nearby sizes (1.7k to 10k lines), so every score forecast below is inference.

## Score conversion (used throughout)

- Word edits per line = WER x 11.3; character edits per line = CER x 62.
- Score = 1 - 0.5 x (11.3 x WER / 12 + 62 x CER / 55) = **1 - 0.471 x WER - 0.564 x CER**.
- Assumption [I]: a CTC reader without a language model has WER of about 3.7 x CER (IAM 15.89 / 4.62 = 3.4 in Retsinas; READ2016 16.5 / 3.9 = 4.2 in HTR-VT). Then score is roughly 1 - 2.31 x CER.
- So 0.90 needs about 4.3% CER, and 0.933 (the winner) about 2.9% CER.
- Check: CER 4%, WER 16% gives 1 - 0.0754 - 0.0226 = 0.902.

## Idea cards

**1. Kraken fine-tuned from CATMuS Medieval, versus Kraken from scratch.**
- Evidence [C]: on 1,666 training lines of a 13th-century hand, Kraken fine-tuned from CATMuS v1.6.0 reached 7.9% CER / 31.2% WER, against 8.0 / 31.9 for a fully fine-tuned TrOCR-large with 333.9M trainable parameters (arXiv 2606.24302). The Kraken row was a reference system, not statistically tested.
- Evidence [S]: at 350 lines, transfer gave 3.3% CER against 18.2% from scratch (arXiv 1804.01527); Kraken's docs recommend fine-tuning when a nearby model exists. I found no Kraken learning curve in lines.
- Licence [C]: the Zenodo records state "Creative Commons Attribution 4.0 International" (v1.5.0 at zenodo.org/records/12743230; v1.6.2 record, file catmus-medieval-1.6.0.mlmodel, at zenodo.org/records/15030337), checked 13 Oct 2026. Kraken itself is Apache 2.0 [S].
- Run: 5 folds each way. The base alphabet differs (NFD normalisation; our `^` and capitals), so the output layer must be resized.
- Expect [I]: fine-tuned beats scratch by 10 to 30% relative CER, mostly on the hard tall batch.
- Cost [I]: under 1 GPU-hour per fold; feasible on a T4.
- Confidence: medium.

**2. Retsinas-style CNN + BiLSTM CTC from scratch.**
- Evidence [C]: Retsinas et al. 2024 (arXiv 2404.11339), IAM lines, no pretraining, no language model, greedy decoding: 4.62% CER / 15.89% WER. Ablation on test CER: no tricks 5.93; padding instead of resizing 5.54; column max-pooling 5.28; auxiliary CTC shortcut 5.11; all three 4.62.
- Run: 128 x 1024 input, median-value padding, auxiliary CTC loss at weight 0.1, 240 epochs.
- Expect [I]: 0.87 to 0.90 alone.
- Cost [U]: the paper gives no training time; I estimate 2 to 4 hours per fold on a 16 GB GPU.
- Confidence: medium.

**3. ConvNeXt V1 with ImageNet weights + CTC (the winner's fourth reader).**
- Evidence: I found no clean ablation of ImageNet versus random initialisation for line-level CTC [U]. Nearest [S]: AttentionHTR, CTC head, IAM validation CER 7.12 with random init against 5.32 with scene-text transfer. HTR-VT [C]: a ResNet-18 stem beats ResNet-50 (4.7 against 4.9 CER), so bigger backbones do not help by themselves.
- Licence [C]: `timm/convnext_large.fb_in22k_ft_in1k` is tagged "apache-2.0". **`timm/convnextv2_large.fcmae_ft_in22k_in1k` is tagged "cc-by-nc-4.0", so ConvNeXt V2 is forbidden.** Both checked 13 Oct 2026.
- Run: pretrained against random init on one fold first.
- Expect [I]: pretrained wins by 10 to 25% relative CER.
- Cost: 1 hour per fold on an A6000 (context pack).
- Confidence: medium that it works, low on the size.

**4. 6-gram character language model in CTC beam search, built from fold-training transcripts only.**
- Evidence [C]: Tarride et al. 2024 (arXiv 2404.18722), PyLaia, 6-gram with Kneser-Ney smoothing, weight 1.5:

| dataset | training lines | CER without / with | WER without / with |
|---|--:|--:|--:|
| IAM | 6,482 | 8.44 / 7.50 | 24.51 / 20.98 |
| RIMES | 10,188 | 4.55 / 3.82 | 14.39 / 10.53 |
| POPP | 3,835 | 16.49 / 16.09 | 36.26 / 34.52 |
| Esposalles | 2,328 | 0.76 / 1.04 | 2.62 / 3.38 (it hurt) |

- The authors say the model hurts when CER is already below about 2%. Companion paper (arXiv 2404.19317) [C]: order 6 was best for characters, weight grid 0 to 5 in steps of 0.5, and word-level models hurt because of unseen words.
- Run: grid over order 4 to 7 and weight 0 to 3, tuned separately for the short and tall batches, since the clean batch may sit in the "hurts" zone.
- Expect [I]: WER down 5 to 15% relative on the tall batch, about zero or negative on the short batch; +0.003 to +0.010 score overall.
- Cost: CPU only, about 10 times slower decoding [C].
- Confidence: medium-high.

**5. StackMix using training transcripts only, plus blots.**
- Evidence [C]: Shonenkov et al. 2021 (arXiv 2108.11667), ResNet-34 stem + BiLSTM, 128 x 2048. Bentham CER 2.99 base, 2.08 with StackMix, 1.73 with both; Digital Peter (6,237 lines) 4.44 to 2.50; IAM 5.80 to 3.77. Word boundaries come from the CTC outputs of a first model.
- Caveat [C]: they generated new lines from external text corpora of 180k to 3M lines. We can only reshuffle our own words, so expect less.
- Run: train reader 2, extract boundaries, recombine word crops within the same scan batch, apply with probability 0.5 to 0.8.
- Expect [I]: 5 to 20% relative CER.
- Cost: their best runs used 1,000 epochs at 21 images per second on one V100 [C]. That is too long; cap at about 200 epochs.
- Confidence: low-medium.

**6. Mild geometric augmentation only.**
- Evidence [C]: TrOCR ablation on READ2016, CER 5.92 without augmentation against 5.16 with it (significant); on the 1,666-line set 8.4 against 8.0 (not significant). [S]: a 2025 study found mild elastic and rotation best (1.86 against 1.93) and heavy dilation or resizing worse.
- Run: small rotation, shear and elastic distortion, with and without blots.
- Expect: +0.002 to +0.006. Cost: free. Confidence: medium.

**7. Normalise scale by character width, and cut a centre band from tall lines.**
- [I]: resizing both batches to 120 px height gives 36 px per character (short) and 26 px (tall), so one height roughly aligns them.
- [I]: at native 56 px height and a width step of 8, a short line yields about 137 frames for up to 120 characters. That is too tight for CTC; upsample or use a step of 4.
- [C]: padding beat resizing in Retsinas (4.85 to 4.62 test CER).
- Run: heights 64, 96 and 128; centre crops of 50%, 65% and 100% of height on the tall batch.
- Expect [I]: the largest single gain on the tall batch. Confidence: medium (no direct paper evidence on centre-band cropping).

**8. HTR-VT (CNN stem + 4-layer ViT encoder + CTC, with sharpness-aware minimisation and span masking) as a diverse voter.**
- Evidence [C]: Li et al. 2024 (arXiv 2409.08573), no pretraining. READ2016 (8,349 lines) 3.9 / 16.5; IAM 4.7 / 14.9. Without the CNN stem IAM validation CER is 20.7; a plain ViT gets 32.4 on IAM test.
- Cost [C]: 16.3 hours on an RTX 4090 for 100k iterations. About 20k iterations should do for our size [I], so 3 to 4 hours per fold.
- Confidence: medium.

**9. TrOCR-large full fine-tune, as a comparison only.**
- Evidence [C]: no better than fine-tuned Kraken at 1,666 lines (card 1). Freezing 6 or more encoder layers hurts (9.7 against 8.0). It used 50 epochs at a peak learning rate of 5.5e-6.
- Licence: the unilm repository LICENSE is "The MIT License (MIT)" [C], but the Hugging Face card text I retrieved shows no licence tag, so the weights' licence is [U]. The handwritten checkpoint was fine-tuned on IAM [C]; ask whether that is acceptable.
- Low priority. Confidence: medium.

## Recommended first reader

**Kraken fine-tuned from CATMuS Medieval, then card 4 on its outputs.**

- It is the cheapest option, with a commercial-use licence I opened.
- It is in the organisers' starter pack, and Kraken was the winner's top-weighted voter (context pack).
- It matched a 334M-parameter TrOCR at 1.7k lines.
- It cannot modernise spellings, because it has no word-level prior [I].
- I did not confirm that stock Kraken supports n-gram decoding; plan to export its per-frame outputs to an external beam search [U].

Expected score [I]:

| case | short batch CER (75%) | tall batch CER (25%) | blended CER | WER at 3.7x | score |
|---|--:|--:|--:|--:|--:|
| central | 2.5% | 12% | 4.9% | 18.0% | 1 - 0.0848 - 0.0276 = 0.888 |
| central + language model (CER -8%, WER -12%) | | | 4.5% | 15.8% | 0.900 |

Range: **0.87 to 0.91**. The tall-batch CER is the main unknown.

## What the evidence says does not work at this size

- Plain ViT or encoder-decoder transformers from scratch: 32.4% CER on IAM; adding a transformer decoder took 4.7 to 7.7-8.1 [C].
- Word-level language models and dictionaries: unseen words make them harmful [C]. This matches our 10% unseen tokens.
- A character language model on an already clean hand (below 2% CER) [C].
- A language model on an attention decoder: neutral for DAN on RIMES lines (WER 10.69 to 10.83) [C]. A vision-language model already carries a stronger, modern-English prior [I]; I found no source testing n-gram fusion on one [U].
- Heavy distortion augmentation [S]; freezing most of a pretrained encoder [C]; larger CNN backbones without pretraining [C].
- DTrOCR: no official weights, and the unofficial MIT-licensed reimplementation has none either [S].
- PP-OCRv5/v6 fine-tuning: no CER result at this size found [U]; one study reports exact-match 37.5% to 50% on Sino-Vietnamese manuscripts [S].

## Sources

**Read in full (HTML):**
- Retsinas et al. 2024: https://arxiv.org/html/2404.11339v1
- HTR-VT: https://arxiv.org/html/2409.08573v1
- StackMix: https://ar5iv.labs.arxiv.org/html/2108.11667
- PyLaia language models: https://arxiv.org/html/2404.18722v1
- N-gram models for PyLaia and DAN: https://ar5iv.labs.arxiv.org/html/2404.19317
- TrOCR medieval ablation: https://arxiv.org/html/2606.24302
- CATMuS records: https://zenodo.org/records/12743230 and https://zenodo.org/records/15030337/latest
- timm cards: https://huggingface.co/timm/convnext_large.fb_in22k_ft_in1k and https://huggingface.co/timm/convnextv2_large.fcmae_ft_in22k_in1k
- unilm licence: https://github.com/microsoft/unilm/blob/master/LICENSE
- TrOCR card (no licence visible): https://huggingface.co/microsoft/trocr-large-handwritten

**Abstract or snippet only:**
- https://arxiv.org/abs/2503.19546 (16 to 256 lines give 10 to 40% relative CER gain)
- https://arxiv.org/abs/2203.11008
- https://arxiv.org/pdf/2201.09390 (AttentionHTR)
- https://arxiv.org/pdf/1804.01527
- https://arxiv.org/pdf/2012.03868
- https://arxiv.org/pdf/2508.11499
- https://arxiv.org/abs/2308.15996 (DTrOCR)
- https://arxiv.org/pdf/2507.05595 (PaddleOCR 3.0)
- https://arxiv.org/abs/2510.04003
- https://kraken.re/7.1/user_guide/training_recognition.html

**Could not open:**
- https://kraken.re/main/ketos.html and https://kraken.re/main/training/rectrain.html (404), so Kraken's default line height and architecture are unverified.
- PDFs of arXiv 2108.11667 and 2203.11008 (binary only).
- The Zindi forum write-up (not found by search).

Not found at all: any PyLaia, DAN or PP-OCR result at about 4k historical lines beyond POPP above; any measured CPU training time.
