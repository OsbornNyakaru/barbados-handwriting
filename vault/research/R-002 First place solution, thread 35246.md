# R-002 First place solution (thread 35246)

Read 2026-10-15 from the Zindi discussion (author Brainiac, posted 7 Oct 2026; 8 replies, none technical except
an unanswered question on how the vote weights were found). Paraphrased; ideas only, never their code, weights or
files ([[RULES]]). Licences for every model the winner names were cleared in [[R-001 Which pretrained models to start from]].

## What the winner built

- **Validation**: 5 folds, twins kept together, best of 8 candidate splits by fold similarity. Every reader trained
  5 times; all 4,098 out-of-fold readings scored. Public LB only a secondary check. Kraken looked useless on fold 0
  and was the most useful reader on all 5.
- **Five families, eight variants, each x 5 folds (40 models)**: Qwen3-VL (8B), Qwen3.5, LightOnOCR-2 (each one
  strip + one "nine views"), ConvNeXt-large CTC, Kraken CTC.
- **Nine views** per line: whole line, six overlapping windows cut at predicted word boundaries, two filtered copies.
  Better alone, and right on different words (large, faded scans).
- **Word boundaries**: a ConvNeXt CTC reader force-aligns training lines to find where words start and end; seven
  ConvNeXt models predict boundaries on test images (word counters at 128 and 192 px, five boundary finders per
  4-px column).
- **Word-span crops** (1 to 5 whole words, full height, the matching substring as label), seven kinds aimed at known
  error types (carets and punctuation, capitals, similar spellings like final -e, numbers and Latin, line ends,
  random spans, five-word spans). Extra training examples for Qwen, LightOnOCR and Kraken.
- **Training**: VLMs 2 epochs, effective batch 8, cosine to the last step; Qwen3-VL LoRA lr 2e-4 on vision and
  language, mergers fully trained at 1e-5 (Unsloth); LightOnOCR full fine-tune 2e-5; ConvNeXt 30 epochs; Kraken 12.
  Vision-side LoRA mattered most. 8B with nine views matched 32B. Fully deterministic runs.
- **Decoding**: VLM beam 4; a reading that hits the length limit and repeats is collapsed, unfinished ones re-read
  greedily. ConvNeXt greedy. Kraken beam 64 with a character 6-gram KenLM from training text only (weight 0.8,
  length bonus 2), keeping its 25 best readings.
- **Vote** (ROVER style), weights summing to 47 (Qwen3-VL 5+5, Qwen3.5 5+5, LightOnOCR 6+6, ConvNeXt 3, Kraken 12
  shared across its 25 readings), set from each reader's contribution on held-out readings:
  1. align all readings letter by letter, vote on each difference, build candidate lines;
  2. pick the candidate with the lowest weighted distance to all readings, distance = word edits/12 + char edits/55
     (the metric itself: minimum expected loss);
  3. swap a word only if at least two families back the alternative. Then fixed whitespace and rewrite rules.
- **Test**: every fold model of every variant reads the test lines; a variant's weight is split across its 5 folds.
- **Metric**: the winner states the edits are averaged with square-root-of-length weights (our `score_sqrt`).
- **Cost**: 1 GPU-hour (ConvNeXt) to about 21 GPU-hours (Qwen3-VL nine views) per fold on an RTX A6000.

## What failed for the winner

Bigger models (Qwen3-VL 32B, InternVL3.5-38B, Gemma-4 31B); windows that ignore word boundaries; zooming everything;
GRPO reinforcement learning (reward up, held-out down); learned re-rankers; pseudo-labels; rare-word specialists
(they "fixed" correct old spellings); prompt rewrites; word-level LMs; dictionary snapping. The KenLM helped Kraken
only, not VLM decoding or re-ranking.

## Against ours (EXP-025: validation 0.9100, Zindi private 0.9090; 1st place 0.933)

| | winner | ours |
|---|---|---|
| Families | 5 (8 variants) | 2 (Qwen 8B, 2B; CRNN CTC) |
| Combining | weighted letter-level vote, minimum expected metric loss | judge picks one whole Qwen reading per line |
| Letter-by-letter reader | Kraken + 6-gram LM, 25-best, the heaviest voter | CRNN + 6-gram LM, judge only, never a candidate |
| Views | strip + nine views | strip |
| Extra training examples | word-span crops | none |
| VLM recipe | 2 epochs, lr 2e-4, mergers trained, beam 4, repeat collapse | 3 epochs, lr 1e-4, LoRA everywhere, greedy, no collapse (one runaway on test) |
| Test readings | 5 fold models each | one model retrained on all lines |

The judge can only choose a whole line some Qwen produced; a vote can build a line no single reader wrote, mixing
the right words from each. That and the weak tall batch (ours 0.858 vs short 0.928) are the two biggest gaps.
