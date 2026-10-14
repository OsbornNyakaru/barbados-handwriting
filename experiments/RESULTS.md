# Results

Validation: 5 random folds (one fixed split), near-identical transcripts kept together, balanced on scan height. Score = 1 - 0.5 * (word edits per line / 12 + character edits per line / 55) on the pooled out-of-fold predictions; ± is a bootstrap over lines. Higher is better.
Adoption: gain over the best >= 0.002 averaged over folds, and better in >= 4/5 folds.

**Best: EXP-001 — 0.1404**

| exp | score | ± | parts | vs best then | adopted | description |
|---|--:|--:|---|---|---|---|
| EXP-001 | 0.1404 | 0.0027 | word_edits 10.5620, char_edits 46.1510, score_sqrt 0.1209, short 0.1189, tall 0.2032 | — | yes | baseline: no reading, every line gets the training transcript closest to the others (medoid) |
| EXP-002 (folds 0 only) | -0.1327 | 0.0073 | word_edits 17.3172, char_edits 45.2239, score_sqrt -0.1492, short -0.1503, tall -0.0821 | -0.2637 (0/1) vs EXP-001 | no | CTC reader trained from scratch (CRNN, no pretrained weights), height 80, 60 epochs, greedy decoding |
| EXP-003 (folds 0 only) | 0.8818 | 0.0043 | word_edits 2.0634, char_edits 3.5423, score_sqrt 0.8802, short 0.9022, tall 0.8235 | +0.7508 (1/1) vs EXP-001 | no | Qwen3-VL-2B-Instruct (Apache 2.0), LoRA on language and vision layers, 3 epochs, greedy; fold 0 only: a smoke test of the vision-language pipeline |
