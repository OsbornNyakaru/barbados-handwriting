# Results

Validation: 5 random folds (one fixed split), near-identical transcripts kept together, balanced on scan height. Score = 1 - 0.5 * (word edits per line / 12 + character edits per line / 55) on the pooled out-of-fold predictions; ± is a bootstrap over lines. Higher is better.
Adoption: gain over the best >= 0.002 averaged over folds, and better in >= 4/5 folds.

**Best: EXP-012 — 0.9100**

| exp | score | ± | parts | vs best then | adopted | description |
|---|--:|--:|---|---|---|---|
| EXP-001 | 0.1404 | 0.0027 | word_edits 10.5620, char_edits 46.1510, score_sqrt 0.1209, short 0.1189, tall 0.2032 | — | yes | baseline: no reading, every line gets the training transcript closest to the others (medoid) |
| EXP-002 (folds 0 only) | -0.1327 | 0.0073 | word_edits 17.3172, char_edits 45.2239, score_sqrt -0.1492, short -0.1503, tall -0.0821 | -0.2637 (0/1) vs EXP-001 | no | CTC reader trained from scratch (CRNN, no pretrained weights), height 80, 60 epochs, greedy decoding |
| EXP-003 (folds 0 only) | 0.8818 | 0.0043 | word_edits 2.0634, char_edits 3.5423, score_sqrt 0.8802, short 0.9022, tall 0.8235 | +0.7508 (1/1) vs EXP-001 | no | Qwen3-VL-2B-Instruct (Apache 2.0), LoRA on language and vision layers, 3 epochs, greedy; fold 0 only: a smoke test of the vision-language pipeline |
| EXP-004 (folds 0 only) | 0.7460 | 0.0057 | word_edits 4.3843, char_edits 7.8420, score_sqrt 0.7425, short 0.7708, tall 0.6751 | +0.6150 (1/1) vs EXP-001 | no | Diagnostic for EXP-002's stuck training: fold 0, 30 epochs, amp=false |
| EXP-005 | 0.8318 | 0.0021 | word_edits 2.9837, char_edits 4.8267, score_sqrt 0.8298, short 0.8529, tall 0.7701 | +0.6914 (5/5) vs EXP-001 | yes | CTC reader from scratch with a LayerNorm and a shortcut around the LSTM (fixes EXP-002's stalled training), 5 folds |
| EXP-007 | 0.8655 | 0.0019 | word_edits 2.3387, char_edits 4.0720, score_sqrt 0.8644, short 0.8861, tall 0.8052 | +0.0338 (5/5) vs EXP-005 | yes | Re-decode EXP-005's held-out log-probabilities: beam 10 with a 6-letter character language model (weight 0.3) built from each fold's training transcripts |
| EXP-006 | 0.8954 | 0.0020 | word_edits 1.8287, char_edits 3.1242, score_sqrt 0.8947, short 0.9147, tall 0.8389 | +0.0299 (5/5) vs EXP-007 | yes | Qwen3-VL-8B-Instruct (Apache 2.0), LoRA on language and vision layers, 3 epochs, greedy; all 5 folds |
| EXP-009 | 0.9035 | 0.0016 | word_edits 1.7067, char_edits 2.7906, score_sqrt 0.9028, short 0.9217, tall 0.8501 | +0.0081 (5/5) vs EXP-006 | yes | Judge: per line, choose among every saved Qwen3-VL-8B reading (EXP-006, two runs on most folds) by CTC log p (EXP-005) + 0.3 x character LM |
| EXP-008 | 0.8844 | 0.0035 | word_edits 1.9966, char_edits 3.5666, score_sqrt 0.8832, short 0.9089, tall 0.8126 | -0.0191 (0/5) vs EXP-009 | no | Qwen3-VL-2B, all folds, beam search keeping 5 readings per line as candidates for combine 'choose' |
| EXP-012 | 0.9100 | 0.0015 | word_edits 1.6096, char_edits 2.5203, score_sqrt 0.9094, short 0.9278, tall 0.8579 | +0.0065 (5/5) vs EXP-009 | yes | Judge over every saved Qwen3-VL-8B (EXP-006) and Qwen3-VL-2B (EXP-008, readings and 5-best lists) reading, by CTC log p (EXP-005) + 0.3 x character LM |
| EXP-010 (folds 0 only) | 0.8125 | 0.0051 | word_edits 3.3060, char_edits 5.4751, score_sqrt 0.8105, short 0.8359, tall 0.7453 | -0.0957 (0/1) vs EXP-012 | no | CTC reader from scratch with rnn_skip at height 128 (was 80): more pixels per letter, a sharper judge |
