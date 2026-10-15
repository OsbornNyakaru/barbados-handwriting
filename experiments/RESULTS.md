# Results

Validation: 5 random folds (one fixed split), near-identical transcripts kept together, balanced on scan height. Score = 1 - 0.5 * (word edits per line / 12 + character edits per line / 55) on the pooled out-of-fold predictions; ± is a bootstrap over lines. Higher is better.
Adoption: gain over the best >= 0.002 averaged over folds, and better in >= 4/5 folds.

**Best: EXP-033 — 0.9170**

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
| EXP-013 | 0.9110 | 0.0015 | word_edits 1.5898, char_edits 2.4995, score_sqrt 0.9103, short 0.9286, tall 0.8596 | +0.0010 (4/5) vs EXP-012 | no | Two CTC judges: EXP-005 (height 80) and EXP-010 (height 128) log p added, over the same Qwen 8B and 2B candidates |
| EXP-011 | 0.8813 | 0.0029 | word_edits 2.0432, char_edits 3.6918, score_sqrt 0.8802, short 0.9045, tall 0.8133 | -0.0287 (0/5) vs EXP-012 | no | Qwen3-VL-4B-Instruct (Apache 2.0), LoRA on language and vision, 3 epochs, greedy, all folds: a third separately trained source of candidate readings (nbest filled with its default 1) |
| EXP-014 | 0.9114 | 0.0015 | word_edits 1.5876, char_edits 2.4739, score_sqrt 0.9107, short 0.9287, tall 0.8607 | +0.0013 (5/5) vs EXP-012 | no | Judge over Qwen 8B, 2B and 4B readings |
| EXP-020 | 0.9098 | 0.0015 | word_edits 1.6137, char_edits 2.5237, score_sqrt 0.9092, short 0.9277, tall 0.8575 | -0.0002 (1/5) vs EXP-012 | no | Judge weight on the character LM: 0.3 -> 0.1 |
| EXP-021 | 0.9099 | 0.0015 | word_edits 1.6125, char_edits 2.5200, score_sqrt 0.9092, short 0.9278, tall 0.8576 | -0.0001 (2/5) vs EXP-012 | no | Judge weight on the character LM: 0.3 -> 0.2 |
| EXP-022 | 0.9098 | 0.0015 | word_edits 1.6142, char_edits 2.5276, score_sqrt 0.9091, short 0.9277, tall 0.8573 | -0.0003 (1/5) vs EXP-012 | no | Judge weight on the character LM: 0.3 -> 0.4 |
| EXP-023 | 0.9092 | 0.0015 | word_edits 1.6227, char_edits 2.5508, score_sqrt 0.9085, short 0.9272, tall 0.8564 | -0.0008 (0/5) vs EXP-012 | no | Judge weight on the character LM: 0.3 -> 0.6 |
| EXP-024 | 0.9100 | 0.0015 | word_edits 1.6096, char_edits 2.5203, score_sqrt 0.9094, short 0.9278, tall 0.8579 | +0.0000 (0/5) vs EXP-012 | no | EXP-012 judge, plus the same judge on the test lines (readers trained on all lines: EXP-015 to EXP-017), for the submission |
| EXP-025 | 0.9100 | 0.0015 | word_edits 1.6096, char_edits 2.5203, score_sqrt 0.9094, short 0.9278, tall 0.8579 | +0.0000 (0/5) vs EXP-012 | no | Submission judge: drop readings longer than any training line (runaway repeats); none left -> CTC beam reading |
| EXP-026 | 0.9145 | 0.0015 | word_edits 1.5234, char_edits 2.4239, score_sqrt 0.9138, short 0.9318, tall 0.8637 | +0.0045 (5/5) vs EXP-012 | yes | Weighted vote (first place, R-002): minimum expected metric loss over every reading, then word-span swaps; Qwen 8B 2, 2B 1, 2B 5-best 1, 4B 1, CTC beam 2 |
| EXP-027 | 0.9136 | 0.0015 | word_edits 1.5347, char_edits 2.4754, score_sqrt 0.9128, short 0.9308, tall 0.8630 | -0.0009 (0/5) vs EXP-026 | no | Vote: add the CTC reader's 10 best readings as a voter (weight 1); first place gave Kraken's 25-best the largest weight |
| EXP-028 | 0.9113 | 0.0015 | word_edits 1.5717, char_edits 2.5578, score_sqrt 0.9105, short 0.9289, tall 0.8596 | -0.0032 (0/5) vs EXP-026 | no | Vote: CTC beam reading weight 2 -> 3 |
| EXP-029 | 0.9132 | 0.0015 | word_edits 1.5425, char_edits 2.4773, score_sqrt 0.9126, short 0.9307, tall 0.8619 | -0.0013 (0/5) vs EXP-026 | no | Vote: Qwen 8B weight 2 -> 3 |
| EXP-018 | 0.8441 | 0.0020 | word_edits 2.7635, char_edits 4.4802, score_sqrt 0.8424, short 0.8640, tall 0.7860 | -0.0704 (0/5) vs EXP-026 | no | CTC reader (the judge): 60 -> 120 epochs |
| EXP-032 | 0.9150 | 0.0015 | word_edits 1.5156, char_edits 2.4051, score_sqrt 0.9143, short 0.9322, tall 0.8645 | +0.0005 (4/5) vs EXP-026 | no | Vote with the 120-epoch CTC reader (EXP-018) as the CTC voter (ctc_nbest is a default fill, unused) |
| EXP-033 | 0.9170 | 0.0015 | word_edits 1.4815, char_edits 2.3407, score_sqrt 0.9163, short 0.9337, tall 0.8679 | +0.0025 (5/5) vs EXP-026 | yes | Vote: add the 60-epoch CTC reader (EXP-005) as a second CTC voter, weight 1 |
| EXP-034 | 0.9121 | 0.0015 | word_edits 1.5625, char_edits 2.5115, score_sqrt 0.9113, short 0.9297, tall 0.8603 | -0.0049 (0/5) vs EXP-033 | no | Vote: stronger CTC voter (EXP-018) weight 2 -> 3 |
| EXP-035 | 0.9176 | 0.0014 | word_edits 1.4693, char_edits 2.3353, score_sqrt 0.9169, short 0.9343, tall 0.8684 | +0.0006 (4/5) vs EXP-033 | no | Vote: add the height-128 CTC reader (EXP-010) as a third CTC voter, weight 1 |

## Leaderboard

| # | file | public | private | cv | note |
|---|---|--:|--:|--:|---|
| 1 | EXP-025_20261015_0704.csv | 0.913924693 | 0.909024825 | 0.91 | late submission 2026-10-15, unranked. Public WER 1.539 CER 2.413; private WER 1.629 CER 2.543 (ours: word 1.610, char 2.520) |
