# EXP-009 The CTC judge over two Qwen3-VL-8B runs scores 0.904

2026-10-14. CPU, from saved runs. For each line: candidates are every saved Qwen3-VL-8B reading (EXP-006; two runners
trained every fold, so two readings per line), choose by CTC log p under EXP-005's held-out log-probabilities plus
0.3 x the fold's 6-letter character language model. No setting tuned on these lines. 80 s on the Mac.

- **0.9035 ± 0.0016** (word edits 1.71, character edits 2.79 per line; short 0.922, tall 0.850).
- Against EXP-006 (Qwen3-VL-8B, latest run per fold, 0.8954): +0.0081, better in 5 of 5 folds. Adopted.
- Path so far: no reading 0.140 → CTC 0.832 → CTC + LM 0.866 → Qwen-8B 0.895 → judge 0.904.
- Gap to the leaderboard: 50th 0.924, 1st 0.933.
- Supports H-005 and [[FND-006 The CTC reader works best as a judge between vision-language readings]].
