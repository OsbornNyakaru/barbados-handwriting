# EXP-012 Adding Qwen3-VL-2B readings lifts the judge to 0.910

2026-10-14. CPU, from saved runs. Same judge as [[EXP-009 The CTC judge over two Qwen3-VL-8B runs scores 0.904]],
with the candidates widened to every saved Qwen3-VL-2B reading (EXP-008: two runs per fold plus 5-best lists).

- **0.9100 ± 0.0015** (word edits 1.61, character edits 2.52 per line; short 0.928, tall 0.858).
- Against EXP-009: +0.0065, better in 5 of 5 folds. Adopted.
- EXP-008 alone (Qwen3-VL-2B, latest run, greedy): 0.8844. The 5-best lists add almost nothing (FND-006); the second
  separately trained run does.
- Gap to the leaderboard: 50th 0.924, 1st 0.933.
