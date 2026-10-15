# FND-007 Our validation predicts the Zindi score to within 0.001

Measured 2026-10-15. A late submission (Zindi scores it, unranked) of EXP-025: the EXP-012 judge, with every reader
retrained on all 4,098 lines (EXP-015 Qwen3-VL-8B, EXP-016 Qwen3-VL-2B 5-best, EXP-017 CTC judge) and a length cap
for runaway readings.

| | score | word edits per line | char edits per line |
|---|--:|--:|--:|
| Our 5-fold validation (4,098 lines) | 0.9100 ± 0.0015 | 1.610 | 2.520 |
| Zindi public | 0.9139 | 1.539 | 2.413 |
| Zindi private | 0.9090 | 1.629 | 2.543 |

- Private is 0.001 below validation, inside the ±0.0015 bootstrap spread; public is 0.004 above (a smaller sample).
  Training on all lines instead of four fifths gained little or nothing on this evidence.
- Zindi's "WER Weighted" and "CER Weighted" match our word and character edits per line, so the metric in
  `prepare.py` is right, parts included. Plain mean (0.9100) and sqrt weights (0.9094) are both within noise of private.
- The public score moved 0.005 from private: use validation, not the public score, to compare runs.

So: [[D-002 Adoption rule (provisional)]] measures what Zindi measures; keep adopting on validation alone. For
reference, the closed leaderboard had 50th at 0.924 and 1st at 0.933.

Ledger: `experiments/leaderboard.jsonl` via `exp.py lb`. Submission file `submissions/EXP-025_20261015_0704.csv`
(git-ignored).
