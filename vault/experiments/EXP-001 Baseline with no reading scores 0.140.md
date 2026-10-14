# EXP-001 Baseline with no reading scores 0.140

Run 2026-10-13 17:03 EAT, harness `5ed07a757d39`, on this machine in 18 seconds.

- What it is: no image is read. Within each fold, every held-out line gets the one training transcript that is
  closest to the other training transcripts on the competition's edit costs (a short witnessing
  formula).
- Score 0.1404 ± 0.0027 on 4,098 held-out lines (folds 0.131, 0.146, 0.146, 0.148, 0.131). With square-root
  weights: 0.1209. Word edits per line 10.56 of 11.28; character edits 46.2 of 62.2.
- By batch: short scans 0.119, tall scans 0.203. Tall lines have shorter transcripts, so a wrong answer costs less
  there; a reader's tall-batch score must be read against that.
- Reference points: an empty prediction scores -0.036; the leaderboard benchmark 0.0425; rank 50 0.9239; the
  winner 0.9331.

**What it shows:** the pipeline runs end to end (folds, scorer, ledger, submission check). The floor is 0.14, so
almost all of the score has to come from reading. A wrong answer to every line still scores above zero, which is
why scores near 0.90 mean roughly 1.8 wrong words per line (scaling the winner's error columns), not "10% wrong".
