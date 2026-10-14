# Results

Validation: 5 random folds (one fixed split), near-identical transcripts kept together, balanced on scan height. Score = 1 - 0.5 * (word edits per line / 12 + character edits per line / 55) on the pooled out-of-fold predictions; ± is a bootstrap over lines. Higher is better.
Adoption: gain over the best >= 0.002 averaged over folds, and better in >= 4/5 folds.

**Best: EXP-001 — 0.1404**

| exp | score | ± | parts | vs best then | adopted | description |
|---|--:|--:|---|---|---|---|
| EXP-001 | 0.1404 | 0.0027 | word_edits 10.5620, char_edits 46.1510, score_sqrt 0.1209, short 0.1189, tall 0.2032 | — | yes | baseline: no reading, every line gets the training transcript closest to the others (medoid) |
