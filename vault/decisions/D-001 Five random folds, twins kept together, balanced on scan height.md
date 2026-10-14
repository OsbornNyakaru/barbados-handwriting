# D-001 Five random folds, twins kept together, balanced on scan height

Decided 2026-10-13 from EDA questions 1 to 3. Status: frozen 2026-10-13 in `autoresearch/prepare.py` (seed 20261013, fold hash `6d2b14640424`, folds of 804, 834, 823, 822 and 815 lines).

## The scheme

- 5 folds over the 4,098 training lines, split at line level with a fixed seed.
- Lines whose normalised text (lower case, single spaces) is 85% alike or more go in the same fold: 167 lines in
  40 groups. `groups()` in `notebooks/eda.py` builds them.
- Folds are balanced on the height batch (over 150 px or not). Trial split: folds of 805 to 842 lines, tall share
  22.8% to 27.0%, mean length 61.8 to 63.0 characters.
- Every model is scored on all 4,098 out-of-fold predictions, overall and per batch, with both score variants
  ([[F-002 The score formula]]).

## Why

- The test set is a random quarter of the same scans ([[FND-001 Test lines are a random quarter of the same scans as training lines]]),
  so random folds mirror it. Holding out pages or writers would not.
- Repeats are rare and are text only, never the same image ([[FND-002 Only 4 percent of training lines repeat another line and no two images are copies]]).
  Keeping twins together is the cautious choice: if the organisers kept repeats out of test, a fold with twins
  split across it would score too high.
- The winner used the same design (thread 35246) and found that settings chosen on one fold did not carry to the
  others, so decisions use all five folds.

## Not yet known

- Repeats and adoption threshold: these need the score's noise, which needs a trained reader (EDA question 5). An
  entrant reports a standard deviation of 0.0048 on 820 lines.
- Whether to drop mislabelled lines from scoring (question 4).
