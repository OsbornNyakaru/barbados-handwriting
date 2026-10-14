# D-002 Adoption rule (provisional)

Decided 2026-10-13 when the harness was built. Provisional until a real reader has been trained.

## The rule

A run replaces the best when, on the same folds, its score is higher by 0.002 or more averaged over the five folds
and it is better in at least 4 of the 5 folds. Set in `scripts/exp.py` (`ADOPT_GAIN`, `ADOPT_WINS`).

## Why these numbers

- There is one fixed split, so there are no repeats to count. The five folds are the paired units, and a bootstrap
  over the 4,098 held-out lines gives each run's spread (the `±` column).
- An entrant measured a standard deviation of 0.0048 on 820 held-out lines at a score near 0.90 (thread 34861).
  Scaled to 4,098 lines that is about 0.0021. 0.002 is one such standard deviation.
- The baseline cannot set this: it predicts the same string for every line, so its spread (0.0027) says nothing
  about a reader's.
- The winner saw settings that helped one fold and not the others, hence the 4-of-5 condition.

## To revisit

After the first reader (EDA question 5): replace 0.0021 with the measured bootstrap spread of the reader, and
measure the spread of a paired difference between two seeds of the same recipe. Paired differences are usually
much tighter than single scores, so the bar may come down.
