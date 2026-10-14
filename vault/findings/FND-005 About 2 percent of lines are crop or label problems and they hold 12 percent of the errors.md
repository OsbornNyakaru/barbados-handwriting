# FND-005 About 2 percent of lines are crop or label problems, and they hold 12 percent of the errors

EDA questions 4 to 6, answered 2026-10-14 from EXP-005's held-out readings (CTC from scratch, 0.8318, all 4,098 lines;
`notebooks/errors.py EXP-005`; the line list is in the git-ignored `experiments/runs/EXP-005/worst_lines.md`).

**Q4 label noise.** 87 lines (2.1%) have more than 30% of characters wrong; they hold 11.5% of all character edits;
47 of them are tall scans (tall is 25% of lines). 22 lines exceed 50%. Read by eye, the worst 15 are mostly not hard
handwriting: the reader writes a fluent line that is a different line from the label (wrong line, or a multi-line
crop labelled with another line), "short" crops 88 to 117 px high that hold more than one line, and one blank
reading. This matches the forum (about 1 line in 70 paired with another line's text, thread 34861). Not dropped:
held-out lines are never dropped from scoring, and dropping training rows needs a rule in code (a later experiment).

**Q5 score noise.** Bootstrap spread 0.0047 at 820 lines (an entrant reported 0.0048) and 0.0021 at 4,098. Fold
scores 0.824 to 0.845 (sd 0.0079): folds differ in difficulty, which paired comparison cancels. Training the same
spec twice moved fold scores by 0.003 and 0.000 (two runners repeated EXP-005's folds by accident). So the
adoption rule of +0.002 averaged over folds and better in 4 of 5 stays ([[D-002 Adoption rule (provisional)]]).

**Q6 error types** (share of 12,227 word edits): spelling with more than one letter wrong 36%, one letter wrong 35%,
missing word 11%, casing 6%, extra word 5%, punctuation 3%, markup (^ ~ : &) 2%, digits 1%. Long lines score worse
(0.79 for 15 words or more against 0.85 for 8 or fewer). Exact lines: 12% short, 3.5% tall.

**What it changes:** most loss is ordinary misreading, so better readers and the vote come first. Then two cheap
tests: a rule that drops training lines whose reading disagrees wildly with the label (in-fold only), and per-batch
handling of tall and odd-height crops.
