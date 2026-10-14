# FND-004 The two height batches are two kinds of document at two scales

- Heights split cleanly: only 129 of 5,472 images are 111 to 199 px high, and none between 168 and 184 px.
- Short scans (3,055 training lines): median 56 px high, 16.7 px of width per character.
  Tall scans (1,043 training lines): median 352 px high, 76.5 px per character. So tall scans are 4.6 times larger
  per letter but 6.3 times taller: a tall crop takes in more of the lines above and below.
- By eye (about 40 training lines read against their labels, as reduced thumbnails, charts 03, 04, 08, 09): short scans are mostly a clean,
  even, later copyist's hand on pale paper; tall scans are older, cramped hands on stained and bled-through paper.
  This is our reading of a small sample, not a measurement. It would explain the data tab's "18th and 19th
  century" beside 1600s dates: later copies of 1600s records.
- Both batches are in test at the same rate ([[FND-001 Test lines are a random quarter of the same scans as training lines]]).

**Consequences:** balance folds on the batch and report every score for each batch as well as overall. Shrinking
tall scans by about 4.6 puts both batches at the same letter size, which is the first input-size experiment.
([[H-002 Bring both batches to one letter size]])

Charts: `docs/eda/01_...`, `04_tall_lines_train_and_test.png`, `08_tall_training_lines_with_labels.png`.
Source: `notebooks/eda.py` section `q1, q3, look`; numbers in `docs/eda/numbers.txt`. Checked 2026-10-13.
