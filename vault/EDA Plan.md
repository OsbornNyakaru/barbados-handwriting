# EDA Plan

Written 2026-10-13, before any modelling. Each question is answered with a number and a chart from
`notebooks/eda.py`, which regenerates everything into `docs/eda/`. Answers go into `vault/EDA Report.md` and one
note per finding in `vault/findings/`.

## What the first look already shows

- 4,098 training lines and 1,374 test lines; test has only an ID ([[F-003 Full lines from the 1600s with scribal markup]]).
- Lines are full sentences of about 11 words; one wrong word costs 4.58 wrong characters ([[F-002 The score formula]]).
- 206 training lines share their first 30 characters with another line; 22 share their whole text.
- The images are in `data/raw/images/`, one JPEG per ID ([[F-004 The images are downloaded]]).

## Questions, in order

| # | question | check | output | what it decides |
|--:|---|---|---|---|
| 1 ✅ FND-001, FND-004 | Do test lines come from the same scans, pages and writers as training lines? | Image height, width, aspect ratio, file size, colour mode and background tone for train and test; a train-against-test classifier on those; cluster images by a pretrained embedding and count test lines per cluster | Overlaid histograms; classifier AUC; share of test lines in clusters with no training line | Whether folds can be random or must hold out whole clusters |
| 2 ✅ FND-002, D-001 | Which lines must stay together in a fold? | Group by exact text, by near-identical text (edit distance) and by near-copy images (perceptual hash) | Group count and sizes | Fold design. The winner grouped on exactly these |
| 3 ✅ FND-003 | Is there a leak? | Target length against file size, image width and file order in the zip; do zip order or timestamps place test lines between training lines of the same page? | Correlations; a plot of train and test IDs along zip order | What to exclude. An entrant cites an image-metadata rule (thread 34321); its text is on no tab and staff never replied, so treat size and order as off limits as features |
| 4 ✅ FND-005 | How bad is the label noise, and where? | Train a small reader on 4 folds, list held-out lines by error, read the worst 50; confirm the 150 px split and the multi-line crops | Share of clearly wrong labels; error by height batch | Which rows to drop (allowed if documented) and the floor on any score |
| 5 ✅ FND-005 | How noisy is a held-out score? | Bootstrap the held-out lines for one model; repeat with seeds | Standard deviation at 820 and 4,098 lines (an entrant reports 0.0048 at 820) | The smallest gain worth adopting |
| 6 ✅ FND-005 | Where does a simple reader go wrong? | Split errors into casing, markup (`^ ~ : &`), modernised spelling, names and numbers, missing or extra words; by line length and height batch | A table of error types by share of word edits | First hypotheses |

Questions 1 to 3 answered 2026-10-13; 4 to 6 on 2026-10-14 from EXP-005 (FND-005).

## Checks specific to this data

- ✅ Character inventory and frequency; characters seen fewer than 10 times (F-005).
- ✅ (FND-002) Vocabulary: share of word tokens seen once; share of held-out words never seen in the other folds.
- ✅ (FND-004) Width in pixels per character, by height batch, to choose input size.
- How often a tall crop holds more than one line, and which convention its label follows.
- ✅ (about 40 labelled training lines, charts 05, 08 and 09; plus 28 unlabelled) Look at 40 images by hand: 10 random, 10 tall, 10 with `^`, 10 of the longest.

## Hunches to test

- A small character-level reader keeps old spellings better than a vision-language model (the winner found this).
- Formula lines ("By this public act...") repeat across documents, so a character language model built from the
  training text helps a CTC reader. The winner found it helped in one reader and nowhere else.
- The tall batch is a different set of source documents, not a scanning artefact. Supported by eye (FND-004).

## Out of scope for EDA

- Any outside handwriting dataset, even to compare scripts. Training on them is forbidden.
- Reading or correcting test images by hand.

## Done when

- Every question above has a number, a chart and a note.
- The fold design is written as a decision note.
