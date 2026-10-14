# F-003 Full lines from the 1600s with scribal markup

Measured on `data/raw/Train.csv` (4,098 rows), 2026-10-13; first noted 2026-08-12 in `research_loop/SHARED_CONTEXT.md`.

- Each target is a full line: mean 11.28 words (median 11), mean 62.3 characters (median 62, max 120).
- Of 148 four-digit years in the targets, 146 are in the 1600s. "six hundred" appears in 35 lines, "seven hundred"
  in none. The data tab says "18th and 19th century".
- 81 distinct characters. `^` (superscript) is in 16.8% of lines; `~` in 87 lines. Old spellings are kept as written.
- 4,086 distinct targets; 22 rows share their exact text with another row; 206 rows share their first 30 characters
  with another row (standard legal opening formulas).
- IDs are random 16-character strings. `Test.csv` has only the ID. There is no document, page or writer column.

**Consequences:** pretrained readers of modern handwriting will modernise spellings, which costs whole-word errors
([[F-002 The score formula]]). Lines with the same or nearly the same text must sit in the same validation fold.
Whether test lines come from the same pages or writers as training lines can only be learned from the images
([[F-004 The images are downloaded]]).
