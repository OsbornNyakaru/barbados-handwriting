# F-005 Transcripts run 37 to 120 characters and some carry stray spaces

Measured on `data/raw/Train.csv` (4,098 rows), 2026-10-13, `notebooks/eda.py` section `look`.

- Every transcript is 37 to 120 characters and 4 to 22 words. There are no short lines, although the data tab says
  "one or more handwritten words". The organisers appear to have kept only lines of 37 characters or more.
- 98 transcripts start or end with a space; 55 contain a double space.
- 81 distinct characters; 7 occur fewer than 10 times (`+ Q * # ? | \`). `^` is in 16.8% of lines.
- `Test.csv` starts with a byte-order mark; read it with `utf-8-sig` or the `ID` column name is wrong.
- 8 crops are squarer than 6:1 (6 train, 2 test). Among the 7 squarest training crops, two hold three written
  lines and are labelled with all three; one holds one labelled line inside a 5929 x 1131 px crop.

**Consequences:** the scorer must state how it treats edge and double spaces (word edits ignore them if words are
split on whitespace; character edits do not). An empty or very short prediction is always wrong, so a length guard
is cheap. The multi-line question (thread 34089) stays open until a reader exists.
