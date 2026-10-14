# First look at `data/raw`

## Files

| file | size | kind |
|---|--:|---|
| SampleSubmission.csv | 44 KB | table: 1374 rows x 2 columns |
| Starters.zip | 28 KB | zip: 8 .py, 3 .yaml, 3 .txt, 3 .sh, 1 .md |
| Test.csv | 24 KB | table: 1374 rows x 1 columns |
| Train.csv | 326 KB | table: 4098 rows x 2 columns |
| manifest-7d1e7efe554232bb4cb70b6040162ba520260703-3588-1i7qw6e.json | 1 KB | json, not a table |

## `SampleSubmission.csv`

1374 rows, 2 columns. Exact duplicate rows: 0.

| column | type | missing | distinct | examples |
|---|---|--:|--:|---|
| ID | str | 0.0% | 1374 | MzQuRiUbPFsq6Azy, 1LrXN25dFz88Spt3, voWkMuyhO8FAlLr5 |
| Target | str | 25.2% | 1011 | (examples removed) |

## `Test.csv`

1374 rows, 1 columns. Exact duplicate rows: 0.

| column | type | missing | distinct | examples |
|---|---|--:|--:|---|
| ID | str | 0.0% | 1374 | MzQuRiUbPFsq6Azy, 1LrXN25dFz88Spt3, voWkMuyhO8FAlLr5 |

## `Train.csv`

4098 rows, 2 columns. Exact duplicate rows: 0.

| column | type | missing | distinct | examples |
|---|---|--:|--:|---|
| ID | str | 0.0% | 4098 | uGI8F9Er0c5XwdnX, Dqt2gc3V5RJqmLE5, eqF93vAKYa4aOeHr |
| Target | str | 0.0% | 4086 | (examples removed: transcripts stay out of the public repository) |

## Train against test (`Train.csv` and `Test.csv`)

- Columns only in train (target candidates): Target
- Columns only in test: none
- Sample submission `SampleSubmission.csv`: 1374 rows; columns ID, Target
- Sample submission IDs found in test: 100.0%

### Target `Target`

- text: 4086 distinct values in 4098 rows; length median 62 characters, max 120
- most repeated value covers 0.1% of rows

### How much of test was seen in train

| column | distinct in train | distinct in test | test rows whose value is in train |
|---|--:|--:|--:|
| ID (ID) | 4098 | 1374 | 0.0% |

Read this table for how the test set was made: a group, place or entity column at 0% means test
holds units never seen in train, so validation must hold out whole units of that kind.

