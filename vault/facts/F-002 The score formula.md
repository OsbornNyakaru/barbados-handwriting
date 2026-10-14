# F-002 The score formula

- Page: weighted WER and weighted CER, 0.5 each; longer references weigh more.
- Forum (threads 33847, 34861): `score = 1 - 0.5 * (W / 12 + C / 55)`; W = mean word edits per line, C = mean
  character edits per line; higher is better. Said to match real submissions to nine decimals. **Not checked by us.**
- **Checked by us, 2026-10-13:** the leaderboard lists "WER Weighted" (W) and "CER Weighted" (C) for every entrant,
  and the formula reproduces the private score to nine decimals at ranks 1, 10 and 50. Rank 1: W 1.2275, C 1.7378.
  Rank 50: W 1.3794, C 2.0467. This confirms how W and C combine, not how each is averaged over lines.
- Useful targets from those rows: the best entry gets about 1.23 word edits and 1.74 character edits per line.
- Weak evidence on the averaging (2026-10-13, `tests/test_harness.py`): the sample submission's garbled strings,
  scored against unrelated training lines, give 0.013 with the plain mean and -0.011 with square-root weights. The
  benchmark row, which those strings appear to come from, scores +0.030 and +0.042. The plain mean is nearer, but
  the pairing is not the real one, so this does not settle it.
- Our scorer (`autoresearch/prepare.py`): plain mean drives adoption, the square-root variant is reported beside
  it. Spaces at the ends are stripped; words are split on any run of spaces.
- The winner (thread 35246) describes W and C as weighted by the square root of the reference word and character counts.
- The starter pack's own scorer uses 0.7 CER + 0.3 WER and imports files that were not shipped. It does not match.
- Our training lines average 11.28 words and 62.3 characters, close to the 12 and 55 in the formula.

**Consequences:** one word edit costs 4.58 character edits, so getting whole words exactly right matters most.
Write the local scorer with both variants (plain mean, square-root weights) and report both, since we cannot
submit to find out which is right.

Source: Info tab and the threads above, read 2026-10-13; averages from `data/raw/Train.csv`.
