# H-002 Bring both batches to one letter size

Status: untested.

- Idea: resize tall scans so a letter is as wide as in short scans (about 4.6 times smaller), and crop tall scans
  to the centre band, since they take in more of the neighbouring lines.
- Evidence so far: 16.7 against 76.5 px per character; tall crops are 6.3 times taller but only 4.6 times larger
  per letter ([[FND-004 The two height batches are two kinds of document at two scales]]). Against it: an entrant
  reports per-batch resolution fixes gained nothing at 0.906 (thread 34861), and the winner trained separate word
  counters at 128 and 192 px.
- Test: one reader, two input recipes, scored per batch.
