# H-001 A small letter-by-letter reader keeps old spellings

Status: untested.

- Idea: a small CTC reader with a character language model built from the training transcripts makes fewer
  whole-word errors on old spellings than a vision-language model, and is the cheapest first model (about 1
  GPU-hour per fold).
- Evidence so far: 10.1% of held-out word tokens never appear in the other folds and 62% of distinct words occur
  once ([[FND-002 Only 4 percent of training lines repeat another line and no two images are copies]]); one wrong
  word costs 4.58 characters ([[F-002 The score formula]]); the winner's Kraken reader carried the largest vote.
- Test: train it under [[D-001 Five random folds, twins kept together, balanced on scan height]]; compare word
  edits per line with and without the language model.
- Research update 2026-10-13 ([[R-001 Which pretrained models to start from]]): train Kraken from scratch for the
  rules-clean score (estimate 0.86 to 0.90, 5 to 10 GPU-hours for 5 folds). Continuing from a CATMuS checkpoint is
  unruled; run it only as a separate side arm. `timm/convnext_large.fb_in22k_ft_in1k` + CTC is the second option.
