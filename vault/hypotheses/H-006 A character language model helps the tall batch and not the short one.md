# H-006 A character language model helps the tall batch and not the short one

Status: untested. From [[R-001 Which pretrained models to start from]].

- Idea: a 6-gram character language model built from the four training folds' transcripts, used in the CTC
  reader's beam search, lowers word errors on the hard tall scans and does nothing or harm on the clean short scans.
- Evidence so far: Tarride et al. 2024 (arXiv 2404.18722): word error 24.5 to 21.0 on IAM, 14.4 to 10.5 on RIMES,
  36.3 to 34.5 on POPP, and worse (2.6 to 3.4) on the cleanest set. The winner used weight 0.8, beam 64.
- Test: decoding only, on saved CTC outputs; grid over order 4 to 7 and weight 0 to 3, per batch, inside folds.
