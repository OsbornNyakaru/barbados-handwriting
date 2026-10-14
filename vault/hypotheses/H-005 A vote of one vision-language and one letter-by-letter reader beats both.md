# H-005 A vote of one vision-language and one letter-by-letter reader beats both

Status: untested. From [[R-001 Which pretrained models to start from]].

- Idea: align the two readers' out-of-fold transcripts character by character and vote; where they differ on an
  old spelling the letter-by-letter reader is right more often.
- Evidence so far: the winner's vote lifted singles near 0.90 to 0.933 with 40 models; a sentence-level vote of
  TrOCR models took character error from 1.86 to 1.60 on 16th-century Latin (arXiv 2508.11499).
- Test: needs [[H-001 A small letter-by-letter reader keeps old spellings]] and
  [[H-004 Qwen3-VL-8B with the vision part trained is the best single reader]] run on the same folds. Vote weights
  are fitted on out-of-fold predictions only. Expected +0.01 to +0.02 (estimate).
