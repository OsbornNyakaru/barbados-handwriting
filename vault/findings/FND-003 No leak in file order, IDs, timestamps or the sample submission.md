# FND-003 No leak in file order, IDs, timestamps or the sample submission

- Zip order: train and test alternate in 1,974 runs against 2,059 expected for a shuffle; neighbours in the zip are
  no more alike in paper tone (11.28 against 11.30 shuffled) or height batch (62.3% against 62.1%). Neighbouring
  training transcripts are 39.9% alike, the same as shuffled. Zip order is not page order.
- Zip timestamps take three values 2 seconds apart; test share is 26.9%, 24.1%, 26.2% across them.
- File-name order and CSV row order: same result, except the 42-row block at the top of `Train.csv` ([[FND-002 Only 4 percent of training lines repeat another line and no two images are copies]]).
- IDs: 16 random characters; no character position relates to transcript length (largest correlation 0.040).
- Sample submission: 25.2% blank, the rest short garbled strings (mean 14 characters); none matches a training
  transcript; its lengths do not follow image width (rho -0.08). It looks like the output of the 0.03 benchmark.
- Image shape against text length (training lines): rho 0.28 on short scans (3,055 lines), 0.59 on tall scans
  (1,043). Size and file size alone predict length to within 7.0 characters on average; guessing the mean gives 10.5.
- All but 8 of 5,472 images share one JPEG setting; the 8 are spread over both splits.

**Consequences:** nothing outside the picture gives the text away, so there is nothing to exclude. Image width is a
weak hint of length and a reader sees it anyway; we will not feed size or order as separate features, in line with
the unconfirmed image-metadata rule (thread 34321).

Chart: `docs/eda/07_image_shape_and_text_length.png`. Source: `notebooks/eda.py` section `q3`; numbers in `docs/eda/numbers.txt`. Checked 2026-10-13.
