# F-004 The images are downloaded

- `data/raw/images.zip` (442,081,009 bytes) downloaded 2026-10-13 from the organisers' link and extracted to
  `data/raw/images/`: 5,472 JPEG files, 432 MB.
- Every one of the 4,098 train IDs and 1,374 test IDs has exactly one image named `<ID>.jpg`; there are no extra files.
- Nothing else about the images has been measured by us. Entrants report two scan batches split cleanly at 150 px
  height, with the taller batch (about 27% of lines) much harder, and about 0.1% of crops with an aspect ratio
  below 6 (thread 34861).

**Consequences:** the image questions in [[EDA Plan]] can now be run.
