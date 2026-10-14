# Rules

Source: https://zindi.world/competitions/road-barbados-historic-handwriting-challenge (read 2026-10-13, signed in for
the forum and leaderboard). Where this file and the page differ, the page wins.

**Status on 2026-10-13: the competition is closed.** The page header reads "Under code review". The leaderboard page says "Submissions are no longer accepted"
and "Scores and ranking are subject to change until 26 October 2026 pending final review". Anything we do here is
practice, scored on our own held-out lines only.

## Scoring

- Page (Info tab): two error metrics, "weighted Word Error Rate (WER), and weighted Character Error Rate (CER)",
  each weighted 0.5. "Longer reference transcriptions are weighted more heavily". Missing, empty or invalid
  predictions count as wrong.
- Forum, entrant J0NNY (thread 33847, 13 Jul): `score = 1 - 0.5 * (W / 12 + C / 55)`, where W is the mean word-level
  edit distance per line and C the mean character-level edit distance per line. **Higher is better.** A second
  entrant (thread 34861) says this reproduces the Benchmark row "to 2e-10" and their own submissions to nine decimals.
- Checked by us (2026-10-13, leaderboard tab): each leaderboard row shows its own "WER Weighted" and "CER Weighted"
  values, and `1 - 0.5 * (WER Weighted / 12 + CER Weighted / 55)` gives the listed private score to nine decimals for
  ranks 1, 10 and 50 (rank 1: 1.227466747 and 1.737800293 give 0.933057367). So the way the two numbers combine is
  confirmed. How each number is averaged over lines (plain mean or square-root length weights) is still not.
- Consequence of the formula: one word edit costs as much as 55/12 = 4.58 character edits.
- Submission columns: `ID`, `Target` (text). 1,374 rows. Row order does not matter.
- Public / private split: the page says about 30% / 70% in the rules summary and about 20% / 80% in the full rules.
- Benchmark row on the leaderboard: 0.0304 public, 0.0425 private.

## What we must not do

- Train, fine-tune, pseudo-label or adapt on any data other than the competition files (staff, thread 34734, 11 Sep:
  "Competition data only may be used for training or fine-tuning").
- Use a pretrained model whose licence is research-only, academic-only or non-commercial (thread 34734). Example
  denied: `stanford-oval/churro-3B`, because its base carries the Qwen Research License (threads 34481, 34712).
  Staff, thread 34053, 5 Aug: "Public availability of the model weights is not, by itself, enough".
- Label or correct anything by hand: no manual transcription of test images, no hand-targeted fixes for individual
  test lines, no Excel edits (rules page; thread 34459).
- Use AutoML tools (rules page). Optuna-style hyperparameter search is allowed (staff, thread 33891).
- Use paid model services or proprietary model APIs, or free trials that need a credit card (rules page; thread 33870).
- Use custom packages in the submitted code (rules page).
- Share code or data privately outside the team, or hold more than one account (rules page).
- Copy, publish or reuse the dataset anywhere, including public repositories and other competitions (data tab).
- Probe the leaderboard or share submissions. Staff removed entrants for "coordinated account activity, submission
  sharing, and leaderboard probing" (thread 35197, 4 Oct).

## Allowed, with conditions

- Pretrained base models that are publicly available under a licence allowing commercial use and deployment by the
  host. We need not audit the upstream training data unless the model's licence passes restrictions down (34734).
  Example approved: PP-OCRv6 medium, Apache 2.0, "fine to use" (staff, thread 34481, 28 Aug).
- Pseudo-labelling the test images, if fully automatic and reproducible, with only general code-driven
  post-processing (staff, thread 34459, 17 Aug).
- Dropping clearly corrupted training rows, if the original file is unchanged and the dropped IDs and reasons are
  documented (staff, thread 33891, 16 Jul).
- Cut-and-paste augmentation from the training images (StackMix style), if automatic and reproducible; no test
  images, no hand-annotated crops (staff, thread 34514, 24 Aug).
- Rented GPUs (RunPod, AWS and similar) as plain infrastructure, with data kept private and hardware and runtime
  documented (staff, thread 33870, 16 Jul).

## Submissions

- 5 per day, 200 in total. A submission that fails processing does not count.
- 2 submissions chosen for the private leaderboard; otherwise the best 2 public ones.
- Tiebreaker: the earlier submission wins.

## Team and accounts

- One account per person. Teams of at most 4. No new members in the final 5 days.

## Review and ownership

- Top 10 on the private leaderboard send code within 48 hours of the request; Zindi may ask anyone at any time (24 hours).
- All model weights and inference code must be handed over and must reproduce the score. Set every seed.
- The top 3 assign copyright in the winning solution to Zindi.
- Penalties: first offence, no prizes for 6 months and 2,000 points removed; second offence, a ban.

## Dates

- Opened 3 Jul 2026. Submissions closed 5 Oct 2026 02:29 EAT (4 Oct 23:29 UTC); staff: "29 mins after midnight
  GMT+1" (thread 35179). Final review until 26 Oct 2026.

## Compute

- Required by the competition: no hardware or runtime limit stated. Hardware and approximate runtime must be documented.
- This machine: Intel Core i7-8569U, 8 threads, 16 GB memory, no usable GPU, 243 GB free disk.
- Also available: a molab notebook with a GPU that pulls specs from GitHub; a private Hugging Face dataset for data
  and results.
- Reference point: the winner reports 1 hour (ConvNeXt) to 21 hours (Qwen3-VL, nine views) per fold on one RTX
  A6000, across 8 model variants x 5 folds (thread 35246).
- Consequence: this machine can do the EDA and score predictions. Any model training needs the GPU notebook. A
  small CTC reader (about 1 GPU-hour per fold) is the realistic first model; a fine-tuned vision-language model is
  a multi-hour run per fold.

## Data

| file | size | what it holds | joins on |
|---|---|---|---|
| `Train.csv` | 326 KB | 4,098 rows: `ID`, `Target` (the transcription) | `ID` = image name |
| `Test.csv` | 24 KB | 1,374 rows: `ID` | `ID` = image name |
| `SampleSubmission.csv` | 44 KB | 1,374 rows: `ID`, `Target`; 25% of example targets are blank | `ID` |
| `Starters.zip` | 28 KB | starter code: VLM, Paddle-OCR, Kraken-OCR | |
| `images.zip` | 442 MB | all train and test line images; extracted to `data/raw/images/` (5,472 JPEGs). `https://storage.googleapis.com/road-handwriting/images.zip` | file name = `ID` |

- Origin: line crops from scanned records in Barbados' archive, digitised by the R.O.A.D. Programme (GovTech Barbados).
  The data tab says "6K cropped images" (we count 5,472, one per train and test ID, none extra) and "Each image
  contains one or more handwritten words".
- Licence: this challenge only. No copying, publishing or reuse.
- Starter: Qwen2-VL-2B with LoRA; a Kraken model continued from a CATMuS checkpoint. The starter's own scoring code
  uses 0.7 CER + 0.3 WER, which does not match the leaderboard.

## Forum (read 2026-10-13, twice; 30 threads over 2 pages; 23 read in full)

### Staff rulings

Listed above with thread numbers: 34734 (external data and licences), 34459 (pseudo-labelling), 33891 (Optuna, row
exclusion), 33870 (rented GPUs), 34514 (StackMix), 35197 (integrity action), 35179 (closing time), 34053 (licences
need review), 34481 (churro-3B denied, PP-OCRv6 approved).

Asked and never answered by staff: hardware, model-size and runtime limits for code review, and whether the review
machine has internet (33702, 11 Jul); where the starter's missing `wer.py` and `cer.py` are (33930); whether a named
outside dataset may be used for adaptation (34481, 28 Aug; 34612; the general ruling in 34734 says no); approval of
`Kansallisarkisto/multicentury-htr-model`, Apache 2.0 (35088; another entrant pointed to 34734).

### Leaks reported (never use)

- None reported. External handwriting datasets raise scores a lot (one entrant claims 75% to 98% character accuracy,
  thread 34304) and are forbidden.

### Other points

- Label noise: about 1 training line in 70 is paired with another line's text; 12 of 820 held-out lines carried 18%
  of one team's character errors (thread 34861). A circulated list of 21 "bad" IDs (thread 33891) is mostly correct
  centre-line labels on tall crops, so do not drop all 21.
- Multi-line crops: most are labelled with the centre line only, some with every line. Staff asked for examples and
  never ruled (thread 34089). Four example IDs are in the thread.
- Two scan batches: lines taller than 150 px (about 27% of lines) score far worse (0.823 against 0.900) and resizing
  does not fix it (thread 34861).
- Held-out noise: standard deviation 0.0048 on 820 lines (thread 34861, an entrant's measurement).
- Dead ends measured by an entrant at about 0.906 (thread 34861): re-casing from a word list, fixing `^ : &` markup,
  snapping rare words to the training vocabulary, guarding against over-long output, per-batch resolution fixes.
- Winner's write-up (thread 35246): 5 base models, 8 variants x 5 folds, a weighted character-level vote; a small
  Kraken reader with a 6-character language model gets the largest vote because it keeps old spellings that
  vision-language models modernise. Dead ends there: bigger models, pseudo-labels, learned re-rankers, dictionaries.

## Our own rules

- Ideas from other entrants and past winners only; never their weights, predictions or submission files.
- Competition data and predictions stay out of any public repository.
- No tokens or keys in code.
- One Zindi account.
- Times reported in EAT (UTC+3).

## Open questions

- Is the length weighting real? The page says longer references weigh more and the winner describes square-root
  weights; the forum formula is a plain mean and is said to match to nine decimals. We cannot test it: no
  submissions are accepted.
- The 30/70 against 20/80 split.
- Whether multi-line test crops are scored on the centre line only (34089, no ruling).
- Whether coding assistants are allowed (thread 34367, no reply).
- An "image-metadata rule" is cited in thread 34321 (6 Aug). Its text is not on the Info, rules or Data tab, and
  no staff member replied in that thread. Treat image size and file order as off limits as predictive features.
- No hardware, model-size or runtime limit for code review was ever stated (question in 33702, unanswered).
- The data tab says "18th and 19th century"; the transcripts carry years from the 1600s (see `vault/facts/F-003`).
- The full rules name a different competition ("Barbados Lands and Surveys Plot Automation Challenge") in the data
  clause, a copy-paste slip.

## Questions for the forum

None worth posting: the competition is closed.

## Change log

- 2026-10-13 16:30 EAT (third read): no rule, date, forum or ranking change. Still "Under code review"; 30 threads,
  no new posts since 12 Oct 19:09 EAT (35246). Joined count 1,894 (was 1,893). New in our notes, not on the page:
  the leaderboard's per-row "WER Weighted" and "CER Weighted" values reproduce the private scores through the forum
  formula (see Scoring). Private scores for reference: 1st 0.933057, 10th 0.927726, 50th 0.923919.

- 2026-10-13 (second read, same day): no rule, date or leaderboard change. Header now reads "Under code review".
  Read 8 more threads (33702, 33930, 34053, 34110, 34321, 34612, 35088, 35202); added the 34053 ruling and the
  PP-OCRv6 approval in 34481. One new reply in 35246 (12 Oct), no content.
- 2026-10-13: first read. Earlier notes in `PROJECT_STATE.md` and `research_loop/` date from 12 Aug 2026; where they
  differ on the metric (they say 0.7 / 0.3), this file is newer.
