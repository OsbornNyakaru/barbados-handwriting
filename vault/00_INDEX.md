# Index

R.O.A.D. Barbados Historic Handwriting Challenge (Zindi). **Closed 5 Oct 2026; practice only.**

## Where we stand (2026-10-13)

- Recon done: `RULES.md`, [[Competition Overview]], [[EDA Plan]], `first_look.md`.
- Refreshed the same day: no change on the page or leaderboard; 23 of 30 forum threads now read (`RULES.md` change log).
- Third read at 16:30 EAT: still no change. The leaderboard's own word and character error columns confirm how
  the score combines them ([[F-002 The score formula]]).
- EDA questions 1 to 3 done ([[EDA Report]]); questions 4 to 6 need a GPU. Validation scheme proposed in
  [[D-001 Five random folds, twins kept together, balanced on scan height]].
- Harness built and frozen 2026-10-13: `autoresearch/prepare.py` hash `5ed07a757d39`, fold hash `6d2b14640424`,
  12 tests pass. Adoption rule in [[D-002 Adoption rule (provisional)]]. Local git repository, no remote.
- **Current best: EXP-007, 0.8655 ± 0.0019** (CTC reader EXP-005 at 0.8318, decoded with a 6-letter character language model, +0.034 in 5 of 5 folds). Floor: EXP-001 0.1404.
- No reader trained, no submission made (none possible). Images downloaded 2026-10-13.
- 2026-10-14: molab set up ([[D-003 GPU runs on molab, code from a public GitHub repository, data in private Hugging Face]]).
  Public GitHub repository for code; data and results in private Hugging Face datasets. Readers written: CTC
  (`autoresearch/readers/ctc.py`), Qwen3-VL (`readers/vlm.py`), character language model (`readers/lm.py`); 15 tests
  pass. EXP-002 (CTC from scratch, 5 folds) and EXP-003 (Qwen3-VL-2B, fold 0) queued. Plan: [[Experiment Plan]].
- Earlier research (12 Aug 2026) is in `PROJECT_STATE.md` and `research_loop/`. Its metric notes are out of date;
  see [[F-002 The score formula]].

## Facts

- [[F-001 The competition is closed]]
- [[F-002 The score formula]]
- [[F-003 Full lines from the 1600s with scribal markup]]
- [[F-004 The images are downloaded]]
- [[F-005 Transcripts run 37 to 120 characters and some carry stray spaces]]

## Findings

- [[FND-006 The CTC reader works best as a judge between vision-language readings]]

- [[FND-005 About 2 percent of lines are crop or label problems and they hold 12 percent of the errors]]

- [[FND-001 Test lines are a random quarter of the same scans as training lines]]
- [[FND-002 Only 4 percent of training lines repeat another line and no two images are copies]]
- [[FND-003 No leak in file order, IDs, timestamps or the sample submission]]
- [[FND-004 The two height batches are two kinds of document at two scales]]

## Hypotheses (untested)

- [[H-001 A small letter-by-letter reader keeps old spellings]]
- [[H-002 Bring both batches to one letter size]]
- [[H-003 Self-training on test images adds little]]
- [[H-004 Qwen3-VL-8B with the vision part trained is the best single reader]]
- [[H-005 A vote of one vision-language and one letter-by-letter reader beats both]]
- [[H-006 A character language model helps the tall batch and not the short one]]

## Research

- [[R-001 Which pretrained models to start from]]: licences, shortlist and the order of the first runs.

- 2026-10-14 first GPU results: [[EXP-003 Qwen3-VL-2B reads fold 0 at 0.882]];
  [[EXP-002 The first CTC reader never learned because the LSTM stalled training]] (cause found and fixed; EXP-005
  re-runs it). Molab installs its own scikit-learn, so the folds travel as `folds.csv` (D-003).

## Next step

Run the molab queue (EXP-002, EXP-003), then `exp.py pull`. Then EDA questions 4 to 6 on EXP-002's held-out errors,
reset the adoption rule from its noise, and follow [[Experiment Plan]].
Run things with `make test`, `make status`, `make run` (CPU), `make submit` / `make pull` (molab).
Environment: `.venv` from `requirements.txt` (Python 3.12). Rerun the EDA with `.venv/bin/python notebooks/eda.py`.
