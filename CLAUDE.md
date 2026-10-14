# CLAUDE.md — agent entry point

R.O.A.D. Barbados Historic Handwriting Challenge (Zindi): read one handwritten line from a scanned 1600s Barbados
record and transcribe it exactly as written. Images and text, 4,098 training lines, 1,374 test lines.
**Closed 5 Oct 2026 02:29 EAT. Practice only: no submissions are accepted, so the only score is our own held-out one.**

## Read in this order

1. `vault/00_INDEX.md`: what is known, the current best, open hypotheses. Record every result in the vault.
2. `RULES.md`: what the competition forbids. Competition data only for training; pretrained models only under a
   licence that allows commercial use; nothing labelled or corrected by hand.
3. `autoresearch/program.md`: the experiment loop.
4. `experiments/RESULTS.md`: the ledger.

## Files

- `autoresearch/prepare.py`: FROZEN. Data loading, folds, the metric, the submission format.
- `autoresearch/experiment.py`: the one file an experiment edits.
- `scripts/exp.py`: `new`, `run`, `submit`, `pull`, `status`, `submission`, `lb`.
- `autoresearch/readers/`: the CTC reader, the Qwen3-VL reader, the character language model.
- `autoresearch/remote.py` and `notebooks/molab_runner.py`: running specs on molab.
- `vault/Experiment Plan.md`: the experiments in order.
- `notebooks/eda.py`: regenerates every chart in `docs/eda/`.

## Principles

1. **Trust the validation score.** There is no leaderboard to check against. The score's spread is the `±` column
   (a bootstrap over the 4,098 held-out lines); see `vault/decisions/D-002`.
2. **Test lines are a random quarter of the same scans as training lines** (`vault/findings/FND-001`), so folds are
   random at line level with near-identical transcripts kept together. Never read or correct a test line by hand.
3. **One change per experiment.** `exp.py new` refuses more unless `--allow-multi`.
4. **Whole words matter most.** One wrong word costs as much as 4.58 wrong characters, and one word in ten in a
   held-out fold was never seen in training, so readers must keep old spellings letter by letter.
5. **Two score variants.** `score` (plain mean over lines) drives adoption; `score_sqrt` is reported beside it
   because the averaging on the real leaderboard was never confirmed. Always report the short and tall batches too.
6. **Reproducible.** Seeds fixed; a spec plus the git commit reproduces a run.

## Environment

- Python: `.venv/bin/python` (create with `make setup`). Do not use `uv run`.
- This machine has no GPU. Readers train on molab: `exp.py submit`, Run queue in `notebooks/molab_runner.py`,
  `exp.py pull` (`vault/decisions/D-003`). Data and results live in private Hugging Face datasets.
- **The GitHub repository is public** (`OsbornNyakaru/barbados-handwriting`, chosen 2026-10-14 so molab can sync).
  Competition data stays out of git (`data/`, `experiments/runs/`, `submissions/*.csv`, the image sheets in
  `docs/eda/`), and no note may quote a transcript: `make hooks` installs `scripts/check_public.py` as a pre-commit
  check. Run `make hooks` in every fresh clone.
- Report times in EAT (UTC+3).
- Commits carry no AI attribution lines. No tokens or keys in code.
