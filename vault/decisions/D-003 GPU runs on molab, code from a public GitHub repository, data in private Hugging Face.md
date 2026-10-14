# D-003 GPU runs on molab, code from a public GitHub repository, data in private Hugging Face

Decided 2026-10-14 by the user. Same pattern as the afro-health-qa project.

- **Code:** public GitHub repository `OsbornNyakaru/barbados-handwriting`. The user chose public so molab can sync the
  runner notebook from GitHub without a token. Everything that runs on molab is fetched from GitHub at the spec's
  commit, so molab always runs the latest pushed code and an old spec still runs its own code.
- **Data:** private Hugging Face dataset `nyakaruosborn/barbados-handwriting-data` (Train.csv, Test.csv,
  SampleSubmission.csv, images.zip; SHA-256 in its manifest). Uploaded 2026-10-14 by `scripts/push_data_to_hf.py`,
  which refuses a public repository.
- **Results:** private Hugging Face dataset `nyakaruosborn/barbados-handwriting-runs`: `queue/EXP-NNN.json` (specs)
  and `runs/EXP-NNN/` (per-fold predictions, CTC log-probabilities, logs, result.json). `exp.py` refuses it if public.
- **Runner:** `notebooks/molab_runner.py` (marimo). Paste a Hugging Face token, press Run queue. It checks
  `prepare.py` against its pinned SHA-256 and the fold hash, then calls `autoresearch/remote.py`, which runs fold by
  fold and uploads each finished fold, so a dead session resumes at the next fold. A claim file stops two open tabs
  from running the same spec.
- **Loop:** `exp.py new` → commit → `git push` → `exp.py submit` → Run queue on molab → `exp.py pull`.
- **Keeping the data out of the public repository:** transcripts quoted in four vault notes were replaced with
  descriptions on 2026-10-14; `scripts/check_public.py` (pre-commit hook, `make hooks`) blocks any five-word run from
  a training transcript, any sample-submission target, and any CSV, image or data folder. The history before the
  public push stays on the local branch `pre-public` and is never pushed.
- **Hardware to record for code review:** the GPU name and memory are written into every result.json by the runner.
- This machine still runs the EDA, the tests, the scorer, `ctc_decode` (re-decoding saved log-probabilities with a
  language model) and every combination of saved predictions.
- **2026-10-14, first molab attempt:** both runs stopped at the fold check (hash `c82a8c6c2563`, frozen
  `6d2b14640424`). molab ignores the notebook's version pins for packages it ships: it ran scikit-learn 1.9.0,
  pandas 3.0.3, numpy 2.4.6 (frozen with 1.5.2, 2.2.3, 1.26.4). Reproduced here: same data, same text-twin groups
  (3,971), same scorer output; only scikit-learn's fold splitter shuffles differently. Fix: the folds travel as data
  (`folds.csv` in the private data repo, written by `scripts/push_data_to_hf.py` in the pinned `.venv`), and
  `remote.py` checks them against the frozen fold hash. The scorer gives identical results under molab's versions.
  Crashed runs are re-queued with `exp.py submit --resubmit EXP-NNN`.
