# program.md — the experiment loop

## Goal

The highest held-out score = 1 - 0.5 * (word edits per line / 12 + character edits per line / 55) on the 4,098
training lines, each predicted by a model that did not train on it. The test set is a random quarter of the same
scans, so this is the best estimate of a leaderboard score we can have; the competition is closed.
Reference points (private leaderboard): 1st 0.9331, 10th 0.9277, 50th 0.9239, benchmark 0.0425.

## Loop

1. Read `vault/00_INDEX.md` and `experiments/RESULTS.md`.
2. Pick ONE hypothesis from `vault/hypotheses/` (or write a new one first).
3. If it needs code, add a config key to `autoresearch/experiment.py` whose default leaves old results unchanged.
   Add a test. Commit.
4. `.venv/bin/python scripts/exp.py new --from best --set key=value --hyp H-00N --desc "what changes and why"`
5. `.venv/bin/python scripts/exp.py run` (CPU) or `submit` then `pull` (GPU, see below)
6. The run is adopted when it beats the best on the same folds by >= 0.002 on average and in >= 4 of 5 folds
   (provisional until a real reader gives the noise: `vault/decisions/D-002`).
7. Record the result in the hypothesis note (supported / rejected, with numbers). Commit.
8. No submissions: the competition is closed. `exp.py submission` still writes and checks a file, as practice.

## Where runs execute

- CPU models (`medoid`, `ctc_decode`, combinations) run here with `exp.py run`.
- GPU readers (`ctc`, `vlm`) run on molab (`vault/decisions/D-003`): commit, `git push`, `exp.py submit`, then the
  user opens `notebooks/molab_runner.py` on molab and presses Run queue; `exp.py pull` scores and records.
- A quick CPU check of a GPU spec before queueing it:
  `.venv/bin/python autoresearch/remote.py EXP-NNN --folds 0 --set epochs=1 --set limit_train=120`.
- The order of experiments is in `vault/Experiment Plan.md`.
- The repository is public. Never commit transcripts, predictions or images; `make hooks` installs the check.

## Hard rules

- Never edit `prepare.py` after EXP-001. `exp.py run` stops if its hash differs from the best run's.
- Train on the competition files only. No outside handwriting data, in any form.
- Pretrained models only under a licence that allows commercial use (Apache 2.0, MIT and similar). Record the
  model and licence in the spec's description.
- No hand labelling or hand fixes. Dropping training rows is allowed only by a rule in code, with IDs and reasons
  written to the vault. Held-out lines are never dropped from scoring without a decision note.
- No AutoML, no paid model services. Image size and file order are not features.
- Anything learned from the transcripts (language models, word lists, pasted-crop augmentation) is built inside
  each fold from the four training folds only.
- Ideas from other entrants only; never their weights, predictions or files.
