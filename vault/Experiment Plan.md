# Experiment Plan

Written 2026-10-14 from [[R-001 Which pretrained models to start from]]. Every experiment is one spec with one change
against its parent, scored on the same five frozen folds (4,098 held-out lines). GPU runs go through molab
([[D-003 GPU runs on molab, code from a public GitHub repository, data in private Hugging Face]]); decoding and
combining run on the Mac from saved outputs. GPU-hours are estimates for molab's RTX Pro 6000 (96 GB) until the
first run measures them. Reference points: no reading 0.140 (EXP-001); leaderboard 50th 0.924, 1st 0.933.

## How each experiment is run

```bash
.venv/bin/python scripts/exp.py new --from <parent> --set <key>=<value> --hyp H-00N --desc "..."
git commit -am "EXP-NNN: ..." && git push        # GPU specs only: molab runs the pushed code
.venv/bin/python scripts/exp.py submit           # GPU specs: queue on Hugging Face
#   molab: open notebooks/molab_runner.py, GPU on, paste the token, Run queue
.venv/bin/python scripts/exp.py pull             # GPU specs: score, adopt or not, update the ledger
.venv/bin/python scripts/exp.py run              # CPU specs (decoding, combining) run here
```

Adoption ([[D-002 Adoption rule (provisional)]]): +0.002 averaged over the folds and better in 4 of 5. Reset from
EXP-002's own noise once it is in. Runs on some folds only are recorded but never adopted.

## Stage A: a letter-by-letter reader (H-001, H-006)

| exp | change | where, cost | what it tells us | then |
|---|---|---|---|---|
| **EXP-002** (queued) | `model=ctc`: CRNN trained from scratch, height 80, 60 epochs, greedy decoding | molab, about 1 to 2 GPU-h for 5 folds | The first real score; expected 0.86 to 0.90. Gives the noise for the adoption rule | Run EDA questions 4 to 6 on its held-out errors (label noise, score noise, error types) |
| EXP-004 | `ctc_decode` of EXP-002: beam 10 with a 6-letter language model, weight 0.5 | Mac, about 20 min | Whether the language model helps, by batch (H-006 says tall yes, short no) | If it helps, try weights 0.3 and 1.0 as EXP-005 and EXP-006 |
| EXP-007 | height 80 → 128 on the best CTC spec | molab, about 2 GPU-h | Whether more pixels per letter help the tall batch (H-002) | Keep whichever is better |
| EXP-008 | epochs 60 → 120 | molab, about 3 GPU-h | Whether the reader is still learning at 60 | |

## Stage B: a vision-language reader (H-004)

| exp | change | where, cost | what it tells us | then |
|---|---|---|---|---|
| **EXP-003** (queued) | `model=vlm`: Qwen3-VL-2B-Instruct, LoRA on language and vision, 3 epochs, greedy, **fold 0 only** | molab, about 0.5 GPU-h | The pipeline works on a GPU; a first VLM score on fold 0 next to EXP-002's fold 0 | Go to 8B |
| EXP-009 | model_id → `Qwen/Qwen3-VL-8B-Instruct`, all 5 folds | molab, about 8 to 15 GPU-h, over several sessions (resumes per fold) | The best single reader? Expected 0.85 to 0.91. Count modernised spellings in its errors | |

## Stage C: combine the two kinds of reader (H-005)

| exp | change | where, cost | what it tells us |
|---|---|---|---|
| EXP-010 | new CPU model `combine`: for each line, score the CTC beam output and the VLM output under the CTC log-probabilities plus the language model, keep the better | Mac, minutes | Whether the two readers' errors differ enough to gain. Expected +0.01 to +0.02 over the best single reader |

## Stage D: a second letter-by-letter reader, then a second vision-language family

| exp | change | where, cost | why |
|---|---|---|---|
| EXP-011 | backbone `crnn` → `convnext_large.fb_in22k_ft_in1k` (Apache 2.0), height 128 | molab, about 5 GPU-h | The winner's largest CTC reader; a third voice for the combination |
| EXP-012 | combine three readers by character-level vote | Mac | Expected 0.915 to 0.925 |
| EXP-013 | LightOnOCR-2-1B (Apache 2.0), only if EXP-012 gained 0.002 or more over EXP-010 | molab, 3 to 5 GPU-h | A second VLM family for the vote |

## Not planned (from R-001)

Nine image views per line for Qwen (about 105 GPU-h), models above 8B, CATMuS or any reader pretrained on outside
handwriting (unruled), pseudo-labels, dictionaries and learned re-rankers (dead ends for the winner).

## Order and stopping

EXP-002 and EXP-003 run in one molab session (queued together). Each later step starts when the step it depends
on is pulled. A step that loses is recorded and its hypothesis marked rejected; the plan goes on from the best spec.
