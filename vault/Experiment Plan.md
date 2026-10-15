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

## Update 2026-10-14 (afternoon): eight readers, then a vote

First GPU results changed the numbering: EXP-005 is the CTC reader with the LSTM fix (EXP-002 never learned),
EXP-006 is Qwen3-VL-8B on all folds. A 200-line test of the character language model on EXP-005's fold 0 gave
+0.033 (0.825 greedy to 0.858). Rerunning the same CTC spec moved a fold's score by about 0.003 (GPU randomness).

The user asked for several readers of both kinds, combined. The lineup, run one after another on molab:

| # | reader | kind | GPU-h, 5 folds |
|--:|---|---|---|
| 1 | CRNN from scratch + rnn_skip (EXP-005) | letter by letter | 1.5 (measured) |
| 2 | the same at height 128 | letter by letter | ~2 |
| 3 | ConvNeXt-large CTC (`convnext_large.fb_in22k_ft_in1k`, Apache 2.0) | letter by letter | 3 to 5 |
| 4 | Qwen3-VL-2B, all folds | vision-language | 1.5 (measured on one fold) |
| 5 | Qwen3-VL-8B (EXP-006) | vision-language | ~5 |
| 6 | Qwen3-VL-4B-Instruct (Apache 2.0) | vision-language | ~3 |
| 7 | LightOnOCR-2-1B (Apache 2.0) | vision-language | 2 to 3 |
| 8 | Qwen3-VL-8B at another image height | vision-language | ~5 |

Combining, on the Mac from saved held-out readings: first a judge (CTC log-probabilities plus the character language
model choose between the CTC and the VLM reading per line), then a character-level vote weighted by each reader's
held-out accuracy. A reader stays in the vote only if the combined score gains at least the noise (about 0.003 per
fold) in most folds. Expected: 0.915 to 0.925 (R-001 estimate).

## Update 2026-10-14 (evening): the CTC reader is the judge

[[FND-006 The CTC reader works best as a judge between vision-language readings]]: choosing per line among Qwen
readings by CTC evidence reached 0.907 on fold 0. The CTC reading itself is not a candidate. Order from here:
EXP-006 (Qwen3-VL-8B, running) → EXP-008 (Qwen3-VL-2B, 5-best, queued) → combine "choose" on all folds → a
stronger judge (ConvNeXt CTC; CRNN at height 128) → more candidates (second 8B seed with 5-best, LightOnOCR-2).

## Update 2026-10-15 (afternoon): the plan after first place's write-up

Where we stand: the vote (EXP-033) 0.9170; 1st place 0.933, 50th 0.924. EXP-019 (8B, 4 epochs) lost on every fold
(0.8866 vs 0.894) and added nothing as a voter (EXP-036, +0.0001), so more training is not the way. EXP-030 (8B with
first place's recipe: 2 epochs, lr 2e-4, merger trained, beam 4, repeat collapse) beat EXP-006 on its first two folds
(0.9023 vs 0.8943, 0.9066 vs 0.8926) and runs about 25 min a fold, faster than EXP-006.

New: spec option `read_test` (top level of config). Each fold model also reads the test lines (fold<k>_test.*), and a
vote spec's `test_from` can map a run to "folds" to use them, the source's weight split over the five fold models
(first place's test readings, R-002). New readers get it; readers already run get one all-lines test run instead
(cheaper, and FND-007 shows it holds up on Zindi).

| # | step | needs | basis | expected |
|--:|---|---|---|---|
| 1 | EXP-030 into the vote (as an extra voter, then replacing EXP-006) | CPU, minutes, after EXP-030 | R-002; +0.008 to +0.014 alone per fold | +0.003 to +0.008 |
| 2 | The recipe on Qwen3-VL-2B (5-best) and 4B, with `read_test` | about 1 + 1.5 GPU-h | R-002; EXP-030 | +0.002 to +0.005 via the vote |
| 3 | EXP-031 ConvNeXt-large CTC into the vote | CPU, after EXP-031 (~17:30 EAT) | R-002 (ConvNeXt weight 3 of 47) | 0 to +0.003 |
| 4 | Word-span crops: word boundaries from forced alignment of each training line against its own held-out CTC log-probabilities (EXP-018, on the Mac), then 1 to 5 word crops as extra Qwen training examples | code (Mac), then one 8B run, about 3 GPU-h | R-002; our tall batch is the weak one (0.869 vs 0.934) | +0.003 to +0.008 |
| 5 | LightOnOCR-2-1B, full fine-tune at 2e-5, 2 epochs | new reader code, check transformers on molab; 2 to 3 GPU-h | R-001 licence cleared; R-002 gave it the largest VLM weight (6+6) | a third family; +0.002 to +0.006 |
| 6 | Second Zindi submission | test runs for the final voters lacking test readings (EXP-018 CTC, 4B or its recipe version, EXP-030), about 2 GPU-h; then a vote spec with test_from | FND-007 | Zindi ~ validation |

Parked: nine views (first place's costliest variant, 21 GPU-h a fold; revisit for tall scans only if step 4 works,
since it needs the same word boundaries); Kraken (our two CTC readers fill its place, and the CTC 10-best as a voter
lost in EXP-027); more 8B epochs (EXP-019); vote weight tuning on the current readings (ten tries, stopped).
