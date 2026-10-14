# FND-006 The CTC reader works best as a judge between vision-language readings

Fold 0 (804 lines), 2026-10-14. Readings: Qwen3-VL-8B twice (EXP-006, two runners repeated the fold: 0.885 and
0.894), Qwen3-VL-2B (EXP-003, 0.882), CTC from scratch with the character language model (EXP-005 + LM, 0.862).

| combination | fold 0 |
|---|--:|
| best single reading | 0.894 |
| choose per line between the two 8B runs, by CTC log p + 0.3 x LM log p | 0.902 |
| the same over all three Qwen readings | **0.907** (halves 0.908 / 0.906) |
| ... adding the CTC beam reading as a fourth candidate (penalty 2 to 10) | 0.864 to 0.878 |
| swap CTC words into the 8B reading (merge, settings from half of fold 0) | 0.892 |
| oracle: best of CTC and 8B per line | 0.905 |

- The CTC evidence is a good judge between readings it did not produce, and a biased one for its own reading.
- Two runs of the same Qwen spec differ by 0.009 on one fold and differ line by line, so repeated runs are useful
  candidates. Qwen-8B against Qwen-2B (+0.003 on one run) is inside that noise.
- Judge settings were not tuned on these lines (w 0.3 is the decoding weight), so 0.907 is a fair fold-0 estimate.

**What it changes:** spend GPU time on more and different vision-language readings per line (n-best lists, seeds,
LightOnOCR-2), and on a stronger CTC judge (ConvNeXt, taller images). EXP-008: Qwen3-VL-2B with 5-best readings.

**Update 2026-10-14 17:05 EAT, folds 0 to 3 (3,283 lines):** judge over Qwen3-VL-8B (two runs) 0.9037; adding
Qwen3-VL-2B's top reading from two runs (EXP-008) 0.9104 (+0.007); adding its 5-best lists as well 0.9106 (+0.0002).
Beam-search alternatives from one model are too alike to help; separately trained runs and models are what add
useful candidates. So GPU time goes to more trained runs (EXP-011: Qwen3-VL-4B), not to n-best decoding.
