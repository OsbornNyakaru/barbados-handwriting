# EXP-003 Qwen3-VL-2B reads fold 0 at 0.882

2026-10-14, molab RTX Pro 6000 (95 GB). Spec: `Qwen/Qwen3-VL-2B-Instruct` (Apache 2.0), LoRA r 16 on 300 modules
(104 in the vision tower, 24.6M trainable), 3 epochs, height 128, greedy. Fold 0 only (804 lines), a smoke test (H-004).

- **Score 0.8818** (± 0.0043 bootstrap): 2.06 word edits and 3.54 character edits per line. Short scans 0.902,
  tall 0.824 (the same gap an entrant reported: 0.900 against 0.823, thread 34861).
- Training loss 0.81 → 0.40 → 0.25 over 3 epochs (5.6 min each); reading 804 lines took 58 s. 18.4 min in all.
- Against the research estimate for the 2B (0.82 to 0.88): at the top. The 8B (estimated 0.85 to 0.91) is next.
- Fold 0 only, so not eligible for adoption.
