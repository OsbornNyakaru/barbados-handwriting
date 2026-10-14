# EXP-002 The first CTC reader never learned because the LSTM stalled training

2026-10-14, molab RTX Pro 6000. Spec: CRNN from scratch, height 80, 60 epochs, greedy (H-001).

- **Result:** −0.133 on fold 0 (worse than an empty answer, −0.036). Loss flat at 3.15 from epoch 2; output mostly
  blank with strings such as "a e e e e"; 89% of columns blank. Folds 1 to 3 the same (two runners repeated it).
- **Cause, found on the Mac:** on 8 rendered lines the network without its LSTM memorised them in about 100 steps;
  with the LSTM it barely moved in 300. On 500 rendered lines the full pipeline read 0 of 100 unseen lines exactly.
  The stacked bidirectional LSTM takes un-normalised projected features and starves the convolution layers of
  gradient, so CTC never leaves its blank-only start.
- **Fix:** `params.rnn_skip` (LayerNorm before the LSTM, a shortcut around it). Same tests: 8 lines memorised (loss
  0.02); 87 of 100 unseen rendered lines exact, 1.8% character errors, with augmentation on. Default off, so old
  specs reproduce. A test now guards it (`test_ctc_reader_memorises_a_few_lines_with_the_lstm_shortcut`).
- **EXP-004** was meant to switch mixed precision off, but `--set epochs=30 --set amp=false` wrote the keys at the
  top level, where no reader looks, so it repeated EXP-002. `exp.py new` now refuses unknown top-level keys.
- Next: EXP-005 = EXP-002 + `rnn_skip`, all 5 folds.
