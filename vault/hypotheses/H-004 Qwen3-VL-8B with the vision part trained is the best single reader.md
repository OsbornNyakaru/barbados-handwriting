# H-004 Qwen3-VL-8B with the vision part trained is the best single reader

Status: untested. From [[R-001 Which pretrained models to start from]].

- Idea: `Qwen/Qwen3-VL-8B-Instruct` (Apache 2.0), LoRA on the language model and the vision tower, 2 epochs,
  learning rate 2e-4, one view, greedy decoding, scores higher than any small reader alone.
- Evidence so far: the winner's recipe (forum thread 35246); freezing the encoder hurt in two studies of
  TrOCR-style readers. Against: it modernises spellings, and a small CTC reader beat a 9B line reader on medieval text.
- Test: one fold of Qwen3-VL-2B first as a smoke test; then 8B on 5 folds; then one fold with the vision part frozen.
- Cost: 8 to 15 GPU-hours on a 48 GB card (estimate).
