# FND-002 Only 4% of training lines repeat another line, and no two images are copies

- Text, 4,098 training lines: 23 lines share their exact text (10 groups); 39 once case and spacing are ignored.
  At 85% alike or more: 167 lines in 40 groups, the largest 34 lines (one witnessing formula
  and its variants). At 70%: 585 lines in 131 groups.
- Images, all 5,472: no pair is closer than 71 of 256 hash bits (unrelated pairs sit near 128; a true copy would be
  under 20). The 50 most look-alike training pairs by embedding are all different lines (0 of 50 share text; median
  text likeness 24%). They are neighbouring lines in the same hand.
- So the same words appear in different documents, but no line image was cropped or scanned twice.
- A held-out fold has 10.1% of its word tokens never seen in the other four folds (9.4% to 10.7% by fold). Of 6,750
  distinct words, 62% occur once.
- The first 42 rows of `Train.csv` are all members of repeat groups, laid out group by group. The other 4,056 rows
  are shuffled and hold 125 grouped lines. The organisers seem to have handled some repeats separately; whether they
  kept repeats out of the test set is unknown and cannot be checked without reading test lines.

**Consequences:** lines 85% alike or more stay in one fold; nothing else needs grouping. One word in ten being new
means a reader must spell from the letters, and snapping to a training word list cannot work (an entrant and the
winner both measured that). ([[D-001 Five random folds, twins kept together, balanced on scan height]],
[[H-001 A small letter-by-letter reader keeps old spellings]])

Charts: `docs/eda/05_most_alike_training_images.png`, `06_repeated_text_is_rare.png`. Source: `notebooks/eda.py` section `q2`; numbers in `docs/eda/numbers.txt`. Checked 2026-10-13.
