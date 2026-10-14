# FND-001 Test lines are a random quarter of the same scans as training lines

- Measured on all 5,472 images (4,098 train, 1,374 test). Only the test images' sizes and pixels were used.
- A classifier asked to tell test from train scores AUC 0.496 on size, file size, paper tone and ink features, and
  0.518 on a ResNet-18 image embedding. Chance is 0.50; the same classifier on shuffled labels gives 0.490.
- Grouping images into 20 clusters of look-alike scans: test share runs 19% to 29% per cluster (overall 25.1%), and
  none of the 20 is outside what chance gives. With 100 clusters, 2 of 100 are outside the 95% band (5 expected by
  chance). No cluster is test-only.
- A test line's nearest look-alike is another test line 25.2% of the time, exactly the chance rate (25.1%).
- Tall scans (over 150 px) are 25.5% of train (1,043 lines) and 25.1% of test (345 lines).
- 4,098 : 1,374 is 74.9 : 25.1. This looks like a plain 75/25 shuffle at line level (inference).

**Consequences:** validation folds can be random at line level. Holding out pages, writers or clusters would make
validation harder than the real test and would mislead us. Test lines share pages and hands with training lines, so
reading test images without labels (pseudo-labelling, allowed) has no domain gap to close.
([[D-001 Five random folds, twins kept together, balanced on scan height]], [[H-003 Self-training on test images adds little]])

Charts: `docs/eda/01_train_and_test_look_alike.png`, `02_no_cluster_is_test_only.png`, `03_...`, `04_...`.
Source: `notebooks/eda.py` section `q1`; numbers in `docs/eda/numbers.txt`. Checked 2026-10-13.
