# H-003 Self-training on test images adds little

Status: untested. A prediction that something will not help.

- Idea: because test lines come from the same scans as training lines, there is no gap for pseudo-labels to close;
  the only gain would be from 34% more images.
- Evidence so far: AUC 0.50 to 0.52 between train and test ([[FND-001 Test lines are a random quarter of the same scans as training lines]]);
  the winner found students "mostly copied their teacher".
- Test: low priority. Try once, late, with a held-out fold standing in for the test set.
