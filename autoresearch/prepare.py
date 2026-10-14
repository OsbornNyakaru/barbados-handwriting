"""prepare.py — FROZEN harness: data loading, validation folds, metric, submission format.

Do not edit after the first recorded experiment: every result in experiments/results.jsonl is only comparable
because this file is the same. `scripts/exp.py` stores this file's hash with each result and refuses to compare
across hashes.

Competition: R.O.A.D. Barbados Historic Handwriting Challenge (Zindi, closed 5 Oct 2026; practice only).
  score = 1 - 0.5 * (W / 12 + C / 55), higher is better
  W = word edits per line, C = character edits per line (Levenshtein, against the reference transcript).
  The way W and C combine is confirmed against the leaderboard (vault/facts/F-002). How they are averaged over
  lines is not: entrants say a plain mean, the winner says weights of sqrt(reference length). Both are computed.
  The plain mean is "score" and drives adoption; the other is reported as "score_sqrt".
  Text is compared after stripping spaces at the two ends; words are split on any run of spaces.

Validation: the test set is a random quarter of the same scans (vault/findings/FND-001), so folds are random at
line level. Lines whose text is 85% alike or more stay in one fold, and folds are balanced on the scan-height batch
(vault/decisions/D-001). One fixed split of 5 folds, because a reader takes GPU-hours to train. A run's score is
the score of the pooled out-of-fold predictions for all 4,098 lines; the five per-fold scores are kept for paired
comparison and a bootstrap over lines gives the spread.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMAGES = RAW / "images"

ID_COL = "ID"
TARGET = "Target"
GROUP_COL = "group"   # lines with the same or nearly the same text share a group
BATCH_COL = "tall"    # True for scans taller than TALL_PX
HIGHER_IS_BETTER = True
N_FOLDS, N_REPEATS, CV_SEED = 5, 1, 20261013
TALL_PX = 150
TWIN_SIMILARITY = 0.85
WORD_NORM, CHAR_NORM = 12.0, 55.0
N_BOOT, BOOT_SEED = 1000, 20261013
MAX_PRED_CHARS = 300
FOLD_HASH = "6d2b14640424"
VALIDATION = (f"{N_FOLDS} random folds (one fixed split), near-identical transcripts kept together, balanced on scan "
              f"height. Score = 1 - 0.5 * (word edits per line / 12 + character edits per line / 55) on the pooled "
              f"out-of-fold predictions; ± is a bootstrap over lines.")


def clean(text) -> str:
    """A transcript as it is compared: a string with no spaces at either end. Missing becomes empty."""
    return "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text).strip()


def norm_text(text: str) -> str:
    """Lower case, single spaces. Used only to find twins for the folds."""
    return " ".join(clean(text).lower().split())


def _groups(texts: list[str]) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    sim = process.cdist(texts, texts, scorer=Levenshtein.normalized_similarity, score_cutoff=TWIN_SIMILARITY,
                        workers=1, dtype=np.float32)
    i, j = np.where(sim >= TWIN_SIMILARITY)
    return connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=sim.shape), directed=False)[1]


def _sizes(ids) -> pd.DataFrame:
    from PIL import Image

    out = []
    for i in ids:
        with Image.open(IMAGES / f"{i}.jpg") as im:
            out.append(im.size)
    return pd.DataFrame(out, columns=["width", "height"])


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) in file order. Both get image (path), width, height and tall; train also gets group."""
    train = pd.read_csv(RAW / "Train.csv", keep_default_na=False)
    test = pd.read_csv(RAW / "Test.csv", encoding="utf-8-sig")  # the file starts with a byte-order mark
    for df in (train, test):
        df["image"] = [str(IMAGES / f"{i}.jpg") for i in df[ID_COL]]
        df[["width", "height"]] = _sizes(df[ID_COL]).to_numpy()
        df[BATCH_COL] = df["height"] > TALL_PX
    train[GROUP_COL] = _groups([norm_text(t) for t in train[TARGET]])
    return train, test


def folds(train: pd.DataFrame) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """N_REPEATS lists of N_FOLDS (train positions, validation positions)."""
    from sklearn.model_selection import StratifiedGroupKFold

    y, g = train[BATCH_COL].to_numpy(), train[GROUP_COL].to_numpy()
    return [list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED + r).split(train, y, g))
            for r in range(N_REPEATS)]


def fold_of(train: pd.DataFrame) -> np.ndarray:
    """The held-out fold number of every training line."""
    out = np.full(len(train), -1)
    for k, (_, held) in enumerate(folds(train)[0]):
        out[held] = k
    return out


def fold_hash(train: pd.DataFrame) -> str:
    text = "\n".join(f"{i},{k}" for i, k in zip(train[ID_COL], fold_of(train)))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def line_edits(y_true, pred) -> pd.DataFrame:
    """Per line: word edits, character edits, and the reference's word and character counts."""
    rows = []
    for ref, hyp in zip(y_true, pred):
        ref, hyp = clean(ref), clean(hyp)
        rw, hw = ref.split(), hyp.split()
        rows.append((Levenshtein.distance(rw, hw), Levenshtein.distance(ref, hyp), len(rw), len(ref)))
    return pd.DataFrame(rows, columns=["word_edits", "char_edits", "n_words", "n_chars"])


def _combine(e: pd.DataFrame, sqrt_weights: bool = False) -> tuple[float, float, float]:
    if sqrt_weights:
        w = np.average(e.word_edits, weights=np.sqrt(e.n_words))
        c = np.average(e.char_edits, weights=np.sqrt(e.n_chars))
    else:
        w, c = e.word_edits.mean(), e.char_edits.mean()
    return float(1 - 0.5 * (w / WORD_NORM + c / CHAR_NORM)), float(w), float(c)


def score(y_true, pred) -> dict:
    """{"score": combined, "parts": {name: value}} with the competition's metric (both averaging variants)."""
    e = line_edits(y_true, pred)
    s, w, c = _combine(e)
    return {"score": s, "parts": {"word_edits": w, "char_edits": c, "score_sqrt": _combine(e, sqrt_weights=True)[0]}}


def cv_score(train: pd.DataFrame, oof: np.ndarray) -> dict:
    """oof: (1, n rows) out-of-fold predictions. Pooled score, bootstrap spread, per-fold scores, per-batch scores.

    "per_repeat" holds the five per-fold scores: with one fixed split they are the paired units exp.py compares.
    """
    pred = np.asarray(oof)[0]
    e = line_edits(train[TARGET].to_numpy(), pred)
    s, w, c = _combine(e)
    tall, fold = train[BATCH_COL].to_numpy(), fold_of(train)
    rng = np.random.default_rng(BOOT_SEED)
    we, ce = e.word_edits.to_numpy(), e.char_edits.to_numpy()
    idx = rng.integers(0, len(e), size=(N_BOOT, len(e)))
    boot = 1 - 0.5 * (we[idx].mean(axis=1) / WORD_NORM + ce[idx].mean(axis=1) / CHAR_NORM)
    return {"score": s, "std": float(boot.std(ddof=1)),
            "parts": {"word_edits": w, "char_edits": c, "score_sqrt": _combine(e, sqrt_weights=True)[0],
                      "short": _combine(e[~tall])[0], "tall": _combine(e[tall])[0]},
            "per_repeat": [round(_combine(e[fold == k])[0], 6) for k in range(N_FOLDS)]}


def build_submission(test: pd.DataFrame, pred) -> pd.DataFrame:
    """ID and Target, in test order. The only step applied is stripping spaces at the two ends."""
    return pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET: [clean(p) for p in pred]})


def validate_submission(sub: pd.DataFrame) -> None:
    sample = pd.read_csv(RAW / "SampleSubmission.csv")
    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(sample) and set(sub[ID_COL]) == set(sample[ID_COL]) and sub[ID_COL].is_unique, "IDs differ from SampleSubmission"
    assert sub.notna().all().all(), "missing predictions"
    text = sub[TARGET]
    assert text.map(lambda t: isinstance(t, str)).all(), "predictions must be text"
    assert (text.str.strip().str.len() > 0).all(), "empty predictions count as wrong: give every line some text"
    assert not text.str.contains(r"[\r\n\t]").any(), "a prediction is one line: no line breaks or tabs"
    assert (text.str.len() <= MAX_PRED_CHARS).all(), f"a prediction longer than {MAX_PRED_CHARS} characters is a runaway (references end at 120)"


def harness_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


if __name__ == "__main__":
    tr, te = load()
    print(f"train {tr.shape}, test {te.shape}")
    print("fold sizes", np.bincount(fold_of(tr)).tolist(), "fold hash", fold_hash(tr), "expected", FOLD_HASH)
    print("harness", harness_hash())
