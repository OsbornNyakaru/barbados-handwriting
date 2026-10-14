"""errors.py — EDA questions 4 to 6 on a reader's held-out predictions.

    .venv/bin/python notebooks/errors.py EXP-002

Q4  label noise: the lines the reader gets most wrong, for reading by eye (written to experiments/runs/<EXP>/,
    which is git-ignored, because it quotes transcripts; the repository is public).
Q5  score noise: bootstrap spread at 820 lines (one fold) and 4,098 lines (all folds), and the fold-to-fold spread.
Q6  error types: each word edit classed as casing, markup, punctuation, digits, missing or extra word, or spelling.
Only counts are printed, so the printout can be quoted in the vault.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))
import prepare as P  # noqa: E402

MARKUP = set("^~:&|\\_")


def word_ops(ref: str, hyp: str):
    """(kind, ref word, hyp word) for every word-level edit."""
    r, h = ref.split(), hyp.split()
    for op in Levenshtein.opcodes(r, h):
        if op.tag == "equal":
            continue
        rw, hw = r[op.src_start:op.src_end], h[op.dest_start:op.dest_end]
        for k in range(max(len(rw), len(hw))):
            yield op.tag, (rw[k] if k < len(rw) else ""), (hw[k] if k < len(hw) else "")


def kind(tag: str, a: str, b: str) -> str:
    if tag == "delete" or not b:
        return "missing word"
    if tag == "insert" or not a:
        return "extra word"
    if a.lower() == b.lower():
        return "casing"
    strip = lambda s, chars: "".join(c for c in s if c not in chars)  # noqa: E731
    if strip(a, MARKUP) == strip(b, MARKUP):
        return "markup (^ ~ : & ...)"
    if re.sub(r"[^\w]", "", a) == re.sub(r"[^\w]", "", b):
        return "punctuation"
    if any(c.isdigit() for c in a + b):
        return "digits"
    if Levenshtein.distance(a.lower(), b.lower()) == 1:
        return "spelling, one letter"
    return "spelling, more"


def main(exp: str) -> None:
    train, _ = P.load()
    fold = P.fold_of(train)
    d = ROOT / "experiments" / "runs" / exp
    pred = pd.concat([pd.read_csv(f, keep_default_na=False, dtype=str) for f in sorted(d.glob("fold?.csv"))])
    m = train.assign(fold=fold).merge(pred, on=P.ID_COL)
    m["ref"] = m[P.TARGET].map(P.clean)
    e = P.line_edits(m["ref"].to_numpy(), m["pred"].to_numpy())
    m = pd.concat([m.reset_index(drop=True), e.reset_index(drop=True)], axis=1)
    m["line"] = 1 - 0.5 * (m["word_edits"] / P.WORD_NORM + m["char_edits"] / P.CHAR_NORM)
    m["cer"] = m["char_edits"] / m["n_chars"].clip(lower=1)
    print(f"{exp}: {len(m)} held-out lines, folds {sorted(m['fold'].unique().tolist())}")
    print(f"score {m['line'].mean():.4f}; word edits/line {m['word_edits'].mean():.3f}; char edits/line {m['char_edits'].mean():.3f}")
    for t, g in m.groupby(P.BATCH_COL):
        print(f"  {'tall' if t else 'short'}: {len(g)} lines, score {g['line'].mean():.4f}, exact {np.mean(g['ref'] == g['pred']):.3f}")

    print("\nQ4 label noise")
    worst = m.sort_values("cer", ascending=False)
    for cut in (0.5, 0.3, 0.2):
        n = int((m["cer"] > cut).sum())
        print(f"  lines with character error rate > {cut}: {n} ({n / len(m):.1%}), "
              f"of which tall {int(((m['cer'] > cut) & m[P.BATCH_COL]).sum())}")
    top = worst.head(int((m["cer"] > 0.3).sum()))
    print(f"  the {len(top)} worst lines hold {top['char_edits'].sum() / m['char_edits'].sum():.1%} of all character edits")
    out = d / "worst_lines.md"
    with out.open("w") as f:
        f.write(f"# {exp}: worst held-out lines (local only: quotes transcripts)\n\n")
        for _, r in worst.head(60).iterrows():
            f.write(f"- `{r[P.ID_COL]}` fold {r['fold']} {'tall' if r[P.BATCH_COL] else 'short'} {r['width']}x{r['height']} "
                    f"CER {r['cer']:.2f}\n  - ref:  {r['ref']}\n  - pred: {r['pred']}\n")
    print(f"  written for reading by eye: {out.relative_to(ROOT)}")

    print("\nQ5 score noise")
    rng = np.random.default_rng(P.BOOT_SEED)
    s = m["line"].to_numpy()
    for n in (820, len(s)):
        print(f"  bootstrap sd at {n} lines: {np.std([s[rng.integers(0, len(s), n)].mean() for _ in range(2000)]):.4f}")
    per = m.groupby("fold")["line"].mean()
    print(f"  per-fold scores: {', '.join(f'{x:.4f}' for x in per)}; sd {per.std(ddof=1):.4f}")

    print("\nQ6 error types (share of word edits)")
    c = Counter(kind(*op) for r, h in zip(m["ref"], m["pred"]) for op in word_ops(r, h))
    tot = sum(c.values())
    for k, v in c.most_common():
        print(f"  {k:24s} {v:6d}  {v / tot:6.1%}")
    by_len = m.groupby(pd.cut(m["n_words"], [0, 8, 11, 14, 30]), observed=True)["line"].agg(["mean", "size"])
    print("\n  score by words per line:\n" + by_len.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "EXP-002")
