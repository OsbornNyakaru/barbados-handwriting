"""Combine readers: a judge that picks, for each line, the candidate reading that best fits the image.

The judge is the CTC reader's own evidence: the CTC probability of a candidate string given the line's saved
log-probabilities (how well the letters line up with what the reader saw), plus the character language model, plus a
small length bonus. Candidates are the CTC beam reading and every vision-language reading of the same line. A
vision-language reading that modernised a spelling scores badly here, because the image does not support the letters
it changed; a CTC reading that misread a smudged letter loses to a fluent reading the image supports equally well.
"""

from __future__ import annotations

import numpy as np

from .lm import CharLM, beam_decode


def ctc_logp(logp: np.ndarray, texts: list[str], charset: list[str]) -> np.ndarray:
    """log p(text | image) under CTC for each text; -inf when a text has a letter the reader cannot write."""
    import torch
    import torch.nn.functional as F

    index = {c: i + 1 for i, c in enumerate(charset)}
    out = np.full(len(texts), -np.inf)
    ok = [i for i, t in enumerate(texts) if t and all(c in index for c in t) and len(t) <= logp.shape[0]]
    if not ok:
        return out
    lp = torch.from_numpy(logp.astype(np.float32)).unsqueeze(1).expand(-1, len(ok), -1)
    tg = torch.cat([torch.tensor([index[c] for c in texts[i]]) for i in ok])
    nll = F.ctc_loss(lp, tg, torch.full((len(ok),), logp.shape[0], dtype=torch.long),
                     torch.tensor([len(texts[i]) for i in ok]), blank=0, reduction="none", zero_infinity=False)
    out[ok] = -nll.numpy()
    return out


def lm_logp(lm: CharLM, text: str) -> float:
    from .lm import EOS

    return sum(lm.logp(text[:i], c) for i, c in enumerate(text)) + lm.logp(text, EOS)


def judge(logp: np.ndarray, charset: list[str], others: list[str], lm: CharLM, weight: float, bonus: float,
          beam: int, ctc_beam: str | None = None) -> tuple[str, int]:
    """(chosen text, index of the source: 0 = CTC beam, 1.. = the other readers in order)."""
    first = ctc_beam if ctc_beam is not None else beam_decode(logp, charset, lm, weight, bonus, beam)
    cands = [first] + [" ".join(str(o).split()) for o in others]
    s = ctc_logp(logp, cands, charset) + np.array([weight * lm_logp(lm, c) + bonus * len(c) for c in cands])
    k = int(np.argmax(s))
    return cands[k], k


def merge_words(logp: np.ndarray, charset: list[str], base: str, other: str, lm: CharLM, a: float, w: float,
                penalty: float, max_spans: int = 12) -> str:
    """Start from `base` (a vision-language reading) and swap in `other`'s words (the CTC beam reading) span by span,
    left to right, wherever the swap raises a * CTC log p + w * LM log p - penalty * swaps. The penalty is the prior
    for trusting the base reader; `a` softens the CTC evidence, which otherwise favours its own beam reading."""
    from rapidfuzz.distance import Levenshtein

    bw, ow = base.split(), other.split()
    spans = [(o.src_start, o.src_end, ow[o.dest_start:o.dest_end]) for o in Levenshtein.opcodes(bw, ow) if o.tag != "equal"]

    def build(choice):
        out, i = [], 0
        for (s, e, rep), c in zip(spans, choice):
            out += bw[i:s] + (rep if c else bw[s:e])
            i = e
        return " ".join(out + bw[i:])

    def score(choice):
        t = build(choice)
        c = ctc_logp(logp, [t], charset)[0]
        return a * (c if np.isfinite(c) else -1e4) + w * lm_logp(lm, t) - penalty * sum(choice)

    choice = [False] * len(spans)
    cur = score(choice)
    for k in range(min(len(spans), max_spans)):
        trial = choice.copy()
        trial[k] = True
        s = score(trial)
        if s > cur:
            cur, choice = s, trial
    return build(choice)


def metric_loss(a: str, b: str) -> float:
    """The competition's cost of writing `a` when the line is `b`: word edits / 12 + character edits / 55."""
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(a.split(), b.split()) / 12.0 + Levenshtein.distance(a, b) / 55.0


def vote(voters: list[tuple[str, float]], rounds: int = 3, max_len: int = 10**9) -> str:
    """Weighted vote (first place's idea, R-002): the line with the lowest weighted metric cost against every reading
    (minimum expected loss). It starts from the best whole reading, then tries swapping one differing word span for
    another reader's words there, keeping a swap while the cost falls, so the answer can be a line no reader wrote.
    Readings longer than max_len (runaway repeats) may vote but are never the answer."""
    from rapidfuzz.distance import Levenshtein

    voters = [(" ".join(str(r).split()), w) for r, w in voters if w > 0]
    cost: dict[str, float] = {}

    def c_of(c):
        if c not in cost:
            cost[c] = sum(w * metric_loss(c, r) for r, w in voters)
        return cost[c]

    ok = sorted({r for r, _ in voters if r and len(r) <= max_len})
    if not ok:
        return min((r for r, _ in voters), key=len, default="")
    best = min(ok, key=c_of)
    for _ in range(rounds):
        bw, cands = best.split(), set()
        for r in ok:
            rw = r.split()
            for o in Levenshtein.opcodes(bw, rw):
                if o.tag != "equal":
                    cands.add(" ".join(bw[:o.src_start] + rw[o.dest_start:o.dest_end] + bw[o.src_end:]))
        nb = min(sorted(cands | {best}), key=c_of)
        if nb == best:
            break
        best = nb
    return best


def choose(logp, charset, candidates: list[str], lm: CharLM, w: float) -> tuple[str, int]:
    """Pick the candidate (vision-language readings only) with the highest CTC log p + w * LM log p. The CTC reader is
    a judge here, never a candidate: its own beam reading wins almost every time under its own evidence (fold 0,
    2026-10-14: adding it dropped 0.907 to 0.865-0.878)."""
    cands = list(dict.fromkeys(" ".join(str(c).split()) for c in candidates if str(c).strip())) or [""]
    # several CTC judges (a list of log-prob arrays with their charsets): their log p are added
    pairs = list(zip(logp, charset)) if isinstance(logp, list) else [(logp, charset)]
    s = sum(np.where(np.isfinite(x), x, -1e4) for x in (ctc_logp(lp, cands, cs) for lp, cs in pairs))
    s = s + w * np.array([lm_logp(lm, c) for c in cands])
    k = int(np.argmax(s))
    return cands[k], k
