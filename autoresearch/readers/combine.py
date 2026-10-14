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
