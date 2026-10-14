"""Character language model and CTC beam search.

The model is a character n-gram with Witten-Bell smoothing, built from the training transcripts of one fold only.
The beam search is the standard CTC prefix search with the language model added to the score:
    log p(ctc) + weight * log p(lm) + bonus * characters
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

BOS, EOS = "\x02", "\x03"


class CharLM:
    def __init__(self, texts, order: int = 6):
        self.n = order
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]  # [len(h)][h][c]
        vocab = set()
        for t in texts:
            s = BOS * (order - 1) + t + EOS
            vocab.update(t)
            for i in range(order - 1, len(s)):
                for k in range(order):
                    self.counts[k][s[i - k:i]][s[i]] += 1
        self.vocab = sorted(vocab) + [EOS]
        self.totals = [{h: (sum(d.values()), len(d)) for h, d in ck.items()} for ck in self.counts]
        self.memo: dict = {}

    def prob(self, h: str, c: str) -> float:
        h = h[-(self.n - 1):] if self.n > 1 else ""
        key = (h, c)
        if key in self.memo:
            return self.memo[key]
        p = 1.0 / (len(self.vocab) + 1)
        for k in range(0, len(h) + 1):
            ctx = h[len(h) - k:] if k else ""
            tot = self.totals[k].get(ctx)
            if not tot:
                break
            n, types = tot
            p = (self.counts[k][ctx].get(c, 0) + types * p) / (n + types)
        self.memo[key] = p
        return p

    def logp(self, h: str, c: str) -> float:
        return math.log(self.prob(BOS * (self.n - 1) + h, c))


def _lse(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_decode(logp: np.ndarray, charset: list[str], lm: CharLM | None, weight: float = 0.5, bonus: float = 0.0,
                beam: int = 10, topk: int = 8, prune: float = -12.0) -> str:
    """logp: [T, 1 + len(charset)] log-probabilities, blank at index 0."""
    beams = {"": (0.0, -math.inf, 0.0)}  # prefix -> (log p blank-ending, log p non-blank-ending, lm log p)
    for t in range(logp.shape[0]):
        row = logp[t]
        cand = [i for i in np.argsort(row)[::-1][:topk] if row[i] > prune]
        nxt: dict[str, list[float]] = {}

        def add(prefix, pb, pnb, lmv):
            if prefix in nxt:
                o = nxt[prefix]
                o[0], o[1] = _lse(o[0], pb), _lse(o[1], pnb)
            else:
                nxt[prefix] = [pb, pnb, lmv]

        for prefix, (pb, pnb, lmv) in beams.items():
            tot = _lse(pb, pnb)
            for i in cand:
                p = float(row[i])
                if i == 0:
                    add(prefix, tot + p, -math.inf, lmv)
                    continue
                c = charset[i - 1]
                if prefix and prefix[-1] == c:
                    add(prefix, -math.inf, pnb + p, lmv)        # repeat collapses
                    new_p = pb + p                              # a blank in between makes a new letter
                else:
                    new_p = tot + p
                if new_p == -math.inf:
                    continue
                extra = lm.logp(prefix, c) if lm else 0.0
                add(prefix + c, -math.inf, new_p, lmv + extra)

        def total(item):
            prefix, (pb, pnb, lmv) = item
            return _lse(pb, pnb) + weight * lmv + bonus * len(prefix)

        beams = {k: tuple(v) for k, v in sorted(nxt.items(), key=total, reverse=True)[:beam]}

    def final(item):
        prefix, (pb, pnb, lmv) = item
        end = weight * lm.logp(prefix, EOS) if lm else 0.0
        return _lse(pb, pnb) + weight * lmv + end + bonus * len(prefix)

    return max(beams.items(), key=final)[0]


def greedy_decode(logp: np.ndarray, charset: list[str]) -> str:
    best = logp.argmax(-1)
    out, prev = [], 0
    for i in best:
        if i != prev and i != 0:
            out.append(charset[i - 1])
        prev = i
    return "".join(out)
