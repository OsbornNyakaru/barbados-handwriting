"""experiment.py — the one file an experiment edits: inputs and model behind a config dict.

A new idea becomes a new config key whose default leaves every earlier result unchanged, so old specs still
reproduce. `scripts/exp.py new --set key=value` then changes exactly one key per experiment.

Models:
  "medoid"      no reading at all: every line gets the one training transcript that is closest, on the competition's
                edit costs, to the other training transcripts. It proves the pipeline and sets the floor. Runs here.
  "ctc"         letter-by-letter reader (readers/ctc.py): from scratch ("crnn") or on a timm backbone. GPU, molab.
  "vlm"         vision-language reader (readers/vlm.py): Qwen3-VL with LoRA. GPU, molab.
  "ctc_decode"  re-decodes the held-out log-probabilities a "ctc" run saved (params.logits_from), for example with a
                character language model. No training, so it runs here on the CPU.
GPU models run fold by fold through `run_fold`, called by autoresearch/remote.py on molab.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

import prepare as P

DEFAULT_CONFIG = {
    "features": ["image", "width", "height", "tall"],   # columns handed to the model; the image is read by the model
    "model": "medoid",
    "params": {"candidates": 300, "sample": 300},
    "seed": 0,
}
GPU_MODELS = {"ctc", "vlm"}
RUNS = Path(__file__).resolve().parents[1] / "experiments" / "runs"


def model_defaults(model: str) -> dict:
    """Every parameter of a model with its default, so a spec stores them all and stays reproducible."""
    if model == "medoid":
        return dict(DEFAULT_CONFIG["params"])
    if model == "ctc":
        from readers.ctc import DEFAULTS
        return dict(DEFAULTS)
    if model == "vlm":
        from readers.vlm import DEFAULTS
        return dict(DEFAULTS)
    if model == "ctc_decode":
        return {"logits_from": "", "decode": "beam", "lm_order": 6, "lm_weight": 0.5, "lm_bonus": 0.0, "beam": 10}
    raise ValueError(f"unknown model {model!r}")


def complete(cfg: dict) -> dict:
    """Fill in the model's missing parameters and drop ones it does not take."""
    d = model_defaults(cfg["model"])
    return {**cfg, "params": {k: cfg.get("params", {}).get(k, v) for k, v in d.items()}}


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Inputs built from the row alone, the same for train and test. Never reads the target."""
    return df[[P.ID_COL, "image", "width", "height", P.BATCH_COL]].reset_index(drop=True)


def _medoid(texts, candidates: int, sample: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    texts = [P.clean(t) for t in texts]
    cand = rng.choice(len(texts), size=min(candidates, len(texts)), replace=False)
    ref = rng.choice(len(texts), size=min(sample, len(texts)), replace=False)
    ref_words = [texts[r].split() for r in ref]
    cost = [np.mean([Levenshtein.distance(rw, texts[c].split()) / P.WORD_NORM + Levenshtein.distance(texts[r], texts[c]) / P.CHAR_NORM
                     for r, rw in zip(ref, ref_words)]) for c in cand]
    return texts[cand[int(np.argmin(cost))]]


def fit_predict(cfg: dict, X_train, y_train, X_apply, seed: int, ctx=None):
    """Fit on one training part and predict another. Everything learned from the target is learned here, inside
    the fold, so validation rows never inform their own prediction. Returns (predictions, extras)."""
    m, prm = cfg["model"], cfg["params"]
    if m == "medoid":
        return np.array([_medoid(y_train, prm["candidates"], prm["sample"], seed)] * len(X_apply), dtype=object), {}
    if m == "ctc":
        from readers import ctc
        y = [P.clean(t) for t in y_train]
        preds, extras = ctc.fit_predict(prm, list(X_train["image"]), y, list(X_apply["image"]), seed, ctx)
        return np.array(preds, dtype=object), extras
    if m == "vlm":
        from readers import vlm
        y = [P.clean(t) for t in y_train]
        preds, extras = vlm.fit_predict(prm, list(X_train["image"]), y, list(X_apply["image"]), seed, ctx)
        return np.array(preds, dtype=object), extras
    raise ValueError(f"unknown model {m!r}")


def _decode_saved(cfg: dict, train: pd.DataFrame, rep) -> np.ndarray:
    """ctc_decode: read each fold's saved log-probabilities and decode them again."""
    from readers.lm import CharLM, beam_decode, greedy_decode

    prm = cfg["params"]
    src = RUNS / prm["logits_from"]
    out = np.empty(len(train), dtype=object)
    for k, (a, b) in enumerate(rep):
        z = np.load(src / f"fold{k}_logp.npz", allow_pickle=True)
        ids, charset = list(z["ids"]), list(z["charset"])
        assert ids == list(train[P.ID_COL].iloc[b]), f"fold {k}: saved log-probabilities are for other lines"
        lm = CharLM([P.clean(t) for t in train[P.TARGET].iloc[a]], prm["lm_order"]) if prm["decode"] == "beam" else None
        out[b] = [beam_decode(lp, charset, lm, prm["lm_weight"], prm["lm_bonus"], prm["beam"]) if lm else
                  greedy_decode(lp, charset) for lp in z["logp"]]
    return out


class _LocalCtx:
    cache: dict = {}
    work_dir = Path("/tmp")

    @staticmethod
    def log(msg):
        print(msg, flush=True)

    @staticmethod
    def should_stop():
        return False


def run(cfg: dict, train: pd.DataFrame, test: pd.DataFrame, folds) -> tuple[np.ndarray, np.ndarray]:
    """Local runs (CPU models). (out-of-fold predictions [repeats, train rows], test predictions [test rows])."""
    if cfg["model"] in GPU_MODELS:
        raise RuntimeError(f"model {cfg['model']!r} needs a GPU: queue it with `exp.py submit` and run it on molab")
    if cfg["model"] == "ctc_decode":
        return np.stack([_decode_saved(cfg, train, rep) for rep in folds]), np.array([""] * len(test), dtype=object)
    X, Xt, y = features(train), features(test), train[P.TARGET].to_numpy()
    oof = np.empty((len(folds), len(train)), dtype=object)
    for r, rep in enumerate(folds):
        for a, b in rep:
            oof[r, b] = fit_predict(cfg, X.iloc[a], y[a], X.iloc[b], cfg["seed"] + r, _LocalCtx)[0]
    return oof, np.asarray(fit_predict(cfg, X, y, Xt, cfg["seed"], _LocalCtx)[0])


def run_fold(cfg: dict, train: pd.DataFrame, fold: int, ctx) -> tuple[pd.DataFrame, dict]:
    """One fold of the fixed split (molab). Returns the held-out predictions [ID, pred] and extras to upload."""
    a, b = P.folds(train)[0][fold]
    X, y = features(train), train[P.TARGET].to_numpy()
    preds, extras = fit_predict(cfg, X.iloc[a], y[a], X.iloc[b], cfg["seed"] + fold, ctx)
    out = pd.DataFrame({P.ID_COL: train[P.ID_COL].iloc[b].to_numpy(), "pred": [P.clean(t) for t in preds]})
    return out, extras
