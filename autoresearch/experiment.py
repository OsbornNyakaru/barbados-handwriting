"""experiment.py — the one file an experiment edits: inputs and model behind a config dict.

A new idea becomes a new config key whose default leaves every earlier result unchanged, so old specs still
reproduce. `scripts/exp.py new --set key=value` then changes exactly one key per experiment.

Models:
  "medoid"      no reading at all: every line gets the one training transcript that is closest, on the competition's
                edit costs, to the other training transcripts. It proves the pipeline and sets the floor. Runs here.
  "ctc"         letter-by-letter reader (readers/ctc.py): from scratch ("crnn") or on a timm backbone. GPU, molab.
  "vlm"         vision-language reader (readers/vlm.py): Qwen3-VL with LoRA. GPU, molab.
  "combine"     CPU, from saved runs (readers/combine.py). mode "choose": for each line, pick among the readings of the
                vision-language runs in params.vlm_from (and their n-best lists) the one with the best CTC evidence
                (params.ctc_from) plus language model. mode "merge": swap CTC words into one vision-language reading.
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
    "read_test": False,   # GPU readers: each fold model also reads the test lines (fold<k>_test.*), see run_fold
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
    if model == "combine":
        return {"mode": "choose", "ctc_from": "", "vlm_from": [], "lm_order": 6, "lm_weight": 0.3, "beam": 10,
                "a": 0.2, "w": 0.3, "penalty": 1.5, "all_versions": True,
                "test_from": {},   # {held-out run: its test run}; set for a submission
                "cap_len": False,
                "vote_weights": {}, "rounds": 3, "ctc_nbest": 10}   # mode "vote" (see _vote_lines)  # drop readings longer than any training line; none left -> the CTC beam reading
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


def _capped(prm: dict, cands: list[str], max_len: int, lp, charset, lm) -> list[str]:
    """cap_len: a reading longer than every training line is a vision-language reader stuck repeating itself (one
    test line, 2026-10-15: every Qwen reading 330 to 432 letters). Drop those; with none left, the CTC beam reading."""
    if not prm.get("cap_len"):
        return cands
    from readers.lm import beam_decode

    keep = [c for c in cands if len(" ".join(str(c).split())) <= max_len]
    return keep or [beam_decode(lp, charset, lm, prm["lm_weight"], 0.0, prm["beam"])]


def _readings(d: Path, stem: str, ids: list[str], nbest: bool) -> dict[str, list[str]]:
    """Every saved reading of these lines in run folder d: <stem>.csv and its timestamped versions (<stem>_HHMM.csv,
    the same spec trained again; identical files counted once), or with nbest the <stem>_nbest.json lists."""
    import json

    out = {i: [] for i in ids}
    if nbest:
        for i, lst in json.loads((d / f"{stem}_nbest.json").read_text()).items():
            if i in out:
                out[i] += [str(x) for x in lst]
        return out
    seen = set()
    for f in [d / f"{stem}.csv"] + sorted(d.glob(f"{stem}_[0-9][0-9][0-9][0-9].csv")):
        v = pd.read_csv(f, keep_default_na=False, dtype=str).set_index(P.ID_COL)["pred"]
        key = tuple(v.get(i, "") for i in ids)
        if key in seen:
            continue
        seen.add(key)
        for i in ids:
            out[i].append(v.get(i, ""))
    return out


def _vote_lines(prm: dict, places, ids: list[str], lm, max_len: int) -> list[str]:
    """mode "vote": params.vote_weights = {source: weight}. A source is a run ("EXP-006": its readings and saved
    versions), a run's n-best lists ("EXP-008:nbest"), or "ctc": the beam reading of ctc_from's saved
    log-probabilities with the character language model (lm_weight). A source's weight is split over its readings.
    places(run) lists the (folder, file stem) pairs holding a run's readings of these lines: one for held-out lines,
    one per fold model for test lines read by the fold models (test_from "folds")."""
    from readers.combine import vote
    from readers.lm import beam_decode

    srcs = {}
    for name in prm["vote_weights"]:
        srcs[name] = {i: [] for i in ids}
        if name in ("ctc", "ctc:nbest") or name.startswith("ctc@"):
            # "ctc:nbest": ctc_from's ctc_nbest best beam readings; "ctc@EXP-NNN": the beam reading of another CTC run
            n = prm["ctc_nbest"] if name == "ctc:nbest" else 1
            for d, stem in places(name[4:] if name.startswith("ctc@") else prm["ctc_from"]):
                zz = dict(np.load(d / f"{stem}_logp.npz", allow_pickle=True).items())
                assert list(zz["ids"]) == ids, f"{name}: saved log-probabilities are for other lines"
                charset = list(zz["charset"])
                for r, i in enumerate(ids):
                    got = beam_decode(zz["logp"][r].astype(np.float32), charset, lm, prm["lm_weight"], 0.0,
                                      max(prm["beam"], n), nbest=n)
                    srcs[name][i] += got if isinstance(got, list) else [got]
        else:
            run, _, kind = name.partition(":")
            for d, stem in places(run):
                for i, rs in _readings(d, stem, ids, kind == "nbest").items():
                    srcs[name][i] += rs
    out = []
    for i in ids:
        voters = [(t, w / len(srcs[n][i])) for n, w in prm["vote_weights"].items() if srcs[n][i] for t in srcs[n][i]]
        out.append(vote(voters, prm["rounds"], max_len))
    return out


def _combine_saved(cfg: dict, train: pd.DataFrame, rep) -> np.ndarray:
    """combine: merge a saved vision-language run with a saved CTC run, fold by fold."""
    import json

    from readers.combine import choose, merge_words
    from readers.lm import CharLM, beam_decode

    prm = cfg["params"]
    vlms = [prm["vlm_from"]] if isinstance(prm["vlm_from"], str) else list(prm["vlm_from"])
    out = np.empty(len(train), dtype=object)
    for k, (a, b) in enumerate(rep):
        judges = [prm["ctc_from"]] if isinstance(prm["ctc_from"], str) else list(prm["ctc_from"])
        # read each array out of the .npz once: z["logp"] decompresses the whole array on every access
        zs = [dict(np.load(RUNS / j / f"fold{k}_logp.npz", allow_pickle=True).items()) for j in judges]
        z = zs[0]
        ids, charset = list(z["ids"]), list(z["charset"])
        for zz in zs:
            assert list(zz["ids"]) == list(train[P.ID_COL].iloc[b]), f"fold {k}: saved log-probabilities are for other lines"
        cands = {i: [] for i in ids}
        for src in vlms:
            # fold<k>.csv plus earlier saved versions of it (fold<k>_HHMM.csv: the same spec trained again, e.g. by a
            # second runner); each is a different reading of the same lines
            files = [RUNS / src / f"fold{k}.csv"] + sorted((RUNS / src).glob(f"fold{k}_[0-9][0-9][0-9][0-9].csv"))
            for f in files if prm.get("all_versions", True) else files[:1]:
                v = pd.read_csv(f, keep_default_na=False, dtype=str).set_index(P.ID_COL)["pred"]
                for i in ids:
                    cands[i].append(v.get(i, ""))
            nb = RUNS / src / f"fold{k}_nbest.json"
            if nb.exists():
                for i, lst in json.loads(nb.read_text()).items():
                    if i in cands:
                        cands[i] += lst
        lm = CharLM([P.clean(t) for t in train[P.TARGET].iloc[a]], prm["lm_order"])
        max_len = int(train[P.TARGET].iloc[a].map(P.clean).str.len().max())
        if prm["mode"] == "vote":
            out[b] = _vote_lines(prm, lambda run: [(RUNS / run, f"fold{k}")], ids, lm, max_len)
            continue
        res = []
        for r, i in enumerate(ids):
            lp = z["logp"][r].astype(np.float32)
            if prm["mode"] == "choose":
                lps = [zz["logp"][r].astype(np.float32) for zz in zs]
                css = [list(zz["charset"]) for zz in zs]
                cs = _capped(prm, cands[i], max_len, lp, charset, lm)
                res.append(choose(lps if len(zs) > 1 else lp, css if len(zs) > 1 else charset, cs, lm, prm["w"])[0])
            else:
                ctc = beam_decode(lp, charset, lm, prm["lm_weight"], 0.0, prm["beam"])
                res.append(merge_words(lp, charset, " ".join(str(cands[i][0]).split()), ctc, lm, prm["a"], prm["w"], prm["penalty"]))
        out[b] = res
    return out


def _combine_test(cfg: dict, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """combine on the test lines: the same judge, reading the test runs named in params.test_from ({held-out run: its
    test run}, each test run being the same spec trained on every training line), with the language model on all of
    them. Modes "choose" and "vote". In "vote", test_from may map a run to "folds": its five fold models' own test
    readings (fold<k>_test.*, spec option read_test), the source's weight split over them as first place did (R-002)."""
    import json

    from readers.combine import choose
    from readers.lm import CharLM

    prm = cfg["params"]
    if prm["mode"] not in ("choose", "vote"):
        raise ValueError("test predictions are built for modes 'choose' and 'vote' only")
    tf = prm["test_from"]
    ids = list(test[P.ID_COL])
    lm = CharLM([P.clean(t) for t in train[P.TARGET]], prm["lm_order"])
    max_len = int(train[P.TARGET].map(P.clean).str.len().max())
    if prm["mode"] == "vote":
        def places(run):
            if tf[run] == "folds":
                return [(RUNS / run, f"fold{k}_test") for k in range(P.N_FOLDS)]
            return [(RUNS / tf[run], "test")]
        return np.array(_vote_lines(prm, places, ids, lm, max_len), dtype=object)
    judges = [prm["ctc_from"]] if isinstance(prm["ctc_from"], str) else list(prm["ctc_from"])
    vlms = [prm["vlm_from"]] if isinstance(prm["vlm_from"], str) else list(prm["vlm_from"])
    zs = [dict(np.load(RUNS / tf[j] / "test_logp.npz", allow_pickle=True).items()) for j in judges]
    for z in zs:
        assert list(z["ids"]) == ids, "saved test log-probabilities are not in test order"
    cands = {i: [] for i in ids}
    for src in vlms:
        d = RUNS / tf[src]
        files = [d / "test.csv"] + sorted(d.glob("test_[0-9][0-9][0-9][0-9].csv"))
        for f in files if prm.get("all_versions", True) else files[:1]:
            v = pd.read_csv(f, keep_default_na=False, dtype=str).set_index(P.ID_COL)["pred"]
            assert set(v.index) == set(ids), f"{f} does not cover the test lines"
            for i in ids:
                cands[i].append(v[i])
        if (d / "test_nbest.json").exists():
            for i, lst in json.loads((d / "test_nbest.json").read_text()).items():
                cands[i] += lst
    out = []
    for r, i in enumerate(ids):
        lps = [z["logp"][r].astype(np.float32) for z in zs]
        css = [list(z["charset"]) for z in zs]
        cs = _capped(prm, cands[i], max_len, lps[0], css[0], lm)
        out.append(choose(lps if len(zs) > 1 else lps[0], css if len(zs) > 1 else css[0], cs, lm, prm["w"])[0])
    return np.array(out, dtype=object)


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
    if cfg["model"] in ("ctc_decode", "combine"):
        fn = _decode_saved if cfg["model"] == "ctc_decode" else _combine_saved
        oof = np.stack([fn(cfg, train, rep) for rep in folds])
        if cfg["model"] == "combine" and cfg["params"].get("test_from"):
            return oof, _combine_test(cfg, train, test)
        return oof, np.array([""] * len(test), dtype=object)
    X, Xt, y = features(train), features(test), train[P.TARGET].to_numpy()
    oof = np.empty((len(folds), len(train)), dtype=object)
    for r, rep in enumerate(folds):
        for a, b in rep:
            oof[r, b] = fit_predict(cfg, X.iloc[a], y[a], X.iloc[b], cfg["seed"] + r, _LocalCtx)[0]
    return oof, np.asarray(fit_predict(cfg, X, y, Xt, cfg["seed"], _LocalCtx)[0])


def _split_extras(extras: dict, n: int) -> tuple[dict, dict]:
    """Per-line extras (logp, nbest) cut at line n; the rest (charset, inner_cer) go to both halves."""
    head, tail = {}, {}
    for k, v in extras.items():
        if k in ("logp", "nbest"):
            head[k], tail[k] = list(v[:n]), list(v[n:])
        else:
            head[k] = tail[k] = v
    return head, tail


def run_fold(cfg: dict, train: pd.DataFrame, fold: int, ctx, positions=None,
             test: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """One fold of the fixed split (molab). `positions` = (train positions, held-out positions); remote.py passes
    them from the pinned fold file. Returns the held-out predictions [ID, pred] and extras to upload. With
    cfg["read_test"] and `test`, the same fold model also reads the test lines (first place's test readings, R-002):
    extras["test"] = (test predictions, their extras)."""
    a, b = positions if positions is not None else P.folds(train)[0][fold]
    X, y = features(train), train[P.TARGET].to_numpy()
    apply = X.iloc[b]
    both = cfg.get("read_test") and test is not None
    if both:
        apply = pd.concat([apply, features(test)], ignore_index=True)
    preds, extras = fit_predict(cfg, X.iloc[a], y[a], apply, cfg["seed"] + fold, ctx)
    preds = [P.clean(t) for t in preds]
    out = pd.DataFrame({P.ID_COL: train[P.ID_COL].iloc[b].to_numpy(), "pred": preds[:len(b)]})
    if both:
        extras, tx = _split_extras(extras, len(b))
        extras["test"] = (pd.DataFrame({P.ID_COL: test[P.ID_COL].to_numpy(), "pred": preds[len(b):]}), tx)
    return out, extras


def run_test(cfg: dict, train: pd.DataFrame, test: pd.DataFrame, ctx) -> tuple[pd.DataFrame, dict]:
    """Train on every training line and read the test lines (molab, for a submission). The seed follows the folds'."""
    preds, extras = fit_predict(cfg, features(train), train[P.TARGET].to_numpy(), features(test),
                                cfg["seed"] + P.N_FOLDS, ctx)
    return pd.DataFrame({P.ID_COL: test[P.ID_COL].to_numpy(), "pred": [P.clean(t) for t in preds]}), extras
