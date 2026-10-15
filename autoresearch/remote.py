"""remote.py — runs one queued spec fold by fold. Called by notebooks/molab_runner.py from the repository as it was at
the spec's git commit, so a run always uses the code the spec was written against.

    store   where results go: exists(name), get(name) -> local path, put({name: bytes | str | DataFrame})
            (names are relative to runs/<EXP>/ in the private Hugging Face runs repo)
    log     callable(str)
    cache   dict that lives as long as the notebook kernel (decoded images, loaded models)

Each finished fold is uploaded at once (fold<k>.csv, plus fold<k>_logp.npz for CTC readers), so a run that dies
resumes at the next fold. Fold -1 (TEST) trains on every training line and reads the test lines into test.csv (and
test_logp.npz, test_nbest.json), for a submission. Locally, `python autoresearch/remote.py EXP-002 --folds 0` runs the same code on the CPU
into experiments/runs/<EXP>/ (use small params such as limit_train and epochs for that).
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import experiment as E  # noqa: E402
import prepare as P  # noqa: E402


class Ctx:
    """What a reader sees. log() also keeps the lines and uploads them as log_live.txt every two minutes, so a
    long fold can be followed from the Mac (`exp.py` users read it from the runs repo)."""

    def __init__(self, log, cache, work_dir, deadline, store=None):
        self._print, self.cache, self.work_dir, self._deadline = log, cache, Path(work_dir), deadline
        self._store, self.lines, self._last = store, [], 0.0

    def log(self, msg):
        self._print(msg)
        self.lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if self._store is not None and time.time() - self._last > 120:
            self.flush()

    def flush(self):
        self._last = time.time()
        try:
            self._store.put({"log_live.txt": "\n".join(self.lines[-3000:])})
        except Exception:
            pass   # never fail a run over a log upload

    def should_stop(self) -> bool:
        return time.time() > self._deadline


def _npz(extras: dict, ids) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, ids=np.array(ids, dtype=object), charset=np.array(extras["charset"], dtype=object),
                        logp=np.array([x.astype(np.float16) for x in extras["logp"]], dtype=object))
    return buf.getvalue()


def pinned_folds(train: pd.DataFrame) -> np.ndarray:
    """The frozen fold of every training line, from data/raw/folds.csv (written by scripts/push_data_to_hf.py with the
    pinned local packages). Molab installs its own scikit-learn, whose fold splitter shuffles differently (EXP-002's
    first attempt, 2026-10-14), so the folds travel as data and are checked against the frozen fold hash."""
    path = P.RAW / "folds.csv"
    if not path.exists():   # an older runner notebook does not download it; fetch it from the private data repo
        import os
        import shutil

        from huggingface_hub import hf_hub_download

        repo = os.environ.get("HF_DATA_REPO", "nyakaruosborn/barbados-handwriting-data")
        shutil.copy(hf_hub_download(repo, "folds.csv", repo_type="dataset"), path)
    f = pd.read_csv(path, keep_default_na=False, dtype={P.ID_COL: str, "fold": int})
    if list(f[P.ID_COL]) != list(train[P.ID_COL]):
        raise RuntimeError("folds.csv is not in Train.csv order")
    text = "\n".join(f"{i},{k}" for i, k in zip(f[P.ID_COL], f["fold"]))
    import hashlib
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    if h != P.FOLD_HASH:
        raise RuntimeError(f"folds.csv hash {h} != frozen {P.FOLD_HASH}")
    return f["fold"].to_numpy()


TEST = -1   # in a spec's "folds": train on every training line, then read the test lines (files test.csv, ...)


def part_name(k: int) -> str:
    return "test" if k == TEST else f"fold{k}"


def run_spec(spec: dict, store, log=print, cache: dict | None = None, work_dir: str = "/tmp/runner",
             max_minutes: float | None = None) -> dict:
    cache = {} if cache is None else cache
    t0 = time.time()
    deadline = t0 + 60 * float(max_minutes or spec.get("max_minutes") or 10_000)
    train, test = P.load()
    fold = pinned_folds(train)
    cfg = spec["config"]
    folds = spec.get("folds") or list(range(P.N_FOLDS))
    ctx = Ctx(log, cache, work_dir, deadline, store)
    done = {}
    for k in folds:
        stem = part_name(k)
        name = f"{stem}.csv"
        if store.exists(name):
            log(f"fold {k}: already done, skipping")
            continue
        if ctx.should_stop():
            log(f"fold {k}: time limit reached before starting")
            break
        if getattr(store, "cancelled", lambda: False)():
            ctx.log(f"fold {k}: cancelled (runs/<EXP>/cancel exists); stopping")
            ctx.flush()
            break
        tf = time.time()
        if k == TEST:
            ctx.log(f"test: start ({cfg['model']}), training on all {len(train)} lines")
            pred, extras = E.run_test(cfg, train, test, ctx)
        else:
            ctx.log(f"fold {k}: start ({cfg['model']})")
            pred, extras = E.run_fold(cfg, train, k, ctx, (np.where(fold != k)[0], np.where(fold == k)[0]), test)
        files = {}
        # the fold model's own test readings (cfg["read_test"]): fold<k>_test.csv, _test_logp.npz, _test_nbest.json
        parts = [(stem, pred, extras)] + ([(f"{stem}_test", *extras.pop("test"))] if "test" in extras else [])
        for s, p, x in parts:
            files[f"{s}.csv"] = p
            if "logp" in x:
                files[f"{s}_logp.npz"] = _npz(x, p[P.ID_COL].tolist())
            if "nbest" in x:
                files[f"{s}_nbest.json"] = json.dumps(dict(zip(p[P.ID_COL].tolist(), x["nbest"])))
        done[k] = {"seconds": round(time.time() - tf, 1), **{x: extras[x] for x in ("inner_cer",) if x in extras}}
        if k != TEST:
            ref = train.set_index(P.ID_COL).loc[pred[P.ID_COL], P.TARGET]
            sc = P.score(ref.to_numpy(), pred["pred"].to_numpy())
            done[k]["score"] = round(sc["score"], 5)
        files[f"{stem}.json"] = json.dumps(done[k])
        store.put(files)
        ctx.log(f"{stem}: " + (f"score {sc['score']:.4f} (word edits {sc['parts']['word_edits']:.3f}, "
                               f"char edits {sc['parts']['char_edits']:.3f}) " if k != TEST else f"read {len(pred)} test lines ")
                + f"in {(time.time() - tf) / 60:.1f} min")
        ctx.flush()
    finished = [k for k in folds if store.exists(f"{part_name(k)}.csv")]
    return {"folds_done": finished, "folds_asked": folds, "this_session": done,
            "status": "ok" if len(finished) == len(folds) else "incomplete", "seconds": round(time.time() - t0, 1)}


class LocalStore:
    """Writes into experiments/runs/<EXP>/ (local dry runs)."""

    def __init__(self, folder: Path):
        self.d = folder
        self.d.mkdir(parents=True, exist_ok=True)

    def exists(self, name):
        return (self.d / name).exists()

    def get(self, name):
        return str(self.d / name)

    def put(self, files: dict):
        for name, v in files.items():
            if isinstance(v, pd.DataFrame):
                v.to_csv(self.d / name, index=False)
            elif isinstance(v, bytes):
                (self.d / name).write_bytes(v)
            else:
                (self.d / name).write_text(v)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--set", action="append", default=[], help="params override key=value for a quick local check")
    a = ap.parse_args()
    root = HERE.parent
    s = json.loads((root / "experiments" / "specs" / f"{a.exp}.json").read_text())
    for kv in a.set:
        k, _, v = kv.partition("=")
        s["config"]["params"][k] = json.loads(v)
    if a.folds is not None:
        s["folds"] = a.folds
    out = root / "experiments" / "runs" / (a.exp + ("-local" if a.set else ""))
    print(json.dumps(run_spec(s, LocalStore(out), cache={}, work_dir=str(out)), indent=2))
