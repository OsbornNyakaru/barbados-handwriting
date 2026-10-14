"""remote.py — runs one queued spec fold by fold. Called by notebooks/molab_runner.py from the repository as it was at
the spec's git commit, so a run always uses the code the spec was written against.

    store   where results go: exists(name), get(name) -> local path, put({name: bytes | str | DataFrame})
            (names are relative to runs/<EXP>/ in the private Hugging Face runs repo)
    log     callable(str)
    cache   dict that lives as long as the notebook kernel (decoded images, loaded models)

Each finished fold is uploaded at once (fold<k>.csv, plus fold<k>_logp.npz for CTC readers), so a run that dies
resumes at the next fold. Locally, `python autoresearch/remote.py EXP-002 --folds 0` runs the same code on the CPU
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
    def __init__(self, log, cache, work_dir, deadline):
        self.log, self.cache, self.work_dir, self._deadline = log, cache, Path(work_dir), deadline

    def should_stop(self) -> bool:
        return time.time() > self._deadline


def _npz(extras: dict, ids) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, ids=np.array(ids, dtype=object), charset=np.array(extras["charset"], dtype=object),
                        logp=np.array([x.astype(np.float16) for x in extras["logp"]], dtype=object))
    return buf.getvalue()


def run_spec(spec: dict, store, log=print, cache: dict | None = None, work_dir: str = "/tmp/runner",
             max_minutes: float | None = None) -> dict:
    cache = {} if cache is None else cache
    t0 = time.time()
    deadline = t0 + 60 * float(max_minutes or spec.get("max_minutes") or 10_000)
    train, _ = P.load()
    fh = P.fold_hash(train)
    if fh != P.FOLD_HASH:
        raise RuntimeError(f"fold hash {fh} != {P.FOLD_HASH}: the folds differ on this machine (package versions?)")
    cfg = spec["config"]
    folds = spec.get("folds") or list(range(P.N_FOLDS))
    ctx = Ctx(log, cache, work_dir, deadline)
    done = {}
    for k in folds:
        name = f"fold{k}.csv"
        if store.exists(name):
            log(f"fold {k}: already done, skipping")
            continue
        if ctx.should_stop():
            log(f"fold {k}: time limit reached before starting")
            break
        tf = time.time()
        log(f"fold {k}: start ({cfg['model']})")
        pred, extras = E.run_fold(cfg, train, k, ctx)
        ref = train.set_index(P.ID_COL).loc[pred[P.ID_COL], P.TARGET]
        sc = P.score(ref.to_numpy(), pred["pred"].to_numpy())
        files = {name: pred}
        if "logp" in extras:
            files[f"fold{k}_logp.npz"] = _npz(extras, pred[P.ID_COL].tolist())
        done[k] = {"score": round(sc["score"], 5), "seconds": round(time.time() - tf, 1),
                   **{x: extras[x] for x in ("inner_cer",) if x in extras}}
        files[f"fold{k}.json"] = json.dumps(done[k])
        store.put(files)
        log(f"fold {k}: score {sc['score']:.4f} (word edits {sc['parts']['word_edits']:.3f}, "
            f"char edits {sc['parts']['char_edits']:.3f}) in {(time.time() - tf) / 60:.1f} min")
    finished = [k for k in folds if store.exists(f"fold{k}.csv")]
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
