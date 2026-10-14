# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.16",
#     "numpy>=1.26,<2.3",
#     "pandas==2.2.3",
#     "pillow>=10.4",
#     "scikit-learn==1.5.2",
#     "scipy>=1.13",
#     "rapidfuzz==3.10.0",
#     "huggingface_hub==0.35.3",
#     "timm==1.0.20",
#     "transformers==4.57.1",
#     "peft==0.17.1",
#     "accelerate==1.10.1",
# ]
# ///
"""molab_runner.py — the experiment runner for the Barbados handwriting loop. You never edit experiments here.

  1. Locally: `exp.py new ...`, commit, `git push`, then `exp.py submit` puts the spec in the private Hugging Face
     dataset nyakaruosborn/barbados-handwriting-runs under queue/.
  2. Here (molab, GPU on): paste a Hugging Face token, press **Run queue**. For each pending spec the runner downloads
     the GitHub repository at the spec's commit, refuses to run if autoresearch/prepare.py is not the frozen one, and
     calls autoresearch/remote.py, which trains and reads fold by fold and uploads each finished fold.
  3. Locally: `exp.py pull` scores the folds with the frozen harness and updates the ledger.

So the code that runs is always the code pushed to GitHub; this notebook only moves files. It never writes to GitHub.
Competition data comes from the private dataset nyakaruosborn/barbados-handwriting-data.
"""

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", app_title="Barbados handwriting — runner")


@app.cell
def _():
    import hashlib
    import importlib
    import io
    import json
    import os
    import sys
    import tarfile
    import threading
    import time
    import traceback
    import urllib.request
    import uuid
    import zipfile
    from datetime import datetime, timezone
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    return (Path, datetime, hashlib, importlib, io, json, mo, os, pd, sys, tarfile, threading, time, timezone,
            traceback, urllib, uuid, zipfile)


@app.cell
def _(mo):
    mo.md("""
    # Barbados handwriting — experiment runner

    1. Turn on the GPU (notebook specs, top right). 2. Paste a Hugging Face token with **read** on
    `barbados-handwriting-data` and **write** on `barbados-handwriting-runs`. 3. Press **Run queue**.
    """)
    return


@app.cell
def _(mo, os):
    hf_token_input = mo.ui.text(kind="password", value="", label="Hugging Face token (this session only)", full_width=True)
    data_repo_input = mo.ui.text(value=os.environ.get("HF_DATA_REPO", "nyakaruosborn/barbados-handwriting-data"), label="data repo", full_width=True)
    runs_repo_input = mo.ui.text(value=os.environ.get("HF_RUNS_REPO", "nyakaruosborn/barbados-handwriting-runs"), label="runs repo", full_width=True)
    mo.vstack([hf_token_input, data_repo_input, runs_repo_input])
    return data_repo_input, hf_token_input, runs_repo_input


@app.cell
def _(data_repo_input, hf_token_input, mo, os, runs_repo_input):
    if hf_token_input.value.strip():
        os.environ["HF_TOKEN"] = hf_token_input.value.strip()
    DATA_REPO = data_repo_input.value.strip()
    RUNS_REPO = runs_repo_input.value.strip()
    GITHUB_REPO = "OsbornNyakaru/barbados-handwriting"
    # The frozen harness: autoresearch/prepare.py must hash to this or the runner refuses (vault/00_INDEX.md).
    PREPARE_SHA256 = "5ed07a757d39d209020988885e15014e0ea44e90d7419200c0b74de3de950279"
    RUNNER_VERSION = "2026-10-14"

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi()
    try:
        _who = api.whoami()["name"]
        _msg = mo.md(f"Hugging Face: signed in as **{_who}** · runner **{RUNNER_VERSION}**")
    except Exception as _e:
        _msg = mo.callout(mo.md(f"Hugging Face: **not signed in** ({type(_e).__name__}). Paste a token above."), kind="danger")
    _msg
    return CommitOperationAdd, DATA_REPO, GITHUB_REPO, PREPARE_SHA256, RUNNER_VERSION, RUNS_REPO, api, hf_hub_download


@app.cell
def _(mo):
    def _gpu():
        try:
            import torch

            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                return {"cuda": True, "name": p.name, "vram_gb": round(p.total_memory / 2**30, 1), "torch": torch.__version__}
        except Exception:
            pass
        return {"cuda": False, "name": "none", "vram_gb": 0.0}

    gpu_info = _gpu()
    mo.md(f"GPU: **{gpu_info['name']}**, {gpu_info['vram_gb']} GB" if gpu_info["cuda"]
          else "GPU: **none**. Turn it on in the notebook specs before running readers.")
    return (gpu_info,)


@app.cell
def _(DATA_REPO, Path, mo, zipfile):
    WORK = Path.cwd() / "runner_work"
    DATA = WORK / "data"
    RAW = DATA / "raw"
    RAW.mkdir(parents=True, exist_ok=True)
    _err = ""
    if not (RAW / "images").is_dir() or len(list((RAW / "images").glob("*.jpg"))) < 5472:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=str(RAW),
                              allow_patterns=["Train.csv", "Test.csv", "SampleSubmission.csv", "images.zip"])
            with zipfile.ZipFile(RAW / "images.zip") as _z:
                _z.extractall(RAW)
            (RAW / "images.zip").unlink()
        except Exception as _e:
            _err = f" ({type(_e).__name__}: {_e})"
    _n = len(list((RAW / "images").glob("*.jpg"))) if (RAW / "images").is_dir() else 0
    data_ok = (RAW / "Train.csv").exists() and _n == 5472
    mo.md(f"Data: **{'ready' if data_ok else 'missing'}**, {_n} images in `{RAW}`{_err}")
    return DATA, WORK, data_ok


@app.cell
def _(
    CommitOperationAdd, DATA, GITHUB_REPO, PREPARE_SHA256, RUNS_REPO, WORK, api, hashlib, hf_hub_download, importlib,
    io, json, pd, sys, tarfile, time, urllib,
):
    KERNEL = {"cache": {}}   # lives as long as the kernel: decoded images, processors

    def fetch_code(sha: str):
        """The repository at `sha` (GitHub's immutable tarball), with data/ linked to the downloaded data."""
        dest = WORK / "code" / sha
        if not (dest / ".ok").exists():
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(f"https://codeload.github.com/{GITHUB_REPO}/tar.gz/{sha}", timeout=120) as r:
                        blob = r.read()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(5 * (attempt + 1))
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
                tf.extractall(dest, filter="data")
            (dest / ".ok").write_text("ok")
        root = next(p for p in dest.iterdir() if p.is_dir())
        if not (root / "data").exists():
            (root / "data").symlink_to(DATA, target_is_directory=True)
        return root

    def check_harness(root):
        h = hashlib.sha256((root / "autoresearch" / "prepare.py").read_bytes()).hexdigest()
        if h != PREPARE_SHA256:
            raise RuntimeError(f"prepare.py hash {h[:12]} != frozen {PREPARE_SHA256[:12]}: harness changed, refusing to run")

    def import_remote(root):
        """remote.py from this checkout; drop modules imported from an earlier checkout first."""
        for name in [m for m in sys.modules if m in {"prepare", "experiment", "remote"} or m.startswith("readers")]:
            del sys.modules[name]
        sys.path[:] = [p for p in sys.path if "/runner_work/code/" not in p]
        sys.path.insert(0, str(root / "autoresearch"))
        importlib.invalidate_caches()
        return importlib.import_module("remote")

    def hf_files() -> list[str]:
        return api.list_repo_files(RUNS_REPO, repo_type="dataset")

    def hf_json(path: str) -> dict:
        return json.loads(open(hf_hub_download(RUNS_REPO, path, repo_type="dataset", force_download=True), encoding="utf-8").read())

    def hf_put(files: dict, message: str):
        ops = []
        for path, content in files.items():
            if isinstance(content, pd.DataFrame):
                content = content.to_csv(index=False).encode("utf-8")
            elif isinstance(content, str):
                content = content.encode("utf-8")
            ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=content))
        for attempt in range(4):
            try:
                return api.create_commit(RUNS_REPO, repo_type="dataset", operations=ops, commit_message=message)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))

    class Store:
        """runs/<EXP>/ in the runs repo, as remote.py expects it."""

        def __init__(self, exp: str):
            self.base = f"runs/{exp}"
            self.files = set(hf_files())

        def exists(self, name):
            return f"{self.base}/{name}" in self.files

        def get(self, name):
            return hf_hub_download(RUNS_REPO, f"{self.base}/{name}", repo_type="dataset")

        def put(self, files: dict):
            hf_put({f"{self.base}/{k}": v for k, v in files.items()}, f"{self.base}: {', '.join(files)}")
            self.files |= {f"{self.base}/{k}" for k in files}

    return KERNEL, Store, check_harness, fetch_code, hf_files, hf_json, hf_put, import_remote


@app.cell
def _(
    KERNEL, RUNNER_VERSION, Store, WORK, check_harness, datetime, fetch_code, gpu_info, hf_files, hf_json, hf_put,
    import_remote, json, mo, threading, time, timezone, traceback, uuid,
):
    SESSION = uuid.uuid4().hex[:8]
    CLAIM_EVERY, CLAIM_STALE = 240, 900   # seconds; a reopened tab must not start a second runner on the same spec

    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _versions():
        out = {}
        for m in ("torch", "transformers", "peft", "timm", "numpy", "pandas", "sklearn"):
            try:
                out[m] = __import__(m).__version__
            except Exception:
                pass
        return out

    def pending() -> list[dict]:
        files = hf_files()
        done = {f.split("/")[1] for f in files if f.startswith("runs/") and f.endswith("/result.json")}
        specs = [hf_json(f) for f in files if f.startswith("queue/") and f.endswith(".json")]
        todo = [s for s in specs if s["exp"] not in done]
        fresh = []
        for s in todo:
            c = f"runs/{s['exp']}/claim.json"
            if c in files:
                try:
                    cl = hf_json(c)
                    if cl.get("session") != SESSION and time.time() - float(cl.get("t", 0)) < CLAIM_STALE:
                        continue
                except Exception:
                    pass
            fresh.append(s)
        return sorted(fresh, key=lambda s: s.get("submitted", ""))

    def _claim(exp):
        hf_put({f"runs/{exp}/claim.json": json.dumps({"session": SESSION, "t": time.time(), "at": _now(), "gpu": gpu_info})},
               f"{exp}: claim {SESSION}")

    def keep_claim(exp):
        stop = threading.Event()

        def beat():
            while not stop.wait(CLAIM_EVERY):
                w = threading.Thread(target=lambda: _quiet(_claim, exp), daemon=True)
                w.start()
                w.join(60)

        threading.Thread(target=beat, daemon=True).start()
        return stop

    def _quiet(fn, *a):
        try:
            fn(*a)
        except Exception:
            pass

    def run_one(spec, log) -> dict:
        exp = spec["exp"]
        lines = []

        def _log(msg):
            line = f"[{time.strftime('%H:%M:%S')}] {exp} {msg}"
            lines.append(line)
            log(line)

        t0 = time.time()
        result = {"exp": exp, "git_sha": spec["git_sha"], "started": _now(), "gpu": gpu_info, "runner": RUNNER_VERSION}
        try:
            root = fetch_code(spec["git_sha"])
            check_harness(root)
            remote = import_remote(root)
            result.update(remote.run_spec(spec, Store(exp), log=_log, cache=KERNEL["cache"], work_dir=str(WORK / "train" / exp)))
        except Exception as e:
            result["status"] = "crash"
            result["error"] = "".join(traceback.format_exception(e))[-4000:]
            _log(f"CRASH: {type(e).__name__}: {e}")
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        result.update(finished=_now(), seconds=round(time.time() - t0, 1), versions=_versions())
        files = {f"runs/{exp}/log_{SESSION}.txt": "\n".join(lines[-2000:])}
        if result.get("status") in ("ok", "crash"):
            files[f"runs/{exp}/result.json"] = json.dumps(result, indent=2, default=str)
        else:
            files[f"runs/{exp}/session_{SESSION}.json"] = json.dumps(result, indent=2, default=str)
        hf_put(files, f"{exp}: {result.get('status')}")
        return result

    def run_queue(watch=True, idle_minutes=30, log=print):
        hf_put({"_runner/ping.json": json.dumps({"at": _now(), "gpu": gpu_info, "version": RUNNER_VERSION})}, "runner ping")
        results, idle_since = [], time.time()
        while True:
            todo = pending()
            if not todo:
                if not watch or time.time() - idle_since > 60 * idle_minutes:
                    break
                time.sleep(60)
                continue
            spec = todo[0]
            _claim(spec["exp"])
            time.sleep(8)
            if hf_json(f"runs/{spec['exp']}/claim.json").get("session") != SESSION:
                log(f"· {spec['exp']} taken by another runner")
                continue
            log(f"▶ {spec['exp']} ({len(todo)} pending): {spec['description']}")
            beat = keep_claim(spec["exp"])
            try:
                results.append(run_one(spec, log))
            finally:
                beat.set()
            log(f"■ {spec['exp']}: {results[-1].get('status')}")
            if results[-1].get("status") == "incomplete":
                break   # the time limit was reached; run the queue again to resume at the next fold
            idle_since = time.time()
        return results

    mo.md("Runner ready.")
    return (run_queue,)


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run queue")
    watch_switch = mo.ui.switch(value=True, label="keep watching for new specs (every 60 s)")
    idle_input = mo.ui.number(start=5, stop=600, step=5, value=30, label="stop after N idle minutes")
    mo.hstack([run_btn, watch_switch, idle_input], justify="start", wrap=True)
    return idle_input, run_btn, watch_switch


@app.cell
def _(data_ok, idle_input, mo, run_btn, run_queue, watch_switch):
    mo.stop(not run_btn.value, mo.md("_Press **Run queue** to start._"))
    mo.stop(not data_ok, mo.callout(mo.md("Data is missing: check the token and the data repo above."), kind="danger"))
    _lines = []

    def _show(line):
        _lines.append(line)
        mo.output.replace(mo.md("```\n" + "\n".join(_lines[-40:]) + "\n```"))

    _res = run_queue(watch=watch_switch.value, idle_minutes=idle_input.value, log=_show)
    mo.md(f"Done: {len(_res)} run(s): " + ", ".join(f"`{r['exp']}` {r.get('status')}" for r in _res)
          + ". Now run `exp.py pull` on the Mac.")
    return


if __name__ == "__main__":
    app.run()
