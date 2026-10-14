"""exp.py — the experiment loop's ledger: spec -> run -> record -> keep or drop -> submission.

    python scripts/exp.py new --from best --set key=value --desc "..." [--hyp H-001]
    python scripts/exp.py run                 # every CPU spec without a result, on this machine
    python scripts/exp.py submit [EXP-002]    # GPU specs: stamp the pushed git commit, queue on Hugging Face
    python scripts/exp.py pull [--wait]       # fetch molab results, score, decide, update the ledger
    python scripts/exp.py status
    python scripts/exp.py submission [--exp EXP-003]
    python scripts/exp.py lb --file <csv> --public 0.81 [--private ...] [--note "..."]

A spec is experiments/specs/EXP-NNN.json: a full config, its parent and the keys it changed. A result is one line of
experiments/results.jsonl. BEST.json is the adopted run: a run replaces it when it beats the best on the same folds by
>= ADOPT_GAIN on average and in >= ADOPT_WINS of the repeats (paired, so fold luck cancels).

Needs autoresearch/prepare.py (frozen) and autoresearch/experiment.py:
    prepare:    load() -> (train, test); folds(train) -> [repeat][fold] (train positions, validation positions);
                cv_score(train, oof) -> {"score", "std", "per_repeat", "parts": {name: value}};
                build_submission(test, preds) -> DataFrame; validate_submission(sub); harness_hash();
                ID_COL, HIGHER_IS_BETTER, N_REPEATS, VALIDATION (one line describing the folds)
    experiment: DEFAULT_CONFIG; run(config, train, test, folds) -> (oof [repeats, rows], test predictions [rows])
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))
import experiment as E  # noqa: E402
import prepare as P  # noqa: E402

EXP = ROOT / "experiments"
SPECS, RUNS, RESULTS, BEST, LB = EXP / "specs", EXP / "runs", EXP / "results.jsonl", EXP / "BEST.json", EXP / "leaderboard.jsonl"
ADOPT_GAIN, ADOPT_WINS = 0.002, 4  # provisional: see vault/decisions/D-002. The paired units are the 5 folds of the one fixed split
SIGN = 1 if P.HIGHER_IS_BETTER else -1
RUNS_REPO = "nyakaruosborn/barbados-handwriting-runs"   # private Hugging Face dataset: queue/ and runs/


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def records() -> list[dict]:
    return [json.loads(x) for x in RESULTS.read_text().splitlines() if x.strip()] if RESULTS.exists() else []


def spec(exp: str) -> dict:
    return json.loads((SPECS / f"{exp}.json").read_text())


def best() -> dict | None:
    return json.loads(BEST.read_text()) if BEST.exists() else None


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def set_key(cfg: dict, key: str, value) -> None:
    """key may be dotted (params.num_leaves)."""
    *path, last = key.split(".")
    for p in path:
        cfg = cfg.setdefault(p, {})
    cfg[last] = value


def flat(cfg: dict, pre: str = "") -> dict:
    out = {}
    for k, v in cfg.items():
        out.update(flat(v, f"{pre}{k}.") if isinstance(v, dict) else {f"{pre}{k}": v})
    return out


def parts(d: dict) -> str:
    return ", ".join(f"{k} {v:.4f}" for k, v in (d.get("parts") or {}).items())


def cmd_new(a) -> None:
    parent = (best() or {}).get("exp") if a.from_ == "best" else (a.from_ or None)
    cfg = json.loads(json.dumps(spec(parent)["config"] if parent else E.DEFAULT_CONFIG))
    cfg = {**json.loads(json.dumps(E.DEFAULT_CONFIG)), **cfg}
    before = flat(cfg)
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        set_key(cfg, k, v)
    if hasattr(E, "complete"):
        cfg = E.complete(cfg)
    after = flat(cfg)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if parent and len(changed) != 1 and not a.allow_multi:
        sys.exit(f"one change per experiment: this changes {changed or 'nothing'} (use --allow-multi to override)")
    n = 1 + max([int(p.stem.split("-")[1]) for p in SPECS.glob("EXP-*.json")], default=0)
    exp = f"EXP-{n:03d}"
    SPECS.mkdir(parents=True, exist_ok=True)
    (SPECS / f"{exp}.json").write_text(json.dumps({"exp": exp, "hypothesis": a.hyp, "description": a.desc, "parent": parent,
                                                   "changed": changed, "config": cfg, "created": now(),
                                                   **({"folds": a.folds} if a.folds else {}),
                                                   **({"max_minutes": a.max_minutes} if a.max_minutes else {})}, indent=2) + "\n")
    print(f"wrote experiments/specs/{exp}.json (changed vs {parent or 'defaults'}: {changed})")
    if cfg["model"] in getattr(E, "GPU_MODELS", ()):
        print("GPU model: commit and push, then `exp.py submit`")


def paired(exp_scores: list[float], best_scores: list[float]) -> tuple[float, int]:
    """Mean improvement over the best and the number of repeats improved (positive = better, whichever way the metric runs)."""
    d = SIGN * (np.array(exp_scores) - np.array(best_scores))
    return float(d.mean()), int((d > 0).sum())


def record(exp: str, s: dict, cv: dict, extra: dict) -> None:
    """Append one result, compare it with the best on the same folds, adopt or not. A partial run (some folds only)
    is recorded and compared on its folds, never adopted; a later full pull of the same EXP replaces it."""
    rec = {"exp": exp, "hypothesis": s.get("hypothesis"), "description": s["description"], "parent": s.get("parent"),
           "changed": s.get("changed"), **cv, "finished": now(), "harness": P.harness_hash(), **extra}
    b = best()
    if b and b["harness"] != rec["harness"]:
        sys.exit("prepare.py changed since the best run: results are not comparable")
    folds = rec.get("folds") or list(range(P.N_FOLDS))
    if b:
        gain, wins = paired(cv["per_repeat"], [b["per_repeat"][k] for k in folds])
        rec["vs_best"] = {"best": b["exp"], "gain": round(gain, 5), "wins": wins}
    else:
        gain, wins, rec["vs_best"] = None, None, None
    full = not rec.get("partial")
    rec["adopted"] = full and (b is None or (gain >= ADOPT_GAIN and wins >= min(ADOPT_WINS, len(cv["per_repeat"]))))
    kept = [r for r in records() if r["exp"] != exp]
    RESULTS.write_text("".join(json.dumps(r) + "\n" for r in kept + [rec]))
    if rec["adopted"]:
        BEST.write_text(json.dumps({k: rec[k] for k in ("exp", "score", "std", "parts", "per_repeat", "harness", "finished")}, indent=2) + "\n")
    vs = f" | vs {b['exp']} on the same folds: {gain:+.4f}, better in {wins}/{len(cv['per_repeat'])}" if b else ""
    tag = "ADOPTED" if rec["adopted"] else ("partial, not eligible" if not full else "not adopted")
    print(f"{exp}: score {cv['score']:.4f} ± {cv['std']:.4f} ({parts(cv)}){vs} -> {tag}")


def cmd_run(a) -> None:
    done = {r["exp"] for r in records()}
    todo = a.exps or sorted(p.stem for p in SPECS.glob("EXP-*.json") if p.stem not in done)
    gpu = getattr(E, "GPU_MODELS", set())
    skip = [e for e in todo if spec(e)["config"]["model"] in gpu]
    todo = [e for e in todo if e not in skip]
    if skip:
        print(f"GPU specs, not run here (exp.py submit, then pull): {', '.join(skip)}")
    if not todo:
        return print("nothing to run")
    train, test = P.load()
    folds = P.folds(train)
    for exp in todo:
        s = spec(exp)
        t0 = time.time()
        oof, test_pred = E.run(s["config"], train, test, folds)
        cv = P.cv_score(train, oof)
        d = RUNS / exp
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "oof.npy", np.asarray(oof), allow_pickle=True)
        pd.DataFrame({P.ID_COL: test[P.ID_COL], "pred": list(test_pred)}).to_csv(d / "test_pred.csv", index=False)
        record(exp, s, cv, {"seconds": round(time.time() - t0, 1), "git_sha": git_sha(), "where": "local"})
    write_results_md()


# ── molab: submit and pull ───────────────────────────────────────────────────


def _hf():
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(RUNS_REPO, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(RUNS_REPO, repo_type="dataset").private:
        sys.exit(f"{RUNS_REPO} is PUBLIC: refusing to use it (predictions are competition data)")
    return api


def _remote_files(api) -> list[str]:
    return api.list_repo_files(RUNS_REPO, repo_type="dataset")


def cmd_submit(a) -> None:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    if subprocess.run(["git", "status", "--porcelain", "--", "autoresearch", "notebooks"], cwd=ROOT,
                      capture_output=True, text=True).stdout.strip():
        sys.exit("autoresearch/ or notebooks/ has uncommitted changes: commit and push first, so molab runs this code")
    sha = git_sha()
    if not subprocess.run(["git", "branch", "-r", "--contains", sha], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        sys.exit(f"commit {sha[:10]} is not on GitHub yet: git push first")
    api = _hf()
    files = set(_remote_files(api))
    done = {r["exp"] for r in records() if not r.get("partial")}
    gpu = getattr(E, "GPU_MODELS", set())
    todo = a.exps or sorted(p.stem for p in SPECS.glob("EXP-*.json")
                            if p.stem not in done and spec(p.stem)["config"]["model"] in gpu and f"queue/{p.stem}.json" not in files)
    if a.resubmit and not a.exps:
        sys.exit("--resubmit needs the EXP ids to run again")
    if not todo:
        return print("nothing to submit")
    ops = []
    for exp in todo:
        s = spec(exp)
        s["git_sha"] = sha if a.resubmit else (s.get("git_sha") or sha)
        s["submitted"] = now()
        (SPECS / f"{exp}.json").write_text(json.dumps(s, indent=2) + "\n")
        ops.append(CommitOperationAdd(path_in_repo=f"queue/{exp}.json", path_or_fileobj=json.dumps(s, indent=2).encode()))
        if a.resubmit:   # a crashed run: clear its end marker so the runner takes it again (logs and finished folds stay)
            ops += [CommitOperationDelete(path_in_repo=f) for f in (f"runs/{exp}/result.json", f"runs/{exp}/claim.json") if f in files]
    api.create_commit(RUNS_REPO, repo_type="dataset", operations=ops, commit_message=f"queue {', '.join(todo)}")
    print(f"queued {', '.join(todo)} at {sha[:10]}. On molab: open the runner notebook and press Run queue.\n"
          f"Then `exp.py pull`. Commit the stamped specs.")


def cmd_pull(a) -> None:
    from huggingface_hub import hf_hub_download

    api = _hf()
    while True:
        files = _remote_files(api)
        queued = sorted(f.split("/")[1][:-5] for f in files if f.startswith("queue/") and f.endswith(".json"))
        have = {r["exp"]: r for r in records()}
        waiting = []
        for exp in queued:
            s = spec(exp) if (SPECS / f"{exp}.json").exists() else None
            if s is None:
                continue
            asked = s.get("folds") or list(range(P.N_FOLDS))
            got = [k for k in asked if f"runs/{exp}/fold{k}.csv" in files]
            prev = have.get(exp)
            if len(got) < len(asked) and f"runs/{exp}/result.json" in files:
                r = json.loads(Path(hf_hub_download(RUNS_REPO, f"runs/{exp}/result.json", repo_type="dataset",
                                                    force_download=True)).read_text())
                print(f"{exp}: {r.get('status')} on molab after folds {got}\n{(r.get('error') or '')[-1500:]}")
            elif len(got) < len(asked):
                waiting.append(exp)
            if not got or (prev and prev.get("folds", list(range(P.N_FOLDS))) == got):
                continue   # nothing yet, or nothing new since the last pull
            d = RUNS / exp
            d.mkdir(parents=True, exist_ok=True)
            for f in files:
                if f.startswith(f"runs/{exp}/") and not f.endswith("claim.json"):
                    src = hf_hub_download(RUNS_REPO, f, repo_type="dataset", local_dir=str(RUNS / "_hf"), force_download=True)
                    Path(src).replace(d / Path(f).name)
            score_pulled(exp, s, got, len(got) < len(asked))
        write_results_md()
        if not (a.wait and waiting):
            if waiting:
                print(f"still running on molab: {', '.join(waiting)}")
            return
        time.sleep(60)


def score_pulled(exp: str, s: dict, folds: list[int], partial: bool) -> None:
    train, _ = P.load()
    fold = P.fold_of(train)
    pred = pd.concat([pd.read_csv(RUNS / exp / f"fold{k}.csv", keep_default_na=False, dtype=str) for k in folds])
    m = train[[P.ID_COL, P.TARGET, P.BATCH_COL]].merge(pred, on=P.ID_COL, how="inner")
    assert len(m) == int(np.isin(fold, folds).sum()), f"{exp}: predictions do not cover the folds {folds}"
    if not partial and len(folds) == P.N_FOLDS:
        oof = np.empty((1, len(train)), dtype=object)
        oof[0] = train[[P.ID_COL]].merge(pred, on=P.ID_COL, how="left")["pred"].fillna("").to_numpy()
        cv = P.cv_score(train, oof)
    else:
        sc = P.score(m[P.TARGET].to_numpy(), m["pred"].to_numpy())
        fm = dict(zip(train[P.ID_COL], fold))
        per = [P.score(g[P.TARGET].to_numpy(), g["pred"].to_numpy())["score"] for _, g in m.groupby(m[P.ID_COL].map(fm))]
        rng = np.random.default_rng(P.BOOT_SEED)
        e = P.line_edits(m[P.TARGET].to_numpy(), m["pred"].to_numpy())
        line = 1 - 0.5 * (e["word_edits"] / P.WORD_NORM + e["char_edits"] / P.CHAR_NORM)
        boots = [line.to_numpy()[rng.integers(0, len(line), len(line))].mean() for _ in range(P.N_BOOT)]
        batch = {name: P.score(g[P.TARGET].to_numpy(), g["pred"].to_numpy())["score"] for name, g in m.groupby(P.BATCH_COL)}
        cv = {"score": sc["score"], "std": float(np.std(boots)), "per_repeat": per,
              "parts": {**sc["parts"], "short": batch.get(False, np.nan), "tall": batch.get(True, np.nan)}}
    meta = [json.loads((RUNS / exp / f"fold{k}.json").read_text()) for k in folds if (RUNS / exp / f"fold{k}.json").exists()]
    extra = {"folds": folds, "partial": bool(partial or len(folds) < P.N_FOLDS), "git_sha": s.get("git_sha"), "where": "molab",
             "seconds": round(sum(x.get("seconds", 0) for x in meta), 1),
             "exact_lines": round(float((m[P.TARGET].map(P.clean) == m["pred"].map(P.clean)).mean()), 4),
             "cer": round(float(sum(Levenshtein.distance(P.clean(t), p) for t, p in zip(m[P.TARGET], m["pred"]))
                               / max(1, sum(len(P.clean(t)) for t in m[P.TARGET]))), 5)}
    record(exp, s, cv, extra)


def write_results_md() -> None:
    rs, b = records(), best()
    lines = ["# Results", "", f"Validation: {P.VALIDATION} {'Higher' if P.HIGHER_IS_BETTER else 'Lower'} is better.",
             f"Adoption: gain over the best >= {ADOPT_GAIN} averaged over folds, and better in >= {ADOPT_WINS}/{P.N_FOLDS} folds.", "",
             f"**Best: {b['exp']} — {b['score']:.4f}**" if b else "", "",
             "| exp | score | ± | parts | vs best then | adopted | description |", "|---|--:|--:|---|---|---|---|"]
    for r in rs:
        v = r.get("vs_best")
        vs = f"{v['gain']:+.4f} ({v['wins']}/{len(r['per_repeat'])}) vs {v['best']}" if v else "—"
        part = f" (folds {','.join(map(str, r['folds']))} only)" if r.get("partial") else ""
        lines.append(f"| {r['exp']}{part} | {r['score']:.4f} | {r['std']:.4f} | {parts(r)} | {vs} | {'yes' if r['adopted'] else 'no'} | {r['description']} |")
    if LB.exists():
        lines += ["", "## Leaderboard", "", "| # | file | public | private | cv | note |", "|---|---|--:|--:|--:|---|"]
        for i, x in enumerate((json.loads(l) for l in LB.read_text().splitlines() if l.strip()), 1):
            lines.append(f"| {i} | {x['file']} | {x['public']} | {x.get('private') or ''} | {x.get('cv') or ''} | {x.get('note') or ''} |")
    (EXP / "RESULTS.md").write_text("\n".join(lines) + "\n")


def cmd_status(a) -> None:
    done = {r["exp"]: r for r in records()}
    b = best()
    for p in sorted(SPECS.glob("EXP-*.json")):
        r = done.get(p.stem)
        s = spec(p.stem)
        state = "queued   " if s.get("submitted") else "not run  "
        part = f" (folds {r['folds']})" if r and r.get("partial") else ""
        print(f"{p.stem}  " + (f"{r['score']:.4f} ± {r['std']:.4f}  {'ADOPTED ' if r['adopted'] else '        '}{s['description']}{part}" if r else f"{state} {s['description']}"))
    print(f"best: {b['exp']} {b['score']:.4f}" if b else "no runs yet")


def cmd_submission(a) -> None:
    exp = a.exp or best()["exp"]
    t = pd.read_csv(RUNS / exp / "test_pred.csv", keep_default_na=False)
    _, test = P.load()
    assert list(t[P.ID_COL]) == list(test[P.ID_COL]), "test predictions are not in test order"
    sub = P.build_submission(test, t["pred"].to_numpy())
    P.validate_submission(sub)
    out = ROOT / "submissions" / f"{exp}_{dt.datetime.now():%Y%m%d_%H%M}.csv"
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"wrote {out.relative_to(ROOT)} ({len(sub)} rows). Upload it to the competition.")


def cmd_lb(a) -> None:
    exp = Path(a.file).name.split("_")[0]
    cv = next((r["score"] for r in records() if r["exp"] == exp), None)
    with LB.open("a") as f:
        f.write(json.dumps({"file": Path(a.file).name, "public": a.public, "private": a.private, "cv": round(cv, 4) if cv else None,
                            "note": a.note, "recorded": now()}) + "\n")
    write_results_md()
    print(f"recorded {Path(a.file).name}: public {a.public}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("--from", dest="from_", default="best", help="'best', an EXP id, or '' for the defaults")
    n.add_argument("--set", action="append", help="config override key=value (JSON values; dotted keys for nested ones)")
    n.add_argument("--desc", required=True)
    n.add_argument("--hyp")
    n.add_argument("--allow-multi", action="store_true")
    n.add_argument("--folds", type=int, nargs="+", help="GPU runs: only these folds (a smoke test); never adopted")
    n.add_argument("--max-minutes", type=float, help="GPU runs: stop starting new folds after this long")
    r = sub.add_parser("run")
    r.add_argument("exps", nargs="*")
    sub.add_parser("status")
    q = sub.add_parser("submit")
    q.add_argument("exps", nargs="*")
    q.add_argument("--resubmit", action="store_true", help="re-queue these EXPs at the current commit (after a crash)")
    pl = sub.add_parser("pull")
    pl.add_argument("--wait", action="store_true", help="poll every minute until every queued run is finished")
    s = sub.add_parser("submission")
    s.add_argument("--exp")
    l = sub.add_parser("lb")
    l.add_argument("--file", required=True)
    l.add_argument("--public", type=float, required=True)
    l.add_argument("--private", type=float)
    l.add_argument("--note")
    a = ap.parse_args()
    {"new": cmd_new, "run": cmd_run, "status": cmd_status, "submission": cmd_submission, "lb": cmd_lb,
     "submit": cmd_submit, "pull": cmd_pull}[a.cmd](a)


if __name__ == "__main__":
    main()
