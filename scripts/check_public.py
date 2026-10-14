"""check_public.py — refuse to commit competition data to the public repository.

The GitHub repository is public (molab syncs the runner notebook from it), and Zindi's terms forbid sharing the data.
This scans files for anything taken from the competition files:
  * any run of 5 words from a training transcript;
  * any sample-submission target of 8 characters or more;
  * images, CSV files and anything under data/ or experiments/runs/.

    python scripts/check_public.py            # the files staged for commit (the pre-commit hook runs this)
    python scripts/check_public.py --all      # every tracked file
Install the hook once with `make hooks`. Without data/raw/Train.csv it only checks file types.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TEXT = {".md", ".py", ".json", ".jsonl", ".txt", ".toml", ".yaml", ".yml", ".cfg", ".ini", ".html", ".ipynb", ""}
BANNED_SUFFIX = {".csv", ".jpg", ".jpeg", ".png", ".zip", ".npy", ".parquet", ".pt", ".safetensors"}
ALLOWED_PNG = "docs/eda/"   # charts of counts and histograms; the image sheets are git-ignored
BANNED_DIRS = ("data/", "experiments/runs/", "submissions/", "reference/")
N = 5


def git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [x for x in out.split("\0") if x]


def norm(s: str) -> str:
    return " ".join(s.split())


def fingerprints() -> tuple[set[str], list[str]]:
    import csv

    grams: set[str] = set()
    if (RAW / "Train.csv").exists():
        with open(RAW / "Train.csv", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                w = norm(row["Target"]).split()
                grams.update(" ".join(w[i:i + N]) for i in range(len(w) - N + 1))
    sample = []
    if (RAW / "SampleSubmission.csv").exists():
        with open(RAW / "SampleSubmission.csv", encoding="utf-8-sig") as f:
            sample = [norm(r["Target"]) for r in csv.DictReader(f) if len(norm(r["Target"])) >= 8]
    return grams, sample


def problems(path: str, text: str | None, grams: set[str], sample: list[str]) -> list[str]:
    out = []
    p = Path(path)
    if path.startswith(BANNED_DIRS):
        out.append("lives in a data folder")
    if p.suffix.lower() in BANNED_SUFFIX and not (p.suffix.lower() == ".png" and path.startswith(ALLOWED_PNG)):
        out.append(f"{p.suffix} files may hold competition data")
    if text is None:
        return out
    t = norm(text)
    if grams:
        w = t.split()
        hits = {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)} & grams
        if hits:
            out.append(f"quotes {len(hits)} five-word runs from training transcripts, e.g. {sorted(hits)[0][:12]!r}...")
    hit = next((s for s in sample if s in t), None)
    if hit:
        out.append("quotes a sample-submission target")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        files = git("ls-files", "-z")
        read = lambda f: (ROOT / f).read_text(errors="ignore") if (ROOT / f).exists() else ""  # noqa: E731
    else:
        files = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
        read = lambda f: subprocess.run(["git", "show", f":{f}"], cwd=ROOT, capture_output=True).stdout.decode(errors="ignore")  # noqa: E731
    grams, sample = fingerprints()
    if not grams:
        print("check_public: data/raw/Train.csv not found; checking file types only", file=sys.stderr)
    bad = 0
    for f in files:
        text = read(f) if Path(f).suffix.lower() in TEXT else None
        for msg in problems(f, text, grams, sample):
            print(f"BLOCKED {f}: {msg}", file=sys.stderr)
            bad += 1
    if bad:
        print(f"check_public: {bad} problem(s). The repository is public; take competition text out and retry.", file=sys.stderr)
        return 1
    print(f"check_public: {len(files)} file(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
