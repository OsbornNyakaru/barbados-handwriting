"""Upload the competition files to a PRIVATE Hugging Face dataset, for the molab runner.

The GitHub repository is public and Zindi's terms forbid sharing the data, so molab downloads it from this private
dataset instead (notebooks/molab_runner.py, with a token pasted in the session). Refuses if the dataset is public.

    .venv/bin/python scripts/push_data_to_hf.py            # -> <you>/barbados-handwriting-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FILES = ("Train.csv", "Test.csv", "SampleSubmission.csv", "images.zip", "folds.csv")

CARD = """---
license: other
pretty_name: barbados-handwriting competition data (private mirror)
---

Private mirror of the Zindi R.O.A.D. Barbados Historic Handwriting Challenge files, for the molab runner only.
Do **not** make this repository public: Zindi's terms forbid redistributing the data. SHA-256 in `manifest.json`.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_folds() -> None:
    """data/raw/folds.csv: the frozen fold of every training line, made with the pinned local packages."""
    sys.path.insert(0, str(ROOT / "autoresearch"))
    import prepare as P

    train, _ = P.load()
    if P.fold_hash(train) != P.FOLD_HASH:
        raise SystemExit(f"local folds {P.fold_hash(train)} != frozen {P.FOLD_HASH}: run this in the pinned .venv")
    import pandas as pd

    pd.DataFrame({P.ID_COL: train[P.ID_COL], "fold": P.fold_of(train)}).to_csv(RAW / "folds.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--only", nargs="*", help="upload only these files (e.g. folds.csv)")
    a = ap.parse_args()
    write_folds()
    missing = [f for f in FILES if not (RAW / f).exists()]
    if missing:
        print(f"missing in {RAW}: {missing}", file=sys.stderr)
        return 1
    api = HfApi()
    repo = a.repo or f"{api.whoami()['name']}/barbados-handwriting-data"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        print(f"refusing to upload: {repo} is PUBLIC", file=sys.stderr)
        return 1
    manifest = {f: sha256(RAW / f) for f in FILES}
    stage = ROOT / "data" / "hf_card"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (stage / "README.md").write_text(CARD)
    for f in (a.only or FILES):
        api.upload_file(path_or_fileobj=str(RAW / f), path_in_repo=f, repo_id=repo, repo_type="dataset", commit_message=f"upload {f}")
    for f in ("manifest.json", "README.md"):
        api.upload_file(path_or_fileobj=str(stage / f), path_in_repo=f, repo_id=repo, repo_type="dataset", commit_message=f"upload {f}")
    remote = {s.rfilename: s for s in api.dataset_info(repo, files_metadata=True).siblings}
    bad = [f for f in FILES if f not in remote or remote[f].size != (RAW / f).stat().st_size]
    print(f"{repo} (private): {sorted(remote)}")
    if bad:
        print(f"size mismatch or missing after upload: {bad}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
