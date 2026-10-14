"""eda.py — exploratory data analysis: writes the figures in docs/eda/ and prints the numbers behind them.

    .venv/bin/python notebooks/eda.py            # everything that runs on this machine (q1 q2 q3)
    .venv/bin/python notebooks/eda.py q1 q3      # only these sections

Every chart and number in vault/findings/ comes from this script. The test set is used only for the
distribution of its images; nothing here reads or labels a test line.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW, CACHE, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "docs" / "eda"
IMG = RAW / "images"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

TRAIN, TEST, THIRD = "#2a78d6", "#eb6834", "#1baf7a"  # train is always blue, test always orange
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.titlelocation": "left", "figure.dpi": 140})
SEED = 0
TALL = 150  # px; entrants report two scan batches split at this height (thread 34861)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote docs/eda/{name}.png")


def say(text: str) -> None:
    """A number that a vault note quotes. Keep the wording stable so notes can be re-checked."""
    print(f"  {text}")


def cached(name: str, make):
    """Run make() once and keep its DataFrame in data/processed/<name>.parquet."""
    path = CACHE / f"{name}.parquet"
    if not path.exists():
        make().to_parquet(path)
    return pd.read_parquet(path)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(RAW / "Train.csv")
    test = pd.read_csv(RAW / "Test.csv", encoding="utf-8-sig")  # Test.csv starts with a byte-order mark
    train["row"], test["row"] = np.arange(len(train)), np.arange(len(test))
    return train, test


# ---------------------------------------------------------------------------------------------------------
# Shared, cached inputs
# ---------------------------------------------------------------------------------------------------------

def _image_features() -> pd.DataFrame:
    """One row per image: size, tone, ink, JPEG settings, and its place and timestamp in images.zip."""
    with zipfile.ZipFile(RAW / "images.zip") as z:
        order = {Path(i.filename).stem: (k, i.date_time) for k, i in enumerate(x for x in z.infolist() if not x.is_dir())}
    rows = []
    for path in sorted(IMG.glob("*.jpg")):
        with Image.open(path) as im:
            w, h = im.size
            quant = sum(sum(q) for q in im.quantization.values())
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
        grey = rgb.mean(axis=2)
        paper = grey >= np.percentile(grey, 75)  # the lightest quarter of pixels stands in for the paper
        r, g, b = (rgb[..., c][paper].mean() for c in range(3))
        cut = (np.percentile(grey, 95) + np.percentile(grey, 2)) / 2
        ink_rows = (grey < cut).mean(axis=1)
        k, stamp = order[path.stem]
        rows.append({
            "ID": path.stem, "width": w, "height": h, "aspect": w / h, "bytes": path.stat().st_size,
            "bytes_per_px": path.stat().st_size / (w * h), "quant": quant,
            "paper_r": r, "paper_g": g, "paper_b": b, "paper_warmth": r - b,
            "grey_mean": grey.mean(), "grey_std": grey.std(), "dark_p2": np.percentile(grey, 2),
            "ink_share": (grey < cut).mean(), "ink_row_peak": ink_rows.max(), "ink_row_spread": ink_rows.std(),
            "sharpness": np.abs(np.diff(grey, axis=1)).mean(),
            "zip_pos": k, "zip_second": stamp[3] * 3600 + stamp[4] * 60 + stamp[5],
        })
    return pd.DataFrame(rows)


def features(train, test) -> pd.DataFrame:
    f = cached("image_features", _image_features)
    split = pd.concat([train.assign(split="train"), test.assign(split="test")])[["ID", "split", "row"]]
    return f.merge(split, on="ID", how="left", validate="one_to_one")


SHAPE = ["width", "height", "aspect", "bytes", "bytes_per_px"]
LOOK = ["paper_r", "paper_g", "paper_b", "paper_warmth", "grey_mean", "grey_std", "dark_p2", "ink_share",
        "ink_row_peak", "ink_row_spread", "sharpness"]


def _embeddings() -> pd.DataFrame:
    """ResNet-18 (ImageNet weights, torchvision, BSD licence) features of three square patches per line.

    Used only to group look-alike scans for this analysis. Each line is resized to 64 px high and three
    64 x 192 patches (left, centre, right) are embedded and averaged, so the vector describes the hand and the
    paper more than the words.
    """
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval()
    mean, std = np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)
    ids, out = [], []
    paths = sorted(IMG.glob("*.jpg"))
    for start in range(0, len(paths), 64):
        batch = []
        for path in paths[start:start + 64]:
            with Image.open(path) as im:
                im = im.convert("RGB")
                w = max(192, round(im.width * 64 / im.height))
                arr = (np.asarray(im.resize((w, 64), Image.BILINEAR), dtype=np.float32) / 255 - mean) / std
            for x0 in (0, (w - 192) // 2, w - 192):
                batch.append(arr[:, x0:x0 + 192].transpose(2, 0, 1))
            ids.append(path.stem)
        with torch.no_grad():
            v = net(torch.from_numpy(np.stack(batch))).numpy()
        out.append(v.reshape(-1, 3, v.shape[1]).mean(axis=1))
    return pd.DataFrame(np.vstack(out), index=pd.Index(ids, name="ID")).rename(columns=str).reset_index()


def _auc_cv(X, y, seed=SEED) -> float:
    """Out-of-fold AUC of a gradient-boosted classifier telling y=1 from y=0."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    model = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05, random_state=seed)
    p = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


# ---------------------------------------------------------------------------------------------------------
# Q1
# ---------------------------------------------------------------------------------------------------------

def q1_train_vs_test(train, test):
    """Do test lines come from the same scans, pages and writers as training lines?"""
    print("Q1 train against test")
    f = features(train, test)
    tr, te = f[f.split == "train"], f[f.split == "test"]
    say(f"images: {len(tr)} train, {len(te)} test")
    q = pd.crosstab(f.quant, [f.height > TALL, f.split])
    say("JPEG quantisation tables (sum of entries) by height batch and split: "
        + "; ".join(f"table {k}: short train {r[(False, 'train')]}, short test {r[(False, 'test')]}, "
                    f"tall train {r[(True, 'train')]}, tall test {r[(True, 'test')]}" for k, r in q.iterrows()))

    # 1a. the two height batches
    for name, d in (("train", tr), ("test", te)):
        say(f"{name}: height median {d.height.median():.0f} px, taller than {TALL} px {100 * (d.height > TALL).mean():.1f}% "
            f"({(d.height > TALL).sum()} lines); width median {d.width.median():.0f} px; aspect below 6: {(d.aspect < 6).sum()} lines")
    gap = f.height[(f.height > 110) & (f.height < 200)]
    say(f"lines with height 111-199 px: {len(gap)} of {len(f)}; "
        f"largest empty stretch of heights between {int(_widest_gap(f.height)[0])} and {int(_widest_gap(f.height)[1])} px")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, col, bins, label in ((axes[0], "height", np.arange(0, 500, 10), "image height (px)"),
                                 (axes[1], "width", np.arange(0, 5600, 100), "image width (px)"),
                                 (axes[2], "paper_warmth", np.linspace(-10, 80, 46), "paper warmth (red minus blue)")):
        ax.hist(tr[col], bins=bins, density=True, color=TRAIN, alpha=0.55, label="train")
        ax.hist(te[col], bins=bins, density=True, histtype="step", color=TEST, linewidth=1.6, label="test")
        ax.set_xlabel(label)
        ax.set_yticks([])
    axes[0].legend(frameon=False)
    axes[0].set_title("Test lines have the same sizes and paper tone as training lines")
    save(fig, "01_train_and_test_look_alike")

    # 1b. can a classifier tell train from test?
    y = (f.split == "test").astype(int).to_numpy()
    for label, cols in (("size and file", SHAPE), ("tone and ink", LOOK), ("all hand-made features", SHAPE + LOOK)):
        aucs = [_auc_cv(f[cols].to_numpy(), y, seed=s) for s in range(3)]
        say(f"train-against-test classifier on {label}: AUC {np.mean(aucs):.3f} (3 seeds, range {min(aucs):.3f}-{max(aucs):.3f})")
    emb = cached("resnet18_embeddings", _embeddings).set_index("ID").loc[f.ID].to_numpy()
    from sklearn.decomposition import PCA
    pcs = PCA(32, random_state=SEED).fit_transform(emb)
    aucs = [_auc_cv(pcs, y, seed=s) for s in range(3)]
    say(f"train-against-test classifier on image embedding (32 components): AUC {np.mean(aucs):.3f} (range {min(aucs):.3f}-{max(aucs):.3f})")
    rng = np.random.default_rng(SEED)
    null = [_auc_cv(pcs, rng.permutation(y), seed=s) for s in range(3)]
    say(f"same classifier with the train/test labels shuffled: AUC {np.mean(null):.3f}")

    # 1c. clusters of look-alike scans: does any cluster hold only test lines?
    from sklearn.cluster import KMeans
    share = y.mean()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for k, colour in ((20, THIRD), (100, MUTED)):
        lab = KMeans(k, n_init=4, random_state=SEED).fit_predict(pcs)
        g = pd.DataFrame({"c": lab, "test": y}).groupby("c").test.agg(["sum", "count"])
        lo, hi = _binom_band(g["count"].to_numpy(), share)
        outside = ((g["sum"] < lo) | (g["sum"] > hi)).sum()
        orphan = g.loc[g["sum"] == g["count"], "count"].sum()
        say(f"{k} clusters: test share per cluster {100 * (g['sum'] / g['count']).min():.0f}%-{100 * (g['sum'] / g['count']).max():.0f}% "
            f"(overall {100 * share:.1f}%); clusters outside the 95% chance band: {outside} of {k}; "
            f"test lines in clusters with no training line: {orphan}")
        ax.scatter(g["count"], 100 * g["sum"] / g["count"], s=22 if k == 20 else 10, color=colour, label=f"{k} clusters",
                   edgecolor=SURFACE, linewidth=0.6, zorder=3)
    n = np.arange(10, 520)
    lo, hi = _binom_band(n, share)
    ax.fill_between(n, 100 * lo / n, 100 * hi / n, color=GRID, zorder=1, label="what chance alone gives (95%)")
    ax.axhline(100 * share, color=MUTED, linewidth=0.8, zorder=2)
    ax.set_ylim(0, 60)
    ax.set_xlabel("lines in the cluster")
    ax.set_ylabel("test lines in the cluster (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Every cluster of look-alike scans holds test lines at the chance rate")
    save(fig, "02_no_cluster_is_test_only")

    # 1d. nearest neighbours: is a test line's closest look-alike as often a training line as chance says?
    from sklearn.neighbors import NearestNeighbors
    unit = pcs / np.linalg.norm(pcs, axis=1, keepdims=True)
    _, idx = NearestNeighbors(n_neighbors=2, metric="cosine").fit(unit).kneighbors(unit)
    nn_is_test = y[idx[:, 1]]
    say(f"nearest look-alike of a test line is a test line {100 * nn_is_test[y == 1].mean():.1f}% of the time; "
        f"of a training line {100 * nn_is_test[y == 0].mean():.1f}% (chance {100 * share:.1f}%)")

    _contact_sheet(tr[tr.height <= TALL].sample(8, random_state=SEED), te[te.height <= TALL].sample(8, random_state=SEED),
                   "03_short_lines_train_and_test", "Short scans (up to 150 px high): train left, test right")
    _contact_sheet(tr[tr.height > TALL].sample(6, random_state=SEED), te[te.height > TALL].sample(6, random_state=SEED),
                   "04_tall_lines_train_and_test", "Tall scans (over 150 px high): train left, test right")


def _widest_gap(values) -> tuple[float, float]:
    v = np.sort(np.unique(values))
    v = v[(v > 60) & (v < 300)]
    i = np.argmax(np.diff(v))
    return v[i], v[i + 1]


def _binom_band(n, p):
    from scipy.stats import binom
    return binom.ppf(0.025, n, p), binom.ppf(0.975, n, p)


def _contact_sheet(left: pd.DataFrame, right: pd.DataFrame, name: str, title: str, captions: dict | None = None,
                   colours=(TRAIN, TEST), row_height: float = 0.75) -> None:
    rows = max(len(left), len(right))
    fig, axes = plt.subplots(rows, 2, figsize=(13, row_height * rows + 0.5), squeeze=False)
    for col, (d, colour) in enumerate(zip((left, right), colours)):
        for r in range(rows):
            ax = axes[r, col]
            ax.axis("off")
            if r < len(d):
                rec = d.iloc[r]
                ax.imshow(Image.open(IMG / f"{rec.ID}.jpg"), aspect="auto")
                text = (captions or {}).get(rec.ID, f"{rec.ID}  {int(rec.width)}x{int(rec.height)}")
                ax.set_title(text, fontsize=6.5, color=colour, loc="left", pad=1, fontweight="normal")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    save(fig, name)


def norm_text(s: str) -> str:
    return " ".join(s.lower().split())


def _components(n: int, pairs) -> np.ndarray:
    """Label connected groups given index pairs."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pairs = np.asarray(list(pairs), dtype=int).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def _text_pairs(texts: list[str], cutoff: float) -> np.ndarray:
    """Pairs of lines whose normalised texts are at least `cutoff` similar (1 - edit distance / longer length)."""
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    sim = process.cdist(texts, texts, scorer=Levenshtein.normalized_similarity, score_cutoff=cutoff, workers=4, dtype=np.float32)
    i, j = np.where(np.triu(sim, 1) >= cutoff)
    return np.column_stack([i, j, np.floor(100 * sim[i, j] + 1e-4)]).astype(int)


def _dhash() -> pd.DataFrame:
    """A 256-bit difference hash of each line squeezed to 8 x 33 grey pixels: near-copies differ in few bits."""
    rows = []
    for path in sorted(IMG.glob("*.jpg")):
        with Image.open(path) as im:
            g = np.asarray(im.convert("L").resize((33, 8), Image.LANCZOS), dtype=np.int16)
        rows.append([path.stem, np.packbits(g[:, 1:] > g[:, :-1]).tobytes()])
    return pd.DataFrame(rows, columns=["ID", "bits"])


def _hash_dist(f: pd.DataFrame) -> np.ndarray:
    h = cached("dhash", _dhash).set_index("ID").loc[f.ID, "bits"]
    bits = np.unpackbits(np.frombuffer(b"".join(h), dtype=np.uint8).reshape(len(h), -1), axis=1).astype(np.float32)
    return bits.shape[1] - (bits @ bits.T + (1 - bits) @ (1 - bits).T)


def _hash_pairs(f: pd.DataFrame, max_bits: int) -> pd.DataFrame:
    dist = _hash_dist(f)
    i, j = np.where(np.triu(dist <= max_bits, 1))
    return pd.DataFrame({"i": i, "j": j, "bits": dist[i, j].astype(int)})


def groups(train: pd.DataFrame, f: pd.DataFrame, text_cutoff: float = 0.85, max_bits: int = 20) -> np.ndarray:
    """Group label per training line: same or near-same text, or near-copy image. The harness reuses this."""
    texts = [norm_text(s) for s in train.Target]
    found = _text_pairs(texts, text_cutoff)
    pairs = [p[:2] for p in found[found[:, 2] >= round(100 * text_cutoff)]]
    pos = pd.Series(np.arange(len(train)), index=train.ID)
    hp = _hash_pairs(f, max_bits)
    ids = f.ID.to_numpy()
    for a, b in zip(ids[hp.i], ids[hp.j]):
        if a in pos.index and b in pos.index:
            pairs.append((pos[a], pos[b]))
    return _components(len(train), pairs)


def q2_validation_unit(train, test):
    """Which lines must stay together in a fold?"""
    print("Q2 validation unit")
    f = features(train, test)
    n = len(train)
    exact = train.Target.map(str.strip)
    texts = [norm_text(s) for s in train.Target]
    say(f"training lines sharing their exact text with another line: {exact.duplicated(keep=False).sum()} "
        f"in {exact[exact.duplicated(keep=False)].nunique()} groups")
    nt = pd.Series(texts)
    say(f"sharing text once case and spacing are ignored: {nt.duplicated(keep=False).sum()} in {nt[nt.duplicated(keep=False)].nunique()} groups")

    pairs = _text_pairs(texts, 0.70)
    rows = []
    for cut in (95, 90, 85, 80, 70):
        lab = _components(n, pairs[pairs[:, 2] >= cut][:, :2])
        sizes = pd.Series(lab).value_counts()
        rows.append((cut, (sizes > 1).sum(), sizes[sizes > 1].sum(), sizes.max()))
        say(f"text at least {cut}% alike: {sizes[sizes > 1].sum()} lines in {(sizes > 1).sum()} groups, largest {sizes.max()}")

    # near-copy images, over train and test together
    hp = _hash_pairs(f, 40)
    split = f.split.to_numpy()
    hp["kind"] = np.where((split[hp.i] == "train") & (split[hp.j] == "train"), "train-train",
                          np.where((split[hp.i] == "test") & (split[hp.j] == "test"), "test-test", "train-test"))
    for bits in (10, 20, 30, 40):
        c = hp[hp.bits <= bits].kind.value_counts()
        say(f"image pairs within {bits} of 256 hash bits: train-train {c.get('train-train', 0)}, "
            f"train-test {c.get('train-test', 0)}, test-test {c.get('test-test', 0)}")
    # for train-train near-copies we may read both labels: are they the same text?
    from rapidfuzz.distance import Levenshtein
    tt = hp[(hp.kind == "train-train")].copy()
    target = train.set_index("ID").Target
    ids = f.ID.to_numpy()
    tt["text_sim"] = [Levenshtein.normalized_similarity(norm_text(target[ids[a]]), norm_text(target[ids[b]])) for a, b in zip(tt.i, tt.j)]
    for lo, hi in ((0, 10), (11, 20), (21, 30), (31, 40)):
        d = tt[(tt.bits >= lo) & (tt.bits <= hi)]
        if len(d):
            say(f"train-train image pairs at {lo}-{hi} bits: {len(d)}; same text (85% alike or more): {(d.text_sim >= 0.85).sum()}")
    nearest = np.sort(np.where(np.eye(len(f), dtype=bool), 999, _hash_dist(f)).min(axis=1))
    say(f"closest any two images come by hash: {int(nearest[0])} bits; median nearest {int(np.median(nearest))} bits (random pairs sit near 128)")

    # second check with the embedding: the 50 closest training pairs, and whether their transcripts match
    emb = cached("resnet18_embeddings", _embeddings).set_index("ID").loc[f.ID].to_numpy()
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cos = unit @ unit.T
    np.fill_diagonal(cos, -1)
    kinds = {"train-train": np.outer(split == "train", split == "train"), "test-test": np.outer(split == "test", split == "test")}
    kinds["train-test"] = ~(kinds["train-train"] | kinds["test-test"])
    for cut in (0.99, 0.98, 0.97):
        say(f"image pairs with embedding similarity above {cut}: " + ", ".join(f"{k} {int((np.triu(cos, 1) > cut)[m].sum())}" for k, m in kinds.items()))
    top = np.dstack(np.unravel_index(np.argsort(np.where(np.triu(kinds["train-train"], 1), cos, -1), axis=None)[::-1][:50], cos.shape))[0]
    sims = [Levenshtein.normalized_similarity(norm_text(target[ids[a]]), norm_text(target[ids[b]])) for a, b in top]
    say(f"the 50 most look-alike training pairs: embedding similarity {cos[top[-1, 0], top[-1, 1]]:.3f}-{cos[top[0, 0], top[0, 1]]:.3f}; "
        f"pairs with the same text (85% alike or more): {sum(v >= 0.85 for v in sims)}; median text likeness {100 * np.median(sims):.0f}%")
    left, right = f.iloc[top[:6, 0]], f.iloc[top[:6, 1]]
    cap = {r.ID: f"{r.ID}: {target[r.ID][:90]}" for r in pd.concat([left, right]).itertuples()}
    _contact_sheet(left, right, "05_most_alike_training_images", "The most look-alike training pairs are different lines (both columns are training lines)", cap,
                   colours=(TRAIN, TRAIN))

    lab = groups(train, f)
    sizes = pd.Series(lab).value_counts()
    say(f"fold groups (text 85% alike or image within 20 bits): {sizes[sizes > 1].sum()} lines in {(sizes > 1).sum()} groups, "
        f"largest {sizes.max()}; {100 * sizes[sizes > 1].sum() / n:.1f}% of training lines")

    # where the duplicates sit in Train.csv
    in_group = pd.Series(lab).map(sizes).to_numpy() > 1
    say(f"grouped lines among the first 42 rows of Train.csv: {in_group[:42].sum()}; among the other {n - 42}: {in_group[42:].sum()}")

    fig, ax = plt.subplots(figsize=(7, 3))
    cuts = [str(r[0]) for r in rows]
    ax.bar(cuts, [r[2] for r in rows], color=TRAIN, width=0.55)
    for x, r in enumerate(rows):
        ax.text(x, r[2] + 8, f"{r[2]} lines\n{r[1]} groups", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_ylim(0, max(r[2] for r in rows) * 1.3)
    ax.set_xlabel("how alike two transcripts must be to count as the same line (%)")
    ax.set_ylabel("training lines with a twin")
    ax.set_title(f"Few of the {n:,} training lines repeat another line's text")
    save(fig, "06_repeated_text_is_rare")

    # how much of a held-out fold's words were never seen in the other folds
    from sklearn.model_selection import GroupKFold
    unseen, once = [], []
    for fit, held in GroupKFold(5).split(texts, groups=lab):
        vocab = {w for k in fit for w in train.Target.iloc[k].split()}
        words = [w for k in held for w in train.Target.iloc[k].split()]
        unseen.append(np.mean([w not in vocab for w in words]))
    counts = pd.Series([w for s in train.Target for w in s.split()]).value_counts()
    say(f"word tokens in a held-out fold never seen in the other four folds: {100 * np.mean(unseen):.1f}% "
        f"(folds {', '.join(f'{100 * u:.1f}' for u in unseen)}); distinct words {len(counts)}, seen once {100 * (counts == 1).mean():.0f}%")
    chars = pd.Series(list("".join(train.Target))).value_counts()
    say(f"distinct characters {len(chars)}; seen fewer than 10 times: {(chars < 10).sum()} ({' '.join(chars[chars < 10].index)})")


def q3_leak(train, test):
    """Does anything outside the picture give the answer away: size, file order, IDs, the sample submission?"""
    print("Q3 leak")
    f = features(train, test)
    tr = f[f.split == "train"].merge(train[["ID", "Target"]], on="ID").sort_values("row")
    tr["chars"] = tr.Target.str.len()
    tr["words"] = tr.Target.str.split().str.len()
    tr["tall"] = tr.height > TALL

    # 3a. length against image size
    from scipy.stats import spearmanr
    for name, d in (("short scans", tr[~tr.tall]), ("tall scans", tr[tr.tall])):
        say(f"{name} ({len(d)} lines): characters against width rho {spearmanr(d.chars, d.width)[0]:.2f}, against aspect ratio "
            f"{spearmanr(d.chars, d.aspect)[0]:.2f}, against file size {spearmanr(d.chars, d.bytes)[0]:.2f}; "
            f"width per character median {np.median(d.width / d.chars):.1f} px; height median {d.height.median():.0f} px")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, (name, d) in zip(axes, (("Short scans", tr[~tr.tall]), ("Tall scans", tr[tr.tall]))):
        ax.scatter(d.aspect, d.chars, s=4, color=TRAIN, alpha=0.35, linewidth=0)
        ax.set_xlabel("image width divided by height")
        ax.set_title(name, fontweight="normal", fontsize=10)
    axes[0].set_ylabel("characters in the transcript")
    fig.suptitle("Image shape only loosely tracks how long the text is", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    save(fig, "07_image_shape_and_text_length")

    # 3b. file order in the zip, timestamps, sorted names, csv row order
    z = f.sort_values("zip_pos")
    is_test = (z.split == "test").to_numpy()
    runs = 1 + (is_test[1:] != is_test[:-1]).sum()
    p = is_test.mean()
    expected = 1 + 2 * (len(z) - 1) * p * (1 - p)
    say(f"zip order: {runs} runs of train/test against {expected:.0f} expected if shuffled; "
        f"test share by zip timestamp: " + ", ".join(f"{100 * v:.1f}%" for v in (z.split == "test").groupby(z.zip_second).mean()))
    rng = np.random.default_rng(SEED)
    for label, order in (("zip order", f.sort_values("zip_pos")), ("file-name order", f.sort_values("ID")),
                         ("Train.csv then Test.csv row order", f.sort_values(["split", "row"]))):
        warm = order.paper_warmth.to_numpy()
        tall = (order.height > TALL).to_numpy()
        say(f"{label}: neighbours differ in paper warmth by {np.abs(np.diff(warm)).mean():.2f} "
            f"(shuffled {np.abs(np.diff(rng.permutation(warm))).mean():.2f}); "
            f"neighbours in the same height batch {100 * (tall[1:] == tall[:-1]).mean():.1f}% "
            f"(shuffled {100 * (tall.mean() ** 2 + (1 - tall.mean()) ** 2):.1f}%)")
    from rapidfuzz import fuzz
    for label, key in (("zip order", "zip_pos"), ("Train.csv row order", "row")):
        s = tr.sort_values(key).Target.map(norm_text).tolist()
        adj = [fuzz.ratio(a, b) for a, b in zip(s[:-1], s[1:])]
        sh = rng.permutation(len(s))
        base = [fuzz.ratio(s[a], s[b]) for a, b in zip(sh[:-1], sh[1:])]
        say(f"{label}: neighbouring training transcripts are {np.mean(adj):.1f}% alike (shuffled {np.mean(base):.1f}%); "
            f"neighbour pairs at least 80% alike: {sum(a >= 80 for a in adj)}")
    s = tr.Target.map(norm_text).tolist()
    adj = np.array([fuzz.ratio(a, b) for a, b in zip(s[:-1], s[1:])])
    say(f"Train.csv: neighbour pairs at least 80% alike in rows 1-42: {(adj[:42] >= 80).sum()}; in rows 43-{len(s)}: {(adj[42:] >= 80).sum()}")

    # 3c. the ID strings
    ids = tr.ID.to_numpy()
    first = pd.Series([i[0] for i in f.ID])
    say(f"IDs: all {f.ID.str.len().unique().tolist()} characters; first character against split: "
        f"test share ranges {100 * (f.split == 'test').groupby(first.to_numpy()).mean().min():.0f}%-"
        f"{100 * (f.split == 'test').groupby(first.to_numpy()).mean().max():.0f}% over {first.nunique()} first characters")
    code = np.array([[ord(c) for c in i] for i in ids])
    rho = [abs(spearmanr(code[:, k], tr.chars)[0]) for k in range(code.shape[1])]
    say(f"IDs: largest correlation between any ID character and transcript length {max(rho):.3f}")

    # 3d. the sample submission
    sub = pd.read_csv(RAW / "SampleSubmission.csv").merge(f[["ID", "width", "aspect"]], on="ID")
    n_chars = sub.Target.fillna("").str.len()
    say(f"sample submission: {100 * sub.Target.isna().mean():.1f}% blank; example text length against test image width rho "
        f"{spearmanr(n_chars, sub.width)[0]:.3f}, against aspect ratio {spearmanr(n_chars, sub.aspect)[0]:.3f}; "
        f"example targets found among training transcripts: {sub.Target.isin(train.Target).sum()}")

    # 3e. can size alone predict the text length? (the best a metadata-only cheat could do)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    pred = cross_val_predict(HistGradientBoostingRegressor(max_depth=4, random_state=SEED), tr[SHAPE], tr.chars,
                             cv=KFold(5, shuffle=True, random_state=SEED))
    say(f"transcript length predicted from size and file size alone: mean error {np.abs(pred - tr.chars).mean():.1f} characters "
        f"(guessing the average: {np.abs(tr.chars.mean() - tr.chars).mean():.1f}); lengths run {tr.chars.min()}-{tr.chars.max()}")


def look(train, test):
    """Training lines beside their transcripts, to read by hand: tall crops, superscripts, the squarest crops."""
    print("Look at training lines with their labels")
    f = features(train, test)
    tr = f[f.split == "train"].merge(train[["ID", "Target"]], on="ID")
    cap = {r.ID: f"{int(r.width)}x{int(r.height)}  |{r.Target}|" for r in tr.itertuples()}
    tall = tr[tr.height > TALL].sample(14, random_state=SEED)
    _contact_sheet(tall.iloc[:7], tall.iloc[7:], "08_tall_training_lines_with_labels",
                   "Tall training crops and their labels (text between bars)", cap, colours=(TRAIN, TRAIN), row_height=1.1)
    caret = tr[tr.Target.str.contains("^", regex=False) & (tr.height <= TALL)].sample(7, random_state=SEED)
    square = tr.sort_values("aspect").head(7)
    _contact_sheet(caret, square, "09_superscripts_and_squarest_crops",
                   "Left: lines with ^ superscripts. Right: the squarest crops", cap, colours=(TRAIN, TRAIN), row_height=1.1)
    ws = train.Target
    say(f"transcripts with a leading or trailing space: {(ws != ws.str.strip()).sum()}; with a double space: {ws.str.contains('  ', regex=False).sum()}; "
        f"with ^: {100 * ws.str.contains('^', regex=False).mean():.1f}%")
    say(f"transcript length: {ws.str.len().min()}-{ws.str.len().max()} characters, {ws.str.split().str.len().min()}-{ws.str.split().str.len().max()} words")


SECTIONS = {"q1": q1_train_vs_test, "q2": q2_validation_unit, "q3": q3_leak, "look": look}
# q4 (label noise), q5 (score noise) and q6 (error types) need a trained reader, which needs a GPU.

if __name__ == "__main__":
    train, test = load()
    for key in sys.argv[1:] or SECTIONS:
        SECTIONS[key](train, test)
