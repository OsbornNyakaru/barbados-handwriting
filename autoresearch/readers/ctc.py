"""Letter-by-letter reader: a convolutional network over the line, a recurrent layer, and CTC.

Backbones:
  "crnn"      trained from scratch (no pretrained weights at all), the same shape as Kraken's default network.
  timm names  e.g. "convnext_large.fb_in22k_ft_in1k" (Apache 2.0): an ImageNet backbone cut after its second stage.

It reads the line left to right and outputs one letter distribution per column, so it spells what it sees and
never modernises a spelling. Decoding is greedy, or a beam search with a character language model built from the
fold's training transcripts (readers/lm.py). Held-out log-probabilities are returned so other decoders can be
tried later on a CPU without training again.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .images import load_lines
from .lm import CharLM, beam_decode, greedy_decode

DEFAULTS = {
    "backbone": "crnn",
    "height": 80,            # pixels; both batches are brought to this height (about 19 and 14 px per letter)
    "max_width": 2600,
    "channels": [64, 128, 256, 256],
    "rnn_hidden": 256,
    "rnn_layers": 3,
    "dropout": 0.2,
    "epochs": 60,
    "batch": 24,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "warmup": 0.05,          # share of steps
    "aug": "basic",          # "none" | "basic"
    "inner_val": 200,        # training-part lines held back to choose the best epoch (never the fold's held-out lines)
    "eval_every": 2,
    "decode": "greedy",      # "greedy" | "beam"
    "lm_order": 6,
    "lm_weight": 0.5,
    "lm_bonus": 0.0,
    "beam": 10,
    "limit_train": 0,        # >0: train on this many lines only (smoke tests)
    "workers": 4,
    "amp": True,
}


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def build_model(p: dict, n_out: int):
    torch, nn = _torch()

    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()
            ch, layers, c_in = p["channels"], [], 1
            pools = [(2, 2), (2, 2), (2, 1), (2, 1)]
            for i, c in enumerate(ch):
                layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                if i >= 2:
                    layers += [nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                layers += [nn.MaxPool2d(pools[i])]
                if i >= 2:
                    layers += [nn.Dropout2d(p["dropout"] / 2)]
                c_in = c
            self.cnn = nn.Sequential(*layers)
            h = p["height"] // 16
            self.proj = nn.Linear(c_in * h, p["rnn_hidden"] * 2)
            self.rnn = nn.LSTM(p["rnn_hidden"] * 2, p["rnn_hidden"], num_layers=p["rnn_layers"], bidirectional=True,
                               batch_first=True, dropout=p["dropout"])
            self.head = nn.Sequential(nn.Dropout(p["dropout"]), nn.Linear(p["rnn_hidden"] * 2, n_out))
            self.stride = 4

        def forward(self, x):                       # x [B, 1, H, W]
            f = self.cnn(x)                         # [B, C, H/16, W/4]
            f = f.permute(0, 3, 1, 2).flatten(2)    # [B, W/4, C*H/16]
            f, _ = self.rnn(self.proj(f))
            return self.head(f)                     # [B, T, n_out]

    class TimmCTC(nn.Module):
        def __init__(self):
            super().__init__()
            import timm

            self.body = timm.create_model(p["backbone"], pretrained=p.get("pretrained", True), features_only=True,
                                          out_indices=(1,))
            info = self.body.feature_info.get_dicts()[-1]
            self.stride = info["reduction"]
            h = p["height"] // self.stride
            self.proj = nn.Linear(info["num_chs"] * h, p["rnn_hidden"] * 2)
            self.rnn = nn.LSTM(p["rnn_hidden"] * 2, p["rnn_hidden"], num_layers=p["rnn_layers"], bidirectional=True,
                               batch_first=True, dropout=p["dropout"])
            self.head = nn.Sequential(nn.Dropout(p["dropout"]), nn.Linear(p["rnn_hidden"] * 2, n_out))

        def forward(self, x):
            f = self.body(x.expand(-1, 3, -1, -1))[-1]
            f = f.permute(0, 3, 1, 2).flatten(2)
            f, _ = self.rnn(self.proj(f))
            return self.head(f)

    return CRNN() if p["backbone"] == "crnn" else TimmCTC()


def augment(a: np.ndarray, rng: np.random.Generator, height: int, max_width: int) -> np.ndarray:
    """Mild, label-preserving changes: width stretch, shear, small rotation, blur, contrast."""
    from PIL import Image, ImageFilter

    im = Image.fromarray(a)
    w = int(np.clip(im.width * rng.uniform(0.85, 1.15), 16, max_width))
    im = im.resize((w, height), Image.BICUBIC)
    shear = rng.uniform(-0.25, 0.25)
    pad = int(abs(shear) * height) + 2
    im = im.transform((w + pad, height), Image.AFFINE, (1, shear, -shear * height / 2 - pad / 2, 0, 1, 0),
                      resample=Image.BICUBIC, fillcolor=int(np.median(a)))
    im = im.rotate(rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, fillcolor=int(np.median(a)))
    if rng.random() < 0.2:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    out = np.asarray(im, dtype=np.float32)
    lo, hi = np.percentile(out, 2), np.percentile(out, 98)
    out = (out - lo) / max(hi - lo, 1.0) * rng.uniform(0.8, 1.0) * 255 + rng.uniform(0, 20)
    return np.clip(out, 0, 255).astype(np.uint8)[:, :max_width]


def to_tensor_batch(arrs, torch):
    """Ink bright on a zero background, right-padded to the widest line."""
    w = max(a.shape[1] for a in arrs)
    h = arrs[0].shape[0]
    x = np.zeros((len(arrs), 1, h, w), dtype=np.float32)
    for i, a in enumerate(arrs):
        x[i, 0, :, : a.shape[1]] = 1.0 - a.astype(np.float32) / 255.0
    return torch.from_numpy(x), [a.shape[1] for a in arrs]


def _batches(order, widths, batch, rng):
    """Batches of similar width (less padding), in random order."""
    idx = np.array(order)
    idx = idx[np.argsort(np.array(widths)[idx] + rng.uniform(0, 60, len(idx)))]
    chunks = [idx[i:i + batch] for i in range(0, len(idx), batch)]
    rng.shuffle(chunks)
    return chunks


def _cer(pred, ref):
    from rapidfuzz.distance import Levenshtein

    return sum(Levenshtein.distance(a, b) for a, b in zip(pred, ref)) / max(1, sum(len(b) for b in ref))


def fit_predict(params: dict, train_paths, train_texts, apply_paths, seed: int, ctx) -> tuple[list[str], dict]:
    """Train on (train_paths, train_texts), read apply_paths. Returns predictions and extras (log-probs, charset)."""
    torch, nn = _torch()
    p = {**DEFAULTS, **params}
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = p["amp"] and dev.type == "cuda"

    texts = [str(t) for t in train_texts]
    if p["limit_train"]:
        keep = rng.choice(len(texts), size=min(p["limit_train"], len(texts)), replace=False)
        train_paths, texts = [train_paths[i] for i in keep], [texts[i] for i in keep]
    charset = sorted(set("".join(texts)))
    index = {c: i + 1 for i, c in enumerate(charset)}

    t0 = time.time()
    tr_img = load_lines(train_paths, p["height"], p["max_width"], ctx.cache)
    ap_img = load_lines(apply_paths, p["height"], p["max_width"], ctx.cache)
    ctx.log(f"ctc: {len(tr_img)} training lines, {len(ap_img)} to read, {len(charset)} letters, images {time.time() - t0:.0f}s")

    n = len(tr_img)
    iv = rng.choice(n, size=min(p["inner_val"], n // 10), replace=False) if p["inner_val"] else np.array([], int)
    fit = np.setdiff1d(np.arange(n), iv)

    model = build_model(p, len(charset) + 1).to(dev)
    stride = model.stride
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    steps = p["epochs"] * math.ceil(len(fit) / p["batch"])
    warm = max(1, int(steps * p["warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    ctx.log(f"ctc: {sum(x.numel() for x in model.parameters()) / 1e6:.1f}M parameters on {dev}, {steps} steps")

    def read(imgs, bs=32):
        model.eval()
        out = []
        with torch.no_grad():
            order = np.argsort([a.shape[1] for a in imgs])
            res = [None] * len(imgs)
            for i in range(0, len(order), bs):
                ids = order[i:i + bs]
                x, ws = to_tensor_batch([imgs[j] for j in ids], torch)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    y = model(x.to(dev))
                lp = y.float().log_softmax(-1).cpu().numpy()
                for k, j in enumerate(ids):
                    res[j] = lp[k, : max(1, math.ceil(ws[k] / stride))]
            out = res
        model.train()
        return out

    widths = [a.shape[1] for a in tr_img]
    best, best_state, step = math.inf, None, 0
    pool = None
    if p["aug"] != "none" and p["workers"] > 0:
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(p["workers"])
    model.train()
    for ep in range(p["epochs"]):
        te, tl = time.time(), 0.0
        for b in _batches(fit, widths, p["batch"], rng):
            seeds = rng.integers(0, 2**31, len(b))
            if p["aug"] == "none":
                arrs = [tr_img[i] for i in b]
            elif pool:
                arrs = list(pool.map(lambda a: augment(tr_img[a[0]], np.random.default_rng(a[1]), p["height"], p["max_width"]),
                                     zip(b, seeds)))
            else:
                arrs = [augment(tr_img[i], np.random.default_rng(s), p["height"], p["max_width"]) for i, s in zip(b, seeds)]
            x, ws = to_tensor_batch(arrs, torch)
            tgt = [torch.tensor([index[c] for c in texts[i]], dtype=torch.long) for i in b]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                y = model(x.to(dev))
            lp = y.float().log_softmax(-1).transpose(0, 1)          # [T, B, C]
            in_len = torch.tensor([min(lp.shape[0], math.ceil(w / stride)) for w in ws], dtype=torch.long)
            loss = ctc_loss(lp, torch.cat(tgt).to(dev), in_len, torch.tensor([len(t) for t in tgt], dtype=torch.long))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            tl += float(loss)
        msg = f"ctc: epoch {ep + 1}/{p['epochs']} loss {tl / max(1, math.ceil(len(fit) / p['batch'])):.3f} {time.time() - te:.0f}s"
        last = ep + 1 == p["epochs"]
        if len(iv) and ((ep + 1) % p["eval_every"] == 0 or last):
            cer = _cer([greedy_decode(lp_, charset) for lp_ in read([tr_img[i] for i in iv])], [texts[i] for i in iv])
            msg += f", inner CER {cer:.4f}"
            if cer < best:
                best, best_state = cer, {k: v.detach().clone() for k, v in model.state_dict().items()}
        ctx.log(msg)
        if ctx.should_stop():
            ctx.log("ctc: time limit reached, stopping training early")
            break
    if pool:
        pool.shutdown()
    if best_state is not None:
        model.load_state_dict(best_state)

    logps = read(ap_img)
    if p["decode"] == "beam":
        lm = CharLM(texts, p["lm_order"])
        preds = [beam_decode(lp_, charset, lm, p["lm_weight"], p["lm_bonus"], p["beam"]) for lp_ in logps]
    else:
        preds = [greedy_decode(lp_, charset) for lp_ in logps]
    return preds, {"logp": logps, "charset": charset, "inner_cer": best}
