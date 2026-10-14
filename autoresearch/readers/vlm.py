"""Vision-language reader: Qwen3-VL (Apache 2.0) fine-tuned with LoRA on the fold's training lines.

The adapter covers both the language layers and the vision tower (research R-001: training the vision part matters
for old hands). Decoding is greedy. Images are resized to a fixed height and a width that is a multiple of the
model's 32-pixel token grid, so the processor does not resize them again.
"""

from __future__ import annotations

import math
import re
import time

import numpy as np

DEFAULTS = {
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "height": 128,
    "max_width": 2560,
    "prompt": "Transcribe the handwritten line exactly as written, keeping the original spelling, capitals and marks.",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|attn\.qkv|attn\.proj|linear_fc1|linear_fc2)$",
    "train_vision": True,
    "epochs": 3,
    "batch": 8,
    "grad_accum": 2,
    "lr": 1e-4,
    "warmup": 0.03,
    "max_new_tokens": 160,
    "gen_batch": 24,
    "limit_train": 0,
    "grad_checkpointing": True,
    "nbest": 1,              # >1: beam search, keep this many readings per line (candidates for combine "choose")
}


def _image(path: str, height: int, max_width: int):
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w = round(im.width * height / im.height)
    w = int(min(max_width, max(64, round(w / 32) * 32)))
    return im.resize((w, height), Image.BICUBIC)


def _load(p: dict, ctx):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    key = ("vlm", p["model_id"])
    if key not in ctx.cache:
        for k in [k for k in ctx.cache if isinstance(k, tuple) and k and k[0] == "vlm"]:
            del ctx.cache[k]
        proc = AutoProcessor.from_pretrained(p["model_id"])
        ctx.cache[key] = proc
    proc = ctx.cache[key]
    model = AutoModelForImageTextToText.from_pretrained(p["model_id"], dtype=torch.bfloat16, device_map="cuda"
                                                        if torch.cuda.is_available() else None)
    ip = proc.image_processor
    # every image is already a multiple of 32 px; open the processor's pixel bounds so it keeps them as they are
    ip.size = {"shortest_edge": 32 * 32, "longest_edge": p["height"] * p["max_width"]}
    for name in ("min_pixels", "max_pixels"):
        if hasattr(ip, name):
            setattr(ip, name, 32 * 32 if name == "min_pixels" else p["height"] * p["max_width"])
    proc.tokenizer.padding_side = "left"
    return proc, model


def _messages(prompt: str, answer: str | None = None):
    m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        m.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return m


def fit_predict(params: dict, train_paths, train_texts, apply_paths, seed: int, ctx) -> tuple[list[str], dict]:
    import torch
    from peft import LoraConfig, get_peft_model

    p = {**DEFAULTS, **params}
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    texts = [str(t) for t in train_texts]
    if p["limit_train"]:
        keep = rng.choice(len(texts), size=min(p["limit_train"], len(texts)), replace=False)
        train_paths, texts = [train_paths[i] for i in keep], [texts[i] for i in keep]

    t0 = time.time()
    proc, model = _load(p, ctx)
    tok = proc.tokenizer
    names = [n for n, _ in model.named_modules() if re.fullmatch(p["lora_targets"], n)]
    if not p["train_vision"]:
        names = [n for n in names if "visual" not in n]
    if not names:
        raise RuntimeError("no LoRA target modules matched; check lora_targets against model.named_modules()")
    model = get_peft_model(model, LoraConfig(r=p["lora_r"], lora_alpha=p["lora_alpha"], lora_dropout=p["lora_dropout"],
                                             target_modules=names, task_type="CAUSAL_LM"))
    if p["grad_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    n_vis = sum("visual" in n for n in names)
    ctx.log(f"vlm: {p['model_id']} loaded in {time.time() - t0:.0f}s; LoRA on {len(names)} modules ({n_vis} in the vision tower); "
            f"{sum(x.numel() for x in model.parameters() if x.requires_grad) / 1e6:.1f}M trainable")

    dev = next(model.parameters()).device
    prompt_txt = proc.apply_chat_template(_messages(p["prompt"]), tokenize=False, add_generation_prompt=True)

    def batch_inputs(idx):
        imgs = [_image(train_paths[i], p["height"], p["max_width"]) for i in idx]
        full = [proc.apply_chat_template(_messages(p["prompt"], texts[i]), tokenize=False) for i in idx]
        tok.padding_side = "right"
        enc = proc(text=full, images=imgs, return_tensors="pt", padding=True)
        pre = proc(text=[prompt_txt] * len(idx), images=imgs, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for r, n_pre in enumerate(pre["attention_mask"].sum(1).tolist()):
            labels[r, :n_pre] = -100
        enc["labels"] = labels
        tok.padding_side = "left"
        return {k: v.to(dev) for k, v in enc.items()}

    opt = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=p["lr"], weight_decay=0.0)
    per_epoch = math.ceil(len(texts) / p["batch"])
    steps = math.ceil(p["epochs"] * per_epoch / p["grad_accum"])
    warm = max(1, int(steps * p["warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    model.train()
    k = 0
    for ep in range(p["epochs"]):
        te, tl, nb = time.time(), 0.0, 0
        order = rng.permutation(len(texts))
        for i in range(0, len(order), p["batch"]):
            out = model(**batch_inputs(order[i:i + p["batch"]]))
            (out.loss / p["grad_accum"]).backward()
            tl, nb, k = tl + float(out.loss), nb + 1, k + 1
            if k % p["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if nb % 50 == 0:
                ctx.log(f"vlm: epoch {ep + 1} batch {nb}/{per_epoch} loss {tl / nb:.4f}")
            if ctx.should_stop():
                break
        ctx.log(f"vlm: epoch {ep + 1}/{p['epochs']} loss {tl / max(nb, 1):.4f} {time.time() - te:.0f}s")
        if ctx.should_stop():
            ctx.log("vlm: time limit reached, stopping training early")
            break

    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    preds = [""] * len(apply_paths)
    nbest: dict[int, list[str]] = {}
    order = np.argsort([_image(pth, p["height"], p["max_width"]).width for pth in apply_paths])
    tg = time.time()
    with torch.no_grad():
        for i in range(0, len(order), p["gen_batch"]):
            ids = order[i:i + p["gen_batch"]]
            imgs = [_image(apply_paths[j], p["height"], p["max_width"]) for j in ids]
            enc = proc(text=[prompt_txt] * len(ids), images=imgs, return_tensors="pt", padding=True).to(dev)
            nb = int(p["nbest"])
            gen = model.generate(**enc, max_new_tokens=p["max_new_tokens"], do_sample=False, num_beams=max(1, nb),
                                 num_return_sequences=max(1, nb))
            outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, j in enumerate(ids):
                group = [" ".join(o.split()) for o in outs[r * max(1, nb):(r + 1) * max(1, nb)]]
                preds[j] = group[0]
                if nb > 1:
                    nbest[j] = group
    ctx.log(f"vlm: read {len(preds)} lines in {time.time() - tg:.0f}s")
    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds, ({"nbest": [nbest.get(j, [preds[j]]) for j in range(len(preds))]} if p["nbest"] > 1 else {})
