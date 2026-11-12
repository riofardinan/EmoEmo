"""One training run for a dataset pair (`--pair xedgo`, `ekfine`, `enghin`, `engsun`;
used by the RP1, RP2 and RP4 pilots in _pilot/). Three modes:

  stage1  train on one dataset with its own labels (head = that inventory).
          This is also the dataset's native reference model.
  stage2  load the stage-1 model of the first dataset (same seed), lay its
          head out on the union label space, then train on the other dataset
          with one continual-learning method:
            ft   masked BCE on the new dataset's columns only;
            lwf  ft + lamda * BCE to the stage-1 model's sigmoid outputs, on the
                 first dataset's columns that the new rows do not annotate;
            er   ft on the new rows plus a buffer of first-dataset rows (up to
                 `er_per_class` per label, emogo's "random" ER), each scored
                 on its own dataset's columns.
            lwf_all  as lwf, but distils *every* first-dataset column, including
                 merged ones that the new rows also annotate (old soft target
                 and new label side by side). The obvious fix for the merge
                 cost; a baseline for HS1/HS2.
            map  (HS2/HS1; separate space only) ft plus distillation *through a
                 learned correspondence*: a linear map M from the new
                 dataset's logits to the first dataset's labels is trained
                 jointly, and lamda * BCE(M(z_new), teacher) keeps the old
                 model's soft targets readable from the new outputs. The old
                 inventory is read as M(z_new) ("readout_*" arrays); M is
                 saved in corr.npz. M has its own Adam (lr `map_lr`), zero
                 weights, and a bias set to the teacher's mean on the new texts.
  joint   train once on both datasets together over the union label space.

All losses are masked: a row is supervised only on the columns of its own
dataset (utils/data.py). Recipe as ../emogo and ../bertgo: bert-base-cased,
batch 16, lr 5e-5, the original BERT optimizer, 10% warmup, gradient clip 1.0,
4 epochs per stage, fp32, LwF lamda 3. Two speed changes, neither changing the
arithmetic: batches are padded to their longest row instead of to 50 tokens
(the attention mask makes this equivalent up to float rounding), and on GPU the
optimizer update runs as one compiled kernel (utils/fast_adam.py, checked
against utils/optimization.py). `--optimizer reference` restores the original.
`--bias-correction` (off by default) adds Adam's bias correction; with lr 2e-5
and 20 epochs it is the RP4 recipe (Mosbach et al., 2021).

`--save-train-logits` also writes logits on the train splits (needed to fit a
correspondence on training texts). Writes <out>/<run>/logits.npz (default <out> = results/) (dev and test logits and labels for both
datasets), meta.json and train.log; stage1 also writes model.pt.

    python train.py --mode stage1 --first xed --seed 0
    python train.py --mode stage2 --first xed --space merge --method lwf --seed 0
    python train.py --mode joint --space separate --seed 0
    python train.py --pair ekfine --mode stage1 --first ekman --seed 0 --out results/try
"""

import argparse
import importlib.util
import json
import logging
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast

from nets.net import EmoNet
from utils.data import PAIRS, LabelSpace, Encoded, batches, labels_of, load
from utils import fast_adam, optimization

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS = sorted({d for pair in PAIRS.values() for d in pair})
N_NATIVE = {d: len(labels_of(d)) for d in DATASETS}


def other_of(pair, first):
    """The other dataset of a pair (a dataset may belong to several pairs)."""
    b, o = PAIRS[pair]
    return o if first == b else b


def run_name(a):
    if a.mode == "stage1":
        return f"stage1_{a.first}_s{a.seed}"
    if a.mode == "joint":
        return f"joint_{a.space}_s{a.seed}"
    return f"stage2_{a.first}2{other_of(a.pair, a.first)}_{a.space}_{a.method}_s{a.seed}"


def masked_bce(z, y, m):
    """Mean over a row's annotated columns, then over rows with any."""
    per = F.binary_cross_entropy_with_logits(z, y, reduction="none") * m
    n = m.sum(1)
    keep = n > 0
    if not keep.any():
        return z.sum() * 0.0
    return (per.sum(1)[keep] / n[keep]).mean()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optim_cfg(a):
    return SimpleNamespace(warmup_proportion=a.warmup, weight_decay_rate=0.01,
                           exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
                           learning_rate=a.lr, adam_beta1=0.9, adam_beta2=0.999,
                           adam_epsilon=1e-6, bias_correction=a.bias_correction)


def fit(model, data, a, device, log, loss_fn, extra_opts=()):
    rng = np.random.default_rng(a.seed)
    steps = (len(data) // a.batch_size) * a.epochs
    lib = fast_adam if a.optimizer == "fused" else optimization
    opt, sched, warm = lib.create_optimizer_and_schedule(model, optim_cfg(a), steps)
    log.info("training on %d rows: %d epochs, %d steps (warmup %d)", len(data), a.epochs, steps, warm)
    history = []
    for ep in range(a.epochs):
        model.train(); t0, tot, n = time.time(), 0.0, 0
        for ids, att, typ, y, m, _ in batches(data, a.batch_size, True, rng, drop_last=True):
            ids, att, typ, y, m = (t.to(device) for t in (ids, att, typ, y, m))
            loss = loss_fn(model(ids, att, typ), y, m, (ids, att, typ))
            opt.zero_grad(set_to_none=True)
            for o in extra_opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for o in extra_opts:
                o.step()
            tot += loss.item(); n += 1
        history.append({"epoch": ep + 1, "loss": tot / max(n, 1), "seconds": time.time() - t0})
        log.info("epoch %d/%d loss %.4f (%.0fs)", ep + 1, a.epochs, tot / max(n, 1), time.time() - t0)
    return history


@torch.no_grad()
def logits_on(model, enc, device, bs=128):
    model.eval(); out = []
    for ids, att, typ, _, _, _ in batches(enc, bs, False):
        out.append(model(ids.to(device), att.to(device), typ.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stage1", "stage2", "joint"], required=True)
    ap.add_argument("--pair", choices=list(PAIRS), default="xedgo")
    ap.add_argument("--first", choices=DATASETS)
    ap.add_argument("--space", choices=["merge", "separate"])
    ap.add_argument("--method", choices=["ft", "lwf", "er", "lwf_all", "map"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="bert-base-cased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--bias-correction", action="store_true",
                    help="Adam bias correction (off in BERT's optimizer); RP4 amendment, Mosbach et al. 2021")
    ap.add_argument("--lwf-lamda", type=float, default=3.0)
    ap.add_argument("--er-per-class", type=int, default=20)
    ap.add_argument("--map-lr", type=float, default=1e-3, help="lr of the correspondence map (method map)")
    ap.add_argument("--save-train-logits", action="store_true", help="also write logits on train splits")
    ap.add_argument("--optimizer", choices=["fused", "reference"], default=None,
                    help="default: fused on GPU, reference on CPU")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: train rows per dataset")
    ap.add_argument("--eval-limit", type=int, default=0, help="smoke test: rows per eval split")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    if a.mode in ("stage1", "stage2") and not a.first:
        ap.error("--first is required")
    if a.mode in ("stage2", "joint") and not a.space:
        ap.error("--space is required")
    if a.mode == "stage2" and not a.method:
        ap.error("--method is required")
    if a.method == "map" and a.space != "separate":
        ap.error("--method map needs --space separate")
    if a.first and a.first not in PAIRS[a.pair]:
        ap.error(f"--first must be one of {PAIRS[a.pair]} for --pair {a.pair}")

    name = run_name(a); out = os.path.join(a.out, name); os.makedirs(out, exist_ok=True)
    log = logging.getLogger(name); log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s")
    for h in (logging.FileHandler(os.path.join(out, "train.log"), mode="w"), logging.StreamHandler()):
        h.setFormatter(fmt); log.addHandler(h)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.optimizer is None:
        # The fused kernel needs torch.compile, i.e. triton (absent on most Windows
        # installs). Both optimizers apply the same update rule (utils/fast_adam.py).
        has_triton = importlib.util.find_spec("triton") is not None
        a.optimizer = "fused" if device.type == "cuda" and has_triton else "reference"
    set_seed(a.seed)
    tok = BertTokenizerFast.from_pretrained(a.model, do_lower_case=False)
    t_start = time.time()

    def take(ds, split, limit):
        texts, y = load(ds, split)
        if limit and len(texts) > limit:
            idx = np.random.default_rng(1234).choice(len(texts), limit, replace=False)
            texts, y = [texts[i] for i in idx], y[idx]
        return texts, y

    space = LabelSpace(a.space, a.pair) if a.space else None
    meta = {"run": name, "args": vars(a), "device": str(device)}

    if a.mode == "stage1":
        cols, size = list(range(N_NATIVE[a.first])), N_NATIVE[a.first]
        model = EmoNet(a.model, size).to(device)
        texts, y = take(a.first, "train", a.limit)
        data = Encoded(texts, y, cols, size, tok)
        meta["history"] = fit(model, data, a, device, log, lambda z, y, m, x: masked_bce(z, y, m))
        torch.save({"model_state_dict": model.state_dict()}, os.path.join(out, "model.pt"))
        layout = {a.first: cols}
        meta["outputs"] = ([f"{a.first}:{i}" for i in cols])

    elif a.mode == "stage2":
        A, B = a.first, other_of(a.pair, a.first)
        ckpt = os.path.join(a.out, f"stage1_{A}_s{a.seed}", "model.pt")
        model = EmoNet(a.model, N_NATIVE[A]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
        teacher = model.frozen_copy() if a.method in ("lwf", "lwf_all", "map") else None
        model.expand_to(space.size, space.cols[A])
        texts, y = take(B, "train", a.limit)
        data = Encoded(texts, y, space.cols[B], space.size, tok)
        if a.method == "er":
            ta, ya = take(A, "train", a.limit)
            rng = np.random.default_rng(a.seed + 1000)
            keep = set()
            for j in range(ya.shape[1]):
                pos = np.flatnonzero(ya[:, j] > 0)
                keep.update(rng.choice(pos, min(a.er_per_class, len(pos)), replace=False).tolist())
            keep = sorted(keep)
            buf = Encoded([ta[i] for i in keep], ya[keep], space.cols[A], space.size, tok)
            meta["er_buffer_rows"] = len(keep)
            log.info("ER buffer: %d rows of %s", len(keep), A)
            data = Encoded.concat([data, buf])
        a_cols = torch.tensor(space.cols[A], device=device)
        b_cols = torch.tensor(space.cols[B], device=device)
        corr, extra = None, ()
        if a.method == "map":
            n_a, n_b = len(space.cols[A]), len(space.cols[B])
            corr = torch.nn.Linear(n_b, n_a).to(device)
            torch.nn.init.zeros_(corr.weight)
            probe = data.subset(list(range(min(2048, len(data)))))
            t_mean = torch.sigmoid(torch.from_numpy(logits_on(teacher, probe, device))).mean(0).clamp(1e-4, 1 - 1e-4)
            with torch.no_grad():
                corr.bias.copy_(torch.log(t_mean / (1 - t_mean)).to(device))
            extra = (torch.optim.Adam(corr.parameters(), lr=a.map_lr),)

        def loss_fn(z, y, m, x):
            loss = masked_bce(z, y, m)
            if a.method in ("lwf", "lwf_all"):
                with torch.no_grad():
                    t = torch.sigmoid(teacher(*x))
                target = torch.zeros_like(z); target[:, a_cols] = t
                dmask = torch.zeros_like(z); dmask[:, a_cols] = 1.0
                if a.method == "lwf":
                    dmask = dmask * (1.0 - m)      # only columns the row does not annotate
                loss = loss + a.lwf_lamda * masked_bce(z, target, dmask)
            elif a.method == "map":
                with torch.no_grad():
                    t = torch.sigmoid(teacher(*x))
                r = corr(z[:, b_cols])
                loss = loss + a.lwf_lamda * masked_bce(r, t, torch.ones_like(t))
            return loss

        meta["history"] = fit(model, data, a, device, log, loss_fn, extra)
        layout = space.cols
        meta["outputs"] = space.names

    else:  # joint
        model = EmoNet(a.model, space.size).to(device)
        parts = []
        for ds in PAIRS[a.pair]:
            texts, y = take(ds, "train", a.limit)
            parts.append(Encoded(texts, y, space.cols[ds], space.size, tok))
        meta["history"] = fit(model, Encoded.concat(parts), a, device, log,
                              lambda z, y, m, x: masked_bce(z, y, m))
        layout = space.cols
        meta["outputs"] = space.names

    arrays = {}
    splits = ("dev", "test") + (("train",) if a.save_train_logits else ())
    for ds in PAIRS[a.pair]:
        for split in splits:
            texts, y = take(ds, split, a.eval_limit if split != "train" else a.limit)
            enc = Encoded(texts, y, list(range(N_NATIVE[ds])), N_NATIVE[ds], tok)
            arrays[f"logits_{ds}_{split}"] = logits_on(model, enc, device)
            arrays[f"y_{ds}_{split}"] = y.astype(np.int8)
    if a.mode == "stage2" and a.method == "map":
        W = corr.weight.detach().cpu().numpy(); bvec = corr.bias.detach().cpu().numpy()
        bc = space.cols[B]
        for k in [k for k in arrays if k.startswith("logits_")]:
            arrays["readout_" + k[len("logits_"):]] = arrays[k][:, bc] @ W.T + bvec
        np.savez(os.path.join(out, "corr.npz"), weight=W, bias=bvec,
                 new_labels=np.array(labels_of(B)), old_labels=np.array(labels_of(A)))
    np.savez_compressed(os.path.join(out, "logits.npz"), **arrays)
    meta["layout"] = layout
    meta["seconds"] = time.time() - t_start
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log.info("done in %.0fs", meta["seconds"])


if __name__ == "__main__":
    main()
