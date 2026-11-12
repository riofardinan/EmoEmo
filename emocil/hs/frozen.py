"""Frozen-encoder baselines for the HS experiments, in the run format of train.py.

The RP3 pilot found a frozen encoder with ridge heads to be a strong paradigm.
In the RP1 and RP2 settings it was below fine-tuned LwF with separate heads,
but close on XED (_pilot/README.md). Here it is a baseline with the same
outputs as a training run:
- a separate label layout, one ridge head per dataset, fitted on that
  dataset's training rows;
- logits for every dataset and split.

Ridge is joint-equivalent, so the result depends on neither the order nor the
seed. One folder per encoder serves every seed and both orders.

Configurations are the ones RP3 chose on GoEmotions dev mAP, not re-tuned here:
  bert   bert-base-cased, mean-pooled layer 1, lambda 1000
  qwen   Qwen/Qwen2.5-1.5B, mean-pooled layer 12, lambda 10
Features come from utils/features.py, the code RP3 used (same pooling and truncation).
Standardisation uses all training rows of the pair, as in the RP3
exploration (_pilot/analysis/frozen_vs_pilots.py).

    cd emocil
    python -m hs.frozen --pair ekfine                  # results/hs_ekfine/frozen_{bert,qwen}/
    python -m hs.frozen --pair ekfine --encoders bert --bert-model prajjwal1/bert-tiny \\
        --limit 300 --eval-limit 200 --out /tmp/hs_smoke   # smoke test
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from utils.features import bert_features, qwen_features
from utils.data import PAIRS, LabelSpace, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # emocil/
CONF = {"bert": ("L1", 1000.0), "qwen": ("L12", 10.0)}


def take(ds, split, limit):
    """As train.py's take(): the same subset under --limit / --eval-limit."""
    texts, y = load(ds, split)
    if limit and len(texts) > limit:
        idx = np.random.default_rng(1234).choice(len(texts), limit, replace=False)
        texts, y = [texts[i] for i in idx], y[idx]
    return texts, y


def qwen_dtype(device):
    """bfloat16 where the GPU has it natively (compute capability >= 8.0, Ampere and
    later), as in RP3. Older GPUs such as the T4 (7.5) get float32: float16 is not
    an option, because Qwen's activations overflowed it in RP3."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float32


def ridge_heads(X, Y, lam, mu, sd):
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1), dtype=np.float32)])
    A = Z.T @ Z + lam * np.eye(Z.shape[1], dtype=np.float64); A[-1, -1] -= lam   # no penalty on the bias
    W = np.linalg.solve(A, Z.T @ Y)
    return lambda X2: np.hstack([(X2 - mu) / sd, np.ones((len(X2), 1), dtype=np.float32)]) @ W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=["ekfine", "xedgo"], default="ekfine")
    ap.add_argument("--encoders", default="bert,qwen")
    ap.add_argument("--bert-model", default="bert-base-cased")
    ap.add_argument("--qwen-model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="default results/hs_<pair>")
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "results", f"hs_{a.pair}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    space = LabelSpace("separate", a.pair)
    for enc in a.encoders.split(","):
        run = os.path.join(out, f"frozen_{enc}")
        if os.path.exists(os.path.join(run, "logits.npz")):
            print(f"frozen_{enc}: exists, skipped"); continue
        os.makedirs(run, exist_ok=True)
        layer, lam = CONF[enc]
        t0, cache, data = time.time(), {}, {}
        for ds in PAIRS[a.pair]:
            for split in ("train", "dev", "test"):
                texts, y = take(ds, split, a.limit if split == "train" else a.eval_limit)
                key = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
                if key not in cache:          # Ekman and fine share GoEmotions dev/test texts
                    f = bert_features(a.bert_model, texts, device) if enc == "bert" else \
                        qwen_features(a.qwen_model, texts, device, dtype=qwen_dtype(device))
                    cache[key] = f[layer].astype(np.float32)
                data[(ds, split)] = (cache[key], y)
                print(f"frozen_{enc} {ds} {split}: {len(texts)} texts", flush=True)
        X_all = np.vstack([data[(ds, "train")][0] for ds in PAIRS[a.pair]])
        mu, sd = X_all.mean(0), X_all.std(0) + 1e-6
        heads = {ds: ridge_heads(*data[(ds, "train")], lam, mu, sd) for ds in PAIRS[a.pair]}
        arrays = {}
        for (ds, split), (X, y) in data.items():
            L = np.zeros((len(X), space.size), dtype=np.float32)
            for hd, f in heads.items():
                L[:, space.cols[hd]] = f(X)
            arrays[f"logits_{ds}_{split}"] = L
            arrays[f"y_{ds}_{split}"] = y.astype(np.int8)
        np.savez_compressed(os.path.join(run, "logits.npz"), **arrays)
        meta = {"run": f"frozen_{enc}", "args": vars(a), "encoder": enc, "layer": layer, "lambda": lam,
                "model": a.bert_model if enc == "bert" else a.qwen_model,
                "dtype": "float32" if enc == "bert" else str(qwen_dtype(device)).replace("torch.", ""),
                "layout": space.cols, "outputs": space.names, "seconds": time.time() - t0}
        with open(os.path.join(run, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"frozen_{enc}: done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
