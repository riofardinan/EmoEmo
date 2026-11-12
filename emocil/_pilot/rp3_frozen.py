"""RP3 pilot: is a frozen pretrained encoder enough for fine-grained multi-label
emotion continual learning? (research problem RP3 in ../research_framework.md;
design and decision rule in README.md, "RP3 pilot".)

A frozen encoder with a closed-form ridge head cannot forget: recursive ridge
equals joint ridge (L3A, ALMA). Its best case is therefore a ridge head fitted
on all 28 GoEmotions labels over frozen features. That best case is compared
with the fine-tuned continual-learning methods of ../emogo on the same test set
and the same metric (their per-class AP, averaged: emogo/utils/metrics.py).

Encoders (frozen when features are extracted):
  bert        bert-base-cased as pretrained (the emogo backbone), all 13 layers,
              mean-pooled, plus the [CLS] pooler output.
  bert_task0  bert-base-cased fine-tuned on task 0 of emogo's B0-I7 protocol
              (the 7 alphabetically first classes, rows carrying one of them,
              emogo/bertgo recipe) and then frozen: the RidgeFT / SEQ* recipe.
  qwen        Qwen/Qwen2.5-1.5B (base, Apache-2.0) as pretrained, mean-pooled
              hidden states of every 4th layer and the last. A frozen LLM is
              the counter-premise (Fang et al., 2026).

For every encoder the layer (or pooler) and the ridge strength are chosen on
the GoEmotions dev set by mAP; test is touched once per encoder.

    python -m _pilot.rp3_frozen finetune-task0          # bert_task0 encoder
    python -m _pilot.rp3_frozen extract --encoder bert  # also bert_task0, qwen
    python -m _pilot.rp3_frozen evaluate                # _pilot/results/rp3/summary.csv, verdict.json
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from utils.data import go_labels, load_go
from utils.features import bert_features, qwen_features  # noqa: F401  (used below and by frozen_vs_pilots)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("RP3_OUT", os.path.join(ROOT, "results", "rp3"))
# emogo, 8 seeds, last-task mAP (analysis/tables/emogrowth_metrics_mean.csv in the repo root)
EMOGO_LAST_MAP = {"B0-I7": {"lwf": 38.08, "agcn": 36.93, "aesl": 33.23, "prs": 32.3, "er": 30.26},
                  "B16-I2": {"lwf": 28.06, "er": 27.18, "prs": 27.02}}
EMOGO_LWF_SD = {"B0-I7": 0.92, "B16-I2": 1.39}
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def average_precision(scores, targets):
    """emogo/utils/metrics.py average_precision, in numpy; returns mean AP x 100."""
    aps = []
    for k in range(scores.shape[1]):
        order = np.argsort(-scores[:, k], kind="stable")
        t = targets[order, k]
        npos = t.sum()
        if npos == 0:
            aps.append(0.0); continue
        prec = np.cumsum(t) / np.arange(1, len(t) + 1)
        aps.append(float((prec * t).sum() / npos))
    return 100.0 * float(np.mean(aps)), aps


def macro_auroc(scores, targets):
    from sklearn.metrics import roc_auc_score
    ks = [k for k in range(targets.shape[1]) if 0 < targets[:, k].sum() < len(targets)]
    return float(np.mean([roc_auc_score(targets[:, k], scores[:, k]) for k in ks]))


def macro_f1_dev_thresholds(s_dev, y_dev, s_test, y_test):
    f1s = []
    for k in range(y_dev.shape[1]):
        cand = np.unique(np.quantile(s_dev[:, k], np.linspace(0, 1, 201)))
        def f1(s, y, t):
            p = s > t; tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = (~p & (y == 1)).sum()
            return 2 * tp / max(2 * tp + fp + fn, 1)
        t = max(cand, key=lambda c: f1(s_dev[:, k], y_dev[:, k], c))
        f1s.append(f1(s_test[:, k], y_test[:, k], t))
    return 100.0 * float(np.mean(f1s))


# ---------------------------------------------------------------- encoders

def task0_classes():
    return sorted(go_labels())[:7]          # emogo B0-I7, alphabetical class order


def finetune_task0(a):
    """Fine-tune bert-base-cased on B0-I7 task 0 and save it (emogo recipe)."""
    from transformers import BertTokenizerFast
    from nets.net import EmoNet
    from train import fit, masked_bce, set_seed
    from utils.data import Encoded
    go = go_labels(); cls = [go.index(c) for c in task0_classes()]
    texts, y = load_go("train")
    keep = np.flatnonzero(y[:, cls].sum(1) > 0)
    if a.limit:
        keep = keep[:a.limit]
    texts, y = [texts[i] for i in keep], y[keep][:, cls]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)
    tok = BertTokenizerFast.from_pretrained(a.model, do_lower_case=False)
    model = EmoNet(a.model, len(cls)).to(device)
    import logging
    log = logging.getLogger("task0"); log.setLevel(logging.INFO); log.addHandler(logging.StreamHandler())
    args = argparse.Namespace(seed=0, batch_size=16, epochs=a.epochs, lr=5e-5, warmup=0.1,
                              optimizer="fused" if device.type == "cuda" else "reference")
    data = Encoded(texts, y, list(range(len(cls))), len(cls), tok)
    hist = fit(model, data, args, device, log, lambda z, yy, m, x: masked_bce(z, yy, m))
    os.makedirs(OUT, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "classes": task0_classes(), "rows": len(keep),
                "history": hist}, os.path.join(OUT, "bert_task0.pt"))
    print(f"task-0 model: {len(keep)} rows, classes {task0_classes()}")




def extract(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    arrays = {}
    for split in ("train", "dev", "test"):
        texts, y = load_go(split)
        if a.limit:
            texts, y = texts[:a.limit], y[:a.limit]
        t0 = time.time()
        if a.encoder == "bert":
            f = bert_features(a.model, texts, device)
        elif a.encoder == "bert_task0":
            st = torch.load(os.path.join(OUT, "bert_task0.pt"), map_location="cpu")["model_state_dict"]
            f = bert_features(a.model, texts, device, state=st)
        else:
            f = qwen_features(a.qwen, texts, device)
        for k, v in f.items():
            arrays[f"{split}__{k}"] = v
        arrays[f"{split}__y"] = y.astype(np.int8)
        print(f"{a.encoder} {split}: {len(texts)} texts, {len(f)} feature sets, {time.time() - t0:.0f}s")
    np.savez(os.path.join(OUT, f"features_{a.encoder}.npz"), **arrays)


# ---------------------------------------------------------------- ridge + evaluation

def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    Zb = np.hstack([Z, np.ones((len(Z), 1))])
    A = Zb.T @ Zb + lam * np.eye(Zb.shape[1]); A[-1, -1] -= lam        # no penalty on the bias
    W = np.linalg.solve(A, Zb.T @ Y)
    return lambda X2: np.hstack([(X2 - mu) / sd, np.ones((len(X2), 1))]) @ W


def protocol_ridge(Xtr, Ytr, lam, order, sizes):
    """Recursive ridge over emogo's MLCIL protocol with absent labels set to 0 (no
    pseudo-labels): task t sees rows carrying a label of its classes, labelled
    on those classes only. Equal to one ridge fit on the stacked task data."""
    blocks_X, blocks_Y = [], []
    start = 0
    for n in sizes:
        cls = order[start:start + n]; start += n
        rows = np.flatnonzero(Ytr[:, cls].sum(1) > 0)
        Y = np.zeros((len(rows), Ytr.shape[1]), dtype=np.float32); Y[:, cls] = Ytr[rows][:, cls]
        blocks_X.append(Xtr[rows]); blocks_Y.append(Y)
    return ridge_fit(np.concatenate(blocks_X), np.concatenate(blocks_Y), lam)


def finetuned_joint_map():
    """Fine-tuned joint reference: RP1 pilot stage-1 GoEmotions models (full train)."""
    vals = []
    for s in (0, 1, 2):
        p = os.path.join(ROOT, "results", "rp1", f"stage1_go_s{s}", "logits.npz")
        if os.path.exists(p):
            z = np.load(p)
            vals.append(average_precision(z["logits_go_test"], z["y_go_test"].astype(np.float32))[0])
    return float(np.mean(vals)) if vals else None


def evaluate(a):
    rows, best = [], {}
    go = go_labels(); order = [go.index(c) for c in sorted(go)]
    for enc in ("bert", "bert_task0", "qwen"):
        p = os.path.join(OUT, f"features_{enc}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        ytr, ydv, yte = (z[f"{s}__y"].astype(np.float32) for s in ("train", "dev", "test"))
        sets = sorted({k.split("__")[1] for k in z.files if not k.endswith("__y")})
        cand = []
        for fs in sets:
            Xtr, Xdv = z[f"train__{fs}"].astype(np.float32), z[f"dev__{fs}"].astype(np.float32)
            for lam in LAMBDAS:
                cand.append((average_precision(ridge_fit(Xtr, ytr, lam)(Xdv), ydv)[0], fs, lam))
        dev_map, fs, lam = max(cand)
        Xtr, Xdv, Xte = (z[f"{s}__{fs}"].astype(np.float32) for s in ("train", "dev", "test"))
        pred = ridge_fit(Xtr, ytr, lam)
        s_dv, s_te = pred(Xdv), pred(Xte)
        r = {"encoder": enc, "features": fs, "lambda": lam, "dev_map": round(dev_map, 2),
             "test_map_joint": round(average_precision(s_te, yte)[0], 2),
             "test_macro_auroc": round(macro_auroc(s_te, yte), 4),
             "test_macro_f1_devthr": round(macro_f1_dev_thresholds(s_dv, ydv, s_te, yte), 2)}
        for proto, sizes in (("B0-I7", [7, 7, 7, 7]), ("B16-I2", [16] + [2] * 6)):
            r[f"test_map_protocol_{proto}"] = round(average_precision(
                protocol_ridge(Xtr, ytr, lam, order, sizes)(Xte), yte)[0], 2)
        rows.append(r); best[enc] = r
    if not rows:
        print("no features yet"); return
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    M, sd = EMOGO_LAST_MAP["B0-I7"]["lwf"], EMOGO_LWF_SD["B0-I7"]
    reach = {e: best[e]["test_map_joint"] >= M - 1.0 for e in best}
    bert_side = any(reach.get(e, False) for e in ("bert", "bert_task0"))
    decision = ("collapsed" if bert_side else
                "encoder-dependent" if reach.get("qwen", False) else
                "proven" if all(e in best for e in ("bert", "bert_task0", "qwen")) else "incomplete")
    verdict = {"lwf_B0-I7_last_map": M, "threshold": round(M - 1.0, 2),
               "frozen_best_joint_map": {e: best[e]["test_map_joint"] for e in best},
               "finetuned_joint_map": finetuned_joint_map(), "decision": decision}
    with open(os.path.join(OUT, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    for r in rows:
        print(r)
    print(json.dumps(verdict, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("finetune-task0"); f.add_argument("--model", default="bert-base-cased")
    f.add_argument("--epochs", type=int, default=4); f.add_argument("--limit", type=int, default=0)
    e = sub.add_parser("extract"); e.add_argument("--encoder", choices=["bert", "bert_task0", "qwen"], required=True)
    e.add_argument("--model", default="bert-base-cased"); e.add_argument("--qwen", default="Qwen/Qwen2.5-1.5B")
    e.add_argument("--limit", type=int, default=0)
    sub.add_parser("evaluate")
    a = ap.parse_args()
    {"finetune-task0": finetune_task0, "extract": extract, "evaluate": evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
