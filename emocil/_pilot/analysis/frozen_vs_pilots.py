"""Exploratory, post-hoc (4 Nov 2026; not pre-registered): the frozen-encoder
paradigm in the RP1 and RP2 settings.

The RP3 pilot found that a frozen encoder with a closed-form ridge head matches
or beats the fine-tuned CL methods on GoEmotions. A ridge head is joint-
equivalent, so with it the sequential amplification found by the RP1 and RP2
pilots cannot occur. This script asks what is left of RP1 and RP2 in practice:
how a frozen encoder with ridge heads (merged or separate label spaces) scores
on each dataset, against the pilots' fine-tuned sequential results.

Encoders: pretrained bert-base-cased (layer 1, lambda 1000) and Qwen2.5-1.5B
(layer 12, lambda 10), the configurations the RP3 pilot selected on GoEmotions
dev. A merged output is fitted on the rows of every dataset that annotates its
name, i.e. masked ridge, solved per output.

Needs results/rp3/features_{bert,qwen}.npz (GoEmotions) and the XED features
this script extracts into results/rp3/features_xed_{bert,qwen}.npz.

    python -m _pilot.analysis.frozen_vs_pilots
"""

import csv
import os

import numpy as np
import torch

from _pilot.analysis.pilot import NATIVE, auroc
from _pilot.rp3_frozen import OUT, bert_features, qwen_features
from utils.data import SHARED_EMOTIONS, go_half, load_go, load_xed, to_ekman

CONF = {"bert": ("L1", 1000.0), "qwen": ("L12", 10.0)}


def xed_features(enc):
    p = os.path.join(OUT, f"features_xed_{enc}.npz")
    if not os.path.exists(p):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arrays = {}
        for split in ("train", "dev", "test"):
            texts, y = load_xed(split)
            f = bert_features("bert-base-cased", texts, device) if enc == "bert" else \
                qwen_features("Qwen/Qwen2.5-1.5B", texts, device)
            arrays[f"{split}__{CONF[enc][0]}"] = f[CONF[enc][0]].astype(np.float32)
            arrays[f"{split}__y"] = y.astype(np.int8)
        np.savez(p, **arrays)
    return np.load(p)


def masked_ridge(X_list, y_list, lam, X_all):
    """One ridge output fitted on the concatenation of the given row blocks;
    standardisation uses all training rows the head can see."""
    mu, sd = X_all.mean(0), X_all.std(0) + 1e-6
    Z = np.vstack([(X - mu) / sd for X in X_list]); Z = np.hstack([Z, np.ones((len(Z), 1))])
    y = np.concatenate(y_list)
    A = Z.T @ Z + lam * np.eye(Z.shape[1]); A[-1, -1] -= lam
    w = np.linalg.solve(A, Z.T @ y)
    return lambda X: np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ w


def score(datasets, space, lam):
    """datasets: {name: (Xtr, Ytr, Xte, Yte, labels)}. Returns mean shared-emotion
    AUROC per dataset for a merged or separate label space."""
    X_all = np.vstack([d[0] for d in datasets.values()])
    out = {}
    for name, (Xtr, Ytr, Xte, Yte, labs) in datasets.items():
        vals = []
        for e in SHARED_EMOTIONS:
            j = labs.index(e)
            if space == "separate":
                f = masked_ridge([Xtr], [Ytr[:, j]], lam, X_all)
            else:
                Xs, ys = [], []
                for o, (Xo, Yo, _, _, lo) in datasets.items():
                    if e in lo:
                        Xs.append(Xo); ys.append(Yo[:, lo.index(e)])
                f = masked_ridge(Xs, ys, lam, X_all)
            vals.append(auroc(f(Xte), Yte[:, j]))
        out[name] = float(np.mean(vals))
    return out


def main():
    rows = []
    for enc, (fs, lam) in CONF.items():
        g = np.load(os.path.join(OUT, f"features_{enc}.npz"))
        Xg_tr, Xg_te = g[f"train__{fs}"].astype(np.float32), g[f"test__{fs}"].astype(np.float32)
        yg_tr, yg_te = g["train__y"].astype(np.float32), g["test__y"].astype(np.float32)
        # RP2: GoEmotions halves, Ekman vs fine (half 0 = ekman, half 1 = fine)
        texts, _ = load_go("train")
        half = np.array([go_half(t) for t in texts])
        ek = (Xg_tr[half == 0], to_ekman(yg_tr[half == 0]), Xg_te, to_ekman(yg_te), NATIVE["ekman"])
        fi = (Xg_tr[half == 1], yg_tr[half == 1], Xg_te, yg_te, NATIVE["fine"])
        for sp in ("separate", "merge"):
            r = score({"ekman": ek, "fine": fi}, sp, lam)
            rows.append({"setting": "RP2 ekman/fine", "encoder": enc, "space": sp,
                         "old_or_first": round(r["ekman"], 4), "other": round(r["fine"], 4)})
        # RP1: XED and GoEmotions
        x = xed_features(enc)
        xd = (x[f"train__{fs}"].astype(np.float32), x["train__y"].astype(np.float32),
              x[f"test__{fs}"].astype(np.float32), x["test__y"].astype(np.float32), NATIVE["xed"])
        gd = (Xg_tr, yg_tr, Xg_te, yg_te, NATIVE["go"])
        for sp in ("separate", "merge"):
            r = score({"xed": xd, "go": gd}, sp, lam)
            rows.append({"setting": "RP1 xed/go", "encoder": enc, "space": sp,
                         "old_or_first": round(r["xed"], 4), "other": round(r["go"], 4)})
    out = os.path.join(OUT, "frozen_vs_pilots.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(open(out).read())


if __name__ == "__main__":
    main()
