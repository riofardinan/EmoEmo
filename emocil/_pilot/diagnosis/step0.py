"""RP1 step 0: are same-named emotion labels equivalent across datasets?

Inference only, no training. The GoEmotions BERT from ../bertgo (run1, test
macro-F1 .491 at threshold 0.3) scores every English XED sentence, and the
scores are compared with XED's own gold labels (8 Plutchik emotions + neutral).
Research question RP1 is in ../../research_framework.md.

What is measured, per XED label j:
  auroc_same  AUROC of the same-named GoEmotions output (6 emotions + neutral).
  auroc_best  best single GoEmotions output, and which one.
  auroc_map   5-fold cross-validated logistic regression from all 28
              GoEmotions logits to XED label j (out-of-fold scores).
  d_cross     auroc_map - auroc_same on XED, with a paired bootstrap 95% CI.
  d_in        the same quantity inside GoEmotions (test split, gold labels),
              where a label is equivalent to itself by construction. This is
              the gain that combining 28 correlated outputs gives with no
              change of meaning, so only d_cross - d_in counts as evidence.
  ratio_0.3   predicted-positive rate at the native threshold 0.3 divided by
              XED prevalence (operating-point mismatch, cf. Fear != Fear).
  f1_0.3 / f1_tuned  F1 of the same-named output at 0.3 and at a threshold
              tuned on a random half and scored on the other half (20 splits).

Decision criteria, fixed before the first run (2 Nov 2026; the script then
lived at analysis/rp1_step0.py and was moved here with only its paths changed). Only the six
shared emotions count (anger, disgust, fear, joy, sadness, surprise):
  C1 meaning: for >= 2 of them, d_cross CI lower bound > 0 and
     d_cross - d_in >= 0.03.
  C2 inferable correspondence: auroc_map >= 0.70 for both anticipation and
     trust, which have no same-named GoEmotions output.
  C3 operating point: ratio_0.3 outside [0.5, 2] for >= 3 of them.
  Decision: C1 true -> continue to the training pilot. C1 false and C3 true ->
  RP1 narrows to the decision layer; reconsider before the pilot. C1 and C3
  false -> the premise does not hold for this pair; stop or test another pair.
  C2 informs the solution idea (correspondence read off old-model
  predictions, as suggested by CLEO), not the go/no-go.

Inputs (all under ../data except the checkpoint):
  xed/en-annotated.tsv, xed/neu_en.txt   https://github.com/Helsinki-NLP/XED
      (AnnotatedData/, CC-BY 4.0). Texts in both files are dropped; duplicate
      neutral lines are kept once.
  goemotions/test.tsv, emotions.txt      copied from ../../bertgo/data.
  --ckpt   the bertgo checkpoint, read as a file (default
           ../../bertgo/output/run1/model.pt). Both datasets are scored by the
           same function, so the in-domain control and XED share one code path.

    cd emocil && ../.venv/bin/python -m _pilot.diagnosis.step0
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # emocil/_pilot/
EMOCIL = os.path.dirname(ROOT)
DATA = os.path.join(EMOCIL, "data")
OUT = os.path.join(ROOT, "results", "step0")
XED_LABELS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness",
              "surprise", "trust", "neutral"]
SHARED = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def load_xed(xed_dir):
    rows = {}
    with open(os.path.join(xed_dir, "en-annotated.tsv"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            text, labs = line.split("\t")
            rows[text] = {int(x) for x in labs.split(",") if x.strip()}
    neutral = set()
    with open(os.path.join(xed_dir, "neu_en.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                neutral.add(line.split("\t", 1)[1])
    clash = neutral & set(rows)
    texts, y = [], []
    for text, labs in rows.items():
        if text in clash:
            continue
        v = np.zeros(len(XED_LABELS), dtype=np.int8)
        for l in labs:                       # 1..8 = anger..trust
            v[l - 1] = 1
        texts.append(text); y.append(v)
    for text in sorted(neutral - clash):
        v = np.zeros(len(XED_LABELS), dtype=np.int8); v[8] = 1
        texts.append(text); y.append(v)
    return texts, np.stack(y), len(clash)


def score_xed(texts, ckpt, batch_size):
    import torch
    from transformers import BertTokenizerFast
    sys.path.insert(0, EMOCIL)
    from nets.bert import BertForMultiLabelEmotion
    torch.set_num_threads(os.cpu_count() or 1)
    tok = BertTokenizerFast.from_pretrained("bert-base-cased", do_lower_case=False)
    model = BertForMultiLabelEmotion("bert-base-cased", num_labels=28)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    order = np.argsort([len(t) for t in texts])          # less padding
    logits = np.zeros((len(texts), 28), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                      max_length=50, return_tensors="pt")
            out, _ = model(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
            logits[idx] = out.numpy()
    return logits


def auroc(score, y):
    """Rank-based AUROC (ties averaged); y in {0,1}."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(score)                  # ties get their average rank
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def oof_map(X, y, seed=0):
    """Out-of-fold scores of a logistic map from all logits to one label."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    return oof


def boot_diff(a, b, y, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(auroc(a[i], y[i]) - auroc(b[i], y[i]))
    return np.percentile(d, [2.5, 97.5])


def f1_at(score, y, thr):
    p = score >= thr
    tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = (~p & (y == 1)).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def f1_tuned(score, y, splits=20, seed=0):
    rng = np.random.default_rng(seed); cand = np.unique(np.quantile(score, np.linspace(0, 1, 201)))
    out = []
    for _ in range(splits):
        m = rng.random(len(y)) < 0.5
        best = max(cand, key=lambda t: f1_at(score[m], y[m], t))
        out.append(f1_at(score[~m], y[~m], best))
    return float(np.mean(out))


def load_goemotions_test():
    """Read test.tsv exactly as ../../bertgo/data.py does. The file uses CSV
    quoting (a text with quotes is wrapped in "..." and inner quotes doubled),
    so it must be parsed with quote handling, not split on tabs."""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA, "goemotions", "test.tsv"), sep="\t", encoding="utf-8",
                     header=None, names=["text", "labels", "id"], dtype={"text": str})
    texts = [t if isinstance(t, str) else "" for t in df["text"].tolist()]
    y = np.zeros((len(texts), 28), dtype=np.int8)
    for i, raw in enumerate(df["labels"].tolist()):
        for l in str(raw).split(","):
            y[i, int(l)] = 1
    return texts, y


def cached_scores(path, texts, ckpt, batch_size):
    if os.path.exists(path):
        c = np.load(path, allow_pickle=True)
        assert list(c["texts"]) == texts, f"{path} does not match the data files"
        return c["logits"]
    L = score_xed(texts, ckpt, batch_size)
    np.savez_compressed(path, logits=L, texts=np.array(texts, dtype=object))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xed-dir", default=os.path.join(DATA, "xed"))
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(EMOCIL), "bertgo", "output", "run1", "model.pt"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()

    go = [l.strip() for l in open(os.path.join(DATA, "goemotions", "emotions.txt")) if l.strip()]
    assert len(go) == 28
    os.makedirs(OUT, exist_ok=True)
    texts, Y, n_clash = load_xed(a.xed_dir)
    L = cached_scores(os.path.join(OUT, "xed_logits.npz"), texts, a.ckpt, a.batch_size)
    g_texts, Yg = load_goemotions_test()
    Lg = cached_scores(os.path.join(OUT, "goemotions_test_logits.npz"), g_texts, a.ckpt, a.batch_size)
    print(f"XED-EN: {len(texts)} texts ({n_clash} dropped as both emotional and neutral)")
    print("prevalence:", {l: round(float(Y[:, j].mean()), 3) for j, l in enumerate(XED_LABELS)})

    with open(os.path.join(OUT, "auroc_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["goemotions_output"] + XED_LABELS)
        for k, name in enumerate(go):
            w.writerow([name] + [f"{auroc(L[:, k], Y[:, j]):.4f}" for j in range(len(XED_LABELS))])

    rows, weights = [], {}
    for j, lab in enumerate(XED_LABELS):
        y = Y[:, j]
        per = np.array([auroc(L[:, k], y) for k in range(28)])
        kb = int(np.nanargmax(per))
        oof = oof_map(L, y)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        Z = StandardScaler().fit_transform(L)
        coef = LogisticRegression(max_iter=2000).fit(Z, y).coef_[0]
        top = np.argsort(-coef)[:5]
        weights[lab] = [(go[k], round(float(coef[k]), 3)) for k in top]
        r = {"xed_label": lab, "prevalence": round(float(y.mean()), 4),
             "best_output": go[kb], "auroc_best": round(per[kb], 4),
             "auroc_map": round(auroc(oof, y), 4),
             "top_map_weights": "; ".join(f"{n} {c:+.2f}" for n, c in weights[lab])}
        if lab in go:
            k = go.index(lab)
            s = L[:, k]
            r["auroc_same"] = round(per[k], 4)
            r["d_cross"] = round(r["auroc_map"] - per[k], 4)
            lo, hi = boot_diff(oof, s, y, n=a.boot)
            r["d_cross_ci"] = f"[{lo:+.4f}, {hi:+.4f}]"
            yg = Yg[:, k]
            r["d_in"] = round(auroc(oof_map(Lg, yg), yg) - auroc(Lg[:, k], yg), 4)
            p = 1 / (1 + np.exp(-s))
            r["ratio_0.3"] = round(float((p >= 0.3).mean() / max(y.mean(), 1e-9)), 3)
            r["f1_0.3"] = round(f1_at(p, y, 0.3), 4)
            r["f1_tuned"] = round(f1_tuned(s, y), 4)
            r["f1_tuned_map"] = round(f1_tuned(oof, y), 4)
            r["_ci_lo"] = lo
        rows.append(r)

    cols = ["xed_label", "prevalence", "auroc_same", "auroc_best", "best_output", "auroc_map",
            "d_cross", "d_cross_ci", "d_in", "ratio_0.3", "f1_0.3", "f1_tuned", "f1_tuned_map",
            "top_map_weights"]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    for r in rows:
        print({c: r.get(c, "") for c in cols})

    shared = [r for r in rows if r["xed_label"] in SHARED]
    c1_labels = [r["xed_label"] for r in shared
                 if r["_ci_lo"] > 0 and r["d_cross"] - r["d_in"] >= 0.03]
    c2 = all(r["auroc_map"] >= 0.70 for r in rows if r["xed_label"] in ("anticipation", "trust"))
    c3_labels = [r["xed_label"] for r in shared if not 0.5 <= r["ratio_0.3"] <= 2]
    c1, c3 = len(c1_labels) >= 2, len(c3_labels) >= 3
    decision = ("continue to the training pilot" if c1 else
                "narrow to the decision layer; reconsider before the pilot" if c3 else
                "premise does not hold for this pair; stop or test another pair")
    verdict = {"C1": [c1, c1_labels], "C2": c2, "C3": [c3, c3_labels], "decision": decision}
    print(json.dumps(verdict, indent=1))
    with open(os.path.join(OUT, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)


if __name__ == "__main__":
    main()
