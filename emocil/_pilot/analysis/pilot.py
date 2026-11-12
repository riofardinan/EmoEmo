"""Analysis and decision rule of the RP1 pilot (fixed before any full run;
see ../README.md, "Training pilot").

For every run and every dataset D on its head, D's test labels are read from
the outputs D maps to (meta.json "layout"), with no knowledge of a text's
source. Metrics, macro-averaged over the six shared emotions (anger, disgust,
fear, joy, sadness, surprise) and over all of D's labels:
  auroc          ranking, threshold-free (primary).
  f1_pooled      one threshold per output, tuned on the pooled dev rows of
                 every dataset that annotates the output: the setting where
                 the source is unknown.
  f1_known       one threshold per (output, dataset), tuned on D's dev only:
                 leaks the source; the gap to f1_pooled is the cost of not
                 knowing it.
  f1_fixed       logit > 0.

Amplification, per order (A = first dataset), method m and seed s, on A's test:
  amp = [sep - merge]_stage2(m, s) - [sep - merge]_joint(s)
The bracket is the cost of merging same-named labels; the joint term is the
static cost, so amp is the extra cost that sequential learning adds.

Decision (per metric, "holds" for (order, m) when amp > 0 in every seed and
its mean >= 0.01):
  confirmed  on auroc, holds for >= 2 of 3 methods in at least one order.
  partial    not confirmed, and in some order >= 2 methods hold on f1_pooled
             but not on f1_known: a decision-layer problem of unknown source.
  collapsed  otherwise.

    python -m _pilot.analysis.pilot --runs _pilot/results/rp1
    python -m _pilot.analysis.pilot --runs _pilot/results/rp2 --pair ekfine
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata

from utils.data import SHARED_EMOTIONS, labels_of, shared_emotions

NATIVE = {d: labels_of(d) for d in ("go", "xed", "fine", "ekman", "b_eng", "b_hin", "b_sun")}
ORDERS_BY_PAIR = {"xedgo": {"xed2go": "xed", "go2xed": "go"},
                  "ekfine": {"ekman2fine": "ekman", "fine2ekman": "fine"},
                  "enghin": {"b_eng2b_hin": "b_eng", "b_hin2b_eng": "b_hin"},
                  "engsun": {"b_eng2b_sun": "b_eng", "b_sun2b_eng": "b_sun"}}
ORDERS = ORDERS_BY_PAIR["xedgo"]
SHARED = SHARED_EMOTIONS          # set per pair in main(); all six for xedgo and ekfine
METHODS = ["ft", "lwf", "er"]


def auroc(s, y):
    y = y.astype(bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def f1(s, y, t):
    p = s > t; y = y.astype(bool)
    tp = (p & y).sum(); fp = (p & ~y).sum(); fn = (~p & y).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def best_threshold(s, y):
    cand = np.unique(np.quantile(s, np.linspace(0, 1, 201)))
    return max(cand, key=lambda t: f1(s, y, t))


def load_runs(root):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*", "logits.npz"))):
        name = os.path.basename(os.path.dirname(d))
        with open(os.path.join(os.path.dirname(d), "meta.json")) as f:
            meta = json.load(f)
        runs[name] = (dict(np.load(d)), meta["layout"])
    return runs


def evaluate(arr, layout, D):
    cols = layout[D]
    out = {}
    per = {"auroc": [], "f1_pooled": [], "f1_known": [], "f1_fixed": []}
    for j, lab in enumerate(NATIVE[D]):
        c = cols[j]
        st, yt = arr[f"logits_{D}_test"][:, c], arr[f"y_{D}_test"][:, j]
        sd, yd = arr[f"logits_{D}_dev"][:, c], arr[f"y_{D}_dev"][:, j]
        ps, py = [], []
        for E, ec in layout.items():
            if c in ec:
                ps.append(arr[f"logits_{E}_dev"][:, c]); py.append(arr[f"y_{E}_dev"][:, ec.index(c)])
        per["auroc"].append(auroc(st, yt))
        per["f1_pooled"].append(f1(st, yt, best_threshold(np.concatenate(ps), np.concatenate(py))))
        per["f1_known"].append(f1(st, yt, best_threshold(sd, yd)))
        per["f1_fixed"].append(f1(st, yt, 0.0))
    shared = [NATIVE[D].index(e) for e in SHARED]
    for k, v in per.items():
        v = np.array(v, dtype=float)
        out[f"{k}_shared"] = float(np.nanmean(v[shared])); out[f"{k}_all"] = float(np.nanmean(v))
    return out


def holds(values):
    return len(values) > 0 and all(v > 0 for v in values) and float(np.mean(values)) >= 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results", "rp1"))
    ap.add_argument("--pair", choices=list(ORDERS_BY_PAIR), default="xedgo",
                    help="xedgo (RP1, default), ekfine (RP2), enghin / engsun (RP4)")
    a = ap.parse_args()
    global ORDERS, SHARED
    ORDERS = ORDERS_BY_PAIR[a.pair]
    SHARED = shared_emotions(a.pair)
    runs = load_runs(a.runs)
    res = {}
    rows = []
    for name, (arr, layout) in runs.items():
        for D in layout:
            m = evaluate(arr, layout, D)
            res[(name, D)] = m
            rows.append({"run": name, "dataset": D, **{k: round(v, 4) for k, v in m.items()}})
    outdir = os.path.join(a.runs, "analysis"); os.makedirs(outdir, exist_ok=True)
    if rows:
        with open(os.path.join(outdir, "runs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    seeds = sorted({int(n.rsplit("_s", 1)[1]) for n in runs})
    amp_rows, rule, used = [], {}, set()
    for metric in ("auroc_shared", "f1_pooled_shared", "f1_known_shared"):
        for order, A in ORDERS.items():
            for mth in METHODS:
                vals = []
                for s in seeds:
                    keys = [(f"stage2_{order}_separate_{mth}_s{s}", A), (f"stage2_{order}_merge_{mth}_s{s}", A),
                            (f"joint_separate_s{s}", A), (f"joint_merge_s{s}", A)]
                    if not all(k in res for k in keys):
                        continue
                    sS, sM, jS, jM = (res[k][metric] for k in keys)
                    amp = (sS - sM) - (jS - jM)
                    vals.append(amp); used.add(s)
                    amp_rows.append({"metric": metric, "order": order, "method": mth, "seed": s,
                                     "seq_sep": round(sS, 4), "seq_merge": round(sM, 4),
                                     "joint_sep": round(jS, 4), "joint_merge": round(jM, 4),
                                     "amp": round(amp, 4)})
                rule[(metric, order, mth)] = (holds(vals), vals)
    if amp_rows:
        with open(os.path.join(outdir, "amplification.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(amp_rows[0])); w.writeheader(); w.writerows(amp_rows)

    verdict = {"seeds_complete": sorted(used), "per_order": {}}   # seeds with every run of the comparison
    confirmed = partial = False
    for order in ORDERS:
        on_auc = [m for m in METHODS if rule.get(("auroc_shared", order, m), (False,))[0]]
        on_pool = [m for m in METHODS if rule.get(("f1_pooled_shared", order, m), (False,))[0]]
        on_known = [m for m in METHODS if rule.get(("f1_known_shared", order, m), (False,))[0]]
        pool_only = [m for m in on_pool if m not in on_known]
        confirmed |= len(on_auc) >= 2
        partial |= len(pool_only) >= 2
        verdict["per_order"][order] = {
            "methods_amplified_auroc": on_auc, "methods_amplified_f1_pooled": on_pool,
            "methods_amplified_f1_known": on_known,
            "mean_amp_auroc": {m: round(float(np.mean(rule[("auroc_shared", order, m)][1])), 4)
                               for m in METHODS if rule.get(("auroc_shared", order, m), (0, []))[1]}}
    verdict["decision"] = ("confirmed" if confirmed else "partial" if partial else "collapsed")
    with open(os.path.join(outdir, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
