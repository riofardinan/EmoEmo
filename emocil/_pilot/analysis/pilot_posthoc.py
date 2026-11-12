"""Post-hoc checks on the RP1 pilot (not part of the pre-registered rule; added
2 Nov 2026 after the verdict). Uses the same run outputs as analysis/pilot.py.

1. shared vs old-only labels. In the merged space LwF distils fewer of the old
   dataset's columns (xed2go: 2 instead of 9), so a merge cost could come from
   a less constrained encoder rather than from overwritten labels. If so, the
   old dataset's own labels (anticipation and trust for XED; the 21 GoEmotions
   labels XED lacks) would drop with merging as well.
2. the new dataset. Whether merging buys anything on the dataset being learned.
3. the static merge cost of joint training on each dataset.

Writes results/rp1/analysis/posthoc_old_labels.csv, posthoc_new_dataset.csv,
posthoc_static_cost.csv.

    python -m _pilot.analysis.pilot_posthoc
"""

import csv
import os

import numpy as np

from _pilot.analysis.pilot import NATIVE, auroc, evaluate, load_runs
from utils.data import SHARED_EMOTIONS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDERS = {"xed2go": ("xed", "go"), "go2xed": ("go", "xed")}
SEEDS = (0, 1, 2)


def per_label(run, D):
    arr, lay = run
    cols = lay[D]
    return {l: auroc(arr[f"logits_{D}_test"][:, cols[j]], arr[f"y_{D}_test"][:, j])
            for j, l in enumerate(NATIVE[D])}


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    runs = load_runs(os.path.join(ROOT, "results", "rp1"))
    out = os.path.join(ROOT, "results", "rp1", "analysis")

    rows = []
    for order, (A, _) in ORDERS.items():
        only = [l for l in NATIVE[A] if l not in SHARED_EMOTIONS and l != "neutral"]
        names = [("native", lambda s: f"stage1_{A}_s{s}")]
        for m in ("ft", "lwf", "er"):
            for sp in ("separate", "merge"):
                names.append((f"{m}_{sp}", lambda s, m=m, sp=sp: f"stage2_{order}_{sp}_{m}_s{s}"))
        for sp in ("separate", "merge"):
            names.append((f"joint_{sp}", lambda s, sp=sp: f"joint_{sp}_s{s}"))
        for label, fn in names:
            d = [per_label(runs[fn(s)], A) for s in SEEDS]
            rows.append({"order": order, "old_dataset": A, "run": label,
                         "auroc_shared": round(float(np.mean([np.mean([x[l] for l in SHARED_EMOTIONS]) for x in d])), 4),
                         "auroc_old_only": round(float(np.mean([np.mean([x[l] for l in only]) for x in d])), 4),
                         "n_old_only": len(only)})
    write(os.path.join(out, "posthoc_old_labels.csv"), rows)

    rows = []
    for order, (_, B) in ORDERS.items():
        names = [("native", lambda s: f"stage1_{B}_s{s}")]
        for m in ("ft", "lwf", "er"):
            for sp in ("separate", "merge"):
                names.append((f"{m}_{sp}", lambda s, m=m, sp=sp: f"stage2_{order}_{sp}_{m}_s{s}"))
        for label, fn in names:
            e = [evaluate(*runs[fn(s)], B) for s in SEEDS]
            rows.append({"order": order, "new_dataset": B, "run": label,
                         **{k: round(float(np.mean([x[k] for x in e])), 4)
                            for k in ("auroc_shared", "auroc_all", "f1_pooled_shared", "f1_known_shared")}})
    write(os.path.join(out, "posthoc_new_dataset.csv"), rows)

    rows = []
    for D in ("xed", "go"):
        for s in SEEDS:
            sep, mrg = evaluate(*runs[f"joint_separate_s{s}"], D), evaluate(*runs[f"joint_merge_s{s}"], D)
            rows.append({"dataset": D, "seed": s,
                         **{f"{k}_sep_minus_merge": round(sep[k] - mrg[k], 4)
                            for k in ("auroc_shared", "f1_pooled_shared", "f1_known_shared")}})
    write(os.path.join(out, "posthoc_static_cost.csv"), rows)
    for f in ("posthoc_old_labels.csv", "posthoc_new_dataset.csv", "posthoc_static_cost.csv"):
        print(open(os.path.join(out, f)).read())


if __name__ == "__main__":
    main()
