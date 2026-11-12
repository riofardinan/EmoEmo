"""RP2 pilot: checks beyond the decision rule (4 Nov 2026).

1. Per-label amplification against the predictions recorded before the run:
   about zero for disgust (Ekman disgust has the same scope as fine disgust),
   growing with scope, largest for joy (12 fine members). Scope sizes come from
   data/goemotions/ekman_mapping.json.
2. Encoder drift. In `fine2ekman` the old inventory (fine) has 21 labels that no
   Ekman label shares a name with. If merging cost only came from a less
   constrained encoder, those would drop with merging too.
3. Native references and the static merge cost (joint separate - merge).

Writes results/rp2/analysis/posthoc_per_label.csv, posthoc_old_labels.csv
and posthoc_static_cost.csv.

    python -m _pilot.analysis.rp2_posthoc
"""

import csv
import json
import os

import numpy as np

from _pilot.analysis.pilot import NATIVE, auroc, evaluate, load_runs
from utils.data import DATA, SHARED_EMOTIONS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "rp2")
ORDERS = {"ekman2fine": ("ekman", "fine"), "fine2ekman": ("fine", "ekman")}
SEEDS = (0, 1, 2)


def label_auroc(run, D, label):
    arr, lay = run
    j = NATIVE[D].index(label)
    return auroc(arr[f"logits_{D}_test"][:, lay[D][j]], arr[f"y_{D}_test"][:, j])


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    runs = load_runs(RUNS)
    out = os.path.join(RUNS, "analysis")
    scope = {e: len(m) for e, m in json.load(open(os.path.join(DATA, "goemotions", "ekman_mapping.json"))).items()}

    rows = []
    for order, (A, _) in ORDERS.items():
        for m in ("ft", "lwf", "er"):
            for lab in SHARED_EMOTIONS:
                amps = []
                for s in SEEDS:
                    seq = (label_auroc(runs[f"stage2_{order}_separate_{m}_s{s}"], A, lab)
                           - label_auroc(runs[f"stage2_{order}_merge_{m}_s{s}"], A, lab))
                    jnt = (label_auroc(runs[f"joint_separate_s{s}"], A, lab)
                           - label_auroc(runs[f"joint_merge_s{s}"], A, lab))
                    amps.append(seq - jnt)
                rows.append({"order": order, "method": m, "label": lab, "ekman_scope": scope[lab],
                             "amp_mean": round(float(np.mean(amps)), 4),
                             "amp_seeds": " ".join(f"{a:+.4f}" for a in amps)})
    write(os.path.join(out, "posthoc_per_label.csv"), rows)

    rows = []
    A = "fine"
    only = [l for l in NATIVE["fine"] if l not in NATIVE["ekman"]]
    for label, fn in [("native", lambda s: f"stage1_fine_s{s}")] + [
            (f"{m}_{sp}", lambda s, m=m, sp=sp: f"stage2_fine2ekman_{sp}_{m}_s{s}")
            for m in ("ft", "lwf", "er") for sp in ("separate", "merge")] + [
            (f"joint_{sp}", lambda s, sp=sp: f"joint_{sp}_s{s}") for sp in ("separate", "merge")]:
        sh = np.mean([np.mean([label_auroc(runs[fn(s)], A, l) for l in SHARED_EMOTIONS]) for s in SEEDS])
        ol = np.mean([np.mean([label_auroc(runs[fn(s)], A, l) for l in only]) for s in SEEDS])
        rows.append({"old_dataset": "fine (order fine2ekman)", "run": label, "auroc_shared": round(float(sh), 4),
                     "auroc_fine_only": round(float(ol), 4), "n_fine_only": len(only)})
    write(os.path.join(out, "posthoc_old_labels.csv"), rows)

    rows = []
    for D in ("ekman", "fine"):
        nat = [evaluate(*runs[f"stage1_{D}_s{s}"], D) for s in SEEDS]
        for s in SEEDS:
            sep, mrg = evaluate(*runs[f"joint_separate_s{s}"], D), evaluate(*runs[f"joint_merge_s{s}"], D)
            rows.append({"dataset": D, "seed": s,
                         "native_auroc_shared": round(nat[s]["auroc_shared"], 4),
                         "native_f1_known_all": round(nat[s]["f1_known_all"], 4),
                         "joint_auroc_shared_sep_minus_merge": round(sep["auroc_shared"] - mrg["auroc_shared"], 4),
                         "joint_f1_pooled_shared_sep_minus_merge": round(sep["f1_pooled_shared"] - mrg["f1_pooled_shared"], 4)})
    write(os.path.join(out, "posthoc_static_cost.csv"), rows)
    for f in ("posthoc_per_label.csv", "posthoc_old_labels.csv", "posthoc_static_cost.csv"):
        print(open(os.path.join(out, f)).read())


if __name__ == "__main__":
    main()
