"""Exploratory check of the solution idea for RP1 (post-hoc, not pre-registered;
3 Nov 2026). No training: uses the logits saved by the pilot.

Idea: instead of protecting each old label's column, read the old inventory
through a correspondence learned from the new one. Here the correspondence is a
ridge map from the new model's new-inventory logits to the old (stage-1) model's
logits, fitted on the NEW dataset's dev texts (the old model's outputs on new
texts are what LwF also uses; no old data). The old dataset's test labels are
then scored through the map and compared with the pilot's own columns.

The dev split stands in for the new dataset's training texts, whose logits the
pilot did not save.

    python -m _pilot.analysis.explore_map
"""

import csv
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from _pilot.analysis.pilot import NATIVE, auroc, load_runs
from utils.data import SHARED_EMOTIONS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDERS = {"xed2go": ("xed", "go"), "go2xed": ("go", "xed")}


def group_auroc(scores, y, D):
    only = [l for l in NATIVE[D] if l not in SHARED_EMOTIONS and l != "neutral"]
    a = [auroc(scores[:, j], y[:, j]) for j in range(y.shape[1])]
    return (float(np.mean([a[NATIVE[D].index(l)] for l in SHARED_EMOTIONS])),
            float(np.mean([a[NATIVE[D].index(l)] for l in only])))


def main():
    runs = load_runs(os.path.join(ROOT, "results", "rp1"))
    rows = []
    for order, (A, B) in ORDERS.items():
        for space, method in (("separate", "ft"), ("merge", "ft"), ("separate", "lwf")):
            mapped, own = [], []
            for s in (0, 1, 2):
                old, _ = runs[f"stage1_{A}_s{s}"]
                new, lay = runs[f"stage2_{order}_{space}_{method}_s{s}"]
                X = new[f"logits_{B}_dev"][:, lay[B]]
                sc = StandardScaler().fit(X)
                reg = Ridge(alpha=1.0).fit(sc.transform(X), old[f"logits_{B}_dev"])
                pred = reg.predict(sc.transform(new[f"logits_{A}_test"][:, lay[B]]))
                mapped.append(group_auroc(pred, new[f"y_{A}_test"], A))
                own.append(group_auroc(new[f"logits_{A}_test"][:, lay[A]], new[f"y_{A}_test"], A))
            rows.append({"order": order, "old_dataset": A, "model": f"{space}_{method}",
                         "own_columns_shared": round(np.mean([r[0] for r in own]), 4),
                         "own_columns_old_only": round(np.mean([r[1] for r in own]), 4),
                         "via_map_shared": round(np.mean([r[0] for r in mapped]), 4),
                         "via_map_old_only": round(np.mean([r[1] for r in mapped]), 4)})
    out = os.path.join(ROOT, "results", "rp1", "analysis", "explore_map.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(open(out).read())


if __name__ == "__main__":
    main()
