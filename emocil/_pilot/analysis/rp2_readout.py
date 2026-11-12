"""RP2, exploratory and post-hoc (4 Nov 2026; not pre-registered): read the old
inventory through a correspondence instead of protecting its own columns.

Order `ekman2fine` only: the old inventory is Ekman, the new one fine, and the
true correspondence is known (an Ekman label is the OR of its fine members).
No training; uses the logits the RP2 pilot saved. For each stage-2 model, the
Ekman test labels are scored from the model's *fine* outputs in three ways:

  oracle_max   max over the member fine logits (OR of the members).
  learned_map  ridge from the 28 fine logits to the old (stage-1 Ekman) model's
               logits, fitted on the dev texts. No labels and no old data, as
               in HS1; the dev split stands in for the new training texts.
  own_columns  the model's own Ekman outputs (what the pilot measured).

    python -m _pilot.analysis.rp2_readout
"""

import csv
import json
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from _pilot.analysis.pilot import auroc, load_runs
from utils.data import DATA, EKMAN_LABELS, SHARED_EMOTIONS, go_labels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "rp2")


def shared_auroc(scores, y):
    return float(np.mean([auroc(scores[:, EKMAN_LABELS.index(e)], y[:, EKMAN_LABELS.index(e)])
                          for e in SHARED_EMOTIONS]))


def main():
    runs = load_runs(RUNS)
    go = go_labels()
    groups = json.load(open(os.path.join(DATA, "goemotions", "ekman_mapping.json")))
    members = [[go.index(m) for m in (groups[e] if e != "neutral" else ["neutral"])] for e in EKMAN_LABELS]
    rows = []
    for m in ("ft", "lwf", "er"):
        for sp in ("separate", "merge"):
            res = {"own_columns": [], "oracle_max": [], "learned_map": []}
            for s in (0, 1, 2):
                arr, lay = runs[f"stage2_ekman2fine_{sp}_{m}_s{s}"]
                old, _ = runs[f"stage1_ekman_s{s}"]
                fcols, ecols = lay["fine"], lay["ekman"]
                y = arr["y_ekman_test"]
                F_test = arr["logits_ekman_test"][:, fcols]          # fine outputs on the (shared) test texts
                F_dev = arr["logits_fine_dev"][:, fcols]
                res["own_columns"].append(shared_auroc(arr["logits_ekman_test"][:, ecols], y))
                res["oracle_max"].append(shared_auroc(np.stack([F_test[:, mm].max(1) for mm in members], 1), y))
                sc = StandardScaler().fit(F_dev)
                reg = Ridge(alpha=1.0).fit(sc.transform(F_dev), old["logits_fine_dev"])
                res["learned_map"].append(shared_auroc(reg.predict(sc.transform(F_test)), y))
            rows.append({"model": f"{m}_{sp}", **{k: round(float(np.mean(v)), 4) for k, v in res.items()}})
    out = os.path.join(RUNS, "analysis", "readout.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(open(out).read())


if __name__ == "__main__":
    main()
