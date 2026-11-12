"""RP1 step 0, post-hoc robustness checks (not pre-registered; added 2 Nov 2026
after reading the step-0 map weights; first run as analysis/rp1_step0_robustness.py
and moved here with only its paths changed). Uses the logits cached by step0.py.

The step-0 map from all 28 GoEmotions logits beat the same-named output on
every shared emotion, but some of its weights are not interpretable (fear <-
grief, sadness <- relief, curiosity positive almost everywhere). Two shortcuts
could produce such a gain without any difference in label meaning:
  * XED's neutral lines come from a separate file, so a generic
    "emotional vs neutral line" signal helps every emotion label;
  * subtitles differ in style from Reddit, so a linear map can re-weight
    outputs to undo the domain shift.

Per shared emotion, all AUROCs out-of-fold (5-fold) on XED:
  R1 d_cross on emotional lines only (neutral-only lines removed).
  R2 map from {same-named, neutral} logits only: the generic shortcut. Its gain
     over auroc_same is the part of d_cross that needs no other emotion.
  R3 map from the same-named output plus its GoEmotions Ekman-group siblings
     (data/goemotions/ekman_mapping.json): a meaning-structured extension. If R3
     recovers most of the full gain, the gain comes from neighbouring emotions,
     i.e. a difference in label extent.

    cd emocil && ../.venv/bin/python -m _pilot.diagnosis.step0_robustness
"""

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # emocil/_pilot/
sys.path.insert(0, os.path.dirname(ROOT))
from _pilot.diagnosis.step0 import DATA, OUT, XED_LABELS, SHARED, auroc, oof_map  # noqa: E402


def main():
    from _pilot.diagnosis.step0 import load_xed
    _, Y, _ = load_xed(os.path.join(DATA, "xed"))
    L = np.load(os.path.join(OUT, "xed_logits.npz"), allow_pickle=True)["logits"]
    go = [l.strip() for l in open(os.path.join(DATA, "goemotions", "emotions.txt")) if l.strip()]
    ekman = json.load(open(os.path.join(DATA, "goemotions", "ekman_mapping.json")))
    emo = Y[:, XED_LABELS.index("neutral")] == 0
    rows = []
    for lab in SHARED:
        j, k = XED_LABELS.index(lab), go.index(lab)
        y = Y[:, j]
        same = auroc(L[:, k], y)
        full = auroc(oof_map(L, y), y)
        r1_same = auroc(L[emo, k], y[emo])
        r1_full = auroc(oof_map(L[emo], y[emo]), y[emo])
        two = auroc(oof_map(L[:, [k, go.index("neutral")]], y), y)
        sib = [go.index(m) for m in ekman[lab]]
        group = auroc(oof_map(L[:, sib], y), y)
        rows.append({
            "xed_label": lab,
            "auroc_same": round(same, 4), "auroc_full": round(full, 4),
            "d_full": round(full - same, 4),
            "R1_d_emotional_only": round(r1_full - r1_same, 4),
            "R2_d_same+neutral": round(two - same, 4),
            "R3_ekman_group": ",".join(ekman[lab]),
            "R3_d_group": round(group - same, 4),
            "R3_share_of_full_gain": round((group - same) / (full - same), 3),
        })
    out = os.path.join(OUT, "robustness.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
