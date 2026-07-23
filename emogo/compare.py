"""Build the EmoGrowth-style results table, with ../bertgo as the Upper-bound.

    python compare.py                    # B0-I7
    python compare.py --protocol B0-I4

Why this script exists: ../bertgo and this folder deliberately report different
things, so their raw output files are not comparable side by side.

  ../bertgo/output/run1/test_table4.txt
      per-emotion precision/recall/F1 at probability > 0.3, for ONE model
      trained on all 28 classes at once. That is the GoEmotions paper's
      reporting format (Table 4) and exists to prove the replication.

  results/<method>/<protocol>/seed<n>/summary.json
      mAP / maF1 / miF1 after every task, at probability > 0.5. That is
      EmoGrowth's format (Tables 1-3) and is what an incremental run produces.

They are not supposed to look alike. The relationship is that bertgo *is* the
**Upper-bound** row of the EmoGrowth table: the same backbone and data with no
incremental constraint at all. And the comparison is legitimate because after
the final task the incremental test set is exactly bertgo's test set — same
5,427 comments, same 28 columns (asserted below).

This script recomputes bertgo's predictions through this folder's metric code
so every number in the table comes from one implementation.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.metrics import MacroF1, MicroF1, average_precision  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BERTGO = os.path.join(HERE, "..", "bertgo")

# Paper Table 3, Audio28 — the 28-class analogue of GoEmotions.
# method -> (Avg mAP, Last maF1, Last miF1, Last mAP)
PAPER_AUDIO28 = {
    "B0-I7": {
        "Upper-bound": (None, 51.4, 61.1, 57.1),
        "finetune": (36.4, 9.2, 14.8, 27.3),
        "lwf": (46.6, 37.9, 51.7, 40.6),
        "aesl": (49.0, 38.4, 51.8, 42.7),
    },
    "B0-I4": {
        "Upper-bound": (None, 51.4, 61.1, 57.1),
        "finetune": (35.3, 5.3, 10.0, 23.3),
        "lwf": (45.8, 49.8, 37.6, 45.0),
        "aesl": (48.7, 41.1, 51.7, 39.8),
    },
    "B16-I3": {
        "Upper-bound": (None, 51.4, 61.1, 57.1),
        "finetune": (29.9, 4.4, 10.3, 22.6),
        "lwf": (45.0, 32.3, 45.2, 40.0),
        "aesl": (47.8, 32.3, 48.0, 42.3),
    },
    "B16-I2": {
        "Upper-bound": (None, 51.4, 61.1, 57.1),
        "finetune": (27.6, 2.8, 8.2, 20.2),
        "lwf": (44.3, 28.8, 41.4, 36.5),
        "aesl": (45.3, 30.8, 45.1, 39.3),
    },
}


def load_test_labels() -> np.ndarray:
    path = os.path.join(BERTGO, "data", "test.tsv")
    y = None
    rows = list(csv.reader(open(path, encoding="utf-8"), delimiter="\t"))
    y = np.zeros((len(rows), 28), dtype=np.float32)
    for i, r in enumerate(rows):
        for j in r[1].split(","):
            y[i, int(j)] = 1.0
    return y


def upper_bound_row():
    """bertgo scored with this folder's metrics: the no-forgetting ceiling."""
    probs_path = os.path.join(BERTGO, "output", "run1", "test_probs.npy")
    if not os.path.isfile(probs_path):
        return None
    probs = np.load(probs_path)
    y = load_test_labels()
    if probs.shape != y.shape:
        raise ValueError(f"bertgo probs {probs.shape} vs labels {y.shape}")
    _, m_ap = average_precision(torch.from_numpy(probs), torch.from_numpy(y))
    # bertgo saves probabilities, so the 0.5 cut here is the same operating
    # point as emogo's logit > 0.
    return (None, 100 * MacroF1(probs, y, 0.5), 100 * MicroF1(probs, y, 0.5),
            100 * float(m_ap))


def method_row(method: str, protocol: str, seed: int):
    path = os.path.join(HERE, "results", method, protocol, f"seed{seed}",
                        "summary.json")
    if not os.path.isfile(path):
        return None, None
    s = json.load(open(path))
    row = (100 * s["avg_acc"]["map"], 100 * s["last_acc"]["macrof1"],
           100 * s["last_acc"]["microf1"], 100 * s["last_acc"]["map"])

    # Sanity: the final task must cover the whole test set and all 28 classes,
    # otherwise the Upper-bound comparison is not like for like. Compare after
    # putting the columns back into emotions.txt order — the class order is a
    # permutation, not a difference in content.
    last = len(s["curves"]["map"]) - 1
    lp = os.path.join(os.path.dirname(path), f"task{last}_labels.npy")
    note = ""
    if os.path.isfile(lp):
        lb = np.load(lp)
        same = (lb.shape == (len(load_test_labels()), 28)
                and np.array_equal(to_canonical(lb, s["class_order"]),
                                   load_test_labels()))
        if not same:
            note = "  (final-task test set != bertgo test set!)"
    return row, note


def load_emotions():
    """Canonical display order: the emotions.txt order, which bertgo uses."""
    path = os.path.join(BERTGO, "data", "emotions.txt")
    return [l for l in open(path).read().splitlines() if l.strip()]


def to_canonical(matrix: np.ndarray, class_order):
    """Reorder an emogo matrix's columns into emotions.txt order.

    emogo's column c holds the c-th class to *arrive*, which under the paper's
    alphabetical protocol is not the emotions.txt position — `neutral` is
    alphabetically 20th but the 28th line of the file. Comparing column-by-
    column without this would silently align the wrong emotions.
    """
    canonical = load_emotions()
    if list(class_order) == canonical:
        return matrix
    pos = {e: i for i, e in enumerate(class_order)}
    perm = [pos[e] for e in canonical]
    return matrix[:, perm]


def per_emotion_table(protocol: str, seed: int, methods, threshold: float):
    """GoEmotions Table 4 format, for bertgo and each incremental method.

    utils/metrics.py never computes per-emotion precision/recall — EmoGrowth
    does not report them — so this reads the saved final-task logits and scores
    them the way ../bertgo/metrics.py does. Valid because the final task's test
    set is bertgo's test set.
    """
    from sklearn.metrics import precision_recall_fscore_support as prf

    emotions = load_emotions()
    y = load_test_labels()

    columns, names = [], []

    probs_path = os.path.join(BERTGO, "output", "run1", "test_probs.npy")
    if os.path.isfile(probs_path):
        columns.append(np.load(probs_path))
        names.append("bertgo(UB)")

    for m in methods:
        rdir = os.path.join(HERE, "results", m, protocol, f"seed{seed}")
        sp = os.path.join(rdir, "summary.json")
        if not os.path.isfile(sp):
            continue
        summary = json.load(open(sp))
        last = len(summary["curves"]["map"]) - 1
        lg = os.path.join(rdir, f"task{last}_logits.npy")
        if not os.path.isfile(lg):
            continue
        logits = np.load(lg)
        if logits.shape != y.shape:
            print(f"skipping {m}: final task is {logits.shape}, not {y.shape}")
            continue
        probs = 1.0 / (1.0 + np.exp(-logits))          # logits -> probabilities
        columns.append(to_canonical(probs, summary["class_order"]))
        names.append(m)

    if not columns:
        print("No predictions found.")
        return

    print(f"\nPer-emotion, final task, probability > {threshold} "
          f"(GoEmotions Table 4 format)\n")
    header = f"{'Emotion':<15}" + "".join(f"{n:>21}" for n in names)
    print(header)
    print(f"{'':<15}" + "".join(f"{'P':>7}{'R':>7}{'F1':>7}" for _ in names))
    print("-" * len(header))

    preds = [(c > threshold).astype(int) for c in columns]
    for j, e in enumerate(emotions):
        line = f"{e:<15}"
        for p in preds:
            P, R, F, _ = prf(y[:, j], p[:, j], average="binary", zero_division=0)
            line += f"{P:>7.2f}{R:>7.2f}{F:>7.2f}"
        print(line)

    print("-" * len(header))
    line = f"{'macro-average':<15}"
    for p in preds:
        P, R, F, _ = prf(y, p, average="macro", zero_division=0)
        line += f"{P:>7.2f}{R:>7.2f}{F:>7.2f}"
    print(line)
    line = f"{'micro-average':<15}"
    for p in preds:
        P, R, F, _ = prf(y, p, average="micro", zero_division=0)
        line += f"{P:>7.2f}{R:>7.2f}{F:>7.2f}"
    print(line)


def fmt(row):
    if row is None:
        return f"{'—':>9}{'—':>9}{'—':>9}{'—':>9}"
    avg, maf1, mif1, m_ap = row
    a = f"{avg:>9.1f}" if avg is not None else f"{'—':>9}"
    return a + f"{maf1:>9.1f}{mif1:>9.1f}{m_ap:>9.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="B0-I7")
    ap.add_argument("--seed", type=int, default=1993)
    ap.add_argument("--methods", nargs="*",
                    default=["finetune", "lwf", "ewc", "replay", "agcn",
                             "krt-r", "aesl"])
    ap.add_argument("--per-emotion", action="store_true",
                    help="Also print the GoEmotions Table 4 style breakdown, "
                         "which utils/metrics.py does not produce.")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="Probability cut for --per-emotion (default 0.3, the "
                         "GoEmotions convention; EmoGrowth uses 0.5).")
    args = ap.parse_args()

    paper = PAPER_AUDIO28.get(args.protocol, {})

    print(f"GoEmotions {args.protocol}, seed {args.seed} — 28 classes\n")
    print(f"{'':<14}{'| ours (GoEmotions)':<38}{'| paper (Audio28, Table 3)':<38}")
    print(f"{'Method':<14}{'Avg mAP':>9}{'maF1':>9}{'miF1':>9}{'mAP':>9}"
          f"   {'Avg mAP':>9}{'maF1':>9}{'miF1':>9}{'mAP':>9}")
    print("-" * 90)

    ub = upper_bound_row()
    print(f"{'Upper-bound':<14}{fmt(ub)}   {fmt(paper.get('Upper-bound'))}")
    if ub is None:
        print("   (run ../bertgo first to fill the Upper-bound row)")

    notes = []
    for m in args.methods:
        row, note = method_row(m, args.protocol, args.seed)
        if row is None and m not in paper:
            continue
        print(f"{m:<14}{fmt(row)}   {fmt(paper.get(m))}")
        if note:
            notes.append(f"{m}:{note}")
        # The paper splits labels alphabetically (Appendix B.1). Results
        # produced under a different order are not comparable with these
        # numbers, or with each other.
        sp = os.path.join(HERE, "results", m, args.protocol,
                          f"seed{args.seed}", "summary.json")
        if os.path.isfile(sp):
            co = json.load(open(sp))["class_order"]
            if list(co) != sorted(co):
                notes.append(
                    f"{m}: class order is NOT alphabetical — stale result, "
                    f"predates the Appendix B.1 fix. Re-run."
                )

    print("-" * 90)
    print("Last-task columns only; Avg mAP is the mean over all tasks.")
    print("Paper columns are a different modality (audio) with a frozen "
          "backbone — read the ordering and the gaps, not the absolute values.")
    for n in notes:
        print("WARNING " + n)

    if args.per_emotion:
        per_emotion_table(args.protocol, args.seed, args.methods,
                          args.threshold)


if __name__ == "__main__":
    main()
