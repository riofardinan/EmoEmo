"""Build the analysis of the emogo runs: EmoGrowth tables + mAP figures +
GoEmotions per-emotion breakdown. Run from the analysis/ directory.

Outputs (into analysis/):
  tables/emogrowth_<protocol>.csv   Avg mAP, Last maF1/miF1/mAP per method
  tables/goemotions_final.csv       per-emotion P/R/F1 at threshold 0.3
  figures/map_curves.png            mAP vs #classes, one panel per protocol
  figures/map_curves_dark.png       same, dark surface
"""

import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EMO = os.path.join(HERE, "emogo")
BERTGO_PROBS = os.path.join(HERE, "bertgo", "output", "run1", "test_probs.npy")
TEST_TSV = os.path.join(HERE, "..", "emogo", "data", "test.tsv")
EMOTIONS_TXT = os.path.join(HERE, "..", "emogo", "data", "emotions.txt")

METHODS = ["finetune", "ewc", "lwf", "er", "rs", "ocdm", "prs", "agcn", "aesl"]
PROTOCOLS = ["B0-I7", "B0-I4", "B16-I3", "B16-I2"]
SEED = 1993

# Paper Table 3 (Audio28), for the "vs paper" column.
PAPER = {
    "B0-I7": {"finetune": (36.4, 9.2, 14.8, 27.3), "ewc": (37.9, 8.3, 14.3, 29.3),
              "lwf": (46.6, 37.9, 51.7, 40.6), "er": (44.7, 8.1, 14.4, 38.0),
              "rs": (43.7, 8.1, 12.3, 36.5), "ocdm": (44.5, 8.7, 12.0, 38.4),
              "prs": (43.3, 10.8, 13.5, 35.5), "agcn": (47.3, 35.3, 50.9, 41.9),
              "aesl": (49.0, 38.4, 51.8, 42.7)},
    "B0-I4": {"finetune": (35.3, 5.3, 10.0, 23.3), "ewc": (37.1, 5.4, 10.5, 26.6),
              "lwf": (45.8, 49.8, 37.6, 45.0), "er": (44.6, 6.5, 5.5, 35.2),
              "rs": (43.6, 5.9, 9.3, 32.0), "ocdm": (44.5, 7.5, 8.8, 31.5),
              "prs": (44.5, 6.8, 8.7, 34.2), "agcn": (47.3, 37.5, 51.0, 38.6),
              "aesl": (48.7, 41.1, 51.7, 39.8)},
    "B16-I3": {"finetune": (29.9, 4.4, 10.3, 22.6), "ewc": (32.2, 4.4, 9.7, 24.7),
               "lwf": (45.0, 32.3, 45.2, 40.0), "er": (41.3, 9.2, 13.3, 36.8),
               "rs": (38.7, 7.5, 11.7, 32.9), "ocdm": (38.2, 5.5, 9.7, 30.1),
               "prs": (38.2, 5.5, 5.8, 32.7), "agcn": (39.5, 22.7, 38.4, 37.0),
               "aesl": (47.8, 32.3, 48.0, 42.3)},
    "B16-I2": {"finetune": (27.6, 2.8, 8.2, 20.2), "ewc": (28.1, 2.8, 8.7, 22.5),
               "lwf": (44.3, 28.8, 41.4, 36.5), "er": (39.4, 10.1, 13.6, 34.1),
               "rs": (38.2, 5.8, 11.6, 31.8), "ocdm": (36.3, 3.7, 7.9, 30.2),
               "prs": (37.7, 4.4, 7.6, 32.5), "agcn": (36.1, 24.4, 36.4, 30.9),
               "aesl": (45.3, 30.8, 45.1, 39.3)},
}


def load_summary(method, protocol):
    path = os.path.join(EMO, method, protocol, f"seed{SEED}", "summary.json")
    return json.load(open(path))


def load_emotions():
    return [l for l in open(EMOTIONS_TXT).read().splitlines() if l.strip()]


def load_test_labels():
    rows = list(csv.reader(open(TEST_TSV, encoding="utf-8"), delimiter="\t"))
    y = np.zeros((len(rows), 28), dtype=np.float64)
    for i, r in enumerate(rows):
        for j in r[1].split(","):
            y[i, int(j)] = 1.0
    return y


def to_canonical(matrix, class_order):
    """Reorder columns from arrival order into emotions.txt order."""
    canonical = load_emotions()
    if list(class_order) == canonical:
        return matrix
    pos = {e: i for i, e in enumerate(class_order)}
    return matrix[:, [pos[e] for e in canonical]]


# --------------------------------------------------------- EmoGrowth tables

def build_emogrowth_tables():
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    for p in PROTOCOLS:
        rows = []
        for m in METHODS:
            s = load_summary(m, p)
            rows.append({
                "method": m,
                "avg_map": 100 * s["avg_acc"]["map"],
                "last_maf1": 100 * s["last_acc"]["macrof1"],
                "last_mif1": 100 * s["last_acc"]["microf1"],
                "last_map": 100 * s["last_acc"]["map"],
                "paper_avg_map": PAPER[p][m][0],
                "paper_last_maf1": PAPER[p][m][1],
                "paper_last_mif1": PAPER[p][m][2],
                "paper_last_map": PAPER[p][m][3],
            })
        out = os.path.join(HERE, "tables", f"emogrowth_{p}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 1) if isinstance(v, float) else v)
                            for k, v in r.items()})
        print_table(p, rows)


def print_table(protocol, rows):
    print(f"\n=== {protocol} — GoEmotions (ours) vs Audio28 (paper) ===")
    print(f"{'method':<10}{'Avg mAP':>9}{'Last maF1':>10}{'Last miF1':>10}"
          f"{'Last mAP':>9}   | {'paper Avg/Last mAP':>18}")
    print("-" * 78)
    best_map = max(r["last_map"] for r in rows)
    for r in rows:
        star = " *" if r["last_map"] == best_map else "  "
        print(f"{r['method']:<10}{r['avg_map']:>9.1f}{r['last_maf1']:>10.1f}"
              f"{r['last_mif1']:>10.1f}{r['last_map']:>9.1f}{star} | "
              f"{r['paper_avg_map']:>8.1f}/{r['paper_last_map']:<8.1f}")


# ------------------------------------------------------ GoEmotions taxonomy

def build_goemotions_table():
    from sklearn.metrics import precision_recall_fscore_support as prf

    emotions = load_emotions()
    y = load_test_labels()
    thr = 0.3

    columns = {}
    if os.path.isfile(BERTGO_PROBS):
        columns["upper-bound"] = np.load(BERTGO_PROBS)
    # The strongest incremental method plus the paper's own method.
    for m in ("lwf", "aesl", "agcn", "finetune"):
        s = load_summary(m, "B0-I7")
        last = len(s["curves"]["map"]) - 1
        lg = np.load(os.path.join(EMO, m, "B0-I7", f"seed{SEED}",
                                  f"task{last}_logits.npy"))
        columns[m] = to_canonical(1.0 / (1.0 + np.exp(-lg)), s["class_order"])

    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    out = os.path.join(HERE, "tables", "goemotions_final.csv")
    names = list(columns.keys())
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        header = ["emotion"] + [f"{n}_{s}" for n in names
                                for s in ("P", "R", "F1")]
        w.writerow(header)
        for j, e in enumerate(emotions):
            row = [e]
            for n in names:
                p = (columns[n] > thr).astype(int)
                P, R, F, _ = prf(y[:, j], p[:, j], average="binary",
                                 zero_division=0)
                row += [round(P, 2), round(R, 2), round(F, 2)]
            w.writerow(row)
        # macro / micro rows
        for avg in ("macro", "micro"):
            row = [f"{avg}-average"]
            for n in names:
                p = (columns[n] > thr).astype(int)
                P, R, F, _ = prf(y, p, average=avg, zero_division=0)
                row += [round(P, 2), round(R, 2), round(F, 2)]
            w.writerow(row)

    # Console view.
    print("\n=== GoEmotions taxonomy, final task (28 classes), threshold 0.3 ===")
    print(f"{'emotion':<15}" + "".join(f"{n:>14}" for n in names))
    print(f"{'':<15}" + "".join(f"{'P':>5}{'R':>5}{'F1':>4}" for _ in names))
    print("-" * (15 + 14 * len(names)))
    for j, e in enumerate(emotions):
        line = f"{e:<15}"
        for n in names:
            p = (columns[n] > thr).astype(int)
            P, R, F, _ = prf(y[:, j], p[:, j], average="binary", zero_division=0)
            line += f"{P:>5.2f}{R:>5.2f}{F:>4.2f}"
        print(line)
    for avg in ("macro", "micro"):
        line = f"{avg+'-avg':<15}"
        for n in names:
            p = (columns[n] > thr).astype(int)
            P, R, F, _ = prf(y, p, average=avg, zero_division=0)
            line += f"{P:>5.2f}{R:>5.2f}{F:>4.2f}"
        print(line)
    return names


if __name__ == "__main__":
    build_emogrowth_tables()
    build_goemotions_table()
    print("\nWrote tables/ ; run plot_maps.py for the figures.")
