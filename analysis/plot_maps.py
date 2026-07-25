"""mAP-vs-#classes comparison figure — the text counterpart of EmoGrowth's
Figure 4. One panel per protocol, every method a line with markers.

Run after build_analysis.py, from the analysis/ directory.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EMO = os.path.join(HERE, "emogo")
PROTOCOLS = ["B0-I7", "B0-I4", "B16-I3", "B16-I2"]
SEED = 1993

# Fixed method order and a colorblind-safe (Okabe-Ito based) assignment.
# CVD separation validated (worst adjacent ΔE 9.6). Markers + line weight are
# the secondary encoding the contrast WARN calls for.
STYLE = {
    #            colour     marker  lw   z
    "finetune": ("#999999", "o", 1.3, 1),
    "ewc":      ("#E69F00", "s", 1.3, 1),
    "er":       ("#56B4E9", "^", 1.3, 1),
    "rs":       ("#009E73", "v", 1.3, 1),
    "ocdm":     ("#8C6D1F", "D", 1.3, 1),
    "prs":      ("#CC79A7", "P", 1.3, 1),
    "agcn":     ("#D55E00", "X", 2.4, 3),   # graph — emphasised
    "aesl":     ("#000000", "*", 2.4, 3),   # the paper's method — emphasised
    "lwf":      ("#0072B2", "d", 2.4, 3),   # best here — emphasised
}
ORDER = ["finetune", "ewc", "lwf", "er", "rs", "ocdm", "prs", "agcn", "aesl"]
LABEL = {"finetune": "Finetune", "ewc": "EWC", "lwf": "LwF", "er": "ER",
         "rs": "RS", "ocdm": "OCDM", "prs": "PRS", "agcn": "AGCN",
         "aesl": "AESL"}


def cumulative_classes(task_sizes):
    xs, total = [], 0
    for s in task_sizes:
        total += s
        xs.append(total)
    return xs


def render(dark=False):
    surface = "#1a1a1a" if dark else "#fcfcfb"
    ink = "#e8e8e8" if dark else "#1a1a1a"
    grid = "#3a3a3a" if dark else "#e5e5e5"
    plt.rcParams.update({
        "figure.facecolor": surface, "axes.facecolor": surface,
        "savefig.facecolor": surface, "text.color": ink,
        "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink,
        "axes.edgecolor": grid, "font.size": 10,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, p in zip(axes.flat, PROTOCOLS):
        for m in ORDER:
            s = json.load(open(os.path.join(EMO, m, p, f"seed{SEED}",
                                            "summary.json")))
            xs = cumulative_classes(s["task_sizes"])
            ys = [100 * v for v in s["curves"]["map"]]
            colour, marker, lw, z = STYLE[m]
            ax.plot(xs, ys, color=colour, marker=marker, linewidth=lw,
                    markersize=6 if lw < 2 else 8, label=LABEL[m], zorder=z,
                    alpha=0.95 if lw >= 2 else 0.75)
        ax.set_title(f"GoEmotions {p}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of classes")
        ax.set_ylabel("mAP (%)")
        ax.grid(True, color=grid, linewidth=0.6, alpha=0.7)
        ax.set_ylim(0, 100)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("Incremental mAP across protocols — 9 methods on GoEmotions "
                 "(28 emotions, seed 1993)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    name = "map_curves_dark.png" if dark else "map_curves.png"
    out = os.path.join(HERE, "figures", name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    render(dark=False)
    render(dark=True)
