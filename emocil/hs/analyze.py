"""HS2 experiment: metrics, readouts and the pre-registered decision.

The rule is in ../README.md, "HS2 experiment". This runs locally on the
downloaded results (logits.npz, meta.json, corr.npz) and needs no GPU.

For each order (A = old inventory, B = new) and each complete seed, every
method is scored on both inventories:
  old   A's test, read the way the method reads A:
          own columns    ft_separate, lwf_separate, lwf_all_merge, joint, frozen_*
          map            M(z_B), the correspondence learned during training ("readout_*")
          map_refit      ridge from the map run's B logits to the stage-1 A model's
                         logits, fitted on B's training texts (secondary)
          map_readout    the same ridge on the ft_separate run (no training change)
          oracle_or      Ekman label = max over its official fine members, from
                         ft_separate (ekfine, order ekman2fine only; reference)
  new   B's test, from the method's B columns.
Metrics: AUROC over the six shared emotions (old, primary), AUROC over all
labels, and macro-F1 with one threshold per label tuned on the dataset's own dev
split (new, primary).

    cd emocil && python -m hs.analyze --pair ekfine     # results/hs_ekfine/analysis/
"""

import argparse
import csv
import json
import os
import re

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from _pilot.analysis.pilot import auroc, best_threshold, f1
from utils.data import DATA, labels_of, shared_emotions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # emocil/
ORDERS = {"ekfine": {"ekman2fine": ("ekman", "fine"), "fine2ekman": ("fine", "ekman")},
          "xedgo": {"xed2go": ("xed", "go"), "go2xed": ("go", "xed")}}
PRIMARY = {"ekfine": "ekman2fine", "xedgo": None}       # no HS1 rule is fixed yet
BASELINES = ["ft_separate", "lwf_separate", "lwf_all_merge"]
FROZEN = ["frozen_bert", "frozen_qwen"]
MARGIN = 0.01
RIDGE_ALPHA = 1.0


def load_run(path):
    arr = dict(np.load(os.path.join(path, "logits.npz")))
    with open(os.path.join(path, "meta.json")) as f:
        layout = json.load(f)["layout"]
    return arr, layout


def score(S_test, Y_test, S_dev, Y_dev, labels, shared):
    au, f = [], []
    for j in range(len(labels)):
        au.append(auroc(S_test[:, j], Y_test[:, j]))
        f.append(f1(S_test[:, j], Y_test[:, j], best_threshold(S_dev[:, j], Y_dev[:, j])))
    au, f = np.array(au, dtype=float), np.array(f, dtype=float)
    sh = [labels.index(e) for e in shared]
    return {"auroc_shared": float(np.nanmean(au[sh])), "auroc_all": float(np.nanmean(au)),
            "f1_all": float(np.nanmean(f))}


def own(run, D, split):
    arr, lay = run
    return arr[f"logits_{D}_{split}"][:, lay[D]]


def ridge_readout(run, teacher, A, B):
    """Map run's B logits -> stage-1 A model's logits, fitted on B's training texts;
    returns A scores for A's dev and test texts, and the map coefficients."""
    X = own(run, B, "train")
    T = teacher[0][f"logits_{B}_train"][:, teacher[1][A]]
    sc = StandardScaler().fit(X)
    reg = Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(X), T)
    arr, lay = run
    out = {s: reg.predict(sc.transform(arr[f"logits_{A}_{s}"][:, lay[B]])) for s in ("dev", "test")}
    return out, reg.coef_ / sc.scale_[None, :]           # coefficients on the raw logit scale


def oracle_or(run, A, B):
    groups = json.load(open(os.path.join(DATA, "goemotions", "ekman_mapping.json")))
    fine = labels_of(B)
    members = [[fine.index(m) for m in (groups[e] if e != "neutral" else ["neutral"])] for e in labels_of(A)]
    arr, lay = run
    return {s: np.stack([arr[f"logits_{A}_{s}"][:, lay[B]][:, mm].max(1) for mm in members], 1)
            for s in ("dev", "test")}


def beats(m, b):
    """m beats b: not worse than b by more than MARGIN on either inventory, and
    better by at least MARGIN on one."""
    no_loss = m["old"] >= b["old"] - MARGIN and m["new"] >= b["new"] - MARGIN
    return bool(no_loss and (m["old"] >= b["old"] + MARGIN or m["new"] >= b["new"] + MARGIN))


def correspondence(weights, A, B):
    """Share of each Ekman label's k largest positive weights (k = group size) that
    fall on its official fine members."""
    groups = json.load(open(os.path.join(DATA, "goemotions", "ekman_mapping.json")))
    fine, out = labels_of(B), {}
    for i, e in enumerate(labels_of(A)):
        mem = set(groups[e]) if e != "neutral" else {"neutral"}
        top = np.argsort(-weights[i])[:len(mem)]
        out[e] = round(float(np.mean([fine[j] in mem for j in top])), 3)
    return out


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=list(ORDERS), default="ekfine")
    ap.add_argument("--runs", default=None, help="default results/hs_<pair>")
    a = ap.parse_args()
    runs_dir = a.runs or os.path.join(ROOT, "results", f"hs_{a.pair}")
    out_dir = os.path.join(runs_dir, "analysis"); os.makedirs(out_dir, exist_ok=True)
    names = sorted(d for d in os.listdir(runs_dir) if os.path.exists(os.path.join(runs_dir, d, "logits.npz")))
    seeds = sorted({int(m.group(1)) for n in names for m in [re.search(r"_s(\d+)$", n)] if m})
    shared = shared_emotions(a.pair)
    frozen = {f: load_run(os.path.join(runs_dir, f)) for f in FROZEN if f in names}

    rows, table, weights_rows = [], {}, []
    for order, (A, B) in ORDERS[a.pair].items():
        LA, LB = labels_of(A), labels_of(B)
        for s in seeds:
            need = {"native_old": f"stage1_{A}_s{s}", "native_new": f"stage1_{B}_s{s}",
                    "joint_separate": f"joint_separate_s{s}",
                    "ft_separate": f"stage2_{order}_separate_ft_s{s}",
                    "lwf_separate": f"stage2_{order}_separate_lwf_s{s}",
                    "lwf_all_merge": f"stage2_{order}_merge_lwf_all_s{s}",
                    "map": f"stage2_{order}_separate_map_s{s}"}
            if not all(v in names for v in need.values()):
                continue
            R = {k: load_run(os.path.join(runs_dir, v)) for k, v in need.items()}
            R.update(frozen)
            yA = {sp: R["native_old"][0][f"y_{A}_{sp}"] for sp in ("dev", "test")}
            yB = {sp: R["native_new"][0][f"y_{B}_{sp}"] for sp in ("dev", "test")}

            def old_from(S):
                return score(S["test"], yA["test"], S["dev"], yA["dev"], LA, shared)

            def new_of(run):
                return score(own(run, B, "test"), yB["test"], own(run, B, "dev"), yB["dev"], LB, shared)

            res = {}
            res["native_old"] = {"old": old_from({sp: own(R["native_old"], A, sp) for sp in ("dev", "test")})}
            res["native_new"] = {"new": new_of(R["native_new"])}
            for k in ["joint_separate", "ft_separate", "lwf_separate", "lwf_all_merge"] + list(frozen):
                res[k] = {"old": old_from({sp: own(R[k], A, sp) for sp in ("dev", "test")}), "new": new_of(R[k])}
            mp = R["map"][0]
            res["map"] = {"old": old_from({sp: mp[f"readout_{A}_{sp}"] for sp in ("dev", "test")}),
                          "new": new_of(R["map"])}
            refit, w_refit = ridge_readout(R["map"], R["native_old"], A, B)
            res["map_refit"] = {"old": old_from(refit), "new": res["map"]["new"]}
            ro, w_ro = ridge_readout(R["ft_separate"], R["native_old"], A, B)
            res["map_readout"] = {"old": old_from(ro), "new": res["ft_separate"]["new"]}
            if a.pair == "ekfine" and A == "ekman":
                res["oracle_or"] = {"old": old_from(oracle_or(R["ft_separate"], A, B)),
                                    "new": res["ft_separate"]["new"]}
                corr = np.load(os.path.join(runs_dir, need["map"], "corr.npz"))
                for src, W in (("map (learned M)", corr["weight"]), ("map_refit", w_refit), ("map_readout", w_ro)):
                    weights_rows.append({"order": order, "seed": s, "map": src, **correspondence(W, A, B)})
            for k, v in res.items():
                rows.append({"order": order, "seed": s, "method": k,
                             **{f"old_{m}": round(x, 4) for m, x in v.get("old", {}).items()},
                             **{f"new_{m}": round(x, 4) for m, x in v.get("new", {}).items()}})
                table[(order, s, k)] = {"old": v["old"]["auroc_shared"] if "old" in v else None,
                                        "new": v["new"]["f1_all"] if "new" in v else None}
    if not rows:
        raise SystemExit(f"no complete seed in {runs_dir}")
    cols = sorted({c for r in rows for c in r}, key=lambda c: (c not in ("order", "seed", "method"), c))
    rows = [{c: r.get(c, "") for c in cols} for r in rows]
    write(os.path.join(out_dir, "scores.csv"), rows)
    summary = {}
    for (order, s, k), v in table.items():
        summary.setdefault((order, k), []).append(v)
    srows = [{"order": o, "method": k, "seeds": len(v),
              "old_auroc_shared": round(float(np.mean([x["old"] for x in v])), 4) if v[0]["old"] is not None else "",
              "new_f1_all": round(float(np.mean([x["new"] for x in v])), 4) if v[0]["new"] is not None else ""}
             for (o, k), v in summary.items()]
    write(os.path.join(out_dir, "summary.csv"), srows)
    if weights_rows:
        write(os.path.join(out_dir, "correspondence.csv"), weights_rows)

    verdict = {"pair": a.pair, "margin": MARGIN, "frozen_runs": sorted(frozen), "orders": {}}
    for order in ORDERS[a.pair]:
        done = sorted({s for (o, s, k) in table if o == order})
        if not done:
            continue
        info = {"seeds_complete": done}
        for cand in ("map", "map_readout"):
            per = []
            for s in done:
                m = table[(order, s, cand)]
                per.append({"seed": s,
                            "beats": {b: beats(m, table[(order, s, b)]) for b in BASELINES},
                            "beaten_by_frozen": [f for f in frozen if beats(table[(order, s, f)], m)]})
            info[cand] = {"per_seed": per,
                          "holds": all(all(p["beats"].values()) and not p["beaten_by_frozen"] for p in per)}
        if a.pair == "ekfine":
            t = lambda k, s: table[(order, s, k)]
            if order == "ekman2fine":
                info["P1_readout_near_native"] = all(
                    t("map_readout", s)["old"] >= t("native_old", s)["old"] - MARGIN and
                    t("map_readout", s)["new"] >= t("native_new", s)["new"] - MARGIN for s in done)
                info["P3_distill_raises_old_keeps_new"] = all(
                    t("map", s)["old"] > t("map_readout", s)["old"] and
                    t("map", s)["new"] >= t("ft_separate", s)["new"] - MARGIN for s in done)
            else:
                info["P2_map_does_not_beat_lwf_separate"] = all(
                    not beats(t("map", s), t("lwf_separate", s)) for s in done)
        verdict["orders"][order] = info
    prim = PRIMARY[a.pair]
    if prim and prim in verdict["orders"]:
        o = verdict["orders"][prim]
        verdict["decision"] = ("supported" if o["map"]["holds"] else
                               "partial (readout only)" if o["map_readout"]["holds"] else "not supported")
        verdict["provisional"] = len(frozen) < len(FROZEN)      # frozen baselines missing
    else:
        verdict["decision"] = "no pre-registered rule for this pair"
    with open(os.path.join(out_dir, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    print(open(os.path.join(out_dir, "summary.csv")).read())
    if weights_rows:
        print(open(os.path.join(out_dir, "correspondence.csv")).read())
    print(json.dumps({k: verdict[k] for k in ("decision",) + (("provisional",) if "provisional" in verdict else ())}))


if __name__ == "__main__":
    main()
