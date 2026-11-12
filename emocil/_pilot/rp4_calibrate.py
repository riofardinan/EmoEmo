"""RP4 pilot: epoch calibration, run before any pilot run (pre-registered 4 Nov 2026).

BRIGHTER training sets are about a tenth the size of XED or GoEmotions. For each
pair, the number of epochs per stage is chosen from {4, 10}. The criterion is
the stage-1 models' dev macro-AUROC, each model on its own language's dev split,
averaged over the pair's two languages. Seed 99 is used, which no pilot run
uses. The test split is never read for the choice; train.py still writes test
logits, and nothing here opens them.

    cd emocil && ../.venv/bin/python -m _pilot.rp4_calibrate            # _pilot/results/rp4_calib/
    cd emocil && ../.venv/bin/python -m _pilot.rp4_calibrate --amended  # _pilot/results/rp4_calib/amended/

Amendment (4 Nov 2026, before any pilot run, test not read). With the
pre-registered recipe, every Hindi and Sundanese stage-1 model collapsed to a
constant output: dev logit std about 0, training loss flat at the label-prior
entropy. English trained normally. This is the failed-run pattern of Mosbach
et al. (2021, ICLR; §7.7: "throughout the training, their training loss stays
close to that at initialization"). Their baseline for small datasets is Adam
with bias correction, lr 2e-5 and 20 epochs. `--amended` adopts that recipe
whole, without a new grid. It checks on dev, with seeds 99 and 98, that no
stage-1 model collapses and that dev macro-AUROC beats the pre-registered
recipe. A stage-1 English model does not depend on the pair (the pair only
changes which eval splits are written), so it is trained once per seed.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from _pilot.analysis.pilot import auroc

ROOT = os.path.dirname(os.path.abspath(__file__))          # emocil/_pilot
EMOCIL = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "results", "rp4_calib")
MODEL = "google-bert/bert-base-multilingual-cased"
SEED, EPOCHS = 99, (4, 10)
PAIRS = {"enghin": ("b_eng", "b_hin"), "engsun": ("b_eng", "b_sun")}


def dev_macro_auroc(run_dir, ds):
    z = np.load(os.path.join(run_dir, "logits.npz"))
    s, y = z[f"logits_{ds}_dev"], z[f"y_{ds}_dev"]
    return float(np.mean([auroc(s[:, j], y[:, j]) for j in range(y.shape[1])]))


AMENDED = ["--bias-correction", "--lr", "2e-5", "--epochs", "20"]
AMENDED_SEEDS = (99, 98)
LANG_PAIR = {"b_eng": "enghin", "b_hin": "enghin", "b_sun": "engsun"}


def collapsed(run_dir, ds):
    z = np.load(os.path.join(run_dir, "logits.npz"))
    return bool(z[f"logits_{ds}_dev"].std(0).max() < 1e-2)


def amended():
    out = os.path.join(OUT, "amended")
    per = {}
    for ds, pair in LANG_PAIR.items():
        for seed in AMENDED_SEEDS:
            run = os.path.join(out, f"stage1_{ds}_s{seed}")
            if not os.path.exists(os.path.join(run, "logits.npz")):
                subprocess.run([sys.executable, "train.py", "--mode", "stage1", "--pair", pair,
                                "--first", ds, "--seed", str(seed), "--model", MODEL, "--out", out]
                               + AMENDED, cwd=EMOCIL, check=True, stdout=subprocess.DEVNULL)
            per[f"{ds}_s{seed}"] = {"dev_macro_auroc": round(dev_macro_auroc(run, ds), 4),
                                    "collapsed": collapsed(run, ds)}
            print(ds, seed, per[f"{ds}_s{seed}"], flush=True)
    result = {"model": MODEL, "recipe": " ".join(AMENDED), "seeds": AMENDED_SEEDS, "runs": per}
    for pair, langs in PAIRS.items():
        result[pair] = {f"mean_s{s}": round(float(np.mean([per[f"{d}_s{s}"]["dev_macro_auroc"]
                                                           for d in langs])), 4) for s in AMENDED_SEEDS}
    result["any_collapsed"] = any(r["collapsed"] for r in per.values())
    with open(os.path.join(out, "calibration_amended.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amended", action="store_true")
    if ap.parse_args().amended:
        return amended()
    os.makedirs(OUT, exist_ok=True)
    result = {"model": MODEL, "seed": SEED, "criterion": "stage-1 dev macro-AUROC, mean over the pair"}
    for pair, langs in PAIRS.items():
        per = {}
        for e in EPOCHS:
            out = os.path.join(OUT, f"{pair}_e{e}")
            scores = {}
            for ds in langs:
                run = os.path.join(out, f"stage1_{ds}_s{SEED}")
                if not os.path.exists(os.path.join(run, "logits.npz")):
                    subprocess.run([sys.executable, "train.py", "--mode", "stage1", "--pair", pair,
                                    "--first", ds, "--seed", str(SEED), "--model", MODEL,
                                    "--epochs", str(e), "--out", out], cwd=EMOCIL, check=True,
                                   stdout=subprocess.DEVNULL)
                scores[ds] = round(dev_macro_auroc(run, ds), 4)
            per[e] = {"per_language": scores, "mean": round(float(np.mean(list(scores.values()))), 4)}
            print(pair, e, per[e], flush=True)
        best = max(EPOCHS, key=lambda e: (per[e]["mean"], -e))   # tie -> fewer epochs
        result[pair] = {"by_epochs": per, "chosen_epochs": best}
    with open(os.path.join(OUT, "calibration.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
