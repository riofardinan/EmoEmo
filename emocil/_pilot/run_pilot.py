"""Runs the whole RP1 (`--pair xedgo`) or RP2 (`--pair ekfine`) pilot, resumably,
then the analysis.

Order: the six stage-1 models (both datasets x 3 seeds) first, since every
stage-2 run starts from one; then, seed by seed, the 12 stage-2 runs (2 orders
x 2 label spaces x 3 methods) and the 2 joint runs, so that complete results
for seed 0 exist early. A run is skipped when its logits.npz already exists,
so the script can be restarted after an interruption. A failed run is logged
and the rest continue.

    python -m _pilot.run_pilot                    # 6 + 3 x 14 = 48 runs, then analysis
    python -m _pilot.run_pilot --model prajjwal1/bert-tiny --limit 300 --eval-limit 200 \
        --epochs 1 --out /tmp/smoke         # smoke test
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))          # emocil/_pilot
EMOCIL = os.path.dirname(ROOT)                               # train.py lives here


PAIRS = {"xedgo": ("xed", "go"), "ekfine": ("ekman", "fine"),       # stage-1 order
         "enghin": ("b_eng", "b_hin"), "engsun": ("b_eng", "b_sun")}
OUT_BY_PAIR = {"xedgo": "rp1", "ekfine": "rp2", "enghin": "rp4_enghin", "engsun": "rp4_engsun"}


def jobs(seeds, pair):
    for first in PAIRS[pair]:
        for s in seeds:
            yield ["--mode", "stage1", "--first", first, "--seed", str(s)]
    for s in seeds:
        for first in PAIRS[pair]:
            for space in ("separate", "merge"):
                for method in ("ft", "lwf", "er"):
                    yield ["--mode", "stage2", "--first", first, "--space", space,
                           "--method", method, "--seed", str(s)]
        for space in ("separate", "merge"):
            yield ["--mode", "joint", "--space", space, "--seed", str(s)]


def name_of(job, pair):
    d = dict(zip(job[::2], job[1::2]))
    if d["--mode"] == "stage1":
        return f"stage1_{d['--first']}_s{d['--seed']}"
    if d["--mode"] == "joint":
        return f"joint_{d['--space']}_s{d['--seed']}"
    a, b = PAIRS[pair]
    other = b if d["--first"] == a else a
    return f"stage2_{d['--first']}2{other}_{d['--space']}_{d['--method']}_s{d['--seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--pair", choices=list(PAIRS), default="xedgo")
    ap.add_argument("--out", default=None, help="default _pilot/results/<rp1|rp2|rp4_enghin|rp4_engsun>")
    ap.add_argument("--model"); ap.add_argument("--epochs"); ap.add_argument("--lr")
    ap.add_argument("--bias-correction", action="store_true", help="RP4 amended recipe")
    ap.add_argument("--limit"); ap.add_argument("--eval-limit")
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(ROOT, "results", OUT_BY_PAIR[a.pair])
    extra = ["--out", a.out, "--pair", a.pair]
    if a.bias_correction:
        extra += ["--bias-correction"]
    for flag in ("model", "epochs", "lr", "limit", "eval_limit"):
        v = getattr(a, flag)
        if v is not None:
            extra += [f"--{flag.replace('_', '-')}", v]
    os.makedirs(a.out, exist_ok=True)
    progress = open(os.path.join(a.out, "progress.log"), "a")
    seeds = [int(s) for s in a.seeds.split(",")]
    all_jobs = list(jobs(seeds, a.pair))
    for k, job in enumerate(all_jobs, 1):
        name = name_of(job, a.pair)
        if os.path.exists(os.path.join(a.out, name, "logits.npz")):
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, "train.py"] + job + extra, cwd=EMOCIL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{k}/{len(all_jobs)}] {name} {status} {time.time() - t0:.0f}s"
        print(line, flush=True); progress.write(line + "\n"); progress.flush()
        if r.returncode != 0:
            progress.write(r.stderr[-3000:] + "\n"); progress.flush()
    subprocess.run([sys.executable, "-m", "_pilot.analysis.pilot", "--runs", a.out, "--pair", a.pair], cwd=EMOCIL)


if __name__ == "__main__":
    main()
