"""HS2 experiment driver: every training run of one dataset pair, resumable.

The design and the decision rule are fixed in ../README.md, "HS2 experiment";
read that first. Per seed, this runs 11 jobs in order:

  stage1   the first and the second dataset (native models, and the stage-2 starts)
  stage2   both orders, each with
             separate ft       no protection (also the input of the post-hoc readout)
             separate lwf      the strongest old-inventory protection in the pilots
             merge    lwf_all  LwF that also distils merged outputs (the obvious fix)
             separate map      HS2: distillation through a learned correspondence
  joint    separate label space (upper reference)

Every run also writes logits on the train splits (`--save-train-logits`),
because the readout baseline fits its map on training texts. A run is skipped
when its logits.npz exists. A failed run is logged in progress.log and the rest
continue. The frozen-encoder baselines are a separate command (hs/frozen.py).

    cd emocil
    python -m hs.run --pair ekfine --seeds 0                 # results/hs_ekfine/
    python -m hs.run --pair ekfine --seeds 1,2               # more seeds later, same folder
    python -m hs.run --pair ekfine --model prajjwal1/bert-tiny --limit 300 \\
        --eval-limit 200 --epochs 1 --out /tmp/hs_smoke      # smoke test
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # emocil/
PAIRS = {"xedgo": ("xed", "go"), "ekfine": ("ekman", "fine")}
STAGE2 = [("separate", "ft"), ("separate", "lwf"), ("merge", "lwf_all"), ("separate", "map")]


def jobs(seeds, pair):
    for s in seeds:
        for first in PAIRS[pair]:
            yield ["--mode", "stage1", "--first", first, "--seed", str(s)]
        for first in PAIRS[pair]:
            for space, method in STAGE2:
                yield ["--mode", "stage2", "--first", first, "--space", space,
                       "--method", method, "--seed", str(s)]
        yield ["--mode", "joint", "--space", "separate", "--seed", str(s)]


def name_of(job, pair):
    d = dict(zip(job[::2], job[1::2]))
    if d["--mode"] == "stage1":
        return f"stage1_{d['--first']}_s{d['--seed']}"
    if d["--mode"] == "joint":
        return f"joint_{d['--space']}_s{d['--seed']}"
    a, b = PAIRS[pair]
    other = b if d["--first"] == a else a
    return f"stage2_{d['--first']}2{other}_{d['--space']}_{d['--method']}_s{d['--seed']}"


def write_env(out):
    """Library versions and GPU, so the local analysis knows what produced the runs."""
    import numpy, sklearn, torch, transformers
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "transformers": transformers.__version__, "numpy": numpy.__version__,
           "scikit-learn": sklearn.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out, f"env_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(env, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=list(PAIRS), default="ekfine")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default=None, help="default results/hs_<pair>")
    ap.add_argument("--model"); ap.add_argument("--epochs"); ap.add_argument("--optimizer")
    ap.add_argument("--limit"); ap.add_argument("--eval-limit")
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "results", f"hs_{a.pair}")
    extra = ["--out", out, "--pair", a.pair, "--save-train-logits"]
    for flag in ("model", "epochs", "optimizer", "limit", "eval_limit"):
        v = getattr(a, flag)
        if v is not None:
            extra += [f"--{flag.replace('_', '-')}", v]
    os.makedirs(out, exist_ok=True)
    write_env(out)
    progress = open(os.path.join(out, "progress.log"), "a")
    all_jobs = list(jobs([int(s) for s in a.seeds.split(",")], a.pair))
    t_all = time.time()
    for k, job in enumerate(all_jobs, 1):
        name = name_of(job, a.pair)
        if os.path.exists(os.path.join(out, name, "logits.npz")):
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, "train.py"] + job + extra, cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{k}/{len(all_jobs)}] {name} {status} {time.time() - t0:.0f}s"
        print(line, flush=True); progress.write(line + "\n"); progress.flush()
        if r.returncode != 0:
            progress.write(r.stderr[-3000:] + "\n"); progress.flush()
    print(f"all jobs handled in {time.time() - t_all:.0f}s; results in {out}")


if __name__ == "__main__":
    main()
