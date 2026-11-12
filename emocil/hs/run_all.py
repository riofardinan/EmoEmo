"""The whole HS2 experiment on one machine (Linux or Windows), in four steps:
check the GPU, the training runs (hs/run.py), the frozen baselines
(hs/frozen.py), the analysis (hs/analyze.py). Step-by-step guide: hs/README.md.

    cd emocil
    python -m hs.run_all --smoke        # setup test, a few minutes (bert-tiny, tiny data) -> results/hs_smoke/
    python -m hs.run_all                # the experiment, seed 0 -> results/hs_ekfine/
    python -m hs.run_all --seeds 1,2    # more seeds later; finished runs are skipped
    python -m hs.run_all --pack         # also write hs_ekfine_results.tgz, to analyse on another machine

Every step is resumable: after an interruption, run the same command again.
"""

import argparse
import importlib.util
import os
import platform
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # emocil/
PAIR = "ekfine"
SMOKE = {"model": "prajjwal1/bert-tiny", "limit": "300", "eval_limit": "200", "epochs": "1"}


def check_gpu(allow_cpu):
    import torch
    print(f"python {platform.python_version()} | torch {torch.__version__} | CUDA build {torch.version.cuda}")
    if not torch.cuda.is_available():
        msg = "No GPU visible to PyTorch. Training on CPU would take days."
        if not allow_cpu:
            raise SystemExit(msg + " Install a CUDA build of PyTorch (hs/README.md, step 2).")
        print("WARNING:", msg); return
    name, cap = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
    print(f"GPU: {name}, compute capability {cap[0]}.{cap[1]}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    try:
        x = torch.randn(64, 64, device="cuda"); (x @ x).sum().item()
    except RuntimeError as e:
        raise SystemExit(f"PyTorch cannot run on this GPU ({e}).\n"
                         "RTX 50-series (compute capability 12.0) needs a PyTorch build for CUDA 12.8 or "
                         "newer, see hs/README.md, step 2.")
    triton = importlib.util.find_spec("triton") is not None
    print("optimizer:", "fused (torch.compile)" if triton else
          "reference (no triton here; slower, same update rule)")
    print("Qwen features in:", "bfloat16" if cap[0] >= 8 else "float32 (no native bfloat16 on this GPU)")


def step(title, args):
    print(f"\n=== {title} ===", flush=True)
    r = subprocess.run([sys.executable, "-m"] + args, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {' '.join(args)} (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--smoke", action="store_true", help="setup test: bert-tiny, 300 rows, 1 epoch")
    ap.add_argument("--pack", action="store_true", help="write a .tgz of the results without weights")
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", "hs_smoke" if a.smoke else f"hs_{PAIR}")

    print("=== 1/4 check ===")
    check_gpu(a.allow_cpu or a.smoke)

    run_args = ["hs.run", "--pair", PAIR, "--seeds", a.seeds, "--out", out]
    frozen_args = ["hs.frozen", "--pair", PAIR, "--out", out]
    if a.smoke:
        run_args += ["--model", SMOKE["model"], "--limit", SMOKE["limit"],
                     "--eval-limit", SMOKE["eval_limit"], "--epochs", SMOKE["epochs"]]
        frozen_args += ["--encoders", "bert", "--bert-model", SMOKE["model"],
                        "--limit", SMOKE["limit"], "--eval-limit", SMOKE["eval_limit"]]
    step("2/4 training runs (hs/run.py; progress in progress.log)", run_args)
    log = os.path.join(out, "progress.log")
    if os.path.exists(log) and "FAILED" in open(log).read():
        raise SystemExit(f"some training runs failed; the error text is in {log}. "
                         "Fix it and run the same command again (finished runs are skipped).")
    step("3/4 frozen baselines (hs/frozen.py)", frozen_args)
    step("4/4 analysis (hs/analyze.py)", ["hs.analyze", "--pair", PAIR, "--runs", out])
    if a.smoke:
        print("\nSMOKE TEST PASSED. Its numbers mean nothing (tiny model, tiny data). "
              "Now run: python -m hs.run_all")
    else:
        print(f"\nDone. Results: {os.path.join(out, 'analysis')} (verdict.json, summary.csv)")
    if a.pack:
        tgz = os.path.join(ROOT, f"{os.path.basename(out)}_results.tgz")
        with tarfile.open(tgz, "w:gz") as t:
            t.add(out, arcname=os.path.basename(out), filter=lambda ti: None if ti.name.endswith(".pt") else ti)
        print(f"Packed without weights: {tgz} ({os.path.getsize(tgz) / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
