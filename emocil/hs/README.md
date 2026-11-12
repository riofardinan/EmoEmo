# HS2 experiment: how to run it

What the experiment tests, its baselines and its decision rule are in
`../README.md`, section "HS2 experiment". That section was fixed before any
run; do not edit it.

This folder uses only `../train.py`, `../utils/`, `../nets/` and `../data/`.
Nothing in `../_pilot/` (the closed pilots) is used.

## What happens, step by step

1. **Data.** GoEmotions train is split into two halves by text hash.
   - Half A is labelled at Ekman level: 7 labels, from the official mapping.
   - Half B keeps the 28 fine labels.
   - Dev and test are the official GoEmotions splits, labelled at both levels.
2. **Stage 1** (2 runs). bert-base-cased is trained once on A (Ekman) and once
   on B (fine). These are the native models and the starting points of stage 2.
3. **Stage 2** (8 runs). Starting from one stage-1 model, train on the other
   half with four methods. Both orders are run: Ekman → fine (primary) and
   fine → Ekman.
4. **Joint** (1 run). Both halves together, as the upper reference.
5. **Frozen baselines** (2). bert-base-cased and Qwen2.5-1.5B are frozen, with
   ridge heads on top.
6. **Analysis.** Every method is scored on both inventories: the old one
   (AUROC) and the new one (macro-F1). The decision goes to `verdict.json`.

| run (seed 0) | what |
|---|---|
| `stage1_ekman_s0`, `stage1_fine_s0` | native models |
| `stage2_ekman2fine_separate_ft_s0` | finetune, no protection of the old labels |
| `stage2_ekman2fine_separate_lwf_s0` | LwF, a separate head per inventory |
| `stage2_ekman2fine_merge_lwf_all_s0` | LwF that also distils merged outputs (the obvious fix) |
| `stage2_ekman2fine_separate_map_s0` | **HS2 method**: distillation through a learned correspondence |
| `stage2_fine2ekman_*_s0` (4 runs) | the same four, in the other order |
| `joint_separate_s0` | joint training |
| `frozen_bert`, `frozen_qwen` | frozen encoder + ridge |

## What you need

- An NVIDIA GPU with at least 8 GB. An RTX 5070 (12 GB) is fine.
- A recent driver. RTX 50-series cards need CUDA 12.8 or newer.
- Python 3.10 or newer (tested with 3.14).
- About 6 GB of free disk: about 3.5 GB of downloaded models, and about 0.9 GB
  of stage-1 weights per seed.
- Internet on the first run, to download bert-base-cased, Qwen2.5-1.5B and,
  for the setup test, bert-tiny.

## Step 1. Get the code

Use either of these:
- **git** (branch `main`), after the commits have been pushed:

      git clone git@github.com:riofardinan/EmoEmo.git      # or `git pull` in an existing clone
      cd EmoEmo

- **a copy** of the `emocil/` folder (code and `data/`, 8 MB). The pilot
  results in `_pilot/results/` are not needed.

## Step 2. Install (once)

Linux:

    cd emocil
    python3 -m venv .venv && source .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install -r hs/requirements.txt

Windows (PowerShell):

    cd emocil
    py -m venv .venv ; .venv\Scripts\Activate.ps1
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install -r hs/requirements.txt

RTX 50-series GPUs need a PyTorch build for CUDA 12.8 or newer (`cu128` above,
or a newer index). Older builds stop with "no kernel image is available".

## Step 3. Setup test (a few minutes)

    python -m hs.run_all --smoke

This checks that PyTorch can use the GPU. It then runs all 11 jobs with
bert-tiny on 300 rows, the frozen baseline and the analysis, and ends with
`SMOKE TEST PASSED`. Its numbers mean nothing. It writes `results/hs_smoke/`,
which can be deleted afterwards.

## Step 4. Run the experiment

    python -m hs.run_all

- **What runs:** seed 0, all 11 training runs, then the frozen baselines, then
  the analysis.
- **Progress:** `results/hs_ekfine/progress.log` gets one line per finished
  run, with its duration.
- **Interrupted?** Run the same command again. Finished runs are skipped.
- **Time:** on the RTX 3050 laptop one run takes about 10–15 minutes, so about
  3 hours for seed 0. The RTX 5070 should be faster; the first lines of
  `progress.log` show the real time per run.

## Step 5. Read the result

Everything is in `results/hs_ekfine/analysis/`:

| file | contents |
|---|---|
| `verdict.json` | `decision`: `supported`, `partial (readout only)` or `not supported`. `provisional`: true if a frozen baseline is missing. |
| `summary.csv` | per order and method: `old_auroc_shared` (old inventory) and `new_f1_all` (new inventory) |
| `correspondence.csv` | whether the map's largest weights fall on each Ekman label's official fine members |
| `scores.csv` | every metric, per seed |

## Step 6 (optional). More seeds

    python -m hs.run_all --seeds 1,2

The analysis uses every complete seed.

## Moving the results to another machine

    python -m hs.run_all --pack        # writes hs_ekfine_results.tgz, without weights

On the other machine, unpack it into `emocil/results/` and run
`python -m hs.analyze --pair ekfine`.

## Troubleshooting

- **The check stops with "no kernel image is available"** (or similar). The
  PyTorch build does not support this GPU. Reinstall torch from the `cu128`
  index (step 2).
- **No triton,** which is usual on Windows. The optimizer switches by itself to
  the reference implementation. It is slower and applies the same update rule.
- **CUDA out of memory.** Close other programs that use the GPU. Do not change
  the batch size (16): it is part of the recipe.
- **A run FAILED.** The error text is at the end of `progress.log`. Fix it, then
  run the same command again.
- **GPUs without bfloat16** (e.g. T4). Qwen features are computed in float32;
  `frozen_qwen/meta.json` records the dtype.

## Files

| file | role |
|---|---|
| `run_all.py` | the four steps above, in one command |
| `run.py` | the 11 training runs (calls `../train.py`), resumable |
| `frozen.py` | the frozen-encoder baselines |
| `analyze.py` | metrics, readouts and the decision rule (CPU only) |
| `requirements.txt` | Python packages besides torch |
