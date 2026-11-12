# emocil — continual learning across emotion label inventories that are not equivalent

This folder holds the work on research problems **RP1** and **RP2**
(`../research_framework.md`).

- **RP1.** A multi-label text emotion model is updated sequentially with new
  datasets whose label inventories differ. It must predict the union of all
  inventories without knowing which dataset a text comes from. Three conditions
  hold at every update:
  - labels of the other inventories are unannotated (absent, not negative);
  - same-named labels are not equivalent across datasets;
  - old and new labels correspond many-to-many, not as a hierarchy.
- **RP2.** The same question inside one dataset: GoEmotions at Ekman level,
  then at fine level. Scope differs and domain does not, so RP2 is the control
  for RP1.

Both problems were confirmed by pilots. The solution hypotheses are HS1 and HS2
in `../research_framework.md`. The literature behind them is in
`../literacy2.md` §14.

Like `../bertgo` and `../emogo`, this folder is self-contained: its data are in
`data/`, and it imports nothing from the other folders.

## Status

| Component | State |
|---|---|
| Pilots: step 0 and RP1–RP4 | closed; designs, amendments and results in `_pilot/README.md` |
| HS2 experiment (RP2, where the true correspondence is known) | design and rule fixed 5 Nov 2026 (see HS2 experiment); code ready, to run on the cloud |
| HS1 experiment (RP1) | after HS2 |

Pilot verdicts: RP1 and RP2 **confirmed**; RP3 and RP4 **collapsed**. Two
results from the closed pilots constrain the HS experiments:
- **RP3 sets a required baseline.** A frozen encoder with ridge heads (bert and
  Qwen2.5-1.5B) removes amplification by construction. In the RP1 and RP2
  settings it is below fine-tuned LwF with separate heads, but close on XED.
- **RP4 marks the boundary.** Across languages with the same label scheme
  (BRIGHTER English–Hindi), merging cost almost nothing. The cost appears when
  the *inventories* differ.

## Layout

```
data/goemotions/   train/dev/test.tsv, emotions.txt   copied from ../bertgo/data
                   ekman_mapping.json                 copied from ../emogo/data
data/xed/          en-annotated.tsv, neu_en.txt, README_XED.md
data/brighter/     RP4 pilot only
nets/net.py        BERT + one head over a label layout (expand_to, frozen_copy)
nets/bert.py       ../bertgo/model.py, copied unchanged (used by step 0)
utils/data.py      datasets, XED split, GoEmotions halves, label spaces, masks
utils/optimization.py   ../emogo/utils/optimization.py (BERT's Adam; optional bias correction)
utils/fast_adam.py      the same optimizer as one compiled kernel (GPU)
train.py           one run: stage1 / stage2 (ft, lwf, er, lwf_all, map) / joint; writes to results/
hs/run.py          HS experiment: all training runs of a pair, resumable
hs/frozen.py       frozen-encoder baselines (RP3 configurations) in the run format
hs/analyze.py      local analysis: both inventories, readouts, the decision rule
hs/cloud_run.sh    one-shot cloud script; hs/requirements.txt
_pilot/            closed pilots: scripts, analyses, results, README
```

**Carried over from the pilots.**
- Stage-1 weights, 3 seeds each:
  - `_pilot/results/rp1/stage1_{xed,go}_s{0,1,2}/model.pt`;
  - `_pilot/results/rp2/stage1_{ekman,fine}_s{0,1,2}/model.pt`.

  A new stage-2 method can start from these instead of retraining stage 1.
- Every run's dev and test logits, under `_pilot/results/rp1/` and
  `_pilot/results/rp2/`.
- Frozen features for the RP3 baseline, under `_pilot/results/rp3/features_*.npz`.
- Metrics and the amplification rule are in `_pilot/analysis/pilot.py`.

## Training recipe (as in the RP1 and RP2 pilots)

- **Model:** bert-base-cased, batch 16, lr 5e-5.
- **Optimizer:** the original BERT Adam (no bias correction), 10% warmup and
  polynomial decay, gradient clip 1.0.
- **Training:** 4 epochs per stage, fp32, LwF λ 3, ER 20 rows per label.
- **Losses** are masked: a row is supervised only on its own dataset's columns.
- **Speed changes** that do not change the arithmetic: dynamic padding, and a
  compiled optimizer kernel on GPU.

Details are in the `train.py` docstring and in `_pilot/README.md`, under
"Training pilot".

    cd emocil
    ../.venv/bin/python train.py --mode stage1 --first xed --seed 0     # results/stage1_xed_s0/

`train.py --mode stage2` loads the stage-1 weights from
`<out>/stage1_<first>_s<seed>/model.pt`. To reuse the pilot's weights, link them
into a new `--out` folder. Do not pass `--out _pilot/results/...`, because that
overwrites the pilot's runs.

## Data

**GoEmotions** (Demszky et al., 2020): 27 emotions + neutral, official
train/dev/test 43,410 / 5,426 / 5,427. The Hugging Face card lists apache-2.0;
Öhman & Koufakou (2026, App. B) state CC BY 4.0. The .tsv files use CSV quoting
(a text containing quotes is wrapped in `"..."` with inner quotes doubled), so
they must be read with quote handling, as `../bertgo/data.py` does with pandas.
Splitting on tabs corrupts 147 of the 5,427 test texts.

**XED, English** (Öhman et al., 2020), from
https://github.com/Helsinki-NLP/XED, `AnnotatedData/`, CC-BY 4.0. Movie
subtitles; 8 Plutchik emotions (`1`..`8` = anger, anticipation, disgust, fear,
joy, sadness, surprise, trust) plus neutral lines in a separate file.

- `en-annotated.tsv`: 17,528 lines, `text<TAB>labels`, no duplicates. 1.28
  labels per line (one 77.9%, two 17.3%, three 4.1%).
- `neu_en.txt`: 9,674 lines, 8,777 unique. Each line is `9<TAB>text`; XED's
  README says neutral is `0`, the file uses `9`.
- No text contains a quote character, so splitting on tabs is exact.
- 4 texts occur in both files and are dropped; the step-0 set has 26,304 texts.
- No official split. Öhman & Koufakou (2026) used an 80/10/10 split by SHA-256
  of normalised text; their exact normalisation is not yet checked. The split
  used here (`utils/data.py`, same idea) is 21,119 / 2,606 / 2,579.

Label names shared with GoEmotions: anger, disgust, fear, joy, sadness,
surprise, neutral. Only in XED: anticipation, trust.

**GoEmotions halves and Ekman level** (RP2). `utils/data.py` splits GoEmotions
train into two disjoint halves by hash (`go_half`). One half is annotated at
Ekman level through `ekman_mapping.json` (Demszky et al.'s grouping, copied from
`../emogo/data`); the other keeps the 27 fine labels + neutral.

**BRIGHTER** (Muhammad et al., 2025; CC-BY-4.0) in `data/brighter/` was used
only by the RP4 pilot. See `data/brighter/README_BRIGHTER.md`.

## HS2 experiment (design fixed 5 Nov 2026, before any run)

**Question** (HS2 in `../research_framework.md`). In `ekman2fine` the old
inventory is Ekman and the new one is fine. Can the old inventory be read
through the new outputs, by a correspondence learned from the old model,
instead of protecting its own columns? Success means both inventories stay
near their native models.

**Setting.** As the RP2 pilot:
- pair `ekfine`: GoEmotions train halves, Ekman labels from the official
  mapping;
- both orders;
- the pilot recipe: bert-base-cased, 4 epochs per stage, lr 5e-5, BERT's
  Adam, batch 16.

Seed 0 first. More seeds use the same command, and the rule uses every
complete seed. Each run also writes logits on the train splits
(`--save-train-logits`).

**Methods and baselines.** Per order there are four stage-2 runs; the
stage-1 and joint runs are shared.

| name | what | role |
|---|---|---|
| `ft_separate` | ft, separate heads, old inventory from its own columns | baseline; input of `map_readout` |
| `lwf_separate` | LwF, separate heads | baseline: the strongest old-inventory protection in the pilots |
| `lwf_all_merge` | merged space, LwF distils every old column, merged ones included | baseline: the obvious fix |
| `map` | ft + λ·BCE(M(z_new), teacher), with a linear map M from new logits to old labels trained jointly; old inventory = M(z_new) | **HS2 method** |
| `map_readout` | ridge from `ft_separate`'s new logits to the stage-1 model's logits, fitted on the new training texts | HS2 without any training change |
| `map_refit` | the same ridge on the `map` run | secondary |
| `oracle_or` | Ekman = max over its official fine members, from `ft_separate` | reference: known correspondence (`ekman2fine` only) |
| `frozen_bert`, `frozen_qwen` | frozen encoder + ridge heads, RP3's configurations | baselines (the RP3 paradigm) |
| `joint_separate`, native stage-1 models | | references |

**Fixed now.**
- λ = 3 for `lwf`, `lwf_all` and `map`, as in the pilots.
- `map`: M is linear with a bias. Its weights start at zero, and its bias at
  the teacher's mean logit on the first 2,048 new training rows. It has its
  own Adam, constant lr 1e-3. The distillation mask covers all old labels on
  every new row.
- Ridge readouts: inputs standardised, alpha 1.0, fitted on the new
  **training** texts (the pilot's post-hoc check used dev).
- Frozen baselines: bert-base-cased layer 1, λ 1000; Qwen2.5-1.5B layer 12,
  λ 10. These were chosen on GoEmotions dev in RP3 and are not re-tuned.

**Metrics.** Each method is scored on both inventories, reading the old one
the way the method reads it.
- *old*: AUROC over the six shared emotions on the old inventory's test labels
  (primary, as in the pilots);
- *new*: macro-F1 over all new labels, one threshold per label tuned on the new
  inventory's dev split (primary);
- also AUROC over all labels, and old-inventory F1.

**Decision rule** (order `ekman2fine`, every complete seed). Method X *beats*
baseline Y when both hold:
- X is not more than 0.01 below Y on either primary metric;
- X is at least 0.01 above Y on one of them.

| decision | condition |
|---|---|
| **supported** | `map` beats `ft_separate`, `lwf_separate` and `lwf_all_merge`, and neither frozen baseline beats `map` |
| **partial (readout only)** | `map_readout` meets the same condition and `map` does not |
| **not supported** | neither |

The decision is marked provisional if a frozen baseline is missing.

**Secondary predictions** (from HS2; reported, not in the rule):
- **P1:** `map_readout` stays within 0.01 of both native models.
- **P2:** in `fine2ekman`, `map` does not beat `lwf_separate`, so an old head is
  still needed there (HS1's equivalence gate).
- **P3:** `map` raises the old inventory above `map_readout` without losing more
  than 0.01 new F1 against `ft_separate`.
- **Correspondence check:** the share of each Ekman label's k largest map
  weights that fall on its official members, with k the group size.

**Checks before the experiment** (smoke tests, not results). The pipeline ran
end to end with bert-tiny. Old methods give bit-identical outputs before and
after the `train.py` change (CPU, ft/lwf/er).

With bert-base on 4,000 training rows for 2 epochs, measured on dev only:
- M learns (largest weight .30);
- Ekman dev AUROC: native .857, `ft` own columns .563, `map` readout .807;
- the learned M does not yet follow the official groups (disgust and fear 0).

With bert-tiny, M hardly moves, because the fine logits carry little signal.
Nothing in the design was changed after these checks.

**Known limits.**
- One seed at first.
- λ and M's lr are not tuned.
- A jointly learned M can carry the teacher in directions that are not
  semantic, which is what the correspondence check is for.
- HS1's equivalence gate and per-inventory intercepts are not part of this
  experiment.

### Running on the cloud

1. **Pack** (on the laptop, from `EmoGrowth/`; about 10 MB, with no pilot
   results or weights):

        tar czf emocil_hs.tgz --exclude='emocil/_pilot/results' --exclude='emocil/results' \
            --exclude='__pycache__' emocil

2. **Run** (on the cloud):

        tar xzf emocil_hs.tgz && cd emocil
        pip install torch --index-url <the wheel index matching the machine's CUDA>
        pip install -r hs/requirements.txt
        bash hs/cloud_run.sh            # seed 0: 11 training runs, then the frozen baselines

   - The script downloads bert-base-cased and Qwen2.5-1.5B from Hugging Face.
   - It is resumable: run it again after an interruption.
   - Any GPU with ≥ 8 GB works.
   - **RTX 50-series** (Blackwell, compute capability 12.0) needs a PyTorch
     build for CUDA ≥ 12.8.
   - **T4** (Turing, 7.5) has no native bfloat16, so `hs/frozen.py` runs Qwen in
     float32 there. Its `meta.json` records the dtype. Float16 is not used,
     because it overflowed on Qwen in RP3.
   - If `torch.compile` fails, run `python -m hs.run --pair ekfine --seeds 0
     --optimizer reference`, then `python -m hs.frozen --pair ekfine`.

3. **Download** `emocil/hs_ekfine_results.tgz`: logits, logs, `corr.npz` and
   `env_*.json`, without weights. Unpack it on the laptop:

        cd emocil && mkdir -p results && tar xzf hs_ekfine_results.tgz -C results

4. **Analyse locally** (CPU, a minute):

        cd emocil && ../.venv/bin/python -m hs.analyze --pair ekfine
        # results/hs_ekfine/analysis/: scores.csv, summary.csv, correspondence.csv, verdict.json

