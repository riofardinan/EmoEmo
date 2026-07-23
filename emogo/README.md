# emogo — EmoGrowth for the text modality

Multi-label class-incremental emotion decoding on **GoEmotions**, following
**Fu et al. (2025), _EmoGrowth: Incremental Multi-label Emotion Decoding with
Augmented Emotional Relation Graph_** (ICML).

EmoGrowth evaluates on video, audio and fMRI. This adds text. The
optimisation configuration comes from `../bertgo`, the verified replication of
the GoEmotions BERT baseline, so any gap against that baseline is attributable
to the incremental setting rather than to a mistuned backbone.

## Status

| Component | State |
|---|---|
| Task protocols, data manager, metrics, trainer | done |
| `finetune`, `lwf` | done |
| `ewc`, `replay` (ER/RS/PRS/OCDM), `agcn`, `krt-r`, `aesl` | not yet implemented |

`utils/factory.py` raises a clear error for a method that is not registered
yet, so nothing fails silently.

### Reference numbers to aim at

GoEmotions has 28 classes, so the paper's **Audio28** table (Table 3) is the
direct analogue. Its B0-I7 column:

| Method | Avg. Acc mAP | Last maF1 | Last miF1 | Last mAP |
|---|---|---|---|---|
| Finetune | 36.4 | 9.2 | 14.8 | 27.3 |
| LwF | 46.6 | 37.9 | 51.7 | 40.6 |
| AESL (theirs) | 49.0 | 38.4 | 51.8 | 42.7 |
| Upper-bound | – | 51.4 | 61.1 | 57.1 |

Do not expect these exact values — different modality, different backbone, and
here the backbone is fine-tuned rather than frozen. What should carry over is
the **ordering and the size of the gap**: LwF must beat Finetune by a wide
margin, and both must sit below an upper bound. `../bertgo` (all 28 classes
trained jointly) is this project's upper bound.

## Run

This folder is self-contained: the GoEmotions splits and the NRC-VAD lexicon
ship in `data/`, and all paths resolve relative to the source files rather than
the working directory.

```bash
pip install -r requirements.txt
python check_config.py                    # assert BERT settings match ../bertgo
python main.py --config exps/finetune_B0-I7.json
python main.py --config exps/lwf_B0-I7.json
./run_all.sh lwf                          # all four protocols
```

On a GPU server, install torch to match the driver before running — see the
note in `../bertgo/README.md`. The default PyPI wheel targets CUDA 13 and falls
back to CPU on older drivers.

Any config field can be overridden:

```bash
python main.py --config exps/finetune_B0-I7.json --seed 1994 --device cuda:1
python main.py --config exps/finetune_B0-I7.json --protocol B16-I3
```

Results per run go to `results/<method>/<protocol>/seed<seed>/`:
`summary.json` (curves plus Avg./Last accuracy), `results.csv` (metrics × tasks,
laid out like EmoGrowth's own csv), `run.log`, and the raw
`task<t>_logits.npy` / `task<t>_labels.npy` for post-hoc analysis at any
threshold.

Quick wiring check on CPU, no GPU needed:

```bash
python main.py --config exps/finetune_B0-I7.json \
  --model_name prajjwal1/bert-tiny --device cpu \
  --init_epochs 1 --epochs 1 --fp16 false --num_workers 0 \
  --output_dir /tmp/smoke
```

## Design decisions

### Protocols

GoEmotions has 27 emotions + `neutral` = **28 classes**, exactly the size of the
paper's Audio28 dataset, so its four protocols transfer unchanged:

| Protocol | Tasks |
|---|---|
| B0-I7 | 7,7,7,7 |
| B0-I4 | 4×7 |
| B16-I3 | 16,3,3,3,3 |
| B16-I2 | 16,2,2,2,2,2,2 |

Setting `"drop_neutral": true` gives 27 classes and unlocks the Video27/Brain27
protocols (B0-I9, B0-I3, B15-I3, B15-I2).

### Task construction

`utils/data_manager.py` reproduces the structure of EmoGrowth's pre-baked
`label_session` .mat files:

* A training task exposes **only its own classes**. At task *b* the label
  matrix has \|C^b\| columns, so earlier labels are invisible
  (**past-missing**) and later ones do not exist yet (**future-missing**).
* **A sample may recur across tasks.** A comment labelled {joy, gratitude}
  appears in the task owning `joy` and again in the task owning `gratitude`,
  each time carrying only the locally visible label — the Figure 1 scenario.
* **Test is cumulative**: after task *b*, every test comment carrying at least
  one class seen so far, scored over all seen classes with full labels.

Class arrival order defaults to the order in `emotions.txt`
(`"class_order": "file"`). `"shuffled"` permutes it with the seed, the
PyCIL/EmoGrowth convention. Whichever you pick, it is recorded in
`summary.json` — comparisons across methods are only meaningful at a fixed
order, so keep it constant within a table.

### Backbone

BERT is fine-tuned **end-to-end** on every task, with the pooled `[CLS]` vector
as the instance representation.

This is a deliberate departure from EmoGrowth, and worth stating in any
write-up: the original never trains its backbone — `sub_data` is a matrix of
precomputed frozen features (1000-d ResNet18 for Video27, voxels for Brain27),
and only the GIN + FDModel head is optimised. Fine-tuning is the stronger and
more standard choice for text, but it means the backbone also drifts between
tasks, so forgetting here has a source that the original setup does not have.
A frozen-feature variant would be the closer ablation, and is easy to add:
freeze `nets/incremental_net.BertBackbone`.

### Affective dimension (AESL Teacher-2)

AESL distils relations from a second teacher living in valence-arousal space.
Video27 and Audio28 ship human VAD ratings per stimulus; GoEmotions does not.
`utils/vad.py` derives them from the comment text with the NRC-VAD lexicon
(mean over matched tokens).

Deriving VAD from the gold emotion labels would be simpler and would look
better, but it leaks: the future-missing setting assumes no access to unseen
classes, and label-derived VAD reintroduces them. Text-derived VAD stays
label-independent. This is the one place where the text setting cannot mirror
the original exactly, and it should be reported as a limitation.

### Metrics

`utils/metrics.py` is a port of EmoGrowth's, so numbers are comparable with its
tables: mAP (ranking-based, threshold-free) plus hamming loss, instance-average
precision, one error, ranking loss, coverage, macro-F1 and micro-F1. The
threshold-based metrics binarise at **logit > 0** (probability > 0.5), as in the
original — note this differs from the 0.3 threshold `../bertgo` uses for the
paper's Table 4. Raw logits are saved per task so both conventions can be
computed after the fact.

Reported as **Avg. Acc** (mean across tasks) and **Last Acc** (after the final
task), matching the paper's columns.

### Methods

**Finetune** — no anti-forgetting mechanism. The detail that makes it the real
multi-label baseline rather than a strawman: at task *b* the targets for
already-seen classes are set to **zero** and the loss covers the whole widened
head (`fake_target_gen` in EmoGrowth's `finetune_ml.py`). Every new sample
therefore actively asserts "none of the old emotions are present", which is
false and is what destroys them.

**LwF** — differs from Finetune in exactly one place:

```
Finetune   L(logits[:, :],       [0…0 | y])
LwF        L(logits[:, known:],  y)  +  lamda * L(logits[:, :known], sigmoid(old_logits))
```

The old columns are never shown ground truth at all. They are supervised by the
frozen previous model's own sigmoid outputs, so old knowledge is preserved by
self-distillation instead of contradicted. `lamda = 3`, the value in both
config blocks of EmoGrowth's `lwf_ml.py`.

Both use `MultiLabelSoftMarginLoss`, as EmoGrowth does — verified numerically
identical to the `BCEWithLogitsLoss(reduction="mean")` that `../bertgo` uses,
so the two folders optimise the same objective.

### Hyperparameters

Every BERT setting is copied from `../bertgo`: `bert-base-cased` (cased),
max length 50, batch 16, LR 5e-5, warmup 10%, BERT's
`AdamWeightDecayOptimizer` (no bias correction — see `utils/optimization.py`),
grad clip 1.0, fp32, 4 epochs per task.

`check_config.py` asserts this match and exits non-zero if the two drift apart;
`run_all.sh` calls it before every sweep. It also prints the four differences
that are intentional: per-task epochs, seed (1993 here, EmoGrowth's convention,
vs 42 in bertgo), the metric threshold, and the absence of a dev split.

EmoGrowth uses a longer first task than subsequent ones (40 vs 30 epochs on
iScience). Here `init_epochs` and `epochs` both default to 4, because the
GoEmotions paper found more than 4 epochs overfits BERT on this data. Each task
gets its own warmup+decay cycle, i.e. every task is a fresh fine-tuning run.

## Adding the remaining methods

Each subclasses `models/finetune.Finetune` and overrides one or two hooks:

* `_compute_loss(out, targets, batch)` — LwF adds distillation on old logits;
  EWC adds the Fisher penalty; AESL adds the label-embedding loss, logit
  distillation and the two relation-based KD terms.
* `_train_dataset_kwargs()` — replay methods return
  `{"appendent": (indices, labels)}`; AESL returns `{"affective": True}`.
* `build_rehearsal_memory(data_manager)` — buffer construction for
  ER / RS / PRS / OCDM.

Then register the class in `utils/factory._REGISTRY`. AESL additionally needs
the CLIF network (GIN + FDModel + grouped `Conv1d` head) ported from
`../EmoGrowth/convs/CLIFModel.py` and `../EmoGrowth/convs/layers.py`, plus the
`sym_conditional_prob` / `sym_conditional_prob_update` graph routines from
`../EmoGrowth/models/base.py`.
