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

### Comparing against ../bertgo

`../bertgo` and this folder **deliberately report different things**, so their
output files are not comparable side by side:

| | reports | threshold | model |
|---|---|---|---|
| `../bertgo/output/run1/test_table4.txt` | per-emotion P/R/F1 | prob > 0.3 | one model, all 28 classes at once |
| `results/*/summary.json` | mAP / maF1 / miF1 per task | prob > 0.5 | incremental |

The first is the GoEmotions paper's format (Table 4) and exists to prove the
replication. The second is EmoGrowth's format (Tables 1–3). They are not
supposed to look alike.

The relationship is that **bertgo is the Upper-bound row** of the EmoGrowth
table — same backbone, same data, no incremental constraint. The comparison is
valid because after the final task the incremental test set is exactly bertgo's
test set: same 5,427 comments, same 28 columns.

```bash
python compare.py                    # builds the table, bertgo as Upper-bound
python compare.py --protocol B0-I4
python compare.py --per-emotion      # bertgo's Table 4 format, for every method
```

`--per-emotion` exists because `utils/metrics.py` reports **no precision or
recall at all** — EmoGrowth does not publish them, so there is no
`macro-average P R F1` line anywhere in `results/`. The flag reads the saved
final-task logits and scores them exactly the way `../bertgo/metrics.py` does,
putting the incremental methods and the upper bound in one table. Use
`--threshold 0.5` to see the same breakdown at EmoGrowth's operating point.

It rescores bertgo's saved predictions through `utils/metrics.py`, so every
number comes from one implementation, and it warns if a final-task test set
ever stops matching bertgo's.

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

### Class order — read this before comparing runs

The default is `"alphabetical"`, which is the paper's protocol. Appendix B.1:
*"In the process of splitting emotion labels for incremental learning, we just
follow the order of the alphabet without interfere."*

**Alphabetical is not the same as the order in `emotions.txt`.** GoEmotions
appends `neutral` as the 28th line, after the 27 alphabetical emotions, so
under `"file"` order `neutral` sits at index 27 instead of its alphabetical
index 20 (between `nervousness` and `optimism`). That matters a great deal:
`neutral` carries **28% of all positive labels**, and putting it in the *final*
task inflates last-task scores for any method that forgets — Finetune's
micro-F1 reads 0.34 under `"file"` order but 0.10 with `neutral` excluded.
Under alphabetical order `neutral` falls in a middle task in all four
protocols, and the artefact disappears.

`"shuffled"` permutes with the seed. The paper does not shuffle, but its
Limitations section calls out class order as unexplored, so this is available
for a robustness check.

The order actually used is recorded in `summary.json`, and `compare.py` warns
when a result was produced under a non-alphabetical order. Comparisons are only
meaningful at a fixed order — keep it constant within a table.

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

### How faithful is this to EmoGrowth?

Verified identical to `EmoGrowth/models/*_ml.py`: the Finetune and LwF loss
functions (transcribed and diffed, 0.0), `lamda = 3`, the KRT instance-splitting
protocol, the cumulative test construction, the metrics, and the protocol
splits.

Deliberately different, and worth stating in any write-up:

| | EmoGrowth (Audio28, App. B.4) | here |
|---|---|---|
| Backbone | frozen precomputed features | BERT fine-tuned end-to-end |
| Optimiser | Adam, β₂ = 0.9999 | BertAdam, β₂ = 0.999 |
| LR / weight decay | 1e-3 / 0 | 5e-5 / 0.01 |
| Schedule | none | warmup + linear decay |
| Epochs | 45 / 40 | 4 / 4 |
| Batch | 128 | 16 |

The optimisation column follows `../bertgo` rather than EmoGrowth on purpose:
these are BERT hyperparameters, and EmoGrowth's LR of 1e-3 would destroy a
pretrained transformer. The frozen-vs-fine-tuned backbone is the deeper
difference — it is why Finetune's mAP here (14.5) falls further than the
paper's (27.3) even though the F1 figures line up (10.6 vs 9.2). Fine-tuning
lets the *features* drift, not just the classifier, so ranking ability is lost
too. A frozen-backbone variant would isolate this.

The datasets also differ in character, which limits how far absolute numbers
travel: Audio28 has 5.27 labels per instance (density 0.19), GoEmotions has
1.17 (density 0.042) — nearly single-label.

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
