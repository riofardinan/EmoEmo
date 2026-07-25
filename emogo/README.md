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
| `finetune`, `ewc`, `lwf`, `er`, `rs`, `ocdm`, `prs`, `agcn`, `aesl` | done |
| `krt-r` | not implemented: no reference code exists |

`utils/factory.py` raises a clear error for a method that is not registered
yet, so nothing fails silently.

**KRT-R is deliberately absent.** EmoGrowth's repo reproduces eight methods and
KRT-R is not among them — it appears in the paper's tables only as a cited
comparison (Dong et al., 2023). Implementing it would mean working from that
paper with no reference implementation, and with no way to verify it the way
every other method here was verified (loss functions diffed against the
original to 0.0). Report it as not replicated.

**AGCN and PRS** were first contributed with bugs (AGCN dropped the `ya + yb`
residual and had an asymmetric graph; PRS's `exp(-N)` underflowed and froze its
buffer). Both were rewritten from the EmoGrowth source and verified: AGCN's
graph branch and `ya+yb` match the original to 1e-8, PRS's sample-in weight
matches to 1e-19 while staying finite at GoEmotions' class counts. See the
method notes below.

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

That row is the *only* thing `compare.py` needs from bertgo, and it is
optional: pass `--upper-bound path/to/test_probs.npy`, or omit it and the row
is left blank. Everything else comes from this folder's own `data/`.

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
| EWC | 37.9 | 8.3 | 14.3 | 29.3 |
| LwF | 46.6 | 37.9 | 51.7 | 40.6 |
| ER | 44.7 | 8.1 | 14.4 | 38.0 |
| RS | 43.7 | 8.1 | 12.3 | 36.5 |
| OCDM | 44.5 | 8.7 | 12.0 | 38.4 |
| PRS | 43.3 | 10.8 | 13.5 | 35.5 |
| AGCN | 47.3 | 35.3 | 50.9 | 41.9 |
| AESL (theirs) | 49.0 | 38.4 | 51.8 | 42.7 |
| Upper-bound | – | 51.4 | 61.1 | 57.1 |

Note the shape of the replay rows: ER and RS reach a respectable **mAP** (38.0,
36.5 — second only to LwF) while their **maF1/miF1 stay down near Finetune's**.
Ranking survives, thresholded prediction does not. If your runs reproduce that
split, the implementation is behaving.

`compare.py` carries these for all four protocols and prints them beside your
own numbers.

Do not expect these exact values — different modality, different backbone, and
here the backbone is fine-tuned rather than frozen. What should carry over is
the **ordering and the size of the gap**: LwF must beat Finetune by a wide
margin, and both must sit below an upper bound. `../bertgo` (all 28 classes
trained jointly) is this project's upper bound.

## Run

This folder is self-contained: the GoEmotions splits, the NRC-VAD lexicon and
EmoBank all ship in `data/`, and every path resolves relative to the source
files rather than the working directory.

```bash
pip install -r requirements.txt
python check_config.py                    # assert BERT settings match ../bertgo
python main.py --config exps/lwf_B0-I7.json     # one run
./run_all.sh                              # finetune + ewc + lwf, 4 protocols each
./run_all.sh lwf                          # one method, 4 protocols
./run_all.sh "finetune lwf" 1994          # subset, different seed
```

`exps/` holds one config per (method, protocol). `run_all.sh` checks each
config against the frozen reference first, then prints the comparison tables
when the sweep finishes.

**AESL needs its affective vectors built once first:**

```bash
python precompute_vad.py --source lexicon    # NRC-VAD, seconds
python precompute_vad.py --source emobank    # BERT regressor, ~5 min on a GPU
./run_all.sh aesl            # lexicon arm
./run_all.sh aesl-emobank    # EmoBank arm
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

**EWC** — keeps the *parameters* near where they were, rather than the outputs.
After each task it estimates the diagonal Fisher information and penalises
movement in the weights that mattered:

```
loss = loss_clf + 1000 * Σ_n fisher[n] · (θ[n] − θ_old[n])² / 2
```

`loss_clf` is Finetune's, zeros and all — EWC does nothing about the
past-missing partial labels, it only slows the drift. The paper's verdict
(§4.2): *"EWC is not suitable for direct application to MLCIL task due to its
poor performance."* Expect it near Finetune, not LwF. `lamda = 1000`,
`fishermax = 1e-4`, both from `ewc_ml.py`.

Costs ~880 MB of extra device memory when fine-tuning BERT, since `fisher` and
`mean` each hold a full copy of the parameters.

**LwF** — differs from Finetune in exactly one place:

```
Finetune   L(logits[:, :],       [0…0 | y])
LwF        L(logits[:, known:],  y)  +  lamda * L(logits[:, :known], sigmoid(old_logits))
```

The old columns are never shown ground truth at all. They are supervised by the
frozen previous model's own sigmoid outputs, so old knowledge is preserved by
self-distillation instead of contradicted. `lamda = 3`, the value in both
config blocks of EmoGrowth's `lwf_ml.py`.

**ER** and **RS** — replay. Both keep a buffer of past samples and mix it into
every later task; they differ only in how it is filled:

* `er` — per class of the current task, keep up to `memory_per_class` (20)
  samples carrying it, chosen uniformly, then union.
* `rs` — reservoir sampling over the stream of all training samples, capped at
  `memory_size` (500). Every sample has the same survival probability, so
  frequent emotions dominate the buffer. That imbalance is exactly what PRS and
  OCDM were designed to fix.

Replay attacks the past-missing problem from a different angle than LwF: the
buffered rows carry their *real* labels for old classes, so the model sees
genuine positive evidence instead of Finetune's zeros. What it does not fix is
that the *current* task's rows still say zero for those classes — the paper's
explanation (§4.2) for why replay disappoints on maF1/miF1: *"just saving the
labels of current task aggravates the partial label problem in subsequent
training."*

The buffer stores row indices plus each sample's global class indices, so the
text is never copied. Buffer construction is skipped after the final task,
matching the original.

**OCDM** — a buffer method again, but one that solves for balance directly:
after each task it searches for the subset of `memory_size` samples whose class
distribution has the lowest KL divergence from uniform. Note the repo's version
is **not** the greedy algorithm of Liang & Li (2022) — EmoGrowth's authors
commented that out and replaced it with a 10,000-trial random search, and since
Table 3 came from the random-search version, that is what is reproduced.
Vectorised here, because the original's Python loop is unusable at GoEmotions'
scale.

**PRS** — Partitioning Reservoir Sampling. Like RS but it steers the buffer
toward class balance: each sample's chance of entering is weighted toward the
rarer of its classes, and eviction targets the most over-represented class.
The rewrite fixes an underflow that the original form hits only at GoEmotions'
scale — the sample-in weight `exp(-N)/Σexp(-N)` is exactly a softmax over the
sample's classes, so it is computed as one (shift by min), matching the
original to 1e-19 without collapsing to 0/0 when class counts reach the
thousands. `rou = 0`, so the target partition is uniform — the point of PRS.

**AGCN** — Augmented Graph Convolutional Network. Is this an "image" method?
No: the graph it convolves over is the emotion label co-occurrence graph, not
an image-region graph, and it is exactly the augmented ERG that AESL and the
paper's title are about. AGCN's image-ness lives only in its CNN backbone,
which — like every method here — is replaced by fine-tuned BERT. The classifier
is a residual `ya + yb`: a plain linear term plus a graph-propagated term over
the label adjacency. Loss is LwF's, read through the graph
(`clf(new) + 1·kd(old)`). The rewrite restores the `ya + yb` residual the first
contribution dropped and normalises the adjacency once rather than inside the
per-sample loop; the branch matches the original to 1e-8.

**AESL** — the paper's own method. Four loss terms (Eq. 15): classification,
emotional-semantics learning over the augmented ERG, logit distillation from
the old model, and relation-based KD against two teachers — the old model's
features and an affective-space representation.

Three things about AESL in the text setting deserve to be stated plainly in a
write-up:

* **The affective teacher is a proxy.** Video27/Brain27 supply 14 human-rated
  appraisal dimensions per stimulus and Audio28 supplies 11; GoEmotions
  supplies none. `precompute_vad.py` builds a 3-dimensional substitute two
  ways, and both are worth running — `--source lexicon` (mean NRC-VAD over
  matched tokens) and `--source emobank` (a BERT regressor trained on EmoBank's
  10k human-rated sentences). Three dimensions give a coarser similarity matrix
  than eleven: on random data the off-diagonal cosine std is 0.58 at 3 dims
  against 0.30 at 11. Neither source touches the gold labels, which matters —
  a label-derived affective signal would leak future classes.
* **RKD is defined over batch statistics, and our batch is 16, not 128.** That
  is 240 off-diagonal RSM entries per step instead of 16,256. AESL is the only
  method here affected: every other loss is per-sample. Batch 16 comes from the
  GoEmotions replication and is kept so the comparison across methods stays
  fair.
* **`atanh` diverges on identical features.** 0.7% of GoEmotions training rows
  are exact duplicates ("Thank you.", "[NAME]"), which produce cosine exactly 1
  off the diagonal. The original's guard — zeroing non-finite entries — is kept
  and tested.

The ERG itself (`utils/graph.py`) was verified against `base.py` to 0.0 and
6e-8. Two details a re-implementation loses easily: the diagonal is zeroed
before symmetrising, and the matrix is symmetrised **last**, so no row
normalisation can break symmetry.

All methods use `MultiLabelSoftMarginLoss`, as EmoGrowth does — verified
numerically identical to the `BCEWithLogitsLoss(reduction="mean")` that
`../bertgo` uses, so the two folders optimise the same objective.

### Hyperparameters

Every BERT setting is copied from `../bertgo`: `bert-base-cased` (cased),
max length 50, batch 16, LR 5e-5, warmup 10%, BERT's
`AdamWeightDecayOptimizer` (no bias correction — see `utils/optimization.py`),
grad clip 1.0, fp32, 4 epochs per task.

**The two folders share no code and no config.** `utils/reference.py` holds a
frozen copy of the verified values as literals, with the provenance of each, so
this folder runs standalone — on a server, in a container, with no `../bertgo`
anywhere. `check_config.py` asserts the run matches that reference and exits
non-zero otherwise; `run_all.sh` calls it before every sweep, so a drifted
config stops the run instead of quietly producing incomparable numbers. It also
prints the four differences that are intentional: per-task epochs, seed (1993
here, EmoGrowth's convention, vs 42 in bertgo), the metric threshold, and the
absence of a dev split.

If a bertgo checkout is at hand, `check_config.py --against ../bertgo`
additionally confirms the frozen snapshot has not fallen behind it.

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
