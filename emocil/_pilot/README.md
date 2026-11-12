# emocil/_pilot — pilots of RP1–RP4 (closed)

This folder keeps the problem-checking pilots: step 0, and the training pilots
of RP1, RP2, RP3 and RP4 from `../../research_framework.md`. Each pilot has its
design (fixed before its runs), its amendments and its results, in the order
they were done. The work that continues (RP1 and RP2, toward HS1 and HS2) is in
the parent folder `emocil/`.

| Pilot | Verdict under its pre-registered rule |
|---|---|
| Step 0: label-equivalence diagnosis (inference only) | done 2 Nov 2026 |
| RP1: XED ↔ GoEmotions, sequential vs joint, merged vs separate labels | **confirmed** (48 runs, 3 seeds) |
| RP2: Ekman → fine inside GoEmotions (control for RP1) | **confirmed** (48 runs, 3 seeds) |
| RP3: frozen encoder + ridge vs fine-tuned CL | **collapsed** |
| RP4: English ↔ Hindi (BRIGHTER, mBERT), amended recipe | **collapsed** (seed 0 only) |

**Paths.** Paths in this file are relative to `_pilot/`. `train.py`, `utils/`,
`nets/` and `data/` are in the parent `emocil/`. Every script runs from
`emocil/` as a module, e.g. `../.venv/bin/python -m _pilot.analysis.pilot`.

```
run_pilot.py               48 runs of a pair, resumable, then the analysis
rp3_frozen.py              RP3: frozen features + ridge vs emogo's fine-tuned CL
rp4_calibrate.py           RP4: epoch calibration and the amended-recipe check (dev only)
analysis/pilot.py          metrics and the decision rule (shared by RP1, RP2, RP4)
analysis/pilot_posthoc.py  RP1 post-hoc checks
analysis/explore_map.py    RP1: old inventory read through a learned map (exploratory)
analysis/rp2_posthoc.py    RP2: per-label amplification, drift check, static cost
analysis/rp2_readout.py    RP2: Ekman read through oracle / learned correspondences
analysis/frozen_vs_pilots.py  frozen + ridge in the RP1 and RP2 settings (exploratory)
diagnosis/                 step0.py, step0_robustness.py
results/step0/             step-0 tables and score caches
results/rp1/, results/rp2/ one folder per run (logits.npz, meta.json, train.log;
                           stage-1 runs also model.pt), progress.log, analysis/
results/rp3/               summary.csv, verdict.json, logs, frozen features of
                           bert-base-cased and Qwen2.5-1.5B (GoEmotions, XED)
results/rp4_calib/, results/rp4_enghin/   logits, logs, analysis (no weights)
results/logs/              driver logs
```

**Removed on 5 Nov 2026, when the folder was tidied.**
- The model weights of the RP4 calibration and pilot (20 files).
- RP3's `bert_task0.pt` and `features_bert_task0.npz`.
- `analysis/rp4_posthoc.py`, which was never run.

All of them can be regenerated with the code here, and every table and verdict
was kept. The result folders were renamed (`pilot` → `rp1`, `rp2_pilot` →
`rp2`). Re-running every analysis from the new layout gave byte-identical
tables. `meta.json` files still record the old absolute `--out` paths.

## Step 0: are same-named labels equivalent? (inference only)

The bertgo GoEmotions BERT (`../bertgo/output/run1/model.pt`, test macro-F1
.491 at threshold 0.3) scores every XED text. For each XED label the scores are
compared with XED's gold labels: the AUROC of the same-named output, of the
best single output, and of a 5-fold cross-validated logistic map from all 28
logits. The same map is fitted inside GoEmotions (test split) as a control,
where a label is equivalent to itself. Details are in the docstring of
`diagnosis/step0.py`.

```
cd emocil
../.venv/bin/python -m _pilot.diagnosis.step0              # writes _pilot/results/step0/
../.venv/bin/python -m _pilot.diagnosis.step0_robustness   # post-hoc checks
```

**Decision criteria, fixed before the first run** (six shared emotions only):

- C1, meaning: for at least 2, the map beats the same-named output on XED
  (bootstrap 95% CI above 0) by at least 0.03 AUROC more than it does inside
  GoEmotions.
- C2, inferable correspondence: the map reaches AUROC ≥ 0.70 for anticipation
  and trust, which have no same-named output.
- C3, operating point: at threshold 0.3 the predicted-positive rate is outside
  [0.5, 2] × the XED prevalence for at least 3.
- C1 → continue to the training pilot; only C3 → narrow RP1 to the decision
  layer; neither → the premise fails for this pair.

**Result: C1, C2 and C3 hold for all six emotions → continue to the pilot.**

| XED | same-named | best single output | map − same on XED [95% CI] | map − same in GoEmotions | positive-rate ratio @0.3 |
|---|---|---|---|---|---|
| anger | .706 | annoyance .738 | +.055 [+.047, +.062] | +.0056 | 0.30 |
| disgust | .725 | annoyance .744 | +.045 [+.038, +.052] | −.0062 | 0.04 |
| fear | .666 | nervousness .666 | +.074 [+.064, +.084] | −.0090 | 0.11 |
| joy | .800 | joy .800 | +.040 [+.035, +.046] | −.0025 | 0.13 |
| sadness | .760 | disappointment .780 | +.045 [+.037, +.052] | −.0009 | 0.21 |
| surprise | .710 | surprise .710 | +.045 [+.038, +.053] | −.0097 | 0.11 |

Trust is best matched by approval (.707; map .775), anticipation by desire
(.649; map .749). The full 28 × 9 AUROC matrix is `results/step0/auroc_matrix.csv`.

**Post-hoc robustness** (`diagnosis/step0_robustness.py`, not pre-registered,
added after reading the map weights, some of which are not interpretable):

- The "emotional vs neutral line" shortcut explains almost none of the gain: a
  map from only the same-named and neutral logits gains −.015 to +.010. On
  emotional lines alone the gain grows to +.064–+.116, so it comes from telling
  emotions apart.
- The same-named output plus its GoEmotions Ekman-group siblings recovers
  39–74% of the gain for anger, joy, sadness and surprise, and about 0% for fear
  and disgust, whose gain comes from outputs outside their group (grief,
  confusion, caring). For those two, a difference in meaning cannot yet be told
  apart from adaptation to the subtitle domain.

**What step 0 does not show.** It is static: the core RP1 claim, that
sequential learning makes label non-equivalence worse, needs the training
pilot. It uses one model, one seed, one direction (GoEmotions model → XED
labels) and one dataset pair. The map is fitted with XED labels
(cross-validated), so it shows that the correspondence can be learned from the
old model's outputs plus some new labels, not that it can be found without them.

**Verification.** `nets/bert.py` with the scorer in `diagnosis/step0.py`
reproduces bertgo's own test probabilities (`../bertgo/output/run1/test_probs.npy`)
to a maximum absolute difference of 3.5e-6. The first run of step 0 (then under
`../analysis/`, scoring GoEmotions from bertgo's saved probabilities) gave the
same verdict, the same AUROC matrix and robustness table, and in-domain gains
equal to within 0.0002.

## Training pilot (design fixed 2 Nov 2026, before any full run)

**Question.** In setting (b) — one head over the union of both inventories,
no knowledge of a text's source — does learning the datasets one after the
other make the cost of non-equivalent same-named labels larger than it already
is when both are learned together?

**Datasets.** XED English (8 Plutchik + neutral; 80/10/10 hash split, see
`utils/data.py`: 21,119 / 2,606 / 2,579) and GoEmotions (27 + neutral;
official split). Six emotion names are shared (anger, disgust, fear, joy,
sadness, surprise), plus neutral.

**Label spaces.**
- `separate`: 37 outputs, same-named labels kept apart.
- `merge`: 30 outputs, the seven shared names share one output.

Every row is supervised only on its own dataset's columns. Labels of the other
inventory are masked, never treated as negative.

**Runs** (`run_pilot.py`, 3 seeds: 0, 1, 2; 48 runs):
- stage 1: each dataset alone with its own labels (6 runs). These are also the
  native reference models.
- stage 2: start from the stage-1 model of dataset A, lay its head out on the
  union space, train on the other dataset B with a method (36 runs = 2 orders ×
  2 label spaces × 3 methods × 3 seeds):
  - `ft`: masked BCE on B's columns;
  - `lwf`: ft + 3 × BCE to the stage-1 model's sigmoid outputs, on A's columns
    that B's rows do not annotate (merge: only A-only labels; separate: all of
    A's columns);
  - `er`: ft on B's rows plus up to 20 A rows per A label (emogo's random ER),
    each scored on its own dataset's columns.
- joint: both datasets together over the union space (6 runs = 2 label
  spaces × 3 seeds).

Orders: `xed2go` (coarse → fine) and `go2xed` (fine → coarse). A seed fixes
stage 1, stage 2 and the joint run of that seed, so differences are paired.

**Recipe** (as `../emogo`, `../bertgo`): bert-base-cased, batch 16, lr 5e-5,
the original BERT optimizer, 10% warmup, gradient clip 1.0, 4 epochs per stage
(joint: 4 epochs over both), fp32, max 50 tokens. Two speed changes that leave
the arithmetic unchanged:
- batches are padded to their longest row;
- on GPU the optimizer update runs as one compiled kernel
  (`utils/fast_adam.py`). On a fixed stream of 300 gradients it matches the
  reference to 1.7e-7 relative (float32 rounding). In real training the
  rounding is amplified, as it is by GPU nondeterminism, so runs are not
  bit-identical to the reference optimizer. A per-tensor `_foreach` version was
  bit-identical but no faster: the update is bound by memory bandwidth (~67 of
  ~140 ms per BERT-base step on the RTX 3050).

**Metrics** (`analysis/pilot.py`), for each dataset D on a run's head, on D's
test set, read from the outputs D maps to, macro over the six shared emotions:
- `auroc` (primary, ranking);
- `f1_pooled`: one threshold per output, tuned on the pooled dev rows of every
  dataset that annotates it (source unknown);
- `f1_known`: one threshold per output and dataset, tuned on D's dev (source
  known);
- `f1_fixed`: logit > 0.

**Amplification**, per order (A = first dataset), method m and seed s, on A's
test set:

    amp = [separate − merge]_stage2(m, s) − [separate − merge]_joint(s)

The bracket is the cost of merging same-named labels; the joint bracket is its
static cost, so `amp` is the extra cost added by sequential learning.

**Decision rule.** A metric "holds" for (order, m) when `amp` > 0 in all three
seeds and its mean is ≥ 0.01.
- **confirmed**: on `auroc`, it holds for ≥ 2 of the 3 methods in at least one
  order. RP1's core claim stands; the method work starts.
- **partial**: not confirmed, but in some order ≥ 2 methods hold on `f1_pooled`
  and not on `f1_known`. The problem is the operating point when the source is
  unknown (a decision-layer problem, cf. Gate 0 and K1′); RP1 narrows.
- **collapsed**: neither. Sequential learning adds no cost beyond the static
  one, which *Fear Is Not Always Fear* already shows; RP1 as a continual-
  learning problem is dropped.

Reported as well, without entering the decision: the static merge cost (joint
`separate − merge` on both datasets), forgetting of A (stage-1 native vs
stage 2), and performance on B.

**Expectation from step 0 and *Fear ≠ Fear* (not part of the rule).** Shared
outputs trained on XED and read on GoEmotions lose ranking that thresholds do
not recover (*Fear ≠ Fear*, XED→GoEmotions F1 .171 → .169), so any
amplification is most likely in `go2xed`, where GoEmotions is the old dataset.

**Running.** Estimated 13 h on the RTX 3050; the actual run took 15.8 h (01:18–17:05, 2 Nov 2026). Resumable: a run
with an existing `logits.npz` is skipped, so the same command continues after
an interruption.

    cd emocil && ../.venv/bin/python -m _pilot.run_pilot          # _pilot/results/rp1/
    ../.venv/bin/python -m _pilot.analysis.pilot --runs _pilot/results/rp1   # re-analyse at any time

## Pilot results (2 Nov 2026)

All 48 runs finished without failure. Tables are in `results/rp1/analysis/`:
`runs.csv` (every run × dataset), `amplification.csv` (per seed),
`verdict.json`, and the post-hoc `posthoc_*.csv`.

**Sanity.** The stage-1 models match their references. GoEmotions native
macro-F1 is .497 with dev-tuned thresholds (bertgo: .49 at 0.3), and XED native
is .492.

**Verdict under the pre-registered rule: confirmed.** In `go2xed`, LwF and ER
show amplification in all three seeds with mean ≥ 0.01; in `xed2go`, only LwF.

AUROC amplification on the old dataset's six shared emotions, mean over seeds
[min, max]:

| order | ft | lwf | er |
|---|---|---|---|
| xed2go (old = XED) | −.102 [−.161, −.068] | **+.058** [+.051, +.062] | +.004 [+.002, +.007] |
| go2xed (old = GoEmotions) | −.018 [−.027, −.011] | **+.045** [+.039, +.049] | **+.013** [+.009, +.017] |

For comparison, the static cost of merging, measured in joint training, is
+.011 AUROC on both datasets. For LwF the sequential cost is about five times
that. On decisions the amplification is larger still: F1 with one pooled
threshold per output gives +.17 to +.21 for LwF and +.04 to +.08 for ER.

**Post-hoc checks** (`analysis/pilot_posthoc.py`, not part of the rule):

- *Encoder drift does not explain it.* With merged labels LwF distils fewer
  old columns, so the encoder is less constrained. The old dataset's own
  labels (not merged) do not move:

  | old-dataset AUROC | lwf separate | lwf merge | er separate | er merge |
  |---|---|---|---|---|
  | XED shared emotions | .859 | .789 | .773 | .758 |
  | XED-only (anticipation, trust) | .838 | .835 | .762 | .765 |
  | GoEmotions shared emotions | .935 | .879 | .894 | .871 |
  | GoEmotions-only (21 labels) | .909 | .904 | .869 | .870 |

  The cost sits on the merged outputs, which the new dataset's labels
  re-supervise.
- *Finetune goes the other way.* Unprotected old columns collapse when kept
  separate (XED shared .662, XED-only .645). Merging at least gives the shared
  outputs supervision from the new dataset (.753). So the amplification is a
  property of methods that protect old knowledge, which are the ones that
  matter.
- *Nothing is gained on the new dataset.* Merging leaves the new dataset's
  AUROC unchanged. The one exception is LwF in `xed2go`, where merging gives
  GoEmotions .940 vs .914, because distilling nine XED columns constrains the
  encoder (a stability–plasticity trade). F1 with pooled thresholds is lower
  with merging in every method and both orders, e.g. ER on GoEmotions .477 vs
  .544: one threshold cannot serve two prevalences.

**What the pilot does not show.**

- *Meaning vs domain.* In stage 2 the merged outputs are supervised only on
  the new dataset's texts. Old-only columns survive the same domain change
  under LwF and ER, which argues against domain shift alone. But no
  same-domain pair with different inventories is available to rule it out.
- *The LwF variant.* In the merged space, LwF does not distil the merged
  outputs, because they carry new labels. A variant that also distils them
  (old soft targets alongside the new labels) is the obvious first method
  candidate and must be a baseline.
- *Scope.* One dataset pair, one backbone, three seeds. The fused optimizer
  and the XED split differ from Öhman & Koufakou (2026), see above.
- *The pre-run expectation was half right.* Amplification was expected mostly
  in `go2xed`; it is stronger there for ER, but LwF shows it in both orders.

## RP2 pilot: Ekman → fine inside GoEmotions (design fixed 4 Nov 2026, before any full run)

RP2 (`../research_framework.md`) is the same question as RP1 inside one corpus.
The text domain is identical, and the difference in scope between same-named
labels is known exactly.

**Question.** When two inventories come from the same texts and differ only in
how much each same-named label covers, does sequential learning still make
merging those labels cost more than it does in joint training?

**Data** (`utils/data.py`, pair `ekfine`). GoEmotions train is halved by
SHA-256 of the normalised text, with no overlap:
- `ekman` (21,858 texts) is labelled at the Ekman level: anger, disgust, fear,
  joy, sadness, surprise and neutral. Each Ekman label is the OR of its fine
  members in GoEmotions' official grouping.
- `fine` (21,552 texts) keeps the 28 GoEmotions labels.
- Both are evaluated on the full GoEmotions dev and test sets, at their own
  level. The test texts are therefore identical for the two inventories.

**Label spaces.** `separate` has 35 outputs; `merge` has 28, because every
Ekman name also exists among the fine labels. Scope of each Ekman label in fine
labels: disgust 1 and neutral 1 (identical), fear 2, anger 3, surprise 4,
sadness 5, joy 12.

**Runs, recipe, metrics, amplification.** As in the RP1 pilot (48 runs, 3
seeds, ft/lwf/er, sequential vs joint, merge vs separate). The orders are
`ekman2fine` (coarse → fine) and `fine2ekman` (fine → coarse). The primary
metric is AUROC over the six shared emotions on the old inventory's test labels.

**Decision rule** (same as RP1). A metric holds for (order, method) when `amp`
> 0 in all three seeds with mean ≥ 0.01.
- **confirmed**: on AUROC, ≥ 2 of 3 methods hold in at least one order. A known
  difference in label scope is enough to produce the amplification, without any
  change of domain. For RP1 this means the effect does not need a domain shift.
- **partial**: only the pooled-threshold F1 shows it. The cost is a decision
  -layer problem.
- **collapsed**: neither. Scope differences alone do not amplify in sequential
  learning. RP1's effect then needs something RP2 lacks, such as a domain shift
  or a difference in annotation.

**Predictions** (recorded now, not part of the rule):
- per-label amplification is about zero for disgust, whose scope is identical
  to its fine namesake;
- it grows with scope, and is largest for joy.

**Running.**

    cd emocil && ../.venv/bin/python -m _pilot.run_pilot --pair ekfine   # _pilot/results/rp2/

## RP3 pilot: frozen encoder vs fine-tuned continual learning (design fixed 4 Nov 2026, before any full run)

**Question** (RP3 in `../research_framework.md`). A frozen pretrained encoder
with a closed-form ridge head cannot forget, and its recursive solution equals
joint ridge (L3A, ALMA). Is that enough for fine-grained multi-label emotion?

**Comparison.** The frozen paradigm's *best case* is a ridge head on all 28
GoEmotions labels over frozen features. It is compared with the fine-tuned
continual-learning methods of `../emogo`, using their 8-seed last-task mAP on
the same GoEmotions test set and the same metric (per-class AP averaged,
`emogo/utils/metrics.py`):

- B0-I7: LwF 38.08 ± 0.92 (best), AGCN 36.93, AESL 33.23;
- B16-I2: LwF 28.06.

The fine-tuned joint reference, from the RP1 pilot's stage-1 GoEmotions models,
is 49.0 mAP.

**Frozen encoders** (`rp3_frozen.py`):
- `bert`: bert-base-cased as pretrained, the emogo backbone. Features are all
  13 layers mean-pooled, plus the [CLS] pooler.
- `bert_task0`: bert-base-cased fine-tuned on task 0 of B0-I7 (7 classes, emogo
  recipe), then frozen. This is the RidgeFT / SEQ* recipe.
- `qwen`: Qwen/Qwen2.5-1.5B (base, Apache-2.0), every 4th layer and the last,
  mean-pooled. This is the counter-premise: frozen LLM layers are strong for
  emotion (Fang et al., 2026).

For each encoder, the feature set and the ridge λ ∈ {1, 10, 100, 1000, 10000}
are chosen by dev mAP. Test is read once per encoder.

**Decision rule.** Threshold = LwF's B0-I7 mAP − 1.0 (about one seed SD) =
37.08.
- **proven** (a frozen encoder is not enough): the best case of all three
  encoders stays below 37.08.
- **encoder-dependent**: both BERT encoders stay below 37.08, but Qwen reaches
  it. Whether freezing suffices depends on the encoder.
- **collapsed**: `bert` or `bert_task0` reaches 37.08. The frozen paradigm
  already matches fine-tuned CL on this protocol.

Reported as well, without entering the decision:
- test macro-AUROC;
- macro-F1 with dev-tuned thresholds;
- the protocol-faithful recursive ridge for B0-I7 and B16-I2 (absent labels set
  to 0, no pseudo-labels), which shows how much MLCIL's partial labels cost the
  frozen paradigm.

**Running** (laptop GPU, after the RP2 pilot):

    cd emocil
    ../.venv/bin/python -m _pilot.rp3_frozen finetune-task0
    ../.venv/bin/python -m _pilot.rp3_frozen extract --encoder bert        # also bert_task0, qwen
    ../.venv/bin/python -m _pilot.rp3_frozen evaluate                      # _pilot/results/rp3/

## RP4 pilot: languages one after another (design fixed 4 Nov 2026, before any run)

**Question** (RP4 in `../research_framework.md`). A multi-label emotion model is
extended from one language to the next, and every language is annotated with
the same emotion scheme. Does sequential learning make sharing same-named
labels across languages cost more than it does when the languages are learned
together?

**Data.** BRIGHTER (Muhammad et al., 2025; CC-BY-4.0), converted to
`data/brighter/*.tsv`, with unannotated cells left empty. Labels are anger,
disgust, fear, joy, sadness and surprise.
- BRIGHTER English leaves disgust unannotated in every split, so English has a
  five-label inventory, and disgust is masked for English rows.
- Train / dev / test sizes: English 2,764 / 230 / 5,528; Hindi 2,556 / 200 /
  2,020; Sundanese 924 / 398 / 1,852.
- Indonesian and Javanese have no training split and cannot be a stage.

**Pairs.** The primary pair `enghin` (English ↔ Hindi) carries the decision.
The secondary pair `engsun` (English ↔ Sundanese, the only Indonesian-region
language with a training split) is reported, but does not enter the decision.
The shared emotions are anger, fear, joy, sadness and surprise.

**Design.** As in the RP1 pilot: merge (one output per emotion name, 6 outputs)
vs separate (a head per language, 11 outputs); ft / lwf / er; sequential vs
joint; both orders; 3 seeds; 48 runs per pair. The encoder is mBERT
(`bert-base-multilingual-cased`), so the code path is the BERT one. The rest of
the recipe is as in RP1.

**Epoch calibration**, before any pilot run. BRIGHTER training sets are about
1–3 k rows, a tenth of XED or GoEmotions. The number of epochs per stage is
chosen from {4, 10} by the stage-1 models' dev macro-AUROC, averaged over both
languages of the pair. A calibration seed (99) that the pilot does not use is
used for this. Test is never read for the choice.

**Decision rule.** As for RP1, on the primary pair. Primary metric: AUROC over
the five shared emotions on the old language's test set.
- **confirmed**: in at least one order, ≥ 2 of the 3 methods show amp > 0 in all
  seeds, with mean ≥ 0.01.
- **partial**: only the pooled-threshold F1 shows it.
- **collapsed**: neither.

**Known limit.** BRIGHTER languages also differ in text source. So like RP1,
this pilot tests the practical cost of sharing labels across languages; it
cannot separate culture from domain.

**Amendment to the recipe** (4 Nov 2026, after the calibration, before any pilot
run; test not read). Calibration with the pre-registered recipe (BERT's Adam
without bias correction, lr 5e-5), seed 99, gave
(`results/rp4_calib/calibration.json`):

| stage-1 dev macro-AUROC | 4 epochs | 10 epochs |
|---|---|---|
| English | .783 | .801 |
| Hindi | .504 | .647 |
| Sundanese | .569 | .559 |

The Hindi and Sundanese numbers are not real scores. All four of those models
collapsed:
- their dev logits are constant (std per label ≤ .001);
- their training loss stays flat at about 0.41, the entropy of the label prior,
  from epoch 2 on.

The AUROC above .5 is numerical noise. A pilot built on such models cannot
measure forgetting of the old language.

This is the failed-run pattern of Mosbach et al. (2021, ICLR), §7.7:
"throughout the training, their training loss stays close to that at
initialization". Their §6 guideline for small datasets is: "Use small learning
rates with bias correction to avoid vanishing gradients early in training" and
"Increase the number of iterations considerably and train to (almost) zero
training loss". Their baseline is Adam with bias correction, lr 2e-5 and 20
epochs, with linear warmup over 10% of steps.

The amendment adopts that recipe whole, with no new search. The remaining
hyperparameters stay as in RP1: batch 16, weight decay 0.01, gradient clip
1.0, and BERT's polynomial (linear) decay schedule. `--bias-correction` is new
in `../train.py`. Its default is off, so RP1–RP3 are unchanged. It was checked in
float64 against `torch.optim.AdamW`, to within 1e-11 relative.

The amended recipe is accepted for the pilot if, on dev with seeds 99 and 98
(`python -m _pilot.rp4_calibrate --amended`, `results/rp4_calib/amended/`):
1. no stage-1 model collapses (max per-label dev logit std ≥ .01), and
2. the mean dev macro-AUROC per pair beats the pre-registered recipe's best.

If it fails, RP4 is reported as untestable with mBERT at this data size, not as
collapsed.

**Result of the check: accepted** (`results/rp4_calib/amended/calibration_amended.json`).
No model collapsed, and training loss falls to .010–.015 for English and Hindi
and to .12 for Sundanese.

| stage-1 dev macro-AUROC | pre-registered recipe, best of 4/10 epochs (s99) | amended, s99 | amended, s98 |
|---|---|---|---|
| English | .801 | .798 | .806 |
| Hindi | .647 (collapsed) | .880 | .881 |
| Sundanese | .569 (collapsed) | .699 | .707 |
| mean, enghin | .724 | .839 | .844 |
| mean, engsun | .680 | .748 | .756 |

**Second amendment: one seed** (4 Nov 2026, Rio's decision; no pilot result had
been read). With the amended recipe a run takes 10–12 min, and a joint run about
20 min, so 48 runs need about 10 h. The first pilot uses seed 0 only (16 runs).
The rule is unchanged in form: `holds` needs amp > 0 in every complete seed and
a mean ≥ 0.01, which for one seed is amp ≥ 0.01. This is weaker evidence. In
RP1 the LwF amplification varied by about ±.006 across seeds. If seed 0 shows
the effect, seeds 1–2 can be added with the same command; their stage-1 models
already exist.

    cd emocil && ../.venv/bin/python -m _pilot.run_pilot --pair enghin \
        --model google-bert/bert-base-multilingual-cased --bias-correction --lr 2e-5 --epochs 20 --seeds 0   # _pilot/results/rp4_enghin/

## RP2 pilot results (4 Nov 2026)

All 48 runs finished without failure in 11.5 h. Tables are in
`results/rp2/analysis/`.

**Verdict under the pre-registered rule: confirmed.** In `ekman2fine`, LwF and ER
show amplification in all seeds; in `fine2ekman`, only LwF.

AUROC amplification on the old inventory's six shared emotions, mean over seeds
[min, max]:

| order | ft | lwf | er |
|---|---|---|---|
| ekman2fine (old = Ekman) | −.066 | **+.076** [+.072, +.080] | **+.017** [+.013, +.022] |
| fine2ekman (old = fine) | −.068 | **+.019** [+.018, +.022] | −.006 |

The static merge cost is nearly zero: +.002 AUROC on Ekman and +.008 on fine.
So with the same texts and a known difference in scope, sequential learning
alone creates the cost. For RP1 this means the amplification does not need a
change of domain. It does not show that domain plays no part there.

**Design note.** In `ekman2fine`, merging gives every Ekman label to a fine
output that the new rows annotate, so LwF has no column left to distil and is
identical to ft. Merging removes LwF's protection entirely; that is the
mechanism of the cost. A LwF variant that also distils merged outputs remains
the obvious baseline (HS1).

**Predictions recorded before the run.**
- *Disgust about zero.* Partly right. Per-label LwF amplification in
  `ekman2fine` is +.027 for disgust and +.011 for fear, the two smallest scopes,
  against +.116 for anger, +.124 for surprise, +.060 for sadness and +.116 for
  joy. In `fine2ekman` disgust is +.000.
- *Growing with scope.* Not monotone: anger (3) and surprise (4) are as large as
  joy (12).

**Encoder drift does not explain it.** In `fine2ekman` the 21 fine labels with
no Ekman namesake are unaffected by merging (LwF .898 separate vs .900 merge),
while the shared labels drop (.936 vs .909). This is the same pattern as RP1.

**Exploratory, post-hoc** (`analysis/rp2_readout.py`): read the Ekman labels
through a correspondence from the model's fine outputs, in `ekman2fine`.

| | Ekman (old) AUROC, shared | fine (new) F1, dev thresholds |
|---|---|---|
| ft separate, own Ekman columns | .781 | .465 |
| ft separate, through the official OR of members | .896 | .465 |
| ft separate, through a ridge map learned on dev texts from the old model's outputs | **.906** | .465 |
| lwf separate, own Ekman columns | .923 | .387 |
| native models | .907 | .459 |

The learned map puts its largest weights on the official members: anger 67%,
disgust 100%, fear 83%, joy 75%, sadness 80%, surprise 75%, as the share of the
top-k weights, with k the group size. Reading the old inventory through it
keeps both inventories at native level. LwF keeps the old one higher but costs
the new one 0.07 F1.

## RP3 pilot results (4 Nov 2026)

`results/rp3/` holds summary.csv, verdict.json and run logs. The feature files
are about 1.2–2.6 GB each.

**Fix before any result was read.** The first evaluation failed: Qwen's hidden
states carry activations of about 7e3, and one test text overflowed float16. The
Qwen features were re-extracted with a bfloat16 forward pass and float32 storage
(`rp3_frozen.py`). BERT features were unaffected (max |x| ≈ 16). The failed
evaluation had written no results.

**Verdict under the pre-registered rule: collapsed.** Every frozen encoder
reaches the threshold of 37.08 mAP:

| frozen encoder + ridge (chosen on dev) | test mAP | protocol ridge B0-I7 | protocol ridge B16-I2 | macro-AUROC |
|---|---|---|---|---|
| bert-base-cased, layer 1, λ 1000 | 38.15 | 37.96 | 38.00 | .878 |
| bert, fine-tuned on B0-I7 task 0 then frozen, layer 10, λ 100 | **42.40** | 42.43 | 42.41 | .892 |
| Qwen2.5-1.5B, layer 12, λ 10 | 39.91 | 39.99 | 40.05 | .899 |
| emogo LwF, fine-tuned CL (8 seeds) | 38.08 | — | 28.06 (B16-I2) | |
| fine-tuned joint (RP1 stage-1 GoEmotions) | 49.0 | | | |

- A frozen encoder with a closed-form head matches or beats the best fine-tuned
  CL method on GoEmotions MLCIL.
- On B16-I2 it beats it by about 10–14 mAP.
- The protocol-faithful ridge, with absent labels set to 0, loses almost
  nothing to MLCIL's partial labels.
- The best frozen encoder is still 6.6 mAP below fine-tuned joint training
  (42.4 vs 49.0). Representation adaptation is worth something; the fine-tuned
  CL methods simply lose more to forgetting than they gain from it. This is an
  observation, not a tested problem.

**Exploratory, post-hoc** (`analysis/frozen_vs_pilots.py`). Frozen encoder plus
ridge heads in the RP1 and RP2 settings, mean shared-emotion AUROC per dataset.
A ridge head is joint-equivalent, so sequential and joint are the same and
there is no amplification:

| setting | bert sep | bert merge | qwen sep | qwen merge | pilot: lwf separate (sequential) | pilot: native |
|---|---|---|---|---|---|---|
| RP1 XED | .802 | .772 | .857 | .851 | **.859** | .839 |
| RP1 GoEmotions | .897 | .883 | .920 | .905 | **.935** | .938 |
| RP2 Ekman | .865 | .862 | .876 | .876 | **.923** | .907 |
| RP2 fine | .891 | .880 | .908 | .902 | | .924 |

In the two-stage cross-inventory settings, fine-tuned sequential learning with
LwF and separate heads still beats the frozen paradigm, narrowly on XED. With a
frozen encoder, merging same-named labels still has its static cost (bert,
XED: −.030). The frozen paradigm wins on many-task MLCIL with partial labels
(RP3) and loses on two-stage cross-inventory learning (RP1, RP2). Any method for
HS1/HS2 must be compared against it.

## RP4 pilot results (4 Nov 2026, seed 0 only)

All 16 seed-0 runs finished without failure (`results/rp4_enghin/analysis/`).
`analysis/pilot.py` now writes in `seeds_complete` only the seeds that have
every run of the comparison. Before, it listed every seed with any run. The
RP1 and RP2 verdicts are unchanged.

**Verdict under the pre-registered rule: collapsed.** On AUROC only LwF shows
amplification, in both orders. "Partial" also fails, because the F1 effect is
not limited to pooled thresholds: LwF and ER show it with per-dataset
thresholds too.

AUROC over the old language's five shared emotions, seed 0:

| order | ft | lwf | er | static cost (joint sep − merge) |
|---|---|---|---|---|
| b_eng2b_hin (old = English) | −.016 | **+.077** | −.005 | +.005 |
| b_hin2b_eng (old = Hindi) | −.039 | **+.016** | −.009 | +.004 |

**Reading.**
- *Sharing an output across English and Hindi costs almost nothing.* Joint
  training with merged outputs loses only .004–.005 AUROC. ER keeps
  old-language rows on the shared outputs and shows no amplification. With
  BRIGHTER English–Hindi and mBERT, there is no sign of same-named labels
  conflicting across the two languages, which was RP4's premise.
- *LwF's amplification is a method artefact here.* In `b_eng2b_hin` every
  English label merges into an output that Hindi rows annotate. LwF then has
  nothing to distil and equals ft: seq-merge is .7133 for both. LwF-separate
  keeps its protection (.795). This is the same design consequence as RP2's
  `ekman2fine`, without RP2's ER effect.
- *Limits.* One seed, one language pair, five shared labels, one encoder. ER's
  values are below zero, not near the .01 threshold, so more seeds are
  unlikely to change the verdict. Sundanese (`engsun`) was not run.
