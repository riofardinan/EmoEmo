# bertgo — GoEmotions BERT baseline, replicated

A PyTorch replication of the baseline in **Demszky et al. (2020), _GoEmotions: A
Dataset of Fine-Grained Emotions_** (ACL). Target: **macro-F1 ≈ .46** on the
full 28-label taxonomy (Table 4).

The original implementation
(`../google-research/goemotions/bert_classifier.py`) is TensorFlow 1.x and
Python 3.7, which will not run on a current machine. This is a port that keeps
every hyperparameter, the exact optimizer, and the exact evaluation protocol —
not a re-interpretation. Its purpose is to establish a trusted configuration
that `../emogo` then reuses for the continual-learning experiments.

## Run

This folder is self-contained: the GoEmotions splits ship in `data/`, and all
paths resolve relative to the source files, not the working directory. Copy the
folder anywhere and it runs.

```bash
pip install -r requirements.txt
python train.py --output_dir ./output/run1
```

**On a GPU server, install torch to match the driver first.** The default
PyPI wheel is built against CUDA 13 and will fail on an older driver with
*"The NVIDIA driver on your system is too old"*, then silently fall back to CPU.
Check with `nvidia-smi`; for a driver reporting CUDA 12.3, use:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Do not start a run until that prints `True`.

Roughly 10.8k optimizer steps; about 25–40 min on a single modern GPU.
Add `--fp16 true` to halve that. For a quick wiring check:

```bash
python train.py --max_train_samples 256 --num_train_epochs 1.0 --output_dir /tmp/smoke
```

Outputs land in `--output_dir`: `test_table4.txt` (Table 4 as printed in the
paper), `test_results.json` (all per-emotion metrics), `test_probs.npy`, and
`model.pt`.

## Configuration provenance

Every value is traced to its source in `config.py`. The ones that matter:

| Setting | Value | Source |
|---|---|---|
| Backbone | `bert-base-cased` | README: "we use the cased base model"; flag `do_lower_case=False` |
| Max sequence length | 50 | flag `max_seq_length` |
| Batch size | 16 | flag `train_batch_size`; §5.3 "a small batch size of 16" |
| Learning rate | 5e-5 | flag `learning_rate`; §5.3 |
| Epochs | 4 | flag `num_train_epochs`; §5.3 "at least 4 epochs [...] more results in overfitting" |
| Warmup | 10% | flag `warmup_proportion` |
| Head | pooled → dropout 0.1 → dense(28) | `create_model()`, `keep_prob=0.9` |
| Loss | sigmoid cross-entropy | `create_model()`, `multilabel=True` |
| Eval threshold | 0.3 | flag `eval_prob_threshold`; `calculate_metrics.py` |
| Labels | 28 | `emotions.txt` already contains `neutral`, so `add_neutral=False` |

Data is the rater-agreement-filtered split shipped with the original repo:
43,410 train / 5,426 dev / 5,427 test. The loader asserts these sizes.

### Three details that commonly break replications

1. **The optimizer is not `torch.optim.AdamW`.** BERT's `AdamWeightDecayOptimizer`
   applies **no bias correction** and places epsilon **outside** the square root
   (`sqrt(v) + eps`, with eps = 1e-6). `optimization.py` reimplements it.
2. **The learning-rate schedule decays from step 0**, not from the end of
   warmup — `polynomial_decay(power=1.0)` is computed over the whole run and
   the warmup value overrides it. This differs from HuggingFace's
   `get_linear_schedule_with_warmup`.
3. **`cased`, not `uncased`.** The corpus is full of emphatic capitalisation
   that carries emotional signal.

## Verification against the original

Each row below was checked by importing and running Google's own code (with
TensorFlow and absl stubbed out) and diffing its output against this port —
not by reading the source and judging it equivalent.

| What | Method | Result |
|---|---|---|
| Tokenization | Ran the original `bert/tokenization.py` `FullTokenizer` + `convert_single_example` over all 43,410 training texts and diffed `input_ids`, `attention_mask`, `token_type_ids` against this port | **0 mismatches** |
| Label construction | Transcribed `DataProcessor._create_examples` and diffed the multi-hot matrices and texts for train/dev/test | **identical, all 3 splits** |
| Evaluation metrics | Ran the original `calculate_metrics.py` on synthetic predictions and diffed all 122 output keys against `metrics.py` | **max abs diff 0.0** |
| Optimizer | Transcribed `AdamWeightDecayOptimizer.apply_gradients` to NumPy, ran 25 steps with and without weight decay | **max abs diff 0.0** |
| LR schedule | Compared against `polynomial_decay(power=1.0)` + warmup at every one of the 10,852 steps | **max abs diff 0.0** |
| Loss | Compared against the `sigmoid_cross_entropy_with_logits` + `reduce_mean` formula | **diff 0.0** |
| Step counts | `int(43410/16*4.0)` = 10,852, warmup 1,085; this loop runs 4 × 2,713 | **10,852, matches** |
| Weight-decay exclusion | Applied the exclusion regexes to real BERT parameter names | 124 tensors excluded, all LayerNorm/bias; embeddings decayed, as in TF |
| Head init | `trunc_normal_(std=0.02, a=-0.04, b=0.04)` | resulting std 0.0176 = 0.02 × 0.8796, the exact 2σ-truncated value TF produces |
| Split sizes | Asserted on load | 43,410 / 5,426 / 5,427 |

One inconsistency exists **in the original itself**: `bert_classifier.py`'s
`metric_fn_multi` binarises at `probabilities >= 0.3`, while
`calculate_metrics.py` uses `> 0.3`. This port follows `calculate_metrics.py`,
since that is the script producing the per-emotion F1 column of Table 4
(`metric_fn_multi` never computes per-emotion F1). With continuous
probabilities the two differ only on an exact tie, so it is numerically moot.

### Deliberate deviations

* **Framework.** TF1 → PyTorch. Numerically equivalent, but not bit-identical:
  weight init RNG, dropout masks, and data shuffling differ, so expect
  macro-F1 within roughly ±.01–.02 of .46 rather than exactly .46.
* **No checkpoint selection.** Like the original, the model is trained for a
  fixed 4 epochs and the final weights are evaluated on test. Dev is scored
  each epoch for monitoring only.
* **Sentiment/correlation regularisers omitted.** Both default to 0 in the
  original and are off for the reported baseline.
* **Shuffling is stronger here.** The original builds its input pipeline as
  `dataset.repeat().shuffle(buffer_size=100)` — a 100-example buffer over an
  infinitely repeated stream, so training order is close to the file order.
  This port does a full per-epoch shuffle, which is standard and at least as
  good, but it is a real difference in the data ordering.
* **Eval batch size 64 vs 16.** The original reuses `train_batch_size` for
  evaluation. Metrics are aggregated over the full split, so this changes
  nothing beyond float accumulation order.
* **Last-batch handling.** `drop_remainder=True` on a repeated stream never
  actually drops anything in the original; here `drop_last=True` drops the
  final 2 examples of each shuffled epoch. The step count is identical
  (10,852) and the dropped examples differ each epoch.
