"""Hyperparameters for the GoEmotions BERT baseline.

Every value here is transcribed from the original TensorFlow implementation
(google-research/goemotions/bert_classifier.py) or from Demszky et al. (2020),
Section 5.3. The `Source:` comment on each field says where it comes from, so
the replication can be audited line by line.
"""

import os
from dataclasses import dataclass, field
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))


def _default_data_dir() -> str:
    """Locate the GoEmotions .tsv files.

    Resolved relative to this file rather than the working directory, so the
    folder can be copied anywhere (a cloud workspace, a scratch disk) and still
    run. Prefers a local `data/` — populate it with `./download_data.sh` — and
    falls back to the checkout of google-research sitting next to the project.
    """
    local = os.path.join(_HERE, "data")
    if os.path.isfile(os.path.join(local, "emotions.txt")):
        return local
    return os.path.join(_HERE, "..", "google-research", "goemotions", "data")


@dataclass
class Config:
    # --- Data -------------------------------------------------------------
    # Source: bert_classifier.py flags `data_dir`, `train_fname`, `dev_fname`,
    # `test_fname`. The .tsv files are the rater-agreement-filtered split
    # (43,410 / 5,426 / 5,427) that the paper trains and reports on.
    data_dir: str = field(default_factory=_default_data_dir)
    train_fname: str = "train.tsv"
    dev_fname: str = "dev.tsv"
    test_fname: str = "test.tsv"
    # Source: flag `emotion_file`. The file holds 28 lines: the 27 emotions
    # plus `neutral`. Flag `add_neutral` defaults to False precisely because
    # neutral is already in the file, so num_labels == 28. Empty means
    # "emotions.txt inside data_dir".
    emotion_file: str = ""

    # --- Model ------------------------------------------------------------
    # Source: README ("In the paper, we use the cased base model") and flag
    # `do_lower_case=False`. Cased matters: the data has emphatic caps such as
    # "WHY THE FUCK IS BAYLESS ISOING".
    model_name: str = "bert-base-cased"
    do_lower_case: bool = False
    # Source: flag `max_seq_length=50`.
    max_seq_length: int = 50
    # Source: create_model(), `tf.nn.dropout(output_layer, keep_prob=0.9)`
    # applied to the pooled output during training only.
    classifier_dropout: float = 0.1
    # Source: `multilabel=True` -> sigmoid_cross_entropy_with_logits.
    multilabel: bool = True

    # --- Optimisation -----------------------------------------------------
    # Source: flags `train_batch_size=16`, `learning_rate=5e-5`,
    # `num_train_epochs=4.0`, `warmup_proportion=0.1`. The paper says: "We find
    # that training for at least 4 epochs is necessary [...] a small batch size
    # of 16 and learning rate of 5e-5 yields the best performance."
    train_batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 5e-5
    num_train_epochs: float = 4.0
    warmup_proportion: float = 0.1
    # Source: bert/optimization.py AdamWeightDecayOptimizer.
    weight_decay_rate: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-6
    max_grad_norm: float = 1.0
    exclude_from_weight_decay: List[str] = field(
        default_factory=lambda: ["LayerNorm", "layer_norm", "bias"]
    )

    # --- Evaluation -------------------------------------------------------
    # Source: flag `eval_prob_threshold=0.3` and calculate_metrics.py
    # `threshold=0.3`. This is the threshold behind the paper's Table 4.
    eval_prob_threshold: float = 0.3

    # --- Run --------------------------------------------------------------
    seed: int = 42
    output_dir: str = "./output"
    device: str = "cuda"
    # Drops the last incomplete training batch, matching `drop_remainder=True`
    # on the training input_fn in the original code.
    drop_last: bool = True
    num_workers: int = 4
    fp16: bool = False

    def __post_init__(self):
        if not self.emotion_file:
            self.emotion_file = os.path.join(self.data_dir, "emotions.txt")
