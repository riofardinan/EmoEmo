"""Run configuration, loaded from a JSON file in exps/.

Optimisation defaults are inherited from the verified GoEmotions replication in
../bertgo (Demszky et al. 2020, Section 5.3) rather than from EmoGrowth's
iScience settings, because the backbone here is a fine-tuned BERT, not a small
MLP over frozen features. The AESL-specific lambdas do come from EmoGrowth.
"""

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import torch

from utils.protocols import resolve_protocol

# Paths resolve relative to this package, not the working directory, so the
# folder can be copied to a server on its own and still find its data.
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # --- experiment identity ---------------------------------------------
    prefix: str = "emogo"
    method: str = "finetune"          # which continual learning algorithm
    protocol: str = "B0-I7"           # see utils/protocols.py
    seed: int = 1993                  # EmoGrowth/PyCIL default

    # --- data -------------------------------------------------------------
    data_dir: str = field(default_factory=lambda: os.path.join(_PKG, "data"))
    train_fname: str = "train.tsv"
    test_fname: str = "test.tsv"
    # Empty means "emotions.txt inside data_dir".
    emotion_file: str = ""
    # GoEmotions ships 27 emotions + neutral. Keeping neutral gives 28 classes,
    # matching the paper's Audio28 protocols.
    drop_neutral: bool = False
    # "file" | "shuffled" | explicit list of emotion names
    class_order: Union[str, List[str]] = "file"

    # Filled in from `protocol` by __post_init__.
    init_cls: int = 0
    increment: int = 0
    total_class: int = 0

    # --- backbone ---------------------------------------------------------
    # Every field from here to `num_workers` is copied from ../bertgo/config.py
    # and must stay identical to it: the verified GoEmotions replication is the
    # reference point (and the Upper-bound row) these experiments are read
    # against, so a drift here would make the comparison meaningless.
    # tests/check_config.py asserts the match.
    model_name: str = "bert-base-cased"
    do_lower_case: bool = False       # cased model — see ../bertgo/README.md
    max_seq_length: int = 50
    classifier_dropout: float = 0.1

    # --- optimisation (from ../bertgo) ------------------------------------
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 5e-5
    init_epochs: int = 4              # first task
    epochs: int = 4                   # subsequent tasks
    warmup_proportion: float = 0.1
    weight_decay_rate: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-6
    max_grad_norm: float = 1.0
    exclude_from_weight_decay: List[str] = field(
        default_factory=lambda: ["LayerNorm", "layer_norm", "bias"]
    )
    drop_last: bool = True
    # The verified bertgo run used fp32. Keep it off so the incremental results
    # differ from that baseline only by the continual-learning setting.
    fp16: bool = False
    num_workers: int = 4

    # --- LwF --------------------------------------------------------------
    # `lamda` in EmoGrowth/models/lwf_ml.py; 3 in both the iScience and PNAS
    # blocks of that file.
    lwf_lamda: float = 3.0

    # --- replay buffer (ER / RS / PRS / OCDM) -----------------------------
    memory_size: int = 500
    memory_per_class: int = 20
    fixed_memory: bool = True
    buffer_type: str = "prs"          # random | rs | prs | ocdm

    # --- AESL / CLIF ------------------------------------------------------
    feature_dim: int = 64
    lamda_le: float = 0.005
    lamda_kd_logits: float = 1.0
    lamda_kd_relation_data: float = 1.0
    lamda_kd_relation_aff: float = 0.1
    ld: bool = False                  # graph-based label disambiguation

    # --- affective dimension (Teacher-2 of the RKD module) ----------------
    # GoEmotions has no per-sample valence/arousal ratings, unlike Video27 and
    # Audio28. We derive them from the text with the NRC-VAD lexicon, which
    # keeps the signal independent of the emotion labels — important, since a
    # label-derived signal would leak future classes.
    vad_lexicon_path: str = field(
        default_factory=lambda: os.path.join(_PKG, "data",
                                             "NRC-VAD-Lexicon-v2.1.txt")
    )
    use_vad_dims: List[int] = field(default_factory=lambda: [0, 1, 2])

    # --- runtime ----------------------------------------------------------
    output_dir: str = "./results"
    device: Any = "cuda"

    def __post_init__(self):
        if not self.emotion_file:
            self.emotion_file = os.path.join(self.data_dir, "emotions.txt")
        init_cls, increment, total_class = resolve_protocol(self.protocol)
        self.init_cls, self.increment, self.total_class = (
            init_cls, increment, total_class
        )
        if isinstance(self.device, str):
            requested = self.device
            if requested.startswith("cuda") and not torch.cuda.is_available():
                requested = "cpu"
            self.device = torch.device(requested)

    @property
    def run_name(self) -> str:
        return f"{self.prefix}_{self.method}_{self.protocol}_seed{self.seed}"

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_dir, self.method, self.protocol,
                            f"seed{self.seed}")

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["device"] = str(self.device)
        return d


def load_config(path: str, overrides: Dict[str, Any] = None) -> Config:
    with open(path, encoding="utf-8") as f:
        params = json.load(f)
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})

    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(params) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return Config(**params)
