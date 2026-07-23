"""Turns GoEmotions into a sequence of multi-label incremental tasks.

This is the text counterpart of EmoGrowth/utils/data_manager_ml.py. That file
reads pre-baked .mat "label sessions"; we build the equivalent structure from
the .tsv splits, reproducing its two defining properties:

  * A training task exposes ONLY its own classes. For task b the label matrix
    has |C^b| columns, so labels from earlier tasks are invisible (past-missing
    partial labels) and labels from later tasks do not exist yet
    (future-missing partial labels). This is the whole problem AESL attacks.

  * A sample may appear in several tasks. A comment labelled {joy, gratitude}
    with joy in task 1 and gratitude in task 3 shows up in both, each time
    carrying only the label visible to that task. That is exactly the
    Figure 1 scenario from the paper.

  * The test set is cumulative: after task b the model is evaluated on all
    classes seen so far, over every test comment that carries at least one of
    them, with full labels for those columns.
"""

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from utils.protocols import build_increments

logger = logging.getLogger(__name__)


def load_emotions(emotion_file: str) -> List[str]:
    with open(emotion_file, encoding="utf-8") as f:
        return [line for line in f.read().splitlines() if line.strip()]


class GoEmotionsDataManager:
    """Holds the tokenized corpus once and slices it per task."""

    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.emotions_all = load_emotions(cfg.emotion_file)

        # Optionally drop `neutral` to get the 27-emotion setting.
        keep = list(range(len(self.emotions_all)))
        if cfg.drop_neutral:
            keep = [i for i, e in enumerate(self.emotions_all) if e != "neutral"]
        self._keep = np.array(keep)
        self.emotions = [self.emotions_all[i] for i in keep]

        if len(self.emotions) != cfg.total_class:
            raise ValueError(
                f"Protocol expects {cfg.total_class} classes but the label set "
                f"has {len(self.emotions)} (drop_neutral={cfg.drop_neutral})."
            )

        self.class_order = self._build_class_order()
        # Column c of every task label matrix refers to emotion
        # self.emotions[self.class_order[c]].
        self.ordered_emotions = [self.emotions[i] for i in self.class_order]
        logger.info("Class order: %s", self.ordered_emotions)

        self._increments = build_increments(
            cfg.init_cls, cfg.increment, cfg.total_class
        )
        logger.info("Task sizes: %s (%d tasks)", self._increments, self.nb_tasks)

        self._splits = {}
        for name, fname in (
            ("train", cfg.train_fname),
            ("test", cfg.test_fname),
        ):
            self._splits[name] = self._load_split(
                os.path.join(cfg.data_dir, fname), tokenizer
            )

    # ------------------------------------------------------------------ setup

    def _build_class_order(self) -> List[int]:
        """Order in which classes arrive.

        'alphabetical' - strict alphabetical order. This is the paper's
            protocol, stated in Appendix B.1: "In the process of splitting
            emotion labels for incremental learning, we just follow the order
            of the alphabet without interfere." It is the default.
        'file'     - the order in emotions.txt. NOT the same thing: GoEmotions
            appends `neutral` as the last line, after the 27 alphabetical
            emotions, so `neutral` lands at index 27 instead of its alphabetical
            index 20 (between `nervousness` and `optimism`). Since `neutral`
            carries 28% of all positive labels, putting it in the final task
            inflates last-task scores for any method that forgets. Kept only
            for comparison with earlier runs.
        'shuffled' - permuted with cfg.seed. The paper does not shuffle, but
            its Limitations section flags class order as unexplored, so this
            is here for a robustness check.
        or an explicit list of emotion names in cfg.class_order.
        """
        order_spec = self.cfg.class_order
        n = len(self.emotions)
        if order_spec == "alphabetical":
            return sorted(range(n), key=lambda i: self.emotions[i])
        if order_spec == "file":
            return list(range(n))
        if order_spec == "shuffled":
            rng = np.random.RandomState(self.cfg.seed)
            return rng.permutation(n).tolist()
        if isinstance(order_spec, list):
            missing = set(self.emotions) - set(order_spec)
            if missing:
                raise ValueError(f"class_order is missing emotions: {sorted(missing)}")
            index = {e: i for i, e in enumerate(self.emotions)}
            return [index[e] for e in order_spec if e in index]
        raise ValueError(f"Unsupported class_order: {order_spec!r}")

    def _load_split(self, path: str, tokenizer):
        df = pd.read_csv(
            path, sep="\t", encoding="utf-8", header=None,
            names=["text", "labels", "id"], dtype={"text": str},
        )
        texts = [t if isinstance(t, str) else "" for t in df["text"].tolist()]

        y_full = np.zeros((len(df), len(self.emotions_all)), dtype=np.float32)
        for i, raw in enumerate(df["labels"].tolist()):
            for idx in str(raw).split(","):
                y_full[i, int(idx)] = 1.0

        # Restrict to the kept emotions, then reorder columns so that column c
        # is the c-th class to arrive.
        y = y_full[:, self._keep][:, self.class_order]

        encoded = tokenizer(
            texts,
            max_length=self.cfg.max_seq_length,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
        )
        split = {
            "texts": texts,
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(encoded["token_type_ids"], dtype=torch.long),
            "y": torch.from_numpy(y),
        }
        logger.info("Loaded %s: %d examples", os.path.basename(path), len(texts))
        return split

    def attach_affective(self, name: str, vectors: np.ndarray):
        """Stores per-sample valence/arousal/dominance for the split.

        Used as Teacher-2 in AESL's relation-based knowledge distillation.
        """
        self._splits[name]["affective"] = torch.from_numpy(
            vectors.astype(np.float32)
        )

    # ------------------------------------------------------------- properties

    @property
    def nb_tasks(self) -> int:
        return len(self._increments)

    def get_task_size(self, task: int) -> int:
        return self._increments[task]

    def get_accumulate_tasksize(self, task: int) -> int:
        return sum(self._increments[: task + 1])

    def get_total_classnum(self) -> int:
        return self.cfg.total_class

    def class_range(self, task: int):
        """[start, end) column indices owned by `task`."""
        start = sum(self._increments[:task])
        return start, start + self._increments[task]

    def task_emotions(self, task: int) -> List[str]:
        start, end = self.class_range(task)
        return self.ordered_emotions[start:end]

    # ------------------------------------------------------------- task views

    def get_dataset(
        self,
        task: int,
        source: str,
        appendent: Optional[tuple] = None,
        ret_data: bool = False,
        affective: bool = False,
    ):
        """Builds the TensorDataset for one task.

        train: rows with >=1 label in C^task; labels restricted to C^task.
        test:  rows with >=1 label in C^0..C^task; labels over all seen classes.

        `appendent` is (indices, label_matrix) from a replay buffer; its labels
        are already expressed over all classes seen so far.
        """
        if source == "train":
            split = self._splits["train"]
            start, end = self.class_range(task)
            mask = split["y"][:, start:end].sum(dim=1) > 0
            idx = torch.nonzero(mask, as_tuple=True)[0]
            y = split["y"][idx][:, start:end]
        elif source == "test":
            split = self._splits["test"]
            end = self.get_accumulate_tasksize(task)
            mask = split["y"][:, :end].sum(dim=1) > 0
            idx = torch.nonzero(mask, as_tuple=True)[0]
            y = split["y"][idx][:, :end]
        else:
            raise ValueError(f"Unknown data source {source!r}")

        tensors = [
            split["input_ids"][idx],
            split["attention_mask"][idx],
            split["token_type_ids"][idx],
            y,
        ]

        if appendent is not None:
            app_idx, app_y = appendent
            app_idx = torch.as_tensor(app_idx, dtype=torch.long)
            app_y = torch.as_tensor(app_y, dtype=torch.float32)
            # The current task's labels only cover C^task, so pad the known
            # classes with zeros before concatenating buffered samples whose
            # labels span every class seen so far.
            known = self.get_accumulate_tasksize(task - 1) if task > 0 else 0
            tensors[3] = torch.hstack([torch.zeros(y.shape[0], known), y])
            tensors = [
                torch.cat([tensors[0], split["input_ids"][app_idx]], dim=0),
                torch.cat([tensors[1], split["attention_mask"][app_idx]], dim=0),
                torch.cat([tensors[2], split["token_type_ids"][app_idx]], dim=0),
                torch.cat([tensors[3], app_y], dim=0),
            ]
            idx = torch.cat([idx, app_idx], dim=0)

        if affective:
            if "affective" not in split:
                raise RuntimeError(
                    "Affective vectors were requested but never attached; "
                    "call attach_affective() first."
                )
            tensors.append(split["affective"][idx])

        dataset = TensorDataset(*tensors)
        logger.info(
            "task %d %s: %d samples, %d label columns",
            task, source, len(dataset), tensors[3].shape[1],
        )
        if ret_data:
            return idx, tensors, dataset
        return dataset
