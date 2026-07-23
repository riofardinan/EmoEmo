"""Shared learner scaffolding, following EmoGrowth/models/base.py."""

import logging
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.metrics import all_metrics, average_precision

logger = logging.getLogger(__name__)


class BaseLearner:
    """State and evaluation shared by every incremental method."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._device = cfg.device

        # Replay buffer: indices into the training split plus their labels
        # over all classes seen so far. Populated by replay-based methods.
        self._data_memory: List[int] = []
        self._targets_memory = None

        self.train_loader = None
        self.test_loader = None

    # ------------------------------------------------------------- interface

    def incremental_train(self, data_manager):
        raise NotImplementedError

    def after_task(self):
        self._known_classes = self._total_classes

    @property
    def exemplar_size(self) -> int:
        return len(self._data_memory)

    @property
    def samples_per_class(self) -> int:
        if self.cfg.fixed_memory:
            return self.cfg.memory_per_class
        return self.cfg.memory_size // max(1, self._total_classes)

    def build_rehearsal_memory(self, data_manager):
        """No-op for methods without a buffer."""

    # ------------------------------------------------------------ evaluation

    def _forward_logits(self, batch) -> torch.Tensor:
        """Logits for one batch. Overridden by methods with a custom forward."""
        input_ids, attention_mask, token_type_ids = (t.to(self._device) for t in batch[:3])
        return self._network(input_ids, attention_mask, token_type_ids)["logits"]

    @torch.no_grad()
    def _collect(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self._network.eval()
        outputs, labels = [], []
        for batch in loader:
            logits = self._forward_logits(batch)
            outputs.append(logits.float().cpu().numpy())
            labels.append(batch[3].numpy())
        return np.concatenate(outputs), np.concatenate(labels)

    def eval_task(self, loader: DataLoader = None):
        """mAP plus the seven metrics EmoGrowth reports, on raw logits.

        The threshold-based metrics binarise at logit > 0 (probability > 0.5),
        which is what utils/metrics.py does in the original code. `mAP` is
        ranking-based and needs no threshold.
        """
        loader = loader if loader is not None else self.test_loader
        outputs, labels = self._collect(loader)
        _, mean_ap = average_precision(
            torch.from_numpy(outputs), torch.from_numpy(labels)
        )
        metrics = all_metrics(outputs, labels, threshold=0.0)
        return float(mean_ap), metrics, outputs, labels
