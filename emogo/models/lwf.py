"""LwF — Learning without Forgetting (EmoGrowth/models/lwf_ml.py).

The multi-label form of LwF differs from Finetune in exactly one way, and that
difference is the whole point:

  Finetune  loss = L(logits[:, :],        [0...0 | y])
  LwF       loss = L(logits[:, known:],   y)
                 + lamda * L(logits[:, :known], sigmoid(old_logits))

Finetune supervises the old columns with **zeros** — it tells the model that
none of the previously learned emotions are present in any new sample, which is
false and is what destroys them (past-missing partial labels). LwF never touches
the old columns with ground truth at all: it supervises them with the previous
model's own sigmoid outputs as soft targets, so old knowledge is preserved by
self-distillation instead of being actively contradicted.

This is why LwF is such a strong baseline on 28 classes — Table 3 of the paper
(Audio28 B0-I7): Finetune ends at mAP 27.3, LwF at 40.6, AESL at 42.7.
"""

import logging

import torch

from models.finetune import Finetune

logger = logging.getLogger(__name__)


class LwF(Finetune):
    def __init__(self, cfg):
        super().__init__(cfg)
        # `trans` in the original: the old model's logits are squashed to
        # probabilities and used as soft targets.
        self.sigmoid = torch.nn.Sigmoid()

    def _compute_loss(self, out, targets, inputs, batch) -> torch.Tensor:
        logits = out["logits"]
        if self._cur_task == 0:
            return self.criterion(logits, targets)

        # New classes: supervised by this task's real labels.
        loss_clf = self.criterion(logits[:, self._known_classes:], targets)

        # Old classes: supervised by the frozen previous model.
        with torch.no_grad():
            old_logits = self._old_network(*inputs)["logits"]
        loss_kd = self.criterion(
            logits[:, : self._known_classes], self.sigmoid(old_logits)
        )

        return self.cfg.lwf_lamda * loss_kd + loss_clf

    def after_task(self):
        """Snapshot the just-trained model as the next task's teacher."""
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
