"""EWC — Elastic Weight Consolidation (EmoGrowth/models/ewc_ml.py).

EWC keeps the *parameters* near where they were, instead of keeping the
*outputs* near where they were the way LwF does. After each task it estimates
how much each weight mattered (the diagonal of the Fisher information) and adds
a quadratic penalty for moving important weights:

    loss = loss_clf + lamda * sum_n  fisher[n] * (theta[n] - theta_old[n])^2 / 2

Note `loss_clf` is Finetune's loss, zeros and all — EWC does nothing about the
past-missing partial label problem, it only slows the drift. The paper's own
verdict (Section 4.2): "EWC is not suitable for direct application to MLCIL
task due to its poor performance." Expect it to land near Finetune, not LwF;
in Table 3 (Audio28 B0-I7) EWC scores 37.9 Avg mAP against Finetune's 36.4 and
LwF's 46.6. It is here as a reference point, not a contender.

Memory note: `fisher` and `mean` each hold a full copy of the backbone's
parameters, so fine-tuning BERT costs ~880 MB of extra device memory on top of
the model and its gradients.
"""

import logging

import torch

from models.finetune import Finetune

logger = logging.getLogger(__name__)


class EWC(Finetune):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fisher = None
        self.mean = None

    # ------------------------------------------------------------------ loss

    def _compute_loss(self, out, targets, inputs, batch) -> torch.Tensor:
        # Classification term is exactly Finetune's, including the zero padding
        # over previously seen classes.
        loss_clf = super()._compute_loss(out, targets, inputs, batch)
        if self._cur_task == 0:
            return loss_clf
        return loss_clf + self.cfg.ewc_lamda * self._ewc_penalty()

    def _ewc_penalty(self) -> torch.Tensor:
        loss = 0.0
        for name, param in self._network.named_parameters():
            if name not in self.fisher:
                continue
            # The classifier grew when the task changed, so score only the rows
            # that existed before. `len()` on a tensor is its first dimension,
            # which for fc.weight is the class axis.
            old = self.mean[name]
            loss = loss + torch.sum(
                self.fisher[name] * (param[: len(old)] - old).pow(2)
            ) / 2
        return loss

    # ------------------------------------------------------ Fisher estimation

    def incremental_train(self, data_manager):
        super().incremental_train(data_manager)
        # Estimated after training, on this task's own data.
        new_fisher = self._fisher_diagonal(self.train_loader)
        if self.fisher is None:
            self.fisher = new_fisher
        else:
            # Running average weighted by how much of the label space is old.
            alpha = self._known_classes / self._total_classes
            for name, tensor in new_fisher.items():
                n_old = len(self.fisher[name])
                tensor[:n_old] = (alpha * self.fisher[name]
                                  + (1 - alpha) * tensor[:n_old])
            self.fisher = new_fisher
        self.mean = {
            name: p.clone().detach()
            for name, p in self._network.named_parameters()
            if p.requires_grad
        }

    def _fisher_diagonal(self, loader):
        """Mean squared gradient per parameter, clamped at `fishermax`.

        The original builds an SGD optimizer here but never steps it — it is
        only used to zero gradients — so this calls zero_grad on the model
        directly. Also note `train()` rather than `eval()`: dropout is active,
        matching the original.
        """
        fisher = {
            name: torch.zeros_like(p)
            for name, p in self._network.named_parameters()
            if p.requires_grad
        }
        self._network.train()
        fishermax = torch.tensor(self.cfg.ewc_fishermax, device=self._device)

        for batch in loader:
            inputs = tuple(t.to(self._device) for t in batch[:3])
            targets = batch[3].to(self._device)
            logits = self._network(*inputs)["logits"]
            # Scored on the current task's classes only.
            loss = self.criterion(logits[:, self._known_classes:], targets)
            self._network.zero_grad(set_to_none=True)
            loss.backward()
            for name, p in self._network.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.detach().pow(2)

        n_batches = max(1, len(loader))
        for name in fisher:
            fisher[name] = torch.min(fisher[name] / n_batches, fishermax)
        self._network.zero_grad(set_to_none=True)
        logger.info("[EWC] Fisher diagonal estimated over %d batches", n_batches)
        return fisher
