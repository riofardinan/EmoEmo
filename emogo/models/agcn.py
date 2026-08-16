"""AGCN — Augmented Graph Convolutional Network (EmoGrowth/models/agcn_ml.py).

A multi-label CIL method that carries label dependencies in an emotional
relation graph which grows as tasks arrive. The loss is LwF's, evaluated
through the graph:

    task 0:   L(logits, targets)
    task b>0: L(logits[:, known:], targets)
            + lamda_kd_logits * L(logits[:, :known], sigmoid(old_model(x, old_adj)))

So AGCN preserves old knowledge exactly the way LwF does — soft labels from the
frozen previous model — but both the student and the teacher read through their
respective label graphs (Section 4.2 of the paper groups it with the graph
methods, and finds it a strong baseline that AESL then improves on).

The graph itself is built by the same `sym_conditional_prob` routines AESL uses,
verified against base.py to 0.0. `lamda_kd_logits = 1`, from agcn_ml.py.
"""

import logging

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.finetune import Finetune
from nets.agcn_net import IncrementalAGCNNet
from utils.graph import ERGBuilder

logger = logging.getLogger(__name__)


class AGCN(Finetune):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._network = IncrementalAGCNNet(cfg)
        self.sigmoid = torch.nn.Sigmoid()
        self.erg = ERGBuilder(cfg)
        self.label_adj = None
        self._old_label_adj = None
        self.soft_label = None

    def _adj(self) -> torch.Tensor:
        return self.label_adj.to(self._device)

    def _forward_logits(self, batch) -> torch.Tensor:
        inputs = tuple(t.to(self._device) for t in batch[:3])
        return self._network(*inputs, self._adj())["logits"]

    # ------------------------------------------------------------- task loop

    def incremental_train(self, data_manager):
        task = self._cur_task + 1
        _, tensors, _ = data_manager.get_dataset(
            task, source="train", ret_data=True
        )
        # Only the graph estimate sees this; the training loop below refetches
        # the full task from the data manager.
        tensors = self.erg.subsample(tensors, task)
        train_y = tensors[3]

        if task == 0:
            self.label_adj = self.erg.first(train_y.to(self._device))
        else:
            soft_logits = self._old_soft_labels(tensors)
            known = self._known_classes
            total = known + data_manager.get_task_size(task)
            self.label_adj, self.soft_label = self.erg.grow(
                soft_logits, self.label_adj, train_y.to(self._device),
                known, total,
            )
        logger.info("[AGCN] ERG is now %s, estimator %s",
                    tuple(self.label_adj.shape), self.erg.describe())

        super().incremental_train(data_manager)

    @torch.no_grad()
    def _old_soft_labels(self, tensors) -> torch.Tensor:
        self._old_network.eval()
        old_adj = self._old_label_adj.to(self._device)
        loader = DataLoader(
            TensorDataset(*tensors[:3]),
            batch_size=self.cfg.eval_batch_size, shuffle=False,
        )
        out = []
        for batch in loader:
            inputs = tuple(t.to(self._device) for t in batch)
            out.append(self._network_forward(self._old_network, inputs, old_adj))
        return torch.cat(out, dim=0)

    @staticmethod
    def _network_forward(net, inputs, adj):
        return net(*inputs, adj)["logits"]

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        self._old_label_adj = self.label_adj

    # ------------------------------------------------------------------ loss

    def _step(self, batch) -> torch.Tensor:
        inputs = tuple(t.to(self._device, non_blocking=True) for t in batch[:3])
        targets = batch[3].to(self._device, non_blocking=True)
        out = self._network(*inputs, self._adj())
        return self._compute_loss(out, targets, inputs, batch)

    def _compute_loss(self, out, targets, inputs, batch) -> torch.Tensor:
        logits = out["logits"]
        if self._cur_task == 0:
            return self.criterion(logits, targets)

        loss_clf = self.criterion(logits[:, self._known_classes:], targets)
        with torch.no_grad():
            old_logits = self._network_forward(
                self._old_network, inputs, self._old_label_adj.to(self._device)
            )
        loss_kd = self.criterion(
            logits[:, : self._known_classes], self.sigmoid(old_logits)
        )
        return loss_clf + self.cfg.lamda_kd_logits * loss_kd
