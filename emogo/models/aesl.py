"""AESL — the paper's method (EmoGrowth/models/clif_ml.py).

Four loss terms (Eq. 15):

    L = L_ce
      + lamda_le            * L_le          emotional semantics learning (Eq. 7)
      + lamda_kd_logits     * L_kd_logits   soft labels from the old model
      + lamda_kd_relation_1 * RKD(old model features,  new features)
      + lamda_kd_relation_2 * RKD(affective features,  new features)

The first three are modality-agnostic. The fourth is not, and it is the one
place where the text setting cannot mirror the paper:

  Video27/Brain27 supply 14 human-rated appraisal dimensions per stimulus and
  Audio28 supplies 11, on a 9-point Likert scale. GoEmotions supplies none, so
  `precompute_vad.py` derives a 3-dimensional valence/arousal/dominance vector
  from the text — either by averaging NRC-VAD lexicon entries, or with a BERT
  regressor trained on EmoBank. Three dimensions make for a coarser similarity
  matrix than eleven; report which source was used.

Two more deviations worth stating in a write-up:

  * RKD compares representational similarity matrices computed **within a
    batch**. EmoGrowth uses batch 128; this project uses 16 throughout, from
    the GoEmotions replication, giving 240 off-diagonal entries per matrix
    instead of 16,256. AESL is the only method here whose loss is defined over
    batch statistics, so it is the only one this affects.
  * `atanh` diverges when two samples have identical features, which duplicate
    texts do produce (0.7% of GoEmotions training rows are exact duplicates).
    The original guards by zeroing non-finite entries; that guard is kept.
"""

import logging

import torch
from torch.utils.data import DataLoader

from models.finetune import Finetune
from nets.clif_net import IncrementalCLIFNet, LinkPredictionLossCosine
from utils.graph import sym_conditional_prob, sym_conditional_prob_update

logger = logging.getLogger(__name__)


class AESL(Finetune):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._network = IncrementalCLIFNet(cfg)
        self.embedding_criterion = LinkPredictionLossCosine()
        self.label_adj = None
        self._old_label_adj = None
        self.soft_label = None

    # --------------------------------------------------------------- helpers

    def _adj(self) -> torch.Tensor:
        return self.label_adj.to(self._device)

    def _forward_logits(self, batch) -> torch.Tensor:
        """Evaluation path; the network returns a tuple, not a dict."""
        inputs = tuple(t.to(self._device) for t in batch[:3])
        logits, _ = self._network(*inputs, self._adj())
        return logits

    def _train_dataset_kwargs(self) -> dict:
        # Pulls the per-sample affective vectors in as a fifth tensor.
        return {"affective": True}

    # ------------------------------------------------------------- task loop

    def incremental_train(self, data_manager):
        """Grow the ERG before training, then hand off to the standard loop."""
        task = self._cur_task + 1
        _, tensors, _ = data_manager.get_dataset(
            task, source="train", ret_data=True, affective=True
        )
        train_y = tensors[3]

        if task == 0:
            self.label_adj = sym_conditional_prob(train_y.to(self._device))
        else:
            soft_logits = self._old_soft_labels(tensors)
            known = self._known_classes
            total = known + data_manager.get_task_size(task)
            self.label_adj, self.soft_label = sym_conditional_prob_update(
                soft_logits, self.label_adj, train_y.to(self._device),
                known, total,
            )
        logger.info("[AESL] ERG is now %s", tuple(self.label_adj.shape))

        super().incremental_train(data_manager)

    @torch.no_grad()
    def _old_soft_labels(self, tensors) -> torch.Tensor:
        """Previous model's logits over old classes, batched to bound memory."""
        self._old_network.eval()
        old_adj = self._old_label_adj.to(self._device)
        loader = DataLoader(
            torch.utils.data.TensorDataset(*tensors[:3]),
            batch_size=self.cfg.eval_batch_size, shuffle=False,
        )
        out = []
        for batch in loader:
            inputs = tuple(t.to(self._device) for t in batch)
            logits, _ = self._old_network(*inputs, old_adj)
            out.append(logits)
        return torch.cat(out, dim=0)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        self._old_label_adj = self.label_adj

    # ------------------------------------------------------------------ loss

    def _step(self, batch) -> torch.Tensor:
        inputs = tuple(t.to(self._device, non_blocking=True) for t in batch[:3])
        targets = batch[3].to(self._device, non_blocking=True)
        affective = batch[4].to(self._device, non_blocking=True)
        adj = self._adj()

        if self._cur_task == 0:
            logits, label_embedding = self._network(*inputs, adj)
            loss_clf = self.criterion(logits, targets)
            loss_le = self.embedding_criterion(label_embedding, self._loss_adj(adj))
            return loss_clf + self.cfg.lamda_le * loss_le

        logits, label_embedding, feature_new = self._network(*inputs, adj, kd=True)
        with torch.no_grad():
            old_logits, _, feature_old = self._old_network(
                *inputs, self._old_label_adj.to(self._device), kd=True
            )

        loss_clf = self.criterion(logits[:, self._known_classes:], targets)
        loss_le = self.embedding_criterion(label_embedding, self._loss_adj(adj))
        loss_kd_logits = self.criterion(
            logits[:, : self._known_classes], torch.sigmoid(old_logits)
        )
        loss_rkd_data = self._relation_kd(feature_old, feature_new)
        loss_rkd_aff = self._relation_kd(affective, feature_new)

        return (loss_clf
                + self.cfg.lamda_le * loss_le
                + self.cfg.lamda_kd_logits * loss_kd_logits
                + self.cfg.lamda_kd_relation_data * loss_rkd_data
                + self.cfg.lamda_kd_relation_aff * loss_rkd_aff)

    @staticmethod
    def _loss_adj(adj: torch.Tensor) -> torch.Tensor:
        """The link-prediction target includes self-loops."""
        return adj + torch.eye(adj.shape[0], dtype=adj.dtype, device=adj.device)

    def _relation_kd(self, feature_old: torch.Tensor,
                     feature_new: torch.Tensor) -> torch.Tensor:
        """Centered-kernel-alignment style loss over batch RSMs (Eq. 10-12).

        Both feature sets are mean-centred over the batch, turned into cosine
        similarity matrices, and compared after an `arctanh` reparameterisation
        that maps (-1, 1) onto the real line.
        """
        feature_old = feature_old - feature_old.mean(dim=0)
        feature_new = feature_new - feature_new.mean(dim=0)

        rsm_old = torch.nn.functional.cosine_similarity(
            feature_old.unsqueeze(1), feature_old.unsqueeze(0), dim=-1)
        rsm_new = torch.nn.functional.cosine_similarity(
            feature_new.unsqueeze(1), feature_new.unsqueeze(0), dim=-1)

        # Drop the diagonal: a sample's similarity with itself is always 1 and
        # would send atanh to infinity.
        rsm_old = rsm_old - torch.diag_embed(torch.diagonal(rsm_old))
        rsm_new = rsm_new - torch.diag_embed(torch.diagonal(rsm_new))

        loss = torch.atanh(rsm_old) - torch.atanh(rsm_new)
        # Duplicate texts give cosine exactly 1 off the diagonal too, so the
        # guard has to stay. The original zeroes inf and nan the same way.
        bad = ~torch.isfinite(loss)
        if bad.any():
            loss = torch.where(bad, torch.zeros_like(loss), loss)
        return torch.mean(torch.pow(loss, 2))
