"""AGCN network — port of EmoGrowth/convs/AGCNModel.py + inc_net_ml.py.

Is AGCN an "image" method? No. The graph AGCN convolves over is the **emotion
label co-occurrence graph** — nodes are emotion classes, edges are how often
they are labelled together. Nothing about it is image-specific. The image-ness
of the original AGCN paper (Du et al., 2022, multi-label image classification)
lives entirely in its CNN backbone, and EmoGrowth already stripped that out:
its AGCNNet takes a precomputed feature vector of any modality. Here that
feature is BERT's pooled [CLS] output, the same substitution every method in
this folder makes. The graph branch transfers unchanged.

Architecture (per EmoGrowth's IncrementalNet_AGCN.forward):

    x    = project(BERT_pooled)                 instance feature, feature_dim
    ya   = fc(x)                                plain linear classifier, [B, C]
    E    = x ⊙ fc.weight                        label-conditioned embeddings, [B, C, d]
    yb   = GCN(E, adj)                           graph branch, [B, C]
    logits = ya + yb

The classifier is thus a residual sum of a direct term and a graph-propagated
term — the colleague's version dropped `ya` and kept only the graph, which is
why its logits looked nothing like the original's.

One deliberate fix: the original normalises the adjacency **inside** its
per-sample Python loop (`H = self.norm(H, add=True)`), so it re-normalises an
already-normalised matrix for every sample after the first — an order-dependent
artifact. This computes the normalisation once, which is plainly the intent,
and vectorises the branch over the batch.
"""

import copy

import torch
import torch.nn as nn
from transformers import BertModel


def gcn_normalize(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation with self-loops: D^-1/2 (A_offdiag + I) D^-1/2.

    Matches `gcn.norm(H, add=True)`: zero the diagonal, add the identity, then
    symmetric-normalise. Negative degrees are clamped and infinities zeroed,
    as in the original.
    """
    n = adj.shape[0]
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
    a = adj * (eye == 0) + eye
    deg = a.sum(dim=1)
    deg = deg.clamp(min=0)
    deg_inv = deg.pow(-0.5)
    deg_inv[torch.isinf(deg_inv)] = 0.0
    d_inv = torch.diag(deg_inv)
    return d_inv @ a @ d_inv


class GraphBranch(nn.Module):
    """The `gcn` module: collapse each class embedding to a scalar, propagate."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, label_embedding: torch.Tensor,
                norm_adj: torch.Tensor) -> torch.Tensor:
        # label_embedding: [B, C, d]  ->  scores [B, C, 1] -> [B, C]
        scores = self.linear(label_embedding).squeeze(-1)          # [B, C]
        # Propagate over the (pre-normalised) label graph: (norm_adj @ s^T)^T
        return scores @ norm_adj.t()                                # [B, C]


class IncrementalAGCNNet(nn.Module):
    """BERT + AGCN head (direct classifier + graph branch), growing per task."""

    def __init__(self, cfg):
        super().__init__()
        self.bert = BertModel.from_pretrained(cfg.model_name)
        self.dropout = nn.Dropout(cfg.classifier_dropout)
        # Project BERT's 768-d pooled output into AGCN's feature_dim space
        # (64), the dimensionality its fc and graph branch assume. EmoGrowth's
        # AGCNNet had a Linear(input_size,512)->ReLU->Linear(512,64) extractor;
        # a single projection suffices here since BERT is fine-tuned.
        self.feature_dim = cfg.feature_dim
        self.project = nn.Linear(self.bert.config.hidden_size, self.feature_dim)
        self.graph_branch = GraphBranch(self.feature_dim)
        self.fc = None

    def update_fc(self, nb_classes: int):
        fc = nn.Linear(self.feature_dim, nb_classes)
        if self.fc is not None:
            n_old = self.fc.out_features
            with torch.no_grad():
                fc.weight[:n_old] = self.fc.weight.data
                fc.bias[:n_old] = self.fc.bias.data
        device = next(self.parameters()).device
        del self.fc
        self.fc = fc.to(device)

    def features(self, input_ids, attention_mask, token_type_ids):
        pooled = self.bert(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).pooler_output
        return self.project(self.dropout(pooled))

    def forward(self, input_ids, attention_mask, token_type_ids, label_adj):
        x = self.features(input_ids, attention_mask, token_type_ids)  # [B, d]
        ya = self.fc(x)                                                # [B, C]

        # Label-conditioned embeddings: x[b] scaled by each class's fc weights.
        # [B, 1, d] * [C, d] -> [B, C, d]
        label_embedding = x.unsqueeze(1) * self.fc.weight
        norm_adj = gcn_normalize(label_adj)
        yb = self.graph_branch(label_embedding, norm_adj)             # [B, C]
        return {"logits": ya + yb}

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
