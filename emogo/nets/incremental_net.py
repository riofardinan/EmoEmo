"""BERT backbone with a classification head that grows as tasks arrive.

The counterpart of EmoGrowth/utils/inc_net_ml.py. Where EmoGrowth feeds a
frozen 1000-d ResNet feature into a small MLP, we fine-tune BERT end-to-end and
treat its pooled [CLS] vector as the instance representation, so the head sits
directly on a 768-d feature.
"""

import copy

import torch
import torch.nn as nn
from transformers import BertModel


class BertBackbone(nn.Module):
    """Pooled [CLS] representation. Shared by every method in models/.

    Returns the *raw* pooled vector. Dropout belongs to the classification path
    and is applied by IncrementalBertNet, not here — methods that distil
    relations between features (AESL's RKD builds a representational
    similarity matrix over them) need a deterministic representation, and
    dropping units inside the backbone would inject noise into that matrix.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.feature_dim = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return out.pooler_output


class IncrementalBertNet(nn.Module):
    """BERT + expanding linear head, for Finetune / EWC / LwF / Replay."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = BertBackbone(cfg.model_name)
        self.dropout = nn.Dropout(cfg.classifier_dropout)
        self.feature_dim = self.backbone.feature_dim
        self.fc = None

    def update_fc(self, nb_classes: int):
        """Widen the head to nb_classes, carrying over the old weights."""
        fc = nn.Linear(self.feature_dim, nb_classes)
        nn.init.trunc_normal_(fc.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.zeros_(fc.bias)
        if self.fc is not None:
            n_old = self.fc.out_features
            with torch.no_grad():
                fc.weight[:n_old] = self.fc.weight.data
                fc.bias[:n_old] = self.fc.bias.data
        device = next(self.parameters()).device
        del self.fc
        self.fc = fc.to(device)

    def extract_vector(self, input_ids, attention_mask, token_type_ids):
        """Deterministic features, for distillation and buffer selection."""
        return self.backbone(input_ids, attention_mask, token_type_ids)

    def forward(self, input_ids, attention_mask, token_type_ids):
        features = self.backbone(input_ids, attention_mask, token_type_ids)
        return {"features": features, "logits": self.fc(self.dropout(features))}

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
