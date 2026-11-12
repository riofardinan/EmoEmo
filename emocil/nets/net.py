"""BERT + one linear head over a label layout, as in ../emogo/nets/incremental_net.py.

Pooled [CLS] -> dropout -> linear. The head is initialised like bertgo/emogo
(truncated normal, std 0.02, cut at +-0.04; zero bias). `expand_to` builds a
wider head and copies trained rows into given positions, which is emogo's
update_fc generalised to non-contiguous positions: a merged label space places
an old dataset's labels among the new dataset's outputs.
"""

import copy

import torch
import torch.nn as nn
from transformers import BertModel


def _new_head(dim, n):
    fc = nn.Linear(dim, n)
    nn.init.trunc_normal_(fc.weight, std=0.02, a=-0.04, b=0.04)
    nn.init.zeros_(fc.bias)
    return fc


class EmoNet(nn.Module):
    def __init__(self, model_name, n_out, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = _new_head(self.bert.config.hidden_size, n_out)

    def forward(self, input_ids, attention_mask, token_type_ids):
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids).pooler_output
        return self.fc(self.dropout(pooled))

    def expand_to(self, n_out, positions):
        """Replace the head by one of size n_out; old row j goes to positions[j]."""
        assert len(positions) == self.fc.out_features
        fc = _new_head(self.fc.in_features, n_out).to(self.fc.weight.device)
        with torch.no_grad():
            fc.weight[positions] = self.fc.weight.data
            fc.bias[positions] = self.fc.bias.data
        self.fc = fc

    def frozen_copy(self):
        twin = copy.deepcopy(self)
        for p in twin.parameters():
            p.requires_grad = False
        return twin.eval()
