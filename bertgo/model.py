"""BERT + a single dense layer, as in create_model() of bert_classifier.py."""

import torch
import torch.nn as nn
from transformers import BertModel


class BertForMultiLabelEmotion(nn.Module):
    """Pooled output -> dropout -> dense(num_labels), sigmoid cross-entropy.

    The original builds the head by hand rather than using a library head:
      output_weights ~ truncated_normal(stddev=0.02), output_bias = zeros,
      logits = output_layer @ output_weights^T + output_bias.
    We reproduce that initialisation because HuggingFace's default for a fresh
    nn.Linear is a different distribution.
    """

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1,
                 multilabel: bool = True):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.num_labels = num_labels
        self.multilabel = multilabel
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.zeros_(self.classifier.bias)

    def pooled_features(self, input_ids, attention_mask, token_type_ids):
        """The [CLS] vector after the pooler (tanh(W·h_CLS)).

        Exposed separately because the incremental experiments in ../emogo
        consume this as their instance representation.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.pooler_output

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        pooled = self.pooled_features(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            if self.multilabel:
                # tf.nn.sigmoid_cross_entropy_with_logits then reduce_mean over
                # *all* elements, i.e. mean over batch and over labels.
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="mean"
                )
            else:
                loss = torch.mean(
                    torch.sum(-labels * torch.log_softmax(logits, dim=-1), dim=-1)
                )
        return logits, loss
