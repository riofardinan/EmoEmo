"""CLIF network for AESL — port of EmoGrowth/convs/{CLIFModel,layers}.py.

Three pieces, following Sections 3.4 and 3.5 of the paper:

  GIN      Label semantic encoding. Runs message passing over the augmented
           emotional relation graph to turn per-class node features into label
           embeddings E^b (Eq. 6).
  FDModel  Semantic-guided feature decoupling. Combines the instance
           representation with each label embedding to produce a per-class
           feature (Eq. 8).
  Conv1d   Grouped classifier, one 1-D kernel per class, so each class scores
           its own decoupled feature (Eq. 9).

The one substitution for the text modality: the instance representation is
BERT's pooled [CLS] vector, fine-tuned end-to-end, where EmoGrowth feeds a
frozen precomputed feature. Everything downstream is unchanged.

Node features are a fixed random Gaussian matrix, not embeddings of the emotion
names. That is the paper's choice (Section 3.4: "emotion category labels are
difficult to obtain word embeddings directly from language models"), and it is
tempting to override it here since we *do* have the emotion names as text — but
the paper's own ablation (Table 5, row "+SE") tried exactly that with LLaMA
3.1-8B sentence embeddings and scored worse: 45.3 against 49.0.
"""

import copy
import math

import torch
import torch.nn as nn
from transformers import BertModel


class MLP(nn.Module):
    """EmoGrowth/convs/layers.py MLP, kaiming-uniform init."""

    def __init__(self, in_features, out_features, hidden_features=None,
                 batch_norm=False, nonlinearity="leaky_relu",
                 negative_slope=0.1, with_output_nonlinearity=True):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.negative_slope = negative_slope
        hidden_features = hidden_features or []
        self.fcs = nn.ModuleList()

        def add_activation(idx, is_last):
            if not (with_output_nonlinearity or not is_last):
                return
            if batch_norm:
                self.fcs.append(nn.BatchNorm1d(idx, track_running_stats=True))
            if nonlinearity == "relu":
                self.fcs.append(nn.ReLU(inplace=True))
            elif nonlinearity == "leaky_relu":
                self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
            else:
                raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

        if hidden_features:
            in_dims = [in_features] + hidden_features
            out_dims = hidden_features + [out_features]
            for i, (a, b) in enumerate(zip(in_dims, out_dims)):
                self.fcs.append(nn.Linear(a, b))
                add_activation(b, i == len(in_dims) - 1)
        else:
            self.fcs.append(nn.Linear(in_features, out_features))
            add_activation(out_features, True)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.fcs:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=self.negative_slope,
                                         nonlinearity=self.nonlinearity)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.fcs:
            # BatchNorm1d needs [N, C]; the FDModel path carries [N, C, H].
            if isinstance(layer, nn.BatchNorm1d) and x.dim() == 3:
                x = layer(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = layer(x)
        return x


class GINLayer(nn.Module):
    """One round of message passing: h <- MLP((1+eps)h + A h) + h."""

    def __init__(self, mlp, eps=0.0, train_eps=True, residual=True):
        super().__init__()
        self.mlp = mlp
        self.initial_eps = eps
        self.residual = residual
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, adj):
        res = (1 + self.eps) * x + torch.matmul(adj, x)
        res = self.mlp(res)
        return res + x if self.residual else res


class GIN(nn.Module):
    """Graph Isomorphism Network, the GAE encoder of Section 3.4."""

    def __init__(self, num_layers, in_features, out_features,
                 hidden_features=None, eps=0.0, train_eps=True, residual=True,
                 batch_norm=True, nonlinearity="leaky_relu",
                 negative_slope=0.1):
        super().__init__()
        hidden_features = hidden_features or []
        self.layers = nn.ModuleList()
        first_residual = in_features == out_features
        self.layers.append(GINLayer(
            MLP(in_features, out_features, hidden_features, batch_norm,
                nonlinearity, negative_slope),
            eps, train_eps, first_residual))
        for _ in range(num_layers - 1):
            self.layers.append(GINLayer(
                MLP(out_features, out_features, hidden_features, batch_norm,
                    nonlinearity, negative_slope),
                eps, train_eps, residual))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x


class FDModel(nn.Module):
    """Semantic-guided feature decoupling (Eq. 8).

    NN1 maps the instance to a latent z; NN2 maps each label embedding to an
    importance vector through a sigmoid; their Hadamard product goes through
    NN3 to give one feature per class.
    """

    def __init__(self, in_features_x, in_features_y, hidden_features,
                 out_features, in_layers1=1, out_layers=1, batch_norm=False,
                 nonlinearity="leaky_relu", negative_slope=0.1):
        super().__init__()
        self.NN1 = MLP(in_features_x, hidden_features,
                       [512] * (in_layers1 - 1), batch_norm, nonlinearity,
                       negative_slope)
        self.NN2 = nn.Linear(in_features_y, hidden_features)
        self.NN3 = MLP(hidden_features, out_features,
                       [hidden_features] * (out_layers - 1), batch_norm,
                       nonlinearity, negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        self.NN1.reset_parameters()
        nn.init.kaiming_uniform_(self.NN2.weight, nonlinearity="sigmoid")
        nn.init.constant_(self.NN2.bias, 0.0)
        self.NN3.reset_parameters()

    def forward(self, x, y):
        x = self.NN1(x)                       # [B, H]  the shared latent z
        feature = x
        y = self.NN2(y).sigmoid()             # [C, H]  importance per class
        out = x.unsqueeze(1) * y.unsqueeze(0)  # [B, C, H]
        return feature, self.NN3(out)


class CLIFNet(nn.Module):
    """BERT + GIN + FDModel. Mirrors CLIFNet_new with a text encoder."""

    def __init__(self, cfg):
        super().__init__()
        n_class = cfg.total_class
        self.bert = BertModel.from_pretrained(cfg.model_name)
        self.dropout = nn.Dropout(cfg.classifier_dropout)
        input_size = self.bert.config.hidden_size

        # Fixed random node features; see the module docstring.
        self.label_embedding = nn.Parameter(torch.empty(n_class, n_class),
                                            requires_grad=False)
        self.gin_encoder = GIN(1, n_class, 256, [math.ceil(256 / 2)])
        self.fd_model = FDModel(
            in_features_x=input_size, in_features_y=256,
            hidden_features=cfg.feature_dim, out_features=cfg.feature_dim,
            in_layers1=2, out_layers=1, batch_norm=False,
            nonlinearity="relu", negative_slope=0.1,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.label_embedding)
        self.gin_encoder.reset_parameters()
        self.fd_model.reset_parameters()

    def forward(self, input_ids, attention_mask, token_type_ids, label_adj):
        pooled = self.bert(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).pooler_output
        label_embedding = self.gin_encoder(
            self.label_embedding[: label_adj.shape[0]], label_adj
        )
        feature, decoupled = self.fd_model(self.dropout(pooled), label_embedding)
        return {
            "uni_features": feature,
            "dis_features": decoupled,
            "label_embedding": label_embedding,
        }


class IncrementalCLIFNet(nn.Module):
    """CLIFNet with the grouped Conv1d head that grows per task."""

    def __init__(self, cfg):
        super().__init__()
        self.clifnet = CLIFNet(cfg)
        self.feature_dim = cfg.feature_dim
        self.fc = None

    def update_fc(self, nb_classes: int):
        """One 1-D kernel per class (Eq. 9), carrying old kernels over."""
        fc = nn.Conv1d(nb_classes, nb_classes, self.feature_dim,
                       groups=nb_classes)
        if self.fc is not None:
            n_old = self.fc.bias.data.shape[0]
            with torch.no_grad():
                fc.weight[:n_old] = self.fc.weight.data
                fc.bias[:n_old] = self.fc.bias.data
        device = next(self.parameters()).device
        del self.fc
        self.fc = fc.to(device)

    def forward(self, input_ids, attention_mask, token_type_ids, label_adj,
                kd: bool = False):
        out = self.clifnet(input_ids, attention_mask, token_type_ids, label_adj)
        logits = self.fc(out["dis_features"]).squeeze(2)
        if kd:
            return logits, out["label_embedding"], out["uni_features"]
        return logits, out["label_embedding"]

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class LinkPredictionLossCosine(nn.Module):
    """Pairwise decoder of Eq. 7: fit cosine(E) to the adjacency."""

    def forward(self, emb, adj):
        emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-6)
        return torch.mean(torch.pow(adj - emb_norm @ emb_norm.t(), 2))
