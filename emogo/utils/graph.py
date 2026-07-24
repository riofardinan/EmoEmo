"""Augmented emotional relation graph (Section 3.3).

Ported from `sym_conditional_prob` / `sym_conditional_prob_update` in
EmoGrowth/models/base.py. Getting these exactly right matters: they are the
"augmented ERG" the paper's title refers to, and the whole AEG-D mechanism for
past-missing partial labels rests on them.

Two details that are easy to lose in a re-implementation, and that a version
reviewed earlier in this project did lose:

  * the diagonal is zeroed before symmetrising, so a class carries no self-loop
    in the adjacency (the GIN's own `(1 + eps) * h` term supplies that); and
  * the matrix is symmetrised **last**, after all the blocks are filled, so
    normalising rows earlier cannot break symmetry.
"""

import torch


def sym_conditional_prob(y: torch.Tensor) -> torch.Tensor:
    """Symmetric conditional co-occurrence over one task's labels (Eq. 1).

    adj[i,j] estimates P(label i | label j), zero on the diagonal, then
    averaged with its transpose.
    """
    adj = torch.matmul(y.t(), y)
    y_sum = torch.sum(y.t(), dim=1, keepdim=True)
    y_sum[y_sum < 1e-6] = 1e-6
    adj = adj / y_sum
    adj.fill_diagonal_(0.0)
    return (adj + adj.t()) * 0.5


def sym_conditional_prob_update(soft_label: torch.Tensor,
                                label_adj_old: torch.Tensor,
                                y: torch.Tensor,
                                known_classes: int,
                                total_classes: int):
    """Grow the graph with the incoming task's classes (Eqs. 4-5).

    `soft_label` is the previous model's **logits** over the old classes for
    this task's data; the sigmoid is applied here, exactly as the original
    does, and the continuous probabilities — not a binarisation — are what the
    dot products consume.

    Layout of the result:
      top-left      the old graph, carried over unchanged
      bottom-right  intra-task co-occurrence among the new classes
      top-right     P(old i | new j), estimated from the soft labels
      bottom-left   the Bayes inversion of the top-right block
    then the whole thing is symmetrised.
    """
    soft_label = torch.sigmoid(soft_label)

    device = label_adj_old.device
    y = y.to(device)
    soft_label = soft_label.to(device)

    adj = torch.zeros(total_classes, total_classes, device=device,
                      dtype=label_adj_old.dtype)
    adj[:known_classes, :known_classes] = label_adj_old
    adj[known_classes:total_classes, known_classes:total_classes] = \
        sym_conditional_prob(y)

    # Upper right: P(old_i | new_j) = <soft_i, y_j> / sum(y_j)
    y_sums = y.sum(dim=0)                                   # [n_new]
    denom = torch.where(y_sums > 0, y_sums, torch.ones_like(y_sums))
    upper = (soft_label.t() @ y) / denom.unsqueeze(0)       # [n_old, n_new]
    upper = torch.where(y_sums.unsqueeze(0) > 0, upper,
                        torch.zeros_like(upper))
    adj[:known_classes, known_classes:total_classes] = upper

    # Lower left, by Bayes: adj[j,i] = adj[i,j] * sum(y_j) / sum(soft_i)
    soft_sums = soft_label.sum(dim=0)                       # [n_old]
    soft_denom = torch.where(soft_sums > 1e-6, soft_sums,
                             torch.full_like(soft_sums, 1e-6))
    lower = upper * y_sums.unsqueeze(0) / soft_denom.unsqueeze(1)
    adj[known_classes:total_classes, :known_classes] = lower.t()

    adj = (adj + adj.t()) * 0.5
    return adj, soft_label
