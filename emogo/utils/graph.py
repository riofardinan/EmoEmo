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

--------------------------------------------------------------------------
Shrinkage (this project's addition; `adj_estimator != "raw"`)

Every entry of the ERG is a ratio of counts, and on GoEmotions those counts
are small. Over the 378 unordered label pairs in the training split: 11% never
co-occur at all, 36% co-occur fewer than five times, 50% fewer than ten. The
median relative standard error of P(i|j) is 30%, and 21% of the estimates have
a standard error larger than half their own value.

Per task it is worse, because each block is estimated from one task's data
over one task's classes. Under B16-I2, three of the six increments have *zero*
co-occurrences: the block the graph branch propagates over is identically
empty. Under B0-I7 the base task has 1013 co-occurrences over 21 pairs and no
empty cell.

So the reference estimator is unbiased but very high variance exactly where
the tasks are small, and the graph branch propagates that variance. Replacing
the raw ratio with a Beta-Binomial posterior mean,

    P(j | i) = (n_ij + alpha * pi_j) / (n_i + alpha)

pulls a pair with little support toward the marginal prevalence pi_j — that
is, toward "no relationship beyond the base rate" — while leaving a
well-supported pair essentially untouched. alpha = 0 recovers the reference
exactly, and the raw path is kept separate so that it does so bit-for-bit.

`ERGEstimator` goes one step further, for `adj_estimator == "shrink_pool"`.
The reference carries the old block of the graph forward unchanged, so once a
block has been estimated from two observations it stays that way for every
later task. Accumulating counts rather than probabilities keeps the effective
sample size attached to each entry, so shrinkage can tell a block built from
two observations apart from one built from two thousand.
"""

from typing import Optional, Tuple

import torch


# --------------------------------------------------------------------- raw

def sym_conditional_prob(y: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
    """Symmetric conditional co-occurrence over one task's labels (Eq. 1).

    adj[i, j] estimates P(label j | label i) — the row index is the
    conditioning class, since the counts are divided by the row sums — zero on
    the diagonal, then averaged with its transpose.

    With `alpha > 0` the ratio becomes a Beta-Binomial posterior mean shrunk
    toward the marginal prevalence of the target class. `alpha = 0` takes the
    reference path unchanged.
    """
    counts = torch.matmul(y.t(), y)
    margin = torch.sum(y.t(), dim=1, keepdim=True)          # [C, 1], n_i

    if alpha <= 0.0:
        margin = margin.clone()
        margin[margin < 1e-6] = 1e-6
        adj = counts / margin
    else:
        adj = _shrink(counts, margin.squeeze(1), y.shape[0], alpha)

    adj.fill_diagonal_(0.0)
    return (adj + adj.t()) * 0.5


def _shrink(counts: torch.Tensor, margin: torch.Tensor, n_samples: int,
            alpha: float) -> torch.Tensor:
    """(n_ij + alpha * pi_j) / (n_i + alpha), broadcast over the matrix.

    `margin[i]` is the number of times class i was seen, `n_samples` the number
    of instances the block was estimated from. pi_j = margin[j] / n_samples is
    the marginal prevalence the estimate is pulled toward.
    """
    total = max(float(n_samples), 1e-6)
    prior = (margin / total).unsqueeze(0)                   # [1, C] -> pi_j
    return (counts + alpha * prior) / (margin.unsqueeze(1) + alpha)


def estimate_alpha(counts: torch.Tensor, margin: torch.Tensor,
                   n_samples: int) -> float:
    """Method-of-moments concentration for the Beta-Binomial prior.

    Offered for `adj_alpha < 0` (auto). A fixed alpha swept over a small grid
    is easier to report and to defend, so this is not the default.
    """
    total = max(float(n_samples), 1e-6)
    n_i = margin.clamp(min=1.0)
    p = counts / n_i.unsqueeze(1)
    off = ~torch.eye(counts.shape[0], dtype=torch.bool, device=counts.device)
    w = n_i.unsqueeze(1).expand_as(p)[off]
    pv = p[off]
    if w.sum() <= 0:
        return 0.0
    mean = float((w * pv).sum() / w.sum())
    var = float((w * (pv - mean) ** 2).sum() / w.sum())
    if var <= 1e-12 or not 0.0 < mean < 1.0:
        return 0.0
    conc = mean * (1.0 - mean) / var - 1.0
    return float(min(max(conc, 0.0), total))


# ------------------------------------------------------------------ update

def sym_conditional_prob_update(soft_label: torch.Tensor,
                                label_adj_old: torch.Tensor,
                                y: torch.Tensor,
                                known_classes: int,
                                total_classes: int,
                                alpha: float = 0.0):
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

    `alpha > 0` applies the same shrinkage to the two blocks that are ratios of
    counts. The carried-over top-left block is not touched here — its support
    is no longer known at this point, which is the limitation `ERGEstimator`
    exists to remove.
    """
    soft_label = torch.sigmoid(soft_label)

    device = label_adj_old.device
    y = y.to(device)
    soft_label = soft_label.to(device)
    n_samples = y.shape[0]

    adj = torch.zeros(total_classes, total_classes, device=device,
                      dtype=label_adj_old.dtype)
    adj[:known_classes, :known_classes] = label_adj_old
    adj[known_classes:total_classes, known_classes:total_classes] = \
        sym_conditional_prob(y, alpha=alpha)

    # Upper right: P(old_i | new_j) = <soft_i, y_j> / sum(y_j)
    y_sums = y.sum(dim=0)                                   # [n_new]
    soft_dot = soft_label.t() @ y                           # [n_old, n_new]
    if alpha <= 0.0:
        denom = torch.where(y_sums > 0, y_sums, torch.ones_like(y_sums))
        upper = soft_dot / denom.unsqueeze(0)
        upper = torch.where(y_sums.unsqueeze(0) > 0, upper,
                            torch.zeros_like(upper))
    else:
        prior_old = soft_label.mean(dim=0).unsqueeze(1)     # [n_old, 1]
        upper = (soft_dot + alpha * prior_old) / (y_sums.unsqueeze(0) + alpha)
    adj[:known_classes, known_classes:total_classes] = upper

    # Lower left, by Bayes: P(new_j | old_i) = n_ij / sum(soft_i)
    soft_sums = soft_label.sum(dim=0)                       # [n_old]
    if alpha <= 0.0:
        soft_denom = torch.where(soft_sums > 1e-6, soft_sums,
                                 torch.full_like(soft_sums, 1e-6))
        lower = upper * y_sums.unsqueeze(0) / soft_denom.unsqueeze(1)
    else:
        prior_new = (y_sums / max(n_samples, 1)).unsqueeze(0)   # [1, n_new]
        lower = ((upper * y_sums.unsqueeze(0) + alpha * prior_new)
                 / (soft_sums.unsqueeze(1) + alpha))
    adj[known_classes:total_classes, :known_classes] = lower.t()

    adj = (adj + adj.t()) * 0.5
    return adj, soft_label


# --------------------------------------------------------- pooled estimator

class ERGEstimator:
    """Count-space ERG for `adj_estimator == "shrink_pool"`.

    Holds the co-occurrence counts and per-class support instead of the
    finished probabilities, so that

      * shrinkage sees the true effective sample size of every block, not just
        of the block estimated most recently; and
      * a block estimated from very little data early on is not frozen at that
        estimate for the rest of the run.

    Old-class counts come from the previous model's sigmoid outputs, the same
    quantity the reference already trusts for the cross block. `pool_old`
    additionally accumulates old-old counts as an outer product of those
    probabilities, which assumes the old labels are conditionally independent
    given the input; it is off by default because that assumption is the whole
    thing the ERG is supposed to model.
    """

    def __init__(self, total_classes: int, alpha: float = 0.0,
                 pool_old: bool = False, device=None, dtype=torch.float32):
        self.alpha = alpha
        self.pool_old = pool_old
        self.count = torch.zeros(total_classes, total_classes,
                                 device=device, dtype=dtype)
        self.margin = torch.zeros(total_classes, device=device, dtype=dtype)
        self.n_samples = 0.0
        self.soft_label: Optional[torch.Tensor] = None

    # -- accumulation ---------------------------------------------------
    def observe_first(self, y: torch.Tensor) -> None:
        k = y.shape[1]
        self.count[:k, :k] += y.t() @ y
        self.margin[:k] += y.sum(dim=0)
        self.n_samples += y.shape[0]

    def observe(self, soft_logits: torch.Tensor, y: torch.Tensor,
                known_classes: int, total_classes: int) -> torch.Tensor:
        soft = torch.sigmoid(soft_logits).to(self.count.device)
        y = y.to(self.count.device)
        k, t = known_classes, total_classes

        self.count[k:t, k:t] += y.t() @ y
        cross = soft.t() @ y                                # [n_old, n_new]
        self.count[:k, k:t] += cross
        self.count[k:t, :k] += cross.t()
        if self.pool_old:
            self.count[:k, :k] += soft.t() @ soft

        self.margin[k:t] += y.sum(dim=0)
        self.margin[:k] += soft.sum(dim=0)
        self.n_samples += y.shape[0]
        self.soft_label = soft
        return soft

    # -- readout --------------------------------------------------------
    def adjacency(self, total_classes: int) -> torch.Tensor:
        c = self.count[:total_classes, :total_classes]
        m = self.margin[:total_classes]
        if self.alpha <= 0.0:
            adj = c / m.clamp(min=1e-6).unsqueeze(1)
        else:
            adj = _shrink(c, m, self.n_samples, self.alpha)
        adj = adj.clone()
        adj.fill_diagonal_(0.0)
        return (adj + adj.t()) * 0.5


# ----------------------------------------------------------------- builder

class ERGBuilder:
    """The ERG for one run, under whichever estimator the config selects.

    AGCN and AESL build the same graph the same way, so the choice of
    estimator lives here rather than twice in the two models. With the
    defaults (`adj_estimator="raw"`, `adj_alpha=0`) every call routes to the
    published formulas unchanged.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = getattr(cfg, "adj_estimator", "raw")
        self.pooled: Optional[ERGEstimator] = None
        if self.mode not in ("raw", "shrink", "shrink_pool"):
            raise ValueError(f"Unsupported adj_estimator: {self.mode!r}")

    # -- helpers --------------------------------------------------------
    def _alpha_for(self, y: torch.Tensor) -> float:
        if self.mode == "raw":
            return 0.0
        a = float(getattr(self.cfg, "adj_alpha", 0.0))
        if a >= 0.0:
            return a
        counts = y.t() @ y
        return estimate_alpha(counts, y.sum(dim=0), y.shape[0])

    def subsample(self, tensors, task: int):
        """Thin the rows the graph is estimated from. Training is untouched."""
        frac = float(getattr(self.cfg, "adj_subsample", 1.0))
        if frac >= 1.0:
            return tensors
        return subsample_rows(tensors, frac, seed=self.cfg.seed * 1000 + task)

    # -- construction ---------------------------------------------------
    def first(self, y: torch.Tensor) -> torch.Tensor:
        alpha = self._alpha_for(y)
        if self.mode == "shrink_pool":
            self.pooled = ERGEstimator(
                self.cfg.total_class, alpha=alpha,
                pool_old=bool(getattr(self.cfg, "adj_pool_old", False)),
                device=y.device, dtype=y.dtype,
            )
            self.pooled.observe_first(y)
            return self.pooled.adjacency(y.shape[1])
        return sym_conditional_prob(y, alpha=alpha)

    def grow(self, soft_logits: torch.Tensor, label_adj_old: torch.Tensor,
             y: torch.Tensor, known_classes: int, total_classes: int):
        alpha = self._alpha_for(y)
        if self.mode == "shrink_pool":
            self.pooled.alpha = alpha
            soft = self.pooled.observe(soft_logits, y, known_classes,
                                       total_classes)
            return self.pooled.adjacency(total_classes), soft
        return sym_conditional_prob_update(
            soft_logits, label_adj_old, y, known_classes, total_classes,
            alpha=alpha,
        )

    def describe(self) -> str:
        if self.mode == "raw":
            return "raw (EmoGrowth Eq. 1)"
        a = float(getattr(self.cfg, "adj_alpha", 0.0))
        alpha = "method-of-moments" if a < 0 else f"alpha={a:g}"
        frac = float(getattr(self.cfg, "adj_subsample", 1.0))
        extra = "" if frac >= 1.0 else f", estimated from {frac:.0%} of rows"
        return f"{self.mode} ({alpha}{extra})"


# ------------------------------------------------------------- subsampling

def subsample_rows(tensors: Tuple[torch.Tensor, ...], fraction: float,
                   seed: int) -> Tuple[torch.Tensor, ...]:
    """Keep a random `fraction` of rows, for the ERG estimate only.

    Used by the adjacency-subsample experiment: the classifier still trains on
    the full task, only the data the graph is estimated from shrinks. That
    separates "how much data estimates the graph" from "how much data trains
    the model", which the benchmark otherwise confounds.
    """
    if fraction >= 1.0:
        return tensors
    n = tensors[0].shape[0]
    keep = max(1, int(round(n * fraction)))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:keep]
    return tuple(t[idx] for t in tensors)
