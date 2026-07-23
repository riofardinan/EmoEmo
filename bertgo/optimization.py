"""Faithful port of bert/optimization.py (AdamWeightDecayOptimizer).

This is deliberately not `torch.optim.AdamW`. Two differences from the original
BERT optimizer routinely cost a point or two of F1 when people replicate this
paper:

  1. No bias correction. TF computes `update = m / (sqrt(v) + eps)` on the raw
     moments. torch.optim.AdamW divides by (1 - beta^t). With only ~10.8k steps
     and warmup this changes the effective early learning rate noticeably.
  2. epsilon sits *outside* the square root: `sqrt(v) + eps`, not
     `sqrt(v + eps)`. And eps is 1e-6, not the 1e-8 torch default.

Weight decay is decoupled (added to the update, not to the gradient) and is
skipped for LayerNorm and bias tensors.
"""

import re
from typing import Iterable, List

import torch
from torch.optim import Optimizer


class BertAdam(Optimizer):
    """AdamWeightDecayOptimizer from the original BERT release."""

    def __init__(
        self,
        params,
        lr: float,
        betas=(0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("BertAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]

                # Standard Adam moments, no bias correction (matches TF).
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                update = m / (v.sqrt() + group["eps"])

                # Decoupled weight decay: added to the update, so it does not
                # interact with the m/v accumulators.
                if group["weight_decay"] > 0.0:
                    update = update + group["weight_decay"] * p

                p.add_(update, alpha=-group["lr"])

        return loss


def build_param_groups(
    named_parameters: Iterable, weight_decay: float, exclude: List[str]
):
    """Splits parameters into decayed / not-decayed groups.

    `exclude` holds regex fragments matched against the parameter name, the
    same way `_do_use_weight_decay` uses re.search in the original.
    """
    decay, no_decay = [], []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if any(re.search(pattern, name) is not None for pattern in exclude):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def bert_lr_lambda(step: int, num_train_steps: int, num_warmup_steps: int) -> float:
    """The TF schedule: linear warmup over a *global* linear decay.

    Note the decay term is measured against the total step count from step 0,
    not from the end of warmup. That is what `polynomial_decay(power=1.0)`
    computes in create_optimizer, and it differs slightly from HuggingFace's
    get_linear_schedule_with_warmup.
    """
    decayed = max(0.0, 1.0 - step / float(max(1, num_train_steps)))
    if num_warmup_steps and step < num_warmup_steps:
        return step / float(max(1, num_warmup_steps))
    return decayed


def create_optimizer_and_schedule(model, cfg, num_train_steps: int):
    """Mirrors create_optimizer() in bert/optimization.py."""
    num_warmup_steps = int(num_train_steps * cfg.warmup_proportion)
    groups = build_param_groups(
        model.named_parameters(),
        cfg.weight_decay_rate,
        cfg.exclude_from_weight_decay,
    )
    optimizer = BertAdam(
        groups,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_epsilon,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: bert_lr_lambda(step, num_train_steps, num_warmup_steps),
    )
    return optimizer, scheduler, num_warmup_steps
