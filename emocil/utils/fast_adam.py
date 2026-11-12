"""BertAdam with one fused update kernel (torch.compile), for the GPU runs.

Same update rule as utils/optimization.py's BertAdam (the original BERT
AdamWeightDecayOptimizer: no bias correction, eps outside the square root,
decoupled weight decay). The reference applies it tensor by tensor in ~26
memory passes over the 110M parameters; on the RTX 3050 that made the
optimizer about half of a BERT-base step (~67 of ~140 ms), bound by memory
bandwidth. Compiling the update fuses it into a few kernels: ~93 ms per step.

Checks (2 Nov 2026, see ../README.md):
  * identical stream of fixed gradients, 300 steps: largest relative
    difference from the reference 1.7e-7, i.e. float32 rounding;
  * real training, 30 steps: trajectories drift apart (2.7e-3 abs), because
    Adam's normalisation amplifies rounding differences on near-zero
    gradients. This is the same kind of drift that GPU nondeterminism in BERT's
    backward pass already causes between two runs of the reference.
A multi-tensor (_foreach) version was bit-identical to the reference but no
faster, which is how the bottleneck was identified as bandwidth.
"""

import torch
from torch.optim import Optimizer

from utils.optimization import bert_lr_lambda, build_param_groups


@torch.compile
def _fused_update(params, grads, m, v, lr, beta1, beta2, eps, wd):
    for p, g, mm, vv in zip(params, grads, m, v):
        mm.mul_(beta1).add_(g * (1.0 - beta1))
        vv.mul_(beta2).add_(g * g * (1.0 - beta2))
        u = mm / (vv.sqrt() + eps)
        if wd > 0.0:
            u = u + wd * p
        p.sub_(u * lr)


@torch.compile
def _fused_update_bc(params, grads, m, v, lr, beta1, beta2, eps, wd, bc1, bc2):
    """As _fused_update with Adam's bias correction; bc1 = 1 - beta1^t and
    bc2 = 1 - beta2^t are 0-d tensors, so the step count does not recompile."""
    for p, g, mm, vv in zip(params, grads, m, v):
        mm.mul_(beta1).add_(g * (1.0 - beta1))
        vv.mul_(beta2).add_(g * g * (1.0 - beta2))
        u = (mm / bc1) / ((vv / bc2).sqrt() + eps)
        if wd > 0.0:
            u = u + wd * p
        p.sub_(u * lr)


class BertAdamFused(Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01,
                 bias_correction=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      bias_correction=bias_correction))
        self._t = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._t += 1
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            for p in params:
                st = self.state[p]
                if len(st) == 0:
                    st["m"] = torch.zeros_like(p); st["v"] = torch.zeros_like(p)
            lr = torch.tensor(group["lr"], device=params[0].device)
            b1, b2 = group["betas"]
            args = (params, [p.grad for p in params], [self.state[p]["m"] for p in params],
                    [self.state[p]["v"] for p in params], lr, b1, b2, group["eps"],
                    group["weight_decay"])
            if group["bias_correction"]:
                dev = params[0].device
                _fused_update_bc(*args, torch.tensor(1.0 - b1 ** self._t, device=dev),
                                 torch.tensor(1.0 - b2 ** self._t, device=dev))
            else:
                _fused_update(*args)
        return loss


def create_optimizer_and_schedule(model, cfg, num_train_steps):
    """utils.optimization.create_optimizer_and_schedule with BertAdamFused."""
    num_warmup_steps = int(num_train_steps * cfg.warmup_proportion)
    groups = build_param_groups(model.named_parameters(), cfg.weight_decay_rate,
                                cfg.exclude_from_weight_decay)
    optimizer = BertAdamFused(groups, lr=cfg.learning_rate,
                              betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_epsilon,
                              bias_correction=getattr(cfg, "bias_correction", False))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: bert_lr_lambda(step, num_train_steps, num_warmup_steps))
    return optimizer, scheduler, num_warmup_steps
