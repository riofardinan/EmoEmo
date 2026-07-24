"""Method registry.

Add a method by importing it here and adding one line to `_REGISTRY`. The
remaining EmoGrowth baselines (EWC, LwF, ER/RS/PRS/OCDM, AGCN, KRT-R, AESL)
plug in the same way — each subclasses Finetune and overrides `_compute_loss`,
`_train_dataset_kwargs`, or `build_rehearsal_memory`.
"""

from models.ewc import EWC
from models.finetune import Finetune
from models.lwf import LwF
from models.replay import Replay

_REGISTRY = {
    "finetune": Finetune,
    "ewc": EWC,
    "lwf": LwF,
    # Both are models/replay.py; the method name picks the buffer strategy.
    "er": Replay,
    "rs": Replay,
    "prs": Replay,
    "replay": Replay,   # honours cfg.buffer_type instead
}


def get_model(method: str, cfg):
    name = method.lower()
    if name not in _REGISTRY:
        raise NotImplementedError(
            f"Method '{method}' is not implemented yet. Available: "
            f"{sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](cfg)


def available_methods():
    return sorted(_REGISTRY)
