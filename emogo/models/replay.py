"""Replay baselines — ER and RS (EmoGrowth/models/replay_ml.py).

Both keep a buffer of past training samples and mix it into every later task.
They differ only in how the buffer is filled, selected with `buffer_type`:

  "random" (ER)  Per class in the current task, keep up to `memory_per_class`
                 samples carrying that class, chosen uniformly. Union across
                 the task's classes, appended to the buffer.
  "rs"     (RS)  Classic reservoir sampling over the stream of all training
                 samples seen so far, capped at `memory_size`. Every sample has
                 the same probability of surviving, so frequent emotions
                 dominate — which is exactly the weakness PRS and OCDM later
                 address.

Replay fixes the past-missing problem differently from LwF: the buffered rows
carry their *real* labels over the old classes, so the model sees genuine
positive evidence for them instead of the zeros Finetune supplies. What it does
not fix is that the *current* task's rows still say zero for old classes. The
paper's read (Section 4.2) is that this is why replay underperforms here:
"just saving the labels of current task aggravates the partial label problem in
subsequent training."

The buffer stores indices into the training split plus each sample's global
class indices — the text itself never needs copying.
"""

import logging
import math
import random
from typing import List

import torch

from models.finetune import Finetune

logger = logging.getLogger(__name__)


class Replay(Finetune):
    # Method name -> buffer type, so `--method er` can never disagree with a
    # stale `buffer_type` left in the config file.
    _BY_METHOD = {"er": "random", "rs": "rs", "prs": "prs"}

    def __init__(self, cfg):
        super().__init__(cfg)
        self.buffer_type = self._BY_METHOD.get(
            cfg.method.lower(), cfg.buffer_type.lower()
        )
        if self.buffer_type not in ("random", "rs", "prs"):
            raise NotImplementedError(
                f"buffer_type '{self.buffer_type}' is not implemented yet. "
                f"Available: random (ER), rs (RS), prs (PRS). OCDM is not built."
            )
        self._data_memory: List[int] = []
        self._targets_memory_ml: List[List[int]] = []
        # Reservoir sampling counts the whole stream, across tasks.
        self.total_sample = 0
        self.running_stats = {}

    # ------------------------------------------------------------ plumbing

    def _get_memory(self):
        if not self._data_memory:
            return None
        return (self._data_memory, self._targets_memory_ml)

    def _train_dataset_kwargs(self) -> dict:
        return {"appendent": self._get_memory()}

    def _compute_loss(self, out, targets, inputs, batch) -> torch.Tensor:
        """No zero-padding here — the data manager already widened the targets.

        When a buffer is attached, `get_dataset` returns labels spanning every
        class seen so far (zeros for old classes on current-task rows, real
        labels on buffered rows). So the loss is taken as-is, matching
        `fake_targets = targets` in replay_ml.py's `_update_representation`.
        """
        logits = out["logits"]
        if targets.shape[1] == logits.shape[1]:
            return self.criterion(logits, targets)
        # Defensive: an empty buffer at task > 0 would leave narrow targets.
        return super()._compute_loss(out, targets, inputs, batch)

    # --------------------------------------------------------- buffer build

    def build_rehearsal_memory(self, data_manager):
        # The original skips buffer construction after the final task — there
        # is no later task to replay into.
        if self._total_classes >= self.cfg.total_class:
            logger.info("[Replay] last task, buffer not updated")
            return

        idx, tensors, _ = data_manager.get_dataset(
            self._cur_task, source="train", ret_data=True
        )
        indices = idx.tolist()
        targets = tensors[3]

        if self.buffer_type == "random":
            self._fill_random(indices, targets, data_manager)
        elif self.buffer_type == "rs":
            self._fill_reservoir(indices, targets)
        elif self.buffer_type == "prs":
            self._fill_prs(indices, targets)

        logger.info("[Replay/%s] buffer size: %d", self.buffer_type,
                    len(self._data_memory))

    def _fill_random(self, indices, targets, data_manager):
        """ER: up to `samples_per_class` per class of the current task."""
        m = self.samples_per_class
        selected = []
        for local_class in range(data_manager.get_task_size(self._cur_task)):
            rows = torch.nonzero(targets[:, local_class] == 1,
                                 as_tuple=True)[0].tolist()
            selected += rows if len(rows) <= m else random.sample(rows, m)
        selected = list(set(selected))

        for row in selected:
            self._data_memory.append(indices[row])
            self._targets_memory_ml.append(self._global_labels(targets[row]))
        logger.info("[Replay/random] added %d exemplars (<=%d per class)",
                    len(selected), m)

    def _fill_reservoir(self, indices, targets):
        """RS: reservoir sampling over the stream, capped at memory_size."""
        m = self.cfg.memory_size
        replaced = 0
        for row in range(len(indices)):
            self.total_sample += 1
            if len(self._data_memory) == m:
                # Accept with probability m / total_sample.
                if random.randint(1, self.total_sample) <= m:
                    slot = random.randint(0, m - 1)
                    self._data_memory[slot] = indices[row]
                    self._targets_memory_ml[slot] = self._global_labels(targets[row])
                    replaced += 1
            else:
                self._data_memory.append(indices[row])
                self._targets_memory_ml.append(self._global_labels(targets[row]))
        logger.info("[Replay/rs] stream=%d, buffer=%d, replaced=%d this task",
                    self.total_sample, len(self._data_memory), replaced)

    def _fill_prs(self, indices, targets):
        """PRS: Partitioning Reservoir Sampling."""
        m = self.cfg.memory_size
        rho = getattr(self.cfg, "prs_rho", 0.5)
        replaced = 0
        
        for row in range(len(indices)):
            self.total_sample += 1
            idx = indices[row]
            global_tgt = self._global_labels(targets[row])
            
            for c in global_tgt:
                self.running_stats[c] = self.running_stats.get(c, 0) + 1
                
            if len(self._data_memory) < m:
                self._data_memory.append(idx)
                self._targets_memory_ml.append(global_tgt)
            else:
                unique_classes = list(self.running_stats.keys())
                sum_n_rho = sum(n_c ** rho for n_c in self.running_stats.values())
                p = {c: (self.running_stats[c] ** rho) / sum_n_rho for c in unique_classes}
                target_m = {c: m * p[c] for c in unique_classes}
                
                sum_y_e_n = sum(math.exp(-self.running_stats[c]) for c in global_tgt)
                w = {c: (math.exp(-self.running_stats[c]) / sum_y_e_n if sum_y_e_n > 0 else 0) for c in global_tgt}
                s = sum((target_m[c] / self.running_stats[c]) * w[c] for c in global_tgt if self.running_stats[c] > 0)
                s = min(max(s, 0.0), 1.0)
                
                if random.random() <= s:
                    l = {c: 0 for c in unique_classes}
                    for mem_tgt in self._targets_memory_ml:
                        for c in mem_tgt:
                            if c in l:
                                l[c] += 1
                                
                    sum_l = sum(l.values())
                    delta = {c: l[c] - p[c] * sum_l for c in unique_classes}
                    
                    pos_delta_classes = [c for c in unique_classes if delta[c] > 0]
                    if not pos_delta_classes:
                        slot = random.randint(0, m - 1)
                        self._data_memory[slot] = idx
                        self._targets_memory_ml[slot] = global_tgt
                        replaced += 1
                        continue
                        
                    delta_tensor = torch.tensor([delta[c] for c in pos_delta_classes], dtype=torch.float)
                    probs = torch.softmax(delta_tensor, dim=0).tolist()
                    sampled_class = random.choices(pos_delta_classes, weights=probs, k=1)[0]
                    
                    Y_indices = [idx_mem for idx_mem, mem_tgt in enumerate(self._targets_memory_ml) if sampled_class in mem_tgt]
                    q = {c: (0 if delta[c] > 0 else 1) for c in unique_classes}
                    
                    max_score = -1
                    K_indices = []
                    for idx_mem in Y_indices:
                        mem_tgt = self._targets_memory_ml[idx_mem]
                        score = sum(q[c] for c in unique_classes if c not in mem_tgt)
                        if score > max_score:
                            max_score = score
                            K_indices = [idx_mem]
                        elif score == max_score:
                            K_indices.append(idx_mem)
                            
                    min_dist = float('inf')
                    best_z = -1
                    for k in K_indices:
                        mem_tgt_k = self._targets_memory_ml[k]
                        C_k = {c: l[c] - (1 if c in mem_tgt_k else 0) for c in unique_classes}
                        sum_C_k = sum(C_k.values())
                        dist = sum(abs(C_k[c] - p[c] * sum_C_k) for c in unique_classes)
                        if dist < min_dist:
                            min_dist = dist
                            best_z = k
                            
                    if best_z != -1:
                        self._data_memory[best_z] = idx
                        self._targets_memory_ml[best_z] = global_tgt
                        replaced += 1

        logger.info("[Replay/prs] stream=%d, buffer=%d, replaced=%d this task",
                    self.total_sample, len(self._data_memory), replaced)

    def _global_labels(self, row: torch.Tensor) -> List[int]:
        """Task-local one-hot row -> global class indices."""
        local = torch.nonzero(row == 1, as_tuple=True)[0].tolist()
        return [c + self._known_classes for c in local]

    def after_task(self):
        super().after_task()
        logger.info("[Replay] exemplar size: %d", len(self._data_memory))
