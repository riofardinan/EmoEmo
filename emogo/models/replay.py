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
    _BY_METHOD = {"er": "random", "rs": "rs", "prs": "prs", "ocdm": "ocdm"}

    def __init__(self, cfg):
        super().__init__(cfg)
        self.buffer_type = self._BY_METHOD.get(
            cfg.method.lower(), cfg.buffer_type.lower()
        )
        if self.buffer_type not in ("random", "rs", "prs", "ocdm"):
            raise NotImplementedError(
                f"buffer_type '{self.buffer_type}' is not implemented. "
                f"Available: random (ER), rs (RS), prs (PRS), ocdm (OCDM)."
            )
        self._data_memory: List[int] = []
        self._targets_memory_ml: List[List[int]] = []
        # Reservoir sampling counts the whole stream, across tasks.
        self.total_sample = 0
        # PRS: per-class positive counts, appended one task at a time
        # (classes are partitioned, so each class's count is finalised in its
        # own task). `running_statistics` in the original.
        self.running_statistics: List[float] = []

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
        elif self.buffer_type == "ocdm":
            self._fill_ocdm(indices, targets)

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
        """PRS — Partitioning Reservoir Sampling (Kim et al., 2020).

        Faithful port of `_construct_exemplar_unified_ml_prs` in EmoGrowth's
        base.py, with one numerical fix that is required at GoEmotions' scale
        and changes nothing where the original already worked:

          The sample-in weight is  W_c = exp(-N_c) / sum_a exp(-N_a)  over the
          sample's active classes a. With N in the thousands (GoEmotions has up
          to 14k positives for a class; Audio28 had at most a few hundred),
          exp(-N) underflows to 0 and the buffer silently freezes. Since only
          differences of N matter, W is exactly softmax(-N) over the active
          classes, computed here by subtracting min(N) first — identical result,
          no underflow.

        `N` is the per-class positive count, accumulated one task at a time.
        `rou` (config `prs_rho`) is 0 in the original, which makes the target
        partition uniform — the whole point of PRS being to balance the buffer.
        """
        import numpy as np

        m = self.cfg.memory_size
        rou = self.cfg.prs_rho

        # Extend running_statistics with this task's per-class counts.
        task_counts = targets.sum(dim=0).cpu().numpy()
        self.running_statistics = self.running_statistics + task_counts.tolist()
        N = np.asarray(self.running_statistics, dtype=np.float64)
        total = len(N)

        P = np.power(N, rou) / np.sum(np.power(N, rou))
        M = P * m
        replaced = 0

        for row in range(len(indices)):
            self.total_sample += 1
            idx = indices[row]
            active = self._global_labels(targets[row])   # global class indices

            if len(self._data_memory) < m:
                self._data_memory.append(idx)
                self._targets_memory_ml.append(active)
                continue

            # --- probability of sample-in --------------------------------
            if not active:
                continue
            n_active = N[active]
            # softmax(-N) over the active classes, underflow-safe.
            shifted = -(n_active - n_active.min())
            w_exp = np.exp(shifted)
            W = w_exp / w_exp.sum()
            s = float(np.sum(M[active] / n_active * W))
            if random.random() >= s:
                continue

            # --- sample-out: evict from the most over-represented class ---
            L = self._buffer_class_counts(total)
            delta = L - P * np.sum(L)
            selected = int(np.argmax(delta))           # argmax(softmax) = argmax

            Y = [i for i, t in enumerate(self._targets_memory_ml)
                 if selected in t]
            if not Y:
                continue

            # Among Y, keep those covering the fewest under-represented classes.
            q = (delta <= 0).astype(np.float64)
            n_star = [float(np.dot(self._inv_multihot(self._targets_memory_ml[i],
                                                      total), q)) for i in Y]
            best = max(n_star)
            K = [Y[i] for i, v in enumerate(n_star) if v == best]

            # Tie-break by whichever eviction leaves the buffer closest to P.
            best_dist, chosen = float("inf"), K[0]
            for k in K:
                others = (self._targets_memory_ml[:k]
                          + self._targets_memory_ml[k + 1:])
                c_k = self._buffer_class_counts(total, targets_list=others)
                dist = float(np.sum(np.abs(c_k - P * np.sum(c_k))))
                if dist < best_dist:
                    best_dist, chosen = dist, k

            self._data_memory[chosen] = idx
            self._targets_memory_ml[chosen] = active
            replaced += 1

        logger.info("[Replay/prs] stream=%d, buffer=%d, replaced=%d this task, "
                    "N range %d-%d", self.total_sample, len(self._data_memory),
                    replaced, int(N.min()), int(N.max()))

    def _buffer_class_counts(self, total, targets_list=None):
        import numpy as np
        counts = np.zeros(total, dtype=np.float64)
        for labels in (self._targets_memory_ml if targets_list is None
                       else targets_list):
            counts[labels] += 1
        return counts

    @staticmethod
    def _inv_multihot(labels, total):
        import numpy as np
        v = np.ones(total, dtype=np.float64)
        v[labels] = 0.0
        return v

    def _fill_ocdm(self, indices, targets):
        """OCDM: pick the buffer whose label distribution is closest to uniform.

        Follows `_construct_exemplar_unified_ml_ocdm` in EmoGrowth's base.py.
        Two things about that implementation are worth knowing:

        * It is **not** the greedy algorithm of Liang & Li (2022). The authors
          commented the greedy version out and replaced it with a random search
          over 10,000 candidate subsets, keeping the one whose class
          distribution has the lowest KL divergence from uniform. Since Table 3
          was produced by the random-search version, that is what is
          reproduced here.
        * The target distribution is uniform over all classes seen so far,
          which is what makes OCDM a *balancing* method like PRS.

        The search is vectorised over the multi-hot matrix; the original loops
        in Python, which would be far too slow at GoEmotions' scale.
        """
        import numpy as np

        m = self.cfg.memory_size
        n_trials = self.cfg.ocdm_trials
        n_seen = self._total_classes

        # Candidate pool = current buffer + this task's rows.
        pool_idx = list(self._data_memory)
        pool_lab = list(self._targets_memory_ml)

        # Top the buffer up first, exactly as the original does.
        rows = list(range(len(indices)))
        if len(pool_idx) < m:
            take = min(len(rows), m - len(pool_idx))
            chosen = random.sample(rows, take)
            for row in chosen:
                pool_idx.append(indices[row])
                pool_lab.append(self._global_labels(targets[row]))
            rows = [r for r in rows if r not in set(chosen)]

        for row in rows:
            pool_idx.append(indices[row])
            pool_lab.append(self._global_labels(targets[row]))

        if len(pool_idx) <= m:
            self._data_memory = pool_idx
            self._targets_memory_ml = pool_lab
            logger.info("[Replay/ocdm] pool %d <= memory %d, kept all",
                        len(pool_idx), m)
            return

        multi_hot = np.zeros((len(pool_idx), n_seen), dtype=np.float64)
        for i, labels in enumerate(pool_lab):
            if labels:
                multi_hot[i, labels] = 1.0

        p_target = np.ones(n_seen) / n_seen
        rng = np.random.default_rng(self.cfg.seed + self._cur_task)

        best_kl, best = np.inf, None
        for _ in range(n_trials):
            cand = rng.choice(len(pool_idx), size=m, replace=False)
            counts = multi_hot[cand].sum(axis=0)
            total = counts.sum()
            if total == 0:
                continue
            p = counts / total
            # KL(p || uniform), skipping zero entries as the original does.
            kl = np.sum(np.where(p != 0, p * np.log(p / p_target), 0.0))
            if kl < best_kl:
                best_kl, best = kl, cand

        self._data_memory = [pool_idx[i] for i in best]
        self._targets_memory_ml = [pool_lab[i] for i in best]
        logger.info("[Replay/ocdm] pool %d -> %d, KL to uniform %.4f "
                    "(%d trials)", len(pool_idx), m, best_kl, n_trials)

    def _global_labels(self, row: torch.Tensor) -> List[int]:
        """Task-local one-hot row -> global class indices."""
        local = torch.nonzero(row == 1, as_tuple=True)[0].tolist()
        return [c + self._known_classes for c in local]

    def after_task(self):
        super().after_task()
        logger.info("[Replay] exemplar size: %d", len(self._data_memory))
