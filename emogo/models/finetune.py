"""Finetune: the no-anti-forgetting baseline (EmoGrowth/models/finetune_ml.py).

Each task simply continues training on the new data. The detail that makes this
the *multi-label* finetune baseline rather than a strawman: at task b the
targets for classes seen earlier are set to zero and the loss is taken over the
full widened head. So the model is actively told "none of the old emotions are
present" on every new sample — which is precisely the past-missing partial
label problem, and why Finetune collapses in Tables 1-3.

Later methods subclass this and override `_compute_loss`.
"""

import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.base import BaseLearner
from nets.incremental_net import IncrementalBertNet
from utils.optimization import create_optimizer_and_schedule

logger = logging.getLogger(__name__)


class Finetune(BaseLearner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._network = IncrementalBertNet(cfg)
        self.criterion = nn.MultiLabelSoftMarginLoss()

    # --------------------------------------------------------------- task loop

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logger.info(
            "Task %d — learning classes %d-%d: %s",
            self._cur_task, self._known_classes, self._total_classes,
            data_manager.task_emotions(self._cur_task),
        )

        train_dataset = data_manager.get_dataset(self._cur_task, source="train",
                                                 **self._train_dataset_kwargs())
        test_dataset = data_manager.get_dataset(self._cur_task, source="test")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=self.cfg.drop_last,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.cfg.device.type == "cuda"),
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

        self._network.to(self._device)
        self._run_epochs()
        self.build_rehearsal_memory(data_manager)

    def _train_dataset_kwargs(self) -> dict:
        """Hook for methods that need replay samples or affective vectors."""
        return {}

    def _run_epochs(self):
        n_epochs = (
            self.cfg.init_epochs if self._cur_task == 0 else self.cfg.epochs
        )
        steps_per_epoch = len(self.train_loader)
        num_train_steps = steps_per_epoch * n_epochs
        optimizer, scheduler, warmup = create_optimizer_and_schedule(
            self._network, self.cfg, num_train_steps
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=self.cfg.fp16 and self._device.type == "cuda"
        )
        logger.info(
            "Task %d — %d epochs x %d steps = %d steps (warmup %d)",
            self._cur_task, n_epochs, steps_per_epoch, num_train_steps, warmup,
        )

        for epoch in range(n_epochs):
            self._network.train()
            total_loss, seen, t0 = 0.0, 0, time.time()
            for batch in self.train_loader:
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    loss = self._step(batch)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._network.parameters(), self.cfg.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()
                seen += 1

            logger.info(
                "Task %d epoch %d/%d — loss %.4f (%.1fs)",
                self._cur_task, epoch + 1, n_epochs,
                total_loss / max(1, seen), time.time() - t0,
            )

    # ------------------------------------------------------------------ losses

    def _step(self, batch) -> torch.Tensor:
        inputs = tuple(t.to(self._device, non_blocking=True) for t in batch[:3])
        targets = batch[3].to(self._device, non_blocking=True)
        out = self._network(*inputs)
        return self._compute_loss(out, targets, inputs, batch)

    def _compute_loss(self, out, targets, inputs, batch) -> torch.Tensor:
        logits = out["logits"]
        if self._cur_task == 0:
            return self.criterion(logits, targets)
        # Pad old classes with zeros and score the whole head.
        fake_targets = torch.hstack(
            [torch.zeros(targets.shape[0], self._known_classes,
                         device=self._device), targets]
        )
        return self.criterion(logits, fake_targets)

    def after_task(self):
        self._known_classes = self._total_classes
