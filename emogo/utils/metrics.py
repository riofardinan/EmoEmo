"""Multi-label metrics, ported from EmoGrowth/utils/metrics.py.

Kept numerically identical to the original so numbers are comparable with the
paper's tables, with three changes:
  * `np.int` (removed in NumPy 1.24) replaced by `int`;
  * a `threshold` argument on the label-based metrics, since the original
    hard-codes `outputs > 0` (i.e. sigmoid probability > 0.5). Pass
    threshold=0.0 on logits to reproduce the paper exactly;
  * vectorised inner loops — same results, but usable on 5k x 28 test matrices.
"""

from typing import List, Tuple

import numpy as np
import torch

FMT = "%.4f"


def average_precision(scores_: torch.Tensor, targets_: torch.Tensor):
    """Per-class average precision and its mean (mAP).

    Ranking-based, so it is threshold-free. This is the paper's headline
    metric ("mAP" in Tables 1-3).
    """
    n, n_class = scores_.shape
    ap = torch.zeros(n_class)
    for k in range(n_class):
        scores = scores_[:, k]
        targets = targets_[:, k]
        _, indices = torch.sort(scores, dim=0, descending=True)
        sorted_targets = targets[indices]
        pos_count = torch.cumsum(sorted_targets, dim=0)
        total_count = torch.arange(1, n + 1, dtype=torch.float32)
        precision_at_i = (pos_count / total_count) * sorted_targets
        n_pos = sorted_targets.sum()
        ap[k] = precision_at_i.sum() / n_pos if n_pos > 0 else 0.0
    return ap, torch.mean(ap)


def AveragePrecision(outputs: np.ndarray, true_labels: np.ndarray) -> float:
    """Instance-based average precision (the 'eAP' column in the logs)."""
    m, _ = true_labels.shape
    ap, all_zero_m = 0.0, 0
    for i in range(m):
        rel_lbl_idx = np.where(true_labels[i] == 1)[0]
        if rel_lbl_idx.size == 0:
            all_zero_m += 1
            continue
        tmp_out = outputs[i]
        sort_idx = np.argsort(-tmp_out)
        rank_of = np.empty_like(sort_idx)
        rank_of[sort_idx] = np.arange(1, len(sort_idx) + 1)
        cnt = 0.0
        for j in rel_lbl_idx:
            cntt = np.count_nonzero(tmp_out[rel_lbl_idx] >= outputs[i, j])
            cnt += cntt / rank_of[j]
        ap += cnt / rel_lbl_idx.size
    denom = m - all_zero_m
    return float(FMT % (ap / denom)) if denom else 0.0


def RankingLoss(outputs: np.ndarray, true_labels: np.ndarray) -> float:
    m, q = true_labels.shape
    rl, all_zero_m = 0.0, 0
    for i in range(m):
        rel_lbl = int(np.count_nonzero(true_labels[i]))
        if rel_lbl == 0:
            all_zero_m += 1
            continue
        sort_idx = np.argsort(-outputs[i, :])
        tmp_true = true_labels[i, :][sort_idx]
        # For each relevant label, count irrelevant labels ranked above it.
        n_zero_before = np.cumsum(tmp_true == 0)
        rl_ins = n_zero_before[tmp_true == 1].sum()
        rl += rl_ins / (rel_lbl * (q - rel_lbl) + 1e-5)
    denom = m - all_zero_m
    return float(FMT % (rl / denom)) if denom else 0.0


def Coverage(outputs: np.ndarray, true_labels: np.ndarray) -> float:
    m, q = true_labels.shape
    cov = 0.0
    for i in range(m):
        sort_idx = np.argsort(-outputs[i, :])
        tmp_true = true_labels[i, :][sort_idx]
        if np.sum(tmp_true) != 0:
            cov += np.max(np.where(tmp_true == 1))
    return float(FMT % (cov / m / q))


def OneError(outputs: np.ndarray, true_labels: np.ndarray) -> float:
    m, _ = true_labels.shape
    top1 = np.argmax(outputs, axis=1)
    oe = np.sum(true_labels[np.arange(m), top1] != 1) / m
    return float(FMT % oe)


def HammingLoss(outputs: np.ndarray, true_labels: np.ndarray,
                threshold: float = 0.0) -> float:
    pre_labels = np.array(outputs > threshold, dtype=int)
    m, q = true_labels.shape
    miss_label = np.sum((pre_labels == true_labels) == False)  # noqa: E712
    return float(FMT % (miss_label / (m * q)))


def MacroF1(outputs: np.ndarray, true_labels: np.ndarray,
            threshold: float = 0.0) -> float:
    pre = np.array(outputs > threshold, dtype=int)
    true = true_labels.astype(int)
    _, q = true.shape
    maf = 0.0
    for i in range(q):
        tp = np.sum(pre[:, i] & true[:, i])
        fp = np.sum(pre[:, i] & (1 - true[:, i]))
        fn = np.sum((1 - pre[:, i]) & true[:, i])
        maf += 0.0 if tp + fp + fn == 0 else (2 * tp) / (2 * tp + fp + fn)
    return float(FMT % (maf / q))


def MicroF1(outputs: np.ndarray, true_labels: np.ndarray,
            threshold: float = 0.0) -> float:
    pre = np.array(outputs > threshold, dtype=int)
    true = true_labels.astype(int)
    tp = np.sum(pre & true)
    fp = np.sum(pre & (1 - true))
    fn = np.sum((1 - pre) & true)
    mif = 0.0 if tp + fp + fn == 0 else (2 * tp) / (2 * tp + fp + fn)
    return float(FMT % mif)


METRIC_NAMES: List[str] = [
    "hamming_loss", "avg_precision", "one_error",
    "ranking_loss", "coverage", "macrof1", "microf1",
]


def all_metrics(outputs: np.ndarray, true_labels: np.ndarray,
                threshold: float = 0.0) -> List[Tuple[str, float]]:
    """The seven metrics EmoGrowth logs per task, in the original order."""
    values = [
        HammingLoss(outputs, true_labels, threshold),
        AveragePrecision(outputs, true_labels),
        OneError(outputs, true_labels),
        RankingLoss(outputs, true_labels),
        Coverage(outputs, true_labels),
        MacroF1(outputs, true_labels, threshold),
        MicroF1(outputs, true_labels, threshold),
    ]
    return list(zip(METRIC_NAMES, values))
