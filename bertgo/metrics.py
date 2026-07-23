"""Port of calculate_metrics.py — the script behind Table 4 of the paper."""

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(
    probs: np.ndarray,
    true: np.ndarray,
    emotions: List[str],
    threshold: float = 0.3,
) -> Dict[str, float]:
    """Binarises probabilities at `threshold` and reports per-label + averages.

    Note the original uses a strict `>` comparison (`pred[pred > t] = 1`), so a
    probability exactly equal to the threshold counts as negative. Kept as-is.

    `accuracy` is subset accuracy: the whole 28-dim label vector must match.
    """
    pred = (probs > threshold).astype(np.float64)
    true = true.astype(np.float64)

    results: Dict[str, float] = {}
    results["accuracy"] = accuracy_score(true, pred)

    for avg in ("macro", "micro", "weighted"):
        p, r, f1, _ = precision_recall_fscore_support(
            true, pred, average=avg, zero_division=0
        )
        results[f"{avg}_precision"] = p
        results[f"{avg}_recall"] = r
        results[f"{avg}_f1"] = f1

    for i, emotion in enumerate(emotions):
        p, r, f1, _ = precision_recall_fscore_support(
            true[:, i], pred[:, i], average="binary", zero_division=0
        )
        results[f"{emotion}_accuracy"] = accuracy_score(true[:, i], pred[:, i])
        results[f"{emotion}_precision"] = p
        results[f"{emotion}_recall"] = r
        results[f"{emotion}_f1"] = f1

    # The paper reports the std of the per-emotion scores alongside the macro
    # average (Table 4: macro F1 .46, std .19).
    for metric in ("precision", "recall", "f1"):
        values = [results[f"{e}_{metric}"] for e in emotions]
        results[f"std_{metric}"] = float(np.std(values))

    return results


def format_table(results: Dict[str, float], emotions: List[str]) -> str:
    """Renders Table 4: per-emotion precision / recall / F1, then the average."""
    lines = [
        f"{'Emotion':<16}{'Precision':>10}{'Recall':>10}{'F1':>10}",
        "-" * 46,
    ]
    for emotion in emotions:
        lines.append(
            f"{emotion:<16}"
            f"{results[f'{emotion}_precision']:>10.2f}"
            f"{results[f'{emotion}_recall']:>10.2f}"
            f"{results[f'{emotion}_f1']:>10.2f}"
        )
    lines.append("-" * 46)
    lines.append(
        f"{'macro-average':<16}"
        f"{results['macro_precision']:>10.2f}"
        f"{results['macro_recall']:>10.2f}"
        f"{results['macro_f1']:>10.2f}"
    )
    lines.append(
        f"{'std':<16}"
        f"{results['std_precision']:>10.2f}"
        f"{results['std_recall']:>10.2f}"
        f"{results['std_f1']:>10.2f}"
    )
    return "\n".join(lines)
