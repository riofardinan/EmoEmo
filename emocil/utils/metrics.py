"""Metrics shared by the HS experiments and the pilots (moved here from
_pilot/analysis/pilot.py on 12 Nov 2026, unchanged)."""

import numpy as np
from scipy.stats import rankdata


def auroc(s, y):
    y = y.astype(bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def f1(s, y, t):
    p = s > t; y = y.astype(bool)
    tp = (p & y).sum(); fp = (p & ~y).sum(); fn = (~p & y).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def best_threshold(s, y):
    cand = np.unique(np.quantile(s, np.linspace(0, 1, 201)))
    return max(cand, key=lambda t: f1(s, y, t))
