"""Estimation of P_text: how often each candidate answer is asserted.

Two estimators, both proxies, to be reported side by side (Section 7.3):

CountEstimator   n-gram counts of the answer string in a corpus index.  Only
                 meaningful when the index is over the model's own training
                 corpus (e.g. the Pile for the Pythia family); otherwise it
                 estimates assertion in a different population.

HumanBaselineEstimator
                 uses human-response data shipped with the benchmark as a proxy
                 for what people assert.  Coarser, but not tied to any corpus.

Both return a vector over the candidate set, normalised to sum to 1.  Neither
is the training distribution itself; conclusions about the decomposition in
eq. (8) are correspondingly indirect (Section 7.5).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

SMOOTHING = 0.5          # add-alpha; keeps zero-count candidates out of log(0)


class CountEstimator:
    """Wraps any count function `count(text) -> int` over a corpus index.

    Plug in an n-gram index over the model's pretraining corpus.  The count
    function is injected so the estimator stays independent of which index is
    used; record the index version in the run metadata.
    """

    def __init__(self, count_fn: Callable[[str], int], context: Optional[str] = None):
        self.count_fn = count_fn
        self.context = context

    def __call__(self, question: str, candidates: Sequence[str]) -> np.ndarray:
        counts = np.array([float(self.count_fn(c)) for c in candidates])
        return _normalise(counts + SMOOTHING)


class HumanBaselineEstimator:
    """Proxy from human-response data: {qid: {candidate: rate}}."""

    def __init__(self, table: dict[str, dict[str, float]]):
        self.table = table

    def __call__(self, qid: str, candidates: Sequence[str]) -> Optional[np.ndarray]:
        row = self.table.get(qid)
        if row is None:
            return None
        rates = np.array([float(row.get(c, 0.0)) for c in candidates])
        if rates.sum() <= 0:
            return None
        return _normalise(rates + SMOOTHING / len(candidates))


def agreement(p_a: np.ndarray, p_b: np.ndarray) -> float:
    """Total-variation distance between two estimates of P_text.

    Reported as a diagnostic: large disagreement means the decomposition should
    be read with caution rather than that one estimator is wrong.
    """
    return float(0.5 * np.abs(np.asarray(p_a) - np.asarray(p_b)).sum())


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    total = v.sum()
    if total <= 0:
        return np.full_like(v, 1.0 / len(v))
    return v / total
