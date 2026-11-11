"""Epistemic calibration metrics.

Implements the quantities of Section 7 of the paper:

    E(x)        = E_{y ~ Q}|C_M(y|x) - C_W(y|x)|        (Definition 2)
    bound       = corpus-bias term + learning-error term (eq. 8)

plus standard expected calibration error for comparison, and AUROC for
the detector conditions.

Conventions
-----------
An *item* is a question with a finite candidate answer set.  For each item we
have three aligned vectors over candidates:

    c_m   model confidence,   normalised over the candidate set (sums to 1)
    c_w   world validity,     1.0 for the gold answer, 0.0 otherwise
    p_txt assertion frequency estimate, normalised over the candidate set;
          may be None when no estimate is available for that item.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# core quantities
# --------------------------------------------------------------------------

def uniform_baseline_E(k: int) -> float:
    """E for a model that is indifferent over k candidates: 2(k-1)/k^2.

    E mixes sharpness with correctness, so a near-uniform model scores close to
    this value whatever the truth is.  Comparing E against it separates 'the
    model is indifferent' from 'the model is confidently wrong'.
    """
    return 2.0 * (k - 1) / (k * k)


def epistemic_calibration_error(c_m: Sequence[float], c_w: Sequence[float]) -> float:
    """E(x): expectation under the model of |C_M - C_W|."""
    c_m = np.asarray(c_m, dtype=float)
    c_w = np.asarray(c_w, dtype=float)
    if c_m.shape != c_w.shape:
        raise ValueError("c_m and c_w must have the same shape")
    return float(np.sum(c_m * np.abs(c_m - c_w)))


def gap_decomposition(
    c_m: Sequence[float],
    c_w: Sequence[float],
    p_txt: Sequence[float],
) -> tuple[float, float]:
    """Corollary 2.2 / eq. (8): (corpus_bias, learning_error).

    Their sum upper-bounds E(x); the slack is reported by ``decomposition_slack``.
    """
    c_m = np.asarray(c_m, dtype=float)
    c_w = np.asarray(c_w, dtype=float)
    p_txt = np.asarray(p_txt, dtype=float)
    corpus_bias = float(np.sum(c_m * np.abs(p_txt - c_w)))
    learning_error = float(np.sum(c_m * np.abs(c_m - p_txt)))
    return corpus_bias, learning_error


def decomposition_slack(
    c_m: Sequence[float], c_w: Sequence[float], p_txt: Sequence[float]
) -> float:
    """Non-negative gap between the bound and E(x); a sanity check on eq. (8)."""
    cb, le = gap_decomposition(c_m, c_w, p_txt)
    return (cb + le) - epistemic_calibration_error(c_m, c_w)


# --------------------------------------------------------------------------
# comparison metrics
# --------------------------------------------------------------------------

def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 15
) -> float:
    """Standard ECE on top-1 confidence (Guo et al., 2017), for contrast with E."""
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    if conf.shape != acc.shape:
        raise ValueError("confidences and correct must have the same shape")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC via the rank (Mann-Whitney) identity; labels are 1 for the positive class.

    In the detector conditions the positive class is 'the answer is wrong', so a
    detector that fires on errors should score above 0.5.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average ranks within ties
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------
# aggregation over a stratum
# --------------------------------------------------------------------------

@dataclass
class StratumResult:
    name: str
    n: int
    E: float
    E_stderr: float
    corpus_bias: Optional[float]
    learning_error: Optional[float]
    ece: float
    accuracy: float
    mean_k: float
    mean_top1_conf: float
    E_uniform: float
    E_rel: float
    mean_conf_gold: float
    mean_conf_gold_se: float
    chance_conf: float

    def as_row(self) -> dict:
        return {
            "stratum": self.name,
            "n": self.n,
            "E": round(self.E, 4),
            "E_se": round(self.E_stderr, 4),
            "corpus_bias": None if self.corpus_bias is None else round(self.corpus_bias, 4),
            "learning_error": None if self.learning_error is None else round(self.learning_error, 4),
            "ECE": round(self.ece, 4),
            "accuracy": round(self.accuracy, 4),
            "mean_k": round(self.mean_k, 2),
            "mean_top1_conf": round(self.mean_top1_conf, 4),
            "E_uniform": round(self.E_uniform, 4),
            "E_rel": round(self.E_rel, 4),
            "mean_conf_gold": round(self.mean_conf_gold, 4),
            "mean_conf_gold_se": round(self.mean_conf_gold_se, 4),
            "chance_conf": round(self.chance_conf, 4),
        }


def aggregate(name: str, items: Iterable[dict]) -> StratumResult:
    """Aggregate per-item records into a stratum-level result.

    Each record must contain 'c_m' and 'c_w'; 'p_txt' is optional.  Strata are
    never pooled (Section 7.4): call this once per stratum.
    """
    es, cbs, les, top_conf, correct = [], [], [], [], []
    ks, unif, conf_gold, chances = [], [], [], []
    for it in items:
        c_m, c_w = it["c_m"], it["c_w"]
        es.append(epistemic_calibration_error(c_m, c_w))
        if it.get("p_txt") is not None:
            cb, le = gap_decomposition(c_m, c_w, it["p_txt"])
            cbs.append(cb)
            les.append(le)
        k = int(np.argmax(c_m))
        top_conf.append(float(np.max(c_m)))
        correct.append(bool(c_w[k] == 1.0))
        n_cand = len(c_m)
        ks.append(n_cand)
        chances.append(1.0 / n_cand)
        unif.append(uniform_baseline_E(n_cand))
        gold = int(np.argmax(c_w))
        conf_gold.append(float(c_m[gold]))
    es = np.asarray(es)
    n = len(es)
    if n == 0:
        raise ValueError(f"stratum {name!r} is empty")
    return StratumResult(
        name=name,
        n=n,
        E=float(es.mean()),
        E_stderr=float(es.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        corpus_bias=float(np.mean(cbs)) if cbs else None,
        learning_error=float(np.mean(les)) if les else None,
        ece=expected_calibration_error(top_conf, correct),
        accuracy=float(np.mean(correct)),
        mean_k=float(np.mean(ks)),
        mean_top1_conf=float(np.mean(top_conf)),
        E_uniform=float(np.mean(unif)),
        E_rel=float(es.mean() / np.mean(unif)),
        mean_conf_gold=float(np.mean(conf_gold)),
        mean_conf_gold_se=float(np.std(conf_gold, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        chance_conf=float(np.mean(chances)),   # mean of 1/k, NOT 1/mean(k)
    )