"""Tests for the metric implementations. Run: python -m pytest tests -q"""

import numpy as np
import pytest

from src import metrics


def test_E_is_zero_for_a_perfectly_warranted_model():
    # all mass on the gold answer
    assert metrics.epistemic_calibration_error([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_E_is_maximal_for_confident_falsehood():
    # all mass on a false answer: E = 1*|1-0| = 1
    assert metrics.epistemic_calibration_error([0.0, 1.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_corollary_1_1_reproductive_case():
    # corpus asserts the falsehood with rate 0.7; at the optimum C_M = P_text
    c_m = [0.3, 0.7]
    c_w = [1.0, 0.0]
    # E = .3*|.3-1| + .7*|.7-0| = .21 + .49
    assert metrics.epistemic_calibration_error(c_m, c_w) == pytest.approx(0.70)


def test_decomposition_bounds_E():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = rng.integers(2, 6)
        c_m = rng.dirichlet(np.ones(k))
        p_txt = rng.dirichlet(np.ones(k))
        c_w = np.zeros(k)
        c_w[rng.integers(k)] = 1.0
        assert metrics.decomposition_slack(c_m, c_w, p_txt) >= -1e-12


def test_decomposition_is_exact_at_the_optimum():
    # C_M = P_text: learning error vanishes, corpus bias equals E
    c_m = p_txt = [0.3, 0.7]
    c_w = [1.0, 0.0]
    cb, le = metrics.gap_decomposition(c_m, c_w, p_txt)
    assert le == pytest.approx(0.0)
    assert cb == pytest.approx(metrics.epistemic_calibration_error(c_m, c_w))


def test_auroc_matches_known_values():
    assert metrics.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert metrics.auroc([1, 2, 3, 4], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert metrics.auroc([1, 1, 1, 1], [0, 0, 1, 1]) == pytest.approx(0.5)


def test_ece_zero_when_perfectly_calibrated():
    conf = [1.0] * 10
    correct = [True] * 10
    assert metrics.expected_calibration_error(conf, correct) == pytest.approx(0.0)


def test_aggregate_reports_stratum_separately():
    items = [
        {"c_m": [0.3, 0.7], "c_w": [1.0, 0.0], "p_txt": [0.3, 0.7]},
        {"c_m": [0.9, 0.1], "c_w": [1.0, 0.0], "p_txt": [0.9, 0.1]},
    ]
    res = metrics.aggregate("low", items)
    assert res.n == 2
    assert res.accuracy == pytest.approx(0.5)
    assert res.learning_error == pytest.approx(0.0)
