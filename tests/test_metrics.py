import numpy as np

from hebbwatch.core.metrics import competition_index, dead_neurons, sparsity, weight_entropy


def test_metrics_are_valid():
    w = np.array([[0.0, 0.1], [2.0, 0.0]])
    assert 0 <= sparsity(w) <= 1
    assert weight_entropy(w) >= 0
    assert 0 <= competition_index(w) <= 1
    assert dead_neurons(w) >= 0
