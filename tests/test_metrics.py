import numpy as np

from aiot.evaluate import auroc, average_precision, metrics


def test_auroc_perfect_and_ties():
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    assert auroc(np.array([1, 1]), np.array([0.1, 0.2])) is None


def test_average_precision():
    assert average_precision(np.array([0, 1]), np.array([0.1, 0.9])) == 1.0
    assert abs(average_precision(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.1])) - (0.5 + 0.5 * 2 / 3)) < 1e-9


def test_metrics_counts():
    m = metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["fpr"] == 0.5
