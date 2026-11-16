"""Falsifiable numerical and target-label checks for E169's winner-only objective."""
import numpy as np
import pandas as pd

from experiments.e169_choice_winner import choice_score, objective


def test_choice_objective():
    A = np.array([[1., 2.], [0., -1.], [2., 0.], [-1., 1.], [0., 3.]])
    groups = [np.array([0, 1]), np.array([2, 3, 4])]
    winners = [0, 4]; weights = [0.7, 1.]
    beta = np.array([0.2, -0.1]); eps = 1e-6
    value, gradient = objective(beta, A, groups, winners, weights)
    numerical = []
    for i in range(len(beta)):
        delta = np.eye(len(beta))[i] * eps
        numerical.append((objective(beta + delta, A, groups, winners, weights)[0]
                          - objective(beta - delta, A, groups, winners, weights)[0]) / (2 * eps))
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numerical, atol=1e-7)
    zero, _ = objective(np.zeros(2), A, groups, winners, weights)
    np.testing.assert_allclose(zero, 0.7 * np.log(2) + np.log(3))


def test_target_labels_unused():
    tr = pd.DataFrame(dict(k=[0, 0, 1, 1, 2, 2], win=[1, 0, 1, 0, 1, 0],
                           d3_order=[1., 10., 2., 9., 1., 8.]))
    te = pd.DataFrame(dict(k=[3, 3], win=[1, 0], d3_order=[1., 9.]))
    before = choice_score(tr, te, ["d3_order"])
    after = choice_score(tr, te.assign(win=[0, 1], gridn=[20, 1], order=[20, 1]), ["d3_order"])
    np.testing.assert_array_equal(before, after)
    assert before[0] < before[1]
    np.testing.assert_allclose(-before.sum(), 1)
    uniform = choice_score(tr, te.assign(d3_order=[3., 3.]), ["d3_order"])
    np.testing.assert_allclose(uniform, [-0.5, -0.5])


if __name__ == "__main__":
    test_choice_objective(); test_target_labels_unused()
    print("choice winner gradient, uniform, and target-label checks passed")
