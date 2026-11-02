"""Metric/harness edge cases (audit route A5): ties, tiny fields, all-equal scores, few classified."""
import numpy as np
import pandas as pd

import f1.walkforward as W


def race(n, order=None):
    order = order if order is not None else list(range(1, n + 1))
    return pd.DataFrame({"driver": [f"d{i:02d}" for i in range(n)], "order": order, "gridn": order})


def test_all_equal_scores_break_ties_by_name_not_grid():
    g = race(20)
    m = W.metrics(g, np.zeros(20), tie=None)                  # name order d00,d01,d02 == true podium here, by construction
    assert m["podium"] == 1.0
    g2 = race(20, order=list(range(20, 0, -1)))               # true podium = d19,d18,d17 -> name tie-break must NOT find them
    assert W.metrics(g2, np.zeros(20), tie=None)["podium"] == 0.0


def test_perfect_and_reversed_scores():
    g = race(20); assert W.metrics(g, g.order.values, tie=None)["podium"] == 1.0
    assert W.metrics(g, -g.order.values, tie=None)["podium"] == 0.0


def test_tiny_field_and_pairwise_denominator():
    g = race(3); m = W.metrics(g, g.order.values, tie=None); assert m["podium"] == 1.0 and m["pairwise"] == 1.0
    m = W.metrics(race(2), np.array([1, 2]), tie=None); assert 0 <= m["ndcg3"] <= 1.0


if __name__ == "__main__":
    for f in (test_all_equal_scores_break_ties_by_name_not_grid, test_perfect_and_reversed_scores, test_tiny_field_and_pairwise_denominator): f()
    print("metric tests ok")
