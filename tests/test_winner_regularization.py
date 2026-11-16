"""E178 selection uses only complete inner comparisons and has a fixed tie-break."""
import numpy as np
import pandas as pd

from experiments.e168_winner import win_score
from experiments.e178_winner_regularization import C_GRID, select_regularization


def test_inner_selection():
    R = pd.DataFrame([dict(season=2020, round=r, model=f"C_{c:g}", winner=float(c == .3))
                      for r in (1, 2) for c in C_GRID])
    assert select_regularization(R) == .3
    assert select_regularization(R.assign(winner=1.)) == .003
    outer = R.assign(season=2025, winner=1. - R.winner)
    combined = pd.concat([R, outer])
    assert select_regularization(combined[combined.season <= 2022]) == .3
    try:
        select_regularization(combined)
    except AssertionError:
        pass
    else:
        raise AssertionError("outer labels were accepted in C selection")


def test_default_and_regularization():
    tr = pd.DataFrame(dict(k=[0, 0, 1, 1, 2, 2], win=[1, 0, 1, 0, 1, 0], d3_order=[1., 9., 2., 8., 1., 9.]))
    te = pd.DataFrame(dict(k=[3, 3], d3_order=[1., 9.]))
    default = win_score(tr, te, ["d3_order"])
    np.testing.assert_array_equal(default, win_score(tr, te, ["d3_order"], C=.03))
    assert not np.allclose(default, win_score(tr, te, ["d3_order"], C=3.))


if __name__ == "__main__":
    test_inner_selection(); test_default_and_regularization()
    print("inner-only selection, tie-break, default and C-response checks passed")
