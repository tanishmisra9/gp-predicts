"""Winner GBM must ignore target-weekend labels and use strictly earlier training rows."""
import numpy as np
import pandas as pd

from experiments.e168_winner import win_score
from experiments.e172_nonlinear_winner import winner_gbm


def test_winner_gbm():
    tr = pd.DataFrame([dict(k=k, win=int(j == 0), d3_order=float(j + 1)) for k in range(20) for j in range(10)])
    te = tr[tr.k.eq(19)].copy().assign(k=20)
    before = winner_gbm(tr, te, ["d3_order"])
    after = winner_gbm(tr, te.assign(win=1 - te.win, podium=1, order=1, gridn=1), ["d3_order"])
    np.testing.assert_array_equal(before, after)
    assert np.argmin(before) == 0
    assert np.isfinite(before).all()
    np.testing.assert_array_equal(win_score(tr, te, ["d3_order"], shuffled=True),
                                  win_score(tr, te, ["d3_order"], shuffled=True, seed=168))


if __name__ == "__main__":
    test_winner_gbm()
    print("winner GBM target-label and historical-null-seed checks passed")
