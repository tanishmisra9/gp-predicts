"""E170's two learned factors must ignore target outcomes and form one race distribution."""
import numpy as np
import pandas as pd

from experiments.e170_team_winner import team_winner


def test_team_winner():
    rows = []
    for k in range(8):
        for j, team in enumerate(("A", "A", "B", "B")):
            rows.append(dict(k=k, team=team, win=int(j == 0), d3_order=float(j + 1),
                             t3_order=1.5 if team == "A" else 3.5, t_season_pts=1. if team == "A" else 0.2))
    tr = pd.DataFrame(rows)
    te = tr[tr.k.eq(7)].copy().assign(k=8)
    F = ["d3_order", "t3_order", "t_season_pts"]
    score = team_winner(tr, te, F)
    changed = team_winner(tr, te.assign(win=[0, 0, 0, 1], gridn=[20, 10, 2, 1], order=[20, 10, 2, 1]), F)
    np.testing.assert_array_equal(score, changed)
    assert np.argmin(score) == 0
    np.testing.assert_allclose(-score.sum(), 1)
    uniform = te.assign(d3_order=2., t3_order=2., t_season_pts=0.5)
    np.testing.assert_allclose(team_winner(tr, uniform, F), [-0.25] * 4)
    null = team_winner(tr, te, F, shuffled=True)
    assert np.isfinite(null).all()
    np.testing.assert_allclose(-null.sum(), 1)


if __name__ == "__main__":
    test_team_winner()
    print("team winner target-label, uniform, and probability checks passed")
