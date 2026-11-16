"""E176 uses same-year pre-season inputs, never target outcomes or future seasons."""
import numpy as np
import pandas as pd

from experiments.e176_testing_winner import TESTING_FILES, attach_testing, testing_score


def test_testing_winner_source():
    X = pd.read_csv("data/results.csv").loc[lambda d: d.season <= 2025].copy()
    source = pd.concat([pd.read_csv(p) for p in TESTING_FILES], ignore_index=True)
    before = attach_testing(X, source)
    mask = X.season.eq(2023) & X["round"].eq(2)
    changed = X.copy()
    changed.loc[changed.season.ge(2023), ["order", "grid", "q_pos", "points"]] = 99
    after = attach_testing(changed, source)
    cols = ["prior_testing_strength", "prior_testing_seen"]
    pd.testing.assert_frame_equal(before.loc[mask, cols], after.loc[mask, cols], check_exact=True)
    later = source.copy(); later.loc[later.season.gt(2023), ["best_gap", "lr_gap"]] = 999
    pd.testing.assert_frame_equal(before.loc[mask, cols], attach_testing(X, later).loc[mask, cols], check_exact=True)
    prior = source.copy(); selected = prior.season.eq(2023) & prior.code.eq("VER")
    prior.loc[selected, ["best_gap", "lr_gap"]] = 999
    assert not before.loc[mask, "prior_testing_strength"].equals(attach_testing(X, prior).loc[mask, "prior_testing_strength"])


def test_testing_winner_fallback():
    te = pd.DataFrame(dict(round=[1] * 4, prior_testing_seen=[True] * 4,
                           prior_testing_strength=[.9, .9, .1, .1]))
    score = np.array([.1, .2, .3, .4])
    assert np.argmin(testing_score(te, score, team_first=True)) == 2
    np.testing.assert_array_equal(testing_score(te.assign(round=4), score), score)
    np.testing.assert_array_equal(testing_score(te.assign(prior_testing_seen=False), score, team_first=True), score)


if __name__ == "__main__":
    test_testing_winner_source(); test_testing_winner_fallback()
    print("testing winner target-label, future-source, sensitivity and fallback checks passed")
