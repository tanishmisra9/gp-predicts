"""E175 winner replacement keeps the selected team and ignores target labels."""
import numpy as np
import pandas as pd

from experiments.e175_portable_winner import swap_winner


def test_portable_winner():
    te = pd.DataFrame(dict(driver=["aaa", "bbb", "ccc", "ddd"], team=["A", "A", "B", "B"],
                           prior_portable_gap=[0.2, -0.2, -10., 10.]), index=[41, 17, 8, 2])
    base = np.array([0.1, 0.2, 0.3, 0.4])
    score = swap_winner(te, base)
    assert te.iloc[np.argmin(score)].driver == "bbb"
    np.testing.assert_array_equal(score, swap_winner(te.assign(win=[1, 0, 0, 0], gridn=[1, 2, 3, 4]), base))
    missing = te.copy(); missing.loc[41, "prior_portable_gap"] = np.nan
    np.testing.assert_array_equal(swap_winner(missing, base), base)
    missing.loc[17, "prior_portable_gap"] = np.nan
    np.testing.assert_array_equal(swap_winner(missing, base), base)
    tied = te.copy(); tied.loc[[41, 17], "prior_portable_gap"] = 0.
    assert tied.iloc[np.argmin(swap_winner(tied, base))].driver == "aaa"


if __name__ == "__main__":
    test_portable_winner()
    print("portable winner selected-team, target-label, missing-history and tie checks passed")
