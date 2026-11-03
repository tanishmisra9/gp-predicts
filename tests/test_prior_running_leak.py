"""Prior-weekend practice running must ignore target and later sessions."""

import pandas as pd
import numpy as np

from experiments.a7_prior_running import PART, add_prior_running
from experiments.common import table


def test_synthetic_cutoff():
    X = pd.DataFrame([(2022, r, r - 1, "AAA", "team") for r in (1, 2, 3)],
                     columns=["season", "round", "k", "code", "team"])
    source = X[["season", "round", "team"]].assign(short=[0.25, 0.5, 1.0], sessions=3, driver_sessions=6)
    base = add_prior_running(X, source)
    changed = source.copy(); changed.loc[changed["round"] >= 2, "short"] = 99
    got = add_prior_running(X, changed)
    assert base.loc[1, "prior_team_short"] == got.loc[1, "prior_team_short"] == 0.25
    changed.loc[changed["round"] == 1, "short"] = 0.75
    assert np.isclose(add_prior_running(X, changed).loc[1, "prior_team_short"], 0.75)
    later = pd.DataFrame([(2022, 7, 6, "AAA", "team")], columns=X.columns)
    assert pd.isna(add_prior_running(pd.concat([X, later], ignore_index=True), source).iloc[-1].prior_team_short)


def test_cached_target_and_later_scramble():
    X = table(); X = X[X.season <= 2025].copy()
    source = pd.read_csv(PART)
    base = add_prior_running(X, source)
    for year, round_ in ((2023, 7), (2025, 20)):
        changed = source.copy()
        later = (changed.season > year) | ((changed.season == year) & (changed["round"] >= round_))
        changed.loc[later, "short"] = 99
        got = add_prior_running(X, changed)
        target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values("code").prior_team_short.reset_index(drop=True)
        pd.testing.assert_series_equal(target(base), target(got), check_exact=True)
        outcomes = X.copy()
        later_x = (outcomes.season > year) | ((outcomes.season == year) & (outcomes["round"] >= round_))
        outcomes.loc[later_x, ["order", "podium", "win"]] = 0
        pd.testing.assert_series_equal(target(base), target(add_prior_running(outcomes, source)), check_exact=True)


if __name__ == "__main__":
    test_synthetic_cutoff()
    test_cached_target_and_later_scramble()
    print("prior running leak test passed")
