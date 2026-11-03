"""A2 joint-corrected pace forecasts must use only earlier race timing."""

import pandas as pd

from experiments.a2_jointpace import PART, prior_joint
from experiments.common import table


def test_synthetic_history():
    X = pd.DataFrame([(2022, r, r - 1, "AAA") for r in (1, 2, 3)],
                     columns=["season", "round", "k", "code"])
    source = X[["season", "round", "code"]].assign(joint_gap=[1.0, 2.0, 3.0])
    base = prior_joint(X, source)
    changed = source.copy()
    changed.loc[changed["round"] >= 2, "joint_gap"] = 99
    got = prior_joint(X, changed)
    assert base.loc[1, "joint_forecast"] == got.loc[1, "joint_forecast"] == 1.0
    changed.loc[changed["round"] == 1, "joint_gap"] = 10
    got = prior_joint(X, changed)
    assert base.loc[1, "joint_forecast"] != got.loc[1, "joint_forecast"]


def test_cached_source_ignores_target_and_later():
    X = table()
    X = X[X.season <= 2025].copy()
    source = pd.read_csv(PART)
    base = prior_joint(X, source)
    for year, round_ in ((2023, 7), (2025, 20)):
        changed = source.copy()
        later = (changed.season > year) | ((changed.season == year) & (changed["round"] >= round_))
        changed.loc[later, "joint_gap"] = 99
        got = prior_joint(X, changed)
        target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values("code").joint_forecast.reset_index(drop=True)
        pd.testing.assert_series_equal(target(base), target(got), check_exact=True)


if __name__ == "__main__":
    test_synthetic_history()
    test_cached_source_ignores_target_and_later()
    print("joint pace leak test passed")
