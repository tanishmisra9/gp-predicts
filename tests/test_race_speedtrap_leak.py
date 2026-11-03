"""Target speed-trap measurements and race labels cannot enter target features."""

import pandas as pd

from experiments.a1_race_speedtrap import PART, prior_speed
from experiments.common import table


def test_synthetic_cutoff():
    X = pd.DataFrame([(2023, r, r - 1, "AAA", "red") for r in (1, 2, 3)],
                     columns=["season", "round", "k", "code", "team"])
    source = X[["season", "round", "code"]].assign(speed_adv=[2.0, 4.0, 6.0])
    base = prior_speed(X, source)
    changed = source.copy()
    changed.loc[changed["round"] >= 2, "speed_adv"] = 999
    got = prior_speed(X, changed)
    assert base.loc[1, "prior_speed_adv"] == got.loc[1, "prior_speed_adv"]
    changed.loc[changed["round"] == 1, "speed_adv"] = -2
    assert prior_speed(X, changed).loc[1, "prior_speed_adv"] != base.loc[1, "prior_speed_adv"]


def test_cached_target_and_later_scramble():
    X = table(); X = X[X.season <= 2025].copy()
    source = pd.read_csv(PART)
    base = prior_speed(X, source)
    for year, round_ in ((2023, 7), (2025, 20)):
        changed = source.copy()
        later = (changed.season > year) | ((changed.season == year) & (changed["round"] >= round_))
        changed.loc[later, "speed_adv"] = 999
        got = prior_speed(X, changed)
        target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values("code").prior_speed_adv.reset_index(drop=True)
        pd.testing.assert_series_equal(target(base), target(got), check_exact=True)
    changed = X.copy()
    later = (changed.season > 2023) | ((changed.season == 2023) & (changed["round"] >= 7))
    changed.loc[later, ["order", "podium", "win", "gridn", "q_pos"]] = 999
    target = lambda d: d[(d.season == 2023) & (d["round"] == 7)].sort_values("code").prior_speed_adv.reset_index(drop=True)
    pd.testing.assert_series_equal(target(base), target(prior_speed(changed, source)), check_exact=True)


if __name__ == "__main__":
    test_synthetic_cutoff()
    test_cached_target_and_later_scramble()
    print("race speed-trap leak test passed")
