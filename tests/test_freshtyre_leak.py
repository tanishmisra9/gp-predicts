"""Fresh-tyre-corrected pace may use only completed prior races."""

import pandas as pd

from experiments.a2_freshtyre import PART, feature_table
from experiments.common import table


def test_synthetic_cutoff():
    rows = pd.DataFrame([(2022, r, r - 1, "AAA") for r in (1, 2, 3)],
                        columns=["season", "round", "k", "code"])
    source = rows.drop(columns="k").assign(joint_gap=[1.0, 2.0, 3.0])
    base = feature_table(rows, source)
    changed = source.copy()
    changed.loc[changed["round"] >= 2, "joint_gap"] = 99
    got = feature_table(rows, changed)
    assert base.loc[1, "fresh_forecast"] == got.loc[1, "fresh_forecast"] == 1
    changed.loc[changed["round"] == 1, "joint_gap"] = 4
    assert feature_table(rows, changed).loc[1, "fresh_forecast"] == 4


def test_cached_source_and_target_outcomes():
    X = table(); X = X[X.season <= 2025].copy()
    source = pd.read_csv(PART)
    base = feature_table(X, source)
    for year, round_ in ((2023, 7), (2025, 20)):
        changed = source.copy()
        later = (changed.season > year) | ((changed.season == year) & (changed["round"] >= round_))
        changed.loc[later, "joint_gap"] = 99
        got = feature_table(X, changed)
        target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values("code").fresh_forecast.reset_index(drop=True)
        pd.testing.assert_series_equal(target(base), target(got), check_exact=True)
        outcomes = X.copy()
        later_x = (outcomes.season > year) | ((outcomes.season == year) & (outcomes["round"] >= round_))
        outcomes.loc[later_x, ["order", "podium", "win"]] = 0
        pd.testing.assert_series_equal(target(base), target(feature_table(outcomes, source)), check_exact=True)


if __name__ == "__main__":
    test_synthetic_cutoff()
    test_cached_source_and_target_outcomes()
    print("fresh tyre leak test passed")
