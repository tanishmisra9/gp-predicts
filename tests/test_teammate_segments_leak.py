"""A5 paired-lap forecast must ignore target and later teammate pace."""

import pandas as pd

from experiments.a5_teammate_segments import PART, prior_pair
from experiments.common import table


def test_synthetic_history():
    rows = pd.DataFrame([(2022, r, r - 1, "TEAM", "AAA", "BBB") for r in (1, 2, 3)],
                        columns=["season", "round", "k", "team", "code", "mate"])
    events = rows.drop(columns="k").assign(gap=[-0.2, 0.1, -0.4], n=10)
    base = prior_pair(rows, events)
    changed = events.copy()
    changed.loc[changed["round"] >= 2, "gap"] = 99
    got = prior_pair(rows, changed)
    assert base.loc[1, "prior_gap"] == got.loc[1, "prior_gap"] == -0.2
    changed.loc[changed["round"] == 1, "gap"] = 1
    got = prior_pair(rows, changed)
    assert base.loc[1, "prior_gap"] != got.loc[1, "prior_gap"]


def test_cached_source_ignores_target_and_later():
    X = table()
    X = X[X.season <= 2025].copy()
    rows = X[["season", "round", "k", "team", "code"]]
    mates = rows.merge(rows, on=["season", "round", "k", "team"], suffixes=("", "_mate"))
    rows = mates[mates.code != mates.code_mate].rename(columns={"code_mate": "mate"})
    events = pd.read_csv(PART)
    base = prior_pair(rows, events)
    for year, round_ in ((2023, 7), (2025, 20)):
        changed = events.copy()
        later = (changed.season > year) | ((changed.season == year) & (changed["round"] >= round_))
        changed.loc[later, "gap"] = 99
        got = prior_pair(rows, changed)
        target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values(["code", "mate"]).prior_gap.reset_index(drop=True)
        pd.testing.assert_series_equal(target(base), target(got), check_exact=True)


if __name__ == "__main__":
    test_synthetic_history()
    test_cached_source_ignores_target_and_later()
    print("teammate segments leak test passed")
