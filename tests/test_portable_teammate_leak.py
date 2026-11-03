"""Portable teammate pace ratings must ignore target-and-later source rows."""

import pandas as pd

from experiments.a5_teammate_segments import PART
from experiments.b6_portable_teammate import feature_table
from experiments.common import table


def test_synthetic_direction_and_cutoff():
    rows = pd.DataFrame([(2022, round_, round_ - 1, code) for round_ in (1, 2, 3)
                         for code in ("AAA", "BBB")], columns=["season", "round", "k", "code"])
    events = pd.DataFrame([(2022, round_, code, mate, gap, 10)
                           for round_ in (1, 2, 3)
                           for code, mate, gap in (("AAA", "BBB", -0.2), ("BBB", "AAA", 0.2))],
                          columns=["season", "round", "code", "mate", "gap", "n"])
    base = feature_table(rows, events)
    target = base[base["round"] == 2].set_index("code").prior_portable_gap
    assert target.AAA < target.BBB
    changed = events.copy()
    changed.loc[changed["round"] >= 2, "gap"] = 99
    got = feature_table(rows, changed)
    pd.testing.assert_series_equal(target, got[got["round"] == 2].set_index("code").prior_portable_gap)
    changed.loc[changed["round"] == 1, "gap"] = 0.2
    got = feature_table(rows, changed)
    assert target.AAA != got[(got["round"] == 2) & (got.code == "AAA")].prior_portable_gap.iloc[0]


def test_cached_target_and_later_scramble():
    X = table(); X = X[X.season <= 2025].copy()
    events = pd.read_csv(PART)
    base = feature_table(X, events)
    for year, round_ in ((2023, 7), (2025, 20)):
        changed = events.copy()
        later = (changed.season > year) | ((changed.season == year) & (changed["round"] >= round_))
        changed.loc[later, "gap"] = 99
        changed.loc[later, "n"] = 1
        got = feature_table(X, changed)
        target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values("code").prior_portable_gap.reset_index(drop=True)
        pd.testing.assert_series_equal(target(base), target(got), check_exact=True)


if __name__ == "__main__":
    test_synthetic_direction_and_cutoff()
    test_cached_target_and_later_scramble()
    print("portable teammate leak test passed")
