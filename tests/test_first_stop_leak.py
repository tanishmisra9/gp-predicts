"""A target race cannot use its own first-stop timing or later timing."""

import glob

import pandas as pd

from experiments.b3_first_stop import past_first_stop, source_table
from experiments.common import table
from f1.data import DATA, load_results


def test_synthetic_cutoff():
    X = pd.DataFrame([(2023, r, r - 1, "red") for r in (1, 2, 3)],
                     columns=["season", "round", "k", "team"])
    source = X.assign(resid=[0.1, -0.2, 0.3], drivers=2)
    base = past_first_stop(X, source)
    changed = source.copy()
    changed.loc[changed["round"] >= 2, "resid"] = 0.9
    got = past_first_stop(X, changed)
    assert base.loc[1, "forecast"] == got.loc[1, "forecast"]
    changed.loc[changed["round"] == 1, "resid"] = -0.1
    assert past_first_stop(X, changed).loc[1, "forecast"] != base.loc[1, "forecast"]


def test_cached_target_and_later_scramble():
    X = table(); X = X[X.season <= 2025].copy()
    pits = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "parts/pits_*.csv"))], ignore_index=True)
    source = source_table(X, pits, load_results())
    base = past_first_stop(X, source)
    for year, round_ in ((2023, 7), (2025, 20)):
        changed = source.copy()
        later = (changed.season > year) | ((changed.season == year) & (changed["round"] >= round_))
        changed.loc[later, "resid"] = 0.9
        got = past_first_stop(X, changed)
        target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values("team").forecast.reset_index(drop=True)
        pd.testing.assert_series_equal(target(base), target(got), check_exact=True)


if __name__ == "__main__":
    test_synthetic_cutoff()
    test_cached_target_and_later_scramble()
    print("first-stop leak test passed")
