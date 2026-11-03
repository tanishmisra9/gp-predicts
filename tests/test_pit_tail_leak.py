"""Past pit-tail risk must ignore the target race's stops and later stops."""

import glob

import pandas as pd

from experiments.b3_pit_tail import prior_tail, source_table
from experiments.common import table
from f1.data import DATA


def test_synthetic_cutoff():
    X = pd.DataFrame([(2022, r, r - 1, "AAA", "team") for r in (1, 2, 3)],
                     columns=["season", "round", "k", "code", "team"])
    source = X.assign(bad=[1.0, 0.0, 1.0])
    base = prior_tail(X, source)
    changed = source.copy(); changed.loc[changed["round"] >= 2, "bad"] = 1.0
    got = prior_tail(X, changed)
    assert base.loc[1, "prior_badstop"] == got.loc[1, "prior_badstop"]
    changed.loc[changed["round"] == 1, "bad"] = 0.0
    assert prior_tail(X, changed).loc[1, "prior_badstop"] != base.loc[1, "prior_badstop"]


def test_cached_target_and_later_scramble():
    X = table(); X = X[X.season <= 2025].copy()
    pits = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "parts/pits_*.csv"))], ignore_index=True)
    source = source_table(X, pits)
    base = prior_tail(X, source)
    for year, round_ in ((2023, 7), (2025, 20)):
        changed = source.copy()
        later = (changed.season > year) | ((changed.season == year) & (changed["round"] >= round_))
        changed.loc[later, "bad"] = 1.0
        got = prior_tail(X, changed)
        target = lambda d: d[(d.season == year) & (d["round"] == round_)].sort_values("code").prior_badstop.reset_index(drop=True)
        pd.testing.assert_series_equal(target(base), target(got), check_exact=True)


if __name__ == "__main__":
    test_synthetic_cutoff()
    test_cached_target_and_later_scramble()
    print("pit-tail leak test passed")
